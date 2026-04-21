"""
Phase 4: Production-Grade Graph Builder
Builds Neo4j graph with SUPERSEDES relationships and temporal constraints.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from tqdm import tqdm

from src.logger import get_logger
from src.config import config
from src.utils import handle_errors, safe_json_load, safe_json_dump, Timer

logger = get_logger(__name__)


@dataclass
class GraphStats:
    """Statistics from graph building."""
    nodes_created: int
    relationships_created: int
    documents_processed: int
    chunks_processed: int
    build_time_seconds: float
    timestamp: str


class Neo4jGraphBuilder:
    """
    Production-grade Neo4j graph builder with validation and error recovery.
    """
    
    def __init__(self):
        self.uri = config.neo4j.uri
        self.user = config.neo4j.user
        self.password = config.neo4j.password
        self.driver = None
        self._connect()
    
    def _connect(self) -> bool:
        """Establish Neo4j connection with retry."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    max_connection_lifetime=3600,
                    max_connection_pool_size=10
                )
                self.driver.verify_connectivity()
                logger.info(f"Connected to Neo4j: {self.uri}")
                return True
                
            except AuthError as e:
                logger.error(f"Authentication failed: {e}")
                return False
            except ServiceUnavailable as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
            except Exception as e:
                logger.error(f"Unexpected connection error: {e}")
                return False
        
        logger.error("Failed to connect to Neo4j after retries")
        return False
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.debug("Neo4j connection closed")
    
    @handle_errors(default_return=False)
    def clear_database(self) -> bool:
        """Clear all nodes and relationships."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")
            return True
    
    @handle_errors(default_return=False)
    def create_indexes(self) -> bool:
        """Create indexes for performance."""
        with self.driver.session() as session:
            # Index on Clause.id
            session.run("CREATE INDEX clause_id IF NOT EXISTS FOR (c:Clause) ON (c.id)")
            # Index on Clause.doc_id
            session.run("CREATE INDEX clause_doc_id IF NOT EXISTS FOR (c:Clause) ON (c.doc_id)")
            # Index on Clause.effective_date
            session.run("CREATE INDEX clause_date IF NOT EXISTS FOR (c:Clause) ON (c.effective_date)")
            logger.info("Indexes created")
            return True
    
    @handle_errors(default_return=None)
    def create_clause_node(self, chunk: Dict) -> Optional[str]:
        """
        Create a single Clause node.
        
        Args:
            chunk: Chunk dictionary with id, doc_id, effective_date, text
            
        Returns:
            Node ID if successful, None otherwise
        """
        chunk_id = chunk.get('chunk_id')
        doc_id = chunk.get('doc_id')
        effective_date = chunk.get('effective_date')
        text = chunk.get('text', '')
        
        # Truncate text for storage (Neo4j string limit)
        text_preview = text[:1000] if text else ''
        
        with self.driver.session() as session:
            result = session.run("""
                MERGE (c:Clause {id: $chunk_id})
                SET c.doc_id = $doc_id,
                    c.effective_date = date($effective_date),
                    c.text = $text,
                    c.char_count = $char_count,
                    c.created_at = datetime()
                RETURN c.id as id
            """,
                chunk_id=chunk_id,
                doc_id=doc_id,
                effective_date=effective_date,
                text=text_preview,
                char_count=len(text)
            )
            record = result.single()
            return record['id'] if record else None
    
    @handle_errors(default_return=0)
    def create_clause_nodes_batch(self, chunks: List[Dict]) -> int:
        """
        Create Clause nodes in batch.
        
        Returns:
            Number of nodes created
        """
        created = 0
        
        with self.driver.session() as session:
            for chunk in tqdm(chunks, desc="Creating nodes", leave=False):
                try:
                    node_id = self.create_clause_node(chunk)
                    if node_id:
                        created += 1
                except Exception as e:
                    logger.error(f"Failed to create node {chunk.get('chunk_id')}: {e}")
        
        return created
    
    @handle_errors(default_return=0)
    def create_supersedes_relationships(self, manifest: pd.DataFrame) -> int:
        """
        Create SUPERSEDES relationships between documents.
        
        Args:
            manifest: DataFrame with doc_id and supersedes_doc_id columns
            
        Returns:
            Number of relationships created
        """
        created = 0
        
        # Filter rows with valid supersedes
        valid_rows = manifest[manifest['supersedes_doc_id'].notna()]
        valid_rows = valid_rows[valid_rows['supersedes_doc_id'] != 'None']
        valid_rows = valid_rows[valid_rows['supersedes_doc_id'] != '']
        
        with self.driver.session() as session:
            for _, row in tqdm(valid_rows.iterrows(), 
                              desc="Creating relationships", 
                              leave=False,
                              total=len(valid_rows)):
                doc_id = row['doc_id']
                supersedes = row['supersedes_doc_id']
                
                try:
                    result = session.run("""
                        MATCH (new:Clause {doc_id: $doc_id})
                        MATCH (old:Clause {doc_id: $supersedes})
                        WHERE new.effective_date > old.effective_date
                        MERGE (new)-[:SUPERSEDES]->(old)
                        RETURN count(*) as count
                    """,
                        doc_id=doc_id,
                        supersedes=supersedes
                    )
                    record = result.single()
                    if record and record['count'] > 0:
                        created += record['count']
                        logger.debug(f"SUPERSEDES: {doc_id} -> {supersedes}")
                        
                except Exception as e:
                    logger.error(f"Failed to create SUPERSEDES for {doc_id}: {e}")
        
        return created
    
    @handle_errors(default_return=0)
    def create_amendment_relationships(self) -> int:
        """
        Create AMENDS relationships between clauses with same section number.
        """
        created = 0
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c1:Clause), (c2:Clause)
                WHERE c1.doc_id <> c2.doc_id
                  AND c1.id CONTAINS c2.id
                  AND c1.effective_date > c2.effective_date
                MERGE (c1)-[:AMENDS]->(c2)
                RETURN count(*) as count
            """)
            record = result.single()
            if record:
                created = record['count']
                logger.info(f"Created {created} AMENDS relationships")
        
        return created
    
    def get_graph_stats(self) -> Dict:
        """Get current graph statistics."""
        with self.driver.session() as session:
            node_count = session.run("MATCH (n:Clause) RETURN count(n) as c").single()['c']
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()['c']
            
            supersedes_count = session.run(
                "MATCH ()-[r:SUPERSEDES]->() RETURN count(r) as c"
            ).single()['c']
            
            return {
                'total_nodes': node_count,
                'total_relationships': rel_count,
                'supersedes_relationships': supersedes_count
            }
    
    def validate_graph(self) -> Tuple[bool, List[str]]:
        """Validate graph integrity."""
        issues = []
        
        with self.driver.session() as session:
            # Check for orphaned nodes
            orphaned = session.run("""
                MATCH (c:Clause)
                WHERE NOT (c)-[:SUPERSEDES]-()
                  AND NOT EXISTS {
                    MATCH (other:Clause)
                    WHERE other.doc_id = c.doc_id AND other.id <> c.id
                  }
                RETURN count(c) as count
            """).single()['count']
            
            if orphaned > 0:
                issues.append(f"Found {orphaned} potentially orphaned nodes")
            
            # Check for circular references
            circular = session.run("""
                MATCH path = (c:Clause)-[:SUPERSEDES*2..5]->(c)
                RETURN count(path) as count
            """).single()['count']
            
            if circular > 0:
                issues.append(f"Found {circular} circular SUPERSEDES references")
        
        return len(issues) == 0, issues
    
    def build(self, chunks_path: Path = None, manifest_path: Path = None) -> Optional[GraphStats]:
        """
        Build complete graph from chunks and manifest.
        
        Args:
            chunks_path: Path to clauses.json
            manifest_path: Path to document_manifest.csv
            
        Returns:
            GraphStats if successful, None otherwise
        """
        if not self.driver:
            logger.error("No Neo4j connection")
            return None
        
        start_time = datetime.now()
        
        # Load data
        chunks_path = chunks_path or config.paths.chunks_dir / "clauses.json"
        manifest_path = manifest_path or config.paths.project_root / "document_manifest.csv"
        
        if not chunks_path.exists():
            logger.error(f"Chunks not found: {chunks_path}")
            return None
        
        if not manifest_path.exists():
            logger.error(f"Manifest not found: {manifest_path}")
            return None
        
        chunks = safe_json_load(chunks_path, default=[])
        manifest = pd.read_csv(manifest_path)
        
        logger.info(f"Loaded {len(chunks)} chunks from {len(manifest)} documents")
        
        with Timer("Graph building"):
            # Clear existing data
            self.clear_database()
            
            # Create indexes
            self.create_indexes()
            
            # Create nodes
            nodes_created = self.create_clause_nodes_batch(chunks)
            logger.info(f"Created {nodes_created} Clause nodes")
            
            # Create SUPERSEDES relationships
            supersedes_created = self.create_supersedes_relationships(manifest)
            logger.info(f"Created {supersedes_created} SUPERSEDES relationships")
            
            # Create AMENDS relationships (optional)
            amends_created = self.create_amendment_relationships()
            
            total_relationships = supersedes_created + amends_created
        
        # Validate
        is_valid, issues = self.validate_graph()
        if not is_valid:
            logger.warning(f"Graph validation issues: {issues}")
        
        # Get final stats
        stats = self.get_graph_stats()
        build_time = (datetime.now() - start_time).total_seconds()
        
        # Save build report
        report = {
            'build_timestamp': datetime.now().isoformat(),
            'nodes_created': nodes_created,
            'relationships_created': total_relationships,
            'supersedes_created': supersedes_created,
            'amends_created': amends_created,
            'documents_processed': len(manifest),
            'chunks_processed': len(chunks),
            'build_time_seconds': build_time,
            'validation_passed': is_valid,
            'validation_issues': issues,
            'final_stats': stats
        }
        
        report_path = config.paths.processed_texts_dir / "graph_build_report.json"
        safe_json_dump(report, report_path)
        
        logger.info(f"✅ Graph build complete in {build_time:.1f}s")
        logger.info(f"   Nodes: {stats['total_nodes']}")
        logger.info(f"   Relationships: {stats['total_relationships']}")
        logger.info(f"📊 Report: {report_path}")
        
        return GraphStats(
            nodes_created=nodes_created,
            relationships_created=total_relationships,
            documents_processed=len(manifest),
            chunks_processed=len(chunks),
            build_time_seconds=build_time,
            timestamp=datetime.now().isoformat()
        )


def main():
    """Main entry point for graph building phase."""
    logger.info("=" * 60)
    logger.info("Phase 4: Production Graph Builder")
    logger.info("=" * 60)
    
    builder = Neo4jGraphBuilder()
    
    try:
        stats = builder.build()
        
        if stats:
            logger.info(f"✅ Successfully built graph with {stats.nodes_created} nodes")
            return 0
        else:
            logger.error("❌ Graph build failed")
            return 1
            
    finally:
        builder.close()


if __name__ == "__main__":
    exit(main())
