"""
Incremental document ingestion for production use.
Processes a single new document and adds it to all indices.
"""

import fitz
import json
import re
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer
import chromadb
from neo4j import GraphDatabase

from src.config import config
from src.logger import get_logger
from src.utils import handle_errors

logger = get_logger(__name__)


def chunk_text_by_size(text: str, max_chars: int = 1200, overlap: int = 200) -> list:
    """Split text into overlapping chunks at paragraph boundaries."""
    paragraphs = text.split('\n\n')
    chunks = []
    current = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) > max_chars and current:
            chunks.append(current.strip())
            current = current[-overlap:] + "\n\n" + para if len(current) > overlap else para + "\n\n"
        else:
            current += para + "\n\n"
    
    if current.strip():
        chunks.append(current.strip())
    
    return chunks


@handle_errors(default_return=False)
def ingest_single_document(doc_id: str, effective_date: str) -> bool:
    """
    Process a single new document and add it to all indices.
    
    Args:
        doc_id: Document ID (e.g., 'doc_009')
        effective_date: Effective date in YYYY-MM-DD format
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Ingesting document: {doc_id} (effective: {effective_date})")
    
    # 1. Parse PDF
    pdf_path = config.RAW_PDFS_DIR / f"{doc_id}.pdf"
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return False
    
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    
    # 2. Save processed text with metadata
    txt_path = config.PROCESSED_TEXTS_DIR / f"{doc_id}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"[DOCUMENT METADATA]\n")
        f.write(f"DOC_ID: {doc_id}\n")
        f.write(f"EFFECTIVE_DATE: {effective_date}\n")
        f.write(f"[END METADATA]\n\n{text}")
    
    logger.info(f"Saved processed text: {txt_path}")
    
    # 3. Chunk the document
    chunks = chunk_text_by_size(text, max_chars=config.chunking.max_chars)
    
    if not chunks:
        logger.warning(f"No chunks generated for {doc_id}")
        return False
    
    # 4. Update clauses.json
    clauses_path = config.CHUNKS_DIR / "clauses.json"
    if clauses_path.exists():
        with open(clauses_path, 'r', encoding='utf-8') as f:
            all_clauses = json.load(f)
    else:
        all_clauses = []
    
    new_chunks = []
    for i, chunk in enumerate(chunks):
        new_chunks.append({
            'chunk_id': f'{doc_id}_chunk_{i+1:03d}',
            'doc_id': doc_id,
            'effective_date': effective_date,
            'text': chunk
        })
    
    all_clauses.extend(new_chunks)
    
    with open(clauses_path, 'w', encoding='utf-8') as f:
        json.dump(all_clauses, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Added {len(new_chunks)} chunks to clauses.json")
    
    # 5. Add to ChromaDB
    try:
        client = chromadb.PersistentClient(path=str(config.VECTORS_DIR))
        collection = client.get_collection("legal_clauses")
        
        embedder = SentenceTransformer(config.embedding.model_name)
        
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for chunk in new_chunks:
            ids.append(chunk['chunk_id'])
            documents.append(chunk['text'])
            embeddings.append(embedder.encode(chunk['text']).tolist())
            metadatas.append({
                'doc_id': doc_id,
                'effective_date': effective_date,
                'chunk_id': chunk['chunk_id']
            })
        
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(new_chunks)} embeddings to ChromaDB")
    except Exception as e:
        logger.error(f"Failed to add to ChromaDB: {e}")
        return False
    
    # 6. Add to Neo4j (if enabled)
    if config.neo4j.is_valid():
        try:
            driver = GraphDatabase.driver(
                config.neo4j.uri,
                auth=(config.neo4j.user, config.neo4j.password)
            )
            
            with driver.session() as session:
                for chunk in new_chunks:
                    session.run("""
                        CREATE (c:Clause {
                            id: $chunk_id,
                            doc_id: $doc_id,
                            effective_date: date($effective_date),
                            text: $text
                        })
                    """, 
                    chunk_id=chunk['chunk_id'],
                    doc_id=doc_id,
                    effective_date=effective_date,
                    text=chunk['text'])
                
                # Create SUPERSEDES relationships if applicable
                manifest_path = config.PROJECT_ROOT / "document_manifest.csv"
                import pandas as pd
                manifest = pd.read_csv(manifest_path)
                row = manifest[manifest['doc_id'] == doc_id]
                
                if not row.empty and pd.notna(row.iloc[0]['supersedes_doc_id']):
                    supersedes = row.iloc[0]['supersedes_doc_id']
                    session.run("""
                        MATCH (new:Clause {doc_id: $doc_id})
                        MATCH (old:Clause {doc_id: $supersedes})
                        MERGE (new)-[:SUPERSEDES]->(old)
                    """, doc_id=doc_id, supersedes=supersedes)
                    logger.info(f"Created SUPERSEDES relationship: {doc_id} -> {supersedes}")
            
            driver.close()
            logger.info(f"Added {len(new_chunks)} nodes to Neo4j")
        except Exception as e:
            logger.warning(f"Neo4j update skipped: {e}")
    
    logger.info(f"✅ Successfully ingested {doc_id}: {len(new_chunks)} chunks")
    return True


def refresh_bm25_index():
    """Refresh the BM25 index after new documents are added."""
    try:
        from src.query_engine import HybridRetriever
        # This will be called by the QueryEngine on next initialization
        logger.info("BM25 index will be refreshed on next QueryEngine init")
    except Exception as e:
        logger.error(f"Failed to refresh BM25 index: {e}")