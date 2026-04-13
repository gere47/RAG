"""
Phase 6: Production-Grade Query Engine
Graph-Grounded Temporal RAG with hybrid search and reranking.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

import chromadb
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from neo4j import GraphDatabase
from rank_bm25 import BM25Okapi

from src.logger import get_logger
from src.config import config
from src.utils import handle_errors

os.environ["CHROMA_TELEMETRY_IMPL"] = "none"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="neo4j")

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """Structured retrieved chunk with metadata."""
    chunk_id: str
    doc_id: str
    text: str
    effective_date: str
    score: float
    is_current: bool = True
    rerank_score: Optional[float] = None


@dataclass
class QueryResult:
    """Complete query result."""
    answer: str
    sources: List[Dict[str, Any]]
    graph_used: bool
    num_chunks_retrieved: int
    retrieval_time_ms: int
    generation_time_ms: int
    total_time_ms: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class HybridRetriever:
    """Combines vector similarity and BM25 keyword matching."""
    
    def __init__(self, collection: chromadb.Collection, embedding_model: SentenceTransformer):
        self.collection = collection
        self.embedder = embedding_model
        self.documents: List[str] = []
        self.ids: List[str] = []
        self.metadatas: List[Dict] = []
        self.bm25: Optional[BM25Okapi] = None
        self._build_index()
    
    @handle_errors()
    def _build_index(self):
        """Build BM25 index from all documents in ChromaDB."""
        results = self.collection.get()
        
        if not results:
            logger.warning("No documents found in collection")
            return
        
        self.ids = results.get('ids', [])
        self.documents = results.get('documents', [])
        self.metadatas = results.get('metadatas', [])
        
        if self.documents:
            tokenized = [doc.split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized)
            logger.info(f"Built BM25 index with {len(self.documents)} documents")
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        scores = np.array(scores)
        min_s, max_s = scores.min(), scores.max()
        if max_s == min_s:
            return np.ones_like(scores) * 0.5
        return ((scores - min_s) / (max_s - min_s)).tolist()
    
    def search(self, query: str, top_k: int = 10, alpha: float = 0.7) -> List[Tuple[str, float]]:
        """Hybrid search combining vector and BM25 scores."""
        if not self.bm25 or not self.documents:
            return self._vector_search_only(query, top_k)
        
        # Vector search
        query_embedding = self.embedder.encode(query, normalize_embeddings=True).tolist()
        n_results = min(len(self.documents), max(top_k * 3, 30))
        
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not vector_results or 'ids' not in vector_results or not vector_results['ids']:
            return []
        
        result_ids = vector_results['ids'][0]
        
        # Vector scores from distances
        if 'distances' in vector_results and vector_results['distances']:
            distances = vector_results['distances'][0]
            vector_scores = self._normalize_scores([1.0 / (1.0 + d) for d in distances])
        else:
            vector_scores = self._normalize_scores([1.0 / (i + 1) for i in range(len(result_ids))])
        
        # BM25 scores
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores_norm = self._normalize_scores(bm25_scores)
        
        # Combine scores
        combined_scores = {}
        id_to_index = {id_: idx for idx, id_ in enumerate(self.ids)}
        
        for i, chunk_id in enumerate(result_ids):
            if chunk_id not in id_to_index:
                continue
            bm25_idx = id_to_index[chunk_id]
            combined = alpha * vector_scores[i] + (1 - alpha) * bm25_scores_norm[bm25_idx]
            combined_scores[chunk_id] = combined
        
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]
    
    def _vector_search_only(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Fallback to pure vector search."""
        query_embedding = self.embedder.encode(query, normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['metadatas']
        )
        if not results or 'ids' not in results or not results['ids']:
            return []
        return [(id_, 1.0 / (i + 1)) for i, id_ in enumerate(results['ids'][0])]
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Retrieve chunk by ID."""
        results = self.collection.get(ids=[chunk_id])
        if results and results['documents']:
            return {
                'id': chunk_id,
                'text': results['documents'][0],
                'metadata': results['metadatas'][0] if results['metadatas'] else {}
            }
        return None


class ReRanker:
    """Cross-encoder based re-ranking."""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)
        logger.info(f"Loaded reranker: {model_name}")
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Re-rank documents using cross-encoder."""
        if not documents:
            return []
        
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class QueryEngine:
    """Main query engine with graph-grounded temporal awareness."""
    
    def __init__(self):
        self.config = config
        
        self._init_vector_db()
        self._init_graph_db()
        self._init_models()
        self._init_retriever()
        
        logger.info("QueryEngine initialized successfully")
    def _init_vector_db(self):
        """Initialize ChromaDB connection."""
        self.chroma_client = chromadb.PersistentClient(
        path=str(self.config.paths.vectors_dir),       
        settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_collection("legal_clauses")
        logger.info(f"Connected to ChromaDB: {self.collection.count()} documents")
    
    def _init_graph_db(self):
        """Initialize Neo4j connection."""
        if self.config.neo4j.is_valid():
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    self.config.neo4j.uri,
                    auth=(self.config.neo4j.user, self.config.neo4j.password)
                )
                self.neo4j_driver.verify_connectivity()
                logger.info(f"Connected to Neo4j: {self.config.neo4j.uri}")
                self.graph_enabled = True
            except Exception as e:
                logger.warning(f"Neo4j connection failed: {e}. Graph features disabled.")
                self.neo4j_driver = None
                self.graph_enabled = False
        else:
            self.neo4j_driver = None
            self.graph_enabled = False
    
    def _init_models(self):
        """Initialize embedding and LLM models."""
        self.embedder = SentenceTransformer(self.config.embedding.model_name)
        self.reranker = ReRanker()
    
    def _init_retriever(self):
        """Initialize hybrid retriever."""
        self.retriever = HybridRetriever(self.collection, self.embedder)
    
    @handle_errors(default_return=(None, None, None))
    def _get_newest_version(self, chunk_id: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Find the most recent version of a clause using graph traversal."""
        if not self.graph_enabled or not self.neo4j_driver:
            chunk = self.retriever.get_chunk_by_id(chunk_id)
            if chunk:
                meta = chunk['metadata']
                return chunk_id, chunk['text'], meta.get('effective_date', 'unknown')
            return chunk_id, None, None
        
        with self.neo4j_driver.session() as session:
            query = """
            MATCH (c:Clause {id: $chunk_id})
            OPTIONAL MATCH path = (c)-[:SUPERSEDES*]->(newest:Clause)
            WHERE NOT (newest)-[:SUPERSEDES]->()
            RETURN COALESCE(newest, c) AS final_node
            """
            result = session.run(query, chunk_id=chunk_id).single()
            if result:
                node = result['final_node']
                return node['id'], node.get('text'), str(node.get('effective_date', ''))
        
        return chunk_id, None, None
    
    def _deduplicate_chunks(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Remove duplicate chunks based on content similarity."""
        seen = set()
        unique = []
        
        for chunk in chunks:
            sig = chunk.text[:200].strip().lower()
            if sig not in seen:
                seen.add(sig)
                unique.append(chunk)
        
        return unique
    
    def answer(self, question: str, target_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Answer a question using Graph-Grounded Temporal RAG.
        """
        total_start = time.perf_counter()
        
        # 1. Hybrid retrieval
        retrieval_start = time.perf_counter()
        retrieved = self.retriever.search(question, top_k=20)
        retrieval_time = int((time.perf_counter() - retrieval_start) * 1000)
        
        if not retrieved:
            return {
                "answer": "No relevant documents found to answer this question.",
                "sources": [],
                "graph_used": self.graph_enabled,
                "num_chunks_retrieved": 0,
                "retrieval_time_ms": retrieval_time,
                "generation_time_ms": 0,
                "total_time_ms": int((time.perf_counter() - total_start) * 1000),
                "timestamp": datetime.now().isoformat()
            }
        
        # 2. Resolve current versions via graph
        chunks: List[RetrievedChunk] = []
        
        for chunk_id, retrieval_score in retrieved[:8]:
            newest_id, newest_text, newest_date = self._get_newest_version(chunk_id)
            
            if newest_text is None:
                continue
            
            meta = self.collection.get(ids=[newest_id], include=['metadatas'])
            doc_id = meta['metadatas'][0].get('doc_id', 'unknown') if meta['metadatas'] else 'unknown'
            
            is_current = True
            if target_date and newest_date and newest_date > target_date:
                is_current = False
            
            chunks.append(RetrievedChunk(
                chunk_id=newest_id,
                doc_id=doc_id,
                text=newest_text,
                effective_date=newest_date or 'unknown',
                score=retrieval_score,
                is_current=is_current
            ))
        
        # 3. Deduplicate
        unique_chunks = self._deduplicate_chunks(chunks)
        
        # 4. Rerank
        chunk_texts = [c.text[:800] for c in unique_chunks[:10]]
        reranked = self.reranker.rerank(question, chunk_texts, top_k=3)
        
        final_chunks = []
        text_to_chunk = {c.text[:800]: c for c in unique_chunks}
        for text, score in reranked:
            chunk = text_to_chunk.get(text)
            if chunk:
                chunk.rerank_score = float(score)
                final_chunks.append(chunk)
        
        # 5. Build context
        context_parts = []
        sources = []
        
        for chunk in final_chunks[:3]:
            status = "CURRENT" if chunk.is_current else "HISTORICAL"
            context_parts.append(
                f"[{status}] {chunk.doc_id} (Effective: {chunk.effective_date}):\n{chunk.text[:600]}"
            )
            sources.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "effective_date": chunk.effective_date,
                "is_current": chunk.is_current,
                "score": chunk.rerank_score or chunk.score
            })
        
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No context available."
        
        prompt = f"""[INST] You are analyzing legal documents. Answer based ONLY on the context below.
If the context doesn't contain the answer, say "Not found in these documents."

Context from documents:
{context}

Question: {question}

Provide a clear, direct answer using only information from the context. [/INST]"""
        
        generation_start = time.perf_counter()
        response = ollama.generate(
            model=self.config.ollama.model,
            prompt=prompt,
            options={"temperature": 0.1, "num_predict": 300}
        )
        generation_time = int((time.perf_counter() - generation_start) * 1000)
        
        answer = response['response'].strip()
        total_time = int((time.perf_counter() - total_start) * 1000)
        
        logger.info(f"Query: {question[:50]}... → {total_time}ms")
        
        return {
            "answer": answer,
            "sources": sources,
            "graph_used": self.graph_enabled,
            "num_chunks_retrieved": len(final_chunks),
            "retrieval_time_ms": retrieval_time,
            "generation_time_ms": generation_time,
            "total_time_ms": total_time,
            "timestamp": datetime.now().isoformat()
        }


def query(question: str, target_date: Optional[str] = None) -> str:
    """Quick query interface."""
    engine = QueryEngine()
    result = engine.answer(question, target_date)
    return result["answer"]


if __name__ == "__main__":
    engine = QueryEngine()
    print("\n" + "=" * 60)
    print("Graph-Grounded Temporal RAG - Query Engine")
    print("=" * 60)
    print(f"Graph enabled: {engine.graph_enabled}")
    print(f"Vector count: {engine.collection.count()}")
    print("=" * 60 + "\n")
    
    while True:
        q = input("Question (or 'exit'): ").strip()
        if q.lower() == 'exit':
            break
        if not q:
            continue
        
        result = engine.answer(q)
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {len(result['sources'])} | Time: {result['total_time_ms']}ms")
        print("-" * 60 + "\n")


# import os
# from typing import List, Tuple, Optional, Dict, Any
# from dataclasses import dataclass

# import chromadb
# import ollama
# import numpy as np
# from sentence_transformers import SentenceTransformer, CrossEncoder
# from neo4j import GraphDatabase
# from rank_bm25 import BM25Okapi

# from src.logger import get_logger
# from src.config import get_config, AppConfig
# from src.utils import handle_errors

# # Suppress ChromaDB telemetryhy
# os.environ["CHROMA_TELEMETRY_IMPL"] = "none"
# os.environ["ANONYMIZED_TELEMETRY"] = "false"

# logger = get_logger(__name__)

# def refresh_index(self):
#     """Rebuild BM25 index after new documents are added."""
#     self._build_index()
#     logger.info("BM25 index refreshed")

# @dataclass
# class RetrievedChunk:
#     """Structured representation of a retrieved chunk."""
#     chunk_id: str
#     text: str
#     doc_id: str
#     effective_date: str
#     score: float
#     is_current: bool = True


# class HybridRetriever:
#     """Combines vector similarity and BM25 keyword matching."""
    
#     def __init__(self, collection: chromadb.Collection, embedding_model: SentenceTransformer):
#         self.collection = collection
#         self.embedder = embedding_model
#         self.documents: List[str] = []
#         self.ids: List[str] = []
#         self.bm25: Optional[BM25Okapi] = None
#         self._build_index()
    
#     @handle_errors(default_return=None)
#     def _build_index(self):
#         """Build BM25 index from all documents in ChromaDB."""
#         results = self.collection.get()
#         self.ids = results['ids']
#         self.documents = results['documents'] if results['documents'] else []
        
#         if self.documents:
#             tokenized = [doc.split() for doc in self.documents]
#             self.bm25 = BM25Okapi(tokenized)
#             logger.info(f"Built BM25 index with {len(self.documents)} documents")
    
#     def search(self, query: str, top_k: int = 10, alpha: float = 0.7) -> List[Tuple[str, float]]:
#         """Hybrid search combining vector and BM25 scores."""
#         if not self.bm25 or not self.documents:
#             logger.warning("BM25 index not available, falling back to vector search")
#             return self._vector_search_only(query, top_k)
        
#         query_embedding = self.embedder.encode(query, normalize_embeddings=True).tolist()
#         n_results = min(len(self.documents), max(top_k * 2, 10))
#         vector_results = self.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=n_results,
#             include=['documents', 'metadatas', 'distances']
#         )
        
#         if not vector_results or 'ids' not in vector_results or not vector_results['ids']:
#             return []
        
#         if not vector_results['ids'][0]:
#             return []
        
#         if 'distances' in vector_results and vector_results['distances']:
#             vector_distances = vector_results['distances'][0]
#             vector_scores = self._normalize([1.0 / (1.0 + d) for d in vector_distances])
#         else:
#             vector_scores = self._normalize([1.0 / (i + 1) for i in range(len(vector_results['ids'][0]))])
        
#         tokenized_query = query.split()
#         bm25_scores = self.bm25.get_scores(tokenized_query)
#         bm25_scores_norm = self._normalize(bm25_scores)
        
#         combined_scores = {}
#         id_to_index = {id_: idx for idx, id_ in enumerate(self.ids)}
        
#         for i, chunk_id in enumerate(vector_results['ids'][0]):
#             if chunk_id not in id_to_index:
#                 continue
#             bm25_idx = id_to_index[chunk_id]
#             combined = alpha * vector_scores[i] + (1 - alpha) * bm25_scores_norm[bm25_idx]
#             combined_scores[chunk_id] = combined
        
#         sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
#         return sorted_items[:top_k]
    
#     def _vector_search_only(self, query: str, top_k: int) -> List[Tuple[str, float]]:
#         """Fallback to pure vector search."""
#         query_embedding = self.embedder.encode(query, normalize_embeddings=True).tolist()
#         results = self.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=top_k,
#             include=['metadatas']
#         )
#         if not results or 'ids' not in results or not results['ids']:
#             return []
#         return [(id_, 1.0 / (i + 1)) for i, id_ in enumerate(results['ids'][0])]
    
#     def _normalize(self, scores: List[float]) -> List[float]:
#         """Normalize scores to [0, 1] range."""
#         import numpy as np
#         scores = np.array(scores)
#         min_s, max_s = scores.min(), scores.max()
#         if max_s == min_s:
#             return np.ones_like(scores) * 0.5
#         return ((scores - min_s) / (max_s - min_s)).tolist()


# class ReRanker:
#     """Cross-encoder based re-ranking for improved precision."""
    
#     def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
#         self.model = CrossEncoder(model_name)
#         logger.info(f"Loaded reranker: {model_name}")
    
#     @handle_errors(default_return=[])
#     def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
#         """
#         Re-rank documents using cross-encoder.
        
#         Returns:
#             List of (document_text, score) tuples sorted by relevance
#         """
#         if not documents:
#             return []
        
#         pairs = [(query, doc) for doc in documents]
#         scores = self.model.predict(pairs)
        
#         ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
#         return ranked[:top_k]


# class QueryEngine:
#     """Main query engine with graph-grounded temporal awareness."""
    
#     def __init__(self, config: Optional[AppConfig] = None):
#         self.config = config or get_config()
        
#         # Initialize components
#         self._init_vector_db()
#         self._init_graph_db()
#         self._init_models()
#         self._init_retrieval()
        
#         logger.info("QueryEngine initialized successfully")
    
#     def _init_vector_db(self):
#         """Initialize ChromaDB connection."""
#         self.chroma_client = chromadb.PersistentClient(
#             path=str(self.config.VECTORS_DIR)
#         )
#         self.collection = self.chroma_client.get_collection("legal_clauses")
#         logger.info(f"Connected to ChromaDB: {self.collection.count()} documents")
    
#     def _init_graph_db(self):
#         """Initialize Neo4j connection."""
#         if self.config.neo4j.is_valid():
#             try:
#                 self.neo4j_driver = GraphDatabase.driver(
#                     self.config.neo4j.uri,
#                     auth=(self.config.neo4j.user, self.config.neo4j.password)
#                 )
#                 self.neo4j_driver.verify_connectivity()
#                 logger.info(f"Connected to Neo4j: {self.config.neo4j.uri}")
#                 self.graph_enabled = True
#             except Exception as e:
#                 logger.warning(f"Neo4j connection failed: {e}. Graph features disabled.")
#                 self.neo4j_driver = None
#                 self.graph_enabled = False
#         else:
#             logger.warning("Neo4j credentials not configured. Graph features disabled.")
#             self.neo4j_driver = None
#             self.graph_enabled = False
    
#     def _init_models(self):
#         """Initialize embedding and LLM models."""
#         self.embedder = SentenceTransformer(self.config.embedding.model_name)
#         self.reranker = ReRanker()
#         logger.info(f"Loaded embedding model: {self.config.embedding.model_name}")
    
#     def _init_retrieval(self):
#         """Initialize hybrid retriever."""
#         self.retriever = HybridRetriever(self.collection, self.embedder)
    
#     @handle_errors(default_return=(None, None, None))
#     def _get_newest_version_graph(self, chunk_id: str) -> Tuple[str, Optional[str], Optional[str]]:
#         """
#         Find the most recent version of a clause using graph traversal.
#         Returns: (newest_id, newest_text, effective_date)
#         """
#         if not self.graph_enabled or not self.neo4j_driver:
#             return chunk_id, None, None
        
#         with self.neo4j_driver.session() as session:
#             query = """
#             MATCH (c:Clause {id: $chunk_id})
#             OPTIONAL MATCH path = (c)-[:SUPERSEDES*]->(newest:Clause)
#             WHERE NOT (newest)-[:SUPERSEDES]->()
#             RETURN COALESCE(newest, c) AS final_node
#             """
#             result = session.run(query, chunk_id=chunk_id).single()
#             if result:
#                 node = result['final_node']
#                 return node['id'], node['text'], str(node['effective_date'])
        
#         return chunk_id, None, None
    
#     def _get_chunk_text_from_vector(self, chunk_id: str) -> Tuple[Optional[str], Optional[str]]:
#         """Fallback: get chunk text from vector DB."""
#         results = self.collection.get(ids=[chunk_id])
#         if results and results['documents']:
#             text = results['documents'][0]
#             meta = results['metadatas'][0] if results['metadatas'] else {}
#             return text, meta.get('effective_date', 'unknown')
#         return None, None
    
#     def _deduplicate_chunks(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
#         """Remove duplicate chunks based on content similarity."""
#         seen_signatures = set()
#         unique_chunks = []
        
#         for chunk in chunks:
#             # Create signature from first 100 chars (normalized)
#             sig = ' '.join(chunk.text[:100].lower().split())
#             if sig not in seen_signatures:
#                 seen_signatures.add(sig)
#                 unique_chunks.append(chunk)
        
#         return unique_chunks
    
#     def answer(self, question: str, target_date: Optional[str] = None) -> Dict[str, Any]:
#         """
#         Answer a question using Graph-Grounded Temporal RAG.
        
#         Args:
#             question: User's question
#             target_date: Optional date for temporal context (YYYY-MM-DD)
        
#         Returns:
#             Dictionary with answer and metadata
#         """
#         logger.info(f"Processing question: {question[:100]}")
        
#         # 1. Hybrid retrieval
#         retrieved = self.retriever.search(
#             question,
#             top_k=self.config.retrieval.top_k,
#             alpha=self.config.retrieval.hybrid_alpha
#         )
        
#         if not retrieved:
#             return {
#                 "answer": "No relevant documents found to answer this question.",
#                 "sources": [],
#                 "graph_used": False
#             }
        
#         # 2. Get current versions and resolve temporal conflicts
#         chunks: List[RetrievedChunk] = []
        
#         for chunk_id, retrieval_score in retrieved:
#             # Get newest version via graph (or fallback)
#             if self.graph_enabled:
#                 newest_id, newest_text, newest_date = self._get_newest_version_graph(chunk_id)
#                 if newest_text is None:
#                     newest_text, newest_date = self._get_chunk_text_from_vector(newest_id)
#             else:
#                 newest_text, newest_date = self._get_chunk_text_from_vector(chunk_id)
#                 newest_id = chunk_id
            
#             if newest_text is None:
#                 continue
            
#             # Get metadata for doc_id
#             meta = self.collection.get(ids=[newest_id], include=['metadatas'])
#             doc_id = meta['metadatas'][0].get('doc_id', 'unknown') if meta['metadatas'] else 'unknown'
            
#             # Check if chunk is current relative to target_date
#             is_current = True
#             if target_date and newest_date and newest_date > target_date:
#                 is_current = False
            
#             chunks.append(RetrievedChunk(
#                 chunk_id=newest_id,
#                 text=newest_text,
#                 doc_id=doc_id,
#                 effective_date=newest_date or 'unknown',
#                 score=retrieval_score,
#                 is_current=is_current
#             ))
        
#         # 3. Deduplicate
#         unique_chunks = self._deduplicate_chunks(chunks)
        
#         # 4. Rerank with cross-encoder
#         if unique_chunks:
#             chunk_texts = [c.text for c in unique_chunks]
#             reranked = self.reranker.rerank(
#                 question,
#                 chunk_texts,
#                 top_k=self.config.retrieval.rerank_top_k
#             )
            
#             # Map back to chunk objects
#             text_to_chunk = {c.text: c for c in unique_chunks}
#             final_chunks = []
#             for text, score in reranked:
#                 chunk = text_to_chunk.get(text)
#                 if chunk:
#                     chunk.score = score
#                     final_chunks.append(chunk)
#         else:
#             final_chunks = unique_chunks
        
#         # 5. Build context and prompt
#         context_parts = []
#         sources = []
        
#         for chunk in final_chunks[:self.config.retrieval.rerank_top_k]:
#             status = "CURRENT" if chunk.is_current else "HISTORICAL"
#             context_parts.append(
#                 f"[{status}] {chunk.chunk_id} (Doc: {chunk.doc_id}, Effective: {chunk.effective_date}):\n"
#                 f"{chunk.text[:800]}\n"
#             )
#             sources.append({
#                 "chunk_id": chunk.chunk_id,
#                 "doc_id": chunk.doc_id,
#                 "effective_date": chunk.effective_date,
#                 "is_current": chunk.is_current,
#                 "score": chunk.score
#             })
        
#         context = "\n---\n".join(context_parts) if context_parts else "No context available."
        
#         # 6. Generate answer with Llama
#         prompt = f"""[INST] You are a precise legal assistant. Answer the question based ONLY on the provided context.
# If the context doesn't contain the answer, say "The answer is not found in the documents."

# Context:
# {context}

# Question: {question}

# Answer: [/INST]"""
        
#         response = ollama.generate(
#             model=self.config.ollama.model,
#             prompt=prompt,
#             options={
#                 "temperature": self.config.ollama.temperature,
#                 "num_predict": self.config.ollama.max_tokens
#             }
#         )
        
#         answer = response['response'].strip()
#         logger.info(f"Generated answer: {answer[:100]}...")
        
#         return {
#             "answer": answer,
#             "sources": sources,
#             "graph_used": self.graph_enabled,
#             "num_chunks_retrieved": len(final_chunks)
#         }


# # Convenience function for quick queries
# def query(question: str, target_date: Optional[str] = None) -> str:
#     """Quick query interface for testing."""
#     engine = QueryEngine()
#     result = engine.answer(question, target_date)
#     return result["answer"]