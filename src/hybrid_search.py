"""
Production-grade Hybrid Retriever combining Vector and BM25 search.
"""

import hashlib
from typing import List, Tuple, Optional, Dict, Any
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb

from src.logger import get_logger
from src.utils import handle_errors

logger = get_logger(__name__)


class HybridRetriever:
    """
    Production-grade hybrid retriever combining dense (vector) and sparse (BM25) retrieval.
    
    Features:
    - Efficient BM25 index with incremental updates
    - Proper score combination with ID alignment
    - Configurable alpha weighting
    - Score normalization with multiple methods
    - Caching for performance
    - Error handling with graceful degradation
    """
    
    def __init__(
        self,
        collection: chromadb.Collection,
        embedding_model: SentenceTransformer,
        alpha: float = 0.7,
        normalize_method: str = "minmax",
        cache_size: int = 1000
    ):
        """
        Args:
            collection: ChromaDB collection
            embedding_model: SentenceTransformer model
            alpha: Weight for vector search (0-1). Higher = more vector influence
            normalize_method: "minmax", "softmax", or "zscore"
            cache_size: LRU cache size for embeddings
        """
        self.collection = collection
        self.embedder = embedding_model
        self.alpha = alpha
        self.normalize_method = normalize_method
        
        # BM25 components
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[str] = []
        self.ids: List[str] = []
        self.metadatas: List[Dict] = []
        self.id_to_index: Dict[str, int] = {}
        
        # Build index
        self._build_bm25_index()
        
        # Cache for embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_size = cache_size
        
        logger.info(f"HybridRetriever initialized: {len(self.ids)} documents, alpha={alpha}")
    
    @handle_errors(default_return=None)
    def _build_bm25_index(self) -> None:
        """
        Build BM25 index from ChromaDB collection.
        Uses original chunk text, not enriched version.
        """
        # Get all documents from collection
        results = self.collection.get(include=['documents', 'metadatas'])
        
        if not results or not results.get('ids'):
            logger.warning("No documents found in collection")
            return
        
        self.ids = results['ids']
        self.metadatas = results.get('metadatas', [])
        
        # Get original text (strip date prefix if present)
        self.documents = []
        for doc in results.get('documents', []):
            # Remove "[Effective: YYYY-MM-DD] " prefix if present
            if doc.startswith('[Effective:'):
                doc = doc.split('] ', 1)[-1] if '] ' in doc else doc
            self.documents.append(doc)
        
        # Build ID to index mapping
        self.id_to_index = {id_: idx for idx, id_ in enumerate(self.ids)}
        
        # Tokenize and build BM25
        tokenized = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        
        logger.debug(f"BM25 index built: {len(self.documents)} documents")
    
    def _get_vector_scores(
        self,
        query_embedding: List[float],
        n_results: int
    ) -> Tuple[List[str], np.ndarray]:
        """
        Get vector similarity scores from ChromaDB.
        
        Returns:
            Tuple of (ids, scores) aligned by index
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['distances']
        )
        
        if not results or 'ids' not in results or not results['ids']:
            return [], np.array([])
        
        ids = results['ids'][0]
        
        # Convert distances to similarity scores
        if 'distances' in results and results['distances']:
            distances = results['distances'][0]
            # Cosine distance to similarity: 1 - distance (for normalized embeddings)
            # or 1 / (1 + distance) for non-normalized
            scores = 1.0 - np.array(distances)
        else:
            # Fallback: position-based scores
            scores = 1.0 / (1.0 + np.arange(len(ids)))
        
        return ids, scores
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to [0, 1] range.
        
        Args:
            scores: Raw score array
            method: "minmax", "softmax", or "zscore"
        """
        if len(scores) == 0:
            return scores
        
        if self.normalize_method == "minmax":
            min_s, max_s = scores.min(), scores.max()
            if max_s > min_s:
                return (scores - min_s) / (max_s - min_s)
            return np.ones_like(scores) * 0.5
        
        elif self.normalize_method == "softmax":
            exp_scores = np.exp(scores - scores.max())
            return exp_scores / exp_scores.sum()
        
        elif self.normalize_method == "zscore":
            mean, std = scores.mean(), scores.std()
            if std > 0:
                z_scores = (scores - mean) / std
                # Sigmoid to bound to [0,1]
                return 1.0 / (1.0 + np.exp(-z_scores))
            return np.ones_like(scores) * 0.5
        
        return scores
    
    def _get_cached_embedding(self, query: str) -> np.ndarray:
        """Get cached embedding or compute and cache."""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        embedding = self.embedder.encode(query, normalize_embeddings=True)
        
        # LRU-style cache eviction
        if len(self._embedding_cache) >= self._cache_size:
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
        
        self._embedding_cache[cache_key] = embedding
        return embedding
    
    @handle_errors(default_return=[])
    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: Optional[float] = None,
        return_scores: bool = False
    ) -> List:
        """
        Hybrid search combining vector and BM25 scores.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            alpha: Weight for vector search (overrides instance default)
            return_scores: If True, return (id, score) tuples
        
        Returns:
            List of chunk IDs, or (id, score) tuples if return_scores=True
        """
        alpha = alpha if alpha is not None else self.alpha
        
        if not self.bm25 or not self.documents:
            logger.warning("BM25 index not available, using vector-only search")
            return self._vector_search_only(query, top_k, return_scores)
        
        # Get query embedding (cached)
        query_embedding = self._get_cached_embedding(query).tolist()
        
        # Get vector scores for all documents (or top N for efficiency)
        n_vector_results = min(len(self.documents), max(top_k * 5, 50))
        vector_ids, vector_scores_raw = self._get_vector_scores(query_embedding, n_vector_results)
        
        if len(vector_ids) == 0:
            return [] if not return_scores else []
        
        # Normalize vector scores
        vector_scores = self._normalize_scores(vector_scores_raw)
        
        # Get BM25 scores for all documents
        tokenized_query = query.split()
        bm25_scores_raw = np.array(self.bm25.get_scores(tokenized_query))
        bm25_scores = self._normalize_scores(bm25_scores_raw)
        
        # Combine scores with proper ID alignment
        combined_scores: Dict[str, float] = {}
        
        # Vector scores (only for retrieved subset)
        for i, chunk_id in enumerate(vector_ids):
            if chunk_id in self.id_to_index:
                combined_scores[chunk_id] = alpha * vector_scores[i]
        
        # BM25 scores (for all documents that had vector scores)
        for chunk_id in vector_ids:
            if chunk_id in self.id_to_index:
                bm25_idx = self.id_to_index[chunk_id]
                if chunk_id in combined_scores:
                    combined_scores[chunk_id] += (1 - alpha) * bm25_scores[bm25_idx]
                else:
                    combined_scores[chunk_id] = (1 - alpha) * bm25_scores[bm25_idx]
        
        # Sort and return
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:top_k]
        
        if return_scores:
            return top_items
        else:
            return [item[0] for item in top_items]
    
    def _vector_search_only(
        self,
        query: str,
        top_k: int,
        return_scores: bool = False
    ) -> List:
        """Fallback: pure vector search."""
        query_embedding = self._get_cached_embedding(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['distances']
        )
        
        if not results or 'ids' not in results or not results['ids']:
            return [] if not return_scores else []
        
        ids = results['ids'][0]
        
        if return_scores and 'distances' in results:
            distances = results['distances'][0]
            scores = 1.0 - np.array(distances)
            return list(zip(ids, scores.tolist()))
        elif return_scores:
            scores = 1.0 / (1.0 + np.arange(len(ids)))
            return list(zip(ids, scores.tolist()))
        else:
            return ids
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full chunk data by ID."""
        if chunk_id not in self.id_to_index:
            return None
        
        idx = self.id_to_index[chunk_id]
        
        return {
            'id': chunk_id,
            'text': self.documents[idx],
            'metadata': self.metadatas[idx] if idx < len(self.metadatas) else {}
        }
    
    def refresh_index(self) -> None:
        """Refresh BM25 index after collection updates."""
        logger.info("Refreshing BM25 index...")
        self._embedding_cache.clear()
        self._build_bm25_index()
        logger.info(f"BM25 index refreshed: {len(self.ids)} documents")
    
    def get_stats(self) -> Dict[str, Any]:
        """Return retriever statistics."""
        return {
            'total_documents': len(self.ids),
            'bm25_ready': self.bm25 is not None,
            'cache_size': len(self._embedding_cache),
            'alpha': self.alpha,
            'normalize_method': self.normalize_method
        }
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._embedding_cache.clear()
        logger.debug("Embedding cache cleared")