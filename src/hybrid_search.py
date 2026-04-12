from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

class HybridRetriever:
    def __init__(self, chroma_collection, embedding_model):
        self.collection = chroma_collection
        self.embedder = embedding_model
        self.bm25 = None
        self.documents = []
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from all documents in ChromaDB."""
        results = self.collection.get()
        self.documents = results['documents']
        tokenized = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query: str, top_k: int = 5, alpha: float = 0.7):
        """
        Hybrid search combining vector and BM25.
        alpha: weight for vector search (0-1). Higher = more vector influence.
        """
        # Vector search
        query_embedding = self.embedder.encode(query).tolist()
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=len(self.documents)
        )
        
        # BM25 search
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize and combine
        vector_scores = self._normalize([1 / (i+1) for i in range(len(vector_results['ids'][0]))])
        bm25_scores_norm = self._normalize(bm25_scores)
        
        combined_scores = {}
        for i, chunk_id in enumerate(vector_results['ids'][0]):
            combined_scores[chunk_id] = alpha * vector_scores[i] + (1-alpha) * bm25_scores_norm[i]
        
        # Sort and return top_k
        sorted_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]
        return sorted_ids
    
    def _normalize(self, scores):
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [0.5] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]