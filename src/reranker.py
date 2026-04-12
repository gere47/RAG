from sentence_transformers import CrossEncoder

class ReRanker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: list, top_k: int = 3):
        """Re‑rank documents using cross‑encoder."""
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        
        # Sort by score descending
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]