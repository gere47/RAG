import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.query_engine import QueryEngine

e = QueryEngine()

# Test what's retrieved
question = "main points of the document"
retrieved = e.retriever.search(question, top_k=5)

print(f"\nQuestion: {question}")
print(f"Retrieved {len(retrieved)} chunks:\n")

for i, (chunk_id, score) in enumerate(retrieved, 1):
    chunk = e.retriever.get_chunk_by_id(chunk_id)
    if chunk:
        doc_id = chunk['metadata'].get('doc_id', 'unknown')
        text_preview = chunk['text'][:150].replace('\n', ' ')
        print(f"{i}. {chunk_id} ({doc_id}) [score: {score:.3f}]")
        print(f"   {text_preview}...\n")