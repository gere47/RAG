import json
import chromadb
from sentence_transformers import SentenceTransformer
import os
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Use a local embedding model (free)
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

def create_index():
    # Initialize ChromaDB (persistent storage)
    client = chromadb.PersistentClient(path="data/vectors")
    collection = client.get_or_create_collection(name="legal_clauses")
    
    # Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Load clauses
    with open('data/chunks/clauses.json', 'r') as f:
        clauses = json.load(f)
    
    # Prepare data for Chroma
    ids = []
    documents = []
    metadatas = []
    
    for clause in clauses:
        ids.append(clause['chunk_id'])
        # Enrich text with date for temporal awareness in vector search
        enriched_text = f"Effective {clause['effective_date']}: {clause['text']}"
        documents.append(enriched_text)
        metadatas.append({
            "doc_id": clause['doc_id'],
            "effective_date": clause['effective_date'],
            "chunk_id": clause['chunk_id']
        })
    
    # Compute embeddings and add in batches (to avoid memory issues)
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        
        embeddings = model.encode(batch_docs).tolist()
        
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=embeddings
        )
        print(f"Indexed batch {i//batch_size + 1}")
    
    print(f"Indexed {len(ids)} clauses in ChromaDB.")

if __name__ == "__main__":
    create_index()