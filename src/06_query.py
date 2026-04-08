import chromadb
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import ollama
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
MODEL_NAME = "llama3.2:3b"

class QueryEngine:
    def __init__(self):
        # Vector DB
        self.chroma_client = chromadb.PersistentClient(path="data/vectors")
        self.collection = self.chroma_client.get_collection("legal_clauses")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        
        # Graph DB
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def get_newest_version(self, chunk_id):
        """Find the most recent version of a clause using graph traversal."""
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
                return node['id'], node['text'], str(node['effective_date'])
            return chunk_id, None, None
    
    def answer(self, question, target_date="2026-04-08"):
        # 1. Vector Search
        query_embedding = self.embedder.encode(question).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=10)
        
        # 2. Filter to active versions (Graph Grounding)
        seen_topics = set()
        context_pieces = []
        historical_pieces = []
        
        for chunk_id in results['ids'][0]:
            newest_id, newest_text, newest_date = self.get_newest_version(chunk_id)
            
            # Simple deduplication by first few words (topic)
            topic_key = newest_text[:50] if newest_text else ""
            if topic_key in seen_topics:
                continue
            seen_topics.add(topic_key)
            
            # Compare dates
            if newest_date and newest_date <= target_date:
                context_pieces.append(f"[CURRENT] Clause {newest_id} (Effective {newest_date}):\n{newest_text[:500]}\n")
            else:
                historical_pieces.append(f"[SUPERSEDED] Clause {newest_id} (Effective {newest_date}):\n{newest_text[:500]}\n")
        
        # 3. Build Prompt
        prompt = f"""[INST] You are a legal assistant. Today's date is {target_date}.
RULES:
1. Only use clauses with dates <= Today.
2. If a clause is marked [SUPERSEDED], ignore it. Use the [CURRENT] clause.

CONTEXT (CURRENT):
{''.join(context_pieces[:3])}

HISTORICAL (IGNORE UNLESS ASKED):
{''.join(historical_pieces[:2])}

Question: {question}
Answer: [/INST]"""
        
        # 4. Get LLM response
        response = ollama.generate(model=MODEL_NAME, prompt=prompt)
        return response['response']

if __name__ == "__main__":
    engine = QueryEngine()
    
    print("Legal Q&A (type 'exit' to quit)")
    while True:
        q = input("\nQuestion: ")
        if q.lower() == 'exit':
            break
        ans = engine.answer(q)
        print(f"\nAnswer: {ans}")