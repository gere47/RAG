import json
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd

load_dotenv()

class LegalGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared.")
    
    def create_clause_node(self, tx, chunk_id, doc_id, effective_date, text):
        query = """
        MERGE (c:Clause {id: $chunk_id})
        SET c.doc_id = $doc_id,
            c.effective_date = date($effective_date),
            c.text = $text
        RETURN c
        """
        tx.run(query, chunk_id=chunk_id, doc_id=doc_id, 
               effective_date=effective_date, text=text)
    
    def create_supersedes_relationship(self, tx, doc_id, supersedes_doc_id):
        # This creates edges between ALL clauses of newer doc and older doc.
        # In a real scenario you might match specific clauses, but for demo this works.
        if pd.isna(supersedes_doc_id) or supersedes_doc_id == 'None':
            return
        query = """
        MATCH (new:Clause {doc_id: $doc_id})
        MATCH (old:Clause {doc_id: $supersedes})
        MERGE (new)-[:SUPERSEDES]->(old)
        """
        tx.run(query, doc_id=doc_id, supersedes=supersedes_doc_id)

def build_graph():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    
    graph = LegalGraph(uri, user, password)
    graph.clear_database()  # Start fresh
    
    # Load clauses
    with open('data/chunks/clauses.json', 'r') as f:
        clauses = json.load(f)
    
    with graph.driver.session() as session:
        # Create clause nodes
        for clause in clauses:
            session.execute_write(
                graph.create_clause_node,
                clause['chunk_id'],
                clause['doc_id'],
                clause['effective_date'],
                clause['text']
            )
        print(f"Created {len(clauses)} Clause nodes.")
        
        # Create SUPERSEDES relationships based on manifest
        manifest = pd.read_csv('document_manifest.csv')
        for _, row in manifest.iterrows():
            if pd.notna(row['supersedes_doc_id']):
                session.execute_write(
                    graph.create_supersedes_relationship,
                    row['doc_id'],
                    row['supersedes_doc_id']
                )
        print("Created SUPERSEDES relationships.")
    
    graph.close()
    print("Graph build complete.")

if __name__ == "__main__":
    build_graph()