import os
import re
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


def chunk_text_by_size(text, max_chars=1200, overlap=200):
    """
    Splits text into overlapping chunks of approximately max_chars.
    Tries to break at paragraph boundaries.
    """
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If adding this paragraph exceeds max_chars, finalise current chunk
        if len(current_chunk) + len(para) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep last `overlap` characters for continuity
            if len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                current_chunk = para + "\n\n"
        else:
            current_chunk += para + "\n\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def chunk_documents():
    manifest = pd.read_csv('document_manifest.csv')
    os.makedirs('data/chunks', exist_ok=True)
    
    all_chunks = []
    
    for _, row in manifest.iterrows():
        doc_id = row['doc_id']
        txt_path = f"data/processed_texts/{doc_id}.txt"
        
        if not os.path.exists(txt_path):
            print(f"Missing: {txt_path}")
            continue
            
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract effective_date from metadata header
        meta_match = re.search(r'EFFECTIVE_DATE:\s*(\S+)', content)
        effective_date = meta_match.group(1) if meta_match else row['effective_date']
        
        # Extract text after [END METADATA]
        parts = content.split('[END METADATA]')
        if len(parts) > 1:
            text_part = parts[1]
        else:
            text_part = content
        
        text_part = re.sub(r'\[TEXT CONTENT\]', '', text_part).strip()
        
        # Generate chunks
        chunks = chunk_text_by_size(text_part, max_chars=1200, overlap=200)
        
        for i, chunk in enumerate(chunks):
            chunk_entry = {
                "chunk_id": f"{doc_id}_chunk_{i+1:03d}",
                "doc_id": doc_id,
                "effective_date": effective_date,
                "text": chunk
            }
            all_chunks.append(chunk_entry)
    
    output_path = "data/chunks/clauses.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
    
    print(f"Created {len(all_chunks)} chunks from {len(manifest)} documents.")
    print(f"Saved to {output_path}")

def chunk_all_documents(manifest, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(chunk_single_document, row) for _, row in manifest.iterrows()]
        results = [f.result() for f in futures]
    return [chunk for doc_chunks in results for chunk in doc_chunks]

if __name__ == "__main__":
    chunk_documents()