import os
import re
import json
import pandas as pd

def chunk_by_sections(text):
    # Legal docs often have "Section X.Y" or "ARTICLE X"
    # This splits while keeping the header attached
    pattern = r'(?=\n\s*(?:Section|ARTICLE)\s+\d+\.?\d*)'
    chunks = re.split(pattern, text)
    
    # First element might be preamble, we keep it if substantial
    result = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) > 50:  # Ignore very short fragments
            result.append(chunk)
    return result

def chunk_documents():
    manifest = pd.read_csv('document_manifest.csv')
    os.makedirs('data/chunks', exist_ok=True)
    
    all_clauses = []
    
    for _, row in manifest.iterrows():
        doc_id = row['doc_id']
        txt_path = f"data/processed_texts/{doc_id}.txt"
        
        if not os.path.exists(txt_path):
            continue
            
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract metadata (already in text but we need it separate)
        meta_match = re.search(r'\[DOCUMENT METADATA\](.*?)\[END METADATA\]', content, re.DOTALL)
        if meta_match:
            meta_text = meta_match.group(1)
            effective_date = re.search(r'EFFECTIVE_DATE:\s*(\S+)', meta_text).group(1)
        else:
            effective_date = row['effective_date']
        
        # Get only text content (after [END METADATA])
        text_part = content.split('[END METADATA]')[-1]
        # Remove the [TEXT CONTENT] marker if present
        text_part = re.sub(r'\[TEXT CONTENT\]', '', text_part).strip()
        
        chunks = chunk_by_sections(text_part)
        
        for i, chunk in enumerate(chunks):
            # Try to extract section number
            sec_match = re.search(r'(?:Section|ARTICLE)\s+([\d\.]+)', chunk)
            section_num = sec_match.group(1) if sec_match else f"unknown_{i}"
            
            clause_entry = {
                "chunk_id": f"{doc_id}_sec_{section_num}",
                "doc_id": doc_id,
                "effective_date": effective_date,
                "text": chunk
            }
            all_clauses.append(clause_entry)
    
    # Save to JSON
    output_path = "data/chunks/clauses.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_clauses, f, indent=2)
    
    print(f"Created {len(all_clauses)} chunks. Saved to {output_path}")

if __name__ == "__main__":
    chunk_documents()