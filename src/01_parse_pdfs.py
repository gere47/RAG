import os
import pandas as pd
import fitz  # PyMuPDF

def parse_pdfs():
    # Load manifest
    manifest = pd.read_csv('document_manifest.csv')
    
    # Ensure output folder exists
    os.makedirs('data/processed_texts', exist_ok=True)
    
    for _, row in manifest.iterrows():
        doc_id = row['doc_id']
        pdf_path = f"data/raw_pdfs/{doc_id}.pdf"
        
        # Check if PDF exists
        if not os.path.exists(pdf_path):
            print(f"WARNING: {pdf_path} not found. Skipping.")
            continue
        
        # Extract text using PyMuPDF
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        
        # Create output text with metadata header
        output_content = f"""[DOCUMENT METADATA]
DOC_ID: {doc_id}
EFFECTIVE_DATE: {row['effective_date']}
SUPERSEDES: {row['supersedes_doc_id'] if pd.notna(row['supersedes_doc_id']) else 'None'}
[END METADATA]

[TEXT CONTENT]
{full_text}
"""
        # Save to file
        output_path = f"data/processed_texts/{doc_id}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        print(f"Parsed: {doc_id} -> {output_path}")

if __name__ == "__main__":
    parse_pdfs()