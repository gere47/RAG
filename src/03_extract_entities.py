import json
import ollama
import pandas as pd
from tqdm import tqdm
import time

MODEL_NAME = "llama3.2:3b"  # or "llama3.1:8b" if you have RAM

def extract_from_chunk(chunk_text, chunk_id, effective_date):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a legal data extractor. Output ONLY valid JSON. No extra text. Extract the following fields from the clause. If missing, use null.

Fields:
- clause_number: The section number (e.g., "2.1")
- subject_party: Party with OBLIGATION
- action: What they must do or pay
- amount: Specific number/percentage
- object_party: Party receiving benefit

Output Format: {{"clause_number": "...", "subject_party": "...", "action": "...", "amount": "...", "object_party": "..."}}
<|eot_id|><|start_header_id|>user<|end_header_id|>
TEXT: {chunk_text[:1500]} 
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    try:
        response = ollama.generate(model=MODEL_NAME, prompt=prompt)
        result_text = response['response'].strip()
        
        # Clean up any markdown code fences
        if result_text.startswith('```json'):
            result_text = result_text[7:]
        if result_text.endswith('```'):
            result_text = result_text[:-3]
        
        data = json.loads(result_text)
        # Add metadata
        data['chunk_id'] = chunk_id
        data['effective_date'] = effective_date
        return data
    except Exception as e:
        print(f"Error on {chunk_id}: {e}")
        return {
            "chunk_id": chunk_id,
            "effective_date": effective_date,
            "clause_number": None,
            "subject_party": None,
            "action": None,
            "amount": None,
            "object_party": None,
            "error": str(e)
        }

def main():
    # Load chunks
    with open('data/chunks/clauses.json', 'r') as f:
        clauses = json.load(f)
    
    # For testing, limit to 10 chunks first
    # Remove this line to run on all chunks (will take time)
    # clauses = clauses[:10]  # <-- UNCOMMENT FOR TEST RUN
    
    extracted_data = []
    for clause in tqdm(clauses, desc="Extracting entities"):
        result = extract_from_chunk(
            clause['text'], 
            clause['chunk_id'], 
            clause['effective_date']
        )
        extracted_data.append(result)
        time.sleep(0.5)  # Be nice to Ollama
    
    # Save results
    os.makedirs('data/extracted', exist_ok=True)
    output_path = 'data/extracted/entities.jsonl'
    with open(output_path, 'w') as f:
        for entry in extracted_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    import os
    main()