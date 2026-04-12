# Graph-Grounded Temporal RAG for Legal Documents

## Overview
Production‑grade RAG system that resolves contradictions across evolving legal documents using Neo4j graph relationships and temporal filtering.

## Features
- ✅ Local LLM (Llama 3.2) – free and private
- ✅ Hybrid search (Vector + BM25)
- ✅ Cross‑encoder re‑ranking
- ✅ Graph‑based contradiction resolution
- ✅ Streamlit web interface
- ✅ Comprehensive error handling and logging

## Quick Start
1. `pip install -r requirements.txt`
2. Start Ollama: `ollama serve`
3. Run pipeline: `python run_pipeline.py`
4. Launch UI: `streamlit run app.py`

## Architecture
[Diagram or description of pipeline phases]