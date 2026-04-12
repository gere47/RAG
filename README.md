<div align="center">

# ⚖️ Graph-Grounded Temporal RAG

### *Contradiction-Resilient Question Answering over Evolving Legal Documents*

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge&logo=database&logoColor=white)](https://trychroma.com)
[![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](CONTRIBUTING.md)
[![Stars](https://img.shields.io/github/stars/gere47/RAG?style=social)](https://github.com/gere47/RAG)

---

<p align="center">
  <img src="https://raw.githubusercontent.com/gere47/RAG/main/assets/banner.png" alt="LexTemporal AI Banner" width="800"/>
</p>

<p align="center">
  <i>A production-grade legal intelligence platform that resolves contradictions across evolving documents using graph-grounded temporal reasoning.</i>
</p>

---

</div>

## 📖 Table of Contents

- [🎯 Overview](#-overview)
- [🏗️ Architecture](#️-architecture)
- [✨ Key Features](#-key-features)
- [🚀 Quick Start](#-quick-start)
- [📊 System Components](#-system-components)
- [🎮 Usage Guide](#-usage-guide)
- [🔬 Technical Deep Dive](#-technical-deep-dive)
- [📈 Performance Metrics](#-performance-metrics)
- [🛠️ Development](#️-development)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

**Graph-Grounded Temporal RAG** is a next-generation Retrieval-Augmented Generation system specifically designed for **evolving legal documents**. Unlike traditional RAG systems that treat all document chunks equally, this system:

- 🕸️ **Tracks Document Evolution** - Maintains a knowledge graph of `SUPERSEDES` relationships
- ⏳ **Resolves Temporal Contradictions** - Automatically identifies the *current active* version of any clause
- 🔍 **Hybrid Retrieval** - Combines vector similarity with BM25 keyword matching
- 🎯 **Cross-Encoder Reranking** - Ensures only the most relevant context reaches the LLM
- 🤖 **Local-First** - Runs entirely on your machine using free, open-source models

### 🎬 The Problem We Solve

| Traditional RAG | Graph-Grounded Temporal RAG |
|-----------------|----------------------------|
| ❌ Returns conflicting answers from old and new documents | ✅ Resolves contradictions via graph traversal |
| ❌ Treats all document versions equally | ✅ Understands temporal order and effective dates |
| ❌ Cannot answer "What was the rule in 2021?" | ✅ Answers temporal queries with historical context |
| ❌ No audit trail for answers | ✅ Full citation chain with version tracking |

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "📥 Ingestion Pipeline"
        A[PDF Documents] --> B[PyMuPDF Parser]
        B --> C[Section-Aware Chunker]
        C --> D[Metadata Extractor]
    end
    
    subgraph "🔍 Retrieval Layer"
        D --> E[Vector Embeddings<br/>all-MiniLM-L6-v2]
        D --> F[BM25 Index]
        E --> G[ChromaDB]
        F --> H[Hybrid Retriever]
    end
    
    subgraph "🕸️ Graph Layer"
        D --> I[Neo4j Graph]
        I --> J[SUPERSEDES Relationships]
        I --> K[Temporal Traversal]
    end
    
    subgraph "🎯 Reranking & Generation"
        H --> L[Cross-Encoder<br/>MS MARCO]
        K --> L
        L --> M[Context Assembly]
        M --> N[Llama 3.2<br/>via Ollama]
        N --> O[Citation-Enhanced Answer]
    end
    
    subgraph "🖥️ Interface"
        O --> P[Streamlit UI]
        P --> Q[REST API]
    end