"""
Central configuration for the entire project.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_TEXTS_DIR = DATA_DIR / "processed_texts"
CHUNKS_DIR = DATA_DIR / "chunks"
EXTRACTED_DIR = DATA_DIR / "extracted"
VECTORS_DIR = DATA_DIR / "vectors"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for dir_path in [RAW_PDFS_DIR, PROCESSED_TEXTS_DIR, CHUNKS_DIR, 
                 EXTRACTED_DIR, VECTORS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""
    uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    
    def is_valid(self) -> bool:
        return bool(self.uri and self.user and self.password)


@dataclass
class OllamaConfig:
    """Ollama LLM configuration."""
    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
    host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    temperature: float = 0.1
    max_tokens: int = 512


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 100


@dataclass
class ChunkingConfig:
    """Document chunking configuration."""
    max_chars: int = 1200
    overlap_chars: int = 200
    min_chunk_chars: int = 100


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    top_k: int = 10
    rerank_top_k: int = 3
    hybrid_alpha: float = 0.7  # Weight for vector vs BM25 (higher = more vector)
    similarity_threshold: float = 0.5


@dataclass
class AppConfig:
    """Master configuration."""
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    
    # Paths as class attributes
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = DATA_DIR
    RAW_PDFS_DIR: Path = RAW_PDFS_DIR
    PROCESSED_TEXTS_DIR: Path = PROCESSED_TEXTS_DIR
    CHUNKS_DIR: Path = CHUNKS_DIR
    EXTRACTED_DIR: Path = EXTRACTED_DIR
    VECTORS_DIR: Path = VECTORS_DIR
    LOGS_DIR: Path = LOGS_DIR
    
    # File paths
    manifest_path: Path = PROJECT_ROOT / "document_manifest.csv"
    clauses_json_path: Path = CHUNKS_DIR / "clauses.json"
    
    @classmethod
    def load(cls) -> "AppConfig":
        return cls()


# Singleton config instance
config = AppConfig.load()


def get_config() -> AppConfig:
    """Return the global configuration instance."""
    return config