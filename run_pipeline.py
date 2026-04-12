#!/usr/bin/env python
"""
One-click pipeline to rebuild the entire RAG system from scratch.
Usage: python run_pipeline.py [--skip-parsing] [--skip-graph]
"""

import sys
import argparse
import subprocess
from pathlib import Path

from src.logger import get_logger, ProgressTracker
from src.utils import validate_manifest, ensure_directory
from src.config import config

logger = get_logger(__name__)


def run_script(script_path: str, description: str) -> bool:
    """Run a Python script and return success status."""
    logger.info(f"Running: {description}")
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        cwd=config.PROJECT_ROOT
    )
    
    if result.returncode != 0:
        logger.error(f"Failed: {description} (exit code {result.returncode})")
        return False
    
    logger.info(f"Completed: {description}")
    return True


def clean_directories():
    """Clean output directories for fresh run."""
    import shutil
    
    dirs_to_clean = [
        config.PROCESSED_TEXTS_DIR,
        config.CHUNKS_DIR,
        config.EXTRACTED_DIR,
        config.VECTORS_DIR
    ]
    
    for dir_path in dirs_to_clean:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True)
            logger.info(f"Cleaned: {dir_path}")


def main():
    parser = argparse.ArgumentParser(description="Rebuild Graph-Grounded Temporal RAG")
    parser.add_argument("--skip-parsing", action="store_true", help="Skip PDF parsing")
    parser.add_argument("--skip-graph", action="store_true", help="Skip Neo4j graph build")
    parser.add_argument("--clean", action="store_true", help="Clean all output before running")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Graph-Grounded Temporal RAG - Pipeline Runner")
    logger.info("=" * 60)
    
    # Validate manifest
    if not validate_manifest(config.manifest_path):
        logger.error("Invalid manifest. Please fix document_manifest.csv")
        return 1
    
    # Clean if requested
    if args.clean:
        clean_directories()
    
    phases = []
    
    if not args.skip_parsing:
        phases.append(("src/01_parse_pdfs.py", "Phase 1: Parse PDFs"))
    
    phases.extend([
        ("src/02_chunk_documents.py", "Phase 2: Chunk Documents"),
        ("src/05_create_index.py", "Phase 3: Build Vector Index"),
    ])
    
    if not args.skip_graph:
        phases.append(("src/04_build_graph.py", "Phase 4: Build Neo4j Graph"))
    
    # Run all phases
    with ProgressTracker(logger, "Pipeline Execution", len(phases)) as tracker:
        for script, description in phases:
            if not run_script(script, description):
                logger.error(f"Pipeline failed at: {description}")
                return 1
            tracker.update(1)
    
    logger.info("=" * 60)
    logger.info("✅ Pipeline completed successfully!")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("  1. Run query: python src/06_query.py")
    logger.info("  2. Or launch UI: streamlit run app.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())