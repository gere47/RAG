"""
Utility functions used across the project.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
import traceback
from datetime import datetime

from src.logger import get_logger

logger = get_logger(__name__)


def handle_errors(default_return: Any = None, reraise: bool = False):
    """
    Decorator to catch and log exceptions gracefully.
    
    Args:
        default_return: Value to return on error
        reraise: If True, re-raise exception after logging
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                logger.debug(traceback.format_exc())
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


def safe_file_read(filepath: Path, encodings: List[str] = None) -> Optional[str]:
    """
    Read file with automatic encoding detection.
    
    Args:
        filepath: Path to file
        encodings: List of encodings to try (default: common encodings)
    
    Returns:
        File contents as string, or None if all encodings fail
    """
    if encodings is None:
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            return None
    
    logger.error(f"Could not decode {filepath} with any encoding: {encodings}")
    return None


def safe_json_load(filepath: Path, default: Any = None) -> Any:
    """Safely load JSON file with error handling."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {filepath}")
        return default
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        return default


def safe_json_dump(data: Any, filepath: Path, indent: int = 2) -> bool:
    """Safely save data as JSON."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        return False


def ensure_directory(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


def validate_manifest(manifest_path: Path) -> bool:
    """
    Validate document_manifest.csv structure.
    
    Required columns: doc_id, doc_title, effective_date, supersedes_doc_id
    """
    import pandas as pd
    
    try:
        df = pd.read_csv(manifest_path)
        required_cols = {'doc_id', 'doc_title', 'effective_date', 'supersedes_doc_id'}
        
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            logger.error(f"Manifest missing columns: {missing}")
            return False
        
        # Validate date format
        date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        invalid_dates = df[~df['effective_date'].astype(str).str.match(date_pattern)]
        if not invalid_dates.empty:
            logger.error(f"Invalid date format in manifest: {invalid_dates['doc_id'].tolist()}")
            return False
        
        logger.info(f"Manifest validated: {len(df)} documents")
        return True
        
    except Exception as e:
        logger.error(f"Failed to validate manifest: {e}")
        return False


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start = None
    
    def __enter__(self):
        self.start = datetime.now()
        return self
    
    def __exit__(self, *args):
        elapsed = (datetime.now() - self.start).total_seconds()
        logger.info(f"{self.name} completed in {elapsed:.2f}s")