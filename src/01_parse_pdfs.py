"""
Phase 1: Production-Grade PDF Parser
Extracts text with metadata preservation, OCR fallback, and validation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import re
import json
import hashlib
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, asdict

import pandas as pd
import fitz  # PyMuPDF

from src.logger import get_logger
from src.config import config
from src.utils import handle_errors, safe_file_read, safe_json_dump, Timer

logger = get_logger(__name__)


@dataclass
class ParsedDocument:
    """Structured representation of a parsed document."""
    doc_id: str
    doc_title: str
    effective_date: str
    supersedes_doc_id: Optional[str]
    source_path: str
    output_path: str
    page_count: int
    char_count: int
    word_count: int
    file_hash: str
    parsed_at: str
    status: str
    error_message: Optional[str] = None


class PDFParser:
    """Production-grade PDF parser with validation and error recovery."""
    
    def __init__(self):
        self.manifest_path = config.manifest_path
        self.raw_pdfs_dir = config.RAW_PDFS_DIR
        self.processed_dir = config.PROCESSED_TEXTS_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    @handle_errors(default_return=None)
    def extract_text_with_metadata(self, pdf_path: Path) -> Tuple[str, Dict]:
        """
        Extract text and metadata from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, metadata_dict)
        """
        doc = fitz.open(pdf_path)
        
        metadata = {
            "page_count": len(doc),
            "pdf_title": doc.metadata.get("title", ""),
            "pdf_author": doc.metadata.get("author", ""),
            "pdf_subject": doc.metadata.get("subject", ""),
            "pdf_creator": doc.metadata.get("creator", ""),
            "pdf_producer": doc.metadata.get("producer", ""),
            "pdf_creation_date": doc.metadata.get("creationDate", ""),
        }
        
        full_text = []
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text()
            if page_text.strip():
                full_text.append(f"[PAGE {page_num}]\n{page_text}")
        
        doc.close()
        
        combined_text = "\n\n".join(full_text)
        
        # Check if PDF is scanned (no text extracted)
        if len(combined_text.strip()) < 100:
            logger.warning(f"Possible scanned PDF detected: {pdf_path.name}")
            combined_text = self._attempt_ocr_fallback(pdf_path)
        
        return combined_text, metadata
    
    def _attempt_ocr_fallback(self, pdf_path: Path) -> str:
        """
        Attempt OCR for scanned PDFs.
        Requires pytesseract and pdf2image installed.
        """
        try:
            from pdf2image import convert_from_path
            import pytesseract
            
            logger.info(f"Attempting OCR on {pdf_path.name}...")
            images = convert_from_path(pdf_path, dpi=150)
            
            ocr_text = []
            for i, image in enumerate(images, 1):
                text = pytesseract.image_to_string(image)
                ocr_text.append(f"[PAGE {i} - OCR]\n{text}")
            
            return "\n\n".join(ocr_text)
            
        except ImportError:
            logger.warning("OCR dependencies not installed. Install: pdf2image, pytesseract")
            return "[SCANNED PDF - OCR NOT AVAILABLE]"
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return "[SCANNED PDF - OCR FAILED]"
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file for integrity verification."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _count_words(self, text: str) -> int:
        """Count words in extracted text."""
        return len(text.split())
    
    def _create_metadata_header(self, doc_id: str, effective_date: str, 
                                 supersedes: Optional[str], pdf_metadata: Dict) -> str:
        """Create structured metadata header for processed text file."""
        header = f"""[DOCUMENT METADATA]
DOC_ID: {doc_id}
EFFECTIVE_DATE: {effective_date}
SUPERSEDES: {supersedes if supersedes and pd.notna(supersedes) else 'None'}
PAGE_COUNT: {pdf_metadata.get('page_count', 0)}
PDF_TITLE: {pdf_metadata.get('pdf_title', 'N/A')}
PDF_AUTHOR: {pdf_metadata.get('pdf_author', 'N/A')}
PROCESSED_AT: {datetime.now().isoformat()}
[END METADATA]

"""
        return header
    
    @handle_errors(default_return=None)
    def parse_single_document(self, doc_id: str, doc_title: str, 
                               effective_date: str, supersedes: Optional[str]) -> Optional[ParsedDocument]:
        """
        Parse a single PDF document.
        
        Args:
            doc_id: Document identifier
            doc_title: Human-readable title
            effective_date: Effective date (YYYY-MM-DD)
            supersedes: ID of document this supersedes (optional)
            
        Returns:
            ParsedDocument object or None on failure
        """
        pdf_path = self.raw_pdfs_dir / f"{doc_id}.pdf"
        
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return None
        
        logger.info(f"Parsing: {doc_id} - {doc_title}")
        
        with Timer(f"Parse {doc_id}"):
            # Extract text and metadata
            extracted_text, pdf_metadata = self.extract_text_with_metadata(pdf_path)
            
            if not extracted_text:
                logger.error(f"No text extracted from {doc_id}")
                return None
            
            # Compute metrics
            char_count = len(extracted_text)
            word_count = self._count_words(extracted_text)
            file_hash = self._compute_file_hash(pdf_path)
            
            # Create header
            header = self._create_metadata_header(doc_id, effective_date, supersedes, pdf_metadata)
            
            # Combine and save
            output_content = header + "[TEXT CONTENT]\n\n" + extracted_text
            output_path = self.processed_dir / f"{doc_id}.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_content)
            
            logger.info(f"Parsed {doc_id}: {pdf_metadata['page_count']} pages, "
                       f"{char_count:,} chars, {word_count:,} words")
            
            return ParsedDocument(
                doc_id=doc_id,
                doc_title=doc_title,
                effective_date=effective_date,
                supersedes_doc_id=supersedes if pd.notna(supersedes) else None,
                source_path=str(pdf_path),
                output_path=str(output_path),
                page_count=pdf_metadata['page_count'],
                char_count=char_count,
                word_count=word_count,
                file_hash=file_hash,
                parsed_at=datetime.now().isoformat(),
                status="success"
            )
    
    def parse_all_documents(self) -> List[ParsedDocument]:
        """
        Parse all documents defined in manifest.
        
        Returns:
            List of successfully parsed documents
        """
        if not self.manifest_path.exists():
            logger.error(f"Manifest not found: {self.manifest_path}")
            return []
        
        manifest = pd.read_csv(self.manifest_path)
        logger.info(f"Found {len(manifest)} documents in manifest")
        
        parsed_docs = []
        failed_docs = []
        
        for _, row in manifest.iterrows():
            doc_id = row['doc_id']
            doc_title = row['doc_title']
            effective_date = row['effective_date']
            supersedes = row.get('supersedes_doc_id', None)
            
            result = self.parse_single_document(doc_id, doc_title, effective_date, supersedes)
            
            if result:
                parsed_docs.append(result)
            else:
                failed_docs.append({
                    'doc_id': doc_id,
                    'doc_title': doc_title,
                    'error': 'Parsing failed'
                })
        
        # Save parsing report
        report = {
            'parsed_at': datetime.now().isoformat(),
            'total_documents': len(manifest),
            'successful': len(parsed_docs),
            'failed': len(failed_docs),
            'parsed_documents': [asdict(d) for d in parsed_docs],
            'failed_documents': failed_docs
        }
        
        report_path = self.processed_dir / "parsing_report.json"
        safe_json_dump(report, report_path)
        
        logger.info(f"Parsing complete: {len(parsed_docs)}/{len(manifest)} successful")
        
        if failed_docs:
            logger.warning(f"Failed documents: {[d['doc_id'] for d in failed_docs]}")
        
        return parsed_docs


def main():
    """Main entry point for PDF parsing phase."""
    logger.info("=" * 60)
    logger.info("Phase 1: Production PDF Parser")
    logger.info("=" * 60)
    
    parser = PDFParser()
    parsed_docs = parser.parse_all_documents()
    
    if parsed_docs:
        logger.info(f"✅ Successfully parsed {len(parsed_docs)} documents")
        logger.info(f"📁 Output directory: {config.PROCESSED_TEXTS_DIR}")
        logger.info(f"📊 Report: {config.PROCESSED_TEXTS_DIR}/parsing_report.json")
    else:
        logger.error("❌ No documents were parsed successfully")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


# import os
# import pandas as pd
# import fitz  # PyMuPDF

# def parse_pdfs():
#     # Load manifest
#     manifest = pd.read_csv('document_manifest.csv')
    
#     # Ensure output folder exists
#     os.makedirs('data/processed_texts', exist_ok=True)
    
#     for _, row in manifest.iterrows():
#         doc_id = row['doc_id']
#         pdf_path = f"data/raw_pdfs/{doc_id}.pdf"
        
#         # Check if PDF exists
#         if not os.path.exists(pdf_path):
#             print(f"WARNING: {pdf_path} not found. Skipping.")
#             continue
        
#         # Extract text using PyMuPDF
#         doc = fitz.open(pdf_path)
#         full_text = ""
#         for page in doc:
#             full_text += page.get_text()
#         doc.close()
        
#         # Create output text with metadata header
#         output_content = f"""[DOCUMENT METADATA]
# DOC_ID: {doc_id}
# EFFECTIVE_DATE: {row['effective_date']}
# SUPERSEDES: {row['supersedes_doc_id'] if pd.notna(row['supersedes_doc_id']) else 'None'}
# [END METADATA]

# [TEXT CONTENT]
# {full_text}
# """
#         # Save to file
#         output_path = f"data/processed_texts/{doc_id}.txt"
#         with open(output_path, 'w', encoding='utf-8') as f:
#             f.write(output_content)
        
#         print(f"Parsed: {doc_id} -> {output_path}")

# if __name__ == "__main__":
#     parse_pdfs()