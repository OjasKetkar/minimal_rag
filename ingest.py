"""
Document ingestion module for minimal RAG baseline.
Handles loading and naive chunking of documents.
"""

import os
from pathlib import Path
from typing import List, Dict
import tiktoken  # for token counting

from config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def load_text_file(file_path: Path) -> str:
    """Load text content from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_pdf_file(file_path: Path) -> str:
    """Load text content from a PDF file."""
    try:
        import PyPDF2
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except ImportError:
        raise ImportError("PyPDF2 is not installed. Please install it using 'pip install PyPDF2'.")

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def chunk_text(text: str, chunk_size: int, overlap: int, source_file: str) -> List[Dict]:
    """
    Naive fixed-size chunking with overlap.
    Returns list of chunk dictionaries.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    start_idx = 0
    chunk_index = 0
    
    while start_idx < len(tokens):
        end_idx = start_idx + chunk_size
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = encoding.decode(chunk_tokens)
        
        chunk_dict = {
            "text": chunk_text,
            "source_file": source_file,
            "chunk_index": chunk_index,
            "start_token": start_idx,
            "end_token": end_idx,
            "token_count": len(chunk_tokens)
        }
        chunks.append(chunk_dict)
        
        chunk_index += 1
        start_idx = end_idx - overlap  # Move forward with overlap
    
    return chunks


def ingest_documents(documents_dir: str = None) -> List[Dict]:
    """
    Load all documents from directory and chunk them.
    Returns list of all chunks from all documents.
    """
    if documents_dir is None:
        documents_dir = DOCUMENTS_DIR
    
    doc_path = Path(documents_dir)
    if not doc_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
    
    all_chunks = []
    
    # Process text files
    for text_file in doc_path.glob("*.txt"):
        text = load_text_file(text_file)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP, text_file.name)
        all_chunks.extend(chunks)
    
    # Process PDF files
    for pdf_file in doc_path.glob("*.pdf"):
        text = load_pdf_file(pdf_file)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP, pdf_file.name)
        all_chunks.extend(chunks)
    
    return all_chunks