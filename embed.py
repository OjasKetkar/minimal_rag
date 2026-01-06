"""
Embedding generation module for minimal RAG baseline.
Converts text chunks into vector embeddings using a static embedding model.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from config import EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDINGS_CACHE


class Embedder:
    """Simple embedding generator with caching."""
    
    def __init__(self, model_name: str = None):
        """Initialize embedding model."""
        if model_name is None:
            model_name = EMBEDDING_MODEL
        
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: pip install sentence-transformers"
            )
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text string."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for a list of chunks.
        Returns chunks with 'embedding' field added.
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        # Add embeddings to chunk dictionaries
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]
        
        return chunks
    
    def save_embeddings(self, chunks_with_embeddings: List[Dict], cache_path: str = None):
        """Save embeddings to disk for caching."""
        if cache_path is None:
            cache_path = EMBEDDINGS_CACHE
        
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(chunks_with_embeddings, f)
    
    def load_embeddings(self, cache_path: str = None) -> Optional[List[Dict]]:
        """Load cached embeddings from disk."""
        if cache_path is None:
            cache_path = EMBEDDINGS_CACHE
        
        cache_file = Path(cache_path)
        if not cache_file.exists():
            return None
        
        with open(cache_file, 'rb') as f:
            return pickle.load(f)


def create_embeddings(chunks: List[Dict], use_cache: bool = True) -> List[Dict]:
    """
    Main function to create embeddings for chunks.
    Checks cache first if use_cache=True.
    """
    embedder = Embedder()
    
    # Check cache
    if use_cache:
        cached = embedder.load_embeddings()
        if cached is not None:
            return cached
    
    # Generate embeddings
    chunks_with_embeddings = embedder.embed_chunks(chunks)
    
    # Save to cache
    if use_cache:
        embedder.save_embeddings(chunks_with_embeddings)
    
    return chunks_with_embeddings