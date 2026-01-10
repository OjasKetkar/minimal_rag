"""
Vector retrieval module for minimal RAG baseline.
Performs naive top-K similarity search using FAISS.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import faiss
import tiktoken

from config import TOP_K, SIMILARITY_METRIC, VECTOR_DB_PATH, EMBEDDING_DIM
from embed import Embedder


class VectorRetriever:
    """Simple vector retriever using FAISS."""
    
    def __init__(self, embedding_dim: int = None):
        """Initialize retriever with embedding dimension."""
        if embedding_dim is None:
            embedding_dim = EMBEDDING_DIM
        
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []  # Store chunks in same order as index
        self.embedder = Embedder()
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def build_index(self, chunks_with_embeddings: List[Dict]):
        """
        Build FAISS index from chunks with embeddings.
        Stores chunks for later retrieval.
        """
        if not chunks_with_embeddings:
            raise ValueError("No chunks provided for indexing")
        
        # Extract embeddings as numpy array
        embeddings = np.array([chunk["embedding"] for chunk in chunks_with_embeddings])
        embeddings = embeddings.astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index (L2 normalized = cosine similarity)
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product on normalized = cosine
        self.index.add(embeddings)
        
        # Store chunks in same order
        self.chunks = chunks_with_embeddings.copy()
    
    def retrieve(self, query: str, top_k: int = None, memory_manager=None, record_context_tokens: bool = True) -> List[Dict]:
        """
        Retrieve top-K most similar chunks for a query.
        Returns chunks with similarity scores.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            memory_manager: Optional MemoryManager instance for tracking memory usage
            record_context_tokens: Whether to record context tokens for retrieved chunks (default: True)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        if top_k is None:
            top_k = TOP_K
        
        # Embed query
        query_embedding = self.embedder.embed_text(query)
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Track query tokens
        if memory_manager is not None:
            query_token_count = len(self.encoding.encode(query))
            memory_manager.record("query", query_token_count)
        
        # Search
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve chunks with similarity scores
        retrieved = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                similarity = float(similarities[0][i])
                chunk["similarity_score"] = similarity
                retrieved.append(chunk)
                
                # Track context tokens for each retrieved chunk (if enabled)
                if memory_manager is not None and record_context_tokens:
                    chunk_token_count = len(self.encoding.encode(chunk["text"]))
                    memory_manager.record(
                        "context",
                        chunk_token_count,
                        metadata={"chunk_id": idx, "score": similarity}
                    )
        
        return retrieved
    
    def save_index(self, index_path: str = None):
        """Save FAISS index and chunks to disk."""
        if index_path is None:
            index_path = VECTOR_DB_PATH
        
        index_dir = Path(index_path)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_dir / "index.faiss"))
        
        # Save chunks metadata
        with open(index_dir / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
    
    def load_index(self, index_path: str = None):
        """Load FAISS index and chunks from disk."""
        if index_path is None:
            index_path = VECTOR_DB_PATH
        
        index_dir = Path(index_path)
        
        if not (index_dir / "index.faiss").exists():
            raise FileNotFoundError(f"Index not found at {index_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        
        # Load chunks metadata
        with open(index_dir / "chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)


def retrieve_chunks(query: str, retriever: VectorRetriever, top_k: int = None, memory_manager=None) -> List[Dict]:
    """
    Convenience function to retrieve chunks for a query.
    
    Args:
        query: Query string
        retriever: VectorRetriever instance
        top_k: Number of chunks to retrieve
        memory_manager: Optional MemoryManager instance for tracking memory usage
    """
    return retriever.retrieve(query, top_k, memory_manager)