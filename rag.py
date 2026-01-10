"""
Main RAG orchestration module for minimal RAG baseline.
Coordinates all components: ingestion, embedding, retrieval, and LLM generation.
"""

import time
from pathlib import Path
from typing import List, Dict, Optional
import tiktoken

from config import DOCUMENTS_DIR, VECTOR_DB_PATH, CONTEXT_MAX_TOKENS
from ingest import ingest_documents
from embed import create_embeddings
from retrieve import VectorRetriever
from llm import LLMInterface
from metrics import MetricsLogger
from memory_manager import MemoryManager


class MinimalRAG:
    """
    Minimal RAG system - baseline implementation.
    No agents, no memory optimization, no multi-step reasoning.
    """
    
    def __init__(self, documents_dir: str = None, use_cache: bool = True):
        """
        Initialize RAG system.
        
        Args:
            documents_dir: Path to documents directory
            use_cache: Whether to use cached embeddings/index
        """
        self.documents_dir = documents_dir or DOCUMENTS_DIR
        self.use_cache = use_cache
        self.retriever = None
        self.llm = None
        self.metrics = MetricsLogger()
        
        # Initialize components
        self._initialize()
    
    def _initialize(self):
        """Initialize all RAG components."""
        # Initialize LLM
        self.llm = LLMInterface()
        
        # Try to load existing index
        index_path = Path(VECTOR_DB_PATH)
        if self.use_cache and (index_path / "index.faiss").exists():
            print("Loading existing vector index...")
            self.retriever = VectorRetriever()
            self.retriever.load_index()
            print("Index loaded successfully.")
        else:
            # Build index from scratch
            print("Building vector index from documents...")
            self._build_index()
    
    def _build_index(self):
        """Build vector index from documents."""
        # Ingest documents
        print(f"Ingesting documents from {self.documents_dir}...")
        chunks = ingest_documents(self.documents_dir)
        
        if not chunks:
            raise ValueError(f"No documents found in {self.documents_dir}")
        
        print(f"Created {len(chunks)} chunks from documents.")
        
        # Create embeddings
        print("Generating embeddings...")
        chunks_with_embeddings = create_embeddings(chunks, use_cache=self.use_cache)
        
        # Build retriever index
        print("Building FAISS index...")
        self.retriever = VectorRetriever()
        self.retriever.build_index(chunks_with_embeddings)
        
        # Save index
        if self.use_cache:
            self.retriever.save_index()
            print(f"Index saved to {VECTOR_DB_PATH}")
        
        print("Index built successfully.")
    
    def query(self, query: str, top_k: int = None) -> Dict:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: User query string
            top_k: Number of chunks to retrieve (defaults to config TOP_K)
        
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Initialize memory manager for token budget tracking
        memory_manager = MemoryManager(max_tokens=CONTEXT_MAX_TOKENS)
        encoding = tiktoken.get_encoding("cl100k_base")
        
        # Retrieve relevant chunks (tracks query tokens, but not context tokens yet)
        retrieved_chunks = self.retriever.retrieve(query, top_k, memory_manager=memory_manager, record_context_tokens=False)
        
        # Filter chunks based on remaining token budget
        # Only include chunks that fit within the budget
        filtered_chunks = []
        for chunk in retrieved_chunks:
            chunk_token_count = len(encoding.encode(chunk["text"]))
            remaining = memory_manager.remaining_budget()
            
            if chunk_token_count <= remaining:
                filtered_chunks.append(chunk)
                # Record context tokens only for chunks that fit within budget
                memory_manager.record(
                    "context",
                    chunk_token_count,
                    metadata={"chunk_id": chunk.get("chunk_index", -1), "score": chunk.get("similarity_score", 0.0)}
                )
            else:
                # Chunk doesn't fit in remaining budget - stop adding chunks
                break
        
        # Generate answer using only filtered chunks that fit within budget
        answer = self.llm.generate(query, filtered_chunks)
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Get prompt for metrics (using filtered chunks)
        prompt = self.llm.build_prompt(query, filtered_chunks)
        
        # Log metrics
        metrics = self.metrics.log_query(
            query=query,
            retrieved_chunks=filtered_chunks,
            prompt=prompt,
            answer=answer,
            latency=latency
        )
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": filtered_chunks,
            "metrics": metrics,
            "memory_snapshot": memory_manager.snapshot()
        }
    
    def get_metrics_summary(self) -> Dict:
        """Get summary statistics of all logged queries."""
        return self.metrics.get_metrics_summary()


def interactive_mode():
    """Interactive terminal mode for querying the RAG system."""
    print("Initializing RAG system...")
    rag = MinimalRAG()
    print("RAG system ready! Type your queries below (type 'exit' or 'quit' to stop).\n")
    
    while True:
        try:
            # Get user query
            query = input("Query: ").strip()
            
            # Check for exit commands
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
            # Skip empty queries
            if not query:
                continue
            
            # Process query
            print("\nProcessing...")
            result = rag.query(query)
            
            # Display answer
            print("\n" + "="*70)
            print("ANSWER:")
            print("="*70)
            print(result["answer"])
            print("="*70)
            
            # Display metrics
            metrics = result['metrics']
            print("\n" + "="*70)
            print("METRICS (logged to data/metrics.jsonl):")
            print("="*70)
            print(f"Query tokens:        {metrics['query_tokens']}")
            print(f"Retrieved chunks:    {metrics['num_retrieved_chunks']}")
            print(f"Context tokens:       {metrics['context_tokens']}")
            print(f"Prompt tokens:       {metrics['prompt_tokens']}")
            print(f"Answer tokens:       {metrics['answer_tokens']}")
            print(f"Total tokens:        {metrics['prompt_tokens'] + metrics['answer_tokens']}")
            print(f"Latency:             {metrics['latency_seconds']:.2f}s")
            print("="*70)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            continue


def main():
    """Example usage of MinimalRAG."""
    # Initialize RAG system
    rag = MinimalRAG()
    
    # Example query
    query = "What is the main topic of the documents?"
    result = rag.query(query)
    
    print("\n" + "="*50)
    print("QUERY:", result["query"])
    print("="*50)
    print("\nANSWER:")
    print(result["answer"])
    metrics = result['metrics']
    print("\n" + "="*50)
    print("METRICS (logged to data/metrics.jsonl):")
    print("="*50)
    print(f"Query tokens:        {metrics['query_tokens']}")
    print(f"Retrieved chunks:    {metrics['num_retrieved_chunks']}")
    print(f"Context tokens:      {metrics['context_tokens']}")
    print(f"Prompt tokens:       {metrics['prompt_tokens']}")
    print(f"Answer tokens:       {metrics['answer_tokens']}")
    print(f"Total tokens:        {metrics['prompt_tokens'] + metrics['answer_tokens']}")
    print(f"Latency:             {metrics['latency_seconds']:.2f}s")
    print("="*50)


if __name__ == "__main__":
    import sys
    
    # Check if interactive mode is requested
    if len(sys.argv) > 1 and sys.argv[1] in ['-i', '--interactive', 'interactive']:
        interactive_mode()
    else:
        # Default: run example or interactive mode
        interactive_mode()

