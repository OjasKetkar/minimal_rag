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
        
        # Sort chunks by similarity score (descending) - highest similarity first
        # This ensures we prioritize the most relevant chunks when memory is limited
        retrieved_chunks.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        
        # Calculate prompt overhead tokens (header + footer with query)
        # This matches what build_prompt will use
        prompt_header = (
            "Answer the following question using the provided context.\n"
            "If the context does not contain enough information, say so.\n\n"
            "Context:\n"
        )
        prompt_footer = f"\n\nQuestion: {query}\n\nAnswer:"
        header_tokens = len(encoding.encode(prompt_header))
        footer_tokens = len(encoding.encode(prompt_footer))
        prompt_overhead = header_tokens + footer_tokens
        
        # Available tokens for context chunks = total budget - prompt overhead
        # Note: query tokens are already recorded in memory_manager, but footer includes query
        # So we use CONTEXT_MAX_TOKENS directly and subtract overhead
        available_for_chunks = CONTEXT_MAX_TOKENS - prompt_overhead
        if available_for_chunks <= 0:
            raise ValueError(f"CONTEXT_MAX_TOKENS ({CONTEXT_MAX_TOKENS}) too small for prompt structure (overhead: {prompt_overhead})")
        
        # Filter chunks based on remaining token budget
        # Account for formatting tokens that build_prompt will add: "[Context {i}]\n" and "\n"
        filtered_chunks = []
        used_context_tokens = 0
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            # Count tokens including the formatting that build_prompt will add
            chunk_text_with_formatting = f"[Context {i}]\n{chunk['text']}\n"
            chunk_token_count = len(encoding.encode(chunk_text_with_formatting))
            
            if used_context_tokens + chunk_token_count <= available_for_chunks:
                filtered_chunks.append(chunk)
                used_context_tokens += chunk_token_count
                # Record context tokens (just the chunk text, not formatting, for memory_manager tracking)
                chunk_text_only_tokens = len(encoding.encode(chunk["text"]))
                memory_manager.record(
                    "context",
                    chunk_text_only_tokens,
                    metadata={"chunk_id": chunk.get("chunk_index", -1), "score": chunk.get("similarity_score", 0.0)}
                )
            else:
                # Chunk doesn't fit in remaining budget - stop adding chunks
                # Lower similarity chunks are dropped first
                break
        
        # Calculate evidence strength: average similarity score of used chunks
        # Normalize to 0-1 range (FAISS cosine similarity is already 0-1 for normalized vectors)
        # Calculate evidence strength from retrieved chunks BEFORE budget filtering
        if retrieved_chunks:
            similarity_scores = [
                chunk.get("similarity_score", 0.0)
                for chunk in retrieved_chunks
                if chunk.get("similarity_score") is not None
            ]

            # Evidence = strongest grounding signal
            evidence_score = max(similarity_scores) if similarity_scores else 0.0
        else:
            evidence_score = 0.0

        # Clamp for safety
        evidence_score = max(0.0, min(1.0, evidence_score))

        
        # Generate answer using only filtered chunks that fit within budget
        answer = self.llm.generate(query, filtered_chunks)
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Get prompt for metrics (using filtered chunks)
        # Pass CONTEXT_MAX_TOKENS to ensure build_prompt enforces the same limit
        prompt = self.llm.build_prompt(query, filtered_chunks, max_tokens=CONTEXT_MAX_TOKENS)
        
        # Log metrics
        metrics = self.metrics.log_query(
            query=query,
            retrieved_chunks=filtered_chunks,
            prompt=prompt,
            answer=answer,
            latency=latency,
            metadata={"evidence_score": evidence_score}
        )
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": filtered_chunks,
            "metrics": metrics,
            "memory_snapshot": memory_manager.snapshot(),
            "evidence_score": evidence_score
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
            evidence_score = result.get('evidence_score', 0.0)
            print("\n" + "="*70)
            print("METRICS (logged to data/metrics.jsonl):")
            print("="*70)
            print(f"Query tokens:        {metrics['query_tokens']}")
            print(f"Retrieved chunks:    {metrics['num_retrieved_chunks']}")
            print(f"Context tokens:       {metrics['context_tokens']}")
            print(f"Prompt tokens:       {metrics['prompt_tokens']}")
            print(f"Answer tokens:       {metrics['answer_tokens']}")
            print(f"Total tokens:        {metrics['prompt_tokens'] + metrics['answer_tokens']}")
            print(f"Evidence score:      {evidence_score:.3f}")
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
    evidence_score = result.get('evidence_score', 0.0)
    print("\n" + "="*50)
    print("METRICS (logged to data/metrics.jsonl):")
    print("="*50)
    print(f"Query tokens:        {metrics['query_tokens']}")
    print(f"Retrieved chunks:    {metrics['num_retrieved_chunks']}")
    print(f"Context tokens:      {metrics['context_tokens']}")
    print(f"Prompt tokens:       {metrics['prompt_tokens']}")
    print(f"Answer tokens:       {metrics['answer_tokens']}")
    print(f"Total tokens:        {metrics['prompt_tokens'] + metrics['answer_tokens']}")
    print(f"Evidence score:      {evidence_score:.3f}")
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

