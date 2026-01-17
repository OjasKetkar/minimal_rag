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

from config import TOP_K

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
    
    def _run_rag_core(self, query: str, top_k: int = None) -> Dict:
        """
        Core RAG execution logic.
        Returns answer, evidence_score, filtered_chunks, memory_manager, and other metadata.
        """
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
        
        return {
            "answer": answer,
            "evidence_score": evidence_score,
            "filtered_chunks": filtered_chunks,
            "memory_manager": memory_manager,
            "retrieved_chunks": retrieved_chunks
        }
    
    def query(self, query: str, top_k: int = None) -> Dict:
        """
        Process a query through the RAG pipeline with agentic decision-making.
        
        Args:
            query: User query string
            top_k: Number of chunks to retrieve (defaults to config TOP_K)
        
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Step 4: Agentic Decision-Making - Dynamic-K policy
        # Start with K=3, increase to K=6, then K=10 if confidence is low (max 2 retries)
        CONFIDENCE_THRESHOLD = 0.6
        
        # Dynamic-K retry policy: K=3 -> K=6 -> K=10
        k_sequence = [3, 6, 10]
        current_k_index = TOP_K
        
        # Run initial RAG with K=3
        initial_k = k_sequence[0]
        result = self._run_rag_core(query, initial_k)
        answer = result["answer"]
        evidence_score = result["evidence_score"]
        filtered_chunks = result["filtered_chunks"]
        memory_manager = result["memory_manager"]
        retrieved_chunks = result["retrieved_chunks"]
        used_query = query
        
        retry_info = None
        retry_count = 0
        max_retries = 2
        
        while evidence_score < CONFIDENCE_THRESHOLD and retry_count < max_retries:
            # Move to next K value
            current_k_index += 1
            if current_k_index >= len(k_sequence):
                break  # No more K values to try
            
            new_k = k_sequence[current_k_index]
            retry_count += 1
            
            print(f"⚠️ Low confidence detected ({evidence_score:.3f} < {CONFIDENCE_THRESHOLD}). Retrying with K={new_k}...")
            
            # Retry with increased K
            result_retry = self._run_rag_core(query, new_k)
            evidence_score_retry = result_retry["evidence_score"]
            
            print(f"   Retry {retry_count} confidence: {evidence_score_retry:.3f} (previous: {evidence_score:.3f})")
            
            # Choose the better result (higher confidence)
            if evidence_score_retry > evidence_score:
                improvement = evidence_score_retry - evidence_score
                print(f"   ✅ Improvement: +{improvement:.3f} (K={new_k})")
                answer = result_retry["answer"]
                evidence_score = evidence_score_retry
                filtered_chunks = result_retry["filtered_chunks"]
                memory_manager = result_retry["memory_manager"]
                retrieved_chunks = result_retry["retrieved_chunks"]
                
                retry_info = {
                    "strategy": "dynamic_k",
                    "original_k": initial_k,
                    "final_k": new_k,
                    "original_confidence": result["evidence_score"],
                    "retry_confidence": evidence_score_retry,
                    "improved": True,
                    "improvement": improvement,
                    "retry_count": retry_count
                }
            else:
                print(f"   ❌ No improvement with K={new_k}. Keeping previous result.")
                # Track that we tried but didn't improve
                if retry_info is None:
                    retry_info = {
                        "strategy": "dynamic_k",
                        "original_k": initial_k,
                        "final_k": new_k,
                        "original_confidence": result["evidence_score"],
                        "improved": False,
                        "improvement": evidence_score_retry - evidence_score,
                        "retry_count": retry_count
                    }
                else:
                    # Update with latest attempt
                    retry_info["final_k"] = new_k
                    retry_info["improvement"] = evidence_score_retry - evidence_score
                    retry_info["retry_count"] = retry_count
        
        # Update retry_info with final confidence if we did retries
        if retry_info:
            retry_info["final_confidence"] = evidence_score
            retry_info["retry_confidence"] = evidence_score  # Final confidence after all retries
        
        # Add warning if confidence is still low after retry
        if evidence_score < CONFIDENCE_THRESHOLD:
            warning = "\n\n⚠️ This answer may be incomplete due to limited or conflicting context."
            answer = answer + warning
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Get prompt for metrics (using filtered chunks)
        # Pass CONTEXT_MAX_TOKENS to ensure build_prompt enforces the same limit
        prompt = self.llm.build_prompt(used_query, filtered_chunks, max_tokens=CONTEXT_MAX_TOKENS)
        
        # Log metrics
        metrics = self.metrics.log_query(
            query=query,
            retrieved_chunks=filtered_chunks,
            prompt=prompt,
            answer=answer,
            latency=latency,
            metadata={"evidence_score": evidence_score}
        )
        
        result = {
            "query": query,
            "answer": answer,
            "retrieved_chunks": filtered_chunks,
            "metrics": metrics,
            "memory_snapshot": memory_manager.snapshot(),
            "evidence_score": evidence_score
        }
        
        if retry_info:
            result["retry_info"] = retry_info
        
        return result
    
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
            retry_info = result.get('retry_info')
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
            if retry_info:
                print(f"\n🔄 AGENTIC RETRY (Dynamic-K):")
                print(f"   Strategy:            {retry_info.get('strategy', 'dynamic_k')}")
                print(f"   Original K:          {retry_info.get('original_k', 'N/A')}")
                print(f"   Final K:             {retry_info.get('final_k', 'N/A')}")
                print(f"   Retries:             {retry_info.get('retry_count', 0)}")
                print(f"   Original confidence: {retry_info['original_confidence']:.3f}")
                print(f"   Final confidence:    {retry_info.get('final_confidence', retry_info['retry_confidence']):.3f}")
                if retry_info['improved']:
                    print(f"   ✅ Improved by:      +{retry_info['improvement']:.3f}")
                else:
                    print(f"   ❌ No improvement:    {retry_info['improvement']:.3f}")
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

