"""
Metrics and logging module for minimal RAG baseline.
Tracks memory-related metrics and performance for baseline comparison.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import tiktoken

from config import METRICS_OUTPUT, LOG_QUERIES


class MetricsLogger:
    """Logs metrics for each RAG query."""
    
    def __init__(self, output_path: str = None):
        """Initialize metrics logger."""
        self.output_path = output_path or METRICS_OUTPUT
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def log_query(
        self,
        query: str,
        retrieved_chunks: list,
        prompt: str,
        answer: str,
        latency: float,
        metadata: Optional[Dict] = None
    ):
        """
        Log metrics for a single query.
        
        Metrics tracked:
        - Query tokens
        - Retrieved context tokens
        - Total prompt tokens
        - Answer tokens
        - Latency
        - Answer length
        """
        # Count tokens
        query_tokens = self.count_tokens(query)
        context_tokens = sum(self.count_tokens(chunk["text"]) for chunk in retrieved_chunks)
        prompt_tokens = self.count_tokens(prompt)
        answer_tokens = self.count_tokens(answer)
        
        # Build metrics dict
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "query_tokens": query_tokens,
            "num_retrieved_chunks": len(retrieved_chunks),
            "context_tokens": context_tokens,
            "prompt_tokens": prompt_tokens,
            "answer_tokens": answer_tokens,
            "answer_length": len(answer),
            "latency_seconds": latency,
            "retrieved_sources": [
                {
                    "source_file": chunk.get("source_file", "unknown"),
                    "chunk_index": chunk.get("chunk_index", -1),
                    "similarity_score": chunk.get("similarity_score", 0.0)
                }
                for chunk in retrieved_chunks
            ]
        }
        
        # Add custom metadata if provided
        if metadata:
            metrics["metadata"] = metadata
        
        # Write to JSONL file
        if LOG_QUERIES:
            self._write_metrics(metrics)
        
        return metrics
    
    def _write_metrics(self, metrics: Dict):
        """Write metrics to JSONL file."""
        output_file = Path(self.output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
    
    def get_metrics_summary(self) -> Dict:
        """Read all metrics and return summary statistics."""
        output_file = Path(self.output_path)
        
        if not output_file.exists():
            return {"error": "No metrics file found"}
        
        metrics_list = []
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    metrics_list.append(json.loads(line))
        
        if not metrics_list:
            return {"error": "No metrics found"}
        
        # Calculate summary statistics
        total_queries = len(metrics_list)
        avg_latency = sum(m["latency_seconds"] for m in metrics_list) / total_queries
        avg_context_tokens = sum(m["context_tokens"] for m in metrics_list) / total_queries
        avg_prompt_tokens = sum(m["prompt_tokens"] for m in metrics_list) / total_queries
        avg_answer_tokens = sum(m["answer_tokens"] for m in metrics_list) / total_queries
        
        return {
            "total_queries": total_queries,
            "avg_latency_seconds": avg_latency,
            "avg_context_tokens": avg_context_tokens,
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_answer_tokens": avg_answer_tokens
        }

