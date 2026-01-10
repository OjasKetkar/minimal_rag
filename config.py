"""
Configuration settings for minimal RAG baseline system.
All parameters are intentionally static and naive for baseline comparison.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Document Processing
CHUNK_SIZE = 500  # tokens per chunk
CHUNK_OVERLAP = 50  # tokens overlap between chunks

# Retrieval Settings
TOP_K = 5  # number of chunks to retrieve (hardcoded, no dynamic adjustment)
SIMILARITY_METRIC = "cosine"  # fixed similarity metric

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # lightweight, proven
EMBEDDING_DIM = 384  # dimension for the chosen model

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # Default: gemini
# Common Gemini models: gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash-exp
# Check https://ai.google.dev/models/gemini or run: poetry run python list_models.py
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")  # Can override via .env
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))  # deterministic for baseline
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))  # limit response length

# Memory Management
CONTEXT_MAX_TOKENS = int(os.getenv("CONTEXT_MAX_TOKENS", "2000"))  # max tokens for input context

# Vector Database
VECTOR_DB_TYPE = "faiss"  # or "chroma"
VECTOR_DB_PATH = "./data/vector_db"  # where to store/load vectors

# Data Paths
DOCUMENTS_DIR = "./data/documents"  # input documents location
EMBEDDINGS_CACHE = "./data/embeddings_cache.pkl"  # cache embeddings

# Metrics & Logging
LOG_QUERIES = True
METRICS_OUTPUT = "./data/metrics.jsonl"  # one metric per line (JSON Lines format)