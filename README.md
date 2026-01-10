# Minimal RAG Baseline

A minimal RAG (Retrieval-Augmented Generation) system designed as a research baseline for studying memory behavior in RAG architectures.

## Purpose

This implementation serves as a baseline RAG system with static memory behavior, used for comparison against memory-aware agentic architectures in subsequent work.

**This is NOT a production chatbot.** This is an experimental baseline designed to:
- Establish a clean, naive RAG implementation
- Demonstrate understanding of RAG fundamentals
- Provide a control group for memory-aware system comparisons
- Enable analysis of memory behavior under static conditions

## Architecture

```
User Query
   ↓
Embed Query
   ↓
Vector Search (Top-K=5)
   ↓
Retrieve Chunks
   ↓
Prompt = Query + Retrieved Context
   ↓
LLM (Google Gemini)
   ↓
Answer
```

**No loops. No intelligence. No memory awareness.**

## Features

- ✅ Document ingestion (PDF, text files)
- ✅ Fixed-size chunking (500 tokens, 50 overlap)
- ✅ Vector embeddings (sentence-transformers)
- ✅ Top-K retrieval (hardcoded K=5)
- ✅ LLM generation (Google Gemini API)
- ✅ Metrics logging (tokens, latency, context size)

**Explicitly excluded:**
- ❌ Agents
- ❌ Tool calling
- ❌ Memory optimization
- ❌ Summarization
- ❌ Multi-step reasoning

## Setup

1. **Install Poetry** (if not already installed):
   ```bash
   # Windows (PowerShell)
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
   
   # macOS/Linux
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install dependencies:**
   ```bash
   poetry install
   ```

3. **Activate Poetry shell:**
   ```bash
   poetry shell
   ```

   Or run commands with Poetry:
   ```bash
   poetry run python rag.py
   ```

4. **Set up environment variables:**
   - Copy the example environment file:
     ```bash
     # Windows PowerShell
     Copy-Item .env.example .env
     
     # macOS/Linux
     cp .env.example .env
     ```
   - Edit `.env` and add your Gemini API key:
     ```
     GEMINI_API_KEY=your_actual_api_key_here
     ```
   - Get your free API key from https://aistudio.google.com/apikey
   
   **Note:** The `.env` file is already in `.gitignore`, so your API key won't be committed.

3. **Add documents:**
   - Place your documents (`.txt` or `.pdf` files) in `data/documents/`

## Usage

```python
from rag import MinimalRAG

# Initialize RAG system
rag = MinimalRAG()

# Query
result = rag.query("What is the main topic?")

print(result["answer"])
print(result["metrics"])
```

Or run the example:
```bash
poetry run python rag.py
```

## Configuration

Edit `config.py` to adjust:
- Chunk size and overlap
- Top-K retrieval count
- Embedding model
- LLM model (gemini-2.5-flash, gemini-2.5-pro, or gemini-2.0-flash-exp)
- Vector database settings

## Metrics

Metrics are logged to `data/metrics.jsonl` (JSON Lines format) and include:
- Query tokens
- Retrieved context tokens
- Total prompt tokens
- Answer tokens
- Latency
- Retrieved sources

View summary:
```python
rag.get_metrics_summary()
```

## Project Structure

```
minimal_rag/
├── data/
│   ├── documents/      # Input documents
│   ├── vector_db/      # FAISS index
│   ├── embeddings_cache.pkl
│   └── metrics.jsonl
├── ingest.py           # Document loading & chunking
├── embed.py            # Embedding generation
├── retrieve.py         # Vector search
├── llm.py              # LLM interface (Gemini)
├── rag.py              # Main orchestration
├── metrics.py          # Metrics logging
├── config.py           # Configuration
├── pyproject.toml      # Poetry configuration
├── requirements.txt    # Alternative: pip requirements
└── README.md
```

## Tech Stack

- **Language:** Python
- **LLM:** Google Gemini API (gemini-2.5-flash / gemini-2.5-pro)
- **Embeddings:** sentence-transformers
- **Vector DB:** FAISS
- **Chunking:** Fixed-size (intentionally naive)

## Research Context

This baseline system intentionally uses:
- Static chunking (no adaptive sizing)
- Fixed retrieval (no dynamic K adjustment)
- Naive prompt assembly (no optimization)
- No memory management

These design choices make memory weaknesses obvious, enabling clear comparison with memory-aware architectures in future work.
