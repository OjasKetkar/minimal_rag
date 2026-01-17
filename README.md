# Memory-Aware Agentic RAG under Real-world Constraints

A minimal RAG (Retrieval-Augmented Generation) system designed as a research baseline for studying memory behavior in RAG architectures.

## Purpose

Modern RAGs often assume that more retrieved contexts implies better answers. In reallity, this assumption fails due to the constraint on availability of RAM/VRAM for LLM inference, where only a finite amount of retrieved info can be loaded into the memory (KV cache). 
The project investigates how memory constraint affect RAG behabour and demonstrates how agentic control can be used to adapt retrieval strategies, rather than blindly increasing context size.

## Quick Reference: Baseline Parameters

| Parameter | Value |
|-----------|-------|
| **Chunk size** | 500 tokens |
| **Chunk overlap** | 50 tokens |
| **Top-K retrieval** | Dynamic (starts at 3, expands/contracts based on confidence) |
| **LLM Model** | allenai/molmo-2-8b:free |
| **Temperature** | 0.0 (deterministic generation) |
| **Max output tokens** | 500 |
| **Agentic Behaviour** | Confidence-driven retry and strategy adaptation |
| **Memory optimization** | Explicit short-term memory management (context-aware) |


## Key System Behaviours

- **Memory Pressure Simulation** : A fixed context token budget limits how much retrieved information can be passed to the LLM, modeling real GPU memory constraints.
Finding: Increasing available context does not monotonically improve answer quality.
- **Memory-Aware Selection** : Retrieved chunks are prioritized by semantic relevance. Under memory pressure, weaker chunks are deliberately discarded rather than accidentally truncated. Finding: Intentional memory allocation stabilizes outputs under large retrieval sizes.
- **Confidence Estimation** : Each answer is assigned a bounded confidence score derived from Evidence relevance (similarity scores), Context coverage, Stability under perturbation
This confidence reflects support under constraints, not objective truth.
- **Agentic Adaptation** : When confidence is low, the system : 

Modifies its retrieval strategy (e.g., dynamic K, query rewrite) --> Retries once --> Selects the result with stronger evidence --> Signals uncertainty if improvement is not possible
Finding: Agentic retries are bounded by semantic memory quality; more retrieval does not guarantee higher confidence.

## System Diagram

```
                ┌──────────────────────────┐
                │   Semantic Memory         │
                │  (FAISS Vector Store)     │
                │  Embedded Document Chunks │
                └───────────┬──────────────┘
                            │
                        Retrieval (Top-K)
                            │
                ┌───────────▼──────────────┐
                │   Memory Manager          │
                │  - Context token budget   │
                │  - Query vs context split │
                └───────────┬──────────────┘
                            │
                ┌───────────▼──────────────┐
                │  Chunk Prioritization     │
                │  - Rank by relevance      │
                │  - Drop weakest first     │
                └───────────┬──────────────┘
                            │
                ┌───────────▼──────────────┐
                │ Short-Term Memory         │
                │ (Final Prompt Context)    │
                └───────────┬──────────────┘
                            │
                ┌───────────▼──────────────┐
                │        LLM                │
                │  Deterministic (Temp=0)   │
                └───────────┬──────────────┘
                            │
                        Answer
                            │
                ┌───────────▼──────────────┐
                │ Confidence Estimation     │
                │ - Evidence strength       │
                │ - Context coverage        │
                │ - Stability               │
                └───────────┬──────────────┘
                            │
           ┌────────────────┴─────────────────┐
           │                                   │
  Confidence ≥ Threshold              Confidence < Threshold
           │                                   │
      Return Answer              Agentic Control Action
                                   (Retry / Dynamic-K /
                                    Query Rewrite / Warn)

```
