# Memory-Aware Agentic RAG under Real-world Constraints

A minimal RAG (Retrieval-Augmented Generation) system designed as a research baseline for studying memory behavior in RAG architectures.

## Purpose

Modern RAGs often assume that more retrieved contexts imply better answers, and that internal environments are implicitly safe. In reality, these assumptions fail due to two major constraints:

Memory Bounds: A finite amount of retrieved info can be loaded into the LLM's KV cache.

Security & Privacy Bounds: Exposing raw vector databases to LLMs introduces critical Data Exfiltration and Prompt Injection vulnerabilities.

This project investigates how memory constraints affect RAG behavior and demonstrates how agentic control and a Defense-in-Depth Security Perimeter can be used to adapt retrieval strategies while mathematically guaranteeing data privacy.

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
| **Inbound Security** | Hybrid WAF (DeBERTa-v3 Semantic ML + Deterministic Regex) |
| **Data Privacy** | In-memory Data Vault (Regex PII Masking & Tokenization) |
| **Egress Security** | Zero-Trust RBAC Rehydration |


## Key System Behaviours

- **Memory Pressure Simulation** : A fixed context token budget limits how much retrieved information can be passed to the LLM, modeling real GPU memory constraints.
Finding: Increasing available context does not monotonically improve answer quality
- **Memory-Aware Selection** : Raw database chunks are intercepted before hitting the LLM. Secrets (AWS keys, IPs) are replaced with synthetic tokens (e.g., <IP_ADDRESS_A1B2>).
Finding: LLMs maintain semantic reasoning capabilities even when operating purely on tokenized synthetic data.
- **Confidence Estimation** : Each answer is assigned a bounded confidence score derived from Evidence relevance (similarity scores), Context coverage, Stability under perturbation
This confidence reflects support under constraints, not objective truth.
- **Agentic Adaptation** : When confidence is low, the system modifies its retrieval strategy (e.g., dynamic K). The confidence score derives from Evidence relevance, Context coverage, and Stability.
- **Bi-Directional Security Perimeter** :
    - Inbound: A Hybrid ML/Regex shield blocks persona hijacks (e.g., "I am the CEO") and prompt injections before database retrieval.
    - Outbound: An RBAC (Role-Based Access Control) gate ensures synthetic tokens are only "rehydrated" into real secrets if the requesting user possesses cryptographic Admin authority.

## System Diagram
```
    ┌──────────────────────────┐
    User Query ────►│  Hybrid Inbound Shield   │──[MALICIOUS]──► DROP & ALERT
                    │ (Semantic ML + Regex)    │
                    └───────────┬──────────────┘
                                │ [SAFE]
                    ┌───────────▼──────────────┐
                    │   Semantic Memory        │
                    │  (FAISS Vector Store)    │
                    └───────────┬──────────────┘
                                │ (Raw Chunks)
                    ┌───────────▼──────────────┐
                    │      Data Vault          │
                    │ - Strip PII & Secrets    │
                    │ - Inject <SYNTH_TOKENS>  │
                    └───────────┬──────────────┘
                                │ (Sanitized Chunks)
                    ┌───────────▼──────────────┐
                    │   Memory Manager &       │
                    │ Chunk Prioritization     │
                    └───────────┬──────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │ Short-Term Memory        │
                    │ (System Prompt Hardened) │
                    └───────────┬──────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │        LLM               │
                    │  Deterministic (Temp=0)  │
                    └───────────┬──────────────┘
                                │ (Raw Answer with Tokens)
                    ┌───────────▼──────────────┐
                    │ Confidence Estimation    │
                    └───────────┬──────────────┘
                                │
               ┌────────────────┴─────────────────┐
               │                                  │
      Confidence ≥ Threshold             Confidence < Threshold
               │                                  │
    ┌──────────▼─────────────────┐     Agentic Control Action
    │ Zero-Trust Rehydration     │      (Retry / Dynamic-K /
    │ (RBAC Evaluation)          │       Query Rewrite)
    └──────────┬─────────────────┘
               │
          ┌────┴────┐
       [ADMIN]   [STANDARD]
          │         │
      Unmask      Keep Tokens &
      Secrets     Append Warning
          │         │
          ▼         ▼
        Final User Output
```