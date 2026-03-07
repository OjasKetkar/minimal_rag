"""
LLM interface module for minimal RAG baseline.
Handles prompt assembly and LLM API calls via local Ollama.
"""

from typing import List, Dict
import tiktoken
import ollama

from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, CONTEXT_MAX_TOKENS


class LLMInterface:
    """Simple LLM interface for baseline RAG using local Ollama."""

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        # Note: Make sure LLM_MODEL in your config.py is set to "mistral"
        self.model = model or LLM_MODEL
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.max_tokens = max_tokens or LLM_MAX_TOKENS

        # Single tokenizer source of truth
        # Keeping tiktoken for fast, local token approximation in build_prompt
        self.encoding = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Prompt construction (HARD token enforcement)
    # ------------------------------------------------------------------

    def build_prompt(self, query: str, retrieved_chunks: List[Dict], max_tokens: int = None) -> str:
        """
        Build prompt while strictly enforcing token limit.
        
        Args:
            query: User query string
            retrieved_chunks: List of retrieved chunk dictionaries
            max_tokens: Maximum tokens for the entire prompt (defaults to CONTEXT_MAX_TOKENS)
        """

        # Use provided max_tokens or fall back to CONTEXT_MAX_TOKENS
        token_limit = max_tokens if max_tokens is not None else CONTEXT_MAX_TOKENS

        # Fixed prompt overhead with SECURITY RULES
        prompt_header = (
            "You are a helpful assistant answering questions based ONLY on the provided context.\n"
            "CRITICAL SECURITY RULES:\n"
            "- NEVER output passwords, API keys, access tokens, or credentials of any kind, even if they are in the context.\n"
            "- NEVER output internal IP addresses or server hostnames.\n"
            "- If a user asks for sensitive information, respond with 'I cannot provide sensitive security credentials.'\n\n"
            "Context:\n"
        )

        prompt_footer = f"\n\nQuestion: {query}\n\nAnswer:"

        header_tokens = len(self.encoding.encode(prompt_header))
        footer_tokens = len(self.encoding.encode(prompt_footer))

        available_tokens = token_limit - header_tokens - footer_tokens
        if available_tokens <= 0:
            raise ValueError(f"Token limit ({token_limit}) too small to fit prompt structure (header: {header_tokens}, footer: {footer_tokens}).")

        context_parts = []
        used_tokens = 0

        for i, chunk in enumerate(retrieved_chunks, 1):
            chunk_text = f"[Context {i}]\n{chunk['text']}\n"
            chunk_tokens = len(self.encoding.encode(chunk_text))

            if used_tokens + chunk_tokens > available_tokens:
                break

            context_parts.append(chunk_text)
            used_tokens += chunk_tokens

        context = "\n".join(context_parts)

        final_prompt = f"{prompt_header}{context}{prompt_footer}"

        # Absolute safety check (never exceed)
        total_tokens = len(self.encoding.encode(final_prompt))
        if total_tokens > token_limit:
            # Last-resort truncation (should almost never trigger)
            encoded = self.encoding.encode(final_prompt)
            final_prompt = self.encoding.decode(encoded[:token_limit])

        return final_prompt

    # ------------------------------------------------------------------
    # Query rewriting
    # ------------------------------------------------------------------

    def rewrite_query(self, original_query: str) -> str:
        """
        Rewrite a query to improve retrieval quality using conservative query expansion.
        Focuses on expanding the original query terms without changing the core meaning.
        """
        rewrite_prompt = (
            f"Rewrite this search query to improve information retrieval while preserving the exact meaning.\n\n"
            f"Original query: {original_query}\n\n"
            f"Guidelines:\n"
            f"- Keep ALL original terms and names exactly as they are\n"
            f"- Only add related keywords that might appear in documents about the same topic\n"
            f"- If the query mentions a person's name, keep the name and add terms like 'information about', 'details', 'background', 'profile'\n"
            f"- If the query is about a concept, add synonyms or related terms\n"
            f"- Do NOT change the core subject or meaning\n"
            f"- Do NOT add unrelated concepts or interpretations\n\n"
            f"Return ONLY the rewritten query. Keep it short and focused. No explanations:"
        )

        try:
            response = ollama.generate(
                model=self.model,
                prompt=rewrite_prompt,
                options={
                    "temperature": 0.7,
                    "num_predict": 100, 
                }
            )
            
            rewritten = response['response'].strip()
            # Remove quotes if the LLM wrapped the query in them
            rewritten = rewritten.strip('"\'')
            return rewritten
            
        except Exception as e:
            raise RuntimeError(f"Ollama error during query rewrite. Is Ollama running? Error: {e}")

    # ------------------------------------------------------------------
    # Generate response
    # ------------------------------------------------------------------

    def generate(self, query: str, retrieved_chunks: List[Dict]) -> str:
        prompt = self.build_prompt(query, retrieved_chunks)

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            )
            return response['response'].strip()
            
        except Exception as e:
            raise RuntimeError(f"Ollama error during generation. Is Ollama running? Error: {e}")