"""
LLM interface module for minimal RAG baseline.
Handles prompt assembly and LLM API calls via OpenRouter.
"""

import os
import json
import requests
from typing import List, Dict

import tiktoken

from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, CONTEXT_MAX_TOKENS


class LLMInterface:
    """Simple LLM interface for baseline RAG using OpenRouter."""

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        self.model = model or LLM_MODEL
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.max_tokens = max_tokens or LLM_MAX_TOKENS

        # Single tokenizer source of truth
        self.encoding = tiktoken.get_encoding("cl100k_base")

        self._init_openrouter()

    # ------------------------------------------------------------------
    # OpenRouter initialization
    # ------------------------------------------------------------------

    def _init_openrouter(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Please set it in your .env file."
            )

        self.site_url = os.getenv("OPENROUTER_SITE_URL", "")
        self.site_name = os.getenv("OPENROUTER_SITE_NAME", "")

        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

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

        # Fixed prompt overhead
        prompt_header = (
            "Answer the following question using the provided context.\n"
            "If the context does not contain enough information, say so.\n\n"
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
        # Conservative query expansion - preserve the original intent
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

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": rewrite_prompt}
            ],
            "temperature": 0.7,  # Slightly higher temperature for variety
            "max_tokens": 100,  # Short response for query rewriting
        }

        response = requests.post(
            self.endpoint,
            headers=headers,
            data=json.dumps(payload),
            timeout=60,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter API error {response.status_code}: {response.text}"
            )

        data = response.json()

        try:
            rewritten = data["choices"][0]["message"]["content"].strip()
            # Remove quotes if the LLM wrapped the query in them
            rewritten = rewritten.strip('"\'')
            return rewritten
        except (KeyError, IndexError):
            raise RuntimeError(f"Malformed OpenRouter response: {data}")

    # ------------------------------------------------------------------
    # Generate response
    # ------------------------------------------------------------------

    def generate(self, query: str, retrieved_chunks: List[Dict]) -> str:
        prompt = self.build_prompt(query, retrieved_chunks)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = requests.post(
            self.endpoint,
            headers=headers,
            data=json.dumps(payload),
            timeout=60,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter API error {response.status_code}: {response.text}"
            )

        data = response.json()

        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            raise RuntimeError(f"Malformed OpenRouter response: {data}")
