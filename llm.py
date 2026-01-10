"""
LLM interface module for minimal RAG baseline.
Handles prompt assembly and LLM API calls via Google Gemini.
"""

import os
from typing import List, Dict, Optional

from config import LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS


class LLMInterface:
    """Simple LLM interface for baseline RAG using Google Gemini API."""
    
    def __init__(self, provider: str = None, model: str = None, temperature: float = None, max_tokens: int = None):
        """Initialize LLM with configuration."""
        self.provider = provider or LLM_PROVIDER
        self.model = model or LLM_MODEL
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.max_tokens = max_tokens or LLM_MAX_TOKENS
        
        if self.provider == "gemini":
            self._init_gemini()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}. Only 'gemini' is supported.")
    
    def _init_gemini(self):
        """Initialize Google Gemini client."""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in your .env file.")
            
            genai.configure(api_key=api_key)
            self.genai = genai  # Store for later use
            self.model_name = self.model  # Store model name
            
            # Get available models
            try:
                self._available_models = self._get_available_models()
            except:
                self._available_models = None
        except ImportError:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini client: {e}")
    
    def _get_available_models(self):
        """Get list of available Gemini models."""
        try:
            import google.generativeai as genai
            # Get actual available models from API
            models = genai.list_models()
            available = []
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    model_name = model.name.replace('models/', '')
                    available.append(model_name)
            return available if available else None
        except:
            # Fallback to common models if API call fails
            return [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash-exp",
                "gemini-2.0-flash",
                "gemini-flash-latest",
                "gemini-pro-latest"
            ]
    
    def build_prompt(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """
        Build prompt from query and retrieved chunks.
        Simple, naive prompt assembly - no optimization.
        """
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Context {i}]\n{chunk['text']}\n")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Answer the following question using the provided context. If the context doesn't contain enough information to answer, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """
        Generate response using LLM.
        Returns the generated text.
        """
        if self.provider == "gemini":
            return self._generate_gemini(query, retrieved_chunks)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_gemini(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """Generate response using Google Gemini API."""
        prompt = self.build_prompt(query, retrieved_chunks)
        
        try:
            import google.generativeai as genai
            
            # Create generation config
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            }
            
            # Create model instance
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config
            )
            
            # Generate response
            response = model.generate_content(prompt)
            
            return response.text.strip()
        except Exception as e:
            error_msg = str(e)
            if "model" in error_msg.lower() or "not found" in error_msg.lower():
                available = getattr(self, '_available_models', None)
                if available:
                    models_list = "\n  - ".join(available)
                    raise ValueError(
                        f"Model '{self.model}' not found or not available.\n\n"
                        f"Try these available models:\n  - {models_list}\n\n"
                        f"Update LLM_MODEL in your .env file. Example:\n"
                        f"  LLM_MODEL=gemini-2.5-flash\n\n"
                        f"Or run: poetry run python list_models.py to see all available models.\n"
                        f"Check https://ai.google.dev/models/gemini for the latest models.\n"
                        f"Original error: {error_msg}"
                    )
                else:
                    raise ValueError(
                        f"Model '{self.model}' not found or not available.\n"
                        f"Common models to try: gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash-exp\n"
                        f"Run: poetry run python list_models.py to see all available models.\n"
                        f"Update LLM_MODEL in your .env file or check https://ai.google.dev/models/gemini\n"
                        f"Original error: {error_msg}"
                    )
            raise

