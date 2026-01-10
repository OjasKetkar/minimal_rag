"""
Utility script to list available Gemini models.
Run this to see what models your API key has access to.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

def list_gemini_models():
    """Fetch and display available Gemini models."""
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in .env file")
        print("\nGet your free API key from: https://aistudio.google.com/apikey")
        return
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # List available models
        models = genai.list_models()
        
        print("Available Gemini Models:")
        print("=" * 50)
        
        # Filter for generation models
        generation_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                generation_models.append(model.name.replace('models/', ''))
                print(f"  - {model.name.replace('models/', '')}")
        
        print("\n" + "=" * 50)
        print(f"Total: {len(generation_models)} generation models available")
        print("\nTo use a model, add to your .env file:")
        print("  LLM_MODEL=model-name-here")
        print("\nRecommended models for RAG:")
        print("  - gemini-2.5-flash (fast, free tier)")
        print("  - gemini-2.5-pro (more capable)")
        print("  - gemini-2.0-flash-exp (experimental)")
        
    except ImportError:
        print("ERROR: google-generativeai package not installed")
        print("Install with: poetry install")
    except Exception as e:
        print(f"Error fetching models: {e}")
        print("\nCommon models to try:")
        print("  - gemini-1.5-flash")
        print("  - gemini-1.5-pro")
        print("  - gemini-2.0-flash-exp")
        print("\nGet your API key from: https://aistudio.google.com/apikey")

if __name__ == "__main__":
    list_gemini_models()

