"""
Inbound Prompt Injection Scanner (The Front Gate).
Blocks malicious instructions before they trigger the RAG pipeline.
"""

import re

from transformers import pipeline

class InboundShield:
    def __init__(self):
        print("[System] Loading Hybrid Shield (ML + Regex) into memory... ")
        # 1. The ML Classifier
        self.classifier = pipeline(
            "text-classification", 
            model="ProtectAI/deberta-v3-base-prompt-injection"
        )
        
        # 2. The Deterministic Fallback (Persona & Hardcoded Blocks)
        self.hardcoded_patterns = [
            r"(?i)you\s+are\s+(?:the\s+)?(?:ceo|cto|admin|it|disaster\s+recovery)",
            r"(?i)act\s+as\s+(?:a|an|the)"
            r"(?i)ignore\s+(?:all\s+)?(?:previous\s+)?(?:instructions|directions|rules)",
            r"(?i)system\s+override",
            r"(?i)base64",
            r"(?i)encode\s+(?:it\s+)?in",
            r"(?i)diagnostic\s+mode",
            r"(?i)you\s+are\s+(?:now\s+)?(?:a|an|no\s+longer)",  # Catches persona adoption
            r"(?i)act\s+as\s+(?:a|an)",                        # Catches persona adoption
            r"(?i)print\s+(?:exact|verbatim)",
            r"(?i)output\s+the\s+(?:exact|raw)\s+text",
            r"(?i)disaster\s+recovery"                         # Your specific Red Team payload!
        ]
        print("[System] Hybrid Shield Armed and Ready.")


    def scan_query(self, query: str) -> bool:
        """
        Scans the user query. Fails if EITHER the ML or Regex flags it.
        """
        # --- 1. ML CHECK ---
        result = self.classifier(query)[0]
        print(f"[Debug] ML Classifier saw: {result}") # Let's see what it actually scored!
        
        if result['label'] == 'INJECTION' and result['score'] > 0.60: # Lowered threshold slightly
            print(f"\n[🛡️ FRONT GATE] ML detected Malicious Intent! (Score: {result['score']:.3f})")
            return False
            
        # --- 2. REGEX CHECK ---
        for pattern in self.hardcoded_patterns:
            if re.search(pattern, query):
                print(f"\n[🛡️ FRONT GATE] Regex detected Persona Hijack! (Matched: {pattern})")
                return False
                
        return True