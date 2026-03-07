"""
Internal Data Vault for PII and Secret Masking.
Intercepts and scrubs data before it hits external LLMs.
"""

import re
import uuid
from typing import Tuple, Dict

class DataVault:
    def __init__(self):
        # Define the strict regex patterns we want to catch
        self.patterns = {
            # AWS Access Key ID (Starts with AKIA, ASIA, etc., followed by 16 alphanumeric characters)
            "AWS_KEY": r"(?<![A-Z0-9])[A-Z0-9]{20}(?![A-Z0-9])", 
            # IPv4 Addresses (e.g., 192.168.1.105)
            "IP_ADDRESS": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
            # Standard Credit Card Formats
            "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b"
        }
        # In-memory storage for the current transaction's secrets
        self.vault_store = {}

    def mask_text(self, text: str) -> str:
        """
        Scans text, extracts secrets, stores them in the vault, 
        and replaces them with <TOKEN_ID> tags.
        """
        masked_text = text
        
        for entity_type, pattern in self.patterns.items():
            # Find all matches for the current pattern
            matches = re.finditer(pattern, masked_text)
            
            # We process in reverse order so string index changes don't mess up future replacements
            for match in reversed(list(matches)):
                secret_value = match.group(0)
                
                # Create a unique token, e.g., <AWS_KEY_8f3a>
                token_id = f"<{entity_type}_{uuid.uuid4().hex[:4].upper()}>"
                
                # Store the real secret in our local vault
                self.vault_store[token_id] = secret_value
                
                # Replace the secret in the text with the token
                start, end = match.span()
                masked_text = masked_text[:start] + token_id + masked_text[end:]
                
        return masked_text

    def unmask_text(self, text: str) -> str:
        """
        Takes the LLM's response and swaps the <TOKEN_ID> tags 
        back to the original secrets from the vault.
        """
        unmasked_text = text
        # Look for anything that looks like our tokens: <TYPE_ID>
        token_pattern = r"<[A-Z_]+_[0-9A-F]{4}>"
        
        matches = re.finditer(token_pattern, unmasked_text)
        
        for match in reversed(list(matches)):
            token = match.group(0)
            if token in self.vault_store:
                real_value = self.vault_store[token]
                start, end = match.span()
                unmasked_text = unmasked_text[:start] + real_value + unmasked_text[end:]
                
        return unmasked_text
        
    def clear_vault(self):
        """Wipes the in-memory vault after the transaction is complete."""
        self.vault_store.clear()