class MemoryManager:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.reset()

    def reset(self):
        self.query_tokens = 0
        self.context_tokens = 0
        self.total_tokens = 0
        self.events = []

    def record(self, event_type: str, tokens: int, metadata: dict = None):
        self.events.append({
            "event": event_type,
            "tokens": tokens,
            "metadata": metadata or {}
        })
        self.total_tokens += tokens
        if event_type == "query":
            self.query_tokens += tokens
        elif event_type == "context":
            self.context_tokens += tokens

    def is_over_budget(self):
        """Check if total tokens exceed the maximum budget."""
        return self.total_tokens > self.max_tokens
    
    def remaining_budget(self):
        """Calculate remaining token budget."""
        return self.max_tokens - self.total_tokens

    def snapshot(self):
        return {
            "query_tokens": self.query_tokens,
            "context_tokens": self.context_tokens,
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "events": self.events
        }
