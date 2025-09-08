from .base_memory_strategy import MemoryStrategy
from ..models.llm_request import LlmRequest
from ..agents.execution_context_ch6 import ExecutionContext


class SlidingWindowStrategy(MemoryStrategy):
    """Keep only the most recent N messages in context"""
    
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        
    async def apply(self, context: ExecutionContext, llm_request: LlmRequest):
        """Apply sliding window to conversation history"""
        contents = llm_request.contents
        
        if len(contents) <= self.max_messages:
            return None
        
        # Keep only recent messages
        recent_contents = contents[-self.max_messages:]
        llm_request.contents = recent_contents
        
        print(f"Trimmed messages")
        print(f"from {len(contents)} to {self.max_messages}") 
        
        return None