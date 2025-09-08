from abc import ABC, abstractmethod

class MemoryStrategy(ABC):
    """Base class for memory management strategies"""
    
    @abstractmethod
    async def apply(self, context, llm_request):  #A
        """Apply memory management strategy to the request"""
        pass
    
    async def __call__(self, context, llm_request):  #B
        """Make strategy callable as a before_llm_callback"""
        return await self.apply(context, llm_request)