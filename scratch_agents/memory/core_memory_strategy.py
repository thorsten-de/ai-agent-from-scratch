from .base_memory_strategy import MemoryStrategy


class CoreMemoryStrategy(MemoryStrategy):
    """Automatically add core memory to LLM context"""
    
    async def apply(self, context, llm_request):
        """Add core memory as instructions if it exists"""
        core_memory = context.session.core_memory  
        
        memory_parts = []
        if core_memory.get("agent"):  
            memory_parts.append(f"[Your Persona]\n{core_memory['agent']}")
        if core_memory.get("user"):  
            memory_parts.append(f"[User Info]\n{core_memory['user']}")
        
        if memory_parts:  
            memory_text = "\n\n".join(memory_parts)
            llm_request.add_instructions([memory_text])  
            
        return None  