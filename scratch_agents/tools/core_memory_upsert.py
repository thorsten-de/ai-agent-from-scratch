from .decorator import tool

@tool
async def core_memory_upsert(
    block: str, 
    content: str, 
    update_content: str = None, 
    context = None 
) -> str:
    """Update or insert content in core memory blocks
    
    Args:
        block: Must be 'agent' or 'user'
        content: Text to find or full replacement
        update_content: New text for partial update
    """
    
    memory = context.session.core_memory
    current = memory.get(block, "")
    
    if update_content:  
        if content in current:
            memory[block] = current.replace(content, update_content)
            return f"Updated {block}"
        else:
            if current:
                memory[block] = f"{current}\n{update_content}"
            else:
                memory[block] = update_content  
                return f"Added to {block}: {update_content}"
    else:
        memory[block] = content
        return f"Set {block}"