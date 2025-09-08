from .decorator import tool
from ..agents.execution_context_ch6 import ExecutionContext
from ..types.contents import Message

@tool
async def conversation_search(
    query: str,
    limit: int = 5,
    context: ExecutionContext = None
):
    """Search through current session's conversation history using exact keyword matching
    
    IMPORTANT: Use SHORT, SPECIFIC KEYWORDS that likely appear in the conversation.
    
    Args:
        query: Short keyword to search for (use simple words that might appear in messages)
        limit: Maximum number of results to return
        context: Execution context with session access
    
    Returns:
        Formatted string with search results or message if none found
    """
    
    query_lower = query.lower()
    results = []  
    
    for event in context.session.events:
        for item in event.content:
            if isinstance(item, Message) and item.content:
                if query_lower in item.content.lower():
                    results.append({
                        "role": item.role,
                        "content": item.content,
                        "event_id": event.id,
                        "timestamp": event.timestamp
                    })
                    break
    
    results = results[-limit:]
    
    if not results:
        return f"No messages found containing '{query}'"
    
    formatted = f"Found {len(results)} message(s) containing '{query}':\n\n"
    for i, result in enumerate(results, 1):
        formatted += f"{i}. [{result['role']}]: {result['content']}"
        formatted += "\n\n"
    
    return formatted