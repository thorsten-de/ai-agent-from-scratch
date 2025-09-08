"""Task-specific cross-session memory management."""

from typing import List, Dict, Any, Optional, Literal
import logging
from pydantic import BaseModel, Field
import json

from .base_cross_session_manager import BaseCrossSessionManager
from ..types.events import Event
from ..types.contents import Message, ToolCall, ToolResult

logger = logging.getLogger(__name__)

MEMORY_EXTRACT_PROMPT = """
You are a Task Memory Extractor specializing in tracking agent actions and problem-solving attempts.
Extract ONLY information about what the agent ACTUALLY DID in this conversation.

Focus on:
1. **Problem Identification**: What issue or challenge was the agent trying to address?
2. **Actions Taken**: What specific actions did the agent perform? (tools used, searches made, code written, etc.)
3. **Key Discoveries**: What important facts or information did the agent discover during the process?
4. **Success Status**: Was the task completed successfully?

DO NOT extract:
- Personal user information (name, preferences, etc.)
- General conversation or greetings
- User opinions or feelings
- Future plans or what should be done

Format each task as a structured memory with:
- problem: Clear description of what the agent was asked to do or investigate
- actions_taken: Specific actions the agent performed (not what it should do)
- key_discoveries: Important information discovered during the task
- success: true/false indicating if the task was completed

Examples of GOOD task memories:
{
  "problem": "User asked about React component not rendering",
  "actions_taken": "Examined useEffect hook, identified missing dependency in array, added state variable to dependency array",
  "key_discoveries": "useEffect was missing 'count' state variable in dependency array causing stale closure",
  "success": true
}

{
  "problem": "User reported database connection timeouts in production",
  "actions_taken": "Checked connection pool configuration, analyzed production logs, increased pool size from 10 to 50, implemented retry logic with exponential backoff",
  "key_discoveries": "Production load peaked at 45 concurrent connections, default pool size was only 10",
  "success": true
}

{
  "problem": "User asked 'What is Mem0 and how does it work?'",
  "actions_taken": "Performed multiple web searches with different query variations to find information about Mem0",
  "key_discoveries": "Found that Mem0 is an open-source memory layer for LLM applications, has a GitHub repo (mem0ai/mem0), provides hybrid data storage and intelligent retrieval",
  "success": false
}
"""

MEMORY_ACTION_PROMPT = """
You are a Task Memory Action Decider specializing in tracking agent actions and problem-solving attempts.
You are given a list of new task memories and a list of existing task memories.
You need to decide whether to ADD, UPDATE, DELETE, or NOOP the new task memories.

Format your response as a list of actions with:
- action: ADD, UPDATE, DELETE, or NOOP
- memory_id: The id of the memory to update or delete

Action:
- ADD: Add the new task memory if it describes a different problem or significantly different approach
- UPDATE: Update the existing task memory if it's the same problem but with better/more complete actions or discoveries
- DELETE: Delete the existing task memory if it's outdated or no longer relevant
- NOOP: Do not add if it's essentially the same problem with similar actions and discoveries

"""


class TaskMemory(BaseModel):
    """Structured task memory."""
    problem: str = Field(description="The problem or task the agent was asked to address")
    actions_taken: str = Field(description="The specific actions the agent performed")
    success: bool = Field(description="Whether the task was completed successfully")
    key_discoveries: Optional[str] = Field(default=None, description="Important information discovered during the task")
    
class MemoryAction(BaseModel):
    """Memory action."""
    action: Literal["ADD", "UPDATE", "DELETE", "NOOP"] = Field(description="The action to take with the memory")
    memory_id: Optional[str] = Field(description="The id of the memory to update or delete")


class TaskCrossSessionManager(BaseCrossSessionManager):
    """Manage task-specific memories across sessions."""
    
    def __init__(self, model, 
                 collection_name="task_memories", 
                 persist_directory="./cross_session_db",
                 ):
        """Initialize task cross-session manager.
        
        Args:
            model: LLM model for memory extraction
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
        """
        super().__init__(model, collection_name, persist_directory)
    
    async def extract_memories(self, events: List[Event]):
        conversation_parts = []
        
        for event in events:
            for item in event.content:
                if isinstance(item, Message):
                    conversation_parts.append(f"{item.role}: {item.content}")
                elif isinstance(item, ToolCall):
                    conversation_parts.append(f"{item.tool_call_id}: {item.name}")
                elif isinstance(item, ToolResult):
                    conversation_parts.append(f"{item.tool_call_id}: {item.name} {item.content}")
        
        conversation = "\n".join(conversation_parts)

        user_prompt = f"""Conversation:
        {conversation}
        """
        messages = [
            {"role": "system", "content": MEMORY_EXTRACT_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self.model.generate_structured(messages, TaskMemory)
            task_memory = TaskMemory.model_validate(response)
            return [task_memory.model_dump()]
            
        except Exception as e:
            logger.error(f"Error extracting task memories: {e}")
            return []
        
    async def find_existing(self, memories: List[Dict], user_id: str) -> List[Dict[str, Any]]:
        existing_memories = []
        for memory in memories:
            query = memory["problem"]
            results = await self.search(query, user_id)
            if results:
                existing_memories.append(results[0])
        return existing_memories
    
    async def decide_actions(self, new_memory: List[Dict], existing: List[Dict[str, Any]], user_id: str) -> List[Dict[str, Any]]:
        system_prompt = MEMORY_ACTION_PROMPT
        user_prompt = f"""
        Existing memory: {existing}
        New memory: {new_memory}
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        action = await self.model.generate_structured(messages, MemoryAction)
        result = []
        if action.action == "UPDATE":

            memory_id = action.memory_id
            if not memory_id:
                logger.error("Cannot update memory: no memory_id available")
                return []
            embeddings = await self.model.embed(self.embedding_model, [new_memory[0]["problem"]])
            # Convert dict to string for ChromaDB document field
            memory_str = json.dumps(new_memory[0], ensure_ascii=False)
            result.append({
                "action": "UPDATE",
                "memory_id": memory_id,
                "memory": memory_str,
                "embedding": embeddings[0],
                "metadata": new_memory[0]  # Store original dict in metadata
            })
        elif action.action == "ADD":
            embeddings = await self.model.embed(self.embedding_model, [new_memory[0]["problem"]])
            # Convert dict to string for ChromaDB document field
            memory_str = json.dumps(new_memory[0], ensure_ascii=False)
            result.append({
                "action": "ADD",
                "memory": memory_str,
                "user_id": user_id,
                "embedding": embeddings[0],
                "metadata": new_memory[0]  # Store original dict in metadata
            })
        elif action.action == "DELETE":
            result.append({
                "action": "DELETE",
                "memory_id": action.memory_id
            })
        elif action.action == "NOOP":
            result.append({
                "action": "NOOP"
            })
        return result