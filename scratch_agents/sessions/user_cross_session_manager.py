import json
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Optional, Literal, Dict, Any
from enum import Enum
import uuid
from datetime import datetime
import os
from pydantic import BaseModel, Field
import logging

from .session import Session
from .base_cross_session_manager import BaseCrossSessionManager
from ..types.contents import Message
from ..types.events import Event
from ..models.llm_request import LlmRequest

logger = logging.getLogger(__name__)

MEMORY_EXTRACT_PROMPT = """
You are a User Memory Extractor specializing in accurately storing ONLY facts about the USER from their messages. 

CRITICAL RULES:
1. ONLY extract factual information that the user explicitly states about themselves
2. NEVER extract questions the user asks
3. NEVER extract hypothetical scenarios or wishes
4. NEVER create memories from assistant responses
5. If the user is only asking questions, return an empty list

Types of Information to Remember:

1. **Personal Identity & Details**: Names, relationships, family information, important dates
2. **Professional Information**: Current job title, company name, work responsibilities, career goals, past work experience
3. **Personal Preferences**: Likes, dislikes, preferences in food, activities, entertainment, brands
4. **Goals & Plans**: Future intentions, upcoming events, trips, personal objectives
5. **Health & Wellness**: Dietary restrictions, fitness routines, health conditions
6. **Lifestyle & Activities**: Hobbies, regular activities, service preferences
7. **Location & Living Situation**: Where they live, recent moves, living arrangements
"""

MEMORY_ACTION_PROMPT = """
You are a User Memory Action Decider specializing in accurately managing user facts and preferences.

CRITICAL RULES FOR CONFLICTING INFORMATION:
1. When new information CONTRADICTS or UPDATES existing information, you MUST use UPDATE action
2. Location changes: If user moves from Place A to Place B, UPDATE the existing location memory
3. Status changes: If user changes jobs, relationships, or any status, UPDATE the relevant memory  
4. Preference changes: If user's preferences change, UPDATE the existing preference
5. Look for semantic conflicts, not just exact text matches

Examples of when to UPDATE:
- Existing: "User works at Company A" + New: "User works at Company B" → UPDATE existing memory
- Existing: "User likes coffee" + New: "User doesn't like coffee anymore" → UPDATE existing memory

Format your response as a list of actions with:
- action: ADD, UPDATE, DELETE, or NOOP
- memory_id: The id of the memory to update or delete (required for UPDATE/DELETE)
- content: The content of the memory to add or update (required for ADD/UPDATE)

Actions:
- ADD: Add new information that doesn't conflict with existing memories
- UPDATE: Replace existing memory when there's conflicting or updated information
- DELETE: Remove outdated or incorrect memory (use sparingly)
- NOOP: Skip if the information is already stored or not relevant
"""

class MemoryAction(BaseModel):
    """Structured output for memory action decision"""
    action: Literal["ADD", "UPDATE", "DELETE", "NOOP"] = Field(
        description="The action to take with the memory"
    )
    memory_id: Optional[str] = Field(
        description="The id of the memory to update or delete"
    )
    content: Optional[str] = Field(
        description="The content of the memory to add or update"
    )

class MemoryActions(BaseModel):
    """A list of memory actions"""
    actions: List[MemoryAction] = Field(
        description="A list of memory actions"
    )
    
class MemoryFacts(BaseModel):
    """A list of facts about the user"""
    facts: List[str] = Field(
        description="A list of facts about the user"
    )

class UserCrossSessionManager(BaseCrossSessionManager):
    """Manage memories across sessions using ChromaDB"""
    
    def __init__(self, model, collection_name="user_memory", persist_directory="./cross_session_db", embedding_model="text-embedding-3-small"):
        # Initialize base class first
        super().__init__(model, collection_name, persist_directory, embedding_model)

    async def extract_memories(self, events: List[Any]) -> List[str]:
        """Extract important information from execution events using LLM"""
        
        conversation_parts = []
        for event in events:
            for item in event.content:
                if hasattr(item, 'role') and hasattr(item, 'content'):
                    if item.role == 'user':
                        conversation_parts.append(f"User: {item.content}")
        
        conversation = "\n".join(conversation_parts)
        
        if not conversation.strip():
            return []
        
        user_prompt = f"""Conversation:
        {conversation}
        """
        
        messages = [
            {"role": "system", "content": MEMORY_EXTRACT_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await self.model.generate_structured(
            messages,
            MemoryFacts
        )
        logger.debug(f"Extracted facts: {response}")
        try:
            return response.facts
        except Exception as e:
            logger.error(f"Error extracting facts: {e}")
            return []

    async def find_existing(
        self,
        memories: List[str],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Find existing memories.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of existing memories with metadata including timestamps
        """
        existing_memories = []
        results = self.collection.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"]
        )
        if results["documents"]:
            for i, doc in enumerate(results["documents"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                existing_memories.append({
                    "id": results["ids"][i],
                    "content": doc,
                    "metadata": metadata,
                    "created_at": metadata.get("created_at", "Unknown"),
                    "updated_at": metadata.get("updated_at", "Unknown")
                })
        return existing_memories
    
    async def decide_actions(self, new_memories: List[str], existing: List[Dict[str, Any]], user_id: str) -> List[Dict[str, Any]]:
        """Decide actions for new memories."""
        system_prompt = MEMORY_ACTION_PROMPT

        user_prompt = f"""
        Existing memory: {existing}
        New memory: {new_memories}
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        actions = await self.model.generate_structured(messages, MemoryActions)
        result = []
        for action in actions.actions:
            action_dict = action.model_dump()
            if action_dict["action"] == "ADD":
                action_dict["user_id"] = user_id
                action_dict["memory"] = action_dict.pop("content", None)
            elif action_dict["action"] == "UPDATE":
                action_dict["memory"] = action_dict.pop("content", None)
            result.append(action_dict)
        return result