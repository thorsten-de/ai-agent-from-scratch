"""Base class for cross-session memory management."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from datetime import datetime
import logging
import os
import uuid
from .session import Session

logger = logging.getLogger(__name__)


class BaseCrossSessionManager(ABC):
    """Abstract base class for cross-session memory management."""
    
    def __init__(
        self,
        model,
        collection_name: str,
        persist_directory: str = "./cross_session_db",
        embedding_model: str = "text-embedding-3-small"
    ):
        """Initialize the base cross-session manager.
        
        Args:
            model: LLM model for memory processing
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            embedding_model: Optional custom embedding model
        """
        self.model = model
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
        )
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=self.embedding_model
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=embedding_function
                )
            logger.info(f"Using existing collection: {collection_name}")
        except Exception:
            logger.error(f"Error getting or creating collection: {collection_name}")
            raise
    
    @abstractmethod
    async def extract_memories(
        self,
        events: List[Dict[str, Any]],
    ) -> List[str]:
        """Extract memories from session events.
        
        Args:
            events: List of session events
            user_id: User identifier
            
        Returns:
            List of extracted memory strings
        """
        pass
    
    async def process_session(
        self,
        session: Session,
        execution_id: str
    ) -> None:
        """Process a completed session and extract/merge memories.
        
        Args:
            session: Session data containing events
            execution_id: Unique execution identifier
        """
        try:
            user_id = session.user_id
            events = session.events
            
            events = [event for event in events if event.execution_id == execution_id]
            
            memories = await self.extract_memories(events)
            
            if memories:
                existing = await self.find_existing(memories, user_id)
                actions = await self.decide_actions(memories, existing, user_id)
                await self.execute_memory_actions(actions)
            else:
                logger.info(f"No memories extracted for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error processing session: {e}")
    
    async def find_existing(
        self,
        memories: List[str],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Find existing memories.
        
        Args:
            memories: List of new memory strings to merge
            user_id: User identifier
            
        Returns:
            List of existing memories with metadata
        """
        existing_memories = []
        for memory in memories:
            existing = await self.search(memory, user_id)
            if existing:    
                existing_memories.append(existing)
        return existing_memories
    
    @abstractmethod
    async def decide_actions(
        self,
        memories: List[str],
        existing: List[Dict[str, Any]],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Decide actions for new memories."""
        pass

    async def execute_memory_actions(
        self,
        actions: List[Dict[str, Any]]
    ) -> None:
        """Execute memory actions."""
        for action in actions:
            if action["action"] == "ADD":
                metadata = action.get("metadata", {})
                await self.add(action["memory"], action["user_id"], action.get("embedding"), metadata)
            elif action["action"] == "UPDATE":
                metadata = action.get("metadata", {})
                await self.update(action["memory_id"], action["memory"], action.get("embedding"), metadata)
            elif action["action"] == "DELETE":
                await self.delete(action["memory_id"])
            elif action["action"] == "NOOP":
                pass
    
    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for relevant memories.
        
        Args:
            query: Search query
            user_id: User identifier
            limit: Maximum number of results
            
        Returns:
            List of relevant memories with metadata
        """
        try:
            # Filter by user_id in metadata
            where = {"user_id": user_id}
            
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=where
            )
            
            memories = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    memory = {
                        "id": results["ids"][0][i] if results["ids"] and results["ids"][0] else None,
                        "content": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0
                    }
                    memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    async def add(
        self,
        memory: str,
        user_id: str,
        embedding: Optional[List[float]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a new memory.
        
        Args:
            memory: Memory content (as string for ChromaDB)
            user_id: User identifier
            embedding: Optional embedding vector
            additional_metadata: Additional metadata to store
            
        Returns:
            Memory ID
        """
        memory_id = f"{uuid.uuid4()}"
        
        final_metadata = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Add any additional metadata (like the original structured data)
        if additional_metadata:
            final_metadata.update(additional_metadata)
        
        if embedding:
            self.collection.upsert(
                documents=[memory],
                ids=[memory_id],
                embeddings=[embedding],
                metadatas=[final_metadata]
            )
        else:
            self.collection.add(
                documents=[memory],
                ids=[memory_id],
                metadatas=[final_metadata]
            )
        
        return memory_id
    
    async def update(
        self,
        memory_id: str,
        memory: str,
        embedding: Optional[List[float]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update an existing memory.
        
        Args:
            memory_id: ID of memory to update
            memory: New memory content (as string for ChromaDB)
            embedding: Optional embedding of the memory
            additional_metadata: Additional metadata to update
        """
        if not memory_id:
            logger.error("Cannot update memory: memory_id is None")
            return
            
        # Get existing metadata
        existing = self.collection.get(ids=[memory_id])
        if existing["metadatas"] and existing["metadatas"][0]:
            final_metadata = existing["metadatas"][0].copy()
            final_metadata["updated_at"] = datetime.now().isoformat()
        else:
            final_metadata = {}
            final_metadata["updated_at"] = datetime.now().isoformat()
        
        # Update with any additional metadata
        if additional_metadata:
            final_metadata.update(additional_metadata)
            
        if embedding:
            self.collection.upsert(
                ids=[memory_id],
                documents=[memory],
                embeddings=[embedding],
                metadatas=[final_metadata]
            )
        else:
            self.collection.upsert(
                ids=[memory_id],
                documents=[memory],
                metadatas=[final_metadata]
            )
    
    async def delete(
        self,
        memory_id: str
    ) -> None:
        """Delete a memory.
        
        Args:
            memory_id: ID of memory to delete
        """
        self.collection.delete(ids=[memory_id])