import uuid
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime
from ..types.contents import ContentItem

class Session(BaseModel):
    """Container for short-term memory during a conversation session"""
    user_id: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    events: List[ContentItem] = Field(default_factory=list)
    state: Dict[str, Any] = Field(default_factory=dict)
    last_updated_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def core_memory(self) -> Dict[str, str]:
        """Access core memory with automatic initialization"""
        if "core_memory" not in self.state:
            self.state["core_memory"] = {
                "persona": "You are a helpful AI assistant",
                "human": ""
            }
        return self.state["core_memory"]