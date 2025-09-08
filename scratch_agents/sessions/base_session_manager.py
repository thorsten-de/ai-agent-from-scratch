from abc import ABC, abstractmethod
from typing import Optional, Dict, List
from scratch_agents.types.events import Event
from scratch_agents.sessions.session import Session


class BaseSessionManager(ABC):
    """Abstract base class for session management"""
    
    @abstractmethod
    def create_session(self, session_id: Optional[str] = None, user_id: str = None) -> Session:
        """Create a new session"""
        pass
    
    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Session]:
        """Load a session from storage"""
        pass
    
    @abstractmethod
    def get_or_create_session(self, session_id: str, user_id: str = None) -> Session:
        """Get an existing session or create a new one"""
        pass
    
    @abstractmethod
    def add_event(self, session: Session, event: Event) -> None:
        """Add an event to the session"""
        pass