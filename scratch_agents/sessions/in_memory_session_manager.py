from .base_session_manager import BaseSessionManager
from .session import Session
from scratch_agents.types.events import Event
from datetime import datetime

class InMemorySessionManager(BaseSessionManager):
    """In-memory session manager"""
    
    def __init__(self):
        self.sessions = {}
        
    def create_session(self, session_id: str, user_id: str = None) -> Session:
        if session_id in self.sessions:
            raise ValueError(f"Session with id {session_id} already exists")
        self.sessions[session_id] = Session(session_id=session_id, user_id=user_id)
        return self.sessions[session_id]
    
    def get_session(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            raise ValueError(f"Session with id {session_id} does not exist")
        return self.sessions[session_id]
    
    def get_or_create_session(self, session_id: str, user_id: str = None) -> Session:
        if session_id not in self.sessions:
            return self.create_session(session_id, user_id)
        return self.sessions[session_id]

    def add_event(self, session: Session, event: Event) -> None:
        session.events.append(event)
        session.last_updated_at = datetime.now()