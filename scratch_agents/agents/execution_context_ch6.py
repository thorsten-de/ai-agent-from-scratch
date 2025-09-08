from dataclasses import dataclass
from ..sessions.session import Session
from ..sessions.in_memory_session_manager import InMemorySessionManager
from ..sessions.base_session_manager import BaseSessionManager
from dataclasses import field
import uuid
from pydantic import BaseModel
from typing import List, Dict, Any
from ..types.events import Event
from ..sessions.base_cross_session_manager import BaseCrossSessionManager

@dataclass
class ExecutionContext:
    session: Session
    session_manager: BaseSessionManager
    cross_session_manager: BaseCrossSessionManager
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_step: int = 0
    max_steps: int = 10
    user_input: str = ""
    final_result: str | BaseModel = ""
    
    def add_event(self, event: Event) -> None:
        self.session_manager.add_event(self.session, event)
    @property
    def events(self) -> List[Event]:
        return self.session.events
    
    @property
    def state(self) -> Dict[str, Any]:
        return self.session.state
    
    def increment_step(self) -> None:
        self.current_step += 1