import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Literal
from config import MAX_CALL_DURATION_S

AgentState = Literal["LISTENING", "THINKING", "SPEAKING"]

@dataclass
class CallSession:
    call_sid: str
    from_number: str = ""
    to_number: str = ""
    start_time: float = field(default_factory=time.time)
    conversation_history: List[Dict] = field(default_factory=list)
    agent_state: AgentState = "LISTENING"
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def duration(self) -> float:
        return time.time() - self.start_time

    def set_state(self, new_state: AgentState):
        with self.lock:
            if self.agent_state != new_state:
                self.agent_state = new_state

    def add_exchange(self, user_input: str, agent_response: str):
        with self.lock:
            self.conversation_history.append({"role": "user", "content": user_input})
            if agent_response:
                self.conversation_history.append({"role": "assistant", "content": agent_response})
            # keep last 10 exchanges only
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

    def get_context(self) -> List[Dict]:
        with self.lock:
            return list(self.conversation_history)

    def should_end(self) -> bool:
        return self.duration >= MAX_CALL_DURATION_S