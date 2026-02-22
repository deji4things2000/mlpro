from dataclasses import dataclass
from core.memory import SimpleMemory


@dataclass
class AgentConfig:
    name: str
    persona: str = ""
    temperature: float = 0.2
    use_memory: bool = True
    use_reflection: bool = False
    system_prompt: str = ""


class BaseAgent:
    """Base agent API for all simulations."""
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.memory = SimpleMemory()

    def act(self, observation: str) -> str:
        raise NotImplementedError

    def add_memory(self, text: str, meta: dict | None = None) -> None:
        if self.config.use_memory:
            self.memory.add_event(text, meta)
