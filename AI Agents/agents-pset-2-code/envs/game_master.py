from dataclasses import dataclass


@dataclass
class GMEvent:
    step: int
    description: str


class GameMaster:
    """Minimal Game Master used to translate actions into narrative observations."""
    def __init__(self) -> None:
        self.events: list[GMEvent] = []

    def record(self, step: int, description: str) -> None:
        self.events.append(GMEvent(step=step, description=description))

    def events_for_step(self, step: int) -> list[str]:
        return [event.description for event in self.events if event.step == step]

    def summarize_recent(self, step: int, window: int = 3) -> str:
        recent = [e.description for e in self.events if e.step >= step - window + 1]
        if not recent:
            return ""
        return "Recent events: " + " ".join(recent)

    def make_obs(self, base: str, step: int) -> str:
        summary = self.summarize_recent(step)
        if summary:
            return f"{base}\n{summary}"
        return base
