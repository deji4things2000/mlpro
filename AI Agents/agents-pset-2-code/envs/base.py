from dataclasses import dataclass


@dataclass
class StepResult:
    observations: dict[str, str]
    rewards: dict[str, float]
    done: bool
    info: dict[str, object]


class BaseEnv:
    """Base environment API for Concordia-inspired simulations.

    Required interface:
      - reset() -> dict[str, str]
      - step(actions: dict[str, object]) -> StepResult
      - get_obs(agent_id: str) -> str
    """

    def __init__(self, agent_ids: list[str], seed: int = 0) -> None:
        self.agent_ids = list(agent_ids)
        self.seed = seed
        self._observations: dict[str, str] = {}

    def reset(self) -> dict[str, str]:
        raise NotImplementedError

    def step(self, actions: dict[str, object]) -> StepResult:
        raise NotImplementedError

    def get_obs(self, agent_id: str) -> str:
        return self._observations.get(agent_id, "")

    def _assert_action_keys(self, actions: dict[str, object]) -> None:
        missing = [a for a in self.agent_ids if a not in actions]
        if missing:
            raise ValueError(f"Missing actions for agents: {missing}")

    def _blank_obs(self) -> dict[str, str]:
        return {agent_id: "" for agent_id in self.agent_ids}
