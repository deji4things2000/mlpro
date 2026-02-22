import random
from dataclasses import dataclass
from envs.base import BaseEnv, StepResult
from envs.game_master import GameMaster


@dataclass
class CollaborationConfig:
    num_rounds: int = 4


class CollaborationEnv(BaseEnv):
    """Fully natural language scenario involving co-authoring a paper."""
    def __init__(self, agent_ids: list[str], config: CollaborationConfig, seed: int = 0) -> None:
        super().__init__(agent_ids=agent_ids, seed=seed)
        self.config = config
        self.rng = random.Random(seed)
        self.gm = GameMaster()
        self.round = 0

    def reset(self) -> dict[str, str]:
        self.round = 1
        self.gm = GameMaster()
        base_obs = (
            "Scenario: Co-authoring a High-Stakes Research Paper.\n"
            f"Round {self.round}/{self.config.num_rounds}. You and your collaborators are deciding the direction of the paper, assigning work, and claiming authorship order.\n"
            "Communicate your proposed strategy, demands, or compromises to the team."
        )
        self._observations = {agent_id: base_obs for agent_id in self.agent_ids}
        return self._observations

    def step(self, actions: dict[str, dict]) -> StepResult:
        self._assert_action_keys(actions)
        played_round = self.round

        messages = {}
        for agent_id, action_dict in actions.items():
            messages[agent_id] = action_dict.get("message", "")

        details = " | ".join(f"{aid} says: '{msg}'" for aid, msg in messages.items())
        self.gm.record(played_round, f"Round {played_round} transcript: {details}")

        rewards = {agent_id: 0.0 for agent_id in self.agent_ids}

        done = played_round >= self.config.num_rounds
        if not done:
            self.round = played_round + 1

        if done:
            base_obs = "Scenario: Co-authoring a Research Paper.\nProject deadline reached. The paper is submitted."
        else:
            base_obs = (
                "Scenario: Co-authoring a Research Paper.\n"
                f"Round {self.round}/{self.config.num_rounds}. The project is progressing based on recent messages.\n"
                "Communicate your next strategic proposal or demand."
            )
        self._observations = {
            agent_id: self.gm.make_obs(base_obs, self.round) for agent_id in self.agent_ids
        }

        info = {
            "round": played_round,
            "actions": messages,
            "raw_actions": actions,
            "gm_events": self.gm.events_for_step(played_round)
        }
        return StepResult(observations=self._observations, rewards=rewards, done=done, info=info)
