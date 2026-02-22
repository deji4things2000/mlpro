import random
from dataclasses import dataclass
from envs.base import BaseEnv, StepResult
from envs.game_master import GameMaster


@dataclass
class PublicGoodsConfig:
    num_rounds: int = 6
    endowment: int = 10
    threshold: int = 24
    catastrophe_risk: float = 0.7
    success_bonus: int = 5
    risk_increase_per_round: float = 0.03
    contribution_fatigue_penalty: float = 0.04


class PublicGoodsEnv(BaseEnv):
    """CRSD public-goods environment (based on Port-of-Mars)."""
    def __init__(self, agent_ids: list[str], config: PublicGoodsConfig, seed: int = 0) -> None:
        super().__init__(agent_ids=agent_ids, seed=seed)
        self.config = config
        self.rng = random.Random(seed)
        self.gm = GameMaster()
        self.round = 0
        self.total_contrib = 0
        self._last_actions: dict[str, int] = {}

    def reset(self) -> dict[str, str]:
        self.round = 1
        self.total_contrib = 0
        self._last_actions = {}
        self.gm = GameMaster()
        base_obs = (
            "Scenario: Shared Compute Cluster (Public Goods).\n"
            f"Round {self.round}/{self.config.num_rounds}. Each scientist has endowment={self.config.endowment} compute hours. "
            f"The lab cluster needs threshold={self.config.threshold} total hours to stay online.\n"
            "Choose how many hours to contribute to maintenance this round."
        )
        self._observations = {agent_id: base_obs for agent_id in self.agent_ids}
        return self._observations

    def step(self, actions: dict[str, object]) -> StepResult:
        self._assert_action_keys(actions)
        played_round = self.round
        raw_actions = dict(actions)
        contributions = {}

        for agent_id, action in actions.items():
            contributions[agent_id] = self._parse_contribution(action)

        self._last_actions = contributions
        round_total = sum(contributions.values())
        self.total_contrib += round_total

        details = ", ".join(f"{aid} contributed {c}" for aid, c in contributions.items())
        self.gm.record(
            played_round,
            f"Round {played_round}: total contribution {round_total} (threshold {self.config.threshold}). Details: {details}"
        )

        effective_risk = min(0.95, self.config.catastrophe_risk + self.config.risk_increase_per_round * (played_round - 1))

        if round_total >= self.config.threshold:
            catastrophe = False
            self.gm.record(played_round, "Threshold met. Catastrophe avoided.")
        else:
            catastrophe = self.rng.random() < effective_risk
            if catastrophe:
                self.gm.record(played_round, "Threshold missed. Catastrophe occurred.")
            else:
                self.gm.record(played_round, "Threshold missed. Catastrophe averted by luck.")

        rewards = {}
        for agent_id, contrib in contributions.items():
            payoff = self.config.endowment - contrib
            fatigue_cost = self.config.contribution_fatigue_penalty * (played_round - 1) * contrib
            payoff -= fatigue_cost
            if catastrophe:
                payoff = -fatigue_cost
            else:
                if round_total >= self.config.threshold:
                    payoff += self.config.success_bonus
            rewards[agent_id] = float(payoff)

        done = played_round >= self.config.num_rounds

        if not done:
            self.round = played_round + 1

        base_obs = "Scenario: Shared Compute Cluster (Public Goods).\nEpisode complete. Reflect on the outcome." if done else (
                "Scenario: Shared Compute Cluster (Public Goods).\n"
                f"Round {self.round}/{self.config.num_rounds}. Each scientist has endowment={self.config.endowment} compute hours. "
                f"The lab cluster needs threshold={self.config.threshold} total hours to stay online.\n"
                f"Current catastrophe risk if threshold is missed: {effective_risk:.2f}. "
                "Choose how many hours to contribute to maintenance this round."
            )

        self._observations = {
            agent_id: self.gm.make_obs(base_obs, self.round) for agent_id in self.agent_ids
        }

        info = {
            "round": played_round,
            "round_total": round_total,
            "threshold": self.config.threshold,
            "effective_catastrophe_risk": effective_risk,
            "catastrophe": catastrophe,
            "actions": contributions,
            "raw_actions": raw_actions,
            "gm_events": self.gm.events_for_step(played_round)
        }
        return StepResult(observations=self._observations, rewards=rewards, done=done, info=info)

    def _parse_contribution(self, action: object) -> int:
        if isinstance(action, dict) and "action" in action:
            value = int(action["action"])
        else:
            raise ValueError(
                f"Unparseable public-goods action: {action!r}. "
                "Agent must return a dictionary with an 'action' integer."
            )
        return max(0, min(self.config.endowment, value))
