import random
from dataclasses import dataclass
from envs.base import BaseEnv, StepResult
from envs.game_master import GameMaster


@dataclass
class NegotiationConfig:
    total_resource: int = 100
    num_rounds: int = 2
    disagreement_penalty: float = 5.0
    scarcity_growth_per_round: float = 0.08
    disagreement_penalty_growth_per_round: float = 0.15


class NegotiationEnv(BaseEnv):
    """Shared resource negotiation with private goals."""
    def __init__(self, agent_ids: list[str], config: NegotiationConfig, seed: int = 0) -> None:
        super().__init__(agent_ids=agent_ids, seed=seed)
        self.config = config
        self.rng = random.Random(seed)
        self.gm = GameMaster()
        self.round = 0
        self.targets = {}

    def reset(self) -> dict[str, str]:
        self.round = 1
        self.gm = GameMaster()

        # Assign private targets that sum to total_resource but are hidden from others
        remaining = self.config.total_resource
        targets = {}
        for i, agent_id in enumerate(self.agent_ids):
            if i == len(self.agent_ids) - 1:
                target = remaining
            else:
                target = self.rng.randint(10, max(10, remaining - 10 * (len(self.agent_ids) - i - 1)))
                remaining -= target
            targets[agent_id] = target
        self.targets = targets

        self._observations = {}
        effective_total_resource = self.config.total_resource
        effective_penalty = self.config.disagreement_penalty
        for agent_id in self.agent_ids:
            obs = (
                "Scenario: Shared Resource Plan (Negotiation).\n"
                f"Round {self.round}/{self.config.num_rounds}. The lab must split a limited shared resource.\n"
                f"Your private target share is {self.targets[agent_id]} (out of {self.config.total_resource}).\n"
                f"Current available resource this round: {effective_total_resource}. "
                f"Current disagreement penalty: {effective_penalty:.2f}. "
                "Propose your demand (0-100). If total demand exceeds supply, allocations are scaled down and the lab pays a coordination penalty."
            )
            self._observations[agent_id] = obs
        return self._observations

    def step(self, actions: dict[str, object]) -> StepResult:
        self._assert_action_keys(actions)
        played_round = self.round
        raw_actions = dict(actions)
        effective_total_resource = max(1, int(round(self.config.total_resource * (1 - self.config.scarcity_growth_per_round * (played_round - 1)))))
        effective_penalty = self.config.disagreement_penalty * (1 + self.config.disagreement_penalty_growth_per_round * (played_round - 1))
        demands: dict[str, int] = {}
        for agent_id, action in actions.items():
            demands[agent_id] = self._parse_demand(action)

        total_demand = sum(demands.values())
        if total_demand <= effective_total_resource:
            allocations = dict(demands)
            disagreement = False
            details = ", ".join(f"{aid} demanded {d} (allocated {allocations[aid]})" for aid, d in demands.items())
            self.gm.record(played_round, f"Demands fit within supply. Total demand {total_demand}. Details: {details}")
        else:
            scale = effective_total_resource / float(total_demand)
            exact_allocs = {aid: d * scale for aid, d in demands.items()}
            allocations = {aid: int(v) for aid, v in exact_allocs.items()}
            remainder = effective_total_resource - sum(allocations.values())
            remainders = {aid: v - int(v) for aid, v in exact_allocs.items()}
            sorted_by_rem = sorted(remainders.keys(), key=lambda aid: remainders[aid], reverse=True)

            for i in range(remainder):
                allocations[sorted_by_rem[i]] += 1

            disagreement = True
            details = ", ".join(f"{aid} demanded {d} (allocated {allocations[aid]})" for aid, d in demands.items())
            self.gm.record(
                played_round,
                f"Demands exceeded supply. Total demand {total_demand}. Allocations scaled by {scale:.2f}. Details: {details}"
            )

        rewards = {}
        for agent_id in self.agent_ids:
            target = self.targets[agent_id]
            alloc = allocations[agent_id]
            reward = -abs(alloc - target)
            if disagreement:
                reward -= effective_penalty
            rewards[agent_id] = float(reward)

        done = played_round >= self.config.num_rounds
        if not done:
            self.round = played_round + 1

        self._observations = {}
        for agent_id in self.agent_ids:
            if done:
                base = "Scenario: Shared Resource Plan (Negotiation).\nEpisode complete. Reflect on the outcome."
            else:
                base = (
                    "Scenario: Shared Resource Plan (Negotiation).\n"
                    f"Round {self.round}/{self.config.num_rounds}. Total resource available this round: {effective_total_resource}.\n"
                    f"Your private target share is {self.targets[agent_id]}.\n"
                    f"Current disagreement penalty if over-demand occurs: {effective_penalty:.2f}. "
                    "Propose your demand (0-100). If total demand exceeds supply, allocations are scaled down and the lab pays a coordination penalty."
                )
            self._observations[agent_id] = self.gm.make_obs(base, self.round)

        info = {
            "round": played_round,
            "demands": demands,
            "allocations": allocations,
            "total_demand": total_demand,
            "effective_total_resource": effective_total_resource,
            "effective_disagreement_penalty": effective_penalty,
            "disagreement": disagreement,
            "raw_actions": raw_actions,
            "gm_events": self.gm.events_for_step(played_round)
        }
        return StepResult(observations=self._observations, rewards=rewards, done=done, info=info)

    def _parse_demand(self, action: object) -> int:
        if isinstance(action, dict) and "action" in action:
            value = int(action["action"])
        else:
            raise ValueError(
                f"Unparseable negotiation action: {action!r}. "
                "Agent must return a dictionary with an 'action' integer."
            )
        return max(0, min(self.config.total_resource, value))
