# =============================================================
# agents/selector.py
# Adaptive Computation Selector with Gumbel-Softmax
# Implements Section III-C of the paper
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SELECTOR_CONFIG, DEVICE


# =============================================================
# Temporal History Encoder
# =============================================================
class TemporalHistoryEncoder(nn.Module):
    """
    Encodes a short history of (q, v, tau) into a fixed vector.
    Uses 1D temporal convolution as described in Section III-C.
    """

    def __init__(
        self,
        nv:          int,
        history_len: int = 10,
        out_dim:     int = 128,
    ):
        super().__init__()
        self.nv          = nv
        self.history_len = history_len
        self.out_dim     = out_dim

        # Input channels: q + v + tau = 3*nv per timestep
        in_channels = 3 * nv

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # pool to single vector
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: [history_len, 3*nv]  (q, v, tau stacked)
        Returns:
            embedding: [out_dim]
        """
        # [history_len, 3*nv] -> [3*nv, history_len] -> [1, 3*nv, T]
        x = history.T.unsqueeze(0).float()
        x = self.conv(x)               # [1, 128, 1]
        x = x.squeeze(-1).squeeze(0)  # [128]
        return self.proj(x)            # [out_dim]


# =============================================================
# Gumbel-Softmax Sampling
# =============================================================
def gumbel_softmax_sample(
    logit:    torch.Tensor,
    temperature: float = 1.0,
    hard:     bool = False,
) -> torch.Tensor:
    """
    Differentiable Bernoulli sample via Gumbel-Softmax.
    Equation (8) in paper.

    Args:
        logit:       scalar logit  (log-odds of selecting full dynamics)
        temperature: tau > 0  (annealed from 1.0 to 0.1)
        hard:        if True, straight-through estimator

    Returns:
        s_t in (0,1)  -- soft during training, hard at inference
    """
    # Two-class Gumbel-Softmax: [p_approx, p_full]
    logits = torch.stack([-logit, logit])         # [2]

    # Sample Gumbel noise
    gumbels = -torch.log(
        -torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)

    y_soft = F.softmax((logits + gumbels) / temperature, dim=0)

    if hard:
        # Straight-through: discrete forward, soft backward
        idx    = y_soft.argmax()
        y_hard = torch.zeros_like(y_soft).scatter_(0, idx.unsqueeze(0), 1.0)
        return (y_hard - y_soft).detach() + y_soft   # [2]
    return y_soft   # [2]  index 1 = prob of full dynamics


# =============================================================
# Adaptive Selector Network
# =============================================================
class AdaptiveSelector(nn.Module):
    """
    Learned selector that predicts whether full rigid-body
    dynamics are needed at each timestep.
    Implements Section III-C of the paper.

    Input:
        z_t       : latent from perception encoder  [latent_dim]
        history   : recent (q, v, tau) history      [T, 3*nv]
        task_id   : one-hot task indicator          [n_tasks]

    Output:
        s_t in (0,1)  -- probability of using full dynamics
    """

    def __init__(
        self,
        nv:          int,
        latent_dim:  int   = 512,
        history_len: int   = 10,
        hidden_dim:  int   = 128,
        n_tasks:     int   = 3,
        temp_start:  float = 1.0,
        temp_end:    float = 0.1,
    ):
        super().__init__()
        self.nv          = nv
        self.latent_dim  = latent_dim
        self.history_len = history_len
        self.temp        = temp_start
        self.temp_start  = temp_start
        self.temp_end    = temp_end

        # History encoder
        self.hist_encoder = TemporalHistoryEncoder(
            nv=nv, history_len=history_len, out_dim=128)

        # Input dim: z_t + history_emb + task_one_hot
        in_dim = latent_dim + 128 + n_tasks

        # 2-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),   # scalar logit
        )

        # Tracking
        self._decisions  = []   # history of s_t values
        self._full_count = 0
        self._total      = 0

    def anneal_temperature(self, step: int, total_steps: int):
        """Linearly anneal Gumbel temperature from start to end."""
        frac      = min(step / max(total_steps, 1), 1.0)
        self.temp = self.temp_start + frac * (
            self.temp_end - self.temp_start)

    def forward(
        self,
        z_t:      torch.Tensor,
        history:  torch.Tensor,
        task_id:  torch.Tensor,
        hard:     bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_t     : perception latent    [latent_dim]
            history : state-action history [history_len, 3*nv]
            task_id : one-hot task vector  [n_tasks]
            hard    : use straight-through (True at inference)

        Returns:
            dict with keys:
                s_t      : selection probability  scalar
                logit    : raw logit              scalar
                use_full : bool decision          scalar
        """
        # Encode history
        h_emb = self.hist_encoder(history)   # [128]

        # Concatenate all inputs
        x = torch.cat([
            z_t.float(),
            h_emb.float(),
            task_id.float()
        ], dim=-1)                            # [latent_dim+128+n_tasks]

        # Predict logit
        logit = self.mlp(x).squeeze(-1)      # scalar

        # Gumbel-Softmax sample
        y = gumbel_softmax_sample(
            logit, temperature=self.temp, hard=hard)
        s_t = y[1]   # probability of full dynamics

        # Hard decision for inference
        use_full = (s_t > 0.5).float()

        # Track statistics
        self._decisions.append(s_t.item())
        self._total      += 1
        self._full_count += int(use_full.item())

        return {
            "s_t":      s_t,
            "logit":    logit,
            "use_full": use_full,
            "temp":     torch.tensor(self.temp),
        }

    def selection_rate(self) -> float:
        """Fraction of timesteps where full dynamics selected."""
        if self._total == 0:
            return 0.0
        return self._full_count / self._total

    def reset_stats(self):
        self._decisions  = []
        self._full_count = 0
        self._total      = 0

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


# =============================================================
# Computation Cost Model
# =============================================================
class ComputationCostModel:
    """
    Models computational cost of full vs approximate dynamics.
    Used in the bi-level optimisation objective (Eq. 9-10).
    """

    def __init__(
        self,
        c_full:   float = 1.0,
        c_approx: float = 0.1,
        alpha:    float = 0.01,
    ):
        self.c_full   = c_full
        self.c_approx = c_approx
        self.alpha    = alpha

    def cost(self, s_t: torch.Tensor) -> torch.Tensor:
        """
        Differentiable cost at timestep t.
        cost = s_t * c_full + (1 - s_t) * c_approx
        """
        return s_t * self.c_full + (1.0 - s_t) * self.c_approx

    def episode_cost(
        self, selections: torch.Tensor
    ) -> torch.Tensor:
        """
        Total cost over an episode.
        selections: [T]  sequence of s_t values
        """
        return self.cost(selections).sum()

    def weighted_loss(
        self,
        task_loss:  torch.Tensor,
        selections: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Combined objective: L_task + alpha * Cost
        Equation (9) in paper.
        """
        comp_cost = self.episode_cost(selections)
        total     = task_loss + self.alpha * comp_cost
        return {
            "total":     total,
            "task":      task_loss,
            "comp_cost": comp_cost,
        }


# =============================================================
# Self-test
# =============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  Adaptive Selector - Self Test")
    print("=" * 55)

    nv          = 6
    latent_dim  = 512
    history_len = 10
    n_tasks     = 3

    selector = AdaptiveSelector(
        nv=nv,
        latent_dim=latent_dim,
        history_len=history_len,
        hidden_dim=128,
        n_tasks=n_tasks,
    )
    print(f"\n  Selector parameters: {selector.count_parameters():,}")

    # ── Test 1: Single forward pass ──────────────────────────
    print(f"\n--- Test 1: Single forward pass ---")
    torch.manual_seed(42)

    z_t     = torch.randn(latent_dim)
    history = torch.randn(history_len, 3 * nv)
    task_id = F.one_hot(torch.tensor(0), n_tasks).float()

    out = selector(z_t, history, task_id, hard=False)
    print(f"  s_t      : {out['s_t'].item():.4f}")
    print(f"  logit    : {out['logit'].item():.4f}")
    print(f"  use_full : {out['use_full'].item():.0f}")
    print(f"  temp     : {out['temp'].item():.4f}")

    # ── Test 2: Gradient flow ─────────────────────────────────
    print(f"\n--- Test 2: Gradient flow ---")
    z_t2    = torch.randn(latent_dim, requires_grad=True)
    out2    = selector(z_t2, history, task_id, hard=False)
    loss2   = out2["s_t"]
    loss2.backward()
    print(f"  Gradient flows to z_t: "
          f"{z_t2.grad is not None}")
    print(f"  grad norm: {z_t2.grad.norm().item():.6f}")

    # ── Test 3: Temperature annealing ────────────────────────
    print(f"\n--- Test 3: Temperature annealing ---")
    temps = []
    for step in [0, 250, 500, 750, 1000]:
        selector.anneal_temperature(step, 1000)
        temps.append((step, round(selector.temp, 4)))
    for s, t in temps:
        print(f"  step={s:4d}  temp={t:.4f}")

    # ── Test 4: Episode simulation ───────────────────────────
    print(f"\n--- Test 4: Episode simulation (50 steps) ---")
    selector.reset_stats()
    selector.temp = 1.0   # reset temperature

    selections = []
    for t in range(50):
        z    = torch.randn(latent_dim)
        hist = torch.randn(history_len, 3 * nv)
        tid  = F.one_hot(torch.tensor(0), n_tasks).float()
        out  = selector(z, hist, tid, hard=True)
        selections.append(out["s_t"])

    selections_t = torch.stack(selections)
    print(f"  Full dynamics rate : "
          f"{selector.selection_rate():.2%}")
    print(f"  Mean s_t           : "
          f"{selections_t.mean().item():.4f}")
    print(f"  Std  s_t           : "
          f"{selections_t.std().item():.4f}")

    # ── Test 5: Cost model ───────────────────────────────────
    print(f"\n--- Test 5: Computation cost model ---")
    cost_model = ComputationCostModel(
        c_full=1.0, c_approx=0.1, alpha=0.01)

    task_loss = torch.tensor(2.5)
    result    = cost_model.weighted_loss(task_loss, selections_t)

    print(f"  Task loss    : {result['task'].item():.4f}")
    print(f"  Compute cost : {result['comp_cost'].item():.4f}")
    print(f"  Total loss   : {result['total'].item():.4f}")
    print(f"  Savings vs always-full: "
          f"{(1 - selector.selection_rate()) * 100:.1f}%")

    # ── Test 6: Consistency check ────────────────────────────
    print(f"\n--- Test 6: Hard vs Soft decisions ---")
    selector.reset_stats()
    selector.temp = 0.1   # low temperature -> more decisive

    soft_vals, hard_vals = [], []
    for _ in range(20):
        z    = torch.randn(latent_dim)
        hist = torch.randn(history_len, 3 * nv)
        tid  = F.one_hot(torch.tensor(1), n_tasks).float()

        out_s = selector(z, hist, tid, hard=False)
        out_h = selector(z, hist, tid, hard=True)
        soft_vals.append(out_s["s_t"].item())
        hard_vals.append(out_h["use_full"].item())

    print(f"  Soft mean s_t    : {np.mean(soft_vals):.4f}")
    print(f"  Hard full-dyn %  : {np.mean(hard_vals):.2%}")

    print(f"\n{'=' * 55}")
    print(f"  All tests passed!")
    print(f"{'=' * 55}")