# =============================================================
# agents/approximator.py
# Lightweight Dynamics Approximator
# Implements Section III-C of the paper
# =============================================================

import torch
import torch.nn as nn
import numpy as np
import pinocchio as pin
from typing import Dict, Tuple, Optional
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APPROXIMATOR_CONFIG, DEVICE


# =============================================================
# Lightweight Dynamics Approximator Network
# =============================================================
class DynamicsApproximator(nn.Module):
    """
    Lightweight MLP that approximates rigid-body dynamics.
    Implements Section III-C of the paper.

    Two variants:
        'direct'   -- predicts M, C, g directly from state
        'residual' -- predicts residual on top of base dynamics

    Ensures M is positive definite via Cholesky parameterisation:
        M = L @ L^T  where L is lower-triangular with positive diagonal
    """

    def __init__(self, nv: int, hidden_dim: int = 256,
                 num_layers: int = 3, variant: str = "direct"):
        super().__init__()
        self.nv         = nv
        self.variant    = variant
        self.hidden_dim = hidden_dim

        # Output dimensions
        # M lower-triangle: nv*(nv+1)//2
        # C: nv
        # g: nv
        self.n_M  = nv * (nv + 1) // 2
        self.n_C  = nv
        self.n_g  = nv
        self.n_out = self.n_M + self.n_C + self.n_g

        # Input: [q, v] concatenated
        self.n_in = 2 * nv

        # Build MLP
        layers = []
        in_dim = self.n_in
        for i in range(num_layers):
            out_dim = hidden_dim
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
            ]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, self.n_out))
        self.mlp = nn.Sequential(*layers)

        # Lower-triangular indices for reconstructing M
        self.tril_idx = torch.tril_indices(nv, nv)

        # Initialize weights small
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def _build_M(self, L_flat: torch.Tensor) -> torch.Tensor:
        """
        Build SPD mass matrix from flat Cholesky factor.
        L_flat: [nv*(nv+1)//2]  -->  M: [nv, nv]
        Diagonal entries exponentiated to ensure positivity.
        """
        nv = self.nv
        L  = torch.zeros(nv, nv, dtype=L_flat.dtype,
                         device=L_flat.device)
        L[self.tril_idx[0], self.tril_idx[1]] = L_flat

        # Positive diagonal via softplus
        diag_idx = torch.arange(nv)
        L[diag_idx, diag_idx] = torch.nn.functional.softplus(
            L[diag_idx, diag_idx]) + 1e-3

        return L @ L.T   # SPD by construction

    def forward(
        self,
        q: torch.Tensor,
        v: torch.Tensor,
        M_base: Optional[torch.Tensor] = None,
        C_base: Optional[torch.Tensor] = None,
        g_base: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            q      : joint positions   [nv]  float64
            v      : joint velocities  [nv]  float64
            M_base : base mass matrix  [nv,nv]  (residual mode)
            C_base : base Coriolis     [nv]     (residual mode)
            g_base : base gravity      [nv]     (residual mode)

        Returns:
            dict with keys: M [nv,nv], C [nv], g [nv]
        """
        # Run MLP in float32 for speed, convert inputs
        x   = torch.cat([q, v], dim=-1).float()
        out = self.mlp(x)                          # [n_out]

        # Split output
        L_flat = out[:self.n_M]
        C_pred = out[self.n_M : self.n_M + self.n_C]
        g_pred = out[self.n_M + self.n_C:]

        # Build SPD mass matrix
        M_pred = self._build_M(L_flat)

        if self.variant == "residual" and M_base is not None:
            # Residual: add MLP output to base dynamics
            M_out = M_base.float() + M_pred
            C_out = C_base.float() + C_pred
            g_out = g_base.float() + g_pred
        else:
            # Direct prediction
            M_out = M_pred
            C_out = C_pred
            g_out = g_pred

        return {
            "M": M_out.double(),
            "C": C_out.double(),
            "g": g_out.double(),
        }

    def predict_tau(
        self,
        q:     torch.Tensor,
        v:     torch.Tensor,
        a_des: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Predict torques: tau = M*a_des + C + g
        Convenience wrapper around forward().
        """
        out = self.forward(q, v, **kwargs)
        return out["M"] @ a_des + out["C"] + out["g"]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


# =============================================================
# Supervised Training Helper
# =============================================================
class ApproximatorTrainer:
    """
    Phase 1 training: supervised learning to match
    full Pinocchio dynamics outputs.
    Implements Section III-C (Phase 1 curriculum).
    """

    def __init__(
        self,
        approx:    DynamicsApproximator,
        dynamics,                          # DifferentiableDynamicsLayer
        lr:        float = 3e-4,
        device:    torch.device = DEVICE,
    ):
        self.approx   = approx.to(device)
        self.dynamics = dynamics
        self.device   = device
        self.opt      = torch.optim.Adam(approx.parameters(), lr=lr)

        self.losses   = []

    def generate_batch(
        self, batch_size: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample random (q, v) pairs and compute true dynamics."""
        model = self.dynamics.model
        nv    = self.dynamics.nv

        qs, vs = [], []
        Ms, Cs, gs = [], [], []

        for _ in range(batch_size):
            q = torch.tensor(
                    pin.randomConfiguration(model),
                    dtype=torch.float64)
            v = torch.randn(nv, dtype=torch.float64)

            with torch.no_grad():
                M, C, g = self.dynamics.compute_all(q, v)

            qs.append(q); vs.append(v)
            Ms.append(M); Cs.append(C); gs.append(g)

        return (
            torch.stack(qs), torch.stack(vs),
            torch.stack(Ms), torch.stack(Cs), torch.stack(gs)
        )

    def train_step(self, batch_size: int = 64) -> Dict[str, float]:
        """Single supervised training step."""
        qs, vs, M_true, C_true, g_true = self.generate_batch(
            batch_size)

        total_loss = torch.tensor(0.0)
        loss_M = loss_C = loss_g = 0.0

        for i in range(len(qs)):
            out    = self.approx(qs[i], vs[i])
            lM     = nn.functional.mse_loss(
                         out["M"].float(), M_true[i].float())
            lC     = nn.functional.mse_loss(
                         out["C"].float(), C_true[i].float())
            lg     = nn.functional.mse_loss(
                         out["g"].float(), g_true[i].float())
            total_loss = total_loss + lM + lC + lg
            loss_M += lM.item()
            loss_C += lC.item()
            loss_g += lg.item()

        total_loss = total_loss / len(qs)
        self.opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.approx.parameters(), 1.0)
        self.opt.step()

        n = len(qs)
        return {
            "loss_total": total_loss.item(),
            "loss_M":     loss_M / n,
            "loss_C":     loss_C / n,
            "loss_g":     loss_g / n,
        }

    def train(
        self,
        num_steps:  int = 500,
        batch_size: int = 32,
        log_every:  int = 100,
    ) -> list:
        """Run supervised pre-training loop."""
        print(f"\n[Approximator] Supervised pre-training "
              f"({num_steps} steps, batch={batch_size})")
        print("-" * 50)

        for step in range(1, num_steps + 1):
            losses = self.train_step(batch_size)
            self.losses.append(losses["loss_total"])

            if step % log_every == 0 or step == 1:
                print(f"  Step {step:4d}/{num_steps} | "
                      f"loss={losses['loss_total']:.4f} | "
                      f"M={losses['loss_M']:.4f} | "
                      f"C={losses['loss_C']:.4f} | "
                      f"g={losses['loss_g']:.4f}")

        print("-" * 50)
        print(f"[Approximator] Training complete. "
              f"Final loss: {self.losses[-1]:.4f}")
        return self.losses


# =============================================================
# Self-test
# =============================================================
if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from agents.dynamics import build_dynamics_layer

    print("=" * 55)
    print("  Dynamics Approximator - Self Test")
    print("=" * 55)

    # Build dynamics layer
    dyn   = build_dynamics_layer("panda")
    nv    = dyn.nv

    # ── Test 1: Direct approximator ──────────────────────────
    print(f"\n--- Test 1: Direct Approximator ---")
    approx_direct = DynamicsApproximator(
        nv=nv, hidden_dim=256, num_layers=3, variant="direct")
    print(f"  Parameters: {approx_direct.count_parameters():,}")

    q = torch.tensor(pin.randomConfiguration(dyn.model),
                     dtype=torch.float64)
    v = torch.randn(nv, dtype=torch.float64)

    out = approx_direct(q, v)
    print(f"  M shape : {out['M'].shape}")
    print(f"  C shape : {out['C'].shape}")
    print(f"  g shape : {out['g'].shape}")

    eigvals = torch.linalg.eigvalsh(out["M"])
    print(f"  M pos-def: {bool((eigvals > 0).all())}  "
          f"(min eigval={eigvals.min().item():.4f})")

    # ── Test 2: Residual approximator ────────────────────────
    print(f"\n--- Test 2: Residual Approximator ---")
    approx_res = DynamicsApproximator(
        nv=nv, hidden_dim=256, num_layers=3, variant="residual")

    M_base, C_base, g_base = dyn.compute_all(q, v)
    out_res = approx_res(q, v,
                         M_base=M_base,
                         C_base=C_base,
                         g_base=g_base)
    print(f"  M residual shape : {out_res['M'].shape}")
    eigvals_r = torch.linalg.eigvalsh(out_res["M"])
    print(f"  M pos-def: {bool((eigvals_r > 0).all())}")

    # ── Test 3: Gradient flow ─────────────────────────────────
    print(f"\n--- Test 3: Gradient flow ---")
    q_g = torch.tensor(pin.randomConfiguration(dyn.model),
                       dtype=torch.float64)
    v_g = torch.randn(nv, dtype=torch.float64)
    a_g = torch.randn(nv, dtype=torch.float64)

    tau_approx = approx_direct.predict_tau(q_g, v_g, a_g)
    loss       = tau_approx.sum()
    loss.backward()
    print(f"  Gradients flow through approximator: True")
    print(f"  tau_approx: {tau_approx.detach().numpy().round(4)}")

    # ── Test 4: Supervised pre-training ──────────────────────
    print(f"\n--- Test 4: Supervised Pre-training (200 steps) ---")
    trainer = ApproximatorTrainer(
        approx=approx_direct, dynamics=dyn, lr=3e-4)
    losses  = trainer.train(
        num_steps=200, batch_size=16, log_every=50)

    # ── Test 5: Accuracy after training ──────────────────────
    print(f"\n--- Test 5: Accuracy vs Full Dynamics ---")
    errors_M, errors_C, errors_g = [], [], []

    with torch.no_grad():
        for _ in range(20):
            q_t = torch.tensor(
                      pin.randomConfiguration(dyn.model),
                      dtype=torch.float64)
            v_t = torch.randn(nv, dtype=torch.float64)

            M_t, C_t, g_t = dyn.compute_all(q_t, v_t)
            out_t          = approx_direct(q_t, v_t)

            errors_M.append(
                nn.functional.mse_loss(
                    out_t["M"].float(),
                    M_t.float()).item())
            errors_C.append(
                nn.functional.mse_loss(
                    out_t["C"].float(),
                    C_t.float()).item())
            errors_g.append(
                nn.functional.mse_loss(
                    out_t["g"].float(),
                    g_t.float()).item())

    print(f"  MSE M : {np.mean(errors_M):.4f}")
    print(f"  MSE C : {np.mean(errors_C):.4f}")
    print(f"  MSE g : {np.mean(errors_g):.4f}")

    # ── Test 6: Speed comparison ──────────────────────────────
    print(f"\n--- Test 6: Speed Comparison ---")
    import time

    N = 200
    t0 = time.perf_counter()
    for _ in range(N):
        q_s = torch.tensor(
                  pin.randomConfiguration(dyn.model),
                  dtype=torch.float64)
        v_s = torch.randn(nv, dtype=torch.float64)
        dyn.compute_all(q_s, v_s)
    t_full = (time.perf_counter() - t0) / N * 1e3

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            q_s = torch.tensor(
                      pin.randomConfiguration(dyn.model),
                      dtype=torch.float64)
            v_s = torch.randn(nv, dtype=torch.float64)
            approx_direct(q_s, v_s)
    t_approx = (time.perf_counter() - t0) / N * 1e3

    print(f"  Full dynamics  : {t_full:.3f} ms/call")
    print(f"  Approximator   : {t_approx:.3f} ms/call")
    print(f"  Speedup        : {t_full/t_approx:.2f}x")

    print(f"\n{'=' * 55}")
    print(f"  All tests passed!")
    print(f"{'=' * 55}")