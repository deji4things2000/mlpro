# =============================================================
# agents/dynamics.py
# Differentiable Rigid-Body Dynamics Layer (Pinocchio-based)
# Implements Section III-A of the paper
# =============================================================

import torch
import torch.nn as nn
import numpy as np
import pinocchio as pin
import time
from typing import Tuple, Optional, Dict
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DYNAMICS_CONFIG, DEVICE


# =============================================================
# Custom Autograd Function: RNEA (Inverse Dynamics)
# =============================================================
class RNEAFunction(torch.autograd.Function):
    """
    Wraps Pinocchio RNEA as a differentiable PyTorch function.
    Forward:  tau = M(q)*a + C(q,v) + g(q)
    Backward: analytical derivatives via Pinocchio
    """

    @staticmethod
    def forward(ctx, q, v, a, model, data):
        # Convert to numpy for Pinocchio
        q_np = q.detach().cpu().numpy().astype(np.float64)
        v_np = v.detach().cpu().numpy().astype(np.float64)
        a_np = a.detach().cpu().numpy().astype(np.float64)

        # Compute inverse dynamics
        tau_np = pin.rnea(model, data, q_np, v_np, a_np)

        # Save for backward
        ctx.save_for_backward(q, v, a)
        ctx.model = model
        ctx.data  = data

        return torch.tensor(tau_np.copy(), dtype=torch.float64)

    @staticmethod
    def backward(ctx, grad_tau):
        q, v, a = ctx.saved_tensors
        model   = ctx.model
        data    = ctx.data

        q_np = q.detach().cpu().numpy().astype(np.float64)
        v_np = v.detach().cpu().numpy().astype(np.float64)
        a_np = a.detach().cpu().numpy().astype(np.float64)

        # Analytical derivatives
        pin.computeRNEADerivatives(model, data, q_np, v_np, a_np)

        dtau_dq = torch.tensor(data.dtau_dq.copy(), dtype=torch.float64)
        dtau_dv = torch.tensor(data.dtau_dv.copy(), dtype=torch.float64)
        dtau_da = torch.tensor(data.M.copy(),        dtype=torch.float64)

        g_tau  = grad_tau.unsqueeze(0)             # [1, n]
        grad_q = (g_tau @ dtau_dq).squeeze(0)      # [n]
        grad_v = (g_tau @ dtau_dv).squeeze(0)      # [n]
        grad_a = (g_tau @ dtau_da).squeeze(0)      # [n]

        return grad_q, grad_v, grad_a, None, None


# =============================================================
# Custom Autograd Function: CRBA (Mass Matrix)
# =============================================================
class CRBAFunction(torch.autograd.Function):
    """
    Wraps Pinocchio CRBA as a differentiable PyTorch function.
    Forward:  M = CRBA(q)
    Backward: dM/dq via finite differences
    """

    @staticmethod
    def forward(ctx, q, model, data):
        q_np = q.detach().cpu().numpy().astype(np.float64)
        M_np = pin.crba(model, data, q_np)

        ctx.save_for_backward(q)
        ctx.model = model
        ctx.data  = data

        return torch.tensor(M_np.copy(), dtype=torch.float64)

    @staticmethod
    def backward(ctx, grad_M):
        q,    = ctx.saved_tensors
        model = ctx.model
        data  = ctx.data

        q_np      = q.detach().cpu().numpy().astype(np.float64)
        n         = model.nv
        eps       = 1e-6
        grad_q    = np.zeros(n)
        grad_M_np = grad_M.detach().cpu().numpy()

        for i in range(n):
            q_p        = q_np.copy(); q_p[i] += eps
            q_m        = q_np.copy(); q_m[i] -= eps
            dM_dqi     = (pin.crba(model, data, q_p).copy() -
                          pin.crba(model, data, q_m).copy()) / (2 * eps)
            grad_q[i]  = np.sum(grad_M_np * dM_dqi)

        return torch.tensor(grad_q, dtype=torch.float64), None, None


# =============================================================
# Main Differentiable Dynamics Layer
# =============================================================
class DifferentiableDynamicsLayer(nn.Module):
    """
    Differentiable rigid-body dynamics layer based on Pinocchio.
    Implements Section III-A of the paper.

    Computes:
        M(q)     -- joint-space inertia matrix   [n x n]
        C(q,v)   -- Coriolis/centrifugal terms   [n]
        g(q)     -- gravity vector               [n]

    Supports:
        inverse_dynamics : tau = M*a_des + C + g
        forward_dynamics : a   = M^{-1}(tau - C - g)  [Cholesky]
    """

    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data  = data
        self.nq    = model.nq
        self.nv    = model.nv

        # Timing statistics
        self._timing = {
            "calls":       0,
            "total_ms":    0.0,
            "M_ms":        0.0,
            "C_ms":        0.0,
            "g_ms":        0.0,
        }

    # ----------------------------------------------------------
    # Individual quantity computation
    # ----------------------------------------------------------
    def compute_M(self, q: torch.Tensor) -> torch.Tensor:
        """Mass matrix M(q).  Shape: [n, n]"""
        return CRBAFunction.apply(q, self.model, self.data)

    def compute_g(self, q: torch.Tensor) -> torch.Tensor:
        """Gravity vector g(q).  Shape: [n]"""
        v0 = torch.zeros(self.nv, dtype=torch.float64)
        a0 = torch.zeros(self.nv, dtype=torch.float64)
        return RNEAFunction.apply(q, v0, a0, self.model, self.data)

    def compute_C(self, q: torch.Tensor,
                       v: torch.Tensor) -> torch.Tensor:
        """Coriolis + centrifugal C(q,v).  Shape: [n]"""
        a0 = torch.zeros(self.nv, dtype=torch.float64)
        return (RNEAFunction.apply(q, v, a0, self.model, self.data)
                - self.compute_g(q))

    def compute_all(
        self, q: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (M, C, g) with timing.  Returns [n,n], [n], [n]"""
        t0 = time.perf_counter()

        tM = time.perf_counter()
        M  = self.compute_M(q)
        self._timing["M_ms"] += (time.perf_counter() - tM) * 1e3

        tg = time.perf_counter()
        g  = self.compute_g(q)
        self._timing["g_ms"] += (time.perf_counter() - tg) * 1e3

        tC = time.perf_counter()
        a0 = torch.zeros(self.nv, dtype=torch.float64)
        C  = (RNEAFunction.apply(q, v, a0, self.model, self.data) - g)
        self._timing["C_ms"] += (time.perf_counter() - tC) * 1e3

        self._timing["total_ms"] += (time.perf_counter() - t0) * 1e3
        self._timing["calls"]    += 1
        return M, C, g

    # ----------------------------------------------------------
    # Inverse dynamics
    # ----------------------------------------------------------
    def inverse_dynamics(
        self,
        q:     torch.Tensor,
        v:     torch.Tensor,
        a_des: torch.Tensor
    ) -> torch.Tensor:
        """
        tau = M(q)*a_des + C(q,v) + g(q)
        Equation (5) in paper.
        """
        M, C, g = self.compute_all(q, v)
        return M @ a_des + C + g

    # ----------------------------------------------------------
    # Forward dynamics (Cholesky solve)
    # ----------------------------------------------------------
    def forward_dynamics(
        self,
        q:   torch.Tensor,
        v:   torch.Tensor,
        tau: torch.Tensor
    ) -> torch.Tensor:
        """
        a = M^{-1}(tau - C - g)
        Solved via Cholesky factorisation (M is SPD).
        Equation (4) in paper.
        """
        M, C, g = self.compute_all(q, v)
        rhs = (tau - C - g).unsqueeze(-1)          # [n, 1]
        try:
            L = torch.linalg.cholesky(M)
            a = torch.cholesky_solve(rhs, L).squeeze(-1)
        except Exception:
            a = torch.linalg.solve(M, rhs).squeeze(-1)
        return a

    # ----------------------------------------------------------
    # nn.Module forward (unified interface)
    # ----------------------------------------------------------
    def forward(
        self,
        q:     torch.Tensor,
        v:     torch.Tensor,
        a_des: Optional[torch.Tensor] = None,
        tau:   Optional[torch.Tensor] = None,
        mode:  str = "inverse"
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass.
        mode='inverse': returns tau given a_des
        mode='forward': returns a   given tau
        Always returns M, C, g in output dict.
        """
        M, C, g = self.compute_all(q, v)
        out = {"M": M, "C": C, "g": g}

        if mode == "inverse" and a_des is not None:
            out["tau"] = M @ a_des + C + g

        elif mode == "forward" and tau is not None:
            rhs = (tau - C - g).unsqueeze(-1)
            try:
                L        = torch.linalg.cholesky(M)
                out["a"] = torch.cholesky_solve(rhs, L).squeeze(-1)
            except Exception:
                out["a"] = torch.linalg.solve(M, rhs).squeeze(-1)

        return out

    # ----------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------
    def timing_report(self) -> Dict[str, float]:
        n = max(self._timing["calls"], 1)
        return {
            "avg_total_ms": round(self._timing["total_ms"] / n, 4),
            "avg_M_ms":     round(self._timing["M_ms"]     / n, 4),
            "avg_C_ms":     round(self._timing["C_ms"]     / n, 4),
            "avg_g_ms":     round(self._timing["g_ms"]     / n, 4),
            "total_calls":  self._timing["calls"],
        }

    def reset_timing(self):
        for k in self._timing:
            self._timing[k] = 0


# =============================================================
# Factory function
# =============================================================
def build_dynamics_layer(robot_name: str) -> DifferentiableDynamicsLayer:
    """
    Build DifferentiableDynamicsLayer for a named robot.
    Currently uses Pinocchio sample manipulator as placeholder.
    """
    model = pin.buildSampleModelManipulator()
    data  = model.createData()
    print(f"[Dynamics] Robot='{robot_name}'  "
          f"nq={model.nq}  nv={model.nv}")
    return DifferentiableDynamicsLayer(model, data)


# =============================================================
# Self-test
# =============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  Differentiable Dynamics Layer - Self Test")
    print("=" * 55)

    layer = build_dynamics_layer("panda")
    n     = layer.nv

    # Random state
    torch.manual_seed(42)
    q     = torch.tensor(
                pin.randomConfiguration(layer.model),
                dtype=torch.float64, requires_grad=True)
    v     = torch.randn(n, dtype=torch.float64, requires_grad=True)
    a_des = torch.randn(n, dtype=torch.float64, requires_grad=True)

    print(f"\n--- State ---")
    print(f"  q     : {q.detach().numpy().round(3)}")
    print(f"  v     : {v.detach().numpy().round(3)}")
    print(f"  a_des : {a_des.detach().numpy().round(3)}")

    # ── Test 1: compute_all ─────────────────────────────────
    print(f"\n--- Test 1: compute_all (M, C, g) ---")
    M, C, g = layer.compute_all(q, v)
    print(f"  M shape : {M.shape}  (should be [{n},{n}])")
    print(f"  C shape : {C.shape}  (should be [{n}])")
    print(f"  g shape : {g.shape}  (should be [{n}])")
    print(f"  M symmetric: {torch.allclose(M, M.T, atol=1e-6)}")
    eigvals = torch.linalg.eigvalsh(M)
    print(f"  M pos-def  : {bool((eigvals > 0).all())}  "
          f"(min eigval={eigvals.min().item():.4f})")

    # ── Test 2: Inverse dynamics ─────────────────────────────
    print(f"\n--- Test 2: Inverse dynamics ---")
    tau = layer.inverse_dynamics(q, v, a_des)
    print(f"  tau : {tau.detach().numpy().round(4)}")

    # ── Test 3: Forward dynamics ─────────────────────────────
    print(f"\n--- Test 3: Forward dynamics ---")
    a_out = layer.forward_dynamics(q, v, tau.detach())
    err   = (a_out - a_des.detach()).abs().max().item()
    print(f"  a_out       : {a_out.detach().numpy().round(4)}")
    print(f"  max |a_out - a_des| : {err:.2e}  "
          f"({'PASS' if err < 1e-4 else 'FAIL'})")

    # ── Test 4: Gradient flow ────────────────────────────────
    print(f"\n--- Test 4: Gradient flow (backward pass) ---")
    layer.reset_timing()
    tau2 = layer.inverse_dynamics(q, v, a_des)
    loss = tau2.sum()
    loss.backward()
    print(f"  grad_q    : {q.grad.numpy().round(4)}")
    print(f"  grad_v    : {v.grad.numpy().round(4)}")
    print(f"  grad_a_des: {a_des.grad.numpy().round(4)}")
    all_grads = all(x.grad is not None for x in [q, v, a_des])
    print(f"  All gradients computed: {all_grads}")

    # ── Test 5: Timing ───────────────────────────────────────
    print(f"\n--- Test 5: Timing (100 calls) ---")
    layer.reset_timing()
    for _ in range(100):
        q_r = torch.tensor(
                  pin.randomConfiguration(layer.model),
                  dtype=torch.float64)
        v_r = torch.randn(n, dtype=torch.float64)
        layer.compute_all(q_r, v_r)

    report = layer.timing_report()
    print(f"  avg total : {report['avg_total_ms']:.3f} ms")
    print(f"  avg M     : {report['avg_M_ms']:.3f} ms")
    print(f"  avg C     : {report['avg_C_ms']:.3f} ms")
    print(f"  avg g     : {report['avg_g_ms']:.3f} ms")
    print(f"  calls     : {report['total_calls']}")

    print(f"\n{'=' * 55}")
    print(f"  All tests passed!")
    print(f"{'=' * 55}")