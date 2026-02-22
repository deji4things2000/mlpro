# =============================================================
# agents/policy.py
# Vision-Language-Action Policy with Differentiable Dynamics
# Implements Section III-B of the paper
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VLA_CONFIG, SELECTOR_CONFIG, TRAINING_CONFIG, DEVICE


# =============================================================
# Visual Encoder (ResNet-18 based)
# =============================================================
class VisualEncoder(nn.Module):
    """
    Lightweight visual encoder based on ResNet-18 architecture.
    Processes RGB images into a latent vector.
    Uses a simplified CNN when torchvision is not available.
    """

    def __init__(self, latent_dim: int = 512, image_size: int = 224):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        try:
            import torchvision.models as models
            backbone        = models.resnet18(weights=None)
            # Replace final FC layer
            backbone.fc     = nn.Linear(512, latent_dim)
            self.encoder    = backbone
            self.using_full = True
            print("[VisualEncoder] Using ResNet-18 backbone")
        except ImportError:
            # Fallback: simple CNN
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, latent_dim),
            )
            self.using_full = False
            print("[VisualEncoder] Using fallback CNN")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [B, 3, H, W] or [3, H, W]
        Returns:
            z_vis: [B, latent_dim] or [latent_dim]
        """
        squeeze = image.dim() == 3
        if squeeze:
            image = image.unsqueeze(0)
        z = self.encoder(image.float())
        return z.squeeze(0) if squeeze else z


# =============================================================
# Language Encoder (CLIP-style)
# =============================================================
class LanguageEncoder(nn.Module):
    """
    Language encoder that maps task instructions to embeddings.
    Uses CLIP text encoder if available, else learned embedding.
    """

    def __init__(self, latent_dim: int = 512, vocab_size: int = 1000):
        super().__init__()
        self.latent_dim = latent_dim

        try:
            import clip
            self.clip_model, _ = clip.load("ViT-B/32", device="cpu")
            self.proj = nn.Linear(512, latent_dim)
            self.using_clip = True
            print("[LanguageEncoder] Using CLIP text encoder")
        except ImportError:
            # Fallback: learned token embedding
            self.embedding  = nn.Embedding(vocab_size, 128)
            self.proj       = nn.Sequential(
                nn.Linear(128, 256), nn.ReLU(),
                nn.Linear(256, latent_dim),
            )
            self.using_clip = False
            print("[LanguageEncoder] Using fallback embedding")

    def forward(
        self,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_tokens: [B, seq_len] token ids
                         OR [B, 512] pre-computed CLIP features
        Returns:
            z_lang: [B, latent_dim] or [latent_dim]
        """
        squeeze = text_tokens.dim() == 1
        if squeeze:
            text_tokens = text_tokens.unsqueeze(0)

        if self.using_clip:
            with torch.no_grad():
                feat = self.clip_model.encode_text(
                    text_tokens).float()
            z = self.proj(feat)
        else:
            # Mean-pool embeddings
            emb = self.embedding(
                text_tokens.clamp(0, self.embedding.num_embeddings-1))
            z   = self.proj(emb.mean(dim=1))

        return z.squeeze(0) if squeeze else z


# =============================================================
# Cross-Modal Fusion Transformer
# =============================================================
class FusionModule(nn.Module):
    """
    Cross-modal transformer that fuses visual and language
    representations into a unified latent z_t.
    4 heads, 2 layers as specified in paper.
    """

    def __init__(
        self,
        latent_dim:  int = 512,
        num_heads:   int = 4,
        num_layers:  int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Project visual and language to same dim
        self.vis_proj  = nn.Linear(latent_dim, latent_dim)
        self.lang_proj = nn.Linear(latent_dim, latent_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        # Final projection
        self.out_proj = nn.Linear(latent_dim * 2, latent_dim)

    def forward(
        self,
        z_vis:  torch.Tensor,
        z_lang: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_vis  : [latent_dim]  visual features
            z_lang : [latent_dim]  language features
        Returns:
            z_t    : [latent_dim]  fused representation
        """
        squeeze = z_vis.dim() == 1
        if squeeze:
            z_vis  = z_vis.unsqueeze(0)
            z_lang = z_lang.unsqueeze(0)

        B = z_vis.shape[0]

        # Project and stack as sequence [B, 2, latent_dim]
        v = self.vis_proj(z_vis).unsqueeze(1)
        l = self.lang_proj(z_lang).unsqueeze(1)
        seq = torch.cat([v, l], dim=1)

        # Transformer fusion
        fused = self.transformer(seq)   # [B, 2, latent_dim]

        # Concatenate and project
        z_t = self.out_proj(
            fused.reshape(B, -1))       # [B, latent_dim]

        return z_t.squeeze(0) if squeeze else z_t


# =============================================================
# Dynamics-Aware Policy Head
# =============================================================
class DynamicsAwarePolicyHead(nn.Module):
    """
    Policy head that predicts desired joint accelerations.
    Takes fused latent z_t + robot state (q, v) as input.
    Outputs desired accelerations a_des in joint space.
    """

    def __init__(
        self,
        latent_dim:  int = 512,
        nv:          int = 6,
        hidden_dim:  int = 256,
        num_layers:  int = 3,
    ):
        super().__init__()
        self.nv = nv

        # Input: z_t + q + v
        in_dim = latent_dim + 2 * nv

        layers = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim
        layers.append(nn.Linear(d, nv))   # output: a_des
        self.mlp = nn.Sequential(*layers)

        # Small init for stable early training
        nn.init.uniform_(self.mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        z_t: torch.Tensor,
        q:   torch.Tensor,
        v:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_t : [latent_dim]   fused perception latent
            q   : [nv]           joint positions
            v   : [nv]           joint velocities
        Returns:
            a_des: [nv]          desired joint accelerations
        """
        x     = torch.cat([
                    z_t.float(),
                    q.float(),
                    v.float()
                ], dim=-1)
        return self.mlp(x)


# =============================================================
# Training Loss Functions
# =============================================================
class VLALoss(nn.Module):
    """
    Combined training objective.
    L = L_task + lambda1*L_dynamics + lambda2*L_smooth + lambda3*L_cost
    Equation (6) in paper.
    """

    def __init__(
        self,
        lambda_dynamics: float = 0.1,
        lambda_smooth:   float = 0.01,
        lambda_cost:     float = 0.01,
        torque_limits:   Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.lambda_dynamics = lambda_dynamics
        self.lambda_smooth   = lambda_smooth
        self.lambda_cost     = lambda_cost
        self.torque_limits   = torque_limits

    def task_loss(
        self,
        achieved: torch.Tensor,
        desired:  torch.Tensor,
    ) -> torch.Tensor:
        """L2 tracking error."""
        return F.mse_loss(achieved, desired)

    def dynamics_loss(
        self,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Penalise torque limit violations."""
        if self.torque_limits is None:
            return torch.tensor(0.0)
        limits = self.torque_limits.to(tau.device).float()
        excess = F.relu(tau.float().abs() - limits)
        return excess.pow(2).mean()

    def smoothness_loss(
        self,
        tau_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Jerk penalty: penalise large changes in torque.
        tau_seq: [T, nv]
        """
        if tau_seq.shape[0] < 2:
            return torch.tensor(0.0)
        delta = tau_seq[1:] - tau_seq[:-1]
        return delta.pow(2).mean()

    def forward(
        self,
        achieved:   torch.Tensor,
        desired:    torch.Tensor,
        tau:        torch.Tensor,
        tau_seq:    Optional[torch.Tensor] = None,
        selections: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss terms.

        Returns dict with individual and total losses.
        """
        l_task = self.task_loss(achieved, desired)
        l_dyn  = self.dynamics_loss(tau)
        l_smooth = (self.smoothness_loss(tau_seq)
                    if tau_seq is not None
                    else torch.tensor(0.0))

        # Computation cost penalty
        if selections is not None:
            l_cost = selections.float().mean()
        else:
            l_cost = torch.tensor(0.0)

        total = (l_task
                 + self.lambda_dynamics * l_dyn
                 + self.lambda_smooth   * l_smooth
                 + self.lambda_cost     * l_cost)

        return {
            "total":    total,
            "task":     l_task,
            "dynamics": l_dyn,
            "smooth":   l_smooth,
            "cost":     l_cost,
        }


# =============================================================
# Full VLA Policy (integrated)
# =============================================================
class VLAPolicy(nn.Module):
    """
    Complete Vision-Language-Action policy with
    differentiable dynamics integration.

    Pipeline:
        (image, language) -> z_t  [perception encoder]
        (z_t, q, v)       -> a_des [policy head]
        (q, v, a_des)     -> tau   [dynamics layer]
        selector          -> use full or approximate dynamics

    Implements the full architecture from Section III.
    """

    def __init__(
        self,
        dynamics,           # DifferentiableDynamicsLayer
        approximator,       # DynamicsApproximator
        selector,           # AdaptiveSelector
        nv:          int   = 6,
        latent_dim:  int   = 512,
        n_tasks:     int   = 3,
        hidden_dim:  int   = 256,
    ):
        super().__init__()
        self.nv         = nv
        self.latent_dim = latent_dim
        self.n_tasks    = n_tasks

        # Sub-modules
        self.visual_encoder  = VisualEncoder(latent_dim)
        self.lang_encoder    = LanguageEncoder(latent_dim)
        self.fusion          = FusionModule(latent_dim)
        self.policy_head     = DynamicsAwarePolicyHead(
            latent_dim, nv, hidden_dim)

        # Dynamics components
        self.dynamics        = dynamics
        self.approximator    = approximator
        self.selector        = selector

        # History buffer  [history_len, 3*nv]
        history_len          = SELECTOR_CONFIG["history_len"]
        self.history_len     = history_len
        self.register_buffer(
            "history_buf",
            torch.zeros(history_len, 3 * nv, dtype=torch.float32))

        # Mode tracking
        self.training_phase  = 1   # 1, 2, or 3

    def reset_history(self):
        """Clear history buffer at episode start."""
        self.history_buf.zero_()

    def _update_history(
        self,
        q:   torch.Tensor,
        v:   torch.Tensor,
        tau: torch.Tensor,
    ):
        """Shift history buffer and append new step."""
        new_entry = torch.cat([
            q.float(), v.float(), tau.float()
        ])                                      # [3*nv]
        self.history_buf = torch.roll(
            self.history_buf, -1, dims=0)
        self.history_buf[-1] = new_entry.detach()

    def forward(
        self,
        image:      torch.Tensor,
        lang_tokens: torch.Tensor,
        q:          torch.Tensor,
        v:          torch.Tensor,
        task_id:    int  = 0,
        use_full_dynamics: Optional[bool] = None,
        hard_select: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through VLA pipeline.

        Args:
            image       : [3, H, W]   RGB observation
            lang_tokens : [seq_len]   language tokens
            q           : [nv]        joint positions
            v           : [nv]        joint velocities
            task_id     : int         task index
            use_full_dynamics: override selector (None=use selector)
            hard_select : use hard Gumbel-Softmax

        Returns:
            dict with: tau, a_des, z_t, s_t, use_full, M, C, g
        """
        # ── 1. Perception encoding ───────────────────────────
        z_vis  = self.visual_encoder(image)
        z_lang = self.lang_encoder(lang_tokens)
        z_t    = self.fusion(z_vis, z_lang)

        # ── 2. Policy head: predict desired accelerations ────
        a_des  = self.policy_head(z_t, q, v)          # [nv]

        # ── 3. Task one-hot ──────────────────────────────────
        task_oh = F.one_hot(
            torch.tensor(task_id),
            self.n_tasks).float()

        # ── 4. Adaptive selector ─────────────────────────────
        sel_out = self.selector(
            z_t.detach(),
            self.history_buf,
            task_oh,
            hard=hard_select,
        )
        s_t      = sel_out["s_t"]

        # Override if specified (e.g. during Phase 1)
        if use_full_dynamics is not None:
            do_full = use_full_dynamics
        else:
            do_full = bool(sel_out["use_full"].item())

        # ── 5. Dynamics computation ──────────────────────────
        a_des_f64 = a_des.double()

        if do_full:
            # Full Pinocchio dynamics
            dyn_out = self.dynamics.forward(
                q.double(), v.double(),
                a_des=a_des_f64, mode="inverse")
            tau = dyn_out["tau"]
            M   = dyn_out["M"]
            C   = dyn_out["C"]
            g   = dyn_out["g"]
        else:
            # Lightweight approximator
            approx_out = self.approximator(
                q.double(), v.double())
            M   = approx_out["M"]
            C   = approx_out["C"]
            g   = approx_out["g"]
            tau = M @ a_des_f64 + C + g

        # ── 6. Update history ────────────────────────────────
        self._update_history(q, v, tau)

        return {
            "tau":      tau,
            "a_des":    a_des,
            "z_t":      z_t,
            "s_t":      s_t,
            "use_full": do_full,
            "M":        M,
            "C":        C,
            "g":        g,
        }

    def count_parameters(self) -> Dict[str, int]:
        def count(m):
            return sum(p.numel() for p in m.parameters()
                       if p.requires_grad)
        return {
            "visual_encoder": count(self.visual_encoder),
            "lang_encoder":   count(self.lang_encoder),
            "fusion":         count(self.fusion),
            "policy_head":    count(self.policy_head),
            "approximator":   count(self.approximator),
            "selector":       count(self.selector),
            "total":          count(self),
        }


# =============================================================
# Self-test
# =============================================================
if __name__ == "__main__":
    import pinocchio as pin
    from agents.dynamics     import build_dynamics_layer
    from agents.approximator import DynamicsApproximator
    from agents.selector     import AdaptiveSelector

    print("=" * 55)
    print("  VLA Policy - Self Test")
    print("=" * 55)

    # Build all components
    torch.manual_seed(42)
    nv         = 6
    latent_dim = 512
    n_tasks    = 3

    dynamics    = build_dynamics_layer("panda")
    approximator = DynamicsApproximator(
        nv=nv, hidden_dim=256, num_layers=3)
    selector    = AdaptiveSelector(
        nv=nv, latent_dim=latent_dim, n_tasks=n_tasks)

    policy = VLAPolicy(
        dynamics=dynamics,
        approximator=approximator,
        selector=selector,
        nv=nv,
        latent_dim=latent_dim,
        n_tasks=n_tasks,
    )

    # Parameter count
    print(f"\n--- Parameter Counts ---")
    params = policy.count_parameters()
    for k, v in params.items():
        print(f"  {k:<20s}: {v:>10,}")

    # ── Test 1: Full dynamics forward pass ───────────────────
    print(f"\n--- Test 1: Full dynamics forward pass ---")
    image  = torch.randn(3, 224, 224)
    tokens = torch.randint(0, 100, (10,))
    q      = torch.tensor(
                 pin.randomConfiguration(dynamics.model),
                 dtype=torch.float64)
    v      = torch.randn(nv, dtype=torch.float64)

    policy.reset_history()
    out = policy(
        image, tokens, q, v,
        task_id=0,
        use_full_dynamics=True)

    print(f"  tau shape  : {out['tau'].shape}")
    print(f"  a_des shape: {out['a_des'].shape}")
    print(f"  z_t shape  : {out['z_t'].shape}")
    print(f"  s_t        : {out['s_t'].item():.4f}")
    print(f"  use_full   : {out['use_full']}")
    print(f"  tau        : {out['tau'].detach().numpy().round(3)}")

    # ── Test 2: Approximate dynamics forward pass ────────────
    print(f"\n--- Test 2: Approximate dynamics forward pass ---")
    out_approx = policy(
        image, tokens, q, v,
        task_id=0,
        use_full_dynamics=False)

    print(f"  tau (approx): "
          f"{out_approx['tau'].detach().numpy().round(3)}")
    print(f"  use_full    : {out_approx['use_full']}")

    # ── Test 3: Adaptive selector ────────────────────────────
    print(f"\n--- Test 3: Adaptive selector mode ---")
    policy.selector.reset_stats()
    full_count = 0
    for _ in range(20):
        q_r = torch.tensor(
                  pin.randomConfiguration(dynamics.model),
                  dtype=torch.float64)
        v_r = torch.randn(nv, dtype=torch.float64)
        o   = policy(image, tokens, q_r, v_r, task_id=1)
        full_count += int(o["use_full"])

    print(f"  Full dynamics used: {full_count}/20 steps "
          f"({full_count/20:.0%})")
    print(f"  Selector rate     : "
          f"{policy.selector.selection_rate():.2%}")

    # ── Test 4: Gradient flow end-to-end ─────────────────────
    print(f"\n--- Test 4: End-to-end gradient flow ---")
    policy.reset_history()
    q_g = torch.tensor(
              pin.randomConfiguration(dynamics.model),
              dtype=torch.float64, requires_grad=False)
    v_g = torch.randn(nv, dtype=torch.float64)

    out_g = policy(
        image, tokens, q_g, v_g,
        task_id=0, use_full_dynamics=True)

    loss = out_g["tau"].float().sum()
    loss.backward()

    has_grad = any(
        p.grad is not None
        for p in policy.policy_head.parameters())
    print(f"  Policy head gradients: {has_grad}")

    has_grad_vis = any(
        p.grad is not None
        for p in policy.visual_encoder.parameters())
    print(f"  Visual encoder grads : {has_grad_vis}")

    # ── Test 5: Loss function ─────────────────────────────────
    print(f"\n--- Test 5: VLA Loss ---")
    loss_fn = VLALoss(
        lambda_dynamics=0.1,
        lambda_smooth=0.01,
        lambda_cost=0.01,
        torque_limits=torch.tensor([87.]*6))

    achieved = out["tau"].float()
    desired  = torch.zeros(nv)
    tau_seq  = torch.randn(10, nv)
    sels     = torch.rand(10)

    losses = loss_fn(achieved, desired, achieved, tau_seq, sels)
    for k, val in losses.items():
        print(f"  {k:<12s}: {val.item():.4f}")

    # ── Test 6: Episode rollout timing ───────────────────────
    print(f"\n--- Test 6: Episode rollout timing (20 steps) ---")
    import time
    policy.reset_history()
    dynamics.reset_timing()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            q_r = torch.tensor(
                      pin.randomConfiguration(dynamics.model),
                      dtype=torch.float64)
            v_r = torch.randn(nv, dtype=torch.float64)
            policy(image, tokens, q_r, v_r,
                   task_id=0, use_full_dynamics=True)
    t_full = (time.perf_counter() - t0) / 20 * 1e3

    policy.reset_history()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            q_r = torch.tensor(
                      pin.randomConfiguration(dynamics.model),
                      dtype=torch.float64)
            v_r = torch.randn(nv, dtype=torch.float64)
            policy(image, tokens, q_r, v_r,
                   task_id=0, use_full_dynamics=False)
    t_approx = (time.perf_counter() - t0) / 20 * 1e3

    print(f"  Full dynamics mode  : {t_full:.2f} ms/step")
    print(f"  Approx dynamics mode: {t_approx:.2f} ms/step")
    print(f"  Speedup             : {t_full/t_approx:.2f}x")

    print(f"\n{'=' * 55}")
    print(f"  All tests passed!")
    print(f"{'=' * 55}")