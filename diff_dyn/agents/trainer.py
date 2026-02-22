# =============================================================
# agents/trainer.py
# Three-Phase Training Curriculum
# Implements Section III-D of the paper
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
from typing import Dict, List, Optional, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAINING_CONFIG, SELECTOR_CONFIG, DEVICE


# =============================================================
# Episode Buffer
# =============================================================
class EpisodeBuffer:
    """
    Stores transitions from a single episode for training.
    """

    def __init__(self, nv: int, max_len: int = 50):
        self.nv      = nv
        self.max_len = max_len
        self.reset()

    def reset(self):
        self.q        = []
        self.v        = []
        self.a_des    = []
        self.tau_full = []
        self.tau_out  = []
        self.s_t      = []
        self.use_full = []
        self.reward   = []
        self.M_list   = []
        self.C_list   = []
        self.g_list   = []

    def add(self, transition: Dict):
        self.q.append(transition["q"])
        self.v.append(transition["v"])
        self.a_des.append(transition["a_des"])
        self.tau_out.append(transition["tau"])
        self.s_t.append(transition["s_t"])
        self.use_full.append(transition["use_full"])
        self.reward.append(float(transition.get("reward", 0.0)))
        if "tau_full" in transition:
            self.tau_full.append(transition["tau_full"])
        if "M" in transition:
            self.M_list.append(transition["M"])
            self.C_list.append(transition["C"])
            self.g_list.append(transition["g"])

    def __len__(self):
        return len(self.q)

    def to_tensors(self) -> Dict[str, torch.Tensor]:
        out = {
            "q":       torch.stack(self.q),
            "v":       torch.stack(self.v),
            "tau_out": torch.stack(self.tau_out),
            "s_t":     torch.stack([
                           torch.tensor(s) if not isinstance(s, torch.Tensor)
                           else s for s in self.s_t]),
            "reward":  torch.tensor(self.reward, dtype=torch.float32),
        }
        if self.tau_full:
            out["tau_full"] = torch.stack(self.tau_full)
        if self.M_list:
            out["M"] = torch.stack(self.M_list)
            out["C"] = torch.stack(self.C_list)
            out["g"] = torch.stack(self.g_list)
        return out


# =============================================================
# Simulated Robot Environment
# =============================================================
class SimulatedRobotEnv:
    """
    Minimal robot environment for training without Gymnasium.
    Simulates a reaching task using forward dynamics.
    """

    def __init__(self, dynamics, nv: int = 6, seed: int = 42):
        self.dynamics   = dynamics
        self.nv         = nv
        self.rng        = np.random.default_rng(seed)
        self.model      = dynamics.model
        self.q_target   = None
        self.q          = None
        self.v          = None
        self.step_count = 0
        self.max_steps  = 50

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        import pinocchio as pin
        self.q = torch.tensor(
            pin.randomConfiguration(self.model),
            dtype=torch.float64)
        self.v = torch.zeros(self.nv, dtype=torch.float64)
        self.q_target = torch.tensor(
            pin.randomConfiguration(self.model),
            dtype=torch.float64)
        self.step_count = 0
        return self.q.clone(), self.v.clone()

    def step(
        self, tau: torch.Tensor, dt: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor, float, bool]:
        """
        Euler integration of forward dynamics.
        Returns: (q_new, v_new, reward, done)
        """
        with torch.no_grad():
            a = self.dynamics.forward_dynamics(
                self.q, self.v, tau.double())

        self.v = self.v + a * dt
        self.q = self.q + self.v * dt

        dist   = (self.q - self.q_target).float().norm().item()
        reward = -dist

        self.step_count += 1
        done = (self.step_count >= self.max_steps) or (dist < 0.05)
        return self.q.clone(), self.v.clone(), reward, done

    def get_obs(self) -> Dict[str, torch.Tensor]:
        return {
            "q":        self.q.clone(),
            "v":        self.v.clone(),
            "q_target": self.q_target.clone(),
            "image":    torch.randn(3, 224, 224),
            "tokens":   torch.randint(0, 100, (10,)),
        }


# =============================================================
# Three-Phase Trainer
# =============================================================
class ThreePhaseTrainer:
    """
    Implements the three-phase training curriculum.
    Section III-D of the paper.
    """

    def __init__(
        self,
        policy,
        env:    SimulatedRobotEnv,
        device: torch.device = DEVICE,
        cfg:    Dict = None,
    ):
        self.policy = policy
        self.env    = env
        self.device = device
        self.cfg    = cfg or TRAINING_CONFIG

        # Separate optimizers
        self.opt_policy = torch.optim.Adam(
            list(policy.visual_encoder.parameters()) +
            list(policy.lang_encoder.parameters())   +
            list(policy.fusion.parameters())          +
            list(policy.policy_head.parameters()),
            lr=self.cfg["lr"])

        self.opt_approx = torch.optim.Adam(
            policy.approximator.parameters(),
            lr=self.cfg["lr"])

        self.opt_selector = torch.optim.Adam(
            policy.selector.parameters(),
            lr=self.cfg["lr"])

        self.logs = {"phase1": [], "phase2": [], "phase3": []}
        self.global_step = 0

    # ----------------------------------------------------------
    # Collect one episode (no grad needed for env interaction)
    # ----------------------------------------------------------
    @torch.no_grad()
    def collect_episode_nograd(
        self,
        use_full: Optional[bool] = True,
    ) -> Tuple[EpisodeBuffer, float]:
        """
        Collect episode transitions WITHOUT gradient tracking.
        Used to gather data for supervised training steps.
        """
        buf = EpisodeBuffer(nv=self.env.nv)
        self.env.reset()
        self.policy.reset_history()
        env_obs = self.env.get_obs()
        q = env_obs["q"]
        v = env_obs["v"]
        total_reward = 0.0

        for _ in range(self.env.max_steps):
            out = self.policy(
                env_obs["image"], env_obs["tokens"],
                q, v,
                task_id=0,
                use_full_dynamics=use_full,
                hard_select=True,
            )
            tau = out["tau"]
            q_new, v_new, reward, done = self.env.step(tau)
            total_reward += reward

            buf.add({
                "q":        q.clone(),
                "v":        v.clone(),
                "a_des":    out["a_des"],
                "tau":      tau,
                "s_t":      out["s_t"],
                "use_full": out["use_full"],
                "reward":   reward,
                "M":        out["M"],
                "C":        out["C"],
                "g":        out["g"],
            })

            q, v    = q_new, v_new
            env_obs = self.env.get_obs()
            env_obs["q"] = q
            env_obs["v"] = v

            if done:
                break

        return buf, total_reward

    # ----------------------------------------------------------
    # Phase 1: Warm-up
    # ----------------------------------------------------------
    def phase1_warmup(
        self,
        num_episodes: int = 20,
        log_every:    int = 5,
    ) -> List[Dict]:
        """
        Train policy with full dynamics always on.
        Train approximator via supervised learning.
        Gradient flows only through differentiable ops.
        """
        print("\n" + "=" * 55)
        print("  PHASE 1: Warm-up (Full Dynamics)")
        print("=" * 55)

        phase_logs = []

        for ep in range(1, num_episodes + 1):
            t0 = time.perf_counter()

            # Collect data without grad (for approximator)
            buf, total_reward = self.collect_episode_nograd(
                use_full=True)
            tensors = buf.to_tensors()

            # ── Policy loss (recompute WITH grad) ─────────────
            self.opt_policy.zero_grad()
            self.policy.reset_history()
            env_obs = self.env.get_obs()

            # Re-run a short forward pass with gradients
            # to get a differentiable loss
            q_ep = tensors["q"][0]
            v_ep = tensors["v"][0]

            out_grad = self.policy(
                env_obs["image"], env_obs["tokens"],
                q_ep, v_ep,
                task_id=0,
                use_full_dynamics=True,
            )

            # Policy loss: minimise predicted torque magnitude
            # (encourages smooth, low-effort control)
            tau_pred  = out_grad["tau"].float()
            policy_loss = tau_pred.pow(2).mean()

            policy_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.policy_head.parameters()) +
                list(self.policy.fusion.parameters()),
                max_norm=1.0)
            self.opt_policy.step()

            # ── Approximator supervised loss ──────────────────
            self.opt_approx.zero_grad()
            approx_losses = []

            n_samples = min(len(tensors["q"]), 10)
            for i in range(n_samples):
                q_i = tensors["q"][i]
                v_i = tensors["v"][i]
                M_i = tensors["M"][i].float()
                C_i = tensors["C"][i].float()
                g_i = tensors["g"][i].float()

                approx_out = self.policy.approximator(q_i, v_i)
                loss_i = (
                    F.mse_loss(approx_out["M"].float(), M_i) +
                    F.mse_loss(approx_out["C"].float(), C_i) +
                    F.mse_loss(approx_out["g"].float(), g_i)
                )
                approx_losses.append(loss_i)

            approx_loss = torch.stack(approx_losses).mean()
            approx_loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy.approximator.parameters(), 1.0)
            self.opt_approx.step()

            elapsed = (time.perf_counter() - t0) * 1e3
            log = {
                "episode":     ep,
                "reward":      total_reward,
                "policy_loss": policy_loss.item(),
                "approx_loss": approx_loss.item(),
                "steps":       len(buf),
                "time_ms":     elapsed,
            }
            phase_logs.append(log)
            self.logs["phase1"].append(log)
            self.global_step += len(buf)

            if ep % log_every == 0 or ep == 1:
                print(f"  Ep {ep:3d}/{num_episodes} | "
                      f"reward={total_reward:8.3f} | "
                      f"policy_loss={policy_loss.item():8.4f} | "
                      f"approx_loss={approx_loss.item():8.4f} | "
                      f"steps={len(buf):2d} | "
                      f"time={elapsed:.0f}ms")

        print(f"\n  Phase 1 complete. "
              f"Final approx_loss: "
              f"{phase_logs[-1]['approx_loss']:.4f}")
        return phase_logs

    # ----------------------------------------------------------
    # Phase 2: Selector pre-training
    # ----------------------------------------------------------
    def phase2_selector_pretrain(
        self,
        num_episodes: int = 20,
        log_every:    int = 5,
    ) -> List[Dict]:
        """
        Label timesteps by approximation error.
        Train selector to predict high-error (need full) steps.
        """
        print("\n" + "=" * 55)
        print("  PHASE 2: Selector Pre-training")
        print("=" * 55)

        phase_logs = []

        for ep in range(1, num_episodes + 1):
            t0 = time.perf_counter()

            # Collect episode data
            buf, total_reward = self.collect_episode_nograd(
                use_full=True)
            tensors = buf.to_tensors()
            n       = min(len(tensors["q"]), 10)

            # ── Compute approximation error labels ────────────
            errors = []
            with torch.no_grad():
                for i in range(n):
                    q_i       = tensors["q"][i]
                    v_i       = tensors["v"][i]
                    tau_full  = tensors["tau_out"][i].float()
                    a_zero    = torch.zeros(
                        self.env.nv, dtype=torch.float64)

                    approx_out = self.policy.approximator(q_i, v_i)
                    tau_approx = (
                        approx_out["M"] @ a_zero +
                        approx_out["C"] +
                        approx_out["g"]
                    ).float()

                    err = F.mse_loss(tau_approx, tau_full).item()
                    errors.append(err)

            errors_t = torch.tensor(errors, dtype=torch.float32)
            if errors_t.max() > errors_t.min() + 1e-8:
                labels = ((errors_t - errors_t.min()) /
                          (errors_t.max() - errors_t.min()))
            else:
                labels = torch.full_like(errors_t, 0.5)

            # ── Train selector with BCE loss ──────────────────
            self.opt_selector.zero_grad()
            sel_losses = []

            env_obs = self.env.get_obs()
            self.policy.reset_history()

            for i in range(n):
                with torch.no_grad():
                    z_vis  = self.policy.visual_encoder(
                        env_obs["image"])
                    z_lang = self.policy.lang_encoder(
                        env_obs["tokens"])
                    z_t    = self.policy.fusion(z_vis, z_lang)

                task_oh = F.one_hot(
                    torch.tensor(0),
                    self.policy.n_tasks).float()

                sel_out = self.policy.selector(
                    z_t.detach(),
                    self.policy.history_buf,
                    task_oh,
                    hard=False,
                )
                s_clamped = sel_out["s_t"].clamp(1e-6, 1.0 - 1e-6)
                loss_i    = F.binary_cross_entropy(
                    s_clamped, labels[i])
                sel_losses.append(loss_i)

            sel_loss = torch.stack(sel_losses).mean()
            sel_loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy.selector.parameters(), 1.0)
            self.opt_selector.step()

            elapsed = (time.perf_counter() - t0) * 1e3
            log = {
                "episode":    ep,
                "reward":     total_reward,
                "sel_loss":   sel_loss.item(),
                "mean_label": labels.mean().item(),
                "time_ms":    elapsed,
            }
            phase_logs.append(log)
            self.logs["phase2"].append(log)
            self.global_step += len(buf)

            if ep % log_every == 0 or ep == 1:
                print(f"  Ep {ep:3d}/{num_episodes} | "
                      f"reward={total_reward:8.3f} | "
                      f"sel_loss={sel_loss.item():.4f} | "
                      f"mean_label={labels.mean().item():.4f} | "
                      f"time={elapsed:.0f}ms")

        print(f"\n  Phase 2 complete. "
              f"Final sel_loss: {phase_logs[-1]['sel_loss']:.4f}")
        return phase_logs

    # ----------------------------------------------------------
    # Phase 3: Joint fine-tuning
    # ----------------------------------------------------------
    def phase3_joint_finetune(
        self,
        num_episodes: int = 20,
        log_every:    int = 5,
    ) -> List[Dict]:
        """
        Fine-tune all components end-to-end.
        Uses Gumbel-Softmax for differentiable selection.
        """
        print("\n" + "=" * 55)
        print("  PHASE 3: Joint Fine-tuning (Gumbel-Softmax)")
        print("=" * 55)

        opt_joint = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.cfg["lr"] * 0.1)

        phase_logs = []

        for ep in range(1, num_episodes + 1):
            t0 = time.perf_counter()

            # Anneal temperature
            self.policy.selector.anneal_temperature(
                ep, num_episodes)

            # Collect WITHOUT grad for env interaction
            buf, total_reward = self.collect_episode_nograd(
                use_full=None)
            tensors = buf.to_tensors()
            n       = min(len(tensors["q"]), 10)

            # ── Recompute WITH grad for backprop ──────────────
            opt_joint.zero_grad()
            self.policy.reset_history()
            env_obs = self.env.get_obs()

            tau_list = []
            s_t_list = []

            for i in range(n):
                q_i = tensors["q"][i]
                v_i = tensors["v"][i]

                out_i = self.policy(
                    env_obs["image"], env_obs["tokens"],
                    q_i, v_i,
                    task_id=0,
                    use_full_dynamics=True,  # always full for grad
                    hard_select=False,
                )
                tau_list.append(out_i["tau"].float())
                s_t_list.append(out_i["s_t"])

            tau_stack = torch.stack(tau_list)   # [n, nv]
            s_t_stack = torch.stack(s_t_list)   # [n]

            # Task loss: minimise torque magnitude
            l_task = tau_stack.pow(2).mean()

            # Smoothness loss
            if tau_stack.shape[0] > 1:
                l_smooth = (
                    tau_stack[1:] - tau_stack[:-1]
                ).pow(2).mean()
            else:
                l_smooth = torch.tensor(0.0)

            # Computation cost: penalise always using full
            l_cost = s_t_stack.mean()

            total_loss = (
                l_task
                + self.cfg["lambda_smooth"] * l_smooth
                + self.cfg["lambda_cost"]   * l_cost
            )

            total_loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), 1.0)
            opt_joint.step()

            sel_rate = s_t_stack.detach().mean().item()
            elapsed  = (time.perf_counter() - t0) * 1e3

            log = {
                "episode":    ep,
                "reward":     total_reward,
                "total_loss": total_loss.item(),
                "l_task":     l_task.item(),
                "l_smooth":   l_smooth.item(),
                "l_cost":     l_cost.item(),
                "sel_rate":   sel_rate,
                "temp":       self.policy.selector.temp,
                "steps":      len(buf),
                "time_ms":    elapsed,
            }
            phase_logs.append(log)
            self.logs["phase3"].append(log)
            self.global_step += len(buf)

            if ep % log_every == 0 or ep == 1:
                print(f"  Ep {ep:3d}/{num_episodes} | "
                      f"reward={total_reward:8.3f} | "
                      f"loss={total_loss.item():8.4f} | "
                      f"sel={sel_rate:.2%} | "
                      f"temp={self.policy.selector.temp:.3f} | "
                      f"time={elapsed:.0f}ms")

        print(f"\n  Phase 3 complete.")
        print(f"  Final selection rate : "
              f"{phase_logs[-1]['sel_rate']:.2%}")
        print(f"  Compute savings      : "
              f"{(1-phase_logs[-1]['sel_rate'])*100:.1f}%")
        return phase_logs

    # ----------------------------------------------------------
    # Full curriculum
    # ----------------------------------------------------------
    def train(
        self,
        phase1_eps: int = 20,
        phase2_eps: int = 20,
        phase3_eps: int = 20,
    ) -> Dict:
        """Run all three training phases sequentially."""
        print("\n" + "#" * 55)
        print("#  Three-Phase Training Curriculum")
        print("#" * 55)
        print(f"  Device      : {self.device}")
        print(f"  Robot DOF   : {self.env.nv}")
        print(f"  Phase 1 eps : {phase1_eps}")
        print(f"  Phase 2 eps : {phase2_eps}")
        print(f"  Phase 3 eps : {phase3_eps}")

        t_total = time.perf_counter()

        p1 = self.phase1_warmup(phase1_eps)
        p2 = self.phase2_selector_pretrain(phase2_eps)
        p3 = self.phase3_joint_finetune(phase3_eps)

        total_time = time.perf_counter() - t_total

        summary = {
            "phase1_final_reward":   p1[-1]["reward"],
            "phase1_approx_loss":    p1[-1]["approx_loss"],
            "phase2_final_sel_loss": p2[-1]["sel_loss"],
            "phase3_final_reward":   p3[-1]["reward"],
            "phase3_sel_rate":       p3[-1]["sel_rate"],
            "compute_savings_pct":   (1 - p3[-1]["sel_rate"]) * 100,
            "total_steps":           self.global_step,
            "total_time_s":          total_time,
        }

        print("\n" + "#" * 55)
        print("#  Training Complete - Summary")
        print("#" * 55)
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k:<30s}: {v:.4f}")
            else:
                print(f"  {k:<30s}: {v}")
        print("#" * 55)
        return summary


# =============================================================
# Self-test
# =============================================================
if __name__ == "__main__":
    import pinocchio as pin
    from agents.dynamics     import build_dynamics_layer
    from agents.approximator import DynamicsApproximator
    from agents.selector     import AdaptiveSelector
    from agents.policy       import VLAPolicy

    print("=" * 55)
    print("  Three-Phase Trainer - Self Test")
    print("=" * 55)

    torch.manual_seed(42)
    np.random.seed(42)

    nv = 6

    dynamics     = build_dynamics_layer("panda")
    approximator = DynamicsApproximator(nv=nv)
    selector     = AdaptiveSelector(nv=nv)
    policy       = VLAPolicy(
        dynamics=dynamics,
        approximator=approximator,
        selector=selector,
        nv=nv,
    )

    env     = SimulatedRobotEnv(dynamics=dynamics, nv=nv)
    trainer = ThreePhaseTrainer(policy=policy, env=env)

    summary = trainer.train(
        phase1_eps=10,
        phase2_eps=10,
        phase3_eps=10,
    )

    print("\n--- Final Verification ---")
    print(f"  Phase 1 reward      : {summary['phase1_final_reward']:.3f}")
    print(f"  Phase 1 approx_loss : {summary['phase1_approx_loss']:.4f}")
    print(f"  Phase 2 sel_loss    : {summary['phase2_final_sel_loss']:.4f}")
    print(f"  Phase 3 reward      : {summary['phase3_final_reward']:.3f}")
    print(f"  Selector rate       : {summary['phase3_sel_rate']:.2%}")
    print(f"  Compute savings     : {summary['compute_savings_pct']:.1f}%")
    print(f"  Total steps         : {summary['total_steps']:,}")