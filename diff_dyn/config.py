# =============================================================
# config.py  --  Central configuration for all experiments
# =============================================================

import torch
import os

# -------------------------------------------------------------
# Device
# -------------------------------------------------------------
DEVICE = torch.device("cpu")   # CPU mode (no CUDA/MPS needed)

# -------------------------------------------------------------
# Project paths
# -------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR    = os.path.join(BASE_DIR, "agents")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
TESTS_DIR     = os.path.join(BASE_DIR, "tests")

# -------------------------------------------------------------
# Robot configurations
# -------------------------------------------------------------
ROBOT_CONFIGS = {
    "panda": {
        "dof": 7,
        "name": "Franka Emika Panda",
        "contact_rich": False,          # Reach task
        "torque_limits": [87, 87, 87, 87, 12, 12, 12],  # Nm
        "velocity_limits": [2.175, 2.175, 2.175, 2.175,
                            2.61,  2.61,  2.61],          # rad/s
    },
    "panda_push": {
        "dof": 7,
        "name": "Franka Emika Panda (Push)",
        "contact_rich": True,
        "torque_limits": [87, 87, 87, 87, 12, 12, 12],
        "velocity_limits": [2.175, 2.175, 2.175, 2.175,
                            2.61,  2.61,  2.61],
    },
    "allegro": {
        "dof": 16,
        "name": "Allegro Hand",
        "contact_rich": True,
        "torque_limits": [0.7] * 16,    # Nm
        "velocity_limits": [10.0] * 16, # rad/s
    },
}

# -------------------------------------------------------------
# Environment configurations
# -------------------------------------------------------------
ENV_CONFIGS = {
    "PandaReach-v3": {
        "robot":         "panda",
        "max_steps":     50,
        "reward_type":   "dense",
        "obs_dim":       25,
        "action_dim":    4,             # 3D delta + gripper
        "image_size":    (224, 224),
        "num_cameras":   1,
    },
    "PandaPush-v3": {
        "robot":         "panda_push",
        "max_steps":     50,
        "reward_type":   "dense",
        "obs_dim":       28,
        "action_dim":    4,
        "image_size":    (224, 224),
        "num_cameras":   1,
    },
    "HandManipulateBlock-v1": {
        "robot":         "allegro",
        "max_steps":     100,
        "reward_type":   "dense",
        "obs_dim":       61,
        "action_dim":    20,
        "image_size":    (224, 224),
        "num_cameras":   1,
    },
}

# -------------------------------------------------------------
# VLA Policy Architecture
# -------------------------------------------------------------
VLA_CONFIG = {
    # Perception encoder
    "visual_encoder":    "resnet18",
    "image_size":        224,
    "latent_dim":        512,           # z_t dimension

    # Policy head
    "policy_hidden":     256,
    "policy_layers":     3,
    "activation":        "relu",

    # Language encoder
    "lang_encoder":      "clip",
    "lang_dim":          512,

    # Fusion
    "fusion_heads":      4,
    "fusion_layers":     2,
    "fusion_dim":        512,
}

# -------------------------------------------------------------
# Differentiable Dynamics Layer
# -------------------------------------------------------------
DYNAMICS_CONFIG = {
    "use_pinocchio":     True,
    "cholesky_solve":    True,          # Avoid explicit M^{-1}
    "compute_M":         True,
    "compute_C":         True,
    "compute_g":         True,
    "dtype":             torch.float64, # Pinocchio uses float64
}

# -------------------------------------------------------------
# Adaptive Selector
# -------------------------------------------------------------
SELECTOR_CONFIG = {
    # Network
    "hidden_dim":        128,
    "num_layers":        2,
    "input_dim":         640,           # 512 (z_t) + 128 (history)
    "history_len":       10,            # timesteps of history

    # Gumbel-Softmax
    "temp_start":        1.0,
    "temp_end":          0.1,
    "threshold":         0.5,           # inference threshold

    # Cost coefficients
    "cost_full":         1.0,           # relative cost of full dynamics
    "cost_approx":       0.1,           # relative cost of approximation
    "alpha":             0.01,          # task vs compute tradeoff
}

# -------------------------------------------------------------
# Approximator Network
# -------------------------------------------------------------
APPROXIMATOR_CONFIG = {
    "hidden_dim":        256,
    "num_layers":        3,
    "activation":        "relu",
    # output = n(n+1)/2 + 2n  (M lower-tri + C + g)
}

# -------------------------------------------------------------
# Training
# -------------------------------------------------------------
TRAINING_CONFIG = {
    # Optimizer
    "lr":                3e-4,
    "lr_schedule":       "cosine",
    "weight_decay":      1e-5,

    # Batching
    "batch_size":        32,            # reduced from 256 (no GPU)
    "episode_len":       50,

    # Scale (reduced for CPU)
    "total_steps":       50_000,        # reduced from 1M
    "eval_every":        1_000,
    "save_every":        5_000,

    # Loss weights
    "lambda_dynamics":   0.1,           # lambda_1
    "lambda_smooth":     0.01,          # lambda_2
    "lambda_cost":       0.01,          # lambda_3 (alpha)

    # Curriculum phases (in steps)
    "phase1_steps":      10_000,        # warm-up
    "phase2_steps":      20_000,        # selector pre-training
    "phase3_steps":      20_000,        # joint fine-tuning

    # Random seed
    "seed":              42,
}

# -------------------------------------------------------------
# Baselines
# -------------------------------------------------------------
BASELINE_CONFIG = {
    "full_dynamics":     True,
    "no_dynamics":       True,
    "fixed_schedule_k":  [2, 5, 10],
    "learned_dynamics":  True,
}

# -------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------
EVAL_CONFIG = {
    "num_episodes":      50,
    "render":            False,
    "record_latency":    True,
    "latency_percentiles": [50, 95, 99],
    "num_trials_hardware": 50,
}

# -------------------------------------------------------------
# Sanity check
# -------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("Configuration loaded successfully")
    print("=" * 50)
    print(f"Device:          {DEVICE}")
    print(f"Robots:          {list(ROBOT_CONFIGS.keys())}")
    print(f"Environments:    {list(ENV_CONFIGS.keys())}")
    print(f"Total steps:     {TRAINING_CONFIG['total_steps']:,}")
    print(f"Batch size:      {TRAINING_CONFIG['batch_size']}")
    print(f"Latent dim:      {VLA_CONFIG['latent_dim']}")
    print(f"Selector input:  {SELECTOR_CONFIG['input_dim']}")
    print("=" * 50)