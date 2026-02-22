# environment_anatomy.py
"""
Detailed inspection of whichever hand env is available.
Priority: Allegro → Adroit → Shadow Hand
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

# ── Priority list ─────────────────────────────────────────────────────────────
CANDIDATE_ENVS = [
    # Allegro (if available)
    "AllegroHandReach-v0",
    "AllegroHand-v0",
    # Adroit (28 DOF)
    "AdroitHandDoor-v1",
    "AdroitHandHammer-v1",
    "AdroitHandPen-v1",
    "AdroitHandRelocate-v1",
    # Shadow (24 DOF)
    "HandReach-v2",
    "HandManipulateBlock-v1",
]

def try_make(env_id):
    try:
        env = gym.make(env_id, render_mode=None)
        return env
    except:
        return None

# Find first working environment
env, chosen_id = None, None
for candidate in CANDIDATE_ENVS:
    env = try_make(candidate)
    if env is not None:
        chosen_id = candidate
        break

if env is None:
    raise RuntimeError("No suitable environment found. Check installation.")

print(f"\n{'=' * 60}")
print(f"Analyzing: {chosen_id}")
print(f"{'=' * 60}")

# ── Spaces ────────────────────────────────────────────────────────────────────
obs, info = env.reset(seed=42)

print(f"\n📐 ACTION SPACE")
print(f"   Type   : {type(env.action_space).__name__}")
print(f"   Shape  : {env.action_space.shape}")
print(f"   Low    : {env.action_space.low[:5]} ...")
print(f"   High   : {env.action_space.high[:5]} ...")
print(f"   DOFs   : {env.action_space.shape[0]}")

print(f"\n👁️  OBSERVATION SPACE")
if hasattr(env.observation_space, 'spaces'):
    # Dict observation space (Goal-conditioned envs)
    for key, space in env.observation_space.spaces.items():
        print(f"   [{key}] shape={space.shape}, dtype={space.dtype}")
else:
    print(f"   Shape  : {env.observation_space.shape}")
    print(f"   Dtype  : {env.observation_space.dtype}")

print(f"\n🎯 GOAL-CONDITIONED?")
is_goal_env = hasattr(env, 'compute_reward')
print(f"   {is_goal_env} — {'GoalEnv interface detected' if is_goal_env else 'Standard env'}")

# ── Reward structure ──────────────────────────────────────────────────────────
print(f"\n🏆 REWARD STRUCTURE (10 random steps)")
rewards = []
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    if terminated or truncated:
        obs, info = env.reset()

print(f"   Rewards seen: {[round(r, 3) for r in rewards]}")
print(f"   Info keys   : {list(info.keys())}")

# ── MuJoCo model internals ────────────────────────────────────────────────────
print(f"\n🔧 MUJOCO MODEL INTERNALS")
try:
    # Access underlying MuJoCo model
    mj_model = env.unwrapped.model
    mj_data  = env.unwrapped.data

    print(f"   n_joints (njnt)  : {mj_model.njnt}")
    print(f"   n_bodies (nbody) : {mj_model.nbody}")
    print(f"   n_geoms  (ngeom) : {mj_model.ngeom}")
    print(f"   n_actuators(nu)  : {mj_model.nu}")
    print(f"   n_sensors (nsensor): {mj_model.nsensor}")
    print(f"   qpos shape       : {mj_data.qpos.shape}")
    print(f"   qvel shape       : {mj_data.qvel.shape}")
except AttributeError as e:
    print(f"   Could not access MuJoCo internals: {e}")

env.close()
print(f"\n{'=' * 60}")
print("✅ Environment anatomy complete")
print(f"{'=' * 60}")