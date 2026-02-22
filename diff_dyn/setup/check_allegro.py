# check_allegro.py
import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

# ── Step 1: Search specifically for Allegro ───────────────────────────────────
all_envs = gym.envs.registry.keys()
allegro_envs = sorted([e for e in all_envs if "Allegro" in e or "allegro" in e])

print("=" * 60)
print("Allegro Environments Found:")
print("=" * 60)

if allegro_envs:
    for env_id in allegro_envs:
        print(f"  ✅  {env_id}")
else:
    print("  ❌  No Allegro environments found in registry")
    print("\n  Available Hand-type environments instead:")
    hand_envs = sorted([e for e in all_envs if "Hand" in e or "hand" in e])
    for env_id in hand_envs:
        print(f"       {env_id}")

# ── Step 2: Try instantiating Allegro (or fallback) ──────────────────────────
TARGET_ENVS = allegro_envs if allegro_envs else [
    "AdroitHandDoor-v1",        # 28 DOF Adroit hand
    "FetchPickAndPlace-v2",     # Fetch robot arm
    "HandReach-v2",             # Shadow Dexterous Hand
]

print("\n" + "=" * 60)
print("Instantiation Tests")
print("=" * 60)

for env_id in TARGET_ENVS[:3]:   # test up to 3
    try:
        env = gym.make(env_id, render_mode=None)
        obs, info = env.reset()

        print(f"\n✅  {env_id}")
        print(f"    Observation space : {env.observation_space}")
        print(f"    Action space      : {env.action_space}")
        print(f"    Action dims (DOF) : {env.action_space.shape[0]}")

        # Run a few random steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

        print(f"    Random rollout    : ✅ (5 steps OK)")
        env.close()

    except Exception as ex:
        print(f"\n❌  {env_id}")
        print(f"    Error: {ex}")