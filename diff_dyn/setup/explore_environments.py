# explore_environments.py
import gymnasium as gym
import gymnasium_robotics

# Register all robotics environments
gym.register_envs(gymnasium_robotics)

# ── List ALL available environments ──────────────────────────────────────────
all_envs = gym.envs.registry.keys()

robotics_envs = sorted([e for e in all_envs if any(
    keyword in e for keyword in [
        "Fetch", "Hand", "Allegro", "Adroit",
        "Shadow", "Ant", "Maze", "Point"
    ]
)])

print("=" * 60)
print("Available Gymnasium-Robotics Environments")
print("=" * 60)
for env_id in robotics_envs:
    print(f"  {env_id}")