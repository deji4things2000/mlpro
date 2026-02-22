# render_test.py
import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

# Use rgb_array for headless servers; "human" opens a window
RENDER_MODE = "rgb_array"   # change to "human" if you have a display

env = gym.make(
    "HandManipulateBlock-v1",   # swap to Allegro if found
    render_mode=RENDER_MODE,
    max_episode_steps=100,
)

obs, _ = env.reset(seed=0)
frames = []

for step in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if RENDER_MODE == "rgb_array":
        frame = env.render()        # numpy array (H, W, 3)
        frames.append(frame)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()

if frames:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, ax in enumerate(axes):
        idx = i * (len(frames) // 5)
        ax.imshow(frames[idx])
        ax.set_title(f"Step {idx}")
        ax.axis("off")
    plt.suptitle("HandManipulateBlock-v1 — Random Policy", fontsize=14)
    plt.tight_layout()
    plt.savefig("render_test.png", dpi=120)
    print("✅  Saved render_test.png")