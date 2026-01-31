# train/collect_expert.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import trange
from env.nav_env import NavEnv
from env.expert_policy import expert_policy

import numpy as np
from tqdm import trange
from env.nav_env import NavEnv
from env.expert_policy import expert_policy

def collect(num_episodes=200, max_steps=200, out_path="data/trajectories.npz"):
    env = NavEnv(render_mode=None)
    all_images, all_lidar, all_state, all_actions, all_ep_ids, all_t = [], [], [], [], [], []
    episode_id = 0

    for _ in trange(num_episodes):
        obs, info = env.reset()
        for step in range(max_steps):
            action = expert_policy(obs, env.goal)
            all_images.append(obs["image"])
            all_lidar.append(obs["lidar"])
            all_state.append(obs["state"])
            all_actions.append(action)
            all_ep_ids.append(episode_id)
            all_t.append(step)

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        episode_id += 1

    np.savez_compressed(
        out_path,
        images=np.array(all_images, dtype=np.uint8),
        lidar=np.array(all_lidar, dtype=np.float32),
        state=np.array(all_state, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.int64),
        episode_ids=np.array(all_ep_ids, dtype=np.int32),
        timesteps=np.array(all_t, dtype=np.int32),
    )

if __name__ == "__main__":
    collect()
