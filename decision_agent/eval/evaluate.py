# eval/evaluate.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from tqdm import trange

from env.nav_env import NavEnv
from env.expert_policy import expert_policy
from utils.preprocessing import build_image_encoder, image_transform
from models.transformer_policy import TransformerPolicy

def run_policy(env, policy_fn, episodes=100):
    successes, collisions, lengths = 0, 0, []
    for _ in trange(episodes):
        obs, info = env.reset()
        steps = 0
        while True:
            action = policy_fn(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                if reward > 0:        # reached goal
                    successes += 1
                elif reward < 0:      # collision
                    collisions += 1
                lengths.append(steps)
                break
    return {
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "avg_len": float(np.mean(lengths)),
    }

def make_random_policy():
    def policy(obs, env):
        return env.action_space.sample()
    return policy

def make_expert_policy():
    def policy(obs, env):
        return expert_policy(obs, env)
    return policy

def make_transformer_policy(model_path, device="cuda", seq_len=8):
    encoder = build_image_encoder(device=device)
    # Build temporary env to infer lidar_dim
    tmp_env = NavEnv()
    lidar_dim = tmp_env.observation_space["lidar"].shape[0]
    input_dim = 128 + lidar_dim + 5 + 4

    policy_net = TransformerPolicy(input_dim=input_dim, num_actions=4, seq_len=seq_len).to(device)
    state_dict = torch.load(model_path, map_location=device)["policy"]
    policy_net.load_state_dict(state_dict)
    policy_net.eval()

    history = []  # list of (image, lidar, state, prev_action_onehot)

    def policy(obs, env):
        nonlocal history
        img = image_transform(obs["image"]).unsqueeze(0)  # (1,3,H,W)
        lidar = torch.from_numpy(obs["lidar"]).float().unsqueeze(0)
        state = torch.from_numpy(obs["state"]).float().unsqueeze(0)
        if len(history) == 0:
            prev_a = torch.zeros(1, 4)
        else:
            last_a = history[-1][3].argmax().item()
            prev_a = torch.eye(4)[last_a].unsqueeze(0)

        history.append((img, lidar, state, prev_a))
        if len(history) > seq_len:
            history = history[-seq_len:]

        # Pad sequence if not full yet
        imgs = torch.cat([h[0] for h in history], dim=0)      # (L',3,H,W)
        lidars = torch.cat([h[1] for h in history], dim=0)    # (L',N)
        states = torch.cat([h[2] for h in history], dim=0)    # (L',5)
        prev_as = torch.cat([h[3] for h in history], dim=0)   # (L',4)

        L = imgs.shape[0]
        if L < seq_len:
            pad = seq_len - L
            imgs = torch.cat([imgs[0:1].repeat(pad,1,1,1), imgs], dim=0)
            lidars = torch.cat([lidars[0:1].repeat(pad,1), lidars], dim=0)
            states = torch.cat([states[0:1].repeat(pad,1), states], dim=0)
            prev_as = torch.cat([prev_as[0:1].repeat(pad,1), prev_as], dim=0)

        imgs = imgs.unsqueeze(0).to(device)    # (1,L,3,H,W)
        lidars = lidars.unsqueeze(0).to(device)
        states = states.unsqueeze(0).to(device)
        prev_as = prev_as.unsqueeze(0).to(device)

        B, L, C, H, W = imgs.shape
        img_flat = imgs.view(B*L, C, H, W)
        with torch.no_grad():
            img_emb_flat = encoder(img_flat)
        img_emb = img_emb_flat.view(B, L, -1)

        tokens = torch.cat([img_emb, lidars, states, prev_as], dim=-1)
        with torch.no_grad():
            logits = policy_net(tokens)
            action = logits.argmax(dim=-1).item()
        return action

    return policy

# ADD THESE TWO FUNCTIONS at the END (before if __name__ == "__main__":)

def transformer_policy(obs, env):
    """Standalone transformer inference (for demo)"""
    # Copy the exact inference logic from make_transformer_policy
    encoder = build_image_encoder(device='cpu')
    tmp_env = NavEnv()
    lidar_dim = tmp_env.observation_space["lidar"].shape[0]
    input_dim = 128 + lidar_dim + 5 + 4
    policy_net = TransformerPolicy(input_dim=input_dim, num_actions=4, seq_len=8).to('cpu')
    state_dict = torch.load("models/transformer_policy.pt", map_location='cpu')["policy"]
    policy_net.load_state_dict(state_dict)
    policy_net.eval()
    
    history = []
    # [Same exact inference code as make_transformer_policy - truncated for brevity]
    # Use the transformer_policy from your make_transformer_policy function
    return make_transformer_policy("models/transformer_policy.pt", device="cpu")(obs, env)

def demo_transformer_video(env, save_path="report/transformer_demo.gif"):
    """🎬 Generate Transformer agent GIF demo"""
    print("🎬 Recording Transformer demo...")
    obs, _ = env.reset()
    frames = []
    
    done = False
    step = 0
    while not done and step < 200:
        frame = env.render()
        frames.append(frame)
        
        action = transformer_policy(obs, env)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
    
    # Create & save GIF
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    fig = plt.figure(figsize=(8, 8))
    
    def animate(i):
        plt.clf()
        plt.imshow(frames[i])
        plt.title(f'Transformer Agent - Step {i+1}/{len(frames)}')
        plt.axis('off')
    
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=100)
    anim.save(save_path, writer=animation.PillowWriter(fps=10))
    plt.close()
    print(f"✅ Video saved: {save_path}")
    return save_path

# REPLACE your if __name__ == "__main__": with this:
if __name__ == "__main__":
    env = NavEnv()
    
    # Your ORIGINAL evaluation (KEEP THIS)
    random_stats = run_policy(env, make_random_policy(), episodes=50)
    expert_stats = run_policy(env, make_expert_policy(), episodes=50)
    transformer_stats = run_policy(env, make_transformer_policy("models/transformer_policy.pt", device="cpu"), episodes=50)
    
    print("Random:", random_stats)
    print("Expert:", expert_stats)
    print("Transformer:", transformer_stats)
    
    # 🎥 NEW: Video demo
    demo_transformer_video(env)


if __name__ == "__main__":
    env = NavEnv()
    random_stats = run_policy(env, make_random_policy(), episodes=50)
    expert_stats = run_policy(env, make_expert_policy(), episodes=50)
    transformer_stats = run_policy(env, make_transformer_policy("models/transformer_policy.pt", device="cpu"), episodes=50)
    print("Random:", random_stats)
    print("Expert:", expert_stats)
    print("Transformer:", transformer_stats)

