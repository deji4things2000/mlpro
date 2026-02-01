# scripts/visualize_rollout.py - CLEAN NO-GRIDLINES VERSION
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # For Mac display
import torch
from env.nav_env import NavEnv
from env.expert_policy import expert_policy
from utils.preprocessing import build_image_encoder, image_transform
from models.transformer_policy import TransformerPolicy

def make_transformer_policy(model_path, device="cpu", seq_len=8):
    encoder = build_image_encoder(device=device)
    tmp_env = NavEnv()
    lidar_dim = tmp_env.observation_space["lidar"].shape[0]
    input_dim = 128 + lidar_dim + 5 + 4

    policy_net = TransformerPolicy(input_dim=input_dim, num_actions=4, seq_len=seq_len).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(checkpoint["policy"])
    policy_net.eval()

    history = []

    def policy(obs, env):
        nonlocal history
        img = image_transform(obs["image"]).unsqueeze(0)
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

        imgs = torch.cat([h[0] for h in history], dim=0)
        lidars = torch.cat([h[1] for h in history], dim=0)
        states = torch.cat([h[2] for h in history], dim=0)
        prev_as = torch.cat([h[3] for h in history], dim=0)

        L = imgs.shape[0]
        if L < seq_len:
            pad = seq_len - L
            imgs = torch.cat([imgs[0:1].repeat(pad,1,1,1), imgs], dim=0)
            lidars = torch.cat([lidars[0:1].repeat(pad,1), lidars], dim=0)
            states = torch.cat([states[0:1].repeat(pad,1), states], dim=0)
            prev_as = torch.cat([prev_as[0:1].repeat(pad,1), prev_as], dim=0)

        imgs = imgs.unsqueeze(0).to(device)
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

if __name__ == "__main__":
    print("🎬 Generating CLEAN expert trajectories (no gridlines)...")
    
    # 1. EXPERT TRAJECTORIES (2x2, no grid)
    env = NavEnv(render_mode="rgb_array")  # ← FIXED: Define env here!
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for ep in range(4):
        obs, _ = env.reset()
        states = []
        
        ax = axes[ep//2, ep%2]
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_title(f"Expert - Episode {ep+1}", fontweight='bold', fontsize=14)
        
        done = False
        step = 0
        while not done and step < 200:
            action = expert_policy(obs, env)
            states.append(env.state.copy())
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            step += 1
        
        states = np.array(states)
        ax.plot(states[:, 0], states[:, 1], 'darkblue', linewidth=5)
        ax.plot(states[0, 0], states[0, 1], 'limegreen', markersize=15, marker='o')
        ax.plot(states[-1, 0], states[-1, 1], 'red', markersize=15, marker='o')
        ax.plot(env.goal[0], env.goal[1], 'limegreen', markersize=20, marker='*')
        
        for rect in env.obstacles:
            x1,y1,x2,y2 = rect
            ax.fill([x1,x1,x2,x2], [y1,y2,y2,y1], 'darkred', alpha=0.7)
        
        ax.grid(False)  # NO GRIDLINES
        ax.axis('equal')
    
    plt.suptitle("Expert Policy Performance (68% Success Rate)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("report/expert_clean.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("✅ Saved: report/expert_clean.png")
    
    print("🎬 Generating CLEAN policy comparison (no gridlines)...")
    
    # 2. POLICY COMPARISON (2x2, no grid)
    env = NavEnv(render_mode="rgb_array")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    policies = [
        (lambda obs, e: e.action_space.sample(), "Random\n(2% Success)"),
        (expert_policy, "Expert\n(68% Success)"), 
        (make_transformer_policy("models/transformer_policy.pt", device="cpu"), "Transformer #1\n(2% Success)"),
        (make_transformer_policy("models/transformer_policy.pt", device="cpu"), "Transformer #2\n(2% Success)")
    ]
    
    for idx, (policy_fn, name) in enumerate(policies):
        ax = axes[idx//2, idx%2]
        obs, _ = env.reset()
        states = []
        
        done = False
        step = 0
        while not done and step < 200:
            action = policy_fn(obs, env)
            states.append(env.state.copy())
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            step += 1
        
        states = np.array(states)
        colors = ['darkred', 'darkgreen', 'darkblue', 'purple']
        ax.plot(states[:, 0], states[:, 1], color=colors[idx], linewidth=6)
        ax.plot(states[0, 0], states[0, 1], 'limegreen', markersize=18, marker='o')
        ax.plot(states[-1, 0], states[-1, 1], 'red', markersize=18, marker='o')
        ax.plot(env.goal[0], env.goal[1], 'limegreen', markersize=25, marker='*')
        
        for rect in env.obstacles:
            x1,y1,x2,y2 = rect
            ax.fill([x1,x1,x2,x2], [y1,y2,y2,y1], 'darkred', alpha=0.7)
        
        ax.set_title(name, fontweight='bold', fontsize=13, pad=15)
        ax.grid(False)  # NO GRIDLINES
        ax.axis('equal')
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
    
    plt.suptitle("Policy Comparison: 98.4% Imitation → 2% Task Success", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("report/policy_comparison_clean.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("✅ Saved: report/policy_comparison_clean.png")
    print("🎉 Publication-ready figures generated!")
