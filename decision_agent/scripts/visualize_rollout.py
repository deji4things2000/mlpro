# scripts/visualize_rollout.py - COMPLETE VERSION
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
from env.nav_env import NavEnv
from env.expert_policy import expert_policy
from utils.preprocessing import build_image_encoder, image_transform
from models.transformer_policy import TransformerPolicy
from torch.utils.data import DataLoader

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

# 2x2 COMPARISON PLOT
if __name__ == "__main__":
    print("🎬 Creating policy comparison...")
    env = NavEnv(render_mode="rgb_array")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    policies = [
        (lambda obs, e: e.action_space.sample(), "Random (Baseline)"),
        (expert_policy, "Expert Policy"), 
        (make_transformer_policy("models/transformer_policy.pt", device="cpu"), "Transformer (98% Acc)"),
        (make_transformer_policy("models/transformer_policy.pt", device="cpu"), "Transformer #2")
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
        color = ['red', 'green', 'blue', 'purple'][idx]
        
        ax.plot(states[:, 0], states[:, 1], color=color, linewidth=4, label=name)
        ax.plot(states[0, 0], states[0, 1], 'go', markersize=15, label='Start')
        ax.plot(states[-1, 0], states[-1, 1], 'ro', markersize=15, label='End')
        ax.plot(env.goal[0], env.goal[1], 'g*', markersize=20, label='Goal')
        
        for rect in env.obstacles:
            x1,y1,x2,y2 = rect
            ax.fill([x1,x1,x2,x2], [y1,y2,y2,y1], 'r', alpha=0.6)
        
        ax.set_title(name, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.axis('equal')
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
    
    plt.suptitle("Transformer Agent vs Expert vs Random (98.4% Imitation Accuracy)", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("report/policy_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Saved: report/policy_comparison.png")
