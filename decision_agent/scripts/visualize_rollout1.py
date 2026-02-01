# scripts/visualize_rollout.py - COMPLETE CLEAN VERSION (NO GRIDLINES)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Mac display
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
           
