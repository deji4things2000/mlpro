# utils/preprocessing.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
from torch.utils.data import Dataset

class Identity(nn.Module):
    def forward(self, x):
        return x

image_transform = T.Compose([
    T.ToTensor(),  # [0,1]
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def build_image_encoder(device="cpu"):
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = Identity()
    for p in resnet.parameters():
        p.requires_grad = False
    proj = nn.Linear(512, 128)
    encoder = nn.Sequential(resnet, proj)
    encoder.to(device)
    encoder.eval()
    return encoder

class TrajectoryDataset(Dataset):
    def __init__(self, npz_path, seq_len=8, device="cpu"):
        data = np.load(npz_path, allow_pickle=True)
        self.images = data["images"]       
        self.lidar = data["lidar"]         
        self.state = data["state"]         
        self.actions = data["actions"]     
        self.episode_ids = data["episode_ids"]
        self.timesteps = data["timesteps"]
        self.seq_len = seq_len
        self.device = device

        # Precompute valid sequence indices
        self.valid_indices = []
        for i in range(len(self.actions)):
            if self.timesteps[i] >= seq_len - 1:
                # Check if we have seq_len contiguous steps in same episode
                start_idx = i - seq_len + 1
                ep_ids = self.episode_ids[start_idx:i+1]
                if len(np.unique(ep_ids)) == 1:  # All same episode
                    self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        start = i - self.seq_len + 1
        end = i + 1
        
        imgs = self.images[start:end]      
        lidar = self.lidar[start:end]      
        state = self.state[start:end]      
        actions = self.actions[start:end]   

        # prev_action one-hot (L, 4)
        prev_a = np.zeros((self.seq_len, 4), dtype=np.float32)
        prev_a[1:] = np.eye(4, dtype=np.float32)[actions[:-1]]

        return {
            "images": imgs,
            "lidar": lidar.astype(np.float32),
            "state": state.astype(np.float32),
            "prev_action": prev_a,
            "target_action": int(actions[-1])
        }
