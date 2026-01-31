import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Now your existing imports work:
from utils.preprocessing import (
    build_image_encoder, image_transform, TrajectoryDataset
)
from models.transformer_policy import TransformerPolicy

def collate_fn(batch):
    # batch: list of dicts from dataset
    imgs = []
    lidar = []
    state = []
    prev_a = []
    targets = []

    for b in batch:
        # Transform images to torch
        # images: (L, H, W, C)
        img_seq = torch.stack(
            [image_transform(i) for i in b["images"]], dim=0
        )  # (L, 3, H, W)
        imgs.append(img_seq)
        lidar.append(torch.from_numpy(b["lidar"]))
        state.append(torch.from_numpy(b["state"]))
        prev_a.append(torch.from_numpy(b["prev_action"]))
        targets.append(torch.tensor(b["target_action"], dtype=torch.long))

    imgs = torch.stack(imgs, dim=0)       # (B, L, 3, H, W)
    lidar = torch.stack(lidar, dim=0)     # (B, L, N)
    state = torch.stack(state, dim=0)     # (B, L, 5)
    prev_a = torch.stack(prev_a, dim=0)   # (B, L, 4)
    targets = torch.stack(targets, dim=0) # (B,)
    return {"images": imgs, "lidar": lidar, "state": state,
            "prev_action": prev_a, "target_action": targets}

def train_il(
    data_path="data/trajectories.npz",
    seq_len=8,
    batch_size=64,
    num_epochs=20,
    lr=1e-4,
    device="cpu"
):
    dataset = TrajectoryDataset(data_path, seq_len=seq_len, device=device)
    # Simple split
    n = len(dataset)
    n_train = int(0.9 * n)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n - n_train])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    encoder = build_image_encoder(device=device)
    lidar_dim = dataset.lidar.shape[1]
    input_dim = 128 + lidar_dim + 5 + 4

    policy = TransformerPolicy(input_dim=input_dim, num_actions=4, seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    best_val = 0.0

    for epoch in range(num_epochs):
        policy.train()
        total_loss = 0.0
        total_correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            imgs = batch["images"].to(device)       # (B, L, 3, H, W)
            lidar = batch["lidar"].to(device)
            state = batch["state"].to(device)
            prev_a = batch["prev_action"].to(device)
            target = batch["target_action"].to(device)

            B, L, C, H, W = imgs.shape
            imgs_flat = imgs.view(B*L, C, H, W)
            with torch.no_grad():
                img_emb_flat = encoder(imgs_flat)
            img_emb = img_emb_flat.view(B, L, -1)   # (B,L,128)

            tokens = torch.cat([img_emb, lidar, state, prev_a], dim=-1)  # (B,L,input_dim)

            logits = policy(tokens)
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            pred = logits.argmax(dim=-1)
            total_correct += (pred == target).sum().item()
            total += B

        train_loss = total_loss / total
        train_acc = total_correct / total

        # Validation
        policy.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                imgs = batch["images"].to(device)
                lidar = batch["lidar"].to(device)
                state = batch["state"].to(device)
                prev_a = batch["prev_action"].to(device)
                target = batch["target_action"].to(device)

                B, L, C, H, W = imgs.shape
                img_flat = imgs.view(B*L, C, H, W)
                img_emb_flat = encoder(img_flat)
                img_emb = img_emb_flat.view(B, L, -1)

                tokens = torch.cat([img_emb, lidar, state, prev_a], dim=-1)
                logits = policy(tokens)
                pred = logits.argmax(dim=-1)
                val_correct += (pred == target).sum().item()
                val_total += B

        val_acc = val_correct / val_total
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"policy": policy.state_dict()}, "models/transformer_policy.pt")

if __name__ == "__main__":
    train_il()
