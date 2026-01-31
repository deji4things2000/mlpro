# models/transformer_policy.py
import torch
import torch.nn as nn

class TransformerPolicy(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=4, num_layers=3, num_actions=4, seq_len=8):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim, d_model)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, num_actions)

    def forward(self, tokens):
        # tokens: (B, L, input_dim)
        x = self.input_proj(tokens) + self.pos_emb[:, :tokens.size(1), :]
        x = self.encoder(x)           # (B, L, d_model)
        x_last = x[:, -1, :]         # last timestep
        logits = self.action_head(x_last)
        return logits
