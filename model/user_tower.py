# model/user_tower_pairwise.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class UserTower(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64):
        super().__init__()

        # 用户 ID embedding
        self.user_embedding = nn.Embedding(num_users + 1, embed_dim)

        # 序列里的 item embedding
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim)

        # 融合 MLP： [user_emb, seq_vec, seq_len] -> user_vec
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, user_ids, seq, seq_len):
        """
        user_ids: [B]
        seq:      [B, L]  （item_id 序列，0 是 padding）
        seq_len:  [B]     （有效长度）
        """
        # 1) user embedding
        u_emb = self.user_embedding(user_ids)          # [B, D]

        # 2) 序列 embedding + mean pooling
        seq_emb = self.item_embedding(seq)             # [B, L, D]
        mask = (seq != 0).float().unsqueeze(-1)        # [B, L, 1]

        if mask.sum() > 0:
            seq_sum = (seq_emb * mask).sum(dim=1)      # [B, D]
            seq_cnt = mask.sum(dim=1) + 1e-9           # [B, 1]
            seq_vec = seq_sum / seq_cnt                # [B, D]
        else:
            seq_vec = torch.zeros_like(u_emb)

        # 3) seq_len 当作一个简单的连续特征
        seq_len_feat = seq_len.unsqueeze(-1).float()   # [B, 1]

        # 4) 融合
        x = torch.cat([u_emb, seq_vec, seq_len_feat], dim=-1)  # [B, 2D+1]
        user_vec = self.mlp(x)                                 # [B, D]

        return user_vec
