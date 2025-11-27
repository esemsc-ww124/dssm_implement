# model/user_tower.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, shared_embedding):
        super().__init__()
        self.item_embedding = shared_embedding  # 共享 embedding

    def forward(self, seq):
        """
        seq: LongTensor [batch, seq_len]
        """
        # [B, seq_len, D]
        item_emb = self.item_embedding(seq)

        # mask 去掉 padding=0
        mask = (seq != 0).float().unsqueeze(-1)   # [B, seq_len, 1]
        masked_emb = item_emb * mask

        sum_emb = masked_emb.sum(dim=1)           # [B, D]
        seq_len = mask.sum(dim=1) + 1e-9          # [B, 1]

        user_emb = sum_emb / seq_len              # [B, D]

        # normalize（保持与 item tower 一致）
        user_emb = F.normalize(user_emb, dim=-1)

        return user_emb
