# model/dual_tower.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DualTower(nn.Module):
    def __init__(self, user_tower, item_tower, temperature=0.07):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.temperature = temperature

    def forward(self, seq, pos_item):
        """
        seq: [B, seq_len]
        pos_item: [B]
        """

        # user 和 item 在各自 tower 中已经 normalize 了
        user_emb = self.user_tower(seq)             # [B, D]
        pos_emb = self.item_tower(pos_item)         # [B, D]

        # In-batch negative
        logits = torch.matmul(user_emb, pos_emb.t()) / self.temperature  # [B, B]
        labels = torch.arange(logits.size(0), device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss, user_emb, pos_emb
