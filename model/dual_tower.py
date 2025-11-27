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

        # 1. user embedding
        user_emb = self.user_tower(seq)  # [B, D]

        # 2. Positive item embedding
        pos_emb = self.item_tower(pos_item)  # [B, D]

        # ⭐ normalize，防止 dot product 受到 norm 干扰
        user_emb = F.normalize(user_emb, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)

        # 3. In-batch negative (InfoNCE)
        """
        logits[i][j] = sim(user_i, pos_item_j)
        """
        logits = torch.matmul(user_emb, pos_emb.t()) / self.temperature  # [B, B]

        # 4. labels = diagonal
        labels = torch.arange(logits.size(0), device=logits.device)

        # 5. InfoNCE loss
        loss = F.cross_entropy(logits, labels)

        return loss, user_emb, pos_emb
