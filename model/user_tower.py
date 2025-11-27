# model/user_tower.py
import torch
import torch.nn as nn

class UserTower(nn.Module):
    def __init__(self, shared_embedding):
        super().__init__()
        self.item_embedding = shared_embedding

    def forward(self, seq):
        """
        seq: LongTensor [batch, seq_len]
        """
        # [B, seq_len, embed_dim]
        item_emb = self.item_embedding(seq)

        # mean pooling：注意要排除 padding=0
        mask = (seq != 0).float().unsqueeze(-1)      # [B, seq_len, 1]
        masked_emb = item_emb * mask                # padding 的 embedding 变成 0

        sum_emb = masked_emb.sum(dim=1)             # [B, embed_dim]
        seq_len = mask.sum(dim=1) + 1e-9            # [B, 1]
        user_emb = sum_emb / seq_len                # [B, embed_dim]

        return user_emb
