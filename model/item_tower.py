# model/item_tower.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ItemTower(nn.Module):
    def __init__(self, shared_embedding):
        super().__init__()
        self.item_embedding = shared_embedding  # 共享 embedding

    def forward(self, item_ids):
        """
        item_ids: LongTensor [batch]
        """
        item_emb = self.item_embedding(item_ids)
        item_emb = F.normalize(item_emb, dim=-1)
        return item_emb
