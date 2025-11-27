# model/item_tower.py
import torch
import torch.nn as nn

class ItemTower(nn.Module):
    def __init__(self, shared_embedding):
        super().__init__()
        self.item_embedding = shared_embedding

    def forward(self, item_ids):
        """
        item_ids: LongTensor [batch]
        """
        # [batch, embed_dim]
        item_emb = self.item_embedding(item_ids)

        return item_emb
