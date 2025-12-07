# model/dual_tower_pairwise.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualTowerPairwise(nn.Module):
    def __init__(self, user_tower, item_tower):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

    def forward(self, batch):
        """
        batch 来自 Dataset __getitem__，需要包含：
          user_id, seq, seq_len,
          pos_item, neg_item,
          pos_cat_ids, neg_cat_ids,
          pos_rating, neg_rating,
          pos_text, neg_text
        """
        user_ids = batch["user_id"]      # [B]
        seq = batch["seq"]               # [B, L]
        seq_len = batch["seq_len"]       # [B]

        pos_item = batch["pos_item"]     # [B]
        neg_item = batch["neg_item"]     # [B]

        pos_cat_ids = batch["pos_cat_ids"]   # [B, C]
        neg_cat_ids = batch["neg_cat_ids"]

        pos_rating = batch["pos_rating"]     # [B, 2]
        neg_rating = batch["neg_rating"]

        pos_text = batch["pos_text"]         # [B, L_txt]
        neg_text = batch["neg_text"]

        # 1) user 向量
        user_vec = self.user_tower(user_ids, seq, seq_len)         # [B, D]

        # 2) 正/负 item 向量
        pos_vec = self.item_tower(pos_item, pos_cat_ids,
                                  pos_rating, pos_text)            # [B, D]
        neg_vec = self.item_tower(neg_item, neg_cat_ids,
                                  neg_rating, neg_text)            # [B, D]

        # 3) 归一化 + BPR loss
        user_vec = F.normalize(user_vec, dim=-1)
        pos_vec = F.normalize(pos_vec, dim=-1)
        neg_vec = F.normalize(neg_vec, dim=-1)

        pos_score = (user_vec * pos_vec).sum(dim=-1)               # [B]
        neg_score = (user_vec * neg_vec).sum(dim=-1)               # [B]

        diff = pos_score - neg_score
        loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()

        return loss, pos_score, neg_score
