# model/item_tower_pairwise.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class ItemTower(nn.Module):
    def __init__(
        self,
        num_items,
        num_cats,
        vocab_size,
        embed_dim=64,
        text_dim=None,
    ):
        super().__init__()

        if text_dim is None:
            text_dim = embed_dim

        # 1) item id embedding
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim)

        # 2) 多类目 embedding
        self.category_embedding = nn.Embedding(num_cats + 1, embed_dim)

        # 3) 文本 embedding (title + description)
        # vocab_size 来自 text_vocab.pkl 的 len(vocab)
        # 约定 0 是 PAD，1..N 是词，UNK 在 vocab 里某个 id
        self.text_embedding = nn.Embedding(
            vocab_size + 1,
            text_dim,
            padding_idx=0
        )

        # 4) rating 特征 (overall_mean, log1p(count))
        self.rating_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, embed_dim),
        )

        # 5) 融合 MLP：item_emb + cat_vec + rating_emb + text_vec
        in_dim = embed_dim + embed_dim + embed_dim + text_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, item_ids, cat_ids, rating_feats, text_ids):
        """
        item_ids:     [B]
        cat_ids:      [B, C]    多类目 id
        rating_feats: [B, 2]    (overall_mean, log(1+count))
        text_ids:     [B, L]    文本 token id 序列
        """
        # 1) item id
        item_emb = self.item_embedding(item_ids)            # [B, D]

        # 2) 多类目 mean pooling
        cat_emb = self.category_embedding(cat_ids)          # [B, C, D]
        cat_mask = (cat_ids != 0).float().unsqueeze(-1)     # [B, C, 1]
        if cat_mask.sum() > 0:
            cat_sum = (cat_emb * cat_mask).sum(dim=1)       # [B, D]
            cat_len = cat_mask.sum(dim=1) + 1e-9            # [B, 1]
            cat_vec = cat_sum / cat_len                     # [B, D]
        else:
            cat_vec = torch.zeros_like(item_emb)

        # 3) rating 特征
        rating_emb = self.rating_mlp(rating_feats)          # [B, D]

        # 4) 文本侧：title + description mean pooling
        txt_emb = self.text_embedding(text_ids)             # [B, L, Dt]
        txt_mask = (text_ids != 0).float().unsqueeze(-1)    # [B, L, 1]
        if txt_mask.sum() > 0:
            txt_sum = (txt_emb * txt_mask).sum(dim=1)       # [B, Dt]
            txt_len = txt_mask.sum(dim=1) + 1e-9            # [B, 1]
            txt_vec = txt_sum / txt_len                     # [B, Dt]
        else:
            txt_vec = torch.zeros(
                txt_emb.size(0), txt_emb.size(-1),
                device=txt_emb.device
            )

        # 5) 融合
        x = torch.cat([item_emb, cat_vec, rating_emb, txt_vec], dim=-1)
        item_vec = self.mlp(x)                              # [B, D]

        return item_vec
