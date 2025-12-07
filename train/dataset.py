# train/dataset_pairwise.py
# -*- coding: utf-8 -*-

import math
import pickle
import random
import os

import torch
from torch.utils.data import Dataset

# 统一从 preprocess 里 import tokenize，避免重复定义
from preprocess.build_text_vocab import tokenize


class AmazonPairDataset(Dataset):
    def __init__(
        self,
        data_dir="../data",
        samples_file="train_samples.pkl",
        item2id_file="item2id.pkl",
        item_meta_file="item_meta.pkl",
        item_stats_file="item_stats.pkl",
        cat2id_file="cat2id.pkl",
        text_vocab_file="text_vocab.pkl",
        max_seq_len=20,
        max_cat_per_item=4,
        max_text_len=32,
    ):
        # ---------- 1. 加载各种字典 ----------
        def _load(name):
            with open(os.path.join(data_dir, name), "rb") as f:
                return pickle.load(f)

        self.samples = _load(samples_file)        # list[dict{user_id, sequence, pos_item}]
        self.item2id = _load(item2id_file)        # asin -> item_id
        self.item_meta = _load(item_meta_file)    # asin -> {title, description, categories}
        self.item_stats = _load(item_stats_file)  # asin -> {overall_mean, review_count}
        self.cat2id = _load(cat2id_file)          # category_str -> cat_id
        self.vocab = _load(text_vocab_file)       # token_str -> token_id

        self.PAD = 0
        self.UNK = self.vocab.get("<UNK>", 1)

        self.max_seq_len = max_seq_len
        self.max_cat_per_item = max_cat_per_item
        self.max_text_len = max_text_len

        self.num_items = len(self.item2id)
        self.num_cats = len(self.cat2id)
        self.vocab_size = len(self.vocab)

        # 反查 asin
        self.id2asin = {v: k for k, v in self.item2id.items()}

        # ---------- 2. 为每个 item 预计算特征 ----------
        # category: [num_items+1, max_cat_per_item]
        self.item_cat_ids = torch.zeros(
            self.num_items + 1, self.max_cat_per_item, dtype=torch.long
        )

        # rating_feat: [overall_mean, log(1+count)] -> [num_items+1, 2]
        self.item_rating_feat = torch.zeros(
            self.num_items + 1, 2, dtype=torch.float32
        )

        # 文本: [num_items+1, max_text_len]
        self.item_text_ids = torch.zeros(
            self.num_items + 1, self.max_text_len, dtype=torch.long
        )

        for item_id, asin in self.id2asin.items():
            meta = self.item_meta.get(asin, {})

            # 2.1 categories 全部用上
            cats = meta.get("categories", []) or []
            cat_ids = [self.cat2id.get(c, 0) for c in cats][: self.max_cat_per_item]
            if cat_ids:
                self.item_cat_ids[item_id, : len(cat_ids)] = torch.tensor(
                    cat_ids, dtype=torch.long
                )

            # 2.2 rating_feat: mean + log(1+count)
            stat = self.item_stats.get(asin)
            if stat is not None:
                mean = float(stat.get("overall_mean", 0.0))
                cnt = float(stat.get("review_count", 0.0))
                log_cnt = math.log1p(cnt)  # ⭐ log(1 + count)
            else:
                mean = 0.0
                log_cnt = 0.0

            self.item_rating_feat[item_id] = torch.tensor(
                [mean, log_cnt], dtype=torch.float32
            )

            # 2.3 文本：title + description
            title = meta.get("title", "") or ""
            desc = meta.get("description", "") or ""
            text = f"{title} {desc}"
            toks = tokenize(text)
            if toks:
                ids = [self.vocab.get(t, self.UNK) for t in toks][: self.max_text_len]
                self.item_text_ids[item_id, : len(ids)] = torch.tensor(
                    ids, dtype=torch.long
                )

        print(
            f"[AmazonPairDataset] samples={len(self.samples)}, "
            f"num_items={self.num_items}, num_cats={self.num_cats}, vocab_size={self.vocab_size}"
        )

    def __len__(self):
        return len(self.samples)

    def _sample_negative(self, pos_item):
        """简单的随机负采样：从 [1, num_items] 中采一个 ≠ pos_item 的 item_id"""
        neg = pos_item
        while neg == pos_item:
            neg = random.randint(1, self.num_items)
        return neg

    def __getitem__(self, idx):
        s = self.samples[idx]

        user_id = s["user_id"]          # int
        seq = s["sequence"]             # list[int], 长度可变
        pos_item = s["pos_item"]        # int

        # 1) 处理序列：截断 + 左侧补 0
        seq = seq[-self.max_seq_len :]
        seq_len = len(seq)
        if seq_len < self.max_seq_len:
            seq = [0] * (self.max_seq_len - seq_len) + seq

        # 2) 负采样
        neg_item = self._sample_negative(pos_item)

        # 3) 取出预计算好的 item 侧特征
        pos_cat_ids = self.item_cat_ids[pos_item]      # [C]
        neg_cat_ids = self.item_cat_ids[neg_item]

        pos_rating = self.item_rating_feat[pos_item]   # [2]
        neg_rating = self.item_rating_feat[neg_item]

        pos_text = self.item_text_ids[pos_item]        # [L_txt]
        neg_text = self.item_text_ids[neg_item]

        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "seq": torch.tensor(seq, dtype=torch.long),
            "seq_len": torch.tensor(seq_len, dtype=torch.float32),

            "pos_item": torch.tensor(pos_item, dtype=torch.long),
            "neg_item": torch.tensor(neg_item, dtype=torch.long),

            "pos_cat_ids": pos_cat_ids,
            "neg_cat_ids": neg_cat_ids,

            "pos_rating": pos_rating,
            "neg_rating": neg_rating,

            "pos_text": pos_text,
            "neg_text": neg_text,
        }
