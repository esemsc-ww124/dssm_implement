# -*- coding: utf-8 -*-
# @Author  : Jerry Wenjie Wu
# @Time    : 19/11/2025 10:26

# preprocess/run_preprocess.py
import pickle

from parse_reviews import build_user_sequences
from parse_metadata import build_item_meta
from build_vocab import build_vocab
from build_samples import build_samples

# 路径（根据你的目录调整）
reviews_path = "../data/Books.json.gz"
meta_path = "../data/meta_Books.json.gz"

# ① 解析用户行为序列
print("Step 1: Parsing reviews...")
user_sequences = build_user_sequences(reviews_path)
print("用户数:", len(user_sequences))

# 保存
with open("../data/user_sequences.pkl", "wb") as f:
    pickle.dump(user_sequences, f)

# ② 解析商品 metadata
print("Step 2: Parsing metadata...")
item_meta = build_item_meta(meta_path)
print("meta 商品数:", len(item_meta))

with open("../data/item_meta.pkl", "wb") as f:
    pickle.dump(item_meta, f)

# ③ 构建 user 和 item 的 vocab
print("Step 3: Building vocab...")
user2id, item2id = build_vocab(user_sequences)
print("user vocab:", len(user2id), "item vocab:", len(item2id))

with open("../data/user2id.pkl", "wb") as f:
    pickle.dump(user2id, f)
with open("../data/item2id.pkl", "wb") as f:
    pickle.dump(item2id, f)

# ④ 构建训练样本
print("Step 4: Building training samples...")
samples = build_samples(user_sequences, user2id, item2id)
print("训练样本数量:", len(samples))

with open("../data/train_samples.pkl", "wb") as f:
    pickle.dump(samples, f)

print("预处理完成！")
