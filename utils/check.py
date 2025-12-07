# preprocess/check_data.py
import pickle
import numpy as np

def load(name):
    with open(f"../data/{name}", "rb") as f:
        return pickle.load(f)

user_sequences = load("user_sequences.pkl")
user2id       = load("user2id.pkl")
item2id       = load("item2id.pkl")
item_meta     = load("item_meta.pkl")
item_stats    = load("item_stats.pkl")
cat2id        = load("cat2id.pkl")
vocab         = load("text_vocab.pkl")
samples       = load("train_samples.pkl")

print("用户数:", len(user2id))
print("物品数:", len(item2id))
print("有行为的用户数:", len(user_sequences))
print("有 meta 的 item 数:", len(item_meta))
print("有评分统计的 item 数:", len(item_stats))
print("类别数:", len(cat2id))
print("词表大小:", len(vocab))
print("训练样本数:", len(samples))
