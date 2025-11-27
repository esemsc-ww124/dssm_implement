# preprocess/build_text_vocab.py
import re
from collections import Counter

def tokenize(text):
    return re.findall(r"[A-Za-z]+", text.lower())

def build_text_vocab(item_meta, min_freq=5):
    counter = Counter()

    for meta in item_meta.values():
        tokens = tokenize(meta["title"]) + tokenize(meta["description"])
        counter.update(tokens)

    # 高频词表
    vocab = {w: i+1 for i, (w, c) in enumerate(counter.items()) if c >= min_freq}
    vocab["<UNK>"] = len(vocab) + 1

    return vocab
