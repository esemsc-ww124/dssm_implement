# preprocess/parse_reviews.py
import gzip
import json
from collections import defaultdict

def parse(path):
    with gzip.open(path, 'r') as f:
        for line in f:
            yield json.loads(line)

def build_user_sequences(reviews_path):
    user_seq = defaultdict(list)
    user_hotness = defaultdict(int)

    for r in parse(reviews_path):
        user = r.get("reviewerID")
        item = r.get("asin")

        if not user or not item:
            continue

        user_seq[user].append(item)
        user_hotness[user] += 1

    # 去掉序列长度 < 2 的用户
    user_seq = {u: seq for u, seq in user_seq.items() if len(seq) >= 2}

    return user_seq, user_hotness
