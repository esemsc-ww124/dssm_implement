# preprocess/check_samples_quality.py
# -*- coding: utf-8 -*-

import pickle
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load(name):
    path = os.path.join(DATA_DIR, name)
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    print("=== 训练样本质量检查 ===")

    user2id = load("user2id.pkl")
    item2id = load("item2id.pkl")
    cat2id = load("cat2id.pkl")
    vocab = load("text_vocab.pkl")
    samples = load("train_samples.pkl")

    num_users = len(user2id)
    num_items = len(item2id)
    num_cats = len(cat2id)
    vocab_size = len(vocab)

    print(f"num_users={num_users}, num_items={num_items}, num_cats={num_cats}, vocab_size={vocab_size}")
    print(f"样本数={len(samples)}")

    bad_user = bad_item = bad_seq = bad_cat = bad_title = bad_desc = bad_rating = 0
    seq_lens = []

    for s in samples:
        uid = s.get("user_id")
        seq = s.get("sequence", [])
        pos = s.get("pos_item")
        cat = s.get("category_id", 0)
        title_ids = s.get("title_ids", [])
        desc_ids = s.get("desc_ids", [])
        rating_feat = s.get("rating_feat", [])

        # 1) user_id 合法
        if not isinstance(uid, int) or not (1 <= uid <= num_users):
            bad_user += 1

        # 2) pos_item 合法
        if not isinstance(pos, int) or not (1 <= pos <= num_items):
            bad_item += 1

        # 3) seq 非空 & 每个 item 在范围内
        if (not isinstance(seq, list)) or len(seq) == 0:
            bad_seq += 1
        else:
            for x in seq:
                if not isinstance(x, int) or x <= 0 or x > num_items:
                    bad_seq += 1
                    break
        if isinstance(seq, list):
            seq_lens.append(len(seq))

        # 4) category_id 合法（允许 0 表示 unknown）
        if not isinstance(cat, int) or cat < 0 or cat > num_cats:
            bad_cat += 1

        # 5) 文本 token 不越界（允许 0 padding）
        for t in title_ids:
            if not isinstance(t, int) or t < 0 or t > vocab_size:
                bad_title += 1
                break

        for t in desc_ids:
            if not isinstance(t, int) or t < 0 or t > vocab_size:
                bad_desc += 1
                break

        # 6) rating_feat 长度为 2，且没有 None
        if not (isinstance(rating_feat, (list, tuple)) and len(rating_feat) == 2):
            bad_rating += 1
        else:
            if any(v is None for v in rating_feat):
                bad_rating += 1

    print("\n=== 异常计数 ===")
    print("bad_user   :", bad_user)
    print("bad_item   :", bad_item)
    print("bad_seq    :", bad_seq)
    print("bad_cat    :", bad_cat)
    print("bad_title  :", bad_title)
    print("bad_desc   :", bad_desc)
    print("bad_rating :", bad_rating)

    if seq_lens:
        print("\n=== 序列长度统计 ===")
        print("min_len:", int(np.min(seq_lens)))
        print("max_len:", int(np.max(seq_lens)))
        print("mean_len:", float(np.mean(seq_lens)))

    print("\n检查完成 ✅")


if __name__ == "__main__":
    main()
