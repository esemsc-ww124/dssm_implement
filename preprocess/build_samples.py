# -*- coding: utf-8 -*-
# @Author  : Jerry Wenjie Wu
# @Time    : 19/11/2025 10:11

# preprocess/build_samples.py

def build_samples(user_sequences, user2id, item2id, min_seq_len=2):
    """
    user_sequences: user -> [item1, item2, item3 ...]
    """
    samples = []

    for user, items in user_sequences.items():
        if len(items) < min_seq_len:
            continue  # 序列太短无法训练

        uid = user2id[user]
        item_ids = [item2id[i] for i in items if i in item2id]

        # 构造训练样本，最后一个item作为正例
        for idx in range(1, len(item_ids)):
            seq = item_ids[:idx]      # 输入序列
            pos = item_ids[idx]       # 正例 item

            samples.append({
                "user": uid,
                "sequence": seq,
                "pos_item": pos
            })

    return samples
