# -*- coding: utf-8 -*-
# @Author  : Jerry Wenjie Wu
# @Time    : 19/11/2025 10:11

# preprocess/build_vocab.py

def build_vocab(user_sequences):
    # user vocab
    users = list(user_sequences.keys())
    user2id = {u: idx+1 for idx, u in enumerate(users)}

    # item vocab
    all_items = set()
    for seq in user_sequences.values():
        all_items.update(seq)

    items = list(all_items)
    item2id = {i: idx+1 for idx, i in enumerate(items)}

    return user2id, item2id
