# preprocess/build_vocab.py
def build_user_item_vocab(user_sequences):
    users = list(user_sequences.keys())
    user2id = {u: i+1 for i, u in enumerate(users)}

    all_items = set()
    for seq in user_sequences.values():
        all_items.update(seq)
    item2id = {i: k+1 for k, i in enumerate(all_items)}

    return user2id, item2id

def build_category_vocab(item_meta):
    cats = {meta["category"] for meta in item_meta.values()}
    cat2id = {c: i+1 for i, c in enumerate(cats)}
    return cat2id
