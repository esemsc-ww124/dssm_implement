# preprocess/build_samples.py
from encode_text import encode_text

def build_samples(user_sequences, user2id, item2id,
                  item_meta, cat2id, item_stats, vocab):
    samples = []

    for user, items in user_sequences.items():
        uid = user2id[user]

        for i in range(1, len(items)):
            seq_items = items[:i]
            pos = items[i]

            if seq_items[-1] not in item2id or pos not in item2id:
                continue

            seq = [item2id[x] for x in seq_items]
            pos_id = item2id[pos]

            # item 特征
            meta = item_meta.get(pos, {
                "title": "",
                "description": "",
                "category": "unknown"
            })

            title_ids = encode_text(meta["title"], vocab)
            desc_ids = encode_text(meta["description"], vocab)
            cat_id = cat2id.get(meta["category"], 0)

            stats = item_stats.get(pos, {"overall_mean": 0.0, "review_count": 0})

            samples.append({
                "user_id": uid,
                "sequence": seq,
                "pos_item": pos_id,
                "category_id": cat_id,
                "title_ids": title_ids,
                "desc_ids": desc_ids,
                "rating_feat": [stats["overall_mean"], stats["review_count"]]
            })

    return samples
