# preprocess/build_samples.py
def build_samples(user_sequences, user2id, item2id):
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

            samples.append({
                "user_id": uid,
                "sequence": seq,
                "pos_item": pos_id
            })

    return samples
