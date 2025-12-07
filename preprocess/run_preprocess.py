# preprocess/run_preprocess.py
import pickle

from parse_reviews import build_user_sequences
from parse_metadata import build_item_meta
from calc_rating_stats import calc_rating_stats
from build_vocab import build_user_item_vocab, build_category_vocab
from build_text_vocab import build_text_vocab
from build_samples import build_samples

reviews_path = "../data/Books.json.gz"
meta_path = "../data/meta_Books.json.gz"

print("Step 1: Parse reviews...")
user_sequences, user_hotness = build_user_sequences(reviews_path)

print("Step 2: Parse metadata...")
item_meta = build_item_meta(meta_path)

print("Step 3: Rating stats...")
item_stats = calc_rating_stats(reviews_path)

print("Step 4: Vocab...")
user2id, item2id = build_user_item_vocab(user_sequences)
cat2id = build_category_vocab(item_meta)
vocab = build_text_vocab(item_meta)

print("Step 5: Build samples...")
samples = build_samples(
    user_sequences, user2id, item2id
)

# 保存所有文件
with open("../data/user_sequences.pkl", "wb") as f:
    pickle.dump(user_sequences, f)

with open("../data/user2id.pkl", "wb") as f:
    pickle.dump(user2id, f)

with open("../data/item2id.pkl", "wb") as f:
    pickle.dump(item2id, f)

with open("../data/item_meta.pkl", "wb") as f:
    pickle.dump(item_meta, f)

with open("../data/item_stats.pkl", "wb") as f:
    pickle.dump(item_stats, f)

with open("../data/cat2id.pkl", "wb") as f:
    pickle.dump(cat2id, f)

with open("../data/text_vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

with open("../data/train_samples.pkl", "wb") as f:
    pickle.dump(samples, f)


print("All DONE!")
