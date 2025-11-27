# preprocess/encode_text.py
from build_text_vocab import tokenize

def encode_text(text, vocab, max_len=32):
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return ids
