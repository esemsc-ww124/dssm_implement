import torch
import pickle
import faiss
from tqdm import tqdm
import torch.nn.functional as F

from model.user_tower import UserTower
from model.item_tower import ItemTower
from model.dual_tower import DualTower

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()
print("Using device:", DEVICE)

def load_vocab(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_samples(path, n=10000):
    with open(path, "rb") as f:
        samples = pickle.load(f)
    return samples[:n]

def build_faiss_index(item_embs):
    item_embs = item_embs.numpy().astype("float32")
    dim = item_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(item_embs)
    return index

def recall_at_k(model, index, test_samples, k=50):
    hits = 0
    user_tower = model.user_tower

    for sample in tqdm(test_samples):
        seq = sample["sequence"][-20:]
        pos_item = sample["pos_item"]

        seq = [0] * (20 - len(seq)) + seq
        seq = torch.tensor(seq).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            user_emb = user_tower(seq)
            user_emb = F.normalize(user_emb, dim=-1).cpu().numpy()

        _, topk = index.search(user_emb, k)

        if pos_item in topk[0]:
            hits += 1

    return hits / len(test_samples)

def main():
    item2id = load_vocab("../data/item2id.pkl")
    num_items = len(item2id)

    # ========== 和训练保持一致（共享 embedding）==========
    shared_item_embedding = torch.nn.Embedding(num_items + 1, 64, padding_idx=0)

    user_tower = UserTower(shared_item_embedding)
    item_tower = ItemTower(shared_item_embedding)
    model = DualTower(user_tower, item_tower)

    model.load_state_dict(torch.load("../data/dual_tower.pt", map_location=DEVICE))
    model = model.to(DEVICE).eval()

    # item embeddings（必须 normalize）
    item_ids = torch.arange(1, num_items + 1).to(DEVICE)
    with torch.no_grad():
        item_embs = model.item_tower(item_ids)
        item_embs = F.normalize(item_embs, dim=-1).cpu()

    index = build_faiss_index(item_embs)

    test_samples = load_samples("../data/train_samples.pkl", n=5000)

    # ===== Debug：正例相似度 =====
    print("\n===== Debug 正例相似度 =====")
    sample = test_samples[0]
    seq = sample["sequence"][-20:]
    pos_item = sample["pos_item"]

    print("pos_item =", pos_item)

    seq = [0] * (20 - len(seq)) + seq
    seq = torch.tensor(seq).unsqueeze(0).to(DEVICE)
    pos_tensor = torch.tensor([pos_item]).to(DEVICE)

    with torch.no_grad():
        user_emb = model.user_tower(seq)
        pos_emb = model.item_towe
