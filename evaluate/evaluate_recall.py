import torch
import pickle
import faiss
from tqdm import tqdm

from model.user_tower import UserTower
from model.item_tower import ItemTower
from model.dual_tower import DualTower
from train.dataset import AmazonDataset

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_vocab(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_samples(path, n=10000):
    """加载一小部分样本作为测试集"""
    with open(path, "rb") as f:
        samples = pickle.load(f)
    return samples[:n]   # 只取 1 万条做评估

def build_faiss_index(item_embs):
    item_embs = item_embs.numpy().astype("float32")
    dim = item_embs.shape[1]

    index = faiss.IndexFlatIP(dim)  # 内积召回
    index.add(item_embs)
    return index

def recall_at_k(model, index, test_samples, k=50):
    hits = 0

    for sample in tqdm(test_samples):
        seq = sample["sequence"][-20:]
        pos_item = sample["pos_item"]

        seq = [0] * (20 - len(seq)) + seq
        seq = torch.tensor(seq).unsqueeze(0).to(DEVICE)

        user_emb = model.user_tower(seq).detach().cpu().numpy()

        D, I = index.search(user_emb, k)
        topk_items = I[0]

        if pos_item in topk_items:
            hits += 1

    return hits / len(test_samples)

def main():
    item2id = load_vocab("../data/item2id.pkl")
    num_items = len(item2id)

    # 加载模型
    user_tower = UserTower()
    item_tower = ItemTower()
    model = DualTower(user_tower, item_tower)

    model.load_state_dict(torch.load("../data/dual_tower.pt", map_location=DEVICE))
    model = model.to(DEVICE).eval()

    trained_item_tower = model.item_tower

    item_ids = torch.arange(1, num_items + 1).to(DEVICE)
    item_embs = trained_item_tower(item_ids).detach().cpu()

    index = build_faiss_index(item_embs)

    test_samples = load_samples("../data/train_samples.pkl", n=5000)

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
        pos_emb = model.item_tower(pos_tensor)

        # normalize to match Faiss
        user_emb_n = torch.nn.functional.normalize(user_emb, dim=-1)
        pos_emb_n = torch.nn.functional.normalize(pos_emb, dim=-1)

        sim = (user_emb_n @ pos_emb_n.T).item()

    print("Similarity(user → pos_item) =", sim)
    print("===== End Debug =====")

    score = recall_at_k(model, index, test_samples, k=50)
    print("Recall@50 =", score)

if __name__ == "__main__":
    main()
