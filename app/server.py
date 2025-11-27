import torch
import faiss
import pickle
from model.user_tower import UserTower
from model.item_tower import ItemTower

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_vocab(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_faiss_index(item_embs):
    item_embs = item_embs.numpy().astype("float32")
    dim = item_embs.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(item_embs)
    return index

class RecallService:
    def __init__(self):
        self.item2id = load_vocab("../data/item2id.pkl")
        self.num_items = len(self.item2id)

        self.user_tower = UserTower(num_items=self.num_items, embed_dim=64)
        self.item_tower = ItemTower(num_items=self.num_items, embed_dim=64)

        self.user_tower.load_state_dict(torch.load("../data/dual_tower.pt", map_location=DEVICE))
        self.item_tower.load_state_dict(torch.load("../data/dual_tower.pt", map_location=DEVICE))  # same file
        self.user_tower.to(DEVICE).eval()
        self.item_tower.to(DEVICE).eval()

        item_ids = torch.arange(1, self.num_items+1).to(DEVICE)
        item_embs = self.item_tower(item_ids).detach().cpu()

        self.index = load_faiss_index(item_embs)

    def recommend(self, sequence, k=20):
        seq = sequence[-20:]
        seq = [0] * (20 - len(seq)) + seq
        seq = torch.tensor(seq).unsqueeze(0).to(DEVICE)

        user_emb = self.user_tower(seq).detach().cpu().numpy()
        D, I = self.index.search(user_emb, k)

        return I[0].tolist()    # top-k item id 返回

# 使用方法
if __name__ == "__main__":
    service = RecallService()
    recs = service.recommend([1234, 5321, 9987], k=10)
    print("推荐结果:", recs)
