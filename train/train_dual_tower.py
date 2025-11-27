# train/train_dual_tower.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import AmazonDataset

from model.user_tower import UserTower
from model.item_tower import ItemTower
from model.dual_tower import DualTower
import pickle

BATCH_SIZE = 128
EPOCHS = 3
LR = 1e-3
EMBED_DIM = 64
MAX_SEQ_LEN = 20

def load_vocab(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    item2id = load_vocab("../data/item2id.pkl")
    num_items = len(item2id)

    dataset = AmazonDataset("../data/train_samples.pkl", max_seq_len=MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    shared_item_embedding = nn.Embedding(
        num_items + 1, EMBED_DIM, padding_idx=0
    )

    # 塔使用同一 embedding
    user_tower = UserTower(shared_item_embedding)
    item_tower = ItemTower(shared_item_embedding)

    model = DualTower(user_tower, item_tower)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # 优化器（包含全部参数）
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (seq, pos_item) in enumerate(dataloader):
            seq = seq.to(device)
            pos_item = pos_item.to(device)

            loss, _, _ = model(seq, pos_item)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"[Epoch {epoch}] batch={batch_idx}, loss={loss.item():.4f}")

        print(f"Epoch {epoch} finished, AvgLoss={total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "../data/dual_tower.pt")
    print("Model saved!")

if __name__ == "__main__":
    main()
