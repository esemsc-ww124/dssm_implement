# train/train_pairwise.py
# -*- coding: utf-8 -*-

import os
import pickle
import torch
from torch.utils.data import DataLoader

from train.dataset import AmazonPairDataset
from model.user_tower import UserTower
from model.item_tower import ItemTower
from model.dual_tower import DualTowerPairwise

BATCH_SIZE = 256
EPOCHS = 3
LR = 1e-3
EMBED_DIM = 64
MAX_SEQ_LEN = 20


def load_pkl(data_dir, name):
    with open(os.path.join(data_dir, name), "rb") as f:
        return pickle.load(f)


def main():
    data_dir = "../data"

    # 1) 读 vocab 大小
    user2id = load_pkl(data_dir, "user2id.pkl")
    item2id = load_pkl(data_dir, "item2id.pkl")
    cat2id = load_pkl(data_dir, "cat2id.pkl")
    text_vocab = load_pkl(data_dir, "text_vocab.pkl")

    num_users = len(user2id)
    num_items = len(item2id)
    num_cats = len(cat2id)
    vocab_size = len(text_vocab)

    print(f"num_users={num_users}, num_items={num_items}, num_cats={num_cats}, vocab_size={vocab_size}")

    # 2) Dataset & DataLoader
    dataset = AmazonPairDataset(
        data_dir=data_dir,
        max_seq_len=MAX_SEQ_LEN,
        max_cat_per_item=4,
        max_text_len=32,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # 3) Model
    user_tower = UserTower(
        num_users=num_users,
        num_items=num_items,
        embed_dim=EMBED_DIM,
    )

    item_tower = ItemTower(
        num_items=num_items,
        num_cats=num_cats,
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
    )

    model = DualTowerPairwise(user_tower, item_tower).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 4) Training loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            # 把 batch 里的 tensor 全部搬到 device
            batch = {k: v.to(device) for k, v in batch.items()}

            loss, pos_score, neg_score = model(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                print(f"[Epoch {epoch}] step={step}, loss={loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} finished, AvgLoss={avg_loss:.4f}")

    # 5) 保存模型
    os.makedirs(data_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(data_dir, "dual_tower_pairwise.pt"))
    print("Model saved at ../data/dual_tower_pairwise.pt")


if __name__ == "__main__":
    main()
