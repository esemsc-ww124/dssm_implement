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

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Apple MPS")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")

def main():
    DEVICE = get_device()

    item2id = load_vocab("../data/item2id.pkl")
    num_items = len(item2id)

    # ===== DataLoader（更快） =====
    dataset = AmazonDataset("../data/train_samples.pkl", max_seq_len=MAX_SEQ_LEN)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True if DEVICE.type == "cuda" else False,
        num_workers=4
    )

    # ===== Shared embedding on device =====
    shared_item_embedding = nn.Embedding(
        num_items + 1, EMBED_DIM, padding_idx=0
    ).to(DEVICE)

    user_tower = UserTower(shared_item_embedding).to(DEVICE)
    item_tower = ItemTower(shared_item_embedding).to(DEVICE)
    model = DualTower(user_tower, item_tower).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # ===== Training loop =====
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()

        for batch_idx, (seq, pos_item) in enumerate(dataloader):

            seq = seq.to(DEVICE, non_blocking=True)
            pos_item = pos_item.to(DEVICE, non_blocking=True)

            # ===== Mixed Precision Training =====
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                loss, _, _ = model(seq, pos_item)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"[Epoch {epoch}] batch={batch_idx}, loss={loss.item():.4f}")

        print(f"Epoch {epoch} finished, AvgLoss={total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "../data/dual_tower.pt")
    print("Model saved at ../data/dual_tower.pt")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
