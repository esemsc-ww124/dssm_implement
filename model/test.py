# model/test_user_tower.py
import torch
from user_tower import UserTower

if __name__ == "__main__":
    tower = UserTower(num_items=200000, embed_dim=64)
    seq = torch.tensor([[1, 2, 3, 0, 0], [5, 6, 7, 8, 9]])
    out = tower(seq)
    print(out.shape)  # [2, 64]
