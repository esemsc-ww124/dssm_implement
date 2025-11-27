# train/dataset.py
import torch
from torch.utils.data import Dataset
import pickle
import os

class AmazonDataset(Dataset):
    def __init__(self, samples_path, max_seq_len=20):
        assert os.path.exists(samples_path), f"{samples_path} not found"
        with open(samples_path, "rb") as f:
            self.samples = pickle.load(f)

        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def _pad_seq(self, seq):
        # 只保留最近 max_seq_len 个
        seq = seq[-self.max_seq_len:]
        # 左侧 padding，0 作为 PAD
        pad_len = self.max_seq_len - len(seq)
        return [0] * pad_len + seq

    def __getitem__(self, idx):
        data = self.samples[idx]
        seq = data["sequence"]      # 已经是 item_id 的整数列表
        pos_item = data["pos_item"] # 单个 item_id

        seq = self._pad_seq(seq)

        seq = torch.tensor(seq, dtype=torch.long)
        pos_item = torch.tensor(pos_item, dtype=torch.long)

        return seq, pos_item
