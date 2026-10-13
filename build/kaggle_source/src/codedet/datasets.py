"""Torch dataset + dynamic-padding collator.
Ported idea from legacy GPADDataset/GPADDataCollator, with three changes:
  * pre-tokenised once (head+tail truncation) instead of per-item padding to max_length
  * style features are precomputed, z-scored arrays (not raw text re-processed inside forward())
  * dynamic padding per batch (big speed-up vs padding every sample to 512)
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import head_tail_ids


class CodeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = 256, head: int | None = None, style=None):
        self.ids = [head_tail_ids(tokenizer, str(t), max_len=max_len, head=head) for t in texts]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.style = None if style is None else np.asarray(style, dtype=np.float32)
        assert len(self.ids) == len(self.labels)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        item = {"input_ids": self.ids[i], "labels": int(self.labels[i])}
        if self.style is not None:
            item["style"] = self.style[i]
        return item


def make_collate(pad_id: int):
    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            mask[i, :L] = 1
        out = {"input_ids": ids, "attention_mask": mask, "labels": torch.tensor([b["labels"] for b in batch])}
        if "style" in batch[0]:
            out["style"] = torch.tensor(np.stack([b["style"] for b in batch]))
        return out

    return collate
