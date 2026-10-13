"""Inference helpers. Ported from legacy/taska/inference.py + model_loader.py (predict_batch, load_*)."""
from __future__ import annotations

import numpy as np
import torch

from .datasets import CodeDataset, make_collate
from .eval import softmax
from .features import StyleScaler, style_matrix
from .trainloop import predict_logits


def predict_texts(model, tokenizer, texts, scaler: StyleScaler | None = None, max_len: int = 256,
                  batch_size: int = 32, temperature: float = 1.0, device: str | None = None):
    """Returns (probs [N,K], logits [N,K]). Labels are dummy zeros; style features are z-scored with `scaler`."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    style = scaler.transform(style_matrix(texts)) if scaler is not None else None
    ds = CodeDataset(texts, np.zeros(len(texts), dtype=int), tokenizer, max_len=max_len, style=style)
    logits, _ = predict_logits(model, ds, make_collate(tokenizer.pad_token_id), batch_size=batch_size, device=device)
    return softmax(logits, temperature), logits
