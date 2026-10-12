"""Losses for imbalanced classification (Task B: ~88% human)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights_from_counts(counts, power: float = 0.5) -> torch.Tensor:
    """w_c ∝ (1/n_c)^power, normalised to mean 1. power=1 is full inverse frequency (can be unstable); 0.5 is a safe default."""
    c = np.asarray(counts, dtype=np.float64)
    w = (1.0 / np.maximum(c, 1)) ** power
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        pt = logp.exp().gather(-1, target[:, None]).squeeze(-1)
        ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")
        return ((1 - pt) ** self.gamma * ce).mean()


class LogitAdjustedCE(nn.Module):
    """Menon et al. logit adjustment: train with logits + tau*log(prior); predict with plain logits."""

    def __init__(self, class_counts, tau: float = 1.0):
        super().__init__()
        prior = np.asarray(class_counts, dtype=np.float64)
        prior = prior / prior.sum()
        self.register_buffer("adj", torch.tensor(tau * np.log(prior + 1e-12), dtype=torch.float32))

    def forward(self, logits, target):
        return F.cross_entropy(logits.float() + self.adj, target)


def make_loss(kind: str = "ce", class_counts=None, gamma: float = 2.0, tau: float = 1.0,
              weight_power: float = 0.5, label_smoothing: float = 0.0) -> nn.Module:
    kind = kind.lower()
    if kind == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if class_counts is None:
        raise ValueError(f"loss {kind!r} needs class_counts")
    if kind == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights_from_counts(class_counts, weight_power),
                                   label_smoothing=label_smoothing)
    if kind == "focal":
        return FocalLoss(gamma, class_weights_from_counts(class_counts, weight_power))
    if kind == "logit_adjust":
        return LogitAdjustedCE(class_counts, tau)
    raise ValueError(f"unknown loss {kind!r}")
