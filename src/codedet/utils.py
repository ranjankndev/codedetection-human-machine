"""Shared helpers. Fixes the missing `utils` module that the old trainer imported
(get_logger, set_seed, compute_metrics_macro, print_classification_report)."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed python, numpy and (if installed) torch/cuda."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def write_json(obj, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def read_json(path: str | os.PathLike):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def compute_metrics_macro(eval_pred) -> dict:
    """HF-Trainer-style metric fn: (logits, labels) -> macro-F1 / accuracy."""
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
    }


def print_classification_report(y_true, y_pred, target_names: Sequence[str] | None = None) -> str:
    from sklearn.metrics import classification_report

    rep = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    print(rep)
    return rep
