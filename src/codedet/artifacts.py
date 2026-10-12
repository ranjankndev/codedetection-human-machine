"""The file contract between stages.

Every model - classical or neural - writes one `probs_<split>.npz` per split holding
(logits, labels, ids). Evaluation NEVER re-runs a model: it only reads these files. That is what makes
"same rows for every model" checkable instead of hopeful, and what lets a crashed run resume.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .eval import macro_f1_fast
from .utils import write_json


def save_preds(path, logits, labels, ids) -> Path:
    """logits [N,K] float32, labels [N] int64, ids [N] str."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    ids = np.asarray([str(i) for i in ids])
    if not (len(logits) == len(labels) == len(ids)):
        raise ValueError(f"ragged prediction arrays: {len(logits)}, {len(labels)}, {len(ids)}")
    np.savez_compressed(path, logits=logits, labels=labels, ids=ids)
    return path


def load_preds(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {"logits": z["logits"], "labels": z["labels"], "ids": z["ids"].astype(str)}


def probs_as_logits(probs) -> np.ndarray:
    """Classical models predict probabilities; store log-probs so softmax()/temperature scaling still apply."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return np.log(p).astype(np.float32)


def write_model_metrics(paths, task: str, model: str, num_labels: int, extra: dict | None = None) -> dict:
    """Quick per-model metrics.json (progress visibility). The authoritative numbers come from evaluate.py."""
    out = {"task": task.upper(), "model": model, "splits": {}}
    model_dir = paths.preds / f"task_{task.lower()}" / model
    for f in sorted(model_dir.glob("probs_*.npz")):
        split = f.stem.replace("probs_", "")
        d = load_preds(f)
        out["splits"][split] = {"n": int(len(d["labels"])),
                                "macro_f1": macro_f1_fast(d["labels"], d["logits"].argmax(1), num_labels)}
    if extra:
        out.update(extra)
    write_json(out, paths.metrics_file(task, model))
    return out
