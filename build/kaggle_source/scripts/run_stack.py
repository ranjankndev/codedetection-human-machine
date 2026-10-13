"""Late fusion: a GBDT on [transformer logits | stylometry | LM features].

    python scripts/run_stack.py --task A --config local_cpu --out results

Why it is here: ensembling is a standard interview topic and this is the cheapest honest version of it -
no extra training of the transformer, just a second-stage model over artifacts that already exist.

Caveat recorded in the metrics file: the stacker is fitted on the TRAIN split, where the transformer's
logits are in-sample and therefore over-confident. The ID-val / OOD numbers are still clean (those rows
were never seen by either stage), but the stacker is biased towards trusting the logits.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from codedet.artifacts import load_preds, probs_as_logits, save_preds, write_model_metrics
from codedet.classical import _gbdt
from codedet.config import SPLITS, Config, RunPaths
from codedet.features import STYLE_NAMES, StyleScaler, style_matrix
from codedet.utils import get_logger, read_json, set_seed

log = get_logger("stack")

BASE_PREFERENCE = ("gpad", "gpad_both", "codebert_weighted", "codebert", "control")


def pick_base(paths: RunPaths, task: str) -> str | None:
    for m in BASE_PREFERENCE:
        if (paths.preds / f"task_{task.lower()}" / m).exists():
            return m
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["A", "B", "both"], default="both")
    ap.add_argument("--config", default="smoke")
    ap.add_argument("--out", default="results")
    ap.add_argument("--base", default=None, help="which neural model's logits to stack on")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    paths = RunPaths.make(a.out)
    report = read_json(Path(a.out) / "data_report.json")
    lm_tag = str(cfg.get("lm.model", "lm")).split("/")[-1]
    rc = 0

    for task in (["A", "B"] if a.task == "both" else [a.task]):
        if task not in report:
            continue
        set_seed(cfg.seed)
        num_labels = int(report[task]["num_labels"])
        base = a.base or pick_base(paths, task)
        if base is None:
            log.warning(f"task {task}: no neural predictions to stack on; skipping")
            rc = 1
            continue
        t0 = time.time()
        feat_names = list(STYLE_NAMES)
        X, y, ids, frames = {}, {}, {}, {}
        for split in SPLITS[task.upper()]:
            f = paths.split_file(task, split)
            pf = paths.pred_file(task, base, split)
            if not (f.exists() and pf.exists()):
                continue
            frame = pd.read_parquet(f)
            pred = load_preds(pf)
            pos = {r: i for i, r in enumerate(pred["ids"])}
            idx = np.array([pos[r] for r in frame["row_id"].values])
            blocks = [pred["logits"][idx], style_matrix(frame["code"].tolist())]
            lm_file = paths.lm_cache(task, split, lm_tag)
            if lm_file.exists():
                arr = np.load(lm_file)
                if len(arr) == len(frame):
                    blocks.append(arr)
            X[split] = np.hstack(blocks).astype(np.float32)
            y[split] = frame["y"].values
            ids[split] = frame["row_id"].values
            frames[split] = frame
        if "train" not in X:
            log.warning(f"task {task}: no train block for stacking")
            rc = 1
            continue
        scaler = StyleScaler().fit(X["train"])
        est, backend = _gbdt(num_labels, cfg.seed, "balanced" if task.upper() == "B" else None)
        est.fit(scaler.transform(X["train"]), y["train"])
        for split in X:
            p = est.predict_proba(scaler.transform(X[split]))
            if p.shape[1] != num_labels:
                full = np.zeros((len(p), num_labels))
                for j, c in enumerate(getattr(est, "classes_", range(p.shape[1]))):
                    full[:, int(c)] = p[:, j]
                p = full
            save_preds(paths.pred_file(task, "stack", split), probs_as_logits(p), y[split], ids[split])
        m = write_model_metrics(paths, task, "stack", num_labels, {
            "base_model": base, "backend": backend, "seconds": round(time.time() - t0, 1),
            "n_features": int(X["train"].shape[1]), "feature_blocks": ["logits", "style"] +
            (["lm"] if X["train"].shape[1] > num_labels + len(feat_names) else []),
            "label_names": report[task]["label_names"],
            "caveat": "fitted on TRAIN logits, which are in-sample for the base transformer",
        })
        log.info(f"task {task} stack on {base}: " +
                 " ".join(f"{k}={v['macro_f1']:.3f}" for k, v in m["splits"].items()))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
