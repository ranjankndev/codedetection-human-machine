"""Fit the classical baselines on the materialised splits and write predictions for every split.

    python scripts/run_classical.py --task A --config local_cpu --out results

Cheap models are run with several seeds (config `seeds.cheap_models`); seed 0 provides the canonical
prediction files and the per-seed macro-F1 spread is recorded in metrics.json, so the report can state
seed variance instead of pretending a single run is the truth.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from codedet.artifacts import probs_as_logits, save_preds, write_model_metrics
from codedet.classical import ClassicalModel, permutation_importance_gbdt, top_ngrams
from codedet.config import SPLITS, Config, RunPaths
from codedet.eval import macro_f1_fast
from codedet.utils import get_logger, read_json, set_seed, write_json

log = get_logger("classical")

CLASSICAL = ("majority", "length_only", "tfidf_lr", "style_gbdt", "lm_gbdt")


def load_splits(paths: RunPaths, task: str) -> dict:
    out = {}
    for split in SPLITS[task.upper()]:
        f = paths.split_file(task, split)
        if f.exists():
            out[split] = pd.read_parquet(f)
    if "train" not in out:
        raise FileNotFoundError(f"no train split for task {task}; run build_splits.py first")
    return out


def load_lm(paths: RunPaths, task: str, splits: dict, lm_tag: str) -> dict | None:
    """LM features are optional: the stage can be skipped (no internet / no GPU) without failing the run."""
    found = {}
    for name, frame in splits.items():
        f = paths.lm_cache(task, name, lm_tag)
        if not f.exists():
            return None
        arr = np.load(f)
        if len(arr) != len(frame):
            log.warning(f"LM cache {f.name} has {len(arr)} rows, split has {len(frame)}; ignoring LM features")
            return None
        found[name] = arr
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["A", "B", "both"], default="both")
    ap.add_argument("--config", default="smoke")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default=None, help="comma separated subset override")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    paths = RunPaths.make(a.out)
    report = read_json(Path(a.out) / "data_report.json")
    lm_tag = str(cfg.get("lm.model", "lm")).split("/")[-1]
    n_seeds = int(cfg.get("seeds.cheap_models", 3))
    status: dict = {}

    for task in (["A", "B"] if a.task == "both" else [a.task]):
        if task not in report:
            log.warning(f"task {task} missing from data_report.json; skipping")
            continue
        num_labels = int(report[task]["num_labels"])
        label_names = report[task]["label_names"]
        splits = load_splits(paths, task)
        lm = load_lm(paths, task, splits, lm_tag)
        wanted = [m for m in (a.models.split(",") if a.models else cfg.models(task)) if m in CLASSICAL]
        balanced = task.upper() == "B"
        for name in wanted:
            if name == "lm_gbdt" and lm is None:
                log.warning("lm_gbdt skipped: no LM feature cache")
                status[f"{task}:{name}"] = "skipped_no_lm_features"
                continue
            t0 = time.time()
            seed_f1, canonical = [], None
            for s in range(max(1, n_seeds) if name != "majority" else 1):
                set_seed(cfg.seed + s)
                model = ClassicalModel(name, num_labels, seed=cfg.seed + s, balanced=balanced)
                model.fit(splits["train"]["code"].tolist(), splits["train"]["y"].values,
                          lm["train"] if lm else None)
                preds = {}
                for split, frame in splits.items():
                    p = model.predict_proba(frame["code"].tolist(), lm[split] if lm else None)
                    preds[split] = p
                val = "id_val" if "id_val" in preds else "train"
                seed_f1.append(macro_f1_fast(splits[val]["y"].values, preds[val].argmax(1), num_labels))
                if s == 0:
                    canonical = (model, preds)
            model, preds = canonical
            for split, p in preds.items():
                save_preds(paths.pred_file(task, name, split), probs_as_logits(p),
                           splits[split]["y"].values, splits[split]["row_id"].values)
            extra = {
                "seconds": round(time.time() - t0, 1), "backend": model.backend,
                "seeds": len(seed_f1), "seed_macro_f1": [round(f, 4) for f in seed_f1],
                "seed_macro_f1_mean": float(np.mean(seed_f1)), "seed_macro_f1_std": float(np.std(seed_f1)),
                "features": model.result().__dict__, "label_names": label_names,
                "uses_lm_features": bool(name == "lm_gbdt" and lm is not None),
            }
            if name == "tfidf_lr" and num_labels == 2:
                extra["top_ngrams"] = top_ngrams(model)
            if name in ("style_gbdt", "lm_gbdt", "length_only"):
                ev = "id_val" if "id_val" in splits else "train"
                extra["permutation_importance_id_val"] = permutation_importance_gbdt(
                    model, splits[ev]["code"].tolist(), splits[ev]["y"].values, lm[ev] if lm else None)
            m = write_model_metrics(paths, task, name, num_labels, extra)
            log.info(f"task {task} {name}: " + " ".join(f"{k}={v['macro_f1']:.3f}" for k, v in m["splits"].items())
                     + f" ({extra['seconds']}s, {model.backend})")
            status[f"{task}:{name}"] = "ok"
    write_json(status, Path(a.out) / "classical_status.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
