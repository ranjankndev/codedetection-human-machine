"""Cache TRUE predictive-entropy / likelihood features from a frozen causal code LM.

    python scripts/extract_lm_features.py --task both --config smoke --out results

This is the zero-shot-detection idea (DetectGPT / Binoculars family) done honestly: NLL and
H(p(.|x_<t)) from a small code LM, no training, no label leakage. It is also the stage most likely to be
unavailable (model download needs internet), so a failure here is reported as `skipped`, never as a crash:
every model that depends on it degrades to style-only features.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from codedet.config import SPLITS, Config, RunPaths
from codedet.lm_features import LM_DIM, LM_FEATURE_NAMES, lm_features
from codedet.utils import get_logger, set_seed, write_json

log = get_logger("lm_features")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["A", "B", "both"], default="both")
    ap.add_argument("--config", default="smoke")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None, help="override lm.n_rows (per split cap)")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    set_seed(cfg.seed)
    paths = RunPaths.make(a.out)
    lm_name = str(cfg.get("lm.model", "Qwen/Qwen2.5-Coder-0.5B"))
    lm_tag = lm_name.split("/")[-1]
    status = {"model": lm_name, "feature_names": LM_FEATURE_NAMES, "dim": LM_DIM, "splits": {}}
    out_file = Path(a.out) / "lm_features_status.json"

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(lm_name)
        model = AutoModelForCausalLM.from_pretrained(lm_name)
        device = "cpu" if a.cpu_only or not torch.cuda.is_available() else "cuda"
        status["load_seconds"] = round(time.time() - t0, 1)
        status["device"] = device
    except Exception as e:  # no internet / model gated / OOM at load
        status["status"] = "skipped"
        status["error"] = f"{type(e).__name__}: {e}"[:300]
        write_json(status, out_file)
        log.warning(f"LM features skipped: {status['error']}")
        print("skipped: " + status["error"])
        return 0

    cap = a.max_rows if a.max_rows is not None else int(cfg.get("lm.n_rows", 25000))
    for task in (["A", "B"] if a.task == "both" else [a.task]):
        for split in SPLITS[task.upper()]:
            f = paths.split_file(task, split)
            if not f.exists():
                continue
            frame = pd.read_parquet(f, columns=["code"])
            cache = paths.lm_cache(task, split, lm_tag)
            if cache.exists() and len(np.load(cache)) == len(frame):
                status["splits"][f"{task}:{split}"] = {"n": int(len(frame)), "cached": True}
                continue
            texts = frame["code"].astype(str).tolist()
            if len(texts) > cap:
                # features must cover the whole split (they are per-row inputs), so a cap means
                # trimming the split is NOT allowed: we just record that the budget was exceeded.
                log.warning(f"{task}:{split} has {len(texts)} rows > lm.n_rows={cap}; extracting anyway")
            t0 = time.time()
            feats = lm_features(model, tok, texts, max_len=int(cfg.get("lm.max_len", 256)),
                                batch_size=int(cfg.get("lm.batch_size", 16)),
                                device=status["device"])
            np.save(cache, feats.astype(np.float32))
            secs = round(time.time() - t0, 1)
            status["splits"][f"{task}:{split}"] = {"n": int(len(texts)), "seconds": secs,
                                                   "nll_mean": float(feats[:, 0].mean()),
                                                   "ent_mean": float(feats[:, 4].mean())}
            log.info(f"{task}:{split} {len(texts)} rows in {secs}s -> {cache.name}")
            write_json(status, out_file)
    status["status"] = "ok"
    write_json(status, out_file)
    print(f"wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
