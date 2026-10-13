"""Turn prediction files into the numbers the README is allowed to quote.

    python scripts/evaluate.py --config local_cpu --out results

Reads `preds/**/probs_<split>.npz` ONLY - it never re-runs a model, which is what makes the comparisons
reproducible and the "identical rows" claim checkable. For every (model, split) it computes metrics with
a bootstrap 95% CI, per-slice breakdowns, calibration before/after temperature scaling (temperature
fitted on ID-val, applied to OOD), and the paired tests that decide which story the README tells:

    gpad vs control   -> does the attention-entropy bias help BEYOND the extra block's capacity?
    gpad vs codebert  -> does the whole GPAD package help at all?
    B: entropy_only vs codebert_weighted -> was the old "entropy-only is best" claim real?

Verdicts come from the CI of the paired difference, never from a point estimate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from codedet.artifacts import load_preds
from codedet.config import SPLITS, Config, RunPaths
from codedet.eval import (bootstrap_ci, ece, fit_temperature, mcnemar, metrics_dict, paired_bootstrap_diff,
                          per_slice, softmax)
from codedet.utils import get_logger, read_json, write_json

log = get_logger("evaluate")

PAIRED_TESTS = {
    "A": [("gpad", "control", "attention-entropy bias beyond extra capacity"),
          ("gpad", "codebert", "whole GPAD package vs plain CodeBERT"),
          ("gpad_sign_neg", "gpad", "sign of the entropy bias"),
          ("lora", "codebert", "LoRA vs full fine-tune"),
          ("lm_gbdt", "style_gbdt", "LM entropy features beyond stylometry"),
          ("tfidf_lr", "length_only", "content beyond length/whitespace shortcut")],
    "B": [("gpad_entropy_only", "codebert_weighted", "old claim: entropy-only is best"),
          ("gpad_both", "control", "entropy+style beyond capacity"),
          ("gpad_both", "codebert_weighted", "GPAD vs weighted CodeBERT"),
          ("tfidf_lr", "majority", "anything above the majority floor")],
}

SLICE_COLUMNS = ("language", "generator", "family", "domain", "seen")


def discover_models(paths: RunPaths, task: str) -> list[str]:
    d = paths.preds / f"task_{task.lower()}"
    return sorted(p.name for p in d.glob("*") if p.is_dir() and any(p.glob("probs_*.npz"))) if d.exists() else []


def split_files(paths: RunPaths, task: str, model: str) -> dict:
    d = paths.preds / f"task_{task.lower()}" / model
    return {f.stem.replace("probs_", ""): f for f in sorted(d.glob("probs_*.npz"))}


def aligned(pred: dict, order: np.ndarray) -> dict:
    """Reindex a prediction block onto a canonical row order (identical rows, identical order)."""
    pos = {rid: i for i, rid in enumerate(pred["ids"])}
    idx = np.array([pos[r] for r in order])
    return {"logits": pred["logits"][idx], "labels": pred["labels"][idx], "ids": pred["ids"][idx]}


def canonical_order(per_model: dict, split: str) -> np.ndarray | None:
    """Rows present for EVERY model that has this split (so no comparison is made on different samples)."""
    sets = [set(p[split]["ids"]) for p in per_model.values() if split in p]
    if not sets:
        return None
    common = set.intersection(*sets)
    any_ids = next(p[split]["ids"] for p in per_model.values() if split in p)
    return np.array([i for i in any_ids if i in common])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="smoke")
    ap.add_argument("--out", default="results")
    ap.add_argument("--task", choices=["A", "B", "both"], default="both")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    paths = RunPaths.make(a.out)
    report = read_json(Path(a.out) / "data_report.json")
    summary = {"preset": cfg.preset, "seed": cfg.seed, "n_boot": cfg.n_boot, "tasks": {}}
    out_file = Path(a.out) / "summary.json"

    for task in (["A", "B"] if a.task == "both" else [a.task]):
        if task not in report:
            continue
        num_labels = int(report[task]["num_labels"])
        names = report[task]["label_names"]
        models = discover_models(paths, task)
        if not models:
            log.warning(f"task {task}: no predictions found")
            continue
        # slice metadata (language / generator / ...) comes from the materialised splits, by row_id
        meta = {}
        for split in SPLITS[task.upper()]:
            f = paths.split_file(task, split)
            if f.exists():
                m = pd.read_parquet(f)
                meta[split] = m.set_index("row_id")
        per_model = {m: {s: load_preds(f) for s, f in split_files(paths, task, m).items()} for m in models}
        all_splits = sorted({s for p in per_model.values() for s in p})
        orders = {s: canonical_order(per_model, s) for s in all_splits}

        tblock: dict = {"num_labels": num_labels, "label_names": names, "models": {},
                        "rows_per_split": {s: (0 if orders[s] is None else int(len(orders[s]))) for s in all_splits},
                        "identical_rows": {}, "paired": {}, "data": report[task]["sizes"]}
        # make the "same rows" property an explicit, checkable claim
        for s in all_splits:
            present = [m for m in models if s in per_model[m]]
            tblock["identical_rows"][s] = {
                "models": present,
                "all_models_same_rows": bool(orders[s] is not None and
                                             all(len(per_model[m][s]["ids"]) == len(orders[s]) for m in present)),
            }

        temps = {}
        for m in models:
            mblock: dict = {"splits": {}}
            mfile = paths.metrics_file(task, m)
            if mfile.exists():
                raw = read_json(mfile)
                for k in ("seconds", "train_seconds", "backend", "device", "loss", "seeds", "seed_macro_f1",
                          "seed_macro_f1_mean", "seed_macro_f1_std", "params", "degradation_rung",
                          "uses_lm_features", "lora", "gpad_config", "stress_macro_f1", "history"):
                    if k in raw:
                        mblock[k] = raw[k]
            # temperature is fitted on ID-val and reused everywhere: that is the point of the experiment
            if "id_val" in per_model[m]:
                d = aligned(per_model[m]["id_val"], orders["id_val"])
                try:
                    temps[m] = fit_temperature(d["logits"], d["labels"])
                except Exception as e:  # pragma: no cover - degenerate logits
                    log.warning(f"{task}/{m}: temperature fit failed ({e})")
                    temps[m] = 1.0
                mblock["temperature_id_val"] = temps[m]
            for s, pred in per_model[m].items():
                order = orders[s]
                d = aligned(pred, order) if order is not None else pred
                y, logits = d["labels"], d["logits"]
                p = logits.argmax(1)
                block = metrics_dict(y, p, num_labels, names)
                block["macro_f1_ci"] = bootstrap_ci(y, p, num_labels, n_boot=cfg.n_boot, seed=cfg.seed)
                probs = softmax(logits)
                block["ece"] = ece(probs, y)
                if m in temps:
                    block["ece_temp_scaled"] = ece(softmax(logits, temps[m]), y)
                base = s.split("__")[0]
                if base in meta:
                    md = meta[base].reindex(d["ids"])
                    for col in SLICE_COLUMNS:
                        if col in md.columns and md[col].nunique() > 1:
                            block.setdefault("per_slice", {})[col] = per_slice(
                                md[col].astype(str).values, y, p, num_labels, min_n=max(10, len(y) // 50))
                mblock["splits"][s] = block
            tblock["models"][m] = mblock
            log.info(f"task {task} {m}: " +
                     " ".join(f"{s}={b['macro_f1']:.3f}" for s, b in mblock["splits"].items() if "__" not in s))

        # --- paired comparisons on identical rows -------------------------------------
        for left, right, question in PAIRED_TESTS[task.upper()]:
            if left not in per_model or right not in per_model:
                continue
            entry = {"question": question, "splits": {}}
            for s in all_splits:
                if "__" in s or s == "train" or s not in per_model[left] or s not in per_model[right]:
                    continue
                order = orders[s]
                dl, dr = aligned(per_model[left][s], order), aligned(per_model[right][s], order)
                assert np.array_equal(dl["labels"], dr["labels"]), "label mismatch after alignment"
                pl, pr = dl["logits"].argmax(1), dr["logits"].argmax(1)
                diff = paired_bootstrap_diff(dl["labels"], pl, pr, num_labels, n_boot=cfg.n_boot, seed=cfg.seed)
                entry["splits"][s] = {"n": int(len(order)), "bootstrap": diff,
                                      "mcnemar": mcnemar(dl["labels"], pl, pr)}
            sig = [s for s, v in entry["splits"].items()
                   if v["bootstrap"]["excludes_zero"] and v["bootstrap"]["diff"] > 0]
            ood = [s for s in sig if s.startswith("ood")]
            entry["significant_splits"] = sig
            entry["verdict"] = ("better on " + ", ".join(sig)) if sig else "no significant difference"
            entry["significant_on_ood"] = bool(ood)
            tblock["paired"][f"{left}_vs_{right}"] = entry
            log.info(f"task {task} {left} vs {right}: {entry['verdict']}")
        summary["tasks"][task] = tblock
        write_json(summary, out_file)

    # --- the decision rule from PLAN A.5, recorded so the README cannot drift from it ---
    a_tests = summary["tasks"].get("A", {}).get("paired", {})
    gpad_control = a_tests.get("gpad_vs_control")
    branch = "entropy_helps" if (gpad_control and gpad_control["significant_on_ood"]) else "no_gain_beyond_capacity"
    summary["readme_branch"] = {
        "branch": branch,
        "rule": "gpad - control paired-bootstrap 95% CI excludes 0 on >= 1 OOD split",
        "evidence": gpad_control["splits"] if gpad_control else None,
    }
    write_json(summary, out_file)
    print(f"wrote {out_file} (README branch: {branch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
