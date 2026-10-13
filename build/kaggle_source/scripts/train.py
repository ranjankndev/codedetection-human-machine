"""Train one neural model on the materialised splits and write predictions for every split.

    python scripts/train.py --task A --model gpad --config local_cpu --out results --cpu-only

One script for every ablation (see codedet/zoo.py for the name -> config table). It also scores the OOD
split under the shortcut perturbations while the model is still in memory, so `analyze.py` never has to
reload a checkpoint: every downstream stage reads prediction files only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from codedet.artifacts import save_preds, write_model_metrics
from codedet.config import SPLITS, Config, RunPaths
from codedet.datasets import CodeDataset, make_collate
from codedet.eval import macro_f1_fast
from codedet.features import StyleScaler, style_matrix
from codedet.losses import make_loss
from codedet.perturb import apply_perturbation
from codedet.trainloop import TrainConfig, fit, predict_logits, run_with_oom_ladder
from codedet.utils import get_logger, read_json, set_seed, write_json
from codedet.zoo import build, loss_kind, uses_style

log = get_logger("train")

STRESS_PERTURBATIONS = ("nocomments", "stripped")


def get_tokenizer(base_model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(base_model)


def load_split_frames(paths: RunPaths, task: str) -> dict:
    frames = {}
    for split in SPLITS[task.upper()]:
        f = paths.split_file(task, split)
        if f.exists():
            frames[split] = pd.read_parquet(f)
    if "train" not in frames:
        raise FileNotFoundError(f"no train split for task {task}; run build_splits.py first")
    return frames


def style_blocks(frames: dict, paths: RunPaths, task: str, lm_tag: str, with_lm: bool):
    """Style (+ optional LM) features, z-scored with TRAIN statistics only. Returns (blocks, scaler, used_lm)."""
    raw = {k: style_matrix(v["code"].tolist()) for k, v in frames.items()}
    used_lm = False
    if with_lm:
        lm = {}
        for k, v in frames.items():
            f = paths.lm_cache(task, k, lm_tag)
            if not f.exists():
                lm = {}
                break
            arr = np.load(f)
            if len(arr) != len(v):
                lm = {}
                break
            lm[k] = arr
        if lm:
            raw = {k: np.hstack([raw[k], lm[k]]) for k in raw}
            used_lm = True
        else:
            log.warning("--with-lm requested but no usable LM cache; continuing with style features only")
    scaler = StyleScaler().fit(raw["train"])
    return {k: scaler.transform(v) for k, v in raw.items()}, scaler, used_lm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["A", "B"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="smoke")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--with-lm", action="store_true", help="concatenate cached LM features into the style vector")
    ap.add_argument("--save-ckpt", action="store_true", help="keep weights (needed only by the demo)")
    ap.add_argument("--seeds", type=int, default=None, help="override seeds.transformers")
    ap.add_argument("--no-stress", action="store_true", help="skip the perturbed OOD scoring")
    ap.add_argument("--max-train-rows", type=int, default=None,
                    help="degradation rung: train on a stratified subsample (eval rows stay identical)")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    paths = RunPaths.make(a.out)
    report = read_json(Path(a.out) / "data_report.json")[a.task]
    num_labels, label_names = int(report["num_labels"]), report["label_names"]
    device = "cpu" if a.cpu_only or not torch.cuda.is_available() else "cuda"
    frames = load_split_frames(paths, a.task)
    if a.max_train_rows and len(frames["train"]) > a.max_train_rows:
        from codedet.data import stratified_subsample

        log.warning(f"degraded: training on {a.max_train_rows} of {len(frames['train'])} rows")
        frames["train"] = stratified_subsample(frames["train"], a.max_train_rows, by=["y"], seed=cfg.seed)
    lm_tag = str(cfg.get("lm.model", "lm")).split("/")[-1]

    need_style = uses_style(a.model) or a.with_lm
    blocks, scaler, used_lm = (style_blocks(frames, paths, a.task, lm_tag, a.with_lm)
                               if need_style else ({k: None for k in frames}, None, False))
    base_model = str(cfg.get("model.base", "microsoft/codebert-base"))
    tok = get_tokenizer(base_model)
    counts = np.bincount(frames["train"]["y"].values, minlength=num_labels).tolist()
    kind = loss_kind(a.model, a.task, cfg)
    n_seeds = a.seeds if a.seeds is not None else int(cfg.get("seeds.transformers", 1))

    tcfg_base = dict(
        batch_size=int(cfg.get("train.batch_size", 16)), grad_accum=int(cfg.get("train.grad_accum", 1)),
        grad_checkpointing=bool(cfg.get("train.grad_checkpointing", False)), max_len=cfg.max_len,
    )

    seed_f1, canonical = [], None
    for s in range(max(1, n_seeds)):
        seed = cfg.seed + s
        set_seed(seed)

        def run_once(batch_size, grad_accum, grad_checkpointing, max_len, _seed=seed):
            """One full training run at a given memory budget; retried cheaper by the OOM ladder."""
            model, info = build(a.model, cfg, num_labels, with_lm=used_lm)
            model.set_loss(make_loss(kind, class_counts=counts,
                                     gamma=float(cfg.get("train.focal_gamma", 2.0)),
                                     tau=float(cfg.get("train.logit_adjust_tau", 1.0))))
            ds = {k: CodeDataset(v["code"].tolist(), v["y"].values, tok, max_len=max_len,
                                 head=min(cfg.head_tokens, max_len), style=blocks[k])
                  for k, v in frames.items()}
            tcfg = TrainConfig(
                epochs=int(cfg.get("train.epochs", 1)), batch_size=batch_size,
                lr=float(cfg.get("train.lr", 2e-5)), head_lr=float(cfg.get("train.head_lr", 1e-4)),
                weight_decay=float(cfg.get("train.weight_decay", 0.01)),
                warmup_ratio=float(cfg.get("train.warmup_ratio", 0.1)), grad_accum=grad_accum,
                fp16=bool(cfg.get("train.fp16", True)) and device == "cuda",
                patience=int(cfg.get("train.patience", 1)), grad_checkpointing=grad_checkpointing, seed=_seed,
            )
            collate = make_collate(tok.pad_token_id)
            val = ds.get("id_val", ds["train"])
            t0 = time.time()
            hist = fit(model, ds["train"], val, collate, tcfg, num_labels, device=device)
            return {"model": model, "info": info, "history": hist, "datasets": ds, "collate": collate,
                    "train_seconds": round(time.time() - t0, 1), "tconfig": tcfg.__dict__, "max_len": max_len}

        res = run_with_oom_ladder(run_once, tcfg_base)
        model, collate = res["model"], res["collate"]
        preds = {}
        for split, dataset in res["datasets"].items():
            logits, y = predict_logits(model, dataset, collate, batch_size=int(cfg.get("eval.batch_size", 64)),
                                       device=device, fp16=bool(cfg.get("train.fp16", True)) and device == "cuda")
            preds[split] = (logits, y)
        val = "id_val" if "id_val" in preds else "train"
        seed_f1.append(macro_f1_fast(preds[val][1], preds[val][0].argmax(1), num_labels))
        log.info(f"seed {seed}: val macro-F1 {seed_f1[-1]:.4f} ({res['train_seconds']}s)")
        if s == 0:
            canonical = (res, preds)
        else:
            del model, res
            if device == "cuda":
                torch.cuda.empty_cache()

    res, preds = canonical
    model = res["model"]
    for split, (logits, y) in preds.items():
        save_preds(paths.pred_file(a.task, a.model, split), logits, y, frames[split]["row_id"].values)

    # --- shortcut stress test: same OOD rows, surface form removed -------------------
    stress = {}
    ood = [k for k in frames if k.startswith("ood")]
    if not a.no_stress and ood:
        oname = ood[0]
        frame = frames[oname]
        for pert in STRESS_PERTURBATIONS:
            codes = apply_perturbation(pert, frame["code"].tolist(), frame.get("language"))
            style = None
            if need_style:
                raw = style_matrix(codes)
                if used_lm:  # LM features are not recomputed for perturbed text (that needs the LM again);
                    pad = model.cfg.style_in_dim - raw.shape[1]   # zero-pad so the input width still matches
                    raw = np.hstack([raw, np.zeros((len(codes), pad), np.float32)])
                style = scaler.transform(raw)
            ds = CodeDataset(codes, frame["y"].values, tok, max_len=res["max_len"],
                             head=min(cfg.head_tokens, res["max_len"]), style=style)
            logits, y = predict_logits(model, ds, res["collate"], device=device,
                                       fp16=bool(cfg.get("train.fp16", True)) and device == "cuda")
            split_name = f"{oname}__{pert}"
            save_preds(paths.pred_file(a.task, a.model, split_name), logits, y, frame["row_id"].values)
            stress[split_name] = macro_f1_fast(y, logits.argmax(1), num_labels)
        log.info("stress: " + " ".join(f"{k}={v:.3f}" for k, v in stress.items()))

    if a.save_ckpt:
        d = paths.model_dir(a.task, a.model)
        model.save(str(d))
        if scaler is not None:
            write_json(scaler.to_dict(), d / "scaler.json")
        log.info(f"checkpoint -> {d}")

    extra = {
        "device": device, "loss": kind, "class_counts": counts, "label_names": label_names,
        "seeds": len(seed_f1), "seed_macro_f1": [round(f, 4) for f in seed_f1],
        "seed_macro_f1_mean": float(np.mean(seed_f1)), "seed_macro_f1_std": float(np.std(seed_f1)),
        "train_seconds": res["train_seconds"], "history": res["history"],
        "degradation_rung": res.get("degradation_rung", 0), "params_used": res.get("params_used", {}),
        "train_config": res["tconfig"], "uses_lm_features": used_lm, "stress_macro_f1": stress,
        **res["info"],
    }
    if scaler is not None:
        extra["style_scaler"] = scaler.to_dict()
    m = write_model_metrics(paths, a.task, a.model, num_labels, extra)
    log.info(f"task {a.task} {a.model}: " + " ".join(f"{k}={v['macro_f1']:.3f}" for k, v in m["splits"].items()))
    print(json.dumps({"task": a.task, "model": a.model,
                      "val_macro_f1": seed_f1[0], "seconds": res["train_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
