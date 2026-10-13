"""Small, explicit PyTorch training loop (replaces the HF Trainer + Accelerator monkey-patches in the legacy code,
which broke across transformers versions: `evaluation_strategy` was renamed `eval_strategy`, etc.).

Features: AdamW + linear warmup/decay, fp16 autocast + GradScaler, grad accumulation, grad clipping,
per-epoch validation macro-F1, early stopping, best-state restore, CUDA-OOM degradation ladder.
"""
from __future__ import annotations

import copy
import gc
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import get_logger

log = get_logger("trainloop")


@dataclass
class TrainConfig:
    epochs: int = 2
    batch_size: int = 16
    lr: float = 2e-5
    head_lr: float = 1e-4           # newly initialised layers learn faster than the pretrained encoder
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_accum: int = 1
    fp16: bool = True
    max_grad_norm: float = 1.0
    patience: int = 2
    num_workers: int = 0
    log_every: int = 50
    grad_checkpointing: bool = False
    seed: int = 42


def _to_device(batch: dict, device):
    return {k: v.to(device) for k, v in batch.items()}


def _macro_f1(y, p, k):
    from .eval import macro_f1_fast

    return macro_f1_fast(y, p, k)


@torch.no_grad()
def predict_logits(model, ds, collate, batch_size=64, device=None, fp16=True):
    device = device or next(model.parameters()).device
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    outs, labs = [], []
    use_amp = fp16 and str(device).startswith("cuda")
    for batch in dl:
        batch = _to_device(batch, device)
        y = batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(**batch)
        outs.append(out["logits"].float().cpu())
        labs.append(y.cpu())
    return torch.cat(outs).numpy(), torch.cat(labs).numpy()


def fit(model, train_ds, val_ds, collate, cfg: TrainConfig, num_labels: int, device=None):
    """Returns history dict. Model is left holding the BEST (by val macro-F1) weights."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if cfg.grad_checkpointing and hasattr(model, "encoder") and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    torch.manual_seed(cfg.seed)

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    new_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    groups = [{"params": enc_params, "lr": cfg.lr}]
    if new_params:
        groups.append({"params": new_params, "lr": cfg.head_lr})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                    num_workers=cfg.num_workers, drop_last=len(train_ds) > cfg.batch_size)
    steps = max(1, (len(dl) // cfg.grad_accum) * cfg.epochs)
    from transformers import get_linear_schedule_with_warmup

    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * steps), steps)
    use_amp = cfg.fp16 and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist = {"train_loss": [], "val_macro_f1": [], "epoch_seconds": []}
    best, best_state, bad = -1.0, None, 0
    for ep in range(cfg.epochs):
        model.train()
        t0, run = time.time(), 0.0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl):
            batch = _to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch)["loss"] / cfg.grad_accum
            scaler.scale(loss).backward()
            run += loss.item() * cfg.grad_accum
            if (i + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if cfg.log_every and (i + 1) % cfg.log_every == 0:
                log.info(f"ep{ep} step {i+1}/{len(dl)} loss {run/(i+1):.4f}")
        logits, y = predict_logits(model, val_ds, collate, device=device, fp16=cfg.fp16)
        f1 = _macro_f1(y, logits.argmax(1), num_labels)
        hist["train_loss"].append(run / max(1, len(dl)))
        hist["val_macro_f1"].append(f1)
        hist["epoch_seconds"].append(time.time() - t0)
        log.info(f"epoch {ep}: loss {hist['train_loss'][-1]:.4f} val macro-F1 {f1:.4f} ({hist['epoch_seconds'][-1]:.0f}s)")
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad > cfg.patience:
                log.info("early stopping")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_val_macro_f1"] = best
    return hist


# ---------------------------------------------------------------- OOM degradation ladder
def oom_ladder(base: dict):
    """Yield parameter dicts, each cheaper than the last. Keys: batch_size, grad_accum, grad_checkpointing, max_len."""
    p = dict(base)
    yield dict(p)
    p["batch_size"], p["grad_accum"] = max(1, p["batch_size"] // 2), p["grad_accum"] * 2
    yield dict(p)
    p["grad_checkpointing"] = True
    yield dict(p)
    p["max_len"] = min(p["max_len"], 192)
    yield dict(p)
    p["max_len"], p["batch_size"], p["grad_accum"] = 128, max(1, p["batch_size"] // 2), p["grad_accum"] * 2
    yield dict(p)


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def run_with_oom_ladder(fn, base: dict):
    """fn(**params) -> result. Retries down the ladder on CUDA OOM; records which rung worked in result['degradation']."""
    last = None
    for rung, params in enumerate(oom_ladder(base)):
        try:
            res = fn(**params)
            if isinstance(res, dict):
                res["degradation_rung"] = rung
                res["params_used"] = params
            return res
        except Exception as e:  # noqa: BLE001
            if not is_oom(e):
                raise
            last = e
            log.warning(f"OOM at rung {rung} with {params}; backing off")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError("OOM at every rung of the ladder") from last
