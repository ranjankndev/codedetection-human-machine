"""Serving numbers: latency, throughput, parameter counts, memory.

Weights are irrelevant to latency, so the benchmark builds the architectures fresh instead of requiring
checkpoints (which are never committed). What it measures is what a deployment actually pays for:
the extra GPAD block, fp16 vs fp32, CPU dynamic int8 quantisation, and the cost of sequence length.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .models.gpad import count_params


def _percentiles(times: list[float]) -> dict:
    t = np.asarray(times) * 1000.0
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
            "mean_ms": float(t.mean()), "n": len(t)}


@torch.no_grad()
def time_forward(model, batch_size: int, seq_len: int, device: str, style_dim: int | None,
                 repeats: int = 20, warmup: int = 3, fp16: bool = False, vocab: int = 1000) -> dict:
    model.eval().to(device)
    ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    mask = torch.ones_like(ids)
    style = torch.randn(batch_size, style_dim, device=device) if style_dim else None
    kwargs = {"input_ids": ids, "attention_mask": mask}
    if style is not None:
        kwargs["style"] = style
    times: list[float] = []
    for i in range(warmup + repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device == "cuda"):
            model(**kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    out = _percentiles(times)
    out["throughput_per_s"] = float(batch_size / (out["p50_ms"] / 1000.0))
    out["batch_size"], out["seq_len"], out["device"] = batch_size, seq_len, device
    out["precision"] = "fp16" if (fp16 and device == "cuda") else "fp32"
    if device == "cuda":
        out["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        torch.cuda.reset_peak_memory_stats()
    return out


def quantize_dynamic_cpu(model):
    """int8 dynamic quantisation of the Linear layers - the cheapest real CPU speed-up for encoders."""
    return torch.quantization.quantize_dynamic(model.cpu(), {torch.nn.Linear}, dtype=torch.qint8)


def benchmark_model(model, name: str, seq_len: int, batch_sizes=(1, 32), repeats: int = 20, warmup: int = 3,
                    style_dim: int | None = None, vocab: int = 1000, include_int8: bool = True) -> dict:
    rows = []
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        for bs in batch_sizes:
            rows.append({"variant": "fp32", **time_forward(model, bs, seq_len, device, style_dim, repeats, warmup)})
            if device == "cuda":
                rows.append({"variant": "fp16",
                             **time_forward(model, bs, seq_len, device, style_dim, repeats, warmup, fp16=True)})
    if include_int8:
        try:
            qmodel = quantize_dynamic_cpu(model)
            for bs in batch_sizes:
                rows.append({"variant": "int8_dynamic",
                             **time_forward(qmodel, bs, seq_len, "cpu", style_dim, repeats, warmup, vocab=vocab)})
        except Exception as e:  # quantisation is unsupported on some builds; never fail the run for it
            rows.append({"variant": "int8_dynamic", "error": str(e)[:200]})
    return {"model": name, "params": count_params(model), "seq_len": seq_len, "rows": rows}
