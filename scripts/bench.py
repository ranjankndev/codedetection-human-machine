"""Latency / throughput / parameter table for the README.

    python scripts/bench.py --config local_cpu --out results --models codebert,gpad,control,lora

Architectures are built fresh (untrained): latency does not depend on the weight values, and this keeps
the benchmark runnable without shipping checkpoints. Sequence length comes from the preset, because
truncation length is the main cost lever in production.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from codedet.bench import benchmark_model
from codedet.config import Config, RunPaths
from codedet.features import STYLE_DIM
from codedet.utils import get_logger, read_json, set_seed, write_json
from codedet.zoo import build, uses_style

log = get_logger("bench")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="smoke")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="codebert,control,gpad,lora")
    ap.add_argument("--task", choices=["A", "B"], default="A")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    paths = RunPaths.make(a.out)
    set_seed(cfg.seed)
    report_file = Path(a.out) / "data_report.json"
    num_labels = int(read_json(report_file)[a.task]["num_labels"]) if report_file.exists() else 2
    batch_sizes = tuple(cfg.get("bench.batch_sizes", [1, 32]))
    out = {"preset": cfg.preset, "base_model": cfg.get("model.base"), "seq_len": cfg.max_len, "models": {}}
    out_file = Path(a.out) / "bench.json"

    for name in [m.strip() for m in a.models.split(",") if m.strip()]:
        try:
            model, info = build(name, cfg, num_labels)
            vocab = int(getattr(model.encoder.config, "vocab_size", 1000))
            res = benchmark_model(model, name, seq_len=cfg.max_len, batch_sizes=batch_sizes,
                                  repeats=int(cfg.get("bench.repeats", 20)),
                                  warmup=int(cfg.get("bench.warmup", 3)),
                                  style_dim=STYLE_DIM if uses_style(name) else None, vocab=vocab)
            res["info"] = {k: v for k, v in info.items() if k != "gpad_config"}
            out["models"][name] = res
            best = min((r for r in res["rows"] if "p50_ms" in r), key=lambda r: r["p50_ms"], default=None)
            log.info(f"{name}: {res['params']['total_M']}M params, fastest p50 {best['p50_ms']:.1f} ms "
                     f"({best['variant']}/{best['device']}/bs{best['batch_size']})" if best else name)
        except Exception as e:
            out["models"][name] = {"error": f"{type(e).__name__}: {e}"[:300]}
            log.warning(f"{name}: benchmark failed ({e})")
        write_json(out, out_file)
    print(f"wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
