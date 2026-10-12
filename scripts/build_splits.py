"""Materialise train / ID-val / OOD splits for a task, once, so every model sees identical rows.

    python scripts/build_splits.py --task both --config local_cpu --data fixtures/ --out results
"""
from __future__ import annotations

import argparse
from pathlib import Path

from codedet.config import Config, RunPaths
from codedet.splits import build
from codedet.utils import get_logger, read_json, set_seed, write_json

log = get_logger("build_splits")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["A", "B", "both"], default="both")
    ap.add_argument("--config", default="smoke")
    ap.add_argument("--data", default=None, help="override data.root from the config")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    set_seed(cfg.seed)
    paths = RunPaths.make(a.out)
    data_root = a.data or cfg.get("data.root", "/kaggle/input")

    report_file = Path(a.out) / "data_report.json"
    report = read_json(report_file) if report_file.exists() else {}
    for task in (["A", "B"] if a.task == "both" else [a.task]):
        r = build(task, cfg, paths, data_root)
        report[task] = r
        log.info(f"task {task}: sizes={r['sizes']} dedup={r['dedup']} labels={r['num_labels']}")
        write_json(report, report_file)   # per task: a later crash never loses the earlier one
    print(f"wrote {report_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
