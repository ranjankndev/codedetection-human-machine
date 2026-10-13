"""Describe whatever parquet data is actually present (never assume Kaggle folder names).

    python scripts/discover_data.py --data /kaggle/input --out results
"""
from __future__ import annotations

import argparse
from pathlib import Path

from codedet.data import discover_parquets, load_split, summarize_split
from codedet.splits import find_label_map, find_task_parquets
from codedet.utils import get_logger, write_json

log = get_logger("discover")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/input", help="root to glob for *.parquet")
    ap.add_argument("--out", default="results")
    ap.add_argument("--peek-rows", type=int, default=2000, help="rows used for column statistics")
    a = ap.parse_args()

    files = discover_parquets(a.data)
    log.info(f"found {len(files)} parquet file(s) under {a.data}")
    report = {"root": a.data, "files": files, "tasks": {}}
    for task in ("A", "B"):
        cands = find_task_parquets(a.data, task)
        info = {"candidates": cands, "label_map": find_label_map(cands[0]) if cands else None}
        if cands:
            try:
                df = load_split(cands[0])
                if len(df) > a.peek_rows:
                    df = df.sample(a.peek_rows, random_state=0)
                info["columns"] = list(df.columns)
                info["summary"] = summarize_split(df, Path(cands[0]).name)
                if "code" in df.columns:
                    chars = df["code"].astype(str).str.len()
                    info["code_chars"] = {"mean": float(chars.mean()), "p50": float(chars.median()),
                                          "p95": float(chars.quantile(0.95)), "max": int(chars.max())}
            except Exception as e:  # discovery must never crash a run
                info["error"] = str(e)
        report["tasks"][task] = info
        log.info(f"task {task}: {len(cands)} candidate file(s)")
    out = Path(a.out) / "data_discovery.json"
    write_json(report, out)
    print(f"wrote {out}  ({len(files)} parquet files)")
    return 0 if files else 1


if __name__ == "__main__":
    raise SystemExit(main())
