"""Kaggle script-kernel entry point. Rendered and pushed by scripts/kaggle_drive.py.

Script kernels take no CLI arguments, so the preset and the source location are substituted into the
placeholders below before the push. The kernel:
  1. finds the competition data under /kaggle/input (never hard-codes a folder name),
  2. gets the project source (attached `codedet-src` dataset, or a git clone when a repo URL is given),
  3. installs only the packages Kaggle lacks - NEVER torch/transformers/numpy,
  4. runs scripts/run_all.py, which owns the stage DAG, the resume markers and the degradation ladder,
  5. copies the small result files to /kaggle/working/results so `kaggle kernels output` can fetch them.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PRESET = "{{PRESET}}"            # rendered by kaggle_drive.py
SOURCE_DATASET = "{{SOURCE}}"    # directory name under /kaggle/input holding the repo, or ""
REPO_URL = "{{REPO_URL}}"        # git URL alternative, or ""

WORK = Path("/kaggle/working")
REPO = WORK / "codedet"
OUT = WORK / "results"
T0 = time.time()


def sh(cmd: list[str], check: bool = True) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, text=True)
    if check and p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd)}")
    return p.returncode


def find_data_root() -> str:
    """The parquet files can sit anywhere under /kaggle/input; discover rather than assume."""
    hits = glob.glob("/kaggle/input/**/*.parquet", recursive=True)
    print(f"found {len(hits)} parquet files under /kaggle/input")
    for h in hits[:10]:
        print("  ", h)
    return "/kaggle/input"


def get_source() -> Path:
    if SOURCE_DATASET:
        src = Path("/kaggle/input") / SOURCE_DATASET
        if src.exists():
            if REPO.exists():
                shutil.rmtree(REPO)
            shutil.copytree(src, REPO)
            print(f"source copied from attached dataset {src}")
            return REPO
        print(f"WARNING: attached dataset {src} not found")
    if REPO_URL:
        sh(["git", "clone", "--depth", "1", REPO_URL, str(REPO)])
        return REPO
    raise SystemExit("no source: attach the codedet-src dataset or set REPO_URL")


def install(repo: Path) -> None:
    for pkg in ("peft", "datasketch", "lightgbm"):
        try:
            __import__(pkg)
        except ImportError:
            sh([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
    sh([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)], check=False)


def main() -> int:
    print(f"preset={PRESET} python={sys.version.split()[0]}")
    try:
        import torch

        print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print("gpu:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("torch unavailable:", e)

    repo = get_source()
    install(repo)
    data_root = find_data_root()
    OUT.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ, PYTHONPATH=str(repo / "src"), PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, str(repo / "scripts" / "run_all.py"), "--preset", PRESET,
           "--data", data_root, "--out", str(OUT), "--readme", str(repo / "README.md")]
    print(f"$ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, env=env, cwd=str(repo), text=True).returncode

    # keep the kernel output small: JSON / MD / PNG only (no parquet, npz, checkpoints)
    for pattern in ("data", "preds", "ckpt", "cache"):
        p = OUT / pattern
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    print(f"run_all returned {rc} after {(time.time() - T0) / 60:.1f} min")
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            print("  result:", f.relative_to(OUT), f"{f.stat().st_size / 1024:.0f} KB")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
