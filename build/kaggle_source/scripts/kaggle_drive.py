"""Drive a Kaggle GPU run end to end from the laptop: upload source -> push kernel -> poll -> download.

    python scripts/kaggle_drive.py --preset smoke
    python scripts/kaggle_drive.py --preset lite --repo-url https://github.com/<user>/<repo>.git

What it does:
  1. packs this repo (code only - no data, no results, no checkpoints) into a private Kaggle dataset
     `<user>/codedet-src`, creating it the first time and versioning it afterwards;
  2. renders `kaggle/kaggle_entry.py` and `kernel-metadata.json` with the preset and the source name;
  3. `kaggle kernels push`, then polls `kaggle kernels status` until it stops running;
  4. `kaggle kernels output` into `--out`, and reports the run's own `status.json`.

Credentials are never read or printed by this script: the `kaggle` CLI reads them itself.
Exit codes: 0 ok, 1 run failed, 75 Kaggle refused the run (quota / limits) - the caller then finalises
with whatever results already exist.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from codedet.utils import get_logger, write_json

log = get_logger("kaggle_drive")
REPO = Path(__file__).resolve().parent.parent
EXIT_QUOTA = 75

SOURCE_INCLUDE = ("src", "scripts", "configs", "tests")
SOURCE_FILES = ("pyproject.toml", "requirements-extra.txt", "README.md", "CLAUDE.md", "PLAN.md")
QUOTA_PATTERNS = (r"quota", r"exceeded", r"too many", r"not allowed", r"limit")


def run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    log.info("$ " + " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")


def kaggle_username() -> str | None:
    """Ask the CLI, not the token file (the token must never be read by this process)."""
    p = run(["kaggle", "config", "view"], timeout=120)
    m = re.search(r"username:\s*(\S+)", (p.stdout or "") + (p.stderr or ""), re.I)
    if m:
        return m.group(1).strip()
    p = run(["kaggle", "datasets", "list", "-m", "--csv"], timeout=180)
    m = re.search(r"^([\w-]+)/", (p.stdout or ""), re.M)
    return m.group(1) if m else None


def pack_source(stage: Path, user: str, slug: str) -> Path:
    """Copy the code (and nothing else) into a dataset folder."""
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for d in SOURCE_INCLUDE:
        src = REPO / d
        if src.exists():
            shutil.copytree(src, stage / d,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.parquet", "*.npz", "*.npy"))
    for f in SOURCE_FILES:
        if (REPO / f).exists():
            shutil.copy2(REPO / f, stage / f)
    write_json({"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "CC0-1.0"}]},
               stage / "dataset-metadata.json")
    return stage


def push_source(stage: Path, user: str, slug: str) -> bool:
    exists = run(["kaggle", "datasets", "status", f"{user}/{slug}"], timeout=300)
    blob = (exists.stdout or "") + (exists.stderr or "")
    if exists.returncode == 0 and "404" not in blob and "not found" not in blob.lower():
        p = run(["kaggle", "datasets", "version", "-p", str(stage), "-m", "source update", "-d", "-r", "zip"], 1800)
    else:
        p = run(["kaggle", "datasets", "create", "-p", str(stage), "-r", "zip"], 1800)
    log.info((p.stdout or p.stderr or "").strip()[:400])
    return p.returncode == 0


def render_kernel(stage: Path, preset: str, user: str, source_slug: str, repo_url: str, dataset: str,
                  gpu: bool) -> Path:
    entry = (REPO / "kaggle" / "kaggle_entry.py").read_text(encoding="utf-8")
    entry = (entry.replace("{{PRESET}}", preset)
                  .replace("{{SOURCE}}", "" if repo_url else source_slug)
                  .replace("{{REPO_URL}}", repo_url or ""))
    kdir = stage.parent / "kernel"
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "kaggle_entry.py").write_text(entry, encoding="utf-8")
    sources = [] if repo_url else [f"{user}/{source_slug}"]
    meta = {
        "id": f"{user}/codedet-{preset}", "title": f"codedet {preset}", "code_file": "kaggle_entry.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": gpu, "enable_internet": True,
        "dataset_sources": [dataset] + sources, "competition_sources": [], "kernel_sources": [],
    }
    write_json(meta, kdir / "kernel-metadata.json")
    return kdir


def poll(kernel_id: str, timeout_min: int, every: int = 60) -> str:
    deadline = time.time() + timeout_min * 60
    last = ""
    while time.time() < deadline:
        p = run(["kaggle", "kernels", "status", kernel_id], timeout=300)
        blob = ((p.stdout or "") + (p.stderr or "")).strip()
        state = "unknown"
        for word in ("complete", "error", "cancelAcknowledged", "running", "queued"):
            if word.lower() in blob.lower():
                state = word
                break
        if blob != last:
            log.info(f"status: {blob[:200]}")
            last = blob
        if state in ("complete", "error", "cancelAcknowledged"):
            return state
        time.sleep(every)
    return "timeout"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="smoke")
    ap.add_argument("--out", default="results")
    ap.add_argument("--dataset", default="daniilor/semeval-2026-task13", help="competition data source")
    ap.add_argument("--repo-url", default="", help="public git URL (skips the source-dataset upload)")
    ap.add_argument("--source-slug", default="codedet-src")
    ap.add_argument("--timeout-min", type=int, default=90)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--skip-source", action="store_true", help="reuse the dataset already on Kaggle")
    a = ap.parse_args()

    if shutil.which("kaggle") is None:
        log.error("kaggle CLI not found; pip install kaggle and configure a token")
        return EXIT_QUOTA
    user = kaggle_username()
    if not user:
        log.error("could not determine the Kaggle username from the CLI (is the token configured?)")
        return EXIT_QUOTA
    log.info(f"kaggle user: {user}")

    stage = REPO / "build" / "kaggle_source"
    if not a.repo_url and not a.skip_source:
        pack_source(stage, user, a.source_slug)
        if not push_source(stage, user, a.source_slug):
            log.error("source dataset upload failed")
            return EXIT_QUOTA
        log.info("waiting 60s for the dataset version to become available")
        time.sleep(60)

    kdir = render_kernel(stage, a.preset, user, a.source_slug, a.repo_url, a.dataset, not a.no_gpu)
    push = run(["kaggle", "kernels", "push", "-p", str(kdir)], timeout=900)
    blob = ((push.stdout or "") + (push.stderr or "")).strip()
    log.info(blob[:400])
    if push.returncode != 0:
        if any(re.search(p, blob, re.I) for p in QUOTA_PATTERNS):
            log.error("Kaggle refused the push (quota or limits); finalise with existing results")
            return EXIT_QUOTA
        return 1

    kernel_id = f"{user}/codedet-{a.preset}"
    state = poll(kernel_id, a.timeout_min, a.poll_seconds)
    log.info(f"kernel finished in state {state!r}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dl = run(["kaggle", "kernels", "output", kernel_id, "-p", str(out)], timeout=1800)
    log.info((dl.stdout or dl.stderr or "")[:400])

    status_file = out / "status.json"
    if status_file.exists():
        status = json.loads(status_file.read_text(encoding="utf-8"))
        log.info(f"run status: {status.get('status')} in {status.get('elapsed_minutes')} min; "
                 f"failed required: {status.get('failed_required')}")
        print(json.dumps({"kernel_state": state, "run_status": status.get("status"),
                          "elapsed_minutes": status.get("elapsed_minutes"),
                          "failed_required": status.get("failed_required")}, indent=2))
        return 0 if status.get("status") == "ok" else 1
    log.error("no status.json in the kernel output")
    return 1 if state == "complete" else EXIT_QUOTA


if __name__ == "__main__":
    raise SystemExit(main())
