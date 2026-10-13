"""One command that runs the whole pipeline, survives partial failure, and records what happened.

    python scripts/run_all.py --preset local_cpu --cpu-only --data fixtures/ --out results

Design rules (PLAN section A.3-A.6):
  * every stage is a separate subprocess, so an OOM or a crash degrades the RUN, not the ORCHESTRATOR;
  * every finished stage drops a marker in `<out>/.done/`, so a re-run resumes instead of repeating;
  * REQUIRED stages failing fails the run; OPTIONAL stages failing are recorded and the run continues;
  * a stage that runs out of the global time budget is skipped, and neural stages first retry once in a
    cheaper configuration (fewer train rows, shorter sequences) before giving up;
  * `<out>/status.json` is the single machine-readable answer to "what state is this run in".
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from codedet.config import Config, RunPaths
from codedet.utils import get_logger, write_json
from codedet.zoo import is_neural

log = get_logger("run_all")
SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent


@dataclass
class Stage:
    name: str
    args: list[str]
    required: bool = True
    retry_cheaper: bool = False     # neural stages: retry once with fewer rows / shorter sequences
    max_minutes: float | None = None  # per-stage cap (PLAN A.6): one slow stage must not eat the run
    note: str = ""
    result: dict = field(default_factory=dict)


def build_stages(cfg: Config, a) -> list[Stage]:
    common = ["--config", a.config_path, "--out", a.out]
    cpu = ["--cpu-only"] if a.cpu_only else []
    optional = set(cfg.optional_stages())
    stages = [
        Stage("discover", ["discover_data.py", "--data", a.data, "--out", a.out], required=False),
        Stage("splits", ["build_splits.py", "--task", "both", "--data", a.data, *common]),
        Stage("lm_features", ["extract_lm_features.py", "--task", "both", *common, *cpu],
              required="lm" not in optional, max_minutes=float(cfg.get("stage_caps.lm_features", 45))),
        Stage("classical", ["run_classical.py", "--task", "both", *common],
              max_minutes=float(cfg.get("stage_caps.classical", 40))),
    ]
    for task in ("A", "B"):
        for model in cfg.models(task):
            if not is_neural(model):
                continue
            stages.append(Stage(
                f"train_{task}_{model}",
                ["train.py", "--task", task, "--model", model, *common, *cpu],
                required=model not in optional and model != "lora",
                retry_cheaper=True, max_minutes=float(cfg.get("stage_caps.train", 90))))
    if "stack" in set(cfg.models("A")) | set(cfg.models("B")):
        stages.append(Stage("stack", ["run_stack.py", "--task", "both", *common],
                            required="stack" not in optional))
    stages += [
        Stage("evaluate", ["evaluate.py", *common]),
        Stage("analyze", ["analyze.py", *common]),
        Stage("bench", ["bench.py", *common], required="bench" not in optional),
        Stage("report", ["make_report.py", "--out", a.out, "--readme", a.readme]),
    ]
    return stages


def run_stage(stage: Stage, timeout: float, log_dir: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, "-u", str(SCRIPTS / stage.args[0]), *stage.args[1:], *(extra or [])]
    t0 = time.time()
    log_file = log_dir / f"{stage.name}.log"
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    # stream to the log file rather than capturing: a long stage (LM features, a training run) must be
    # observable while it runs, not only after it finishes
    with open(log_file, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO), env=env)
        try:
            rc = proc.wait(timeout=max(30.0, timeout))
            timed_out = False
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            rc, timed_out = -9, True
    tail = ""
    try:
        tail = log_file.read_text(encoding="utf-8", errors="replace")[-600:]
    except OSError:
        pass
    if timed_out:
        return {"status": "failed", "seconds": round(time.time() - t0, 1), "returncode": rc,
                "error": f"timeout after {timeout:.0f}s", "log": str(log_file)}
    return {"status": "ok" if rc == 0 else "failed", "seconds": round(time.time() - t0, 1),
            "returncode": rc, "error": None if rc == 0 else tail, "log": str(log_file)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="smoke", help="config name or path")
    ap.add_argument("--data", default=None, help="data root (overrides the preset)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--readme", default=str(REPO / "README.md"))
    ap.add_argument("--only", default=None, help="comma separated stage names to run")
    ap.add_argument("--skip", default="", help="comma separated stage names to skip")
    ap.add_argument("--force", action="store_true", help="ignore .done markers")
    a = ap.parse_args()

    cfg = Config.load(a.preset)
    a.config_path = cfg.path or a.preset
    a.data = a.data or cfg.get("data.root", "/kaggle/input")
    paths = RunPaths.make(a.out)
    log_dir = Path(a.out) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    budget = float(cfg.get("budget_minutes", 60)) * 60
    started = time.time()
    stages = build_stages(cfg, a)
    only = {s.strip() for s in a.only.split(",")} if a.only else None
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}

    status = {"preset": cfg.preset, "config": a.config_path, "data_root": a.data,
              "budget_minutes": budget / 60, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "python": sys.version.split()[0], "stages": []}
    status_file = Path(a.out) / "status.json"

    try:
        import torch

        status["torch"] = torch.__version__
        status["cuda"] = bool(torch.cuda.is_available()) and not a.cpu_only
        if status["cuda"]:
            status["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        status["torch"] = None

    failed_required = []
    for stage in stages:
        marker = paths.done / f"{stage.name}.done"
        entry = {"stage": stage.name, "required": stage.required}
        if (only and stage.name not in only) or stage.name in skip:
            entry.update(status="skipped", seconds=0.0, error="not selected")
        elif marker.exists() and not a.force:
            entry.update(status="cached", seconds=0.0, error=None)
        else:
            remaining = budget - (time.time() - started)
            if remaining <= 60 and not stage.required:
                entry.update(status="skipped", seconds=0.0, error="out of time budget")
            else:
                allowed = remaining if remaining > 60 else 600
                if stage.max_minutes:
                    allowed = min(allowed, stage.max_minutes * 60)
                res = run_stage(stage, allowed, log_dir)
                entry.update(res)
                if res["status"] == "failed" and stage.retry_cheaper:
                    # degradation ladder rung: fewer train rows + shorter sequences, one retry
                    log.warning(f"{stage.name} failed; retrying cheaper")
                    res2 = run_stage(stage, max(120.0, budget - (time.time() - started)), log_dir,
                                     extra=["--max-train-rows", "1000", "--seeds", "1", "--no-stress"])
                    if res2["status"] == "ok":
                        res2["degradation"] = "retried with --max-train-rows 1000 --seeds 1 --no-stress"
                    entry.update(res2)
                if entry["status"] == "ok":
                    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
        if entry["status"] == "failed" and stage.required:
            failed_required.append(stage.name)
        log.info(f"{stage.name}: {entry['status']} ({entry.get('seconds', 0):.0f}s)"
                 + (f" :: {entry['error'][:120]}" if entry.get("error") else ""))
        status["stages"].append(entry)
        status["elapsed_minutes"] = round((time.time() - started) / 60, 2)
        status["failed_required"] = failed_required
        write_json(status, status_file)

    status["status"] = "ok" if not failed_required else "failed"
    degraded = [s for s in status["stages"] if s.get("degradation")]
    # stages excluded by --only / --skip are a user choice, not a gap in the project
    gap_stages = [s for s in status["stages"]
                  if s["status"] in ("skipped", "failed") and s.get("error") != "not selected"]
    skipped = [s["stage"] for s in gap_stages]
    status["degraded_stages"] = [s["stage"] for s in degraded]
    status["skipped_or_failed"] = skipped
    write_json(status, status_file)

    # KNOWN_GAPS.md is auto-written so the report can never silently omit a missing stage. Only the
    # generated block is replaced; notes written by hand around it survive.
    begin, end = "<!-- BEGIN GENERATED GAPS -->", "<!-- END GENERATED GAPS -->"
    gaps = [begin, "",
            f"Generated by `scripts/run_all.py` for preset `{cfg.preset}` ({status['started']}), "
            f"elapsed {status['elapsed_minutes']:.0f} min.", ""]
    if skipped:
        for s in gap_stages:
            if True:
                gaps.append(f"- **{s['stage']}** - {s['status']}"
                            + (" (required)" if s["required"] else " (optional)")
                            + (f": {(s.get('error') or '').strip()[:200]}" if s.get("error") else ""))
        gaps += ["", "Everything not listed here ran to completion."]
    else:
        gaps += ["Every stage of this run completed; nothing was skipped or degraded."]
    gaps += ["", end]
    block = "\n".join(gaps)
    gaps_file = REPO / "KNOWN_GAPS.md"
    prev = gaps_file.read_text(encoding="utf-8") if gaps_file.exists() else ""
    if begin in prev and end in prev:
        text = prev.split(begin)[0] + block + prev.split(end, 1)[1]
    else:
        text = (prev.rstrip() + "\n\n" if prev.strip() else "# Known gaps\n\n") + block + "\n"
    gaps_file.write_text(text, encoding="utf-8")

    print(json.dumps({"status": status["status"], "elapsed_minutes": status["elapsed_minutes"],
                      "failed_required": failed_required, "skipped_or_failed": skipped}, indent=2))
    return 0 if not failed_required else 1


if __name__ == "__main__":
    raise SystemExit(main())
