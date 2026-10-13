"""Run configuration: YAML preset + CLI overrides + the on-disk layout of a run.

Why a class instead of a dict: every stage needs the same derived paths (splits, predictions,
checkpoints, caches) and the same "where do I write my metrics" answer. Keeping that in one place is
what makes the stages independently re-runnable and the orchestrator resumable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


class Config:
    """Thin dotted-key view over a preset YAML."""

    def __init__(self, raw: dict, path: str | None = None):
        self.raw = raw
        self.path = path

    @classmethod
    def load(cls, preset_or_path: str) -> "Config":
        p = Path(preset_or_path)
        if not p.exists():
            p = CONFIG_DIR / f"{preset_or_path}.yaml"
        with open(p) as f:
            raw = yaml.safe_load(f) or {}
        return cls(raw, str(p))

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cur = self.raw
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    # --- frequently used shortcuts -------------------------------------------------
    @property
    def preset(self) -> str:
        return str(self.get("preset", "custom"))

    @property
    def seed(self) -> int:
        return int(self.get("seed", 42))

    @property
    def max_len(self) -> int:
        return int(self.get("text.max_len", 256))

    @property
    def head_tokens(self) -> int:
        return int(self.get("text.head_tokens", int(self.max_len * 0.75)))

    @property
    def n_boot(self) -> int:
        return int(self.get("bootstrap.n_boot", 1000))

    def models(self, task: str) -> list[str]:
        return list(self.get(f"models.task_{task.lower()}", []) or [])

    def optional_stages(self) -> list[str]:
        return list(self.get("optional_stages", []) or [])

    def to_dict(self) -> dict:
        return {"preset": self.preset, "config_path": self.path, "seed": self.seed, **self.raw}


# Split names are part of the artifact contract: evaluation joins models by (task, split).
# "official_val" is present only when the release ships a labeled validation parquet; every stage
# iterates over the splits that actually exist on disk.
SPLITS = {"A": ["train", "id_val", "ood_lang", "official_val"],
          "B": ["train", "id_val", "ood_gen", "official_val"]}


@dataclass
class RunPaths:
    """Everything a stage needs to know about where files live.

    Heavy artifacts (parquet / npz / checkpoints) live under the run root but are git-ignored;
    only the small JSON / MD / PNG files are meant to be committed.
    """

    root: Path

    @classmethod
    def make(cls, out: str | os.PathLike) -> "RunPaths":
        p = cls(Path(out))
        for d in (p.root, p.data, p.preds, p.ckpt, p.cache, p.figures, p.done):
            d.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def preds(self) -> Path:
        return self.root / "preds"

    @property
    def ckpt(self) -> Path:
        return self.root / "ckpt"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def done(self) -> Path:
        return self.root / ".done"

    # --- artifact helpers ---------------------------------------------------------
    def split_file(self, task: str, split: str) -> Path:
        return self.data / f"task_{task.lower()}" / f"{split}.parquet"

    def pred_file(self, task: str, model: str, split: str) -> Path:
        return self.preds / f"task_{task.lower()}" / model / f"probs_{split}.npz"

    def model_dir(self, task: str, model: str) -> Path:
        return self.ckpt / f"task_{task.lower()}" / model

    def metrics_file(self, task: str, model: str) -> Path:
        return self.preds / f"task_{task.lower()}" / model / "metrics.json"

    def lm_cache(self, task: str, split: str, lm_tag: str) -> Path:
        return self.cache / f"lm_{lm_tag}_task{task.lower()}_{split}.npy"
