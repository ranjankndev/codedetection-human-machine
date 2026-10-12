"""Label handling. Ported from legacy/taskb (normalize_family, mark_seen, id map).

IMPORTANT: the official `label_to_id.json` in each task folder is the source of truth.
FAMILIES below is only a fallback that mirrors the order used in the old Task B scripts.
"""
from __future__ import annotations

import pandas as pd

HUMAN = "human"

FAMILIES = [  # fallback ordering (old Task B scripts)
    HUMAN, "deepseek", "qwen", "01-ai", "bigcode", "gemma", "phi",
    "meta-llama", "ibm-granite", "mistral", "openai",
]
ID2FAMILY = {i: f for i, f in enumerate(FAMILIES)}
FAMILY2ID = {f: i for i, f in ID2FAMILY.items()}


def is_human(generator: str) -> bool:
    return str(generator).strip().lower() == HUMAN


def normalize_family(gen: str) -> str:
    """'meta-llama/CodeLlama-7b' -> 'meta-llama'; 'gpt-4o-mini' -> 'openai'; 'Human' -> 'human'."""
    gen = str(gen)
    if is_human(gen):
        return HUMAN
    fam = gen.split("/")[0].strip().lower()
    if fam.startswith(("gpt", "openai")):
        return "openai"
    if fam.startswith("mistral"):
        return "mistral"
    if fam.startswith(("gemma", "google")):
        return "gemma"
    if fam.startswith(("microsoft", "phi")):
        return "phi"
    if fam.startswith("deepseek"):
        return "deepseek"
    return fam


def add_family(df: pd.DataFrame, gen_col: str = "generator") -> pd.DataFrame:
    out = df.copy()
    out["family"] = out[gen_col].map(normalize_family)
    return out


def binary_labels(df: pd.DataFrame) -> pd.Series:
    """Task A labels: 0 = human, 1 = machine. Uses `label` if present, else derives from `generator`.
    (Check the official label map once: do not assume the polarity blindly.)"""
    if "label" in df.columns:
        return df["label"].astype(int)
    return (~df["generator"].map(is_human)).astype(int)


def mark_seen(df: pd.DataFrame, seen_generators, gen_col: str = "generator") -> pd.DataFrame:
    """Add column `seen` in {'seen','unseen'}; human code is always 'seen'."""
    seen = {str(s).lower() for s in seen_generators}
    out = df.copy()
    out["seen"] = out[gen_col].map(lambda g: "seen" if is_human(g) or str(g).lower() in seen else "unseen")
    return out
