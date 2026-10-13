"""Build honest train / ID-val / OOD splits and record exactly what was built.

Three things here are not negotiable, because they are the fixes for the audit findings:
  * every model is evaluated on IDENTICAL rows per split -> rows carry a stable `row_id` and the
    splits are materialised once, to disk, before any model runs (old code compared n=349 vs n=17,226);
  * OOD is a held-out LANGUAGE (task A) / held-out GENERATORS (task B), not a random split;
  * train is deduplicated against every eval split (exact, plus MinHash near-dup when datasketch is
    installed), because near-duplicate competitive-programming solutions inflate scores.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (cap_human_share, dedup_exact, holdout_generator_split, holdout_language_split,
                   load_split, near_duplicate_mask, stratified_subsample, summarize_split)
from .labels import FAMILIES, HUMAN, add_family, is_human
from .utils import get_logger

log = get_logger("splits")

REQUIRED_COLUMNS = ("code",)


# ------------------------------------------------------------------ discovery
def find_task_parquets(root: str, task: str) -> list[str]:
    """Parquet files that look like the training data of `task`. Globs; never hard-codes a Kaggle path."""
    cands = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    tag = f"task_{task.lower()}"
    scoped = [p for p in cands if tag in p.replace("\\", "/").lower()] or cands
    train = [p for p in scoped if "train" in Path(p).name.lower()]
    return train or scoped


def find_validation_parquets(root: str, task: str) -> list[str]:
    """Official validation files, if the release ships them WITH labels.

    They are an in-distribution check against the organisers' own split, which is worth reporting next
    to our proxy OOD splits; `build()` keeps them only when the labels are actually present.
    """
    cands = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    tag = f"task_{task.lower()}"
    scoped = [p for p in cands if tag in p.replace("\\", "/").lower()] or cands
    return [p for p in scoped if "valid" in Path(p).name.lower()]


def find_label_map(parquet_path: str) -> dict | None:
    """The official `label_to_id.json` (if shipped next to the parquet) is the source of truth for names."""
    d = Path(parquet_path).parent
    for cand in (d, d.parent):
        f = cand / "label_to_id.json"
        if f.exists():
            try:
                with open(f) as fh:
                    return json.load(fh)
            except Exception as e:  # pragma: no cover - corrupt file
                log.warning(f"could not read {f}: {e}")
    return None


# ------------------------------------------------------------------ frame preparation
def row_ids(df: pd.DataFrame, task: str) -> pd.Series:
    """Stable id from the code text (so the same row keeps its id across presets and re-runs)."""
    h = df["code"].astype(str).map(lambda s: hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()[:16])
    pos = pd.Series(np.arange(len(df)), index=df.index).astype(str)
    return task.upper() + "-" + h + "-" + pos


def prepare_frame(df: pd.DataFrame, task: str, label_map: dict | None = None):
    """Normalise columns, derive the target `y`, and report anything surprising.

    Returns (frame, label_names, notes). `notes` records the label-polarity check: the old code derived
    the task-A label from the generator column without ever verifying the official polarity.
    """
    out = df.copy()
    notes: dict = {}
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"data is missing required column(s) {missing}; have {list(out.columns)}")
    for col in ("language", "generator", "domain"):
        if col not in out.columns:
            out[col] = "unknown"
            notes[f"missing_{col}"] = True
    out["language"] = out["language"].astype(str)
    out["generator"] = out["generator"].astype(str)
    out = add_family(out)
    out["row_id"] = row_ids(out, task)

    if task.upper() == "A":
        has_human_col = bool(out["generator"].map(is_human).any())
        machine = (~out["generator"].map(is_human)).astype(int)
        if "label" in out.columns and pd.api.types.is_numeric_dtype(out["label"]) and out["label"].nunique() <= 2:
            given = out["label"].astype(int)
            if has_human_col:
                agree = float((given == machine).mean())
                notes["label_vs_generator_agreement"] = round(agree, 4)
                if agree < 0.5:
                    notes["label_polarity_flipped"] = True
                    given = 1 - given
                out["y"] = machine          # generator column is unambiguous, prefer it
            else:
                out["y"] = given
        else:
            out["y"] = machine
        names = ["human", "machine"]
    else:
        present = set(out["family"])
        names = [f for f in FAMILIES if f in present] + sorted(present - set(FAMILIES))
        if label_map:
            official = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
            ordered = [n for n in official if n in present]
            if ordered:
                names = ordered + [n for n in names if n not in ordered]
                notes["label_map_used"] = True
        if HUMAN in names:  # human stays class 0: the imbalance analysis assumes it
            names = [HUMAN] + [n for n in names if n != HUMAN]
        idx = {n: i for i, n in enumerate(names)}
        out["y"] = out["family"].map(idx).astype(int)
    return out, names, notes


# ------------------------------------------------------------------ split builders
def _train_val_split(df: pd.DataFrame, n_train: int, n_val: int, stratify_by, seed: int):
    """Stratified subsample to n_train+n_val, then a stratified train/val cut on the same keys."""
    by = [("y" if c == "label" else c) for c in (stratify_by or ["y"])]
    by = [c for c in by if c in df.columns] or ["y"]
    pool = stratified_subsample(df, n_train + n_val, by=by, seed=seed)
    key = pool[by].astype(str).agg("|".join, axis=1)
    counts = key.value_counts()
    frac_val = n_val / max(n_train + n_val, 1)
    rng = np.random.default_rng(seed)
    val_parts = []
    for k, g in pool.groupby(key, sort=False):
        k_val = int(round(len(g) * frac_val))
        if counts[k] >= 2:
            k_val = min(max(k_val, 1), len(g) - 1)
        if k_val > 0:
            val_parts.append(g.sample(n=k_val, random_state=int(rng.integers(1 << 30))))
    val = pd.concat(val_parts).reset_index(drop=True) if val_parts else pool.iloc[:0].copy()
    train = pool[~pool["row_id"].isin(set(val["row_id"]))].reset_index(drop=True)
    return train, val


def pick_holdout_generators(df: pd.DataFrame, n_families: int, seed: int = 42) -> list[str]:
    """One generator from each of up to `n_families` families that have >= 2 generators.

    Holding out a whole family would make that class unlearnable; holding out one checkpoint per family
    measures generalisation to an unseen model of a known lineage.
    """
    rng = np.random.default_rng(seed)
    machine = df[~df["generator"].map(is_human)]
    if not len(machine):
        return []
    per_family = machine.groupby("family")["generator"].agg(lambda s: sorted(set(s)))
    multi = {f: gens for f, gens in per_family.items() if len(gens) >= 2}
    chosen = []
    for fam in sorted(multi, key=lambda f: -len(multi[f]))[:max(0, n_families)]:
        chosen.append(multi[fam][int(rng.integers(len(multi[fam])))])
    if not chosen:  # degenerate data (fixtures): hold out the rarest generator
        counts = machine["generator"].value_counts()
        if len(counts) >= 2:
            chosen = [counts.index[-1]]
    return chosen


def build_task_a(df: pd.DataFrame, cfg, seed: int):
    want = str(cfg.get("data.task_a.holdout_language", "Java"))
    langs = df["language"].astype(str)
    notes = {"languages_available": sorted(langs.unique())}
    if want.lower() not in {l.lower() for l in notes["languages_available"]}:
        counts = langs.value_counts()
        want = counts.index[-1]
        notes["holdout_language_fallback"] = want
    rest, ood = holdout_language_split(df, want)
    notes["holdout_language"] = want
    if len(rest) == 0:  # a single-language dataset: fall back to a random OOD slice, and say so
        rest = df.sample(frac=0.8, random_state=seed)
        ood = df.drop(rest.index)
        notes["holdout_language_degraded_to_random"] = True
    train, val = _train_val_split(rest, int(cfg.get("data.task_a.n_train", 2000)),
                                  int(cfg.get("data.task_a.n_val", 500)),
                                  cfg.get("data.task_a.stratify_by", ["label", "language"]), seed)
    ood = stratified_subsample(ood, int(cfg.get("data.task_a.n_ood", 800)), by=["y"], seed=seed)
    return {"train": train, "id_val": val, "ood_lang": ood}, notes


def build_task_b(df: pd.DataFrame, cfg, seed: int):
    notes = {}
    holdout = pick_holdout_generators(df, int(cfg.get("data.task_b.n_unseen_families", 3)), seed)
    notes["holdout_generators"] = holdout
    if holdout:
        rest, ood = holdout_generator_split(df, holdout, human_frac=0.1, seed=seed)
    else:
        rest = df.sample(frac=0.8, random_state=seed)
        ood = df.drop(rest.index)
        notes["holdout_generator_degraded_to_random"] = True
    train, val = _train_val_split(rest, int(cfg.get("data.task_b.n_train", 2500)),
                                  int(cfg.get("data.task_b.n_val", 600)), ["y", "language"], seed)
    # the human cap is applied in build(), AFTER deduplication: dedup removes machine rows
    # preferentially (generated code is more self-similar) and would otherwise undo the cap
    notes["human_cap"] = float(cfg.get("data.task_b.human_cap", 0.4))
    n_ood = int(cfg.get("data.task_b.n_ood", cfg.get("data.task_b.n_val", 600)))
    ood = stratified_subsample(ood, n_ood, by=["y"], seed=seed)
    return {"train": train, "id_val": val, "ood_gen": ood}, notes


# ------------------------------------------------------------------ leakage control + writing
def deduplicate(splits: dict, use_minhash: bool = True) -> dict:
    """Drop train rows that also appear (exactly, then near-exactly) in any eval split."""
    train = splits["train"]
    evals = [v for k, v in splits.items() if k != "train"]
    train, n_exact = dedup_exact(train, evals)
    n_near = -1  # -1 == check not run (datasketch missing)
    if use_minhash and len(train) and evals:
        try:
            ref = pd.concat(evals)["code"].astype(str).tolist()
            mask = near_duplicate_mask(train["code"].astype(str).tolist(), ref)
            n_near = int(mask.sum())
            train = train[~mask].reset_index(drop=True)
        except ImportError:
            pass
    splits["train"] = train
    return {"exact_removed": n_exact, "near_dup_removed": n_near}


def apply_human_cap(splits: dict, cap: float, seed: int) -> dict:
    """Cap the human share of TRAIN only. Eval splits keep the true (heavily human) prior on purpose:
    training needs to see the 10 LLM families, reporting needs the real-world class balance."""
    train = splits["train"]
    before = int(train["generator"].map(is_human).sum())
    splits["train"] = cap_human_share(train, max_share=cap, seed=seed)
    after = int(splits["train"]["generator"].map(is_human).sum())
    return {"human_rows_train_before_cap": before, "human_rows_train_after_cap": after,
            "human_share_train": round(float(splits["train"]["generator"].map(is_human).mean()), 4)}


def write_splits(paths, task: str, splits: dict) -> dict:
    sizes = {}
    for name, frame in splits.items():
        f = paths.split_file(task, name)
        f.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(f, index=False)
        sizes[name] = int(len(frame))
    return sizes


def load_official_validation(data_root: str, task: str, names: list[str], n: int, seed: int, notes: dict):
    """Use the organisers' validation parquet as an extra eval split when it carries usable labels."""
    files = find_validation_parquets(data_root, task)
    if not files:
        return None
    try:
        df = pd.concat([load_split(f) for f in files], ignore_index=True)
        frame, got_names, _ = prepare_frame(df, task)
        if task.upper() == "B":
            idx = {n_: i for i, n_ in enumerate(names)}
            known = frame["family"].isin(idx)
            notes["official_val_rows_dropped_unknown_family"] = int((~known).sum())
            frame = frame[known].copy()
            frame["y"] = frame["family"].map(idx).astype(int)
        if frame["y"].nunique() < 2:
            notes["official_val_skipped"] = "labels missing or single-class"
            return None
        notes["official_val_files"] = files
        return stratified_subsample(frame, n, by=["y"], seed=seed)
    except Exception as e:  # an unlabeled or differently shaped release must not break the run
        notes["official_val_skipped"] = f"{type(e).__name__}: {e}"[:200]
        return None


def build(task: str, cfg, paths, data_root: str, seed: int | None = None) -> dict:
    """Full split pipeline for one task. Returns the per-task block of results/data_report.json."""
    seed = cfg.seed if seed is None else seed
    files = find_task_parquets(data_root, task)
    if not files:
        raise FileNotFoundError(f"no parquet found under {data_root!r}")
    frames, used = [], []
    for f in files:
        d = load_split(f)
        if "code" in d.columns:
            frames.append(d)
            used.append(f)
    if not frames:
        raise FileNotFoundError(f"no parquet with a `code` column under {data_root!r}")
    df = pd.concat(frames, ignore_index=True)
    df, names, notes = prepare_frame(df, task, find_label_map(used[0]))
    splits, split_notes = (build_task_a if task.upper() == "A" else build_task_b)(df, cfg, seed)
    notes.update(split_notes)
    official = load_official_validation(data_root, task, names, int(cfg.get(f"data.task_{task.lower()}.n_val", 500)),
                                        seed, notes)
    if official is not None:
        splits["official_val"] = official
    dup = deduplicate(splits)
    if task.upper() == "B":
        notes.update(apply_human_cap(splits, float(notes.get("human_cap", 0.4)), seed))
    sizes = write_splits(paths, task, splits)
    tr = set(splits["train"]["row_id"])
    for k, v in splits.items():   # the leakage guarantee is asserted, not assumed
        if k != "train":
            assert not (tr & set(v["row_id"])), f"row overlap between train and {k}"
    return {
        "task": task.upper(), "preset": cfg.preset, "seed": seed, "source_files": used,
        "rows_total": int(len(df)), "label_names": names, "num_labels": len(names),
        "sizes": sizes, "dedup": dup, "notes": notes,
        "splits": {k: summarize_split(v, k) for k, v in splits.items()},
    }
