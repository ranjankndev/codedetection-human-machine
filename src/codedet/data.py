"""Data utilities (torch-free). Ported/extended from legacy/taskb (load_split, sample_* helpers)."""
from __future__ import annotations

import hashlib
import glob
import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .labels import is_human


# ---------------------------------------------------------------- loading / discovery
def load_split(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    return df.reset_index(drop=True)


def discover_parquets(root: str) -> dict:
    """Find parquet files under `root` (e.g. /kaggle/input) and describe them. Never hard-code folders."""
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)):
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(p)
            out[p] = {"rows": pf.metadata.num_rows, "columns": pf.schema_arrow.names}
        except Exception as e:  # keep going; report the error
            out[p] = {"error": str(e)}
    return out


def summarize_split(df: pd.DataFrame, name: str = "split") -> dict:
    s = {"name": name, "rows": int(len(df))}
    for col in ("label", "language", "generator", "family", "domain"):
        if col in df.columns:
            s[f"{col}_counts"] = df[col].value_counts().head(30).to_dict()
    return s


# ---------------------------------------------------------------- sampling
def stratified_subsample(df: pd.DataFrame, n: int, by: Sequence[str], seed: int = 42) -> pd.DataFrame:
    """Approximately proportional sample of n rows keeping every stratum (>=1 row each)."""
    if n >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    frac = n / len(df)
    parts = []
    for _, g in df.groupby(list(by), sort=False):
        k = min(len(g), max(1, int(round(len(g) * frac))))
        parts.append(g.sample(n=k, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def sample_balanced(df: pd.DataFrame, n_human: int, n_machine: int, seed: int = 42, gen_col="generator"):
    """Old helper: fixed human / machine counts."""
    h_mask = df[gen_col].map(is_human)
    h, m = df[h_mask], df[~h_mask]
    h = h.sample(n=min(n_human, len(h)), random_state=seed) if len(h) else h
    m = m.sample(n=min(n_machine, len(m)), random_state=seed) if len(m) else m
    return pd.concat([h, m]).sample(frac=1, random_state=seed).reset_index(drop=True)


def sample_per_label(df: pd.DataFrame, label_col="family", n_per_label=100, seed=42) -> pd.DataFrame:
    parts = [g.sample(n=min(n_per_label, len(g)), random_state=seed) for _, g in df.groupby(label_col) if len(g)]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def cap_human_share(df: pd.DataFrame, max_share: float = 0.4, seed: int = 42, gen_col="generator"):
    """Task B: humans are ~88% of train. Cap them so every LLM family is visible. Keep the TRUE prior in eval sets."""
    h_mask = df[gen_col].map(is_human)
    h, m = df[h_mask], df[~h_mask]
    n_h = int(max_share / (1 - max_share) * len(m))
    h = h.sample(n=min(n_h, len(h)), random_state=seed)
    return pd.concat([h, m]).sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------- leakage control
def _hash_code(s: str) -> str:
    return hashlib.md5(" ".join(str(s).split()).encode("utf-8", "ignore")).hexdigest()


def dedup_exact(train: pd.DataFrame, others: Iterable[pd.DataFrame], col: str = "code"):
    """Drop duplicates inside `train` and any train row whose (whitespace-normalised) code appears in `others`.
    Returns (clean_train, n_removed)."""
    ref = set()
    for o in others:
        ref.update(o[col].map(_hash_code))
    h = train[col].map(_hash_code)
    keep = ~h.isin(ref) & ~h.duplicated()
    return train[keep].reset_index(drop=True), int((~keep).sum())


def near_duplicate_mask(codes: Sequence[str], ref_codes: Sequence[str], threshold=0.9, num_perm=64, shingle=5):
    """MinHash near-dup check (needs `datasketch`). True where a row is near-identical to something in ref."""
    from datasketch import MinHash, MinHashLSH

    def mh(text):
        m = MinHash(num_perm=num_perm)
        toks = text.split()
        for i in range(max(1, len(toks) - shingle + 1)):
            m.update(" ".join(toks[i:i + shingle]).encode("utf8", "ignore"))
        return m

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, c in enumerate(ref_codes):
        lsh.insert(str(i), mh(c))
    return np.array([len(lsh.query(mh(c))) > 0 for c in codes])


# ---------------------------------------------------------------- OOD split builders
def holdout_language_split(df: pd.DataFrame, holdout: str, lang_col: str = "language"):
    """Proxy 'unseen language': returns (train_without_lang, ood_lang_rows)."""
    mask = df[lang_col].astype(str).str.lower() == holdout.lower()
    if mask.sum() == 0:
        raise ValueError(f"language {holdout!r} not found; have {sorted(df[lang_col].astype(str).unique())[:20]}")
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def holdout_generator_split(df: pd.DataFrame, holdout_generators: Sequence[str], gen_col="generator",
                            human_frac: float = 0.1, seed: int = 42):
    """Proxy 'unseen generator' (Task B). Returns (train, unseen_eval).
    unseen_eval = all rows of the held-out generators + `human_frac` of human rows (removed from train)."""
    hold = {g.lower() for g in holdout_generators}
    is_hold = df[gen_col].astype(str).str.lower().isin(hold)
    if is_hold.sum() == 0:
        raise ValueError(f"none of {holdout_generators} found in column {gen_col!r}")
    humans = df[df[gen_col].map(is_human)]
    h_eval = humans.sample(frac=human_frac, random_state=seed)
    unseen = pd.concat([df[is_hold], h_eval]).reset_index(drop=True)
    train = df[~is_hold & ~df.index.isin(h_eval.index)].reset_index(drop=True)
    return train, unseen


# ---------------------------------------------------------------- tokenisation helper
def add_special_tokens(tokenizer, ids: list[int]) -> list[int]:
    """Wrap ids in the tokenizer's special tokens.

    `build_inputs_with_special_tokens` existed on every tokenizer up to transformers 4.x and was dropped
    from the fast tokenizers in 5.x, so fall back to the cls/sep (or bos/eos) ids.
    """
    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        try:
            return list(tokenizer.build_inputs_with_special_tokens(list(ids)))
        except (AttributeError, NotImplementedError):
            pass
    cls = getattr(tokenizer, "cls_token_id", None)
    sep = getattr(tokenizer, "sep_token_id", None)
    cls = getattr(tokenizer, "bos_token_id", None) if cls is None else cls
    sep = getattr(tokenizer, "eos_token_id", None) if sep is None else sep
    out = list(ids)
    if cls is not None:
        out = [cls] + out
    if sep is not None:
        out = out + [sep]
    return out


def head_tail_ids(tokenizer, text: str, max_len: int = 256, head: int | None = None, max_chars: int = 20000):
    """Token ids with HEAD+TAIL truncation (imports at the top and the ending carry style signal).
    Returns ids WITH special tokens, length <= max_len."""
    if len(text) > max_chars:
        text = text[: max_chars // 2] + "\n" + text[-max_chars // 2:]
    ids = tokenizer.encode(text, add_special_tokens=False)
    n_special = len(add_special_tokens(tokenizer, []))
    budget = max_len - n_special
    if len(ids) > budget:
        head = int(budget * 0.75) if head is None else min(head, budget)
        tail = budget - head
        ids = ids[:head] + (ids[-tail:] if tail > 0 else [])
    return add_special_tokens(tokenizer, ids)
