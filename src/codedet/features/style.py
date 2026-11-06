"""Stylometric features (torch-free, language-agnostic, cached by the caller).

Replaces the old per-batch `StylometricExtractor.extract_features` (computed inside forward(), unscaled,
Python-biased). The old `radon` 20-dim features are kept below as `radon_features` but are PYTHON-ONLY:
for C++/Java they silently become all zeros, which turns them into a language detector (shortcut!).
"""
from __future__ import annotations

import re

import numpy as np

STYLE_NAMES = [
    "log_lines", "mean_line_len", "max_line_len", "blank_ratio", "mean_indent", "tab_ratio",
    "trailing_ws_ratio", "comment_line_ratio", "mean_ident_len", "unique_ident_ratio",
    "snake_ratio", "camel_ratio", "digit_ratio", "punct_density", "long_line_share", "upper_ratio",
]
STYLE_DIM = len(STYLE_NAMES)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"^[a-z]+[A-Z]")
_COMMENT_PREFIX = ("#", "//", "/*", "*", "--")


def stylometric_features(code: str) -> np.ndarray:
    code = str(code)
    lines = code.split("\n")
    n_lines = max(len(lines), 1)
    n_chars = max(len(code), 1)
    lens = np.array([len(l) for l in lines], dtype=np.float32)
    stripped = [l.strip() for l in lines]
    idents = _IDENT.findall(code)
    n_id = max(len(idents), 1)
    indents = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip()]
    punct = sum(code.count(c) for c in "{}()[];,.:<>=+-*/&|!%^")
    f = [
        np.log1p(n_lines),
        float(lens.mean()) if len(lens) else 0.0,
        float(min(lens.max(), 400)) if len(lens) else 0.0,
        sum(1 for s in stripped if not s) / n_lines,
        float(np.mean(indents)) if indents else 0.0,
        code.count("\t") / n_chars,
        sum(1 for l in lines if l != l.rstrip()) / n_lines,
        sum(1 for s in stripped if s.startswith(_COMMENT_PREFIX)) / n_lines,
        sum(len(i) for i in idents) / n_id,
        len(set(idents)) / n_id,
        sum(1 for i in idents if "_" in i[1:-1]) / n_id,
        sum(1 for i in idents if _CAMEL.match(i)) / n_id,
        sum(c.isdigit() for c in code) / n_chars,
        punct / n_chars,
        float((lens > 80).mean()) if len(lens) else 0.0,
        sum(c.isupper() for c in code) / n_chars,
    ]
    return np.nan_to_num(np.asarray(f, dtype=np.float32))


def style_matrix(codes) -> np.ndarray:
    return np.stack([stylometric_features(c) for c in codes]) if len(codes) else np.zeros((0, STYLE_DIM), np.float32)


class StyleScaler:
    """z-score with TRAIN statistics only; save/load as a plain dict next to the checkpoint."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> "StyleScaler":
        self.mean = X.mean(0)
        self.std = np.maximum(X.std(0), 1e-6)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "StyleScaler":
        s = cls()
        s.mean, s.std = np.asarray(d["mean"], np.float32), np.asarray(d["std"], np.float32)
        return s


def radon_features(code: str, dim: int = 20) -> np.ndarray:
    """LEGACY (python-only). Ported from legacy/taskb/codebert-entropy-stylometry-classifier.py.
    raw(loc,lloc,comments,multi,blank) + cyclomatic(mean,max,count) + halstead(5). Zeros if not parseable."""
    import radon.complexity as rcc
    import radon.metrics as rmetrics
    import radon.raw as rraw

    feats: list[float] = []
    try:
        r = rraw.analyze(code)
        feats += [r.loc, r.lloc, r.comments, r.multi, r.blank]
    except Exception:
        feats += [0.0] * 5
    try:
        cc = [c.complexity for c in rcc.cc_visit(code)]
        feats += [np.mean(cc), np.max(cc), len(cc)] if cc else [0.0] * 3
    except Exception:
        feats += [0.0] * 3
    try:
        h = rmetrics.h_visit(code).total
        feats += [h.operators, h.operands, h.distinct_operators, h.distinct_operands, h.volume]
    except Exception:
        feats += [0.0] * 5
    feats += [0.0] * (dim - len(feats))
    return np.asarray(feats[:dim], dtype=np.float32)
