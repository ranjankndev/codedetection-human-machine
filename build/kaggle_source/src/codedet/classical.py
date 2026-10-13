"""Classical baselines. They exist for three reasons, not as filler:

  * `majority` and `length_only` are SHORTCUT baselines. If a 2-feature logistic regression on code
    length and whitespace is close to a fine-tuned transformer, the "detector" is reading formatting,
    not authorship. (Audit finding P7: 0.99 in-distribution -> 0.57 on an unseen language.)
  * `tfidf_lr` (char 3-5 grams) is the strong cheap baseline every honest report needs.
  * `style_gbdt` / `lm_gbdt` are interpretable: permutation importance over named features tells you
    WHICH artifact the model uses, which a transformer will not.

All of them expose the same (fit, predict_proba) shape so `run_classical.py` stays a thin loop.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .features import STYLE_NAMES, StyleScaler, style_matrix

LENGTH_NAMES = ["n_chars", "n_lines", "mean_line_len", "space_ratio", "tab_ratio", "blank_line_ratio"]


def length_matrix(codes) -> np.ndarray:
    """Deliberately content-free features: size and whitespace only."""
    rows = []
    for c in codes:
        s = str(c)
        lines = s.split("\n")
        n_lines = max(len(lines), 1)
        n_chars = max(len(s), 1)
        rows.append([
            np.log1p(len(s)),
            np.log1p(n_lines),
            float(np.mean([len(l) for l in lines])) if lines else 0.0,
            s.count(" ") / n_chars,
            s.count("\t") / n_chars,
            sum(1 for l in lines if not l.strip()) / n_lines,
        ])
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, len(LENGTH_NAMES)), np.float32)


def _gbdt(num_labels: int, seed: int = 42, class_weight=None):
    """LightGBM if available (faster, better), else sklearn's HistGradientBoosting. Both are fine; log which."""
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(n_estimators=300, learning_rate=0.08, num_leaves=31, random_state=seed,
                              class_weight=class_weight, verbose=-1), "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed,
                                              class_weight=class_weight), "hist_gbdt"


@dataclass
class ClassicalResult:
    name: str
    backend: str
    n_features: int
    feature_names: list[str]
    extra: dict


class MajorityClassifier:
    """The floor. Task B's accuracy 0.64 / macro-F1 0.29 was this model wearing a transformer costume."""

    def __init__(self, num_labels: int):
        self.num_labels = num_labels
        self.prior = np.full(num_labels, 1.0 / num_labels)

    def fit(self, X, y):
        counts = np.bincount(np.asarray(y, int), minlength=self.num_labels).astype(float)
        self.prior = counts / max(counts.sum(), 1.0)
        return self

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.tile(self.prior, (n, 1))


class ClassicalModel:
    """A (featuriser, estimator) pair with a frozen train-time scaler where one is needed."""

    def __init__(self, name: str, num_labels: int, seed: int = 42, balanced: bool = False):
        self.name, self.num_labels, self.seed = name, num_labels, seed
        self.balanced = balanced
        self.backend = "sklearn"
        self.feature_names: list[str] = []
        self.scaler: StyleScaler | None = None
        self.vectorizer = None
        self.est = None
        self.extra: dict = {}

    # --- featurisation -------------------------------------------------------------
    def _features(self, codes, lm: np.ndarray | None, fit: bool):
        if self.name == "majority":
            return np.zeros((len(codes), 1), np.float32)
        if self.name == "length_only":
            X = length_matrix(codes)
            self.feature_names = LENGTH_NAMES
        elif self.name == "tfidf_lr":
            from sklearn.feature_extraction.text import TfidfVectorizer

            texts = [str(c) for c in codes]
            if fit:
                self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                                  max_features=200_000, sublinear_tf=True)
                X = self.vectorizer.fit_transform(texts)
                self.feature_names = ["char_ngram"]
            else:
                X = self.vectorizer.transform(texts)
            return X
        elif self.name in ("style_gbdt", "lm_gbdt"):
            X = style_matrix(list(codes))
            self.feature_names = list(STYLE_NAMES)
            if self.name == "lm_gbdt":
                if lm is None:
                    raise ValueError("lm_gbdt needs LM features; run extract_lm_features.py first")
                from .lm_features import LM_FEATURE_NAMES

                X = np.hstack([X, np.asarray(lm, np.float32)])
                self.feature_names = list(STYLE_NAMES) + list(LM_FEATURE_NAMES)
        else:
            raise ValueError(f"unknown classical model {self.name!r}")
        # dense feature blocks are z-scored with TRAIN statistics only
        if fit:
            self.scaler = StyleScaler().fit(X)
        return self.scaler.transform(X)

    # --- sklearn-ish surface -------------------------------------------------------
    def fit(self, codes, y, lm: np.ndarray | None = None):
        X = self._features(codes, lm, fit=True)
        y = np.asarray(y, int)
        if self.name == "majority" or len(set(y.tolist())) < 2:
            # a single-class training split cannot fit a discriminative model; predicting the prior is
            # the honest fallback (and keeps the pipeline running on degenerate subsamples)
            self.est = MajorityClassifier(self.num_labels).fit(X, y)
            self.backend = "prior" if self.name == "majority" else "prior_fallback_single_class"
        elif self.name in ("length_only", "tfidf_lr"):
            from sklearn.linear_model import LogisticRegression

            self.est = LogisticRegression(max_iter=2000, C=1.0, n_jobs=None,
                                          class_weight="balanced" if self.balanced else None)
            self.est.fit(X, y)
            self.backend = "logreg"
        else:
            self.est, self.backend = _gbdt(self.num_labels, self.seed,
                                           "balanced" if self.balanced else None)
            self.est.fit(X, y)
        self.extra["n_features"] = int(X.shape[1]) if hasattr(X, "shape") else 0
        self.extra["classes_seen"] = sorted(set(int(v) for v in y))
        return self

    def predict_proba(self, codes, lm: np.ndarray | None = None) -> np.ndarray:
        X = self._features(codes, lm, fit=False)
        p = self.est.predict_proba(X)
        return self._expand(p)

    def _expand(self, p: np.ndarray) -> np.ndarray:
        """sklearn drops classes absent from train; macro-F1 over range(K) needs full-width columns."""
        if p.shape[1] == self.num_labels:
            return p
        full = np.zeros((p.shape[0], self.num_labels), dtype=np.float64)
        classes = getattr(self.est, "classes_", np.arange(p.shape[1]))
        for j, c in enumerate(classes):
            full[:, int(c)] = p[:, j]
        return full

    def result(self) -> ClassicalResult:
        return ClassicalResult(self.name, self.backend, int(self.extra.get("n_features", 0)),
                               self.feature_names, dict(self.extra))


def top_ngrams(model: ClassicalModel, k: int = 15) -> dict:
    """Most human-indicative / machine-indicative char n-grams from the TF-IDF model (binary task only)."""
    if model.name != "tfidf_lr" or model.vectorizer is None or not hasattr(model.est, "coef_"):
        return {}
    coef = model.est.coef_
    if coef.shape[0] != 1:
        return {}
    names = np.asarray(model.vectorizer.get_feature_names_out())
    w = coef[0]
    order = np.argsort(w)
    clean = lambda arr: [{"ngram": re.sub(r"\s+", " ", names[i]), "weight": float(w[i])} for i in arr]
    return {"machine_indicative": clean(order[::-1][:k]), "human_indicative": clean(order[:k])}


def permutation_importance_gbdt(model: ClassicalModel, codes, y, lm=None, n_repeats: int = 5, seed: int = 0) -> list:
    """Permutation importance in macro-F1 points over the NAMED dense features (style / style+LM)."""
    from .eval import macro_f1_fast

    if model.name not in ("style_gbdt", "lm_gbdt", "length_only"):
        return []
    X = model._features(codes, lm, fit=False)
    y = np.asarray(y, int)
    base = macro_f1_fast(y, model._expand(model.est.predict_proba(X)).argmax(1), model.num_labels)
    rng = np.random.default_rng(seed)
    out = []
    for j, name in enumerate(model.feature_names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            f1 = macro_f1_fast(y, model._expand(model.est.predict_proba(Xp)).argmax(1), model.num_labels)
            drops.append(base - f1)
        out.append({"feature": name, "macro_f1_drop": float(np.mean(drops)), "std": float(np.std(drops))})
    return sorted(out, key=lambda d: -d["macro_f1_drop"])


def fit_stack(logit_blocks: dict, extra_blocks: dict, y, num_labels: int, seed: int = 42, balanced=False):
    """Late fusion: GBDT on [transformer logits | style | LM features] (one row per example).

    `logit_blocks` / `extra_blocks` map split -> array; the TRAIN split is used for fitting.
    """
    est, backend = _gbdt(num_labels, seed, "balanced" if balanced else None)
    X = np.hstack([logit_blocks["train"], extra_blocks["train"]])
    est.fit(X, np.asarray(y, int))
    return est, backend
