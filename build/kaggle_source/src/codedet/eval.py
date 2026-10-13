"""Evaluation: metrics, bootstrap CIs, paired tests, calibration.
Ported from legacy evaluation_metrics.py / inference.py (macro P/R/F1, confusion matrix) and extended."""
from __future__ import annotations

import numpy as np


def confusion(y, p, k):
    y, p = np.asarray(y), np.asarray(p)
    return np.bincount(y * k + p, minlength=k * k).reshape(k, k)


def macro_f1_fast(y, p, k) -> float:
    """Macro-F1 over ALL k classes (a class absent from y and p scores 0, like sklearn with labels=range(k))."""
    cm = confusion(y, p, k)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return float(f1.mean())


def metrics_dict(y, p, k, label_names=None) -> dict:
    from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

    y, p = np.asarray(y), np.asarray(p)
    labels = list(range(k))
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    names = label_names or [str(i) for i in labels]
    return {
        "n": int(len(y)),
        "macro_f1": float(f1.mean()), "macro_precision": float(pr.mean()), "macro_recall": float(rc.mean()),
        "accuracy": float((y == p).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "per_class": {names[i]: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])}
                      for i in labels},
        "confusion_matrix": confusion(y, p, k).tolist(),
    }


def bootstrap_ci(y, p, k, n_boot=1000, alpha=0.05, seed=0, fn=macro_f1_fast):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = fn(y[idx], p[idx], k)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"point": float(fn(y, p, k)), "lo": float(lo), "hi": float(hi)}


def paired_bootstrap_diff(y, p_a, p_b, k, n_boot=1000, alpha=0.05, seed=0, fn=macro_f1_fast):
    """Macro-F1(A) - Macro-F1(B) on identical rows. `excludes_zero` drives the README branch decision (PLAN A.5)."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        d[b] = fn(y[idx], p_a[idx], k) - fn(y[idx], p_b[idx], k)
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(fn(y, p_a, k) - fn(y, p_b, k)), "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def mcnemar(y, p_a, p_b) -> dict:
    """Exact McNemar test on paired correctness."""
    from scipy.stats import binomtest

    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    a_ok, b_ok = p_a == y, p_b == y
    n01, n10 = int((~a_ok & b_ok).sum()), int((a_ok & ~b_ok).sum())
    n = n01 + n10
    pval = 1.0 if n == 0 else float(binomtest(n10, n, 0.5).pvalue)
    return {"a_only_correct": n10, "b_only_correct": n01, "p_value": pval}


def per_slice(values, y, p, k, min_n: int = 30) -> dict:
    """Macro-F1 per slice value (e.g. per language / per generator / seen vs unseen)."""
    values, y, p = np.asarray(values), np.asarray(y), np.asarray(p)
    out = {}
    for v in np.unique(values):
        m = values == v
        if m.sum() >= min_n:
            out[str(v)] = {"n": int(m.sum()), "macro_f1": macro_f1_fast(y[m], p[m], k), "accuracy": float((y[m] == p[m]).mean())}
    return out


# ---------------------------------------------------------------- calibration
def softmax(z, T: float = 1.0):
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs, y, n_bins: int = 15) -> float:
    probs, y = np.asarray(probs), np.asarray(y)
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(total)


def fit_temperature(logits, y) -> float:
    """Temperature scaling: fit on ID validation logits, then evaluate ECE on OOD to show calibration under shift."""
    from scipy.optimize import minimize_scalar

    logits, y = np.asarray(logits, dtype=np.float64), np.asarray(y)

    def nll(logT):
        p = softmax(logits, np.exp(logT))
        return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()

    r = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(r.x))
