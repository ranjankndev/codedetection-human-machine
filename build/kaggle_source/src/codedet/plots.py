"""Figures for the report. matplotlib only (no seaborn dependency), Agg backend, one consistent style.

Every figure is generated from a prediction file, so a figure can never disagree with the tables.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PALETTE = ["#2f6f8f", "#c25e3a", "#5b8c5a", "#8b6bb1", "#b5903a", "#7a7a7a"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "axes.titlesize": 10, "axes.titleweight": "bold",
    })
    return plt


def save(fig, path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    _plt().close(fig)
    return str(path)


def confusion_heatmap(cm, labels, path, title="Confusion matrix", normalize: bool = True) -> str:
    plt = _plt()
    cm = np.asarray(cm, dtype=float)
    shown = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(4.5, 0.55 * n + 2.5), max(4.0, 0.55 * n + 2.0)))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(n), labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    ax.grid(False)
    thresh = (shown.max() or 1) * 0.6
    for i in range(n):
        for j in range(n):
            if shown[i, j] > 0.004:
                ax.text(j, i, f"{shown[i, j]:.2f}" if normalize else int(cm[i, j]),
                        ha="center", va="center", fontsize=7,
                        color="white" if shown[i, j] > thresh else "#222")
    fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85,
                 label="row-normalised rate" if normalize else "count")
    return save(fig, path)


def reliability_diagram(prob_sets: dict, y, path, title="Reliability (ID-fitted temperature)", n_bins=12) -> str:
    """prob_sets: label -> probs [N,K]. Plots confidence vs accuracy; the diagonal is perfect calibration."""
    plt = _plt()
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="perfect")
    bins = np.linspace(0, 1, n_bins + 1)
    for i, (name, probs) in enumerate(prob_sets.items()):
        conf, pred = probs.max(1), probs.argmax(1)
        acc = (pred == y).astype(float)
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 5:
                xs.append(conf[m].mean())
                ys.append(acc[m].mean())
        ax.plot(xs, ys, "o-", color=PALETTE[i % len(PALETTE)], lw=1.4, ms=4, label=name)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False)
    return save(fig, path)


def pr_curve(curves: dict, path, title="Precision-recall, P(machine)", operating_points=None) -> str:
    """curves: label -> (recall, precision). operating_points: label -> (recall, precision, text)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    for i, (name, (rec, prec)) in enumerate(curves.items()):
        ax.plot(rec, prec, color=PALETTE[i % len(PALETTE)], lw=1.5, label=name)
    for i, (name, (r, p, text)) in enumerate((operating_points or {}).items()):
        ax.plot([r], [p], "*", ms=11, color=PALETTE[i % len(PALETTE)])
        ax.annotate(text, (r, p), textcoords="offset points", xytext=(6, -10), fontsize=7)
    ax.set_xlabel("recall (machine)")
    ax.set_ylabel("precision (machine)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    return save(fig, path)


def grouped_bars(data: dict, categories, path, title="", ylabel="macro-F1", ylim=(0, 1)) -> str:
    """data: series label -> list of values aligned with `categories`."""
    plt = _plt()
    cats = list(categories)
    fig, ax = plt.subplots(figsize=(max(4.5, 1.1 * len(cats) + 1.5), 3.4))
    n = max(len(data), 1)
    width = 0.8 / n
    x = np.arange(len(cats))
    for i, (name, vals) in enumerate(data.items()):
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=name, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x, cats, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    if len(data) > 1:
        ax.legend(fontsize=7, frameon=False, ncol=min(len(data), 3))
    return save(fig, path)


def horizontal_bars(labels, values, path, title="", xlabel="") -> str:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, max(2.4, 0.28 * len(labels) + 1.0)))
    order = np.argsort(values)
    ax.barh([labels[i] for i in order], [values[i] for i in order], color=PALETTE[0])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return save(fig, path)


def forest_plot(rows: list, path, title="macro-F1 with 95% bootstrap CI", xlabel="macro-F1") -> str:
    """rows: list of (label, point, lo, hi) - the CI picture that replaces 'model X is best'."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.4, max(2.4, 0.34 * len(rows) + 1.0)))
    ys = np.arange(len(rows))
    for i, (_, point, lo, hi) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color="#999", lw=1.5)
        ax.plot([point], [i], "o", color=PALETTE[0], ms=5)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return save(fig, path)
