"""The analysis that turns a results table into evidence of ML judgment.

    python scripts/analyze.py --config local_cpu --out results

Five things, all generated from prediction files:
  1. shortcut stress test - the SAME OOD rows with comments stripped / whitespace normalised; a model
     that collapses was reading surface form (audit finding P7);
  2. shortcut ladder - how close the length/whitespace-only baseline gets to the transformer;
  3. imbalance - task B per-class table + confusion matrix, macro-F1 vs accuracy side by side;
  4. calibration - reliability diagram and ECE before/after ID-fitted temperature scaling, on ID and OOD;
  5. operating point - PR curve and the threshold that minimises an explicitly stated cost ratio
     (a false "machine" accusation of a student is costlier than a miss), on ID and on OOD.
Plus `error_analysis.md`: the 10 most confident mistakes, with a one-line cause each.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from codedet.artifacts import load_preds
from codedet.config import SPLITS, Config, RunPaths
from codedet.eval import ece, fit_temperature, macro_f1_fast, softmax
from codedet.plots import confusion_heatmap, forest_plot, grouped_bars, horizontal_bars, pr_curve, reliability_diagram
from codedet.utils import get_logger, read_json, write_json

log = get_logger("analyze")

FP_COST = 5.0   # cost of wrongly calling human code "machine" (a false accusation)
FN_COST = 1.0   # cost of missing machine code


def ood_split(task: str) -> str:
    return "ood_lang" if task.upper() == "A" else "ood_gen"


def preds_for(paths: RunPaths, task: str, model: str) -> dict:
    d = paths.preds / f"task_{task.lower()}" / model
    return {f.stem.replace("probs_", ""): load_preds(f) for f in sorted(d.glob("probs_*.npz"))}


def best_neural(summary_task: dict, prefer=("gpad", "gpad_both", "codebert", "codebert_weighted", "control")) -> str | None:
    for m in prefer:
        if m in summary_task.get("models", {}):
            return m
    return next(iter(summary_task.get("models", {})), None)


# ---------------------------------------------------------------- 1 + 2 shortcut analysis
def shortcut_analysis(paths: RunPaths, task: str, summary_task: dict) -> dict:
    """Perturbation drop per model + the gap between the content-free baseline and the transformer."""
    ood = ood_split(task)
    out: dict = {"ood_split": ood, "perturbations": {}, "shortcut_ladder": {}}
    for model, block in summary_task["models"].items():
        splits = block["splits"]
        if ood not in splits:
            continue
        base = splits[ood]["macro_f1"]
        row = {"ood_macro_f1": base}
        for name, s in splits.items():
            if name.startswith(f"{ood}__"):
                pert = name.split("__", 1)[1]
                row[pert] = s["macro_f1"]
                row[f"{pert}_drop"] = round(base - s["macro_f1"], 4)
        if len(row) > 1:
            out["perturbations"][model] = row
    for model, block in summary_task["models"].items():
        s = block["splits"]
        out["shortcut_ladder"][model] = {k: s[k]["macro_f1"] for k in ("id_val", ood) if k in s}
    floor, ceiling = out["shortcut_ladder"].get("length_only"), None
    for cand in ("gpad", "gpad_both", "codebert", "codebert_weighted"):
        if cand in out["shortcut_ladder"]:
            ceiling = out["shortcut_ladder"][cand]
            out["ceiling_model"] = cand
            break
    if floor and ceiling:
        out["length_only_vs_transformer"] = {
            k: round(ceiling.get(k, float("nan")) - floor.get(k, float("nan")), 4)
            for k in ("id_val", ood) if k in floor and k in ceiling}
    return out


# ---------------------------------------------------------------- 3 imbalance
def imbalance_analysis(paths: RunPaths, task: str, summary_task: dict, figures: Path) -> dict:
    model = best_neural(summary_task)
    if model is None:
        return {}
    names = summary_task["label_names"]
    out = {"model": model, "splits": {}, "figures": {}}
    for split in ("id_val", ood_split(task)):
        block = summary_task["models"][model]["splits"].get(split)
        if not block:
            continue
        out["splits"][split] = {
            "macro_f1": block["macro_f1"], "accuracy": block["accuracy"],
            "balanced_accuracy": block["balanced_accuracy"],
            "accuracy_minus_macro_f1": round(block["accuracy"] - block["macro_f1"], 4),
            "per_class": block["per_class"],
        }
        out["figures"][split] = confusion_heatmap(
            block["confusion_matrix"], names, figures / f"confusion_task{task.lower()}_{split}.png",
            title=f"Task {task.upper()} {model} - {split} (row-normalised)")
    # the "majority-class illusion": accuracy high, macro-F1 low
    maj = summary_task["models"].get("majority", {}).get("splits", {}).get("id_val")
    if maj:
        out["majority_baseline_id_val"] = {"macro_f1": maj["macro_f1"], "accuracy": maj["accuracy"]}
    return out


# ---------------------------------------------------------------- 4 calibration
def calibration_analysis(paths: RunPaths, task: str, summary_task: dict, figures: Path) -> dict:
    model = best_neural(summary_task)
    if model is None:
        return {}
    preds = preds_for(paths, task, model)
    if "id_val" not in preds:
        return {}
    T = fit_temperature(preds["id_val"]["logits"], preds["id_val"]["labels"])
    out = {"model": model, "temperature_fitted_on_id_val": T, "ece": {}, "figures": {}}
    for split in ("id_val", ood_split(task)):
        if split not in preds:
            continue
        d = preds[split]
        before, after = softmax(d["logits"]), softmax(d["logits"], T)
        out["ece"][split] = {"before": ece(before, d["labels"]), "after_temperature": ece(after, d["labels"])}
        out["figures"][split] = reliability_diagram(
            {"raw": before, f"T={T:.2f}": after}, d["labels"],
            figures / f"reliability_task{task.lower()}_{split}.png",
            title=f"Task {task.upper()} {model} - {split}")
    if len(out["ece"]) == 2:
        id_e = out["ece"]["id_val"]["after_temperature"]
        ood_e = out["ece"][ood_split(task)]["after_temperature"]
        out["calibration_breaks_under_shift"] = bool(ood_e > id_e * 1.5)
    return out


# ---------------------------------------------------------------- 5 operating point (task A)
def threshold_analysis(paths: RunPaths, task: str, summary_task: dict, figures: Path,
                       fp_cost=FP_COST, fn_cost=FN_COST) -> dict:
    if summary_task["num_labels"] != 2:
        return {}
    model = best_neural(summary_task)
    preds = preds_for(paths, task, model)
    if "id_val" not in preds:
        return {}
    from sklearn.metrics import precision_recall_curve

    T = fit_temperature(preds["id_val"]["logits"], preds["id_val"]["labels"])
    curves, points, out = {}, {}, {"model": model, "cost_ratio_fp_to_fn": fp_cost / fn_cost,
                                   "temperature": T, "splits": {}}
    # threshold chosen on ID-val only, then applied unchanged to OOD (that is the honest protocol)
    d = preds["id_val"]
    p_machine = softmax(d["logits"], T)[:, 1]
    grid = np.unique(np.round(np.concatenate([[0.01, 0.99], np.quantile(p_machine, np.linspace(0, 1, 199))]), 4))
    y = d["labels"]
    costs = [(float(t), float(fp_cost * ((p_machine >= t) & (y == 0)).sum() + fn_cost * ((p_machine < t) & (y == 1)).sum()))
             for t in grid]
    thr = min(costs, key=lambda kv: kv[1])[0]
    out["threshold"] = thr
    out["cost_curve_sample"] = costs[:: max(1, len(costs) // 25)]
    for split in ("id_val", ood_split(task)):
        if split not in preds:
            continue
        d = preds[split]
        pm = softmax(d["logits"], T)[:, 1]
        y = d["labels"]
        for name, t in (("argmax_0.5", 0.5), ("cost_optimal", thr)):
            pred = (pm >= t).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            out["splits"].setdefault(split, {})[name] = {
                "threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision_machine": tp / max(tp + fp, 1), "recall_machine": tp / max(tp + fn, 1),
                "false_accusation_rate": fp / max(fp + tn, 1),
                "expected_cost_per_1000": round(1000 * (fp_cost * fp + fn_cost * fn) / max(len(y), 1), 2),
            }
        prec, rec, _ = precision_recall_curve(y, pm)
        curves[split] = (rec, prec)
        b = out["splits"][split]["cost_optimal"]
        points[split] = (b["recall_machine"], b["precision_machine"], f"t={thr:.2f}")
    out["figure"] = pr_curve(curves, figures / f"pr_curve_task{task.lower()}.png",
                             title=f"Task {task.upper()} {model}: PR curve (cost {fp_cost:.0f}:1 FP:FN)",
                             operating_points=points)
    return out


# ---------------------------------------------------------------- error analysis
def error_analysis(paths: RunPaths, task: str, summary_task: dict, k: int = 10) -> str:
    model = best_neural(summary_task)
    split = ood_split(task)
    preds = preds_for(paths, task, model)
    if split not in preds:
        split = "id_val"
    if split not in preds:
        return ""
    d = preds[split]
    frame_file = paths.split_file(task, split)
    frame = pd.read_parquet(frame_file).set_index("row_id") if frame_file.exists() else None
    probs = softmax(d["logits"])
    pred = probs.argmax(1)
    conf = probs.max(1)
    wrong = np.where(pred != d["labels"])[0]
    order = wrong[np.argsort(-conf[wrong])][:k]
    names = summary_task["label_names"]
    lines = [f"# Error analysis - task {task.upper()} ({model}, split `{split}`)", "",
             f"The {len(order)} most *confident* mistakes (confidence after softmax, no temperature).",
             "Confident errors are the ones that would actually harm a user, so they are the ones worth reading.", ""]
    for rank, i in enumerate(order, 1):
        rid = d["ids"][i]
        row = frame.loc[rid] if frame is not None and rid in frame.index else None
        code = str(row["code"]) if row is not None else ""
        excerpt = "\n".join(code.strip().split("\n")[:12])[:700]
        meta = (f"language={row.get('language', '?')}, generator={row.get('generator', '?')}"
                if row is not None else "metadata unavailable")
        lines += [f"## {rank}. true `{names[d['labels'][i]]}` -> predicted `{names[pred[i]]}` "
                  f"(confidence {conf[i]:.2f})", f"- {meta}", f"- row_id `{rid}`", "", "```", excerpt, "```", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- feature importance (interpretable models)
def feature_importance(paths: RunPaths, task: str) -> dict:
    out = {}
    for model in ("style_gbdt", "lm_gbdt", "length_only"):
        f = paths.metrics_file(task, model)
        if f.exists():
            m = read_json(f)
            if m.get("permutation_importance_id_val"):
                out[model] = m["permutation_importance_id_val"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="smoke")
    ap.add_argument("--out", default="results")
    ap.add_argument("--fp-cost", type=float, default=FP_COST)
    ap.add_argument("--fn-cost", type=float, default=FN_COST)
    a = ap.parse_args()

    cfg = Config.load(a.config)
    paths = RunPaths.make(a.out)
    summary = read_json(Path(a.out) / "summary.json")
    figures = paths.figures
    analysis: dict = {"preset": cfg.preset, "cost_ratio_fp_to_fn": a.fp_cost / a.fn_cost, "tasks": {}}

    for task, st in summary["tasks"].items():
        block = {
            "shortcut": shortcut_analysis(paths, task, st),
            "imbalance": imbalance_analysis(paths, task, st, figures),
            "calibration": calibration_analysis(paths, task, st, figures),
            "threshold": threshold_analysis(paths, task, st, figures, a.fp_cost, a.fn_cost),
            "feature_importance": feature_importance(paths, task),
        }
        # CI forest plot over models on the OOD split: the one picture that shows what is a tie
        ood = ood_split(task)
        rows = [(m, b["splits"][ood]["macro_f1"], b["splits"][ood]["macro_f1_ci"]["lo"],
                 b["splits"][ood]["macro_f1_ci"]["hi"])
                for m, b in st["models"].items() if ood in b["splits"]]
        if rows:
            block["figures"] = {"forest_ood": forest_plot(
                sorted(rows, key=lambda r: -r[1]), figures / f"forest_task{task.lower()}_{ood}.png",
                title=f"Task {task.upper()} - {ood}: macro-F1 with 95% CI")}
        # per-language / per-generator slices of the strongest model
        model = best_neural(st)
        slices = st["models"].get(model, {}).get("splits", {}).get("id_val", {}).get("per_slice", {})
        for col, vals in slices.items():
            if len(vals) > 1:
                cats = list(vals)
                block.setdefault("figures", {})[f"slice_{col}"] = grouped_bars(
                    {model: [vals[c]["macro_f1"] for c in cats]}, cats,
                    figures / f"slice_{col}_task{task.lower()}.png",
                    title=f"Task {task.upper()} {model}: macro-F1 by {col} (ID-val)")
        imp = block["feature_importance"].get("style_gbdt") or block["feature_importance"].get("lm_gbdt")
        if imp:
            top = imp[:12]
            block.setdefault("figures", {})["feature_importance"] = horizontal_bars(
                [d["feature"] for d in top], [d["macro_f1_drop"] for d in top],
                figures / f"feature_importance_task{task.lower()}.png",
                title=f"Task {task.upper()}: permutation importance", xlabel="macro-F1 drop when shuffled")
        analysis["tasks"][task] = block
        md = error_analysis(paths, task, st)
        if md:
            (Path(a.out) / f"error_analysis_task{task.lower()}.md").write_text(md, encoding="utf-8")
        log.info(f"task {task}: analysis done ({len(block.get('figures', {}))} figures)")

    write_json(analysis, Path(a.out) / "analysis.json")
    print(f"wrote {Path(a.out) / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
