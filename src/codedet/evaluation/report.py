"""Turn results JSON into markdown. The ONLY place that writes numbers into prose.

Rule for this project: no number is ever typed by hand. README and results/REPORT.md embed the same
short generated block, and a test regenerates it from `results/*.json` and fails if the committed text
has drifted. Everything too detailed for that block goes to results/APPENDIX.md (`render_appendix`).
The verdict sentences are chosen from confidence intervals, not from point estimates, so the report
cannot claim "better" where the data says "tie".
"""
from __future__ import annotations

from typing import Iterable

GENERATED_BEGIN = "<!-- BEGIN GENERATED RESULTS - produced by scripts/make_report.py, do not edit by hand -->"
GENERATED_END = "<!-- END GENERATED RESULTS -->"

# plain names for readers; the A*/B* codes are the ids used in docs/ARCHITECTURE.md and the appendix
MODEL_LABELS = {
    "majority": "Majority class (floor)",
    "length_only": "Length + whitespace only (shortcut probe)",
    "tfidf_lr": "Character n-grams + logistic regression",
    "style_gbdt": "Style features + gradient boosting",
    "lm_gbdt": "Style + LM entropy + gradient boosting",
    "codebert": "CodeBERT fine-tune",
    "codebert_weighted": "CodeBERT, class-weighted loss",
    "lora": "CodeBERT + LoRA",
    "gpad": "GPAD (entropy bias + style)",
    "gpad_both": "GPAD (entropy + style)",
    "control": "GPAD with the bias switched off (control)",
    "gpad_entropy_only": "GPAD, entropy only",
    "gpad_style_only": "GPAD, style only",
    "gpad_sign_neg": "GPAD, entropy sign flipped",
    "stack": "Stacked ensemble",
}
MODEL_CODES = {
    "majority": "A0a", "length_only": "A0b", "tfidf_lr": "A1", "style_gbdt": "A2", "lm_gbdt": "A3",
    "codebert": "A4", "lora": "A5", "gpad": "A6", "gpad_sign_neg": "A6b", "control": "A7", "stack": "A8",
    "codebert_weighted": "B2", "gpad_entropy_only": "B3", "gpad_style_only": "B4", "gpad_both": "B5",
}

SPLIT_LABELS = {
    "id_val": "in-distribution", "ood_lang": "unseen language", "ood_gen": "unseen generator",
    "official_val": "official val set", "train": "train (in-sample)",
}
TASK_TITLES = {"A": "Task A - human or machine?", "B": "Task B - which model family wrote it?"}
CONSTANT_MODELS = {"majority"}   # nothing to calibrate or threshold


def label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def split_label(split: str) -> str:
    if "__" in split:
        base, pert = split.split("__", 1)
        return f"{SPLIT_LABELS.get(base, base)} ({pert})"
    return SPLIT_LABELS.get(split, split)


def fmt_ci(block: dict | None, digits: int = 3) -> str:
    if not block:
        return "-"
    ci = block.get("macro_f1_ci")
    if not ci:
        return f"{block['macro_f1']:.{digits}f}"
    return f"{ci['point']:.{digits}f} [{ci['lo']:.{digits}f}, {ci['hi']:.{digits}f}]"


def _table(header: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    header = list(header)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def eval_splits(task_block: dict) -> list[str]:
    """Reported splits: ID-val first, then OOD; perturbed and in-sample splits live in their own tables."""
    seen = {s for m in task_block["models"].values() for s in m["splits"]}
    return [s for s in ("id_val", "official_val", "ood_lang", "ood_gen") if s in seen]


def headline_splits(task_block: dict) -> list[str]:
    """The two columns a reader needs: in-distribution and the shifted split."""
    return [s for s in eval_splits(task_block) if s != "official_val"]


def _sorted_models(task_block: dict, split: str | None):
    return sorted(task_block["models"].items(),
                  key=lambda kv: -(kv[1]["splits"].get(split, {}).get("macro_f1", 0) if split else 0))


def pair_verdict(bootstrap: dict, p_value: float) -> str:
    """Two tests, one word. Call it a difference only when the CI and McNemar agree."""
    ci, mc = bool(bootstrap["excludes_zero"]), p_value < 0.05
    if ci and mc:
        return "**different**"
    if ci or mc:
        return "unclear (tests disagree)"
    return "tie"


def missing_models(status: dict | None, task: str) -> list[str]:
    prefix = f"train_{task.upper()}_"
    return [s["stage"][len(prefix):] for s in (status or {}).get("stages", [])
            if s["stage"].startswith(prefix) and s["status"] in ("failed", "skipped", "timeout")]


# ---------------------------------------------------------------- tables
def headline_table(task_block: dict) -> str:
    splits = headline_splits(task_block)
    rows = [[label(m), *[fmt_ci(b["splits"].get(s)) for s in splits]]
            for m, b in _sorted_models(task_block, splits[-1] if splits else None)]
    return _table(["model", *[split_label(s) for s in splits]], rows)


def results_table(task_block: dict) -> str:
    splits = eval_splits(task_block)
    rows = []
    for model, block in _sorted_models(task_block, splits[-1] if splits else None):
        seeds = block.get("seeds") or 1
        spread = f"{block['seed_macro_f1_std']:.3f}" if seeds > 1 and block.get("seed_macro_f1_std") is not None else "-"
        rows.append([MODEL_CODES.get(model, "-"), label(model), *[fmt_ci(block["splits"].get(s)) for s in splits],
                     f"{seeds}", spread])
    return _table(["id", "model", *[split_label(s) for s in splits], "seeds", "seed sd"], rows)


def paired_table(task_block: dict) -> str:
    rows = []
    for key, entry in task_block.get("paired", {}).items():
        left, right = key.split("_vs_")
        for split, v in entry["splits"].items():
            b, p = v["bootstrap"], v["mcnemar"]["p_value"]
            rows.append([f"{left} - {right}", split_label(split), f"{v['n']}",
                         f"{b['diff']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]", f"{p:.3g}", pair_verdict(b, p)])
    return _table(["comparison", "split", "n", "macro-F1 diff [95% CI]", "McNemar p", "verdict"], rows)


def per_class_table(task_block: dict, split: str = "id_val", model: str | None = None) -> str:
    models = task_block["models"]
    model = model or next((m for m in ("gpad_both", "gpad", "codebert_weighted", "codebert") if m in models), None)
    block = models.get(model, {}).get("splits", {}).get(split)
    if not block:
        return ""
    rows = [[name, f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}", v["support"]]
            for name, v in block["per_class"].items()]
    return f"{label(model)}, {split_label(split)}:\n\n" + _table(["class", "precision", "recall", "F1", "support"], rows)


def shortcut_table(analysis_task: dict) -> str:
    """Change = perturbed minus original, so a negative number means the model got worse."""
    rows = []
    for model, row in analysis_task.get("shortcut", {}).get("perturbations", {}).items():
        rows.append([label(model), f"{row['ood_macro_f1']:.3f}",
                     f"{row.get('nocomments', float('nan')):.3f}", f"{-row.get('nocomments_drop', 0):+.3f}",
                     f"{row.get('stripped', float('nan')):.3f}", f"{-row.get('stripped_drop', 0):+.3f}"])
    if not rows:
        return ""
    return ("Same OOD rows, macro-F1. *change* is perturbed minus original: negative = the model got worse.\n\n" +
            _table(["model", "original", "comments removed", "change", "comments + formatting stripped", "change"],
                   rows))


def calibration_table(analysis_task: dict) -> str:
    cal = analysis_task.get("calibration", {})
    if not cal.get("ece") or cal.get("model") in CONSTANT_MODELS:
        return ""
    rows = [[split_label(s), f"{v['before']:.3f}", f"{v['after_temperature']:.3f}"] for s, v in cal["ece"].items()]
    return (f"{label(cal['model'])}; temperature fitted on in-distribution data: "
            f"**T = {cal['temperature_fitted_on_id_val']:.2f}**. Lower ECE is better.\n\n" +
            _table(["split", "ECE raw", "ECE after temperature"], rows))


def threshold_table(analysis_task: dict) -> str:
    th = analysis_task.get("threshold", {})
    if not th.get("splits"):
        return ""
    rows = []
    for split, variants in th["splits"].items():
        for name, v in variants.items():
            rows.append([split_label(split), name, f"{v['threshold']:.2f}", f"{v['precision_machine']:.3f}",
                         f"{v['recall_machine']:.3f}", f"{v['false_accusation_rate']:.3f}",
                         f"{v['expected_cost_per_1000']:.0f}"])
    return (f"A false accusation costs **{th['cost_ratio_fp_to_fn']:.0f}x** a miss. The threshold is chosen on "
            f"in-distribution data only, then applied unchanged to OOD.\n\n" +
            _table(["split", "rule", "threshold", "precision (machine)", "recall (machine)",
                    "false-accusation rate", "cost / 1000 snippets"], rows))


def threshold_summary(analysis_task: dict) -> str:
    """Two rows for the headline: what the default and the cost-aware threshold do to false accusations."""
    th = analysis_task.get("threshold", {})
    splits = th.get("splits", {})
    if len(splits) < 2:
        return ""
    names = {"argmax_0.5": "default (0.5)", "cost_optimal": "cost-aware"}
    order = list(splits)
    rows = []
    for rule, nice in names.items():
        if all(rule in splits[s] for s in order):
            rows.append([nice, f"{splits[order[0]][rule]['threshold']:.2f}",
                         *[f"{splits[s][rule]['false_accusation_rate']:.1%}" for s in order]])
    return (f"**Humans wrongly flagged as machine** (a false accusation costs {th['cost_ratio_fp_to_fn']:.0f}x a "
            f"miss; threshold picked on in-distribution data only, model `{th.get('model', '?')}`):\n\n" +
            _table(["threshold", "value", *[split_label(s) for s in order]], rows))


def bench_table(bench: dict) -> str:
    if not bench or not bench.get("models"):
        return ""
    rows = []
    for name, res in bench["models"].items():
        if "error" in res:
            rows.append([label(name), "-", "-", "-", "-", "-", res["error"][:40]])
            continue
        for r in res["rows"]:
            if "p50_ms" not in r:
                continue
            rows.append([label(name), f"{res['params']['total_M']:.1f}M / {res['params']['trainable_M']:.1f}M",
                         f"{r['device']}/{r['variant']}", r["batch_size"], f"{r['p50_ms']:.1f}",
                         f"{r['p95_ms']:.1f}", f"{r['throughput_per_s']:.0f}"])
    return (f"Sequence length {bench.get('seq_len')} tokens, base model `{bench.get('base_model')}`; "
            "architectures built fresh (latency is weight-independent).\n\n" +
            _table(["model", "params total/trainable", "device/precision", "batch", "p50 ms", "p95 ms", "snippets/s"],
                   rows))


def feature_importance_table(analysis_task: dict, k: int = 10) -> str:
    fi = analysis_task.get("feature_importance", {})
    block = fi.get("style_gbdt") or fi.get("lm_gbdt") or fi.get("length_only")
    if not block:
        return ""
    rows = [[d["feature"], f"{d['macro_f1_drop']:+.4f}"] for d in block[:k]]
    return _table(["feature", "macro-F1 lost when shuffled (in-distribution)"], rows)


# ---------------------------------------------------------------- generated prose
def verdict_lines(summary: dict, analysis: dict) -> list[str]:
    """Sentences that are allowed only because a CI supports them."""
    out = []
    branch = summary.get("readme_branch", {})
    a = summary["tasks"].get("A", {})
    gc = a.get("paired", {}).get("gpad_vs_control")
    if gc:
        if branch.get("branch") == "entropy_helps":
            sig = ", ".join(split_label(s) for s in gc["significant_splits"])
            out.append(f"**The entropy bias survives its control.** GPAD beats the same model with the bias "
                       f"switched off on {sig}; the paired 95% CI excludes zero.")
        else:
            diffs = ", ".join(f"{v['bootstrap']['diff']:+.3f} ({split_label(s)})" for s, v in gc["splits"].items())
            out.append(f"**The entropy bias adds nothing beyond its extra parameters.** GPAD minus the same model "
                       f"with the bias switched off: {diffs}. No interval rules out zero, so the honest reading is "
                       f"a tie. The original report credited the entropy mechanism for a gain this comparison "
                       f"cannot support.")
    sc = analysis.get("tasks", {}).get("A", {}).get("shortcut", {})
    gap = sc.get("length_only_vs_transformer")
    ladder = sc.get("shortcut_ladder", {})
    if gap and "length_only" in ladder:
        ood_key = sc.get("ood_split", "ood_lang")
        lo = ladder["length_only"]
        gap_ood = gap.get(ood_key, 0.0)
        line = (f"**A length-and-whitespace shortcut is hard to beat.** Six content-free features score "
                f"{lo.get('id_val', float('nan')):.3f} in-distribution and {lo.get(ood_key, float('nan')):.3f} on "
                f"the {split_label(ood_key)}; the transformer (`{sc.get('ceiling_model', '?')}`) is "
                f"{gap.get('id_val', 0):+.3f} and {gap_ood:+.3f} against that.")
        if gap_ood <= 0:
            line += " Out of distribution it learned formatting, not authorship."
        out.append(line)
    calib = analysis.get("tasks", {}).get("A", {}).get("calibration", {})
    items = list(calib.get("ece", {}).items())
    if len(items) >= 2 and calib.get("model") not in CONSTANT_MODELS:
        (s0, v0), (s1, v1) = items[0], items[1]
        out.append(f"**Confidence does not survive the shift.** Calibration error (ECE) is "
                   f"{v0['after_temperature']:.3f} {split_label(s0)} but {v1['after_temperature']:.3f} on the "
                   f"{split_label(s1)}, even after temperature scaling.")
    th = analysis.get("tasks", {}).get("A", {}).get("threshold", {}).get("splits", {})
    if len(th) >= 2:
        s0, s1 = list(th)[:2]
        d, c = th[s1].get("argmax_0.5"), th[s1].get("cost_optimal")
        if d and c:
            out.append(f"**The default threshold accuses too many humans.** On the {split_label(s1)}, 0.5 flags "
                       f"{d['false_accusation_rate']:.0%} of human snippets as machine; a threshold chosen for a 5:1 "
                       f"cost brings that to {c['false_accusation_rate']:.0%}, still far from usable.")
    b = summary["tasks"].get("B", {})
    if b:
        maj = b["models"].get("majority", {}).get("splits", {}).get("id_val")
        best = max(((m, v["splits"]["id_val"]) for m, v in b["models"].items()
                    if "id_val" in v["splits"] and m not in CONSTANT_MODELS),
                   key=lambda kv: kv[1]["macro_f1"], default=None)
        if maj and best:
            out.append(f"**Task B is an imbalance problem.** Always guessing the biggest class gives accuracy "
                       f"{maj['accuracy']:.3f} but macro-F1 {maj['macro_f1']:.3f}; the best model here "
                       f"(`{best[0]}`) reaches macro-F1 {best[1]['macro_f1']:.3f}. Accuracy alone would hide this.")
        eo = b.get("paired", {}).get("gpad_entropy_only_vs_codebert_weighted")
        if eo:
            out.append(f"**The old 'entropy-only is best' claim**, re-tested on identical rows: {eo['verdict']}.")
    return out


def render_results_block(summary: dict, analysis: dict, bench: dict | None = None,
                         status: dict | None = None) -> str:
    """The short block embedded verbatim in README.md and results/REPORT.md. A test re-renders and diffs it."""
    parts = [GENERATED_BEGIN, ""]
    if str(summary.get("preset")) == "local_cpu":
        parts += ["> **These numbers are a plumbing dry run, not results.** Preset `local_cpu` trains a "
                  "2-layer *randomly initialised* encoder on ~300 rows of synthetic fixture code. It exists "
                  "to prove every stage runs and every artifact is produced.", ""]
    encoder = f", transformer rows use `{bench['base_model']}`" if (bench or {}).get("base_model") else ""
    parts += [f"*Preset `{summary.get('preset')}`{encoder}. Scores are macro-F1 with a 95% bootstrap interval "
              f"({summary.get('n_boot')} resamples); every model is scored on the same rows. "
              f"Generated from `results/*.json`.*", "",
              "### Key findings", ""] + [f"- {line}" for line in verdict_lines(summary, analysis)] + [""]
    for task, block in summary["tasks"].items():
        sizes = ", ".join(f"{split_label(s)} n={block['rows_per_split'].get(s, 0)}" for s in headline_splits(block))
        parts += [f"### {TASK_TITLES.get(task.upper(), task)}", "", f"Rows: {sizes}.", "", headline_table(block), ""]
        missing = missing_models(status, task)
        if missing:
            parts += ["Not in this run (training stage failed or was skipped): " +
                      ", ".join(f"`{m}`" for m in missing) + ".", ""]
        ts = threshold_summary(analysis.get("tasks", {}).get(task, {})) if task.upper() == "A" else ""
        if ts:
            parts += [ts, ""]
    parts += [GENERATED_END]
    return "\n".join(parts)


def render_appendix(summary: dict, analysis: dict, bench: dict | None = None, status: dict | None = None,
                    figures: Iterable[str] = ()) -> str:
    """Every detail table, for the reader who wants to check the headline."""
    parts = ["# Appendix - every table behind the report", "",
             "Generated by `scripts/make_report.py` from `results/*.json`. Do not edit by hand. "
             "The short version is [REPORT.md](REPORT.md).", ""]
    for task, block in summary["tasks"].items():
        ablock = analysis.get("tasks", {}).get(task, {})
        parts += [f"## {TASK_TITLES.get(task.upper(), task)}", "", "### All models, all splits", "",
                  results_table(block), ""]
        sections = (
            ("Paired comparisons", "Same rows. *different* = the bootstrap CI excludes zero **and** McNemar "
                                   "p < 0.05; if only one test says so, *unclear*.\n\n" + paired_table(block)
             if block.get("paired") else ""),
            ("Stress test: strip comments and formatting", shortcut_table(ablock)),
            ("Calibration", calibration_table(ablock)),
            ("Operating point", threshold_table(ablock)),
            ("Per-class breakdown", per_class_table(block)),
            ("Which style features matter", feature_importance_table(ablock)),
        )
        for heading, text in sections:
            if text:
                parts += [f"### {heading}", "", text, ""]
        tag = f"task{task.lower()}"
        figs = [f for f in figures if tag in f]
        if figs:
            parts += ["### Figures", ""] + [f"![{f}](figures/{f})" for f in figs] + [""]
    bt = bench_table(bench or {})
    if bt:
        parts += ["## Speed and size", "", bt, ""]
    if status:
        parts += ["## Run status", "", f"Overall: **{status.get('status')}**, "
                  f"{status.get('elapsed_minutes', 0):.0f} min. Logs are in `results/logs/`.", "",
                  _table(["stage", "status", "seconds"],
                         [[s["stage"], s["status"], f"{s.get('seconds', 0):.0f}"] for s in status.get("stages", [])]),
                  ""]
    return "\n".join(parts)


def splice_generated(text: str, block: str) -> str:
    """Replace the generated region of a document, keeping the hand-written parts untouched."""
    if GENERATED_BEGIN in text and GENERATED_END in text:
        head = text.split(GENERATED_BEGIN)[0]
        tail = text.split(GENERATED_END, 1)[1]
        return head + block + tail
    return text.rstrip() + "\n\n" + block + "\n"
