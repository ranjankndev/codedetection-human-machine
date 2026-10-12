"""Turn results JSON into markdown. The ONLY place that writes numbers into prose.

Rule for this project: no number is ever typed by hand. README and summary.md embed a generated block,
and a test regenerates that block from `results/*.json` and fails if the committed text has drifted.
The verdict sentences are chosen from confidence intervals, not from point estimates, so the report
cannot claim "better" where the data says "tie".
"""
from __future__ import annotations

from typing import Iterable

GENERATED_BEGIN = "<!-- BEGIN GENERATED RESULTS - produced by scripts/make_report.py, do not edit by hand -->"
GENERATED_END = "<!-- END GENERATED RESULTS -->"

MODEL_LABELS = {
    "majority": "A0a majority class",
    "length_only": "A0b length/whitespace LR (shortcut probe)",
    "tfidf_lr": "A1 char 3-5-gram TF-IDF + LR",
    "style_gbdt": "A2 stylometry + GBDT",
    "lm_gbdt": "A3 stylometry + LM entropy/NLL + GBDT",
    "codebert": "A4 CodeBERT fine-tune",
    "codebert_weighted": "B2 CodeBERT + weighted CE",
    "lora": "A5 CodeBERT + LoRA",
    "gpad": "A6 GPAD (entropy bias + style)",
    "gpad_both": "B5 GPAD (entropy + style)",
    "control": "A7 capacity control (extra block, bias off)",
    "gpad_entropy_only": "B3 GPAD entropy-only",
    "gpad_style_only": "B4 GPAD style-only",
    "gpad_sign_neg": "A6b GPAD, entropy sign flipped",
    "stack": "A8 stacking (logits + features)",
}

SPLIT_LABELS = {
    "id_val": "ID val", "ood_lang": "OOD: unseen language", "ood_gen": "OOD: unseen generator",
    "train": "train (in-sample)",
}


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
    order = [s for s in ("id_val", "ood_lang", "ood_gen") if s in seen]
    return order


def results_table(task_block: dict) -> str:
    splits = eval_splits(task_block)
    rows = []
    for model, block in sorted(task_block["models"].items(),
                               key=lambda kv: -(kv[1]["splits"].get(splits[-1], {}).get("macro_f1", 0) if splits else 0)):
        cells = [fmt_ci(block["splits"].get(s)) for s in splits]
        seeds = block.get("seeds")
        spread = (f"{block['seed_macro_f1_std']:.3f}" if block.get("seed_macro_f1_std") is not None else "-")
        rows.append([label(model), *cells, f"{seeds or 1}", spread])
    return _table(["model", *[split_label(s) for s in splits], "seeds", "seed sd"], rows)


def paired_table(task_block: dict) -> str:
    rows = []
    for key, entry in task_block.get("paired", {}).items():
        left, right = key.split("_vs_")
        for split, v in entry["splits"].items():
            b = v["bootstrap"]
            rows.append([f"{left} - {right}", split_label(split), f"{v['n']}",
                         f"{b['diff']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]",
                         f"{v['mcnemar']['p_value']:.3g}",
                         "**significant**" if b["excludes_zero"] else "tie"])
    return _table(["comparison (macro-F1 diff)", "split", "n", "diff [95% CI]", "McNemar p", "verdict"], rows)


def per_class_table(task_block: dict, split: str = "id_val", model: str | None = None) -> str:
    models = task_block["models"]
    model = model or next((m for m in ("gpad_both", "gpad", "codebert_weighted", "codebert") if m in models), None)
    block = models.get(model, {}).get("splits", {}).get(split)
    if not block:
        return ""
    rows = [[name, f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}", v["support"]]
            for name, v in block["per_class"].items()]
    return f"Model `{model}`, split `{split}`:\n\n" + _table(["class", "precision", "recall", "F1", "support"], rows)


def shortcut_table(analysis_task: dict) -> str:
    sc = analysis_task.get("shortcut", {})
    rows = []
    for model, row in sc.get("perturbations", {}).items():
        rows.append([label(model), f"{row['ood_macro_f1']:.3f}",
                     f"{row.get('nocomments', float('nan')):.3f}", f"{row.get('nocomments_drop', 0):+.3f}",
                     f"{row.get('stripped', float('nan')):.3f}", f"{row.get('stripped_drop', 0):+.3f}"])
    if not rows:
        return ""
    return _table(["model", "OOD macro-F1", "no comments", "drop", "comments+format stripped", "drop"], rows)


def calibration_table(analysis_task: dict) -> str:
    cal = analysis_task.get("calibration", {})
    if not cal.get("ece"):
        return ""
    rows = [[split_label(s), f"{v['before']:.3f}", f"{v['after_temperature']:.3f}"] for s, v in cal["ece"].items()]
    return (f"Temperature fitted on ID-val: **T = {cal['temperature_fitted_on_id_val']:.2f}** "
            f"(model `{cal['model']}`)\n\n" + _table(["split", "ECE raw", "ECE after temperature"], rows))


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
    return (f"Cost ratio: a false accusation (FP) is **{th['cost_ratio_fp_to_fn']:.0f}x** the cost of a miss. "
            f"Threshold chosen on ID-val only, then applied unchanged to OOD.\n\n" +
            _table(["split", "rule", "threshold", "precision (machine)", "recall (machine)",
                    "false-accusation rate", "cost / 1000 snippets"], rows))


def bench_table(bench: dict) -> str:
    if not bench or not bench.get("models"):
        return ""
    rows = []
    for name, res in bench["models"].items():
        if "error" in res:
            rows.append([label(name), "-", "-", "-", "-", res["error"][:40]])
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
    return _table(["feature", "macro-F1 drop when shuffled (ID-val)"], rows)


# ---------------------------------------------------------------- generated prose
def verdict_lines(summary: dict, analysis: dict) -> list[str]:
    """Sentences that are allowed only because a CI supports them."""
    out = []
    branch = summary.get("readme_branch", {})
    a = summary["tasks"].get("A", {})
    gc = a.get("paired", {}).get("gpad_vs_control")
    if gc:
        if branch.get("branch") == "entropy_helps":
            sig = ", ".join(gc["significant_splits"])
            out.append(f"**The attention-entropy bias survives its control.** GPAD beats the parameter-matched "
                       f"capacity control on {sig} with a paired-bootstrap 95% CI that excludes zero.")
        else:
            worst = ", ".join(f"{split_label(s)}: {v['bootstrap']['diff']:+.3f} "
                              f"[{v['bootstrap']['lo']:+.3f}, {v['bootstrap']['hi']:+.3f}]"
                              for s, v in gc["splits"].items())
            out.append(f"**The attention-entropy bias does not survive its control.** Against the "
                       f"parameter-matched control (same extra block, bias off), the macro-F1 difference is "
                       f"{worst} - every interval contains zero, so the honest reading is *no gain beyond capacity*. "
                       f"The original report credited the entropy mechanism for a gain this comparison cannot support.")
    sc = analysis.get("tasks", {}).get("A", {}).get("shortcut", {})
    gap = sc.get("length_only_vs_transformer")
    ladder = sc.get("shortcut_ladder", {})
    if gap and "length_only" in ladder:
        ood_key = sc.get("ood_split", "ood_lang")
        lo = ladder["length_only"]
        out.append(f"**Shortcut probe.** A logistic regression on nothing but length and whitespace reaches "
                   f"macro-F1 {lo.get('id_val', float('nan')):.3f} in-distribution and "
                   f"{lo.get(ood_key, float('nan')):.3f} on the held-out language; the transformer "
                   f"(`{sc.get('ceiling_model', '?')}`) is {gap.get('id_val', 0):+.3f} / "
                   f"{gap.get(ood_key, 0):+.3f} above that. The size of that gap is how much of the task is "
                   f"actually about authorship rather than formatting.")
    pert = sc.get("perturbations", {})
    if pert:
        worst = max(pert.items(), key=lambda kv: kv[1].get("stripped_drop", 0))
        out.append(f"**Stress test.** Stripping comments and normalising whitespace on the *same* OOD rows moves "
                   f"`{worst[0]}` by {worst[1].get('stripped_drop', 0):+.3f} macro-F1 "
                   f"({worst[1]['ood_macro_f1']:.3f} -> {worst[1].get('stripped', float('nan')):.3f}).")
    cal = analysis.get("tasks", {}).get("A", {}).get("calibration", {})
    if cal.get("ece"):
        items = list(cal["ece"].items())
        if len(items) >= 2:
            (s0, v0), (s1, v1) = items[0], items[1]
            out.append(f"**Calibration breaks under shift.** After temperature scaling fitted on ID-val, ECE is "
                       f"{v0['after_temperature']:.3f} on {split_label(s0)} but {v1['after_temperature']:.3f} on "
                       f"{split_label(s1)}: a confidence number calibrated in-distribution is not trustworthy "
                       f"off-distribution.")
    b = summary["tasks"].get("B", {})
    if b:
        maj = b["models"].get("majority", {}).get("splits", {}).get("id_val")
        best = max(((m, v["splits"]["id_val"]) for m, v in b["models"].items() if "id_val" in v["splits"]),
                   key=lambda kv: kv[1]["macro_f1"], default=None)
        if maj and best:
            out.append(f"**Task B is an imbalance problem.** The majority-class baseline already scores accuracy "
                       f"{maj['accuracy']:.3f} with macro-F1 {maj['macro_f1']:.3f}; the best model reaches macro-F1 "
                       f"{best[1]['macro_f1']:.3f} (accuracy {best[1]['accuracy']:.3f}, `{best[0]}`). Reporting "
                       f"accuracy here would have hidden the whole problem - which is what the original run did.")
        eo = b.get("paired", {}).get("gpad_entropy_only_vs_codebert_weighted")
        if eo:
            out.append(f"**The old 'entropy-only is best' claim.** Re-tested against weighted CodeBERT on identical "
                       f"rows: {eo['verdict']}.")
    return out


def render_results_block(summary: dict, analysis: dict, bench: dict | None = None) -> str:
    """The block embedded verbatim in README.md and summary.md. A test re-renders and diffs it."""
    parts = [GENERATED_BEGIN, ""]
    if str(summary.get("preset")) == "local_cpu":
        parts += ["> **These numbers are a plumbing dry run, not results.** Preset `local_cpu` trains a "
                  "2-layer *randomly initialised* encoder on ~300 rows of synthetic fixture code. It exists "
                  "to prove every stage runs and every artifact is produced. The portfolio numbers come from "
                  "the `lite` preset on the official SemEval-2026 Task 13 data.", ""]
    parts += [f"*Preset `{summary.get('preset')}`, seed {summary.get('seed')}, "
              f"{summary.get('n_boot')} bootstrap resamples. Every number below is read from "
              f"`results/summary.json` / `results/analysis.json`.*", ""]
    for task, block in summary["tasks"].items():
        title = "Task A - human vs machine" if task.upper() == "A" else "Task B - which LLM family"
        parts += [f"### {title}", "",
                  f"Rows per split: " + ", ".join(f"`{split_label(s)}` n={block['rows_per_split'].get(s, 0)}"
                                                  for s in eval_splits(block)) +
                  ". Every model is scored on the identical rows.", "",
                  results_table(block), ""]
        pt = paired_table(block)
        if pt.count("\n") > 1:
            parts += ["**Paired comparisons** (same rows, 95% bootstrap CI of the macro-F1 difference + McNemar):",
                      "", pt, ""]
        ablock = analysis.get("tasks", {}).get(task, {})
        for heading, text in (("Shortcut stress test", shortcut_table(ablock)),
                              ("Calibration", calibration_table(ablock)),
                              ("Operating point", threshold_table(ablock)),
                              ("Per-class breakdown", per_class_table(block)),
                              ("Feature importance", feature_importance_table(ablock))):
            if text:
                parts += [f"**{heading}**", "", text, ""]
    bt = bench_table(bench or {})
    if bt:
        parts += ["### Efficiency", "", bt, ""]
    parts += ["### What the numbers say", ""] + [f"- {line}" for line in verdict_lines(summary, analysis)]
    parts += ["", GENERATED_END]
    return "\n".join(parts)


def splice_generated(text: str, block: str) -> str:
    """Replace the generated region of a document, keeping the hand-written parts untouched."""
    if GENERATED_BEGIN in text and GENERATED_END in text:
        head = text.split(GENERATED_BEGIN)[0]
        tail = text.split(GENERATED_END, 1)[1]
        return head + block + tail
    return text.rstrip() + "\n\n" + block + "\n"
