"""Build the updated deck from the generated results — no number is typed into a slide by hand.

    python scripts/make_slides.py --out results --deck docs/codedet_slides.pptx

Structure follows `docs/slides_update.md`: the original story, plus the three slides the rebuild earned
(audit & fixes, controlled ablation with CIs, shortcut learning & calibration). Figures come from
`results/figures/`; tables come from `results/summary.json` and `results/analysis.json`.

Skips cleanly (exit 0) when python-pptx is not installed - this is a packaging nicety, not a pipeline
stage, and it must never fail a run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from codedet.report import eval_splits, fmt_ci, label, split_label
from codedet.utils import get_logger, read_json

log = get_logger("slides")

TITLE = "Detecting machine-generated code"
SUBTITLE = "SemEval-2026 Task 13 - same idea, honest evaluation"


def _load(out: Path, name: str) -> dict:
    f = out / name
    return read_json(f) if f.exists() else {}


def _bullets(slide, lines, size=16):
    from pptx.util import Pt

    body = slide.placeholders[1].text_frame
    body.clear()
    for i, line in enumerate(lines):
        p = body.paragraphs[0] if i == 0 else body.add_paragraph()
        p.text = line
        p.level = 0 if not line.startswith("  ") else 1
        for run in p.runs:
            run.font.size = Pt(size)


def _table_slide(prs, title: str, header: list[str], rows: list[list[str]], note: str = ""):
    from pptx.util import Inches, Pt

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title
    n_rows = len(rows) + 1
    shape = slide.shapes.add_table(n_rows, len(header), Inches(0.4), Inches(1.5),
                                   Inches(9.2), Inches(0.3 * n_rows))
    table = shape.table
    for j, h in enumerate(header):
        cell = table.cell(0, j)
        cell.text = h
        cell.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            cell = table.cell(i, j)
            cell.text = str(v)
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)
    if note:
        box = slide.shapes.add_textbox(Inches(0.4), Inches(1.5 + 0.3 * n_rows + 0.15), Inches(9.2), Inches(0.8))
        tf = box.text_frame
        tf.word_wrap = True
        tf.text = note
        tf.paragraphs[0].runs[0].font.size = Pt(11)
    return slide


def _picture_slide(prs, title: str, images: list[Path], caption: str = ""):
    from pptx.util import Inches, Pt

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title
    images = [p for p in images if p.exists()][:2]
    width = Inches(4.4 if len(images) > 1 else 6.5)
    for i, img in enumerate(images):
        slide.shapes.add_picture(str(img), Inches(0.5 + i * 4.7), Inches(1.6), width=width)
    if caption:
        box = slide.shapes.add_textbox(Inches(0.5), Inches(6.2), Inches(9.0), Inches(0.9))
        box.text_frame.word_wrap = True
        box.text_frame.text = caption
        box.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    return slide


def build(out: Path, deck: Path) -> Path:
    from pptx import Presentation
    from pptx.util import Inches, Pt

    summary, analysis, bench = _load(out, "summary.json"), _load(out, "analysis.json"), _load(out, "bench.json")
    figures = out / "figures"
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(10), Inches(7.5)

    # 1 title
    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = TITLE
    s.placeholders[1].text = f"{SUBTITLE}\npreset `{summary.get('preset', '?')}`, seed {summary.get('seed', '?')}"

    # 2 the mechanism
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "What GPAD actually computes"
    _bullets(s, [
        "CodeBERT encoder -> 1 extra block -> MLP head",
        "Bias on pre-softmax attention scores:  score_qk += sign . alpha . sigma(mean(H) - H_k)",
        "  H_k = entropy of a LEARNED softmax over key self-similarity (not predictive entropy)",
        "  at sign=+1 this boosts LOW-entropy keys - the opposite of the original write-up",
        "Gated fusion of 16 z-scored stylometric features",
        "True predictive entropy / NLL from a frozen code LM is a separate, named feature set",
    ])

    # 3 audit & fixes
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Audit of my own project, and the fixes"
    _bullets(s, [
        "'Entropy' meant three different things -> one name per mechanism",
        "No control for the extra block -> parameter-matched capacity control",
        "Tables compared 349 vs 17,226 rows -> identical rows by id, n printed",
        "Task B accuracy 0.64 / macro-F1 0.29 -> majority-class illusion, class-aware loss",
        "'Entropy-only is best' from single runs -> paired bootstrap + McNemar, ties called ties",
        "Python-only stylometry acted as a language detector -> language-agnostic features",
    ])

    # 4-5 results tables
    for task, block in summary.get("tasks", {}).items():
        splits = eval_splits(block)
        rows = []
        for model, mb in sorted(block["models"].items(),
                                key=lambda kv: -kv[1]["splits"].get(splits[-1], {}).get("macro_f1", 0)):
            rows.append([label(model)] + [fmt_ci(mb["splits"].get(sp)) for sp in splits])
        title = "Task A - human vs machine" if task.upper() == "A" else "Task B - which LLM family"
        _table_slide(prs, f"{title} (macro-F1, 95% bootstrap CI)",
                     ["model"] + [split_label(sp) for sp in splits], rows,
                     note="Same rows for every model. Overlapping intervals mean 'tie', not 'better'.")

    # 6 controlled ablation
    paired = summary.get("tasks", {}).get("A", {}).get("paired", {})
    rows = []
    for key, entry in paired.items():
        for sp, v in entry["splits"].items():
            b = v["bootstrap"]
            rows.append([key.replace("_vs_", " - "), split_label(sp), f"{v['n']}",
                         f"{b['diff']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]",
                         "significant" if b["excludes_zero"] else "tie"])
    if rows:
        branch = summary.get("readme_branch", {}).get("branch", "?")
        _table_slide(prs, "Controlled ablation: does the mechanism beat its own control?",
                     ["comparison", "split", "n", "macro-F1 diff [95% CI]", "verdict"], rows,
                     note=f"Decision rule: GPAD - control CI must exclude 0 on an OOD split. Outcome: {branch}.")

    # 7 shortcut + calibration figures
    task_a = analysis.get("tasks", {}).get("A", {})
    sc = task_a.get("shortcut", {})
    ladder = sc.get("shortcut_ladder", {})
    cap = ""
    if "length_only" in ladder:
        lo = ladder["length_only"]
        cap = (f"A length/whitespace-only regression scores {lo.get('id_val', float('nan')):.3f} in-distribution "
               f"and {lo.get(sc.get('ood_split', 'ood_lang'), float('nan')):.3f} on the held-out language. "
               "The gap to the transformer is how much of the task is about authorship rather than formatting.")
    _picture_slide(prs, "Shortcut learning and calibration under shift",
                   [figures / "forest_taska_ood_lang.png", figures / "reliability_taska_ood_lang.png"], cap)

    # 8 imbalance
    _picture_slide(prs, "Task B is an imbalance problem",
                   [figures / "confusion_taskb_id_val.png", figures / "confusion_taskb_ood_gen.png"],
                   "Macro-F1 next to accuracy; the majority-class baseline is on the results slide.")

    # 9 efficiency
    rows = []
    for name, res in (bench.get("models") or {}).items():
        if "error" in res:
            continue
        for r in res["rows"]:
            if "p50_ms" in r and r["batch_size"] == min(b["batch_size"] for b in res["rows"] if "p50_ms" in b):
                rows.append([label(name), f"{res['params']['total_M']:.1f}M / {res['params']['trainable_M']:.1f}M",
                             f"{r['device']}/{r['variant']}", f"{r['p50_ms']:.1f}", f"{r['p95_ms']:.1f}"])
    if rows:
        _table_slide(prs, "Serving cost", ["model", "params total/trainable", "device/precision",
                                           "p50 ms", "p95 ms"], rows[:12])

    # 10 original results, clearly labelled
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Original competition-time results (not re-validated)"
    _bullets(s, [
        "Task A macro-F1 reported in 2025: 0.9731 baseline -> 0.9957 GPAD-2 (seen/seen)",
        "  unseen language 0.5676 -> 0.7268; unseen/unseen 0.5606 -> 0.6538",
        "Task B macro-F1: 0.29 baseline, 0.34 entropy-only, 0.20 style-only",
        "Measured before the fixes on the audit slide: different row counts, no control, no CIs",
        "Kept here for honesty, not comparison - see docs/ORIGINAL_RESULTS.md",
    ])

    # 11 next
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "What I would do next"
    _bullets(s, [
        "Full data, multiple seeds, the lite preset on a GPU session",
        "Larger code LM for the entropy features; the true Binoculars ratio with a second model",
        "Domain-adversarial training against the language shortcut",
        "Open-set rejection for Task B so an unseen generator can answer 'unknown'",
        "Distillation + ONNX export, and re-measurement whenever a new model family ships",
    ])

    deck.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(deck))
    return deck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--deck", default="docs/codedet_slides.pptx")
    a = ap.parse_args()
    out = Path(a.out)
    if not (out / "summary.json").exists():
        log.warning("no summary.json; run scripts/evaluate.py first")
        return 0
    try:
        deck = build(out, Path(a.deck))
    except ImportError:
        log.warning("python-pptx not installed; skipping the deck (pip install python-pptx)")
        return 0
    log.info(f"wrote {deck}")
    print(f"wrote {deck}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
