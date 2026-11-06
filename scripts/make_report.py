"""Write results/REPORT.md (one page) and results/APPENDIX.md (every table), and refresh README.md.

    python scripts/make_report.py --out results --readme README.md

README.md and REPORT.md share one generated block, delimited by HTML comments; everything outside it is
hand-written and preserved. `tests/test_report_matches_results.py` re-renders the block from
results/*.json and fails if either file drifted - that is the mechanism behind "no number in this repo
was typed by hand".
"""
from __future__ import annotations

import argparse
from pathlib import Path

from codedet.evaluation.report import render_appendix, render_results_block, splice_generated
from codedet.utils import get_logger, read_json

log = get_logger("make_report")

# the three pictures that carry the story; the appendix links the rest
KEY_FIGURES = (
    ("forest_taska_ood_lang.png", "Every model on the unseen language, with 95% intervals"),
    ("reliability_taska_ood_lang.png", "Confidence vs. accuracy on the unseen language"),
    ("confusion_taska_ood_lang.png", "Where the mistakes go on the unseen language"),
)


def load(path: Path) -> dict:
    return read_json(path) if path.exists() else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--readme", default="README.md")
    a = ap.parse_args()

    out = Path(a.out)
    summary, analysis, bench = load(out / "summary.json"), load(out / "analysis.json"), load(out / "bench.json")
    status = load(out / "status.json")
    if not summary:
        log.warning("no summary.json; run scripts/stages/evaluate.py first")
        return 1
    block = render_results_block(summary, analysis, bench, status)
    figures = sorted(p.name for p in (out / "figures").glob("*.png"))

    md = ["# Results", "", block, ""]
    key = [(f, cap) for f, cap in KEY_FIGURES if f in figures]
    if key:
        md += ["## Key figures", ""] + [f"**{cap}**\n\n![{cap}](figures/{f})\n" for f, cap in key]
    md += ["Every other table (paired tests, stress test, calibration, per-class, speed, run status) and "
           "figure: [APPENDIX.md](APPENDIX.md). The ten most confident mistakes per task: "
           "[error_analysis_taska.md](error_analysis_taska.md), [error_analysis_taskb.md](error_analysis_taskb.md).",
           ""]
    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    (out / "APPENDIX.md").write_text(render_appendix(summary, analysis, bench, status, figures), encoding="utf-8")
    (out / "summary.md").unlink(missing_ok=True)   # the old all-in-one report, replaced by the two above
    log.info(f"wrote {out / 'REPORT.md'} and {out / 'APPENDIX.md'}")

    readme = Path(a.readme)
    if readme.exists():
        readme.write_text(splice_generated(readme.read_text(encoding="utf-8"), block), encoding="utf-8")
        log.info(f"refreshed generated block in {readme}")
    print(f"wrote {out / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
