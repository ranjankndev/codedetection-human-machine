"""The anti-drift test: the numbers in README.md must be exactly what the JSON says.

If someone (human or model) edits a result into the README by hand, this fails. If `results/` has not
been produced in this checkout yet, there is nothing to compare and the test skips.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from codedet.report import GENERATED_BEGIN, GENERATED_END, render_results_block

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def _load(name: str) -> dict:
    f = RESULTS / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _generated_region(text: str) -> str | None:
    if GENERATED_BEGIN not in text or GENERATED_END not in text:
        return None
    return GENERATED_BEGIN + text.split(GENERATED_BEGIN, 1)[1].split(GENERATED_END, 1)[0] + GENERATED_END


@pytest.fixture(scope="module")
def expected() -> str:
    summary = _load("summary.json")
    if not summary.get("tasks"):
        pytest.skip("no results/summary.json in this checkout; run scripts/run_all.py first")
    return render_results_block(summary, _load("analysis.json"), _load("bench.json"))


def test_readme_generated_block_matches_results(expected):
    readme = REPO / "README.md"
    assert readme.exists(), "README.md is missing"
    region = _generated_region(readme.read_text(encoding="utf-8"))
    assert region is not None, "README.md has no generated results block"
    if "Run `python scripts/run_all.py" in region:
        pytest.skip("README still holds the placeholder block; run scripts/make_report.py")
    assert region.strip() == expected.strip(), (
        "README results differ from results/*.json - run `python scripts/make_report.py`")


def test_summary_md_matches_results(expected):
    f = RESULTS / "summary.md"
    if not f.exists():
        pytest.skip("results/summary.md not generated yet")
    region = _generated_region(f.read_text(encoding="utf-8"))
    assert region is not None and region.strip() == expected.strip(), (
        "results/summary.md is stale - run `python scripts/make_report.py`")


def test_every_reported_split_has_identical_rows_across_models():
    summary = _load("summary.json")
    if not summary.get("tasks"):
        pytest.skip("no results yet")
    for task, block in summary["tasks"].items():
        for split, info in block["identical_rows"].items():
            assert info["all_models_same_rows"], (
                f"task {task} split {split}: models were scored on different rows ({info['models']})")
