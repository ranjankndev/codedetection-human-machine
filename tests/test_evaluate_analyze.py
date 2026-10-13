"""End-to-end test of the evaluation and analysis stages on synthetic prediction files.

No model is trained here: the stages read `probs_*.npz` by contract, which is exactly the property that
makes them testable. What is asserted is what the project claims: identical rows across models, a CI on
every reported number, a paired comparison that calls a tie a tie, and the stress/calibration/threshold
blocks that the README quotes.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codedet.artifacts import probs_as_logits, save_preds
from codedet.config import RunPaths
from codedet.fixtures import make_fixture
from codedet.splits import prepare_frame
from codedet.utils import write_json

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_cli(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_cli", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_cli"] = mod
    spec.loader.exec_module(mod)
    return mod


def _probs(y, accuracy: float, k: int, rng) -> np.ndarray:
    """Predictions that are right `accuracy` of the time, with plausible confidences."""
    n = len(y)
    pred = np.where(rng.random(n) < accuracy, y, rng.integers(0, k, n))
    p = np.full((n, k), 0.1 / max(k - 1, 1))
    p[np.arange(n), pred] = 0.9
    return p / p.sum(1, keepdims=True)


@pytest.fixture
def run(tmp_path):
    """A complete fake run: task A splits on disk plus predictions for four models."""
    rng = np.random.default_rng(0)
    paths = RunPaths.make(tmp_path)
    frame, names, _ = prepare_frame(make_fixture(400, seed=3), "A")
    splits = {"train": frame.iloc[:200], "id_val": frame.iloc[200:300], "ood_lang": frame.iloc[300:]}
    for split, f in splits.items():
        file = paths.split_file("A", split)
        file.parent.mkdir(parents=True, exist_ok=True)
        f.to_parquet(file, index=False)
    write_json({"A": {"task": "A", "num_labels": 2, "label_names": names, "preset": "test", "seed": 0,
                      "sizes": {k: len(v) for k, v in splits.items()}}}, tmp_path / "data_report.json")

    accuracies = {"majority": 0.5, "length_only": 0.7, "control": 0.85, "gpad": 0.86}
    for model, acc in accuracies.items():
        for split, f in splits.items():
            y = f["y"].values
            save_preds(paths.pred_file("A", model, split), probs_as_logits(_probs(y, acc, 2, rng)),
                       y, f["row_id"].values)
        if model in ("gpad", "control"):      # the perturbed OOD scoring train.py would have written
            y = splits["ood_lang"]["y"].values
            save_preds(paths.pred_file("A", model, "ood_lang__stripped"),
                       probs_as_logits(_probs(y, acc - 0.2, 2, rng)), y, splits["ood_lang"]["row_id"].values)
        write_json({"task": "A", "model": model, "seeds": 1, "seed_macro_f1_std": 0.0,
                    "splits": {}}, paths.metrics_file("A", model))
    return tmp_path


def _run_cli(mod, argv):
    old = sys.argv
    sys.argv = argv
    try:
        return mod.main()
    finally:
        sys.argv = old


@pytest.fixture
def evaluated(run):
    mod = _load_cli("evaluate")
    assert _run_cli(mod, ["evaluate.py", "--config", "local_cpu", "--out", str(run), "--task", "A"]) == 0
    return run, json.loads((run / "summary.json").read_text(encoding="utf-8"))


def test_every_model_is_scored_on_identical_rows(evaluated):
    _, summary = evaluated
    block = summary["tasks"]["A"]
    for split, info in block["identical_rows"].items():
        assert info["all_models_same_rows"], split
    assert block["rows_per_split"]["id_val"] == 100


def test_every_reported_score_carries_a_bootstrap_interval(evaluated):
    _, summary = evaluated
    for model, mb in summary["tasks"]["A"]["models"].items():
        for split, block in mb["splits"].items():
            ci = block["macro_f1_ci"]
            assert ci["lo"] <= ci["point"] <= ci["hi"], (model, split)


def test_paired_comparison_of_two_similar_models_is_a_tie(evaluated):
    _, summary = evaluated
    entry = summary["tasks"]["A"]["paired"]["gpad_vs_control"]
    # gpad and control were generated with almost the same accuracy: the honest verdict is "tie"
    assert entry["verdict"] == "no significant difference"
    assert not entry["significant_on_ood"]
    assert summary["readme_branch"]["branch"] == "no_gain_beyond_capacity"
    for split, v in entry["splits"].items():
        assert "mcnemar" in v and 0.0 <= v["mcnemar"]["p_value"] <= 1.0


def test_temperature_is_fitted_on_id_val_and_reported(evaluated):
    _, summary = evaluated
    gpad = summary["tasks"]["A"]["models"]["gpad"]
    assert gpad["temperature_id_val"] > 0
    assert "ece_temp_scaled" in gpad["splits"]["ood_lang"]


def test_analysis_produces_the_blocks_the_readme_quotes(evaluated):
    run, _ = evaluated
    mod = _load_cli("analyze")
    assert _run_cli(mod, ["analyze.py", "--config", "local_cpu", "--out", str(run)]) == 0
    analysis = json.loads((run / "analysis.json").read_text(encoding="utf-8"))
    block = analysis["tasks"]["A"]
    assert block["shortcut"]["perturbations"]["gpad"]["stripped_drop"] > 0      # stress test bites
    assert "length_only" in block["shortcut"]["shortcut_ladder"]
    assert block["calibration"]["ece"]["id_val"]["after_temperature"] >= 0
    th = block["threshold"]
    assert 0 < th["threshold"] < 1 and th["cost_ratio_fp_to_fn"] == 5.0
    assert th["splits"]["ood_lang"]["cost_optimal"]["false_accusation_rate"] >= 0
    assert (run / "figures").glob("*.png")
    assert (run / "error_analysis_taska.md").exists()


def test_report_block_renders_from_the_generated_json(evaluated):
    run, summary = evaluated
    mod = _load_cli("analyze")
    _run_cli(mod, ["analyze.py", "--config", "local_cpu", "--out", str(run)])
    rep = _load_cli("make_report")
    assert _run_cli(rep, ["make_report.py", "--out", str(run), "--readme", str(run / "README.md")]) == 0
    text = (run / "summary.md").read_text(encoding="utf-8")
    assert "A7 capacity control" in text and "95% bootstrap CI" in text
    assert "macro-F1" in text
