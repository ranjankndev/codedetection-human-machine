"""Config presets and the orchestrator's stage plan.

Cheap guards against the two mistakes that cost the most time in practice: a typo in a YAML preset that
only surfaces 40 minutes into a GPU session, and a stage plan that silently drops a required model.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from codedet.classical import CLASSICAL_MODELS
from codedet.config import SPLITS, CONFIG_DIR, Config, RunPaths
from codedet.zoo import NEURAL_MODELS, is_neural

PRESETS = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
EXTRA_STAGE_MODELS = {"stack"}        # produced by run_stack.py, not by the classical or neural stages


def _load_run_all():
    """Import scripts/run_all.py as a module (it is a CLI, not part of the package)."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_all.py"
    spec = importlib.util.spec_from_file_location("run_all_cli", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_all_cli"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_presets_exist():
    assert {"smoke", "lite", "local_cpu"} <= set(PRESETS)


@pytest.mark.parametrize("preset", PRESETS)
def test_preset_is_loadable_and_complete(preset):
    cfg = Config.load(preset)
    assert cfg.preset == preset
    assert cfg.seed and cfg.max_len > 0 and cfg.n_boot > 0
    assert cfg.head_tokens <= cfg.max_len
    assert float(cfg.get("budget_minutes", 0)) > 0
    assert cfg.get("model.base"), "every preset must name its base encoder"
    for task in ("A", "B"):
        assert cfg.models(task), f"{preset} lists no models for task {task}"


@pytest.mark.parametrize("preset", PRESETS)
def test_preset_model_names_are_known(preset):
    cfg = Config.load(preset)
    known = set(CLASSICAL_MODELS) | set(NEURAL_MODELS) | EXTRA_STAGE_MODELS
    for task in ("A", "B"):
        unknown = [m for m in cfg.models(task) if m not in known]
        assert not unknown, f"{preset} task {task} lists unknown model(s): {unknown}"


@pytest.mark.parametrize("preset", PRESETS)
def test_task_b_caps_humans_and_task_a_holds_out_a_language(preset):
    cfg = Config.load(preset)
    cap = float(cfg.get("data.task_b.human_cap", 1.0))
    assert 0 < cap <= 0.5, f"{preset}: task B must cap the human share of train (got {cap})"
    assert cfg.get("data.task_a.holdout_language"), f"{preset}: task A needs a held-out language"


def test_stage_plan_covers_every_configured_model(tmp_path):
    mod = _load_run_all()
    cfg = Config.load("local_cpu")
    args = types.SimpleNamespace(config_path="local_cpu", out=str(tmp_path), cpu_only=True,
                                 data="fixtures", readme=str(tmp_path / "README.md"))
    stages = mod.build_stages(cfg, args)
    names = [s.name for s in stages]
    for task in ("A", "B"):
        for model in cfg.models(task):
            if is_neural(model):
                assert f"train_{task}_{model}" in names
    assert names[:2] == ["discover", "splits"]
    assert names.index("report") == len(names) - 2 and names[-1] == "slides"
    assert "evaluate" in names and "analyze" in names


def test_required_and_optional_stages_match_the_plan(tmp_path):
    mod = _load_run_all()
    cfg = Config.load("local_cpu")
    args = types.SimpleNamespace(config_path="local_cpu", out=str(tmp_path), cpu_only=True,
                                 data="fixtures", readme=str(tmp_path / "README.md"))
    by_name = {s.name: s for s in mod.build_stages(cfg, args)}
    # the capacity control and the two models it is compared against can never be optional:
    # without them the project's central claim cannot be evaluated at all
    for stage in ("train_A_codebert", "train_A_control", "train_A_gpad", "evaluate", "analyze", "report"):
        assert by_name[stage].required, f"{stage} must be a required stage"
    assert not by_name["train_A_lora"].required          # PEFT comparison is a nice-to-have
    assert by_name["train_A_gpad"].retry_cheaper         # neural stages degrade before they fail


def test_run_paths_layout(tmp_path):
    p = RunPaths.make(tmp_path / "run")
    assert p.split_file("A", "id_val").name == "id_val.parquet"
    assert p.pred_file("B", "gpad", "ood_gen").parts[-2:] == ("gpad", "probs_ood_gen.npz")
    assert p.metrics_file("A", "codebert").name == "metrics.json"
    for d in (p.data, p.preds, p.ckpt, p.cache, p.figures, p.done):
        assert d.exists()


def test_split_names_are_the_contract():
    assert SPLITS["A"][:3] == ["train", "id_val", "ood_lang"]
    assert SPLITS["B"][:3] == ["train", "id_val", "ood_gen"]
    assert "official_val" in SPLITS["A"] and "official_val" in SPLITS["B"]
