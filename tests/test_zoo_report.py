"""The ablation table (zoo) and the report renderer - the two places where a silent change would
turn an honest claim into a dishonest one."""
from __future__ import annotations

import numpy as np
import pytest

from codedet.config import Config
from codedet.features import STYLE_DIM
from codedet.lm_features import LM_DIM
from codedet.report import (GENERATED_BEGIN, GENERATED_END, fmt_ci, render_results_block, splice_generated,
                            verdict_lines)
from codedet.zoo import build, gpad_config, loss_kind, uses_style

torch = pytest.importorskip("torch")


@pytest.fixture
def cfg():
    return Config({"preset": "test", "seed": 0, "model": {"entropy_alpha": 0.2}})


def test_control_has_the_extra_block_but_no_entropy_or_style(cfg):
    c = gpad_config("control", cfg, 2)
    assert c.num_gpad_layers == 1 and c.entropy_alpha == 0.0 and c.use_style is False


def test_gpad_turns_both_mechanisms_on(cfg):
    c = gpad_config("gpad", cfg, 2)
    assert c.num_gpad_layers == 1 and c.entropy_alpha == 0.2 and c.use_style is True


def test_codebert_has_no_extra_block(cfg):
    assert gpad_config("codebert", cfg, 2).num_gpad_layers == 0


def test_sign_ablation_flips_only_the_sign(cfg):
    a, b = gpad_config("gpad", cfg, 2), gpad_config("gpad_sign_neg", cfg, 2)
    assert b.entropy_sign == -1.0 and a.entropy_sign == 1.0
    assert (b.entropy_alpha, b.use_style, b.num_gpad_layers) == (a.entropy_alpha, a.use_style, a.num_gpad_layers)


def test_style_input_width_matches_the_feature_modules(cfg):
    assert gpad_config("gpad", cfg, 2).style_in_dim == STYLE_DIM
    assert gpad_config("gpad", cfg, 2, with_lm=True).style_in_dim == STYLE_DIM + LM_DIM


def test_task_b_defaults_to_a_class_aware_loss(cfg):
    assert loss_kind("codebert", "B", cfg) == "weighted_ce"
    assert loss_kind("codebert", "A", cfg) == "ce"


def test_uses_style_matches_the_spec():
    assert uses_style("gpad") and uses_style("gpad_style_only")
    assert not uses_style("control") and not uses_style("codebert")


def test_build_control_and_gpad_differ_only_by_the_mechanism(cfg, tiny_encoder):
    import copy

    ctrl, _ = build("control", cfg, 2, encoder=copy.deepcopy(tiny_encoder))
    gpad, _ = build("gpad", cfg, 2, encoder=copy.deepcopy(tiny_encoder))
    assert ctrl.layers[0].attn.entropy is None          # control carries no entropy parameters
    assert gpad.layers[0].attn.entropy is not None
    ids = torch.randint(0, 90, (2, 12))
    mask = torch.ones_like(ids)
    out = ctrl(input_ids=ids, attention_mask=mask)
    assert out["logits"].shape == (2, 2)
    out_g = gpad(input_ids=ids, attention_mask=mask, style=torch.randn(2, STYLE_DIM))
    assert out_g["logits"].shape == (2, 2)


# ---------------------------------------------------------------- report rendering
def _summary(diff_lo=-0.05, diff_hi=0.05):
    ci = lambda p: {"point": p, "lo": p - 0.03, "hi": p + 0.03}
    def mblock(f1):
        return {"splits": {s: {"macro_f1": f1, "accuracy": f1, "balanced_accuracy": f1,
                               "macro_f1_ci": ci(f1), "per_class": {"human": {"precision": f1, "recall": f1,
                                                                             "f1": f1, "support": 10}},
                               "confusion_matrix": [[5, 1], [1, 5]]}
                           for s in ("id_val", "ood_lang")}, "seeds": 1, "seed_macro_f1_std": 0.0}
    return {
        "preset": "test", "seed": 0, "n_boot": 10,
        "tasks": {"A": {"num_labels": 2, "label_names": ["human", "machine"],
                        "rows_per_split": {"id_val": 10, "ood_lang": 10},
                        "models": {"gpad": mblock(0.8), "control": mblock(0.79), "length_only": mblock(0.6)},
                        "paired": {"gpad_vs_control": {
                            "question": "q", "verdict": "v", "significant_splits": [],
                            "significant_on_ood": diff_lo > 0,
                            "splits": {"ood_lang": {"n": 10,
                                                    "bootstrap": {"diff": 0.01, "lo": diff_lo, "hi": diff_hi,
                                                                  "excludes_zero": diff_lo > 0},
                                                    "mcnemar": {"p_value": 0.4}}}}}}},
        "readme_branch": {"branch": "no_gain_beyond_capacity" if diff_lo <= 0 else "entropy_helps"},
    }


def test_fmt_ci_formats_point_and_interval():
    assert fmt_ci({"macro_f1": 0.5, "macro_f1_ci": {"point": 0.5, "lo": 0.4, "hi": 0.6}}) == "0.500 [0.400, 0.600]"
    assert fmt_ci(None) == "-"


def test_generated_block_is_delimited_and_contains_the_table():
    block = render_results_block(_summary(), {"tasks": {}}, {})
    assert block.startswith(GENERATED_BEGIN) and block.rstrip().endswith(GENERATED_END)
    assert "A6 GPAD" in block and "A7 capacity control" in block


def test_verdict_follows_the_confidence_interval_not_the_point_estimate():
    tie = " ".join(verdict_lines(_summary(-0.05, 0.05), {"tasks": {}}))
    assert "does not survive its control" in tie
    win = " ".join(verdict_lines(_summary(0.02, 0.09), {"tasks": {}}))
    assert "survives its control" in win


def test_splice_replaces_only_the_generated_region():
    doc = f"intro\n\n{GENERATED_BEGIN}\nold\n{GENERATED_END}\n\noutro\n"
    new = splice_generated(doc, f"{GENERATED_BEGIN}\nnew\n{GENERATED_END}")
    assert "intro" in new and "outro" in new and "new" in new and "old" not in new
