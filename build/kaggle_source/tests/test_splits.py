"""Splits: the leakage guarantee and the OOD construction are the things that must never silently break."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from codedet.config import Config, RunPaths
from codedet.data import dedup_exact
from codedet.fixtures import make_fixture
from codedet.splits import build, pick_holdout_generators, prepare_frame, row_ids


@pytest.fixture
def cfg():
    c = Config({"preset": "test", "seed": 0,
                "data": {"task_a": {"n_train": 60, "n_val": 20, "n_ood": 20, "holdout_language": "Java",
                                    "stratify_by": ["label", "language"]},
                         "task_b": {"n_train": 60, "n_val": 20, "n_ood": 20, "human_cap": 0.5,
                                    "n_unseen_families": 2}},
                "bootstrap": {"n_boot": 20}})
    return c


@pytest.fixture
def data_root(tmp_path):
    make_fixture(300, seed=0).to_parquet(tmp_path / "task_a_training_set.parquet")
    make_fixture(300, seed=1).to_parquet(tmp_path / "task_b_training_set.parquet")
    return str(tmp_path)


def test_row_ids_are_stable_and_unique():
    df = make_fixture(50, seed=3)
    a, b = row_ids(df, "A"), row_ids(df, "A")
    assert list(a) == list(b)
    assert a.nunique() == len(df)


def test_prepare_frame_task_a_polarity():
    df = make_fixture(60, seed=2)
    out, names, notes = prepare_frame(df, "A")
    assert names == ["human", "machine"]
    assert notes["label_vs_generator_agreement"] == 1.0
    assert set(out["y"]) <= {0, 1}
    # a flipped official label must be detected, not silently trusted
    flipped = df.copy()
    flipped["label"] = 1 - flipped["label"]
    _, _, n2 = prepare_frame(flipped, "A")
    assert n2["label_polarity_flipped"] is True


def test_prepare_frame_task_b_human_is_class_zero():
    out, names, _ = prepare_frame(make_fixture(120, seed=4), "B")
    assert names[0] == "human"
    assert (out.loc[out["generator"] == "Human", "y"] == 0).all()


def test_pick_holdout_generators_picks_one_per_family():
    out, _, _ = prepare_frame(make_fixture(400, seed=5), "B")
    chosen = pick_holdout_generators(out, 2, seed=0)
    assert 1 <= len(chosen) <= 2
    fams = {out.loc[out["generator"] == g, "family"].iloc[0] for g in chosen}
    assert len(fams) == len(chosen)          # one generator per family, never a whole family


def test_build_task_a_has_no_train_eval_overlap(cfg, data_root, tmp_path):
    paths = RunPaths.make(tmp_path / "run")
    rep = build("A", cfg, paths, data_root)
    train = pd.read_parquet(paths.split_file("A", "train"))
    ood = pd.read_parquet(paths.split_file("A", "ood_lang"))
    val = pd.read_parquet(paths.split_file("A", "id_val"))
    assert not set(train["row_id"]) & set(val["row_id"])
    assert not set(train["row_id"]) & set(ood["row_id"])
    assert (ood["language"] == "Java").all()            # the OOD split really is the held-out language
    assert "Java" not in set(train["language"])         # and the model never sees it in training
    assert rep["num_labels"] == 2
    assert rep["sizes"]["train"] > 0


def test_build_task_b_caps_human_share_in_train_only(cfg, data_root, tmp_path):
    paths = RunPaths.make(tmp_path / "run")
    rep = build("B", cfg, paths, data_root)
    train = pd.read_parquet(paths.split_file("B", "train"))
    held = set(g.lower() for g in rep["notes"]["holdout_generators"])
    assert train["generator"].str.lower().isin(held).sum() == 0
    human_share = (train["generator"] == "Human").mean()
    assert human_share <= 0.55 + 1e-6                    # cap applies to train
    assert rep["notes"]["human_cap"] == 0.5


def test_dedup_exact_removes_whitespace_variants():
    train = pd.DataFrame({"code": ["def f():\n  return 1", "def g():\n  return 2"]})
    other = pd.DataFrame({"code": ["def   f():\n       return    1"]})
    clean, removed = dedup_exact(train, [other])
    assert removed == 1 and len(clean) == 1
