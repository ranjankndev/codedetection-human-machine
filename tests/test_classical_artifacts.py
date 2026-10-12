"""Classical baselines, the prediction-file contract, and the perturbations used by the stress test."""
from __future__ import annotations

import numpy as np
import pytest

from codedet.artifacts import load_preds, probs_as_logits, save_preds
from codedet.classical import ClassicalModel, length_matrix, permutation_importance_gbdt, top_ngrams
from codedet.eval import macro_f1_fast, softmax
from codedet.fixtures import make_fixture
from codedet.perturb import normalize_whitespace, strip_and_normalize, strip_comments


@pytest.fixture
def data():
    df = make_fixture(160, seed=7)
    return df["code"].tolist(), df["label"].values


def test_majority_predicts_the_train_prior(data):
    codes, y = data
    m = ClassicalModel("majority", 2).fit(codes, y)
    p = m.predict_proba(codes)
    assert p.shape == (len(codes), 2)
    assert np.allclose(p[0], p[1])
    assert p.argmax(1)[0] == np.bincount(y).argmax()


@pytest.mark.parametrize("name", ["length_only", "tfidf_lr", "style_gbdt"])
def test_classical_models_learn_the_synthetic_signal(data, name):
    codes, y = data
    m = ClassicalModel(name, 2, seed=0).fit(codes, y)
    p = m.predict_proba(codes)
    assert p.shape == (len(codes), 2)
    assert np.allclose(p.sum(1), 1.0, atol=1e-6)
    assert macro_f1_fast(y, p.argmax(1), 2) > 0.7     # fixtures are separable on purpose


def test_probability_columns_are_full_width_when_a_class_is_missing():
    codes = [c for c in make_fixture(40, seed=1, human_share=1.0)["code"]]
    y = np.zeros(len(codes), dtype=int)
    m = ClassicalModel("length_only", 5, seed=0).fit(codes, y)
    p = m.predict_proba(codes)
    assert p.shape[1] == 5                            # macro-F1 over range(K) needs every column


def test_length_matrix_is_content_free():
    a = length_matrix(["def f():\n  return 1\n"])
    b = length_matrix(["def g():\n  return 2\n"])
    assert np.allclose(a, b, atol=1e-6)                # same shape of code, different content -> same features


def test_permutation_importance_names_real_features(data):
    codes, y = data
    m = ClassicalModel("style_gbdt", 2, seed=0).fit(codes, y)
    imp = permutation_importance_gbdt(m, codes, y, n_repeats=2)
    assert imp and {d["feature"] for d in imp} <= set(m.feature_names)
    assert imp == sorted(imp, key=lambda d: -d["macro_f1_drop"])


def test_top_ngrams_only_for_binary_tfidf(data):
    codes, y = data
    m = ClassicalModel("tfidf_lr", 2, seed=0).fit(codes, y)
    ng = top_ngrams(m, k=5)
    assert len(ng["machine_indicative"]) == 5 and len(ng["human_indicative"]) == 5


def test_prediction_roundtrip_and_softmax_recovery(tmp_path):
    probs = np.array([[0.2, 0.8], [0.7, 0.3]])
    f = save_preds(tmp_path / "probs_id_val.npz", probs_as_logits(probs), [1, 0], ["r1", "r2"])
    d = load_preds(f)
    assert list(d["ids"]) == ["r1", "r2"]
    assert np.allclose(softmax(d["logits"]), probs, atol=1e-5)


def test_save_preds_rejects_ragged_arrays(tmp_path):
    with pytest.raises(ValueError):
        save_preds(tmp_path / "x.npz", np.zeros((3, 2)), [0, 1], ["a", "b"])


def test_strip_comments_removes_python_and_c_style():
    py = "# header\ndef f():\n    '''doc'''\n    return 1  # trailing\n"
    out = strip_comments(py, "Python")
    assert "#" not in out and "doc" not in out and "return 1" in out
    c = "/* block */\nint f(){ // line\n return 1;}\n"
    out_c = strip_comments(c, "C++")
    assert "block" not in out_c and "line" not in out_c and "return 1;" in out_c


def test_normalize_whitespace_collapses_indent_and_blanks():
    code = "def f():\n\n\t\treturn    1   \n"
    out = normalize_whitespace(code, "Python")
    assert "\t" not in out and "\n\n" not in out and out.endswith("return 1")


def test_strip_and_normalize_is_idempotent():
    code = "// c\nint  f(){\n\n   return  1;}\n"
    once = strip_and_normalize(code, "Java")
    assert strip_and_normalize(once, "Java") == once
