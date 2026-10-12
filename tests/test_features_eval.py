import numpy as np
import pytest
from codedet.features import stylometric_features, style_matrix, StyleScaler, STYLE_DIM
from codedet.eval import (macro_f1_fast, bootstrap_ci, paired_bootstrap_diff, mcnemar, ece, fit_temperature,
                          softmax, metrics_dict)
from codedet.fixtures import make_fixture


def test_style():
    f = stylometric_features("")
    assert f.shape == (STYLE_DIM,) and np.isfinite(f).all()
    X = style_matrix(make_fixture(40).code.tolist())
    s = StyleScaler().fit(X)
    Z = s.transform(X)
    assert abs(Z.mean()) < 1e-3
    s2 = StyleScaler.from_dict(s.to_dict())
    assert np.allclose(s2.transform(X), Z, atol=1e-5)


def test_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 5, 500), rng.integers(0, 5, 500)
    assert macro_f1_fast(y, p, 5) == pytest.approx(f1_score(y, p, average="macro", labels=range(5)), abs=1e-9)
    assert metrics_dict(y, p, 5)["macro_f1"] == pytest.approx(macro_f1_fast(y, p, 5), abs=1e-9)


def test_bootstrap_and_tests():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 600)
    good = np.where(rng.random(600) < 0.9, y, 1 - y)
    bad = np.where(rng.random(600) < 0.6, y, 1 - y)
    ci = bootstrap_ci(y, good, 2, n_boot=200)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    d = paired_bootstrap_diff(y, good, bad, 2, n_boot=200)
    assert d["excludes_zero"] and d["diff"] > 0
    assert mcnemar(y, good, bad)["p_value"] < 0.01
    assert mcnemar(y, good, good)["p_value"] == 1.0


def test_calibration():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    logits = np.stack([-(2 * y - 1) * 0.5, (2 * y - 1) * 0.5], 1) * 1.0
    flip = rng.random(2000) < 0.25
    y_noisy = np.where(flip, 1 - y, y)
    over = logits * 10                       # overconfident
    T = fit_temperature(over, y_noisy)
    assert T > 2
    assert ece(softmax(over, T), y_noisy) < ece(softmax(over), y_noisy)
