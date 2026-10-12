import numpy as np
import torch
from codedet.lm_features import lm_stats_from_logits, predictive_entropy, LM_DIM


def test_entropy_bounds():
    V = 50
    uniform = torch.zeros(3, V)
    assert torch.allclose(predictive_entropy(uniform), torch.full((3,), float(np.log(V))), atol=1e-4)
    peaked = torch.zeros(3, V); peaked[:, 0] = 50
    assert (predictive_entropy(peaked) < 1e-3).all()


def test_stats_respect_padding():
    torch.manual_seed(0)
    B, S, V = 2, 10, 30
    logits = torch.randn(B, S, V)
    ids = torch.randint(0, V, (B, S))
    mask = torch.ones(B, S, dtype=torch.long); mask[1, 6:] = 0
    a = lm_stats_from_logits(logits, ids, mask)
    ids2, logits2 = ids.clone(), logits.clone()
    ids2[1, 6:] = 0; logits2[1, 6:] = 123.0           # garbage at padded positions must not matter
    b = lm_stats_from_logits(logits2, ids2, mask)
    assert a.shape == (B, LM_DIM) and np.allclose(a, b, atol=1e-5) and np.isfinite(a).all()
