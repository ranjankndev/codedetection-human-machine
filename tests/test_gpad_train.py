import numpy as np
import pytest
import torch
from codedet.models.gpad import GPADClassifier, GPADConfig, count_params
from codedet.datasets import CodeDataset, make_collate
from codedet.features import style_matrix, StyleScaler, STYLE_DIM
from codedet.fixtures import make_fixture
from codedet.losses import make_loss
from codedet.trainloop import fit, TrainConfig, predict_logits, run_with_oom_ladder
from codedet.eval import macro_f1_fast


def _batch(B=3, S=12, V=100):
    torch.manual_seed(0)
    ids = torch.randint(5, V, (B, S))
    mask = torch.ones(B, S, dtype=torch.long); mask[1, 8:] = 0
    return ids, mask, torch.randn(B, STYLE_DIM)


def _model(enc, **kw):
    cfg = GPADConfig(num_labels=2, style_in_dim=STYLE_DIM, style_dim=8, classifier_hidden=16, **kw)
    return GPADClassifier(cfg, encoder=enc).eval()


@pytest.mark.parametrize("kw", [dict(), dict(num_gpad_layers=0), dict(entropy_alpha=0.0, use_style=False)])
def test_forward_shapes(tiny_encoder, kw):
    ids, mask, st = _batch()
    out = _model(tiny_encoder, **kw)(ids, mask, st, labels=torch.tensor([0, 1, 0]))
    assert out["logits"].shape == (3, 2) and out["loss"].isfinite()


def test_padding_invariance(tiny_encoder):
    ids, mask, st = _batch()
    m = _model(tiny_encoder)
    a = m(ids, mask, st)["logits"]
    ids2 = ids.clone(); ids2[1, 8:] = 7                    # change tokens at masked positions only
    b = m(ids2, mask, st)["logits"]
    assert torch.allclose(a, b, atol=1e-4)


def test_ablation_flags_change_output(tiny_encoder):
    ids, mask, st = _batch()
    torch.manual_seed(1); on = _model(tiny_encoder, entropy_alpha=2.0)
    torch.manual_seed(1); off = _model(tiny_encoder, entropy_alpha=0.0)
    off.load_state_dict({k: v for k, v in on.state_dict().items() if k in off.state_dict()}, strict=False)
    assert not torch.allclose(on(ids, mask, st)["logits"], off(ids, mask, st)["logits"], atol=1e-6)
    neg = _model(tiny_encoder, entropy_alpha=2.0, entropy_sign=-1.0)
    neg.load_state_dict(on.state_dict())
    assert not torch.allclose(on(ids, mask, st)["logits"], neg(ids, mask, st)["logits"], atol=1e-6)
    assert count_params(off)["total_M"] <= count_params(on)["total_M"]


def test_end_to_end_tiny_training(tiny_encoder, tok):
    df = make_fixture(160, seed=3)
    tr, va = df.iloc[:120], df.iloc[120:]
    X = style_matrix(tr.code.tolist()); sc = StyleScaler().fit(X)
    ds_tr = CodeDataset(tr.code, tr.label, tok, 64, style=sc.transform(X))
    ds_va = CodeDataset(va.code, va.label, tok, 64, style=sc.transform(style_matrix(va.code.tolist())))
    model = GPADClassifier(GPADConfig(num_labels=2, style_in_dim=STYLE_DIM, style_dim=8, classifier_hidden=16),
                           encoder=tiny_encoder).set_loss(make_loss("weighted_ce", class_counts=[60, 60]))
    hist = fit(model, ds_tr, ds_va, make_collate(tok.pad_token_id),
               TrainConfig(epochs=4, batch_size=16, lr=5e-4, head_lr=2e-3, fp16=False, log_every=0), num_labels=2, device="cpu")
    logits, y = predict_logits(model, ds_va, make_collate(tok.pad_token_id), device="cpu", fp16=False)
    assert macro_f1_fast(y, logits.argmax(1), 2) > 0.7, hist
    model.save  # persistence API exists


def test_oom_ladder():
    calls = []

    def fn(batch_size, grad_accum, grad_checkpointing, max_len):
        calls.append(batch_size)
        if len(calls) < 3:
            raise RuntimeError("CUDA out of memory")
        return {"ok": True}
    res = run_with_oom_ladder(fn, dict(batch_size=32, grad_accum=1, grad_checkpointing=False, max_len=256))
    assert res["ok"] and res["degradation_rung"] == 2 and calls == [32, 16, 16]
