"""Checkpoint round-trip and the LoRA variant.

Both paths were previously untested (`docs/dev/OLD_CODE_AUDIT.md` flags the reload path): the demo loads a
checkpoint with `GPADClassifier.load`, and the PEFT comparison is a README row. Neither downloads
anything here - the tiny random encoder is saved and rebuilt from its own config.
"""
from __future__ import annotations

import json

import pytest

from codedet.config import Config
from codedet.features.style import STYLE_DIM
from codedet.models.gpad import GPADClassifier, GPADConfig, count_params
from codedet.models.lora import LoRALinear, build_lora_classifier

torch = pytest.importorskip("torch")


@pytest.fixture
def cfg():
    return Config({"preset": "test", "seed": 0, "model": {"entropy_alpha": 0.2}, "lora": {"r": 4, "alpha": 8}})


def _inputs(batch=2, length=10, style=True):
    kw = {"input_ids": torch.randint(5, 90, (batch, length)), "attention_mask": torch.ones(batch, length, dtype=torch.long)}
    if style:
        kw["style"] = torch.randn(batch, STYLE_DIM)
    return kw


def test_save_load_roundtrip_reproduces_logits(tiny_encoder, tmp_path):
    gcfg = GPADConfig(num_labels=2, style_in_dim=STYLE_DIM, num_gpad_layers=1, entropy_alpha=0.2,
                      classifier_hidden=32)
    model = GPADClassifier(gcfg, encoder=tiny_encoder).eval()
    kw = _inputs()
    with torch.no_grad():
        before = model(**kw)["logits"]
    model.save(str(tmp_path / "ckpt"))
    saved = json.loads((tmp_path / "ckpt" / "gpad_config.json").read_text())
    assert saved["entropy_alpha"] == 0.2 and saved["num_gpad_layers"] == 1

    reloaded = GPADClassifier.load(str(tmp_path / "ckpt"))
    with torch.no_grad():
        after = reloaded(**kw)["logits"]
    assert torch.allclose(before, after, atol=1e-5)


def test_loaded_model_is_in_eval_mode(tiny_encoder, tmp_path):
    GPADClassifier(GPADConfig(num_labels=2, style_in_dim=STYLE_DIM, classifier_hidden=32),
                   encoder=tiny_encoder).save(str(tmp_path / "c"))
    assert not GPADClassifier.load(str(tmp_path / "c")).training


def test_lora_trains_far_fewer_parameters_than_full_fine_tuning(cfg, tiny_encoder):
    import copy

    full = GPADClassifier(GPADConfig(num_labels=2, style_in_dim=STYLE_DIM, num_gpad_layers=0,
                                     use_style=False, classifier_hidden=32),
                          encoder=copy.deepcopy(tiny_encoder))
    lora, info = build_lora_classifier(
        GPADConfig(num_labels=2, style_in_dim=STYLE_DIM, num_gpad_layers=0, use_style=False,
                   classifier_hidden=32),
        encoder=copy.deepcopy(tiny_encoder), r=4, alpha=8)
    assert info["backend"] in ("peft", "fallback")
    enc_trainable = sum(p.numel() for p in lora.encoder.parameters() if p.requires_grad)
    enc_total = sum(p.numel() for p in lora.encoder.parameters())
    assert enc_trainable < 0.5 * enc_total, "the encoder should be mostly frozen"
    assert count_params(lora)["trainable_M"] < count_params(full)["trainable_M"]
    head_trainable = all(p.requires_grad for p in lora.head.parameters())
    assert head_trainable, "the classifier head must stay trainable"


def test_lora_forward_and_backward_work(cfg, tiny_encoder):
    lora, _ = build_lora_classifier(
        GPADConfig(num_labels=2, style_in_dim=STYLE_DIM, num_gpad_layers=0, use_style=False,
                   classifier_hidden=32), encoder=tiny_encoder, r=4, alpha=8)
    kw = _inputs(style=False)
    out = lora(**kw, labels=torch.tensor([0, 1]))
    out["loss"].backward()
    grads = [p.grad is not None for p in lora.head.parameters()]
    assert all(grads) and out["logits"].shape == (2, 2)


def test_fallback_lora_layer_starts_as_identity():
    base = torch.nn.Linear(8, 8)
    wrapped = LoRALinear(base, r=4, alpha=8, dropout=0.0)
    x = torch.randn(3, 8)
    assert torch.allclose(base(x), wrapped(x), atol=1e-6)   # B is zero-initialised
    assert not any(p.requires_grad for p in wrapped.base.parameters())
