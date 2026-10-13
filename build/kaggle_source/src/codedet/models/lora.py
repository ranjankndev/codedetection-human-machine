"""CodeBERT + LoRA, for the PEFT comparison (trainable parameters, time, F1).

LoRA: W + (alpha/r) * B A, with A [r, d_in], B [d_out, r] trainable and W frozen. Applied to the
attention query/value projections (the standard choice from the paper) while the classifier head stays
fully trainable - freezing a randomly initialised head would make the comparison meaningless.

`peft` is optional. If it is missing we fall back to a small, explicit LoRA implementation so the
ablation still runs on a machine with no extra installs; metrics record which path was taken.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .gpad import GPADClassifier, GPADConfig, count_params

TARGET_MODULES = ("query", "value")


class LoRALinear(nn.Module):
    """Fallback wrapper: frozen base linear + trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scaling = r, alpha / r
        self.a = nn.Parameter(torch.zeros(r, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))   # B stays zero => identity at step 0
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.drop(x) @ self.a.t() @ self.b.t() * self.scaling


def _inject_fallback(encoder: nn.Module, r: int, alpha: int) -> int:
    n = 0
    for module in encoder.modules():
        for name, child in list(module.named_children()):
            if name in TARGET_MODULES and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
                n += 1
    return n


def build_lora_classifier(cfg: GPADConfig, encoder: nn.Module | None = None, r: int = 8, alpha: int = 16,
                          dropout: float = 0.05) -> tuple[GPADClassifier, dict]:
    """Returns (model, info). Head + LoRA params train; the rest of the encoder is frozen."""
    model = GPADClassifier(cfg, encoder=encoder)
    info: dict = {"r": r, "alpha": alpha, "target_modules": list(TARGET_MODULES)}
    try:
        from peft import LoraConfig, get_peft_model

        lcfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                          target_modules=list(TARGET_MODULES))
        model.encoder = get_peft_model(model.encoder, lcfg)
        info["backend"] = "peft"
    except Exception as e:  # ImportError, or a peft/transformers version mismatch
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        info["backend"] = "fallback"
        info["peft_error"] = str(e)[:200]
        info["layers_wrapped"] = _inject_fallback(model.encoder, r, alpha)
    info["params"] = count_params(model)
    enc_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    info["encoder_trainable_M"] = round(enc_trainable / 1e6, 4)
    return model, info
