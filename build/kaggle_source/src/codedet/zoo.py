"""One place that maps a model NAME to a configuration, so every ablation is a table row and not a
near-duplicate script (the old repo had ~25 of those).

The names matter for the central claim of the project:

    codebert            plain encoder + MLP head (no extra block)        <- the old baseline
    control             extra block, entropy bias OFF, no style fusion   <- CAPACITY CONTROL
    gpad                extra block, entropy bias ON, style fusion ON    <- the project idea
    gpad_entropy_only   extra block, bias ON, no style
    gpad_style_only     extra block, bias OFF, style ON
    gpad_sign_neg       bias ON with the sign flipped (boosts HIGH-entropy keys)
    lora                frozen encoder + LoRA adapters + head

`gpad` adds parameters as well as a mechanism, so a gain over `codebert` proves nothing on its own:
only gpad - control isolates the entropy bias. That comparison drives the README branch (PLAN A.5).
"""
from __future__ import annotations

from .features import STYLE_DIM
from .lm_features import LM_DIM
from .models.gpad import GPADClassifier, GPADConfig, count_params

NEURAL_MODELS = ("codebert", "codebert_weighted", "control", "gpad", "gpad_both", "gpad_entropy_only",
                 "gpad_style_only", "gpad_sign_neg", "lora")

# (num_gpad_layers, entropy_alpha_on, use_style, entropy_sign, weighted_loss_default)
_SPECS = {
    "codebert":          dict(layers=0, entropy=False, style=False, sign=1.0, weighted=False),
    "codebert_weighted": dict(layers=0, entropy=False, style=False, sign=1.0, weighted=True),
    "control":           dict(layers=1, entropy=False, style=False, sign=1.0, weighted=False),
    "gpad":              dict(layers=1, entropy=True, style=True, sign=1.0, weighted=False),
    "gpad_both":         dict(layers=1, entropy=True, style=True, sign=1.0, weighted=False),
    "gpad_entropy_only": dict(layers=1, entropy=True, style=False, sign=1.0, weighted=False),
    "gpad_style_only":   dict(layers=1, entropy=False, style=True, sign=1.0, weighted=False),
    "gpad_sign_neg":     dict(layers=1, entropy=True, style=True, sign=-1.0, weighted=False),
    "lora":              dict(layers=0, entropy=False, style=False, sign=1.0, weighted=False),
}


def is_neural(name: str) -> bool:
    return name in _SPECS


def spec(name: str) -> dict:
    if name not in _SPECS:
        raise ValueError(f"unknown neural model {name!r}; known: {sorted(_SPECS)}")
    return dict(_SPECS[name])


def uses_style(name: str) -> bool:
    return bool(spec(name)["style"])


def gpad_config(name: str, cfg, num_labels: int, with_lm: bool = False) -> GPADConfig:
    s = spec(name)
    style_in = STYLE_DIM + (LM_DIM if with_lm else 0)
    return GPADConfig(
        base_model=str(cfg.get("model.base", "microsoft/codebert-base")),
        num_labels=num_labels,
        style_in_dim=style_in,
        style_dim=int(cfg.get("model.style_dim", 32)),
        num_gpad_layers=int(s["layers"]),
        entropy_alpha=float(cfg.get("model.entropy_alpha", 0.15)) if s["entropy"] else 0.0,
        entropy_sign=float(s["sign"]),
        use_style=bool(s["style"]),
        hidden_dropout=float(cfg.get("model.hidden_dropout", 0.1)),
        attn_dropout=float(cfg.get("model.attn_dropout", 0.1)),
        classifier_hidden=int(cfg.get("model.classifier_hidden", 384)),
    )


def loss_kind(name: str, task: str, cfg) -> str:
    """Task B is ~88% human: weighted CE by default there, plain CE for the balanced binary task."""
    s = spec(name)
    if task.upper() == "B":
        return str(cfg.get("train.loss_task_b", "weighted_ce"))
    return "weighted_ce" if s["weighted"] else str(cfg.get("train.loss_task_a", "ce"))


def build(name: str, cfg, num_labels: int, encoder=None, with_lm: bool = False):
    """Returns (model, info). `encoder=None` downloads the base model; tests inject a tiny one."""
    gcfg = gpad_config(name, cfg, num_labels, with_lm=with_lm)
    info = {"model": name, "gpad_config": gcfg.__dict__.copy(), "with_lm_features": bool(with_lm)}
    if name == "lora":
        from .models.lora import build_lora_classifier

        model, lora_info = build_lora_classifier(gcfg, encoder=encoder,
                                                 r=int(cfg.get("lora.r", 8)),
                                                 alpha=int(cfg.get("lora.alpha", 16)))
        info["lora"] = lora_info
    else:
        model = GPADClassifier(gcfg, encoder=encoder)
    info["params"] = count_params(model)
    return model, info
