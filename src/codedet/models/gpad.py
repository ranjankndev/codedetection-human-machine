"""GPAD, cleaned. Ported from legacy/taska/gpad_v2_model.py.

What the mechanism REALLY is (the old README described something else):
  For every token we compute the entropy of a *learned softmax over key self-similarity* (`AttnEntropy`),
  centre it per sequence, squash with a sigmoid and ADD alpha * sign * that value to the pre-softmax attention
  scores of every query for that key. => a content-driven additive per-key attention bias (same family as
  ALiBi / relative-position biases). With sign=+1, LOW-entropy tokens are boosted.
It is NOT token-level predictive entropy (see lm_features.py for that).

Changes vs legacy: padding-aware entropy, explicit `sign`, precomputed style input, no PreTrainedModel subclass,
encoder injectable (tests use a tiny random encoder, no downloads), losses pluggable.

Ablations are config flags:
  plain CodeBERT + MLP head : num_gpad_layers=0
  capacity CONTROL          : num_gpad_layers=1, entropy_alpha=0.0, use_style=False   (same extra block, bias off)
  entropy-only / style-only / both : entropy_alpha / use_style
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPADConfig:
    base_model: str = "microsoft/codebert-base"
    num_labels: int = 2
    style_in_dim: int = 16          # must equal features.STYLE_DIM (+ LM_DIM if LM features are concatenated)
    style_dim: int = 32
    num_gpad_layers: int = 1
    entropy_alpha: float = 0.15     # 0.0 => bias off (capacity control)
    entropy_sign: float = 1.0       # +1 boosts low-entropy tokens, -1 boosts high-entropy tokens
    use_style: bool = True
    hidden_dropout: float = 0.1
    attn_dropout: float = 0.1
    classifier_hidden: int = 384


class AttnEntropy(nn.Module):
    """Per-token entropy of a learned self-similarity softmax; returns a bias in (0,1) per (batch, head, key)."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.q = nn.Linear(head_dim, head_dim)
        self.k = nn.Linear(head_dim, head_dim)
        self.scale = head_dim ** -0.5

    def forward(self, keys: torch.Tensor, key_mask: torch.Tensor | None):
        scores = torch.matmul(self.q(keys), self.k(keys).transpose(-2, -1)) * self.scale   # [B,H,S,S]
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        p = F.softmax(scores.float(), dim=-1)
        ent = -(p * torch.log(p + 1e-9)).sum(-1)                                            # [B,H,S]
        if key_mask is not None:
            m = key_mask[:, None, :].float()
            mean = (ent * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp(min=1.0)
        else:
            mean = ent.mean(-1, keepdim=True)
        return torch.sigmoid(mean - ent).to(keys.dtype)


class EntropyWeightedAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, alpha: float, sign: float, dropout: float):
        super().__init__()
        assert hidden % heads == 0
        self.hidden, self.heads, self.dim = hidden, heads, hidden // heads
        self.scale, self.alpha, self.sign = self.dim ** -0.5, alpha, sign
        self.q, self.k, self.v, self.o = (nn.Linear(hidden, hidden) for _ in range(4))
        self.entropy = AttnEntropy(self.dim) if alpha != 0.0 else None   # no params in the control model
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, mask):
        b, s, _ = x.shape
        split = lambda t: t.view(b, s, self.heads, self.dim).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale                          # [B,H,S,S]
        key_mask = mask.bool() if mask is not None else None
        if self.entropy is not None:
            bias = self.entropy(k, key_mask)                                                 # [B,H,S] per key
            scores = scores + self.sign * self.alpha * bias.unsqueeze(2)
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = self.drop(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, self.hidden)
        return self.norm(x + self.drop(self.o(out)))


class StyleFusion(nn.Module):
    """h + gate(h, s) * W s  (s = projected style vector, broadcast over tokens)."""

    def __init__(self, hidden: int, style_dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(style_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1))
        self.gate = nn.Sequential(nn.Linear(hidden + style_dim, hidden), nn.Sigmoid())

    def forward(self, h, s):
        L = h.size(1)
        s_tok = s.unsqueeze(1).expand(-1, L, -1)
        return h + self.gate(torch.cat([h, s_tok], -1)) * self.proj(s).unsqueeze(1)


class GPADLayer(nn.Module):
    def __init__(self, cfg: GPADConfig, hidden: int, heads: int, inter: int):
        super().__init__()
        self.attn = EntropyWeightedAttention(hidden, heads, cfg.entropy_alpha, cfg.entropy_sign, cfg.attn_dropout)
        self.style = StyleFusion(hidden, cfg.style_dim) if cfg.use_style else None
        self.ff = nn.Sequential(nn.Linear(hidden, inter), nn.GELU(), nn.Dropout(cfg.hidden_dropout),
                                nn.Linear(inter, hidden), nn.Dropout(cfg.hidden_dropout))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, s, mask):
        h = self.attn(h, mask)
        if self.style is not None and s is not None:
            h = self.style(h, s)
        return self.norm(h + self.ff(h))


class GPADClassifier(nn.Module):
    def __init__(self, cfg: GPADConfig, encoder: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        if encoder is None:
            from transformers import AutoModel

            encoder = AutoModel.from_pretrained(cfg.base_model)
        self.encoder = encoder
        ec = encoder.config
        hidden, heads, inter = ec.hidden_size, ec.num_attention_heads, ec.intermediate_size
        self.style_proj = (
            nn.Sequential(nn.Linear(cfg.style_in_dim, cfg.style_dim), nn.LayerNorm(cfg.style_dim), nn.GELU(), nn.Dropout(0.1))
            if cfg.use_style and cfg.num_gpad_layers > 0 else None
        )
        self.layers = nn.ModuleList([GPADLayer(cfg, hidden, heads, inter) for _ in range(cfg.num_gpad_layers)])
        ch = cfg.classifier_hidden
        self.head = nn.Sequential(
            nn.Linear(hidden, ch), nn.LayerNorm(ch), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(ch, ch // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(ch // 2, cfg.num_labels),
        )
        self.loss_fn: nn.Module = nn.CrossEntropyLoss()

    def set_loss(self, loss_fn: nn.Module):
        self.loss_fn = loss_fn
        return self

    def forward(self, input_ids, attention_mask, style=None, labels=None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        s = self.style_proj(style) if (self.style_proj is not None and style is not None) else None
        for layer in self.layers:
            h = layer(h, s, attention_mask)
        logits = self.head(h[:, 0])
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}

    # ---- persistence (plain torch; no PreTrainedModel machinery)
    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_dir, "gpad.pt"))
        with open(os.path.join(out_dir, "gpad_config.json"), "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)
        self.encoder.config.save_pretrained(out_dir)

    @classmethod
    def load(cls, in_dir: str, map_location="cpu"):
        from transformers import AutoConfig, AutoModel

        with open(os.path.join(in_dir, "gpad_config.json")) as f:
            cfg = GPADConfig(**json.load(f))
        enc = AutoModel.from_config(AutoConfig.from_pretrained(in_dir))
        model = cls(cfg, encoder=enc)
        model.load_state_dict(torch.load(os.path.join(in_dir, "gpad.pt"), map_location=map_location))
        return model.eval()


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_M": round(total / 1e6, 2), "trainable_M": round(train / 1e6, 2)}
