"""TRUE predictive-entropy / likelihood features from a frozen causal code LM (e.g. Qwen2.5-Coder-0.5B).

This replaces the three different (and mis-named) 'entropy' definitions in the old code:
  1. taskb:  softmax over the 768 hidden *dimensions* of each token (token_entropy(hidden_states))
  2. taska:  entropy of a learned softmax over key self-similarity (EntropyCalculator)
  3. README: 'token uncertainty' (not what the code computed)
Here entropy means H(p(. | x_<t)) of the LM's next-token distribution, and NLL is -log p(x_t | x_<t).
Machine-written code tends to have lower NLL and lower entropy (the DetectGPT / Binoculars family of ideas).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

LM_FEATURE_NAMES = [
    "nll_mean", "nll_std", "nll_max", "nll_p90", "ent_mean", "ent_std", "top1_hit", "nll_over_ent",
]
LM_DIM = len(LM_FEATURE_NAMES)


def predictive_entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def lm_stats_from_logits(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    """logits [B,S,V], input_ids [B,S], mask [B,S] (right padded). Returns [B, LM_DIM] float32."""
    logits = logits[:, :-1].float()
    targets = input_ids[:, 1:]
    valid = attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    ent = -(logp.exp() * logp).sum(-1)
    hit = (logits.argmax(-1) == targets).float()
    out = []
    for i in range(nll.size(0)):
        m = valid[i]
        if m.sum() == 0:
            out.append(np.zeros(LM_DIM, np.float32))
            continue
        n, e, h = nll[i][m], ent[i][m], hit[i][m]
        out.append(np.array([
            n.mean(), n.std(unbiased=False), n.max(), torch.quantile(n, 0.9), e.mean(), e.std(unbiased=False),
            h.mean(), n.mean() / e.mean().clamp(min=1e-6),
        ], dtype=np.float32))
    return np.stack(out)


@torch.no_grad()
def lm_features(model, tokenizer, texts, max_len: int = 256, batch_size: int = 8, device: str | None = None):
    """Run a frozen causal LM over texts; returns [N, LM_DIM]. Sort by length for speed, restore order."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    order = np.argsort([len(t) for t in texts])
    feats = np.zeros((len(texts), LM_DIM), np.float32)
    use_amp = device.startswith("cuda")
    for s in range(0, len(texts), batch_size):
        idx = order[s:s + batch_size]
        enc = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        feats[idx] = lm_stats_from_logits(logits, enc["input_ids"], enc["attention_mask"])
    return feats
