"""
gpad_v2_model.py
GPAD V2 model – entropy-weighted attention + stylometric fusion
Compatible with Trainer-based fine-tuning (Task A/B)
UPDATED: Uses true Shannon entropy calculation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
@dataclass
class GPADConfig:
    base_model: str = "microsoft/codebert-base"
    num_labels: int = 2
    stylometric_dim: int = 32
    num_gpad_layers: int = 1
    entropy_alpha: float = 0.15
    max_length: int = 512
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    use_stylometric: bool = True
    use_gradient_checkpointing: bool = True
    classifier_hidden_dim: int = 384


# -------------------------------------------------------------------------
# Stylometric feature extractor
# -------------------------------------------------------------------------
class StylometricExtractor(nn.Module):
    """Extract 16 handcrafted code-style features → projected to stylometric_dim."""

    def __init__(self, feature_dim: int = 32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(16, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def extract_features(self, code_text: str) -> torch.Tensor:
        try:
            lines = code_text.split("\n")
            tokens = code_text.split()
            char_count = max(len(code_text), 1)
            line_count = max(len(lines), 1)
            token_count = max(len(tokens), 1)

            f = [
                char_count / line_count,
                token_count / char_count,
                code_text.count(" ") / char_count,
                code_text.count("\t") / char_count,
                (code_text.count("if ") + code_text.count("for ") + code_text.count("while ")) / line_count,
                code_text.count("def ") / line_count,
                code_text.count("class ") / line_count,
                len(set(tokens)) / token_count,
                code_text.count("import ") / line_count,
                code_text.count("#") / line_count,
                code_text.count("(") / char_count,
                code_text.count("{") / char_count,
                code_text.count('"') / char_count,
                len([t for t in tokens if len(t) > 10]) / token_count,
                sum(1 for c in code_text if c.isupper()) / char_count,
                sum(1 for l in lines if len(l.strip()) > 80) / line_count,
            ]
            return torch.tensor(f, dtype=torch.float32)
        except Exception:
            return torch.zeros(16, dtype=torch.float32)

    def forward(self, code_texts: List[str]) -> torch.Tensor:
        feats = [self.extract_features(t) for t in code_texts]
        device = next(self.proj.parameters()).device
        return self.proj(torch.stack(feats).to(device))


# -------------------------------------------------------------------------
# Entropy modules (UPDATED: True Shannon Entropy)
# -------------------------------------------------------------------------
class EntropyCalculator(nn.Module):
    """Calculate true distributional Shannon entropy from attention patterns."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.query_net = nn.Linear(head_dim, head_dim)
        self.key_net = nn.Linear(head_dim, head_dim)
        self.scale = head_dim ** -0.5
        
    def forward(self, keys, values):
        """Calculate entropy from self-similarity distribution
        
        Args:
            keys: [B, H, S, D] key vectors
            values: [B, H, S, D] value vectors (for compatibility)
            
        Returns:
            entropy: [B, H, S] normalized entropy scores
        """
        b, h, s, d = keys.shape
        
        # Compute self-attention distribution for each token
        q = self.query_net(keys)
        k = self.key_net(keys)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        probs = F.softmax(scores, dim=-1)
        
        # Shannon entropy: measures uncertainty in attention distribution
        # H(X) = -sum(p(x) * log(p(x)))
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        
        # Normalize: high entropy = uncertain/generic token
        # Invert so that high entropy tokens get lower weights
        return torch.sigmoid(entropy.mean(dim=-1, keepdim=True) - entropy)


class EntropyWeightedAttention(nn.Module):
    """Multi-head attention with true Shannon entropy bias."""

    def __init__(self, hidden: int, heads: int, alpha: float, dropout: float):
        super().__init__()
        self.h, self.heads, self.dim = hidden, heads, hidden // heads
        self.scale = self.dim ** -0.5
        self.alpha = alpha
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.o = nn.Linear(hidden, hidden)
        self.entropy = EntropyCalculator(self.dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, mask=None):
        b, s, _ = x.shape
        res = x
        
        q = self.q(x).view(b, s, self.heads, self.dim).transpose(1, 2)
        k = self.k(x).view(b, s, self.heads, self.dim).transpose(1, 2)
        v = self.v(x).view(b, s, self.heads, self.dim).transpose(1, 2)
        
        # Attention with Shannon entropy bias
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        e = self.entropy(k, v)
        scores = scores + self.alpha * e.unsqueeze(2)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, torch.finfo(scores.dtype).min)
        
        attn = self.drop(F.softmax(scores, dim=-1))
        # Handle NaN/Inf values
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, self.h)
        out = self.o(out)
        return self.norm(res + self.drop(out))


# -------------------------------------------------------------------------
# Style fusion + FFN block
# -------------------------------------------------------------------------
class StyleIntegration(nn.Module):
    def __init__(self, hidden: int, style_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(style_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden + style_dim, hidden),
            nn.Sigmoid(),
        )

    def forward(self, h, s):
        B, L, H = h.shape
        s_proj = self.proj(s).unsqueeze(1).expand(-1, L, -1)
        g = self.gate(torch.cat([h, s.unsqueeze(1).expand(-1, L, -1)], dim=-1))
        return h + g * s_proj


class GPADLayer(nn.Module):
    def __init__(self, cfg: GPADConfig, base_cfg):
        super().__init__()
        self.attn = EntropyWeightedAttention(
            hidden=base_cfg.hidden_size,
            heads=base_cfg.num_attention_heads,
            alpha=cfg.entropy_alpha,
            dropout=cfg.attention_probs_dropout_prob,
        )
        self.style = StyleIntegration(base_cfg.hidden_size, cfg.stylometric_dim) if cfg.use_stylometric else None
        self.ff = nn.Sequential(
            nn.Linear(base_cfg.hidden_size, base_cfg.intermediate_size),
            nn.GELU(),
            nn.Dropout(cfg.hidden_dropout_prob),
            nn.Linear(base_cfg.intermediate_size, base_cfg.hidden_size),
            nn.Dropout(cfg.hidden_dropout_prob),
        )
        self.norm = nn.LayerNorm(base_cfg.hidden_size)

    def forward(self, h, s=None, mask=None):
        # Apply entropy-weighted attention
        h = self.attn(h, mask)
        
        # Apply style fusion if available
        if self.style and s is not None:
            h = self.style(h, s)
        
        # Feed-forward network
        return self.norm(h + self.ff(h))


# -------------------------------------------------------------------------
# Main model
# -------------------------------------------------------------------------
class GPADModel(PreTrainedModel):
    """Full GPAD architecture for authorship detection with Shannon entropy."""

    config_class = RobertaConfig

    def __init__(self, cfg: GPADConfig):
        roberta_cfg = RobertaConfig.from_pretrained(cfg.base_model)
        roberta_cfg.num_labels = cfg.num_labels
        super().__init__(roberta_cfg)

        self.cfg = cfg
        self.encoder = RobertaModel.from_pretrained(cfg.base_model, config=roberta_cfg, add_pooling_layer=False)
        if cfg.use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        self.style = StylometricExtractor(cfg.stylometric_dim) if cfg.use_stylometric else None
        self.layers = nn.ModuleList([GPADLayer(cfg, roberta_cfg) for _ in range(cfg.num_gpad_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(roberta_cfg.hidden_size, cfg.classifier_hidden_dim),
            nn.LayerNorm(cfg.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.classifier_hidden_dim, cfg.classifier_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.classifier_hidden_dim // 2, cfg.num_labels),
        )

        self.init_weights()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"✓ GPAD model ready ({trainable/1e6:.1f} M trainable / {total/1e6:.1f} M total)")

    def forward(self, input_ids, attention_mask=None, code_texts=None, labels=None, output_hidden_states=False):
        B = input_ids.size(0)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        h = out.last_hidden_state
        
        # Extract stylometric features
        style = self.style(code_texts) if self.style else None
        
        # Apply GPAD layers with Shannon entropy-weighted attention
        for layer in self.layers:
            h = layer(h, style, attention_mask)
        
        # Classification using [CLS] token
        cls = h[:, 0, :]
        logits = self.classifier(cls)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.cfg.num_labels), labels.view(-1))

        # Only return hidden_states if explicitly requested to save memory
        return SequenceClassifierOutput(
            loss=loss, 
            logits=logits, 
            hidden_states=h if output_hidden_states else None
        )