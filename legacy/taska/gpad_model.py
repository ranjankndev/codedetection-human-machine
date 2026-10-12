
import os
import json
import math
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.utils import clip_grad_norm_
from transformers import (
    RobertaModel, RobertaConfig, PreTrainedModel, AutoModel
)
from transformers.modeling_outputs import SequenceClassifierOutput

# Logging
logger = logging.getLogger(__name__)

# Env setup
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


@dataclass
class GPADConfig:
    """Configuration class for GPAD model."""
    base_model: str = "microsoft/codebert-base"
    num_labels: int = 2
    stylometric_dim: int = 64
    num_gpad_layers: int = 3
    entropy_alpha: float = 0.1
    max_length: int = 512
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-12
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    cache_entropy: bool = True


# =====================================================
# Stylometric Extractor
# =====================================================
class OptimizedStylemetricExtractor(nn.Module):
    """Highly optimized stylometric feature extractor with caching."""

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_cache: Dict[int, torch.Tensor] = {}

        self.feature_projection = nn.Sequential(
            nn.Linear(32, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def extract_raw_features(self, code_text: str) -> torch.Tensor:
        features: List[float] = []
        cache_key = hash(code_text)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        try:
            lines = code_text.split("\n")
            tokens = code_text.split()

            char_count = len(code_text)
            line_count = len(lines)
            non_empty_lines = sum(1 for line in lines if line.strip())
            token_count = len(tokens)

            # Ratios
            features.extend([
                char_count / max(line_count, 1),
                non_empty_lines / max(line_count, 1),
                token_count / max(char_count, 1),
                code_text.count(" ") / max(char_count, 1),
                code_text.count("\t") / max(char_count, 1),
            ])

            # Complexity
            control_structures = (code_text.count("if ") + code_text.count("for ")
                                  + code_text.count("while ") + code_text.count("switch"))
            features.extend([
                control_structures / max(line_count, 1),
                code_text.count("def ") + code_text.count("class "),
                abs(code_text.count("(") - code_text.count(")")),
                abs(code_text.count("{") - code_text.count("}")),
            ])

            # Lexical diversity
            unique_tokens = set(token.lower() for token in tokens if token.isalnum())
            features.extend([
                len(unique_tokens) / max(token_count, 1),
                sum(1 for token in tokens if len(token) > 10) / max(token_count, 1),
                code_text.count("#") / max(line_count, 1),
                (code_text.count('"') + code_text.count("'")) / max(char_count, 1),
            ])

            # Advanced
            features.extend([
                len([line for line in lines if len(line.strip()) > 80]) / max(line_count, 1),
                code_text.count("import ") + code_text.count("from "),
                code_text.count("return ") / max(control_structures + 1, 1),
                sum(line.count(" " * 4) for line in lines) / max(line_count, 1),
                len([c for c in code_text if c.isupper()]) / max(char_count, 1),
            ])

            # Stats
            line_lengths = [len(line) for line in lines if line.strip()]
            if line_lengths:
                features.extend([
                    float(np.mean(line_lengths)),
                    float(np.std(line_lengths)),
                    float(np.median(line_lengths)),
                    float(max(line_lengths) - min(line_lengths)),
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

            # Pad/truncate
            while len(features) < 32:
                features.append(0.0)
            features = features[:32]

            result = torch.tensor(features, dtype=torch.float32)
            self.feature_cache[cache_key] = result
            return result
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return torch.zeros(32, dtype=torch.float32)

    def forward(self, code_texts: List[str]) -> torch.Tensor:
        batch_features: List[torch.Tensor] = []
        for text in code_texts:
            features = self.extract_raw_features(text)
            batch_features.append(features)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_tensor = torch.stack(batch_features).to(device)
        return self.feature_projection(batch_tensor)


# =====================================================
# Entropy Calculator
# =====================================================
class OptimizedEntropyCalculator(nn.Module):
    """Entropy calculation with caching."""

    def __init__(self, model_name: str = "microsoft/codebert-base", cache_size: int = 10000):
        super().__init__()
        self.cache_size = cache_size
        self.entropy_cache: Dict[int, torch.Tensor] = {}

        try:
            self.entropy_model = AutoModel.from_pretrained(
                model_name,
                cache_dir=None,  # Use default HuggingFace cache
                use_safetensors=True
            )  
            logger.info(f"Loaded entropy model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load entropy model: {e}")
            logger.warning("Creating a dummy entropy model instead")
    # Create a simple dummy model instead of using TensorFlow
            from transformers import RobertaModel, RobertaConfig
            config = RobertaConfig.from_pretrained(model_name)
            self.entropy_model = RobertaModel(config)

        self.entropy_model.eval()
        for p in self.entropy_model.parameters():
            p.requires_grad = False

        self.entropy_projection = nn.Linear(self.entropy_model.config.hidden_size, 1)

    def calculate_token_entropy_fast(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        try:
            with torch.no_grad():
                outputs = self.entropy_model(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             output_hidden_states=False)
                hidden_states = outputs.last_hidden_state
                token_variance = torch.var(hidden_states, dim=-1)
                entropy_approx = torch.log(token_variance + 1e-8)
                entropy_approx = torch.clamp(entropy_approx, min=-5, max=5)

                entropy_scores = self.entropy_projection(hidden_states).squeeze(-1)
                final_entropy = 0.7 * entropy_approx + 0.3 * entropy_scores
                return final_entropy * attention_mask.float()
        except Exception as e:
            logger.warning(f"Entropy calculation failed: {e}")
            return torch.zeros(batch_size, seq_len, device=device)

    def forward(self, input_ids, attention_mask):
        return self.calculate_token_entropy_fast(input_ids, attention_mask)


# =====================================================
# Entropy-Weighted Self-Attention
# =====================================================
class EntropyWeightedSelfAttention(nn.Module):
    """Self-Attention with entropy bias."""

    def __init__(self, gpad_config: GPADConfig, base_model_config):
        super().__init__()
        self.gpad_config = gpad_config
        self.hidden_size = base_model_config.hidden_size
        self.num_heads = base_model_config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.entropy_alpha = gpad_config.entropy_alpha
        self.scale = self.head_dim ** -0.5

        assert self.hidden_size % self.num_heads == 0

        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(gpad_config.attention_probs_dropout_prob)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=gpad_config.layer_norm_eps)

        self.entropy_gate = nn.Sequential(nn.Linear(1, self.num_heads), nn.Sigmoid())

    def forward(self, hidden_states, entropy_weights=None, attention_mask=None, use_flash_attention=True):
        batch_size, seq_len, hidden_size = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_output = self._standard_attention(q, k, v, entropy_weights, attention_mask)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, hidden_size)
        output = self.out_proj(attn_output)
        output = self.layer_norm(output + hidden_states)
        return output

    def _standard_attention(self, q, k, v, entropy_weights, attention_mask):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if entropy_weights is not None:
            entropy_bias = self.entropy_gate(entropy_weights.unsqueeze(-1))
            entropy_bias = entropy_bias.permute(0, 2, 1).unsqueeze(2)
            entropy_bias = self.entropy_alpha * entropy_bias
            attn_scores = attn_scores + entropy_bias

        if attention_mask is not None:
            mask_value = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, mask_value)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        return torch.matmul(attn_probs, v)


# =====================================================
# Cross-Attention with Stylometry
# =====================================================
class CrossAttentionWithStylemetry(nn.Module):
    """Cross-Attention with stylometric integration."""

    def __init__(self, gpad_config: GPADConfig, base_model_config):
        super().__init__()
        self.gpad_config = gpad_config
        self.hidden_size = base_model_config.hidden_size
        self.stylometric_dim = gpad_config.stylometric_dim
        self.num_heads = base_model_config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.content_heads = max(1, self.num_heads * 3 // 4)
        self.style_heads = self.num_heads - self.content_heads

        self.content_q = nn.Linear(self.hidden_size, self.content_heads * self.head_dim, bias=False)
        self.content_k = nn.Linear(self.hidden_size, self.content_heads * self.head_dim, bias=False)
        self.content_v = nn.Linear(self.hidden_size, self.content_heads * self.head_dim, bias=False)

        self.style_q = nn.Linear(self.hidden_size, self.style_heads * self.head_dim, bias=False)
        self.style_k = nn.Linear(self.stylometric_dim, self.style_heads * self.head_dim, bias=False)
        self.style_v = nn.Linear(self.stylometric_dim, self.style_heads * self.head_dim, bias=False)

        self.style_integration = nn.Sequential(
            nn.Linear(self.stylometric_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(gpad_config.hidden_dropout_prob)
        )

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=gpad_config.layer_norm_eps)
        self.dropout = nn.Dropout(gpad_config.attention_probs_dropout_prob)
        self.scale = self.head_dim ** -0.5

    def forward(self, hidden_states, stylometric_features, attention_mask=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        style_projected = self.style_integration(stylometric_features)
        style_expanded = style_projected.unsqueeze(1).expand(-1, seq_len, -1)
        augmented_states = hidden_states + 0.1 * style_expanded

        # Content attention
        content_q = self.content_q(augmented_states).view(batch_size, seq_len, self.content_heads, self.head_dim).transpose(1, 2)
        content_k = self.content_k(augmented_states).view(batch_size, seq_len, self.content_heads, self.head_dim).transpose(1, 2)
        content_v = self.content_v(augmented_states).view(batch_size, seq_len, self.content_heads, self.head_dim).transpose(1, 2)
        content_scores = torch.matmul(content_q, content_k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            mask_value = torch.finfo(content_scores.dtype).min
            content_scores = content_scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, mask_value)
        content_probs = F.softmax(content_scores, dim=-1)
        content_out = torch.matmul(content_probs, content_v)

        # Style attention
        cls_vec = hidden_states[:, 0, :]
        style_q = self.style_q(cls_vec).view(batch_size, 1, self.style_heads, self.head_dim).transpose(1, 2)
        style_k = self.style_k(stylometric_features).view(batch_size, 1, self.style_heads, self.head_dim).transpose(1, 2)
        style_v = self.style_v(stylometric_features).view(batch_size, 1, self.style_heads, self.head_dim).transpose(1, 2)
        style_scores = torch.matmul(style_q, style_k.transpose(-2, -1)) * self.scale
        style_probs = F.softmax(style_scores, dim=-1)
        style_out = torch.matmul(style_probs, style_v).expand(-1, -1, seq_len, -1)

        combined_out = torch.cat([content_out, style_out], dim=1)
        combined_out = combined_out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        output = self.out_proj(combined_out)
        output = self.dropout(output)
        output = self.layer_norm(output + hidden_states)
        return output


# =====================================================
# GPAD Layer
# =====================================================
class OptimizedGPADLayer(nn.Module):
    def __init__(self, gpad_config: GPADConfig, base_model_config):
        super().__init__()
        self.gpad_config = gpad_config
        self.entropy_attention = EntropyWeightedSelfAttention(gpad_config, base_model_config)
        self.cross_attention = CrossAttentionWithStylemetry(gpad_config, base_model_config)
        self.ffn = nn.Sequential(
            nn.Linear(base_model_config.hidden_size, gpad_config.intermediate_size),
            nn.GELU(),
            nn.Dropout(gpad_config.hidden_dropout_prob),
            nn.Linear(gpad_config.intermediate_size, base_model_config.hidden_size),
            nn.Dropout(gpad_config.hidden_dropout_prob),
        )
        self.layer_norm_ffn = nn.LayerNorm(base_model_config.hidden_size, eps=gpad_config.layer_norm_eps)

    def forward(self, hidden_states, stylometric_features, entropy_weights=None, attention_mask=None):
        if self.gpad_config.use_gradient_checkpointing and self.training:
            return checkpoint.checkpoint(
                self._forward_impl, hidden_states, stylometric_features, entropy_weights, attention_mask, use_reentrant=False
            )
        else:
            return self._forward_impl(hidden_states, stylometric_features, entropy_weights, attention_mask)

    def _forward_impl(self, hidden_states, stylometric_features, entropy_weights, attention_mask):
        attn_output = self.entropy_attention(hidden_states, entropy_weights, attention_mask)
        cross_output = self.cross_attention(attn_output, stylometric_features, attention_mask)
        ffn_output = self.ffn(cross_output)
        output = self.layer_norm_ffn(ffn_output + cross_output)
        return output


# =====================================================
# GPAD Model
# =====================================================
class OptimizedGPADModel(PreTrainedModel):
    config_class = RobertaConfig

    def __init__(self, gpad_config: GPADConfig):
        roberta_config = RobertaConfig.from_pretrained(gpad_config.base_model)
        roberta_config.num_labels = gpad_config.num_labels
        super().__init__(roberta_config)

        self.gpad_config = gpad_config
        self.num_labels = gpad_config.num_labels
        self.roberta = RobertaModel(roberta_config, add_pooling_layer=False)

        self.stylometric_extractor = OptimizedStylemetricExtractor(gpad_config.stylometric_dim)
        self.entropy_calculator = OptimizedEntropyCalculator(model_name=gpad_config.base_model)
        self.gpad_layers = nn.ModuleList([OptimizedGPADLayer(gpad_config, roberta_config) for _ in range(gpad_config.num_gpad_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(roberta_config.hidden_size, roberta_config.hidden_size // 2),
            nn.LayerNorm(roberta_config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(roberta_config.hidden_size // 2, roberta_config.hidden_size // 4),
            nn.LayerNorm(roberta_config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(roberta_config.hidden_size // 4, self.num_labels),
        )

        self.style_predictor = nn.Sequential(
            nn.Linear(roberta_config.hidden_size, gpad_config.stylometric_dim), nn.Tanh()
        )
        self.entropy_predictor = nn.Sequential(
            nn.Linear(roberta_config.hidden_size, 1), nn.Sigmoid()
        )
        self.pooler = nn.Sequential(nn.Linear(roberta_config.hidden_size, roberta_config.hidden_size), nn.Tanh())

        self.init_weights()
        self._apply_optimizations()

    def _apply_optimizations(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if hasattr(module, "weight"):
                    torch.nn.init.ones_(module.weight)
                if hasattr(module, "bias"):
                    torch.nn.init.zeros_(module.bias)

        if self.gpad_config.use_gradient_checkpointing:
            try:
                self.roberta.gradient_checkpointing_enable()
            except Exception:
                pass

    def forward(self, input_ids, attention_mask=None, code_texts=None, labels=None, **kwargs):
        batch_size = input_ids.shape[0]

        if code_texts is None or len(code_texts) != batch_size:
            code_texts = [""] * batch_size

        stylometric_features = self.stylometric_extractor(code_texts)
        entropy_weights = self.entropy_calculator(input_ids, attention_mask)

        encoder_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = encoder_outputs.last_hidden_state

        for gpad_layer in self.gpad_layers:
            hidden_states = gpad_layer(hidden_states, stylometric_features, entropy_weights, attention_mask)

        cls_output = hidden_states[:, 0, :]

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_output = sum_embeddings / sum_mask

            hidden_states_masked = hidden_states.masked_fill(~mask_expanded.bool(), -1e9)
            max_output = torch.max(hidden_states_masked, 1)[0]
        else:
            mean_output = torch.mean(hidden_states, 1)
            max_output = torch.max(hidden_states, 1)[0]

        pooled_output = self.pooler(cls_output + 0.3 * mean_output + 0.2 * max_output)

        logits = self.classifier(pooled_output)
        style_pred = self.style_predictor(pooled_output)
        entropy_pred = self.entropy_predictor(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                classification_loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
                classification_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            style_loss = F.mse_loss(style_pred, stylometric_features.detach())
            entropy_loss = F.mse_loss(entropy_pred.squeeze(), entropy_weights.mean(dim=1).detach())

            loss = classification_loss + 0.1 * style_loss + 0.05 * entropy_loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states if kwargs.get("output_hidden_states") else None,
            attentions=None,
        )