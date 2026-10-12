# =============================
# INSTALL DEPENDENCIES
# =============================
!pip install radon pytz
#!pip install -U "transformers==4.44.2"
!pip install --upgrade transformers==4.45.0

# =============================
# IMPORTS & UTILITIES
# =============================
import pytz
from datetime import datetime

def log_message(*args):
    msg = " ".join(str(a) for a in args)
    ist = pytz.timezone("Asia/Kolkata")
    timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{timestamp}] {msg}")

# Trial run configuration for quick testing vs full training
TRIAL_RUN = False  # Set to True for quick testing with small datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import ast
import keyword
import radon.complexity as rcc
import radon.metrics as rmetrics
import tokenize
import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os
import pandas as pd
from collections import Counter, defaultdict
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import sys, time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm

# Enhanced GPU detection and multi-GPU setup
gpu_count = 0
if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_count = torch.cuda.device_count()
    log_message(f"🔥 CUDA available! Found {gpu_count} GPU(s)")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        log_message(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    if gpu_count > 1:
        log_message(f"🚀 Multi-GPU training will be enabled with {gpu_count} GPUs")
    else:
        log_message(f"📱 Single GPU training on {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    log_message(f"💻 Using CPU (CUDA not available)")

log_message(f"Primary device: {device}")

# =============================
# CONSTANTS
# =============================
MODEL_NAME = "microsoft/codebert-base"
TOKENIZER_NAME = MODEL_NAME

human_label = "human"

label2id = {
  human_label: "0",
  "deepseek": "1",
  "qwen": "2",
  "01-ai": "3",
  "bigcode": "4",
  "gemma": "5",
  "phi": "6",
  "meta-llama": "7",
  "ibm-granite": "8",
  "mistral": "9",
  "openai": "10"
}

id2label = {
  "0": human_label,
  "1": "deepseek",
  "2": "qwen",
  "3": "01-ai",
  "4": "bigcode",
  "5": "gemma",
  "6": "phi",
  "7": "meta-llama",
  "8": "ibm-granite",
  "9": "mistral",
  "10": "openai"
}

# =============================
# STYLOMETRIC FEATURE EXTRACTION
# =============================
def extract_stylometric_features(code, stylo_dim=20):
    import radon.complexity as rcc
    import radon.metrics as rmetrics
    import radon.raw as rraw
    import numpy as np
    features = []
    try:
        raw = rraw.analyze(code)
        features.extend([raw.loc, raw.lloc, raw.comments, raw.multi, raw.blank])
    except Exception:
        features.extend([0.0]*5)
    try:
        cc_list = rcc.cc_visit(code)
        if cc_list:
            cc_vals = [c.complexity for c in cc_list]
            features.extend([np.mean(cc_vals), np.max(cc_vals), len(cc_vals)])
        else:
            features.extend([0.0, 0.0, 0.0])
    except Exception:
        features.extend([0.0, 0.0, 0.0])
    try:
        hal = rmetrics.h_visit(code)
        features.extend([
            hal.total.operators, hal.total.operands,
            hal.total.distinct_operators, hal.total.distinct_operands,
            hal.total.volume
        ])
    except Exception:
        features.extend([0.0]*5)
    while len(features) < stylo_dim:
        features.append(0.0)
    features = features[:stylo_dim]
    return np.array(features, dtype=np.float32)

def _max_ast_depth(node, depth=0):
    if not isinstance(node, ast.AST) or not list(ast.iter_child_nodes(node)):
        return depth
    return max(_max_ast_depth(child, depth + 1) for child in ast.iter_child_nodes(node))

# ============================================
# Stylometric Feature Caching Utility
# ============================================

def compute_and_cache_stylo_features(df, cache_path, stylo_dim=20):
    """
    Compute or load stylometric features for the given dataframe.

    Parameters:
        df : pd.DataFrame with a 'code' column
        cache_path : str, path to .npy or .pkl file
        stylo_dim : int, dimension to pad/trim stylometric features

    Returns:
        np.ndarray of shape [len(df), stylo_dim]
    """
    if os.path.exists(cache_path):
        log_message(f"✅ Loaded cached stylometric features from {cache_path}")
        if cache_path.endswith(".npy"):
            return np.load(cache_path)
        else:
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    log_message(f"⚙️ Computing stylometric features for {len(df)} samples...")
    all_features = []
    for code in tqdm(df["code"].astype(str).values, total=len(df)):
        feats = extract_stylometric_features(code, stylo_dim=stylo_dim)
        all_features.append(feats)

    all_features = np.vstack(all_features)

    # Save to disk
    if cache_path.endswith(".npy"):
        np.save(cache_path, all_features)
    else:
        with open(cache_path, "wb") as f:
            pickle.dump(all_features, f)

    log_message(f"✅ Saved computed stylometric features to {cache_path}")
    return all_features

# =============================
# ENTROPY COMPUTATION
# =============================
def token_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

# =============================
# ENTROPY-WEIGHTED SELF-ATTENTION
# =============================
class EntropyWeightedSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Learnable entropy scaling parameter
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        # Scaling factor for attention scores
        self.scale = self.head_dim ** -0.5

    def forward(self, hidden_states, entropy_bias=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores: Q * K^T / sqrt(d_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        
        # Apply entropy bias if provided
        if entropy_bias is not None:
            # entropy_bias shape: [batch_size, seq_len]
            # We need to add bias to each attention head
            bias = self.alpha * entropy_bias  # Shape: [batch_size, seq_len]
            
            # Expand bias to match attention scores dimensions
            # We add bias to the diagonal (self-attention) and broadcast across heads
            bias_expanded = bias.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, seq_len, 1]
            bias_expanded = bias_expanded.expand(-1, self.num_heads, -1, seq_len)  # [batch_size, num_heads, seq_len, seq_len]
            
            # Add entropy bias to attention scores BEFORE softmax
            attn_scores = attn_scores + bias_expanded
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        # Shape: [batch_size, num_heads, seq_len, head_dim]
        
        # Concatenate heads and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        attn_output = self.out_proj(attn_output)
        
        # Return averaged attention weights across heads for visualization
        attn_weights_avg = attn_weights.mean(dim=1)  # Average across heads
        
        return attn_output, attn_weights_avg

# =============================
# STYLOMETRY CROSS-ATTENTION
# =============================
class StylometryCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, stylo_dim):
        super().__init__()
        self.stylo_proj = nn.Linear(stylo_dim, hidden_size)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

    def forward(self, decoder_states, encoder_states, stylo_features):
        stylo_proj = self.stylo_proj(stylo_features)
        stylo_proj = stylo_proj.unsqueeze(1).expand(-1, encoder_states.size(1), -1)
        encoder_aug = encoder_states + stylo_proj
        output, attn_weights = self.cross_attn(decoder_states, encoder_aug, encoder_aug)
        return output, attn_weights

# =============================
# MAIN MODEL
# =============================
class CodeBERTAuthorshipClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_labels=12, stylo_dim=20, hidden_size=768, num_heads=8):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj_stylo = nn.Linear(stylo_dim, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)
        self.stylo_regressor = nn.Linear(hidden_size, stylo_dim)
        self.entropy_attn = EntropyWeightedSelfAttention(hidden_size, num_heads=num_heads)

    def forward(self, input_ids, attention_mask, stylo_features, labels=None, stylo_targets=None):
        enc_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        entropy = token_entropy(enc_outputs.last_hidden_state)
        attn_out, _ = self.entropy_attn(enc_outputs.last_hidden_state, entropy_bias=entropy)
        pooled_output = attn_out.mean(dim=1)
        stylo_proj = self.proj_stylo(stylo_features)
        combined = torch.cat([pooled_output, stylo_proj], dim=-1)
        combined = self.dropout(combined)
        class_logits = self.classifier(combined)
        stylo_pred = self.stylo_regressor(pooled_output)
        loss = None
        if labels is not None and stylo_targets is not None:
            ce_loss = F.cross_entropy(class_logits, labels)
            mse_loss = F.mse_loss(stylo_pred, stylo_targets)
            loss = ce_loss + 0.3 * mse_loss
        return {"loss": loss, "logits": class_logits, "stylo_pred": stylo_pred}

# =============================
# TRAINING FUNCTION
# =============================
def train_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    log_interval: int = 200,
    patience: int = 3,
    weight_decay: float = 0.01,
    gradient_clip_norm: float = 1.0,
    warmup_ratio: float = 0.1,
    scheduler_type: str = 'linear_warmup',
    save_checkpoints: bool = True,
    checkpoint_dir: str = None,
    monitor_metric: str = 'f1_macro',
    accumulation_steps: int = 1,
    use_amp: bool = True
):
    """
    Enhanced training function with comprehensive improvements.
    
    IMPROVEMENTS MADE:
    1. CODE READABILITY & MAINTAINABILITY:
       - Added comprehensive type hints for all parameters and return value
       - Enhanced docstring with detailed parameter descriptions
       - Separated validation logic into helper function
       - Better variable naming and code organization
       - Added training history tracking for analysis
    
    2. PERFORMANCE OPTIMIZATION:
       - Added gradient clipping to prevent exploding gradients
       - Gradient accumulation for effective larger batch sizes
       - Memory-efficient operations with proper device handling
       - Enhanced optimizer configuration with weight decay
       - Multiple scheduler options with configurable warmup
       - Performance monitoring with detailed timing metrics
    
    3. BEST PRACTICES & PATTERNS:
       - Comprehensive input validation with descriptive errors
       - Professional checkpoint saving system
       - Flexible monitoring metrics (F1, accuracy, loss)
       - Resource cleanup and memory management
       - Structured logging with progress tracking
       - Exception handling for robust training
    
    4. ERROR HANDLING & EDGE CASES:
       - Input parameter validation
       - GPU memory error handling with cleanup
       - Graceful handling of training interruptions
       - Device compatibility checks
       - Robust error recovery mechanisms
    
    Args:
        model: The neural network model to train
        dataloader: Training data loader
        val_dataloader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to run training on (cpu/cuda)
        log_interval: Interval for logging training progress (default: 200)
        patience: Number of epochs to wait before early stopping (default: 3)
        weight_decay: L2 regularization factor (default: 0.01)
        gradient_clip_norm: Maximum gradient norm for clipping (default: 1.0)
        warmup_ratio: Ratio of total steps for warmup (default: 0.1)
        scheduler_type: Learning rate scheduler type (default: 'linear_warmup')
        save_checkpoints: Whether to save model checkpoints (default: True)
        checkpoint_dir: Directory to save checkpoints (optional)
        monitor_metric: Metric to monitor for early stopping (default: 'f1_macro')
                       RECOMMENDED FOR CODE AUTHORSHIP DETECTION:
                       - 'f1_macro': Best for imbalanced multi-class (RECOMMENDED)
                       - 'f1_micro': Good for overall performance
                       - 'accuracy': Simple but can be misleading with class imbalance
                       - 'val_loss': Can overfit, less interpretable for this task
        accumulation_steps: Gradient accumulation steps (default: 1)
        use_amp: Whether to use Automatic Mixed Precision for faster training (default: True)
    
    Returns:
        tuple: (trained_model, training_history)
        
    Raises:
        ValueError: If invalid parameters are provided
        RuntimeError: If training fails due to device or memory issues
    
    METRIC SELECTION GUIDE FOR CODE AUTHORSHIP DETECTION:
    
    **F1-Macro (RECOMMENDED - DEFAULT)**:
    - Treats all classes equally regardless of frequency
    - Perfect for imbalanced datasets with multiple authors/generators
    - Ensures model performs well on minority classes (rare authors)
    - Most appropriate for authorship attribution tasks
    
    **F1-Micro**:
    - Weighted by class frequency, good for overall performance
    - May favor majority classes in imbalanced scenarios
    - Use when overall accuracy across all samples is most important
    
    **Accuracy**:
    - Simple percentage of correct predictions
    - Can be misleading with class imbalance (high accuracy by predicting majority class)
    - Use only when classes are well-balanced
    
    **Validation Loss**:
    - Can lead to overfitting and poor generalization
    - Less interpretable for classification tasks
    - May not correlate well with actual classification performance
    - Generally NOT recommended for authorship detection
    """
    
    # =============================
    # 1. INPUT VALIDATION - Enhanced error handling
    # =============================
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if lr <= 0 or lr > 1:
        raise ValueError(f"learning rate must be in (0, 1], got {lr}")
    if patience < 0:
        raise ValueError(f"patience must be non-negative, got {patience}")
    if log_interval <= 0:
        raise ValueError(f"log_interval must be positive, got {log_interval}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if gradient_clip_norm < 0:
        raise ValueError(f"gradient_clip_norm must be non-negative, got {gradient_clip_norm}")
    if not 0 <= warmup_ratio <= 1:
        raise ValueError(f"warmup_ratio must be in [0, 1], got {warmup_ratio}")
    if accumulation_steps < 1:
        raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")
    if monitor_metric not in ['f1_macro', 'f1_micro', 'accuracy', 'val_loss']:
        raise ValueError(f"monitor_metric must be one of ['f1_macro', 'f1_micro', 'accuracy', 'val_loss'], got {monitor_metric}")
    
    # Validate data loaders
    if len(dataloader) == 0:
        raise ValueError("Training dataloader cannot be empty")
    if len(val_dataloader) == 0:
        raise ValueError("Validation dataloader cannot be empty")
    
    # =============================
    # 2. SETUP AND CONFIGURATION - Better organization
    # =============================
    log_message("✅ Started Enhanced Training with Advanced Early Stopping")
    
    # Setup checkpoint directory
    if save_checkpoints and checkpoint_dir:
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            log_message(f"📁 Checkpoints will be saved to: {checkpoint_dir}")
        except OSError as e:
            log_message(f"⚠️ Warning: Could not create checkpoint directory: {e}")
            save_checkpoints = False
    
    # Enhanced optimizer configuration with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # Calculate total steps for scheduler
    total_steps = len(dataloader) * epochs // accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    
    # Enhanced scheduler setup with multiple options
    if scheduler_type == 'linear_warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=lr * 0.01
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // 3), gamma=0.1
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
    
    # Move model to device with error handling
    try:
        model.to(device)
        log_message(f"🔧 Model moved to device: {device}")
        if torch.cuda.is_available() and device.type == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
            log_message(f"💾 GPU memory available: {gpu_memory:.1f} GB")
    except RuntimeError as e:
        raise RuntimeError(f"Failed to move model to device {device}: {str(e)}")
    
    # Initialize Mixed Precision Training (AMP) for faster training
    scaler = None
    if use_amp and torch.cuda.is_available() and device.type == 'cuda':
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        log_message("🚀 Mixed Precision Training (AMP) enabled for faster training")
    elif use_amp:
        log_message("⚠️ AMP requested but CUDA not available, falling back to FP32")
        use_amp = False
    
    # Enable optimizations for PyTorch 2.0+
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            # Compile model for faster execution (PyTorch 2.0+)
            if not isinstance(model, torch.nn.DataParallel):
                model = torch.compile(model, mode='reduce-overhead')
                log_message("⚡ Model compiled with torch.compile for faster execution")
        except Exception as e:
            log_message(f"⚠️ torch.compile failed: {e}, continuing without compilation")
    
    # Enhanced tracking variables
    best_metric = 0.0 if monitor_metric != 'val_loss' else float('inf')
    patience_counter = 0
    best_model_state = None
    global_step = 0
    
    # Comprehensive training history tracking
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_f1_macro': [],
        'val_f1_micro': [],
        'val_acc': [],
        'learning_rates': [],
        'epoch_times': [],
        'gradient_norms': []
    }
    
    # Log enhanced training configuration
    log_message(f"🎯 Enhanced Training Configuration:")
    log_message(f"   Epochs: {epochs}, LR: {lr}, Weight Decay: {weight_decay}")
    log_message(f"   Scheduler: {scheduler_type}, Warmup Steps: {warmup_steps}")
    log_message(f"   Gradient Clipping: {gradient_clip_norm}, Accumulation: {accumulation_steps}")
    log_message(f"   Mixed Precision (AMP): {use_amp}, GPUs: {gpu_count}")
    log_message(f"   Patience: {patience}, Monitor Metric: {monitor_metric}")
    log_message(f"   Training samples: {len(dataloader.dataset)}")
    log_message(f"   Validation samples: {len(val_dataloader.dataset)}")
    
    # =============================
    # 3. ENHANCED TRAINING LOOP
    # =============================
    try:
        for epoch in range(epochs):
            epoch_start_time = time.time()
            model.train()
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            batch_start_time = time.time()
            gradient_norms = []
            
            # For F1-macro calculation during training
            train_preds_running = []
            train_labels_running = []
            
            # Reset accumulated gradients
            optimizer.zero_grad()
            
            for step, batch in enumerate(dataloader):
                try:
                    # Move batch to device with non-blocking transfer for performance
                    batch_device = {
                        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                        for key, value in batch.items()
                    }
                    
                    # Forward pass with optional mixed precision
                    if use_amp and scaler is not None:
                        with autocast():
                            outputs = model(
                                input_ids=batch_device["input_ids"],
                                attention_mask=batch_device["attention_mask"],
                                stylo_features=batch_device["stylo_features"],
                                labels=batch_device["labels"],
                                stylo_targets=batch_device["stylo_targets"],
                            )
                            loss = outputs["loss"]
                            
                            # Handle DataParallel loss aggregation
                            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                                loss = loss.mean()
                            
                            # Scale loss for gradient accumulation
                            if accumulation_steps > 1:
                                loss = loss / accumulation_steps
                        
                        # Backward pass with gradient scaling
                        scaler.scale(loss).backward()
                    else:
                        # Standard forward pass
                        outputs = model(
                            input_ids=batch_device["input_ids"],
                            attention_mask=batch_device["attention_mask"],
                            stylo_features=batch_device["stylo_features"],
                            labels=batch_device["labels"],
                            stylo_targets=batch_device["stylo_targets"],
                        )
                        
                        loss = outputs["loss"]
                        
                        # Handle DataParallel loss aggregation
                        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                            loss = loss.mean()
                        
                        # Scale loss for gradient accumulation
                        if accumulation_steps > 1:
                            loss = loss / accumulation_steps
                        
                        # Backward pass
                        loss.backward()
                    
                    # Update metrics
                    total_loss += loss.item() * accumulation_steps  # Unscale for logging
                    
                    # Calculate accuracy and collect predictions for F1
                    with torch.no_grad():
                        logits = outputs["logits"]
                        predictions = torch.argmax(logits, dim=-1)
                        correct_predictions += (predictions == batch_device["labels"]).sum().item()
                        total_samples += batch_device["labels"].size(0)
                        
                        # Store predictions and labels for F1-macro calculation
                        train_preds_running.extend(predictions.cpu().numpy())
                        train_labels_running.extend(batch_device["labels"].cpu().numpy())
                    
                    # Gradient accumulation and optimization step
                    if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                        if use_amp and scaler is not None:
                            # AMP gradient clipping and optimization
                            if gradient_clip_norm > 0:
                                scaler.unscale_(optimizer)
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), gradient_clip_norm
                                )
                                gradient_norms.append(grad_norm.item())
                            
                            # AMP optimizer step
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            # Standard gradient clipping and optimization
                            if gradient_clip_norm > 0:
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), gradient_clip_norm
                                )
                                gradient_norms.append(grad_norm.item())
                            
                            # Standard optimizer step
                            optimizer.step()
                        
                        # Scheduler step and reset gradients
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                        
                        # Periodic memory cleanup for multi-GPU training
                        if gpu_count > 1 and global_step % 100 == 0:
                            torch.cuda.empty_cache()
                    
                    # Enhanced logging with more metrics including F1-macro
                    if (step + 1) % log_interval == 0 or (time.time() - batch_start_time) > 60:
                        elapsed = time.time() - batch_start_time
                        avg_loss = total_loss / (step + 1)
                        current_acc = correct_predictions / total_samples
                        current_lr = optimizer.param_groups[0]['lr']
                        eta = (len(dataloader) - step - 1) * (elapsed / (step + 1))
                        avg_grad_norm = np.mean(gradient_norms[-10:]) if gradient_norms else 0.0
                        
                        # Calculate running F1-macro
                        current_f1_macro = f1_score(
                            train_labels_running, train_preds_running,
                            average='macro', zero_division=0
                        ) if len(train_labels_running) > 0 else 0.0
                        
                        log_message(
                            f"[Epoch {epoch+1}/{epochs}] Step {step+1}/{len(dataloader)} "
                            f"| Loss: {avg_loss:.4f} | Acc: {current_acc:.4f} | F1 macro: {current_f1_macro:.4f} "
                            f"| LR: {current_lr:.2e} | Grad: {avg_grad_norm:.3f} "
                            f"| ETA: {eta/60:.1f}min"
                        )
                        sys.stdout.flush()
                        batch_start_time = time.time()  # Reset timer after each log
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        log_message(f"⚠️ GPU out of memory at step {step}. Clearing cache...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise RuntimeError(f"GPU out of memory. Try reducing batch size or sequence length.")
                    else:
                        log_message(f"❌ Runtime error at step {step}: {str(e)}")
                        raise
            
            # Calculate epoch metrics
            epoch_time = time.time() - epoch_start_time
            avg_train_loss = total_loss / len(dataloader)
            train_accuracy = correct_predictions / total_samples
            current_lr = optimizer.param_groups[0]['lr']
            avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0.0
            
            log_message(
                f"✅ Epoch {epoch+1}/{epochs} completed in {epoch_time:.1f}s. "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}"
            )
            
            # ===== ENHANCED VALIDATION =====
            val_metrics = _validate_model(model, val_dataloader, device, log_interval)
            
            # Store comprehensive training history
            training_history['train_loss'].append(avg_train_loss)
            training_history['train_acc'].append(train_accuracy)
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_f1_macro'].append(val_metrics['f1_macro'])
            training_history['val_f1_micro'].append(val_metrics['f1_micro'])
            training_history['val_acc'].append(val_metrics['accuracy'])
            training_history['learning_rates'].append(current_lr)
            training_history['epoch_times'].append(epoch_time)
            training_history['gradient_norms'].append(avg_grad_norm)
            
            # Enhanced validation logging
            log_message(
                f"📊 Validation - Loss: {val_metrics['loss']:.4f}, "
                f"F1-macro: {val_metrics['f1_macro']:.4f}, "
                f"F1-micro: {val_metrics['f1_micro']:.4f}, "
                f"Acc: {val_metrics['accuracy']:.4f}"
            )
            sys.stdout.flush()
            
            # ===== ENHANCED EARLY STOPPING =====
            current_metric = val_metrics[monitor_metric.replace('val_', '')]
            is_better = (
                current_metric > best_metric if monitor_metric != 'val_loss'
                else current_metric < best_metric
            )
            
            if is_better:
                best_metric = current_metric
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                log_message(f"💾 New best {monitor_metric}: {current_metric:.4f} — model saved.")
                
                # Save checkpoint if enabled
                if save_checkpoints and checkpoint_dir:
                    checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_model_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_metric': best_metric,
                        'training_history': training_history
                    }, checkpoint_path)
                    log_message(f"💾 Checkpoint saved to {checkpoint_path}")
            else:
                patience_counter += 1
                log_message(
                    f"⚠️ No improvement in {monitor_metric}. "
                    f"Patience: {patience_counter}/{patience} "
                    f"(Best: {best_metric:.4f})"
                )
                
                if patience_counter >= patience:
                    log_message("🛑 Early stopping triggered.")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        log_message("🔄 Restored best model weights.")
                    break
    
    except KeyboardInterrupt:
        log_message("⚠️ Training interrupted by user.")
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            log_message("🔄 Restored best model weights.")
    
    except Exception as e:
        log_message(f"❌ Training failed with error: {str(e)}")
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            log_message("🔄 Restored best model weights.")
        raise
    
    # Load best model before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final summary
    total_time = sum(training_history['epoch_times'])
    log_message("=" * 60)
    log_message("🎉 TRAINING COMPLETED SUCCESSFULLY")
    log_message("=" * 60)
    log_message(f"Total training time: {total_time:.1f}s ({total_time/60:.1f}m)")
    log_message(f"Best {monitor_metric}: {best_metric:.4f}")
    log_message(f"Final learning rate: {training_history['learning_rates'][-1]:.2e}")
    log_message("=" * 60)
    
    return model, training_history


def _validate_model(
    model,
    val_dataloader,
    device,
    log_interval=200
):
    """
    Enhanced validation function with comprehensive metrics and progress logging.
    
    Args:
        model: The model to validate
        val_dataloader: Validation data loader
        device: Device to run validation on
        log_interval: Interval for logging validation progress (default: 200)
        
    Returns:
        Dictionary containing validation metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    correct_predictions = 0
    total_samples = 0
    
    # Initialize timing for progress logging
    val_start_time = time.time()
    batch_start_time = time.time()
    
    log_message("🔍 Starting validation...")
    
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            try:
                # Move batch to device
                batch_device = {
                    key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                
                # Forward pass
                outputs = model(
                    input_ids=batch_device["input_ids"],
                    attention_mask=batch_device["attention_mask"],
                    stylo_features=batch_device["stylo_features"],
                )
                
                # Calculate loss if labels available
                if "labels" in batch_device:
                    loss_outputs = model(
                        input_ids=batch_device["input_ids"],
                        attention_mask=batch_device["attention_mask"],
                        stylo_features=batch_device["stylo_features"],
                        labels=batch_device["labels"],
                        stylo_targets=batch_device.get("stylo_targets", batch_device["stylo_features"]),
                    )
                    loss = loss_outputs["loss"]
                    # Handle DataParallel loss aggregation
                    if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                        loss = loss.mean()
                    total_loss += loss.item()
                
                # Collect predictions and labels
                preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
                labels = batch_device["labels"].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
                
                # Update running accuracy for progress logging
                batch_correct = np.sum(preds == labels)
                correct_predictions += batch_correct
                total_samples += len(labels)
                
                # Progress logging with F1-macro similar to training loop
                if (step + 1) % log_interval == 0 or (time.time() - batch_start_time) > 60:
                    elapsed = time.time() - batch_start_time
                    current_acc = correct_predictions / total_samples
                    avg_loss = total_loss / (step + 1) if step > 0 else 0.0
                    eta = (len(val_dataloader) - step - 1) * (elapsed / (step + 1)) if step > 0 else 0
                    
                    # Calculate running F1-macro for validation progress
                    current_f1_macro = f1_score(
                        all_labels, all_preds,
                        average='macro', zero_division=0
                    ) if len(all_labels) > 0 else 0.0
                    
                    log_message(
                        f"🔍 Validation Step {step+1}/{len(val_dataloader)} "
                        f"| Loss: {avg_loss:.4f} | Acc: {current_acc:.4f} | F1 macro: {current_f1_macro:.4f} "
                        f"| ETA: {eta/60:.1f}min"
                    )
                    sys.stdout.flush()
                    batch_start_time = time.time()  # Reset timer after each log
                
            except RuntimeError as e:
                log_message(f"⚠️ Validation error at step {step}: {str(e)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
    
    # Calculate comprehensive metrics
    val_time = time.time() - val_start_time
    avg_loss = total_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0.0
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    
    log_message(f"✅ Validation completed in {val_time:.1f}s")
    
    return {
        'loss': avg_loss,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'accuracy': accuracy
    }


# ==============================
# Dataset Class
# ==============================

class CodeAuthorshipDataset(Dataset):
    def __init__(self, codes, labels, stylo_features, tokenizer_name=TOKENIZER_NAME, max_length=256):
        """
        codes: list of code snippets
        labels: list of author IDs (ints)
        stylo_features: tensor or list of shape [N, stylo_dim]
        """
        self.codes = codes
        self.labels = labels
        self.stylo_features = torch.tensor(stylo_features, dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = str(self.codes[idx])
        label = int(self.labels[idx])
        stylo_feat = self.stylo_features[idx]

        # Tokenize input for encoder
        enc = self.tokenizer(
            code,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "stylo_features": stylo_feat,
            "labels": torch.tensor(label, dtype=torch.long),
            "stylo_targets": stylo_feat,  # supervised regression
        }


# ==============================
# Dataloader Builder
# ==============================

def get_dataloaders(
    train_codes, train_labels, train_stylo,
    val_codes, val_labels, val_stylo,
    test_codes, test_labels, test_stylo,
    batch_size=16, max_length=256, tokenizer_name=TOKENIZER_NAME
):
    # Train dataset
    train_dataset = CodeAuthorshipDataset(
        train_codes, train_labels, train_stylo,
        tokenizer_name=tokenizer_name, max_length=max_length
    )

    # Validation dataset
    val_dataset = CodeAuthorshipDataset(
        val_codes, val_labels, val_stylo,
        tokenizer_name=tokenizer_name, max_length=max_length
    )

    # Test dataset
    test_dataset = CodeAuthorshipDataset(
        test_codes, test_labels, test_stylo,
        tokenizer_name=tokenizer_name, max_length=max_length
    )

    # Enhanced DataLoaders with multi-GPU optimizations
    # Determine optimal num_workers based on available CPUs and GPUs
    import multiprocessing
    available_cpus = multiprocessing.cpu_count()
    # "With GPUs: Uses up to 4 workers, but caps at half the available CPUs"
    # The logic min(4, available_cpus // 2) when GPUs are present is based on several performance considerations:
    # 1. GPU-CPU Workload Separation
    # When GPUs are available:
    # GPUs handle: The computationally intensive neural network operations (forward/backward passes)
    # CPUs handle: Data loading, preprocessing, and moving data to GPU memory
    # Since the CPU is freed from heavy computation, more CPU cores can be dedicated to data loading without creating bottlenecks.

    # 2. Why Cap at Half the CPUs (available_cpus // 2)?
    # Reserve CPU cores: Even with GPUs doing computation, the main training process still needs CPU resources for:
    # Orchestrating the training loop
    # Memory management
    # GPU-CPU data transfers
    # Other system processes
    # Prevent over-subscription: Using all CPU cores for data loading could starve the main process

    # 3. Why Maximum of 4 Workers?
    # Diminishing returns: Beyond 4 workers, additional parallelism often doesn't improve performance due to:
    # I/O bottlenecks (disk/network speed limits)
    # Memory bandwidth limitations
    # Context switching overhead
    # Memory considerations: Each worker process consumes memory; too many can cause out-of-memory errors

    # 4. Practical Example
    # On a system with 16 CPU cores and GPUs:
    # available_cpus // 2 = 8
    # min(4, 8) = 4 workers
    # Result: 4 workers for data loading, 12 cores available for main process and system
    # This strikes an optimal balance between data loading speed and resource availability for the training process.
    optimal_workers = min(4, available_cpus // 2) if gpu_count > 0 else min(2, available_cpus // 4)
    persistent_workers = optimal_workers > 0
    pin_memory = torch.cuda.is_available()
    prefetch_factor = 2 if optimal_workers > 0 else None
    log_message(f"""🔧 DataLoader optimization: available_cpus={available_cpus}, num_workers={optimal_workers}, 
                pin_memory={pin_memory}, prefetch_factor={prefetch_factor}""")
    
    # Enhanced DataLoaders with performance optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=optimal_workers,
        pin_memory=pin_memory,  # Faster CPU-GPU transfers
        prefetch_factor=prefetch_factor,  # Prefetch batches
        persistent_workers=persistent_workers  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=optimal_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=optimal_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )

    return train_loader, val_loader, test_loader



# =============================
# PART 3 — Dataset Loading, Training, and Evaluation
# =============================
def load_split(path, show_samples=True, n_samples=3):
    """
    Load dataset split from parquet file with optional sample display.
    
    Args:
        path: Path to the parquet file
        show_samples: Whether to display first/last few records (default: True)
        n_samples: Number of first/last samples to show (default: 3)
        
    Returns:
        DataFrame with reset index
    """
    log_message(f"📂 Loading dataset from: {path}")
    df = pd.read_parquet(path)
    df = df.reset_index(drop=True)
    
    log_message(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")
    log_message(f"📊 Columns: {list(df.columns)}")
    
    if show_samples and len(df) > 0:
        log_message(f"🔍 First {min(n_samples, len(df))} records:")
        for i in range(min(n_samples, len(df))):
            record = df.iloc[i]
            # Show key fields in a compact format
            generator = record.get('generator', 'N/A')
            code_preview = str(record.get('code', 'N/A'))[:100] + "..." if len(str(record.get('code', ''))) > 100 else str(record.get('code', 'N/A'))
            log_message(f"  [{i}] Generator: {generator}")
            log_message(f"      Code: {code_preview}")
            if 'family' in record:
                log_message(f"      Family: {record['family']}")
        
        if len(df) > n_samples:
            log_message(f"🔍 Last {min(n_samples, len(df))} records:")
            start_idx = max(0, len(df) - n_samples)
            for i in range(start_idx, len(df)):
                record = df.iloc[i]
                # Show key fields in a compact format
                generator = record.get('generator', 'N/A')
                code_preview = str(record.get('code', 'N/A'))[:100] + "..." if len(str(record.get('code', ''))) > 100 else str(record.get('code', 'N/A'))
                log_message(f"  [{i}] Generator: {generator}")
                log_message(f"      Code: {code_preview}")
                if 'family' in record:
                    log_message(f"      Family: {record['family']}")
    
    return df


def sample_for_balanced_size(df, target_total, seed=42):
    humans = df[df["generator"].str.lower() == human_label.lower()]
    machines = df[df["generator"].str.lower() != human_label.lower()]

    n_machine = len(machines)
    n_human = target_total - n_machine
    if n_human < 0:
        raise ValueError(f"Target {target_total} smaller than machine count {n_machine}")

    sampled_humans = humans.sample(n=n_human, random_state=seed)
    balanced = pd.concat([sampled_humans, machines]).sample(frac=1, random_state=seed)
    return balanced.reset_index(drop=True)

def sample_for_balanced_size_strict(df, seed=42):
    """
    Balance all classes (human and all machine generators) based on the smallest class count.
    
    Args:
        df: DataFrame with 'generator' and 'family' columns
        target_total: Ignored - kept for backward compatibility
        seed: Random seed for sampling
        
    Returns:
        Balanced DataFrame where all classes have the same count as the smallest class
    """
    # Get class counts for all families (human + all machine types)
    class_counts = df['family'].value_counts()
    
    # Find the minimum count across all classes
    min_count = class_counts.min()
    
    log_message(f"📊 Class distribution before balancing:")
    for family, count in class_counts.items():
        log_message(f"   {family}: {count}")
    
    log_message(f"🎯 Balancing all classes to minimum count: {min_count}")
    
    # Sample from each class to match the minimum count
    balanced_dfs = []
    for family in class_counts.index:
        family_df = df[df['family'] == family]
        
        if len(family_df) >= min_count:
            # Sample down to min_count
            sampled_df = family_df.sample(n=min_count, random_state=seed)
        else:
            # This shouldn't happen since min_count is the minimum, but handle it
            sampled_df = family_df
            
        balanced_dfs.append(sampled_df)
        log_message(f"   Sampled {len(sampled_df)} from {family} (original: {len(family_df)})")
    
    # Combine all balanced classes and shuffle
    balanced = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=seed)
    
    # Verify the balancing
    final_counts = balanced['family'].value_counts()
    log_message(f"✅ Final balanced distribution:")
    for family, count in final_counts.items():
        log_message(f"   {family}: {count}")
    
    log_message(f"📈 Total samples: {len(balanced)} (was {len(df)})")
    
    return balanced.reset_index(drop=True)

def mark_seen(df, seen_authors):
    def _mark(row):
        if row["generator"].lower() == human_label:
            return "seen"
        return "seen" if row["generator"] in seen_authors else "unseen"
    df["seen"] = df.apply(_mark, axis=1)
    return df


# =============================
# Family Normalization
# =============================
def normalize_family(gen):
    """
    Normalize generator family names to standard labels.
    
    Args:
        gen: Generator name (e.g., "meta-llama/CodeLlama-7b", "Human", "gpt-4o-mini")
        
    Returns:
        Normalized family name
    """
    if gen.lower() == human_label:
        return human_label
    
    fam = gen.split("/")[0]
    fam_lower = fam.lower()
    
    # Custom mappings for specific patterns
    if fam_lower.startswith("gpt-4o"):
        return "openai"
    elif fam_lower.startswith("mistral"):
        return "mistral"
    elif fam_lower.startswith("gemma") or fam_lower.startswith("google"):
        return "gemma"
    elif fam_lower.startswith("microsoft"):
        return "phi"
    elif fam_lower.startswith("deepseek"):
        return "deepseek"
    return fam_lower



# =============================
# Load Dataset
# =============================
data_path = "/kaggle/input/semeval-2026-task13/SemEval-2026-Task13/task_b"

train_df_full = load_split(os.path.join(data_path, "task_b_training_set.parquet"))
val_df_full = load_split(os.path.join(data_path, "task_b_validation_set.parquet"))
test_df_full = load_split(os.path.join(data_path, "task_b_test_set_sample.parquet"))

log_message("Loaded train, val, test full")

for split in [train_df_full, val_df_full, test_df_full]:
    split["family"] = split["generator"].apply(normalize_family)

log_message("Added family column to train, val, test full")

# train_df = train_df_full
# val_df = val_df_full
# test_df = test_df_full
# balance classes to fix class imbalance and also use f1 macro for eval 
# Conditional dataset sizing based on TRIAL_RUN flag
if TRIAL_RUN:
    # Quick trial run with small datasets for testing
    train_df = train_df_full[:70]
    val_df = val_df_full[:15]
    test_df = test_df_full[:10]
    log_message("🧪 TRIAL_RUN mode: Using small datasets for quick testing")
else:
    # Full training with balanced datasets
    # train_df = sample_for_balanced_size(train_df_full, target_total=70000)
    # val_df = sample_for_balanced_size(val_df_full, target_total=15000)
    # test_df = test_df_full
    train_df = sample_for_balanced_size_strict(train_df_full)
    val_df = sample_for_balanced_size_strict(val_df_full)
    test_df = test_df_full

    log_message("🚀 FULL_RUN mode: Using balanced datasets for complete training")

total_train, total_val, total_test = len(train_df_full), len(val_df_full), len(test_df_full)
filtered_train, filtered_val, filtered_test = len(train_df), len(val_df), len(test_df)
log_message(f"Loaded {filtered_train} out of {total_train} train, {filtered_val} out of {total_val} val, {filtered_test} out of {total_test} test examples from parquet dataset")

seen_authors = set(train_df[train_df["generator"] != "Human"]["generator"])
train_df = mark_seen(train_df, seen_authors)
val_df = mark_seen(val_df, seen_authors)
test_df = mark_seen(test_df, seen_authors)


# =============================
# Label Mapping
# =============================
for split in [train_df, val_df, test_df]:
    split["label_id"] = split["family"].map(label2id)


# =============================
# Prepare Data
# =============================
train_codes = train_df["code"].tolist()
val_codes = val_df["code"].tolist()
test_codes = test_df["code"].tolist()

# train_stylo_features = np.array([extract_stylometric_features(c) for c in train_codes])
# val_stylo_features = np.array([extract_stylometric_features(c) for c in val_codes])
# test_stylo_features = np.array([extract_stylometric_features(c) for c in test_codes])

# Paths to cached stylometric files
#cache_path = "/kaggle/working"
cache_path = "/kaggle/input/stylo-features"
train_cache = f"{cache_path}/train_stylo.npy"
val_cache = f"{cache_path}/val_stylo.npy"
test_cache = f"{cache_path}/test_stylo.npy"

# Compute or load
train_stylo_features = compute_and_cache_stylo_features(train_df, train_cache)
val_stylo_features = compute_and_cache_stylo_features(val_df, val_cache)
test_stylo_features = compute_and_cache_stylo_features(test_df, test_cache)

stylo_dim = train_stylo_features.shape[1]
log_message("Finished extracting stylometric features")


# =============================
# Dataloaders & Model
# =============================
train_labels = train_df["label_id"].tolist()
val_labels = val_df["label_id"].tolist()
test_labels = test_df["label_id"].tolist()

per_cpu_gpu_batch_size = 16
# Optimize batch size based on available GPUs
if gpu_count > 1:
    # With 2 GPUs, we can use larger batch sizes
    optimal_batch_size = per_cpu_gpu_batch_size * gpu_count  # 16 per GPU (32 total)
    log_message(f"🚀 Multi-GPU detected: Using batch_size={optimal_batch_size} (split across {gpu_count} GPUs)")
else:
    # Single GPU or CPU
    optimal_batch_size = 16
    log_message(f"📱 Single device: Using batch_size={optimal_batch_size}")

train_loader, val_loader, test_loader = get_dataloaders(
    train_codes, train_labels, train_stylo_features,
    val_codes, val_labels, val_stylo_features,
    test_codes, test_labels, test_stylo_features,
    batch_size=optimal_batch_size
)
log_message("Finished train val test loaders creation")

base_model = AutoModel.from_pretrained(MODEL_NAME)
hidden_size = base_model.config.hidden_size
num_heads = 8  # must divide hidden_size evenly
#num_labels=len(set(train_labels))
#num_labels should match the final number of unique families — ensure this is passed consistently:
num_labels = len(label2id)

model = CodeBERTAuthorshipClassifier(
    model_name=MODEL_NAME,
    num_labels=num_labels,
    stylo_dim=stylo_dim,
    hidden_size=hidden_size,
    num_heads=num_heads
)

# Enable multi-GPU training if available
if gpu_count > 1:
    log_message(f"🚀 Enabling DataParallel for {gpu_count} GPUs")
    model = torch.nn.DataParallel(model)
    log_message(f"✅ Model wrapped with DataParallel")

log_message("Finished model creation")


# Optimize gradient accumulation for multi-GPU training
optimal_accumulation_steps = 1
if gpu_count > 1:
    # With multiple GPUs, we can reduce accumulation steps since effective batch size is already larger
    optimal_accumulation_steps = 1
    log_message(f"🔧 Multi-GPU: Using accumulation_steps={optimal_accumulation_steps}")
else:
    # Single GPU might benefit from gradient accumulation for larger effective batch size
    optimal_accumulation_steps = 2
    log_message(f"🔧 Single GPU: Using accumulation_steps={optimal_accumulation_steps} for larger effective batch size")

# Conditional epoch configuration based on TRIAL_RUN flag
# Increased from 1 to 10 for proper training
epochs = 1 if TRIAL_RUN else 10

model, training_history = train_model(
    model=model,
    dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=epochs,  
    lr=5e-5,
    device=device,
    checkpoint_dir="/kaggle/working/cpts",
    accumulation_steps=optimal_accumulation_steps,
    use_amp=True  # Enable mixed precision training
)

# Log training completion with history summary
log_message("🎉 Training completed successfully!")
log_message(f"📊 Training History Summary:")
log_message(f"   Final train loss: {training_history['train_loss'][-1]:.4f}")
log_message(f"   Final train accuracy: {training_history['train_acc'][-1]:.4f}")
log_message(f"   Final validation F1-macro: {training_history['val_f1_macro'][-1]:.4f}")
log_message(f"   Final validation accuracy: {training_history['val_acc'][-1]:.4f}")
log_message(f"   Total epochs completed: {len(training_history['train_loss'])}")


# =============================
# Evaluation
# =============================
def detailed_report(labels, preds, split_name):
    log_message(f"\n=== Detailed Report for {split_name} ===")
    all_labels = sorted(id2label.keys())
    target_names = [id2label[i] for i in all_labels]

    log_message(classification_report(
        labels, preds, labels=all_labels,
        target_names=target_names, zero_division=0
    ))

    cm = confusion_matrix(labels, preds, labels=list(id2label.keys()))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d",
                xticklabels=id2label.values(),
                yticklabels=id2label.values(),
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {split_name}")
    plt.show()


def predict_model(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            stylo_features = batch["stylo_features"].to(device)
            y = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stylo_features=stylo_features
            )
            logits = outputs["logits"]
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    return np.array(labels), np.array(preds)


# =============================
# Run Final Evaluation
# =============================
log_message("\nRunning Final Test Evaluation...")
y_true, y_pred = predict_model(model, test_loader, device)

detailed_report(y_true, y_pred, "Test (Overall)")

seen_mask = (test_df["seen"].values == "seen")
unseen_mask = (test_df["seen"].values == "unseen")

seen_true, seen_pred = y_true[seen_mask], y_pred[seen_mask]
unseen_true, unseen_pred = y_true[unseen_mask], y_pred[unseen_mask]

detailed_report(seen_true, seen_pred, "Test (Seen Authors)")
detailed_report(unseen_true, unseen_pred, "Test (Unseen Authors)")
