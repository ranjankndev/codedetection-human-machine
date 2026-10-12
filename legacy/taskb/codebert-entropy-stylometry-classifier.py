# =============================
# INSTALL DEPENDENCIES
# =============================
!pip install radon pytz
!pip install -U "transformers==4.44.2"

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
FIXME: handle na 
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_message(f"device: {device}")

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
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden_states, entropy_bias=None):
        q = k = v = hidden_states
        attn_output, attn_weights = self.attn(q, k, v, need_weights=True)
        if entropy_bias is not None:
            bias = self.alpha * entropy_bias.unsqueeze(-1)
            attn_weights = attn_weights + bias
        return attn_output, attn_weights

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
    def __init__(self, model_name=MODEL_NAME, num_labels=12, stylo_dim=20, hidden_size=768, num_heads=8,
                 enable_entropy=True, enable_style_features=True):
        super().__init__()
        self.enable_entropy = enable_entropy
        self.enable_style_features = enable_style_features
        self.stylo_dim = stylo_dim
        
        # Core encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Conditional components based on flags
        if self.enable_style_features:
            self.proj_stylo = nn.Linear(stylo_dim, hidden_size)
            self.stylo_regressor = nn.Linear(hidden_size, stylo_dim)
        
        if self.enable_entropy:
            self.entropy_attn = EntropyWeightedSelfAttention(hidden_size, num_heads=num_heads)
        
        # Determine classifier input size based on enabled features
        classifier_input_size = hidden_size
        if self.enable_style_features:
            classifier_input_size += hidden_size
            
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(classifier_input_size, num_labels)

    def forward(self, input_ids, attention_mask, stylo_features=None, labels=None, stylo_targets=None):
        # Encode input
        enc_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Apply entropy-weighted attention if enabled
        if self.enable_entropy:
            entropy = token_entropy(enc_outputs.last_hidden_state)
            attn_out, _ = self.entropy_attn(enc_outputs.last_hidden_state, entropy_bias=entropy)
            pooled_output = attn_out.mean(dim=1)
        else:
            # Simple mean pooling without entropy weighting
            pooled_output = enc_outputs.last_hidden_state.mean(dim=1)
        
        # Handle stylometric features if enabled
        if self.enable_style_features:
            if stylo_features is None:
                raise ValueError("stylo_features must be provided when enable_style_features=True")
            stylo_proj = self.proj_stylo(stylo_features)
            combined = torch.cat([pooled_output, stylo_proj], dim=-1)
        else:
            combined = pooled_output
        
        # Final classification
        combined = self.dropout(combined)
        class_logits = self.classifier(combined)
        
        # Stylometric regression (only if style features enabled)
        stylo_pred = None
        if self.enable_style_features:
            stylo_pred = self.stylo_regressor(pooled_output)
        
        # Compute loss
        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(class_logits, labels)
            
            if self.enable_style_features and stylo_targets is not None:
                mse_loss = F.mse_loss(stylo_pred, stylo_targets)
                loss = ce_loss + 0.3 * mse_loss
            else:
                loss = ce_loss
                
        return {"loss": loss, "logits": class_logits, "stylo_pred": stylo_pred}

# =============================
# TRAINING FUNCTION
# =============================
def train_model(model, dataloader, val_dataloader, epochs, lr, device, log_interval=200, patience=3,
                enable_entropy=True, enable_style_features=True, checkpoint_dir=None, save_checkpoints=True,
                accumulation_steps=1):
    log_message(f"✅ Started training with Early Stopping (entropy={enable_entropy}, style={enable_style_features})")
    
    # Setup checkpoint directory with timestamp
    if save_checkpoints:
        if checkpoint_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = f"/kaggle/working/cpts/training_{timestamp}"
        
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            log_message(f"📁 Checkpoint directory ready: {checkpoint_dir}")
        except Exception as e:
            log_message(f"⚠️ Failed to create checkpoint directory: {e}")
            save_checkpoints = False
    
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    model.to(device)
    best_f1 = 0.0
    patience_counter = 0
    best_model_state = None
    training_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_preds, train_labels = [], []
        start_time = time.time()

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Prepare inputs based on model configuration
            forward_kwargs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            
            if enable_style_features:
                forward_kwargs["stylo_features"] = batch["stylo_features"].to(device)
                forward_kwargs["stylo_targets"] = batch["stylo_targets"].to(device)
            
            outputs = model(**forward_kwargs)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
            # Collect predictions for training metrics
            with torch.no_grad():
                preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                train_preds.extend(preds)
                train_labels.extend(labels)

            # Log progress every few batches or every ~minute
            if (step + 1) % log_interval == 0 or (time.time() - start_time) > 60:
                elapsed = time.time() - start_time
                avg_loss = total_loss / (step + 1)
                eta = (len(dataloader) - step - 1) * (elapsed / (step + 1))
                
                # Calculate training metrics for logged steps
                if len(train_preds) > 0:
                    train_acc = np.mean(np.array(train_preds) == np.array(train_labels))
                    train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
                    log_message(f"[Epoch {epoch+1}/{epochs}] Step {step+1}/{len(dataloader)} "
                                f"| Loss: {avg_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f} "
                                f"| ETA: {eta/60:.1f} min")
                else:
                    log_message(f"[Epoch {epoch+1}/{epochs}] Step {step+1}/{len(dataloader)} "
                                f"| Loss: {avg_loss:.4f} | ETA: {eta/60:.1f} min")
                
                sys.stdout.flush()
                start_time = time.time()  # reset timer after each log

        avg_loss = total_loss / len(dataloader)
        
        # Calculate training F1-macro for the entire epoch
        train_f1_macro = f1_score(train_labels, train_preds, average='macro', zero_division=0)
        train_accuracy = np.mean(np.array(train_preds) == np.array(train_labels))
        
        log_message(f"✅ Epoch {epoch+1}/{epochs} completed. Train Loss: {avg_loss:.4f}, "
                   f"Train Acc: {train_accuracy:.4f}, Train F1: {train_f1_macro:.4f}")

        # ===== Validation =====
        model.eval()
        all_preds, all_labels = [], []
        val_start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                # Prepare validation inputs
                val_kwargs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                
                if enable_style_features:
                    val_kwargs["stylo_features"] = batch["stylo_features"].to(device)
                
                outputs = model(**val_kwargs)
                preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
                
                # Log validation progress every few batches or every ~minute
                if (batch_idx + 1) % log_interval == 0 or (time.time() - val_start_time) > 60:
                    elapsed = time.time() - val_start_time
                    eta = (len(val_dataloader) - batch_idx - 1) * (elapsed / (batch_idx + 1))
                    
                    # Calculate current validation metrics
                    if len(all_preds) > 0:
                        current_val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
                        current_val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
                        log_message(f"[Epoch {epoch+1}/{epochs}] Validation Step {batch_idx+1}/{len(val_dataloader)} "
                                    f"| Acc: {current_val_acc:.4f} | F1: {current_val_f1:.4f} "
                                    f"| ETA: {eta/60:.1f} min")
                    else:
                        log_message(f"[Epoch {epoch+1}/{epochs}] Validation Step {batch_idx+1}/{len(val_dataloader)} "
                                    f"| ETA: {eta/60:.1f} min")
                    
                    sys.stdout.flush()
                    val_start_time = time.time()  # reset timer after each log

        f1_macro = f1_score(all_labels, all_preds, average='macro')
        log_message(f"📊 Validation F1-macro: {f1_macro:.4f}")
        sys.stdout.flush()

        # Record training history
        epoch_history = {
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_f1_macro': train_f1_macro,
            'train_accuracy': train_accuracy,
            'val_f1_macro': f1_macro,
            'timestamp': datetime.now().isoformat()
        }
        training_history.append(epoch_history)

        # ===== Early Stopping Check =====
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            log_message(f"💾 New best F1-macro: {f1_macro:.4f} — model saved.")
            
            # Save best model checkpoint
            if save_checkpoints and checkpoint_dir:
                best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                best_checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_f1': best_f1,
                    'training_history': training_history,
                    'model_config': {
                        'model_name': MODEL_NAME,
                        'num_labels': len(label2id),
                        'stylo_dim': model.stylo_dim if hasattr(model, 'stylo_dim') else 20,
                        'hidden_size': model.encoder.config.hidden_size,
                        'enable_entropy': enable_entropy,
                        'enable_style_features': enable_style_features
                    },
                    'training_config': {
                        'lr': lr,
                        'epochs': epochs,
                        'patience': patience,
                        'log_interval': log_interval
                    }
                }
                
                try:
                    # Also save a timestamped best model
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    timestamped_path = os.path.join(checkpoint_dir, f'best_model_{timestamp}_epoch_{epoch+1}.pt')
                    torch.save(best_checkpoint, timestamped_path)
                    log_message(f"📅 Timestamped best model saved to {timestamped_path}")
                    
                except Exception as e:
                    log_message(f"⚠️ Failed to save best model checkpoint: {e}")
        else:
            patience_counter += 1
            log_message(f"⚠️ No improvement. Early stopping patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                log_message("🛑 Early stopping triggered.")
                if best_model_state:
                    model.load_state_dict(best_model_state)
                break

    # Load best model before returning
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save final training summary
    if save_checkpoints and checkpoint_dir:
        summary_path = os.path.join(checkpoint_dir, 'training_summary.txt')
        try:
            with open(summary_path, 'w') as f:
                f.write("=== Training Summary ===\n")
                f.write(f"Best F1-macro: {best_f1:.4f}\n")
                f.write(f"Total epochs: {len(training_history)}\n")
                f.write(f"Final patience counter: {patience_counter}\n")
                f.write(f"Enable entropy: {enable_entropy}\n")
                f.write(f"Enable style features: {enable_style_features}\n")
                f.write(f"Learning rate: {lr}\n")
                f.write(f"Training completed at: {datetime.now().isoformat()}\n\n")
                
                f.write("=== Epoch History ===\n")
                for hist in training_history:
                    f.write(f"Epoch {hist['epoch']}: Loss={hist['train_loss']:.4f}, "
                           f"Train F1={hist['train_f1_macro']:.4f}, "
                           f"Val F1={hist['val_f1_macro']:.4f}, Time={hist['timestamp']}\n")
            
            log_message(f"📋 Training summary saved to {summary_path}")
        except Exception as e:
            log_message(f"⚠️ Failed to save training summary: {e}")

    log_message("✅ Training completed.")
    return model


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

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

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
    train_df = sample_for_balanced_size(train_df_full, target_total=70000)
    val_df = sample_for_balanced_size(val_df_full, target_total=15000)
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

# Configuration flags for ablation studies
ENABLE_ENTROPY = True
ENABLE_STYLE_FEATURES = False

# Conditional stylometric feature computation
if ENABLE_STYLE_FEATURES:
    # Paths to cached stylometric files
    cache_path = "/kaggle/working"  # Use writable directory
    train_cache = f"{cache_path}/train_stylo.npy"
    val_cache = f"{cache_path}/val_stylo.npy"
    test_cache = f"{cache_path}/test_stylo.npy"

    # Compute or load
    train_stylo_features = compute_and_cache_stylo_features(train_df, train_cache)
    val_stylo_features = compute_and_cache_stylo_features(val_df, val_cache)
    test_stylo_features = compute_and_cache_stylo_features(test_df, test_cache)

    stylo_dim = train_stylo_features.shape[1]
    log_message("Finished extracting stylometric features")
else:
    # Create dummy stylometric features when disabled
    stylo_dim = 20  # default dimension
    train_stylo_features = np.zeros((len(train_df), stylo_dim), dtype=np.float32)
    val_stylo_features = np.zeros((len(val_df), stylo_dim), dtype=np.float32)
    test_stylo_features = np.zeros((len(test_df), stylo_dim), dtype=np.float32)
    log_message("Skipped stylometric feature extraction (enable_style_features=False)")


# =============================
# Dataloaders & Model
# =============================
train_labels = train_df["label_id"].tolist()
val_labels = val_df["label_id"].tolist()
test_labels = test_df["label_id"].tolist()

train_loader, val_loader, test_loader = get_dataloaders(
    train_codes, train_labels, train_stylo_features,
    val_codes, val_labels, val_stylo_features,
    test_codes, test_labels, test_stylo_features,
    batch_size=16
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
    num_labels=len(label2id),  # Use total number of families, not just those in train set
    stylo_dim=stylo_dim,
    hidden_size=hidden_size,
    num_heads=num_heads,
    enable_entropy=ENABLE_ENTROPY,
    enable_style_features=ENABLE_STYLE_FEATURES
)
log_message("Finished model creation")

# Conditional epoch configuration based on TRIAL_RUN flag
# Increased from 1 to 10 for proper training
epochs = 1 if TRIAL_RUN else 5

model = train_model(
    model=model,
    dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=epochs,
    lr=5e-5,
    device=device,
    enable_entropy=ENABLE_ENTROPY, 
    enable_style_features=ENABLE_STYLE_FEATURES
)


# =============================
# Evaluation
# =============================
def detailed_report(labels, preds, split_name):
    log_message(f"\n=== Detailed Report for {split_name} ===")
    all_labels = sorted(id2label.keys())
    target_names = [id2label[i] for i in all_labels]

    print(classification_report(
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
