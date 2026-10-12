!pip install radon

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, get_linear_schedule_with_warmup
from torch.optim import AdamW

import ast
import keyword
import radon.complexity as rcc
import radon.metrics as rmetrics
import tokenize
import io
import numpy as np

def extract_stylometric_features(code, stylo_dim=20):
    import radon.complexity as rcc
    import radon.metrics as rmetrics
    import radon.raw as rraw
    import numpy as np

    features = []

    try:
        # Raw metrics
        raw = rraw.analyze(code)
        features.extend([
            raw.loc, raw.lloc, raw.comments, raw.multi, raw.blank
        ])
    except Exception:
        features.extend([0.0]*5)

    try:
        # Cyclomatic complexity
        cc_list = rcc.cc_visit(code)
        if cc_list:
            cc_vals = [c.complexity for c in cc_list]
            features.extend([
                np.mean(cc_vals),
                np.max(cc_vals),
                len(cc_vals)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
    except Exception:
        features.extend([0.0, 0.0, 0.0])

    try:
        # Halstead metrics
        hal = rmetrics.h_visit(code)
        features.extend([
            hal.total.operators, hal.total.operands,
            hal.total.distinct_operators, hal.total.distinct_operands,
            hal.total.volume
        ])
    except Exception:
        features.extend([0.0]*5)

    # Pad or truncate to fixed size
    while len(features) < stylo_dim:
        features.append(0.0)
    features = features[:stylo_dim]

    return np.array(features, dtype=np.float32)

def _max_ast_depth(node, depth=0):
    """Recursive AST depth calculator."""
    if not isinstance(node, ast.AST) or not list(ast.iter_child_nodes(node)):
        return depth
    return max(_max_ast_depth(child, depth + 1) for child in ast.iter_child_nodes(node))


# =============================
# Utility: Entropy Computation
# =============================
def token_entropy(logits):
    """
    Compute token-level entropy given logits.
    logits: [batch, seq_len, vocab_size]
    returns: [batch, seq_len]
    """
    probs = F.softmax(logits, dim=-1)  # [B, L, V]
    log_probs = torch.log(probs + 1e-12)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, L]
    return entropy


# =============================
# Custom Entropy-Weighted Self-Attention
# =============================
class EntropyWeightedSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # trainable entropy scaling

    def forward(self, hidden_states, entropy_bias=None):
        # hidden_states: [B, L, H]
        # entropy_bias: [B, L] -> optional external entropy
        q = k = v = hidden_states
        attn_output, attn_weights = self.attn(q, k, v, need_weights=True)

        if entropy_bias is not None:
            # Add entropy bias to attention weights before softmax
            bias = self.alpha * entropy_bias.unsqueeze(-1)  # [B, L, 1]
            attn_weights = attn_weights + bias

        return attn_output, attn_weights


# =============================
# Generative Stylometry Cross-Attention
# =============================
class StylometryCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, stylo_dim):
        super().__init__()
        self.stylo_proj = nn.Linear(stylo_dim, hidden_size)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

    def forward(self, decoder_states, encoder_states, stylo_features):
        # stylo_features: [B, stylo_dim]
        stylo_proj = self.stylo_proj(stylo_features)  # [B, H]
        stylo_proj = stylo_proj.unsqueeze(1).expand(-1, encoder_states.size(1), -1)  # [B, L, H]

        encoder_aug = encoder_states + stylo_proj
        output, attn_weights = self.cross_attn(decoder_states, encoder_aug, encoder_aug)
        return output, attn_weights


# =============================
# Full Integrated Model
# =============================
class CodeT5AuthorshipClassifier(nn.Module):
    def __init__(self, model_name, num_labels, stylo_dim, hidden_size=768, num_heads=12):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Replace decoder self-attention with entropy-weighted variant
        self.ew_self_attn = EntropyWeightedSelfAttention(hidden_size, num_heads)

        # Add stylometry cross-attention
        self.stylo_cross_attn = StylometryCrossAttention(hidden_size, num_heads, stylo_dim)

        # Classification head
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Auxiliary regression head (multi-task)
        self.stylo_regressor = nn.Linear(hidden_size, stylo_dim)


    def forward(self, input_ids, attention_mask, decoder_input_ids=None,
                stylo_features=None, labels=None, stylo_targets=None):
    
        # If decoder_input_ids not provided (e.g., during predict), use padding
        if decoder_input_ids is None:
            decoder_input_ids = torch.zeros_like(input_ids)
    
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden = encoder_outputs.last_hidden_state  # [B, L, H]
    
        # Decoder initial embeddings
        decoder_embeddings = self.model.decoder.embed_tokens(decoder_input_ids)
    
        # Compute entropy bias using encoder logits
        encoder_logits = self.model.lm_head(encoder_hidden)  # [B, L, V]
        entropy_bias = token_entropy(encoder_logits)  # [B, L]
    
        # Apply Entropy-Weighted Self-Attention
        ew_out, _ = self.ew_self_attn(decoder_embeddings, entropy_bias=entropy_bias)
    
        # Cross-Attention with stylometry
        cross_out, _ = self.stylo_cross_attn(ew_out, encoder_hidden, stylo_features)
    
        # Pooling
        pooled = cross_out.mean(dim=1)  # [B, H]
    
        # Classification
        class_logits = self.classifier(pooled)
    
        # Auxiliary stylometry regression
        stylo_pred = self.stylo_regressor(pooled)
    
        loss = None
        if labels is not None and stylo_targets is not None:
            ce_loss = F.cross_entropy(class_logits, labels)
            mse_loss = F.mse_loss(stylo_pred, stylo_targets)
            loss = ce_loss + 0.3 * mse_loss
    
        return {
            "loss": loss,
            "logits": class_logits,
            "stylo_pred": stylo_pred
        }

# =============================
# Training Function
# =============================
def train_model(model, dataloader, val_dataloader, epochs, lr, device):
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                decoder_input_ids=batch["decoder_input_ids"].to(device),
                stylo_features=batch["stylo_features"].to(device),
                labels=batch["labels"].to(device),
                stylo_targets=batch["stylo_targets"].to(device),
            )

            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_dataloader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    decoder_input_ids=batch["decoder_input_ids"].to(device),
                    stylo_features=batch["stylo_features"].to(device),
                )
                preds = torch.argmax(outputs["logits"], dim=-1)
                correct += (preds == batch["labels"].to(device)).sum().item()
                total += batch["labels"].size(0)

        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")
    return model

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch

class CodeAuthorshipDataset(Dataset):
    def __init__(self, codes, labels, stylo_features, tokenizer_name="Salesforce/codet5-small", max_length=256):
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

        # Tokenize decoder input (shifted code, here we reuse input code tokens)
        dec = self.tokenizer(
            code,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "decoder_input_ids": dec["input_ids"].squeeze(0),
            "stylo_features": stylo_feat,
            "labels": torch.tensor(label, dtype=torch.long),
            "stylo_targets": stylo_feat,  # supervised regression
        }


def get_dataloaders(train_codes, train_labels, train_stylo,
                    val_codes, val_labels, val_stylo,
                    test_codes, test_labels, test_stylo,
                    batch_size=16, max_length=256,
                    tokenizer_name="Salesforce/codet5-small"):
    # Train dataset
    train_dataset = CodeAuthorshipDataset(train_codes, train_labels, train_stylo,
                                          tokenizer_name=tokenizer_name, max_length=max_length)
    # Validation dataset
    val_dataset = CodeAuthorshipDataset(val_codes, val_labels, val_stylo,
                                        tokenizer_name=tokenizer_name, max_length=max_length)
    # Test dataset
    test_dataset = CodeAuthorshipDataset(test_codes, test_labels, test_stylo,
                                         tokenizer_name=tokenizer_name, max_length=max_length)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

import pytz
from datetime import datetime

def log_message(*args):
    msg = " ".join(str(a) for a in args)
    ist = pytz.timezone("Asia/Kolkata")
    timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{timestamp}] {msg}")
    
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from collections import Counter, defaultdict

# ==========================
# Dataset utilities
# ==========================
def load_split(path):
    """Load a parquet split."""
    df = pd.read_parquet(path)
    return df.reset_index(drop=True)



def sample_for_balanced_size(df, target_total, seed=42):
    """
    Downsample Human examples so that total = target_total 
    (keeping all Machine examples).
    """
    # Split human and machine
    humans = df[df["generator"] == "Human"]
    machines = df[df["generator"] != "Human"]

    # How many humans to keep?
    n_machine = len(machines)
    n_human = target_total - n_machine
    if n_human < 0:
        raise ValueError(f"Target {target_total} is smaller than machine count {n_machine}")

    # Sample humans
    sampled_humans = humans.sample(n=n_human, random_state=seed)
    balanced = pd.concat([sampled_humans, machines]).sample(frac=1, random_state=seed)
    return balanced.reset_index(drop=True)

def mark_seen(df, seen_authors):
    """Mark whether each row is seen/unseen (Humans always seen)."""
    def _mark(row):
        if row["generator"] == "Human":
            return "seen"
        return "seen" if row["generator"] in seen_authors else "unseen"
    df["seen"] = df.apply(_mark, axis=1)
    return df

def summarize_split(df, split_name="train"):
    """Print stats for a split: total, human/machine, families, authors, seen/unseen."""
    total = len(df)
    human = (df["generator"] == "Human").sum()
    machine = total - human

    print(f"\n=== {split_name.upper()} SPLIT ===")
    print(f"Total examples : {total}")
    print(f"Human examples : {human}")
    print(f"Machine examples: {machine}")

    if "seen" in df.columns:
        seen_counts = Counter(df["seen"])
        print(f"\nSeen/Unseen distribution:")
        for k, v in seen_counts.items():
            print(f"  {k:6s}: {v}")

    if machine > 0:
        if "family" not in df.columns:
            df["family"] = df["generator"].apply(lambda g: g.split("/")[0] if isinstance(g, str) else "UNKNOWN")
        if "author" not in df.columns:
            df["author"] = df["generator"]

        fam_counts = Counter(df[df["generator"] != "Human"]["family"])
        print("\nMachine families:")
        for fam, count in fam_counts.most_common():
            print(f"  {fam:15s}: {count}")

        fam_author_counts = defaultdict(Counter)
        for _, row in df[df["generator"] != "Human"].iterrows():
            fam_author_counts[row["family"]][row["author"]] += 1

        print("\nTop authors per family:")
        for fam, authors in fam_author_counts.items():
            top = ", ".join([f"{a} ({c})" for a, c in authors.most_common(5)])
            print(f"  {fam:15s}: {top}")

mapping = {
    "deepseek-ai": "DeepSeek-AI",
    "qwen": "Qwen",
    "01-ai": "01-ai",
    "bigcode": "BigCode",
    "gemma": "Gemma",
    #"phi": "Phi",
    "meta-llama": "Meta-LLaMA",
    "ibm-granite": "IBM-Granite",
    "mistral": "Mistral",
    #"openai": "OpenAI"
}
    
# ==========================
# Family Normalization
# ==========================
def normalize_family(gen):
    if gen == "Human":
        return "Human"
    fam = gen.split("/")[0]
    fam_lower = fam.lower()
    for key, val in mapping.items():
        if key in fam_lower:
            return val
    return "Unknown"

# ==========================
# Load dataset
# ==========================
data_path = "/kaggle/input/semeval-2026-task13/SemEval-2026-Task13/task_b"

train_df_full = load_split(os.path.join(data_path, "task_b_training_set.parquet"))
val_df_full   = load_split(os.path.join(data_path, "task_b_validation_set.parquet"))
test_df_full  = load_split(os.path.join(data_path, "task_b_test_set_sample.parquet"))

log_message("loaded train val test full")
# Normalize to families
for split in [train_df_full, val_df_full, test_df_full]:
    split["family"] = split["generator"].apply(normalize_family)
log_message("added family col to train val test full")

# Train: keep ~70k (57904 machine + 13096 human)
train_df = sample_for_balanced_size(train_df_full, target_total=70000)

# Val: keep ~15k (11510 machine + 3490 human)
val_df = sample_for_balanced_size(val_df_full, target_total=15000)
test_df = test_df_full

total_train, total_val, total_test = len(train_df_full), len(val_df_full), len(test_df_full)
filtered_train, filtered_val, filtered_test = len(train_df), len(val_df), len(test_df)
log_message(f"Loaded {filtered_train} out of {total_train} train, {filtered_val} out of {total_val} val, {filtered_test} out of {total_test} test examples from parquet dataset")

# Mark seen/unseen (based on train generators)
seen_authors = set(train_df[train_df["generator"] != "Human"]["generator"])
train_df = mark_seen(train_df, seen_authors)
val_df   = mark_seen(val_df,   seen_authors)
test_df  = mark_seen(test_df,  seen_authors)

# ==========================
# Label mapping
# ==========================
# Build families list
families = ["Human"] + sorted(set(mapping.values())) + ["Unknown"]

label2id = {fam: i for i, fam in enumerate(families)}
id2label = {i: fam for fam, i in label2id.items()}

for split in [train_df, val_df, test_df]:
    split["label_id"] = split["family"].map(label2id)

import numpy as np

# Extract code snippets
train_labels = train_df["label_id"].tolist()
val_labels   = val_df["label_id"].tolist()
test_labels  = test_df["label_id"].tolist()

# Extract code snippets
train_codes = train_df["code"].tolist()
val_codes   = val_df["code"].tolist()
test_codes  = test_df["code"].tolist()

# # Placeholder stylometric features (random or precomputed)
# # TODO: replace with actual computed stylometric features
# stylo_dim = 20
# train_stylo_features = np.random.rand(len(train_codes), stylo_dim)
# val_stylo_features   = np.random.rand(len(val_codes), stylo_dim)
# test_stylo_features  = np.random.rand(len(test_codes), stylo_dim)

train_stylo_features = np.array([extract_stylometric_features(c) for c in train_codes])
val_stylo_features   = np.array([extract_stylometric_features(c) for c in val_codes])
test_stylo_features  = np.array([extract_stylometric_features(c) for c in test_codes])

stylo_dim = train_stylo_features.shape[1]
log_message("finished extract_stylometric_features ")


# Example usage
train_loader, val_loader, test_loader = get_dataloaders(
    train_codes, train_labels, train_stylo_features,
    val_codes, val_labels, val_stylo_features,
    test_codes, test_labels, test_stylo_features,
    batch_size=16
)

from transformers import AutoModelForSeq2SeqLM

base_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")
hidden_size = base_model.config.d_model  # 512 for codet5-small
num_heads = 8  # 8 * 64 = 512 Using 12 heads will also throw an error, because 512/12 is not integer

model = CodeT5AuthorshipClassifier(
    model_name="Salesforce/codet5-small",
    num_labels=len(set(train_labels)),
    stylo_dim=train_stylo_features.shape[1],  # e.g. 20 stylometric features
    hidden_size=hidden_size,
    num_heads=num_heads
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = train_model(
    model=model,
    dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=5,
    lr=5e-5,
    device=device
)

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def detailed_report(labels, preds, split_name):
    log_message(f"\n=== Detailed Report for {split_name} ===")
    all_labels = sorted(id2label.keys())
    target_names = [id2label[i] for i in all_labels]
    log_message(
        classification_report(
            labels,
            preds,
            labels=all_labels,
            target_names=target_names,
            zero_division=0
        )
    )
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

            logits = outputs["logits"]   # ✅ extract logits from dict
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    return np.array(labels), np.array(preds)


log_message("\nRunning Final Test Evaluation...")

y_true, y_pred = predict_model(model, test_loader, device)

# Overall report
detailed_report(y_true, y_pred, "Test (Overall)")

# Seen vs Unseen subsets
seen_mask = (test_df["seen"].values == "seen")
unseen_mask = (test_df["seen"].values == "unseen")

seen_true, seen_pred = y_true[seen_mask], y_pred[seen_mask]
unseen_true, unseen_pred = y_true[unseen_mask], y_pred[unseen_mask]

detailed_report(seen_true, seen_pred, "Test (Seen Authors)")
detailed_report(unseen_true, unseen_pred, "Test (Unseen Authors)")

