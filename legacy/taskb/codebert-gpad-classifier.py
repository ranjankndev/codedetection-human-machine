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
def train_model(model, dataloader, val_dataloader, epochs, lr, device, log_interval=200, patience=3):
    log_message("✅ Started training with Early Stopping")
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    model.to(device)
    best_f1 = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                stylo_features=batch["stylo_features"].to(device),
                labels=batch["labels"].to(device),
                stylo_targets=batch["stylo_targets"].to(device),
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            # Log progress every few batches or every ~minute
            if (step + 1) % log_interval == 0 or (time.time() - start_time) > 60:
                elapsed = time.time() - start_time
                avg_loss = total_loss / (step + 1)
                eta = (len(dataloader) - step - 1) * (elapsed / (step + 1))
                log_message(f"[Epoch {epoch+1}/{epochs}] Step {step+1}/{len(dataloader)} "
                            f"| Loss: {avg_loss:.4f} | ETA: {eta/60:.1f} min")
                sys.stdout.flush()
                start_time = time.time()  # reset timer after each log

        avg_loss = total_loss / len(dataloader)
        log_message(f"✅ Epoch {epoch+1}/{epochs} completed. Train Loss: {avg_loss:.4f}")

        # ===== Validation =====
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    stylo_features=batch["stylo_features"].to(device),
                )
                preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        f1_macro = f1_score(all_labels, all_preds, average='macro')
        log_message(f"📊 Validation F1-macro: {f1_macro:.4f}")
        sys.stdout.flush()

        # ===== Early Stopping Check =====
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model_state = model.state_dict()
            patience_counter = 0
            log_message("💾 Improved F1-macro — model saved.")
        else:
            patience_counter += 1
            log_message(f"⚠️ No improvement. Early stopping patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                log_message("🛑 Early stopping triggered.")
                if best_model_state:
                    model.load_state_dict(best_model_state)
                return model

    # Load best model before returning
    if best_model_state:
        model.load_state_dict(best_model_state)

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
def load_split(path):
    df = pd.read_parquet(path)
    return df.reset_index(drop=True)


def sample_for_balanced_size(df, target_total, seed=42):
    humans = df[df["generator"] == "Human"]
    machines = df[df["generator"] != "Human"]

    n_machine = len(machines)
    n_human = target_total - n_machine
    if n_human < 0:
        raise ValueError(f"Target {target_total} smaller than machine count {n_machine}")

    sampled_humans = humans.sample(n=n_human, random_state=seed)
    balanced = pd.concat([sampled_humans, machines]).sample(frac=1, random_state=seed)
    return balanced.reset_index(drop=True)


def mark_seen(df, seen_authors):
    def _mark(row):
        if row["generator"] == "Human":
            return "seen"
        return "seen" if row["generator"] in seen_authors else "unseen"
    df["seen"] = df.apply(_mark, axis=1)
    return df


def summarize_split(df, split_name="train"):
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


# =============================
# Family Normalization
# =============================
mapping = {
    "deepseek-ai": "DeepSeek-AI",
    "qwen": "Qwen",
    "01-ai": "01-ai",
    "bigcode": "BigCode",
    "gemma": "Gemma",
    "meta-llama": "Meta-LLaMA",
    "ibm-granite": "IBM-Granite",
    "mistral": "Mistral",
}

def normalize_family(gen):
    if gen == "Human":
        return "Human"
    fam = gen.split("/")[0]
    fam_lower = fam.lower()
    for key, val in mapping.items():
        if key in fam_lower:
            return val
    return "Unknown"


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

# train_df = sample_for_balanced_size(train_df_full, target_total=70000)
# val_df = sample_for_balanced_size(val_df_full, target_total=15000)
# test_df = test_df_full
# no need to balance as we're using f1 macro to account for class imbalance
train_df = train_df_full[:7000]
val_df = val_df_full[:1500]
test_df = test_df_full[:1000]

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
families = ["Human"] + sorted(set(mapping.values())) + ["Unknown"]
label2id = {fam: i for i, fam in enumerate(families)}
id2label = {i: fam for fam, i in label2id.items()}

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
    num_labels=len(set(train_labels)),
    stylo_dim=stylo_dim,
    hidden_size=hidden_size,
    num_heads=num_heads
)
log_message("Finished model creation")


model = train_model(
    model=model,
    dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=3,
    lr=5e-5,
    device=device
)


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

