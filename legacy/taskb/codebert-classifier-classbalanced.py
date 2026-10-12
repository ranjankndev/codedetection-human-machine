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

def sample_balanced(df, n_human, n_machine, seed=42):
    """Sample balanced human and machine data."""
    humans = df[df["generator"] == "Human"]
    machines = df[df["generator"] != "Human"]

    sampled_humans = (
        humans.sample(n=min(n_human, len(humans)), random_state=seed)
        if len(humans) > 0 else pd.DataFrame(columns=df.columns)
    )
    sampled_machines = (
        machines.sample(n=min(n_machine, len(machines)), random_state=seed)
        if len(machines) > 0 else pd.DataFrame(columns=df.columns)
    )

    balanced = pd.concat([sampled_humans, sampled_machines]).sample(frac=1, random_state=seed)
    return balanced.reset_index(drop=True)

def sample_per_label(df, label_col="family", n_per_label=100, seed=42):
    """Sample exactly n_per_label per label from df, if available."""
    sampled = []
    for label, group in df.groupby(label_col):
        if len(group) == 0:
            continue
        n = min(n_per_label, len(group))
        sampled.append(group.sample(n=n, random_state=seed))
    return pd.concat(sampled).sample(frac=1, random_state=seed).reset_index(drop=True)

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

# Balanced subsampling
# train_df = sample_balanced(train_df_full, n_human=4000, n_machine=4000)
# val_df   = sample_balanced(val_df_full,   n_human=500,  n_machine=500)
# test_df  = sample_balanced(test_df_full,  n_human=500,  n_machine=500)
# train_df = sample_balanced(train_df_full, n_human=8, n_machine=8)
# val_df   = sample_balanced(val_df_full,   n_human=1,  n_machine=1)
# test_df  = sample_balanced(test_df_full,  n_human=1,  n_machine=1)

# Balanced sampling
# train_df = sample_per_label(train_df_full, n_per_label=1000)  # 12k
# val_df   = sample_per_label(val_df_full,   n_per_label=100)   # 1.2k
# test_df  = sample_per_label(test_df_full,  n_per_label=100)   # 1.2k

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

# Summaries
summarize_split(train_df, "train")
summarize_split(val_df, "val")
summarize_split(test_df, "test")

# ==========================
# Label mapping
# ==========================
# families = ["Human", "DeepSeek-AI", "Qwen", "01-ai", "BigCode", "Gemma",
#            "Phi", "Meta-LLaMA", "IBM-Granite", "Mistral", "OpenAI", "Unknown"]
# Build families list
families = ["Human"] + sorted(set(mapping.values())) + ["Unknown"]

label2id = {fam: i for i, fam in enumerate(families)}
id2label = {i: fam for fam, i in label2id.items()}

for split in [train_df, val_df, test_df]:
    split["label_id"] = split["family"].map(label2id)

# ==========================
# Torch Dataset
# ==========================
class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts = df["code"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ==========================
# Trainer
# ==========================

from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}
    
def train_and_eval(train_df, val_df, exp_name):
    # ==========================
    # Model
    # ==========================
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    train_dataset = CodeDataset(train_df, tokenizer)
    val_dataset   = CodeDataset(val_df, tokenizer)
    test_dataset = CodeDataset(test_df, tokenizer)

    training_args = TrainingArguments(
        output_dir=f"./results/{exp_name}",
        num_train_epochs=5, 
        eval_strategy="epoch",
        save_strategy="epoch",
        #logging_strategy="epoch",
        logging_strategy="steps", 
        logging_steps=100,
        #per_device_train_batch_size=8,
        #per_device_eval_batch_size=8,
        per_device_train_batch_size=32,   
        per_device_eval_batch_size=64,
        learning_rate=3e-5,
        weight_decay=0.01, 
        warmup_ratio=0.06,
        fp16=True,
        load_best_model_at_end=True,
        #metric_for_best_model="accuracy",
        metric_for_best_model="f1_macro",        # 🔹 choose metric
        greater_is_better=True,
        save_total_limit=3,                      # 🔹 keep only top 3 models
        report_to="none"                 
    )


    #with Trainer, no need .to(device) — it is GPU aware and handles device placement.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # 🔹 stop after 3 no-improve epochs
    )

    trainer.train()
    results = trainer.evaluate()
    log_message(f"\n=== {exp_name} RESULTS ===")
    log_message(results)
    return trainer, test_dataset

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def detailed_report(labels, preds, split_name):
    log_message(f"\n=== Detailed Report for {split_name} ===")
    # Ensure consistent label ordering
    all_labels = sorted(id2label.keys())
    target_names = [id2label[i] for i in all_labels]
    #log_message(classification_report(labels, preds, target_names=[id2label[i] for i in sorted(id2label.keys())]))
    # Full classification report (keeps all classes, even if missing in preds)
    log_message(
        classification_report(
            labels,
            preds,
            labels=all_labels,          # ✅ Force all labels to be considered
            target_names=target_names,  # ✅ Names aligned with labels
            zero_division=0
        )
    )
    # Confusion Matrix
    cm = confusion_matrix(labels, preds, labels=list(id2label.keys()))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d", xticklabels=id2label.values(), yticklabels=id2label.values(), cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {split_name}")
    plt.show()

# ==========================
# Run experiments
# ==========================
trainer, test_dataset = train_and_eval(train_df, val_df, "train_and_eval")

# Final test evaluation
log_message("\nRunning Final Test Evaluation...")

preds = trainer.predict(test_dataset)
log_message(f"\n=== Final Test RESULTS ===")
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=-1)

# Overall test metrics
detailed_report(y_true, y_pred, "Test (Overall)")

# Seen vs Unseen subsets
#seen_mask = test_df["seen"] == "seen"
seen_mask = (test_df["seen"].values == "seen")

#unseen_mask = test_df["seen"] == "unseen"
unseen_mask = (test_df["seen"].values == "unseen")

seen_true, seen_pred = y_true[seen_mask], y_pred[seen_mask]
unseen_true, unseen_pred = y_true[unseen_mask], y_pred[unseen_mask]

detailed_report(seen_true, seen_pred, "Test (Seen Authors)")
detailed_report(unseen_true, unseen_pred, "Test (Unseen Authors)")

