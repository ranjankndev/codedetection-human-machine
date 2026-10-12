import os
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime
import pytz

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


def log_message(message):
    """Enhanced logging function with IST timestamp"""
    ist = pytz.timezone("Asia/Kolkata")
    timestamp = datetime.now(ist).strftime("%d-%m-%Y %H:%M:%S %Z")
    print(f"[{timestamp}] {message}")

# =============================
# PART 3 — Dataset Loading, Training, and Evaluation
# =============================
def load_split(path):
    df = pd.read_parquet(path)
    return df.reset_index(drop=True)


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


def summarize_split(df, split_name="train"):
    """
    Enhanced function to summarize dataset split with comprehensive logging.
    
    Args:
        df: DataFrame to summarize
        split_name: Name of the split (train, validation, test)
        label2id: Optional dictionary mapping family names to label IDs for validation
    """
    total = len(df)
    human = (df["generator"].str.lower() == human_label.lower()).sum()
    machine = total - human

    log_message(f"=== {split_name.upper()} SPLIT SUMMARY ===")
    log_message(f"Total examples: {total}")
    log_message(f"Human examples: {human} ({human/total*100:.1f}%)")
    log_message(f"Machine examples: {machine} ({machine/total*100:.1f}%)")

    # Enhanced seen/unseen statistics logging
    if "seen" in df.columns:
        seen_counts = Counter(df["seen"])
        log_message("Seen/Unseen distribution:")
        
        total_seen_unseen = sum(seen_counts.values())
        for status, count in seen_counts.items():
            percentage = (count / total_seen_unseen) * 100 if total_seen_unseen > 0 else 0
            log_message(f"  {status.capitalize():6s}: {count:6d} ({percentage:.1f}%)")
        
        # Additional breakdown by generator type for seen/unseen
        if len(seen_counts) > 1:  # Only if we have both seen and unseen
            log_message("Seen/Unseen breakdown by generator type:")
            
            # Human breakdown
            human_seen = len(df[(df["generator"].str.lower() == human_label.lower()) & (df["seen"] == "seen")])
            human_total = len(df[df["generator"].str.lower() == human_label.lower()])
            if human_total > 0:
                log_message(f"  Human - Seen: {human_seen}/{human_total} ({human_seen/human_total*100:.1f}%)")
            
            # Machine breakdown
            machine_seen = len(df[(df["generator"].str.lower() != human_label.lower()) & (df["seen"] == "seen")])
            machine_unseen = len(df[(df["generator"].str.lower() != human_label.lower()) & (df["seen"] == "unseen")])
            machine_total = machine_seen + machine_unseen
            if machine_total > 0:
                log_message(f"  Machine - Seen: {machine_seen}/{machine_total} ({machine_seen/machine_total*100:.1f}%)")
                log_message(f"  Machine - Unseen: {machine_unseen}/{machine_total} ({machine_unseen/machine_total*100:.1f}%)")

    if machine > 0:
        if "family" not in df.columns:
            #df["family"] = df["generator"].apply(lambda g: g.split("/")[0] if isinstance(g, str) else unknown_label)
            df["family"] = df["generator"].apply(lambda g: g.split("/")[0])
        if "author" not in df.columns:
            df["author"] = df["generator"]

        # Overall machine statistics
        machine_df = df[df["generator"].str.lower() != human_label.lower()]
        fam_counts = Counter(machine_df["family"])
        log_message(f"All Machine families ({len(fam_counts)} unique):")
        for fam, count in fam_counts.most_common():
            percentage = (count / machine) * 100
            log_message(f"  {fam:15s}: {count:6d} ({percentage:.1f}%)")

        fam_author_counts = defaultdict(Counter)
        for _, row in machine_df.iterrows():
            fam_author_counts[row["family"]][row["author"]] += 1

        log_message("All Machine - Top authors per family:")
        for fam, authors in fam_author_counts.items():
            total_authors = len(authors)
            top_5 = authors.most_common(5)
            top_str = ", ".join([f"{a} ({c})" for a, c in top_5])
            log_message(f"  {fam:15s} ({total_authors} authors): {top_str}")

        # Seen vs Unseen breakdown if "seen" column exists
        if "seen" in df.columns:
            # Seen machine authors
            seen_machine_df = machine_df[machine_df["seen"] == "seen"]
            if len(seen_machine_df) > 0:
                seen_fam_counts = Counter(seen_machine_df["family"])
                log_message(f"Seen Machine families ({len(seen_fam_counts)} unique):")
                for fam, count in seen_fam_counts.most_common():
                    percentage = (count / len(seen_machine_df)) * 100
                    log_message(f"  {fam:15s}: {count:6d} ({percentage:.1f}%)")

                seen_fam_author_counts = defaultdict(Counter)
                for _, row in seen_machine_df.iterrows():
                    seen_fam_author_counts[row["family"]][row["author"]] += 1

                log_message("Seen Machine - Top authors per family:")
                for fam, authors in seen_fam_author_counts.items():
                    total_authors = len(authors)
                    top_5 = authors.most_common(5)
                    top_str = ", ".join([f"{a} ({c})" for a, c in top_5])
                    log_message(f"  {fam:15s} ({total_authors} authors): {top_str}")

            # Unseen machine authors
            unseen_machine_df = machine_df[machine_df["seen"] == "unseen"]
            if len(unseen_machine_df) > 0:
                unseen_fam_counts = Counter(unseen_machine_df["family"])
                log_message(f"Unseen Machine families ({len(unseen_fam_counts)} unique):")
                for fam, count in unseen_fam_counts.most_common():
                    percentage = (count / len(unseen_machine_df)) * 100
                    log_message(f"  {fam:15s}: {count:6d} ({percentage:.1f}%)")

                unseen_fam_author_counts = defaultdict(Counter)
                for _, row in unseen_machine_df.iterrows():
                    unseen_fam_author_counts[row["family"]][row["author"]] += 1

                log_message("Unseen Machine - Top authors per family:")
                for fam, authors in unseen_fam_author_counts.items():
                    total_authors = len(authors)
                    top_5 = authors.most_common(5)
                    top_str = ", ".join([f"{a} ({c})" for a, c in top_5])
                    log_message(f"  {fam:15s} ({total_authors} authors): {top_str}")
    
    # Check for families not present in label2id dictionary
    if label2id is not None and machine > 0:
        log_message("Label Mapping Validation:")
        
        # Get all unique families in the dataset
        all_families = set(machine_df["family"].unique())
        
        # Get families defined in label2id
        mapped_families = set(label2id.keys())
        
        # Find families in data but not in label2id
        unmapped_families = all_families - mapped_families
        
        if unmapped_families:
            log_message(f"⚠️  WARNING: Families in data but NOT in label2id ({len(unmapped_families)}):")
            for family in sorted(unmapped_families):
                count = len(machine_df[machine_df["family"] == family])
                log_message(f"    {family}: {count} examples")
        else:
            log_message("✅ All families in data are present in label2id mapping")
    
    log_message("=" * 50)  # Separator line for readability


def save_balanced_dataset(train_df, val_df, test_df, train_original_size=None, val_original_size=None,
                         test_original_size=None, output_dir="balanced_datasets", file_format="parquet"):
    """
    Save the class balanced datasets to disk with target size and percentage in filename.
    
    Args:
        train_df (pd.DataFrame): Balanced training dataset
        val_df (pd.DataFrame): Balanced validation dataset
        test_df (pd.DataFrame): Test dataset
        train_original_size (int): Original size of training dataset before balancing
        val_original_size (int): Original size of validation dataset before balancing
        test_original_size (int): Original size of test dataset before balancing
        output_dir (str): Directory to save the datasets
        file_format (str): Format to save files ('parquet' or 'csv')
    
    Returns:
        dict: Paths to saved files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = {}
    
    # Define file extension based on format
    ext = "parquet" if file_format == "parquet" else "csv"
    
    # Calculate size suffixes for train and val
    def get_size_suffix(current_size, original_size):
        if original_size is None:
            return ""
        
        # Format target total (e.g., 70000 -> 70k, 15000 -> 15k)
        if current_size >= 1000:
            size_str = f"{current_size // 1000}k"
        else:
            size_str = str(current_size)
        
        # Calculate percentage
        percentage = round((current_size / original_size) * 100)
        
        return f"_{size_str}_{percentage}pc"
    
    # Save each split
    splits_info = {
        "train": (train_df, train_original_size),
        "val": (val_df, val_original_size),
        "test": (test_df, test_original_size)
    }
    
    for split_name, (df, original_size) in splits_info.items():
        size_suffix = get_size_suffix(len(df), original_size)
        filename = f"task_b_{split_name}_balanced{size_suffix}.{ext}"
        filepath = os.path.join(output_dir, filename)
        
        # Save based on format
        if file_format == "parquet":
            df.to_parquet(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)
            
        saved_paths[split_name] = filepath
        
        # Log with percentage info if available
        if original_size:
            percentage = round((len(df) / original_size) * 100)
            log_message(f"Saved {split_name} split ({len(df)} examples, {percentage}% of original {original_size}) to {filepath}")
        else:
            log_message(f"Saved {split_name} split ({len(df)} examples) to {filepath}")
    
    # Generate dynamic summary filename with size and percentage info
    def get_summary_suffix(current_size, original_size):
        if original_size is None:
            # For test dataset or when no original size provided
            if current_size >= 1000:
                size_str = f"{current_size // 1000}k"
            else:
                size_str = str(current_size)
            return f"{size_str}_100pc"
        
        # Format size (e.g., 70000 -> 70k, 15000 -> 15k)
        if current_size >= 1000:
            size_str = f"{current_size // 1000}k"
        else:
            size_str = str(current_size)
        
        # Calculate percentage
        percentage = round((current_size / original_size) * 100)
        
        return f"{size_str}_{percentage}pc"
    
    # Build summary filename
    summary_parts = []
    for split_name, (df, original_size) in splits_info.items():
        suffix = get_summary_suffix(len(df), original_size)
        summary_parts.append(f"{split_name}_{suffix}")
    
    summary_filename = f"dataset_summary_{'_'.join(summary_parts)}.txt"
    summary_path = os.path.join(output_dir, summary_filename)
    
    with open(summary_path, "w") as f:
        f.write("=== BALANCED DATASET SUMMARY ===\n\n")
        for split_name, (df, _) in splits_info.items():
            f.write(f"{split_name.upper()} SPLIT:\n")
            f.write(f"  Total examples: {len(df)}\n")
            f.write(f"  Human examples: {(df['generator'].str.lower() == human_label.lower()).sum()}\n")
            f.write(f"  Machine examples: {(df['generator'].str.lower() != human_label.lower()).sum()}\n")
            
            if "family" in df.columns:
                fam_counts = Counter(df[df["generator"].str.lower() != human_label.lower()]["family"])
                f.write(f"  Machine families: {len(fam_counts)}\n")
                for fam, count in fam_counts.most_common(5):
                    f.write(f"    {fam}: {count}\n")
            f.write("\n")
    
    saved_paths["summary"] = summary_path
    log_message(f"Saved dataset summary to {summary_path}")
    
    return saved_paths


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

train_df = train_df_full
val_df = val_df_full
test_df = test_df_full
seen_authors = set(train_df[train_df["generator"] != "Human"]["generator"])
train_df = mark_seen(train_df, seen_authors)
val_df = mark_seen(val_df, seen_authors)
test_df = mark_seen(test_df, seen_authors)
summarize_split(train_df, "train")
summarize_split(val_df, "validation")
summarize_split(test_df, "test")

train_df = sample_for_balanced_size(train_df_full, target_total=70000)
val_df = sample_for_balanced_size(val_df_full, target_total=15000)
test_df = test_df_full
# train_df = train_df_full[:70]
# val_df = val_df_full[:15]
# test_df = test_df_full[:10]

total_train, total_val, total_test = len(train_df_full), len(val_df_full), len(test_df_full)
filtered_train, filtered_val, filtered_test = len(train_df), len(val_df), len(test_df)
log_message(f"Loaded {filtered_train} out of {total_train} train, {filtered_val} out of {total_val} val, {filtered_test} out of {total_test} test examples from parquet dataset")

def mark_seen(df, seen_authors):
    def _mark(row):
        if row["generator"].lower() == human_label:
            return "seen"
        return "seen" if row["generator"] in seen_authors else "unseen"
    df["seen"] = df.apply(_mark, axis=1)
    return df

seen_authors = set(train_df[train_df["generator"] != "Human"]["generator"])
train_df = mark_seen(train_df, seen_authors)
val_df = mark_seen(val_df, seen_authors)
test_df = mark_seen(test_df, seen_authors)

# # =============================
# # Save Balanced Datasets
# # =============================
# # Save the balanced datasets with original size information
# saved_files = save_balanced_dataset(train_df, val_df, test_df,
#                                    train_original_size=total_train,
#                                    val_original_size=total_val,
#                                    test_original_size=total_test)
# log_message(f"All balanced datasets saved successfully: {list(saved_files.keys())}")

# =============================
# Dataset Summary
# =============================
# Display detailed statistics for each split
summarize_split(train_df, "train")
summarize_split(val_df, "validation")
summarize_split(test_df, "test")

train_df = sample_for_balanced_size_strict(train_df_full)
val_df = sample_for_balanced_size_strict(val_df_full)
test_df = test_df_full
seen_authors = set(train_df[train_df["generator"] != "Human"]["generator"])
train_df = mark_seen(train_df, seen_authors)
val_df = mark_seen(val_df, seen_authors)
test_df = mark_seen(test_df, seen_authors)

# # =============================
# # Save Balanced Datasets
# # =============================
# # Save the balanced datasets with original size information
# saved_files = save_balanced_dataset(train_df, val_df, test_df,
#                                    train_original_size=total_train,
#                                    val_original_size=total_val,
#                                    test_original_size=total_test)
# log_message(f"All balanced datasets saved successfully: {list(saved_files.keys())}")

# =============================
# Dataset Summary
# =============================
# Display detailed statistics for each split
summarize_split(train_df, "train")
summarize_split(val_df, "validation")
summarize_split(test_df, "test")
