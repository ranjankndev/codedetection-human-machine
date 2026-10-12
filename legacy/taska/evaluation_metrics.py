# evaluation_metrics.py
"""
Simplified metrics computation - Only F1, Precision, Recall (Macro)
"""
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def compute_extended_metrics(y_true, logits, label_names):
    """
    Compute essential metrics: F1 (Macro), Precision (Macro), Recall (Macro)
    
    Args:
        y_true: True labels
        logits: Model output logits
        label_names: List of label names
    
    Returns:
        dict: Contains macro_f1, macro_precision, macro_recall
    """
    # Get predictions from logits
    y_pred = np.argmax(logits, axis=1) if logits.ndim > 1 else logits
    
    # Ensure numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate macro-averaged metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, 
        y_pred, 
        labels=list(range(len(label_names))),
        average='macro',
        zero_division=0
    )
    
    # Return only the 3 essential metrics
    metrics = {
        "macro_f1": float(f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall)
    }
    
    return metrics


def save_metrics_artifacts(out_dir, split_name, metrics):
    """
    Save metrics to JSON file
    
    Args:
        out_dir: Output directory
        split_name: Name of the split (e.g., 'validation', 'test')
        metrics: Dictionary of metrics
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    # Save as JSON
    metrics_file = f"{out_dir}/metrics_{split_name}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Saved metrics to {metrics_file}")
