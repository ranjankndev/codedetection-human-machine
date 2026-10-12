"""
inference.py
Complete inference and evaluation script for GPAD V2 and baselines
Compatible with both GPAD and CodeBERT models
"""

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    accuracy_score,
    classification_report,
    confusion_matrix
)
from model_loader import load_gpad, load_base, load_codebert_pretrained
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def predict_batch(model, tokenizer, texts, device, max_length=512, batch_size=32, model_type="gpad"):
    """
    Predict with batching for efficiency.
    
    Args:
        model: GPAD or CodeBERT model
        tokenizer: Tokenizer
        texts: List of code snippets
        device: 'cuda' or 'cpu'
        max_length: Maximum sequence length
        batch_size: Batch size for inference
        model_type: 'gpad' or 'base'
        
    Returns:
        predictions: numpy array of predicted labels
        probabilities: numpy array of class probabilities
    """
    model.eval()
    all_preds = []
    all_probs = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        
        # Predict
        with torch.no_grad():
            if model_type == "gpad":
                # GPAD model needs code_texts
                outputs = model(
                    input_ids=enc['input_ids'],
                    attention_mask=enc['attention_mask'],
                    code_texts=batch_texts
                )
            else:
                # Standard HF model
                outputs = model(**enc)
            
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(-1)
        
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_probs)


def evaluate(model_path, test_file, model_type="gpad", batch_size=32, max_length=512):
    """
    Evaluate model on test set with comprehensive metrics.
    
    Args:
        model_path: Path to saved model or HF model name
        test_file: Path to test parquet/csv file
        model_type: 'gpad' or 'base'
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    if model_type == "gpad":
        model, tokenizer = load_gpad(model_path, device=device)
    else:
        model, tokenizer = load_base(model_path, device=device)
    
    # Load test data
    print(f"\nLoading test data from: {test_file}")
    if test_file.endswith('.parquet'):
        df = pd.read_parquet(test_file)
    elif test_file.endswith('.csv'):
        df = pd.read_csv(test_file)
    else:
        raise ValueError("Test file must be .parquet or .csv")
    
    print(f"Test samples: {len(df)}")
    
    # Check for required columns
    if 'code' not in df.columns:
        raise ValueError("Test file must have 'code' column")
    
    # Prepare labels
    if 'label' in df.columns:
        y_true = df['label'].values
    elif 'generator' in df.columns:
        # Convert generator to binary labels (Human=0, LLM=1)
        y_true = (df['generator'] != 'Human').astype(int).values
    else:
        raise ValueError("Test file must have 'label' or 'generator' column")
    
    # Get predictions
    texts = df['code'].tolist()
    preds, probs = predict_batch(
        model, tokenizer, texts, device, 
        max_length=max_length, 
        batch_size=batch_size,
        model_type=model_type
    )
    
    # Calculate metrics
    print("\n" + "="*70)
    print(f"📊 EVALUATION RESULTS: {model_path}")
    print("="*70)
    
    accuracy = accuracy_score(y_true, preds)
    f1_macro = f1_score(y_true, preds, average="macro")
    f1_weighted = f1_score(y_true, preds, average="weighted")
    precision_macro = precision_score(y_true, preds, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, preds, average="macro", zero_division=0)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  F1 (Macro):        {f1_macro:.4f}")
    print(f"  F1 (Weighted):     {f1_weighted:.4f}")
    print(f"  Precision (Macro): {precision_macro:.4f}")
    print(f"  Recall (Macro):    {recall_macro:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    unique_labels = np.unique(y_true)
    for label in unique_labels:
        mask = y_true == label
        if mask.sum() > 0:
            class_acc = (preds[mask] == label).mean()
            print(f"  Class {label}: Accuracy = {class_acc:.4f}, Support = {mask.sum()}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_true, preds)
    print(cm)
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    target_names = ['Human', 'LLM'] if len(unique_labels) == 2 else [f'Class_{i}' for i in unique_labels]
    print(classification_report(y_true, preds, target_names=target_names, zero_division=0))
    
    print("="*70 + "\n")
    
    # Return metrics dictionary
    metrics = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'confusion_matrix': cm.tolist(),
        'predictions': preds.tolist(),
        'probabilities': probs.tolist()
    }
    
    return metrics


def evaluate_codebert_baseline(test_file, num_labels=2, batch_size=32, max_length=512):
    """
    Evaluate fresh CodeBERT baseline (without fine-tuning) on test set.
    This is for baseline comparison only.
    
    Args:
        test_file: Path to test file
        num_labels: Number of classes
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load fresh CodeBERT
    model, tokenizer = load_codebert_pretrained(num_labels=num_labels, device=device)
    
    # Load test data
    print(f"\nLoading test data from: {test_file}")
    if test_file.endswith('.parquet'):
        df = pd.read_parquet(test_file)
    elif test_file.endswith('.csv'):
        df = pd.read_csv(test_file)
    else:
        raise ValueError("Test file must be .parquet or .csv")
    
    print(f"Test samples: {len(df)}")
    
    # Prepare labels
    if 'label' in df.columns:
        y_true = df['label'].values
    elif 'generator' in df.columns:
        y_true = (df['generator'] != 'Human').astype(int).values
    else:
        raise ValueError("Test file must have 'label' or 'generator' column")
    
    # Get predictions
    texts = df['code'].tolist()
    preds, probs = predict_batch(
        model, tokenizer, texts, device, 
        max_length=max_length, 
        batch_size=batch_size,
        model_type="base"
    )
    
    # Calculate metrics
    print("\n" + "="*70)
    print(f"📊 CODEBERT BASELINE (No Fine-tuning)")
    print("="*70)
    
    accuracy = accuracy_score(y_true, preds)
    f1_macro = f1_score(y_true, preds, average="macro", zero_division=0)
    precision_macro = precision_score(y_true, preds, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, preds, average="macro", zero_division=0)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  F1 (Macro):        {f1_macro:.4f}")
    print(f"  Precision (Macro): {precision_macro:.4f}")
    print(f"  Recall (Macro):    {recall_macro:.4f}")
    
    print("="*70 + "\n")
    
    return {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro)
    }


def compare_models(test_file, gpad_path=None, baseline_path=None, batch_size=32):
    """
    Compare GPAD V2 vs CodeBERT baseline on same test set.
    
    Args:
        test_file: Path to test file
        gpad_path: Path to trained GPAD model
        baseline_path: Path to trained baseline model
        batch_size: Batch size for inference
    """
    print("\n" + "="*70)
    print("🔬 MODEL COMPARISON")
    print("="*70)
    
    results = {}
    
    if gpad_path:
        print("\n[1/2] Evaluating GPAD V2...")
        results['gpad'] = evaluate(gpad_path, test_file, model_type="gpad", batch_size=batch_size)
    
    if baseline_path:
        print("\n[2/2] Evaluating CodeBERT Baseline...")
        results['baseline'] = evaluate(baseline_path, test_file, model_type="base", batch_size=batch_size)
    
    # Summary comparison
    if len(results) > 1:
        print("\n" + "="*70)
        print("📊 COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Metric':<20} {'GPAD V2':<15} {'Baseline':<15} {'Improvement':<15}")
        print("-"*70)
        
        for metric in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
            gpad_val = results.get('gpad', {}).get(metric, 0)
            base_val = results.get('baseline', {}).get(metric, 0)
            improvement = ((gpad_val - base_val) / base_val * 100) if base_val > 0 else 0
            
            print(f"{metric:<20} {gpad_val:<15.4f} {base_val:<15.4f} {improvement:+.2f}%")
        
        print("="*70 + "\n")
    
    return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Evaluate GPAD V2 model
    # evaluate(
    #     model_path="/workspace/results/both_combined",
    #     test_file="/workspace/data/test.parquet",
    #     model_type="gpad"
    # )
    
    # Example 2: Evaluate CodeBERT baseline
    # evaluate(
    #     model_path="/workspace/results/codebert_baseline",
    #     test_file="/workspace/data/test.parquet",
    #     model_type="base"
    # )
    
    # Example 3: Evaluate fresh CodeBERT (no fine-tuning)
    # evaluate_codebert_baseline(
    #     test_file="/workspace/data/test.parquet",
    #     num_labels=2
    # )
    
    # Example 4: Compare models
    # compare_models(
    #     test_file="/workspace/data/test.parquet",
    #     gpad_path="/workspace/results/both_combined",
    #     baseline_path="/workspace/results/codebert_baseline"
    # )
    
    pass