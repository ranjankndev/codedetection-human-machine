"""
model_loader.py
Compatible model loading for GPAD V2 and CodeBERT baseline
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gpad_v2_model import GPADModel, GPADConfig
import json
import os


def load_gpad(model_path, device="cuda"):
    """
    Load GPAD V2 model from checkpoint.
    
    Args:
        model_path: Path to saved model directory
        device: 'cuda' or 'cpu'
        
    Returns:
        model, tokenizer
    """
    print(f"Loading GPAD V2 model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load training config to reconstruct model
    config_path = os.path.join(model_path, "training_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            train_config = json.load(f)
        
        # Create GPAD config
        gpad_config = GPADConfig(
            base_model=train_config.get('base_model', 'microsoft/codebert-base'),
            num_labels=train_config.get('num_labels', 2),
            stylometric_dim=train_config.get('stylometric_dim', 32),
            num_gpad_layers=train_config.get('num_gpad_layers', 1),
            entropy_alpha=train_config.get('entropy_alpha', 0.15),
            max_length=train_config.get('max_length', 512),
            use_stylometric=train_config.get('use_stylometric', True),
            classifier_hidden_dim=train_config.get('classifier_hidden_dim', 384),
            use_gradient_checkpointing=False  # Disable for inference
        )
    else:
        # Default config if training_config.json not found
        print("⚠️  Warning: training_config.json not found, using defaults")
        gpad_config = GPADConfig(
            base_model='microsoft/codebert-base',
            num_labels=2,
            stylometric_dim=32,
            num_gpad_layers=1,
            entropy_alpha=0.15,
            use_gradient_checkpointing=False
        )
    
    # Load model
    model = GPADModel(gpad_config)
    
    # Load weights
    model_weights_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(model_weights_path):
        state_dict = torch.load(model_weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        # Try loading from model.safetensors
        print("⚠️  pytorch_model.bin not found, trying from_pretrained...")
        model = GPADModel.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    print(f"✅ GPAD V2 model loaded successfully")
    return model, tokenizer


def load_base(model_path, device="cuda"):
    """
    Load CodeBERT baseline model (fine-tuned for classification).
    
    Args:
        model_path: Path to saved model directory or HF model name
        device: 'cuda' or 'cpu'
        
    Returns:
        model, tokenizer
    """
    print(f"Loading CodeBERT baseline from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    print(f"✅ CodeBERT baseline loaded successfully")
    return model, tokenizer


def load_codebert_pretrained(num_labels=2, device="cuda"):
    """
    Load fresh CodeBERT model (not fine-tuned) for baseline comparison.
    
    Args:
        num_labels: Number of output classes
        device: 'cuda' or 'cpu'
        
    Returns:
        model, tokenizer
    """
    print(f"Loading pre-trained CodeBERT (microsoft/codebert-base)")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base",
        num_labels=num_labels,
        ignore_mismatched_sizes=True  # Allow classifier head initialization
    )
    
    model.to(device)
    model.eval()
    
    print(f"⚠️  Note: This is a fresh model without fine-tuning")
    return model, tokenizer