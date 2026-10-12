"""
gpad_v2_trainer.py
Training pipeline for GPAD V2 (Enhanced Code Authorship Detection)

• Supports Task A (binary) and Task B (multi-class)
• Integrates evaluation_metrics.py for extended metrics
• Uses utils.py for logging + seed + macro-F1 computation
• FIXED: Proper handling of predictions tuple
• FIXED: Memory management during evaluation with custom Trainer
• FIXED: Immediate GPU->CPU transfer to prevent OOM
• FIXED: Accelerate compatibility issue
"""

import os
import json
import gc
import numpy as np
import pandas as pd
import torch

# FIX: Set environment variable to avoid dispatch_batches issue
os.environ['ACCELERATE_USE_FSDP'] = '0'
os.environ['ACCELERATE_USE_DEEPSPEED'] = '0'

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# CRITICAL FIX: Patch Accelerator to remove incompatible parameters
try:
    from accelerate import Accelerator
    original_init = Accelerator.__init__
    
    def patched_init(self, *args, **kwargs):
        # Remove all potentially incompatible parameters
        incompatible_params = [
            'dispatch_batches',
            'even_batches',
            'use_seedable_sampler',
        ]
        for param in incompatible_params:
            kwargs.pop(param, None)
        return original_init(self, *args, **kwargs)
    
    Accelerator.__init__ = patched_init
    print("✅ Applied Accelerator compatibility patch (removed: dispatch_batches, even_batches, use_seedable_sampler)")
except Exception as e:
    print(f"⚠️ Could not patch Accelerator: {e}")


from sklearn.model_selection import train_test_split
from dataclasses import dataclass, asdict
from typing import Optional, List
from datetime import datetime

from gpad_v2_model import GPADModel, GPADConfig
from utils import (
    get_logger,
    set_seed,
    compute_metrics_macro,
    print_classification_report
)
from evaluation_metrics import compute_extended_metrics, save_metrics_artifacts

logger = get_logger("GPAD_V2_Trainer")


# -------------------------------------------------------------------------
# Custom Trainer with Memory Management
# -------------------------------------------------------------------------
class MemoryEfficientTrainer(Trainer):
    """Custom Trainer that clears cache to prevent OOM during evaluation."""
    
    def evaluation_loop(self, *args, **kwargs):
        """Override evaluation loop to clear cache periodically."""
        # Clear cache before evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Run the standard evaluation loop
        output = super().evaluation_loop(*args, **kwargs)
        
        # Clear cache after evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        return output
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction step to immediately move outputs to CPU."""
        # Run standard prediction
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )
        
        # Move to CPU immediately to free GPU memory
        if logits is not None:
            logits = logits.detach().cpu()
        if labels is not None:
            labels = labels.detach().cpu()
        if loss is not None:
            loss = loss.detach().cpu()
            
        return loss, logits, labels


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
@dataclass
class GPADTrainingConfig:
    task: str = "A"
    num_labels: int = 2
    base_model: str = "microsoft/codebert-base"
    stylometric_dim: int = 32
    num_gpad_layers: int = 1
    entropy_alpha: float = 0.15
    max_length: int = 512
    use_stylometric: bool = True
    classifier_hidden_dim: int = 384
    num_epochs: int = 5
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    val_split: float = 0.2
    sample_size: Optional[int] = None
    output_dir: str = "./gpad_v2_results"
    log_interval: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    early_stopping_patience: int = 3
    seed: int = 42
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_private: bool = False


# -------------------------------------------------------------------------
# Dataset helpers
# -------------------------------------------------------------------------
class GPADDataset(torch.utils.data.Dataset):
    """Standard dataset that keeps code_texts for stylometric analysis."""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "code_texts": str(self.texts[idx])
        }


class GPADDataCollator:
    """Custom collator that preserves raw text for stylometric feature extractor."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "code_texts": [b["code_texts"] for b in batch]
        }


# -------------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------------
class GPADTrainer:
    """Main training and evaluation class for GPAD V2."""
    def __init__(self, config: GPADTrainingConfig):
        self.config = config
        set_seed(config.seed)
        logger.info(f"🧩 Initialized trainer for Task {config.task}")

    # -------------------------------
    def _load_dataframe(self, path: str) -> pd.DataFrame:
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            raise ValueError("File must be .parquet or .csv")
        logger.info(f"Loaded {len(df):,} records from {path}")
        return df

    # -------------------------------
    def _prepare_labels(self, df: pd.DataFrame):
        """Prepares text + labels depending on Task A or B."""
        texts = df["code"].tolist()
        task = self.config.task.upper()

        if task == "A":
            labels = [0 if g == "Human" else 1 for g in df["generator"]]
            label_names = ["Human", "LLM"]
        elif task == "B":
            families = df["generator"].apply(lambda g: g.split("/")[0] if "/" in g else g)
            unique = sorted(families.unique())
            label_map = {n: i for i, n in enumerate(unique)}
            labels = [label_map[f] for f in families]
            label_names = unique
        else:
            raise ValueError(f"Invalid task: {task}")

        self.config.num_labels = len(label_names)
        logger.info(f"Task {task}: {self.config.num_labels} classes → {label_names}")
        return texts, labels, label_names

    # -------------------------------
    def train(self, train_file: str, validation_file: Optional[str] = None):
        """Complete training → evaluation → HF Hub (optional) pipeline."""
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)

        df = self._load_dataframe(train_file)
        if self.config.sample_size and self.config.sample_size < len(df):
            df = df.sample(self.config.sample_size, random_state=self.config.seed)
            logger.info(f"Sampled {self.config.sample_size} examples for testing")

        texts, labels, label_names = self._prepare_labels(df)

        # Validation split
        if validation_file:
            val_df = self._load_dataframe(validation_file)
            val_texts, val_labels, _ = self._prepare_labels(val_df)
        else:
            texts, val_texts, labels, val_labels = train_test_split(
                texts, labels,
                test_size=self.config.val_split,
                random_state=self.config.seed,
                stratify=labels
            )
            logger.info(f"Split → Train: {len(texts)}, Val: {len(val_texts)}")
            
            # CRITICAL: Verify validation has all classes
            val_unique = np.unique(val_labels)
            if len(val_unique) != len(label_names):
                raise ValueError(
                    f"❌ VALIDATION ERROR: Validation set only has {len(val_unique)} classes "
                    f"but expected {len(label_names)} classes!\n"
                    f"Your dataset is too small or imbalanced.\n"
                    f"Current size: {len(texts) + len(val_texts)} samples\n"
                    f"Solutions:\n"
                    f"  1. Use a larger dataset\n"
                    f"  2. Remove --sample_size parameter\n"
                    f"  3. Provide a separate --validation_file"
                )

        train_ds = GPADDataset(texts, labels, tokenizer, self.config.max_length)
        val_ds = GPADDataset(val_texts, val_labels, tokenizer, self.config.max_length)

        model_cfg = GPADConfig(
            base_model=self.config.base_model,
            num_labels=self.config.num_labels,
            stylometric_dim=self.config.stylometric_dim,
            num_gpad_layers=self.config.num_gpad_layers,
            entropy_alpha=self.config.entropy_alpha,
            max_length=self.config.max_length,
            use_stylometric=self.config.use_stylometric,
            classifier_hidden_dim=self.config.classifier_hidden_dim,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing
        )
        model = GPADModel(model_cfg)

        # TrainingArguments with compatibility fixes
        args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            logging_steps=self.config.log_interval,
            save_steps=self.config.save_steps,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_total_limit=2,
            fp16=self.config.use_mixed_precision and torch.cuda.is_available(),
            report_to="none",
            load_best_model_at_end=True,
            # Simplified settings for compatibility
            dataloader_num_workers=0,
        )

        trainer = MemoryEfficientTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=GPADDataCollator(tokenizer),
            compute_metrics=lambda p: compute_metrics_macro((p.predictions, p.label_ids), self.config.task),
            callbacks=[EarlyStoppingCallback(self.config.early_stopping_patience)]
        )

        # ---------------- Train ----------------
        logger.info("⚙️ Training started …")
        trainer.train()
        logger.info("🏁 Training complete → evaluating best checkpoint")

        # ---------------- Evaluate ----------------
        eval_output = trainer.predict(val_ds)
        
        # FIXED: Handle predictions (can be tuple or array)
        predictions = eval_output.predictions
        if isinstance(predictions, tuple):
            logits = predictions[0]  # First element is logits
        else:
            logits = predictions
            
        y_true = eval_output.label_ids
        
        # Compute extended metrics
        metrics = compute_extended_metrics(np.array(y_true), logits, label_names)
        save_metrics_artifacts(self.config.output_dir, "validation", metrics)

        # Save human-readable report
        y_pred = np.argmax(logits, axis=1)
        print_classification_report(y_true, y_pred, target_names=label_names)

        # ---------------- Save ----------------
        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        with open(os.path.join(self.config.output_dir, "training_config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        # ---------------- Optional HF Hub push ----------------
        if self.config.push_to_hub and self.config.hub_model_id:
            logger.info(f"☁️ Pushing model to Hugging Face Hub: {self.config.hub_model_id}")
            trainer.push_to_hub(
                repo_id=self.config.hub_model_id,
                private=self.config.hub_private,
                commit_message=f"Upload GPAD V2 ({datetime.now().strftime('%Y-%m-%d')})"
            )

        logger.info("✅ All done – metrics saved and model exported.")
        logger.info("\n" + "="*60)
        logger.info("📊 FINAL VALIDATION METRICS")
        logger.info("="*60)
        logger.info(f"F1 (Macro):        {metrics['macro_f1']:.4f}")
        logger.info(f"Precision (Macro): {metrics['macro_precision']:.4f}")
        logger.info(f"Recall (Macro):    {metrics['macro_recall']:.4f}")
        logger.info("="*60)
        return trainer, metrics