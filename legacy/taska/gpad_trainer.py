# gpad_trainer.py (production-ready, cleaned & optimized)

import random
import numpy as np
import math
import os
import json
import shutil
import logging
import warnings

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, get_linear_schedule_with_warmup,
    Trainer, TrainingArguments, EarlyStoppingCallback, EvalPrediction
)
from torch.optim import AdamW as TorchAdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import wandb
from evaluation import compute_metrics as eval_compute_metrics
from gpad_model import OptimizedGPADModel, GPADConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")


# ============================================================
# Config
# ============================================================
@dataclass
class GPADTrainingConfig:
    base_model: str = "microsoft/codebert-base"
    num_labels: int = 2
    stylometric_dim: int = 64
    num_gpad_layers: int = 3
    entropy_alpha: float = 0.1
    max_length: int = 512

    num_epochs: int = 5
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    optimizer_type: str = "adamw"
    scheduler_type: str = "linear"

    val_split: float = 0.2
    sample_size: Optional[int] = None
    shuffle_data: bool = True
    num_workers: int = 4

    label_smoothing: float = 0.1
    dropout_rate: float = 0.1
    layer_wise_lr_decay: float = 0.9
    use_swa: bool = False
    early_stopping_patience: int = 3

    use_wandb: bool = False
    wandb_project: str = "gpad-code-detection"
    wandb_run_name: Optional[str] = None
    log_interval: int = 100
    save_strategy: str = "steps"
    save_steps: int = 500
    eval_strategy: str = "steps"
    eval_steps: int = 500

    output_dir: str = "./gpad_model"
    cache_dir: str = "./cache"
    seed: int = 42


# ============================================================
# Dataset
# ============================================================
class OptimizedGPADDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512, cache_tokenization: bool = True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization
        self._tokenization_cache: Dict[int, Dict[str, torch.Tensor]] = {}

        if cache_tokenization and len(texts) < 10000:
            self._pre_tokenize()

    def _pre_tokenize(self):
        logger.info("Pre-tokenizing dataset for faster training...")
        for idx in tqdm(range(len(self.texts)), desc="Tokenizing"):
            self._tokenization_cache[idx] = self._tokenize_text(self.texts[idx])

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        try:
            enc = self.tokenizer(
                str(text),
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0)
            }
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}")
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long)
            }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if idx in self._tokenization_cache:
            encoding = self._tokenization_cache[idx]
        else:
            encoding = self._tokenize_text(self.texts[idx])
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'code_texts': str(self.texts[idx])
        }


class GPADCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        code_texts = [item['code_texts'] for item in batch]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'code_texts': code_texts
        }


# ============================================================
# Trainer
# ============================================================
class OptimizedGPADTrainer:
    def __init__(self, config: GPADTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._set_seeds()
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.best_metric = -1.0
        self.training_history = []

        os.makedirs(self.config.output_dir, exist_ok=True)
        logger.info(f"Initialized GPAD trainer on device: {self.device}")
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        seed = self.config.seed or 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Determinism (not inside CUDA if)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.info(f"Seeds set to {seed}")
    def sanity_check(self, code_sample: str, label_sample: int):
        """Run a forward pass on a single sample to validate model pipeline."""
        if self.model is None or self.tokenizer is None:
            logger.info("Model or tokenizer not initialized. Initializing...")
            self.initialize_model_and_tokenizer()

        self.model.eval()
        inputs = self.tokenizer(
            code_sample,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        labels = torch.tensor([label_sample], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels
            )

        loss = outputs.loss.item()
        logits = outputs.logits.cpu()

        logger.info(f"Sanity check completed - Loss: {loss:.4f}, Logits shape: {logits.shape}")
        return loss, logits

    # --------------------------------------------------------
    # Improved model/tokenizer initialization
    # --------------------------------------------------------
    def initialize_model_and_tokenizer(self):
        logger.info(f"Initializing tokenizer & model: {self.config.base_model}")

        local_model_path = "./codebert-base"
        os.makedirs(self.config.cache_dir, exist_ok=True)  # ✅ always ensure cache exists

        # --- Tokenizer ---
        try:
            if os.path.isdir(local_model_path):
                logger.info(f"Loading tokenizer from local path: {local_model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            else:
                logger.info("Loading tokenizer from Hugging Face cache/hub...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.base_model,
                    cache_dir=self.config.cache_dir
                )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            try:
                logger.warning(f"Removing corrupted cache: {self.config.cache_dir}")
                shutil.rmtree(self.config.cache_dir, ignore_errors=True)
                os.makedirs(self.config.cache_dir, exist_ok=True)

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.base_model,
                    cache_dir=self.config.cache_dir,
                    force_download=True,
                    local_files_only=False
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Tokenizer re-downloaded successfully")
            except Exception as e2:
                logger.critical(f"Tokenizer recovery failed: {e2}")
                raise

        # --- Model ---
        try:
            gpad_config = GPADConfig(
                base_model=self.config.base_model,
                num_labels=self.config.num_labels,
                stylometric_dim=self.config.stylometric_dim,
                num_gpad_layers=self.config.num_gpad_layers,
                entropy_alpha=self.config.entropy_alpha,
                max_length=self.config.max_length,
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                use_flash_attention=self.config.use_flash_attention
            )

            self.model = OptimizedGPADModel(gpad_config)
            logger.info("Model created successfully")
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise

        try:
            self.model.to(self.device)
            logger.info(f"Model moved to device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to move model to device: {e}")
            raise

        if self.config.use_gradient_checkpointing:
            try:
                if hasattr(self.model, "gradient_checkpointing_enable"):
                    self.model.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled")
            except Exception as e:
                logger.warning(f"Could not enable gradient checkpointing: {e}")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

    def load_and_prepare_data(self, train_file: str, validation_file: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare training and validation data."""
        logger.info(f"Loading data from {train_file}")
        
        try:
            if train_file.endswith('.parquet'):
                train_df = pd.read_parquet(train_file)
            elif train_file.endswith('.csv'):
                train_df = pd.read_csv(train_file)
            else:
                raise ValueError("Unsupported file format. Use .parquet or .csv")
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise

        required_columns = ['code', 'label']
        missing_columns = [c for c in required_columns if c not in train_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Data cleaning
        train_df = train_df.dropna(subset=required_columns).copy()
        
        if len(train_df) == 0:
            raise ValueError("Training data is empty after removing NaN values")
        
        train_df['code'] = train_df['code'].astype(str)
        train_df['label'] = train_df['label'].astype(int)

        # Validate labels
        unique_labels = train_df['label'].unique()
        if len(unique_labels) < 2:
            logger.warning(f"Only {len(unique_labels)} unique label(s) found in training data")
        if not all(label in [0, 1] for label in unique_labels):
            logger.warning(f"Labels contain values other than 0 and 1: {unique_labels}")

        if self.config.sample_size and self.config.sample_size < len(train_df):
            logger.info(f"Sampling {self.config.sample_size} examples for testing")
            train_df = train_df.sample(n=self.config.sample_size, random_state=self.config.seed)

        if validation_file:
            try:
                if validation_file.endswith('.parquet'):
                    val_df = pd.read_parquet(validation_file)
                else:
                    val_df = pd.read_csv(validation_file)
                val_df = val_df.dropna(subset=required_columns).copy()
                
                if len(val_df) == 0:
                    raise ValueError("Validation data is empty after removing NaN values")
                
                val_df['code'] = val_df['code'].astype(str)
                val_df['label'] = val_df['label'].astype(int)
            except Exception as e:
                logger.error(f"Failed to load validation data: {e}")
                raise
        else:
            try:
                train_df, val_df = train_test_split(
                    train_df,
                    test_size=self.config.val_split,
                    random_state=self.config.seed,
                    stratify=train_df['label']
                )
            except Exception as e:
                logger.warning(f"Stratified split failed: {e}. Using non-stratified split.")
                train_df, val_df = train_test_split(
                    train_df,
                    test_size=self.config.val_split,
                    random_state=self.config.seed
                )

        logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
        logger.info(f"Label distribution - Train: {train_df['label'].value_counts().to_dict()}")
        logger.info(f"Label distribution - Val: {val_df['label'].value_counts().to_dict()}")

        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    def initialize_model_and_tokenizer(self):
        """Initialize tokenizer and model."""
        logger.info(f"Initializing tokenizer & model: {self.config.base_model}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                cache_dir=None  # Use default HuggingFace cache instead of corrupted ./cache
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.token
            logger.info("Tokenizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

        # Create GPAD config and model
        try:
            gpad_config = GPADConfig(
                base_model=self.config.base_model,
                num_labels=self.config.num_labels,
                stylometric_dim=self.config.stylometric_dim,
                num_gpad_layers=self.config.num_gpad_layers,
                entropy_alpha=self.config.entropy_alpha,
                max_length=self.config.max_length,
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                use_flash_attention=self.config.use_flash_attention
            )

            self.model = OptimizedGPADModel(gpad_config)
            logger.info("Model created successfully")
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise

        # Move model to device
        try:
            self.model.to(self.device)
            logger.info(f"Model moved to device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to move model to device: {e}")
            raise

        # Try enabling gradient checkpointing safely
        if self.config.use_gradient_checkpointing:
            try:
                if hasattr(self.model, "gradient_checkpointing_enable"):
                    self.model.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled")
            except Exception as e:
                logger.warning(f"Could not enable gradient checkpointing: {e}")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")
        
    def create_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Create PyTorch datasets from dataframes."""
        logger.info("Creating datasets (tokenization may be cached)...")
        
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError("Cannot create datasets from empty dataframes")
        
        try:
            self.train_dataset = OptimizedGPADDataset(
                texts=train_df['code'].tolist(),
                labels=train_df['label'].tolist(),
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
                cache_tokenization=True
            )
            logger.info(f"Training dataset created with {len(self.train_dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to create training dataset: {e}")
            raise
        
        try:
            self.val_dataset = OptimizedGPADDataset(
                texts=val_df['code'].tolist(),
                labels=val_df['label'].tolist(),
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
                cache_tokenization=True
            )
            logger.info(f"Validation dataset created with {len(self.val_dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to create validation dataset: {e}")
            raise

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        logger.info("Creating optimizer & scheduler (layer-wise lr decay support)...")
        # Parameter groups
        if self.config.layer_wise_lr_decay < 1.0 and hasattr(self.model, 'gpad_layers') and hasattr(self.model, 'classifier'):
            try:
                param_groups = []
                # gpad layers
                for i, layer in enumerate(self.model.gpad_layers):
                    lr_scale = self.config.layer_wise_lr_decay ** (len(self.model.gpad_layers) - i - 1)
                    param_groups.append({
                        'params': layer.parameters(),
                        'lr': self.config.learning_rate * lr_scale
                    })
                # classifier
                param_groups.append({
                    'params': self.model.classifier.parameters(),
                    'lr': self.config.learning_rate
                })
                # other params
                other_params = []
                for name, param in self.model.named_parameters():
                    if not any(name.startswith(prefix) for prefix in ['gpad_layers', 'classifier']):
                        other_params.append(param)
                if other_params:
                    param_groups.append({
                        'params': other_params,
                        'lr': self.config.learning_rate * 0.5
                    })
                optimizer = TorchAdamW(
                    param_groups,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    eps=1e-8
                )
                logger.info(f"Created optimizer with {len(param_groups)} parameter groups (layer-wise LR decay)")
            except Exception as e:
                logger.warning(f"Failed to create layer-wise optimizer: {e}. Falling back to standard optimizer.")
                optimizer = TorchAdamW(
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    eps=1e-8
                )
        else:
            optimizer = TorchAdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                eps=1e-8
            )
            logger.info("Created standard optimizer (no layer-wise LR decay)")

        # Scheduler
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        scheduler = None
        try:
            if self.config.scheduler_type == "linear":
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps
                )
                logger.info(f"Created linear scheduler with {warmup_steps} warmup steps")
            elif self.config.scheduler_type == "cosine":
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=max(1, num_training_steps // 4),
                    T_mult=2
                )
                logger.info("Created cosine annealing scheduler")
            elif self.config.scheduler_type == "onecycle":
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=self.config.learning_rate,
                    total_steps=max(1, num_training_steps),
                    pct_start=self.config.warmup_ratio
                )
                logger.info("Created OneCycle scheduler")
            else:
                logger.warning(f"Unknown scheduler type: {self.config.scheduler_type}, using default linear")
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps
                )
        except Exception as e:
            logger.warning(f"Failed to create scheduler: {e}. Training will proceed without scheduler.")
            scheduler = None
        
        return optimizer, scheduler

    def compute_metrics(self, p: EvalPrediction) -> Dict[str, float]:
        """Compute metrics using the evaluation module (more comprehensive)."""
        try:
            return eval_compute_metrics((p.predictions, p.label_ids))
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return {}
    

    def train(self, train_file: str, validation_file: Optional[str] = None):
        """Main training method."""
        logger.info("Starting training pipeline...")
        
        # Load and prepare data
        train_df, val_df = self.load_and_prepare_data(train_file, validation_file)
        
        # Initialize model and tokenizer
        self.initialize_model_and_tokenizer()
        
        # Create datasets
        self.create_datasets(train_df, val_df)

        # Compute training steps robustly
        train_examples = len(self.train_dataset)
        batch_effective = max(1, self.config.train_batch_size * self.config.gradient_accumulation_steps)
        steps_per_epoch = math.ceil(train_examples / batch_effective)
        num_training_steps = max(1, steps_per_epoch * self.config.num_epochs)
        
        logger.info(f"Train examples: {train_examples}, Steps/epoch: {steps_per_epoch}, Total steps: {num_training_steps}")

        # TrainingArguments
        try:
            # Handle save_strategy and load_best_model_at_end compatibility
            load_best = True
            if self.config.save_strategy == "no":
                logger.warning("save_strategy is 'no', setting load_best_model_at_end=False")
                load_best = False
            
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.train_batch_size,
                per_device_eval_batch_size=self.config.eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                warmup_ratio=self.config.warmup_ratio,
                max_grad_norm=self.config.max_grad_norm,
                eval_strategy=self.config.eval_strategy,
                eval_steps=max(1, self.config.eval_steps) if self.config.eval_strategy == "steps" else None,
                save_strategy=self.config.save_strategy,
                save_steps=max(1, self.config.save_steps) if self.config.save_strategy == "steps" else None,
                save_total_limit=3,
                load_best_model_at_end=load_best,
                metric_for_best_model="f1",  # Always set for EarlyStoppingCallback
                greater_is_better=True,
                fp16=self.config.use_mixed_precision and torch.cuda.is_available(),
                dataloader_num_workers=self.config.num_workers,
                remove_unused_columns=False,
                logging_dir=os.path.join(self.config.output_dir, "logs"),
                logging_steps=max(1, self.config.log_interval),
                report_to="wandb" if self.config.use_wandb else "none",
                run_name=self.config.wandb_run_name,
                seed=self.config.seed,
                data_seed=self.config.seed,
            )
            logger.info("TrainingArguments created successfully")
        except Exception as e:
            logger.error(f"Failed to create TrainingArguments: {e}")
            raise

        data_collator = GPADCollator(self.tokenizer)

        # Optionally create custom optimizer+scheduler (for layer-wise LR)
        optimizers = None
        try:
            optimizer, scheduler = self.create_optimizer_and_scheduler(num_training_steps)
            if optimizer is not None:
                optimizers = (optimizer, scheduler)
                logger.info("Using custom optimizer and scheduler")
            else:
                logger.info("Using Trainer's default optimizer")
        except Exception as e:
            logger.warning(f"Could not create custom optimizer/scheduler: {e}. Using Trainer defaults.")
            optimizers = None

        # Add callbacks conditionally
        callbacks = []
        # Only use early stopping when saving checkpoints (it needs checkpoints to work)
        if self.config.early_stopping_patience > 0 and self.config.save_strategy != "no":
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience))
            logger.info(f"Early stopping enabled with patience={self.config.early_stopping_patience}")
        elif self.config.early_stopping_patience > 0 and self.config.save_strategy == "no":
            logger.warning("Early stopping disabled because save_strategy='no' (early stopping requires saving checkpoints)")

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers  # pass our optimizer+scheduler when available
        )

        # W&B init (optional)
        if self.config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name or f"gpad_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=asdict(self.config)
                )
            except Exception as e:
                logger.warning(f"W&B init failed: {e}")

        # Train
        logger.info("Beginning training loop...")
        try:
            train_result = trainer.train()
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

        # Save
        try:
            trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            config_path = os.path.join(self.config.output_dir, "training_config.json")
            with open(config_path, "w") as f:
                json.dump(asdict(self.config), f, indent=2)
            logger.info(f"Model and config saved to {self.config.output_dir}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

        logger.info("Running final evaluation...")
        try:
            eval_result = trainer.evaluate()
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            eval_result = {}

        results = {
            'train_result': train_result.metrics if hasattr(train_result, 'metrics') else {},
            'eval_result': eval_result,
            'config': asdict(self.config),
            'model_size': {
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }

        try:
            results_path = os.path.join(self.config.output_dir, "training_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {results_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

        logger.info(f"Training completed! Results saved to {self.config.output_dir}")
        
        # Safe access to metrics
        eval_f1 = eval_result.get('eval_f1', None)
        eval_acc = eval_result.get('eval_accuracy', None)
        
        if eval_f1 is not None:
            logger.info(f"Final validation F1: {eval_f1:.4f}")
        if eval_acc is not None:
            logger.info(f"Final validation accuracy: {eval_acc:.4f}")

        if self.config.use_wandb:
            wandb.finish()

        return trainer, results


def create_training_config_from_args(args) -> GPADTrainingConfig:
    """Create training config from command line arguments."""
    config = GPADTrainingConfig()

    # Update config with provided arguments
    for key, value in vars(args).items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)

    return config