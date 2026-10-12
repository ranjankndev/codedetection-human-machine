#!/usr/bin/env python3
"""
gpad_main.py - Main training script for GPAD model (vast.ai compatible)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

from gpad_trainer import GPADTrainingConfig, OptimizedGPADTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GPAD model for code authorship detection')
    
    # Generate default output directory with timestamp
    timestamp = datetime.now().strftime('%d_%H%M')  # Format: DD_HHMM
    default_output_dir = f'./persistent_{timestamp}'
    
    # Data arguments
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to training data (parquet or csv)')
    parser.add_argument('--val_file', type=str, default=None,
                        help='Path to validation data (optional)')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Sample size for quick testing')
    
    # Model arguments
    parser.add_argument('--base_model', type=str, default='microsoft/codebert-base',
                        help='Base model name')
    parser.add_argument('--num_labels', type=int, default=2,
                        help='Number of classification labels')
    parser.add_argument('--stylometric_dim', type=int, default=64,
                        help='Stylometric feature dimension')
    parser.add_argument('--num_gpad_layers', type=int, default=3,
                        help='Number of GPAD layers')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='Evaluation batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    
    # Optimization arguments
    parser.add_argument('--use_mixed_precision', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--use_gradient_checkpointing', action='store_true', default=True,
                        help='Use gradient checkpointing')
    parser.add_argument('--scheduler_type', type=str, default='linear',
                        choices=['linear', 'cosine', 'onecycle'],
                        help='Learning rate scheduler type')
    
    # Logging and saving
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                        help=f'Output directory for model and logs (default: {default_output_dir})')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Cache directory for pretrained models')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval')
    parser.add_argument('--eval_strategy', type=str, default='steps',
                        choices=['steps', 'epoch'],
                        help='Evaluation strategy')
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='Evaluation steps')
    parser.add_argument('--save_strategy', type=str, default='steps',
                        choices=['steps', 'epoch', 'no'],
                        help='Save strategy')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save steps')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Early stopping patience (0 to disable)')
    
    # W&B arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='gpad-code-detection',
                        help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--sanity_check', action='store_true',
                        help='Run sanity check before training')
    parser.add_argument('--dry_run', action='store_true',
                        help='Quick dry run with minimal data')
    
    return parser.parse_args()


def prepare_data(train_file: str, val_file: str = None, sample_size: int = None):
    """Load and prepare training data."""
    logger.info(f"Loading training data from {train_file}")
    
    # Load training data
    if train_file.endswith('.parquet'):
        train_df = pd.read_parquet(train_file)
    elif train_file.endswith('.csv'):
        train_df = pd.read_csv(train_file)
    else:
        raise ValueError("Training file must be .parquet or .csv")
    
    # Check columns
    logger.info(f"Training data columns: {train_df.columns.tolist()}")
    
    # Rename columns if needed (adjust based on your actual column names)
    column_mapping = {
        'text': 'code',
        'source_code': 'code',
        'content': 'code',
        # Add other possible column names
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in train_df.columns and new_col not in train_df.columns:
            train_df = train_df.rename(columns={old_col: new_col})
            logger.info(f"Renamed column '{old_col}' to '{new_col}'")
    
    # Verify required columns
    if 'code' not in train_df.columns or 'label' not in train_df.columns:
        raise ValueError(f"Required columns 'code' and 'label' not found. Available: {train_df.columns.tolist()}")
    
    # Sample for quick testing
    if sample_size and sample_size < len(train_df):
        logger.info(f"Sampling {sample_size} examples from training data")
        train_df = train_df.sample(n=sample_size, random_state=42)
    
    # Load validation data if provided
    val_df = None
    if val_file:
        logger.info(f"Loading validation data from {val_file}")
        if val_file.endswith('.parquet'):
            val_df = pd.read_parquet(val_file)
        else:
            val_df = pd.read_csv(val_file)
        
        # Apply same column mapping
        for old_col, new_col in column_mapping.items():
            if old_col in val_df.columns and new_col not in val_df.columns:
                val_df = val_df.rename(columns={old_col: new_col})
        
        if sample_size and sample_size < len(val_df):
            val_df = val_df.sample(n=min(sample_size // 2, len(val_df)), random_state=42)
    
    logger.info(f"Training samples: {len(train_df)}")
    if val_df is not None:
        logger.info(f"Validation samples: {len(val_df)}")
    
    logger.info(f"Label distribution:\n{train_df['label'].value_counts()}")
    
    return train_df, val_df


def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("="*80)
    logger.info("GPAD Model Training")
    logger.info("="*80)
    logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Dry run mode
    if args.dry_run:
        logger.info("DRY RUN MODE - Using minimal settings")
        args.num_epochs = 1
        args.train_batch_size = 2
        args.eval_batch_size = 2
        args.sample_size = 50
        args.save_strategy = 'no'
        args.early_stopping_patience = 0
        args.eval_steps = 5
        args.log_interval = 2
    
    # Prepare data
    train_df, val_df = prepare_data(args.train_file, args.val_file, args.sample_size)
    
    # Determine num_labels from data
    if args.num_labels == 2:
        args.num_labels = train_df['label'].nunique()
        logger.info(f"Detected {args.num_labels} unique labels")
    
    # Save processed data temporarily
    temp_train = os.path.join(args.output_dir, 'temp_train.parquet')
    temp_val = os.path.join(args.output_dir, 'temp_val.parquet') if val_df is not None else None
    
    train_df.to_parquet(temp_train, index=False)
    if val_df is not None:
        val_df.to_parquet(temp_val, index=False)
    
    # Create config
    config = GPADTrainingConfig(
        base_model=args.base_model,
        num_labels=args.num_labels,
        stylometric_dim=args.stylometric_dim,
        num_gpad_layers=args.num_gpad_layers,
        max_length=args.max_length,
        num_epochs=args.num_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_mixed_precision=args.use_mixed_precision,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        scheduler_type=args.scheduler_type,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        log_interval=args.log_interval,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        early_stopping_patience=args.early_stopping_patience,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
    )
    
    logger.info("Configuration:")
    logger.info(f"  Base model: {config.base_model}")
    logger.info(f"  Number of labels: {config.num_labels}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Batch size: {config.train_batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Output dir: {config.output_dir}")
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = OptimizedGPADTrainer(config)
    
    # Sanity check
    if args.sanity_check or args.dry_run:
        logger.info("Running sanity check...")
        try:
            loss, logits = trainer.sanity_check(
                train_df.iloc[0]["code"],
                train_df.iloc[0]["label"]
            )
            logger.info(f"Sanity check passed - Loss: {loss:.4f}, Logits shape: {logits.shape}")
        except Exception as e:
            logger.error(f"Sanity check failed: {e}")
            return
    
    # Train
    logger.info("Starting training...")
    try:
        trainer_obj, results = trainer.train(
            train_file=temp_train,
            validation_file=temp_val
        )
        
        logger.info("="*80)
        logger.info("Training completed successfully!")
        logger.info("="*80)
        
        # Log final results
        if 'eval_result' in results:
            eval_results = results['eval_result']
            logger.info("Final Evaluation Results:")
            for key, value in eval_results.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
        
        logger.info(f"Model saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        # Cleanup temp files
        if os.path.exists(temp_train):
            os.remove(temp_train)
        if temp_val and os.path.exists(temp_val):
            os.remove(temp_val)
    
    logger.info(f"Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()