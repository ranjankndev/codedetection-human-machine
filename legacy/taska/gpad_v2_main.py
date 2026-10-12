"""
gpad_v2_main.py
Kaggle/Vast-compatible entry point for GPAD V2 training.
Supports:
  • Task A → Binary (Human vs LLM)
  • Task B → Multi-class (11 LLM families)
"""

import sys
import argparse
import torch
from gpad_v2_trainer import GPADTrainer, GPADTrainingConfig
from utils import get_logger, set_seed

logger = get_logger("GPAD_V2_Main")


def parse_args():
    parser = argparse.ArgumentParser(description="GPAD V2 Training & Evaluation")

    # === Task & Data ===
    parser.add_argument("--task", type=str, choices=["A", "B"], required=True,
                        help="Task A = Binary detection, Task B = Multi-class detection.")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training file (.parquet or .csv).")
    parser.add_argument("--validation_file", type=str, default=None,
                        help="Optional validation file (.parquet or .csv).")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Optional sample size for testing runs.")

    # === Model configuration ===
    parser.add_argument("--base_model", type=str, default="microsoft/codebert-base",
                        help="Base model name (CodeBERT / GraphCodeBERT / UniXCoder).")
    parser.add_argument("--use_entropy", action="store_true",
                        help="Enable entropy-weighted attention layers.")
    parser.add_argument("--stylometric_dim", type=int, default=32,
                        help="Stylometric feature dimension (default: 32).")
    parser.add_argument("--num_gpad_layers", type=int, default=1,
                        help="Number of entropy-weighted GPAD layers (1–2 recommended).")
    parser.add_argument("--classifier_hidden_dim", type=int, default=384,
                        help="Hidden dimension for classifier head.")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum tokenized sequence length.")
    parser.add_argument("--disable_stylometric", action="store_true",
                        help="Disable stylometric feature fusion.")
    parser.add_argument("--disable_gradient_checkpointing", action="store_true",
                        help="Disable gradient checkpointing for debugging.")

    # === Training hyperparameters ===
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--disable_mixed_precision", action="store_true",
                        help="Disable FP16 mixed precision training.")

    # === Output & Logging ===
    parser.add_argument("--output_dir", type=str, default="./gpad_v2_results",
                        help="Output directory for model + metrics.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    # === Hugging Face Hub ===
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model and metrics to Hugging Face Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Repository ID on HF Hub (e.g., yourname/gpad-v2-taskA).")
    parser.add_argument("--hub_token", type=str, default=None,
                        help="Hugging Face access token (optional if already logged in).")
    parser.add_argument("--hub_private", action="store_true",
                        help="Upload as private repo.")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info("=" * 90)
    logger.info(f"🚀 Starting GPAD V2 | Task={args.task} | Base={args.base_model}")
    logger.info("=" * 90)

    # === Build config ===
    config = GPADTrainingConfig(
        task=args.task.upper(),
        base_model=args.base_model,
        stylometric_dim=args.stylometric_dim,
        num_gpad_layers=args.num_gpad_layers,
        entropy_alpha=0.15 if args.use_entropy else 0.0,
        max_length=args.max_length,
        use_stylometric=not args.disable_stylometric,
        classifier_hidden_dim=args.classifier_hidden_dim,
        use_gradient_checkpointing=not args.disable_gradient_checkpointing,
        num_epochs=args.epochs,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_mixed_precision=not args.disable_mixed_precision,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_private=args.hub_private,
    )

    # === GPU info ===
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✅ GPU: {device_name} ({total_mem:.1f} GB)")
    else:
        logger.warning("⚠ No GPU detected — training will be slow.")

    # === Trainer ===
    trainer = GPADTrainer(config)
    trainer.train(args.train_file, args.validation_file)

    logger.info("✅ Training complete — metrics and model saved.")
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
