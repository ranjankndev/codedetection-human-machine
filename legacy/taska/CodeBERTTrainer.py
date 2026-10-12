
# CodeBERTTrainer.py - Production Version for Vast.ai
import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from tqdm.auto import tqdm
import logging
import warnings
from datetime import datetime
import json

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeBERTTrainer:
    def __init__(self, task_subset='A', max_length=512, model_name="microsoft/codebert-base"):
        self.task_subset = task_subset
        self.max_length = max_length
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.num_labels = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_and_prepare_data(self, train_file=None, validation_file=None):
        logger.info(f"Loading dataset for task {self.task_subset}...")

        # Load parquet files
        try:
            train_df = pd.read_parquet(train_file)
            val_df = pd.read_parquet(validation_file)
        except Exception as e:
            logger.error(f"Error reading parquet files: {e}")
            raise

        # Check necessary columns
        for df, name in zip([train_df, val_df], ["train", "val"]):
            if 'code' not in df.columns or 'label' not in df.columns:
                raise ValueError(f"{name}_df must contain 'code' and 'label' columns")
            df.dropna(subset=['code', 'label'], inplace=True)
            df['code'] = df['code'].astype(str)
            df['label'] = df['label'].astype(int)

        self.num_labels = max(train_df['label'].nunique(), val_df['label'].nunique())
        logger.info(f"Number of unique labels: {self.num_labels}")
        logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    def initialize_model_and_tokenizer(self):
        logger.info(f"Initializing {self.model_name} model and tokenizer...")

            # Ensure num_labels is set
        if self.num_labels is None:
            self.num_labels = 2  # Default to binary classification
            
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification"
        )
        logger.info(f"Model initialized with {self.num_labels} labels")

    def tokenize_function(self, examples):
            codes = [str(c) for c in examples['code']]
            return self.tokenizer(
            codes,
            truncation=True,
            padding=True,
            max_length=self.max_length
            )

    def prepare_datasets(self, train_df, val_df):
        logger.info("Preparing datasets with tokenization...")

            # Reset index to avoid __index_level_0__ column
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        train_dataset = Dataset.from_pandas(train_df[['code', 'label']])
        val_dataset = Dataset.from_pandas(val_df[['code', 'label']])

        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['code'],
            desc="Tokenizing train"
        )
        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['code'],
            desc="Tokenizing val"
        )

        train_dataset = train_dataset.rename_column('label', 'labels')
        val_dataset = val_dataset.rename_column('label', 'labels')

        return train_dataset, val_dataset

    def compute_metrics(self, eval_pred):
        from evaluation import compute_metrics
        return compute_metrics(eval_pred)

    def train(self, train_dataset, val_dataset, output_dir="./results", 
              num_epochs=3, batch_size=16, learning_rate=2e-5, 
              use_wandb=False, run_name=None):
        logger.info("Starting training...")

        # Setup WandB if enabled
        report_to = "wandb" if use_wandb else "none"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="steps", 
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            remove_unused_columns=False,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            report_to=report_to,
            run_name=run_name or f"task_{self.task_subset}_{self.timestamp}",
            save_total_limit=2,
            push_to_hub=False
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Training completed. Model saved to {output_dir}")
        return trainer

    def evaluate_model(self, trainer, val_dataset):
        logger.info("Evaluating model...")
        predictions = trainer.predict(val_dataset)
        from evaluation import print_classification_report
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        print_classification_report(y_true, y_pred)
        return predictions


def load_model_and_tokenizer(model_path, device="cpu"):
    """Load trained model and tokenizer from checkpoint."""
    from transformers import RobertaForSequenceClassification, RobertaTokenizer
    import torch
    
    logger.info(f"Loading model from: {model_path}")
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully on {device}")
    
    return model, tokenizer