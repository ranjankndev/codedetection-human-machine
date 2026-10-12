import argparse
from CodeBERTTrainer import CodeBERTTrainer

parser = argparse.ArgumentParser()

parser.add_argument("--train_file", type=str, required=True)
parser.add_argument("--validation_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="./codebert_baseline_clean")

args = parser.parse_args()

trainer = CodeBERTTrainer(
    task_subset='A',
    max_length=512,
    model_name="microsoft/codebert-base"
)

# Load Data
train_df, val_df = trainer.load_and_prepare_data(
    train_file=args.train_file,
    validation_file=args.validation_file
)

trainer.initialize_model_and_tokenizer()

train_dataset, val_dataset = trainer.prepare_datasets(train_df, val_df)

trainer_obj = trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    output_dir=args.output_dir,
    num_epochs=5,
    batch_size=16,
    learning_rate=2e-5,
    use_wandb=False,
    run_name="codebert_clean_split_baseline"
)

trainer.evaluate_model(trainer_obj, val_dataset)
