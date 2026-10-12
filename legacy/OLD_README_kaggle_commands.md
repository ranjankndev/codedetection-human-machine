# Codedetection-Human-Machine

Training and prediction for SemEval-2026 Task 13

!python create_datasubset.py --task task_b --sample_pct 10


%cd /kaggle/working/Codedetection-Human-Machine/code
!python create_datasubset.py --task task_b --sample_pct 10

%cd /kaggle/working/Codedetection-Human-Machine/code
!python train_lr.py --task task_b --sample_pct 10 \
  --train_path /kaggle/working/task_b_train_10pct.parquet \
  --val_path /kaggle/working/task_b_val_10pct.parquet



  %cd /kaggle/working/Codedetection-Human-Machine/code
!python train_codebert.py --task B --epochs 3 --batch_size 16 --learning_rate 2e-5 --output_dir /kaggle/working/codebert_b_results


#the predict function hf compatabel

python predict.py \
  --model_dir ./results/codebert_taskA_baseline \
  --test_file /kaggle/input/semeval-2026/task_a/test_seen.parquet \
  --output_csv ./submission_taskA_seen.csv \
  --forverify_file /kaggle/input/semeval-2026/task_a/test_seen_forverify.parquet
  --hf_repo_id ranjan56cse/codebert-taskA-baseline \
  --hf_token <your_token>
  
python predict.py \
  --model_dir ./results/codebert_taskB_baseline \
  --test_file /kaggle/input/semeval-2026/task_b/test_unseen.parquet \
  --output_csv ./submissions/taskB_unseen.csv \
  --forverify_file /kaggle/input/semeval-2026/task_b/test_unseen_forverify.parquet \
  --hf_repo_id ranjan56cse/codebert-taskB-baseline \
  --hf_token <your_token>
