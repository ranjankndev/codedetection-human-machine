# Detecting machine-generated code, and naming the model that wrote it

**SemEval-2026 Task 13.** Task A: is this snippet written by a human or a machine? Task B: which of
10 LLM families wrote it (11 classes, human included)?

This rebuilds a competition project. The model idea is the same (CodeBERT plus one extra transformer
block with an **attention-entropy bias**, fused with style features: "GPAD"). The **evaluation** is new:
every model on the same rows, confidence intervals, a capacity control for the extra block, and test
splits the model has never seen (a held-out language, held-out generators).

A code detector is a tool for accusing someone of cheating, so the questions here are *when does it
fail*, *how sure should it be*, and *what does a wrong call cost*.

## Results

The current numbers come from a laptop-CPU run on the real data (`configs/local_real.yaml`, small
samples, a small encoder). The full GPU run (`configs/lite.yaml`) has not been done yet.
One-page report with figures: [`results/REPORT.md`](results/REPORT.md). Every table:
[`results/APPENDIX.md`](results/APPENDIX.md).

<!-- BEGIN GENERATED RESULTS - produced by scripts/make_report.py, do not edit by hand -->

*Preset `local_real`, transformer rows use `huggingface/CodeBERTa-small-v1`. Scores are macro-F1 with a 95% bootstrap interval (1000 resamples); every model is scored on the same rows. Generated from `results/*.json`.*

### Key findings

- **The entropy bias adds nothing beyond its extra parameters.** GPAD minus the same model with the bias switched off: +0.003 (in-distribution), +0.007 (official val set), +0.002 (unseen language). No interval rules out zero, so the honest reading is a tie. The original report credited the entropy mechanism for a gain this comparison cannot support.
- **A length-and-whitespace shortcut is hard to beat.** Six content-free features score 0.903 in-distribution and 0.513 on the unseen language; the transformer (`gpad`) is +0.047 and -0.036 against that. Out of distribution it learned formatting, not authorship.
- **Confidence does not survive the shift.** Calibration error (ECE) is 0.027 in-distribution but 0.337 on the unseen language, even after temperature scaling.
- **The default threshold accuses too many humans.** On the unseen language, 0.5 flags 85% of human snippets as machine; a threshold chosen for a 5:1 cost brings that to 38%, still far from usable.
- **Task B is an imbalance problem.** Always guessing the biggest class gives accuracy 0.880 but macro-F1 0.085; the best model here (`stack`) reaches macro-F1 0.148. Accuracy alone would hide this.
- **The Task B transformers did not learn the task.** `codebert_weighted` (2 of 11 classes), `gpad_entropy_only` (2 of 11 classes) predict only a few classes and score within 0.02 macro-F1 of always guessing the biggest class, so comparisons between them say nothing about the architectures.

### Task A - human or machine?

Rows: in-distribution n=300, unseen language n=300.

| model | in-distribution | unseen language |
|---|---|---|
| Style features + gradient boosting | 0.957 [0.933, 0.977] | 0.754 [0.703, 0.798] |
| Stacked ensemble | 0.963 [0.943, 0.983] | 0.747 [0.700, 0.793] |
| Length + whitespace only (shortcut probe) | 0.903 [0.869, 0.936] | 0.513 [0.456, 0.569] |
| CodeBERT fine-tune | 0.943 [0.917, 0.967] | 0.509 [0.452, 0.566] |
| GPAD (entropy bias + style) | 0.950 [0.923, 0.973] | 0.477 [0.421, 0.531] |
| GPAD with the bias switched off (control) | 0.947 [0.920, 0.970] | 0.475 [0.419, 0.527] |
| Majority class (floor) | 0.344 [0.320, 0.367] | 0.344 [0.320, 0.368] |
| Character n-grams + logistic regression | 0.840 [0.797, 0.877] | 0.341 [0.317, 0.364] |

**Humans wrongly flagged as machine** (a false accusation costs 5x a miss; threshold picked on in-distribution data only, model `gpad`):

| threshold | value | in-distribution | unseen language |
|---|---|---|---|
| default (0.5) | 0.50 | 6.3% | 85.3% |
| cost-aware | 0.90 | 3.5% | 37.8% |

### Task B - which model family wrote it?

Rows: in-distribution n=300, unseen generator n=300.

| model | in-distribution | unseen generator |
|---|---|---|
| Stacked ensemble | 0.148 [0.093, 0.201] | 0.151 [0.094, 0.190] |
| Character n-grams + logistic regression | 0.144 [0.091, 0.202] | 0.125 [0.088, 0.159] |
| Majority class (floor) | 0.085 [0.083, 0.087] | 0.087 [0.086, 0.089] |
| GPAD, entropy only | 0.099 [0.085, 0.115] | 0.085 [0.082, 0.086] |
| CodeBERT, class-weighted loss | 0.092 [0.082, 0.104] | 0.082 [0.080, 0.085] |

<!-- END GENERATED RESULTS -->

## How GPAD works

`src/codedet/models/gpad.py` adds one transformer block on top of the encoder. For each token it
computes the entropy of a learned softmax over that token's similarity to the others, and adds
`sign * alpha * sigmoid(mean(H) - H_k)` to the attention scores. With `sign = +1` this boosts
**low**-entropy tokens, the opposite of what the original write-up claimed; `gpad_sign_neg` flips it.
The `control` model is the same block with the bias switched off, so any GPAD gain can be split into
"more parameters" vs. "the mechanism". Real predictive entropy from a code LM lives separately in
`src/codedet/features/lm.py`. Diagrams and the full ablation list: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Run it

```bash
pip install -e ".[dev,extra]"      # torch/transformers come from your environment
pytest -q                          # CPU only, no downloads

# dry run on synthetic data (proves every stage works, ~10 min on a laptop)
python scripts/make_fixture.py
python scripts/run_all.py --preset local_cpu --cpu-only --data fixtures --out /tmp/dry_run

# real data on a laptop CPU (parquet files in data/)
python scripts/run_all.py --preset local_real --cpu-only --data data --out results

# real data on a free GPU: see cloud/README.md (Kaggle or Colab)
```

`run_all.py` runs each stage in `scripts/stages/` as its own process, resumes from where it stopped,
degrades instead of crashing (smaller batch, shorter sequences, skip optional stages), writes
`results/status.json`, then rebuilds `results/REPORT.md`, `results/APPENDIX.md` and the block above.
No number in this README is typed by hand; a test fails if it drifts from `results/*.json`.

## Repo layout

```
src/codedet/
  data/          load parquet, labels, train/val/OOD splits + dedup, perturbations, synthetic fixtures
  features/      style features (style.py), code-LM entropy features (lm.py)
  models/        GPAD, LoRA, the model registry (zoo.py), classical models, losses, training loop
  evaluation/    metrics + bootstrap/McNemar, plots, report rendering, latency benchmark
  config.py      presets and output paths
scripts/
  run_all.py     the one command: runs every stage below and the report
  stages/        one script per pipeline stage (splits, classical, train, evaluate, analyze, ...)
  make_report.py results/*.json -> REPORT.md, APPENDIX.md, README block
  make_slides.py results/*.json -> docs/codedet_slides.pptx
configs/         local_cpu (dry run), local_real (laptop, real data), smoke + lite (GPU)
cloud/           Kaggle and Colab notebooks, Kaggle kernel driver
app/demo.py      Gradio demo
tests/           CPU-only pytest
results/         REPORT.md, APPENDIX.md, JSON metrics, figures (data, predictions, checkpoints are not committed)
docs/            ARCHITECTURE, MODEL_CARD, KNOWN_GAPS, INTERVIEW_NOTES; docs/dev/ holds build notes
```

The original competition scripts are kept at git tag `legacy-code` (`git checkout legacy-code -- legacy/`).

## Limitations

- Official SemEval-2026 Task 13 data only; no third-party AI-code detectors (competition rule).
- The CPU run uses 500 training rows and a 6-layer encoder; read its numbers as a comparison between
  models, not as the best this approach can do. The Task B transformers barely beat guessing the biggest class with that little data.
- The OOD splits (held-out language, held-out generator) stand in for the organisers' hidden test sets.
- A detector score is evidence, not proof. Read the false-accusation numbers above before using any
  such model on real people. What was not done: [`docs/KNOWN_GAPS.md`](docs/KNOWN_GAPS.md).
  Intended use and slices: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Credits

SemEval-2026 Task 13 organisers and the dataset (`daniilor/semeval-2026-task13`). Encoders:
`microsoft/codebert-base`, `huggingface/CodeBERTa-small-v1`. LM features: `Qwen/Qwen2.5-Coder-0.5B`.
The original competition results are kept, clearly labeled and not re-validated, in
[`docs/dev/ORIGINAL_RESULTS.md`](docs/dev/ORIGINAL_RESULTS.md).
