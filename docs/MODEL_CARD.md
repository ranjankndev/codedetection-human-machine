# Model card — codedet (SemEval-2026 Task 13)

**Status:** research / portfolio artifact. Not a product, not validated for use on real people.

## Model details

| | |
|---|---|
| Architecture | `microsoft/codebert-base` encoder → *N* GPAD blocks (attention-entropy bias + gated stylometric fusion) → MLP head |
| Variants | `codebert` (N=0), `control` (N=1, bias off, no style — the capacity control), `gpad` (N=1, bias + style), `gpad_entropy_only`, `gpad_style_only`, `gpad_sign_neg`, `lora` (frozen encoder + rank-8 adapters on query/value) |
| Inputs | a code snippet (any language present in the data), truncated head+tail to the preset's `max_len` (192 head + 64 tail at 256) |
| Auxiliary inputs | 16 language-agnostic stylometric features, z-scored with **train** statistics; optionally 8 LM likelihood/entropy features from a frozen `Qwen/Qwen2.5-Coder-0.5B` |
| Outputs | Task A: P(machine). Task B: distribution over 11 classes (human + 10 LLM families) |
| Losses | Task A cross-entropy; Task B class-weighted CE by default (focal and logit-adjusted CE are implemented and configurable) |
| Training | AdamW, linear warmup/decay, fp16 autocast, dynamic padding, early stopping on validation macro-F1; separate LR for the pretrained encoder (2e-5) and new layers (1e-4) |

The "entropy" in GPAD is **not** token-level predictive entropy. It is the entropy of a learned softmax
over key self-similarity, turned into an additive per-key bias on pre-softmax attention scores:
`score_qk += sign · α · σ(mean(H) − H_k)`. With `sign=+1` low-entropy keys are boosted. True predictive
entropy is a separate, explicitly named feature set (`codedet/lm_features.py`).

## Intended use

- Research into machine-generated-code detection and the evaluation of such detectors.
- Teaching material for distribution shift, calibration, ablation design and cost-sensitive thresholds.

## Out-of-scope / prohibited use

- **Academic-integrity decisions about an individual.** The measured out-of-distribution macro-F1 and the
  false-accusation rate in `results/analysis.json` are the reason. A score is evidence at best; it is not
  proof, and it is not admissible on its own.
- Any setting where the input language, domain or generation style differs from the training data
  without re-measuring first. The project exists partly to show how large that drop is.
- Ranking or scoring people, hiring, or moderation at scale.

## Training data

Official SemEval-2026 Task 13 training parquet files only (`code`, `label`, `language`, `generator`,
`domain`). No third-party pretrained AI-code detectors are used, per competition rules. Counts, per-class
and per-language distributions, deduplication counts and the exact split construction for each run are in
`results/data_report.json`.

Split policy:
- **Task A OOD** = a held-out language (Java by default); the model never sees it in training.
- **Task B OOD** = held-out *generators* (one checkpoint from each of up to 3 families), plus a sample of
  human rows. Holding out whole families would make those classes unlearnable.
- Train is deduplicated against every evaluation split: exact (whitespace-normalised hash) and
  near-duplicate (MinHash/Jaccard ≥ 0.9) when `datasketch` is installed.
- Task B caps the human share of **train only** (default 40%); evaluation splits keep the true, heavily
  human-dominated prior.

## Metrics

Primary: macro-F1 with a 95% bootstrap confidence interval (1000 resamples at `lite`). Reported alongside:
balanced accuracy, accuracy, per-class precision/recall/F1, per-language and per-generator slices, ECE
before and after temperature scaling, and the operating point under a 5:1 false-accusation cost ratio.
Model-vs-model claims use a paired bootstrap on identical rows plus McNemar; differences whose interval
contains zero are reported as ties.

Current numbers: `results/summary.md` (generated). If that file is absent, the run has not been executed
in this checkout — no numbers are quoted anywhere by hand.

## Ethical considerations

The cost structure is asymmetric and the model card should say so plainly: a false "machine" verdict on a
student's own work is far more damaging than a missed detection. The default operating point is therefore
not `argmax`; it is the threshold minimising a stated 5:1 cost, chosen on in-distribution validation data
and then applied unchanged out of distribution (where it degrades — see the analysis).

Detectors also degrade silently as new models are released. Any deployment would need periodic
re-measurement against newly released generators, and the `unseen-generator` split in this repo is the
cheap proxy for that test.

## Limitations

- Budget presets (≈20k training rows, max_len 256, 1–2 epochs) sized for a free Kaggle GPU session.
- Single seed for transformer models at `lite` (3 seeds for the cheap models); seed spread is reported.
- Proxy OOD splits built from the training release; the organisers' hidden test sets are not used.
- Stylometric features are language-agnostic by design, but the legacy `radon` features (Python-only,
  kept in `features.py` for the ablation) act as a language detector and must not be used for claims.
