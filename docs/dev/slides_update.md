# Slide deck update (text only — numbers are filled from `results/summary.json`)

Keeps the original 8-slide structure and adds three slides. Slides 4–7 tables are regenerated; the
original competition numbers survive on one clearly labelled slide so the story is "here is what I
claimed, here is what held up", which is stronger than quietly replacing them.

---

### Slide 1 — Title
**Detecting machine-generated code, and naming the model that wrote it**
SemEval-2026 Task 13 · Task A: human vs machine · Task B: human + 10 LLM families
*Rebuild: same idea, honest evaluation.*

### Slide 2 — Architecture (replace the old diagram)
CodeBERT encoder → **1 GPAD block** → MLP head.
Inside the block: additive per-key attention bias `score_qk += sign·α·σ(mean(H) − H_k)` where `H_k` is
the entropy of a *learned softmax over key self-similarity* — a content-driven attention bias in the
ALiBi family. Gated fusion of 16 z-scored stylometric features. **At `sign=+1` the bias boosts
low-entropy keys** (the old slide said the opposite).
Separately: true predictive entropy / NLL features from a frozen Qwen2.5-Coder-0.5B.

### Slide 3 — Data and splits
Official training parquet only. Train / ID-val / **OOD** materialised once, with row ids.
OOD-A = held-out language (Java). OOD-B = held-out generator checkpoints from 3 families.
Exact + MinHash dedup between train and every eval split. Task B caps the human share in **train only**;
evaluation keeps the true ~88% human prior.

### Slide 4 (NEW) — Audit and fixes
| Found | Fix |
|---|---|
| "Entropy" meant 3 different things across code, README and slides | One name per mechanism; real predictive entropy added under its own name |
| GPAD adds a transformer block, no control | Parameter-matched `control` (bias off, no style) |
| Tables compared models on 349 vs 17,226 rows | Identical rows by id, `n` printed in every table |
| Accuracy 0.64 / macro-F1 0.29 on Task B | Majority-class illusion; class-aware loss + macro-F1 |
| "Entropy-only is best" from single runs | Paired bootstrap + McNemar; ties called ties |

### Slide 5 — Results, Task A (regenerated, with CIs)
Table from `results/summary.md`: majority / length-only probe / TF-IDF+LR / stylometry GBDT / LM-feature
GBDT / CodeBERT / LoRA / GPAD / capacity control, each as macro-F1 with a 95% bootstrap CI on ID-val and
on the held-out language.

### Slide 6 — Results, Task B (regenerated)
Same shape, 11 classes, plus per-class F1 and the confusion matrix figure. Macro-F1 **and** accuracy, to
show why accuracy alone was misleading.

### Slide 7 (NEW) — Controlled ablation
GPAD − control, GPAD − CodeBERT, entropy-only − weighted CodeBERT: point difference with its 95% CI and
the McNemar p-value, on the same rows. One sentence verdict taken verbatim from the generated report.

### Slide 8 (NEW) — Shortcut learning and calibration
Left: the length/whitespace-only probe next to the transformer, and the comment-stripping stress test on
the same OOD rows. Right: reliability diagram before/after temperature scaling, ID vs OOD, and the PR
curve with the cost-chosen operating point (5:1 false-accusation cost).

### Slide 9 — Efficiency
Params total/trainable (full fine-tune vs LoRA vs GPAD), p50/p95 latency at batch 1 and 32, fp32 / fp16 /
int8-dynamic, and the cost of sequence length.

### Slide 10 — Original competition-time results (unchanged, labelled)
The 2025 numbers exactly as reported then (`docs/ORIGINAL_RESULTS.md`), with the caption:
*"Measured before the fixes on this slide deck's slide 4; not comparable to the regenerated tables."*

### Slide 11 — What I'd do next
Full data and multi-seed; a larger code LM for the entropy features (true Binoculars ratio);
domain-adversarial training against the language shortcut; open-set rejection for unseen generators;
distillation + ONNX for serving.
