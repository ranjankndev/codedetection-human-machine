# Interview notes

Talking points for this project. Every number referenced lives in `results/summary.json`; if a number is
not there, do not say it.

## 60-second pitch

"SemEval-2026 Task 13 is: given a code snippet, say whether a human or an LLM wrote it, and if an LLM,
which family. I'd already shipped a version during the competition — CodeBERT with an extra attention
block that biases attention by a token-entropy signal, plus stylometric features. When I went back to it,
the modelling was fine but the *evidence* was not: the extra block added parameters and nobody had
controlled for that, the in-distribution and out-of-distribution test sets had different numbers of rows
so the tables weren't comparable, and 'entropy-only is best' rested on single runs with no confidence
intervals. So I rebuilt the evaluation around the claims: parameter-matched control, identical rows for
every model, bootstrap CIs and paired tests, held-out-language and held-out-generator splits, calibration
under shift, and a cost-based threshold because a false accusation is much more expensive than a miss.
The whole thing runs end to end on a free Kaggle GPU in under an hour and every number in the README is
generated from JSON by a script."

## 3-minute deep dive

1. **The data honesty layer.** Splits are materialised to disk once, with a stable `row_id` per snippet,
   before any model trains. Every model's predictions are saved as `(logits, labels, ids)`; evaluation
   reads files only and aligns on row ids, so "same rows for every model" is something I can check rather
   than assert. Train is deduplicated against every eval split, exact and MinHash — near-duplicate
   solutions are everywhere in competitive-programming-style corpora.
2. **The mechanism, correctly named.** The bias is `score_qk += sign·α·σ(mean(H) − H_k)` where `H_k` is
   the entropy of a learned softmax over key self-similarity. It is an additive content-driven attention
   bias, the same family as ALiBi, and at `sign=+1` it boosts *low*-entropy keys — the opposite of what
   the original README claimed. I kept the sign as an ablation rather than quietly fixing the text.
3. **The control.** `control` = the same extra block with `α=0` and no style fusion. If GPAD beats plain
   CodeBERT but not the control, the gain was capacity, not entropy. The README branch is chosen
   automatically from whether the paired-bootstrap CI of (gpad − control) excludes zero on an OOD split.
4. **Real entropy, separately.** `lm_features.py` computes per-token NLL and `H(p(·|x<t))` from a frozen
   Qwen2.5-Coder-0.5B — the DetectGPT/Binoculars idea. Machine code tends to be lower-perplexity under
   another model. These feed a GBDT and can be fused into GPAD, which is also a nice interpretability
   story because the features have names.
5. **Imbalance.** Task B is ~88% human, which is why the original accuracy 0.64 / macro-F1 0.29 looked
   contradictory: that model was close to a majority-class predictor. I cap the human share in training
   only, keep the true prior in evaluation, default to class-weighted CE (focal and logit adjustment are
   there too), and report macro-F1 and balanced accuracy next to accuracy.
6. **Deployment reality.** Temperature scaling fitted on ID-val, ECE measured on OOD; PR curve and a
   threshold chosen for a 5:1 FP:FN cost; latency p50/p95 for fp32/fp16/int8-dynamic at batch 1 and 32;
   LoRA vs full fine-tune on trainable parameters and time.

## What I'd do with more compute

Full training data instead of a 20k-row subsample; 3+ seeds for the transformers; a larger code LM for
the entropy features (and the true Binoculars ratio with a second model); domain-adversarial training
over language to attack the shortcut directly; open-set rejection for Task B so an unseen generator can
be answered "unknown" instead of forced into a known family; distillation and ONNX export for serving.

## Failures I own

- **The entropy naming was wrong.** Three different things were called entropy across the old code and
  README. Fixed by renaming the mechanism and introducing real predictive entropy under its own name.
- **No capacity control.** The headline claim of the original project could not be supported by its own
  experiments. The rebuild tests it and reports whatever comes out.
- **Non-comparable tables.** Task A compared models on 349 vs 17,226 rows. Now every table states `n` and
  the rows are aligned by id.
- **Unsupported superlatives.** "Entropy-only is best" came from 0.02–0.07 macro-F1 gaps on single runs.
  Re-tested with a paired bootstrap; the report uses the word "tie" when the interval contains zero.
- **Python-only stylometry.** The old `radon` features silently returned zeros for C++ and Java, which
  made them a language detector. Replaced with language-agnostic features; `radon` is kept only as a
  labelled legacy ablation.

## Ten questions I should be able to answer

1. *Why macro-F1 and not accuracy?* — Task B is 88% human; accuracy rewards predicting "human". Macro-F1
   weights all 11 classes equally, balanced accuracy is reported next to it.
2. *Why a capacity control?* — GPAD adds a transformer block. Without `α=0, no style` at the same
   parameter count, any gain is confounded with capacity.
3. *Why a bootstrap CI and not a t-test?* — macro-F1 is not a mean of i.i.d. per-sample scores; resampling
   rows gives an interval without distributional assumptions. McNemar is reported alongside because it is
   the right paired test for correct/incorrect on the same items.
4. *Why head+tail truncation?* — imports and the closing lines carry style signal; plain head truncation
   throws the tail away. 192+64 at `max_len=256`.
5. *How does LoRA work and what did it cost?* — `W + (α/r)·BA` on query/value, rank 8, encoder frozen,
   head trainable; the table reports trainable parameters, wall-clock and macro-F1 versus full fine-tune.
6. *What is the difference between your attention entropy and perplexity-based detection?* — The first is
   an architectural bias computed from the encoder's own keys; the second is a property of a frozen
   language model's predictive distribution over the actual tokens. Only the second is "entropy" in the
   information-theoretic sense the detection literature uses.
7. *Why is the model worse on an unseen language?* — Because a lot of the in-distribution signal is
   formatting and comment conventions that do not transfer. The `length_only` probe and the
   comment-stripping stress test quantify how much.
8. *How would you deploy it?* — Threshold by cost, not argmax; temperature-scale and re-fit per domain;
   monitor the score distribution for drift; re-measure whenever a new model family is released; never
   expose a verdict without the uncertainty and the caveat.
9. *How would you handle a brand-new LLM?* — The held-out-generator split is exactly this test. Mitigation:
   open-set rejection via max-softmax/energy, periodic re-training, and keeping the task binary where
   possible since attribution degrades faster than detection.
10. *What would you delete from this project?* — The stacking stage: it is cheap and standard, and the
    honest caveat (fitted on in-sample logits) makes it the weakest row in the table.
