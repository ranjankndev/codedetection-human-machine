# Original competition-time results (from the team deck; NOT re-validated)
Use only in a clearly labelled "competition-time results" section. Re-run everything for the portfolio numbers.

## Task A, macro-F1 (Precision / Recall in the deck)
| Model | Split | Samples | F1 (macro) |
|---|---|---|---|
| Baseline (CodeBERT) | seen lang, seen dom | 37,117 | 0.9731 |
| | unseen lang, seen dom | 27,722 | 0.5676 |
| | seen lang, unseen dom | 46,512 | 0.8692 |
| | unseen lang, unseen dom | **349** | 0.5606 |
| GPAD-1 | seen / seen | 37,117 | 0.9459 |
| | unseen lang / seen dom | 27,722 | 0.5194 |
| | seen lang / unseen dom | 46,512 | 0.8548 |
| | unseen / unseen | **349** | 0.4898 |
| GPAD-2 | seen / seen | 37,117 | 0.9957 |
| | unseen lang / seen dom | 27,722 | 0.7268 |
| | seen lang / unseen dom | 46,512 | 0.9579 |
| | unseen / unseen | **17,226** | 0.6538 |

Caveat: the last row of each model uses different sample counts (349 vs 17,226): not comparable.

## Task B (single runs, no confidence intervals)
| Variant | Overall Acc | Overall F1-M | Seen Acc | Seen F1-M | Unseen Acc | Unseen F1-M |
|---|---|---|---|---|---|---|
| CodeBERT baseline | 0.64 | 0.29 | 0.85 | 0.47 | 0.41 | 0.14 |
| + Entropy + Style | 0.60 | 0.26 | 0.80 | 0.37 | 0.37 | 0.14 |
| + Style only | 0.62 | 0.20 | 0.76 | 0.19 | 0.46 | 0.12 |
| + Entropy only | 0.63 | 0.34 | 0.85 | 0.50 | 0.38 | 0.21 |

Deck claims to re-test: "entropy-only generalises best to unseen authors", "style features limit performance", "language shift is harder than domain shift".
