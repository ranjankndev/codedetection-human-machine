# PROGRESS (Claude: update after every step; a fresh session resumes from the first unchecked box)

Last updated: 2026-10-13 07:10 IST | Kaggle relaunches used: 2 / 3 (both blocked by CLI auth, see below) | README branch: pending the real run

## Phase P0 — build locally (CPU)
- [x] S0  Baseline: install, `pytest -q` green on the provided code (17 tests, py3.14 / torch 2.10 / transformers 5.15)
- [x] S1  Hygiene: starter pack promoted to the repo root, `.gitignore`, git init, first commit
- [x] S2  scripts/discover_data.py, scripts/make_fixture.py (fixtures now randomised so dedup has work to do)
- [x] S3  scripts/build_splits.py + codedet/splits.py (+ data_report.json, official_val split when labeled)
- [x] S4  classical.py + scripts/run_classical.py (majority, length-only probe, TF-IDF+LR, style GBDT, LM GBDT)
- [x] S5  scripts/extract_lm_features.py (skippable stage; records why it was skipped)
- [x] S6  scripts/train.py + codedet/zoo.py (codebert / control / gpad / entropy-only / style-only / sign-flip / lora)
- [x] S7  models/lora.py (peft, with an explicit fallback implementation when peft is unavailable)
- [x] S8  scripts/evaluate.py (bootstrap CIs, paired bootstrap + McNemar, per-slice, ECE, temperature scaling)
- [x] S9  scripts/run_all.py (stage DAG, resume markers, status.json, per-stage caps, degradation ladder)
- [x] S10 kaggle/kaggle_entry.py + scripts/kaggle_drive.py (+ Colab and Kaggle fallback notebooks)
- [x] S11 GATE P0: `run_all --preset local_cpu` completed every stage in 10 min on fixtures; pytest green

## Phase P1/P2 — Kaggle
- [ ] S12 Kaggle smoke — **blocked**: the installed Kaggle CLI accepts the configured legacy key for read
      operations but returns "Authentication required" for `datasets create` and `kernels push`. Needs one
      interactive `kaggle auth login` by the repo owner; see KNOWN_GAPS.md for the exact commands.
- [ ] S13 Kaggle lite — blocked by the same step.
- [x] S12b Fallback instead of Kaggle: the official data was fetched from the Hugging Face mirror
      (`daniilor/SemEval-2026-Task13`) and a real-data CPU run (`configs/local_real.yaml`) is in progress.

## Phase P3/P4 — analyze and finalize
- [x] S14 scripts/analyze.py (stress test, permutation importance, confusion, reliability, PR + cost threshold, errors)
- [x] S15 scripts/bench.py + app/demo.py
- [x] S16 make_report.py, README, MODEL_CARD, INTERVIEW_NOTES, slides_update.md, KNOWN_GAPS.md
- [ ] S17 Publish the real-data numbers into README/summary.md and tag `v1.0-portfolio`

## Decisions / notes log (append; one line each)
- 2026-10-13 Promoted `codedet-starter/` to the repo root; the starter copy stays on disk, untracked.
- 2026-10-13 `head_tail_ids` fixed for transformers 5 (`build_inputs_with_special_tokens` was dropped from fast tokenizers).
- 2026-10-13 Artifact contract: every model writes `probs_<split>.npz` of (logits, labels, ids); evaluation reads files only.
- 2026-10-13 Task B train is built to a target composition (machines first, then humans up to the cap); a proportional sample of an 88%-human corpus starved the 10 LLM families.
- 2026-10-13 Human cap moved to AFTER dedup — near-dup removal drops machine rows preferentially and was undoing the cap.
- 2026-10-13 Kaggle write API refused (CLI wants the newer access token); dataset fetched from the HF mirror instead. 2 of 3 relaunches used.
- 2026-10-13 GitHub repo created and pushed: https://github.com/ranjankndev/codedetection-human-machine (needed for the kernel-clone path).
- 2026-10-13 No GPU on this machine: CodeBERT costs ~6 s/step on CPU, so `local_real` uses 1k train rows / len 128 / 1 epoch. `lite` is unchanged for the GPU run.
- 2026-10-13 LM-feature stage killed on CPU: Qwen2.5-Coder-0.5B over ~5k rows projected >3 h. Stage is optional and recorded as skipped; a smaller code LM is the fallback if time allows.
