# CLAUDE.md — Machine-generated code detection (SemEval-2026 Task 13), portfolio revamp

Read this file at the start of EVERY session, then read `PROGRESS.md` and continue from the first unchecked step.
Full background: `PLAN.md` (Section A = autonomy rules, overrides everything else), `docs/design_doc.html`, `docs/OLD_CODE_AUDIT.md`.

## 1. What this project is
Given a code snippet: Task A = human vs machine (binary); Task B = human vs 10 LLM families (11-way).
Original idea ("GPAD"): CodeBERT + one extra attention block with an additive entropy bias + stylometric fusion.
Revamp goal: same project made **honest, reproducible, small enough for a free Kaggle GPU (smoke ≈5 min, lite ≈35–50 min)**,
and showing interview-grade ML judgment (OOD splits, ablation with a capacity control, bootstrap CIs, calibration, imbalance, LoRA, latency).
This is a portfolio project, NOT a leaderboard attempt. Never invent numbers: every number in README/slides comes from `results/*.json`.

## 2. Operating contract (autonomy)
- **Do not ask the user questions.** Use the decision rules in PLAN.md §A.5, the degradation ladder in §A.6, and the phase gates in §A.4.
- Work in small steps (Section 5). After each step: run tests, update `PROGRESS.md` (tick the box + one line of what changed), `git commit`.
- Sessions can end at any time (usage limits). Therefore: keep `PROGRESS.md` current, commit often, never leave the repo broken at a commit.
- Use **plan → implement → test → commit**. Keep context small: don't paste big files/logs into the chat; read with line ranges, `tail -n 40` logs.
- Max **3 Kaggle relaunches** in total. If blocked after that: finish with what exists and write `KNOWN_GAPS.md`.
- Never print, log or commit credentials. Never read `~/.kaggle/*`. Hugging Face upload is OFF by default.

## 3. Repo map
```
CLAUDE.md  PLAN.md  PROGRESS.md  KNOWN_GAPS.md(created when needed)
pyproject.toml  requirements-extra.txt
configs/      smoke.yaml  lite.yaml  local_cpu.yaml
src/codedet/  (PORTED + TESTED, extend don't rewrite — see docs/OLD_CODE_AUDIT.md)
  utils.py labels.py data.py features.py lm_features.py datasets.py losses.py
  models/gpad.py  trainloop.py  eval.py  inference.py  fixtures.py
tests/        pytest, CPU only, no downloads (tiny random encoder + fake tokenizer)
scripts/      TO BUILD: discover_data, build_splits, run_classical, extract_lm_features, train, evaluate,
              analyze, bench, make_report, run_all, kaggle_drive
kaggle/       TO BUILD: kernel-metadata.json, kaggle_entry.py
docs/         original_deck_Presentation.pdf, design_doc.html, OLD_CODE_AUDIT.md, ORIGINAL_RESULTS.md, LAUNCH_PROMPT.md, SETUP_CHECKLIST.md
legacy/       old taska/ taskb/ scripts, kept verbatim for reference. NEVER import from legacy/. Don't edit.
results/      committed small JSON/MD/PNG only (no checkpoints, no data)
```

## 4. Commands
```bash
pip install -e ".[dev]"                       # package + pytest (torch/transformers: use the environment's own)
pip install -r requirements-extra.txt         # only if a stage needs peft/datasketch/gradio/matplotlib
pytest -q                                     # must be green before every commit
python scripts/make_fixture.py                # synthetic parquet for CPU dry runs
python scripts/run_all.py --preset smoke --cpu-only --data fixtures/      # P0 gate
python scripts/kaggle_drive.py --preset smoke                              # P1 (needs kaggle CLI + token)
python scripts/kaggle_drive.py --preset lite                               # P2
```

## 5. Step-by-step process (tick them in PROGRESS.md)
Each step lists: what, which existing code to reuse, and the acceptance check.

**S0 Baseline.** Create venv, install, run `pytest -q`. All provided tests must pass before you touch anything. If one fails, fix it first (versions drift).
**S1 Hygiene.** `.gitignore` is provided. Create `README.md` skeleton (title + "work in progress"), `git init` if needed, first commit.
**S2 Data discovery + fixtures.** `scripts/discover_data.py` (use `data.discover_parquets`, print columns/label maps/language/generator counts, write `results/data_discovery.json`). `scripts/make_fixture.py` (use `fixtures.make_fixture`, write `fixtures/task_a.parquet`, `fixtures/task_b.parquet`). Never hard-code Kaggle paths; old code used `/kaggle/input/semeval-2026-task13/SemEval-2026-Task13/task_{a,b}/task_*_training_set.parquet` — verify, don't assume.
**S3 Splits.** `scripts/build_splits.py --task A|B --config configs/X.yaml`: load → `data.stratified_subsample` (Task B also `cap_human_share` for TRAIN only) → `dedup_exact` train vs every eval split (+ `near_duplicate_mask` if datasketch installed) → OOD splits: Task A `holdout_language_split(df,"Java")`; Task B `holdout_generator_split` with ≥1 held-out generator from up to 3 families that have ≥2 generators → write parquet + `results/data_report.json` (sizes, class counts, per-language counts, dup counts). If official labeled OOD test files exist, also evaluate on them using the same rows for every model.
**S4 Classical baselines.** `src/codedet/classical.py` + `scripts/run_classical.py`: A0 majority + length/whitespace-only LR (shortcut baseline), A1 char 3–5-gram TF-IDF + LR, A2 `style_matrix` + HistGradientBoosting (LightGBM only if installed), B1 TF-IDF + class-weighted LR. Save probs `.npz` + metrics JSON per (model, split).
**S5 LM features.** `scripts/extract_lm_features.py`: `lm_features.lm_features(Qwen2.5-Coder-0.5B,...)` → cached `.npy`; then A3 = HistGradientBoosting on `[style | lm]`. Skippable stage if the model can't be downloaded (record in status.json).
**S6 Neural training.** `scripts/train.py --task A|B --model codebert|gpad|control|gpad_entropy_only|gpad_style_only|lora`. Build with `GPADClassifier(GPADConfig(...))`; flags: codebert = `num_gpad_layers=0`; control = `entropy_alpha=0, use_style=False`; entropy-only = `use_style=False`; style-only = `entropy_alpha=0`. `datasets.CodeDataset` + `make_collate`, `StyleScaler` fit on TRAIN only (save `scaler.json`), `losses.make_loss` (Task B default `weighted_ce`), `trainloop.fit`, wrap in `run_with_oom_ladder`. Save `probs_{split}.npz` (logits, labels, ids), `metrics.json`, `history.json`; do NOT commit checkpoints.
**S7 LoRA (optional).** `models/lora.py`: peft `LoraConfig(r=8, target_modules=["query","value"])` on the encoder, head trainable; report `count_params`, time, F1.
**S8 Evaluation + stats.** `scripts/evaluate.py`: for every model × split: `eval.metrics_dict`, `bootstrap_ci`, `per_slice` (language / generator / seen), ECE, `fit_temperature` on ID-val then ECE on OOD; paired tests `paired_bootstrap_diff` + `mcnemar` for (gpad vs control), (gpad vs codebert), Task B (entropy-only vs codebert_weighted). Write `results/summary.json`.
**S9 Orchestrator.** `scripts/run_all.py`: stage DAG (data → classical → lm → neural → eval → analyze → bench → report) with per-stage marker files in `<out>/.done/`, `results/status.json` (`stage,status,seconds,error,degradation`), required vs optional stages (PLAN §A.5), per-stage try/except, global time budget from config.
**S10 Kaggle bundle.** `kaggle/kernel-metadata.json` (template in `kaggle/`), `kaggle/kaggle_entry.py` (script kernel: pip install only missing extras, clone repo or read attached dataset, `pip install -e .`, run `run_all.py`), `scripts/kaggle_drive.py` (wrap `kaggle kernels push|status|output`; poll every 60 s with timeout; exit code from status.json; exit 75 if quota exhausted).
**S11 P0 gate.** CPU dry run on fixtures finishes and writes every results file. `pytest -q` green.
**S12 Kaggle smoke (P1)** then **S13 Kaggle lite (P2)** via `kaggle_drive.py`. Download outputs into `results/`. Gates in PLAN §A.4.
**S14 Analysis.** `scripts/analyze.py`: shortcut stress test (strip comments / normalise whitespace on OOD set, re-score), feature importance (permutation on style features), Task B confusion matrix + per-class table, reliability diagram, PR curve + threshold for a stated cost ratio, 10 hand-inspected errors → `results/error_analysis.md`.
**S15 Bench + demo.** `scripts/bench.py` (latency p50/p95, batch 1 & 32, fp32/fp16/int8-dynamic on CPU via `torch.quantization.quantize_dynamic`, params, peak mem); `app/demo.py` Gradio (loads checkpoint from path/env var, shows temperature-scaled P(machine), LM gauges, out-of-range warning).
**S16 Docs + release.** `scripts/make_report.py` → `results/summary.md` + figures; README (pick the branch by the A6-vs-A7 test: PLAN §A.5), `docs/MODEL_CARD.md`, `docs/INTERVIEW_NOTES.md`, `docs/slides_update.md`; update PROGRESS; tag `v1.0-portfolio`; push only if `gh auth status` succeeds.

## 6. How to code here (conventions)
- Python ≥3.10, type hints on public functions, short docstrings that say WHY. `from __future__ import annotations`.
- Package code in `src/codedet/`, thin CLI wrappers in `scripts/`. Scripts parse args with argparse + a YAML config; no `input()`, no globals, no notebooks-as-source.
- Torch-free modules stay torch-free (`data.py`, `features.py`, `eval.py`, `labels.py`) so CPU tests are fast and CLI stages don't import torch needlessly.
- Determinism: `utils.set_seed(cfg.seed)` at every entrypoint; log the seed and config into each result JSON.
- Everything a stage produces is a file (`.npz` probs, `.json` metrics). Evaluation reads files only; it never re-runs a model.
- Evaluate every model on **identical rows** per split (store row ids). Report macro-F1 with 95% bootstrap CI; use `eval.macro_f1_fast` (labels = range(K)) so missing classes count as 0.
- Write results JSON after **each** model, not at the end.
- Tests: one test file per module, tiny data (`fixtures.make_fixture`), tiny random encoder + `FakeTokenizer` from `tests/conftest.py`; no network in tests.
- Logging via `utils.get_logger`; never log code samples at INFO beyond 100 chars; never log tokens.
- Commits: imperative subject (≤72 chars), body says what/why. Run `pytest -q` first.
- Prefer editing the ported modules over adding parallel versions. No new `*_v2.py` files.
- Don't add dependencies without need; Kaggle already has torch/transformers/sklearn/pandas/pyarrow/scipy. Never pin torch/pyarrow.

## 7. Domain facts and traps (from auditing the old code)
1. **Entropy was three different things** in the old code (hidden-dim softmax entropy in Task B; learned key-similarity entropy in `gpad_v2_model.py`; "token uncertainty" in the README). In the revamp: `AttnEntropy` = attention-entropy bias (document it exactly as in `models/gpad.py`); `lm_features.py` = true predictive entropy. Use these names; never call the first "token entropy".
2. `sigmoid(mean(H) − H)` boosts **low**-entropy tokens at `entropy_sign=+1`. The old README claimed the opposite. Keep the `sign` ablation.
3. GPAD adds an extra transformer block → **always run the capacity control** (`entropy_alpha=0, use_style=False`) or the entropy claim is meaningless.
4. Old Task A table compared models on different test sizes (349 vs 17,226 rows for unseen/unseen) → not comparable. Always same rows.
5. Task B: humans ≈ 88% of train; accuracy 0.64 vs macro-F1 0.29 = majority-class illusion. Use macro-F1, balanced accuracy, weighted CE / focal / logit adjustment; cap human share in TRAIN only.
6. Old `radon` stylometry only parses Python → zeros for C++/Java → acts as a language detector (shortcut). Use `features.stylometric_features` (language-agnostic); keep `radon_features` only as a labeled legacy ablation.
7. Old code padded every sample to `max_length` and computed stylometry inside `forward()` per batch. Use dynamic padding, head+tail truncation (`data.head_tail_ids`), cached z-scored features.
8. Old trainer used `evaluation_strategy` (renamed `eval_strategy` in recent transformers) and monkey-patched Accelerate. Use `trainloop.fit` instead of HF Trainer.
9. Old data columns: `code`, `generator` ("Human" or e.g. "Qwen/Qwen2.5-Coder-7B"), plus `label`, `language`. Task A label derived from generator in old code; check polarity against the official label map before trusting `label`.
10. Old Task B README says 20-dim stylometry, v2 code used 16→32, cached `.npy` were 20-dim. Now: `STYLE_DIM=16` from `features.py`; `GPADConfig.style_in_dim` must match (+`LM_DIM` if concatenated).
11. Competition rules (still honour them): only official training data; no third-party pretrained AI-code detectors; general code models (CodeBERT, Qwen-Coder) are OK.
12. `from_pretrained` needs internet: Kaggle needs Internet ON (or models attached). A missing download must mark the stage failed/skipped, not crash the run.

## 8. Kaggle specifics
- Kernel: script kernel, GPU on, Internet on, `dataset_sources: ["daniilor/semeval-2026-task13"]`, private. Outputs under `/kaggle/working`.
- Never `pip install` torch/transformers/numpy. `pip install -q peft datasketch` only if import fails.
- Check the current Kaggle quota/limits at run time via CLI errors; don't hard-code limits. GPU quota exhausted → exit 75, finalize with CPU results and note "GPU runs pending".
- Presets: `smoke` ≈5 min (2k rows, len 128, 1 epoch), `lite` ≈35–50 min (≈20k rows, len 256, 2 epochs). If over budget shrink rows first; never drop CIs or the control.

## 9. Definition of done
All boxes in PLAN.md §7 plus: `pytest -q` green; `results/summary.md` generated; README numbers match `results/*.json` (a test asserts it);
`KNOWN_GAPS.md` lists anything skipped; no secrets/data/checkpoints committed; tag `v1.0-portfolio`.
At the end, print a 10-line summary: results table, README branch chosen and why, what was skipped, reproduce commands.

## 10. Do NOT
- Do not rewrite the ported modules from scratch or "modernise" them without a failing test that demands it.
- Do not import from `legacy/` or copy its monkey-patches.
- Do not train on anything but the official training data. Do not use pretrained AI-code detectors.
- Do not claim "entropy helps" unless the paired bootstrap against the control excludes 0.
- Do not stop to ask the user. Decide by rule, record the decision in PROGRESS.md.
