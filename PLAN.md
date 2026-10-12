# PLAN v2 (autonomous): Revamp "Code Detection – Human vs Machine" (SemEval-2026 Task 13) into an interview-ready portfolio project

Audience: Claude Code (executor). Owner: Ranjan (ranjan56cse). Repo: `Codedetection-Human-Machine`.
Goal: same project, **cleaner, honest, reproducible, runnable end-to-end on Kaggle in < 1 hour**, and visibly demonstrating the ML concepts an AI-engineer interviewer asks about.
Non-goal: leaderboard chasing. Do not rewrite the idea; fix and rigorously evaluate it.

---

## A. AUTONOMOUS EXECUTION DESIGN (read this first; it overrides any "stop and ask the user" wording below)

**Goal:** after one ~10-minute setup, the user starts Claude Code once and comes back to a finished repo: code, tests, Kaggle results, README, model card, slide text, all committed locally (and pushed if GitHub is authenticated).

### A.1 Where can Claude run this by itself? (verified vs. assumed)
| Route | Verdict | Notes |
|---|---|---|
| **1. Claude Code on the user's laptop/VM + Kaggle CLI** | **Recommended** | Claude Code writes code, tests on CPU locally, then `kaggle kernels push` → polls `kaggle kernels status` → `kaggle kernels output` → reads JSON → fixes/relaunches → writes README. Needs outbound internet to kaggle.com (a normal laptop has it). |
| 2. Claude Code headless *inside* a Kaggle notebook (`ANTHROPIC_API_KEY` in Kaggle Secrets, `claude -p`) | Fallback only | It can run the pipeline but cannot relaunch its own notebook and burns API credits; not needed because the pipeline itself is deterministic. |
| 3. Google Colab | Manual fallback only | No supported programmatic way to start a free Colab GPU runtime. Ship `colab/run_colab.ipynb` (same code, data pulled from Hugging Face `DaniilOr/SemEval-2026-Task13` via `huggingface_hub`); user clicks "Run all". |
| 4. This claude.ai chat sandbox | **Not possible** | Tested: kaggle.com, colab.research.google.com and huggingface.co are blocked from the sandbox shell, and browser control is not on the free plan. Use it only for planning/doc work. |

### A.2 One-time setup the USER does (the only required involvement)
1. Kaggle account, **phone-verified** (required to enable GPU + Internet in notebooks). Accept the dataset/competition terms for `daniilor/semeval-2026-task13` once in the browser.
2. Create a Kaggle API token (Settings → API) and store it per the current Kaggle CLI docs (`~/.kaggle/kaggle.json`, `chmod 600`, or the env var the installed CLI version documents). Never commit it.
3. `pip install kaggle` (and `gh auth login` if they want Claude to push to GitHub).
4. Put the project folder in a git repo; start Claude Code in it with the permissions allowlist from A.7; paste the **Launch Prompt** (A.8).
5. Keep the machine awake until Claude reports done (expect ~2–4 h wall time, mostly waiting on Kaggle).

Optional, ≈10 min at the very end: skim README/model card before posting publicly.

### A.3 Autonomy principles
1. **Never block on the user.** Every former "stop and ask" is replaced by a decision rule (A.5) or a documented default.
2. **Everything non-interactive:** no `input()`, no prompts, no browser steps, no secrets needed (Hugging Face push is **off by default**, so no HF token is required).
3. **Machine-readable state:** every stage writes `results/status.json` (`stage`, `status: ok|failed|skipped`, `seconds`, `error`, `degradation_applied`). Claude Code decides the next action only from these files.
4. **Idempotent, resumable stages:** each stage writes a marker file; re-running the kernel skips completed stages (cache lives in the kernel's `/kaggle/working`; Claude Code also downloads outputs after each run and re-uploads `cache/` as an input dataset only if a relaunch is needed).
5. **Cheapest-first, fail-fast:** local CPU tests → Kaggle `smoke` → Kaggle `lite`. A phase must pass its gate (A.4) before the next starts.
6. **Graceful degradation ladder** instead of crashing (A.6).
7. **Claims are generated from numbers, never typed by hand** (README has templated branches for each outcome).

### A.4 Autonomous phases and pass/fail gates
| Phase | Where | What Claude Code does | Gate to continue |
|---|---|---|---|
| P0 Build | local CPU | Implement WP1–WP3, WP6 code, `kaggle/` bundle, `run_all.py`; write fixtures (200-row synthetic parquet) | `pytest -q` green; `python scripts/run_all.py --preset smoke --cpu-only --data fixtures/` completes and writes all result JSONs |
| P1 Kaggle smoke | Kaggle GPU kernel | `kaggle_drive.py push --preset smoke` → poll → download | `status.json`: all *required* stages ok; every model's macro-F1 > majority baseline; runtime < 10 min. Else apply fix/degrade (A.6), max 3 relaunches |
| P2 Kaggle lite | Kaggle GPU kernel | Same with `--preset lite` | Required stages ok; results table complete with CIs; if some *optional* stage failed it is listed in `KNOWN_GAPS.md` and the run still counts |
| P3 Analyze | local | `make_report.py`, error analysis, figures, README branch selection (A.5) | `summary.md` and figures exist; README numbers match JSON (a test asserts this) |
| P4 Finalize | local | Model card, INTERVIEW_NOTES, slide text (`docs/slides_update.md`, plus `.pptx` via python-pptx if installed), commit, tag `v1.0-portfolio`, push if `gh` is authenticated | `git status` clean; DoD checklist (section 7) all ticked or listed in `KNOWN_GAPS.md` |

### A.5 Decision rules replacing human checkpoints
- **GPAD vs capacity control (A6 vs A7):** if the 95% CI of the macro-F1 difference (paired bootstrap) excludes 0 on ≥1 OOD split → README branch "entropy bias helps"; otherwise branch "no gain beyond capacity; LM-entropy features / class-aware loss / shortcut analysis are the findings". Both branches are pre-written templates.
- **Task B "entropy-only best" claim:** keep only if it beats B2 with non-overlapping CI; else report as "not significant".
- **Seeds:** use 3 seeds for cheap models; for transformers use 2 seeds if the run is projected < 45 min total, else 1 seed + CI and state it.
- **Labeled official OOD files missing?** Use proxy OOD splits (WP2); no question asked.
- **Kaggle dataset folder differs from expectation?** `discover_data()` globs and logs; fail only if no parquet with `code` and `label` is found.
- **Optional stages** (skippable without failing the run): LoRA, stacking, DANN, open-set, quantisation bench, demo build.
- **Required stages:** data+splits, classical baselines, CodeBERT, GPAD, capacity control, evaluation, report.

### A.6 Self-healing ladder (implemented in `run_all.py`, not by hand)
1. CUDA OOM → batch ÷2 and grad-accum ×2 (keep effective batch) → enable gradient checkpointing → `max_length` 256→192→128.
2. Stage time over its budget (smoke 10 min, lite 60 min total; per-stage caps) → reduce train rows ×0.5 and retry once.
3. HF model download fails (no internet) → read from attached Kaggle Model/dataset path if present, else mark stage failed and continue with the stages that don't need it (e.g. skip LM-feature stage).
4. Package missing (`peft`, `lightgbm`, `datasketch`) → `pip install` once, else skip dependent optional stage.
5. NaN loss → lower LR ×0.5 once, then switch to fp32 for that model.
6. Kernel dies or exceeds limits → Claude Code relaunches with the next degradation preset (`lite` → `lite_small`), **max 3 relaunches total**; afterwards finalize with what exists plus `KNOWN_GAPS.md`.

### A.7 Files to add for autonomy
```
kaggle/
  kernel-metadata.json     # id: <kaggle-username>/codedet-<preset>, code_file: kaggle_entry.py,
                           # kernel_type: script, enable_gpu: true, enable_internet: true,
                           # dataset_sources: ["daniilor/semeval-2026-task13"], is_private: true
  kaggle_entry.py          # entry: pip install -q peft lightgbm datasketch (only if missing);
                           # git clone <repo> or read attached "codedet-src" dataset; pip install -e .;
                           # python scripts/run_all.py --preset $PRESET --out /kaggle/working/results
scripts/
  run_all.py               # stage DAG, resume markers, degradation ladder, status.json
  kaggle_drive.py          # wraps the kaggle CLI: push -> poll (every 60 s, timeout) -> output download -> exit code from status.json
  discover_data.py         # globs parquet, prints/records columns, label maps, language & generator counts
colab/run_colab.ipynb      # manual fallback, HF datasets download
.claude/settings.json      # permission allowlist: Bash(kaggle:*), Bash(python:*), Bash(pytest:*), Bash(pip:*),
                           # Bash(git:*), Bash(gh:*); deny reading ~/.kaggle/* contents into output, deny rm -rf
KNOWN_GAPS.md              # auto-written when a stage was skipped/degraded
```
- Prefer the allowlist over `--dangerously-skip-permissions`; use the latter only inside a throwaway VM/container.
- Kernel `PRESET` is passed through the kernel's environment/arguments embedded by `kaggle_drive.py` when it renders `kaggle_entry.py` (script kernels have no CLI args).
- Check current Kaggle limits at run time and encode them in `kaggle_drive.py` rather than assuming (weekly GPU quota, max session length, whether T4×2 or P100 is default). If GPU quota is exhausted, `kaggle_drive.py` exits with code 75 and Claude Code finalizes with the CPU-only results plus a clear "GPU runs pending" note; this is the one case that genuinely needs the user later.

### A.8 Launch Prompt (the user pastes this once into Claude Code)
> Read PLAN.md fully and execute it autonomously with minimal interaction. Follow Section A for autonomy rules: do not ask me questions; use the decision rules, the degradation ladder and the phase gates. Build locally first (P0), then run Kaggle smoke (P1) and lite (P2) through the Kaggle CLI, then analyze and finalize (P3–P4). Use at most 3 Kaggle relaunches. Never print or commit credentials. Write `KNOWN_GAPS.md` for anything skipped. When finished, give me a 10-line summary: results table, which README branch was chosen and why, what was skipped, and the commands to reproduce.

### A.9 What can still require the user (be upfront in the README/summary)
- Kaggle phone verification, terms acceptance, API token (one time).
- GPU quota exhausted or account restrictions on Kaggle.
- `git push` if GitHub is not authenticated on the machine.
- Final human read-through before publishing.

---

## 0. Ground rules (read first)

1. **Minimal run on Kaggle.** One notebook, free tier (T4 16 GB or P100, 2 CPU cores, ~7 GB RAM here / ~29 GB on Kaggle). Three presets:
   - `smoke` ≤ 5 min (2k train samples, proves the pipeline works)
   - `lite` ≈ 35–50 min total (the portfolio run; numbers go in README)
   - `full` (optional, documented only, not required)
2. **Competition rules still apply to the data/model choice:** only official SemEval training data; no third-party pretrained "AI-code detectors". General code models (CodeBERT, Qwen2.5-Coder, StarCoder) are fine.
3. **No secrets in the repo.** Tokens come from Kaggle Secrets / env vars. (Audit found none committed; keep it so. README currently has `<your_token>` placeholders and a hard-coded HF namespace; make both configurable.)
4. **Don't invent numbers.** Every number in README/slides must come from a JSON in `results/` produced by the code. Slide numbers from the old report are kept only in a clearly labeled "original competition-time results" section.
5. **Don't pin `torch`/`pyarrow` upper bounds** (current `requirements.txt` has `torch<3`, `pyarrow<20`, which can trigger reinstalls and break Kaggle's preinstalled CUDA stack). Use whatever Kaggle ships; list versions in README after the first successful run.
6. Keep git history; move old scripts to `legacy/` rather than deleting, so nothing is lost.

---

## 1. Audit: what exists and what is wrong (fix these)

Project facts
- Task A (binary human vs machine) and Task B (11-way: human + 10 LLM families). Backbone `microsoft/codebert-base`.
- "GPAD" = CodeBERT encoder → 1 extra transformer layer with an **additive entropy bias on attention scores** → gated stylometric fusion → MLP head. GPAD-1 / GPAD-2 are variants.
- Reported (Task A, macro-F1): seen/seen 0.9731 baseline → 0.9957 GPAD-2; unseen-lang 0.5676 → 0.7268; unseen-domain 0.8692 → 0.9579; unseen/unseen 0.5606 → 0.6538. Task B macro-F1: baseline 0.29, entropy-only 0.34, style-only 0.20, entropy+style 0.26; unseen-author F1 0.14–0.21.

Problems (each becomes a work item)
| # | Problem | Why an interviewer will poke it | Fix (section) |
|---|---|---|---|
| P1 | `taska/gpad_v2_main.py` and `gpad_v2_trainer.py` import `utils` (`get_logger`, `set_seed`, …) which **is not in the repo**. The main path cannot run. | "Does it run?" | WP1 |
| P2 | ~10.7k lines across ~25 near-duplicate scripts (`codebert-*-classifier-*.py`, `*-cpu`, `*-gpu2`, `*-eq-balanced`). | Reads as experiment debris | WP1 |
| P3 | The "entropy" is **not token/predictive entropy**. In code it's the entropy of a learned softmax over key self-similarity (`EntropyCalculator`), then `sigmoid(mean − entropy)`, which **up-weights low-entropy tokens**. The README says the opposite ("high entropy → high attention") and says entropy is computed from hidden states. | Conceptual correctness of the central idea | WP3 |
| P4 | No control for the real confound: GPAD adds an **extra transformer layer + params**. Gains may come from capacity, not entropy. | Ablation hygiene | WP4 |
| P5 | Task A table compares models on **different sample counts**: unseenlang_unseendom is n=349 for Baseline/GPAD-1 but n=17,226 for GPAD-2. Not comparable. | Evaluation validity | WP4 |
| P6 | Task B: accuracy 0.64 but macro-F1 0.29 → majority-class (human ≈ 88% of train) is dominating. "Entropy-only is best" rests on single runs with differences of 0.02–0.07 F1 and no CIs; "style features hurt" is likewise unsupported. | Imbalance + significance | WP2, WP4 |
| P7 | Big seen→unseen collapse (0.99 → 0.57 on unseen languages for the baseline) strongly suggests **shortcut learning** (language/format artifacts). Never analyzed. | Distribution shift | WP5 |
| P8 | Stylometric features are raw ratios, **unscaled**, computed per batch inside `forward()` on the main process; dims are inconsistent across docs/code (README says 20, v2 code uses 16→32, `.npy` files are 20-dim). | Feature hygiene, efficiency | WP3 |
| P9 | `taska/README.md` is "TBD". Root README is Kaggle command scratch notes. Committed 7 MB `.npy` features, `New Text Document.txt`. | Portfolio first impression | WP8 |
| P10 | No calibration, no per-language/per-class breakdown, no error analysis, no latency numbers, no tracked configs. | "What would you do in prod?" | WP5–WP7 |

---

## 2. Target repo layout

```
Codedetection-Human-Machine/
├── README.md                    # portfolio README (WP8)
├── pyproject.toml               # package "codedet", no hard torch pin
├── requirements-extra.txt       # only what Kaggle lacks: peft, lightgbm (if missing), gradio
├── configs/
│   ├── smoke.yaml               # 2k train / 1k val / 1 epoch / len 128
│   ├── lite.yaml                # ≈20k train / 5k val / 2 epochs / len 256   <- portfolio run
│   └── full.yaml                # documented only
├── src/codedet/
│   ├── data.py                  # load parquet, stratified subsample, dedup, OOD splits, head+tail truncation
│   ├── features.py              # stylometry (vectorized, scaled), LM NLL/entropy features
│   ├── models/
│   │   ├── gpad.py              # cleaned GPAD (rename entropy, controls, documented)
│   │   ├── lora.py              # CodeBERT + LoRA classifier
│   │   └── classical.py         # TF-IDF+LR, stylometry+LightGBM, stacking
│   ├── train.py                 # one trainer for A and B, YAML-driven, fp16, grad ckpt, early stop
│   ├── losses.py                # CE, class-weighted CE, focal, logit adjustment
│   ├── eval.py                  # macro-F1/P/R, balanced acc, per-slice, bootstrap CI, ECE, confusion matrix, McNemar
│   ├── calibrate.py             # temperature scaling
│   ├── bench.py                 # latency/throughput: fp32 vs fp16 vs int8-dynamic
│   └── utils.py                 # get_logger, set_seed  (fixes P1)
├── scripts/
│   ├── run_experiments.py       # runs the experiment matrix for a preset, writes results/*.json
│   └── make_report.py           # results/*.json -> results/summary.md + figures/*.png
├── kaggle/ + scripts/run_all.py + scripts/kaggle_drive.py   # autonomous Kaggle launcher (Section A.7)
├── notebooks/kaggle_minimal.ipynb   # manual fallback (same code); colab/run_colab.ipynb for Colab
├── app/demo.py                  # Gradio demo (paste code -> P(machine), top features)
├── results/                     # committed: small JSONs, summary.md, figures (NOT checkpoints, NOT data)
├── docs/MODEL_CARD.md, docs/ARCHITECTURE.md, docs/INTERVIEW_NOTES.md
├── tests/                       # tiny pytest: shapes, masking, determinism, metric correctness
└── legacy/                      # old taska/ taskb/ scripts moved verbatim; stylo-features/*.npy removed (regenerable)
```

---

## 3. Kaggle minimal-run design

**Setup (cell 1–3 of the notebook)**
- Add the competition dataset as Input (`daniilor/semeval-2026-task13`, mounted under `/kaggle/input/...`). Cell 1 must **discover paths** (`glob` for `*.parquet`, print columns, label maps, language/generator counts) and write them into the config rather than hard-coding folder names. Columns known from the task page: `code`, `label`, plus metadata such as `language`, `generator`.
- Settings: GPU on, Internet on (needed to download `microsoft/codebert-base` and, for WP3, `Qwen/Qwen2.5-Coder-0.5B`). Optional: attach those models as Kaggle Models to run offline.
- `pip install -q peft lightgbm gradio` only if missing; **never reinstall torch/transformers**.
- Clone the repo (or attach it as a dataset) → `pip install -e .`.
- HF token (only if pushing model) via `UserSecretsClient().get_secret("HF_TOKEN")`. `report_to="none"`.

**Budget for the `lite` preset (target, verify on first run and adjust)**
| Step | Device | Target time |
|---|---|---|
| Data prep + subsample + dedup + splits | CPU | ~3 min |
| Classical baselines (Task A + B) | CPU | ~5 min |
| LM features (Qwen2.5-Coder-0.5B fwd only, ≈25k snippets, len 256) | GPU | ~6–8 min |
| CodeBERT full fine-tune, Task A (2 epochs, 20k) | GPU fp16 | ~8–10 min |
| GPAD + controls, Task A (see matrix) | GPU fp16 | ~10–15 min |
| Task B (balanced subsample, 2–3 models) | GPU fp16 | ~10 min |
| Eval, calibration, bench, report | GPU/CPU | ~3 min |

Rules to keep it inside budget: max_len 256 with **head+tail truncation** (first 192 + last 64 tokens); dynamic padding; fp16 autocast; gradient checkpointing only if OOM; stylometric features **precomputed once** (vectorized, cached to `/kaggle/working/cache/*.npy`, not committed); save only the best checkpoint; write results JSON after every model so a crash loses nothing; `preset=smoke` is the first cell that runs end-to-end before `lite` is launched.

**Output:** `/kaggle/working/results/` (JSONs + `summary.md` + PNGs) and optionally a pushed HF model. User downloads `results/` and commits it.

---

## 4. Work packages (execute in order; each has acceptance criteria)

### WP1 — Make it run, then make it small
- Create `src/codedet/utils.py` (`get_logger`, `set_seed` with torch/numpy/random/cuda + `PYTHONHASHSEED`; add whatever else the two `from utils import (...)` lines at `gpad_v2_trainer.py:60` need).
- Move all current `taska/` and `taskb/` code to `legacy/` unchanged; extract only the logic worth keeping (GPAD layer, stylometry, label normalization `normalize_family`, seen/unseen marking `mark_seen`) into the new package.
- Replace the ~25 scripts with one `train.py` driven by YAML (`task: A|B`, `model: codebert|gpad|lora`, flags for each ablation).
- Remove committed `taskb/stylo-features/*.npy` and `New Text Document.txt`; update `.gitignore` (`results/*.pt`, `cache/`, `*.parquet`, `.env`).
- **Accept:** `pytest -q` passes on CPU in < 60 s; `python scripts/run_experiments.py --preset smoke --cpu-only` finishes on a laptop with a 200-row synthetic parquet fixture.

### WP2 — Data pipeline with honest splits
- Stratified subsample by (label × language) for `lite`; Task B subsample must **cap human** (e.g. ≤ 40% of rows) so the 10 LLM families are visible; keep the true prior in the *eval* set and report both macro-F1 and balanced accuracy.
- **Dedup + leakage check:** exact hash dedup, then MinHash/Jaccard near-dup (datasketch, or a cheap shingle hash) between train and every eval split; log the count and drop dups.
- **Create OOD evaluation yourself, with labels** (the official unseen-language/domain sets may be unlabeled locally; check first and use them if labeled):
  - Task A: `train = {C++, Python}`, `OOD-lang = Java` (held-out language), plus `ID-val`. If official labeled unseen-language/domain files exist, evaluate on them too, **same rows for every model** (fixes P5).
  - Task B: hold out 1–2 specific generators from several families (`unseen-generator`), plus `seen-generator` val. Print the generator→family map.
- **Accept:** `results/data_report.json` lists split sizes, class counts, per-language counts, dup counts; unit test asserts zero train/eval overlap by hash.

### WP3 — Fix and clarify the model (the "GPAD" story)
In `models/gpad.py`:
1. Rename the current mechanism **`attn_entropy_bias`** and document exactly what it is: an additive, per-key bias `α·σ(mean(H) − H_k)` on pre-softmax scores (an attention-bias idea like ALiBi/relative-position bias, but content-driven). Fix README/ARCHITECTURE so text, diagram, and code agree (low-entropy keys are boosted — or flip the sign and test both: `sign ∈ {+1,−1}` is a 2-line ablation).
2. Add **true predictive-entropy features** from a small causal code LM (`Qwen2.5-Coder-0.5B`, fwd only, no training): per snippet compute mean/std/max/quantiles of token NLL, mean token entropy `H(p(·|x<t))`, mean rank / top-1 hit-rate, and Binoculars-style NLL-over-cross-entropy if a second model is cheap enough. Cache them. These feed (a) a GBDT head and (b) the GPAD stylometric-fusion input. This is the zero-shot-detection idea (DetectGPT/Binoculars family) done properly, and it makes "entropy" true to its name.
3. Stylometry: vectorize, **z-score with train stats**, one fixed dimensionality (document it; the code decides, not the README), computed once in the Dataset/cache, not in `forward()`.
4. Keep `use_entropy`, `use_stylometric`, `num_gpad_layers`, `alpha` as config flags so ablations are one-line YAML changes.
5. Loss options for Task B: class-weighted CE, focal (γ=2), and **logit adjustment** (add `τ·log prior` at train / subtract at test). Default Task B to weighted CE.
- **Accept:** shape/mask unit tests; `alpha=0` and `num_gpad_layers=0` reproduce the plain CodeBERT head within tolerance; docs and code describe the same sign and formula.

### WP4 — Experiment matrix (the core of the portfolio)
Run in `lite`, results to `results/*.json`. Same rows for every model within a task.

Task A (binary)
| ID | Model | Purpose |
|---|---|---|
| A0 | length-only / whitespace-only logistic regression | **shortcut baseline** (P7) |
| A1 | TF-IDF char 3–5-gram + LogisticRegression | strong cheap baseline |
| A2 | Stylometry + LightGBM | interpretable baseline |
| A3 | LM NLL/entropy features + LightGBM (zero-shot-style) | proper "entropy" idea |
| A4 | CodeBERT full fine-tune | the old baseline, re-run |
| A5 | CodeBERT + **LoRA** (r=8, q/v + classifier) | PEFT comparison: trainable params, time, F1 |
| A6 | GPAD (attn-entropy bias + style) | the project idea |
| A7 | **Capacity control:** same extra layer, `alpha=0`, no style | isolates whether entropy bias matters (P4) |
| A8 | A4 head + A2/A3 features **stacked** (late fusion via LightGBM on A4 logits + features) | ensembling concept |

Task B (11-way)
| ID | Model |
|---|---|
| B1 | TF-IDF + LR (class-weighted) |
| B2 | CodeBERT, weighted CE |
| B3 | GPAD entropy-only (matches the old "best") |
| B4 | GPAD style-only |
| B5 | GPAD entropy+style |
| B6 | B2 + LM-feature stacking |

Statistics (mandatory)
- **Bootstrap 95% CI** (1,000 resamples) for macro-F1 on every row of the results table.
- **3 seeds** for the cheap models (A0–A3, B1) and for A4/A6/A7 on the `smoke`-sized data *or* 2 seeds on `lite` if time allows; otherwise 1 seed + bootstrap and **say so** in the README.
- **McNemar test** (A4 vs A6, A6 vs A7; B2 vs B3) on identical rows.
- Verdict language in README must follow the statistics: if the CI overlaps, write "no significant difference", even if the old slide said "best".
- **Accept:** `results/summary.md` has one table per task: rows = models, columns = ID-val, OOD split(s), each as `F1 [lo, hi]`; capacity-control row present; a short "what actually helped" paragraph generated from the numbers.

### WP5 — Analysis that shows ML judgment
- **Shortcut/robustness (P7):** per-language F1 on Task A; A0 vs A4 gap; feature importance (LightGBM gain, permutation importance on stylometry); top char-n-grams from A1 (e.g. comment style, trailing whitespace, snake_case); a **normalization stress test**: strip comments and normalize whitespace/identifiers in the OOD set and re-score A4/A6 (does the model rely on formatting?). Write 5–8 sentences interpreting it.
- **Imbalance (P6):** Task B confusion matrix, per-class P/R/F1 table, macro-F1 vs accuracy explanation; which families get confused (expect same-lineage clusters, e.g. Qwen/DeepSeek/01-ai).
- **Calibration:** reliability diagram + ECE before/after **temperature scaling** (fit on ID-val, evaluate on OOD); show that ID calibration breaks under shift.
- **Thresholding (Task A):** PR curve, threshold chosen for a stated cost ratio (a false "machine" accusation of a student is costlier than a miss); report the operating point on ID and OOD.
- **Error analysis:** 10 hand-inspected false positives / false negatives with a one-line cause each (`results/error_analysis.md`).
- **Stretch (only if time remains, mark as optional):** gradient-reversal **domain-adversarial** head over language (DANN) on A6 to reduce language shortcutting; report OOD-lang F1 delta honestly even if negative. Open-set rejection for Task B using max-softmax / energy score for "unknown generator".

### WP6 — Efficiency and serving numbers
`bench.py`: latency p50/p95 and throughput for batch 1 and 32 on GPU (fp32 vs fp16) and on CPU (fp32 vs **dynamic int8 quantization** via `torch.quantization.quantize_dynamic`); trainable-parameter counts (full FT vs LoRA vs GPAD head); peak GPU memory; training time per epoch. One table in README. Mention (not necessarily implement) ONNX export and why truncation to 256 matters for cost.

### WP7 — Demo
`app/demo.py`: Gradio app: paste code → P(machine) with temperature-scaled probability, the LM-feature gauge (mean NLL, mean entropy), top stylometric contributors, and a visible warning on language/length out of training range. Runs on CPU with the LoRA or distilled checkpoint; optional HF Space instructions in README. No checkpoint committed; `app/` loads from HF Hub id in env var.

### WP8 — Portfolio packaging
- **README.md** (top-to-bottom): 1-paragraph problem + why it matters; results table (from `summary.md`, with CIs); a single architecture diagram (corrected); "What I learned / what didn't work" (honest: e.g. capacity control outcome, shortcut findings, imbalance fix); reproduce-in-Kaggle badge + 5-line instructions; repo map; limitations; ethics note on false accusations; credits (SemEval-2026 Task 13 organizers, dataset license).
- `docs/MODEL_CARD.md` (intended use, data, metrics by slice, calibration, limitations).
- `docs/INTERVIEW_NOTES.md` (see section 6).
- **Slides:** keep the 8-slide deck's structure, add 3 slides: (1) "Audit & fixes" (P3–P6), (2) "Controlled ablation with CIs", (3) "Shortcut learning & calibration". Replace slide 4–7 tables with the regenerated ones; keep original numbers on one clearly labeled "competition-time results" slide. Fix slide 2's diagram to match the corrected architecture.
- Commit `results/` and `figures/`; tag `v1.0-portfolio`.

---

## 5. ML concepts map (what to be able to explain, and where it lives)

| Concept (2026 AI-engineer interview staples) | Where in the project | 30-second talking point |
|---|---|---|
| Distribution shift / OOD generalization / shortcut learning | WP2 splits, A0 baseline, WP5 stress test | "0.99 in-distribution but 0.57 on a new language told me the model learned format artifacts; here's the test that shows it." |
| Class imbalance & metric choice | Task B macro-F1 vs accuracy, weighted CE/focal/logit adjustment | "88% human ⇒ 64% accuracy was meaningless; I optimized macro-F1 and fixed the loss." |
| Ablation design & controls | A7 capacity control, `sign` ablation | "Before crediting entropy I added a parameter-matched control." |
| Statistical rigor | bootstrap CIs, McNemar, seeds | "Differences inside CI are reported as ties." |
| Attention mechanics | additive attention bias, masking, scaled dot-product | "My bias is added pre-softmax per key — same family as ALiBi/relative bias." |
| Perplexity / token entropy / zero-shot detection (LLM concepts) | LM features (A3) | "NLL and predictive entropy from a small code LM; Binoculars/DetectGPT idea; machine code tends to be lower-perplexity." |
| Transfer learning vs PEFT | A4 vs A5 (LoRA) | "LoRA: ΔW = (α/r)·BA, rank 8, trains ~1% of parameters; I measured the F1/time trade-off." |
| Training efficiency | fp16 autocast, grad checkpointing/accumulation, dynamic padding, head+tail truncation | "Memory/compute trade-offs and what I turned on and why." |
| Classical ML still matters | TF-IDF+LR, LightGBM, stacking | "Strong baselines first; stacking transformer logits with features." |
| Calibration | ECE, temperature scaling, shift | "Calibrated on ID, miscalibrated on OOD; threshold set by cost ratio." |
| Data leakage / dedup | MinHash near-dup check | "Near-duplicate problems in LeetCode-style data inflate scores." |
| Inference optimization | int8 dynamic quantization, latency p50/p95 | "Numbers on CPU and GPU, truncation cost." |
| Reproducibility / MLOps | YAML configs, seeds, results JSON → report, tests, model card | "One command regenerates every number in the README." |
| Responsible AI | threshold choice, false accusation cost, model card | "Detector output is evidence, not proof; FPR matters more than recall." |

---

## 6. `docs/INTERVIEW_NOTES.md` should contain
- 60-second pitch, 3-minute deep dive, and a "what I'd do with more compute" list (full data, DANN, larger code LM, multi-seed).
- Honest failure list: the original entropy naming mismatch, non-comparable test sizes, unsupported "entropy-only is best" claim, and what the controlled results showed. Owning these is a strength.
- 10 likely follow-up questions with short answers drawn from the actual results (why macro-F1, why LoRA, why not just fine-tune bigger, how would you deploy, how would you handle a new LLM release, how to avoid false accusations).

---

## 7. Definition of done
- [ ] `python scripts/kaggle_drive.py --preset lite` runs push → poll → download unattended; `smoke` < 5 min, `lite` ≤ ~50 min, no manual edits; `status.json` shows all required stages ok.
- [ ] Manual fallbacks work: `notebooks/kaggle_minimal.ipynb` on Kaggle and `colab/run_colab.ipynb` on Colab.
- [ ] `KNOWN_GAPS.md` exists (even if empty) and lists any skipped or degraded stage.
- [ ] `pytest -q` green on CPU.
- [ ] `results/summary.md`, `results/*.json`, figures generated by code; README numbers match them.
- [ ] README/ARCHITECTURE/code agree on what "entropy" is (P3 closed).
- [ ] Capacity-control row and CIs present (P4, P6 closed); all models in a table evaluated on identical rows (P5 closed).
- [ ] Shortcut stress test, calibration, imbalance analysis, latency table present.
- [ ] No tokens, data files, or checkpoints committed; `legacy/` holds the old scripts.
- [ ] Slides updated; honest framing of original vs re-run results.

## 8. Suggested order of work for Claude Code (so partial progress is still useful)
1. WP1 + WP2 (repo runs, splits, tests) → 2. classical baselines A0–A3/B1 + `eval.py` (CPU only, immediately demoable) → 3. `train.py` + A4/B2 → 4. WP3 GPAD fix + A6/A7/B3–B5 → 5. LoRA A5 + stacking A8/B6 → 6. WP5 analysis → 7. WP6 bench → 8. notebook, README, model card, demo, slides.
Commit after each step with a descriptive message. Do NOT stop for user review after steps 2 or 4: the ablation outcome selects the README branch automatically via Section A.5. If GPAD does not beat the capacity control, the honest story is "entropy bias did not help beyond extra capacity; LM entropy features and class-aware training did", which is still a strong interview narrative.

## 9. Open items Claude Code must resolve by inspection, not assumption
- Exact Kaggle input folder names and whether any **labeled** official unseen-language/domain/author test files exist locally (the task page says the public test set is on Kaggle; labels may be hidden). The proxy OOD splits in WP2 are the fallback and are valid either way.
- Which metadata columns exist in each parquet (`language`, `generator`, `domain`?) and the label maps (`label_to_id.json`).
- Whether the first `lite` timing matches the budget in section 3; if not, reduce train size before reducing rigor (keep CIs and the control).
