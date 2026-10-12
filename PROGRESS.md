# PROGRESS (Claude: update after every step; a fresh session resumes from the first unchecked box)

Last updated: (never)   |   Kaggle relaunches used: 0 / 3   |   README branch: (undecided)

## Phase P0 — build locally (CPU)
- [ ] S0  Baseline: venv, `pip install -e ".[dev]"`, `pytest -q` green on the provided code
- [ ] S1  Hygiene: README skeleton, git init, first commit
- [ ] S2  scripts/discover_data.py, scripts/make_fixture.py
- [ ] S3  scripts/build_splits.py (+ data_report.json)
- [ ] S4  classical.py + scripts/run_classical.py (A0 A1 A2 B1)
- [ ] S5  scripts/extract_lm_features.py (+ A3)
- [ ] S6  scripts/train.py (codebert / gpad / control / entropy-only / style-only)
- [ ] S7  models/lora.py (optional)
- [ ] S8  scripts/evaluate.py (CIs, paired tests, calibration)
- [ ] S9  scripts/run_all.py (stages, markers, status.json, ladder)
- [ ] S10 kaggle/ bundle + scripts/kaggle_drive.py
- [ ] S11 GATE P0: CPU dry run on fixtures OK, pytest green

## Phase P1/P2 — Kaggle
- [ ] S12 Kaggle smoke passed gate P1
- [ ] S13 Kaggle lite passed gate P2 (results/ downloaded)

## Phase P3/P4 — analyze and finalize
- [ ] S14 scripts/analyze.py (stress test, importance, confusion, reliability, PR, error analysis)
- [ ] S15 scripts/bench.py + app/demo.py
- [ ] S16 make_report.py, README, MODEL_CARD, INTERVIEW_NOTES, slides_update.md, KNOWN_GAPS.md, tag v1.0-portfolio

## Decisions / notes log (append; one line each)
