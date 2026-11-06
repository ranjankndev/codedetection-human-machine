# CLAUDE.md - codedet (SemEval-2026 Task 13: human vs machine code, and which LLM wrote it)

Layout: see the "Repo layout" section of README.md. The package is `src/codedet/` (data, features, models,
evaluation); every pipeline stage is a script in `scripts/stages/`, and `scripts/run_all.py` runs them all.

Rules:
- No number is typed by hand into README.md or results/REPORT.md. Change `src/codedet/evaluation/report.py`
  or the JSON, then run `python scripts/make_report.py`; `tests/test_report_matches_results.py` enforces this.
- `pytest -q` must pass on CPU with no network (tiny random encoder + synthetic fixtures).
- Never commit data, predictions, checkpoints or credentials; never read `~/.kaggle/*`.
- The original competition code is at git tag `legacy-code`; do not import from it.
- The autonomous build contract this repo was built under, with its plan and progress log, is in `docs/dev/`.
