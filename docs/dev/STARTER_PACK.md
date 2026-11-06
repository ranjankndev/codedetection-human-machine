# codedet starter pack: start here

This folder is a ready-to-run repo skeleton for the SemEval-2026 Task 13 portfolio revamp, built so Claude Code can finish it with minimal help from you.

**Do this:**
1. Read `docs/SETUP_CHECKLIST.md` (10-15 min, one time: Claude Code on Pro, Kaggle token).
2. `cd` into this folder, run `claude`, paste the prompt from `docs/LAUNCH_PROMPT.md`.
3. Come back later. If the Pro usage limit interrupts, type `/progress` after the reset.

**What is inside**
| Path | Purpose |
|---|---|
| `CLAUDE.md` | Project memory Claude Code loads every session: process, coding rules, traps, definition of done |
| `PLAN.md` | Full plan; Section A = autonomous execution rules, phases, gates, self-healing |
| `PROGRESS.md` | Checklist Claude updates; lets a new session resume exactly where the last stopped |
| `docs/design_doc.html` | Design document (open in a browser) |
| `docs/OLD_CODE_AUDIT.md` | What was reused from your earlier code and why |
| `docs/ORIGINAL_RESULTS.md` | Numbers from your original deck (labelled, not re-validated) |
| `docs/LAUNCH_PROMPT.md`, `docs/SETUP_CHECKLIST.md` | Prompts and one-time setup |
| `src/codedet/` | Ported, cleaned and tested building blocks (17 passing CPU tests) |
| `configs/` | `smoke`, `lite`, `local_cpu` presets |
| `.claude/` | Permission allowlist (no destructive commands, `~/.kaggle` unreadable) and `/progress`, `/gate` commands |
| `docs/original_deck_Presentation.pdf` | Your original 8-slide deck, for Claude to base the updated slide text on |
| `legacy/` | Your original scripts, untouched, for reference only |
| `kaggle/` | Kernel template; Claude completes the entry script and driver |

Run the tests yourself any time: `pip install -e ".[dev]" && pytest -q` (needs torch + transformers installed).
