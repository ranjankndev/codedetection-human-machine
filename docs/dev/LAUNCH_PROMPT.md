# Prompts to paste into Claude Code

## First launch (once)
```
Read CLAUDE.md, PROGRESS.md and PLAN.md (Section A first). Execute the project autonomously with minimal interaction.
Do not ask me questions: use the decision rules (PLAN §A.5), the degradation ladder (§A.6) and the phase gates (§A.4).
Build locally first (steps S0-S11), then run the Kaggle smoke and lite runs through the kaggle CLI (S12-S13), then analyze and finalize (S14-S16).
Update PROGRESS.md and commit after every step. At most 3 Kaggle relaunches. Never print or commit credentials; never read ~/.kaggle.
Write KNOWN_GAPS.md for anything skipped. When finished, give me a 10-line summary: results table, README branch chosen and why, what was skipped, reproduce commands.
```

## Resume after a usage-limit pause or a closed session
```
/progress
```
(or paste: "Read CLAUDE.md and PROGRESS.md and continue from the first unchecked step. Do not ask questions.")

## Check a gate manually
```
/gate
```

## If Claude Code asks you something anyway
Answer: "Decide by the rules in PLAN.md section A.5; record the decision in PROGRESS.md and continue."
