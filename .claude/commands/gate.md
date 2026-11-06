---
description: Check the current phase gate (docs/dev/PLAN.md section A.4) and decide go / fix / degrade
---
Determine the current phase from docs/dev/PROGRESS.md. Run `pytest -q` and, if results exist, read `results/status.json`.
Evaluate the gate criteria in docs/dev/PLAN.md §A.4 for that phase. Output PASS or FAIL with the exact failing criteria, then apply the fix or the next rung of the degradation ladder (§A.6) and continue. Record the outcome in docs/dev/PROGRESS.md.
