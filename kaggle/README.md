# kaggle/ (to be completed by Claude Code in step S10)
- `kernel-metadata.template.json`: fill `<KAGGLE_USERNAME>` and `<PRESET>`; `scripts/kaggle_drive.py` renders it.
- `kaggle_entry.py` (to write): script kernel entry. Install only missing extras, get the repo (git clone of the public repo, or an attached
  private dataset `codedet-src` created with `kaggle datasets create`), `pip install -e .`, run `python scripts/run_all.py --preset $PRESET --out /kaggle/working/results`.
- Verify current CLI syntax with `kaggle kernels push --help` before relying on flags; GPU accelerator selection options can change.
