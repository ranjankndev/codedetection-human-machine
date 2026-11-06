# cloud/ - running on a GPU (Kaggle or Colab)

| file | what it is |
|---|---|
| `kaggle.ipynb` | Import into a Kaggle notebook (GPU T4, Internet on, competition data attached) and run all cells. |
| `colab.ipynb` | Same run on Colab; it downloads the data from the Hugging Face mirror, no Kaggle key needed. |
| `kaggle_drive.py` | Pushes the repo and runs it as a Kaggle script kernel from your laptop: `python cloud/kaggle_drive.py --preset lite`. Needs `~/.kaggle/kaggle.json` with write access. |
| `kaggle_entry.py` | The kernel entry point that `kaggle_drive.py` fills in and uploads. Not run by hand. |
| `kernel-metadata.template.json` | Kernel settings template that `kaggle_drive.py` fills in. |

All of them end in the same command: `python scripts/run_all.py --preset <preset> --data <parquet dir> --out results`.
Check current flags with `kaggle kernels push --help` before relying on them; the CLI changes.
