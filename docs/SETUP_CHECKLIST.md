# One-time setup (about 10-15 minutes)

## 0. Your machine needs (check first)
- Python 3.10+ and git. (Node.js only if you install Claude Code through npm.)
- Windows: use WSL2 or Git Bash; the permission allowlist and all scripts assume a bash shell.
- Internet access (Kaggle, Hugging Face, PyPI).
- Local CPU test environment: `pip install torch transformers pyarrow pytest` (CPU build is fine). `pyproject.toml` deliberately does not install torch, because Kaggle ships its own.
  Claude Code will do this for you in step S0 if you let it; it only needs about 1-2 GB of disk.
- About 5 GB free disk (fixtures, caches, downloaded results). No local GPU needed.
- Optional: an empty GitHub repo for your portfolio. If you skip it, Claude uploads the code to Kaggle as a private dataset instead (no GitHub needed).
- No Hugging Face token, no Anthropic API key, no paid services required. CodeBERT and Qwen2.5-Coder-0.5B are openly downloadable.

## A. Claude Code on a Claude Pro plan
1. Install Claude Code on your laptop (Windows/macOS/Linux) following the official quickstart: https://code.claude.com/docs/en/quickstart
   (an npm install of `@anthropic-ai/claude-code` also works if you have Node.js).
2. Run `claude` in a terminal and sign in with the **same account you use for Claude (Pro)**. If you ever used Console/API billing, run `/login` and choose the subscription.
3. Make sure the environment variable `ANTHROPIC_API_KEY` is **not set**. If it is, Claude Code bills the API instead of your Pro plan. Check usage with `/status`.
4. Pro limits: one five-hour session limit and a weekly limit, **shared** with claude.ai chat and Claude Code. A long autonomous run can hit the five-hour limit.
   This project is built for that: state lives in `PROGRESS.md` and git, so after the reset just run `/progress`.
   Tips: avoid chatting on claude.ai while a run is going; use `/clear` between phases (state is in files); use `/model` to pick a lighter model for bulk coding if the limit is tight.
   Details: https://support.claude.com/en/articles/11145838-use-claude-code-with-your-pro-or-max-plan and https://support.claude.com/en/articles/9797557-usage-limit-best-practices

## B. Kaggle (needed for GPU runs)
1. Create/sign in to a Kaggle account and **verify your phone number** (required to enable GPU and Internet in notebooks).
2. Open the dataset https://www.kaggle.com/datasets/daniilor/semeval-2026-task13 once and accept any terms/competition rules it asks for.
3. Settings → API → create a token. Save it as `~/.kaggle/kaggle.json` (Linux/macOS) or `C:\Users\<you>\.kaggle\kaggle.json` (Windows); on Linux/macOS run `chmod 600 ~/.kaggle/kaggle.json`.
   Use the method shown on Kaggle's current API page if it differs.
4. `pip install kaggle` then test with `kaggle datasets list -s semeval` (should list results, not an auth error).
   **Note from the 2026-10 build run:** a read test passing is NOT enough. Recent Kaggle CLI versions accept the
   legacy `kaggle.json` key for read endpoints but reject *write* endpoints (`kernels push`, `datasets create`) with
   "Authentication required". Test the write path too:
   ```bash
   kaggle kernels list --mine          # read
   kaggle auth login                   # OAuth in the browser - needed before any push
   ```
   If `kaggle auth login` is not available in your version, generate a token at https://www.kaggle.com/settings/api
   and save it to `~/.kaggle/access_token`. Until this is done, `scripts/kaggle_drive.py` exits 75 and the run
   falls back to CPU / Colab (`colab/run_colab.ipynb`) or the manual notebook (`notebooks/kaggle_minimal.ipynb`).
5. Put your Kaggle username in the repo's config when Claude asks nothing: it reads it with `kaggle config view` (username only).

## C. GitHub (optional, for the final push)
`gh auth login` (or leave unauthenticated; Claude will commit locally and tell you the push command).

## D. Start
1. Unzip the starter pack, `cd codedet-starter`, `git init` (if not already a repo).
2. `python -m venv .venv`, activate it (Claude will do the rest), or let Claude create it.
3. Run `claude` in this folder, accept the project's `.claude/settings.json` permissions, paste the first-launch prompt from `docs/LAUNCH_PROMPT.md`.
4. Keep the computer awake. Expect 2-4 hours, mostly waiting on Kaggle. Come back, read the 10-line summary.

## E. What may still need you
Kaggle GPU quota exhausted or account restricted; GitHub not authenticated; final read-through of README/model card before publishing.
