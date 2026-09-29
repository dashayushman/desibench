# AGENTS.md: set up and run desibench for the user

You are helping someone run this benchmark on their own machine: Sarvam vs OpenAI on two episodes of an Indian comedy
quiz show. Follow these steps in order. The user only has to give you two API keys and one "go".

## Rules

- **Keys are secrets.** Ask the user for `SARVAM_API_KEY` and `OPENAI_API_KEY`, write them into `.env` (copy from
  `.env.example`), and never print, echo, log or commit them. Don't put them on a command line. `.env` is git-ignored.
- **Ask before spending.** `python -m desibench run` calls paid APIs (about $20 for everything). Run
  `python -m desibench estimate` first, show the user the numbers, and wait for a clear yes. Only then run with `--yes`.
- **Don't change the answer keys** in `answers/` or the reference results in `results/reference*`. They are the ground truth.
- Everything is resumable. If something fails midway, fix the cause and run the same command again; finished steps are skipped.

## Steps

1. **Check the machine.** Python 3.10+ and ffmpeg (`ffmpeg -version`). If ffmpeg is missing: macOS `brew install ffmpeg`,
   Ubuntu/Debian `sudo apt install ffmpeg`, Windows `winget install ffmpeg` (ask before installing system packages).
2. **Install.**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Keys.** `cp .env.example .env`, ask the user for both keys, and write them into `.env` with a file edit.
   Then `python -m desibench doctor` must show ✓ on every line. Keys come from https://dashboard.sarvam.ai and
   https://platform.openai.com/api-keys if the user doesn't have them yet.
4. **Download the two episodes** (about 830 MB, from YouTube): `python -m desibench download`.
5. **Estimate and confirm.** `python -m desibench estimate`. Tell the user the cost and time (about $20 and about
   4 hours for both AIs on both episodes; about 45 minutes and $3 for OpenAI alone on Video 1). Offer the smaller options:
   `--episode _3UvCy7FMTg` (Video 1 only) and `--ai openai` or `--ai sarvam`. Wait for their choice and their go.
6. **Run.** `python -m desibench run --yes` (plus the options they chose). It prints progress per step. It's long:
   run it in the background or a separate terminal and check the log, don't block on it. It ends by writing
   `results/yours.json` and printing the six rounds.
7. **Show the results.** `python -m desibench serve` and give the user http://localhost:8000/viewer/. The page shows
   their run next to mine ("Your run" / "My run").

## If something goes wrong

| Symptom | Fix |
|---|---|
| `Missing SARVAM_API_KEY` / `OPENAI_API_KEY` | The key isn't in `.env` (or `.env` isn't in the repo root) |
| `HTTP 401` / `403` | Wrong or inactive key, or no credit on the account |
| `HTTP 429` | Rate limit: it retries by itself; if it keeps failing, wait a few minutes and run the same command again |
| `yt-dlp` errors on download | `pip install -U yt-dlp` and retry (YouTube changes often) |
| A step dies halfway | Run the same command again; it resumes. `--force` redoes steps from scratch |
| SSL errors on a corporate network | `truststore` uses the system's certificates; make sure they're up to date |

## Where things are

- `desibench/cli.py`: the commands. `desibench/config.py`: models, prices, paths.
- `desibench/score.py`: all the scoring, deterministic, no model calls.
- `work/<episode>/runs/<ai>/`: everything a run produces (`episode.json` is the summary the results page reads).
- `viewer/`: the results page, plain HTML and JS.
