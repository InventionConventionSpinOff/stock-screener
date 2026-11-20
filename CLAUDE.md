# Stock Screener - notes for Claude

This folder is a personal stock screener (Python, Windows). It scans tickers for chart
patterns, cross-references the user's Schwab / thinkorswim / Fidelity positions CSV, and
writes a multi-sheet Excel report to `outputs/`. It is pattern detection plus backtested
stats, not financial advice.

## If the user asks to "update" / "get the latest version"

Run this in PowerShell (it works whether or not the screener is already installed, and it
prints the folder path when done):

```powershell
irm https://raw.githubusercontent.com/InventionConventionSpinOff/stock-screener/master/get-screener.ps1 | iex
```

- Updates in place: `git pull --ff-only` if this folder is a git checkout, otherwise a zip
  download copied over the code.
- Never touches the user's own data: `inputs/`, `outputs/`, `cache/`, `logs/`.
- To confirm where it is installed, read the "Folder:" line it prints (default
  `%USERPROFILE%\stock-screener`). Tell the user that path.
- If `git pull` refuses because tracked files were edited locally, do NOT discard their
  edits. Show what differs (`git status`, `git diff`) and ask what they want to keep.

## Running it

- `RUN_SCREENER.bat` - daily run, ~518 tickers (DJIA + S&P 500 + Nasdaq-100), about 1 minute.
  Auto-installs Python if missing.
- `RUN_SCREENER_WEEKLY_ALLUS.bat` - every US-listed ticker (~10,000+), 30-45+ minutes,
  heavily rate-limited by Yahoo Finance. Run about weekly, not daily.
- `RUN_SCREENER_CUSTOM.bat` - scans your holdings plus extra tickers you give it.
- Put the Positions CSV anywhere in `inputs/` (any filename). The reader auto-detects it.
- Start with `python config.py` - its "System health" section shows ML model status,
  backtest cache age, and all-US ticker coverage.

## Layout

- `run_screener.py` - main entry point (`--universe full` or `--universe all-us`).
- `modules/` - pattern detectors (`pattern_*.py`), `recommendation.py`, `positions.py`,
  `options.py`, `ticker_universe.py`, `market_data.py`, `report_screener.py`,
  `backtest.py`, `tracking.py`, `notes.py`, etc.
- `backtest_patterns.py`, `analyze_*.py`, `tune_pattern_thresholds.py`,
  `train_recommendation_model.py` - research/analysis scripts.
- `tests/` - pytest suite.

## Making changes

- Run `python -m pytest tests/` after any change to pattern detectors, recommendation or
  positions logic, or backtest helpers. All tests should pass before the user relies on it.
- Position/option parsing: Fidelity's option format is verified. Schwab and thinkorswim
  parsing is best-effort. If a holding or option doesn't show up right, get ONE real example
  row from the user's export (they can blank out dollar amounts) and fix the column names /
  regex in `modules/options.py` and `modules/ticker_universe.py`. Don't guess.
- Today's still-forming bar can have `Close = NaN` mid-session. `get_history()` in
  `modules/market_data.py` drops those rows - keep that behavior.
- Findings so far: Bathtub and Cup & Handle are the only patterns with a real backtested
  edge (~53% win rate at 20 days); Death Cross is consistently bad. The ML win-probability
  model does not beat the simple backtest stats (AUC ~0.50) and stays gated off unless
  AUC >= 0.55. Don't oversell any signal.

## Privacy

- Treat the user's positions CSV, `outputs/` reports, and `logs/` as private. Don't paste
  their contents (tickers held, quantities, dollar amounts) into external services,
  issues, or commits.
- `.gitignore` already excludes `inputs/*.csv`, `outputs/`, `logs/`, and `cache/`. Check
  `git status` before any commit or push, and never commit those.
