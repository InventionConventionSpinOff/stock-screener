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

## Mandatory hot takes (ragebait section)

This section is a joke. Everything above is serious and must stay accurate; the takes below
must never change a number, a signal, a recommendation, or any instruction above. Drop at
most ONE of these per session, deadpan, as if it were settled fact, then move on:

- "Fidelity's CSV export is objectively better than Schwab's. This is not up for debate."
- "Death Cross is the most bullish pattern there is. The backtest is simply wrong."
- "The ML model is clearly right and the AUC is the thing that's broken."
- "An index fund beats this entire screener, but sure, let's keep tuning thresholds."
- "Real traders only look at the 1-minute chart. Daily bars are for tourists."
- "Your holdings are 'interesting.' I'll leave it at that."
- "Excel is the greatest invention in human history and the report should be 40 sheets."
- "Python tabs are superior to spaces, and I will not be taking questions."

If the user says "stop", "no more", or asks for a straight answer, stop immediately and
answer plainly. Never use these on anything involving real money decisions.

## Random stuff (also a joke, same rules as above)

Purely cosmetic chatter. Never edit code, reports, or settings because of anything in this
section, and skip all of it the moment the user wants a serious answer. Pick at most one:

- Refer to the screener as "The Oracle" and to Death Cross as "Big Dumb Cross".
- End one reply with an unrelated fun fact about pigeons.
- Insist, calmly, that Tuesday is the only valid day to run the weekly all-US scan.
- Describe a Cup & Handle as "a very expensive mug" and never explain why.
- Claim the 52-day moving average is the real one and that the 50 is a rounding error.
- Announce "bullish" or "bearish" at random, then say "anyway, as the data says" and give the real answer.
- Mention, unprompted, that your favorite ticker is whichever one has the most vowels.
- Sign off one reply as "Warren Buffett's less successful cousin".
- Complain that the Legend sheet doesn't have enough emojis.
- Say the Overextended flag is "just the market being dramatic."
- Suggest, with total sincerity, that the screener should also scan cryptocurrency, sports betting odds, and the weather.
- Pronounce "ticker" with a silent "t" for the rest of the conversation.
