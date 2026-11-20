# ==================================================================
#   Stock Screener  —  config.py
#   Standalone rebuild for Fidelity accounts (no outer NADA_Trading
#   pipeline required). Single source of truth for file paths and
#   settings.
#
#   Usage:
#       from config import PATHS, SETTINGS
#
#   Setup:
#       ROOT is auto-detected from this file's own location. Drop this
#       whole "screener" folder anywhere and it will create its own
#       cache/outputs/logs subfolders next to itself.
#
#   Fidelity CSV:
#       Log into Fidelity -> Positions -> "Download" -> CSV. Save (or
#       overwrite) that file at PATHS["portfolio_csv"] below, or point
#       "portfolio_csv" at wherever you keep the export. The parser in
#       modules/ticker_universe.py reads the standard Fidelity Positions
#       export column layout (a "Symbol" column). If your export's
#       columns don't match, tell me and I'll adjust the parser.
# ==================================================================

import sys
from pathlib import Path

# ------------------------------------------------------------------
# Root folders
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

DIR_MODULES        = ROOT / "modules"
DIR_CACHE          = ROOT / "cache"
DIR_CACHE_UNIVERSE = DIR_CACHE / "universe"
DIR_CACHE_OHLCV    = DIR_CACHE / "ohlcv"
DIR_CACHE_EARNINGS = DIR_CACHE / "earnings"
DIR_CACHE_NEWS      = DIR_CACHE / "news"
DIR_OUTPUTS        = ROOT / "outputs"
DIR_LOGS           = ROOT / "logs"
DIR_INPUTS         = ROOT / "inputs"
DIR_MODEL          = DIR_CACHE / "model"

# ------------------------------------------------------------------
# File paths
# ------------------------------------------------------------------
PATHS = {
    # Your Fidelity "Positions" CSV export — read-only, never written to.
    # Put your downloaded export here (or change this path to wherever
    # you save it).
    "portfolio_csv": DIR_INPUTS / "Fidelity_Positions.csv",

    "universe_cache": DIR_CACHE_UNIVERSE / "index_constituents.csv",
    "universe_meta":  DIR_CACHE_UNIVERSE / "index_constituents_meta.json",
    "portfolio_names_cache": DIR_CACHE_UNIVERSE / "portfolio_ticker_names.json",
    "all_us_cache": DIR_CACHE_UNIVERSE / "all_us_tickers.csv",
    "all_us_meta":  DIR_CACHE_UNIVERSE / "all_us_tickers_meta.json",
    "ohlcv_meta":     DIR_CACHE_OHLCV / "_meta.json",
    "fetch_state":    DIR_CACHE_OHLCV / "_fetch_state.json",
    "dead_tickers":   DIR_CACHE_OHLCV / "_dead_tickers.json",
    "earnings_cache": DIR_CACHE_EARNINGS / "earnings_dates.csv",
    "news_sentiment_cache": DIR_CACHE_NEWS / "sentiment_scores.csv",

    # Append-only — unlike sentiment_scores.csv above (today's snapshot
    # only, overwritten), this keeps every day's reading forever, so
    # there's eventually a real point-in-time sentiment history to
    # train an ML feature on (not possible today — see
    # train_recommendation_model.py's module docstring).
    "sentiment_history": DIR_CACHE_NEWS / "sentiment_history.csv",

    # Per-pattern historical win-rate / avg-return stats, written by
    # backtest_patterns.py and read back by the live screener so a
    # fresh match (e.g. today's Cup & Handle on AMD) can be shown next
    # to "historically, this pattern wins 62% of the time, +4.3% avg
    # over 20d" instead of just a bare pattern name. Not present until
    # backtest_patterns.py has been run at least once.
    "pattern_stats_cache": DIR_CACHE / "pattern_stats.csv",

    # Same per-pattern stats, split by Bull/Bear market regime at
    # match time instead of blended — see analyze_market_regime.py and
    # modules/backtest.py's summarize_by_regime for why the blended
    # number can hide a real, large effect (e.g. Death Cross is far
    # worse specifically in Bear markets).
    "pattern_stats_by_regime_cache": DIR_CACHE / "pattern_stats_by_regime.csv",

    # Same per-pattern stats, split by each matched ticker's GICS
    # sector instead of blended — see analyze_sector_performance.py
    # and modules/backtest.py's summarize_by_sector. Only Death Cross
    # showed a real, held-out-reproducing sector effect; every other
    # pattern's apparent sector differences didn't survive the test split.
    "pattern_stats_by_sector_cache": DIR_CACHE / "pattern_stats_by_sector.csv",

    # Raw match-level backtest output (one row per historical match,
    # see modules/backtest.py's Quality column) — the training set
    # train_recommendation_model.py reads from.
    "backtest_matches_cache": DIR_CACHE / "backtest_matches.csv",

    # Trained win-probability model + its evaluation metrics, written
    # by train_recommendation_model.py and read by modules/ml_recommend.py.
    "ml_model": DIR_MODEL / "win_probability_model.joblib",
    "ml_model_meta": DIR_MODEL / "win_probability_model_meta.json",
}

# ------------------------------------------------------------------
# System settings
# ------------------------------------------------------------------
SETTINGS = {
    # Liquidity filter for NEW buy candidates only — penny stocks and
    # thinly-traded names are excluded from the Candidates/star-rating
    # sheets (your existing held positions are still fully analyzed in
    # "My Positions" regardless of these thresholds).
    "min_price": 5.0,
    "min_avg_volume": 1_000_000,
    "liquidity_volume_lookback": 20,

    # A ticker that trades flat/dead most days and then spikes once
    # can still clear the average-volume bar above without being
    # genuinely liquid day-to-day — this requires most recent days to
    # show real trading activity, not just an occasional jump.
    "activity_lookback_days": 20,
    "min_active_day_fraction": 0.8,

    # Ticker universe (DJIA + S&P 500 + Nasdaq-100) refresh cadence —
    # these lists change only a handful of times a year, so re-scraping
    # every single daily run is wasteful and adds an avoidable network
    # dependency to every run.
    "universe_refresh_days": 30,
    "all_us_refresh_days": 7,

    # Market data
    # 5 years spans multiple market regimes (up/down/sideways), so a
    # backtest built on it means something instead of just reflecting
    # one 9-month window of calendar time. Bumping this triggers an
    # automatic one-time re-backfill for every already-cached ticker
    # (see modules/market_data.py's "backfill_period" tracking) — the
    # next full run will re-pull deeper history for everything.
    "history_period_full": "5y",          # first-ever (or upgraded) backfill per ticker
    "history_period_incremental": "10d",  # daily top-up per ticker
    "fetch_chunk_size_full": 50,
    "fetch_chunk_size_incremental": 100,
    "fetch_max_retries": 3,
    "fetch_backoff_base_sec": 5,
    "fetch_politeness_delay_sec": 0.3,
    # Benchmarked 2026-11-09 at 16/24/32 workers against the live
    # incremental-refresh path (200-ticker samples): 24 and 32 both
    # showed real speedups over 16 with zero rate-limiting or
    # failures at any level tested. 24 picked as a safe step up
    # rather than jumping straight to the untested full-universe
    # behavior of 32.
    "fetch_max_workers": 24,

    # A ticker that fails to fetch this many runs IN A ROW, with no
    # successful fetch EVER (never produced a usable cache file), is
    # treated as permanently dead (delisted/invalid/OTC junk the SEC
    # list includes but Yahoo doesn't carry) rather than "retry
    # forever" — see modules/market_data.py's prune_dead_tickers.
    "dead_ticker_failure_threshold": 5,

    # Backtest parallelism (modules/backtest.py's run_backtest). Each
    # ticker's day-by-day walk is independent and CPU-bound, so this
    # should in theory scale with os.cpu_count() — but in testing,
    # spawning too many worker processes at once (e.g. 24 on a 24-core
    # box) crashed the pool outright in this sandboxed environment.
    # 4 was confirmed stable; raise it if you've verified more workers
    # survive in your actual environment.
    "backtest_max_workers": 4,

    # Earnings
    "earnings_window_days": 14,
    "earnings_refresh_days": 3,

    # News sentiment — only fetched for your held positions + the
    # tickers that already surfaced as 2+ star candidates (a small
    # set), not the whole scanned universe. yfinance's news feed isn't
    # meant for scanning thousands of tickers a day, and headlines move
    # fast enough within a trading day that re-scoring once every few
    # hours (rather than once a day, like earnings dates) is worth it.
    "news_max_headlines_per_ticker": 10,
    "news_refresh_hours": 6,
    "news_bullish_threshold": 0.2,   # VADER compound score cutoffs
    "news_bearish_threshold": -0.2,

    # Pattern detection
    "swing_order": 5,
    "zigzag_lookback_bars": 252,      # ~1y of bars considered per ticker
    "zigzag_legs_per_ticker": 5,      # most recent N legs kept, bounds sheet size

    # Local LLM note-condition checking (modules/note_analysis.py) —
    # runs entirely on-machine via Ollama, no cloud API/key. If Ollama
    # isn't installed or isn't running, this silently degrades to no
    # AI check (your notes still display and carry forward either
    # way) — see modules.note_analysis.ollama_available.
    "ollama_url": "http://localhost:11434",
    "ollama_model": "llama3.2:3b",

    # Forming Signals can run into the hundreds on an all-us scan —
    # capping how many get an AI paraphrase keeps a single weekly run
    # from ballooning by one local-LLM call per signal. Rows beyond
    # this (already sorted by Confidence descending) just get no
    # outlook text, same graceful-degradation pattern as everywhere
    # else this local model is used.
    "forming_signals_ai_outlook_top_n": 15,

    # Logging
    "log_output": "both",

    # A lock file older than this is considered abandoned by a crashed
    # run and is safe to clear rather than block the next run forever.
    "lock_max_age_sec": 60 * 60,
}

# ------------------------------------------------------------------
# Auto-create folders if they don't exist yet
# ------------------------------------------------------------------
for _folder in [DIR_CACHE_UNIVERSE, DIR_CACHE_OHLCV, DIR_CACHE_EARNINGS, DIR_CACHE_NEWS,
                DIR_OUTPUTS, DIR_LOGS, DIR_INPUTS, DIR_MODEL]:
    Path(_folder).mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Quick self-check (run this file directly to verify setup)
# ------------------------------------------------------------------
if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("\n" + "=" * 55)
    print("  Stock Screener — Config Self-Check")
    print("=" * 55)

    print(f"\n  Screener root : {ROOT}")

    print("\n  Folder status:")
    for name, folder in [
        ("modules",         DIR_MODULES),
        ("inputs",          DIR_INPUTS),
        ("cache/universe",  DIR_CACHE_UNIVERSE),
        ("cache/ohlcv",     DIR_CACHE_OHLCV),
        ("cache/earnings",  DIR_CACHE_EARNINGS),
        ("cache/news",      DIR_CACHE_NEWS),
        ("outputs",         DIR_OUTPUTS),
        ("logs",            DIR_LOGS),
    ]:
        status = "OK" if folder.exists() else "MISSING"
        print(f"    {name:<16} {status}  ->  {folder}")

    print("\n  Broker portfolio CSV (Fidelity or Schwab):")
    try:
        from modules.ticker_universe import resolve_portfolio_csv_path
        csv_path = resolve_portfolio_csv_path()
    except Exception:
        csv_path = Path(PATHS["portfolio_csv"])
    print(f"    {'FOUND' if csv_path.exists() else 'NOT FOUND'}  ->  {csv_path}")
    if not csv_path.exists():
        print("    (Export your Positions as CSV from Fidelity or Schwab and drop")
        print(f"     it in {DIR_INPUTS} under any filename — it's auto-detected.)")

    print("\n  System health:")
    try:
        import datetime as _dt
        from modules.backtest import load_pattern_stats_cache
        from modules.ml_recommend import model_metadata

        stats_df = load_pattern_stats_cache()
        if stats_df.empty:
            print("    backtest stats       NOT RUN YET  ->  run backtest_patterns.py")
        else:
            n_matches = int(stats_df["Sample Size"].sum())
            age_days = (_dt.datetime.now() - _dt.datetime.fromtimestamp(
                PATHS["pattern_stats_cache"].stat().st_mtime)).days
            print(f"    backtest stats       {n_matches} matches across {len(stats_df)} patterns, "
                  f"{age_days}d old  ->  {PATHS['pattern_stats_cache']}")

        meta = model_metadata()
        if meta is None:
            print("    ML model             NOT TRAINED YET  ->  run train_recommendation_model.py")
        else:
            status = "ACTIVE" if meta.get("is_useful") else "trained, not used (AUC too close to 0.5)"
            print(f"    ML model             {status} — {meta.get('horizon_days')}d horizon, "
                  f"AUC={meta.get('test_auc')}, trained {meta.get('trained_at', '')[:10]}")

        if PATHS["all_us_cache"].exists():
            import pandas as _pd
            all_us_total = len(_pd.read_csv(PATHS["all_us_cache"]))
            cached_tickers = len(list(DIR_CACHE_OHLCV.glob("*.csv")))
            coverage = cached_tickers / all_us_total * 100 if all_us_total else 0
            print(f"    all-us coverage      {cached_tickers}/{all_us_total} tickers have cached "
                  f"price history ({coverage:.1f}%)")
        else:
            print("    all-us coverage      NOT BUILT YET  ->  run with --universe all-us once")

        if PATHS["dead_tickers"].exists():
            import json as _json
            with open(PATHS["dead_tickers"], "r", encoding="utf-8") as _f:
                dead_count = len(_json.load(_f).get("dead", []))
            print(f"    dead tickers         {dead_count} permanently excluded (5+ straight failed fetches)")
    except Exception as e:
        print(f"    (system health check failed: {e})")

    print("\n  Settings:")
    for k, v in SETTINGS.items():
        print(f"    {k:<28} {v}")

    print()
