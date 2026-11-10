# ==================================================================
#   Stock Screener  —  backtest_patterns.py
#   Answers the question "do these pattern detectors actually predict
#   anything, or are they just noise?" by replaying each ticker's
#   cached price history day by day and checking what the price
#   actually did after every historical match.
#
#   Usage:
#       python backtest_patterns.py                      <- backtest your fast/liquid universe
#       python backtest_patterns.py --universe portfolio  <- just your 4 holdings (small sample)
#       python backtest_patterns.py --universe all-us     <- every cached ticker (slow, biggest sample)
#       python backtest_patterns.py --limit 200            <- cap ticker count
#
#   Needs price history already cached — run run_screener.py at least
#   once first (any --universe mode) so there's data to replay.
#
#   Depth note: backtests against whatever's cached (5 years as of
#   writing — config.SETTINGS["history_period_full"]), which spans
#   multiple regimes but is still one period of history, not true
#   out-of-sample validation. Good enough to catch an actively broken
#   pattern; not proof a pattern works in all future conditions.
# ==================================================================
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from config import SETTINGS
from modules.backtest import (
    run_backtest, summarize, summarize_by_regime, save_backtest_report,
    save_pattern_stats_cache, save_pattern_stats_by_regime_cache, save_raw_matches_cache,
)


def main():
    args = [a.lower() for a in sys.argv[1:]]
    universe_mode = "fast"
    if "--universe" in args:
        idx = args.index("--universe")
        if idx + 1 < len(sys.argv[1:]):
            universe_mode = sys.argv[1:][idx + 1].lower()
    limit = None
    if "--limit" in args:
        idx = args.index("--limit")
        if idx + 1 < len(sys.argv[1:]):
            limit = int(sys.argv[1:][idx + 1])

    from modules.ticker_universe import (
        build_cached_liquid_universe, build_all_us_universe,
        build_full_universe, get_traded_tickers,
    )

    print(f"Building '{universe_mode}' ticker universe for backtesting...")
    if universe_mode == "portfolio":
        tickers = get_traded_tickers()
    elif universe_mode == "all-us":
        tickers = build_all_us_universe()["Ticker"].tolist()
    elif universe_mode == "full":
        tickers = build_full_universe()["Ticker"].tolist()
    else:
        tickers = build_cached_liquid_universe()["Ticker"].tolist()

    if limit:
        tickers = tickers[:limit]

    print(f"Backtesting {len(tickers)} tickers (this replays each day of their cached history)...")
    matches_df = run_backtest(tickers, log=print, max_workers=SETTINGS["backtest_max_workers"])
    print(f"\nTotal historical pattern matches found: {len(matches_df)}")

    summary_df = summarize(matches_df)
    print("\n" + "=" * 70)
    print("  PATTERN PERFORMANCE SUMMARY")
    print("  (WinRate = % of matches where price moved the predicted")
    print("   direction; AvgReturn = average signed return, positive =")
    print("   the pattern's predicted direction was profitable)")
    print("=" * 70)
    if summary_df.empty:
        print("No matches found — try a larger --universe or check that price history is cached.")
    else:
        print(summary_df.to_string(index=False))

    out_path = save_backtest_report(matches_df, summary_df)
    print(f"\nFull report saved -> {out_path}")

    save_pattern_stats_cache(summary_df, log=print)
    save_raw_matches_cache(matches_df, log=print)

    regime_summary_df = summarize_by_regime(matches_df)
    save_pattern_stats_by_regime_cache(regime_summary_df, log=print)


if __name__ == "__main__":
    main()
