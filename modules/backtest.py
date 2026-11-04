# ==================================================================
#   modules/backtest.py
#   Walks each ticker's cached price history day by day, re-running
#   every pattern detector on only the data that would have been known
#   "as of" that day, and checks what the price actually did over the
#   following N days whenever a fresh match fires. This is the only
#   way to know whether a pattern/threshold has real predictive value
#   on this data, instead of trusting the textbook definition.
#
#   Depth note: backtests against whatever's cached per ticker, which
#   is config.SETTINGS["history_period_full"] (5 years as of writing) —
#   enough to span multiple market regimes, but still only one period
#   of history, not an out-of-sample, multi-cycle validation. Good
#   enough to catch a pattern that's actively wrong (see Death Cross
#   results); not a substitute for real out-of-sample testing if you
#   want to trust this with real money.
# ==================================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import DIR_OUTPUTS
from modules.market_data import get_history
from modules.pattern_bathtub import detect_bathtub
from modules.pattern_cup_handle import detect_cup_handle
from modules.pattern_flag import detect_flag
from modules.pattern_double import detect_double
from modules.pattern_golden_cross import detect_golden_cross
from modules.pattern_breakout import detect_breakout
from modules.pattern_overextension import detect_overextension
from modules.priority import _match_quality

DEFAULT_DETECTORS = [
    detect_cup_handle, detect_bathtub, detect_flag,
    detect_double, detect_golden_cross, detect_breakout,
    # Live in run_screener.py's own detector list (see step_detect_patterns)
    # but was missing here — without it, neither the pattern-stats cache
    # nor the ML training set below ever see an Overextended match.
    detect_overextension,
]
DEFAULT_FORWARD_DAYS = [5, 10, 20]

# After a match fires, skip this many bars before allowing another
# match of the SAME pattern on the SAME ticker — a pattern's geometric
# conditions often stay true for several consecutive days in a row,
# and counting each of those as a separate "prediction" would wildly
# overstate the sample size and double-count what is really one signal.
MIN_GAP_BARS = 10

# Minimum bars of history needed before we start testing a ticker at
# all — the slowest detector (Golden Cross) needs 200 (slow MA) + 10
# (lookback) = 210; give it a little headroom.
MIN_HISTORY_BARS = 220


def backtest_ticker(ticker: str, detectors: list = None, forward_days: list = None) -> list:
    """Returns a list of match-outcome dicts for one ticker: each dict
    is one historical pattern match plus its actual forward return."""
    detectors = detectors or DEFAULT_DETECTORS
    forward_days = forward_days or DEFAULT_FORWARD_DAYS
    max_forward = max(forward_days)

    df = get_history(ticker)
    if len(df) < MIN_HISTORY_BARS + max_forward:
        return []

    results = []
    last_match_bar = {}  # pattern name -> bar index of its last counted match

    # Every detector only ever looks backward a bounded amount (Golden/
    # Death Cross is the hungriest, at 210 bars) — passing the full
    # growing history each iteration instead of this fixed slice made
    # the loop accidentally O(n^2) per ticker (each day re-copies and
    # re-scans everything seen so far), which is why a 5-year/1250-bar
    # history took tens of minutes per ticker instead of seconds. 260
    # bars gives every detector a 50-bar margin over what it needs.
    window_size = 260

    for t in range(MIN_HISTORY_BARS, len(df) - max_forward):
        window = df.iloc[max(0, t + 1 - window_size):t + 1].reset_index(drop=True)
        today = df["Date"].iloc[t]

        for detect in detectors:
            try:
                m = detect(window)
            except Exception:
                continue
            if not m:
                continue
            # Only count a match that's fresh "as of today" — a detector
            # re-run on a slightly earlier window may report the same
            # historical match with an end_date in the past, which
            # would double-count it once per day it stays visible.
            if pd.Timestamp(m.end_date) != pd.Timestamp(today):
                continue

            last_bar = last_match_bar.get(m.pattern, -MIN_GAP_BARS - 1)
            if t - last_bar < MIN_GAP_BARS:
                continue
            last_match_bar[m.pattern] = t

            entry_price = df["Close"].iloc[t]
            sign = 1 if m.direction == "bullish" else -1
            row = {
                "Ticker": ticker,
                "Pattern": m.pattern,
                "Direction": m.direction,
                "Date": today,
                "Entry": round(entry_price, 2),
                "VolumeConfirms": m.volume_confirms,
                # Same 0-1 "how clean is this specific match" score
                # modules/priority.py uses for star ratings — folded in
                # here too so a future ML model can learn on the same
                # per-match signal instead of just the bare pattern name
                # (a "textbook" Cup & Handle and a "barely qualifies" one
                # currently get treated identically by this flattened row).
                "Quality": round(_match_quality(m), 3),
            }
            for fd in forward_days:
                exit_price = df["Close"].iloc[t + fd]
                raw_return_pct = (exit_price - entry_price) / entry_price * 100
                # Signed so "positive" always means "the trade would
                # have made money in the direction the pattern predicted" —
                # a bearish call followed by a price drop is a win, not a loss.
                row[f"SignedReturn_{fd}d_pct"] = round(raw_return_pct * sign, 2)
            results.append(row)

    return results


def run_backtest(tickers: list, detectors: list = None, forward_days: list = None, log=print,
                  max_workers: int = None) -> pd.DataFrame:
    """Backtests every ticker in the list, returns one row per
    historical match with its forward returns.

    Parallelized across processes — each ticker's day-by-day walk is
    completely independent and CPU-bound (no shared state, no I/O to
    serialize around), so this is close to a free max_workers-times
    speedup on a multi-core machine rather than something that needs
    careful synchronization."""
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    forward_days = forward_days or DEFAULT_FORWARD_DAYS
    max_workers = max_workers or min(32, os.cpu_count() or 4)

    all_rows = []
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(backtest_ticker, t, detectors, forward_days): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                rows = future.result()
            except Exception as e:
                log(f"{ticker}: backtest failed ({e})")
                rows = []
            all_rows.extend(rows)
            completed += 1
            if completed % 200 == 0 or completed == len(tickers):
                log(f"Backtested {completed}/{len(tickers)} tickers, {len(all_rows)} matches found so far")

    return pd.DataFrame(all_rows)


def summarize(matches_df: pd.DataFrame, forward_days: list = None) -> pd.DataFrame:
    """Rolls raw matches into a per-pattern summary: sample size, win
    rate, and average signed return for each forward-day horizon."""
    forward_days = forward_days or DEFAULT_FORWARD_DAYS
    if matches_df.empty:
        return pd.DataFrame()

    rows = []
    for pattern, group in matches_df.groupby("Pattern"):
        row = {"Pattern": pattern, "Sample Size": len(group)}
        for fd in forward_days:
            col = f"SignedReturn_{fd}d_pct"
            if col not in group.columns:
                continue
            win_rate = (group[col] > 0).mean() * 100
            avg_return = group[col].mean()
            row[f"WinRate_{fd}d_pct"] = round(win_rate, 1)
            row[f"AvgReturn_{fd}d_pct"] = round(avg_return, 2)
        rows.append(row)

    out = pd.DataFrame(rows)
    sort_col = f"WinRate_{forward_days[-1]}d_pct"
    if sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return out


def save_pattern_stats_cache(summary_df: pd.DataFrame, log=print) -> None:
    """Writes the per-pattern summary to a fixed-name cache file (not
    dated, unlike the Excel report) so the live screener can look up
    "historically, how has this pattern actually performed" on every
    run without re-running the backtest itself."""
    from config import PATHS

    cache_path = PATHS["pattern_stats_cache"]
    if summary_df.empty:
        log("No backtest matches to cache — pattern_stats_cache left untouched.")
        return
    summary_df.to_csv(cache_path, index=False)
    log(f"Pattern stats cache updated -> {cache_path}")


def save_raw_matches_cache(matches_df: pd.DataFrame, log=print) -> None:
    """Writes the full match-level backtest output (one row per
    historical match, including the Quality feature) to a fixed-name
    cache — this is the training set for train_recommendation_model.py,
    as opposed to save_pattern_stats_cache's per-pattern rollup."""
    from config import PATHS

    cache_path = PATHS["backtest_matches_cache"]
    if matches_df.empty:
        log("No backtest matches to cache — backtest_matches_cache left untouched.")
        return
    matches_df.to_csv(cache_path, index=False)
    log(f"Raw backtest matches cache updated -> {cache_path} ({len(matches_df)} rows)")


def load_pattern_stats_cache() -> pd.DataFrame:
    """Returns the cached per-pattern backtest summary (Pattern, Sample
    Size, WinRate_Nd_pct, AvgReturn_Nd_pct columns), or an empty
    DataFrame if backtest_patterns.py hasn't been run yet."""
    from config import PATHS

    cache_path = PATHS["pattern_stats_cache"]
    if not os.path.exists(cache_path):
        return pd.DataFrame()
    try:
        return pd.read_csv(cache_path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()


# The forward-day horizon used for the single "is this pattern
# historically weak" gate shared by modules/recommendation.py (new
# candidates) and modules/positions.py (held positions) — 20 trading
# days is roughly a month, long enough for a pattern's thesis to play
# out without the read going stale.
PRIMARY_HORIZON_DAYS = 20

# A pattern counts as "historically weak" only below this win rate AND
# with a negative average return — a sub-50% win rate alone isn't
# damning if the wins are much bigger than the losses. Requiring a
# minimum sample size keeps this from firing off a thin, noisy slice
# of history (e.g. Flag's ~44-match sample) the way it legitimately
# should for Death Cross (~38% win rate across 300+ matches).
WEAK_BACKTEST_WIN_RATE_FLOOR = 45.0
WEAK_BACKTEST_MIN_SAMPLE = 30


def pattern_backtest_is_weak(pattern: str, stats_df: pd.DataFrame) -> bool:
    """True if this pattern has a measured, trustworthy track record of
    losing more often than winning. The one factor shared by both the
    Recommendation verdict (new candidates) and the Action verdict
    (held positions) that's backed by actual outcomes rather than a
    same-day heuristic."""
    if stats_df.empty or "Pattern" not in stats_df.columns:
        return False
    row = stats_df[stats_df["Pattern"] == pattern]
    win_col = f"WinRate_{PRIMARY_HORIZON_DAYS}d_pct"
    ret_col = f"AvgReturn_{PRIMARY_HORIZON_DAYS}d_pct"
    if row.empty or win_col not in row.columns or pd.isna(row.iloc[0][win_col]):
        return False
    r = row.iloc[0]
    if r["Sample Size"] < WEAK_BACKTEST_MIN_SAMPLE:
        return False
    return r[win_col] < WEAK_BACKTEST_WIN_RATE_FLOOR and r[ret_col] < 0


def ticker_has_weak_backtest_pattern(matches: list, stats_df: pd.DataFrame) -> bool:
    """Convenience wrapper — True if ANY of this ticker's matched
    patterns individually qualifies as historically weak."""
    return any(pattern_backtest_is_weak(m.pattern, stats_df) for m in matches)


def save_backtest_report(matches_df: pd.DataFrame, summary_df: pd.DataFrame) -> "os.PathLike":
    import datetime as dt
    today = dt.datetime.now().strftime("%Y-%m-%d")
    out_path = DIR_OUTPUTS / f"BacktestReport_{today}.xlsx"

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        if summary_df.empty:
            pd.DataFrame({"Note": ["No matches found — try a larger ticker universe."]}).to_excel(
                writer, sheet_name="Summary", index=False)
        else:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
        if matches_df.empty:
            pd.DataFrame({"Note": ["No matches found."]}).to_excel(
                writer, sheet_name="All Matches", index=False)
        else:
            matches_df.sort_values(["Pattern", "Date"]).to_excel(
                writer, sheet_name="All Matches", index=False)

    return out_path


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    rows = backtest_ticker(ticker)
    df = pd.DataFrame(rows)
    print(df)
    print()
    print(summarize(df))
