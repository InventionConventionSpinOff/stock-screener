# ==================================================================
#   modules/tracking.py
#   Every other "does this work" signal in this project is backtested
#   against history (modules/backtest.py). This is the one thing that
#   checks the LIVE recommendation engine against reality going
#   forward: log today's BUY/WATCH calls, then come back in N trading
#   days and see what the ticker actually did.
#
#   This is slower than a backtest (it only accumulates one real data
#   point per call per day) but it's the only honest answer to "is
#   this system's advice actually any good," since it can't be
#   gamed by curve-fitting against the same history it was built on.
# ==================================================================
import os
import sys
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import DIR_CACHE

TRACKING_LOG_PATH = DIR_CACHE / "track_record.csv"
LOG_COLUMNS = ["Ticker", "LoggedDate", "Recommendation", "Priority", "EntryPrice",
               "EvalDate", "EvalPrice", "ActualReturnPct", "Correct"]

# How many trading days later to check whether a BUY actually paid off —
# matches the "20d" primary horizon already used everywhere else in
# the report (HistoricalEdge, the ML model's default horizon) so this
# stays comparable to those numbers instead of introducing a new one.
EVAL_HORIZON_DAYS = 20


def _load_log() -> pd.DataFrame:
    if not os.path.exists(TRACKING_LOG_PATH):
        return pd.DataFrame(columns=LOG_COLUMNS)
    try:
        df = pd.read_csv(TRACKING_LOG_PATH, parse_dates=["LoggedDate", "EvalDate"])
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame(columns=LOG_COLUMNS)
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def _save_log(df: pd.DataFrame) -> None:
    df.to_csv(TRACKING_LOG_PATH, index=False)


def log_todays_calls(candidates_df: pd.DataFrame, log=print) -> None:
    """Appends today's BUY/WATCH candidates to the permanent tracking
    log, one row per ticker per day — skips a ticker already logged
    today so re-running the screener twice in one day doesn't double
    count it."""
    if candidates_df is None or candidates_df.empty:
        return
    if "Recommendation" not in candidates_df.columns:
        return

    today = pd.Timestamp(dt.date.today())
    existing = _load_log()
    already_logged_today = set(
        existing[existing["LoggedDate"] == today]["Ticker"]
    ) if not existing.empty else set()

    from modules.market_data import get_history
    new_rows = []
    for _, row in candidates_df.iterrows():
        if row["Recommendation"] not in ("BUY", "WATCH"):
            continue
        ticker = row["Ticker"]
        if ticker in already_logged_today:
            continue
        hist = get_history(ticker)
        if hist.empty:
            continue
        entry_price = hist["Close"].iloc[-1]
        new_rows.append({
            "Ticker": ticker,
            "LoggedDate": today,
            "Recommendation": row["Recommendation"],
            "Priority": row.get("Priority"),
            "EntryPrice": round(entry_price, 2),
            "EvalDate": pd.NaT,
            "EvalPrice": None,
            "ActualReturnPct": None,
            "Correct": None,
        })

    if not new_rows:
        return
    merged = pd.DataFrame(new_rows) if existing.empty else pd.concat(
        [existing, pd.DataFrame(new_rows)], ignore_index=True)
    _save_log(merged)
    log(f"  Track record: logged {len(new_rows)} new call(s) for today")


def resolve_due_calls(log=print) -> pd.DataFrame:
    """For any logged call that's at least EVAL_HORIZON_DAYS of actual
    trading-day history old and hasn't been resolved yet, fills in
    EvalPrice/ActualReturnPct/Correct using whatever price history is
    cached now. Returns the full (updated) log."""
    df = _load_log()
    if df.empty:
        return df

    from modules.market_data import get_history
    unresolved = df[df["EvalPrice"].isna()]
    if unresolved.empty:
        return df

    resolved_count = 0
    for idx, row in unresolved.iterrows():
        ticker = row["Ticker"]
        hist = get_history(ticker)
        if hist.empty:
            continue
        logged_date = pd.Timestamp(row["LoggedDate"])
        bars_since = hist[hist["Date"] > logged_date]
        if len(bars_since) < EVAL_HORIZON_DAYS:
            continue  # not enough trading days have passed yet

        eval_row = bars_since.iloc[EVAL_HORIZON_DAYS - 1]
        eval_price = eval_row["Close"]
        entry_price = row["EntryPrice"]
        if pd.isna(entry_price) or entry_price == 0:
            continue
        actual_return = (eval_price - entry_price) / entry_price * 100

        df.loc[idx, "EvalDate"] = eval_row["Date"]
        df.loc[idx, "EvalPrice"] = round(eval_price, 2)
        df.loc[idx, "ActualReturnPct"] = round(actual_return, 2)
        # A BUY call is "correct" if the stock actually went up; WATCH
        # calls don't carry a directional promise, so they're not
        # scored right/wrong, just logged for reference.
        df.loc[idx, "Correct"] = bool(actual_return > 0) if row["Recommendation"] == "BUY" else None
        resolved_count += 1

    if resolved_count:
        _save_log(df)
        log(f"  Track record: resolved {resolved_count} call(s) that reached their {EVAL_HORIZON_DAYS}d check-in")
    return df


def track_record_summary(df: pd.DataFrame = None) -> pd.DataFrame:
    """One-row-per-status summary: how many BUY calls have actually
    resolved, what their real (not backtested) win rate and average
    return have been so far."""
    if df is None:
        df = _load_log()
    if df.empty:
        return pd.DataFrame()

    buys = df[df["Recommendation"] == "BUY"]
    resolved = buys[buys["Correct"].notna()]
    pending = buys[buys["Correct"].isna()]

    rows = [{
        "Metric": "Total BUY calls logged", "Value": len(buys),
    }, {
        "Metric": "Resolved (20+ trading days old)", "Value": len(resolved),
    }, {
        "Metric": "Still pending", "Value": len(pending),
    }]
    if not resolved.empty:
        rows.append({"Metric": "Live win rate (resolved BUYs)",
                      "Value": f"{resolved['Correct'].mean() * 100:.1f}%"})
        rows.append({"Metric": "Live avg return (resolved BUYs)",
                      "Value": f"{resolved['ActualReturnPct'].mean():+.2f}%"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = resolve_due_calls()
    print(f"{len(df)} total logged calls")
    print(track_record_summary(df).to_string(index=False))
