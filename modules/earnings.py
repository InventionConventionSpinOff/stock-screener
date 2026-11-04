# ==================================================================
#   modules/earnings.py
#   Caches each ticker's next earnings date via yfinance, refreshing
#   as dates approach (no need to keep re-checking a ticker whose
#   earnings are 3 months out).
# ==================================================================
import os
import sys
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf

from config import PATHS, SETTINGS


def _load_cache() -> pd.DataFrame:
    path = PATHS["earnings_cache"]
    empty = pd.DataFrame(columns=["Ticker", "NextEarningsDate", "CheckedAt"])
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
        except (pd.errors.EmptyDataError, OSError):
            return empty
        # A cache written by a run with zero tickers is missing these
        # columns entirely — treat it as empty rather than crashing.
        if "NextEarningsDate" not in df.columns or "CheckedAt" not in df.columns:
            return empty
        df["NextEarningsDate"] = pd.to_datetime(df["NextEarningsDate"], errors="coerce")
        df["CheckedAt"] = pd.to_datetime(df["CheckedAt"], errors="coerce")
        return df
    return empty


def _save_cache(df: pd.DataFrame) -> None:
    df.to_csv(PATHS["earnings_cache"], index=False)


def _needs_refresh(row, refresh_days: int) -> bool:
    if pd.isna(row["CheckedAt"]):
        return True
    age_days = (dt.datetime.now() - row["CheckedAt"]).days
    return age_days >= refresh_days


def refresh_earnings_cache(tickers: list, log=print) -> pd.DataFrame:
    cache = _load_cache()
    cache = cache.set_index("Ticker") if not cache.empty else cache
    refresh_days = SETTINGS["earnings_refresh_days"]

    rows = {}
    for ticker in tickers:
        if not cache.empty and ticker in cache.index and not _needs_refresh(cache.loc[ticker], refresh_days):
            rows[ticker] = cache.loc[ticker].to_dict()
            continue

        next_date = None
        try:
            cal = yf.Ticker(ticker).calendar
            if isinstance(cal, dict):
                dates = cal.get("Earnings Date")
                if dates:
                    next_date = pd.to_datetime(dates[0])
            elif cal is not None and not cal.empty and "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"].iloc[0]
                next_date = pd.to_datetime(val)
        except Exception as e:
            log(f"{ticker}: could not fetch earnings date ({e})")

        rows[ticker] = {"NextEarningsDate": next_date, "CheckedAt": dt.datetime.now()}

    out = pd.DataFrame.from_dict(rows, orient="index").reset_index().rename(columns={"index": "Ticker"})
    _save_cache(out)
    return out


def in_earnings_window(next_earnings_date, window_days: int = None) -> bool:
    window_days = window_days or SETTINGS["earnings_window_days"]
    if pd.isna(next_earnings_date):
        return False
    days_out = (pd.to_datetime(next_earnings_date) - pd.Timestamp.now()).days
    return 0 <= days_out <= window_days


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    df = refresh_earnings_cache([ticker])
    print(df)
    if not df.empty:
        print("In window:", in_earnings_window(df.iloc[0]["NextEarningsDate"]))
