# ==================================================================
#   modules/market_data.py
#   Per-ticker OHLCV cache under cache/ohlcv/<TICKER>.csv. New tickers
#   get a full backfill; cached tickers get a short incremental
#   top-up, deduped by date. Per-ticker fetch errors are isolated so
#   one bad/delisted ticker never blocks the rest of the run.
# ==================================================================
import os
import sys
import json
import time
import threading
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf

from config import PATHS, SETTINGS, DIR_CACHE_OHLCV


def _cache_path(ticker: str):
    return DIR_CACHE_OHLCV / f"{ticker}.csv"


def _load_meta() -> dict:
    if os.path.exists(PATHS["ohlcv_meta"]):
        try:
            with open(PATHS["ohlcv_meta"], "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_meta(meta: dict) -> None:
    # Write-then-rename so a concurrent reader (e.g. another screener
    # run happening at the same time) never sees a half-written file.
    path = PATHS["ohlcv_meta"]
    tmp_path = path.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp_path, path)
    except OSError:
        pass


def get_history(ticker: str) -> pd.DataFrame:
    """Read the cached OHLCV history for a ticker. Empty DataFrame if
    nothing is cached (e.g. it failed every fetch attempt so far)."""
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()
    # Belt-and-suspenders against any already-cached incomplete bar
    # (see _fetch_one's dropna) from before this check existed, or any
    # other way a NaN close ended up on disk — every detector in this
    # project assumes the last row's Close is a real, closed price.
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df.sort_values("Date").reset_index(drop=True)


def passes_liquidity_filter(ticker: str, min_price: float = None, min_avg_volume: float = None) -> bool:
    """True if a ticker's latest close and recent average volume clear
    the configured thresholds — used to keep penny stocks and thinly
    traded names out of the new-candidate recommendations. Returns
    False (excluded) if there's no cached history to check."""
    min_price = SETTINGS["min_price"] if min_price is None else min_price
    min_avg_volume = SETTINGS["min_avg_volume"] if min_avg_volume is None else min_avg_volume
    lookback = SETTINGS["liquidity_volume_lookback"]

    hist = get_history(ticker)
    if hist.empty:
        return False

    last_price = hist["Close"].iloc[-1]
    avg_volume = hist["Volume"].iloc[-lookback:].mean()

    return bool(last_price >= min_price and avg_volume >= min_avg_volume)


def passes_activity_filter(ticker: str, lookback: int = None, min_active_fraction: float = None) -> bool:
    """True if a ticker actually traded (moved) on most recent days,
    rather than sitting flat/dead and then spiking once — a single
    spike day can inflate an average-volume number without the ticker
    being genuinely liquid day-to-day. Returns False if there's no
    cached history to check."""
    lookback = SETTINGS["activity_lookback_days"] if lookback is None else lookback
    min_active_fraction = SETTINGS["min_active_day_fraction"] if min_active_fraction is None else min_active_fraction

    hist = get_history(ticker)
    if len(hist) < lookback + 1:
        return False

    window = hist.iloc[-lookback:]
    prev_close = hist["Close"].iloc[-(lookback + 1):-1].values
    active_days = (window["Volume"].values > 0) & (window["Close"].values != prev_close)
    active_fraction = active_days.mean()

    return bool(active_fraction >= min_active_fraction)


def _classify_fetch_error(reason: str) -> str:
    """Buckets a raw exception message into one of a handful of
    categories so refresh_all_history can log one grouped line (e.g.
    "1,461 failed: 1,200 rate-limited, 261 not-found") instead of one
    line per ticker — at all-us scale (10,000+ tickers) the per-ticker
    version drowned out everything else in the log."""
    if not reason:
        return "unknown"
    r = reason.lower()
    if "too many requests" in r or "rate limit" in r:
        return "rate-limited"
    if ("404" in r or "not found" in r or "delisted" in r or "no price data" in r
            or "empty history" in r):
        return "not-found"
    if "timeout" in r or "timed out" in r:
        return "timeout"
    return "other"


def _fetch_one(ticker: str, period: str, retries: int, backoff_base: int) -> tuple:
    """Returns (DataFrame, error_reason) — error_reason is None on
    success. No per-ticker logging here anymore (see
    _classify_fetch_error) — refresh_all_history logs one grouped
    summary at the end instead."""
    last_err = None
    for attempt in range(retries):
        try:
            hist = yf.Ticker(ticker).history(period=period, auto_adjust=False)
            if hist.empty:
                return pd.DataFrame(), "empty history"
            hist = hist.reset_index()
            hist = hist[["Date", "Open", "High", "Low", "Close", "Volume"]]
            hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
            # Yahoo returns a row for today even mid-session, before the
            # bar has actually closed — Close (and sometimes High/Low)
            # comes back NaN for that still-forming bar. Caching it
            # corrupts every detector that reads the latest close (e.g.
            # Overextended's gain_pct). Drop any such incomplete rows;
            # tomorrow's fetch picks up the real closed bar instead.
            hist = hist.dropna(subset=["Open", "High", "Low", "Close"])
            if hist.empty:
                return pd.DataFrame(), "empty history"
            return hist, None
        except Exception as e:
            last_err = e
            if attempt + 1 < retries:
                time.sleep(backoff_base * (attempt + 1))
    return pd.DataFrame(), str(last_err)


def _refresh_one_ticker(ticker: str, retries: int, backoff: int, delay: float, force_full: bool) -> dict:
    """Runs on a worker thread. Fetches + writes cache for one ticker,
    returns a small status dict instead of mutating shared state directly
    (meta/result dict updates happen back on the main thread).

    force_full=True means this ticker's existing cache was backfilled
    under an older (shorter) history_period_full setting — e.g. it was
    fetched back when the setting was "1y" and it's now "5y" — so it
    gets a fresh full backfill (overwriting, not merging) instead of
    just an incremental top-up, to actually deepen its history.
    """
    cache_path = _cache_path(ticker)
    is_new = not os.path.exists(cache_path)
    do_full = is_new or force_full
    period = SETTINGS["history_period_full"] if do_full else SETTINGS["history_period_incremental"]

    # A small per-thread delay (instead of one global sequential delay)
    # still spaces out request bursts across the whole pool without
    # serializing everything back to one-at-a-time.
    if delay:
        time.sleep(delay)

    try:
        fresh, reason = _fetch_one(ticker, period, retries, backoff)
        if fresh.empty:
            return {"ticker": ticker, "status": "failed", "reason": reason}

        if do_full:
            fresh.to_csv(cache_path, index=False)
            return {"ticker": ticker, "status": "full_backfill"}
        else:
            existing = get_history(ticker)
            combined = pd.concat([existing, fresh], ignore_index=True)
            combined = combined.drop_duplicates(subset="Date", keep="last")
            combined = combined.sort_values("Date").reset_index(drop=True)
            combined.to_csv(cache_path, index=False)
            return {"ticker": ticker, "status": "incremental"}

    except Exception as e:
        # Isolate per-ticker failures (e.g. a delisted symbol) so one
        # bad ticker never derails the rest of the batch. No per-ticker
        # log call — the reason is aggregated into refresh_all_history's
        # one grouped summary line instead.
        return {"ticker": ticker, "status": "failed", "reason": str(e)}


def _load_fetch_state() -> dict:
    path = PATHS["fetch_state"]
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_fetch_state(failed_tickers: list, failure_rate: float) -> None:
    with open(PATHS["fetch_state"], "w", encoding="utf-8") as f:
        json.dump({
            "failed_tickers": failed_tickers,
            "failure_rate": round(failure_rate, 4),
            "updated_at": dt.datetime.now().isoformat(),
        }, f)


def _load_dead_tracking() -> dict:
    path = PATHS["dead_tickers"]
    if not os.path.exists(path):
        return {"consecutive_failures": {}, "dead": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("consecutive_failures", {})
        data.setdefault("dead", [])
        return data
    except (json.JSONDecodeError, OSError):
        return {"consecutive_failures": {}, "dead": []}


def _save_dead_tracking(data: dict) -> None:
    with open(PATHS["dead_tickers"], "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def update_dead_ticker_tracking(succeeded: list, failed: list, log=print) -> list:
    """Increments/resets each ticker's consecutive-failure count, and
    marks any ticker that's failed dead_ticker_failure_threshold times
    in a row AND never once produced a usable cache file as dead.
    Returns the list of tickers newly marked dead this call."""
    threshold = SETTINGS.get("dead_ticker_failure_threshold", 5)
    data = _load_dead_tracking()
    counts = data["consecutive_failures"]

    for t in succeeded:
        counts.pop(t, None)

    newly_dead = []
    for t in failed:
        if t in data["dead"]:
            continue
        counts[t] = counts.get(t, 0) + 1
        if counts[t] >= threshold and not os.path.exists(_cache_path(t)):
            newly_dead.append(t)

    if newly_dead:
        data["dead"] = sorted(set(data["dead"]) | set(newly_dead))
        for t in newly_dead:
            counts.pop(t, None)
        log(f"Marking {len(newly_dead)} ticker(s) as permanently dead after "
            f"{threshold}+ straight failed fetches with no cache ever: {newly_dead[:10]}"
            f"{'...' if len(newly_dead) > 10 else ''}")

    data["consecutive_failures"] = counts
    _save_dead_tracking(data)
    return newly_dead


def exclude_dead_tickers(tickers: list) -> list:
    """Filters out tickers already confirmed dead — no point spending
    retry budget/rate-limit headroom on a symbol that's failed 5+
    straight times with nothing to show for it. A ticker can still
    come back: delete cache/ohlcv/_dead_tickers.json to reset this."""
    dead = set(_load_dead_tracking()["dead"])
    if not dead:
        return tickers
    return [t for t in tickers if t not in dead]


def prioritize_previously_failed(tickers: list) -> list:
    """Reorders tickers so ones that failed to fetch last run go
    first. Without this, a run that hits rate limits partway through
    an alphabetical ticker list means late-alphabet tickers (or
    whatever was last in whatever order) never get a real first
    attempt — the failures just repeat in the same spot every time
    instead of eventually covering the whole universe."""
    state = _load_fetch_state()
    failed_last_run = set(state.get("failed_tickers", []))
    if not failed_last_run:
        return tickers
    failed_first = [t for t in tickers if t in failed_last_run]
    rest = [t for t in tickers if t not in failed_last_run]
    return failed_first + rest


def refresh_all_history(tickers: list, log=print) -> dict:
    """Fetch/refresh OHLCV for every ticker in the list, in parallel
    (SETTINGS["fetch_max_workers"] threads at a time — network/API-bound
    work, so threads are effective despite the GIL). Returns a summary
    dict: {full_backfill, incremental, failed}.

    Self-tunes across runs: if the LAST run's failure rate was high
    (rate-limiting), this run automatically eases off — fewer
    concurrent workers, a longer per-request delay — rather than
    hammering an already-throttling endpoint the same way again."""
    meta = _load_meta()
    result = {"full_backfill": 0, "incremental": 0, "failed": []}
    current_period = SETTINGS["history_period_full"]

    retries = SETTINGS["fetch_max_retries"]
    backoff = SETTINGS["fetch_backoff_base_sec"]
    delay = SETTINGS["fetch_politeness_delay_sec"]
    max_workers = SETTINGS.get("fetch_max_workers", 12)

    prior_failure_rate = _load_fetch_state().get("failure_rate", 0.0)
    if prior_failure_rate > 0.3:
        eased_workers = max(4, int(max_workers * (1 - prior_failure_rate)))
        eased_delay = delay * (1 + prior_failure_rate)
        log(f"Last run's failure rate was {prior_failure_rate * 100:.0f}% (likely rate-limited) — "
            f"easing off this run: {max_workers}->{eased_workers} workers, "
            f"{delay}s->{eased_delay:.1f}s delay")
        max_workers, delay = eased_workers, eased_delay

    tickers = exclude_dead_tickers(prioritize_previously_failed(tickers))
    meta_lock = threading.Lock()

    def _stored_period(ticker):
        entry = meta.get(ticker)
        # Old meta format was a bare ISO-timestamp string with no period
        # recorded — treat that as "unknown", which forces a re-backfill
        # under the current (deeper) setting, same as a brand-new ticker.
        if isinstance(entry, dict):
            return entry.get("backfill_period")
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _refresh_one_ticker, t, retries, backoff, delay,
                _stored_period(t) != current_period,
            ): t
            for t in tickers
        }
        failure_reasons = {}
        succeeded = []
        for future in as_completed(futures):
            outcome = future.result()
            status = outcome["status"]
            if status == "failed":
                result["failed"].append(outcome["ticker"])
                category = _classify_fetch_error(outcome.get("reason"))
                failure_reasons[category] = failure_reasons.get(category, 0) + 1
            else:
                succeeded.append(outcome["ticker"])
                result[status] += 1
                with meta_lock:
                    prior = meta.get(outcome["ticker"])
                    prior_period = prior.get("backfill_period") if isinstance(prior, dict) else None
                    meta[outcome["ticker"]] = {
                        "updated_at": dt.datetime.now().isoformat(),
                        "backfill_period": current_period if status == "full_backfill" else prior_period,
                    }

    _save_meta(meta)
    if failure_reasons:
        breakdown = ", ".join(f"{count} {cat}" for cat, count in
                               sorted(failure_reasons.items(), key=lambda kv: -kv[1]))
        log(f"{len(result['failed'])} ticker(s) failed to fetch ({breakdown}) — will retry next run")
    result["failure_reasons"] = failure_reasons

    failure_rate = len(result["failed"]) / len(tickers) if tickers else 0.0
    _save_fetch_state(result["failed"], failure_rate)
    update_dead_ticker_tracking(succeeded, result["failed"], log=log)
    return result


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    r = refresh_all_history([ticker])
    print(r)
    print(get_history(ticker).tail())
