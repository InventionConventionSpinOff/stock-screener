# ==================================================================
#   modules/news_sentiment.py
#   Free headline sentiment: pulls each ticker's recent headlines from
#   Yahoo Finance's public RSS feed (no API key, no yfinance — its
#   .news/.get_news() endpoint currently 404s on Yahoo's end) and
#   scores them with VADER (a small, offline sentiment lexicon bundled
#   in the vaderSentiment package — no network calls beyond the RSS
#   fetch itself).
#
#   Only meant to be called for a small set of tickers (open
#   positions + that run's surfaced candidates) — fine for dozens of
#   tickers, not built for scanning thousands a day.
# ==================================================================
import os
import sys
import datetime as dt
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import PATHS, SETTINGS

_analyzer = SentimentIntensityAnalyzer()

RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
RSS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}

CACHE_COLUMNS = ["Ticker", "HeadlineCount", "AvgSentiment", "SentimentLabel", "TopHeadline", "CheckedAt"]


def _label(score: float) -> str:
    if score >= SETTINGS["news_bullish_threshold"]:
        return "Bullish"
    if score <= SETTINGS["news_bearish_threshold"]:
        return "Bearish"
    return "Neutral"


def _fetch_one(ticker: str, log=print) -> dict:
    max_headlines = SETTINGS["news_max_headlines_per_ticker"]
    try:
        resp = requests.get(RSS_URL.format(ticker=ticker), headers=RSS_HEADERS, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        log(f"{ticker}: could not fetch news ({e})")
        return None

    headlines = [item.findtext("title") for item in root.findall("./channel/item")]
    headlines = [h.strip() for h in headlines if h and h.strip()][:max_headlines]

    if not headlines:
        return {
            "Ticker": ticker, "HeadlineCount": 0, "AvgSentiment": 0.0,
            "SentimentLabel": "No news", "TopHeadline": "",
        }

    scores = [_analyzer.polarity_scores(h)["compound"] for h in headlines]
    avg_score = sum(scores) / len(scores)
    return {
        "Ticker": ticker,
        "HeadlineCount": len(headlines),
        "AvgSentiment": round(avg_score, 3),
        "SentimentLabel": _label(avg_score),
        "TopHeadline": headlines[0],
    }


def _load_cache() -> pd.DataFrame:
    path = PATHS["news_sentiment_cache"]
    empty = pd.DataFrame(columns=CACHE_COLUMNS)
    if not os.path.exists(path):
        return empty
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return empty
    if not set(CACHE_COLUMNS).issubset(df.columns):
        return empty
    df["CheckedAt"] = pd.to_datetime(df["CheckedAt"], errors="coerce")
    return df


def _save_cache(df: pd.DataFrame) -> None:
    df.to_csv(PATHS["news_sentiment_cache"], index=False)


def get_news_sentiment(tickers: list, log=print) -> pd.DataFrame:
    """Returns Ticker/HeadlineCount/AvgSentiment/SentimentLabel/TopHeadline
    for the given tickers, reusing any still-fresh cached score
    (SETTINGS["news_refresh_hours"]) instead of re-fetching every run."""
    if not tickers:
        return pd.DataFrame(columns=CACHE_COLUMNS).drop(columns=["CheckedAt"])

    cache = _load_cache().set_index("Ticker")
    refresh_hours = SETTINGS["news_refresh_hours"]
    now = dt.datetime.now()

    rows = {}
    changed = False
    for ticker in tickers:
        if ticker in cache.index:
            checked_at = cache.loc[ticker, "CheckedAt"]
            if pd.notna(checked_at) and (now - checked_at).total_seconds() / 3600 < refresh_hours:
                rows[ticker] = {"Ticker": ticker, **cache.loc[ticker].to_dict()}
                continue

        result = _fetch_one(ticker, log=log)
        if result is None:
            if ticker in cache.index:
                rows[ticker] = {"Ticker": ticker, **cache.loc[ticker].to_dict()}
            continue
        result["CheckedAt"] = now
        rows[ticker] = result
        changed = True

    out = pd.DataFrame.from_dict(rows, orient="index").reset_index(drop=True)

    if changed:
        cache_rows = cache.reset_index()
        if cache_rows.empty:
            merged = out
        else:
            merged = pd.concat([cache_rows, out], ignore_index=True)
        merged = merged.drop_duplicates(subset="Ticker", keep="last")
        _save_cache(merged[CACHE_COLUMNS])
        _append_history(out[out["CheckedAt"] == now])

    return out.drop(columns=["CheckedAt"], errors="ignore")


def _append_history(fresh_rows: pd.DataFrame) -> None:
    """Appends today's freshly-fetched (not cache-reused) sentiment
    readings to the permanent history log, one row per ticker per
    calendar day — skips a ticker already logged today so re-running
    within the same day doesn't duplicate rows."""
    if fresh_rows.empty:
        return
    path = PATHS["sentiment_history"]
    today = dt.date.today().isoformat()
    new_rows = fresh_rows[["Ticker", "AvgSentiment", "SentimentLabel", "HeadlineCount"]].copy()
    new_rows.insert(1, "Date", today)

    if os.path.exists(path):
        try:
            existing = pd.read_csv(path)
        except (pd.errors.EmptyDataError, OSError):
            existing = pd.DataFrame(columns=new_rows.columns)
        already = set(existing[existing["Date"] == today]["Ticker"]) if not existing.empty else set()
        new_rows = new_rows[~new_rows["Ticker"].isin(already)]
        if new_rows.empty:
            return
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined.to_csv(path, index=False)


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(get_news_sentiment([ticker]))
