# ==================================================================
#   modules/ticker_universe.py
#   Builds the ticker universe: your own Fidelity holdings, and/or the
#   DJIA + S&P 500 + Nasdaq-100 constituents scraped from Wikipedia
#   (cached, since these lists barely change).
# ==================================================================
import os
import sys
import csv
import json
import time
import io
import datetime as dt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import requests

from config import PATHS, SETTINGS, DIR_INPUTS

WIKI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}

WIKI_URLS = {
    # Both of these used to have the components table inline on the main
    # index page; Wikipedia has since split it out to a dedicated list
    # article, which is the only place the Symbol/Ticker column still
    # shows up in pd.read_html.
    "DJIA":       "https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies",
    "S&P500":     "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "Nasdaq-100": "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
}

# SEC's official list of every ticker registered with EDGAR — free,
# no API key, ~10,000+ US-listed companies. Updated by the SEC roughly
# daily; we cache it locally per SETTINGS["all_us_refresh_days"] since
# a screener run doesn't need same-day awareness of new listings.
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_HEADERS = {
    # SEC requires a descriptive User-Agent identifying the requester —
    # a generic browser UA gets throttled/blocked.
    "User-Agent": "Stock Screener (personal use) contact: not-provided@example.com"
}


# ------------------------------------------------------------------
# Your own holdings (broker "Positions" CSV export — Fidelity or
# Schwab, see module/options.py docstring for format support notes)
# ------------------------------------------------------------------
# Thinkorswim's "Position Statement" export stacks SEPARATE tables in
# one CSV — an Equities table, then an Options table, then a Futures
# table, etc., each with its OWN header row, rather than Fidelity/
# Schwab's one flat table. read_broker_csv() below handles both: a
# flat file is just the trivial case of one section. Best-effort —
# NOT yet verified against a real thinkorswim export, since none was
# available when this was written. If it doesn't parse your real
# file, tell me a real example row (redact dollar amounts if you
# want) and I'll correct it.
_SECTION_HEADER_HINTS = {"symbol", "instrument", "ticker"}


def read_broker_csv(csv_path) -> pd.DataFrame:
    """Reads a broker Positions CSV that may contain either one flat
    table (Fidelity, Schwab) or several stacked tables each with their
    own header row (thinkorswim's Equities/Options/Futures sections).
    Returns a single DataFrame — the union of every section's columns,
    missing values as NaN for rows from a section that didn't have
    that column. Downstream code (get_traded_tickers, get_open_positions,
    modules.options.get_option_positions) is section-agnostic: it just
    looks for whichever column names it recognizes, wherever they are."""
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        raw_rows = list(csv.reader(f))

    sections = []
    current_header = None
    current_rows = []
    for row in raw_rows:
        cells = [c.strip() for c in row]
        non_empty = [c for c in cells if c]
        is_header = len(non_empty) >= 2 and any(c.lower() in _SECTION_HEADER_HINTS for c in cells)
        if is_header:
            if current_header is not None and current_rows:
                sections.append((current_header, current_rows))
            current_header = cells
            current_rows = []
        elif current_header is not None and len(non_empty) >= 2:
            # A lone single-cell line (e.g. a section title like
            # "Options" sitting on its own line above its real column
            # header, or a blank spacer row) is never a real data row
            # in any broker's position export — skip it rather than
            # let it bleed into the PRIOR section as a bogus row with
            # that one word as the "Symbol".
            current_rows.append(row)
    if current_header is not None and current_rows:
        sections.append((current_header, current_rows))

    if not sections:
        # No recognizable header anywhere — fall back to pandas'
        # normal single-table assumption so a plain, simple export
        # (or one this heuristic doesn't understand) still has a
        # chance of working rather than returning nothing.
        return pd.read_csv(csv_path, index_col=False)

    frames = []
    for header, rows in sections:
        width = len(header)
        normalized_rows = [r[:width] + [""] * (width - len(r)) for r in rows]
        frames.append(pd.DataFrame(normalized_rows, columns=header))
    return pd.concat(frames, ignore_index=True, sort=False)


def resolve_portfolio_csv_path():
    """PATHS["portfolio_csv"] if it exists; otherwise the first *.csv
    found in inputs/ — so a Schwab (or any other broker) export can
    just be dropped in under its own natural filename instead of
    needing to be renamed to match the Fidelity-specific default."""
    configured = Path(PATHS["portfolio_csv"])
    if configured.exists():
        return configured
    candidates = sorted(DIR_INPUTS.glob("*.csv"))
    return candidates[0] if candidates else configured


def get_traded_tickers() -> list:
    """Read your Fidelity Positions CSV export and return the list of
    ticker symbols you hold.

    Fidelity's standard Positions export has a "Symbol" column, plus a
    handful of non-ticker summary rows at the bottom (cash, "Pending
    Activity", disclaimer text) which we filter out heuristically: a
    real ticker is short, uppercase, and doesn't contain spaces.
    """
    csv_path = resolve_portfolio_csv_path()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Broker positions CSV not found at {csv_path}. "
            f"Export your Positions as CSV and drop it in the inputs/ folder."
        )

    df = read_broker_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "Symbol" not in df.columns:
        raise ValueError(
            f"Expected a 'Symbol' column in {csv_path}, found: {list(df.columns)}. "
            f"If your export uses different column names, tell me and "
            f"I'll adjust this parser."
        )

    symbols = df["Symbol"].dropna().astype(str).str.strip().str.upper()
    symbols = symbols[symbols.str.match(r"^[A-Z][A-Z0-9.\-]{0,6}$")]
    # Common non-ticker rows Fidelity includes: cash sweep positions.
    symbols = symbols[~symbols.isin(_CASH_SWEEP_SYMBOLS)]

    return sorted(symbols.unique().tolist())


# Column name candidates across the brokers this has been checked
# against (Fidelity, verified; Schwab and thinkorswim, best-effort
# from their publicly documented export formats — NOT yet verified
# against a real export from either, since none was available when
# this was written. If a real export doesn't parse, tell me the real
# header names from the actual file and I'll correct these.
_QUANTITY_COLS = ["Quantity", "Qty"]
# Already a PER-SHARE cost basis in Fidelity, Schwab, and thinkorswim
# exports (thinkorswim calls it "Trade Price" — the price you actually
# paid, not a current/live figure).
_AVG_COST_COLS = ["Average Cost Basis", "Cost Basis Per Share", "Trade Price"]
# Schwab's "Cost Basis" column is the TOTAL position cost, not
# per-share — handled separately below (divided by Quantity) rather
# than lumped in with _AVG_COST_COLS, which would silently use the
# total as if it were a per-share price.
_TOTAL_COST_BASIS_COLS = ["Cost Basis"]
_CURRENT_VALUE_COLS = ["Current Value", "Market Value", "Mark Value"]
# thinkorswim's current per-share price is called "Mark".
_LAST_PRICE_COLS = ["Last Price", "Price", "Mark"]
# Cash-sweep / money-market symbols to exclude from the traded-ticker
# universe across brokers — these aren't equities with chart patterns.
_CASH_SWEEP_SYMBOLS = {"SPAXX", "FDRXX", "CASH", "SWVXX", "SWGXX", "SNVXX", "SNAXX", "SNSXX", "MMDA1"}


def _first_present(df: pd.DataFrame, candidates: list):
    """Case-insensitive column lookup — Fidelity's real export uses
    lowercase headers like 'Last price' / 'Average cost basis', not
    the title-cased names you might expect."""
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def get_open_positions() -> pd.DataFrame:
    """Read your Fidelity Positions CSV export and return a DataFrame
    of your actual open positions: Ticker, Quantity, AvgCost,
    LastPrice (if present in the export). Used to compute unrealized
    gain/loss for Buy/Sell/Hold recommendations — see modules/positions.py.
    """
    csv_path = resolve_portfolio_csv_path()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Broker positions CSV not found at {csv_path}. "
            f"Export your Positions as CSV and drop it in the inputs/ folder."
        )

    df = read_broker_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "Symbol" not in df.columns:
        raise ValueError(f"Expected a 'Symbol' column in {csv_path}, found: {list(df.columns)}")

    qty_col = _first_present(df, _QUANTITY_COLS)
    cost_col = _first_present(df, _AVG_COST_COLS)
    total_cost_col = _first_present(df, _TOTAL_COST_BASIS_COLS)
    price_col = _first_present(df, _LAST_PRICE_COLS)

    def _clean_money(series):
        return pd.to_numeric(
            series.astype(str).str.replace(r"[$,]", "", regex=True).str.strip(),
            errors="coerce",
        )

    out = pd.DataFrame()
    out["Ticker"] = df["Symbol"].astype(str).str.strip().str.upper()
    out["Quantity"] = _clean_money(df[qty_col]) if qty_col else None
    if cost_col:
        out["AvgCost"] = _clean_money(df[cost_col])
    elif total_cost_col:
        # Schwab-style total cost basis — divide by quantity to get
        # the same per-share figure Fidelity's column already is.
        out["AvgCost"] = _clean_money(df[total_cost_col]) / out["Quantity"]
    else:
        out["AvgCost"] = None
    out["LastPrice"] = _clean_money(df[price_col]) if price_col else None

    out = out[out["Ticker"].str.match(r"^[A-Z][A-Z0-9.\-]{0,6}$", na=False)]
    out = out[~out["Ticker"].isin(_CASH_SWEEP_SYMBOLS)]
    out = out.dropna(subset=["Quantity"])
    out = out[out["Quantity"] != 0]

    if qty_col is None or (cost_col is None and total_cost_col is None):
        missing = [name for name, col in
                   [("Quantity", qty_col), ("AvgCost/Cost Basis", cost_col or total_cost_col)] if col is None]
        raise ValueError(
            f"Could not find columns for {missing} in {csv_path} "
            f"(found: {list(df.columns)}). Tell me your export's real header "
            f"names for these and I'll adjust the parser."
        )

    return out.reset_index(drop=True)


def attach_names(df: pd.DataFrame, log=print) -> pd.DataFrame:
    """Attach a best-effort company Name column, cached to avoid
    hammering yfinance for names on every run."""
    cache_path = PATHS["portfolio_names_cache"]
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            cache = {}

    import yfinance as yf

    names = []
    updated = False
    for ticker in df["Ticker"]:
        if ticker in cache:
            names.append(cache[ticker])
            continue
        try:
            info = yf.Ticker(ticker).info
            name = info.get("shortName") or info.get("longName") or ""
        except Exception as e:
            log(f"Could not fetch name for {ticker}: {e}")
            name = ""
        cache[ticker] = name
        names.append(name)
        updated = True

    if updated:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        except OSError as e:
            log(f"Could not write name cache: {e}")

    df = df.copy()
    df["Name"] = names
    return df


# ------------------------------------------------------------------
# Index constituents (DJIA / S&P 500 / Nasdaq-100), scraped + cached
# ------------------------------------------------------------------
def _scrape_index(name: str, url: str) -> list:
    import io

    resp = requests.get(url, headers=WIKI_HEADERS, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))

    # The constituents table is the first table containing a "Symbol"
    # or "Ticker" column — Wikipedia page layout varies per index.
    for table in tables:
        cols = [str(c).strip() for c in table.columns]
        for candidate in ("Symbol", "Ticker", "Ticker symbol"):
            if candidate in cols:
                symbols = (
                    table[candidate]
                    .astype(str)
                    .str.strip()
                    .str.replace(".", "-", regex=False)  # BRK.B -> BRK-B (yfinance style)
                    .str.upper()
                )
                return sorted(symbols.unique().tolist())
    raise ValueError(f"Could not find a ticker column in the {name} Wikipedia page")


def _cache_is_fresh(meta_path, max_age_days) -> bool:
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        fetched = dt.datetime.fromisoformat(meta["fetched_at"])
    except (json.JSONDecodeError, OSError, KeyError, ValueError):
        return False
    return (dt.datetime.now() - fetched).days < max_age_days


def build_full_universe(force_refresh: bool = False, log=print) -> pd.DataFrame:
    """Returns a DataFrame with Ticker / Source / Index / Name columns
    covering DJIA + S&P 500 + Nasdaq-100, using a cached scrape unless
    force_refresh is set or the cache is stale."""
    cache_path = PATHS["universe_cache"]
    meta_path = PATHS["universe_meta"]
    max_age = SETTINGS["universe_refresh_days"]

    if not force_refresh and _cache_is_fresh(meta_path, max_age) and os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        log(f"Using cached universe ({len(df)} tickers)")
        return df

    rows = []
    for index_name, url in WIKI_URLS.items():
        try:
            symbols = _scrape_index(index_name, url)
            for s in symbols:
                rows.append({"Ticker": s, "Source": "Index", "Index": index_name})
            log(f"{index_name}: {len(symbols)} tickers scraped")
        except Exception as e:
            log(f"Failed to scrape {index_name} ({e}) — will fall back to stale cache if available")

    if not rows:
        if os.path.exists(cache_path):
            log("All scrapes failed — falling back to stale cache")
            return pd.read_csv(cache_path)
        raise RuntimeError("Could not scrape any index constituents and no cache exists")

    df = pd.DataFrame(rows)
    # A ticker in multiple indices (e.g. AAPL in both S&P 500 and
    # Nasdaq-100) should appear once, with its Index column listing all
    # memberships.
    df = (
        df.groupby("Ticker")["Index"]
        .apply(lambda idxs: "+".join(sorted(set(idxs))))
        .reset_index()
    )
    df["Source"] = "Index"

    df = attach_names(df, log=log)

    df.to_csv(cache_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": dt.datetime.now().isoformat()}, f)

    log(f"Universe cache refreshed: {len(df)} unique tickers")
    return df


# ------------------------------------------------------------------
# Every US-listed ticker (SEC EDGAR full company list)
# ------------------------------------------------------------------
def build_all_us_universe(force_refresh: bool = False, log=print) -> pd.DataFrame:
    """Returns a DataFrame with Ticker / Source / Index / Name covering
    every company registered with the SEC (~10,000+ tickers) — the
    broadest universe this screener supports. Company names come
    straight from the SEC file, so no per-ticker yfinance name lookup
    is needed (that would be far too slow at this scale)."""
    cache_path = PATHS["all_us_cache"]
    meta_path = PATHS["all_us_meta"]
    max_age = SETTINGS["all_us_refresh_days"]

    if not force_refresh and _cache_is_fresh(meta_path, max_age) and os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        log(f"Using cached all-US ticker list ({len(df)} tickers)")
        return df

    try:
        resp = requests.get(SEC_TICKERS_URL, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        if os.path.exists(cache_path):
            log(f"Failed to fetch SEC ticker list ({e}) — falling back to stale cache")
            return pd.read_csv(cache_path)
        raise RuntimeError(f"Could not fetch SEC ticker list and no cache exists: {e}")

    rows = []
    for entry in data.values():
        ticker = str(entry.get("ticker", "")).strip().upper().replace(".", "-")
        name = str(entry.get("title", "")).strip()
        if ticker:
            rows.append({"Ticker": ticker, "Source": "AllUS", "Index": "SEC", "Name": name})

    df = pd.DataFrame(rows).drop_duplicates(subset="Ticker").reset_index(drop=True)

    df.to_csv(cache_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": dt.datetime.now().isoformat()}, f)

    log(f"All-US ticker list refreshed: {len(df)} tickers")
    return df


# ------------------------------------------------------------------
# Fast repeat scans — reuse tickers already proven to have real,
# liquid yfinance data from a past run instead of re-touching the
# full ~10,400 SEC list (which includes thousands of illiquid/
# delisted/never-fetchable names that just burn time and trip rate
# limits for nothing).
# ------------------------------------------------------------------
def build_cached_liquid_universe(log=print) -> pd.DataFrame:
    """Returns a DataFrame of every ticker that already has cached
    price history AND currently passes the liquidity filter
    (SETTINGS["min_price"] / ["min_avg_volume"]) — i.e. tickers already
    known-good from a prior full all-us run. Much smaller and faster
    than re-scanning the entire SEC list, at the cost of never
    discovering brand-new tickers that haven't been fetched before
    (run --universe all-us periodically to pick those up)."""
    from modules.market_data import passes_liquidity_filter, DIR_CACHE_OHLCV

    cached_tickers = sorted(p.stem for p in DIR_CACHE_OHLCV.glob("*.csv"))
    if not cached_tickers:
        raise RuntimeError(
            "No cached price history found — run --universe all-us at least "
            "once first so there's something to reuse."
        )

    liquid = [t for t in cached_tickers if passes_liquidity_filter(t)]
    log(f"Cached tickers: {len(cached_tickers)}, liquid: {len(liquid)}")

    # Pull names from the all-US cache if we have it, so the report
    # still shows company names instead of just tickers.
    name_map = {}
    if os.path.exists(PATHS["all_us_cache"]):
        try:
            all_us = pd.read_csv(PATHS["all_us_cache"])
            name_map = dict(zip(all_us["Ticker"], all_us["Name"]))
        except (pd.errors.EmptyDataError, OSError, KeyError):
            pass

    df = pd.DataFrame({
        "Ticker": liquid,
        "Source": "CachedLiquid",
        "Index": "",
        "Name": [name_map.get(t, "") for t in liquid],
    })
    return df


if __name__ == "__main__":
    print("Traded tickers:", get_traded_tickers())
    u = build_full_universe(force_refresh=True)
    print(u.head(20))
    print(f"Total: {len(u)}")
