# ==================================================================
#   modules/sector_exposure.py
#   Flags sector concentration across your OPEN positions — e.g. if
#   60% of your invested dollars sit in Technology, a fresh "BUY" call
#   on another tech name is a different decision than it looks in
#   isolation. Sector lookups are cached (they basically never change)
#   to avoid re-hitting yfinance every run.
# ==================================================================
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import DIR_CACHE_UNIVERSE

SECTOR_CACHE_PATH = DIR_CACHE_UNIVERSE / "ticker_sectors.json"

# Above this share of invested dollars in one sector, flag it as
# concentrated — a round number, not a formal risk model.
CONCENTRATION_THRESHOLD_PCT = 30.0


def _load_cache() -> dict:
    if not os.path.exists(SECTOR_CACHE_PATH):
        return {}
    try:
        with open(SECTOR_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(SECTOR_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except OSError:
        pass


def get_sectors(tickers: list, log=print) -> dict:
    """Returns {ticker: sector_or_''}, using a persistent cache."""
    cache = _load_cache()
    updated = False

    import yfinance as yf
    for ticker in tickers:
        if ticker in cache:
            continue
        try:
            sector = yf.Ticker(ticker).info.get("sector") or "Unknown"
        except Exception as e:
            log(f"{ticker}: could not fetch sector ({e})")
            sector = "Unknown"
        cache[ticker] = sector
        updated = True

    if updated:
        _save_cache(cache)

    return {t: cache.get(t, "Unknown") for t in tickers}


def build_sector_exposure(positions_df: pd.DataFrame, log=print) -> pd.DataFrame:
    """positions_df: Ticker/Quantity/LastPrice (from get_open_positions).
    Returns Sector / MarketValue / PortfolioPct, sorted largest first."""
    if positions_df is None or positions_df.empty:
        return pd.DataFrame()

    df = positions_df.copy()
    df["MarketValue"] = df["Quantity"] * df["LastPrice"]
    sectors = get_sectors(df["Ticker"].tolist(), log=log)
    df["Sector"] = df["Ticker"].map(sectors)

    total_value = df["MarketValue"].sum()
    if not total_value:
        return pd.DataFrame()

    out = (
        df.groupby("Sector")["MarketValue"]
        .sum()
        .reset_index()
        .sort_values("MarketValue", ascending=False)
        .reset_index(drop=True)
    )
    out["PortfolioPct"] = round(out["MarketValue"] / total_value * 100, 1)
    out["MarketValue"] = out["MarketValue"].round(2)
    out["Concentrated"] = out["PortfolioPct"] >= CONCENTRATION_THRESHOLD_PCT
    return out


if __name__ == "__main__":
    fake = pd.DataFrame({
        "Ticker": ["AAPL", "NVDA", "SCHD"],
        "Quantity": [2, 1, 6],
        "LastPrice": [341.07, 225.07, 33.21],
    })
    print(build_sector_exposure(fake))
