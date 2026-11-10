# ==================================================================
#   tests/test_sector_exposure.py
#   Sector concentration should pool stock AND option exposure by the
#   option's UNDERLYING sector — a portfolio that's "only" 20% Tech in
#   shares but also holds a large Tech call position is still
#   concentrated in Tech.
# ==================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import modules.sector_exposure as sector_exposure
from modules.sector_exposure import build_sector_exposure


def _stub_sectors(monkeypatch, mapping):
    monkeypatch.setattr(sector_exposure, "get_sectors",
                         lambda tickers, log=print: {t: mapping.get(t, "Unknown") for t in tickers})


def test_stocks_only_unaffected(monkeypatch):
    _stub_sectors(monkeypatch, {"AAPL": "Technology", "SCHD": "Financials"})
    positions = pd.DataFrame({
        "Ticker": ["AAPL", "SCHD"],
        "Quantity": [10, 10],
        "LastPrice": [200.0, 30.0],
    })
    out = build_sector_exposure(positions)
    tech = out[out["Sector"] == "Technology"].iloc[0]
    assert tech["MarketValue"] == 2000.0


def test_options_contribute_to_underlyings_sector(monkeypatch):
    _stub_sectors(monkeypatch, {"AAPL": "Technology"})
    options = pd.DataFrame({
        "Underlying": ["AAPL"],
        "Quantity": [2],
        "LastPrice": [5.0],
    })
    out = build_sector_exposure(None, options)
    assert not out.empty
    tech = out[out["Sector"] == "Technology"].iloc[0]
    # 2 contracts * $5.00 * 100-share multiplier.
    assert tech["MarketValue"] == 1000.0


def test_stocks_and_options_pool_into_the_same_sector(monkeypatch):
    _stub_sectors(monkeypatch, {"AAPL": "Technology", "MSFT": "Technology"})
    positions = pd.DataFrame({"Ticker": ["AAPL"], "Quantity": [10], "LastPrice": [200.0]})
    options = pd.DataFrame({"Underlying": ["MSFT"], "Quantity": [1], "LastPrice": [10.0]})
    out = build_sector_exposure(positions, options)
    assert len(out) == 1
    tech = out.iloc[0]
    # $2000 equity + ($10 * 100 * 1 contract) = $3000 total Technology.
    assert tech["MarketValue"] == 3000.0
    assert tech["Concentrated"]


def test_no_positions_or_options_returns_empty(monkeypatch):
    _stub_sectors(monkeypatch, {})
    assert build_sector_exposure(None, None).empty
    assert build_sector_exposure(pd.DataFrame(), pd.DataFrame()).empty
