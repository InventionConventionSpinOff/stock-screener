# ==================================================================
#   tests/test_ticker_universe.py
#   modules.ticker_universe.get_open_positions against both a
#   Fidelity-shaped CSV (verified format) and a Schwab-shaped one
#   (best-effort, unverified — see modules/options.py's module
#   docstring for the same caveat on the options side).
# ==================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from config import PATHS
import modules.ticker_universe as ticker_universe


@pytest.fixture
def fidelity_csv(tmp_path, monkeypatch):
    path = tmp_path / "Fidelity_Positions.csv"
    path.write_text(
        "Account number,Account name,Symbol,Description,Quantity,Last price,Average cost basis\n"
        "Z123,Individual,AAPL,APPLE INC,10,$200.00,$150.00\n"
        "Z123,Individual,SPAXX**,HELD IN MONEY MARKET,,,\n"
    )
    monkeypatch.setitem(PATHS, "portfolio_csv", path)
    yield path


@pytest.fixture
def schwab_csv(tmp_path, monkeypatch):
    path = tmp_path / "Schwab_Positions.csv"
    path.write_text(
        "Symbol,Description,Quantity,Price,Cost Basis,Market Value\n"
        "AAPL,APPLE INC,10,200.00,1500.00,2000.00\n"
    )
    monkeypatch.setitem(PATHS, "portfolio_csv", path)
    yield path


@pytest.fixture
def thinkorswim_csv(tmp_path, monkeypatch):
    # Multi-section layout: a lone section-title line ("Equities",
    # "Options") sits ABOVE each section's own column-header row —
    # this shape is what originally leaked "Options" into the ticker
    # list as a bogus Symbol before the section-boundary fix.
    path = tmp_path / "tos_positions.csv"
    path.write_text(
        "Account Statement for XXXX1234 on 11/03/26\n"
        "\n"
        "Cash & Sweep Vehicle\n"
        "Symbol,Description,Qty,Mark Value\n"
        "MMDA1,MONEY MARKET,1,500.00\n"
        "\n"
        "Equities\n"
        "Symbol,Description,Qty,Trade Price,Mark,Mark Value\n"
        "AAPL,APPLE INC,15,150.00,210.50,3157.50\n"
        "MSFT,MICROSOFT CORP,8,250.00,340.20,2721.60\n"
        "\n"
        "Options\n"
        "Symbol,Description,Qty,Trade Price,Mark,Mark Value\n"
        "AAPL  260116C00220000,APPLE INC 16 JAN 26 220 CALL,3,3.00,4.80,1440.00\n"
    )
    monkeypatch.setitem(PATHS, "portfolio_csv", path)
    yield path


def test_fidelity_avg_cost_used_directly(fidelity_csv):
    df = ticker_universe.get_open_positions()
    row = df[df["Ticker"] == "AAPL"].iloc[0]
    assert row["AvgCost"] == 150.0
    assert row["LastPrice"] == 200.0
    assert row["Quantity"] == 10


def test_fidelity_excludes_cash_sweep(fidelity_csv):
    df = ticker_universe.get_open_positions()
    assert "SPAXX**" not in df["Ticker"].values
    assert "SPAXX" not in df["Ticker"].values


def test_schwab_total_cost_basis_divided_by_quantity(schwab_csv):
    df = ticker_universe.get_open_positions()
    row = df[df["Ticker"] == "AAPL"].iloc[0]
    # Cost Basis (1500.00) / Quantity (10) == 150.00 per share —
    # matching the Fidelity case above despite the different source
    # column semantics (total vs. already-per-share).
    assert row["AvgCost"] == 150.0
    assert row["LastPrice"] == 200.0


def test_thinkorswim_multi_section_equities_parsed(thinkorswim_csv):
    df = ticker_universe.get_open_positions()
    tickers = set(df["Ticker"])
    assert tickers == {"AAPL", "MSFT"}
    aapl = df[df["Ticker"] == "AAPL"].iloc[0]
    assert aapl["AvgCost"] == 150.0   # Trade Price
    assert aapl["LastPrice"] == 210.5  # Mark
    assert aapl["Quantity"] == 15


def test_thinkorswim_section_title_line_not_leaked_as_ticker(thinkorswim_csv):
    # Regression test: "Options" (the lone section-title line sitting
    # above its own column-header row) used to bleed into the prior
    # section as a bogus one-cell "data" row before the section-
    # boundary fix, producing a fake "OPTIONS" ticker.
    tickers = ticker_universe.get_traded_tickers()
    assert "OPTIONS" not in tickers
    assert "EQUITIES" not in tickers


def test_thinkorswim_excludes_cash_sweep(thinkorswim_csv):
    tickers = ticker_universe.get_traded_tickers()
    assert "MMDA1" not in tickers


def test_thinkorswim_option_section_parsed_via_options_module(thinkorswim_csv):
    from modules.options import get_option_positions
    opts = get_option_positions()
    assert len(opts) == 1
    row = opts.iloc[0]
    assert row["Underlying"] == "AAPL"
    assert row["OptionType"] == "CALL"
    assert row["Strike"] == 220.0
    assert row["AvgCost"] == 3.0   # Trade Price
    assert row["LastPrice"] == 4.8  # Mark


def test_resolver_picks_newest_csv_regardless_of_name(tmp_path, monkeypatch):
    # An OLDER export happens to sit at the configured default
    # filename; a NEWER export (re-downloaded under the broker's own
    # auto-generated name) sits right next to it. The newer one must
    # win — filename is not a proxy for freshness.
    monkeypatch.setitem(PATHS, "portfolio_csv", tmp_path / "Fidelity_Positions.csv")

    old_default = tmp_path / "Fidelity_Positions.csv"
    old_default.write_text("Symbol,Quantity\nAAPL,1\n")
    new_export = tmp_path / "Portfolio_Positions_Nov-09-2026.csv"
    new_export.write_text("Symbol,Quantity\nMSFT,1\n")

    import os
    import time
    old_time = time.time() - 3600
    os.utime(old_default, (old_time, old_time))

    assert ticker_universe.resolve_portfolio_csv_path() == new_export


def test_resolver_picks_newest_even_when_it_is_the_configured_name(tmp_path, monkeypatch):
    # Same setup, but this time the configured-named file is the
    # NEWER one — it should still win on recency, not because of its
    # name.
    monkeypatch.setitem(PATHS, "portfolio_csv", tmp_path / "Fidelity_Positions.csv")

    older_export = tmp_path / "Portfolio_Positions_Oct-01-2026.csv"
    older_export.write_text("Symbol,Quantity\nMSFT,1\n")
    import os
    import time
    old_time = time.time() - 3600
    os.utime(older_export, (old_time, old_time))

    newer_default = tmp_path / "Fidelity_Positions.csv"
    newer_default.write_text("Symbol,Quantity\nAAPL,1\n")

    assert ticker_universe.resolve_portfolio_csv_path() == newer_default


def test_resolver_falls_back_to_any_csv_in_inputs_dir(tmp_path, monkeypatch):
    # The configured default doesn't exist, but a differently-named
    # CSV (e.g. a Schwab export under its own natural filename) does —
    # this is the actual bug caught by testing the friend-facing
    # package end-to-end: run_screener.py's step_check_config used to
    # bypass this resolver and check the hardcoded Fidelity path
    # directly, so a Schwab user always saw a false "not found" warning.
    missing_default = tmp_path / "Fidelity_Positions.csv"
    monkeypatch.setitem(PATHS, "portfolio_csv", missing_default)
    other_csv = tmp_path / "Schwab_Export_Whatever_Name.csv"
    other_csv.write_text("Symbol,Quantity\nAAPL,1\n")
    assert ticker_universe.resolve_portfolio_csv_path() == other_csv


def test_resolver_returns_configured_path_when_nothing_exists(tmp_path, monkeypatch):
    missing = tmp_path / "Fidelity_Positions.csv"
    monkeypatch.setitem(PATHS, "portfolio_csv", missing)  # tmp_path is empty — no CSV to fall back to
    assert ticker_universe.resolve_portfolio_csv_path() == missing
