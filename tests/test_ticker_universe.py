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


def test_resolver_prefers_configured_path_when_it_exists(fidelity_csv):
    assert ticker_universe.resolve_portfolio_csv_path() == fidelity_csv


def test_resolver_falls_back_to_any_csv_in_inputs_dir(tmp_path, monkeypatch):
    # The configured default doesn't exist, but a differently-named
    # CSV (e.g. a Schwab export under its own natural filename) does —
    # this is the actual bug caught by testing the friend-facing
    # package end-to-end: run_screener.py's step_check_config used to
    # bypass this resolver and check the hardcoded Fidelity path
    # directly, so a Schwab user always saw a false "not found" warning.
    missing_default = tmp_path / "Fidelity_Positions.csv"
    monkeypatch.setitem(PATHS, "portfolio_csv", missing_default)
    monkeypatch.setattr(ticker_universe, "DIR_INPUTS", tmp_path)
    other_csv = tmp_path / "Schwab_Export_Whatever_Name.csv"
    other_csv.write_text("Symbol,Quantity\nAAPL,1\n")
    assert ticker_universe.resolve_portfolio_csv_path() == other_csv


def test_resolver_returns_configured_path_when_nothing_exists(tmp_path, monkeypatch):
    missing = tmp_path / "Fidelity_Positions.csv"
    monkeypatch.setitem(PATHS, "portfolio_csv", missing)
    monkeypatch.setattr(ticker_universe, "DIR_INPUTS", tmp_path)  # empty — no CSV to fall back to
    assert ticker_universe.resolve_portfolio_csv_path() == missing
