# ==================================================================
#   tests/test_options.py
#   modules.options.parse_option_symbol — no real option position was
#   available to verify the OCC symbol format against when this was
#   built, so these are the only safety net confirming the regex does
#   what the module docstring claims.
# ==================================================================
import sys
from pathlib import Path
import datetime as dt

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.options import parse_option_symbol


def test_parses_call_with_leading_dash():
    r = parse_option_symbol("-AAPL251219C00200000")
    assert r == {"underlying": "AAPL", "expiration": dt.date(2025, 12, 19),
                 "option_type": "CALL", "strike": 200.0}


def test_parses_put_without_leading_dash():
    r = parse_option_symbol("TSLA260117P00450000")
    assert r["option_type"] == "PUT"
    assert r["underlying"] == "TSLA"
    assert r["strike"] == 450.0
    assert r["expiration"] == dt.date(2026, 1, 17)


def test_parses_fractional_strike():
    r = parse_option_symbol("SPY251031C00599500")
    assert r["strike"] == 599.5


def test_tolerates_embedded_spaces():
    r = parse_option_symbol("- AAPL 251219 C 00200000")
    assert r is not None
    assert r["underlying"] == "AAPL"


def test_rejects_plain_equity_symbol():
    assert parse_option_symbol("AAPL") is None


def test_rejects_cash_sweep_symbol():
    assert parse_option_symbol("SPAXX**") is None


def test_rejects_garbage():
    assert parse_option_symbol("not an option") is None
    assert parse_option_symbol("") is None
    assert parse_option_symbol(None) is None


def test_rejects_invalid_calendar_date():
    # Month 13 doesn't exist — should fail the date construction, not raise.
    assert parse_option_symbol("AAPL251319C00200000") is None


# Schwab's space-separated human-readable format — best-effort,
# unverified against a real Schwab export (see module docstring).

def test_parses_schwab_call():
    r = parse_option_symbol("AAPL 01/17/2025 150.00 C")
    assert r == {"underlying": "AAPL", "expiration": dt.date(2025, 1, 17),
                 "option_type": "CALL", "strike": 150.0}


def test_parses_schwab_put_without_decimal():
    r = parse_option_symbol("TSLA 11/21/2025 450 P")
    assert r["option_type"] == "PUT"
    assert r["strike"] == 450.0
    assert r["expiration"] == dt.date(2025, 11, 21)


def test_parses_schwab_lowercase_and_extra_spaces():
    r = parse_option_symbol("spy  10/31/2026   599.5   c")
    assert r is not None
    assert r["underlying"] == "SPY"
    assert r["strike"] == 599.5


def test_schwab_format_rejects_invalid_calendar_date():
    assert parse_option_symbol("AAPL 13/17/2025 150.00 C") is None
