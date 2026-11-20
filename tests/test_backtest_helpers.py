# ==================================================================
#   tests/test_backtest_helpers.py
#   modules.backtest's pattern_backtest_is_weak gate — shared by both
#   modules.recommendation (new candidates) and modules.positions
#   (held positions). A regression here silently breaks the downgrade
#   logic in BOTH places at once.
# ==================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from modules.backtest import (pattern_backtest_is_weak, ticker_has_weak_backtest_pattern,
                               pattern_backtest_is_weak_now, ticker_has_weak_backtest_pattern_now)
from modules.pattern_types import PatternMatch


def _stats_row(pattern, win_rate, avg_return, sample_size=100):
    return pd.DataFrame([{
        "Pattern": pattern, "Sample Size": sample_size,
        "WinRate_20d_pct": win_rate, "AvgReturn_20d_pct": avg_return,
    }])


def test_low_win_rate_and_negative_return_is_weak():
    stats = _stats_row("Death Cross", 38.0, -2.5)
    assert pattern_backtest_is_weak("Death Cross", stats)


def test_low_win_rate_but_positive_return_is_not_weak():
    # Big wins, small/rare losses can still have a sub-50% win rate
    # and be profitable overall — win rate alone isn't damning.
    stats = _stats_row("Flag", 40.0, 1.5)
    assert not pattern_backtest_is_weak("Flag", stats)


def test_decent_win_rate_is_not_weak():
    stats = _stats_row("Cup & Handle", 55.0, 1.2)
    assert not pattern_backtest_is_weak("Cup & Handle", stats)


def test_thin_sample_is_not_weak_even_if_bad():
    # Below WEAK_BACKTEST_MIN_SAMPLE — too little data to trust.
    stats = _stats_row("Flag", 30.0, -3.0, sample_size=10)
    assert not pattern_backtest_is_weak("Flag", stats)


def test_empty_stats_is_not_weak():
    assert not pattern_backtest_is_weak("Death Cross", pd.DataFrame())


def test_unknown_pattern_is_not_weak():
    stats = _stats_row("Death Cross", 38.0, -2.5)
    assert not pattern_backtest_is_weak("Some New Pattern", stats)


def test_ticker_weak_if_any_matched_pattern_is_weak():
    stats = pd.concat([
        _stats_row("Death Cross", 38.0, -2.5),
        _stats_row("Cup & Handle", 55.0, 1.2),
    ], ignore_index=True)
    matches = [
        PatternMatch(pattern="Cup & Handle", direction="bullish", start_date=None, end_date=None),
        PatternMatch(pattern="Death Cross", direction="bearish", start_date=None, end_date=None),
    ]
    assert ticker_has_weak_backtest_pattern(matches, stats)


def test_ticker_not_weak_if_no_matched_pattern_is_weak():
    stats = _stats_row("Cup & Handle", 55.0, 1.2)
    matches = [PatternMatch(pattern="Cup & Handle", direction="bullish", start_date=None, end_date=None)]
    assert not ticker_has_weak_backtest_pattern(matches, stats)


# ------------------------------------------------------------------
# Raised floor (45.0 -> 46.0), added after the 2026-11-09 fresh
# backtest showed Double Top at 45.9% win / -1.55% avg return — a real
# losing pattern the old floor missed by a tenth of a point.
# ------------------------------------------------------------------
def test_double_top_at_fresh_backtest_numbers_is_now_weak():
    stats = _stats_row("Double Top", 45.9, -1.55, sample_size=20409)
    assert pattern_backtest_is_weak("Double Top", stats)


def test_breakout_at_fresh_backtest_numbers_stays_not_weak():
    # Positive avg return keeps this excluded regardless of the floor.
    stats = _stats_row("Breakout", 48.7, 0.13, sample_size=25777)
    assert not pattern_backtest_is_weak("Breakout", stats)


# ------------------------------------------------------------------
# pattern_backtest_is_weak_now — the regime-aware gate.
# ------------------------------------------------------------------
def _regime_row(pattern, regime, win_rate, avg_return, sample_size=200):
    return pd.DataFrame([{
        "Pattern": pattern, "Regime": regime, "Sample Size": sample_size,
        "WinRate_20d_pct": win_rate, "AvgReturn_20d_pct": avg_return,
    }])


def test_regime_sensitive_pattern_uses_bear_specific_stats_when_trustworthy():
    # Blended stats look fine, but the Bear-specific slice is bad and
    # has enough sample to trust — should override the blended "not weak."
    blended = _stats_row("Death Cross", 55.0, 1.0)
    regime = _regime_row("Death Cross", "Bear", 30.0, -8.33, sample_size=300)
    assert pattern_backtest_is_weak_now("Death Cross", blended, regime, "Bear")


def test_regime_sensitive_pattern_falls_back_to_blended_when_regime_sample_too_thin():
    blended = _stats_row("Death Cross", 38.0, -2.5)
    regime = _regime_row("Death Cross", "Bear", 30.0, -8.33, sample_size=50)  # below MIN_REGIME_SAMPLE
    assert pattern_backtest_is_weak_now("Death Cross", blended, regime, "Bear") == \
        pattern_backtest_is_weak("Death Cross", blended)


def test_non_regime_sensitive_pattern_ignores_regime_stats_entirely():
    blended = _stats_row("Flag", 38.0, -2.5)
    regime = _regime_row("Flag", "Bear", 70.0, 5.0, sample_size=300)  # would flip the verdict if used
    assert pattern_backtest_is_weak_now("Flag", blended, regime, "Bear") == \
        pattern_backtest_is_weak("Flag", blended)


def test_unknown_regime_falls_back_to_blended():
    blended = _stats_row("Death Cross", 38.0, -2.5)
    regime = _regime_row("Death Cross", "Bear", 30.0, -8.33, sample_size=300)
    assert pattern_backtest_is_weak_now("Death Cross", blended, regime, "Unknown") == \
        pattern_backtest_is_weak("Death Cross", blended)


def test_ticker_has_weak_backtest_pattern_now_matches_any_matched_pattern():
    blended = _stats_row("Death Cross", 55.0, 1.0)
    regime = _regime_row("Death Cross", "Bear", 30.0, -8.33, sample_size=300)
    matches = [PatternMatch(pattern="Death Cross", direction="bearish", start_date=None, end_date=None)]
    assert ticker_has_weak_backtest_pattern_now(matches, blended, regime, "Bear")


# ------------------------------------------------------------------
# Sector-conditioned gate — Death Cross is the only pattern with a
# validated sector effect (analyze_sector_performance.py), worst in
# Financial Services, least-bad in Healthcare.
# ------------------------------------------------------------------
def _sector_row(pattern, sector, win_rate, avg_return, sample_size=200):
    return pd.DataFrame([{
        "Pattern": pattern, "Sector": sector, "Sample Size": sample_size,
        "WinRate_20d_pct": win_rate, "AvgReturn_20d_pct": avg_return,
    }])


def test_sector_sensitive_pattern_uses_sector_specific_stats_when_trustworthy():
    blended = _stats_row("Death Cross", 55.0, 1.0)  # looks fine blended
    sector = _sector_row("Death Cross", "Financial Services", 30.0, -5.0, sample_size=300)
    assert pattern_backtest_is_weak_now("Death Cross", blended, sector_stats_df=sector,
                                         ticker_sector="Financial Services")


def test_sector_sensitive_pattern_falls_back_when_sector_sample_too_thin():
    blended = _stats_row("Death Cross", 38.0, -2.5)
    sector = _sector_row("Death Cross", "Financial Services", 30.0, -5.0, sample_size=50)
    assert pattern_backtest_is_weak_now("Death Cross", blended, sector_stats_df=sector,
                                         ticker_sector="Financial Services") == \
        pattern_backtest_is_weak("Death Cross", blended)


def test_non_sector_sensitive_pattern_ignores_sector_stats_entirely():
    blended = _stats_row("Flag", 38.0, -2.5)
    sector = _sector_row("Flag", "Financial Services", 70.0, 5.0, sample_size=300)
    assert pattern_backtest_is_weak_now("Flag", blended, sector_stats_df=sector,
                                         ticker_sector="Financial Services") == \
        pattern_backtest_is_weak("Flag", blended)


def test_no_ticker_sector_falls_back_to_blended():
    blended = _stats_row("Death Cross", 38.0, -2.5)
    sector = _sector_row("Death Cross", "Financial Services", 30.0, -5.0, sample_size=300)
    assert pattern_backtest_is_weak_now("Death Cross", blended, sector_stats_df=sector,
                                         ticker_sector=None) == \
        pattern_backtest_is_weak("Death Cross", blended)


def test_regime_and_sector_signals_are_independent_either_can_flag_weak():
    # Good regime (Bull), but bad sector (Financial Services) — should
    # still be flagged weak off the sector signal alone.
    blended = _stats_row("Death Cross", 55.0, 1.0)
    regime = _regime_row("Death Cross", "Bull", 60.0, 2.0, sample_size=300)  # looks fine in Bull
    sector = _sector_row("Death Cross", "Financial Services", 30.0, -5.0, sample_size=300)
    assert pattern_backtest_is_weak_now("Death Cross", blended, regime, "Bull", sector, "Financial Services")


def test_sector_in_a_good_sector_is_not_weak_even_with_bad_blended():
    # Healthcare is the "least-bad" sector for Death Cross — a good
    # sector-specific reading should override a bad blended one, same
    # as the regime gate's precedence.
    blended = _stats_row("Death Cross", 30.0, -5.0)
    sector = _sector_row("Death Cross", "Healthcare", 55.0, 1.0, sample_size=300)
    assert not pattern_backtest_is_weak_now("Death Cross", blended, sector_stats_df=sector,
                                             ticker_sector="Healthcare")
