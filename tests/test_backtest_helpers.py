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

from modules.backtest import pattern_backtest_is_weak, ticker_has_weak_backtest_pattern
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
