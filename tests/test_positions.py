# ==================================================================
#   tests/test_positions.py
#   modules.positions.recommend_action's Buy More/Hold/Trim/Sell/Watch
#   verdict matrix for held positions, including the
#   weak_backtest_history downgrade path added this session.
# ==================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.pattern_types import PatternMatch
from modules.positions import recommend_action

BULLISH = [PatternMatch(pattern="Cup & Handle", direction="bullish",
                         start_date=None, end_date=None, volume_confirms=True)]
BEARISH = [PatternMatch(pattern="Death Cross", direction="bearish",
                         start_date=None, end_date=None, volume_confirms=True)]


def test_deep_loss_with_bearish_signal_is_sell():
    r = recommend_action(avg_cost=100, last_price=80, matches=BEARISH)
    assert r["action"] == "SELL"


def test_strong_gain_with_bearish_signal_is_trim():
    r = recommend_action(avg_cost=100, last_price=130, matches=BEARISH)
    assert r["action"] == "TRIM"


def test_gain_with_bullish_signal_is_buy_more():
    r = recommend_action(avg_cost=100, last_price=110, matches=BULLISH)
    assert r["action"] == "BUY MORE"


def test_gain_with_bullish_signal_but_weak_backtest_is_hold_not_buy_more():
    r = recommend_action(avg_cost=100, last_price=110, matches=BULLISH, weak_backtest_history=True)
    assert r["action"] == "HOLD"


def test_loss_with_bullish_signal_is_hold():
    r = recommend_action(avg_cost=100, last_price=90, matches=BULLISH)
    assert r["action"] == "HOLD"


def test_loss_with_bullish_signal_but_weak_backtest_is_watch_not_hold():
    r = recommend_action(avg_cost=100, last_price=90, matches=BULLISH, weak_backtest_history=True)
    assert r["action"] == "WATCH"


def test_bearish_signal_without_crossing_thresholds_is_watch():
    r = recommend_action(avg_cost=100, last_price=102, matches=BEARISH)
    assert r["action"] == "WATCH"


def test_no_signal_is_hold():
    r = recommend_action(avg_cost=100, last_price=101, matches=[])
    assert r["action"] == "HOLD"


def test_gain_pct_computed_correctly():
    r = recommend_action(avg_cost=50, last_price=60, matches=[])
    assert r["gain_pct"] == 20.0


def test_volume_confirmed_matches_weighted_double():
    confirmed = [PatternMatch(pattern="Golden Cross", direction="bullish",
                               start_date=None, end_date=None, volume_confirms=True)]
    unconfirmed = [PatternMatch(pattern="Golden Cross", direction="bullish",
                                 start_date=None, end_date=None, volume_confirms=False)]
    r_confirmed = recommend_action(avg_cost=100, last_price=101, matches=confirmed)
    r_unconfirmed = recommend_action(avg_cost=100, last_price=101, matches=unconfirmed)
    assert r_confirmed["signal"] == 2 * r_unconfirmed["signal"]
