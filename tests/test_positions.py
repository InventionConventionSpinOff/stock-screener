# ==================================================================
#   tests/test_positions.py
#   modules.positions.recommend_action's Buy More/Hold/Trim/Sell/Watch
#   verdict matrix for held positions, including the
#   weak_backtest_history downgrade path added this session.
# ==================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from modules.pattern_types import PatternMatch
from modules.positions import build_positions_report, recommend_action

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


_ONE_POSITION = pd.DataFrame([{"Ticker": "AAPL", "Quantity": 10, "AvgCost": 100, "LastPrice": 105}])


def test_my_note_and_note_alert_columns_present_even_without_notes():
    df = build_positions_report(_ONE_POSITION, {})
    assert "My Note" in df.columns
    assert "Note Alert" in df.columns
    assert df.iloc[0]["My Note"] == ""
    assert df.iloc[0]["Note Alert"] == ""


def test_note_carried_forward_without_ollama_running(monkeypatch):
    from modules import note_analysis
    monkeypatch.setattr(note_analysis, "ollama_available", lambda: False)
    df = build_positions_report(_ONE_POSITION, {}, notes={"AAPL": "hold until breakout"})
    # Note displays even when the AI check can't run — it just gets no alert.
    assert df.iloc[0]["My Note"] == "hold until breakout"
    assert df.iloc[0]["Note Alert"] == ""


def test_note_alert_set_when_ollama_flags_condition_met(monkeypatch):
    from modules import note_analysis
    monkeypatch.setattr(note_analysis, "ollama_available", lambda: True)
    monkeypatch.setattr(note_analysis, "check_note_condition",
                         lambda *a, **k: {"status": "MET", "explanation": "Breakout completed today."})
    df = build_positions_report(_ONE_POSITION, {}, notes={"AAPL": "hold until breakout"})
    assert "Breakout completed today." in df.iloc[0]["Note Alert"]


def test_deterministic_pattern_match_overrides_a_wrong_llm_not_met(monkeypatch):
    # Guards the exact failure mode a live test caught: the LLM
    # non-deterministically called this NOT_MET on a rerun of
    # identical input. The note names a pattern that genuinely
    # completed today, so the alert must fire regardless of what the
    # (mocked, here deliberately wrong) LLM says.
    from modules import note_analysis
    monkeypatch.setattr(note_analysis, "ollama_available", lambda: True)
    monkeypatch.setattr(note_analysis, "check_note_condition",
                         lambda *a, **k: {"status": "NOT_MET", "explanation": "wrong answer"})
    matches = {"AAPL": [PatternMatch(pattern="Cup & Handle", direction="bullish",
                                      start_date=None, end_date=None)]}
    df = build_positions_report(_ONE_POSITION, matches, notes={"AAPL": "hold until Cup & Handle finishes"})
    assert "Cup & Handle" in df.iloc[0]["Note Alert"]
    assert "completed today" in df.iloc[0]["Note Alert"]


def test_deterministic_match_does_not_require_ollama_running():
    matches = {"AAPL": [PatternMatch(pattern="Cup & Handle", direction="bullish",
                                      start_date=None, end_date=None)]}
    df = build_positions_report(_ONE_POSITION, matches, notes={"AAPL": "hold until Cup & Handle finishes"})
    assert "Cup & Handle" in df.iloc[0]["Note Alert"]


def test_no_note_skips_ai_check_entirely(monkeypatch):
    from modules import note_analysis
    calls = []
    monkeypatch.setattr(note_analysis, "ollama_available", lambda: calls.append("checked") or True)
    df = build_positions_report(_ONE_POSITION, {}, notes={})
    assert calls == []  # never even bothered checking Ollama — nothing to check it for
    assert df.iloc[0]["Note Alert"] == ""
