# ==================================================================
#   tests/test_pattern_detectors.py
#   Synthetic-data sanity checks for the completed-pattern detectors —
#   each one should fire on a price series deliberately shaped to
#   match its definition, and NOT fire on flat/random noise. This
#   doesn't replace the real backtest validation (modules/backtest.py
#   against 5 years of real history) — it just catches an outright
#   broken detector (e.g. an off-by-one in a window slice) before it
#   ships, which flat noise + a deliberate shape is enough to do.
# ==================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from modules.pattern_golden_cross import detect_golden_cross
from modules.pattern_breakout import detect_breakout
from modules.pattern_bathtub import detect_bathtub
from modules.pattern_overextension import detect_overextension


def _make_df(closes, volumes=None, start="2020-01-01"):
    n = len(closes)
    dates = pd.bdate_range(start=start, periods=n)
    closes = np.array(closes, dtype=float)
    volumes = np.array(volumes, dtype=float) if volumes is not None else np.full(n, 1_000_000.0)
    return pd.DataFrame({
        "Date": dates,
        "Open": closes, "High": closes * 1.005, "Low": closes * 0.995,
        "Close": closes, "Volume": volumes,
    })


def test_flat_noise_triggers_nothing():
    rng = np.random.default_rng(42)
    closes = 100 + rng.normal(0, 0.3, 300).cumsum() * 0.05  # gentle random walk, no real trend
    df = _make_df(closes)
    assert detect_golden_cross(df) is None
    assert detect_breakout(df) is None
    assert detect_bathtub(df) is None
    assert detect_overextension(df) is None


def test_golden_cross_fires_on_a_clean_uptrend_after_a_downtrend():
    # 220 days declining (so the 200MA sits low), then a rally sized
    # so the 50/200MA crossover lands within the detector's own
    # 10-trading-day lookback window (verified empirically — the
    # cross point moves with rally length/steepness, it isn't fixed).
    decline = np.linspace(100, 80, 220)
    rally = np.linspace(80, 160, 20)
    df = _make_df(np.concatenate([decline, rally]))
    m = detect_golden_cross(df)
    assert m is not None
    assert m.pattern == "Golden Cross"
    assert m.direction == "bullish"


def test_death_cross_fires_on_a_clean_downtrend_after_an_uptrend():
    rally = np.linspace(100, 120, 220)
    decline = np.linspace(120, 60, 20)
    df = _make_df(np.concatenate([rally, decline]))
    m = detect_golden_cross(df)  # same detector, direction depends on data
    assert m is not None
    assert m.pattern == "Death Cross"
    assert m.direction == "bearish"


def test_breakout_fires_on_a_tight_range_then_a_volume_spike_close_above():
    # needs range_window(20) + volume_lookback(20) + 1 bars minimum
    flat = np.full(45, 100.0)
    breakout_day = [106.0]
    closes = np.concatenate([flat, breakout_day])
    volumes = np.concatenate([np.full(45, 1_000_000.0), [5_000_000.0]])
    df = _make_df(closes, volumes)
    m = detect_breakout(df)
    assert m is not None
    assert m.direction == "bullish"
    assert m.volume_confirms is True


def test_breakout_does_not_fire_without_a_volume_spike():
    flat = np.full(45, 100.0)
    closes = np.concatenate([flat, [106.0]])  # same price move, no volume confirmation
    df = _make_df(closes)  # flat 1M volume throughout, no spike
    assert detect_breakout(df) is None


def test_bathtub_fires_on_a_symmetric_recovered_bowl():
    # Leading padding keeps the detector's own lookback_bars=90 window
    # (the last 90 bars only) landing exactly on the bowl shape below —
    # without it, the window clips into the middle of the decline leg
    # and the rim-similarity check fails on the clipped, asymmetric shape.
    pad = np.full(20, 100.0)
    left = np.linspace(100, 80, 30)
    bottom = np.full(10, 80.0)
    right = np.linspace(80, 99, 50)  # recovers ~97% of the way back
    df = _make_df(np.concatenate([pad, left, bottom, right]))
    m = detect_bathtub(df)
    assert m is not None
    assert m.direction == "bullish"


def test_overextension_fires_on_a_parabolic_run():
    base = np.full(60, 50.0)
    run = np.linspace(50, 160, 61)[1:]  # +220% over the window
    df = _make_df(np.concatenate([base, run]))
    m = detect_overextension(df)
    assert m is not None
    assert m.direction == "bearish"  # risk flag, not a buy signal
