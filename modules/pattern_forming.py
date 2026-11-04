# ==================================================================
#   modules/pattern_forming.py
#   Every detector in pattern_*.py (other than this one) only fires
#   once a pattern has fully completed — Golden Cross requires the MA
#   crossover to have ALREADY happened, Breakout requires the close to
#   have ALREADY cleared the range, Bathtub requires recovery to have
#   ALREADY hit 85%. By the time any of those fire, some of the move
#   is usually already gone.
#
#   This module detects the SAME setups while they're still in
#   progress — not a different pattern, a leading-edge version of the
#   same geometry, with a `confidence` (0-1, see PatternMatch) that
#   reflects how far along / how trustworthy the in-progress read is.
#   Each detector explicitly bails out if the completed version would
#   already fire, so a ticker is never double-counted as both
#   "forming" and "complete" for the same setup.
#
#   Important: these are NOT validated by modules/backtest.py the way
#   the completed patterns are — "will this cup actually finish
#   forming" is a materially different (harder) question than "did
#   this cup's forward return make money," and backtesting a
#   not-yet-complete shape requires its own walk-forward design. Treat
#   these as early heads-up flags to watch, not a backtested edge.
# ==================================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from modules.pattern_types import PatternMatch
from modules.pattern_golden_cross import PARAMS as GC_PARAMS, detect_golden_cross
from modules.pattern_breakout import PARAMS as BREAKOUT_PARAMS, detect_breakout
from modules.pattern_bathtub import PARAMS as BATHTUB_PARAMS, detect_bathtub
from modules.pattern_flag import PARAMS as FLAG_PARAMS, detect_flag

# A forming signal below this confidence is too weak/noisy to be
# worth a user's attention — filtered out at the source rather than
# left for the report to sort through.
MIN_CONFIDENCE = 0.5


def detect_golden_cross_forming(df: pd.DataFrame, params: dict = None) -> PatternMatch:
    """Projects the 50MA/200MA gap forward using a linear fit over the
    last `trend_window` days — if it's closing at a steady rate and on
    track to cross within `max_days_ahead` trading days, flags it
    ahead of the actual crossover rather than after."""
    p = {**GC_PARAMS, **(params or {})}
    trend_window = 10
    max_days_ahead = 15
    needed = p["slow_window"] + trend_window
    if len(df) < needed:
        return None
    if detect_golden_cross(df, params) is not None:
        return None  # already crossed — that's the completed detector's job

    d = df.copy()
    d["fast_ma"] = d["Close"].rolling(p["fast_window"]).mean()
    d["slow_ma"] = d["Close"].rolling(p["slow_window"]).mean()
    d = d.dropna(subset=["fast_ma", "slow_ma"]).reset_index(drop=True)
    if len(d) < trend_window:
        return None

    recent = (d["fast_ma"] - d["slow_ma"]).values[-trend_window:]
    x = np.arange(trend_window)
    slope, intercept = np.polyfit(x, recent, 1)
    if slope == 0:
        return None

    diff_today = recent[-1]
    zero_at = -intercept / slope
    days_ahead = zero_at - (trend_window - 1)
    if not (0 < days_ahead <= max_days_ahead):
        return None

    if diff_today < 0 and slope > 0:
        direction, label = "bullish", "Golden Cross"
    elif diff_today > 0 and slope < 0:
        direction, label = "bearish", "Death Cross"
    else:
        return None

    fitted = intercept + slope * x
    ss_res = float(np.sum((recent - fitted) ** 2))
    ss_tot = float(np.sum((recent - recent.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    confidence = max(0.0, min(1.0, r2))
    if confidence < MIN_CONFIDENCE:
        return None

    return PatternMatch(
        pattern=f"{label} (Forming)",
        direction=direction,
        start_date=d["Date"].iloc[-trend_window],
        end_date=d["Date"].iloc[-1],
        key_levels={"days_to_cross": round(days_ahead, 1), "trend_r2": round(r2, 2)},
        volume_confirms=False,
        confidence=round(confidence, 2),
        note=f"50/200MA gap closing steadily — projected {label.lower()} in "
             f"~{round(days_ahead)} trading days (trend fit R²={r2:.2f})",
    )


def detect_breakout_forming(df: pd.DataFrame, params: dict = None) -> PatternMatch:
    """Price within `proximity_pct` of the same N-day range boundary
    detect_breakout watches, with volume already building above its
    own recent baseline (even if not yet at the hard 1.8x gate)."""
    p = {**BREAKOUT_PARAMS, **(params or {})}
    proximity_pct = 2.0
    needed = p["range_window"] + p["volume_lookback"] + 1
    if len(df) < needed:
        return None
    if detect_breakout(df, params) is not None:
        return None  # already broken out

    today = df.iloc[-1]
    prior = df.iloc[-(p["range_window"] + 1):-1]
    prior_high = prior["High"].max()
    prior_low = prior["Low"].min()

    vol_baseline = df["Volume"].iloc[-(p["volume_lookback"] + 1):-1].mean()
    vol_ratio = today["Volume"] / vol_baseline if vol_baseline else 0.0
    if vol_ratio < 1.0:
        return None  # volume isn't even building yet — too early to flag

    dist_to_high_pct = (prior_high - today["Close"]) / today["Close"] * 100
    dist_to_low_pct = (today["Close"] - prior_low) / today["Close"] * 100

    if 0 <= dist_to_high_pct <= proximity_pct:
        direction, dist, level = "bullish", dist_to_high_pct, prior_high
    elif 0 <= dist_to_low_pct <= proximity_pct:
        direction, dist, level = "bearish", dist_to_low_pct, prior_low
    else:
        return None

    proximity_score = 1 - dist / proximity_pct
    volume_score = min(vol_ratio / p["volume_spike_mult"], 1.0)
    confidence = max(0.0, min(1.0, 0.6 * proximity_score + 0.4 * volume_score))
    if confidence < MIN_CONFIDENCE:
        return None

    return PatternMatch(
        pattern="Breakout (Approaching)",
        direction=direction,
        start_date=prior["Date"].iloc[0],
        end_date=today["Date"],
        key_levels={"distance_pct": round(dist, 2), "level": round(level, 2), "volume_ratio": round(vol_ratio, 2)},
        volume_confirms=vol_ratio >= p["volume_spike_mult"],
        confidence=round(confidence, 2),
        note=f"Within {round(dist, 1)}% of {'resistance' if direction == 'bullish' else 'support'} "
             f"({round(level, 2)}), volume already {round(vol_ratio, 1)}x avg and building",
    )


def detect_bathtub_forming(df: pd.DataFrame, params: dict = None) -> PatternMatch:
    """Same bowl geometry as detect_bathtub, but recovery is still
    under way (below the 85% confirmation threshold) rather than
    already there — the natural precursor to both Bathtub and Cup &
    Handle completions, since both start from this same recovering
    bowl shape."""
    p = {**BATHTUB_PARAMS, **(params or {})}
    min_recovery_pct_forming = 40.0  # below this, it's just noise near the bottom, not a real recovery yet
    if len(df) < p["lookback_bars"] + p["volume_lookback"]:
        return None
    if detect_bathtub(df, params) is not None:
        return None  # already fully recovered

    window = df.iloc[-p["lookback_bars"]:].reset_index(drop=True)
    left_rim = window["Close"].iloc[0]
    right_rim = window["Close"].iloc[-1]
    bottom_idx = window["Low"].idxmin()
    bottom = window["Low"].iloc[bottom_idx]

    if not (len(window) * 0.25 < bottom_idx < len(window) * 0.75):
        return None

    rim_avg = (left_rim + right_rim) / 2
    depth_pct = (rim_avg - bottom) / rim_avg * 100
    if not (p["min_depth_pct"] <= depth_pct <= p["max_depth_pct"]):
        return None

    recovery_pct = (right_rim - bottom) / (rim_avg - bottom) * 100
    if not (min_recovery_pct_forming <= recovery_pct < p["min_recovery_pct"]):
        return None

    confidence = max(0.0, min(1.0, recovery_pct / p["min_recovery_pct"]))
    if confidence < MIN_CONFIDENCE:
        return None

    return PatternMatch(
        pattern="Bathtub (Recovering)",
        direction="bullish",
        start_date=window["Date"].iloc[0],
        end_date=window["Date"].iloc[-1],
        key_levels={"depth_pct": round(depth_pct, 1), "recovery_pct": round(recovery_pct, 1)},
        volume_confirms=False,
        confidence=round(confidence, 2),
        note=f"{round(depth_pct, 1)}% deep bowl, {round(recovery_pct, 1)}% of the way back to rim "
             f"(needs {p['min_recovery_pct']:.0f}% to confirm as a completed Bathtub/Cup)",
    )


def detect_flag_forming(df: pd.DataFrame, params: dict = None) -> PatternMatch:
    """Same pole-then-consolidation shape as detect_flag, checked
    partway through the consolidation window instead of waiting for
    the full flag_window to elapse."""
    p = {**FLAG_PARAMS, **(params or {})}
    forming_bars = max(3, p["flag_window"] // 2)
    needed = p["pole_window"] + forming_bars + p["volume_lookback"]
    if len(df) < needed:
        return None
    if detect_flag(df, params) is not None:
        return None  # already a complete flag

    flag = df.iloc[-forming_bars:]
    pole = df.iloc[-(forming_bars + p["pole_window"]):-forming_bars]
    if len(pole) < 2 or len(flag) < 2:
        return None

    pole_start, pole_end = pole["Close"].iloc[0], pole["Close"].iloc[-1]
    pole_move_pct = (pole_end - pole_start) / pole_start * 100
    if abs(pole_move_pct) < p["pole_min_move_pct"]:
        return None
    direction = "bullish" if pole_move_pct > 0 else "bearish"

    flag_range_pct = (flag["High"].max() - flag["Low"].min()) / pole_end * 100
    if flag_range_pct > p["flag_max_range_pct"]:
        return None

    confidence = max(0.0, min(1.0, forming_bars / p["flag_window"]))
    if confidence < MIN_CONFIDENCE:
        return None

    return PatternMatch(
        pattern="Flag (Forming)",
        direction=direction,
        start_date=pole["Date"].iloc[0],
        end_date=flag["Date"].iloc[-1],
        key_levels={"pole_move_pct": round(pole_move_pct, 1), "flag_range_pct": round(flag_range_pct, 1),
                    "bars_in_flag": forming_bars},
        volume_confirms=False,
        confidence=round(confidence, 2),
        note=f"{round(pole_move_pct, 1)}% pole, {forming_bars}/{p['flag_window']} days into a tight "
             f"consolidation so far ({round(flag_range_pct, 1)}% range)",
    )


FORMING_DETECTORS = [
    detect_golden_cross_forming,
    detect_breakout_forming,
    detect_bathtub_forming,
    detect_flag_forming,
]


def detect_all_forming(df: pd.DataFrame) -> list:
    matches = []
    for detect in FORMING_DETECTORS:
        try:
            m = detect(df)
        except Exception:
            continue
        if m:
            matches.append(m)
    return matches


if __name__ == "__main__":
    from modules.market_data import get_history
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    hist = get_history(ticker)
    if hist.empty:
        print(f"No cached history for {ticker}")
    else:
        matches = detect_all_forming(hist)
        if not matches:
            print("No forming signals")
        for m in matches:
            print(f"{m.pattern:<24} {m.direction:<8} confidence={m.confidence}  {m.note}")
