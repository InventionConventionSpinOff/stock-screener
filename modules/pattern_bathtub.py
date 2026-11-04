# ==================================================================
#   modules/pattern_bathtub.py
#   "Bathtub": a broad U-shaped bowl recovering back near its starting
#   price, with no distinct handle/pullback afterward (that's what
#   distinguishes it from a Cup & Handle — see run_screener.py's
#   step_detect_patterns, which checks cup_handle first and only
#   falls back to bathtub if no handle formed).
# ==================================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from modules.pattern_types import PatternMatch

PARAMS = {
    "lookback_bars": 90,
    "min_depth_pct": 12.0,       # bowl must dip at least this much below its rim
    "max_depth_pct": 50.0,       # too deep isn't a bowl, it's a crash
    "rim_similarity_pct": 8.0,   # left/right rim prices must be within this % of each other
    "min_recovery_pct": 85.0,    # % of the way back up from the bottom to the rim
    "volume_confirm_mult": 1.2,
    "volume_lookback": 20,
}


def detect_bathtub(df: pd.DataFrame, params: dict = None) -> PatternMatch:
    p = {**PARAMS, **(params or {})}
    if len(df) < p["lookback_bars"] + p["volume_lookback"]:
        return None

    window = df.iloc[-p["lookback_bars"]:].reset_index(drop=True)
    left_rim = window["Close"].iloc[0]
    right_rim = window["Close"].iloc[-1]
    bottom_idx = window["Low"].idxmin()
    bottom = window["Low"].iloc[bottom_idx]

    # Bottom should sit roughly in the middle third, not right at either edge.
    if not (len(window) * 0.25 < bottom_idx < len(window) * 0.75):
        return None

    rim_avg = (left_rim + right_rim) / 2
    depth_pct = (rim_avg - bottom) / rim_avg * 100
    if not (p["min_depth_pct"] <= depth_pct <= p["max_depth_pct"]):
        return None

    rim_similarity_pct = abs(left_rim - right_rim) / rim_avg * 100
    if rim_similarity_pct > p["rim_similarity_pct"]:
        return None

    recovery_pct = (right_rim - bottom) / (rim_avg - bottom) * 100
    if recovery_pct < p["min_recovery_pct"]:
        return None

    vol_baseline = df["Volume"].iloc[-(p["volume_lookback"] + 1):-1].mean()
    volume_confirms = bool(vol_baseline and df["Volume"].iloc[-1] > vol_baseline * p["volume_confirm_mult"])

    return PatternMatch(
        pattern="Bathtub",
        direction="bullish",
        start_date=window["Date"].iloc[0],
        end_date=window["Date"].iloc[-1],
        key_levels={
            "left_rim": round(left_rim, 2),
            "right_rim": round(right_rim, 2),
            "bottom": round(bottom, 2),
            "depth_pct": round(depth_pct, 1),
        },
        volume_confirms=volume_confirms,
        note=f"{round(depth_pct, 1)}% deep bowl, recovered {round(recovery_pct, 1)}% back to rim",
    )


if __name__ == "__main__":
    from modules.market_data import get_history
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    hist = get_history(ticker)
    if hist.empty:
        print(f"No cached history for {ticker}")
    else:
        m = detect_bathtub(hist)
        print(m if m else "No match")
