# ==================================================================
#   modules/pattern_cup_handle.py
#   Cup & Handle: same broad U-shaped bowl as pattern_bathtub.py
#   (a cup is a strict superset of the bathtub shape), PLUS a small
#   pullback ("handle") in the final stretch after the right rim.
#   Checked before pattern_bathtub.py in run_screener.py so a cup
#   match is never double-counted as a plain bathtub too.
# ==================================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from modules.pattern_types import PatternMatch
from modules.pattern_bathtub import PARAMS as BATHTUB_PARAMS, detect_bathtub

PARAMS = {
    **BATHTUB_PARAMS,
    "handle_bars": 10,           # bars checked for the handle, tail end of the bowl window
    "handle_max_depth_pct": 15.0,  # handle pullback must be shallower than the cup itself
    "handle_min_depth_pct": 2.0,   # a flat non-pullback isn't a handle either
}


def detect_cup_handle(df: pd.DataFrame, params: dict = None) -> PatternMatch:
    p = {**PARAMS, **(params or {})}

    bowl_params = {k: v for k, v in p.items() if k in BATHTUB_PARAMS}
    bowl_match = detect_bathtub(df, bowl_params)
    if bowl_match is None:
        return None

    handle = df.iloc[-p["handle_bars"]:]
    if len(handle) < 3:
        return None

    handle_high = handle["High"].iloc[0]
    handle_low = handle["Low"].min()
    if handle_high <= 0:
        return None
    handle_depth_pct = (handle_high - handle_low) / handle_high * 100

    if not (p["handle_min_depth_pct"] <= handle_depth_pct <= p["handle_max_depth_pct"]):
        return None

    # Handle should be forming near the cup's right rim, not down near
    # the bottom of the cup — otherwise it's just cup noise.
    right_rim = bowl_match.key_levels["right_rim"]
    if handle_low < right_rim * (1 - p["handle_max_depth_pct"] / 100 * 1.5):
        return None

    last_close = df["Close"].iloc[-1]
    breakout = last_close > handle_high

    match = PatternMatch(
        pattern="Cup & Handle",
        direction="bullish",
        start_date=bowl_match.start_date,
        end_date=df["Date"].iloc[-1],
        key_levels={
            **bowl_match.key_levels,
            "handle_high": round(handle_high, 2),
            "handle_low": round(handle_low, 2),
            "handle_depth_pct": round(handle_depth_pct, 1),
            "breakout": breakout,
        },
        volume_confirms=bowl_match.volume_confirms,
        note=f"Cup depth {bowl_match.key_levels['depth_pct']}%, handle depth "
             f"{round(handle_depth_pct, 1)}%{' (breakout)' if breakout else ''}",
    )
    return match


if __name__ == "__main__":
    from modules.market_data import get_history
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    hist = get_history(ticker)
    if hist.empty:
        print(f"No cached history for {ticker}")
    else:
        m = detect_cup_handle(hist)
        print(m if m else "No match")
