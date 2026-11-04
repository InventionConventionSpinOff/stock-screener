# ==================================================================
#   modules/pattern_flag.py
#   Flag: a sharp directional "pole" move followed by a tight,
#   low-volatility consolidation ("flag") drifting counter to the
#   pole. Direction of the match = direction of the pole (a bullish
#   pole implies a likely continuation breakout upward).
# ==================================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from modules.pattern_types import PatternMatch

PARAMS = {
    "pole_window": 10,          # bars over which the pole move is measured
    "pole_min_move_pct": 12.0,  # minimum % move to count as a pole
    "flag_window": 10,          # bars of consolidation checked after the pole
    "flag_max_range_pct": 6.0,  # consolidation range must stay this tight (% of pole end price)
    "volume_confirm_mult": 1.5,  # pole should show above-average volume (advisory)
    "volume_lookback": 20,
}


def detect_flag(df: pd.DataFrame, params: dict = None) -> PatternMatch:
    p = {**PARAMS, **(params or {})}
    needed = p["pole_window"] + p["flag_window"] + p["volume_lookback"]
    if len(df) < needed:
        return None

    flag = df.iloc[-p["flag_window"]:]
    pole = df.iloc[-(p["flag_window"] + p["pole_window"]):-p["flag_window"]]
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

    vol_baseline = df["Volume"].iloc[-(needed):-p["flag_window"] - p["pole_window"]].mean() \
        if len(df) > needed else df["Volume"].mean()
    pole_vol = pole["Volume"].mean()
    volume_confirms = bool(vol_baseline and pole_vol > vol_baseline * p["volume_confirm_mult"])

    return PatternMatch(
        pattern="Flag",
        direction=direction,
        start_date=pole["Date"].iloc[0],
        end_date=flag["Date"].iloc[-1],
        key_levels={
            "pole_move_pct": round(pole_move_pct, 1),
            "flag_range_pct": round(flag_range_pct, 1),
            "flag_high": round(flag["High"].max(), 2),
            "flag_low": round(flag["Low"].min(), 2),
        },
        volume_confirms=volume_confirms,
        note=f"{round(pole_move_pct, 1)}% pole then {round(flag_range_pct, 1)}% tight consolidation",
    )


if __name__ == "__main__":
    from modules.market_data import get_history
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    hist = get_history(ticker)
    if hist.empty:
        print(f"No cached history for {ticker}")
    else:
        m = detect_flag(hist)
        print(m if m else "No match")
