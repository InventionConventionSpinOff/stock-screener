# ==================================================================
#   modules/pattern_golden_cross.py
#   Golden cross (bullish, 50MA crosses above 200MA) / death cross
#   (bearish, 50MA crosses below 200MA), flagged only if the cross
#   happened recently (within lookback_days) so stale crosses from
#   months ago don't keep showing up every run.
# ==================================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from modules.pattern_types import PatternMatch

PARAMS = {
    "fast_window": 50,
    "slow_window": 200,
    "lookback_days": 10,     # cross must have happened within this many bars of the end
    "volume_confirm_mult": 1.2,
    "volume_lookback": 20,
}


def detect_golden_cross(df: pd.DataFrame, params: dict = None) -> PatternMatch:
    p = {**PARAMS, **(params or {})}
    if len(df) < p["slow_window"] + p["lookback_days"]:
        return None

    df = df.copy()
    df["fast_ma"] = df["Close"].rolling(p["fast_window"]).mean()
    df["slow_ma"] = df["Close"].rolling(p["slow_window"]).mean()
    df = df.dropna(subset=["fast_ma", "slow_ma"]).reset_index(drop=True)
    if len(df) < 2:
        return None

    diff = df["fast_ma"] - df["slow_ma"]
    diffs = diff.values
    cross_idx = None
    direction = None
    for i in range(len(diffs) - 1, max(0, len(diffs) - p["lookback_days"] - 1), -1):
        if diffs[i - 1] <= 0 < diffs[i]:
            cross_idx = i
            direction = "bullish"
            break
        if diffs[i - 1] >= 0 > diffs[i]:
            cross_idx = i
            direction = "bearish"
            break

    if cross_idx is None:
        return None

    vol_recent = df["Volume"].iloc[max(0, cross_idx - p["volume_lookback"]):cross_idx].mean()
    vol_confirms = bool(df["Volume"].iloc[cross_idx] > vol_recent * p["volume_confirm_mult"]) if vol_recent else False

    return PatternMatch(
        pattern="Golden Cross" if direction == "bullish" else "Death Cross",
        direction=direction,
        start_date=df["Date"].iloc[cross_idx],
        end_date=df["Date"].iloc[-1],
        key_levels={
            "fast_ma": round(df["fast_ma"].iloc[cross_idx], 2),
            "slow_ma": round(df["slow_ma"].iloc[cross_idx], 2),
            "cross_price": round(df["Close"].iloc[cross_idx], 2),
        },
        volume_confirms=vol_confirms,
        note=f"{p['fast_window']}MA crossed {'above' if direction == 'bullish' else 'below'} "
             f"{p['slow_window']}MA {len(df) - 1 - cross_idx} bars ago",
    )


if __name__ == "__main__":
    from modules.market_data import get_history
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    hist = get_history(ticker)
    if hist.empty:
        print(f"No cached history for {ticker}")
    else:
        m = detect_golden_cross(hist)
        print(m if m else "No match")
