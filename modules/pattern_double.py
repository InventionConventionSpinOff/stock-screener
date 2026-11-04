# ==================================================================
#   modules/pattern_double.py
#   Double top (bearish) / double bottom (bullish): two swing highs
#   (or lows) of similar price, separated by an opposite swing in
#   between, with the most recent close having broken the "neckline"
#   (the swing in between).
# ==================================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from modules.pattern_types import PatternMatch
from modules.swing_points import swing_sequence

PARAMS = {
    "swing_order": 5,
    "peak_tolerance_pct": 3.0,   # how close the two tops/bottoms must be to each other
    "min_separation_bars": 10,   # minimum bars between the two peaks/troughs
    "volume_confirm_mult": 1.2,
    "volume_lookback": 20,
}


def detect_double(df: pd.DataFrame, params: dict = None) -> PatternMatch:
    p = {**PARAMS, **(params or {})}
    if len(df) < p["volume_lookback"] + p["min_separation_bars"]:
        return None

    swings = swing_sequence(df, order=p["swing_order"])
    if len(swings) < 3:
        return None

    # Look at the last 3 alternating swing points: X - Y - Z, where X
    # and Z are the same type (both highs or both lows) — that's the
    # double-top/bottom shape, with Y as the neckline.
    tail = swings.tail(3).reset_index(drop=True)
    if tail.iloc[0]["Type"] != tail.iloc[2]["Type"]:
        return None

    peak_type = tail.iloc[0]["Type"]
    price_a, price_b = tail.iloc[0]["Price"], tail.iloc[2]["Price"]
    neckline_price = tail.iloc[1]["Price"]

    if tail.iloc[2]["Index"] - tail.iloc[0]["Index"] < p["min_separation_bars"]:
        return None

    similarity_pct = abs(price_a - price_b) / price_a * 100
    if similarity_pct > p["peak_tolerance_pct"]:
        return None

    last_close = df["Close"].iloc[-1]
    if peak_type == "high":
        direction = "bearish"
        broke_neckline = last_close < neckline_price
    else:
        direction = "bullish"
        broke_neckline = last_close > neckline_price

    if not broke_neckline:
        return None

    vol_baseline = df["Volume"].iloc[-(p["volume_lookback"] + 1):-1].mean()
    volume_confirms = bool(vol_baseline and df["Volume"].iloc[-1] > vol_baseline * p["volume_confirm_mult"])

    return PatternMatch(
        pattern="Double Top" if peak_type == "high" else "Double Bottom",
        direction=direction,
        start_date=tail.iloc[0]["Date"],
        end_date=df["Date"].iloc[-1],
        key_levels={
            "peak_a": round(price_a, 2),
            "peak_b": round(price_b, 2),
            "neckline": round(neckline_price, 2),
            "similarity_pct": round(similarity_pct, 2),
        },
        volume_confirms=volume_confirms,
        note=f"Neckline at {round(neckline_price, 2)} broken, peaks {round(similarity_pct, 1)}% apart",
    )


if __name__ == "__main__":
    from modules.market_data import get_history
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    hist = get_history(ticker)
    if hist.empty:
        print(f"No cached history for {ticker}")
    else:
        m = detect_double(hist)
        print(m if m else "No match")
