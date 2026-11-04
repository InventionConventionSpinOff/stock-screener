# ==================================================================
#   modules/swing_points.py
#   Pivot (swing high/low) detection via scipy argrelextrema. Shared
#   building block for pattern_double.py, the ZigZag report sheet, and
#   any future pivot-based detector.
# ==================================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from config import SETTINGS


def find_swing_points(df: pd.DataFrame, order: int = None) -> pd.DataFrame:
    """Returns a DataFrame of swing highs/lows: columns Date, Price,
    Type ("high"/"low"), Index (positional index into df)."""
    order = order or SETTINGS["swing_order"]
    if len(df) < order * 2 + 1:
        return pd.DataFrame(columns=["Date", "Price", "Type", "Index"])

    highs_idx = argrelextrema(df["High"].values, np.greater_equal, order=order)[0]
    lows_idx = argrelextrema(df["Low"].values, np.less_equal, order=order)[0]

    rows = []
    for i in highs_idx:
        rows.append({"Date": df["Date"].iloc[i], "Price": df["High"].iloc[i], "Type": "high", "Index": i})
    for i in lows_idx:
        rows.append({"Date": df["Date"].iloc[i], "Price": df["Low"].iloc[i], "Type": "low", "Index": i})

    out = pd.DataFrame(rows).sort_values("Index").reset_index(drop=True)
    return out


def swing_sequence(df: pd.DataFrame, order: int = None, max_legs: int = None) -> pd.DataFrame:
    """Collapses find_swing_points() into a strictly alternating
    high/low sequence (zigzag legs), keeping only the most recent
    max_legs if given."""
    points = find_swing_points(df, order=order)
    if points.empty:
        return points

    alternating = [points.iloc[0]]
    for _, row in points.iloc[1:].iterrows():
        if row["Type"] == alternating[-1]["Type"]:
            # Same type back-to-back — keep the more extreme one.
            if row["Type"] == "high" and row["Price"] > alternating[-1]["Price"]:
                alternating[-1] = row
            elif row["Type"] == "low" and row["Price"] < alternating[-1]["Price"]:
                alternating[-1] = row
        else:
            alternating.append(row)

    out = pd.DataFrame(alternating).reset_index(drop=True)
    if max_legs:
        out = out.tail(max_legs).reset_index(drop=True)
    return out


if __name__ == "__main__":
    from market_data import get_history
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    hist = get_history(ticker)
    if hist.empty:
        print(f"No cached history for {ticker} — run market_data.py {ticker} first")
    else:
        print(swing_sequence(hist))
