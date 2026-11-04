# ==================================================================
#   modules/pattern_breakout.py
#   Price breaks above (bullish) or below (bearish) its N-day range,
#   gated hard on a volume spike — a bare price breakout with no
#   volume behind it is too noisy to be worth surfacing at all.
# ==================================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from modules.pattern_types import PatternMatch

PARAMS = {
    "range_window": 20,       # bars used to define the prior range (excludes today)
    "volume_spike_mult": 1.8,  # HARD GATE — no match without this
    "volume_lookback": 20,
}


def detect_breakout(df: pd.DataFrame, params: dict = None) -> PatternMatch:
    p = {**PARAMS, **(params or {})}
    needed = p["range_window"] + p["volume_lookback"] + 1
    if len(df) < needed:
        return None

    today = df.iloc[-1]
    prior = df.iloc[-(p["range_window"] + 1):-1]
    prior_high = prior["High"].max()
    prior_low = prior["Low"].min()

    vol_baseline = df["Volume"].iloc[-(p["volume_lookback"] + 1):-1].mean()
    volume_confirms = bool(vol_baseline and today["Volume"] > vol_baseline * p["volume_spike_mult"])

    if not volume_confirms:
        return None  # hard gate

    if today["Close"] > prior_high:
        direction = "bullish"
        level = prior_high
    elif today["Close"] < prior_low:
        direction = "bearish"
        level = prior_low
    else:
        return None

    return PatternMatch(
        pattern="Breakout",
        direction=direction,
        start_date=prior["Date"].iloc[0],
        end_date=today["Date"],
        key_levels={
            "range_high": round(prior_high, 2),
            "range_low": round(prior_low, 2),
            "breakout_price": round(today["Close"], 2),
            "volume_mult": round(today["Volume"] / vol_baseline, 2) if vol_baseline else None,
        },
        volume_confirms=True,
        note=f"Closed {'above' if direction == 'bullish' else 'below'} "
             f"{p['range_window']}-day range on {round(today['Volume'] / vol_baseline, 1)}x avg volume",
    )


if __name__ == "__main__":
    from modules.market_data import get_history
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    hist = get_history(ticker)
    if hist.empty:
        print(f"No cached history for {ticker}")
    else:
        m = detect_breakout(hist)
        print(m if m else "No match")
