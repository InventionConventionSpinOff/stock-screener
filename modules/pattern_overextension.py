# ==================================================================
#   modules/pattern_overextension.py
#   Not a "buy" pattern — a risk signal. Flags a ticker that has run
#   up hard recently (default: 100%+ over the last ~3 months) as
#   elevated pullback risk. Feeds into modules/priority.py's confluence
#   scoring (a bullish setup on an already-parabolic mover gets its
#   star rating dragged down instead of chased blindly) and, since
#   held positions run through the same detector list, into
#   modules/positions.py's Buy/Sell/Hold recommendation for real
#   money you already have in a name that's run up huge.
# ==================================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from modules.pattern_types import PatternMatch

PARAMS = {
    "lookback_bars": 60,       # ~3 trading months
    "min_gain_pct": 100.0,     # must at least double to count as "overextended"
    "volume_lookback": 20,
    "volume_confirm_mult": 1.2,
}


def detect_overextension(df: pd.DataFrame, params: dict = None) -> PatternMatch:
    p = {**PARAMS, **(params or {})}
    needed = p["lookback_bars"] + 1
    if len(df) < needed:
        return None

    start_price = df["Close"].iloc[-needed]
    end_price = df["Close"].iloc[-1]
    if start_price <= 0:
        return None

    gain_pct = (end_price - start_price) / start_price * 100
    if gain_pct < p["min_gain_pct"]:
        return None

    vol_baseline = df["Volume"].iloc[-(p["volume_lookback"] + 1):-1].mean()
    volume_confirms = bool(vol_baseline and df["Volume"].iloc[-1] > vol_baseline * p["volume_confirm_mult"])

    return PatternMatch(
        pattern="Overextended",
        direction="bearish",
        start_date=df["Date"].iloc[-needed],
        end_date=df["Date"].iloc[-1],
        key_levels={
            "gain_pct": round(gain_pct, 1),
            "lookback_bars": p["lookback_bars"],
        },
        volume_confirms=volume_confirms,
        note=f"Up {round(gain_pct, 1)}% over the last {p['lookback_bars']} trading days — elevated pullback risk",
    )


if __name__ == "__main__":
    from modules.market_data import get_history
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    hist = get_history(ticker)
    if hist.empty:
        print(f"No cached history for {ticker}")
    else:
        m = detect_overextension(hist)
        print(m if m else "No match")
