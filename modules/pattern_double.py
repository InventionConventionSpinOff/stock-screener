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
    # X must be within this many bars of today — roughly a quarter —
    # or "within 3% of some high from years ago" stops meaning
    # anything like a double top and just means "a round number
    # exists somewhere in 5 years of history," which matched ~10x
    # more often than every other detector here.
    "max_lookback_bars": 90,
    "volume_confirm_mult": 1.2,
    "volume_lookback": 20,
}


def _candidate(df: pd.DataFrame, swings: pd.DataFrame, peak_type: str, p: dict) -> dict:
    """X and Y are a CONSECUTIVE confirmed-swing pair (X the top/
    bottom, Y the trough/peak right after it) — both safely in the
    past, so confirmation lag doesn't matter for them. Y is fixed once
    picked, specifically the swing immediately following X, rather
    than "whatever the lowest/highest point since X has been" —
    recomputing Y as a global extreme would let it silently drift to
    a point AFTER the pattern's own breakout (e.g. the neckline
    "sinking" to match a later, post-break low), which isn't the
    neckline at all. Z (the second top/bottom) is deliberately NOT
    required to be its own confirmed swing: it's just "the most
    extreme price since Y", read fresh off raw price data every day.
    Requiring Z to be its own confirmed swing (order=5 bars with no
    more extreme point after it) means that by the time it would
    confirm, a newer swing has almost always already taken its place
    in swing_sequence()'s alternating output, shifting X/Y out of the
    last-N window before the neckline ever gets a chance to break —
    verified this made the original design structurally unable to
    ever fire a single match across 4 major tickers' full history.
    Scans X candidates from most recent backwards so the freshest
    valid setup wins."""
    today_index = len(df) - 1
    same_type_positions = swings.index[swings["Type"] == peak_type].tolist()
    for pos in reversed(same_type_positions):
        if pos + 1 >= len(swings):
            continue  # nothing confirmed after this X yet to serve as Y
        x = swings.iloc[pos]
        if today_index - int(x["Index"]) > p["max_lookback_bars"]:
            break  # positions are time-ordered — every older X is too far too
        y = swings.iloc[pos + 1]
        after_y = df.iloc[int(y["Index"]) + 1:]
        if after_y.empty:
            continue
        if peak_type == "high":
            z_pos = after_y["High"].values.argmax()
            price_b = after_y["High"].iloc[z_pos]
        else:
            z_pos = after_y["Low"].values.argmin()
            price_b = after_y["Low"].iloc[z_pos]
        z_index = int(y["Index"]) + 1 + z_pos

        if z_index - int(x["Index"]) < p["min_separation_bars"]:
            continue
        price_a = x["Price"]
        similarity_pct = abs(price_a - price_b) / price_a * 100
        if similarity_pct > p["peak_tolerance_pct"]:
            continue

        last_close = df["Close"].iloc[-1]
        broke_neckline = last_close < y["Price"] if peak_type == "high" else last_close > y["Price"]
        if not broke_neckline:
            continue

        # Must be a FRESH break — yesterday's close still on the
        # un-broken side of the neckline. Without this, the same
        # historical X-Y-Z setup matches on every single day price
        # happens to remain past the neckline (which, once broken, it
        # often does for weeks), turning one real event into hundreds
        # of duplicate matches. Verified this was the actual failure
        # mode once the lack-of-any-match bug above was fixed: >95% of
        # days matched instead of ~0%.
        if len(df) < 2:
            continue
        prev_close = df["Close"].iloc[-2]
        prev_broke = prev_close < y["Price"] if peak_type == "high" else prev_close > y["Price"]
        if prev_broke:
            continue

        return {
            "x": x, "neckline_price": y["Price"], "price_a": price_a,
            "price_b": price_b, "similarity_pct": similarity_pct,
        }
    return None


def detect_double(df: pd.DataFrame, params: dict = None) -> PatternMatch:
    p = {**PARAMS, **(params or {})}
    if len(df) < p["volume_lookback"] + p["min_separation_bars"]:
        return None

    swings = swing_sequence(df, order=p["swing_order"])
    if swings.empty:
        return None

    # Check double-top and double-bottom independently — in the rare
    # case both would validly complete on the same day, prefer the
    # cleaner (tighter peak-similarity) match rather than an arbitrary
    # pick.
    candidates = {}
    for peak_type in ("high", "low"):
        c = _candidate(df, swings, peak_type, p)
        if c is not None:
            candidates[peak_type] = c
    if not candidates:
        return None
    peak_type = min(candidates, key=lambda k: candidates[k]["similarity_pct"])
    c = candidates[peak_type]

    direction = "bearish" if peak_type == "high" else "bullish"
    vol_baseline = df["Volume"].iloc[-(p["volume_lookback"] + 1):-1].mean()
    volume_confirms = bool(vol_baseline and df["Volume"].iloc[-1] > vol_baseline * p["volume_confirm_mult"])

    return PatternMatch(
        pattern="Double Top" if peak_type == "high" else "Double Bottom",
        direction=direction,
        start_date=c["x"]["Date"],
        end_date=df["Date"].iloc[-1],
        key_levels={
            "peak_a": round(c["price_a"], 2),
            "peak_b": round(c["price_b"], 2),
            "neckline": round(c["neckline_price"], 2),
            "similarity_pct": round(c["similarity_pct"], 2),
        },
        volume_confirms=volume_confirms,
        note=f"Neckline at {round(c['neckline_price'], 2)} broken, peaks {round(c['similarity_pct'], 1)}% apart",
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
