# ==================================================================
#   modules/positions.py
#   Turns your open Fidelity positions (avg cost basis) + that day's
#   detected chart patterns into a per-ticker Buy More / Hold / Trim /
#   Sell recommendation.
#
#   This is a heuristic, not financial advice — it combines two
#   signals:
#     1. Unrealized gain/loss % vs. your average cost basis.
#     2. Net pattern signal that day (bullish matches minus bearish
#        matches, weighted by whether each match's move already
#        confirmed with volume).
# ==================================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

PARAMS = {
    "deep_loss_pct": -15.0,     # underwater by more than this + bearish signal -> SELL
    "strong_gain_pct": 20.0,    # up more than this + bearish signal -> TRIM (take profit)
    "signal_confirm_min": 1,    # net signal strength needed to call it "bearish"/"bullish"
}


def _net_signal(matches: list) -> int:
    """+1 per bullish match, -1 per bearish match, volume-confirmed
    matches count double (more weight, since they're more reliable)."""
    score = 0
    for m in matches:
        weight = 2 if m.volume_confirms else 1
        score += weight if m.direction == "bullish" else -weight
    return score


def recommend_action(avg_cost: float, last_price: float, matches: list, params: dict = None,
                      weak_backtest_history: bool = False) -> dict:
    """Returns {"action": str, "gain_pct": float, "signal": int, "reason": str}.

    action is one of: "SELL", "TRIM", "HOLD", "BUY MORE", "WATCH"

    weak_backtest_history: True if one of this ticker's matched
    patterns has a measured losing track record (see
    modules/report_screener.py's _pattern_backtest_is_weak and
    modules/recommendation.py, which gates new-stock BUY calls the
    same way) — a bullish signal riding a pattern that's historically
    lost money shouldn't trigger buying MORE of something you already
    own any more than it should trigger a fresh BUY.
    """
    p = {**PARAMS, **(params or {})}
    gain_pct = (last_price - avg_cost) / avg_cost * 100 if avg_cost else 0.0
    signal = _net_signal(matches)

    if gain_pct <= p["deep_loss_pct"] and signal <= -p["signal_confirm_min"]:
        action = "SELL"
        reason = f"Down {round(gain_pct, 1)}% and pattern signal is bearish — cut the loss"
    elif gain_pct >= p["strong_gain_pct"] and signal <= -p["signal_confirm_min"]:
        action = "TRIM"
        reason = f"Up {round(gain_pct, 1)}% but pattern signal has turned bearish — consider taking profit"
    elif signal >= p["signal_confirm_min"] and gain_pct >= 0:
        if weak_backtest_history:
            action = "HOLD"
            reason = (f"Up {round(gain_pct, 1)}% with a bullish signal, but that pattern has a measured "
                       f"losing track record in backtesting — hold rather than add")
        else:
            action = "BUY MORE"
            reason = f"Up {round(gain_pct, 1)}% with a bullish pattern signal — trend intact"
    elif signal >= p["signal_confirm_min"] and gain_pct < 0:
        if weak_backtest_history:
            action = "WATCH"
            reason = (f"Down {round(gain_pct, 1)}% — bullish signal present, but it's a historically "
                       f"losing pattern, so don't count on it alone to recover")
        else:
            action = "HOLD"
            reason = f"Down {round(gain_pct, 1)}% but pattern signal has turned bullish — give it room"
    elif signal <= -p["signal_confirm_min"]:
        action = "WATCH"
        reason = f"Bearish pattern signal but not yet at the loss/gain threshold — watch closely"
    else:
        action = "HOLD"
        reason = "No strong pattern signal either way — no action needed"

    return {
        "action": action,
        "gain_pct": round(gain_pct, 2),
        "signal": signal,
        "reason": reason,
    }


def build_positions_report(positions_df: pd.DataFrame, matches_by_ticker: dict,
                            pattern_stats_df: pd.DataFrame = None, notes: dict = None,
                            forming_by_ticker: dict = None, regime_stats_df: pd.DataFrame = None,
                            current_regime: str = None, sector_stats_df: pd.DataFrame = None) -> pd.DataFrame:
    """positions_df: Ticker/Quantity/AvgCost/LastPrice (from
    modules.ticker_universe.get_open_positions()).
    matches_by_ticker: {ticker: [PatternMatch, ...]}
    pattern_stats_df: modules.backtest.load_pattern_stats_cache()'s
    output, used to gate BUY MORE the same way
    modules.recommendation.recommend_buy gates new BUY calls.
    regime_stats_df, current_regime: modules.backtest.load_pattern_stats_by_regime_cache()
    and modules.market_data.get_current_regime() — when both are
    available, the weak-pattern gate below checks today's Bull/Bear-
    specific track record instead of the blended one for the patterns
    with a validated regime effect (see
    modules.backtest.pattern_backtest_is_weak_now).
    sector_stats_df: modules.backtest.load_pattern_stats_by_sector_cache() —
    same idea as regime_stats_df, but conditions on this ticker's own
    GICS sector instead of the market-wide regime, for Death Cross
    specifically (the only pattern with a validated sector effect).
    notes: {ticker: note_text} carried forward from the previous
    report by modules.notes.load_previous_notes() — a free-text note
    you wrote on this position (e.g. "hold until Cup & Handle
    finishes").
    forming_by_ticker: {ticker: [FormingMatch, ...]} for today's
    still-in-progress patterns, so the note check below can see "X is
    forming" as well as "X completed."
    """
    from modules.backtest import ticker_has_weak_backtest_pattern_now

    notes = notes or {}
    forming_by_ticker = forming_by_ticker or {}

    ticker_sector_map = {}
    if sector_stats_df is not None and not sector_stats_df.empty and not positions_df.empty:
        from modules.sector_exposure import get_sectors
        ticker_sector_map = get_sectors(positions_df["Ticker"].tolist())

    # One cheap health check per report, not per note — if Ollama
    # isn't up, every note still displays and carries forward; it
    # just skips straight to no alert instead of trying and failing
    # once per held position with a note.
    ollama_up = False
    if notes:
        from modules.note_analysis import ollama_available
        ollama_up = ollama_available()

    rows = []
    for _, pos in positions_df.iterrows():
        ticker = pos["Ticker"]
        last_price = pos.get("LastPrice")
        avg_cost = pos.get("AvgCost")
        if pd.isna(last_price) or pd.isna(avg_cost):
            continue
        matches = matches_by_ticker.get(ticker, [])
        weak = ticker_has_weak_backtest_pattern_now(matches, pattern_stats_df, regime_stats_df, current_regime,
                                                      sector_stats_df, ticker_sector_map.get(ticker)) \
            if pattern_stats_df is not None else False
        rec = recommend_action(avg_cost, last_price, matches, weak_backtest_history=weak)

        note_text = notes.get(ticker, "")
        note_alert = ""
        if note_text:
            completed_patterns = sorted({m.pattern for m in matches})
            forming_patterns = sorted({fm.pattern for fm in forming_by_ticker.get(ticker, [])})

            # Deterministic, zero-hallucination safety net: if a pattern
            # name the note literally mentions is one that completed
            # today, that's a MET condition no matter what the LLM
            # below says. Added after a live test showed the same
            # clear-cut "pattern named in note + that exact pattern
            # completed today" input non-deterministically flip from a
            # correct MET to a wrong NOT_MET on an identical rerun — a
            # small local model answering this freeform isn't reliable
            # enough on its own, and a false NOT_MET silently suppresses
            # an alert that should have fired, which is the one failure
            # mode worth actively guarding against.
            note_lower = note_text.lower()
            mentioned_completed = [p for p in completed_patterns if p.lower() in note_lower]
            if mentioned_completed:
                note_alert = f"Looks done -- {', '.join(mentioned_completed)} completed today, matches your note"
            elif ollama_up:
                from modules.note_analysis import check_note_condition
                result = check_note_condition(ticker, note_text, rec["action"], rec["reason"],
                                               rec["gain_pct"], completed_patterns, forming_patterns)
                if result["status"] == "MET":
                    note_alert = f"AI read (unconfirmed) -- {result['explanation']}"
                elif result["status"] == "UNSURE":
                    note_alert = f"? {result['explanation']}"
                # NOT_MET / UNAVAILABLE: nothing worth flagging, leave blank.

        rows.append({
            "Ticker": ticker,
            "Quantity": pos.get("Quantity"),
            "AvgCost": round(avg_cost, 2),
            "LastPrice": round(last_price, 2),
            "GainPct": rec["gain_pct"],
            "Signal": rec["signal"],
            "Action": rec["action"],
            "Reason": rec["reason"],
            "Patterns": ", ".join(m.pattern for m in matches) if matches else "",
            "WeakBacktestPattern": weak,
            "My Note": note_text,
            "Note Alert": note_alert,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        action_order = {"SELL": 0, "TRIM": 1, "WATCH": 2, "HOLD": 3, "BUY MORE": 4}
        df["_sort"] = df["Action"].map(action_order).fillna(5)
        df = df.sort_values(["_sort", "GainPct"]).drop(columns="_sort").reset_index(drop=True)
    return df


if __name__ == "__main__":
    from modules.pattern_types import PatternMatch
    fake_matches = [PatternMatch(pattern="Golden Cross", direction="bullish", start_date=None, end_date=None, volume_confirms=True)]
    print(recommend_action(avg_cost=100, last_price=95, matches=fake_matches))
    print(recommend_action(avg_cost=100, last_price=130, matches=[PatternMatch(pattern="Double Top", direction="bearish", start_date=None, end_date=None, volume_confirms=True)]))
