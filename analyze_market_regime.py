# ==================================================================
#   Stock Screener  —  analyze_market_regime.py
#   Checks whether a pattern's historical win rate/avg return differs
#   meaningfully depending on the BROADER market's trend at the time
#   the pattern fired — not the stock's own trend (that's already
#   baked into most detectors), SPY's. Bullish continuation patterns
#   (Bathtub, Cup & Handle, Golden Cross) are widely believed in
#   technical-analysis practice to work better when the tide is
#   already going out in the same direction; this has never actually
#   been checked against this project's own backtested data.
#
#   Regime definition: "Bull" if SPY's close is above its own 200-day
#   moving average on that date, "Bear" otherwise — the standard,
#   widely-used broad-market trend heuristic (the same mechanism
#   modules/pattern_golden_cross.py already applies to individual
#   stocks, just applied to the index itself).
#
#   Same discipline as tune_pattern_thresholds.py and
#   train_recommendation_model.py: for each pattern, within EACH
#   regime, a chronological 80/20 split checks whether a regime effect
#   seen in the early data actually reproduces on later, unseen data —
#   not just something that looks good on the one period used to find
#   it. This script reports the evidence; it does not change any
#   detector or recommendation logic.
#
#   Usage:
#       python analyze_market_regime.py
#       python analyze_market_regime.py --pattern Bathtub
#
#   Needs:
#     - cache/backtest_matches.csv (run backtest_patterns.py first)
#     - cache/ohlcv/SPY.csv (run modules/market_data.refresh_all_history(['SPY'])
#       once if missing — this script does NOT fetch it itself, to
#       avoid a network call inside what's meant to be a fast,
#       offline analysis pass)
# ==================================================================
import argparse
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from config import PATHS
from modules.market_data import get_regime_lookup

HORIZON_DAYS = 20

MIN_BUCKET_SAMPLE = 100
MIN_WIN_RATE_EFFECT_PP = 5.0
MIN_AVG_RETURN_EFFECT_PP = 1.0

PATTERNS = ["Bathtub", "Cup & Handle", "Breakout", "Golden Cross",
            "Death Cross", "Flag", "Overextended"]


def load_matches_with_regime() -> pd.DataFrame:
    path = PATHS["backtest_matches_cache"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run backtest_patterns.py first.")
    df = pd.read_csv(path, parse_dates=["Date"])

    regime = get_regime_lookup()
    if regime.empty:
        raise FileNotFoundError(
            "cache/ohlcv/SPY.csv not found — run once: "
            "python -c \"from modules.market_data import refresh_all_history; "
            "refresh_all_history(['SPY'])\""
        )
    df["Regime"] = df["Date"].map(regime)
    # A match on a date outside SPY's own covered range (e.g. the
    # first 200 bars of SPY's history, consumed by the MA warm-up)
    # has no regime label — drop rather than silently mis-tag.
    before = len(df)
    df = df.dropna(subset=["Regime"])
    dropped = before - len(df)
    if dropped:
        print(f"(Dropped {dropped} match(es) outside SPY's regime-labeled date range.)")
    return df


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Chronological split — see module docstring for why this isn't
    a random train_test_split."""
    cutoff = int(len(df) * (1 - test_frac))
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def regime_report(pattern: str, df: pd.DataFrame) -> dict:
    target_col = f"SignedReturn_{HORIZON_DAYS}d_pct"
    sub = df[df["Pattern"] == pattern].sort_values("Date").reset_index(drop=True)
    if sub.empty:
        return {"pattern": pattern, "error": "no matches for this pattern"}

    rows = {}
    for regime in ("Bull", "Bear"):
        regime_sub = sub[sub["Regime"] == regime]
        if len(regime_sub) < 2 * MIN_BUCKET_SAMPLE:
            rows[regime] = {"error": f"only {len(regime_sub)} matches (need >={2 * MIN_BUCKET_SAMPLE})"}
            continue
        train, test = time_split(regime_sub)
        if len(train) < MIN_BUCKET_SAMPLE or len(test) < MIN_BUCKET_SAMPLE:
            rows[regime] = {"error": f"train n={len(train)}, test n={len(test)}, below {MIN_BUCKET_SAMPLE} floor"}
            continue
        rows[regime] = {
            "train_n": len(train), "test_n": len(test),
            "train_win_pct": round((train[target_col] > 0).mean() * 100, 1),
            "test_win_pct": round((test[target_col] > 0).mean() * 100, 1),
            "train_avg_pct": round(train[target_col].mean(), 2),
            "test_avg_pct": round(test[target_col].mean(), 2),
        }
    return {"pattern": pattern, "regimes": rows}


def print_report(report: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {report['pattern']} — Bull vs. Bear regime ({HORIZON_DAYS}d horizon)")
    print("=" * 72)
    if "error" in report:
        print(f"  SKIPPED: {report['error']}")
        return

    print(f"  {'Regime':<8} {'Train n':>8} {'TrWin%':>7} {'TrAvg%':>7}   "
          f"{'Test n':>8} {'TeWin%':>7} {'TeAvg%':>7}")
    for regime in ("Bull", "Bear"):
        r = report["regimes"].get(regime, {})
        if "error" in r:
            print(f"  {regime:<8} SKIPPED: {r['error']}")
            continue
        print(f"  {regime:<8} {r['train_n']:>8} {r['train_win_pct']:>7} {r['train_avg_pct']:>7}   "
              f"{r['test_n']:>8} {r['test_win_pct']:>7} {r['test_avg_pct']:>7}")


def evaluate_decision(report: dict) -> str:
    if "error" in report:
        return f"NO EFFECT — {report['error']}"
    bull, bear = report["regimes"].get("Bull"), report["regimes"].get("Bear")
    if not bull or not bear or "error" in bull or "error" in bear:
        missing = [name for name, r in [("Bull", bull), ("Bear", bear)]
                   if not r or "error" in r]
        return f"NO EFFECT — insufficient sample in {', '.join(missing)} regime"

    train_effect = bull["train_win_pct"] - bear["train_win_pct"]
    test_effect = bull["test_win_pct"] - bear["test_win_pct"]
    train_ret_effect = bull["train_avg_pct"] - bear["train_avg_pct"]
    test_ret_effect = bull["test_avg_pct"] - bear["test_avg_pct"]

    # Same direction (Bull outperforming Bear, or vice versa) must
    # hold in BOTH train and test — a sign flip between the two means
    # whatever showed up in train didn't generalize.
    same_direction = (train_effect > 0) == (test_effect > 0)
    train_big_enough = abs(train_effect) >= MIN_WIN_RATE_EFFECT_PP or abs(train_ret_effect) >= MIN_AVG_RETURN_EFFECT_PP
    test_big_enough = abs(test_effect) >= MIN_WIN_RATE_EFFECT_PP or abs(test_ret_effect) >= MIN_AVG_RETURN_EFFECT_PP

    if not (same_direction and train_big_enough and test_big_enough):
        return (f"NO EFFECT — regime difference too small or inconsistent "
                 f"(train Bull-Bear {train_effect:+.1f}pp, test {test_effect:+.1f}pp)")

    better = "Bull" if test_effect > 0 else "Bear"
    return (f"REAL REGIME EFFECT — {report['pattern']} performs better in {better} markets "
            f"(train winrate gap {train_effect:+.1f}pp, test {test_effect:+.1f}pp, "
            f"reproduces on held-out data)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", choices=PATTERNS, default=None)
    args = parser.parse_args()

    df = load_matches_with_regime()
    patterns = [args.pattern] if args.pattern else PATTERNS

    for pattern in patterns:
        report = regime_report(pattern, df)
        print_report(report)
        print(f"\n  VERDICT: {evaluate_decision(report)}")


if __name__ == "__main__":
    main()
