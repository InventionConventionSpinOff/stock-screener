# ==================================================================
#   Stock Screener  —  analyze_sector_performance.py
#   Checks whether a pattern's historical win rate/avg return differs
#   meaningfully depending on WHICH SECTOR the matched ticker is in
#   (e.g. is Breakout better in Technology than in Utilities?) — a
#   stock's sector is static (unlike market regime, which changes day
#   to day), but detector behavior could still plausibly differ by
#   sector (volatility regime, typical volume patterns, how often a
#   name gaps on news, etc.). This has never actually been checked
#   against this project's own backtested data.
#
#   Same discipline as analyze_market_regime.py (this script's direct
#   template) and tune_pattern_thresholds.py: for each pattern, within
#   each sector with adequate sample, a chronological 80/20 split
#   checks whether a sector effect seen in the early data actually
#   reproduces on later, unseen data — not just something that looks
#   good on the one period used to find it. Best/worst sector is
#   picked from TRAIN data only (never from test), specifically so the
#   decision can't cherry-pick whichever sector happened to win on the
#   held-out slice. This script reports the evidence; it does not
#   change any detector or recommendation logic.
#
#   Usage:
#       python analyze_sector_performance.py
#       python analyze_sector_performance.py --pattern Breakout
#
#   Needs:
#     - cache/backtest_matches.csv (run backtest_patterns.py first)
#     - Sector lookups for every ticker in that cache (modules/
#       sector_exposure.get_sectors) — already-cached tickers
#       (cache/universe/ticker_sectors.json) are free; anything not
#       yet cached is a live yfinance call made THE FIRST TIME this
#       runs. On a fresh fast-universe backtest (~1600 tickers) this
#       can mean a four-figure number of one-time network calls — a
#       real cost, but one that's then cached permanently.
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
from modules.sector_exposure import get_sectors

HORIZON_DAYS = 20

MIN_SECTOR_SAMPLE = 100
MIN_WIN_RATE_EFFECT_PP = 5.0
MIN_AVG_RETURN_EFFECT_PP = 1.0

PATTERNS = ["Bathtub", "Cup & Handle", "Breakout", "Golden Cross",
            "Death Cross", "Flag", "Overextended"]


def load_matches_with_sector() -> pd.DataFrame:
    path = PATHS["backtest_matches_cache"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run backtest_patterns.py first.")
    df = pd.read_csv(path, parse_dates=["Date"])

    sectors = get_sectors(sorted(df["Ticker"].unique()))
    df["Sector"] = df["Ticker"].map(sectors)
    # "Unknown" (yfinance had no sector for this ticker — ETFs, some
    # ADRs, delisted names) isn't a real sector bucket — drop rather
    # than silently lumping unrelated tickers together under one label.
    before = len(df)
    df = df[df["Sector"].notna() & (df["Sector"] != "Unknown")]
    dropped = before - len(df)
    if dropped:
        print(f"(Dropped {dropped} match(es) with no known sector.)")
    return df


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Chronological split — see module docstring for why this isn't
    a random train_test_split."""
    cutoff = int(len(df) * (1 - test_frac))
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def sector_report(pattern: str, df: pd.DataFrame) -> dict:
    target_col = f"SignedReturn_{HORIZON_DAYS}d_pct"
    sub = df[df["Pattern"] == pattern].sort_values("Date").reset_index(drop=True)
    if sub.empty:
        return {"pattern": pattern, "error": "no matches for this pattern"}

    rows = {}
    skipped = []
    for sector, sector_sub in sub.groupby("Sector"):
        if len(sector_sub) < 2 * MIN_SECTOR_SAMPLE:
            skipped.append(f"{sector} (n={len(sector_sub)})")
            continue
        train, test = time_split(sector_sub)
        if len(train) < MIN_SECTOR_SAMPLE or len(test) < MIN_SECTOR_SAMPLE:
            skipped.append(f"{sector} (train n={len(train)}, test n={len(test)})")
            continue
        rows[sector] = {
            "train_n": len(train), "test_n": len(test),
            "train_win_pct": round((train[target_col] > 0).mean() * 100, 1),
            "test_win_pct": round((test[target_col] > 0).mean() * 100, 1),
            "train_avg_pct": round(train[target_col].mean(), 2),
            "test_avg_pct": round(test[target_col].mean(), 2),
        }
    return {"pattern": pattern, "sectors": rows, "skipped": skipped}


def print_report(report: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {report['pattern']} — by sector ({HORIZON_DAYS}d horizon)")
    print("=" * 72)
    if "error" in report:
        print(f"  SKIPPED: {report['error']}")
        return

    sectors = report["sectors"]
    if not sectors:
        print("  No sector had adequate sample (>=100 matches in both train and test).")
    else:
        print(f"  {'Sector':<24} {'Train n':>8} {'TrWin%':>7} {'TrAvg%':>7}   "
              f"{'Test n':>8} {'TeWin%':>7} {'TeAvg%':>7}")
        for sector, r in sorted(sectors.items(), key=lambda kv: -kv[1]["train_win_pct"]):
            print(f"  {sector:<24} {r['train_n']:>8} {r['train_win_pct']:>7} {r['train_avg_pct']:>7}   "
                  f"{r['test_n']:>8} {r['test_win_pct']:>7} {r['test_avg_pct']:>7}")
    if report["skipped"]:
        print(f"\n  Insufficient sample, skipped: {', '.join(report['skipped'])}")


def evaluate_decision(report: dict) -> str:
    if "error" in report:
        return f"NO EFFECT — {report['error']}"
    sectors = report["sectors"]
    if len(sectors) < 2:
        return f"NO EFFECT — fewer than 2 sectors with adequate sample (have {len(sectors)})"

    # Best/worst picked from TRAIN data only — never from test — so
    # this can't cherry-pick whichever sector happened to win on the
    # held-out slice. The question is strictly "does the gap seen in
    # train reproduce on data the selection never looked at."
    by_train = sorted(sectors.items(), key=lambda kv: kv[1]["train_win_pct"])
    worst_name, worst = by_train[0]
    best_name, best = by_train[-1]
    if best_name == worst_name:
        return "NO EFFECT — no train-set spread between sectors"

    train_effect = best["train_win_pct"] - worst["train_win_pct"]
    test_effect = best["test_win_pct"] - worst["test_win_pct"]
    train_ret_effect = best["train_avg_pct"] - worst["train_avg_pct"]
    test_ret_effect = best["test_avg_pct"] - worst["test_avg_pct"]

    # Same direction (best staying better than worst) must hold in
    # BOTH train and test — a sign flip means whatever showed up in
    # train didn't generalize.
    same_direction = (train_effect > 0) == (test_effect > 0)
    train_big_enough = abs(train_effect) >= MIN_WIN_RATE_EFFECT_PP or abs(train_ret_effect) >= MIN_AVG_RETURN_EFFECT_PP
    test_big_enough = abs(test_effect) >= MIN_WIN_RATE_EFFECT_PP or abs(test_ret_effect) >= MIN_AVG_RETURN_EFFECT_PP

    if not (same_direction and train_big_enough and test_big_enough):
        return (f"NO EFFECT — {best_name} vs {worst_name} gap too small or inconsistent "
                f"(train {train_effect:+.1f}pp, test {test_effect:+.1f}pp)")

    return (f"REAL SECTOR EFFECT — {report['pattern']} performs best in {best_name}, "
            f"worst in {worst_name} (train gap {train_effect:+.1f}pp, test {test_effect:+.1f}pp, "
            f"reproduces on held-out data)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", choices=PATTERNS, default=None)
    args = parser.parse_args()

    df = load_matches_with_sector()
    patterns = [args.pattern] if args.pattern else PATTERNS

    for pattern in patterns:
        report = sector_report(pattern, df)
        print_report(report)
        print(f"\n  VERDICT: {evaluate_decision(report)}")


if __name__ == "__main__":
    main()
