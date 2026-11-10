# ==================================================================
#   Stock Screener  —  analyze_confluence.py
#   Checks whether a ticker having 2+ DISTINCT, AGREEING patterns
#   match on the same day actually wins more often / returns more
#   than a single pattern alone — the core assumption behind
#   modules/priority.py's star rating (more agreeing patterns = more
#   stars, conflicting bullish+bearish patterns cancel out). This has
#   never actually been checked against this project's own
#   backtested data.
#
#   Confluence definition: group cache/backtest_matches.csv rows by
#   (Ticker, Date). If every pattern in that group agrees on
#   Direction, ConfluenceCount = the number of DISTINCT patterns in
#   the group. Groups with a direction CONFLICT (one bullish pattern,
#   one bearish) are dropped entirely — that's priority.py's
#   "cancel out" case, not a form of confluence, and mixing it in
#   would muddy the comparison. Each (Ticker, Date) group collapses
#   to ONE row before bucketing — the patterns in an agreeing group
#   share the same entry price/direction/forward outcome, so keeping
#   every per-pattern row would just count the same trading outcome
#   several times and inflate the higher-confluence buckets' sample
#   size without adding any real independent evidence.
#
#   Same discipline as tune_pattern_thresholds.py and
#   analyze_market_regime.py: a chronological 80/20 split checks
#   whether an effect seen in the early data actually reproduces on
#   later, unseen data — not just something that looks good on the
#   one period used to find it. This script reports the evidence; it
#   does not change modules/priority.py's scoring.
#
#   Usage:
#       python analyze_confluence.py
#       python analyze_confluence.py --horizon 10
#
#   Needs cache/backtest_matches.csv — run backtest_patterns.py first.
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

MIN_BUCKET_SAMPLE = 100
MIN_WIN_RATE_EFFECT_PP = 5.0
MIN_AVG_RETURN_EFFECT_PP = 1.0

# 3+ patterns agreeing on the same ticker/day is rare enough that it
# needs its own bucket collapsed with anything higher, same reasoning
# as tune_pattern_thresholds.py's bucket-floor handling elsewhere.
BUCKET_LABELS = {1: "1 pattern", 2: "2 patterns"}
BUCKET_FALLBACK = "3+ patterns"


def load_matches(horizon: int) -> pd.DataFrame:
    path = PATHS["backtest_matches_cache"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run backtest_patterns.py first.")
    df = pd.read_csv(path, parse_dates=["Date"])
    target_col = f"SignedReturn_{horizon}d_pct"
    if target_col not in df.columns:
        raise ValueError(f"No '{target_col}' column in {path} — that horizon wasn't "
                          f"part of the backtest run that produced this cache.")
    return df


def build_confluence_table(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """One row per (Ticker, Date) group that unanimously agrees on
    Direction, with ConfluenceCount = distinct patterns in that
    group and Return = that group's forward outcome (averaged across
    its rows, which should already agree almost exactly since they
    share the same ticker/date/direction/entry)."""
    target_col = f"SignedReturn_{horizon}d_pct"
    sub = df.dropna(subset=["Direction", target_col])

    rows = []
    for (ticker, date), g in sub.groupby(["Ticker", "Date"]):
        if g["Direction"].nunique() > 1:
            continue  # conflicting patterns cancel out -- not confluence
        rows.append({
            "Ticker": ticker,
            "Date": date,
            "ConfluenceCount": g["Pattern"].nunique(),
            "Return": g[target_col].mean(),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Date").reset_index(drop=True)
    return out


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Chronological split -- see module docstring for why this isn't
    a random train_test_split."""
    cutoff = int(len(df) * (1 - test_frac))
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def _bucket_label(count: int) -> str:
    return BUCKET_LABELS.get(count, BUCKET_FALLBACK)


def _summarize_buckets(d: pd.DataFrame) -> list:
    order = list(BUCKET_LABELS.values()) + [BUCKET_FALLBACK]
    rows = []
    for label, group in d.groupby("Bucket", observed=True):
        if len(group) == 0:
            continue
        rows.append({
            "bucket": label,
            "n": len(group),
            "win_rate_pct": round((group["Return"] > 0).mean() * 100, 1),
            "avg_return_pct": round(group["Return"].mean(), 2),
        })
    return sorted(rows, key=lambda r: order.index(r["bucket"]))


def confluence_report(conf_df: pd.DataFrame, horizon: int) -> dict:
    if conf_df.empty:
        return {"error": "no unanimous-direction (Ticker, Date) groups found"}

    train, test = time_split(conf_df)
    if len(train) < 2 * MIN_BUCKET_SAMPLE or len(test) < MIN_BUCKET_SAMPLE:
        return {"error": f"not enough data (train n={len(train)}, test n={len(test)})"}

    train = train.copy()
    test = test.copy()
    train["Bucket"] = train["ConfluenceCount"].apply(_bucket_label)
    test["Bucket"] = test["ConfluenceCount"].apply(_bucket_label)

    return {
        "horizon": horizon,
        "train": _summarize_buckets(train),
        "test": _summarize_buckets(test),
        "train_n": len(train), "test_n": len(test),
    }


def print_report(report: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"  Confluence (agreeing-pattern count) -- {report.get('horizon')}d horizon")
    print("=" * 72)
    if "error" in report:
        print(f"  SKIPPED: {report['error']}")
        return

    print(f"  Train n={report['train_n']}, Test n={report['test_n']}\n")
    print(f"  {'Bucket':<14} {'Train n':>8} {'TrWin%':>7} {'TrAvg%':>7}   "
          f"{'Test n':>8} {'TeWin%':>7} {'TeAvg%':>7}")
    train_by_bucket = {r["bucket"]: r for r in report["train"]}
    test_by_bucket = {r["bucket"]: r for r in report["test"]}
    order = list(BUCKET_LABELS.values()) + [BUCKET_FALLBACK]
    for label in order:
        tr = train_by_bucket.get(label, {"n": 0, "win_rate_pct": "-", "avg_return_pct": "-"})
        te = test_by_bucket.get(label, {"n": 0, "win_rate_pct": "-", "avg_return_pct": "-"})
        if tr["n"] == 0 and te["n"] == 0:
            continue
        print(f"  {label:<14} {tr['n']:>8} {tr['win_rate_pct']!s:>7} {tr['avg_return_pct']!s:>7}   "
              f"{te['n']:>8} {te['win_rate_pct']!s:>7} {te['avg_return_pct']!s:>7}")


def evaluate_decision(report: dict) -> str:
    if "error" in report:
        return f"NO EFFECT -- {report['error']}"

    train, test = report["train"], report["test"]
    if len(train) < 2 or len(test) < 2:
        return "NO EFFECT -- fewer than 2 confluence buckets survived"
    if any(r["n"] < MIN_BUCKET_SAMPLE for r in train) or any(r["n"] < MIN_BUCKET_SAMPLE for r in test):
        return f"NO EFFECT -- a bucket has fewer than {MIN_BUCKET_SAMPLE} matches"

    def monotonic_enough(rates, max_inversions=1):
        inversions = sum(1 for i in range(1, len(rates)) if rates[i] < rates[i - 1])
        return inversions <= max_inversions

    train_rates = [r["win_rate_pct"] for r in train]
    test_rates = [r["win_rate_pct"] for r in test]
    if not monotonic_enough(train_rates) or not monotonic_enough(test_rates):
        return "NO EFFECT -- win rate isn't consistently rising with confluence count in both train and test"

    train_effect = train_rates[-1] - train_rates[0]
    test_effect = test_rates[-1] - test_rates[0]
    train_ret_effect = train[-1]["avg_return_pct"] - train[0]["avg_return_pct"]
    test_ret_effect = test[-1]["avg_return_pct"] - test[0]["avg_return_pct"]

    train_big_enough = train_effect >= MIN_WIN_RATE_EFFECT_PP or train_ret_effect >= MIN_AVG_RETURN_EFFECT_PP
    test_big_enough = test_effect >= MIN_WIN_RATE_EFFECT_PP or test_ret_effect >= MIN_AVG_RETURN_EFFECT_PP
    if not (train_big_enough and test_big_enough):
        return (f"NO EFFECT -- effect too small (train winrate {train_effect:+.1f}pp/"
                f"avgret {train_ret_effect:+.1f}pp, test winrate {test_effect:+.1f}pp/"
                f"avgret {test_ret_effect:+.1f}pp)")

    return (f"REAL EFFECT -- win rate rises {train_rates[0]:.1f}%->{train_rates[-1]:.1f}% (train), "
            f"{test_rates[0]:.1f}%->{test_rates[-1]:.1f}% (test) with more agreeing patterns -- "
            f"confluence reproduces on held-out data.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=20, choices=[5, 10, 20])
    args = parser.parse_args()

    df = load_matches(args.horizon)
    conf_df = build_confluence_table(df, args.horizon)
    report = confluence_report(conf_df, args.horizon)
    print_report(report)
    print(f"\n  VERDICT: {evaluate_decision(report)}")


if __name__ == "__main__":
    main()
