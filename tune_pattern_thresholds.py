# ==================================================================
#   Stock Screener  —  tune_pattern_thresholds.py
#   Checks whether a SPECIFIC threshold value (e.g. "require 2.2x
#   volume instead of 1.8x" for Breakout) would have actually produced
#   better results historically — not a new guess, a check against
#   real backtested outcomes in cache/backtest_matches.csv.
#
#   For each target pattern: bucket matches into quintiles (or more,
#   see --bins) by a raw metric (Breakout/Death Cross's VolumeMult,
#   Overextension's GainPct), using a CHRONOLOGICAL 80/20 split — same
#   discipline as train_recommendation_model.py's time_split(), never
#   a random split, to avoid look-ahead bias. Bucket edges are derived
#   ONLY from the train slice, then applied as-is to the test slice —
#   "does this threshold actually generalize" needs a real answer on
#   data the bucketing itself never saw, not just a pattern that looks
#   good on the exact data used to find it.
#
#   This script reports the evidence; it does NOT auto-edit PARAMS.
#   The plan's decision rule (monotonic in train AND test, minimum
#   sample/effect size) gates whether a change is justified at all —
#   the specific new threshold value is a human judgment call informed
#   by the printed bucket table, applied by hand to the relevant
#   pattern_*.py file plus the matching modules/priority.py quality
#   function (which hardcodes the same threshold as a duplicate
#   literal, not an import — must be kept in sync manually).
#
#   Usage:
#       python tune_pattern_thresholds.py
#       python tune_pattern_thresholds.py --pattern Breakout --bins 10
#
#   Needs cache/backtest_matches.csv with the VolumeMult/GainPct
#   columns (added to modules/backtest.py this session) — run
#   backtest_patterns.py after that change, not before, or these
#   columns will be entirely NaN.
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

HORIZON_DAYS = 20

# Each target's hypothesis is "higher is better" — these are all
# one-sided hard/soft gates (volume spike, extension size) where the
# question is whether RAISING the floor would have filtered out the
# weaker matches. If a future target needed the opposite direction,
# the monotonicity check below would need a per-pattern flip.
TARGETS = {
    "Breakout": {"metric": "VolumeMult", "current_threshold": 1.8},
    "Death Cross": {"metric": "VolumeMult", "current_threshold": 1.2},
    "Overextended": {"metric": "GainPct", "current_threshold": 100.0},
}

MIN_BUCKET_SAMPLE = 100
MIN_WIN_RATE_EFFECT_PP = 5.0
MIN_AVG_RETURN_EFFECT_PP = 1.0


def load_matches() -> pd.DataFrame:
    path = PATHS["backtest_matches_cache"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run backtest_patterns.py first.")
    df = pd.read_csv(path, parse_dates=["Date"])
    target_col = f"SignedReturn_{HORIZON_DAYS}d_pct"
    if target_col not in df.columns:
        raise ValueError(f"No {target_col} column in {path}.")
    for required in ("VolumeMult", "GainPct"):
        if required not in df.columns:
            raise ValueError(
                f"No '{required}' column in {path} — this cache predates the "
                f"modules/backtest.py instrumentation added for threshold tuning. "
                f"Re-run backtest_patterns.py to regenerate it."
            )
    return df.sort_values("Date").reset_index(drop=True)


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Chronological split — see module docstring for why this isn't
    a random train_test_split."""
    cutoff = int(len(df) * (1 - test_frac))
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def _summarize_buckets(d: pd.DataFrame, target_col: str) -> list:
    rows = []
    for bucket, group in d.groupby("Bucket", observed=True):
        if len(group) == 0:
            continue
        rows.append({
            "bucket": str(bucket),
            "left": bucket.left,
            "n": len(group),
            "win_rate_pct": round((group[target_col] > 0).mean() * 100, 1),
            "avg_return_pct": round(group[target_col].mean(), 2),
        })
    return sorted(rows, key=lambda r: r["left"])


def bucket_report(pattern: str, metric: str, df: pd.DataFrame, bins: int) -> dict:
    target_col = f"SignedReturn_{HORIZON_DAYS}d_pct"
    sub = (df[df["Pattern"] == pattern]
           .dropna(subset=[metric, target_col])
           .sort_values("Date").reset_index(drop=True))
    if sub.empty:
        return {"pattern": pattern, "metric": metric, "error": "no matches with this metric populated"}

    train, test = time_split(sub)
    if len(train) < bins * MIN_BUCKET_SAMPLE or len(test) < bins * MIN_BUCKET_SAMPLE:
        return {
            "pattern": pattern, "metric": metric,
            "error": (f"not enough data for {bins} bins (train n={len(train)}, test n={len(test)}, "
                      f"need >={bins * MIN_BUCKET_SAMPLE} each)"),
        }

    # Bucket edges derived ONLY from train — the leakage-avoidance step.
    try:
        _, edges = pd.qcut(train[metric], q=bins, retbins=True, duplicates="drop")
    except ValueError as e:
        return {"pattern": pattern, "metric": metric, "error": f"could not bucket train slice: {e}"}

    # Widen the outer edges so an out-of-train-range value (in either
    # slice) lands in the nearest bucket instead of becoming NaN, then
    # apply these SAME widened edges to BOTH train and test — using
    # qcut's own (unwidened) categorical for train would give train
    # and test different bucket labels at the ends, breaking the
    # side-by-side alignment in the printed table below.
    edges = edges.copy()
    edges[0], edges[-1] = -float("inf"), float("inf")
    train = train.copy()
    train["Bucket"] = pd.cut(train[metric], bins=edges)
    test = test.copy()
    test["Bucket"] = pd.cut(test[metric], bins=edges)

    return {
        "pattern": pattern, "metric": metric,
        "train": _summarize_buckets(train, target_col),
        "test": _summarize_buckets(test, target_col),
        "train_n": len(train), "test_n": len(test),
        "current_threshold": TARGETS[pattern]["current_threshold"],
    }


def print_report(report: dict) -> None:
    print(f"\n{'=' * 78}")
    print(f"  {report['pattern']} — bucketed by {report['metric']} ({HORIZON_DAYS}d horizon)")
    print("=" * 78)
    if "error" in report:
        print(f"  SKIPPED: {report['error']}")
        return

    print(f"  Current threshold: {report['metric']} >= {report['current_threshold']}")
    print(f"  Train n={report['train_n']}, Test n={report['test_n']}\n")
    print(f"  {'Bucket (' + report['metric'] + ')':<26} {'Train n':>8} {'TrWin%':>7} {'TrAvg%':>7}   "
          f"{'Test n':>8} {'TeWin%':>7} {'TeAvg%':>7}")
    train_by_left = {r["left"]: r for r in report["train"]}
    test_by_left = {r["left"]: r for r in report["test"]}
    for left in sorted(set(train_by_left) | set(test_by_left)):
        tr = train_by_left.get(left, {"n": 0, "bucket": "-", "win_rate_pct": "-", "avg_return_pct": "-"})
        te = test_by_left.get(left, {"n": 0, "bucket": "-", "win_rate_pct": "-", "avg_return_pct": "-"})
        label = tr.get("bucket") if tr.get("bucket") != "-" else te.get("bucket", "-")
        print(f"  {label:<26} {tr['n']:>8} {tr['win_rate_pct']!s:>7} {tr['avg_return_pct']!s:>7}   "
              f"{te['n']:>8} {te['win_rate_pct']!s:>7} {te['avg_return_pct']!s:>7}")


def evaluate_decision(report: dict) -> str:
    """Macro triage verdict per the plan's decision rule — a quick
    "does this even clear the bar" check. The exact new threshold
    value (which bucket edge to adopt) is a human call from the
    printed table above, not computed here."""
    if "error" in report:
        return f"NO CHANGE — {report['error']}"

    train, test = report["train"], report["test"]
    if len(train) < 2 or len(test) < 2:
        return "NO CHANGE — fewer than 2 buckets survived"
    if any(r["n"] < MIN_BUCKET_SAMPLE for r in train) or any(r["n"] < MIN_BUCKET_SAMPLE for r in test):
        return f"NO CHANGE — a bucket has fewer than {MIN_BUCKET_SAMPLE} matches"

    def monotonic_enough(rates, max_inversions=1):
        inversions = sum(1 for i in range(1, len(rates)) if rates[i] < rates[i - 1])
        return inversions <= max_inversions

    train_rates = [r["win_rate_pct"] for r in train]
    test_rates = [r["win_rate_pct"] for r in test]
    if not monotonic_enough(train_rates) or not monotonic_enough(test_rates):
        return "NO CHANGE — win rate isn't consistently rising across buckets in both train and test"

    train_effect = train_rates[-1] - train_rates[0]
    test_effect = test_rates[-1] - test_rates[0]
    train_ret_effect = train[-1]["avg_return_pct"] - train[0]["avg_return_pct"]
    test_ret_effect = test[-1]["avg_return_pct"] - test[0]["avg_return_pct"]

    train_big_enough = train_effect >= MIN_WIN_RATE_EFFECT_PP or train_ret_effect >= MIN_AVG_RETURN_EFFECT_PP
    test_big_enough = test_effect >= MIN_WIN_RATE_EFFECT_PP or test_ret_effect >= MIN_AVG_RETURN_EFFECT_PP
    if not (train_big_enough and test_big_enough):
        return (f"NO CHANGE — effect too small (train winrate {train_effect:+.1f}pp/"
                f"avgret {train_ret_effect:+.1f}pp, test winrate {test_effect:+.1f}pp/"
                f"avgret {test_ret_effect:+.1f}pp)")

    return (f"CHANGE WORTH CONSIDERING — win rate rises {train_rates[0]:.1f}%->{train_rates[-1]:.1f}% "
            f"(train), {test_rates[0]:.1f}%->{test_rates[-1]:.1f}% (test) across {report['metric']} buckets. "
            f"Review the table above to pick a specific new threshold value.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", choices=list(TARGETS.keys()), default=None)
    parser.add_argument("--bins", type=int, default=None)
    args = parser.parse_args()

    df = load_matches()
    patterns = [args.pattern] if args.pattern else list(TARGETS.keys())

    for pattern in patterns:
        cfg = TARGETS[pattern]
        bins = args.bins if args.bins else (10 if pattern == "Breakout" else 5)
        report = bucket_report(pattern, cfg["metric"], df, bins)
        print_report(report)
        print(f"\n  VERDICT: {evaluate_decision(report)}")


if __name__ == "__main__":
    main()
