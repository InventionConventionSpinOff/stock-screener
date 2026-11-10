# ==================================================================
#   Stock Screener  —  train_recommendation_model.py
#   Trains a classifier on modules/backtest.py's raw match cache
#   (cache/backtest_matches.csv) to predict P(win) for a fresh pattern
#   match, using the same features available the day the pattern
#   fires: pattern type, the 0-1 "how clean is this match" Quality
#   score already used for star ratings (modules/priority.py), and
#   volume confirmation.
#
#   Tried adding the broad market's Bull/Bear regime
#   (modules.market_data.get_regime_lookup) as a 4th feature after
#   analyze_market_regime.py found real, held-out-reproducing regime
#   effects (e.g. Death Cross is far worse in Bear markets). It didn't
#   move held-out AUC at any horizon (5d/10d/20d all stayed ~0.50-0.51)
#   — the effect is real but per-pattern, and gets diluted away when
#   blended into one classifier across all 7 patterns. Reverted; the
#   regime finding is used directly in modules/report_screener.py
#   instead (per-pattern stat lookup, no ML needed) rather than forcing
#   it through this model.
#
#   News sentiment is deliberately NOT a feature yet — the sentiment
#   cache (modules/news_sentiment.py) only ever stores today's
#   snapshot, so there's no point-in-time historical sentiment to
#   train against. Add it once a real historical log exists.
#
#   Split is chronological, not random: the model trains on the
#   earlier 80% of matches by date and is evaluated on the most
#   recent 20% it has never seen, the same discipline backtest.py
#   itself uses ("as of" each day, no hindsight). A random split would
#   let the model see matches from both before AND after a given test
#   match in training, which is a subtler form of the same look-ahead
#   bias backtest.py was built to avoid.
#
#   Usage:
#       python train_recommendation_model.py                  <- 20-day horizon (default)
#       python train_recommendation_model.py --horizon 5       <- 5 or 10-day horizon instead
#
#   Needs cache/backtest_matches.csv — run backtest_patterns.py
#   (ideally --universe full, for the biggest sample) at least once.
# ==================================================================
import argparse
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import PATHS, DIR_MODEL

FEATURE_COLS = ["Pattern", "Quality", "VolumeConfirms"]

# Below this AUC, the model isn't meaningfully better than a coin flip
# on held-out data — not worth trusting for real recommendations, just
# because it technically produces a number.
MIN_USEFUL_AUC = 0.55


def load_training_data(horizon: int) -> pd.DataFrame:
    path = PATHS["backtest_matches_cache"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run backtest_patterns.py (--universe full "
            f"recommended for the biggest sample) first."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    target_col = f"SignedReturn_{horizon}d_pct"
    if target_col not in df.columns:
        raise ValueError(
            f"No '{target_col}' column in {path} — that horizon wasn't part "
            f"of the backtest run that produced this cache."
        )

    df = df.dropna(subset=FEATURE_COLS + [target_col]).sort_values("Date").reset_index(drop=True)
    df["Win"] = (df[target_col] > 0).astype(int)
    return df


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Chronological split — see module docstring for why this isn't
    a random train_test_split."""
    cutoff = int(len(df) * (1 - test_frac))
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def build_candidate_pipelines() -> dict:
    """Two candidates, kept deliberately simple given ~3 features and a
    few hundred to a few thousand rows — not enough data to justify
    anything heavier, and a heavier model here would mostly be
    overfitting noise with extra confidence. Logistic regression is
    the interpretable baseline; HistGradientBoostingClassifier can
    pick up pattern/quality interactions the linear model can't, if
    there are any real ones to find. Whichever wins on held-out AUC
    is what gets saved — decided empirically, not assumed."""
    pre = ColumnTransformer([
        ("pattern", OneHotEncoder(handle_unknown="ignore"), ["Pattern"]),
    ], remainder="passthrough")
    return {
        "logistic_regression": Pipeline([
            ("pre", pre),
            # class_weight="balanced" because some patterns (e.g. Death
            # Cross) have a real, lopsided win/loss split — without this
            # the model could get lazy and lean on the base rate instead
            # of learning per-pattern signal.
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "gradient_boosting": Pipeline([
            ("pre", pre),
            ("clf", HistGradientBoostingClassifier(max_depth=3, random_state=0)),
        ]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=20, choices=[5, 10, 20])
    args = parser.parse_args()

    print(f"Loading backtest matches for the {args.horizon}-day horizon...")
    df = load_training_data(args.horizon)
    print(f"{len(df)} total matches across {df['Pattern'].nunique()} pattern types, "
          f"{df['Date'].min().date()} -> {df['Date'].max().date()}")
    print("Per-pattern sample sizes:")
    print(df["Pattern"].value_counts().to_string())

    train_df, test_df = time_split(df)
    if len(test_df) < 20 or train_df["Win"].nunique() < 2:
        print("\nNot enough data yet to train/evaluate meaningfully — "
              "run backtest_patterns.py on a bigger universe first.")
        sys.exit(1)

    print(f"\nTrain: {len(train_df)} matches ({train_df['Date'].min().date()} -> "
          f"{train_df['Date'].max().date()})")
    print(f"Test:  {len(test_df)} matches ({test_df['Date'].min().date()} -> "
          f"{test_df['Date'].max().date()})")

    candidates = build_candidate_pipelines()
    results = {}
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON (on held-out, later-in-time matches)")
    print("=" * 60)
    for name, pipe in candidates.items():
        pipe.fit(train_df[FEATURE_COLS], train_df["Win"])
        test_probs = pipe.predict_proba(test_df[FEATURE_COLS])[:, 1]
        test_preds = (test_probs >= 0.5).astype(int)
        auc = round(roc_auc_score(test_df["Win"], test_probs), 4) if test_df["Win"].nunique() > 1 else None
        results[name] = {
            "pipeline": pipe,
            "test_probs": test_probs,
            "test_accuracy": round(accuracy_score(test_df["Win"], test_preds), 4),
            "test_auc": auc,
            "test_brier_score": round(brier_score_loss(test_df["Win"], test_probs), 4),
        }
        print(f"  {name:<22} AUC={auc}  accuracy={results[name]['test_accuracy']}  "
              f"brier={results[name]['test_brier_score']}")

    # Pick whichever actually scored higher on held-out AUC — not a
    # fixed preference for one algorithm. A None AUC (degenerate
    # single-class test set) sorts last.
    best_name = max(results, key=lambda n: results[n]["test_auc"] if results[n]["test_auc"] is not None else -1)
    best = results[best_name]
    pipe = best["pipeline"]
    print(f"\n  Best: {best_name}")

    metrics = {
        "model_type": best_name,
        "horizon_days": args.horizon,
        "feature_cols": FEATURE_COLS,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_date_range": [str(train_df["Date"].min().date()), str(train_df["Date"].max().date())],
        "test_date_range": [str(test_df["Date"].min().date()), str(test_df["Date"].max().date())],
        "test_accuracy": best["test_accuracy"],
        "test_auc": best["test_auc"],
        "test_brier_score": best["test_brier_score"],
        "naive_baseline_winrate": round(train_df["Win"].mean(), 4),
        "trained_at": pd.Timestamp.now().isoformat(),
    }

    print("\n" + "=" * 60)
    print("  FINAL MODEL EVALUATION")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:<24} {v}")

    useful = metrics["test_auc"] is not None and metrics["test_auc"] >= MIN_USEFUL_AUC
    if not useful:
        print(f"\n  NOTE: test AUC is at or near 0.5 (coin flip) — this model isn't")
        print(f"  demonstrating real predictive power on held-out data yet. It will")
        print(f"  still be saved so you can inspect/compare it, but modules/ml_recommend.py")
        print(f"  reports this via metadata so report_screener.py can show the prediction")
        print(f"  as informational only rather than treating it as trustworthy.")

    DIR_MODEL.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, PATHS["ml_model"])
    metrics["is_useful"] = useful
    with open(PATHS["ml_model_meta"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nModel saved -> {PATHS['ml_model']}")
    print(f"Metadata saved -> {PATHS['ml_model_meta']}")


if __name__ == "__main__":
    main()
