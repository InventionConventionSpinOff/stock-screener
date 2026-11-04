# ==================================================================
#   modules/ml_recommend.py
#   Loads the model train_recommendation_model.py produces and scores
#   a fresh pattern match with it. Falls back cleanly to "no
#   prediction" (None) rather than erroring if training hasn't been
#   run yet, or if the trained model's own held-out evaluation showed
#   it isn't actually better than a coin flip (see is_useful in the
#   metadata file) — a confident-looking number from an unvalidated
#   model is worse than no number at all.
# ==================================================================
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import PATHS

_model = None
_metadata = None
_loaded = False


def _load():
    global _model, _metadata, _loaded
    if _loaded:
        return
    _loaded = True

    meta_path = PATHS["ml_model_meta"]
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                _metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            _metadata = None

    model_path = PATHS["ml_model"]
    if os.path.exists(model_path) and _metadata is not None and _metadata.get("is_useful"):
        import joblib
        _model = joblib.load(model_path)


def model_metadata() -> dict:
    """Returns the trained model's held-out evaluation metrics (AUC,
    accuracy, sample sizes, date ranges), or None if no model has been
    trained yet. Useful for surfacing "is this even any good" in the
    report rather than just a bare probability."""
    _load()
    return _metadata


def predict_win_probability(pattern: str, quality: float, volume_confirms: bool) -> float:
    """Returns a 0-1 probability, per the trained model, that this
    match's predicted direction plays out over its training horizon.
    Returns None if no usable model exists yet (not trained, or
    trained but its own held-out AUC didn't clear the "better than a
    coin flip" bar — see train_recommendation_model.py's MIN_USEFUL_AUC)."""
    _load()
    if _model is None:
        return None
    row = pd.DataFrame([{"Pattern": pattern, "Quality": quality, "VolumeConfirms": bool(volume_confirms)}])
    return float(_model.predict_proba(row)[0, 1])


if __name__ == "__main__":
    meta = model_metadata()
    print("Model metadata:", meta or "(no model trained yet)")
    if meta:
        print(predict_win_probability("Cup & Handle", 0.75, True))
        print(predict_win_probability("Death Cross", 0.6, False))
