# ==================================================================
#   modules/pattern_types.py
#   Shared result type every pattern detector returns.
# ==================================================================
from dataclasses import dataclass, field


@dataclass
class PatternMatch:
    pattern: str            # e.g. "Cup & Handle", "Golden Cross"
    direction: str           # "bullish" or "bearish"
    start_date: object        # pandas Timestamp
    end_date: object           # pandas Timestamp
    key_levels: dict = field(default_factory=dict)
    volume_confirms: bool = False
    note: str = ""
    ticker: str = ""          # filled in by the caller after detection
    # 0-1, how far along / how trustworthy an in-progress (not yet
    # complete) pattern looks — see modules/pattern_forming.py. Always
    # 1.0 for the regular completed-pattern detectors; only the
    # "forming" detectors set this to something less than 1.0.
    confidence: float = 1.0
