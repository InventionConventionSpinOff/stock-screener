# ==================================================================
#   modules/run_lock.py
#   Simple PID/timestamp lock file so two screener runs never overlap
#   (e.g. a manual run started while the scheduled watcher is mid-run).
# ==================================================================
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def acquire_lock(lock_path: Path, max_age_sec: int) -> bool:
    """Try to create the lock file. Returns True if acquired.

    If a lock file already exists but is older than max_age_sec, it's
    assumed to be left over from a crashed run and is cleared rather
    than blocking every future run forever.
    """
    lock_path = Path(lock_path)

    if lock_path.exists():
        try:
            age = time.time() - lock_path.stat().st_mtime
        except OSError:
            age = max_age_sec + 1
        if age <= max_age_sec:
            return False
        # Stale lock — clear it and proceed.
        try:
            lock_path.unlink()
        except OSError:
            pass

    try:
        with open(lock_path, "w", encoding="utf-8") as f:
            f.write(f"pid={os.getpid()} time={time.time()}")
    except OSError:
        return False

    return True


def release_lock(lock_path: Path) -> None:
    lock_path = Path(lock_path)
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


if __name__ == "__main__":
    p = Path("._test.lock")
    print("acquire:", acquire_lock(p, 3600))
    print("acquire again (should be False):", acquire_lock(p, 3600))
    release_lock(p)
    print("released; acquire again (should be True):", acquire_lock(p, 3600))
    release_lock(p)
