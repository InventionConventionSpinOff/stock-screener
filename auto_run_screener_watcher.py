# ==================================================================
#   Stock Screener  —  auto_run_screener_watcher.py
#   Checks whether your Fidelity Positions CSV has changed since the
#   last time the screener processed it; if so, runs run_screener.py.
#   Safe to run this as often as you like — it's a no-op when the CSV
#   hasn't changed. Intended to be wired up as a Windows Scheduled Task
#   polling every 30 minutes or so.
# ==================================================================
import sys
import subprocess
import datetime as dt
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from config import DIR_LOGS
from modules.notify import notify_failure
from modules.ticker_universe import resolve_portfolio_csv_path

MARKER_FILE = ROOT_DIR / ".last_processed_csv_mtime"
WATCHER_LOG = DIR_LOGS / "auto_run_screener_watcher.log"


def log(msg: str) -> None:
    line = f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line)
    with open(WATCHER_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    csv_path = resolve_portfolio_csv_path()
    if not csv_path.exists():
        log("No broker positions CSV found — nothing to do.")
        return

    current_mtime = csv_path.stat().st_mtime
    last_mtime = None
    if MARKER_FILE.exists():
        try:
            last_mtime = float(MARKER_FILE.read_text().strip())
        except ValueError:
            last_mtime = None

    if last_mtime is not None and current_mtime <= last_mtime:
        return  # CSV unchanged since last screener run — nothing to do

    log(f"New/updated positions CSV detected (mtime {current_mtime}) — running run_screener.py")
    result = subprocess.run(
        [sys.executable, str(ROOT_DIR / "run_screener.py"), "--universe", "full"],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
    )
    log(f"run_screener.py exited with code {result.returncode}")
    if result.returncode != 0:
        log("STDOUT tail:\n" + "\n".join(result.stdout.splitlines()[-20:]))
        log("STDERR tail:\n" + "\n".join(result.stderr.splitlines()[-20:]))
        stderr_tail = "\n".join(result.stderr.splitlines()[-5:]) or "See log file for details."
        notify_failure("Stock Screener", stderr_tail)
        # Do NOT update the marker on failure — retry next tick.
        return

    MARKER_FILE.write_text(str(current_mtime))
    log("Marker updated — CSV marked as processed by the screener.")


if __name__ == "__main__":
    main()
