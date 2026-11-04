# ==================================================================
#   modules/notify.py
#   Windows toast notification on failure (used by the watcher script
#   when it runs headlessly and no one is watching the console).
#   Degrades gracefully to a console print if winotify isn't installed
#   or toast delivery fails for any reason — a notification failure
#   should never crash the pipeline.
# ==================================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def notify_failure(title: str, message: str) -> None:
    try:
        from winotify import Notification

        toast = Notification(
            app_id="Stock Screener",
            title=title,
            msg=message,
        )
        toast.show()
    except Exception as e:
        print(f"[notify_failure] Could not show toast ({e}). {title}: {message}")


def notify_success(title: str, message: str) -> None:
    try:
        from winotify import Notification

        toast = Notification(
            app_id="Stock Screener",
            title=title,
            msg=message,
        )
        toast.show()
    except Exception as e:
        print(f"[notify_success] Could not show toast ({e}). {title}: {message}")


def notify_urgent(title: str, message: str) -> None:
    """Same delivery as notify_success, but with duration='long' and
    its own audio cue — reserved for things that actually need
    same-day attention (see build_urgent_alerts), so it reads as
    distinct from the routine daily-summary toast."""
    try:
        from winotify import Notification, audio

        toast = Notification(
            app_id="Stock Screener",
            title=f"⚠ {title}",
            msg=message,
            duration="long",
        )
        toast.set_audio(audio.Reminder, loop=False)
        toast.show()
    except Exception as e:
        print(f"[notify_urgent] Could not show toast ({e}). {title}: {message}")


# Within this many trading days counts as "needs attention now" rather
# than "worth knowing about eventually" — short enough that missing it
# actually costs you the window to act.
URGENT_DAYS_THRESHOLD = 3


def build_urgent_alerts(forming_matches: list = None, positions_df=None, options_df=None) -> list:
    """Scans forming signals on HELD positions and soon-to-expire
    options for anything that needs same-day attention, distinct from
    the routine daily summary — a Death Cross forming in 2 days on a
    stock you own, or an option expiring in 2 days, is worth a louder
    nudge than "today's scan found 45 candidates."""
    alerts = []
    held_tickers = set(positions_df["Ticker"]) if positions_df is not None and not positions_df.empty else set()

    if forming_matches and held_tickers:
        for m in forming_matches:
            if m.ticker not in held_tickers:
                continue
            days = m.key_levels.get("days_to_cross")
            if days is not None and days <= URGENT_DAYS_THRESHOLD:
                alerts.append(f"{m.ticker} (held): {m.pattern} in ~{days:.0f} trading days — {m.note}")

    if options_df is not None and not options_df.empty:
        soon = options_df[options_df["DaysToExpiration"] <= URGENT_DAYS_THRESHOLD]
        for _, row in soon.iterrows():
            alerts.append(f"{row['Underlying']} {row['OptionType']} ${row['Strike']} expires in "
                          f"{row['DaysToExpiration']}d")

    return alerts


def notify_urgent_alerts(alerts: list) -> None:
    if not alerts:
        return
    message = "\n".join(alerts[:5])
    if len(alerts) > 5:
        message += f"\n...and {len(alerts) - 5} more (see report)"
    notify_urgent("Stock Screener — needs attention today", message)


if __name__ == "__main__":
    notify_failure("Stock Screener", "Test notification — self-test OK if you saw a toast.")
    notify_success("Stock Screener", "Test notification — self-test OK if you saw a toast.")
