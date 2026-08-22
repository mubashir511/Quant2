import concurrent.futures
from typing import Callable, TypeVar

T = TypeVar("T")


def run_with_timeout(
    fn: Callable[[], T], timeout_seconds: float, default: T, catch_exceptions: bool = True
) -> T:
    """Runs `fn` (a zero-arg callable) with a hard wall-clock timeout,
    returning `default` if it doesn't finish in time — for guarding
    against a blocking call (typically network I/O) that can hang far
    longer than any timeout it thinks it's honoring on its own. Confirmed
    live as a real failure mode, not a hypothetical: yfinance's own
    internal cookie/crumb negotiation (data/news_source.py's
    fetch_recent_headlines) silently hung well past its own per-request
    10-30s timeouts under a Windows Scheduled Task's non-interactive
    process context, which killed the unattended daily "mega market
    analysis" job for 24+ hours straight with no exception, no error log,
    and no state ever recorded — it just stopped producing output
    mid-symbol and was never heard from again until the next day's
    scheduled attempt hit the exact same wall.

    `catch_exceptions=True` (default) treats ANY failure — the timeout,
    or a genuine exception raised inside `fn` — the same way, returning
    `default` for both; right for a caller whose own contract is already
    "best-effort, empty/default on any failure" (the two fetch functions
    above). Pass `catch_exceptions=False` when the caller needs to tell
    those apart — a real exception from `fn` propagates normally in that
    case, and ONLY the timeout itself is swallowed into `default`; used
    by the unattended mega-analysis job, which records a different state
    ("timeout" vs "error", with the real exception's own message) for
    each, and would otherwise silently mislabel a genuine MT5/Claude CLI
    failure as "timed out" if this weren't distinguished.

    Uses a plain ThreadPoolExecutor rather than signal-based timeouts
    (which don't work on Windows, or off the main thread). Python has no
    way to forcibly kill a running thread, so a timed-out call's thread
    is abandoned (`shutdown(wait=False)`) rather than blocked on — it may
    keep running in the background until it eventually finishes or the
    process exits, but the caller gets `default` back immediately instead
    of hanging with it."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return executor.submit(fn).result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        return default
    except Exception:
        if catch_exceptions:
            return default
        raise
    finally:
        executor.shutdown(wait=False)
