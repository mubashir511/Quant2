"""Re-hunt ledger (Rockefeller's re-entry rule, 2026-09-25 position-hunting review).

Rockefeller's trading plan has a rule for what happens AFTER a trade ends — re-enter only on a stated
condition, never on impulse. This system had none: an entry that died unfilled (price reached the target
without ever coming back, or the setup was already through its stop, or a breakout was already extended)
was simply gone, and the next Mega Session had no memory of it.

The Clerk records such a dead entry here — once per symbol per UTC day, with the drafted levels and the exact
reason — and the next Mega Session's prompt lists them as candidates for ONE re-issue: fresh levels from the
fresh data, verified by the same final live re-check as any other entry (ai.live_recheck). Nothing here
places or changes an order, and nothing here invents a level: the ledger only remembers what was drafted and
why it died. Best-effort throughout — every function swallows its own errors, so a ledger problem can never
affect trading.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def _path() -> Path:
    return Path(config.REHUNT_LEDGER_FILE)


def _load() -> dict:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(data: dict) -> None:
    path = _path()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def record_dead_entry(symbol: str, side: str, price, stop_loss, take_profit, reason: str, now_utc: datetime | None = None) -> bool:
    """Remember that `symbol`'s drafted entry died unfilled, and why. At most ONE record per symbol per UTC
    day (the first cause wins); returns True only when a new record was written. Never raises."""
    try:
        now = now_utc or datetime.now(timezone.utc)
        day = now.strftime("%Y-%m-%d")
        data = _load()
        key = f"{day}|{symbol}"
        if key in data:
            return False
        data[key] = {
            "symbol": symbol, "side": side, "price": price, "stop_loss": stop_loss, "take_profit": take_profit,
            "reason": reason, "recorded_utc": now.isoformat(),
        }
        cutoff = now - timedelta(days=config.REHUNT_LEDGER_KEEP_DAYS)
        data = {k: v for k, v in data.items() if _recorded(v) is None or _recorded(v) >= cutoff}
        _save(data)
        return True
    except Exception:  # noqa: BLE001 - a ledger problem must never touch trading
        logger.debug("re-hunt ledger: could not record %s", symbol, exc_info=True)
        return False


def _recorded(record: dict) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(record["recorded_utc"])
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (KeyError, ValueError, TypeError):
        return None


def recent_dead_entries(now_utc: datetime | None = None, max_age_hours: float | None = None) -> list[dict]:
    """Dead-entry records newer than `max_age_hours` (default config.REHUNT_MAX_AGE_HOURS), newest first."""
    try:
        now = now_utc or datetime.now(timezone.utc)
        limit = now - timedelta(hours=max_age_hours if max_age_hours is not None else config.REHUNT_MAX_AGE_HOURS)
        records = [r for r in _load().values() if (_recorded(r) or limit) > limit]
        return sorted(records, key=lambda r: r.get("recorded_utc", ""), reverse=True)
    except Exception:  # noqa: BLE001
        return []


def _fmt(value) -> str:
    return f"{value:.5g}" if isinstance(value, (int, float)) else "n/a"


def format_rehunt_block(now_utc: datetime | None = None) -> str:
    """The Mega Session prompt block: '' when nothing died recently."""
    records = recent_dead_entries(now_utc)
    if not records:
        return ""
    lines = [
        "RE-HUNT CANDIDATES (Rockefeller's re-entry rule; entries drafted by an earlier session that DIED without a "
        "fill, recorded by the Clerk with the reason). Each may be re-issued ONCE, only if its thesis still holds on "
        "TODAY's fresh data, with fresh levels chosen from today's printed structure (the drafted levels below are "
        "history, not a level to reuse) and an entry_mode that fits where the market is now — the final live re-check "
        "verifies it like any other entry. If the thesis no longer holds, leave it out and say why:"
    ]
    for r in records:
        lines.append(
            f"- {r['symbol']} {str(r.get('side', '')).upper()} (drafted entry {_fmt(r.get('price'))} / stop {_fmt(r.get('stop_loss'))} "
            f"/ target {_fmt(r.get('take_profit'))}), died {str(r.get('recorded_utc', ''))[:16].replace('T', ' ')} UTC: {r.get('reason', '')}"
        )
    return "\n".join(lines)
