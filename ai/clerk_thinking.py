"""The Clerk's THINKING results, kept apart from its ACTING.

Why (2026-09-26, direct user concern: "due to slow clerk speed and thinking steps the execution of trades ... should not be delayed"):
one Clerk poll used to do two very different jobs in one go - deterministic work (guards, placing/amending/closing orders,
circuit-breakers, the profit ratchet: seconds) and model work (judging free-text triggers, invalidation conditions and tactical
verdicts with a small local model: several minutes, one call after another). Everything waited for the slowest model call, and the
execution lock was held the whole time, so a failed entry, a partial close or a stop ratchet could sit for a full candle.

Now the model work runs in its own job (clerk_think_job.py, its own lock, `run_clerk_execution_check(think=True)`) and only WRITES its
verdicts here. Every acting pass (the one-minute fast lane and the regular poll, both `run_clerk_execution_check` without any model call)
reads the freshest unused verdict from this cache and acts on it in seconds. A verdict is used at most once, only while it is fresh
(config.CLERK_THINK_CACHE_TTL_MINUTES) and only for the Mega suggestion it was made for.

Pure file helpers; the verdict <-> dict conversion lives in ai.clerk_execution where the verdict types are defined.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config


def _path() -> Path:
    return Path(config.CLERK_THINK_CACHE_FILE)


def load_cache() -> dict:
    try:
        data = json.loads(_path().read_text())
    except (OSError, json.JSONDecodeError):
        return {"generated_utc": None, "last_think_completed_utc": None, "items": {}}
    if not isinstance(data, dict) or not isinstance(data.get("items"), dict):
        return {"generated_utc": None, "last_think_completed_utc": None, "items": {}}
    return data


def save_cache(cache: dict) -> None:
    path = _path()
    try:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(cache, indent=2))
        os.replace(tmp, path)
    except OSError:
        pass  # a missing cache only means the next thinking pass runs again


def _key(kind: str, symbol: str) -> str:
    return f"{kind}:{symbol}"


def for_suggestion(cache: dict, generated_utc: str | None) -> dict:
    """The cache emptied when it belongs to a different (older) Mega suggestion - a new session never inherits old verdicts."""
    if cache.get("generated_utc") != generated_utc:
        return {"generated_utc": generated_utc, "last_think_completed_utc": cache.get("last_think_completed_utc"), "items": {}}
    return cache


def store(cache: dict, kind: str, symbol: str, payload: dict, now: datetime) -> None:
    cache["items"][_key(kind, symbol)] = {**payload, "checked_utc": now.isoformat(), "consumed_utc": None}


def take_fresh(cache: dict, kind: str, symbol: str, now: datetime, ttl_minutes: float | None = None) -> dict | None:
    """The verdict for (kind, symbol) if it exists, is unused and is younger than the TTL - and marks it used. None otherwise."""
    ttl = config.CLERK_THINK_CACHE_TTL_MINUTES if ttl_minutes is None else ttl_minutes
    item = cache.get("items", {}).get(_key(kind, symbol))
    if not item or item.get("consumed_utc"):
        return None
    try:
        checked = datetime.fromisoformat(item["checked_utc"])
    except (KeyError, TypeError, ValueError):
        return None
    if now - checked > timedelta(minutes=ttl):
        return None
    item["consumed_utc"] = now.isoformat()
    return item


def is_think_due(cache: dict, suggestion_generated_utc: str | None, now: datetime, interval_minutes: float) -> bool:
    """True when the thinking pass should run: a NEW Mega suggestion has no verdicts yet (run at once), or the review interval has
    passed since the last completed thinking pass."""
    if cache.get("generated_utc") != suggestion_generated_utc:
        return True
    last = cache.get("last_think_completed_utc")
    if not last:
        return True
    try:
        return now >= datetime.fromisoformat(last) + timedelta(minutes=interval_minutes)
    except ValueError:
        return True


def mark_think_completed(cache: dict, now: datetime | None = None) -> None:
    cache["last_think_completed_utc"] = (now or datetime.now(timezone.utc)).isoformat()
