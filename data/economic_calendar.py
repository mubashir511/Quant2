"""High-impact economic-calendar events (FOMC, CPI, NFP, central-bank
speeches ...) for the intraday decision tier (2026-09-24). There was no
event-time awareness anywhere in this codebase before: an M5/M15 entry could
be placed minutes before a rate decision.

Source: the free, keyless weekly JSON calendar published at
nfs.faireconomy.media (fields: title, country [a CURRENCY code, e.g. USD],
date [ISO-8601 with UTC offset], impact [High/Medium/Low/Holiday], forecast,
previous). It is an unofficial feed — every function here is fail-soft: a
network failure, a bad payload or a missing week returns fewer/no events
(never raises, never fabricates), and the caller simply loses event
awareness rather than a trade. A shared on-disk cache (default 60 min TTL)
keeps every process (app, Clerk job, Mega job) from re-hitting the feed and
lets a stale copy bridge a short outage.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_FEED_URLS = (
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
)
_FETCH_TIMEOUT_SECONDS = 10
_STALE_CACHE_MAX_AGE = timedelta(days=8)

_FX_CODES = frozenset({"USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "CNY", "SEK", "NOK", "MXN", "ZAR", "TRY", "PLN", "HKD", "SGD", "CZK", "HUF"})

# Non-US instruments whose price is driven mainly by another currency's
# data. Everything not matched (metals, oil, agriculture, crypto, US
# equities/indices) is USD-driven by default.
_SYMBOL_CURRENCY_OVERRIDES = {
    "LVMH": {"EUR"}, "SAP": {"EUR"}, "ASML": {"EUR"}, "AIR": {"EUR"}, "BMW": {"EUR"}, "SIE": {"EUR"},
    "DE40": {"EUR"}, "GER40": {"EUR"}, "DAX": {"EUR"}, "FRA40": {"EUR"}, "EU50": {"EUR"}, "ESP35": {"EUR"},
    "UK100": {"GBP"}, "JP225": {"JPY"}, "NI225": {"JPY"}, "AUS200": {"AUD"}, "HK50": {"HKD", "CNY"},
}


@dataclass(frozen=True)
class CalendarEvent:
    title: str
    currency: str
    time_utc: datetime
    impact: str  # "High" / "Medium" / "Low" / "Holiday"
    forecast: str = ""
    previous: str = ""


@dataclass(frozen=True)
class BlackoutStatus:
    active: bool
    event: CalendarEvent | None
    minutes_to_event: float | None  # negative once the event has passed
    reason: str


def _parse_events(raw: list) -> list[CalendarEvent]:
    events: list[CalendarEvent] = []
    for item in raw:
        try:
            when = datetime.fromisoformat(item["date"]).astimezone(timezone.utc)
            events.append(
                CalendarEvent(
                    title=str(item.get("title", "")).strip(),
                    currency=str(item.get("country", "")).strip().upper(),
                    time_utc=when,
                    impact=str(item.get("impact", "")).strip().title(),
                    forecast=str(item.get("forecast", "") or ""),
                    previous=str(item.get("previous", "") or ""),
                )
            )
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
    return events


def _fetch_raw() -> list:
    raw: list = []
    for url in _FEED_URLS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, list):
                raw.extend(payload)
        except Exception as e:
            logger.info("economic_calendar: %s unavailable (%s: %s)", url, type(e).__name__, e)
    return raw


def _cache_path() -> Path:
    return Path(config.ECONOMIC_CALENDAR_CACHE_FILE)


def _read_cache() -> tuple[datetime, list] | None:
    try:
        data = json.loads(_cache_path().read_text(encoding="utf-8"))
        return datetime.fromisoformat(data["fetched_utc"]), list(data["events"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(raw: list, now: datetime) -> None:
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps({"fetched_utc": now.isoformat(), "events": raw}), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        logger.warning("economic_calendar: could not write cache (non-fatal).", exc_info=True)


def fetch_calendar_events(now: datetime | None = None) -> list[CalendarEvent]:
    """All parsed events for this and next week, from the shared cache when
    fresh, else the live feed; a stale cache bridges an outage. [] on total
    failure — never raises."""
    now = now or datetime.now(timezone.utc)
    try:
        cached = _read_cache()
        ttl = timedelta(minutes=config.ECONOMIC_CALENDAR_CACHE_MINUTES)
        if cached is not None and now - cached[0] < ttl:
            return _parse_events(cached[1])
        raw = _fetch_raw()
        if raw:
            _write_cache(raw, now)
            return _parse_events(raw)
        if cached is not None and now - cached[0] < _STALE_CACHE_MAX_AGE:
            return _parse_events(cached[1])
    except Exception:
        logger.warning("economic_calendar: fetch failed (non-fatal).", exc_info=True)
    return []


def currencies_for_symbol(symbol: str) -> set[str]:
    """Which calendar currencies move this instrument. FX pairs -> both legs;
    known non-US names -> their own currency; everything else (metals, oil,
    agriculture, crypto, US equities/indices) -> USD."""
    base = symbol.upper().split(".")[0]
    if base in _SYMBOL_CURRENCY_OVERRIDES:
        return set(_SYMBOL_CURRENCY_OVERRIDES[base])
    if len(base) == 6 and base[:3] in _FX_CODES and base[3:] in _FX_CODES:
        return {base[:3], base[3:]}
    return {"USD"}


def upcoming_events(
    symbol: str,
    now: datetime,
    hours_ahead: float = 24.0,
    impacts: tuple[str, ...] = ("High",),
    events: list[CalendarEvent] | None = None,
    hours_back: float = 0.0,
) -> list[CalendarEvent]:
    """Events for this symbol's currencies between now-hours_back and
    now+hours_ahead, soonest first."""
    events = fetch_calendar_events(now) if events is None else events
    currencies = currencies_for_symbol(symbol)
    lo, hi = now - timedelta(hours=hours_back), now + timedelta(hours=hours_ahead)
    hits = [e for e in events if e.currency in currencies and e.impact in impacts and lo <= e.time_utc <= hi]
    return sorted(hits, key=lambda e: e.time_utc)


def blackout_status(
    symbol: str, now: datetime, events: list[CalendarEvent] | None = None
) -> BlackoutStatus:
    """Inside the configured window around a High-impact event for this
    symbol's currencies (default T-30min to T+10min)? Also reports the
    nearest such event's distance for size-scaling/advisory use."""
    before = timedelta(minutes=config.EVENT_BLACKOUT_BEFORE_MINUTES)
    after = timedelta(minutes=config.EVENT_BLACKOUT_AFTER_MINUTES)
    nearby = upcoming_events(
        symbol, now, hours_ahead=6.0, impacts=tuple(config.EVENT_BLACKOUT_IMPACTS), events=events,
        hours_back=after.total_seconds() / 3600,
    )
    if not nearby:
        return BlackoutStatus(False, None, None, "no high-impact event nearby")
    nearest = min(nearby, key=lambda e: abs((e.time_utc - now).total_seconds()))
    minutes = (nearest.time_utc - now).total_seconds() / 60
    active = -after.total_seconds() / 60 <= minutes <= before.total_seconds() / 60
    reason = (
        f"{nearest.impact}-impact {nearest.currency} event '{nearest.title}' "
        f"{'in' if minutes >= 0 else ''} {abs(minutes):.0f} min{'' if minutes >= 0 else ' ago'} "
        f"({nearest.time_utc:%Y-%m-%d %H:%M} UTC)"
    )
    return BlackoutStatus(active, nearest, minutes, reason)


def format_calendar_block(now: datetime, hours_ahead: float = 24.0, events: list[CalendarEvent] | None = None) -> str:
    """Account-wide list of upcoming High-impact events (Mega Session /
    Clerk context). Empty-feed and no-event cases say so plainly — an empty
    list is never presented as 'no risk'."""
    events = fetch_calendar_events(now) if events is None else events
    if not events:
        return "Economic calendar: feed unavailable right now — event risk UNKNOWN (do not assume none)."
    hi = now + timedelta(hours=hours_ahead)
    upcoming = sorted((e for e in events if e.impact == "High" and now <= e.time_utc <= hi), key=lambda e: e.time_utc)
    if not upcoming:
        return f"Economic calendar: no High-impact events in the next {hours_ahead:.0f}h (UTC)."
    lines = [f"Economic calendar — High-impact events, next {hours_ahead:.0f}h (UTC):"]
    for e in upcoming:
        detail = f" (forecast {e.forecast}, previous {e.previous})" if e.forecast or e.previous else ""
        lines.append(f"  - {e.time_utc:%a %H:%M} {e.currency}: {e.title}{detail}")
    return "\n".join(lines)


def format_symbol_events(symbol: str, now: datetime, hours_ahead: float = 24.0, events: list[CalendarEvent] | None = None) -> str | None:
    hits = upcoming_events(symbol, now, hours_ahead=hours_ahead, events=events)
    if not hits:
        return None
    parts = [f"{e.time_utc:%a %H:%M} UTC {e.currency} {e.title}" for e in hits]
    return f"  Upcoming High-impact events for {symbol} ({'/'.join(sorted(currencies_for_symbol(symbol)))}): " + "; ".join(parts)
