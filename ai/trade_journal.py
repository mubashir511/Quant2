"""Trade Journal — an append-only, per-trade lifecycle record spanning
every real touch a symbol's own trade idea gets: from the moment Mega
Session first proposes it, through every Clerk check/action along the
way, to however it ultimately resolves (filled and closed, rejected at
the order, or superseded/expired before ever filling).

Direct user request 2026-09-19: "note down all the reasoning with which
the mega session selected that position... then record all the clerk
activity with that position throughout its life cycle... then record
how that trade ends, win or loss, why" — explicitly including
suggestions that never get filled at all ("EVERY trade suggestion...
whether it was filled at MT5 or not"). That's why the FIRST event
(`proposed`) is written the moment Mega Session's own suggestion is
parsed (see record_proposals, called from ai.ftmo_suggest.py's
_write_latest_suggestion), not when Clerk successfully executes an
order — a rejected or never-triggered suggestion would never reach that
later point at all, and would otherwise be invisible to this journal
entirely.

Storage: one JSON file per trade story under config.TRADE_JOURNAL_DIR
(records/ftmo_trade_journal/, same local/gitignored/account-independent
convention as ai.curiosity's own CURIOSITY_RECORDS_DIR) — the single
source of truth. The Obsidian vault note for the same story
(obsidian_vault/Trades/Lifecycle/) is a deterministic RENDER of this
JSON, fully regenerated on every append — never hand-edited, never
parsed back for logic, so the two can never drift apart. Deliberately a
NEW, separate note location from the existing obsidian_vault/Trades/
*.md files ai.clerk_execution._export_closed_trade_notes already
writes — this module runs ALONGSIDE that mechanism, not in place of it;
see this module's own "never puncture existing logic" contract below.

Every public function here is a best-effort side effect, the same
"cosmetic, must never affect a real trading decision" contract already
established for ai.clerk_execution._export_closed_trade_notes and
ai.researcher._export_research_note — direct user instruction this time
too ("do not puncture any existing logic in the mega session or
clerk"). Every public function swallows its own exceptions internally
and never raises to its caller; a call site never needs its own
try/except around these. Nothing in this module changes what Mega
Session proposes or what Clerk decides to execute — it only ever
observes and records.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)

# Statuses a story can never leave once reached — a fresh Mega Session
# mentioning this symbol again is unambiguously a NEW trade idea, not a
# continuation, once the prior one has resolved.
_TERMINAL_STATUSES = frozenset({"rejected", "closed_won", "closed_lost", "closed_unknown", "expired_unfilled"})


STATUS_LABELS = {
    "proposed": "Proposed",
    "order_placed": "Order Placed",
    "rejected": "Order Rejected",
    "open": "Filled — Open",
    "closed_won": "Closed — Won",
    "closed_lost": "Closed — Lost",
    "closed_unknown": "Closed — P&L not yet matched",
    "expired_unfilled": "Expired Unfilled",
}


@dataclass
class TradeStoryEvent:
    type: str  # "proposed" | "carried_forward" | "order_result" | "clerk_check" | "tactical_action" | "filled" | "closed" | "closed_reconciled" | "superseded"
    timestamp_utc: str
    data: dict


@dataclass
class TradeStory:
    symbol: str
    story_id: str  # f"{symbol}_{first-proposed timestamp %Y-%m-%d_%H%M%S}" -- stable even before a real MT5 ticket exists
    status: str
    events: list[TradeStoryEvent] = field(default_factory=list)


def _story_dir() -> Path:
    return Path(config.TRADE_JOURNAL_DIR)


def _story_path(story_id: str) -> Path:
    return _story_dir() / f"{story_id}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_story(story: TradeStory) -> None:
    path = _story_path(story.story_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(asdict(story), indent=2), encoding="utf-8")
    import os

    os.replace(tmp_path, path)


def _load_story(path: Path) -> TradeStory | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        events = [TradeStoryEvent(**e) for e in data.get("events", [])]
        return TradeStory(symbol=data["symbol"], story_id=data["story_id"], status=data["status"], events=events)
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def list_all_stories() -> list[TradeStory]:
    """Every real trade story on record, most-recently-proposed first.
    Empty (never raises) if the directory doesn't exist yet — the normal
    "nothing recorded yet" case, same convention as every other records
    reader in this codebase.

    Real bug found live 2026-09-20 (caught on a direct "recheck your
    implementation" challenge, confirmed with a real multi-symbol test
    before fixing, not assumed): this used to sort the raw filenames
    themselves (`sorted(glob(...), reverse=True)`), which sorts by
    SYMBOL NAME first — filenames are "{symbol}_{timestamp}.json" — so
    across different symbols the order was reverse-alphabetical-by-
    ticker, not remotely chronological (only correct BY COINCIDENCE
    for find_open_story's own single-symbol glob, where every candidate
    already shares the same symbol prefix). Fixed by sorting on the
    timestamp segment of story_id alone (split on the first "_" only,
    so a symbol name is never mistaken for part of the timestamp).
    ai.trade_audit.find_unaudited_closed_stories() and this function's
    own webapp caller (app.py's Trading Journal table) both depend on
    this actually being chronological — the former for fair, oldest-
    first processing of a backlog under its own daily per-run cap, the
    latter just for a sane display order."""
    story_dir = _story_dir()
    if not story_dir.exists():
        return []
    stories = [s for p in story_dir.glob("*.json") if (s := _load_story(p)) is not None]
    stories.sort(key=lambda s: s.story_id.split("_", 1)[-1], reverse=True)
    return stories


def find_open_story(symbol: str) -> TradeStory | None:
    """The most recent NON-terminal story for this symbol, if one
    exists — "still an active trade idea" in the sense that a fresh
    Mega Session mention of this symbol should be treated as a
    continuation of it, not a brand-new proposal. None if every story
    for this symbol has already resolved (or none exists at all)."""
    story_dir = _story_dir()
    if not story_dir.exists():
        return None
    candidates = [
        s
        for p in story_dir.glob(f"{symbol}_*.json")
        if (s := _load_story(p)) is not None and s.status not in _TERMINAL_STATUSES
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda s: s.story_id)


def find_story_by_id(story_id: str) -> TradeStory | None:
    """Direct lookup by the exact id a caller already holds — for a caller that recorded/received a specific
    story_id earlier (e.g. ai.continuation_hunter, which watches a specific CLOSED story after its own win)
    and must not risk attaching to a different, newer story that happens to share the same symbol."""
    return _load_story(_story_path(story_id))


def record_continuation_watch(story_id: str, outcome: str, data: dict) -> None:
    """Continuation Watch (2026-09-28 plan, point 7): appends the checklist/model reasoning for a
    CONTINUATION candidate check to the SAME story that just closed. `outcome` is one of "no_continuation"
    (the checklist's own timeliness/move gates never passed in time), "vetoed_cost" (passed those but the
    real round-trip cost was too high), "proposed" (a fresh trade was judged worth it — LOG-ONLY today, see
    `data["log_only"]`) or "skipped" (a model looked and said no). Like record_audit, takes story_id
    directly rather than looking the story up by symbol — find_open_story() would return None here (the
    story is already terminal), and a newer story on the same symbol must never be mistaken for this one."""
    try:
        story = find_story_by_id(story_id)
        if story is None:
            return
        _append_event(story, "continuation_watch", {"outcome": outcome, **data})
    except Exception:
        logger.warning("Trade journal: record_continuation_watch failed for %s (cosmetic only).", story_id, exc_info=True)


def find_unaudited_closed_stories() -> list[TradeStory]:
    """Every real, resolved trade (status "closed_won"/"closed_lost")
    that doesn't yet have an "audit" event on it — the exact "is there
    real pending work" check ai.trade_audit.run_trade_audit_check must
    perform FIRST, before importing/calling anything that touches a
    model (direct user request: zero API calls on a day nothing new
    closed). Deliberately does NOT reuse find_open_story's own filtering
    — that function explicitly EXCLUDES every terminal status (closed_
    won/closed_lost included), by design, since it answers "should a
    fresh Mega Session mention continue this story," the opposite
    question from the one this function answers.

    OLDEST-unaudited-first, deliberately the REVERSE of list_all_
    stories()'s own newest-first order — real gap caught on a direct
    "recheck your implementation" challenge: ai.trade_audit.run_trade_
    audit_check caps how many of this list it actually processes per
    day (config.TRADE_AUDIT_MAX_STORIES_PER_RUN). Returning newest-first
    and letting the caller take the first N would mean a symbol closing
    fresh today always wins that day's slots over an older trade still
    waiting in the backlog — the older one could starve indefinitely if
    trades keep closing faster than the daily cap. Oldest-first
    guarantees the backlog actually drains in order, never starves."""
    unaudited = [
        s for s in list_all_stories() if s.status in ("closed_won", "closed_lost") and not any(e.type == "audit" for e in s.events)
    ]
    return list(reversed(unaudited))


def record_audit(story: TradeStory, block: str, audit_available: bool) -> None:
    """Appends the retrospective 3-model coaching audit for an already-
    CLOSED story. Takes the TradeStory object directly rather than a
    symbol — unlike every other record_* function in this module — since
    find_open_story() explicitly excludes every terminal status (see its
    own docstring), so looking this story up by symbol the usual way
    would always return None for exactly the stories this function
    exists to handle. The caller (ai.trade_audit) is expected to have
    obtained `story` from find_unaudited_closed_stories() moments
    earlier in the same run. Never changes story.status — the story is
    already terminal; an audit doesn't change what actually happened."""
    try:
        _append_event(story, "audit", {"block": block, "audit_available": audit_available})
    except Exception:
        logger.warning("Trade journal: record_audit failed for %s (cosmetic only).", story.story_id, exc_info=True)


def _append_event(story: TradeStory, event_type: str, data: dict, new_status: str | None = None) -> TradeStory:
    story.events.append(TradeStoryEvent(type=event_type, timestamp_utc=_now_iso(), data=data))
    if new_status is not None:
        story.status = new_status
    _save_story(story)
    _write_story_note(story)
    return story


def _first_event(story: TradeStory, types: tuple) -> TradeStoryEvent | None:
    return next((e for e in story.events if e.type in types), None)


def _last_event(story: TradeStory, types: tuple) -> TradeStoryEvent | None:
    return next((e for e in reversed(story.events) if e.type in types), None)


def summarize_story(story: TradeStory) -> dict:
    """One flat, display-ready dict per trade story — the single place
    the webapp's Trading Journal table and the Obsidian note's summary
    header both read from, so the two can never disagree about a
    trade's status, timestamps, terms or P&L. Every timestamp is the raw
    UTC ISO string (None when that lifecycle step never happened); no
    value is ever inferred or fabricated."""
    proposed = _first_event(story, ("proposed",))
    plan = proposed.data if proposed else {}
    latest_plan_event = _last_event(story, ("carried_forward", "proposed"))
    latest_plan = latest_plan_event.data if latest_plan_event else {}
    order = next(
        (
            e for e in reversed(story.events)
            if e.type == "order_result" and e.data.get("success") and e.data.get("action") in ("open", "increase")
        ),
        None,
    )
    terms = (order.data.get("terms") if order else None) or {}
    filled = _first_event(story, ("filled",))
    closed = _last_event(story, ("closed_reconciled", "closed"))
    reconciled = _last_event(story, ("closed_reconciled",))
    analysis_event = _last_event(story, ("tactical_action",)) or _last_event(story, ("clerk_check",))
    if analysis_event is not None:
        if analysis_event.type == "tactical_action":
            label = f"Clerk tactical {str(analysis_event.data.get('tier', '?')).upper()} ({'applied' if analysis_event.data.get('applied') else 'not applied'})"
        else:
            label = f"Clerk {analysis_event.data.get('kind', 'check')} ({'CONFIRMED' if analysis_event.data.get('confirmed') else 'not confirmed'})"
        analysis = {"label": label, "at": analysis_event.timestamp_utc, "text": (analysis_event.data.get("raw_text") or "").strip()}
    else:
        analysis = {"label": "", "at": None, "text": ""}
    audit = _last_event(story, ("audit",))
    return {
        "symbol": story.symbol,
        "story_id": story.story_id,
        "status": story.status,
        "status_label": STATUS_LABELS.get(story.status, story.status),
        "side": plan.get("side"),
        "proposed_at": proposed.timestamp_utc if proposed else None,
        "order_placed_at": order.timestamp_utc if order else None,
        "filled_at": filled.timestamp_utc if filled else None,
        # When Clerk's own poll noticed the close (UTC, accurate to one
        # poll) — NOT the reconcile event's own timestamp, which is merely
        # when the P&L was matched later.
        "closed_at": (_last_event(story, ("closed",)) or closed).timestamp_utc if closed else None,
        "last_activity_at": story.events[-1].timestamp_utc if story.events else None,
        "planned_price": plan.get("price"),
        "planned_stop": plan.get("stop_loss"),
        "planned_target": plan.get("take_profit"),
        "planned_pct": plan.get("pct"),
        "order_price": terms.get("price"),
        "order_stop": terms.get("stop_loss"),
        "order_target": terms.get("take_profit"),
        "order_volume": terms.get("volume"),
        "fill_price": (reconciled.data.get("open_price") if reconciled else None) or (filled.data.get("fill_price") if filled else None),
        "close_price": closed.data.get("close_price") if closed else None,
        "net_pnl": closed.data.get("net_pnl") if closed else None,
        "cause": closed.data.get("cause") if closed else None,
        "thesis": plan.get("reason"),
        "latest_thesis": latest_plan.get("reason"),
        "trigger_condition": plan.get("trigger_condition"),
        "invalidation_condition": latest_plan.get("invalidation_condition") or plan.get("invalidation_condition"),
        "analysis": analysis,
        "audit": (audit.data.get("block") if audit and audit.data.get("audit_available") else None),
        "events": len(story.events),
    }


# --- Obsidian rendering -----------------------------------------------

_EVENT_RENDERERS = {
    "proposed": lambda d: (
        f"**Proposed** — {d.get('side', '?')} {d.get('pct', '?')}% at {d.get('price', '?')}, "
        f"stop {d.get('stop_loss', '?')}, target {d.get('take_profit', '?')}"
        + (f", trigger: {d['trigger_condition']}" if d.get("trigger_condition") else "")
        + f"\n> {d.get('reason') or '(no reason given)'}"
        + (f"\n**Invalidation condition:** {d['invalidation_condition']}" if d.get("invalidation_condition") else "")
    ),
    "carried_forward": lambda d: f"**Carried forward** by a later Mega Session — {d.get('reason') or '(no reason given)'}",
    "order_result": lambda d: (
        f"**Order {d.get('action', '?')}** — {'succeeded' if d.get('success') else 'FAILED'}: {d.get('detail', '')}"
        + (
            f"\nActually sent: price {t.get('price')}, stop {t.get('stop_loss')}, "
            f"target {t.get('take_profit')}, volume {t.get('volume')}"
            + "".join(f"\n  - {g}" for g in (t.get("guards") or []))
            if (t := d.get("terms"))
            else ""
        )
    ),
    "clerk_check": lambda d: (
        f"**Clerk {d.get('kind', 'check')}** — "
        f"{'CONFIRMED' if d.get('confirmed') else 'not confirmed'}: {(d.get('raw_text') or '')[:400]}"
    ),
    "tactical_action": lambda d: (
        f"**Clerk tactical: {d.get('tier', '?').upper()}** — "
        f"{'applied' if d.get('applied') else 'not applied'}"
        + (f" ({d['skipped_reason']})" if d.get("skipped_reason") else "")
        + f"\n> {(d.get('raw_text') or '')[:400]}"
    ),
    "filled": lambda d: f"**Filled** — ticket {d.get('ticket', '?')} at {d.get('fill_price', '?')}",
    "closed": lambda d: (
        f"**Closed** — net P&L {d.get('net_pnl') if d.get('net_pnl') is not None else 'not yet matched to an MT5 deal'}, "
        f"cause: {d.get('cause', 'unknown')}"
    ),
    "closed_reconciled": lambda d: (
        f"**Closed (P&L reconciled from MT5 deal history)** — net P&L {d.get('net_pnl', '?')}, "
        f"cause: {d.get('cause', 'unknown')}, filled {d.get('open_price', '?')} -> closed {d.get('close_price', '?')}, "
        f"volume {d.get('volume', '?')}"
    ),
    "superseded": lambda d: "**Superseded** — no longer mentioned by a fresh Mega Session suggestion; treated as expired unfilled.",
    "audit": lambda d: (
        f"**Retrospective audit (3 independent models)**\n\n{d.get('block', '')}"
        if d.get("audit_available")
        else "**Retrospective audit** — no independent model was available this run; no coaching review recorded."
    ),
}


def _render_event(event: TradeStoryEvent) -> str:
    renderer = _EVENT_RENDERERS.get(event.type)
    body = renderer(event.data) if renderer else str(event.data)
    return f"### {event.timestamp_utc} — {event.type}\n{body}\n"


def _fmt(v, default="—"):
    return default if v is None or v == "" else str(v)


def format_ts(raw: str | None) -> str:
    """Journal's own UTC ISO timestamp -> "YYYY-MM-DD HH:MM:SS UTC"."""
    if not raw:
        return "—"
    try:
        return datetime.fromisoformat(raw).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return "—"


def _render_summary(story: TradeStory) -> list[str]:
    m = summarize_story(story)
    pnl = m["net_pnl"]
    lines = [
        "## Summary",
        "",
        f"- **Status:** {m['status_label']}"
        + (f" — net P&L **{pnl:+.2f}**" if isinstance(pnl, (int, float)) else "")
        + (f" (cause: {m['cause']})" if m["cause"] else ""),
        f"- **Side:** {_fmt(m['side'])}",
        f"- **Proposed:** {format_ts(m['proposed_at'])}",
        f"- **Order placed:** {format_ts(m['order_placed_at'])}",
        f"- **Filled:** {format_ts(m['filled_at'])}" + (f" at {m['fill_price']}" if m["filled_at"] and m["fill_price"] else ""),
        f"- **Closed:** {format_ts(m['closed_at'])}" + (f" at {m['close_price']}" if m["close_price"] else ""),
        f"- **Mega Session plan:** entry {_fmt(m['planned_price'])}, stop {_fmt(m['planned_stop'])}, "
        f"target {_fmt(m['planned_target'])}, {_fmt(m['planned_pct'])}% of equity at risk",
    ]
    if m["order_price"] is not None:
        lines.append(
            f"- **Order Clerk actually sent:** entry {m['order_price']}, stop {_fmt(m['order_stop'])}, "
            f"target {_fmt(m['order_target'])}, volume {_fmt(m['order_volume'])}"
        )
    if m["trigger_condition"]:
        lines.append(f"- **Trigger:** {m['trigger_condition']}")
    if m["invalidation_condition"]:
        lines.append(f"- **Invalidation:** {m['invalidation_condition']}")
    lines += ["", f"**Thesis:** {_fmt(m['thesis'], '(no reason recorded)')}"]
    if m["latest_thesis"] and m["latest_thesis"] != m["thesis"]:
        lines += ["", f"**Latest thesis (carried forward):** {m['latest_thesis']}"]
    a = m["analysis"]
    if a["text"]:
        lines += ["", f"**Latest analysis — {a['label']} at {format_ts(a['at'])}:**", f"> {a['text'][:1200]}"]
    lines += ["", "## Event log", ""]
    return lines


def _render_events(events: list[TradeStoryEvent]) -> list[str]:
    """Consecutive identical-outcome Clerk checks (e.g. 200+ "not
    confirmed" polls on a setup that simply hasn't triggered) collapse
    into one line for the NOTE only — the JSON keeps every event."""
    out: list[str] = []
    i = 0
    while i < len(events):
        e = events[i]
        if e.type == "clerk_check":
            j = i
            while (
                j + 1 < len(events)
                and events[j + 1].type == "clerk_check"
                and events[j + 1].data.get("kind") == e.data.get("kind")
                and events[j + 1].data.get("confirmed") == e.data.get("confirmed")
            ):
                j += 1
            if j > i:
                last = events[j]
                outcome = "CONFIRMED" if e.data.get("confirmed") else "not confirmed"
                out.append(
                    f"### {e.timestamp_utc} → {last.timestamp_utc} — clerk_check ×{j - i + 1}\n"
                    f"**Clerk {e.data.get('kind', 'check')}** — {j - i + 1} consecutive checks, all {outcome}. "
                    f"Most recent: {(last.data.get('raw_text') or '')[:400]}\n"
                )
                i = j + 1
                continue
        out.append(_render_event(e))
        i += 1
    return out


def render_story_note(story: TradeStory) -> str:
    """Pure formatting -> Obsidian-vault markdown for one full trade
    story. Regenerated wholesale on every append (see _write_story_note)
    rather than incrementally edited, so the note can never drift from
    the underlying JSON regardless of how it's viewed/edited elsewhere."""
    is_audited = any(e.type == "audit" for e in story.events)
    tags = f"tags: [trade, lifecycle, {story.status}]" if not is_audited else f"tags: [trade, lifecycle, {story.status}, audited]"
    lines = [
        "---",
        tags,
        "---",
        "",
        f"# {story.symbol} — {story.status.replace('_', ' ')}",
        "",
    ]
    lines += _render_summary(story)
    lines += _render_events(story.events)
    lines += [f"[[{story.symbol}]]", "[[Trade Journal]]"]
    if is_audited:
        # Without this, an audited story's retrospective coaching content
        # was only reachable indirectly (via the symbol/Trade Journal
        # links) -- real gap found 2026-09-20: unlike News and Trade
        # Journal, Trade Audit had no hub node of its own in the Obsidian
        # graph, so audited trades never visually clustered together.
        lines.append("[[Trade Audit]]")
    return "\n".join(lines)


def _write_story_note(story: TradeStory) -> None:
    try:
        vault_dir = Path(config.OBSIDIAN_VAULT_PATH) / "Trades" / "Lifecycle"
        vault_dir.mkdir(parents=True, exist_ok=True)
        path = vault_dir / f"{story.story_id}.md"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        import os

        tmp_path.write_text(render_story_note(story), encoding="utf-8")
        os.replace(tmp_path, path)
    except OSError:
        logger.warning("Trade journal: could not write vault note for %s (cosmetic only).", story.story_id, exc_info=True)


# --- Public recording API (all best-effort; never raise) --------------


def record_proposals(payload: dict) -> None:
    """Called once per fresh Mega Session suggestion (ai.ftmo_suggest.py
    ._write_latest_suggestion, right after building `payload`) — the
    ONLY place a `proposed` event is ever created, precisely so a
    suggestion that later gets rejected or never triggers still has a
    real record (see this module's own docstring for why that couldn't
    live inside Clerk's own execution loop instead).

    For every symbol still in `payload` with real exposure: an existing
    open story for it gets a `carried_forward` event (a continuation,
    not a new idea); no existing story gets a brand-new one. Any
    PREVIOUSLY open story whose symbol is no longer mentioned at all in
    this fresh payload is marked `superseded` -> "expired_unfilled" —
    Mega Session dropped it, so it's done, one way or another, from this
    journal's point of view."""
    try:
        mentioned: dict[str, dict] = {}
        for symbol, entry in payload.get("immediate_allocation", {}).items():
            # CASH is a pseudo-symbol (uninvested %), never a real trade
            # idea — same exclusion ai.clerk_execution.py already applies
            # everywhere else it iterates immediate_allocation.
            if symbol.upper() != "CASH" and entry.get("pct", 0) > 0:
                mentioned[symbol] = {**entry, "kind": "immediate"}
        for s in payload.get("pending_setups", []):
            if s["symbol"].upper() != "CASH":
                mentioned.setdefault(s["symbol"], {**s, "kind": "pending_setup"})

        for symbol, data in mentioned.items():
            existing = find_open_story(symbol)
            if existing is None:
                now = datetime.now(timezone.utc)
                story = TradeStory(
                    symbol=symbol,
                    story_id=f"{symbol}_{now:%Y-%m-%d_%H%M%S}",
                    status="proposed",
                )
                _append_event(story, "proposed", data)
            else:
                _append_event(existing, "carried_forward", data)

        # Only a story that never became a REAL, live position can be
        # "expired unfilled" by simply not being re-mentioned — "proposed"
        # (an idea Clerk hasn't acted on yet) or "order_placed" (a resting
        # order still waiting to fill). A story already at "open" is a
        # real, currently-held position: Mega Session silently reaffirming
        # an already-well-managed position without repeating its own
        # reasoning that cycle is normal, expected behavior (the exact
        # same real gap ai.clerk_execution._backfill_settlement_for_held_
        # positions exists to cover for settlement.json), NOT evidence the
        # trade is done. Real bug caught on a direct user challenge before
        # this ever ran against a real open position: the original version
        # here applied to every non-terminal status, which would have
        # mislabeled a genuinely open, real position as "expired_unfilled"
        # the first time a later session simply didn't repeat its symbol.
        _supersedable_statuses = frozenset({"proposed", "order_placed"})
        for story in list_all_stories():
            if story.status not in _supersedable_statuses or story.symbol in mentioned:
                continue
            _append_event(story, "superseded", {}, new_status="expired_unfilled")
    except Exception:
        logger.warning("Trade journal: record_proposals failed (cosmetic only, continuing).", exc_info=True)


def record_order_result(
    symbol: str, action: str, success: bool, detail: str, terms: dict | None = None
) -> None:
    """Called from Clerk's execution loop right after attempting to open/
    increase a position. Advances status to "order_placed" on success,
    or to "rejected" on a failed FIRST attempt (a still-"proposed" story
    that never even got as far as a resting order).

    `terms` (price/stop_loss/take_profit/volume) is the order Clerk
    ACTUALLY sent. Real gap found 2026-09-24: the `proposed` event holds
    Mega Session's original numbers, but Clerk's guards (M5 entry
    refinement, ATR stop floor, sizing) routinely change them before the
    order goes out -- MSFT was proposed at 494.065 and really ordered at
    498.53 with a different stop and size, and nothing in the journal
    said so."""
    try:
        story = find_open_story(symbol)
        if story is None:
            return
        new_status = None
        if action in ("open", "increase"):
            if success and story.status == "proposed":
                new_status = "order_placed"
            elif not success and story.status == "proposed":
                new_status = "rejected"
        data = {"action": action, "success": success, "detail": detail}
        if terms:
            data["terms"] = terms
        _append_event(story, "order_result", data, new_status)
    except Exception:
        logger.warning("Trade journal: record_order_result failed for %s (cosmetic only).", symbol, exc_info=True)


def record_clerk_check(symbol: str, kind: str, confirmed: bool, raw_text: str | None) -> None:
    """Called from each of Clerk's three verdict branches (pending-setup
    trigger, invalidation, and — separately, via record_tactical_action
    below — tactical). `kind` is a short label ("pending_setup_trigger" /
    "invalidation") for the rendered note, informational only."""
    try:
        story = find_open_story(symbol)
        if story is None:
            return
        _append_event(story, "clerk_check", {"kind": kind, "confirmed": confirmed, "raw_text": raw_text})
    except Exception:
        logger.warning("Trade journal: record_clerk_check failed for %s (cosmetic only).", symbol, exc_info=True)


def record_tactical_action(
    symbol: str, tier: str, applied: bool, skipped_reason: str, raw_text: str | None
) -> None:
    try:
        story = find_open_story(symbol)
        if story is None:
            return
        _append_event(
            story,
            "tactical_action",
            {"tier": tier, "applied": applied, "skipped_reason": skipped_reason, "raw_text": raw_text},
        )
    except Exception:
        logger.warning("Trade journal: record_tactical_action failed for %s (cosmetic only).", symbol, exc_info=True)


def record_filled(symbol: str, ticket: int | None, fill_price: float | None) -> None:
    try:
        story = find_open_story(symbol)
        if story is None:
            return
        _append_event(story, "filled", {"ticket": ticket, "fill_price": fill_price}, new_status="open")
    except Exception:
        logger.warning("Trade journal: record_filled failed for %s (cosmetic only).", symbol, exc_info=True)


def _status_from_pnl(net_pnl: float | None) -> str:
    if net_pnl is None:
        return "closed_unknown"
    return "closed_lost" if net_pnl < 0 else "closed_won"


def record_closed(symbol: str, net_pnl: float | None, cause: str) -> None:
    """`cause` is determined by the CALLER (ai.clerk_execution.py — it
    has the position/verdict context this module deliberately doesn't
    duplicate), e.g. "stop_loss_hit" / "take_profit_hit" /
    "clerk_tactical_exit" / "manual_or_unknown"."""
    try:
        story = find_open_story(symbol)
        if story is None:
            return
        # An unmatched deal (net_pnl None) is NOT a win -- it used to fall
        # through to "closed_won", labeling MSFT/WHEAT winners with no P&L
        # at all. "closed_unknown" is reconciled to won/lost later by
        # reconcile_unknown_closures once the real deal is found.
        new_status = _status_from_pnl(net_pnl)
        _append_event(story, "closed", {"net_pnl": net_pnl, "cause": cause}, new_status=new_status)
    except Exception:
        logger.warning("Trade journal: record_closed failed for %s (cosmetic only).", symbol, exc_info=True)


def _has_unmatched_close(story: TradeStory) -> bool:
    # Status alone isn't enough: stories closed before "closed_unknown"
    # existed were stamped "closed_won" with a null P&L (MSFT, WHEAT).
    if any(e.type == "closed_reconciled" for e in story.events):
        return False
    closed_event = next((e for e in reversed(story.events) if e.type == "closed"), None)
    return closed_event is not None and closed_event.data.get("net_pnl") is None


def has_unknown_closures() -> bool:
    return any(_has_unmatched_close(s) for s in list_all_stories())


# MT5 deal timestamps come from the broker's server clock (observed a few
# hours ahead of UTC, see data.mt5_source.get_history_deals), so matching a
# deal to a story needs a window wider than any real skew.
_RECONCILE_WINDOW = timedelta(hours=12)


def reconcile_unknown_closures(closed_trades: list, resolve_cause=None) -> list:
    """Fills in the real P&L for stories closed with no matching deal at
    the time (status "closed_unknown"), once MT5 deal history has it.
    `closed_trades` are data.mt5_source.ClosedTrade-shaped objects
    (symbol/position_id/closed_at/profit/...). A story's own filled ticket
    is matched to position_id first (in netting mode a position's id is
    its opening order's ticket); otherwise the same-symbol trade whose
    close lands nearest this story's recorded close. Returns a
    (story, matched_trade) pair per story reconciled (so the caller can
    also repair anything else that recorded the same close, e.g. the
    vault's Trades/ note). Never raises."""
    fixed: list = []
    try:
        for story in list_all_stories():
            if not _has_unmatched_close(story):
                continue
            closed_event = next((e for e in reversed(story.events) if e.type == "closed"), None)
            if closed_event is None:
                continue
            closed_at = datetime.fromisoformat(closed_event.timestamp_utc)
            filled_event = next((e for e in story.events if e.type == "filled"), None)
            ticket = filled_event.data.get("ticket") if filled_event else None
            filled_at = datetime.fromisoformat(filled_event.timestamp_utc) if filled_event else closed_at
            candidates = [
                t for t in closed_trades
                if t.symbol == story.symbol
                and filled_at - _RECONCILE_WINDOW <= t.closed_at <= closed_at + _RECONCILE_WINDOW
            ]
            match = next((t for t in candidates if ticket and t.position_id == ticket), None)
            if match is None and candidates:
                match = min(candidates, key=lambda t: abs((t.closed_at - closed_at).total_seconds()))
            if match is None:
                continue
            cause = resolve_cause(match, story) if resolve_cause else closed_event.data.get("cause", "manual_or_unknown")
            _append_event(
                story,
                "closed_reconciled",
                {
                    "net_pnl": match.profit, "cause": cause, "close_price": match.close_price,
                    "open_price": match.open_price, "volume": match.volume,
                },
                new_status=_status_from_pnl(match.profit),
            )
            fixed.append((story, match))
    except Exception:
        logger.warning("Trade journal: reconcile_unknown_closures failed (cosmetic only).", exc_info=True)
    return fixed
