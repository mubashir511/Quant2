"""Trade Audit — the fourth, independent role in this app's automation
family (alongside Mega Session, Clerk, and Researcher): a once-daily,
after-close retrospective coach reviewing every CLOSED trade story in
the Trade Journal (ai/trade_journal.py) that hasn't been reviewed yet,
using the exact same 3-model concurrent audit pool ai/ftmo_suggest.py
already uses for its own forward-looking SUGGESTION audit
(ai.portfolio_suggest.build_audit_block + AUDIT_MODELS) — reused here
completely unmodified, just fed a retrospective case file instead of a
forward-looking draft, with a new retrospective-specific instruction and
focus-directive set (TRADE_AUDIT_INSTRUCTION/TRADE_AUDIT_FOCUS_GROUPS
below) instead of ai.ftmo_suggest.AUDIT_INSTRUCTION/AUDIT_FOCUS_GROUPS.

Direct user request 2026-09-20: audit every already-closed trade once,
retrospectively, splitting the review into three distinct coaching
angles run by three independent models in parallel — Mega Session's own
original decision, Clerk's real management through the position's life,
and a dedicated hunt for a specific, previously-named failure mode:
correct data or an explicit warning being available somewhere in the
record but ignored or contradicted in the final decision anyway. Per
direct user decision the same conversation ("option b"), this is
PURELY a vault/journal-side feature for now — nothing here feeds back
into a future Mega Session's own prompt; that is a separate, deferred
step, not implied or started by this file.

Kept deliberately separate from ai/trade_journal.py: that module's own
contract is "lightweight, dependency-light, cosmetic side effect,
imported from hot paths on every proposal/check/fill/close" — pulling in
build_audit_block's network/threading/retry machinery there would blur
that contract. This file owns the whole feature instead, mirroring how
ai/researcher.py, ai/clerk_execution.py, and ai/mega_analysis.py each
own one automation role.

Zero-API-calls-when-idle contract (direct user request — this must be
the FIRST thing that happens after the enabled/due gate, before any
model is ever touched): run_trade_audit_check's own first action is
ai.trade_journal.find_unaudited_closed_stories() — a cheap local
directory scan. A day with nothing newly closed since the last check
costs exactly that scan, nothing else.

Trigger timing is a DELIBERATE DEPARTURE from this codebase's own
established "fixed UTC hour/minute, accepted DST drift" convention (see
config.RESEARCHER_TRIGGER_HOUR_UTC's own comment, and config.
TRADE_AUDIT_TRIGGER_HOUR_LOCAL's) — see _ny_close_trigger_utc below."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import config
from ai.portfolio_suggest import AUDIT_MODELS, build_audit_block
from ai.trade_journal import TradeStory, find_unaudited_closed_stories, record_audit

logger = logging.getLogger(__name__)

TRADE_AUDIT_LOCK_PATH = Path(__file__).resolve().parent.parent / "trade_audit.lock"
# Comfortably past this job's own worst-case ceiling (TRADE_AUDIT_RUN_
# TIMEOUT_SECONDS, 30 min default) — same "a lock still fresher than this
# genuinely could be a real, still-running attempt" reasoning as every
# other job's own stale-lock constant in this codebase.
TRADE_AUDIT_LOCK_STALE_AFTER_SECONDS = 40 * 60

_TIMEOUT_SENTINEL = object()

_CASE_FILE_CONTEXT_STUB = (
    "This is a retrospective review of ONE already-closed FTMO trade, "
    "for coaching/learning purposes only — the position is fully "
    "resolved and nothing about this review can change its outcome. "
    "The full case file (Mega Session's original reasoning, every real "
    "Clerk check/action taken during the position's life, and the real "
    "final result) is given below as the material to audit.\n\n"
    "HOW TO READ THE NUMBERS (real misreadings in past audits): a \"pct\" "
    "field is the % of ACCOUNT EQUITY RISKED if the stop-loss is hit "
    "(0.15 means 0.15% of the account lost at the stop) — it is NOT the "
    "position's size relative to the account, so never infer an "
    "\"enormous account\" or a leveraged position from it, and never "
    "multiply it by the stop distance again. Volatility figures quoted "
    "inside a Clerk raw text (ATR, RSI, ...) belong to the timeframe that "
    "text names — a daily (D1) ATR is several times an H1 ATR and dozens "
    "of times an M5 ATR — so never attribute one timeframe's figure to "
    "another. STOP DIRECTION: for a LONG (buy) a MORE protective stop is a "
    "HIGHER price, for a SHORT (sell) a LOWER price; a tactical_action "
    "rejected under the \"never-widen-stop rule\" means the local model "
    "proposed a stop that was LESS protective than the current one (a "
    "lower stop on a long) and the guard correctly refused it — that is "
    "the system working as designed, not a comparison bug; the finding "
    "there is that the model kept proposing wrong-direction stops. The "
    "final outcome (WIN or LOSS, with the real net P&L) and the REAL time "
    "the position closed are stated at the top of section 3 of the case "
    "file; a later \"reconciled\" timestamp is only when the P&L was "
    "matched to the closing deal, not when the trade closed."
)

TRADE_AUDIT_INSTRUCTION = (
    "You are an independent trading coach conducting a RETROSPECTIVE "
    "audit of one already-closed FTMO trade. This is a REAL prop-firm "
    "1-Stage-Challenge account — the trade below is fully resolved (won "
    "or lost, with a real recorded net P&L); nothing in your review can "
    "change its outcome, so your job is not to grade a live decision but "
    "to extract genuine, specific coaching lessons that make the NEXT "
    "trade better. You do NOT have web search or live data access — "
    "reason only from the case file below, a full, real, chronological "
    "record: (1) Mega Session's own original reasoning for proposing "
    "this trade — symbol, side, price levels, lot size/position %, "
    "trigger condition, stop-loss, take-profit, invalidation condition, "
    "and its stated WHY (\"pct\" = % of account equity risked at the "
    "stop, see the note above); (2) every real Clerk check and tactical action "
    "taken (or explicitly NOT taken, with its own stated reason) while "
    "the position was open; and (3) the real final outcome — net P&L "
    "and the determined cause of closure (stop_loss_hit / "
    "take_profit_hit / clerk_tactical_exit / manual_or_unknown).\n\n"
    "This is not a generic \"was this a good trade\" review. Think like "
    "a real trading coach sitting down with the desk after a trade "
    "closes — ask yourself specific, falsifiable questions, and answer "
    "them using ONLY what's actually in the case file below (not "
    "hindsight knowledge of what the market did AFTER this trade "
    "closed, which isn't given to you and shouldn't be assumed):\n"
    "- Given ONLY what Mega Session actually knew and stated at the "
    "moment it proposed this trade, was the entry thesis internally "
    "consistent with the price levels, stop, and target it chose — or "
    "did the stop/target imply a different read of the setup than the "
    "stated reasoning did?\n"
    "- Was the position size proportionate to the stop distance and the "
    "instrument's own real volatility, or does the case file suggest a "
    "templated size that ignored either?\n"
    "- Did Clerk's own checks and tactical actions (or its explicit "
    "decisions NOT to act) actually track the invalidation condition "
    "and stop/target Mega Session originally set, or did management "
    "drift from the original plan without a clearly stated new reason?\n"
    "- If the trade was a loss: was it a good trade that lost anyway "
    "(the setup was sound, the market just didn't cooperate — no real "
    "lesson beyond \"this happens\"), or a genuinely flawed decision "
    "that happened to lose (a real, fixable lesson)? These are NOT the "
    "same finding and must not be conflated.\n"
    "- If the trade was a win: was it a good decision that worked, or a "
    "flawed decision that got lucky (a stop objectively too tight/wide "
    "for the setup, sized wrong, or held past its own stated "
    "invalidation condition, that still happened to land in profit)? A "
    "win with a real underlying flaw is exactly the finding that's easy "
    "to skip past, and exactly what this exercise exists to catch — do "
    "not give a winning trade a pass on rigor just because it was "
    "profitable.\n"
    "- Cite the SPECIFIC event(s) in the case file — by type and "
    "approximate timestamp — that support each finding you raise. A "
    "finding with nothing specific behind it is not useful coaching.\n\n"
    "Structure your written review as three clearly labeled sections, "
    "even though your own assigned focus below concentrates your real "
    "depth on ONE of them — a short one- or two-line placeholder for "
    "the other two is fine if you have nothing substantive to add there, "
    "so the combined output always keeps all three angles visible in a "
    "consistent structure:\n"
    "#### 1. Mega Session's Original Decision\n"
    "#### 2. Clerk's Management Through the Trade's Life\n"
    "#### 3. Ignored/Contradicted Warning Hunt\n\n"
    "Keep your total written review under 500 words — depth on your own "
    "assigned focus, not equal-length coverage of all three."
)

TRADE_AUDIT_FOCUS_GROUPS: list[str] = [
    (
        "YOUR PRIMARY FOCUS THIS REVIEW (Group A — Mega Session's "
        "Original Decision): three independent models are reviewing "
        "this same closed-trade case file in parallel, each with a "
        "different coaching angle, specifically so the same observation "
        "doesn't get written up three times while a narrower one goes "
        "unnoticed. Your job is to put Mega Session's OWN original "
        "entry decision on trial, using only what it actually knew at "
        "the time (not the outcome). Work through, specifically: was "
        "the stated entry thesis actually internally consistent with "
        "the chosen price, stop-loss, and take-profit — e.g. if the "
        "reasoning described a tight, well-defined range, did the stop "
        "respect that range, or was it set wider/tighter than the "
        "reasoning itself implied? Was the stop-loss placed at a real "
        "invalidation point for the stated thesis, or at an arbitrary "
        "distance unrelated to the actual structure/volatility "
        "described? Was the position size (lot size/% of account) "
        "proportionate to the stop distance and this instrument's own "
        "real volatility? Was the invalidation condition specific and "
        "checkable, or vague enough that Clerk could never have cleanly "
        "detected a genuine invalidation even if it occurred? Given the "
        "REAL final outcome, what is the ONE most specific, actionable "
        "change to how a similar setup should be sized, stopped, or "
        "targeted next time — not a vague \"be more careful,\" but "
        "something concrete enough to check on the next similar trade. "
        "You do not need to also write out full analysis of Clerk's own "
        "management activity or hunt for ignored warnings (the other "
        "two reviewers own those) unless you spot something unambiguous "
        "and urgent there."
    ),
    (
        "YOUR PRIMARY FOCUS THIS REVIEW (Group B — Clerk's Management "
        "Through the Trade's Life): three independent models are "
        "reviewing this same closed-trade case file in parallel, each "
        "with a different coaching angle. Your job is to put Clerk's "
        "REAL, real-time management of this position on trial across "
        "its full recorded lifecycle. Work through, specifically: at "
        "each recorded clerk_check/tactical_action event, did Clerk's "
        "own verdict (confirmed/not confirmed, applied/not applied, and "
        "its stated reason when one was given) track the ORIGINAL "
        "invalidation condition, stop, and target Mega Session set — or "
        "did management drift onto a different, unstated rationale "
        "partway through the position's life? Did Clerk defend the "
        "position too late, too early, or not at all where its own "
        "tactical framework plausibly should have acted given what its "
        "own checks were finding at the time — cite the specific "
        "event(s) where a different action, or an action at a different "
        "point in time, would have served the position better? "
        "Conversely, did Clerk over-manage — acting on a tactical "
        "trigger that wasn't actually a real threat, cutting a position "
        "short of a target it was genuinely on track to reach? Does the "
        "gap in time between the position opening and the first real "
        "clerk_check/tactical_action (or between successive ones) "
        "suggest the position went unattended while real conditions "
        "were changing? What is the ONE most specific, actionable "
        "change to how Clerk should manage a position with this exact "
        "profile next time? You do not need to also write out full "
        "analysis of Mega Session's own original entry decision or hunt "
        "for ignored warnings (the other two reviewers own those) "
        "unless you spot something unambiguous and urgent there."
    ),
    (
        "YOUR PRIMARY FOCUS THIS REVIEW (Group C — Ignored/Contradicted "
        "Warning Hunt): three independent models are reviewing this "
        "same closed-trade case file in parallel. Your assigned angle "
        "is DIFFERENT IN KIND from the other two reviewers' — you are "
        "not judging whether the original strategy or the management "
        "was a good IDEA, you are hunting for a specific, named failure "
        "mode: a case where CORRECT data or an EXPLICIT warning was "
        "already present, in writing, somewhere in this case file — in "
        "Mega Session's own stated reasoning, in a clerk_check's raw "
        "text, or in a tactical_action's stated reason — but the FINAL "
        "decision (the entry itself, a management action, or the "
        "choice not to act) went against, ignored, or contradicted "
        "that exact same piece of correct information, rather than "
        "genuinely being undermined by an information gap or bad luck. "
        "This is a real, previously observed failure mode in this "
        "system worth naming plainly: getting correct data or an "
        "explicit warning and still taking the wrong step in the final "
        "decision anyway. Work through, specifically: read every "
        "event's own raw text/reason field for any explicit warning, "
        "caveat, contradiction, or uncertainty the system itself "
        "already stated out loud, then check whether the very next "
        "action taken (or NOT taken) actually respected that warning "
        "or simply proceeded as if it had never been raised. Check the "
        "stop-loss/take-profit/lot-size math itself against the "
        "instrument's own price levels stated elsewhere for a plain "
        "arithmetic or logical inconsistency. If Clerk logged a check "
        "as \"not confirmed\" or flagged a real invalidation-adjacent "
        "condition but no corresponding tactical action followed, treat "
        "that gap as a candidate finding — was there a stated reason, "
        "and does it hold up? For each finding, quote or closely "
        "paraphrase the SPECIFIC contradicting text and name exactly "
        "which later step ignored it — a vague \"there might have been "
        "a warning somewhere\" is not usable. If you find NO genuine "
        "instance of this specific failure mode (a real possibility), "
        "say so plainly rather than manufacturing a weak example — \"no "
        "ignored-warning pattern found this trade\" is itself a useful "
        "finding. You do not need to also write out full strategic "
        "analysis of Mega Session's entry decision or Clerk's overall "
        "management quality (the other two reviewers own those) unless "
        "it's inseparable from a warning-ignored finding you're making."
    ),
]

if len(TRADE_AUDIT_FOCUS_GROUPS) != len(AUDIT_MODELS):
    raise AssertionError(
        f"TRADE_AUDIT_FOCUS_GROUPS has {len(TRADE_AUDIT_FOCUS_GROUPS)} entries, "
        f"but AUDIT_MODELS has {len(AUDIT_MODELS)} slots — build_audit_block requires exactly one per slot."
    )


# --- enable/state -------------------------------------------------------


def read_trade_audit_enabled() -> bool:
    """Whether Trade Audit is currently enabled — defaults to True on a
    missing/corrupt file, same opt-out-not-opt-in posture as every other
    role's own enabled flag in this codebase."""
    path = Path(config.TRADE_AUDIT_ENABLED_FILE)
    if not path.exists():
        return True
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return True
    return bool(data.get("enabled", True))


def set_trade_audit_enabled(enabled: bool) -> None:
    try:
        Path(config.TRADE_AUDIT_ENABLED_FILE).write_text(json.dumps({"enabled": enabled}))
    except OSError as e:
        logger.warning("Could not write trade-audit enabled-flag file %s: %s", config.TRADE_AUDIT_ENABLED_FILE, e)


def read_trade_audit_state() -> dict:
    """Best-effort: the outcome of the last Trade Audit run. `{}` on
    anything missing/unreadable, same safe-default convention as every
    other role's own state reader in this codebase."""
    path = Path(config.TRADE_AUDIT_STATE_FILE)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_trade_audit_state(status: str, now_utc: datetime, audited_count: int = 0, failed_count: int = 0) -> None:
    """`last_run_date_utc` is set to today ONLY when `status == "success"`
    — mirrors ai.researcher._write_researcher_state exactly: a failed
    attempt genuinely allows a retry later the same day, within the
    grace window, instead of permanently burning that day's one chance.
    A "success" with zero pending stories still counts as success — the
    job correctly did its one required check, there was just nothing to
    audit that day."""
    prior = read_trade_audit_state()
    payload = {
        "last_attempt_utc": now_utc.isoformat(),
        "last_status": status,
        "last_audited_count": audited_count,
        "last_failed_count": failed_count,
        "last_run_date_utc": now_utc.date().isoformat() if status == "success" else prior.get("last_run_date_utc"),
        "total_runs": prior.get("total_runs", 0) + 1,
    }
    try:
        Path(config.TRADE_AUDIT_STATE_FILE).write_text(json.dumps(payload))
    except OSError as e:
        logger.warning("Could not write trade-audit state file %s: %s", config.TRADE_AUDIT_STATE_FILE, e)


# --- trigger timing -------------------------------------------------


def _ny_close_trigger_utc(for_date: date) -> datetime:
    """The real US/FTMO forex-session daily close for `for_date`, in
    UTC — computed FRESH via stdlib zoneinfo rather than a fixed UTC
    hour/minute (see this module's own top-of-file docstring and
    config.TRADE_AUDIT_TRIGGER_HOUR_LOCAL's own comment for the full
    "deliberate departure from this codebase's usual fixed-UTC
    convention" reasoning). Correctly returns 21:00 UTC during EDT and
    22:00 UTC during EST, verified live against both a real EDT date and
    a real EST date on this machine."""
    ny_local = datetime(
        for_date.year, for_date.month, for_date.day,
        config.TRADE_AUDIT_TRIGGER_HOUR_LOCAL, config.TRADE_AUDIT_TRIGGER_MINUTE_LOCAL,
        tzinfo=ZoneInfo(config.TRADE_AUDIT_TRIGGER_TZ),
    )
    return ny_local.astimezone(timezone.utc)


def is_trade_audit_due(now_utc: datetime, state: dict | None = None) -> bool:
    """True only within the config.TRADE_AUDIT_GRACE_MINUTES window
    starting config.TRADE_AUDIT_SETTLEMENT_BUFFER_MINUTES after today's
    real NY-close instant, and only if this job hasn't already completed
    a real check today — same once-daily/grace-window/same-day-dedup
    shape as ai.researcher.is_researcher_due, just with a dynamically-
    computed (DST-aware) trigger instant instead of a fixed one."""
    state = state if state is not None else read_trade_audit_state()
    today = now_utc.date()
    if state.get("last_run_date_utc") == today.isoformat():
        return False
    due_from = _ny_close_trigger_utc(today) + timedelta(minutes=config.TRADE_AUDIT_SETTLEMENT_BUFFER_MINUTES)
    due_until = due_from + timedelta(minutes=config.TRADE_AUDIT_GRACE_MINUTES)
    return due_from <= now_utc <= due_until


def next_trade_audit_check_utc(now_utc: datetime, state: dict | None = None) -> datetime:
    """Best-effort next Trade Audit instant, purely for display (the real
    trigger is trade_audit_job.py's own OS-level poll) — parity with
    ai.researcher.next_researcher_check_utc."""
    state = state if state is not None else read_trade_audit_state()
    today = now_utc.date()
    ran_today = state.get("last_run_date_utc") == today.isoformat()
    today_due_from = _ny_close_trigger_utc(today) + timedelta(minutes=config.TRADE_AUDIT_SETTLEMENT_BUFFER_MINUTES)
    if not ran_today and now_utc < today_due_from:
        return today_due_from
    return _ny_close_trigger_utc(today + timedelta(days=1)) + timedelta(minutes=config.TRADE_AUDIT_SETTLEMENT_BUFFER_MINUTES)


# --- case file -----------------------------------------------------

_ORIGINAL_REASONING_EVENTS = ("proposed", "carried_forward")
_MANAGEMENT_EVENTS = ("order_result", "clerk_check", "tactical_action", "filled")


def _outcome_headline(closed_data: dict, closed_at: str | None = None) -> str:
    """The one-line verdict an audit model can not miss: WIN/LOSS and the real net P&L. Found on audit
    (2026-09-24): the first two real audits were written from a case file whose outcome was just
    {"net_pnl": null, "cause": "manual_or_unknown"} (the P&L had not been reconciled yet), so the models
    said things like "we never learn whether the thesis was invalidated" instead of judging a win or a loss."""
    pnl = closed_data.get("net_pnl")
    when = f" The position actually closed at {closed_at}." if closed_at else ""
    if not isinstance(pnl, (int, float)):
        return "OUTCOME: net P&L NOT YET KNOWN (the closing deal could not be matched) — do not guess win or loss." + when
    verdict = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"
    return f"OUTCOME: {verdict} — real net P&L {pnl:+.2f}, cause of closure {closed_data.get('cause', 'unknown')}." + when


def build_case_file(story: TradeStory) -> str:
    """Pure formatting, no I/O — a full-fidelity plain-text case file
    for the audit pool, chronological, in three sections. Deliberately
    does NOT reuse ai.trade_journal._EVENT_RENDERERS (those are markdown
    renderers tuned for a human Obsidian reader, including a raw_text
    [:400] truncation) — the audit models get each event's FULL raw
    text/data, untruncated, since a truncated warning is exactly the
    kind of thing Group C's ignored-warning hunt needs to see in full."""
    lines = [f"# Case file: {story.symbol} ({story.status})", ""]

    lines.append("## 1. Mega Session's original reasoning")
    for event in story.events:
        if event.type in _ORIGINAL_REASONING_EVENTS:
            lines.append(f"### {event.timestamp_utc} — {event.type}")
            lines.append(json.dumps(event.data, indent=2))
            lines.append("")

    lines.append("## 2. Clerk's real management through the position's life")
    management_events = [e for e in story.events if e.type in _MANAGEMENT_EVENTS]
    if not management_events:
        lines.append("(No Clerk activity was recorded for this trade.)")
    for event in management_events:
        lines.append(f"### {event.timestamp_utc} — {event.type}")
        lines.append(json.dumps(event.data, indent=2))
        lines.append("")

    lines.append("## 3. Real final outcome")
    closed_event = next((e for e in reversed(story.events) if e.type in ("closed_reconciled", "closed")), None)
    if closed_event is not None:
        # The real close moment is the ORIGINAL "closed" event; "closed_reconciled" is stamped whenever the P&L
        # was matched to the closing deal (found on audit: a model read a reconciliation 89 hours later as the
        # close time and wrote about an "89-hour unmanaged gap" that never existed).
        original_close = next((e for e in story.events if e.type == "closed"), closed_event)
        lines.append(_outcome_headline(closed_event.data, original_close.timestamp_utc))
        lines.append(f"### {original_close.timestamp_utc} — closed")
        lines.append(json.dumps(closed_event.data, indent=2))
        if closed_event is not original_close:
            lines.append(f"(net P&L reconciled against the closing deal at {closed_event.timestamp_utc}; not the close time)")
    else:
        lines.append(f"(No 'closed' event found, despite status={story.status!r} — this shouldn't happen.)")

    return "\n".join(lines)


# --- the real work ---------------------------------------------------


def _audit_one_story(story: TradeStory) -> bool:
    """Returns True if a real audit was recorded. On a total pool
    outage (audit_available=False), deliberately does NOT call
    record_audit — leaves the story eligible for a future day's run
    instead of permanently marking a total-outage day as "audited" with
    no usable content (a deliberate refinement flagged to, and approved
    by, the user during planning: see this feature's own plan doc)."""
    case_file = build_case_file(story)
    result = build_audit_block(
        _CASE_FILE_CONTEXT_STUB,
        case_file,
        audit_instruction=TRADE_AUDIT_INSTRUCTION,
        include_copilot=False,
        focus_directives=TRADE_AUDIT_FOCUS_GROUPS,
    )
    if not result.audit_available:
        logger.warning(
            "Trade Audit: no model produced a real review for %s — leaving it unaudited for a future run.",
            story.story_id,
        )
        return False
    record_audit(story, result.block, result.audit_available)
    return True


def run_trade_audit_check() -> None:
    """Wrapped by trade_audit_job.py's run_with_timeout. FIRST action:
    find_unaudited_closed_stories() — zero model calls if empty (direct
    user request)."""
    pending = find_unaudited_closed_stories()
    if not pending:
        logger.info("Trade Audit: nothing unaudited — zero model calls made.")
        _write_trade_audit_state("success", datetime.now(timezone.utc))
        return

    audited = failed = 0
    for story in pending[: config.TRADE_AUDIT_MAX_STORIES_PER_RUN]:
        try:
            if _audit_one_story(story):
                audited += 1
            else:
                failed += 1
        except Exception:
            failed += 1
            logger.exception("Trade Audit: failed auditing story %s — continuing with the rest.", story.story_id)

    _write_trade_audit_state("success", datetime.now(timezone.utc), audited_count=audited, failed_count=failed)
