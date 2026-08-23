"""The "clerk/executioner" half of the FTMO boardroom architecture —
direct user request 2026-08-22/23: Claude (senior analyst) runs only in
the once-daily mega analysis session; GitHub Copilot's own separate role
(the one already carved out of the audit pool earlier the same day — see
ai.mega_analysis.run_mega_analysis's include_copilot=False) is to run
on a short interval (config.COPILOT_EXECUTION_CHECK_INTERVAL_MINUTES,
default 15 — tightened from an original hourly cadence on direct user
request 2026-08-23, since the check itself is cheap), read that mega
session's own guidance, check it against fresh live MT5 data/technicals,
and if a setup has genuinely come true,
execute it itself with position sizing recomputed from live equity. No
human confirmation step in this path — this module IS the confirmation
step, standing in for the "Confirm and Execute" button the manual
Apply Suggestion dialog would otherwise require.

Two categories of trade come from ai.mega_analysis.read_latest_suggestion():
- `immediate_allocation` — feasible right now, per the mega session. No
  new schema needed for this; it's re-applied every poll via
  compute_rebalance_plan's own existing diff logic (open/increase/
  reduce/close/hold), which is naturally idempotent (a target that
  already matches what's held resolves to "hold", so repeatedly
  re-attempting it is safe).
- `pending_setups` — conditional, not yet feasible; only added to the
  target mix once a live Copilot verdict confirms the stated
  trigger_condition against fresh technicals.

**A safety-critical fact this design is built around**: `open_position`
(data/mt5_execution.py) never places a market order — it always places
a GTC PENDING LIMIT order, and MT5 tracks pending orders
(get_pending_orders) and filled positions (get_open_positions) as
separate concepts. A flat "executed once, don't touch again" design
would let an unfilled order silently orphan itself the moment it drops
out of consideration. `SymbolSettlement` tracks each symbol this job has
ever placed an order for through an explicit state machine instead:

    order_placed --(ticket fills)--> filled --(position closes)--> closed_after_fill
        |
        +--(ticket rejected/cancelled, never filled)--> dropped entirely

`closed_after_fill` is permanent for the rest of THIS mega-session cycle
(the settlement file's own `generated_utc` field) — this is the actual
"don't reopen a trade that already ran its course" guarantee, and it
applies uniformly to `immediate_allocation` entries too, not just
Pending Setups: "recommend trades feasible at the time of analysis" is a
one-shot instruction, not "keep relentlessly reopening this every hour."
A brand-new `order_placed` entry (a just-fired Pending Setup with zero
held volume yet) is deliberately NOT included in the merged target mix
fed to compute_rebalance_plan — that function only diffs against HELD
positions, so an unfilled fresh-open order shows zero held volume and
would otherwise look like "still needs opening," stacking a second order
on top of the first every single poll. An `order_placed` entry that
originated from `immediate_allocation`, by contrast, IS still carried
forward into the merge — real bug found on audit: excluding it too would
make an already-HELD position (a filled base plus a still-unfilled top-
up "increase" order) look like "held but missing from the target," which
defaults to a 0% target and force-closes the whole real position. Not
resubmitting a duplicate order for that same still-open delta is handled
separately, at the execution-loop level (skip the order_send call for a
symbol that already has an outstanding order_placed record), never by
hiding the symbol from the plan.

On a FRESH mega session (the suggestion file's own generated_utc
changes), any symbol still `order_placed` gets its order cancelled
(cancel_pending_order) before the settlement record resets — otherwise a
stale, yesterday-sized order would survive into a new cycle, untracked
by anything.

Every safety gate this job enforces (is_demo/ALLOW_LIVE_EXECUTION,
is_trading_permitted(), FTMO daily-loss headroom) is the exact same,
shared risk/apply_suggestion.py::check_execution_safety_gates the manual
"Apply Suggestion" dialog uses — direct user request: no NEW autonomous-
specific safety cap beyond what already exists.

2026-08-23 update — user-controlled enable/disable toggle and review-
frequency picker (app.py's Copilot Execution Clerk panel heading):
read_copilot_execution_enabled() is checked as the VERY FIRST thing in
run_copilot_execution_check(), before even the mega-session-live check
— the single, authoritative gate all three of this function's call
sites (the standalone poll, mega_analysis_job.py's inline pass, app.py's
manual-button inline pass) share automatically, so a disabled clerk is
disabled everywhere at once, not just on the scheduled poll. The review
frequency itself (previously a fixed config.COPILOT_EXECUTION_CHECK_
INTERVAL_MINUTES) is now similarly overridable via read_copilot_
execution_interval_minutes(), read by _interval_start (and therefore by
both is_execution_due and the new next_execution_check_utc, used for
the panel's own "next review" countdown)."""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import config
from ai.copilot_cli import CLI_FAILED_PREFIX as COPILOT_FAILED_PREFIX
from ai.copilot_cli import CLI_MISSING_MESSAGE as COPILOT_MISSING_MESSAGE
from ai.copilot_cli import run_copilot
from ai.ftmo_suggest import (
    analyze_ftmo_asset_live,
    fetch_ftmo_status,
    format_ftmo_asset_context,
    read_latest_suggestion,
)
from ai.mega_analysis import read_progress as read_mega_progress
from ai.mega_analysis import read_state as read_mega_state
from ai.portfolio_suggest import AllocationEntry, PendingSetup
from data.mt5_execution import MT5ConnectionError, OrderResult, cancel_pending_order, close_position, open_position
from data.mt5_source import (
    PendingOrder,
    Position,
    connect,
    get_account_summary,
    get_contract_spec,
    get_market_watch,
    get_open_positions,
    get_pending_orders,
    is_trading_permitted,
)
from risk.apply_suggestion import check_execution_safety_gates, compute_aggregate_heat_pct, compute_rebalance_plan
from risk.ftmo_rules import DEFAULT_HEADROOM_FRACTION, would_breach_daily_loss_headroom

logger = logging.getLogger(__name__)

# The shared mutual-exclusion lock guarding every call to
# run_copilot_execution_check(), regardless of which of its TWO call
# sites triggers it — copilot_execution_job.py's own standalone hourly
# poll, AND mega_analysis_job.py's inline "run once immediately after a
# successful mega session" pass. Real concurrency bug found on audit:
# these are two independent OS processes: the inline call used to invoke
# run_copilot_execution_check() directly with zero lock protection, so a
# standalone hourly poll landing at the same moment an inline pass was
# still running could execute the exact same trade-decision logic
# concurrently — risking a duplicate order for the same symbol or a lost
# update to the settlement/state files (a read-modify-write race, even
# though each individual file write is itself atomic). Defined here
# (not duplicated as a private constant in each job script) so both
# callers acquire the literal same lock file, and can never drift apart
# on path or staleness window.
EXECUTION_LOCK_PATH = Path(__file__).resolve().parent.parent / "copilot_execution.lock"
# Comfortably above COPILOT_EXECUTION_RUN_TIMEOUT_SECONDS (both callers
# bound the guarded work to that ceiling), so a lock still fresher than
# this genuinely could be a real, still-running attempt.
EXECUTION_LOCK_STALE_AFTER_SECONDS = 20 * 60


@dataclass
class SymbolSettlement:
    origin: str  # "immediate" | "pending_setup" — informational/audit only
    state: str  # "order_placed" | "filled" | "closed_after_fill"
    entry: dict  # {"pct", "price", "stop_loss", "take_profit", "side"} — the values the order was placed with
    order_ticket: int | None = None


# --- run-state / live-progress (mirrors ai.mega_analysis's own pattern) ---


def read_execution_state() -> dict:
    """Best-effort: the outcome of the last execution-check poll.
    `{}` on anything missing/unreadable, same safe-default convention as
    ai.mega_analysis.read_state."""
    path = Path(config.COPILOT_EXECUTION_STATE_FILE)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_execution_state(
    status: str,
    detail: str = "",
    last_verdicts: dict | None = None,
    last_execution_results: dict | None = None,
) -> None:
    """Records this poll's outcome. Preserves `last_run_interval_utc`
    from whatever was there before (this function never touches it —
    only _mark_interval_ran does) so a call here can never accidentally
    erase the due-check's own dedup marker regardless of call order.
    `last_verdicts` and `last_execution_results` are both MERGED onto
    the prior record (a fresh entry for a symbol overwrites its own old
    one; other symbols' entries from an earlier poll survive) rather
    than replaced wholesale — otherwise a poll that only touches some
    symbols (e.g. the COPILOT_EXECUTION_MAX_PENDING_SETUPS cap truncated
    the Pending Setups list, or a plan this poll simply had fewer
    entries than a prior one) would silently discard a still-relevant
    entry recorded a poll or two earlier for a symbol this poll didn't
    touch.

    `last_execution_results` — added 2026-08-23 direct user request
    ("I see position in 5 assets but only 2 assets are showing up in the
    clerk section") — is the immediate_allocation counterpart to
    last_verdicts: one entry per symbol in the LATEST compute_rebalance_
    plan (open/increase/reduce/close/hold/infeasible), not just the ones
    that resulted in a placed order. Without this, a FAILED attempt (no
    settlement record ever gets created for those) was invisible outside
    the log file — app.py's panel only ever showed Pending Setups, never
    what happened to the actual immediate_allocation targets."""
    prior = read_execution_state()
    now = datetime.now(timezone.utc)
    merged_verdicts = {**prior.get("last_verdicts", {}), **(last_verdicts or {})}
    merged_results = {**prior.get("last_execution_results", {}), **(last_execution_results or {})}
    payload = {
        "last_attempt_utc": now.isoformat(),
        "last_status": status,
        "last_detail": detail,
        "last_verdicts": merged_verdicts,
        "last_execution_results": merged_results,
        "last_run_interval_utc": prior.get("last_run_interval_utc"),
    }
    try:
        Path(config.COPILOT_EXECUTION_STATE_FILE).write_text(json.dumps(payload, indent=2))
    except OSError as e:
        logger.warning("Could not write execution state file %s: %s", config.COPILOT_EXECUTION_STATE_FILE, e)


def read_execution_progress() -> dict:
    """Best-effort read of this job's own LIVE, in-progress status —
    mirrors ai.mega_analysis.read_progress exactly, same standing
    "automated runs must be as visible as a manual button click"
    principle applied to this second unattended job."""
    path = Path(config.COPILOT_EXECUTION_PROGRESS_FILE)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_execution_progress(message: str) -> None:
    payload = {"message": message, "updated_utc": datetime.now(timezone.utc).isoformat()}
    try:
        Path(config.COPILOT_EXECUTION_PROGRESS_FILE).write_text(json.dumps(payload))
    except OSError as e:
        logger.warning("Could not write progress file %s: %s", config.COPILOT_EXECUTION_PROGRESS_FILE, e)


# --- user-controlled enable/disable + review-frequency overrides (2026-08-23) ---


def read_copilot_execution_enabled() -> bool:
    """Whether the Copilot Execution Clerk is currently enabled — defaults
    to True on a missing/corrupt file, same opt-out-not-opt-in posture as
    ai.mega_analysis.read_mega_analysis_enabled. Checked once, at the top
    of run_copilot_execution_check itself (not duplicated at each of its
    three call sites — the standalone poll, mega_analysis_job.py's inline
    pass, and app.py's manual-button inline pass) so a single toggle
    genuinely governs the whole role regardless of what triggered it,
    matching this same session's earlier "only the trigger should differ"
    correction (see ai.ftmo_suggest.suggest_ftmo_portfolio's own
    docstring)."""
    path = Path(config.COPILOT_EXECUTION_ENABLED_FILE)
    if not path.exists():
        return True
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return True
    return bool(data.get("enabled", True))


def set_copilot_execution_enabled(enabled: bool) -> None:
    try:
        Path(config.COPILOT_EXECUTION_ENABLED_FILE).write_text(json.dumps({"enabled": enabled}))
    except OSError as e:
        logger.warning("Could not write enabled-flag file %s: %s", config.COPILOT_EXECUTION_ENABLED_FILE, e)


def read_copilot_execution_interval_minutes() -> int:
    """The review-frequency minutes — user-overridable via app.py's
    picker, falling back to config.COPILOT_EXECUTION_CHECK_INTERVAL_
    MINUTES on a missing/corrupt/non-positive value."""
    path = Path(config.COPILOT_EXECUTION_INTERVAL_FILE)
    default = config.COPILOT_EXECUTION_CHECK_INTERVAL_MINUTES
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text())
        minutes = int(data["minutes"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return default
    return minutes if minutes > 0 else default


def set_copilot_execution_interval_minutes(minutes: int) -> None:
    try:
        Path(config.COPILOT_EXECUTION_INTERVAL_FILE).write_text(json.dumps({"minutes": minutes}))
    except OSError as e:
        logger.warning("Could not write interval file %s: %s", config.COPILOT_EXECUTION_INTERVAL_FILE, e)


# --- interval due-check (mirrors ai.mega_analysis.is_due, keyed on a fixed-size window) ---


def _interval_start(now_utc: datetime) -> datetime:
    """The start instant of the review-frequency window `now_utc` falls
    inside — e.g. with a 15-minute interval, 13:00-13:14 all resolve to
    13:00, 13:15-13:29 all resolve to 13:15. A pure function of the clock
    plus the current interval-minutes setting, not of any stored state,
    so it can never disagree with itself between is_execution_due and
    _mark_interval_ran."""
    interval = read_copilot_execution_interval_minutes()
    bucket_minute = (now_utc.minute // interval) * interval
    return now_utc.replace(minute=bucket_minute, second=0, microsecond=0)


def is_execution_due(now_utc: datetime, state: dict | None = None) -> bool:
    """True only within the first COPILOT_EXECUTION_GRACE_MINUTES minutes
    of a review-frequency-sized window this job hasn't already run in —
    the same trigger/grace-window/dedup shape as ai.mega_analysis.is_due's
    daily version, just keyed on a fixed-size recurring window instead of
    one clock time per day, since this job's own OS-level poll interval
    is set at the Task Scheduler level, not here. Direct user request
    2026-08-23: originally hourly, tightened to check more often since
    the work itself is cheap; the frequency itself became user-editable
    (read_copilot_execution_interval_minutes) the same day."""
    state = state if state is not None else read_execution_state()
    interval = read_copilot_execution_interval_minutes()
    current_interval_key = _interval_start(now_utc).isoformat()
    if state.get("last_run_interval_utc") == current_interval_key:
        return False
    return now_utc.minute % interval <= config.COPILOT_EXECUTION_GRACE_MINUTES


def next_execution_check_utc(now_utc: datetime, state: dict | None = None) -> datetime:
    """Best-effort next execution-check instant, purely for display (the
    real trigger is copilot_execution_job.py's own OS-level poll, not a
    precise clock instant): `now_utc` itself if the current window hasn't
    run yet (i.e. due now, or as soon as the next OS-level poll lands),
    otherwise the start of the next review-frequency window."""
    state = state if state is not None else read_execution_state()
    interval = read_copilot_execution_interval_minutes()
    current_start = _interval_start(now_utc)
    if state.get("last_run_interval_utc") != current_start.isoformat():
        return now_utc
    return current_start + timedelta(minutes=interval)


def _mark_interval_ran(now_utc: datetime) -> None:
    prior = read_execution_state()
    prior["last_run_interval_utc"] = _interval_start(now_utc).isoformat()
    try:
        Path(config.COPILOT_EXECUTION_STATE_FILE).write_text(json.dumps(prior, indent=2))
    except OSError as e:
        logger.warning("Could not write execution state file %s: %s", config.COPILOT_EXECUTION_STATE_FILE, e)


# --- settlement (the pending-order lifecycle state machine) ---


def _load_settlement() -> dict:
    path = Path(config.COPILOT_EXECUTION_SETTLEMENT_FILE)
    if not path.exists():
        return {"generated_utc": None, "settled": {}}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"generated_utc": None, "settled": {}}
    if not isinstance(data, dict) or "settled" not in data:
        return {"generated_utc": None, "settled": {}}
    return data


def read_settlement() -> dict:
    """Public read of the per-symbol settlement-tracking file, for UI
    display (app.py's Copilot execution panel) — same safe-default
    behavior as _load_settlement, just under a public name since this
    one's meant for an external caller rather than this module's own
    internal use."""
    return _load_settlement()


def _save_settlement(settlement: dict) -> None:
    path = Path(config.COPILOT_EXECUTION_SETTLEMENT_FILE)
    try:
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(json.dumps(settlement, indent=2))
        os.replace(tmp_path, path)
    except OSError as e:
        logger.warning("Could not write settlement file %s: %s", path, e)


def _reconcile_settlement(
    settled: dict, positions: list[Position], pending_orders: list[PendingOrder]
) -> dict:
    """Advances every tracked symbol's state machine from the SAME fresh
    positions/pending-orders snapshot the rest of this poll already
    fetched — see this module's own docstring for the full transition
    table. A symbol whose order was rejected/cancelled and never filled
    is dropped entirely (not carried forward as any state), making it
    eligible for a fresh Copilot check on a later poll."""
    held_symbols = {p.symbol for p in positions}
    pending_tickets_by_symbol: dict[str, set] = {}
    for o in pending_orders:
        pending_tickets_by_symbol.setdefault(o.symbol, set()).add(o.ticket)

    updated: dict = {}
    for symbol, rec in settled.items():
        state = rec.get("state")
        ticket = rec.get("order_ticket")
        if state == "order_placed":
            still_pending = ticket is not None and ticket in pending_tickets_by_symbol.get(symbol, set())
            if still_pending:
                updated[symbol] = rec
            elif symbol in held_symbols:
                updated[symbol] = {**rec, "state": "filled"}
            # else: order gone and never filled -- dropped, eligible for a fresh check.
        elif state == "filled":
            updated[symbol] = rec if symbol in held_symbols else {**rec, "state": "closed_after_fill"}
        elif state == "closed_after_fill":
            updated[symbol] = rec  # permanent for this mega-session cycle
    return updated


def _reset_settlement_for_new_session(settled: dict) -> dict:
    """Cancels any still-unfilled order before discarding old tracking —
    otherwise a stale, yesterday-sized GTC limit order would survive
    into a new mega-session cycle, untracked by anything (see this
    module's own docstring). filled/closed_after_fill symbols need no
    action here — the reset only clears TRACKING, it never touches an
    already-filled real position."""
    for symbol, rec in settled.items():
        if rec.get("state") != "order_placed":
            continue
        ticket = rec.get("order_ticket")
        if ticket is None:
            continue
        try:
            result = cancel_pending_order(ticket)
            if not result.success:
                logger.warning(
                    "Could not cancel superseded pending order %d (%s) on session reset: %s",
                    ticket, symbol, result.comment,
                )
        except MT5ConnectionError as e:
            logger.warning("Could not cancel superseded pending order %d (%s): %s", ticket, symbol, e)
    return {}


def _allocation_entry_from_dict(d: dict) -> AllocationEntry:
    return AllocationEntry(
        pct=float(d["pct"]),
        price=d.get("price"),
        stop_loss=d.get("stop_loss"),
        take_profit=d.get("take_profit"),
        side=d.get("side", "buy"),
    )


def _build_carried_forward_allocation(immediate_allocation_raw: dict, settled: dict) -> dict[str, AllocationEntry]:
    """The base target mix before any THIS-POLL Copilot verdicts are
    merged in: every immediate_allocation symbol EXCEPT one that's
    already closed_after_fill this cycle (don't reopen a trade that
    already ran its course), PLUS every settled symbol in state "filled"
    that isn't already covered by immediate_allocation (so
    compute_rebalance_plan's normal hold/resize/close diffing keeps
    managing an already-open, previously-conditional position).

    An immediate_allocation symbol with an outstanding order_placed
    record is DELIBERATELY still carried forward here, not excluded —
    real bug found on audit: excluding it would make compute_rebalance_
    plan see an ALREADY-HELD symbol (e.g. one with a filled base
    position plus a still-unfilled top-up "increase" order) as "held but
    missing from the target," which defaults to a 0% target and would
    force-close the entire real position just because an unrelated
    top-up order happened to still be pending. Not resubmitting a
    DUPLICATE order for that same still-open delta is handled instead at
    the execution-loop level (see run_copilot_execution_check), by
    skipping only the actual order_send call, not by hiding the whole
    symbol from the plan. A settled symbol NOT in immediate_allocation
    (a fired Pending Setup) in state order_placed is correctly excluded
    by the second loop below (only "filled" is carried forward there) —
    it has zero held volume yet, so there is nothing to protect from a
    forced close, and including it here would just resubmit a duplicate
    fresh-open order every poll instead."""
    carried: dict[str, AllocationEntry] = {}
    for symbol, raw in immediate_allocation_raw.items():
        if symbol.upper() == "CASH":
            continue
        rec = settled.get(symbol)
        if rec is not None and rec.get("state") == "closed_after_fill":
            continue
        carried[symbol] = _allocation_entry_from_dict(raw)

    for symbol, rec in settled.items():
        if symbol in carried or symbol in immediate_allocation_raw:
            continue
        if rec.get("state") != "filled":
            continue
        carried[symbol] = _allocation_entry_from_dict(rec["entry"])

    return carried


# --- Copilot verdict: a hard, fail-safe boolean, not a feeling ---

_FINAL_VERDICT_PATTERN = re.compile(r"FINAL_VERDICT:\s*(CONFIRMED|NOT_CONFIRMED)", re.IGNORECASE)


def parse_copilot_verdict(response_text: str) -> bool:
    """Extracts a hard execute/no-execute boolean from Copilot's free-
    text response — a new contract, unlike build_copilot_verification's
    existing usage elsewhere (which never programmatically parses
    Copilot's response, just embeds it as text for a LATER Claude call
    to weigh). Fails safe to False (never execute) on every ambiguous
    branch: the CLI missing/failed sentinels, an empty response, no
    FINAL_VERDICT token anywhere, or a value other than CONFIRMED. If
    the model restates itself and multiple tokens appear, the LAST one
    wins — same "last occurrence wins" convention
    ai.portfolio_suggest._last_allocation_match already establishes —
    but only an unambiguous final CONFIRMED flips this to True; anything
    else (including NOT_CONFIRMED, or a stray unrecognized word) does
    not."""
    if not response_text:
        return False
    if response_text == COPILOT_MISSING_MESSAGE or response_text.startswith(COPILOT_FAILED_PREFIX):
        return False
    if not response_text.strip():
        return False
    matches = _FINAL_VERDICT_PATTERN.findall(response_text)
    if not matches:
        return False
    return matches[-1].upper() == "CONFIRMED"


def _build_verdict_prompt(
    setup: PendingSetup, technical_context: str, account_equity: float, elapsed_description: str
) -> str:
    return (
        "You are an automated trading clerk for an FTMO account. A "
        "senior analyst (Claude) identified the following NOT-YET-"
        "TRIGGERED trade setup during today's mega market analysis, "
        "worth watching for until the next analysis session:\n\n"
        f"Symbol: {setup.symbol}\n"
        f"Side: {setup.side}\n"
        f"Trigger condition (from the senior analyst): {setup.trigger_condition}\n"
        f"Planned entry: {setup.price}\n"
        f"Planned stop-loss: {setup.stop_loss}\n"
        f"Planned take-profit: {setup.take_profit}\n"
        f"Reason given: {setup.reason}\n"
        f"Time since that analysis was run: {elapsed_description}.\n\n"
        "Below is this symbol's REAL, LIVE technical picture right now "
        "(the same data source that powers this account's own Asset "
        "Health analysis). Use it, plus your own live web access if "
        "useful (e.g. checking for any major news since the analysis "
        "ran that would invalidate this idea), to judge two things: "
        "(1) has the stated trigger condition genuinely been met, and "
        "(2) does the planned stop-loss/take-profit still make sense "
        "against the current price and volatility, not just whether "
        "the trigger fired. Do not invent or assume data you don't "
        "have — if the live technical picture below is inconclusive or "
        "contradicts the trigger condition, that means NOT_CONFIRMED.\n\n"
        "This is a genuinely continuous, thinking task, not a one-shot "
        "check — you may be asked about this same setup again on a "
        "later poll, possibly after a real delay (a missed poll, a "
        "computer that was off, or simply the market drifting for "
        "hours). Explicitly distinguish two different NOT_CONFIRMED "
        "cases in your reasoning: (a) the setup simply hasn't triggered "
        "yet and is still worth watching — say so plainly, no warning "
        "needed; versus (b) enough has genuinely changed since the "
        "senior analyst's own timestamp above (price has moved well "
        "past where the stop/target still make sense, the technical "
        "structure that justified this setup no longer holds, or "
        "real news has overtaken it) that this setup should now be "
        "treated as STALE and abandoned rather than kept watching. For "
        "case (b) specifically, start your response with a single "
        "line beginning exactly with \"STALE SETUP WARNING:\" followed "
        "by one concise sentence stating what changed and why this is "
        "no longer a valid trade — this line is what a human will see "
        "in the UI and the log, so make it clear and specific, not "
        "generic.\n\n"
        f"{technical_context}\n\n"
        f"Account equity for context: {account_equity:.2f}. This does "
        "not change your verdict on the trigger itself.\n\n"
        "End your response with exactly one line, and nothing after "
        'it: either "FINAL_VERDICT: CONFIRMED" or "FINAL_VERDICT: '
        'NOT_CONFIRMED". Do not use this exact token anywhere else in '
        "your response."
    )


def _fetch_technical_context(
    setup: PendingSetup, market_prices: dict, account_equity: float
) -> str | None:
    """The MT5-bound half of checking one setup — fetches its live
    4-timeframe technicals (analyze_ftmo_asset_live) and formats them
    into the same rich narrative Asset Health/the mega session's own
    prompt already use. MUST be called sequentially by the caller, once
    per not-yet-settled setup, never from inside a thread pool — the
    MetaTrader5 Python API isn't documented thread-safe, and this
    project's whole connection model already assumes one connection,
    called sequentially. Returns None if the symbol isn't currently
    visible in Market Watch (skipped, logged by the caller) rather than
    raising."""
    asset = market_prices.get(setup.symbol)
    if asset is None:
        return None
    analysis = analyze_ftmo_asset_live(setup.symbol, asset.bid, asset.ask, asset.description)
    return format_ftmo_asset_context([analysis], account_equity=account_equity)


def _run_copilot_verdict(
    setup: PendingSetup, technical_context: str, account_equity: float, elapsed_description: str
) -> tuple[PendingSetup, bool, str]:
    """The Copilot-bound half — pure CLI subprocess call, no MT5
    involvement, safe to run from inside a thread pool (see
    run_copilot_execution_check's own parallel round). Returns (setup,
    confirmed, raw_copilot_text) — the raw text always travels back for
    the audit-trail log/UI, even on a NOT_CONFIRMED or failed verdict."""
    prompt = _build_verdict_prompt(setup, technical_context, account_equity, elapsed_description)
    raw = run_copilot(prompt, timeout=config.COPILOT_VERIFICATION_TIMEOUT_SECONDS)
    return setup, parse_copilot_verdict(raw), raw


def _describe_elapsed(generated_utc: str | None, now_utc: datetime) -> str:
    """Human-readable elapsed time since the mega session that produced
    this Pending Setup — fed into the verdict prompt so Copilot can
    reason about genuine staleness (a real delay: a missed poll, the PC
    being off, hours of drift) rather than just "not yet triggered"."""
    if not generated_utc:
        return "unknown"
    try:
        generated_dt = datetime.fromisoformat(generated_utc)
    except ValueError:
        return "unknown"
    elapsed = now_utc - generated_dt
    total_minutes = max(0, int(elapsed.total_seconds() // 60))
    hours, minutes = divmod(total_minutes, 60)
    if hours == 0:
        return f"{minutes} minute(s)"
    return f"{hours} hour(s) {minutes} minute(s)"


def _mega_session_is_live(mega_progress: dict, mega_state: dict) -> bool:
    """Same freshness heuristic app.py's own countdown widget already
    uses (progress timestamp fresher than 5 minutes AND newer than the
    last COMPLETED attempt) — reused here so this job never acts
    mid-transition while a mega session is actively writing its own
    latest-suggestion file."""
    progress_ts = mega_progress.get("updated_utc")
    if not progress_ts:
        return False
    try:
        progress_dt = datetime.fromisoformat(progress_ts)
    except ValueError:
        return False
    now_utc = datetime.now(timezone.utc)
    age_seconds = (now_utc - progress_dt).total_seconds()
    last_attempt = mega_state.get("last_attempt_utc")
    newer_than_last_attempt = last_attempt is None or progress_ts > last_attempt
    return 0 <= age_seconds < 300 and newer_than_last_attempt


def run_copilot_execution_check(on_stage: Callable[[str], None] | None = None) -> None:
    """The full execution-check orchestration — see this module's own docstring
    for the design this implements. Never raises for an anticipated
    failure mode (a missing suggestion, a blocked safety gate, an
    unreachable MT5 terminal all resolve to a logged, recorded outcome);
    a genuinely unexpected exception is left to propagate so the caller
    (copilot_execution_job.py, wrapped in run_with_timeout) can record
    it distinctly, matching ai.mega_analysis.run_scheduled_mega_analysis's
    own "never silently swallow a real failure" convention."""

    def _notify(message: str) -> None:
        logger.info(message)
        _write_execution_progress(message)
        if on_stage:
            on_stage(message)

    if not read_copilot_execution_enabled():
        _notify("Copilot Execution Clerk is disabled via the app's toggle — skipping this check.")
        _write_execution_state("disabled")
        return

    if _mega_session_is_live(read_mega_progress(), read_mega_state()):
        _notify("A mega analysis session is currently running — skipping this check.")
        _write_execution_state("skipped_mega_live")
        return

    suggestion = read_latest_suggestion()
    if not suggestion:
        _notify("No mega-analysis suggestion on file yet — nothing to check.")
        _write_execution_state("no_suggestion")
        return

    generated_utc = suggestion.get("generated_utc")
    immediate_allocation_raw = suggestion.get("immediate_allocation", {})
    pending_setups_raw = suggestion.get("pending_setups", [])
    pending_setups = [
        PendingSetup(
            symbol=s["symbol"], side=s["side"], pct=s["pct"],
            trigger_condition=s["trigger_condition"], price=s.get("price"),
            stop_loss=s.get("stop_loss"), take_profit=s.get("take_profit"),
            reason=s.get("reason", ""),
        )
        for s in pending_setups_raw
    ]

    _notify("Connecting to the FTMO MT5 account...")
    connect(login=config.FTMO_MT5_LOGIN, password=config.FTMO_MT5_PASSWORD, server=config.FTMO_MT5_SERVER)

    settlement = _load_settlement()
    if settlement.get("generated_utc") != generated_utc:
        _notify("A new mega session superseded the prior one — resetting pending-setup tracking...")
        settled = _reset_settlement_for_new_session(settlement.get("settled", {}))
        settlement = {"generated_utc": generated_utc, "settled": settled}

    _notify("Fetching FTMO account and market data...")
    account = get_account_summary()
    positions = get_open_positions()
    pending_orders = get_pending_orders()
    assets = get_market_watch()
    market_prices = {a.symbol: a for a in assets}

    settlement["settled"] = _reconcile_settlement(settlement.get("settled", {}), positions, pending_orders)
    _save_settlement(settlement)

    carried_forward = _build_carried_forward_allocation(immediate_allocation_raw, settlement["settled"])

    not_yet_settled = [s for s in pending_setups if s.symbol not in settlement["settled"]]
    if len(not_yet_settled) > config.COPILOT_EXECUTION_MAX_PENDING_SETUPS:
        logger.warning(
            "%d not-yet-settled Pending Setups exceeds the cap of %d — checking only the first %d.",
            len(not_yet_settled), config.COPILOT_EXECUTION_MAX_PENDING_SETUPS,
            config.COPILOT_EXECUTION_MAX_PENDING_SETUPS,
        )
        not_yet_settled = not_yet_settled[: config.COPILOT_EXECUTION_MAX_PENDING_SETUPS]

    last_verdicts: dict = {}
    merged_allocation = dict(carried_forward)
    if not_yet_settled:
        _notify(f"Checking {len(not_yet_settled)} pending setup(s) against live technicals and Copilot...")
        # Phase 1 (sequential, MT5-bound): fetch every not-yet-settled
        # setup's live technicals one at a time — never in parallel, see
        # _fetch_technical_context's own docstring.
        checkable: list[tuple[PendingSetup, str]] = []
        for setup in not_yet_settled:
            technical_context = _fetch_technical_context(setup, market_prices, account.equity)
            if technical_context is None:
                logger.warning("%s is not currently visible in Market Watch — skipped.", setup.symbol)
                continue
            checkable.append((setup, technical_context))

        # Phase 2 (parallel, Copilot-bound): pure CLI subprocess calls,
        # no MT5 involvement — bounds wall-clock to roughly the slowest
        # single call rather than their sum, mirroring
        # ai.portfolio_suggest.build_audit_block's own proven pattern.
        if checkable:
            elapsed_description = _describe_elapsed(generated_utc, datetime.now(timezone.utc))
            with ThreadPoolExecutor(max_workers=len(checkable)) as pool:
                futures = [
                    pool.submit(_run_copilot_verdict, setup, technical_context, account.equity, elapsed_description)
                    for setup, technical_context in checkable
                ]
                results = [f.result() for f in futures]
            # Merged on the main thread, in original list order, after
            # every future resolves — never inside a worker thread — so
            # completion order can never make this non-deterministic.
            for setup, confirmed, raw in results:
                last_verdicts[setup.symbol] = {
                    "confirmed": confirmed, "raw_text": raw,
                    "checked_utc": datetime.now(timezone.utc).isoformat(),
                }
                # The full reasoning text is logged here too (direct user
                # request 2026-08-23: "also write this in a log
                # somewhere") — previously it only ever landed in
                # copilot_execution_state.json's last_verdicts, never in
                # the actual human-readable log file.
                logger.info(
                    "Copilot verdict for %s: %s — %s", setup.symbol,
                    "CONFIRMED" if confirmed else "NOT_CONFIRMED",
                    raw[:1000] if raw else "(no response text)",
                )
                if confirmed:
                    merged_allocation[setup.symbol] = AllocationEntry(
                        pct=setup.pct, price=setup.price, stop_loss=setup.stop_loss,
                        take_profit=setup.take_profit, side=setup.side,
                    )

    planned_heat_pct = compute_aggregate_heat_pct(merged_allocation)
    ftmo_status = fetch_ftmo_status(account)
    ftmo_heat_blocked = would_breach_daily_loss_headroom(ftmo_status, planned_heat_pct)
    ftmo_heat_blocked_reason = ""
    if ftmo_heat_blocked:
        allowed_pct = max(0.0, ftmo_status.daily_loss_headroom_pct) * DEFAULT_HEADROOM_FRACTION
        ftmo_heat_blocked_reason = (
            f"Execution blocked: this plan's aggregate heat ({planned_heat_pct:.2f}% of "
            f"equity at risk if every stop is hit) would exceed {DEFAULT_HEADROOM_FRACTION:.0%} "
            f"of this account's REAL remaining daily-loss headroom ({ftmo_status.daily_loss_headroom_pct:.2f}%)."
        )

    is_demo = bool(config.FTMO_MT5_SERVER) and "demo" in config.FTMO_MT5_SERVER.lower()
    trading_permitted, trading_blocked_reason = (
        (True, "") if config.USE_MOCK_DATA else is_trading_permitted()
    )
    execution_ok, block_reason = check_execution_safety_gates(
        is_demo=is_demo,
        allow_live_execution=config.ALLOW_LIVE_EXECUTION,
        trading_permitted=trading_permitted,
        trading_blocked_reason=trading_blocked_reason,
        ftmo_heat_blocked=ftmo_heat_blocked,
        ftmo_heat_blocked_reason=ftmo_heat_blocked_reason,
    )
    if not execution_ok:
        _notify(f"Execution blocked this poll: {block_reason}")
        _write_execution_state("blocked", block_reason, last_verdicts=last_verdicts)
        _mark_interval_ran(datetime.now(timezone.utc))
        return

    plan = compute_rebalance_plan(
        positions, account, merged_allocation, get_contract_spec, market_prices,
        price_sanity_band_pct=config.PRICE_SANITY_BAND_PCT,
    )

    executed_count = 0
    # One entry per symbol in THIS plan, regardless of outcome — direct
    # user request 2026-08-23 ("I see position in 5 assets but only 2
    # assets are showing up in the clerk section"): a FAILED open/close
    # never gets a settlement record (that only tracks successfully-
    # placed orders), so without this, app.py's panel had no way to show
    # why a symbol didn't get an order — only the log file did.
    results_this_poll: dict[str, dict] = {}
    checked_utc = datetime.now(timezone.utc).isoformat()

    def _record_result(symbol: str, action: str, success: bool, detail: str) -> None:
        results_this_poll[symbol] = {
            "action": action, "success": success, "detail": detail, "checked_utc": checked_utc,
        }

    for o in plan:
        if o.action in ("open", "increase"):
            existing = settlement["settled"].get(o.symbol)
            if existing is not None and existing.get("state") == "order_placed":
                # An order for this exact symbol is already outstanding
                # (see _build_carried_forward_allocation's own docstring
                # for why that symbol is still deliberately in the plan
                # rather than hidden) — skip sending a second, duplicate
                # order on top of it. The next poll re-evaluates fresh
                # once this one fills, gets rejected, or gets cancelled.
                logger.info(
                    "%s (%s): skipped — an order for this symbol is already outstanding (ticket %s).",
                    o.symbol, o.action, existing.get("order_ticket"),
                )
                _record_result(o.symbol, o.action, True, f"already has an outstanding order (ticket {existing.get('order_ticket')})")
                continue
            try:
                result = open_position(o.symbol, o.side, o.volume, o.price, o.stop_loss, o.take_profit)
            except MT5ConnectionError as e:
                result = OrderResult(False, None, str(e), None)
            logger.info("%s (%s): %s", o.symbol, o.action, "placed" if result.success else f"failed ({result.comment})")
            _record_result(o.symbol, o.action, result.success, "placed" if result.success else result.comment)
            if result.success:
                executed_count += 1
                origin = "pending_setup" if o.symbol in last_verdicts else "immediate"
                settlement["settled"][o.symbol] = asdict(
                    SymbolSettlement(
                        origin=origin, state="order_placed",
                        entry={
                            "pct": merged_allocation[o.symbol].pct,
                            "price": o.price,
                            "stop_loss": o.stop_loss,
                            "take_profit": o.take_profit,
                            "side": o.side,
                        },
                        order_ticket=result.ticket,
                    )
                )
        elif o.action in ("reduce", "close"):
            any_failed = False
            last_detail = ""
            for ticket, ticket_volume in o.tickets_to_close:
                matching = next((p for p in positions if p.ticket == ticket), None)
                if matching is None:
                    logger.warning("%s (%s): ticket %d not found among fetched positions.", o.symbol, o.action, ticket)
                    any_failed = True
                    last_detail = f"ticket {ticket} not found among fetched positions"
                    continue
                try:
                    result = close_position(matching, volume=ticket_volume)
                except MT5ConnectionError as e:
                    result = OrderResult(False, None, str(e), None)
                logger.info("%s (%s): %s", o.symbol, o.action, "closed" if result.success else f"failed ({result.comment})")
                if result.success:
                    executed_count += 1
                else:
                    any_failed = True
                    last_detail = result.comment
            _record_result(o.symbol, o.action, not any_failed, "closed" if not any_failed else last_detail)
        elif o.action == "infeasible":
            # Silent in the manual dialog too (the human sees it in the
            # preview table instead) — but nobody reviews a preview table
            # here, so this is the ONLY place this reason would ever
            # surface for an unattended run. Logged, not raised: an
            # infeasible entry is an expected, valid plan outcome (e.g.
            # "can't afford the minimum lot at this risk%"), not a
            # failure of this job itself.
            logger.info("%s (infeasible): %s", o.symbol, o.reason)
            _record_result(o.symbol, o.action, False, o.reason)
        else:
            _record_result(o.symbol, o.action, True, "on target, no change needed")

    _save_settlement(settlement)
    _notify(f"Execution-check complete — {executed_count} order(s) sent this poll.")
    _write_execution_state(
        "success", f"{executed_count} order(s) sent",
        last_verdicts=last_verdicts, last_execution_results=results_this_poll,
    )
    _mark_interval_ran(datetime.now(timezone.utc))
