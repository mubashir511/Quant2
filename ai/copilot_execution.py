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
the panel's own "next review" countdown).

2026-08-23/24 update — Claude and Copilot can now re-assess an ALREADY-
suggested position, not just propose fresh ones (direct user request:
"claude should be able to scan the existing suggested positions... and
comment on these positions whether to keep them, update it... or
completely stop the active order or drop the limit order", with
Copilot's own periodic judgment explicitly bounded to "the thesis from
the mega session" rather than an open-ended second opinion). Two new
optional `AllocationEntry` fields, `reason` and `invalidation_condition`
(mirroring `PendingSetup`'s own `reason`/`trigger_condition`), let
Claude's daily mega session write a per-position thesis and a specific,
mechanically-checkable exit condition. `compute_rebalance_plan`
(risk/apply_suggestion.py) is now pending-order-aware and can emit three
new actions handled by the execution loop below: `"cancel"` (a resting
pending order the fresh target no longer wants), `"amend_pending"`
(cancel + reopen at new terms — safe, since an unfilled order has no
realized exposure), and `"amend_position"` (a true in-place SL/TP amend
via the new data/mt5_execution.py::modify_position_sltp, deliberately
never a close-then-reopen — see that function's own docstring for why).
`watched_positions` (built in run_copilot_execution_check, just before
the existing Pending-Setups verdict phase) collects every already-
settled (`order_placed`/`filled`) immediate_allocation symbol carrying a
non-empty `invalidation_condition`, fetches its live technicals the same
sequential way Pending Setups already do, and runs a NEW
`_build_invalidation_prompt`/`_run_copilot_invalidation_check` pair in
the SAME parallel ThreadPoolExecutor phase as the existing Pending-
Setups verdict calls — deliberately a mechanical trip-wire check on
Claude's own stated condition, never Copilot's own fresh opinion on
whether the trade "still feels right." On CONFIRMED, this forces the
symbol's `pct` to 0 in `merged_allocation` rather than calling
`cancel_pending_order`/`close_position` directly — `compute_rebalance_
plan` stays the ONE place that ever actually decides to cancel/close, so
every existing safety gate still runs first automatically, and a same-
poll collision with a brand-new mega session's own fresh `pct: 0` for
the same symbol resolves unambiguously ("0% wins") instead of two
independent execution paths racing.

Added 2026-08-25, direct user request, after the `copilot` CLI's
monthly quota was confirmed genuinely exhausted (every real verdict/
invalidation call that day failing with "You have exceeded your
monthly quota"), leaving the whole mechanism above dark: both
`_run_copilot_verdict` and `_run_copilot_invalidation_check` now call a
shared `_run_clerk_prompt` helper instead of `run_copilot` directly.
Copilot is still tried first; only if it's genuinely unavailable
(quota/auth/crash — never for a real NOT_CONFIRMED, which is a legitimate
answer, not a failure) does it fall through `COPILOT_BACKUP_MODELS`, 5
free OpenRouter models (reusing ai/openrouter_client.py, the same client
the audit pool already relies on) in a fixed priority order, first
success wins. That list was chosen by reading the actual last 4 FTMO
mega-session audit transcripts for real per-model availability, not by
trusting AUDIT_MODELS' own capacity ranking blindly — see
COPILOT_BACKUP_MODELS' own comment for the exact reasoning and which 3
of AUDIT_MODELS' 10 models were excluded for unreliability. This makes
6 possible reviewers total for the same bounded check. A backup loses
Copilot's own live web access (the "check for breaking news" angle both
prompts invite); it still fully judges the stated condition against the
real MT5 technical context already in the prompt."""

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
from ai.openrouter_client import FAILED_MESSAGE as OPENROUTER_FAILED_MESSAGE
from ai.openrouter_client import MISSING_KEY_MESSAGE as OPENROUTER_MISSING_KEY_MESSAGE
from ai.openrouter_client import run_openrouter
from ai.ftmo_suggest import (
    analyze_ftmo_asset_live,
    fetch_ftmo_status,
    format_ftmo_asset_context,
    read_latest_suggestion,
)
from ai.mega_analysis import mega_session_is_live as _mega_session_is_live
from ai.mega_analysis import read_progress as read_mega_progress
from ai.mega_analysis import read_state as read_mega_state
from ai.portfolio_suggest import AllocationEntry, PendingSetup
from data.book_wisdom import format_trend_wisdom
from data.mt5_execution import (
    MT5ConnectionError,
    OrderResult,
    cancel_pending_order,
    close_position,
    modify_position_sltp,
    open_position,
)
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
from risk.apply_suggestion import (
    check_execution_safety_gates,
    compute_aggregate_heat_pct,
    compute_rebalance_plan,
    pct_for_target_lots,
)
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
    # {"last_action_utc", "last_stop_loss", "adverse_move_pct_at_last_action",
    # "defend_count", "tier_reached"} — the Clerk's own tactical-defense
    # history for this symbol, added 2026-08-27 (see this module's own
    # docstring for the real gold-trade incident that motivated it).
    # Default-safe: every existing construction site above stays valid
    # unmodified, and an existing JSON settlement record simply reads
    # back with this key absent (falsy checks already treat that as "no
    # prior tactical action").
    tactical: dict | None = None


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
    last_tactical_verdicts: dict | None = None,
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
    what happened to the actual immediate_allocation targets.

    `last_tactical_verdicts` — added 2026-08-27 for the Clerk's new
    tactical-defense authority (see this module's own docstring) — kept
    as its OWN top-level key, merged the same way, rather than folded
    into `last_verdicts`: a tactical verdict's shape (tier/new_stop_loss/
    partial_close_fraction/citations) is richer than last_verdicts' own
    plain {"confirmed": bool} shape, and conflating them would break
    existing `verdict.get("confirmed")` reads elsewhere."""
    prior = read_execution_state()
    now = datetime.now(timezone.utc)
    merged_verdicts = {**prior.get("last_verdicts", {}), **(last_verdicts or {})}
    merged_results = {**prior.get("last_execution_results", {}), **(last_execution_results or {})}
    merged_tactical_verdicts = {
        **prior.get("last_tactical_verdicts", {}), **(last_tactical_verdicts or {})
    }
    payload = {
        "last_attempt_utc": now.isoformat(),
        "last_status": status,
        "last_detail": detail,
        "last_verdicts": merged_verdicts,
        "last_execution_results": merged_results,
        "last_tactical_verdicts": merged_tactical_verdicts,
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


def execution_check_is_live(progress: dict, state: dict) -> bool:
    """True iff `progress` (from read_execution_progress()) reflects an
    execution-check poll that's still genuinely in flight — same shape
    of check as ai.mega_analysis.mega_session_is_live (see its own
    docstring for the full reasoning), applied here to this job's own
    progress/state files instead. Found live 2026-08-27 while fixing
    that exact bug for the mega session: app.py's own execution-check
    panel had an independent THIRD copy of the identical hardcoded
    300-second cutoff, which could just as easily go stale mid-run —
    this job's own verdict checks can call an OpenRouter backup chain
    (confirmed live the same day: all 10 free OpenRouter models can go
    down and retry for several minutes at once) — and wrongly hide the
    live-progress banner while a check was still genuinely running.
    Consolidated to one ceiling, config.COPILOT_EXECUTION_RUN_TIMEOUT_
    SECONDS — the same hard ceiling this job's own single pass is
    already bounded by, so this can never time out a check that's still
    within its own allowed budget."""
    progress_ts = progress.get("updated_utc")
    if not progress_ts:
        return False
    try:
        progress_dt = datetime.fromisoformat(progress_ts)
    except ValueError:
        return False
    age_seconds = (datetime.now(timezone.utc) - progress_dt).total_seconds()
    last_attempt = state.get("last_attempt_utc")
    newer_than_last_attempt = last_attempt is None or progress_ts > last_attempt
    return 0 <= age_seconds < config.COPILOT_EXECUTION_RUN_TIMEOUT_SECONDS and newer_than_last_attempt


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


def read_tactical_defense_enabled() -> bool:
    """Whether the Clerk's tactical-defense authority (DEFEND/EXIT on an
    already-filled position's short-term "trend" read, independent of
    Claude's own invalidation_condition) is enabled — defaults to FALSE,
    a deliberate break from read_copilot_execution_enabled's own usual
    opt-out-not-opt-in posture: this is fresh unattended authority over
    real money, added 2026-08-27 direct user request after a real gold
    position went from +$28 to -$61 while the mega session hadn't run in
    days and the Clerk had zero authority to react (see this module's
    own docstring). Checked separately from the whole-Clerk enabled
    toggle above so the long-proven Pending-Setup/invalidation
    mechanisms keep running unaffected by this new, separately-staged
    one — see run_copilot_execution_check's own shadow-mode behavior
    when this is False: the full tactical pipeline still runs and
    reports every poll, it just never touches merged_allocation."""
    path = Path(config.COPILOT_TACTICAL_DEFENSE_ENABLED_FILE)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("enabled", False))


def set_tactical_defense_enabled(enabled: bool) -> None:
    try:
        Path(config.COPILOT_TACTICAL_DEFENSE_ENABLED_FILE).write_text(json.dumps({"enabled": enabled}))
    except OSError as e:
        logger.warning(
            "Could not write tactical-defense enabled-flag file %s: %s",
            config.COPILOT_TACTICAL_DEFENSE_ENABLED_FILE, e,
        )


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


def _backfill_settlement_for_held_positions(
    settled: dict,
    positions: list[Position],
    immediate_allocation_raw: dict,
    account_equity: float,
    get_spec: Callable,
) -> dict:
    """Found on audit (2026-08-27): _reset_settlement_for_new_session
    wipes ALL tracking — including "filled" records — on every new mega
    session, by its own explicit design ("the reset only clears
    TRACKING, it never touches an already-filled real position"). But
    neither "hold" nor "amend_position" (see run_copilot_execution_
    check's own execution loop) ever writes a FRESH settlement record —
    only "open"/"increase"/"amend_pending" do. So a position the fresh
    mega session simply reaffirms unchanged (a common, even typical
    outcome for an already-well-managed position) would have NO
    settlement record at all for the rest of that mega-session cycle,
    making it invisible to BOTH the invalidation-condition watcher AND
    the tactical-defense check above — precisely the "Clerk has zero
    authority over an already-open position between mega sessions" gap
    this whole initiative exists to close (see this module's own
    docstring for the real gold-trade incident). Called right after
    _reconcile_settlement, before anything else reads settlement, so
    every real held position always has at least a "filled" record by
    the time watched_positions/tactical_candidates are built.

    A symbol the fresh suggestion still mentions gets a record sourced
    from that FRESH entry (so reason/invalidation_condition stay
    genuinely current, not stale). A symbol NOT mentioned (a Pending-
    Setup-originated position the fresh session simply didn't comment on
    this cycle) gets its `pct` back-computed via pct_for_target_lots so
    it EXACTLY reproduces the position's own already-held size at its
    own current stop — a safe, conservative default (compute_rebalance_
    plan then resolves it to "hold", not an unintended resize purely
    from a backfilled guess). If that sizing can't be done (no usable
    spec/stop), the symbol is left unbackfilled rather than writing an
    unusable record — degrading to this same gap's own prior, non-
    crashing behavior for that one symbol, never a crash."""
    held_symbols = {p.symbol for p in positions}
    backfilled = dict(settled)
    for symbol in held_symbols:
        if symbol in backfilled or symbol.upper() == "CASH":
            continue
        raw = immediate_allocation_raw.get(symbol)
        if raw is not None:
            entry = {
                "pct": raw.get("pct"), "price": raw.get("price"), "stop_loss": raw.get("stop_loss"),
                "take_profit": raw.get("take_profit"), "side": raw.get("side", "buy"),
                "reason": raw.get("reason", ""), "invalidation_condition": raw.get("invalidation_condition"),
            }
            backfilled[symbol] = asdict(
                SymbolSettlement(origin="immediate", state="filled", entry=entry, order_ticket=None)
            )
            continue

        position = next((p for p in positions if p.symbol == symbol), None)
        if position is None or position.sl is None:
            continue
        # position.price_current, not price_open -- this backfilled
        # entry feeds straight into compute_rebalance_plan THIS SAME
        # poll (via carried_forward), which re-clamps whatever price it
        # carries to within the live sanity band; price_open (fixed at
        # whenever the position first opened, possibly long ago) risks
        # exactly the same stale-price/wrong-stop-distance mismatch
        # found and fixed in _validate_and_apply_tactical_verdict above.
        pct = pct_for_target_lots(symbol, position.volume, position.price_current, position.sl, account_equity, get_spec)
        if pct is None:
            continue
        entry = {
            "pct": pct, "price": position.price_current, "stop_loss": position.sl,
            "take_profit": position.tp, "side": position.side, "reason": "", "invalidation_condition": None,
        }
        backfilled[symbol] = asdict(
            SymbolSettlement(origin="pending_setup", state="filled", entry=entry, order_ticket=None)
        )
    return backfilled


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
        reason=d.get("reason", ""),
        invalidation_condition=d.get("invalidation_condition"),
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
    symbol: str, market_prices: dict, account_equity: float
) -> str | None:
    """The MT5-bound half of checking one symbol — fetches its live
    4-timeframe technicals (analyze_ftmo_asset_live) and formats them
    into the same rich narrative Asset Health/the mega session's own
    prompt already use. MUST be called sequentially by the caller, once
    per not-yet-settled setup or watched position, never from inside a
    thread pool — the MetaTrader5 Python API isn't documented thread-
    safe, and this project's whole connection model already assumes one
    connection, called sequentially. Returns None if the symbol isn't
    currently visible in Market Watch (skipped, logged by the caller)
    rather than raising. Takes a bare symbol (not a whole PendingSetup)
    since 2026-08-23 — the invalidation-check phase (see
    _build_invalidation_prompt) needed this same fetch for an already-
    settled immediate_allocation symbol, which has no PendingSetup
    object at all."""
    asset = market_prices.get(symbol)
    if asset is None:
        return None
    analysis = analyze_ftmo_asset_live(symbol, asset.bid, asset.ask, asset.description)
    return format_ftmo_asset_context([analysis], account_equity=account_equity)


# Backup reviewers for the verdict/invalidation checks below, tried in
# this exact order ONLY when Copilot itself is genuinely unavailable
# (quota/auth/crash — never for a real NOT_CONFIRMED answer). Selected
# 2026-08-25, direct user request, after the `copilot` CLI's monthly
# quota was confirmed exhausted (every real call this session failing
# with "You have exceeded your monthly quota") left the whole verdict
# mechanism dark. Chosen from AUDIT_MODELS' own 10-model OpenRouter pool
# by reading the actual last 4 FTMO mega-session audit transcripts
# (records/ftmo/portfolio_suggestion_2026-08-{20,22,23,25}*.md) rather
# than trusting AUDIT_MODELS' own capacity ranking blindly: 7 of the 10
# answered successfully in EVERY one of those 4 real runs (openai/gpt-
# oss-20b was down in all 4; google/gemma-4-26b-a4b-it was down in 3 of
# 4; nvidia/nemotron-3-nano-30b-a3b was down in 1 of 4 — all three
# excluded here for availability, not capacity). Of those 7 reliable
# ones, these are the top 5 by AUDIT_MODELS' own declared-parameter-
# count proxy, highest first, matching the user's explicit "highest
# capacity available model to lowest" ordering — this makes 6 possible
# reviewers total for the same bounded check: Copilot first, then these
# 5 in order, first success wins. Re-verify against fresh mega-session
# transcripts before trusting this list again if it's ever revisited —
# the same "this roster churns, don't rely on memory" caveat AUDIT_
# MODELS' own comment already makes applies here too.
COPILOT_BACKUP_MODELS: list[tuple[str, str]] = [
    ("Nvidia Nemotron-Ultra-550B", "nvidia/nemotron-3-ultra-550b-a55b:free"),
    ("Nvidia Nemotron-Super-120B", "nvidia/nemotron-3-super-120b-a12b:free"),
    ("Dots Studio Dots3-Note Preview", "dots-studio/dots-3-note-preview:free"),
    ("Nvidia Nemotron-Nano-Omni-30B-Reasoning", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"),
    ("Poolside Laguna S 2.1", "poolside/laguna-s-2.1:free"),
]


def _is_copilot_unavailable(raw: str) -> bool:
    return raw == COPILOT_MISSING_MESSAGE or raw.startswith(COPILOT_FAILED_PREFIX)


def _is_openrouter_unavailable(raw: str) -> bool:
    return raw == OPENROUTER_MISSING_KEY_MESSAGE or raw == OPENROUTER_FAILED_MESSAGE


def _run_clerk_prompt(prompt: str, timeout: int) -> str:
    """Runs `prompt` through Copilot first (the primary clerk) and, only
    if Copilot is genuinely unavailable — not a real NOT_CONFIRMED
    answer, which is a legitimate verdict, not a failure — falls through
    COPILOT_BACKUP_MODELS in priority order via the same free OpenRouter
    pool the audit phase already uses, stopping at the first one that
    responds. The exact same prompt (built for Copilot) is reused
    unchanged for a backup: it already asks for a plain FINAL_VERDICT:
    CONFIRMED/NOT_CONFIRMED line, a format any capable instruction-
    following model can honor, so parse_copilot_verdict works unmodified
    on a backup's response too.

    Known, accepted trade-off: a backup loses Copilot's own live web
    access, so the "check for breaking news since the analysis ran"
    angle both verdict prompts explicitly invite Copilot to use is
    unavailable on a backup — it still mechanically judges the stated
    condition against the real, live MT5 technical context already
    embedded in the prompt, just without that one extra check. Not
    fixed here: none of AUDIT_MODELS' free OpenRouter models have live
    web access either (see AUDIT_MODELS' own docstring note).

    If every backup also fails, returns Copilot's own original failure
    message (the most informative single failure to surface), and
    parse_copilot_verdict's existing fail-safe handling of that sentinel
    still applies — a total outage still resolves to NOT_CONFIRMED,
    never a silent guess."""
    raw = run_copilot(prompt, timeout=timeout)
    if not _is_copilot_unavailable(raw):
        return raw
    for label, model in COPILOT_BACKUP_MODELS:
        backup_raw = run_openrouter(prompt, model=model, timeout=config.OPENROUTER_TIMEOUT_SECONDS)
        if not _is_openrouter_unavailable(backup_raw):
            return f"[Copilot unavailable — backup reviewer {label} responded]\n\n{backup_raw}"
    return raw


def _run_copilot_verdict(
    setup: PendingSetup, technical_context: str, account_equity: float, elapsed_description: str
) -> tuple[PendingSetup, bool, str]:
    """The Copilot-bound half — pure CLI subprocess call (with an
    OpenRouter backup chain, see _run_clerk_prompt), no MT5 involvement,
    safe to run from inside a thread pool (see
    run_copilot_execution_check's own parallel round). Returns (setup,
    confirmed, raw_copilot_text) — the raw text always travels back for
    the audit-trail log/UI, even on a NOT_CONFIRMED or failed verdict."""
    prompt = _build_verdict_prompt(setup, technical_context, account_equity, elapsed_description)
    raw = _run_clerk_prompt(prompt, timeout=config.COPILOT_VERIFICATION_TIMEOUT_SECONDS)
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


def _build_invalidation_prompt(
    symbol: str,
    entry: dict,
    invalidation_condition: str,
    technical_context: str,
    account_equity: float,
    elapsed_description: str,
    position_state: str,
) -> str:
    """The exit-side counterpart to _build_verdict_prompt (added
    2026-08-23, direct user request: Claude's daily mega session writes
    an explicit invalidation_condition per already-suggested position —
    same style as a Pending Setup's own trigger_condition — and this
    lets Copilot mechanically re-check ONLY that stated condition every
    poll, bounded exactly as confirmed with the user: a trip-wire on one
    specific condition, never an open-ended fresh opinion on the trade.
    `entry` is the raw immediate_allocation dict for this symbol (pct/
    price/stop_loss/take_profit/side/reason — see
    ai/ftmo_suggest.py::_write_latest_suggestion's own payload shape)."""
    holding_description = (
        "a still-UNFILLED pending limit order (not yet a real open position)"
        if position_state == "order_placed"
        else "a REAL, currently OPEN position"
    )
    return (
        "You are an automated trading clerk for an FTMO account. A senior "
        "analyst (Claude) placed the following trade during today's mega "
        "market analysis and wrote down a SPECIFIC condition that would "
        f"mean this trade should now be abandoned. This is {holding_description}.\n\n"
        f"Symbol: {symbol}\n"
        f"Side: {entry.get('side')}\n"
        f"Entry/trigger price: {entry.get('price')}\n"
        f"Stop-loss: {entry.get('stop_loss')}\n"
        f"Take-profit: {entry.get('take_profit')}\n"
        f"Reason given (from the senior analyst): {entry.get('reason', '')}\n"
        f"Invalidation condition (from the senior analyst): {invalidation_condition}\n"
        f"Time since that analysis was run: {elapsed_description}.\n\n"
        "Below is this symbol's REAL, LIVE technical picture right now "
        "(the same data source that powers this account's own Asset "
        "Health analysis). Your ONLY job is to mechanically check whether "
        "the SPECIFIC invalidation condition above has genuinely occurred "
        "— nothing more. Do NOT exit this position just because you "
        "personally would have managed it differently, because you have a "
        "new, independent opinion on the trade, or because the market has "
        "moved against it without the stated condition itself being met — "
        "none of those are grounds for CONFIRMED here. This position was "
        "already deliberately sized and risk-managed by the senior "
        "analyst's own full research process; you are a mechanical "
        "trip-wire check on ONE stated condition, not a second opinion on "
        "the trade itself. If the live technical picture below is "
        "inconclusive or only partially matches the stated condition, "
        "that means NOT_CONFIRMED — the condition must have genuinely, "
        "unambiguously occurred, not merely be trending toward occurring.\n\n"
        f"{technical_context}\n\n"
        f"Account equity for context: {account_equity:.2f}. This does not "
        "change your verdict on the condition itself.\n\n"
        "End your response with exactly one line, and nothing after it: "
        'either "FINAL_VERDICT: CONFIRMED" (the invalidation condition has '
        'genuinely occurred — this position should be exited/cancelled '
        'now) or "FINAL_VERDICT: NOT_CONFIRMED" (the condition has not '
        "occurred — keep holding/watching this position unchanged). Do "
        "not use this exact token anywhere else in your response."
    )


def _run_copilot_invalidation_check(
    symbol: str,
    entry: dict,
    invalidation_condition: str,
    technical_context: str,
    account_equity: float,
    elapsed_description: str,
    position_state: str,
) -> tuple[str, bool, str]:
    """The Copilot-bound half of an invalidation check — pure CLI
    subprocess call (with an OpenRouter backup chain, see
    _run_clerk_prompt), no MT5 involvement, safe to run from inside a
    thread pool (see run_copilot_execution_check's own parallel round).
    Returns (symbol, confirmed, raw_copilot_text) — deliberately a bare
    symbol string in position 0, not a PendingSetup, so the merge loop
    can tell the two verdict kinds apart via isinstance()."""
    prompt = _build_invalidation_prompt(
        symbol, entry, invalidation_condition, technical_context,
        account_equity, elapsed_description, position_state,
    )
    raw = _run_clerk_prompt(prompt, timeout=config.COPILOT_VERIFICATION_TIMEOUT_SECONDS)
    return symbol, parse_copilot_verdict(raw), raw


# --- tactical-defense check: bounded DEFEND/EXIT authority between mega
# sessions (added 2026-08-27, direct user request after a real gold
# position went from +$28 to -$61 while the mega session hadn't run in
# days and the Clerk had zero authority to react — see this module's own
# docstring). Unlike the invalidation check above (a mechanical trip-
# wire on Claude's OWN stated condition), this is the Clerk's own
# judgment on the SHORT-TERM/"trend" book-wisdom tier (data/book_wisdom.
# py::format_trend_wisdom), tiered exactly as confirmed with the user:
# DEFEND (tighten a stop and/or take a partial profit) is the default,
# preferred response; EXIT (full close) is an escalation, not a first
# resort. Every DEFEND/EXIT must cite a specific book rule and specific
# real numbers — parse_tactical_verdict fails safe to HOLD otherwise. ---


@dataclass
class TacticalVerdict:
    tier: str  # "hold" | "defend" | "exit"
    new_stop_loss: float | None = None
    partial_close_fraction: float | None = None
    rule_citation: str = ""
    numbers_citation: str = ""
    raw_text: str = ""


_TACTICAL_VERDICT_PATTERN = re.compile(r"FINAL_VERDICT:\s*(HOLD|DEFEND|EXIT)", re.IGNORECASE)
_TACTICAL_NEW_STOP_PATTERN = re.compile(r"NEW_STOP_LOSS:\s*([^\n\r]*)", re.IGNORECASE)
_TACTICAL_PARTIAL_FRACTION_PATTERN = re.compile(r"PARTIAL_CLOSE_FRACTION:\s*([^\n\r]*)", re.IGNORECASE)
_TACTICAL_RULE_PATTERN = re.compile(r"RULE:\s*([^\n\r]*)", re.IGNORECASE)
_TACTICAL_NUMBERS_PATTERN = re.compile(r"NUMBERS:\s*([^\n\r]*)", re.IGNORECASE)


_LEADING_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_optional_tactical_float(raw: str | None) -> float | None:
    """Extracts the first number on the line rather than requiring the
    ENTIRE remainder to be a bare float — found on audit: a model that
    otherwise follows the prompt's format can still append a short
    parenthetical ("1960.0 (tightened from 1950)"), which a strict
    `float(raw)` would reject outright, silently losing an otherwise
    well-formed DEFEND to the fail-safe HOLD. A genuinely wrong number
    extracted this way is still caught downstream by
    _validate_and_apply_tactical_verdict's own never-widen-stop and
    sizing checks — this only widens what counts as parseable, it never
    widens what counts as safe to apply."""
    if raw is None:
        return None
    match = _LEADING_NUMBER_PATTERN.search(raw)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def parse_tactical_verdict(response_text: str) -> TacticalVerdict:
    """Extracts a HOLD/DEFEND/EXIT tactical verdict from Copilot's (or a
    backup model's) free-text response — the DEFEND/EXIT counterpart to
    parse_copilot_verdict's own hard boolean. Fails safe to HOLD (never
    DEFEND/EXIT) on every ambiguous branch, mirroring parse_copilot_
    verdict's own discipline, rather than ever guessing at a real-money
    action: the CLI missing/failed sentinels, an empty response, no
    FINAL_VERDICT token, a DEFEND/EXIT missing its RULE/NUMBERS citation
    (a real book rule and real numbers are mandatory, direct user
    request: never a "random guess, malfunctioning, or misunderstanding"),
    a DEFEND with BOTH NEW_STOP_LOSS and PARTIAL_CLOSE_FRACTION absent
    (nothing to actually defend with), or a PARTIAL_CLOSE_FRACTION
    outside config's configured [min, max] bounds — REJECTED outright,
    never silently clamped into range. If multiple FINAL_VERDICT tokens
    appear, the LAST one wins, same convention as parse_copilot_verdict."""
    fail_safe = TacticalVerdict(tier="hold", raw_text=response_text or "")
    if not response_text or not response_text.strip():
        return fail_safe
    if response_text == COPILOT_MISSING_MESSAGE or response_text.startswith(COPILOT_FAILED_PREFIX):
        return fail_safe

    matches = _TACTICAL_VERDICT_PATTERN.findall(response_text)
    if not matches:
        return fail_safe
    tier = matches[-1].lower()
    if tier == "hold":
        return TacticalVerdict(tier="hold", raw_text=response_text)

    rule_matches = _TACTICAL_RULE_PATTERN.findall(response_text)
    numbers_matches = _TACTICAL_NUMBERS_PATTERN.findall(response_text)
    rule_citation = rule_matches[-1].strip() if rule_matches else ""
    numbers_citation = numbers_matches[-1].strip() if numbers_matches else ""
    if not rule_citation or not numbers_citation:
        return fail_safe

    if tier == "exit":
        return TacticalVerdict(
            tier="exit", rule_citation=rule_citation, numbers_citation=numbers_citation,
            raw_text=response_text,
        )

    # tier == "defend"
    stop_matches = _TACTICAL_NEW_STOP_PATTERN.findall(response_text)
    fraction_matches = _TACTICAL_PARTIAL_FRACTION_PATTERN.findall(response_text)
    new_stop_loss = _parse_optional_tactical_float(stop_matches[-1]) if stop_matches else None
    partial_close_fraction = _parse_optional_tactical_float(fraction_matches[-1]) if fraction_matches else None
    if new_stop_loss is None and partial_close_fraction is None:
        return fail_safe
    if partial_close_fraction is not None and not (
        config.COPILOT_TACTICAL_MIN_PARTIAL_CLOSE_FRACTION
        <= partial_close_fraction
        <= config.COPILOT_TACTICAL_MAX_PARTIAL_CLOSE_FRACTION
    ):
        return fail_safe
    return TacticalVerdict(
        tier="defend", new_stop_loss=new_stop_loss, partial_close_fraction=partial_close_fraction,
        rule_citation=rule_citation, numbers_citation=numbers_citation, raw_text=response_text,
    )


def _build_tactical_prompt(
    symbol: str,
    entry: AllocationEntry,
    position: Position,
    technical_context: str,
    account_equity: float,
    prior_tactical: dict | None,
) -> str:
    """The short-term/tactical counterpart to _build_invalidation_prompt.
    Unlike that check (a mechanical trip-wire on Claude's OWN stated
    condition), this gives the Clerk bounded, tiered authority of its
    own: DEFEND (tighten the stop and/or take a partial profit) as the
    default response to a deteriorating short-term/"trend" read,
    escalating to EXIT (full close) only when defense wouldn't be enough
    or was already tried this cycle and the position kept deteriorating
    — the user's own exact framing, confirmed directly. Every DEFEND/
    EXIT must cite a specific book rule and specific real numbers."""
    if prior_tactical:
        history = (
            "This position has ALREADY had a tactical action taken on it this "
            f"mega-session cycle: {prior_tactical.get('defend_count', 0)} prior DEFEND "
            f"action(s), most recently at {prior_tactical.get('last_action_utc', 'an unknown time')}, "
            f"moving the stop to {prior_tactical.get('last_stop_loss')} when the adverse "
            f"move stood at {prior_tactical.get('adverse_move_pct_at_last_action', 0):.2f}%. "
            "Reason honestly about escalation: if the position kept "
            "deteriorating after that defense, EXIT may now be warranted; if "
            "it has since stabilized or improved, HOLD or a further DEFEND "
            "may be enough."
        )
    else:
        history = "No prior tactical action has been taken on this position yet this cycle."

    return (
        "You are an automated trading clerk for an FTMO account with NEW, "
        "bounded tactical authority over an already-open position — this is "
        "a DIFFERENT check from the mechanical invalidation trip-wire check "
        "this same position may also receive (that check, if present, owns "
        "ONLY the senior analyst's own stated invalidation_condition below; "
        "THIS check owns the short-term/tactical technical read instead).\n\n"
        f"Symbol: {symbol}\n"
        f"Side: {position.side}\n"
        f"Entry price: {position.price_open}\n"
        f"Current price: {position.price_current}\n"
        f"Live floating P&L: {position.profit:.2f}\n"
        f"Adverse move so far: {position.adverse_move_pct:.2f}% (positive = against this position)\n"
        f"Current stop-loss: {position.sl}\n"
        f"Current take-profit: {position.tp}\n"
        f"Opened at: {position.opened_at}\n"
        f"Senior analyst's original reason: {entry.reason}\n"
        "Senior analyst's own invalidation condition (a SEPARATE, "
        f"mechanical check owns this — do not re-judge it here): "
        f"{entry.invalidation_condition or '(none stated)'}\n\n"
        f"{history}\n\n"
        f"{format_trend_wisdom()}\n\n"
        "Below is this symbol's REAL, LIVE technical picture right now (the "
        "same data source that powers this account's own Asset Health "
        "analysis):\n\n"
        f"{technical_context}\n\n"
        f"Account equity for context: {account_equity:.2f}.\n\n"
        "This account trades intraday, closing out within a single trading "
        "day at most — judge this position against a day trader's own "
        "single-session thesis (per the holding-period principle above), "
        "not a swing- or position-trader's much longer patience.\n\n"
        "Decide ONE of three tiers, using the short-term/tactical "
        "principles above (cite the specific one driving your decision, "
        "with specific real numbers from this position/technical picture "
        "— never a vague or unexplained judgment call):\n\n"
        "HOLD — the position's short-term thesis still holds; no action "
        "needed right now.\n\n"
        "DEFEND (the default, preferred response to a deteriorating "
        "short-term read) — tighten the stop-loss (NEVER propose a level "
        "less protective than the current stop-loss above) and/or take a "
        "partial profit, sized/triggered by a specific book rule. Prefer "
        "this over EXIT whenever it would meaningfully reduce risk while "
        "leaving the position room to recover.\n\n"
        "EXIT (an ESCALATION, not a first resort) — close the position "
        "fully. Only choose this when a DEFEND action genuinely wouldn't "
        "be enough, or one was already tried this cycle (see history "
        "above) and the position kept deteriorating regardless — the goal "
        "is capturing the best remaining outcome (maximum remaining "
        "profit, or minimum further loss), not giving up early.\n\n"
        "Respond with your reasoning, then end with EXACTLY this block and "
        "nothing after it (omit a line entirely if it doesn't apply):\n\n"
        "FINAL_VERDICT: HOLD\n\n"
        "-- or --\n\n"
        "FINAL_VERDICT: DEFEND\n"
        "NEW_STOP_LOSS: <a price more protective than the current stop, or NONE>\n"
        "PARTIAL_CLOSE_FRACTION: <a fraction between 0 and 1 of the current "
        "position to close now, or NONE — at least one of these two lines "
        "must be a real value, not both NONE>\n"
        "RULE: <author — the specific one-line book rule driving this>\n"
        "NUMBERS: <the specific real numbers from above driving this decision>\n\n"
        "-- or --\n\n"
        "FINAL_VERDICT: EXIT\n"
        "RULE: <author — the specific one-line book rule driving this>\n"
        "NUMBERS: <the specific real numbers from above driving this "
        "decision, including why a DEFEND alone would not be enough>\n\n"
        'Do not use the token "FINAL_VERDICT:" anywhere else in your response.'
    )


def _run_copilot_tactical_check(
    symbol: str,
    entry: AllocationEntry,
    position: Position,
    technical_context: str,
    account_equity: float,
    prior_tactical: dict | None,
) -> tuple[str, TacticalVerdict, str]:
    """The Copilot-bound half of a tactical-defense check — pure CLI
    subprocess call (with an OpenRouter backup chain, see
    _run_clerk_prompt), no MT5 involvement, safe to run from inside a
    thread pool (see run_copilot_execution_check's own parallel round).
    Returns (symbol, verdict, raw_text) — a bare symbol string in
    position 0 like _run_copilot_invalidation_check, but a TacticalVerdict
    (not a bool) in position 1, which is exactly how the merge loop tells
    the two kinds of bare-symbol result apart."""
    prompt = _build_tactical_prompt(symbol, entry, position, technical_context, account_equity, prior_tactical)
    raw = _run_clerk_prompt(prompt, timeout=config.COPILOT_VERIFICATION_TIMEOUT_SECONDS)
    return symbol, parse_tactical_verdict(raw), raw


def _validate_and_apply_tactical_verdict(
    symbol: str,
    verdict: TacticalVerdict,
    position: Position,
    existing_entry: AllocationEntry | None,
    prior_tactical: dict | None,
    account_equity: float,
    get_spec: Callable,
) -> tuple[AllocationEntry | None, dict | None, str]:
    """The real safety-critical guardrail layer for the Clerk's new
    tactical-defense authority — never trusts the LLM's own numbers past
    parse_tactical_verdict; every hard rule here is deterministic Python,
    not a second AI opinion. Returns (new_allocation_entry_or_None,
    new_tactical_state_or_None, rejected_reason): the first is merged
    into merged_allocation only when not None (None means "no change,
    leave the existing target alone"); the second replaces this symbol's
    SymbolSettlement.tactical dict only when not None; the third is a
    non-empty string ONLY when a DEFEND/EXIT was genuinely REJECTED by
    one of the deterministic checks below (never set for HOLD, and never
    set for a DEFEND/EXIT that resolved successfully).

    Found on audit: without this third value, the caller (run_copilot_
    execution_check's merge loop) could not tell "this DEFEND was
    rejected outright by a guardrail and will NEVER be applied" apart
    from "this DEFEND is valid and would be applied the moment tactical
    defense is enabled" — both looked identical (new_entry is None,
    applied stays False) to app.py's own last_tactical_verdicts display.
    During exactly the shadow-mode observation period this feature's own
    staged rollout depends on, that ambiguity could mask a DEFEND that
    would ALWAYS be rejected (e.g. a stop repeatedly landing on the
    wrong side of price, or a cooldown that never lets it re-fire) as if
    it were a healthy, pending action merely waiting on the toggle.

    HOLD -> (None, None, ""), always.

    EXIT -> forces pct to 0 (mirrors the existing invalidation-CONFIRMED
    pattern exactly — never calls close_position directly here;
    compute_rebalance_plan stays the one place that ever decides to
    close, so every existing safety gate still runs first automatically).

    DEFEND is gated by four deterministic checks, ALL of which must
    pass or the whole DEFEND is rejected to "no change" (never a partial
    apply):
    1. Cooldown/no-thrash: suppressed unless enough time has passed AND
       the position has genuinely deteriorated further since the last
       tactical action (both required together to re-fire inside the
       cooldown window) — config.COPILOT_TACTICAL_DEFEND_COOLDOWN_
       MINUTES / _MIN_RETRIGGER_PCT.
    2. Never-widen-stop: a proposed new stop must be strictly MORE
       protective than the position's own current stop (higher for a
       buy, lower for a sell) — defense-in-depth alongside the prompt's
       own instruction not to propose one.
    3. Sane side of the LIVE price: "more protective than the OLD stop"
       alone doesn't rule out a stop placed past the CURRENT price (an
       over-eager tighten proposing a buy's stop ABOVE the live price,
       which isn't a valid protective stop at all) — found on audit,
       rejected explicitly here rather than relying on compute_
       rebalance_plan's own incidental wrong-side-of-entry check
       downstream (which would silently no-op it as "infeasible" with
       no clear reason surfaced at this layer).
    4. Real sizing math: reuses risk/apply_suggestion.py::
       pct_for_target_lots (the exact inverse of compute_rebalance_
       plan's own risk formula) against the position's LIVE price_
       current (never existing_entry.price, which can be days-stale by
       exactly the scenario this feature exists to defend against, and
       would silently desync from what compute_rebalance_plan's own
       price-sanity clamp later uses) rather than naively reusing the
       position's existing pct — tightening a stop while keeping the
       SAME pct would silently INCREASE lot size (smaller stop distance
       / same risk-% = more lots), the opposite of "same size, tighter
       stop." A partial-close fraction is applied to the CURRENT held
       lots BEFORE solving for pct, so the same helper covers both a
       pure stop-tighten and a partial-close/stop-tighten combo."""
    if verdict.tier == "hold":
        return None, None, ""

    if verdict.tier == "exit":
        new_entry = AllocationEntry(
            pct=0.0,
            price=existing_entry.price if existing_entry else position.price_open,
            stop_loss=existing_entry.stop_loss if existing_entry else position.sl,
            take_profit=existing_entry.take_profit if existing_entry else position.tp,
            side=existing_entry.side if existing_entry else position.side,
            reason=f"Tactical EXIT: {verdict.rule_citation} — {verdict.numbers_citation}",
        )
        tactical_state = {
            **(prior_tactical or {}),
            "last_action_utc": datetime.now(timezone.utc).isoformat(),
            "tier_reached": "exit",
        }
        return new_entry, tactical_state, ""

    # tier == "defend"
    now = datetime.now(timezone.utc)
    if prior_tactical and prior_tactical.get("last_action_utc"):
        try:
            last_action_dt = datetime.fromisoformat(prior_tactical["last_action_utc"])
        except ValueError:
            last_action_dt = None
        if last_action_dt is not None:
            minutes_since = (now - last_action_dt).total_seconds() / 60
            worsened_by = position.adverse_move_pct - prior_tactical.get("adverse_move_pct_at_last_action", 0.0)
            if (
                minutes_since < config.COPILOT_TACTICAL_DEFEND_COOLDOWN_MINUTES
                and worsened_by < config.COPILOT_TACTICAL_DEFEND_MIN_RETRIGGER_PCT
            ):
                reason = (
                    f"suppressed by cooldown (only {minutes_since:.1f}m since the last "
                    f"action, cooldown is {config.COPILOT_TACTICAL_DEFEND_COOLDOWN_MINUTES}m; "
                    f"adverse move only worsened by {worsened_by:.2f}%, needs "
                    f">= {config.COPILOT_TACTICAL_DEFEND_MIN_RETRIGGER_PCT:.2f}%)"
                )
                logger.info("%s: tactical DEFEND %s.", symbol, reason)
                return None, None, reason

    # Found on audit: this MUST be the position's own live price, never
    # existing_entry.price (the mega session's own possibly days-old
    # suggestion price — exactly the value that has, by construction,
    # drifted the most in precisely the "mega session hasn't run in
    # days" scenario this whole feature exists to defend against).
    # compute_rebalance_plan will re-clamp whatever price this entry
    # carries to within PRICE_SANITY_BAND_PCT of the LIVE market price
    # before it sizes anything — using a stale price here that's already
    # drifted past that band would silently size against a DIFFERENT
    # stop distance than compute_rebalance_plan actually uses, breaking
    # the very "same lots, tighter stop" invariant pct_for_target_lots
    # exists to guarantee. position.price_current is fetched this same
    # poll, so it's always within that band of itself.
    entry_price = position.price_current
    current_stop = position.sl
    new_stop = verdict.new_stop_loss
    if new_stop is not None:
        if current_stop is None:
            reason = "rejected: proposed a new stop but the position has no current stop to compare against"
            logger.warning("%s: tactical DEFEND %s.", symbol, reason)
            return None, None, reason
        more_protective = new_stop > current_stop if position.side == "buy" else new_stop < current_stop
        if not more_protective:
            reason = (
                f"rejected: proposed stop {new_stop:.5f} is not more protective than the "
                f"current stop {current_stop:.5f} (never-widen-stop rule)"
            )
            logger.warning("%s: tactical DEFEND %s.", symbol, reason)
            return None, None, reason
        # Found on audit: "more protective than the OLD stop" alone
        # doesn't rule out a stop placed past the CURRENT live price —
        # e.g. an over-eager tighten proposing a buy's stop ABOVE the
        # current price, which is not a valid protective stop at all
        # (it would trigger immediately, or be rejected by the broker).
        # Reject explicitly here, at the guardrail layer, rather than
        # relying on compute_rebalance_plan's own incidental wrong-side-
        # of-entry check downstream (which would silently no-op this as
        # "infeasible" — safe, but with no clear reason surfaced here,
        # and last_tactical_verdicts would misleadingly show "applied").
        sane_side = new_stop < entry_price if position.side == "buy" else new_stop > entry_price
        if not sane_side:
            reason = (
                f"rejected: proposed stop {new_stop:.5f} is on the wrong side of the "
                f"current live price {entry_price:.5f}"
            )
            logger.warning("%s: tactical DEFEND %s.", symbol, reason)
            return None, None, reason
        resolved_stop = new_stop
    else:
        resolved_stop = current_stop

    if resolved_stop is None:
        reason = "rejected: no usable stop (current or proposed) to size against"
        logger.warning("%s: tactical DEFEND %s.", symbol, reason)
        return None, None, reason

    fraction = verdict.partial_close_fraction
    target_lots = position.volume * (1.0 - fraction) if fraction is not None else position.volume

    new_pct = pct_for_target_lots(symbol, target_lots, entry_price, resolved_stop, account_equity, get_spec)
    if new_pct is None:
        reason = "rejected: could not be sized (no usable contract spec or stop distance)"
        logger.warning("%s: tactical DEFEND %s.", symbol, reason)
        return None, None, reason

    new_entry = AllocationEntry(
        pct=new_pct,
        price=entry_price,
        stop_loss=resolved_stop,
        take_profit=existing_entry.take_profit if existing_entry else position.tp,
        side=existing_entry.side if existing_entry else position.side,
        reason=f"Tactical DEFEND: {verdict.rule_citation} — {verdict.numbers_citation}",
        invalidation_condition=existing_entry.invalidation_condition if existing_entry else None,
    )
    tactical_state = {
        "last_action_utc": now.isoformat(),
        "last_stop_loss": resolved_stop,
        "adverse_move_pct_at_last_action": position.adverse_move_pct,
        "defend_count": (prior_tactical or {}).get("defend_count", 0) + 1,
        "tier_reached": "defend",
    }
    return new_entry, tactical_state, ""


# _mega_session_is_live used to be its own local copy of the same
# freshness heuristic app.py's live-progress display also needs — see
# ai.mega_analysis.mega_session_is_live's own docstring for why keeping
# two independent copies was itself a real risk (a fix applied to only
# one, as almost happened here) and for the age-ceiling bug found and
# fixed in both at once, 2026-08-27. Now a single shared implementation,
# imported above under the original name so this module's own call site
# and the tests that patch "ai.copilot_execution._mega_session_is_live"
# keep working unchanged.


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
    # First position found per symbol — same accepted "compare only the
    # first ticket" simplification risk/apply_suggestion.py::compute_
    # rebalance_plan's own amend-in-place path already documents for a
    # rare multi-ticket-per-symbol position.
    positions_by_symbol: dict[str, Position] = {}
    for p in positions:
        positions_by_symbol.setdefault(p.symbol, p)

    settlement["settled"] = _reconcile_settlement(settlement.get("settled", {}), positions, pending_orders)
    settlement["settled"] = _backfill_settlement_for_held_positions(
        settlement["settled"], positions, immediate_allocation_raw, account.equity, get_contract_spec,
    )
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

    # Already-settled (order_placed/filled) immediate_allocation symbols
    # carrying their own invalidation_condition — added 2026-08-23,
    # direct user request: Copilot should watch the mega session's own
    # stated exit condition for an already-suggested position, not just
    # check brand-new Pending Setups. Bounded exactly as confirmed with
    # the user: only symbols Claude itself flagged with a real condition,
    # never an open-ended re-opinion on every held position.
    watched_positions: list[tuple[str, dict, str, str]] = []  # (symbol, entry, invalidation_condition, state)
    for symbol, raw in immediate_allocation_raw.items():
        if symbol.upper() == "CASH":
            continue
        rec = settlement["settled"].get(symbol)
        if rec is None or rec.get("state") not in ("order_placed", "filled"):
            continue
        invalidation_condition = raw.get("invalidation_condition")
        if not invalidation_condition:
            continue
        watched_positions.append((symbol, raw, invalidation_condition, rec["state"]))
    if len(watched_positions) > config.COPILOT_EXECUTION_MAX_WATCHED_POSITIONS:
        logger.warning(
            "%d watched positions exceeds the cap of %d — checking only the first %d.",
            len(watched_positions), config.COPILOT_EXECUTION_MAX_WATCHED_POSITIONS,
            config.COPILOT_EXECUTION_MAX_WATCHED_POSITIONS,
        )
        watched_positions = watched_positions[: config.COPILOT_EXECUTION_MAX_WATCHED_POSITIONS]

    # Tactical-defense candidates — added 2026-08-27, direct user request
    # after a real gold position went from +$28 to -$61 while the mega
    # session hadn't run in days and the Clerk had zero authority to
    # react (see this module's own docstring). Deliberately broader than
    # watched_positions above: every already-FILLED symbol with a real
    # live MT5 position gets a tactical check, with NO invalidation_
    # condition gate — this answers a different question (the short-
    # term/"trend" read) than that mechanical trip-wire does, and a
    # symbol can legitimately be checked by both independently (see the
    # merge loop's own "0% wins" collision handling below).
    tactical_enabled = read_tactical_defense_enabled()
    tactical_candidates = [
        symbol for symbol, rec in settlement["settled"].items()
        if rec.get("state") == "filled" and symbol in positions_by_symbol
    ]
    if len(tactical_candidates) > config.COPILOT_EXECUTION_MAX_TACTICAL_CANDIDATES:
        logger.warning(
            "%d tactical-defense candidates exceeds the cap of %d — checking only the first %d.",
            len(tactical_candidates), config.COPILOT_EXECUTION_MAX_TACTICAL_CANDIDATES,
            config.COPILOT_EXECUTION_MAX_TACTICAL_CANDIDATES,
        )
        tactical_candidates = tactical_candidates[: config.COPILOT_EXECUTION_MAX_TACTICAL_CANDIDATES]

    last_verdicts: dict = {}
    last_tactical_verdicts: dict = {}
    merged_allocation = dict(carried_forward)
    if not_yet_settled or watched_positions or tactical_candidates:
        if not_yet_settled:
            _notify(f"Checking {len(not_yet_settled)} pending setup(s) against live technicals and Copilot...")
        if watched_positions:
            _notify(f"Checking {len(watched_positions)} watched position(s) for invalidation...")
        if tactical_candidates:
            _notify(f"Checking {len(tactical_candidates)} position(s) for tactical defense...")
        # Phase 1 (sequential, MT5-bound): fetch every not-yet-settled
        # setup's, every watched position's, AND every tactical
        # candidate's live technicals one at a time — never in parallel,
        # see _fetch_technical_context's own docstring. technical_
        # context_cache is shared across all three so a symbol appearing
        # in more than one list (e.g. a watched position that's also a
        # tactical candidate) is only ever fetched once.
        technical_context_cache: dict[str, str] = {}

        def _cached_technical_context(symbol: str) -> str | None:
            if symbol not in technical_context_cache:
                technical_context_cache[symbol] = _fetch_technical_context(symbol, market_prices, account.equity)
            return technical_context_cache[symbol]

        checkable: list[tuple[PendingSetup, str]] = []
        for setup in not_yet_settled:
            technical_context = _cached_technical_context(setup.symbol)
            if technical_context is None:
                logger.warning("%s is not currently visible in Market Watch — skipped.", setup.symbol)
                continue
            checkable.append((setup, technical_context))

        checkable_watched: list[tuple[str, dict, str, str, str]] = []
        for symbol, raw, invalidation_condition, state in watched_positions:
            technical_context = _cached_technical_context(symbol)
            if technical_context is None:
                logger.warning("%s (watched) is not currently visible in Market Watch — skipped.", symbol)
                continue
            checkable_watched.append((symbol, raw, invalidation_condition, state, technical_context))

        checkable_tactical: list[tuple[str, AllocationEntry, Position, str, dict | None]] = []
        for symbol in tactical_candidates:
            technical_context = _cached_technical_context(symbol)
            if technical_context is None:
                logger.warning("%s (tactical) is not currently visible in Market Watch — skipped.", symbol)
                continue
            entry = carried_forward.get(symbol)
            if entry is None:
                continue
            prior_tactical = settlement["settled"].get(symbol, {}).get("tactical")
            checkable_tactical.append((symbol, entry, positions_by_symbol[symbol], technical_context, prior_tactical))

        # Phase 2 (parallel, Copilot-bound): pure CLI subprocess calls,
        # no MT5 involvement — bounds wall-clock to roughly the slowest
        # single call rather than their sum, mirroring
        # ai.portfolio_suggest.build_audit_block's own proven pattern.
        # All three kinds of check share ONE pool — splitting into more
        # would only add wall-clock (bounded by the slower pool's own
        # slowest call, sequentially after the first) for no isolation
        # benefit, since none touches shared mutable state until this
        # single-threaded merge below. Submitted in this exact order —
        # Pending-Setup triggers, then invalidation checks, then tactical
        # checks last — so the merge loop below processes results in
        # that same fixed priority order regardless of which future
        # resolves first, generalizing the existing "0% wins" collision
        # rule: by the time a tactical result is merged, any invalidation
        # CONFIRMED (or the mega session's own fresh 0% target) for the
        # same symbol has already been applied.
        if checkable or checkable_watched or checkable_tactical:
            elapsed_description = _describe_elapsed(generated_utc, datetime.now(timezone.utc))
            with ThreadPoolExecutor(
                max_workers=len(checkable) + len(checkable_watched) + len(checkable_tactical)
            ) as pool:
                futures = [
                    pool.submit(_run_copilot_verdict, setup, technical_context, account.equity, elapsed_description)
                    for setup, technical_context in checkable
                ] + [
                    pool.submit(
                        _run_copilot_invalidation_check, symbol, raw, cond, technical_context,
                        account.equity, elapsed_description, state,
                    )
                    for symbol, raw, cond, state, technical_context in checkable_watched
                ] + [
                    pool.submit(
                        _run_copilot_tactical_check, symbol, entry, position, technical_context,
                        account.equity, prior_tactical,
                    )
                    for symbol, entry, position, technical_context, prior_tactical in checkable_tactical
                ]
                results = [f.result() for f in futures]
            # Merged on the main thread, in original list order, after
            # every future resolves — never inside a worker thread — so
            # completion order can never make this non-deterministic.
            # Discriminated by result[0]'s type: PendingSetup for a
            # trigger-check; for the remaining two kinds, both return a
            # bare symbol string in position 0 (see _run_copilot_
            # invalidation_check's and _run_copilot_tactical_check's own
            # docstrings for why), so result[1]'s type — bool for an
            # invalidation verdict, TacticalVerdict for a tactical one —
            # is what tells THOSE two apart.
            for result in results:
                if isinstance(result[0], PendingSetup):
                    setup, confirmed, raw = result
                    last_verdicts[setup.symbol] = {
                        "confirmed": confirmed, "raw_text": raw,
                        "checked_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    # The full reasoning text is logged here too (direct
                    # user request 2026-08-23: "also write this in a log
                    # somewhere") — previously it only ever landed in
                    # copilot_execution_state.json's last_verdicts, never
                    # in the actual human-readable log file.
                    logger.info(
                        "Copilot verdict for %s: %s — %s", setup.symbol,
                        "CONFIRMED" if confirmed else "NOT_CONFIRMED",
                        raw[:1000] if raw else "(no response text)",
                    )
                    if confirmed:
                        merged_allocation[setup.symbol] = AllocationEntry(
                            pct=setup.pct, price=setup.price, stop_loss=setup.stop_loss,
                            take_profit=setup.take_profit, side=setup.side, reason=setup.reason,
                        )
                elif isinstance(result[1], TacticalVerdict):
                    symbol, verdict, raw = result
                    logger.info(
                        "Copilot tactical check for %s: %s — %s", symbol, verdict.tier.upper(),
                        raw[:1000] if raw else "(no response text)",
                    )
                    existing_target = merged_allocation.get(symbol)
                    prior_tactical = settlement["settled"].get(symbol, {}).get("tactical")
                    position = positions_by_symbol.get(symbol)
                    applied = False
                    skipped_reason = ""
                    if existing_target is not None and existing_target.pct <= 0.0:
                        # "0% wins": an earlier phase THIS SAME POLL (a
                        # Pending-Setup trigger, an invalidation CONFIRMED,
                        # or the mega session's own fresh 0% target already
                        # carried into merged_allocation) already decided
                        # to close this symbol for an unrelated reason —
                        # generalizes the existing invalidation-vs-mega-
                        # session collision rule to also cover tactical.
                        skipped_reason = "already targeted at 0% this poll by an unrelated decision"
                        logger.info("%s: tactical %s skipped — %s.", symbol, verdict.tier.upper(), skipped_reason)
                    elif position is None:
                        skipped_reason = "no live position found for this symbol"
                        logger.warning("%s: tactical verdict returned but %s — skipped.", symbol, skipped_reason)
                    else:
                        new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
                            symbol, verdict, position, existing_target, prior_tactical,
                            account.equity, get_contract_spec,
                        )
                        # Found on audit: a genuinely REJECTED DEFEND/EXIT
                        # (cooldown suppression, never-widen-stop, wrong-
                        # side-of-price, sizing failure) and a VALID one
                        # merely waiting on the disabled toggle both used
                        # to look identical here (new_entry is None either
                        # way) — surfacing rejected_reason lets app.py's
                        # panel tell "this will never fire" apart from
                        # "this would fire the moment tactical defense is
                        # enabled," which matters most during exactly the
                        # shadow-mode observation window this feature's
                        # own rollout plan depends on.
                        skipped_reason = rejected_reason
                        if tactical_enabled:
                            if new_entry is not None:
                                merged_allocation[symbol] = new_entry
                                applied = True
                            if tactical_state is not None:
                                rec = settlement["settled"].get(symbol)
                                if rec is not None:
                                    rec["tactical"] = tactical_state
                        # else: shadow mode — new_entry/tactical_state are
                        # still computed above (a faithful rehearsal of
                        # the real pipeline against real live data) for
                        # visibility in last_tactical_verdicts below, but
                        # deliberately never applied to merged_allocation
                        # or the settlement record while disabled.
                    last_tactical_verdicts[symbol] = {
                        "tier": verdict.tier,
                        "new_stop_loss": verdict.new_stop_loss,
                        "partial_close_fraction": verdict.partial_close_fraction,
                        "rule_citation": verdict.rule_citation,
                        "numbers_citation": verdict.numbers_citation,
                        "raw_text": raw,
                        "checked_utc": datetime.now(timezone.utc).isoformat(),
                        "applied": applied,
                        "shadow_mode": not tactical_enabled,
                        "skipped_reason": skipped_reason,
                    }
                else:
                    symbol, confirmed, raw = result
                    last_verdicts[symbol] = {
                        "confirmed": confirmed, "raw_text": raw,
                        "checked_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    logger.info(
                        "Copilot invalidation check for %s: %s — %s", symbol,
                        "CONFIRMED (invalidated)" if confirmed else "NOT_CONFIRMED (still holds)",
                        raw[:1000] if raw else "(no response text)",
                    )
                    if confirmed:
                        # Force this symbol's target to 0 rather than
                        # calling cancel_pending_order/close_position
                        # directly here — compute_rebalance_plan (below)
                        # stays the ONE place that ever decides to
                        # cancel/close, so every existing safety gate
                        # still runs first automatically, and a same-poll
                        # collision with a brand-new mega session's own
                        # fresh pct:0 for this symbol resolves
                        # unambiguously ("0% wins") instead of racing two
                        # independent execution paths.
                        existing = merged_allocation.get(symbol)
                        merged_allocation[symbol] = AllocationEntry(
                            pct=0.0,
                            price=existing.price if existing else None,
                            stop_loss=existing.stop_loss if existing else None,
                            take_profit=existing.take_profit if existing else None,
                            side=(existing.side if existing else "buy"),
                            reason=f"Invalidation condition fired: {raw[:200] if raw else ''}",
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
        _write_execution_state(
            "blocked", block_reason, last_verdicts=last_verdicts, last_tactical_verdicts=last_tactical_verdicts,
        )
        _mark_interval_ran(datetime.now(timezone.utc))
        return

    plan = compute_rebalance_plan(
        positions, account, merged_allocation, get_contract_spec, market_prices,
        price_sanity_band_pct=config.PRICE_SANITY_BAND_PCT,
        pending_orders=pending_orders,
        amend_tolerance_pct=config.AMEND_TOLERANCE_PCT,
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
                            "reason": merged_allocation[o.symbol].reason,
                            "invalidation_condition": merged_allocation[o.symbol].invalidation_condition,
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
        elif o.action == "cancel":
            any_failed = False
            last_detail = ""
            for ticket in o.pending_tickets_to_cancel:
                try:
                    result = cancel_pending_order(ticket)
                except MT5ConnectionError as e:
                    result = OrderResult(False, None, str(e), None)
                logger.info("%s (cancel): %s", o.symbol, "cancelled" if result.success else f"failed ({result.comment})")
                if result.success:
                    executed_count += 1
                else:
                    any_failed = True
                    last_detail = result.comment
            if not any_failed:
                # Real bug found on self-review 2026-08-24: this used to
                # pop unconditionally. That's correct when the settlement
                # record itself was tracking the now-cancelled order
                # (state "order_placed" — the symbol should be eligible
                # for a fresh check right away, not wait a whole poll for
                # _reconcile_settlement to notice the ticket is gone).
                # But this "cancel" action ALSO fires for a stray/
                # superseded pending order sitting alongside an already-
                # held, already-at-target position (settlement state
                # "filled") — popping there would wipe the tracking for a
                # real, completely unaffected position, silently breaking
                # its ongoing watched_positions/invalidation-condition
                # eligibility for the rest of this mega-session cycle.
                # Only clear the record when it was actually about THIS
                # cancelled order.
                existing = settlement["settled"].get(o.symbol)
                if existing is not None and existing.get("state") == "order_placed":
                    settlement["settled"].pop(o.symbol, None)
            _record_result(o.symbol, o.action, not any_failed, "cancelled" if not any_failed else last_detail)
        elif o.action == "amend_pending":
            cancel_ok = True
            last_detail = ""
            for ticket in o.pending_tickets_to_cancel:
                try:
                    result = cancel_pending_order(ticket)
                except MT5ConnectionError as e:
                    result = OrderResult(False, None, str(e), None)
                if not result.success:
                    cancel_ok = False
                    last_detail = result.comment
            if not cancel_ok:
                # Don't stack a fresh order on top of one that failed to
                # cancel — leave both alone this poll; the next poll
                # recomputes the same amend_pending decision from scratch.
                logger.warning(
                    "%s (amend_pending): cancel failed, skipping the replacement open this poll: %s",
                    o.symbol, last_detail,
                )
                _record_result(o.symbol, o.action, False, f"cancel failed, not replaced this poll: {last_detail}")
            else:
                try:
                    result = open_position(o.symbol, o.side, o.volume, o.price, o.stop_loss, o.take_profit)
                except MT5ConnectionError as e:
                    result = OrderResult(False, None, str(e), None)
                logger.info(
                    "%s (amend_pending): %s", o.symbol, "replaced" if result.success else f"failed ({result.comment})"
                )
                _record_result(o.symbol, o.action, result.success, "replaced" if result.success else result.comment)
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
                                "reason": merged_allocation[o.symbol].reason,
                                "invalidation_condition": merged_allocation[o.symbol].invalidation_condition,
                            },
                            order_ticket=result.ticket,
                        )
                    )
        elif o.action == "amend_position":
            any_failed = False
            last_detail = ""
            for ticket in o.position_tickets_to_amend:
                matching = next((p for p in positions if p.ticket == ticket), None)
                if matching is None:
                    any_failed = True
                    last_detail = f"ticket {ticket} not found among fetched positions"
                    continue
                try:
                    result = modify_position_sltp(matching, o.stop_loss, o.take_profit)
                except MT5ConnectionError as e:
                    result = OrderResult(False, None, str(e), None)
                logger.info(
                    "%s (amend_position): %s", o.symbol, "updated" if result.success else f"failed ({result.comment})"
                )
                if result.success:
                    executed_count += 1
                else:
                    any_failed = True
                    last_detail = result.comment
            _record_result(o.symbol, o.action, not any_failed, "updated" if not any_failed else last_detail)
        else:
            _record_result(o.symbol, o.action, True, "on target, no change needed")

    _save_settlement(settlement)
    _notify(f"Execution-check complete — {executed_count} order(s) sent this poll.")
    _write_execution_state(
        "success", f"{executed_count} order(s) sent",
        last_verdicts=last_verdicts, last_execution_results=results_this_poll,
        last_tactical_verdicts=last_tactical_verdicts,
    )
    _mark_interval_ran(datetime.now(timezone.utc))
