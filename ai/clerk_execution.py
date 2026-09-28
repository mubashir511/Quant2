"""The "clerk/executioner" half of the FTMO boardroom architecture —
direct user request 2026-08-22/23: Claude (senior analyst) runs only in
the once-daily mega analysis session; the Clerk's own separate role is
to run on a short interval (config.CLERK_EXECUTION_CHECK_INTERVAL_MINUTES,
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
  target mix once a live Clerk verdict confirms the stated
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
frequency picker (app.py's Execution Clerk panel heading):
read_clerk_execution_enabled() is checked as the VERY FIRST thing in
run_clerk_execution_check(), before even the mega-session-live check
— the single, authoritative gate all three of this function's call
sites (the standalone poll, mega_analysis_job.py's inline pass, app.py's
manual-button inline pass) share automatically, so a disabled clerk is
disabled everywhere at once, not just on the scheduled poll. The review
frequency itself (previously a fixed config.CLERK_EXECUTION_CHECK_
INTERVAL_MINUTES) is now similarly overridable via read_clerk_
execution_interval_minutes(), read by both is_execution_due and
next_execution_check_utc (used for the panel's own "next review"
countdown) — see is_execution_due's own docstring for the 2026-09-21
switch from fixed clock-aligned windows to a genuine rolling cooldown
timed from the last run's own completion.

2026-08-23/24 update — Claude and the Clerk can now re-assess an
ALREADY-suggested position, not just propose fresh ones (direct user
request: "claude should be able to scan the existing suggested
positions... and comment on these positions whether to keep them,
update it... or completely stop the active order or drop the limit
order", with the Clerk's own periodic judgment explicitly bounded to
"the thesis from the mega session" rather than an open-ended second
opinion). Two new optional `AllocationEntry` fields, `reason` and
`invalidation_condition` (mirroring `PendingSetup`'s own `reason`/
`trigger_condition`), let Claude's daily mega session write a
per-position thesis and a specific, mechanically-checkable exit
condition. `compute_rebalance_plan` (risk/apply_suggestion.py) is now
pending-order-aware and can emit three new actions handled by the
execution loop below: `"cancel"` (a resting pending order the fresh
target no longer wants), `"amend_pending"` (cancel + reopen at new
terms — safe, since an unfilled order has no realized exposure), and
`"amend_position"` (a true in-place SL/TP amend via the new
data/mt5_execution.py::modify_position_sltp, deliberately never a
close-then-reopen — see that function's own docstring for why).
`watched_positions` (built in run_clerk_execution_check, just before
the existing Pending-Setups verdict phase) collects every already-
settled (`order_placed`/`filled`) immediate_allocation symbol carrying a
non-empty `invalidation_condition`, fetches its live technicals the same
sequential way Pending Setups already do, and runs a NEW
`_build_invalidation_prompt`/`_run_clerk_invalidation_check` pair in
the SAME parallel ThreadPoolExecutor phase as the existing Pending-
Setups verdict calls — deliberately a mechanical trip-wire check on
Claude's own stated condition, never the Clerk's own fresh opinion on
whether the trade "still feels right." On CONFIRMED, this forces the
symbol's `pct` to 0 in `merged_allocation` rather than calling
`cancel_pending_order`/`close_position` directly — `compute_rebalance_
plan` stays the ONE place that ever actually decides to cancel/close, so
every existing safety gate still runs first automatically, and a same-
poll collision with a brand-new mega session's own fresh `pct: 0` for
the same symbol resolves unambiguously ("0% wins") instead of two
independent execution paths racing.

2026-08-25 update (superseded 2026-08-30, kept for history) — after the
GitHub `copilot` CLI's monthly quota was confirmed genuinely exhausted
(every real verdict/invalidation call that day failing with "You have
exceeded your monthly quota"), this job was changed to try Copilot
first, then fall through 5 free OpenRouter backup models, first success
wins. That mostly worked, but a second failure mode showed up live
2026-08-27: all 10 of the mega session's own free OpenRouter audit
models went unavailable AT ONCE during a real run (a shared-pool
outage, not a per-model one) — the same shared pool this job's own
backup chain drew from.

2026-08-30 update, direct user request — Copilot CLI and the OpenRouter
backup chain are both removed from this job entirely and replaced with
two models running on a local Ollama server (ai/ollama_client.py):
config.CLERK_PRIMARY_MODEL (gemma4:12b, thinking disabled) tried first,
config.CLERK_BACKUP_MODEL (qwen2.5:7b) as the fallback if the primary
genuinely isn't available. Both run entirely on this machine — no API
key, no shared rate limit, no per-token cost, and no dependency on any
external service's own uptime, directly answering the two real failure
modes above. Live-tested across all three verdict tiers before this
swap (see _run_clerk_prompt's own docstring for the numbers): gemma4:12b
with thinking enabled took 8-13+ minutes per call and once never
finished at all — a category mismatch for this job's explicit "quick
and correct" design goal, not "eventually correct after extended
deliberation" — but with thinking disabled it answered in 30-70s with
correct stop-tightening arithmetic on the one case a smaller model
(qwen2.5:7b alone) got backwards. GitHub Copilot itself keeps its
entirely separate, still-active role verifying the mega session's own
web-research claims (ai.portfolio_suggest.build_copilot_verification) —
that feature is untouched by this change, it was always independent of
this module.

2026-09-15/16 update, direct user request after a real, confirmed
incident — external stop-loss drift (see _iter_stop_drift's own
docstring for the full 2026-09-08 XAUUSD detection history) is no
longer detection-only. A second real incident on 2026-09-14 (XAGUSD:
Clerk's own tactical DEFEND had tightened the live stop to 63.97,
something outside this app reverted it back to the original 63.13, the
existing detector correctly flagged this on every poll for ~4 hours, and
nothing ever corrected it — the position was then stopped out at the
worse, reverted level) prompted the direct instruction: "do not wait for
any human intervention just act and save the equity."
`_restore_external_stop_drift` now runs where the old detect-only call
used to, every poll, unconditionally on either drift direction, and
actively writes the app's own last-recorded stop back onto the live
position via modify_position_sltp the moment drift is found — see that
function's own docstring for why this is safe (it only ever re-asserts a
value Clerk's own logic already validated, never invents a new one) and
why it deliberately runs even while FTMO-heat-blocked (a risk-reducing
restore is never new exposure)."""

import json
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

import config
from ai import continuation_hunter, trade_journal
from ai.ollama_client import FAILED_MESSAGE as OLLAMA_FAILED_MESSAGE
from ai.ollama_client import run_ollama
from ai.ftmo_suggest import (
    FtmoAssetAnalysis,
    IntradayBacktests,
    intraday_edge_verdicts,
    effective_stop_floor,
    _HIGH_CORRELATION_THRESHOLD,
    _MIN_CORRELATION_OBSERVATIONS,
    aligned_h1_h4_trend_direction,
    aligned_m5_trend_direction,
    analyze_ftmo_asset_live,
    fetch_ftmo_status,
    format_clerk_context,
    format_ftmo_asset_context,
    read_latest_suggestion,
)
from ai.mega_analysis import mega_session_is_live as _mega_session_is_live
from ai.mega_analysis import read_progress as read_mega_progress
from ai.mega_analysis import read_state as read_mega_state
from ai.portfolio_suggest import AllocationEntry, AssetAnalysis, PendingSetup
from analysis.backtest import classify_backtest_favorability
from analysis.edge_stats import SUPPORTED as EDGE_SUPPORTED
from analysis.entry_mode import normalise_entry_mode, resolve_entry_mode
from analysis.triggers import evaluate_trigger, normalise_trigger
from ai import clerk_thinking
from ai.sentinel import apply_sentinel_ratchet, ingest_sentinel_stops, read_sentinel_state
from analysis.trail import original_risk as trail_original_risk, trail_decision
from ai import rehunt
from analysis.chart_structure import (
    ChartStructureSnapshot,
    LiquiditySweepEvent,
    SRLevel,
    StructureBreak,
    compute_chart_structure,
    find_swing_points,
)
from analysis.setup_classifier import classify_setups
from analysis.technical import (
    TRADING_DAYS_PER_YEAR,
    TechnicalStats,
    classify_m5_velocity_tier,
    classify_rsi_tier,
    classify_velocity_tier,
    compute_atr,
    compute_regime_segments,
    compute_technical_stats,
    detect_rsi_divergence,
)
from analysis.timeframe_profiles import M5_PROFILE
from analysis.trade_zone import construct_trade_zone
from data.book_wisdom import format_trend_wisdom
from data import economic_calendar, symbol_news
from data.mt5_execution import (
    MT5ConnectionError,
    OrderResult,
    cancel_pending_order,
    close_position,
    modify_position_sltp,
    open_position,
)
from ai.chart_overlay import OverlaySymbolInputs, build_overlay_lines, write_chart_overlay
from data.mt5_source import (
    ClosedTrade,
    PendingOrder,
    Position,
    TradeCost,
    connect,
    fetch_mt5_price_history,
    fetch_mt5_price_history_range,
    get_account_summary,
    get_contract_spec,
    get_history_deals,
    get_market_watch,
    get_open_positions,
    get_pending_orders,
    get_server_time_offset,
    get_symbol_category,
    get_terminal_commondata_path,
    get_trade_economics,
    group_closed_trades,
    is_symbol_tradable_now,
    is_trading_permitted,
)
from risk.apply_suggestion import (
    PlannedOrder,
    check_execution_safety_gates,
    compute_aggregate_heat_pct,
    compute_rebalance_plan,
    pct_for_target_lots,
)
from risk.ftmo_rules import DEFAULT_HEADROOM_FRACTION, would_breach_daily_loss_headroom

logger = logging.getLogger(__name__)

# The shared mutual-exclusion lock guarding every call to
# run_clerk_execution_check(), regardless of which of its TWO call
# sites triggers it — clerk_execution_job.py's own standalone hourly
# poll, AND mega_analysis_job.py's inline "run once immediately after a
# successful mega session" pass. Real concurrency bug found on audit:
# these are two independent OS processes: the inline call used to invoke
# run_clerk_execution_check() directly with zero lock protection, so a
# standalone hourly poll landing at the same moment an inline pass was
# still running could execute the exact same trade-decision logic
# concurrently — risking a duplicate order for the same symbol or a lost
# update to the settlement/state files (a read-modify-write race, even
# though each individual file write is itself atomic). Defined here
# (not duplicated as a private constant in each job script) so both
# callers acquire the literal same lock file, and can never drift apart
# on path or staleness window.
EXECUTION_LOCK_PATH = Path(__file__).resolve().parent.parent / "clerk_execution.lock"
# Comfortably above CLERK_EXECUTION_RUN_TIMEOUT_SECONDS (both callers
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
    path = Path(config.CLERK_EXECUTION_STATE_FILE)
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
    preserve_headline: bool = False,
) -> None:
    """Records this poll's outcome. `preserve_headline` (the fast lane): keep the prior last_attempt_utc / status / detail and
    only merge the per-symbol maps. Preserves `last_run_completed_utc`
    from whatever was there before (this function never touches it —
    only _mark_interval_ran does) so a call here can never accidentally
    erase the due-check's own rolling-cooldown marker regardless of call
    order.
    `last_verdicts` and `last_execution_results` are both MERGED onto
    the prior record (a fresh entry for a symbol overwrites its own old
    one; other symbols' entries from an earlier poll survive) rather
    than replaced wholesale — otherwise a poll that only touches some
    symbols (e.g. the CLERK_EXECUTION_MAX_PENDING_SETUPS cap truncated
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
    if preserve_headline:
        status = prior.get("last_status", status)
        detail = prior.get("last_detail", detail)
    payload = {
        "last_attempt_utc": prior.get("last_attempt_utc", now.isoformat()) if preserve_headline else now.isoformat(),
        "last_status": status,
        "last_detail": detail,
        "last_verdicts": merged_verdicts,
        "last_execution_results": merged_results,
        "last_tactical_verdicts": merged_tactical_verdicts,
        "last_run_completed_utc": prior.get("last_run_completed_utc"),
    }
    try:
        Path(config.CLERK_EXECUTION_STATE_FILE).write_text(json.dumps(payload, indent=2))
    except OSError as e:
        logger.warning("Could not write execution state file %s: %s", config.CLERK_EXECUTION_STATE_FILE, e)


def read_execution_progress() -> dict:
    """Best-effort read of this job's own LIVE, in-progress status —
    mirrors ai.mega_analysis.read_progress exactly, same standing
    "automated runs must be as visible as a manual button click"
    principle applied to this second unattended job."""
    path = Path(config.CLERK_EXECUTION_PROGRESS_FILE)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_execution_progress(message: str) -> None:
    payload = {"message": message, "updated_utc": datetime.now(timezone.utc).isoformat()}
    try:
        Path(config.CLERK_EXECUTION_PROGRESS_FILE).write_text(json.dumps(payload))
    except OSError as e:
        logger.warning("Could not write progress file %s: %s", config.CLERK_EXECUTION_PROGRESS_FILE, e)


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
    Consolidated to one ceiling, config.CLERK_EXECUTION_RUN_TIMEOUT_
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
    return 0 <= age_seconds < config.CLERK_EXECUTION_RUN_TIMEOUT_SECONDS and newer_than_last_attempt


# --- user-controlled enable/disable + review-frequency overrides (2026-08-23) ---


def read_clerk_execution_enabled() -> bool:
    """Whether the Execution Clerk is currently enabled — defaults
    to True on a missing/corrupt file, same opt-out-not-opt-in posture as
    ai.mega_analysis.read_mega_analysis_enabled. Checked once, at the top
    of run_clerk_execution_check itself (not duplicated at each of its
    three call sites — the standalone poll, mega_analysis_job.py's inline
    pass, and app.py's manual-button inline pass) so a single toggle
    genuinely governs the whole role regardless of what triggered it,
    matching this same session's earlier "only the trigger should differ"
    correction (see ai.ftmo_suggest.suggest_ftmo_portfolio's own
    docstring)."""
    path = Path(config.CLERK_EXECUTION_ENABLED_FILE)
    if not path.exists():
        return True
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return True
    return bool(data.get("enabled", True))


def set_clerk_execution_enabled(enabled: bool) -> None:
    try:
        Path(config.CLERK_EXECUTION_ENABLED_FILE).write_text(json.dumps({"enabled": enabled}))
    except OSError as e:
        logger.warning("Could not write enabled-flag file %s: %s", config.CLERK_EXECUTION_ENABLED_FILE, e)


def read_tactical_defense_enabled() -> bool:
    """Whether the Clerk's tactical-defense authority (DEFEND/EXIT on an
    already-filled position's short-term "trend" read, independent of
    Claude's own invalidation_condition) is enabled — defaults to FALSE,
    a deliberate break from read_clerk_execution_enabled's own usual
    opt-out-not-opt-in posture: this is fresh unattended authority over
    real money, added 2026-08-27 direct user request after a real gold
    position went from +$28 to -$61 while the mega session hadn't run in
    days and the Clerk had zero authority to react (see this module's
    own docstring). Checked separately from the whole-Clerk enabled
    toggle above so the long-proven Pending-Setup/invalidation
    mechanisms keep running unaffected by this new, separately-staged
    one — see run_clerk_execution_check's own shadow-mode behavior
    when this is False: the full tactical pipeline still runs and
    reports every poll, it just never touches merged_allocation."""
    path = Path(config.CLERK_TACTICAL_DEFENSE_ENABLED_FILE)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("enabled", False))


def set_tactical_defense_enabled(enabled: bool) -> None:
    try:
        Path(config.CLERK_TACTICAL_DEFENSE_ENABLED_FILE).write_text(json.dumps({"enabled": enabled}))
    except OSError as e:
        logger.warning(
            "Could not write tactical-defense enabled-flag file %s: %s",
            config.CLERK_TACTICAL_DEFENSE_ENABLED_FILE, e,
        )


def read_clerk_execution_interval_minutes() -> int:
    """The review-frequency minutes — user-overridable via app.py's
    picker, falling back to config.CLERK_EXECUTION_CHECK_INTERVAL_
    MINUTES on a missing/corrupt/non-positive value."""
    path = Path(config.CLERK_EXECUTION_INTERVAL_FILE)
    default = config.CLERK_EXECUTION_CHECK_INTERVAL_MINUTES
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text())
        minutes = int(data["minutes"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return default
    return minutes if minutes > 0 else default


def set_clerk_execution_interval_minutes(minutes: int) -> None:
    try:
        Path(config.CLERK_EXECUTION_INTERVAL_FILE).write_text(json.dumps({"minutes": minutes}))
    except OSError as e:
        logger.warning("Could not write interval file %s: %s", config.CLERK_EXECUTION_INTERVAL_FILE, e)


# --- interval due-check: a genuine rolling cooldown from the last run's
# own completion, not a fixed clock-aligned window ---------------------
#
# Replaced 2026-09-21, direct user report ("clerk seems to fire much
# faster than the interval, maybe continuously" — then, once that was
# traced to a grace-window bug and fixed, the direct follow-up "why not
# have the counter restart only after the clerk session finishes").
# That follow-up is a genuinely better design for THIS job, not just a
# preference: the OLD fixed-clock-window scheme (mirroring ai.mega_
# analysis.is_due's own once-daily version — a reasonable shape to COPY
# at the time, but that job genuinely needs calendar-day alignment,
# which this one never did) marked a window "ran" using the run's own
# COMPLETION time, not its start time. A run that starts in window
# [15:00-15:04] but takes 6 minutes (slow local-LLM calls are common in
# this account's own logs) completes at 15:06 — inside the NEXT window,
# [15:05-15:09] — so THAT fresh window gets silently marked "already
# ran" by a check that was really answering the PREVIOUS window's
# question, and the window it actually started in never gets marked at
# all. A rolling cooldown sidesteps this whole class of bug entirely:
# "due" simply means "at least `interval` minutes have passed since the
# last run finished," which is exactly what a repeating check wants and
# needs no fixed clock alignment, no grace-window sizing, and no
# separate bucket-key bookkeeping to keep in sync with it.


def is_execution_due(now_utc: datetime, state: dict | None = None) -> bool:
    """True once at least the configured review-frequency (read_clerk_
    execution_interval_minutes) has elapsed since the last run's own
    real completion (state["last_run_completed_utc"]) — or immediately,
    if this job has never completed a run before. A pure rolling
    cooldown: however long a given run actually takes, the NEXT one is
    always due exactly `interval` minutes after THAT run finished, never
    sooner and never meaningfully later (bounded only by however often
    clerk_execution_job.py's own OS-level/tray-timer poll actually
    lands) — see this section's own module-level comment for the real
    incident (a slow run silently consuming the wrong window's dedup
    slot) this replaced a fixed-clock-window scheme to fix."""
    state = state if state is not None else read_execution_state()
    last_completed = state.get("last_run_completed_utc")
    if not last_completed:
        return True
    try:
        last_completed_dt = datetime.fromisoformat(last_completed)
    except ValueError:
        return True
    interval = read_clerk_execution_interval_minutes()
    return now_utc >= last_completed_dt + timedelta(minutes=interval)


def next_execution_check_utc(now_utc: datetime, state: dict | None = None) -> datetime:
    """Best-effort next execution-check instant, purely for display (the
    real trigger is clerk_execution_job.py's own OS-level poll, not a
    precise clock instant): `now_utc` itself if a check is already due
    (i.e. due now, or as soon as the next OS-level poll lands), otherwise
    `interval` minutes after the last run's own real completion."""
    state = state if state is not None else read_execution_state()
    if is_execution_due(now_utc, state=state):
        return now_utc
    last_completed_dt = datetime.fromisoformat(state["last_run_completed_utc"])
    interval = read_clerk_execution_interval_minutes()
    return last_completed_dt + timedelta(minutes=interval)


def _mark_interval_ran(now_utc: datetime) -> None:
    prior = read_execution_state()
    prior["last_run_completed_utc"] = now_utc.isoformat()
    try:
        Path(config.CLERK_EXECUTION_STATE_FILE).write_text(json.dumps(prior, indent=2))
    except OSError as e:
        logger.warning("Could not write execution state file %s: %s", config.CLERK_EXECUTION_STATE_FILE, e)


def is_pre_weekend_cleanup_due(now_utc: datetime) -> bool:
    """True on a Friday, at/after config.CLERK_PRE_WEEKEND_CLEANUP_HOUR_
    UTC — a pure time-of-day/day-of-week check, deliberately with NO
    once-per-day dedup (see risk/apply_suggestion.py::
    compute_rebalance_plan's own docstring, and config.py's own comment,
    for the real incident — an INTC and a USDCHF pending order both
    survived into a weekend — this exists to prevent).

    Real gap caught on a self-recheck (2026-09-12): an earlier version
    of this function DID dedup to "once per Friday," tracked via a
    persisted state field. That's a genuine problem, not just extra
    complexity: this account's mega session can be re-run manually at
    any time (confirmed live, multiple times, in this exact project's
    own history) — if it's re-run AFTER the one-time window already
    fired and proposes a fresh non-crypto pending setup, that dedup
    would have let it sit uncaught through the whole weekend, silently
    recreating the exact incident this feature exists to prevent.
    Re-checking on every poll for the rest of Friday has no real
    downside: cancelling an order that's already gone (already
    cancelled by an earlier pass) is a harmless no-op — get_pending_
    orders() reflects real, current MT5 state, so there's nothing left
    to act on the second time around. Simpler AND safer than the
    dedup'd version it replaces."""
    if not config.CLERK_PRE_WEEKEND_CLEANUP_ENABLED:
        return False
    if now_utc.weekday() != 4:  # Monday=0 .. Friday=4
        return False
    return now_utc.hour >= config.CLERK_PRE_WEEKEND_CLEANUP_HOUR_UTC


# --- settlement (the pending-order lifecycle state machine) ---


def _load_settlement() -> dict:
    path = Path(config.CLERK_EXECUTION_SETTLEMENT_FILE)
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
    display (app.py's Execution Clerk panel) — same safe-default
    behavior as _load_settlement, just under a public name since this
    one's meant for an external caller rather than this module's own
    internal use."""
    return _load_settlement()


def _save_settlement(settlement: dict) -> None:
    path = Path(config.CLERK_EXECUTION_SETTLEMENT_FILE)
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
    eligible for a fresh Clerk check on a later poll."""
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


def _detect_newly_closed_symbols(old_settled: dict, new_settled: dict) -> list[str]:
    """Symbols whose settlement state transitioned from "filled" to
    "closed_after_fill" during THIS poll's _reconcile_settlement call —
    the only moment a real close can be detected at all, since
    closed_after_fill is otherwise permanent for the rest of this mega-
    session cycle (see _reconcile_settlement's own docstring). Doesn't
    distinguish Clerk's own close_position call from MT5 hitting the
    stop/target from a manual close in the terminal — all three collapse
    to the same "not in positions anymore" signal, which is fine for a
    trade journal (it records what happened, not who did it)."""
    newly_closed = []
    for symbol, new_rec in new_settled.items():
        if new_rec.get("state") != "closed_after_fill":
            continue
        old_rec = old_settled.get(symbol)
        if old_rec is not None and old_rec.get("state") == "filled":
            newly_closed.append(symbol)
    return newly_closed


def _detect_newly_filled_symbols(old_settled: dict, new_settled: dict) -> list[tuple[str, dict]]:
    """Trade Journal support (direct user request 2026-09-19): the same
    "transition detection" pattern as _detect_newly_closed_symbols
    above, one state earlier — "order_placed" to "filled". Returns the
    new record alongside the symbol (not just the symbol) since the
    caller needs order_ticket/entry.price from it, and this function is
    the one place that already knows exactly which records qualify."""
    newly_filled = []
    for symbol, new_rec in new_settled.items():
        if new_rec.get("state") != "filled":
            continue
        old_rec = old_settled.get(symbol)
        if old_rec is not None and old_rec.get("state") == "order_placed":
            newly_filled.append((symbol, new_rec))
    return newly_filled


def _compute_realized_r(entry: dict, open_price: float, close_price: float) -> float | None:
    """Realized R-multiple from the position's own ORIGINAL recorded stop
    distance (entry["stop_loss"]) against its real close price — None
    (never a fabricated number) when stop_loss is missing/falsy, side is
    neither "buy" nor "sell", or the implied risk is non-positive (bad/
    stale data, e.g. a stop on the wrong side of entry)."""
    side = entry.get("side")
    stop_loss = entry.get("stop_loss")
    if not stop_loss or side not in ("buy", "sell"):
        return None
    if side == "buy":
        risk = open_price - stop_loss
        reward = close_price - open_price
    else:
        risk = stop_loss - open_price
        reward = open_price - close_price
    if risk <= 0:
        return None
    return reward / risk


def _format_closed_trade_note(symbol: str, entry: dict, closed_trade: ClosedTrade | None) -> str:
    """Pure formatting -> Obsidian-vault markdown for one closed trade.
    Never fabricates a price/P&L/R figure it doesn't actually have —
    degrades to an honest disclosure line instead (see closed_trade=None
    branch) rather than skipping the note or guessing."""
    side = entry.get("side", "?")
    reason = entry.get("reason") or "(no thesis recorded)"
    invalidation = entry.get("invalidation_condition")

    lines = ["---", "tags: [trade, closed]", "---", "", f"# {symbol} — {side} closed", ""]

    if closed_trade is not None:
        r = _compute_realized_r(entry, closed_trade.open_price, closed_trade.close_price)
        r_text = f"{r:+.2f}R" if r is not None else "not computable (no recorded stop distance)"
        lines += [
            f"Opened {closed_trade.opened_at.isoformat()} at {closed_trade.open_price}, "
            f"closed {closed_trade.closed_at.isoformat()} at {closed_trade.close_price} "
            f"({closed_trade.volume} lots).",
            "",
            f"**Realized P&L: {closed_trade.profit:+.2f}** (net of swap/commission). Realized: {r_text}.",
        ]
    else:
        lines += [
            "Real closing price/P&L could not be matched to an MT5 deal record this poll "
            "(a real, disclosed gap — not fabricated) — only the original order's own "
            "intended terms are recorded below.",
            "",
            f"Intended entry: {entry.get('price')}, stop: {entry.get('stop_loss')}, "
            f"target: {entry.get('take_profit')}.",
        ]

    lines += ["", f"**Original thesis:** {reason}"]
    if invalidation:
        lines.append(f"**Invalidation condition:** {invalidation}")
    lines += ["", f"[[{symbol}]]", "[[Trade Journal]]"]
    return "\n".join(lines)


def _export_closed_trade_notes(newly_closed_symbols: list[str], old_settled: dict) -> None:
    """Best-effort side effect, same posture as _export_chart_overlay:
    writes one real markdown note per newly-closed symbol into the
    Obsidian vault's Trades/ folder. Deliberately swallows every
    exception at both the per-symbol and outer level — a locked file, an
    unmounted vault path, or an MT5 deal-history hiccup must never
    interrupt or fail the real execution check that called this."""
    if not newly_closed_symbols:
        return
    try:
        vault_dir = Path(config.OBSIDIAN_VAULT_PATH) / "Trades"
        vault_dir.mkdir(parents=True, exist_ok=True)

        now_utc = datetime.now(timezone.utc)
        closed_trades_by_symbol: dict[str, ClosedTrade] = {}
        try:
            # No explicit date_to: get_history_deals' own default deliberately reaches a day into
            # the future to absorb the broker's server-clock skew — passing `now_utc` here excluded a
            # just-closed deal, so every closure came back P&L-less (real bug, 2026-09-22).
            deals = get_history_deals(now_utc - timedelta(days=7))
            for t in group_closed_trades(deals):
                closed_trades_by_symbol.setdefault(t.symbol, t)  # already newest-first
        except Exception:
            logger.warning(
                "Vault trade-journal: could not fetch MT5 deal history this poll — "
                "notes below will degrade to intended-terms-only.", exc_info=True,
            )

        for symbol in newly_closed_symbols:
            try:
                entry = old_settled.get(symbol, {}).get("entry", {})
                closed_trade = closed_trades_by_symbol.get(symbol)
                text = _format_closed_trade_note(symbol, entry, closed_trade)
                timestamp = (closed_trade.closed_at if closed_trade else now_utc).strftime("%Y-%m-%d_%H%M%S")
                filename = f"{symbol} {timestamp} {entry.get('side', 'trade')}.md"
                path = vault_dir / filename
                tmp_path = path.with_suffix(path.suffix + ".tmp")
                tmp_path.write_text(text, encoding="utf-8")
                os.replace(tmp_path, path)
            except Exception:
                logger.warning(
                    "Vault trade-journal export failed for %s this poll (cosmetic only, continuing).",
                    symbol, exc_info=True,
                )
    except Exception:
        logger.warning("Vault trade-journal export failed this poll (cosmetic only, continuing).", exc_info=True)


def _determine_close_cause(entry: dict, closed_trade: ClosedTrade | None, story) -> str:
    """Best-effort classification of why a trade closed. The Trade
    Journal's own last recorded tactical action (a Clerk EXIT genuinely
    applied shortly before this poll noticed the close) takes priority
    over inferring from price — it's a direct causal record, not a
    guess. Falls back to comparing the real close price against the
    ORIGINALLY recorded stop/target (a real, if imperfect, proxy — a
    stop/target amended mid-life by a tactical action would need
    comparing against its latest value instead, which is out of scope
    here) before giving up and calling it manual/unknown rather than
    fabricating a specific cause it can't actually support."""
    if story is not None:
        for event in reversed(story.events):
            if event.type == "tactical_action" and event.data.get("tier") == "exit" and event.data.get("applied"):
                return "clerk_tactical_exit"
    if closed_trade is not None:
        stop_loss = entry.get("stop_loss")
        take_profit = entry.get("take_profit")
        close_price = closed_trade.close_price
        tolerance = close_price * 0.001  # 0.1% of price -- real fills rarely land exactly on the level
        if stop_loss and abs(close_price - stop_loss) <= tolerance:
            return "stop_loss_hit"
        if take_profit and abs(close_price - take_profit) <= tolerance:
            return "take_profit_hit"
    return "manual_or_unknown"


def _maybe_register_continuation_watch(symbol: str, story_id: str, closed_trade: ClosedTrade) -> None:
    """Continuation Watch (2026-09-28 plan, point 1): only ever called right above for a WINNING close.
    Best-effort, same posture as the rest of this function's own callers — a failure here must never break
    real close-journaling, so it gets its own try/except and never propagates."""
    try:
        if not config.CONTINUATION_WATCH_ENABLED:
            return
        atr_window = fetch_mt5_price_history_range(
            symbol, "M5", closed_trade.closed_at - timedelta(hours=2), closed_trade.closed_at,
        )
        atr_at_close = compute_atr(atr_window) if not atr_window.empty else None
        continuation_hunter.register_watch(
            symbol, story_id, closed_trade.side, closed_trade.close_price, closed_trade.closed_at, atr_at_close,
        )
    except Exception:
        logger.warning("Continuation watch: could not register a watch for %s (cosmetic only).", symbol, exc_info=True)


def _record_trade_journal_closures(newly_closed_symbols: list[str], old_settled: dict) -> None:
    """Trade Journal support (direct user request 2026-09-19): a
    separate, parallel best-effort side effect alongside
    _export_closed_trade_notes above — deliberately NOT a modification
    of that function, so its own existing behavior/tests stay completely
    untouched. Does its own independent MT5 deal-history fetch (same
    pattern; the real duplicate cost is negligible since this only runs
    on an actual close, not every poll) rather than sharing state with
    that function, keeping the two fully decoupled."""
    if not newly_closed_symbols:
        return
    try:
        now_utc = datetime.now(timezone.utc)
        closed_trades_by_symbol: dict[str, ClosedTrade] = {}
        try:
            # No explicit date_to: get_history_deals' own default deliberately reaches a day into
            # the future to absorb the broker's server-clock skew — passing `now_utc` here excluded a
            # just-closed deal, so every closure came back P&L-less (real bug, 2026-09-22).
            deals = get_history_deals(now_utc - timedelta(days=7))
            for t in group_closed_trades(deals):
                closed_trades_by_symbol.setdefault(t.symbol, t)
        except Exception:
            logger.warning(
                "Trade journal: could not fetch MT5 deal history this poll — "
                "closed events below will degrade to a bare P&L-less record.", exc_info=True,
            )
        for symbol in newly_closed_symbols:
            try:
                entry = old_settled.get(symbol, {}).get("entry", {})
                closed_trade = closed_trades_by_symbol.get(symbol)
                story = trade_journal.find_open_story(symbol)
                cause = _determine_close_cause(entry, closed_trade, story)
                net_pnl = closed_trade.profit if closed_trade is not None else None
                trade_journal.record_closed(symbol, net_pnl, cause)
                if net_pnl is not None and net_pnl > 0 and story is not None and closed_trade is not None:
                    _maybe_register_continuation_watch(symbol, story.story_id, closed_trade)
            except Exception:
                logger.warning("Trade journal: record_closed failed for %s (cosmetic only).", symbol, exc_info=True)
    except Exception:
        logger.warning("Trade journal: closure recording failed this poll (cosmetic only).", exc_info=True)


def _reconcile_journal_unknown_closures(settled: dict) -> None:
    """Retries P&L matching for journal stories that closed with no
    matching MT5 deal at the time (status "closed_unknown"). Cheap when
    there's nothing to fix: no MT5 call unless such a story exists."""
    try:
        if not trade_journal.has_unknown_closures():
            return
        now_utc = datetime.now(timezone.utc)
        closed_trades = group_closed_trades(get_history_deals(now_utc - timedelta(days=7)))

        def _cause(closed_trade: ClosedTrade, story) -> str:
            record = settled.get(story.symbol) or {}
            entry = dict(record.get("entry") or {})
            live_stop = (record.get("tactical") or {}).get("last_stop_loss")
            if live_stop:
                entry["stop_loss"] = live_stop
            return _determine_close_cause(entry, closed_trade, story)

        reconciled = trade_journal.reconcile_unknown_closures(closed_trades, _cause)
        if config.CLERK_VAULT_JOURNAL_ENABLED:
            for story, closed_trade in reconciled:
                _repair_closed_trade_note(story, closed_trade, (settled.get(story.symbol) or {}).get("entry"))
    except Exception:
        logger.warning("Trade journal: unknown-closure reconciliation failed this poll (cosmetic only).", exc_info=True)


def _repair_closed_trade_note(story, closed_trade: ClosedTrade, settled_entry: dict | None) -> None:
    """Rewrites the vault's Trades/ note that was written P&L-less when
    this story's close was first noticed (its filename carries the close
    poll's own timestamp, same as the journal's `closed` event) now that
    the real deal is matched. No note found -> nothing to repair."""
    try:
        closed_event = next((e for e in reversed(story.events) if e.type == "closed"), None)
        if closed_event is None:
            return
        stamp = datetime.fromisoformat(closed_event.timestamp_utc).astimezone(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        vault_dir = Path(config.OBSIDIAN_VAULT_PATH) / "Trades"
        for path in vault_dir.glob(f"{story.symbol} {stamp} *.md"):
            summary = trade_journal.summarize_story(story)
            entry = dict(settled_entry) if settled_entry else {
                "side": summary["side"], "price": summary["planned_price"], "stop_loss": summary["planned_stop"],
                "take_profit": summary["planned_target"], "reason": summary["thesis"],
                "invalidation_condition": summary["invalidation_condition"],
            }
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(_format_closed_trade_note(story.symbol, entry, closed_trade), encoding="utf-8")
            os.replace(tmp_path, path)
    except Exception:
        logger.warning("Vault trade note repair failed for %s (cosmetic only).", story.symbol, exc_info=True)


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
    neither "hold" nor "amend_position" (see run_clerk_execution_
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
        # position.price_open, not price_current -- this backfilled entry
        # feeds straight into compute_rebalance_plan THIS SAME poll (via
        # carried_forward), whose own sizing for an ALREADY-HELD position
        # anchors to the position's real price_open, never a live-
        # tracking price (see that function's own "sizing_price" comment
        # for the real 2026-09-08 incident this same coupling caused
        # elsewhere — a live anchor here would size against a DIFFERENT
        # stop distance than compute_rebalance_plan actually uses,
        # breaking the "same lots" invariant pct_for_target_lots exists
        # to guarantee, same as the fix in _validate_and_apply_tactical_
        # verdict above).
        pct = pct_for_target_lots(symbol, position.volume, position.price_open, position.sl, account_equity, get_spec)
        if pct is None:
            continue
        entry = {
            "pct": pct, "price": position.price_open, "stop_loss": position.sl,
            "take_profit": position.tp, "side": position.side, "reason": "", "invalidation_condition": None,
        }
        backfilled[symbol] = asdict(
            SymbolSettlement(origin="pending_setup", state="filled", entry=entry, order_ticket=None)
        )
    return backfilled


def _within_pct_tolerance(a: float, b: float, tolerance_pct: float) -> bool:
    reference = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / reference * 100 <= tolerance_pct


def _iter_stop_drift(
    positions_by_symbol: dict[str, Position],
    settled: dict,
    carried_forward: dict[str, AllocationEntry],
    tolerance_pct: float = config.AMEND_TOLERANCE_PCT,
) -> list[tuple[str, Position, float]]:
    """Shared, pure comparison core for both _detect_external_stop_drift
    (warn-only) and _restore_external_stop_drift (warn + auto-correct) —
    kept in one place so the two can never drift apart on what counts as
    genuine, unexplained drift. See either caller's own docstring for the
    full incident history and reasoning. Returns (symbol, live position,
    recorded_sl) for every currently-held symbol with a real, unexplained
    live-vs-recorded stop-loss mismatch beyond `tolerance_pct`.

    Compares each currently-held position's REAL, live MT5 stop-loss
    against the stop-loss THIS APP most recently recorded for it — a
    tactical DEFEND's own `tactical.persisted_stop_loss` if one has fired
    this mega-session cycle (see _validate_and_apply_tactical_verdict),
    else the settlement record's original `entry.stop_loss`.

    Real bug found on self-review (2026-09-09), fixed here: comparing
    directly against `entry.stop_loss` alone is UNSOUND on its own —
    per _backfill_settlement_for_held_positions' own docstring, neither
    a plain "hold" nor a genuine, app-driven "amend_position" (a FRESH
    mega session legitimately changing an already-held position's stop)
    ever refreshes that field once a settlement record exists for a
    symbol. Comparing the live stop straight against a `entry.stop_loss`
    that's gone stale purely because this app itself already moved on
    to a new target would flag every ordinary, legitimate stop update as
    "external interference" — exactly the false-alarm noise that would
    make this detector worthless. The fix: also resolve `carried_forward`
    (this SAME poll's fully-resolved target for the symbol, built moments
    ago by _build_carried_forward_allocation using the exact same
    tactical-persistence-then-entry precedence) — if the target ITSELF
    already differs from the recorded value, this app's own normal
    execution flow already intends to change the stop this poll (that's
    not drift, that's just pending, expected work, and skipping it here
    doesn't hide anything: the amend's own success/failure is logged
    separately when compute_rebalance_plan's plan actually runs). Only
    when the target and the recorded value AGREE — meaning nothing about
    this app's own intent has changed — does a live-stop mismatch have no
    other explanation, i.e. it's genuinely unaccounted-for drift.

    Called right after settlement AND carried_forward are both built,
    before this poll's own invalidation/tactical logic runs, so it only
    ever surfaces drift that happened BETWEEN polls. A freshly-backfilled
    symbol (first time this app has ever seen it) trivially matches,
    since backfill sources its recorded stop straight from the same live
    position — never a false positive there.

    Pure and side-effect-free so it's cheap to unit test."""
    drifted: list[tuple[str, Position, float]] = []
    for symbol, position in positions_by_symbol.items():
        if position.sl is None:
            continue
        rec = settled.get(symbol)
        if rec is None:
            continue
        tactical = rec.get("tactical") or {}
        recorded_sl = tactical.get("sentinel_stop")
        if recorded_sl is None:
            recorded_sl = tactical.get("persisted_stop_loss")
        if recorded_sl is None:
            recorded_sl = (rec.get("entry") or {}).get("stop_loss")
        if recorded_sl is None:
            continue

        target_entry = carried_forward.get(symbol)
        target_sl = target_entry.stop_loss if target_entry is not None else None
        if target_sl is not None and not _within_pct_tolerance(target_sl, recorded_sl, tolerance_pct):
            # This app's own fresh target already disagrees with what was
            # last recorded — normal execution flow will amend toward it
            # this poll (or already tried and logged its own outcome
            # separately). Nothing unexplained to flag here.
            continue

        if not _within_pct_tolerance(position.sl, recorded_sl, tolerance_pct):
            drifted.append((symbol, position, recorded_sl))
    return drifted


def _detect_external_stop_drift(
    positions_by_symbol: dict[str, Position],
    settled: dict,
    carried_forward: dict[str, AllocationEntry],
    tolerance_pct: float = config.AMEND_TOLERANCE_PCT,
) -> list[str]:
    """Detection-only safety net for a real, confirmed incident class:
    2026-09-08, a live XAUUSD position's real MT5 stop-loss was found (on
    forensic review, AFTER the trade had already closed) to have moved
    from 4378.00 — exactly what this app's own mega session placed it at
    — to 4390.03, with NO matching entry anywhere in this app's own logs
    (clerk_execution_log.txt, mega_analysis_log.txt, streamlit_log.txt)
    for any Clerk tactical action, mega-session amend, or manual "Apply
    Suggestion" click in that window. The leading explanation is MT5's
    own terminal-side trailing-stop feature (or some other actor entirely
    outside this codebase) modifying the position directly — this app
    had no way to notice until the numbers stopped adding up under
    after-the-fact review, by which point the trade had already been
    stopped out by the drifted level on perfectly ordinary noise.

    Thin formatter over _iter_stop_drift's own shared comparison core —
    see that function's docstring for the full detection reasoning.

    Returns one human-readable warning string per symbol with genuine,
    unexplained drift beyond `tolerance_pct` (same convention as config.
    AMEND_TOLERANCE_PCT elsewhere in this codebase). Detection only —
    never modifies anything itself; this is now the warn-only half of a
    pair — see _restore_external_stop_drift (added 2026-09-15/16) for the
    auto-correcting half, which actually acts on exactly what this
    function detects. Pure and side-effect-free so it's cheap to unit
    test; still called on its own by nothing in the live poll loop as of
    this pass, kept for direct unit-test coverage of the pure detection
    logic in isolation."""
    return [
        f"{symbol}: live MT5 stop-loss ({position.sl}) no longer matches this app's own "
        f"last-recorded stop-loss ({recorded_sl}) for this position. Something outside this "
        "app's own Clerk/mega-session actions changed it directly (a terminal-side trailing "
        "stop is the leading suspect — see this function's own docstring for the 2026-09-08 "
        "XAUUSD incident this was built from). Not auto-corrected — review the position and "
        "MT5 terminal settings directly."
        for symbol, position, recorded_sl in _iter_stop_drift(
            positions_by_symbol, settled, carried_forward, tolerance_pct
        )
    ]


def _restore_external_stop_drift(
    positions_by_symbol: dict[str, Position],
    settled: dict,
    carried_forward: dict[str, AllocationEntry],
    tolerance_pct: float = config.AMEND_TOLERANCE_PCT,
) -> list[str]:
    """Auto-corrects genuine external stop-loss drift instead of only
    ever warning about it — direct user decision 2026-09-15/16, after a
    second, now-confirmed real incident on top of the 2026-09-08 XAUUSD
    one _detect_external_stop_drift/_iter_stop_drift were originally
    built from: XAGUSD's live stop silently reverted from Clerk's own
    tactical-DEFEND-tightened 63.97 back down to the original 63.13
    sometime between 03:07 and 09:40 UTC on 2026-09-14, was correctly
    flagged by the detector on every single poll for the next ~4 hours,
    and was never corrected because detection alone never acts — the
    position was then stopped out at the worse, reverted level instead
    of the break-even-or-better level Clerk had actually already
    achieved. Direct instruction: "do not wait for any human
    intervention just act and save the equity."

    Restores the LIVE MT5 stop-loss straight back to `recorded_sl` — the
    exact same value _iter_stop_drift already computes as "what this app
    itself last, deliberately decided" (a tactical DEFEND's own
    persisted_stop_loss if one has fired this cycle, else the original
    entry.stop_loss) — via modify_position_sltp. Never invents a new
    number and never re-derives one from scratch: this re-asserts a
    value Clerk's own logic already validated (a tactical DEFEND's own
    proposed stop already passed the never-widen/sane-side-of-price
    guardrails in _validate_and_apply_tactical_verdict BEFORE it was
    ever persisted, and the original entry.stop_loss was itself already
    vetted at trade-open time) — not proposing a fresh one, so there's no
    new judgment call being made here that a human would need to review,
    only re-applying one already made.

    Restores unconditionally, on either side (tighter-than-live or
    looser-than-live): drift in EITHER direction is a real anomaly this
    app didn't intend — an unwanted tightening can trigger a premature
    stop-out on ordinary noise (the 2026-09-08 XAUUSD case), an unwanted
    loosening can turn a well-managed trade into a bigger loss than
    intended (the 2026-09-14 XAGUSD case) — and the fix in both cases is
    the same: put back exactly what Clerk itself last decided, rather
    than second-guessing which direction is "worse" this time.

    take_profit is ALWAYS resent as the position's own current live tp
    (never touched) — modify_position_sltp's own contract requires both
    fields on every call, and this function only ever intends to correct
    the stop.

    Deliberately NOT gated on ftmo_heat_blocked (same reasoning as a pure
    "cancel" elsewhere in this module, see run_clerk_execution_check's
    own heat-block handling): restoring this app's own already-validated,
    already-more-defensive-by-construction stop is a risk-REDUCING
    action, never a new/increased-exposure one, so a heat block meant to
    stop new risk from being added must never hold it back.

    A restore can genuinely fail even though nothing is wrong with this
    app's own logic: the whole point of `recorded_sl` is that it was
    valid at the moment Clerk last set it, but by the time drift is
    caught, live price may have already moved PAST it (a real, live-
    reconstructed scenario found 2026-09-16 while replaying the XAGUSD
    incident: a DEFEND had tightened the stop to within 0.008 of price
    at the instant it fired, and price kept sliding afterward — by the
    time a restore would run, 63.97 could already be on the wrong side
    of the live price for a buy, which MT5 rejects outright as an
    invalid stop). Direct follow-up instruction after that replay: "Clerk
    should fall back to closing the position at market immediately
    whenever the intended restore is rejected as invalid, rather than
    just logging the failure and stopping there... what worst could
    happen, i would lose some more cash but its still better than bigger
    disaster." So ANY restore failure — not only an invalid-stops
    rejection specifically, since a broker can reject an SL/TP amend for
    other reasons too and this app has no reliable way to tell those
    apart from the outside, and the same "still exposed at an unintended
    stop" risk applies regardless of why — now falls back to closing the
    position outright via close_position. This deliberately accepts a
    strictly worse worst case than doing nothing (a market-order exit at
    whatever price is live, versus staying open at an unintended stop)
    in exchange for eliminating the far worse case this whole mechanism
    exists to prevent: a position left running, unprotected, indefinitely,
    because a stop couldn't be silently fixed and nothing else was tried.

    Returns one human-readable outcome string per symbol attempted
    (restored, closed-as-fallback, or a rare failure of both) for the
    caller to log/notify — same shape as _detect_external_stop_drift's
    own return, so callers can treat both uniformly."""
    outcomes: list[str] = []
    for symbol, position, recorded_sl in _iter_stop_drift(positions_by_symbol, settled, carried_forward, tolerance_pct):
        live_sl = position.sl
        try:
            result = modify_position_sltp(position, stop_loss=recorded_sl, take_profit=position.tp)
        except MT5ConnectionError as e:
            outcomes.append(_close_after_failed_stop_restore(symbol, position, live_sl, recorded_sl, str(e)))
            continue
        if result.success:
            outcomes.append(
                f"{symbol}: detected external stop-loss drift (live was {live_sl}, this app's own "
                f"recorded value is {recorded_sl}) — something outside this app's own Clerk/mega-"
                "session actions changed it directly (a terminal-side trailing stop is the leading "
                f"suspect); automatically restored the stop-loss to {recorded_sl}."
            )
        else:
            outcomes.append(
                _close_after_failed_stop_restore(symbol, position, live_sl, recorded_sl, result.comment)
            )
    return outcomes


def _close_after_failed_stop_restore(
    symbol: str, position: Position, live_sl: float, recorded_sl: float, restore_failure: str
) -> str:
    """The fallback half of _restore_external_stop_drift's own contract
    (see its docstring for the full reasoning and the direct 2026-09-16
    instruction this implements) — called only once a restore attempt
    has already genuinely failed. Closes the WHOLE position at market
    (never a partial) via close_position, since a partially-fixed
    exposure is still an unintended, un-vetted risk sitting on the
    account. Never raises: an MT5ConnectionError from close_position
    itself is caught and folded into the returned outcome string like
    every other failure here, so a bad connection can't crash the poll
    loop or silently swallow the fact that BOTH the restore and the
    close-fallback failed — that combination is the one genuinely
    dangerous residual case (a position left open, unprotected, with
    neither fix having landed), so its own message says so explicitly
    rather than reading like an ordinary failure."""
    try:
        close_result = close_position(position)
    except MT5ConnectionError as e:
        return (
            f"{symbol}: detected external stop-loss drift (live {live_sl} vs this app's own "
            f"recorded {recorded_sl}) — auto-restore failed ({restore_failure}), and the market-close "
            f"fallback ALSO failed: {e}. This position is STILL OPEN with an unintended stop-loss — "
            "review the position and MT5 terminal directly, urgently."
        )
    if close_result.success:
        return (
            f"{symbol}: detected external stop-loss drift (live {live_sl} vs this app's own "
            f"recorded {recorded_sl}) — the intended restore was no longer valid ({restore_failure}), "
            "so this app closed the position at market instead of leaving it exposed at an unintended "
            "stop level."
        )
    return (
        f"{symbol}: detected external stop-loss drift (live {live_sl} vs this app's own "
        f"recorded {recorded_sl}) — auto-restore failed ({restore_failure}), and the market-close "
        f"fallback ALSO failed: {close_result.comment}. This position is STILL OPEN with an unintended "
        "stop-loss — review the position and MT5 terminal directly, urgently."
    )


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
        entry_mode=normalise_entry_mode(d.get("entry_mode")),
    )


def _apply_persisted_tactical(entry: AllocationEntry, tactical: dict | None) -> AllocationEntry:
    """The persisted tactical state (a DEFEND's reduced risk %, tightened stop and target) laid over a settlement-record entry.
    The immediate-allocation branch of _build_carried_forward_allocation always did this; the pending-setup-origin branches read
    rec["entry"] as-is, so on the poll AFTER a DEFEND the baseline snapped back to the ORIGINAL stop and size and the plan
    amended a trailed stop back out (or re-bought a partial close). Only ever tightens: applies when the persisted risk is
    below the entry's own."""
    tactical = tactical or {}
    persisted_pct = tactical.get("persisted_pct")
    if persisted_pct is None or persisted_pct >= entry.pct:
        return entry
    return _with_changes(
        entry, pct=persisted_pct,
        stop_loss=tactical.get("persisted_stop_loss", entry.stop_loss),
        take_profit=tactical.get("persisted_take_profit", entry.take_profit),
    )


def _build_carried_forward_allocation(
    immediate_allocation_raw: dict, settled: dict, held_symbols: frozenset[str] = frozenset()
) -> dict[str, AllocationEntry]:
    """The base target mix before any THIS-POLL Clerk verdicts are
    merged in: every immediate_allocation symbol EXCEPT one that's
    already closed_after_fill this cycle (don't reopen a trade that
    already ran its course), PLUS every settled symbol in state "filled"
    (OR with a real live position per `held_symbols` — see that
    parameter's own note below) that isn't already covered by
    immediate_allocation (so compute_rebalance_plan's normal hold/
    resize/close diffing keeps managing an already-open, previously-
    conditional position).

    `held_symbols` (added 2026-09-09, hardening against a real incident
    class found live 2026-09-08/09 — see run_clerk_execution_check's own
    tactical_candidates comment for the specific trade this was traced
    from): the set of symbols with a REAL, currently-open MT5 position,
    fetched fresh this same poll — passed in so the SECOND loop below
    (settled-but-not-in-immediate_allocation symbols, e.g. a fired
    Pending Setup) doesn't wrongly exclude a symbol whose settlement
    record is stuck in "order_placed" for a reason unrelated to whether
    it's actually held: settlement tracks exactly one record per symbol,
    and a separate, still-resting "top-up" pending order on an already-
    filled symbol keeps that record in "order_placed" indefinitely, even
    though a real position exists needing to be carried forward and
    managed. The real 2026-09-08 USDCAD trade that surfaced this pattern
    happened to also be a key of immediate_allocation_raw that cycle, so
    it was actually saved by the FIRST loop below regardless of state —
    this parameter closes the same gap for a symbol that ISN'T (a
    Pending-Setup-originated position with no fresh mega-session mention
    this cycle), which the second loop's own state=="filled" gate alone
    cannot protect. Defaults to empty (unchanged behavior) for any
    caller that hasn't been updated to pass it.

    An immediate_allocation symbol with an outstanding order_placed
    record is DELIBERATELY still carried forward here, not excluded —
    real bug found on audit: excluding it would make compute_rebalance_
    plan see an ALREADY-HELD symbol (e.g. one with a filled base
    position plus a still-unfilled top-up "increase" order) as "held but
    missing from the target," which defaults to a 0% target and would
    force-close the entire real position just because an unrelated
    top-up order happened to still be pending. Not resubmitting a
    DUPLICATE order for that same still-open delta is handled instead at
    the execution-loop level (see run_clerk_execution_check), by
    skipping only the actual order_send call, not by hiding the whole
    symbol from the plan. A settled symbol NOT in immediate_allocation
    (a fired Pending Setup) in state order_placed is correctly excluded
    by the second loop below (only "filled" is carried forward there) —
    it has zero held volume yet, so there is nothing to protect from a
    forced close, and including it here would just resubmit a duplicate
    fresh-open order every poll instead.

    Real incident, 2026-09-10 (INTC) — the OTHER direction of the same
    "don't let one order's own bookkeeping force-close an unrelated real
    position" family of bug: when a symbol IS a key of immediate_
    allocation_raw for a reason unrelated to its current real position —
    a stale "cancel this now-superseded pending order" pct=0 directive,
    while a SEPARATE Pending Setup for the SAME symbol fired hours later
    the SAME mega-session cycle and has since filled — this loop now
    detects that via settlement's own origin == "pending_setup" field
    and carries forward THAT record's real entry instead of the stale
    immediate_allocation_raw one. Without this, the first loop's own
    pct=0 silently won (this function runs before the merge loop's own
    per-poll Clerk verdicts, so nothing downstream ever saw the real,
    freshly-filled position as anything other than "target: 0%") and a
    brand-new position was closed 3 minutes after opening.

    A symbol whose tactical-defense state carries a persisted_pct (see
    _validate_and_apply_tactical_verdict's own DEFEND branch) gets THAT
    reduced pct/stop/target as its baseline here instead of the raw
    mega-session JSON's original, unreduced ones. Real bug found live
    2026-09-03 (USDCHF oscillated 0.13, 0.09, 0.13, 0.06, 0.13 lots
    across three separate tickets in about an hour, on a chart with no
    real trend to justify any of it): without this, a tactical DEFEND's
    protective size reduction only ever lasted until the next poll,
    which rebuilt the baseline straight from the untouched original
    target and bought the reduced amount right back. The persisted
    values are wiped the moment a genuinely fresh mega session runs (see
    _reset_settlement_for_new_session, called before this function on
    every new generated_utc), so a reduction never outlives the session
    that produced it -- a new session's own explicit target always wins."""
    carried: dict[str, AllocationEntry] = {}
    for symbol, raw in immediate_allocation_raw.items():
        if symbol.upper() == "CASH":
            continue
        rec = settled.get(symbol)
        if rec is not None and rec.get("state") == "closed_after_fill":
            continue
        # Real incident, 2026-09-10 (INTC): a symbol can appear in
        # immediate_allocation_raw for a reason entirely UNRELATED to a
        # real, currently-held position that originated from a SEPARATE
        # Pending Setup trigger fired later in the SAME mega-session
        # cycle — e.g. "cancel this now-stale pending order" (pct=0), a
        # directive about a ticket that had already gone unfilled for
        # too long, superseded hours later by a fresh Pending Setup on
        # the SAME symbol that then triggered, filled, and became a real
        # position. Because this loop runs BEFORE the second one below
        # (which is where a Pending-Setup-originated "filled" symbol
        # would normally be carried forward from its own settlement
        # record), and the second loop explicitly skips anything already
        # claimed here, the STALE immediate_allocation_raw entry (still
        # pct=0 from the old, already-cancelled ticket, since the mega
        # session's own generated_utc hasn't changed) silently won —
        # closing the brand-new real position 3 minutes after it filled,
        # citing "already targeted at 0% this poll by an unrelated
        # decision." rec.get("origin") == "pending_setup" is the exact
        # discriminator: it means this symbol's real, live position (or
        # its settlement record, once filled) did NOT come from this
        # immediate_allocation entry at all, so that entry's own pct is
        # simply stale here and must never be allowed to override the
        # real, separately-tracked Pending-Setup target.
        if rec is not None and rec.get("origin") == "pending_setup" and (
            rec.get("state") == "filled" or symbol in held_symbols
        ):
            carried[symbol] = apply_sentinel_ratchet(
                _apply_persisted_tactical(_allocation_entry_from_dict(rec["entry"]), rec.get("tactical")), rec.get("tactical")
            )
            continue
        entry = _allocation_entry_from_dict(raw)
        tactical = (rec or {}).get("tactical") or {}
        persisted_pct = tactical.get("persisted_pct")
        if persisted_pct is not None and persisted_pct < entry.pct:
            entry = AllocationEntry(
                pct=persisted_pct,
                price=entry.price,
                stop_loss=tactical.get("persisted_stop_loss", entry.stop_loss),
                take_profit=tactical.get("persisted_take_profit", entry.take_profit),
                side=entry.side,
                reason=f"{entry.reason} (tactically reduced to {persisted_pct:.2f}% risk this session)",
                invalidation_condition=entry.invalidation_condition,
                entry_mode=entry.entry_mode,
            )
        carried[symbol] = apply_sentinel_ratchet(entry, tactical)

    for symbol, rec in settled.items():
        if symbol in carried or symbol in immediate_allocation_raw:
            continue
        if rec.get("state") != "filled" and symbol not in held_symbols:
            continue
        carried[symbol] = apply_sentinel_ratchet(
            _apply_persisted_tactical(_allocation_entry_from_dict(rec["entry"]), rec.get("tactical")), rec.get("tactical")
        )

    return carried


# --- Clerk verdict: a hard, fail-safe boolean, not a feeling ---

_FINAL_VERDICT_PATTERN = re.compile(r"FINAL_VERDICT:\s*(CONFIRMED|NOT_CONFIRMED)", re.IGNORECASE)


def parse_clerk_verdict(response_text: str) -> bool:
    """Extracts a hard execute/no-execute boolean from the Clerk's
    free-text response — a new contract, unlike build_copilot_
    verification's existing usage elsewhere (which never
    programmatically parses Copilot's response, just embeds it as text
    for a LATER Claude call to weigh — a genuinely separate,
    independent feature, see ai/portfolio_suggest.py). Fails safe to
    False (never execute) on every ambiguous branch: the local-model
    failure sentinel, an empty response, no FINAL_VERDICT token
    anywhere, or a value other than CONFIRMED. If the model restates
    itself and multiple tokens appear, the LAST one wins — same "last
    occurrence wins" convention ai.portfolio_suggest._last_allocation_
    match already establishes — but only an unambiguous final CONFIRMED
    flips this to True; anything else (including NOT_CONFIRMED, or a
    stray unrecognized word) does not."""
    if not response_text:
        return False
    if response_text == OLLAMA_FAILED_MESSAGE:
        return False
    if not response_text.strip():
        return False
    matches = _FINAL_VERDICT_PATTERN.findall(response_text)
    if not matches:
        return False
    return matches[-1].upper() == "CONFIRMED"


def _fetch_clerk_news_block(symbol: str, description: str) -> str:
    """Real, live headlines for `symbol` — closes the real gap where the
    Clerk's verdict prompt invited "checking for major news" but neither
    local model (qwen3:8b/phi4-mini) can actually browse the web (see
    _run_clerk_prompt's own docstring).

    Real bug fixed 2026-09-20: this used to resolve `symbol` via data.
    underlying.resolve_yahoo_ticker (a PMEX-symbol keyword map matched
    against `description`) — confirmed live against this account's real
    symbol mix that 17 of 22 symbols (77%) silently resolved to NO
    ticker at all under it (every equity/forex/crypto symbol; only a
    couple of metals/commodities worked, by coincidence, since their MT5
    description happens to literally contain a matching keyword like
    "Gold"). Now delegates entirely to data.symbol_news.get_symbol_
    news_block, the same shared fetch/cache/vault-history Researcher and
    Mega Session use — this is also where "avoid double-calling the same
    symbol's news" is actually solved: Clerk, Researcher, and Mega
    Session now share ONE cache (config.SYMBOL_NEWS_CACHE_MINUTES),
    keyed by the real Yahoo ticker, regardless of which of the three
    resolves/asks for it first. `description` is now ALSO passed straight
    through to get_symbol_news_block's own "intelligent search" fallback
    (added 2026-09-20) — a symbol whose category matches no known ticker
    convention still gets real news via a Google search built from this
    same real MT5 description, instead of nothing.

    Returns "" (never fabricated) when no news resolves for this symbol
    (precisely or via the generic fallback) or the real fetch comes back
    empty on every source — the caller renders an honest "no recent
    headlines found" placeholder for that case, same convention as ai.
    researcher's own "(no category-specialty feed for ...)"."""
    category = get_symbol_category(symbol)
    items = symbol_news.get_symbol_news_block(
        symbol, category, limit=config.NEWS_HEADLINES_PER_ASSET, description=description
    )
    return "\n".join(f"- {item['title']}" for item in items)


# Shared by all three local-LLM prompts (verdict / invalidation / tactical) —
# intraday decision-tier upgrade, 2026-09-24: the technical context they are
# handed is now split into a DECISION TIER (M5 only — H1 joined the context
# tier the same day) and a CONTEXT TIER (D1/H4/H1), and the trigger/invalidation
# conditions Claude writes are now in M5 terms.
_TIMEFRAME_ROLES_NOTE = (
    "TIMEFRAME ROLES: the technical picture below is split into two tiers. The DECISION TIER "
    "(M5 only) is what you judge triggers, invalidation conditions, stops, targets and "
    "defense against — the senior analyst wrote them in M5 terms (levels, trigger and timing all "
    "come from the M5 read). The CONTEXT TIER (D1/H4/H1) is regime context only: use it to say "
    "whether the setup is WITH or AGAINST the larger trend, never to decide that an M5 condition "
    "has or has not happened. EXCEPTION for older setups: a trigger or invalidation condition that "
    "the senior analyst explicitly wrote on H1, H4 or D1 (setups carried over from before this "
    "timeframe split) is judged against that named timeframe's read in the context tier — do not call "
    "it unevaluable just because it is not an M5 condition. If an 'Upcoming High-impact events' line "
    "appears for this instrument, say so plainly in your reasoning."
)


def _build_verdict_prompt(
    setup: PendingSetup, technical_context: str, account_equity: float, elapsed_description: str, news_block: str
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
        "Health analysis), plus its real recent headlines. Use both to "
        "judge two things: "
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
        f"{_TIMEFRAME_ROLES_NOTE}\n\n{technical_context}\n\n"
        f"Recent headlines for {setup.symbol}:\n"
        f"{news_block if news_block else '(no recent headlines found)'}\n\n"
        f"Account equity for context: {account_equity:.2f}. This does "
        "not change your verdict on the trigger itself.\n\n"
        "Keep your reasoning short (at most four sentences). "
        "End your response with exactly one line, and nothing after "
        'it: either "FINAL_VERDICT: CONFIRMED" or "FINAL_VERDICT: '
        'NOT_CONFIRMED". Do not use this exact token anywhere else in '
        "your response."
    )


def _fetch_technical_context(
    symbol: str, market_prices: dict, account_equity: float, lean: bool = False
) -> tuple[FtmoAssetAnalysis, str] | None:
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
    object at all.

    Returns the structured FtmoAssetAnalysis alongside the formatted
    string (not just the string) since 2026-08-30 — the tactical-
    candidate phase's deterministic pre-screen (_compute_tactical_
    signals) needs the analysis's own h1_stats.atr, and this fetch
    already computes it internally; re-fetching separately would double
    the real MT5 round-trips for no reason.

    include_favorable_excursion=False (added 2026-09-12, real gap caught
    on a self-recheck): this formatted text becomes `technical_context`
    for the tactical-verdict prompts below, which never include
    ai/ftmo_suggest.py's own _INSTRUCTION_HEAD — the ONLY place the
    favorable-excursion figure's critical misread warning ("a positive
    number is NOT independent evidence the setup works") and its HOLDING
    HORIZON scale-mismatch caveat actually live. Without this, the local
    model here would see a bare, uncaveated positive-looking magnitude
    figure with a dangling "see the TP-sizing instruction above"
    reference that doesn't exist in its own prompt at all — exactly the
    "never re-derive real evidence from prose when it's one fact among
    many" risk TacticalSignals already exists to prevent for every other
    fact in this same string, not a new exception to that rule.

    include_market_status=False (same real-gap self-recheck pattern):
    the "market CLOSED (weekend)" tag exists to support _INSTRUCTION_
    HEAD's "don't propose a NEW trade on a closed market" guidance, which
    this prompt also never includes. Worse than the excursion case:
    Clerk never proposes new trades at all — every candidate here is
    already an existing position or a setup Claude already proposed
    (see run_clerk_execution_check's own candidate pools) — so an
    unexplained "market CLOSED" tag would be pure, unactionable noise
    with real hallucination risk for this weak local model, not omitting
    it a loss of anything it could actually act on."""
    asset = market_prices.get(symbol)
    if asset is None:
        return None
    analysis = (
        analyze_ftmo_asset_live(symbol, asset.bid, asset.ask, asset.description, lean=True)
        if lean else analyze_ftmo_asset_live(symbol, asset.bid, asset.ask, asset.description)
    )
    now = datetime.now(timezone.utc)
    try:
        calendar_events = economic_calendar.fetch_calendar_events(now)
    except Exception:
        calendar_events = []
    if config.CLERK_COMPACT_CONTEXT:
        return analysis, format_clerk_context(analysis, calendar_events=calendar_events, now_utc=now)
    return analysis, format_ftmo_asset_context(
        [analysis], account_equity=account_equity, include_favorable_excursion=False, include_market_status=False,
        calendar_events=calendar_events, now_utc=now,
    )


# Local-model roster, replacing the old Copilot-CLI-then-OpenRouter-
# backup chain entirely (direct user request 2026-08-30, after two real,
# already-observed failure points for a job that needs to be available
# every poll, not just most polls: Copilot's own monthly CLI quota was
# confirmed genuinely exhausted 2026-08-25, and the 10-model free
# OpenRouter pool was confirmed capable of going down as a WHOLE block
# at once, 2026-08-27 — 10/10 models unavailable simultaneously during
# a real mega-session run). Both models below run entirely on this
# machine via a local Ollama server (ai/ollama_client.py) — no API key,
# no shared rate limit, no per-token cost, and no dependency on any
# external service's own uptime.
#
# qwen3:8b is primary as of 2026-08-31 (see config.CLERK_PRIMARY_MODEL's
# own comment for the full rationale/history — gemma4:12b, primary until
# then, proved a poor fit for this machine's 4GB-VRAM GPU: only ~24-38%
# of it fits in VRAM depending on variant, versus qwen3:8b's ~39%, and
# gemma4's larger absolute size means far more of it lands on the slow
# CPU path regardless of the percentage split — a real, live-measured
# multi-candidate-poll timeout incident, not a hypothetical). Thinking
# DISABLED here too (see ai/ollama_client.py's own docstring for why —
# the same latency trap that made gemma4:12b's own default thinking mode
# unusable applies to any Ollama "thinking" model; also why qwen3:4b was
# tried and rejected as a candidate — its Ollama packaging ignores the
# think:false request entirely). phi4-mini is the backup (see config.
# CLERK_BACKUP_MODEL's own comment for the full rationale) — a direct
# user correction that a backup answering only "sometimes, and slowly"
# is worse than no backup: gemma4:12b was the most reliable model tested
# but too slow to be USABLE as a fallback in this job's own timeframe,
# so it was removed from this machine's Ollama install entirely.
_HTF_CONDITION_RE = re.compile(r"\b(d1|h4|h1|1h|4h|daily|weekly)\b", re.IGNORECASE)


def _condition_needs_htf(condition: str | None) -> bool:
    """True when the analyst wrote a trigger / invalidation condition on H1, H4 or D1 (older setups): judging it needs the higher-
    timeframe reads the compact Clerk context leaves out."""
    return bool(condition and _HTF_CONDITION_RE.search(condition))


def _context_for_condition(analysis, compact_context: str, condition: str | None, account_equity: float) -> str:
    """The compact context, or the FULL one when `condition` is written on a higher timeframe. Falls back to the compact text if
    the full formatting cannot be built."""
    if not (config.CLERK_COMPACT_CONTEXT and _condition_needs_htf(condition)):
        return compact_context
    try:
        now = datetime.now(timezone.utc)
        try:
            events = economic_calendar.fetch_calendar_events(now)
        except Exception:
            events = []
        return format_ftmo_asset_context(
            [analysis], account_equity=account_equity, include_favorable_excursion=False, include_market_status=False,
            calendar_events=events, now_utc=now,
        )
    except Exception:  # noqa: BLE001
        logger.warning("Could not build the full context for a higher-timeframe condition - using the compact one.", exc_info=True)
        return compact_context


_STRICT_M5_INVALIDATION_RE = re.compile(
    r"^\s*(?:the\s+)?m5\s+(?:candle\s+)?close[sd]?\s+(?:back\s+)?(above|below)\s+\$?(\d[\d,]*(?:\.\d+)?)\s*\.?\s*$", re.IGNORECASE
)


def deterministic_invalidation(entry_raw: dict, condition: str | None, analysis) -> tuple[bool, str] | None:
    """(confirmed, reasoning text) for an invalidation condition that is PURELY mechanical - "M5 closes above 64.49" for a sell,
    "M5 closes below 745.54" for a buy - decided from the last COMPLETED M5 close, with no model call; None for anything else
    (extra clauses, another timeframe, a level on the favourable side), which the model still judges. The model used to be
    asked "did the M5 close beyond X?" - a comparison Python does exactly, and a call that cost 15-250 seconds."""
    if not config.CLERK_DETERMINISTIC_INVALIDATION or not condition:
        return None
    match = _STRICT_M5_INVALIDATION_RE.match(condition)
    if match is None:
        return None
    direction, level = match.group(1).lower(), float(match.group(2).replace(",", ""))
    side = (entry_raw or {}).get("side")
    if not ((side == "sell" and direction == "above") or (side == "buy" and direction == "below")):
        return None
    bars = getattr(analysis, "m5_recent", None)
    if bars is None or len(bars) == 0 or "Close" not in bars.columns:
        return None
    last_close = float(bars["Close"].iloc[-1])
    confirmed = last_close > level if direction == "above" else last_close < level
    verdict = "CONFIRMED" if confirmed else "NOT_CONFIRMED"
    text = (
        _DETERMINISTIC_CIRCUIT_BREAKER_PREFIX
        + f"Invalidation \"{condition.strip()}\": the last completed M5 close is {last_close:.5g} against the level {level:.5g} - "
        + ("it HAS closed beyond it." if confirmed else "it has not closed beyond it.")
        + f"\n\nFINAL_VERDICT: {verdict}"
    )
    return confirmed, text


def _run_clerk_prompt(prompt: str, timeout: int) -> str:
    """Runs `prompt` through the Clerk's primary local model
    (config.CLERK_PRIMARY_MODEL, thinking disabled) and, only if that's
    genuinely unavailable — not a real HOLD/NOT_CONFIRMED answer, which
    is a legitimate verdict, not a failure — falls back to
    config.CLERK_BACKUP_MODEL on the same local Ollama server. The exact
    same prompt is reused unchanged for the backup: both models are
    asked for the same plain FINAL_VERDICT: line, a format any capable
    instruction-following model can honor.

    Known, accepted gap versus the old Copilot-CLI path: neither local
    model has live web access of its own, so the verdict/invalidation
    prompts can't invite an open-ended "go check the web for news"
    action. As of 2026-09-17, the pending-setup verdict prompt closes
    the practical gap that mattered by embedding real, pre-fetched
    headlines directly in the prompt text instead (see
    _fetch_clerk_news_block) — the model still doesn't browse anything
    itself, but it does see real, current news for that symbol, not a
    dangling, unfulfillable instruction to go find some. The
    invalidation/tactical prompts remain deliberately news-free — both
    are narrow, mechanical trip-wire checks on a single stated
    condition, and injecting news there would invite exactly the kind
    of "new independent opinion" override those prompts explicitly
    forbid.

    If both models fail, returns the primary's own failure message (the
    most informative single failure to surface), and the existing
    fail-safe verdict parsing still applies — a total outage still
    resolves to NOT_CONFIRMED/HOLD, never a silent guess."""
    raw = run_ollama(
        prompt, model=config.CLERK_PRIMARY_MODEL, timeout=timeout, keep_alive=config.CLERK_LLM_KEEP_ALIVE,
        max_tokens=config.CLERK_LLM_MAX_TOKENS,
    )
    if raw != OLLAMA_FAILED_MESSAGE:
        return raw
    backup_raw = run_ollama(prompt, model=config.CLERK_BACKUP_MODEL, timeout=timeout)
    if backup_raw != OLLAMA_FAILED_MESSAGE:
        return (
            f"[{config.CLERK_PRIMARY_MODEL} unavailable — backup model "
            f"{config.CLERK_BACKUP_MODEL} responded]\n\n{backup_raw}"
        )
    return raw


def _detect_ollama_outage(results: list[tuple]) -> str:
    """Real gap found live 2026-09-16, direct user request: unlike the
    AutoTrading-off gate (an immediate, visible "Execution blocked"
    status), a fully unreachable local Ollama server had no equivalent —
    every check still "completes" from run_clerk_execution_check's own
    point of view (parse_clerk_verdict/parse_tactical_verdict fail safe
    to NOT_CONFIRMED/HOLD on OLLAMA_FAILED_MESSAGE, same as any other
    ambiguous response), so nothing otherwise distinguishes "the model
    genuinely judged this NOT_CONFIRMED" from "Ollama itself never
    answered."

    `results` is this poll's own raw ThreadPoolExecutor results list —
    each a (PendingSetup-or-symbol, verdict, raw_text) tuple regardless
    of which of the three check kinds produced it (see the merge loop's
    own docstring for why raw_text is always at index 2). Excludes this
    poll's own deterministic circuit-breaker verdicts (hard-exit, trend-
    flip — see _DETERMINISTIC_CIRCUIT_BREAKER_PREFIX) from the count
    entirely: those never call Ollama at all, so a poll made up ENTIRELY
    of them must never be misread as "every real attempt failed."

    Returns a real, human-readable warning when this poll made at least
    one genuine LLM attempt and EVERY one of them failed with Ollama's
    own total-failure sentinel — "" (nothing to report) when there were
    no real attempts at all, or at least one succeeded (a partial
    failure is more likely an unlucky single call than the server being
    down, and _run_clerk_prompt's own primary/backup cascade already
    absorbs a single flaky model)."""
    llm_attempts = [r[2] for r in results if not r[2].startswith(_DETERMINISTIC_CIRCUIT_BREAKER_PREFIX)]
    ollama_failures = sum(1 for raw in llm_attempts if raw == OLLAMA_FAILED_MESSAGE)
    if not llm_attempts or ollama_failures != len(llm_attempts):
        return ""
    return (
        f"Ollama's local model server appears unreachable this poll "
        f"({ollama_failures}/{len(llm_attempts)} check(s) got no real response) — "
        "pending-setup/invalidation/tactical checks are all falling back to their "
        "fail-safe verdicts (NOT_CONFIRMED/HOLD) until it's back. Check that the "
        "Ollama server is running (localhost:11434)."
    )


def _run_clerk_verdict(
    setup: PendingSetup, technical_context: str, account_equity: float, elapsed_description: str, description: str
) -> tuple[PendingSetup, bool, str]:
    """The LLM-bound half — pure local HTTP call (with a same-server
    backup model, see _run_clerk_prompt), no MT5 involvement, safe to
    run from inside a thread pool (see run_clerk_execution_check's own
    parallel round). Returns (setup, confirmed, raw_text) — the raw
    text always travels back for the audit-trail log/UI, even on a
    NOT_CONFIRMED or failed verdict.

    `description` (the symbol's real Market Watch description, e.g.
    "Euro vs United States Dollar") is only used to resolve a Yahoo
    ticker for the real news fetch below — the news call itself is
    plain HTTP (yfinance), the same thread-safety class as the LLM call
    already made from here, so it's safe to do inline rather than
    forcing it into the earlier, MT5-bound sequential phase."""
    news_block = _fetch_clerk_news_block(setup.symbol, description)
    prompt = _build_verdict_prompt(setup, technical_context, account_equity, elapsed_description, news_block)
    raw = _run_clerk_prompt(prompt, timeout=config.CLERK_LLM_TIMEOUT_SECONDS)
    return setup, parse_clerk_verdict(raw), raw


def _describe_elapsed(generated_utc: str | None, now_utc: datetime) -> str:
    """Human-readable elapsed time since the mega session that produced
    this Pending Setup — fed into the verdict prompt so the Clerk can
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
    lets the Clerk mechanically re-check ONLY that stated condition every
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
        f"{_TIMEFRAME_ROLES_NOTE}\n\n{technical_context}\n\n"
        f"Account equity for context: {account_equity:.2f}. This does not "
        "change your verdict on the condition itself.\n\n"
        "Keep your reasoning SHORT - at most two sentences, naming the level and the latest M5 price or close you compared - "
        "then the final line. "
        "End your response with exactly one line, and nothing after it: "
        'either "FINAL_VERDICT: CONFIRMED" (the invalidation condition has '
        'genuinely occurred — this position should be exited/cancelled '
        'now) or "FINAL_VERDICT: NOT_CONFIRMED" (the condition has not '
        "occurred — keep holding/watching this position unchanged). Do "
        "not use this exact token anywhere else in your response."
    )


def _run_clerk_invalidation_check(
    symbol: str,
    entry: dict,
    invalidation_condition: str,
    technical_context: str,
    account_equity: float,
    elapsed_description: str,
    position_state: str,
) -> tuple[str, bool, str]:
    """The LLM-bound half of an invalidation check — pure local HTTP
    call (with a same-server backup model, see
    _run_clerk_prompt), no MT5 involvement, safe to run from inside a
    thread pool (see run_clerk_execution_check's own parallel round).
    Returns (symbol, confirmed, raw_text) — deliberately a bare
    symbol string in position 0, not a PendingSetup, so the merge loop
    can tell the two verdict kinds apart via isinstance()."""
    prompt = _build_invalidation_prompt(
        symbol, entry, invalidation_condition, technical_context,
        account_equity, elapsed_description, position_state,
    )
    raw = _run_clerk_prompt(prompt, timeout=config.CLERK_LLM_TIMEOUT_SECONDS)
    return symbol, parse_clerk_verdict(raw), raw


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
    # True for a deterministic circuit-breaker verdict _run_clerk_
    # tactical_check constructs directly without an LLM call — the
    # O'Neil hard stop-loss ceiling (TacticalSignals.hard_exit_required)
    # or the sustained-trend-flip-against-the-position escalation
    # (TacticalSignals.trend_flip_against_count, added 2026-09-16, name
    # kept as "hard_exit" rather than renamed since both are the exact
    # same kind of rule: a genuine "no exceptions" circuit-breaker, not a
    # discretionary LLM judgment call, so both must apply even while
    # config.read_tactical_defense_enabled() is OFF (the default shadow
    # mode for every other DEFEND/EXIT verdict). See the merge loop in
    # run_clerk_execution_check for the actual bypass.
    hard_exit: bool = False


# Shared prefix for every synthetic, no-model-call verdict this module
# constructs directly (the O'Neil hard-exit ceiling, the trend-flip
# partial/exit escalation) — used both to label those verdicts' own raw
# text and to EXCLUDE them from the Ollama-health check below (a
# deliberately-skipped model call is not a failed one).
_DETERMINISTIC_CIRCUIT_BREAKER_PREFIX = "[deterministic circuit-breaker — no model call made] "

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
    """Extracts a HOLD/DEFEND/EXIT tactical verdict from the Clerk's (or
    its backup model's) free-text response — the DEFEND/EXIT counterpart
    to parse_clerk_verdict's own hard boolean. Fails safe to HOLD (never
    DEFEND/EXIT) on every ambiguous branch, mirroring parse_clerk_
    verdict's own discipline, rather than ever guessing at a real-money
    action: the local-model failure sentinel, an empty response, no
    FINAL_VERDICT token, a DEFEND/EXIT missing its RULE/NUMBERS citation
    (a real book rule and real numbers are mandatory, direct user
    request: never a "random guess, malfunctioning, or misunderstanding"),
    a DEFEND with BOTH NEW_STOP_LOSS and PARTIAL_CLOSE_FRACTION absent
    (nothing to actually defend with), or a PARTIAL_CLOSE_FRACTION
    outside config's configured [min, max] bounds — REJECTED outright,
    never silently clamped into range. If multiple FINAL_VERDICT tokens
    appear, the LAST one wins, same convention as parse_clerk_verdict."""
    fail_safe = TacticalVerdict(tier="hold", raw_text=response_text or "")
    if not response_text or not response_text.strip():
        return fail_safe
    if response_text == OLLAMA_FAILED_MESSAGE:
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
        config.CLERK_TACTICAL_MIN_PARTIAL_CLOSE_FRACTION
        <= partial_close_fraction
        <= config.CLERK_TACTICAL_MAX_PARTIAL_CLOSE_FRACTION
    ):
        return fail_safe
    return TacticalVerdict(
        tier="defend", new_stop_loss=new_stop_loss, partial_close_fraction=partial_close_fraction,
        rule_citation=rule_citation, numbers_citation=numbers_citation, raw_text=response_text,
    )


@dataclass
class TacticalSignals:
    """Deterministic, Python-computed numbers for a single tactical
    candidate — handed to the Clerk's LLM alongside the free-text
    technical context so its job shifts from "read prose and derive
    arithmetic" to "weigh already-computed candidates and decide". Added
    2026-08-30 after a real, observed failure: the backup model once
    proposed LOOSENING a stop while calling it "tightening" the position.
    Every field that can't be computed from real data degrades to None
    (or, for the two bool "due" flags, False) rather than a guess — same
    convention as analysis.technical.TechnicalStats."""

    favorable_move_pct: float  # positive = in this position's favor
    h1_atr: float | None
    velocity_tier: str | None  # "fast" / "slow" / None — see analysis.technical.classify_velocity_tier
    atr_stop_multiple_used: float  # the ACTUAL multiple applied below — velocity-tiered, not always config.CLERK_TACTICAL_ATR_STOP_MULTIPLE; always populated (defaults to the slow-tier constant when velocity can't be classified)
    atr_stop_candidate: float | None
    atr_stop_is_tighter_than_current: bool | None
    profit_lock_due: bool
    target_captured_pct: float | None
    days_held: float | None
    partial_profit_due: bool
    hard_exit_required: bool
    h1_rsi: float | None  # added 2026-09-09 — see h1_rsi_tier's own comment
    h1_rsi_tier: str | None  # "overbought" / "oversold" / "neutral" / None — a deterministic Python classification (see analysis.technical.classify_rsi_tier), never left for the model's own free-text reasoning to derive. Real, observed failure this fixes: qwen3:8b repeatedly mislabeled a deeply oversold H4 RSI (11, then 15, then 15 again) as "overbought" across three consecutive live polls on a real USDCAD position.
    h4_rsi: float | None
    h4_rsi_tier: str | None  # same fix, H4 timeframe
    # --- added 2026-09-09: real structural S/R, extracted out of the
    # free-text technical_context prose the same way RSI/velocity already
    # were above. The tactical clerk's own technical_context string
    # ALREADY contains this same data (format_ftmo_asset_context calls
    # ai.ftmo_suggest.format_chart_structure internally) — but buried
    # among ~15 other lines per timeframe (daily/feasibility/backtest,
    # setup signals, MTF confluence, trading cost), which this project's
    # own history shows a weak local model (or even Claude, see build_
    # trend_radar's own docstring for the real 2026-09-05 EEURUSD/GBPUSD
    # incident) can correctly compute yet still fail to genuinely engage
    # with when it's one fact among many. Pulling out just the SINGLE
    # nearest resistance/support level explicitly mirrors the RSI-tier
    # fix's own reasoning: never re-derive real evidence from prose when
    # it's already been computed once, deterministically, in Python.
    nearest_resistance: SRLevel | None = None
    nearest_support: SRLevel | None = None
    # --- added 2026-09-12: real historical backtest evidence for this
    # position's OWN side, extracted the same way RSI-tier/velocity-tier/
    # nearest-S-R already were above — see analysis.backtest.classify_
    # backtest_favorability's own docstring for the full SUPPORTED/
    # CONTRADICTED/MIXED rule. Real gap this closes: Clerk's own
    # technical_context deliberately excludes the raw favorable-
    # excursion figure (see _fetch_technical_context's own
    # include_favorable_excursion=False comment) because its safety
    # caveats live only in ai/ftmo_suggest.py's _INSTRUCTION_HEAD, which
    # Clerk never sees — these two fields are the safe replacement: a
    # deterministic verdict instead of raw prose, with the excursion
    # figure mechanically gated to never appear unless the underlying
    # win-rate evidence already supports the setup.
    backtest_favorability: str | None = None  # "supported" / "contradicted" / "mixed" / None
    favorable_excursion_median_r: float | None = None  # only ever non-None when backtest_favorability == "supported"
    # --- added 2026-09-16: real incident this closes (NVDA, held long for
    # most of a trading day while H1+H4 explicitly, repeatedly read
    # downtrend — clerk_execution_log.txt:22101-23099, 2026-09-10/11).
    # A PERSISTED counter (unlike every other field above, which is pure/
    # stateless per-poll) of how many CONSECUTIVE polls this position's
    # own aligned_h1_h4_trend_direction has contradicted its side —
    # incremented here from prior_tactical's own last-saved value, reset
    # to 0 the instant the trend no longer contradicts (re-aligns, or
    # reads ambiguous/flat). _run_clerk_tactical_check uses this to force
    # de-risking (see config.CLERK_TREND_FLIP_PARTIAL_AFTER_POLLS/_EXIT_
    # AFTER_POLLS) regardless of what the LLM tactical verdict says —
    # the whole point being that a self-acknowledged, sustained
    # contradiction shouldn't be allowed to just keep getting HELD.
    trend_flip_against_count: int = 0
    # --- added 2026-09-18: direct user request for occasional small,
    # real, realized-profit events (a new, aspirant trader's own stated
    # need), separate from and much earlier than partial_profit_due's own
    # 50%-of-target milestone above, which is only ever advisory and has
    # never reliably converted into a real banked gain. A PERSISTED,
    # ONE-TIME sticky flag (unlike trend_flip_against_count, which
    # decays) — once True, stays True for the life of this settled
    # position; see _run_clerk_tactical_check's own deterministic
    # circuit breaker for it and config.CLERK_QUICK_PROFIT_LOCK_COST_
    # MULTIPLE's own comment for why 3x real round-trip cost, not a
    # guessed flat percentage.
    quick_profit_lock_due: bool = False
    # --- added 2026-09-21, Phase 5c of the charting-expert technical-
    # analysis upgrade: the most recent real BOS/CHOCH structure break
    # and liquidity sweep on H1 (analysis.chart_structure.detect_
    # structure_breaks/detect_liquidity_sweeps, Phase 2/3 of that same
    # upgrade) — pulled out of technical_context's own buried prose the
    # same way nearest_resistance/nearest_support already were above, so
    # a genuine "structure just changed against you" fact can't get lost
    # among the ~15 other lines per timeframe. Distinct from, not a
    # replacement for, trend_flip_against_count: a CHOCH is a one-shot
    # structural EVENT (it either happened recently or it didn't), while
    # trend_flip_against_count tracks SUSTAINED contradiction building up
    # poll over poll — genuinely complementary signals, not redundant
    # ones (see _build_tactical_prompt's own comment for how both are
    # surfaced together).
    nearest_structure_break: StructureBreak | None = None
    nearest_liquidity_sweep: LiquiditySweepEvent | None = None
    # --- added 2026-09-24, intraday decision-tier upgrade; re-based on M5 ONLY the same day (H1/H4/D1 are
    # context). The ATR stop candidate is sized off the M5 ATR (atr_timeframe says which ATR was
    # actually used — "H1" only when no M5 read exists), velocity is classified from the M5 ATR%,
    # nearest_resistance/nearest_support and the structure break/sweep come from the M5 structure
    # (structure_timeframe says which; "H1" only when no M5 read exists), the trend-flip circuit
    # breaker reads the M5 trend+regime (trend_flip_basis says which; "H1+H4" only when no M5 read
    # exists), and defense gets the M5 RSI and a structure-trail stop candidate (just beyond the
    # nearest M5 support/resistance). H1/H4 RSI stay in the prompt as context lines.
    m5_atr: float | None = None
    atr_timeframe: str = "H1"
    structure_timeframe: str = "H1"
    trend_flip_basis: str = "H1+H4"
    velocity_timeframe: str = "H1"  # "M5" when velocity came from the median M5 ATR%, else the H1 fallback
    m5_rsi: float | None = None
    m5_rsi_tier: str | None = None
    nearest_m5_resistance: SRLevel | None = None
    nearest_m5_support: SRLevel | None = None
    structure_trail_candidate: float | None = None
    structure_trail_is_tighter: bool | None = None
    event_note: str | None = None
    # --- added 2026-09-25 (position-hunting review): deterministic profit trail. MEASURED on 20 real symbols x
    # 6000 M5 bars (38,960 paired random trades, stop 2 ATR / target 2R / hold 96 bars, spread charged): a stop
    # trailed 1.5 M5 ATR behind the price once the trade is +1R improved net expectancy from -0.174R to -0.148R
    # (17/20 symbols), better than breakeven-at-1R (-0.163R, 16/20), partial-50%+breakeven (-0.155R, 16/20) and
    # breakeven-at-1.5R (-0.172R, 13/20), while lifting the win share from 34% to 48%. `profit_trail_r0` is the
    # trade's ORIGINAL risk distance (persisted in the tactical state, since a later tightened stop would shrink
    # it), `profit_trail_r` the current progress in those R, `profit_trail_stop` the stop to ratchet to (None
    # unless the trade is past config.CLERK_PROFIT_TRAIL_START_R AND the new stop is tighter than the current one
    # by at least config.CLERK_PROFIT_TRAIL_MIN_STEP_ATR M5 ATR).
    profit_trail_r0: float | None = None
    profit_trail_r: float | None = None
    profit_trail_stop: float | None = None


def _compute_tactical_signals(
    position: Position,
    entry: AllocationEntry,
    h1_stats: TechnicalStats | None,
    h4_stats: TechnicalStats | None = None,
    h1_structure: ChartStructureSnapshot | None = None,
    base: AssetAnalysis | None = None,
    prior_tactical: dict | None = None,
    trade_cost: TradeCost | None = None,
    m5_stats: TechnicalStats | None = None,
    m5_structure: ChartStructureSnapshot | None = None,
    event_note: str | None = None,
    intraday_backtests: IntradayBacktests | None = None,
    m5_atr_pct_median: float | None = None,
) -> TacticalSignals:
    """Pure, no-I/O — every input is already-fetched data the caller
    holds (never entry.price: that's the mega-session's OWN suggestion
    price and can be stale by the time this runs, exactly the discipline
    _validate_and_apply_tactical_verdict already follows for the same
    reason; position.price_open/price_current are the real, live MT5
    numbers).

    The ATR-multiple used for atr_stop_candidate is velocity-tiered
    (added 2026-09-09 — see analysis.technical.classify_velocity_tier's
    own docstring for the real XAUUSD incident and data comparison this
    is built from): an instrument currently classified "fast" by its own
    live H1 atr_pct gets config.CLERK_TACTICAL_ATR_STOP_MULTIPLE_FAST
    (wider — its own ordinary noise covers more ground faster), anything
    else (including "slow" or an unclassifiable None, i.e. no h1_atr_pct
    reading available) keeps the original config.
    CLERK_TACTICAL_ATR_STOP_MULTIPLE — the same, unchanged default this
    already used before velocity-tiering existed, so a caller with no
    atr_pct reading (e.g. a test fixture, or a genuinely thin history)
    degrades to exactly the prior behavior rather than guessing a tier.

    `h4_stats` (added 2026-09-09, optional/defaults to None so any
    existing caller keeps working unchanged) feeds h4_rsi/h4_rsi_tier —
    see TacticalSignals' own field comments for the real, repeated
    mislabeling incident (qwen3:8b calling a deeply oversold RSI
    "overbought", three polls running) this was built to stop at the
    source rather than merely hope a future prompt tweak discourages.

    `h1_structure` (added 2026-09-09, same optional/backward-compatible
    default) feeds nearest_resistance/nearest_support — see TacticalSignals'
    own field comment for why this is pulled out of prose rather than
    left for the model to find on its own.

    `base` (added 2026-09-12, same optional/backward-compatible default)
    feeds backtest_favorability/favorable_excursion_median_r via
    analysis.backtest.classify_backtest_favorability — see that
    function's own docstring and TacticalSignals' own field comment for
    the full rule and the real gap this closes.

    `prior_tactical` (added 2026-09-16, same optional/backward-compatible
    default) feeds TacticalSignals.trend_flip_against_count — see that
    field's own comment. The ONLY stateful field this function computes;
    every other field above is pure/stateless given just this poll's own
    data."""
    favorable_move_pct = -position.adverse_move_pct

    h1_rsi = h1_stats.rsi if h1_stats is not None else None
    h1_rsi_tier = classify_rsi_tier(h1_rsi)
    h4_rsi = h4_stats.rsi if h4_stats is not None else None
    h4_rsi_tier = classify_rsi_tier(h4_rsi)

    h1_atr = h1_stats.atr if h1_stats is not None else None
    h1_atr_pct = h1_stats.atr_pct if h1_stats is not None else None
    # Decision-tier basis: the M5 ATR with the M5 multiples (2.0x, 3.0x fast — the same stop distance the
    # earlier 1.5x/2.5x M15 ATR gave); the H1-era 1.5x/2.5x multiples apply only when no M5 read exists.
    m5_atr = m5_stats.atr if m5_stats is not None else None
    use_m5_atr = m5_atr is not None and m5_atr > 0
    # Velocity is a property of the instrument, so it is read from its MEDIAN M5 ATR% (a stable ~2-day
    # baseline), never the instantaneous value that dips in quiet hours (audited: gold/copper/BTC would
    # flip fast<->slow on 18-28% of bars); H1 ATR% only when no M5 baseline exists.
    if m5_atr_pct_median is not None:
        velocity_tier, velocity_timeframe = classify_m5_velocity_tier(m5_atr_pct_median), "M5"
    else:
        velocity_tier, velocity_timeframe = classify_velocity_tier(h1_atr_pct), "H1"
    if use_m5_atr:
        atr_stop_multiple_used = (
            config.CLERK_TACTICAL_M5_ATR_STOP_MULTIPLE_FAST
            if velocity_tier == "fast"
            else config.CLERK_TACTICAL_M5_ATR_STOP_MULTIPLE
        )
    else:
        atr_stop_multiple_used = (
            config.CLERK_TACTICAL_ATR_STOP_MULTIPLE_FAST
            if velocity_tier == "fast"
            else config.CLERK_TACTICAL_ATR_STOP_MULTIPLE
        )
    atr_stop_candidate: float | None = None
    atr_stop_is_tighter_than_current: bool | None = None
    stop_atr = m5_atr if use_m5_atr else h1_atr
    atr_timeframe = "M5" if use_m5_atr else "H1"
    if stop_atr is not None:
        distance = atr_stop_multiple_used * stop_atr
        if position.side == "buy":
            atr_stop_candidate = position.price_current - distance
            atr_stop_is_tighter_than_current = (
                position.sl is None or atr_stop_candidate > position.sl
            )
        else:
            atr_stop_candidate = position.price_current + distance
            atr_stop_is_tighter_than_current = (
                position.sl is None or atr_stop_candidate < position.sl
            )

    target_captured_pct: float | None = None
    if position.tp is not None and position.tp != position.price_open:
        target_distance = position.tp - position.price_open
        moved = position.price_current - position.price_open
        target_captured_pct = moved / target_distance * 100

    days_held: float | None = None
    try:
        opened_at = position.opened_at
        now = datetime.now() if opened_at.tzinfo is None else datetime.now(timezone.utc)
        days_held = (now - opened_at).total_seconds() / 86400.0
    except (TypeError, OSError):
        days_held = None

    partial_profit_due = (
        target_captured_pct is not None
        and days_held is not None
        and target_captured_pct >= config.CLERK_TACTICAL_PARTIAL_PROFIT_TARGET_PCT
        and days_held <= config.CLERK_TACTICAL_PARTIAL_PROFIT_MAX_DAYS
    )

    # favorable_move_pct and trade_cost.spread_pct_of_price are both
    # already expressed as a % of price_open (see Position.adverse_move_
    # pct's own docstring and TradeCost's own field comment) — directly
    # comparable with no unit conversion. A one-time, sticky flag (see
    # TacticalSignals.quick_profit_lock_due's own comment) — once
    # already_locked, never re-fires for this same settled position.
    already_locked = (prior_tactical or {}).get("quick_profit_lock_done", False)
    quick_profit_lock_due = (
        not already_locked
        and trade_cost is not None
        and trade_cost.spread_pct_of_price > 0
        and favorable_move_pct >= config.CLERK_QUICK_PROFIT_LOCK_COST_MULTIPLE * trade_cost.spread_pct_of_price
    )

    # Nearest level on each side — resistance_levels are all ABOVE price
    # (distance_pct > 0, smallest = nearest), support_levels all BELOW
    # (distance_pct < 0, largest/least-negative = nearest) — see
    # analysis.chart_structure.compute_sr_levels' own sort-by-touches
    # convention; this is a SEPARATE "nearest" read for tactical
    # relevance, not a re-ranking of that function's own "strongest
    # first" ordering.
    # The structure read is the M5 one (decision timeframe); H1 only when the caller has no M5 read.
    structure_source = m5_structure if m5_structure is not None else h1_structure
    structure_timeframe = "M5" if m5_structure is not None else "H1"
    nearest_resistance: SRLevel | None = None
    nearest_support: SRLevel | None = None
    if structure_source is not None and structure_source.sr_levels is not None:
        if structure_source.sr_levels.resistance_levels:
            nearest_resistance = min(structure_source.sr_levels.resistance_levels, key=lambda lvl: lvl.distance_pct)
        if structure_source.sr_levels.support_levels:
            nearest_support = max(structure_source.sr_levels.support_levels, key=lambda lvl: lvl.distance_pct)

    # Most recent (smallest bars_ago) real structure break/liquidity
    # sweep on the same structure read — see TacticalSignals.nearest_structure_break's own
    # field comment for why this is pulled out of prose the same way
    # nearest_resistance/nearest_support already were.
    nearest_structure_break: StructureBreak | None = None
    nearest_liquidity_sweep: LiquiditySweepEvent | None = None
    if structure_source is not None:
        if structure_source.structure_breaks:
            nearest_structure_break = min(structure_source.structure_breaks, key=lambda b: b.bars_ago)
        if structure_source.liquidity_sweeps:
            nearest_liquidity_sweep = min(structure_source.liquidity_sweeps, key=lambda s: s.bars_ago)

    nearest_m5_resistance: SRLevel | None = None
    nearest_m5_support: SRLevel | None = None
    structure_trail_candidate: float | None = None
    structure_trail_is_tighter: bool | None = None
    if m5_structure is not None and m5_structure.sr_levels is not None:
        if m5_structure.sr_levels.resistance_levels:
            nearest_m5_resistance = min(m5_structure.sr_levels.resistance_levels, key=lambda lvl: lvl.distance_pct)
        if m5_structure.sr_levels.support_levels:
            nearest_m5_support = max(m5_structure.sr_levels.support_levels, key=lambda lvl: lvl.distance_pct)
    if use_m5_atr:
        buffer = 0.4 * m5_atr  # ~0.25 of the old M15 ATR (median M15/M5 ATR ratio 1.42 -> 0.35; 0.4 rounds up)
        if position.side == "buy" and nearest_m5_support is not None:
            level = nearest_m5_support.low if nearest_m5_support.low is not None else nearest_m5_support.price
            candidate = level - buffer
            if candidate < position.price_current:
                structure_trail_candidate = candidate
                structure_trail_is_tighter = position.sl is None or candidate > position.sl
        elif position.side == "sell" and nearest_m5_resistance is not None:
            level = nearest_m5_resistance.high if nearest_m5_resistance.high is not None else nearest_m5_resistance.price
            candidate = level + buffer
            if candidate > position.price_current:
                structure_trail_candidate = candidate
                structure_trail_is_tighter = position.sl is None or candidate < position.sl
    m5_rsi = m5_stats.rsi if m5_stats is not None else None

    profit_trail_r0: float | None = None
    profit_trail_r: float | None = None
    profit_trail_stop: float | None = None
    if config.CLERK_PROFIT_TRAIL_ENABLED and use_m5_atr:
        r0 = trail_original_risk(
            position.price_open, entry.stop_loss if entry is not None else None, position.sl,
            (prior_tactical or {}).get("profit_trail_r0"),
        )
        decision = trail_decision(
            position.side, position.price_open, position.price_current, position.sl, r0, m5_atr,
            trade_cost.min_stop_distance_pct if trade_cost is not None else 0.0,
        )
        profit_trail_r0, profit_trail_r, profit_trail_stop = decision.r0, decision.progress_r, decision.new_stop

    backtest_favorability: str | None = None
    favorable_excursion_median_r: float | None = None
    if base is not None:
        # M5 is the decision tier, so the M5 setups are judged against their random-entry baseline
        # (a raw avg-R sign near the null result is not evidence). The D1 sign read stays only as the
        # fallback for a symbol with no M5 backtest at all.
        intraday = intraday_backtests if intraday_backtests is not None else IntradayBacktests()
        m5_verdicts = {
            key: verdict.verdict for key, verdict in intraday_edge_verdicts(intraday, position.side).items()
        }
        if m5_verdicts:
            backtest_favorability, favorable_excursion_median_r = classify_backtest_favorability(
                position.side,
                intraday.rsi_overbought_backtest,
                intraday.rsi_oversold_backtest,
                intraday.support_resistance_backtest,
                None,
                None,
                edge_verdicts=m5_verdicts,
            )
        else:
            backtest_favorability, favorable_excursion_median_r = classify_backtest_favorability(
                position.side,
                base.rsi_overbought_backtest,
                base.rsi_oversold_backtest,
                base.support_resistance_backtest,
                base.double_bottom_backtest,
                base.double_top_backtest,
            )

    # Risk management reads the M5 trend+regime (aligned_m5_trend_direction); the H1+H4 alignment is
    # only the fallback for a caller with no M5 read.
    if m5_stats is not None and m5_stats.trend is not None:
        aligned_trend = aligned_m5_trend_direction(m5_stats)
        trend_flip_basis = "M5"
    else:
        aligned_trend = aligned_h1_h4_trend_direction(
            h4_stats.trend if h4_stats is not None else None,
            h1_stats.trend if h1_stats is not None else None,
        )
        trend_flip_basis = "H1+H4"
    trend_contradicts_position = (
        (position.side == "buy" and aligned_trend == "down")
        or (position.side == "sell" and aligned_trend == "up")
    )
    prior_trend_flip_count = (prior_tactical or {}).get("trend_flip_against_count", 0)
    trend_flip_against_count = prior_trend_flip_count + 1 if trend_contradicts_position else 0

    return TacticalSignals(
        favorable_move_pct=favorable_move_pct,
        h1_atr=h1_atr,
        velocity_tier=velocity_tier,
        atr_stop_multiple_used=atr_stop_multiple_used,
        atr_stop_candidate=atr_stop_candidate,
        atr_stop_is_tighter_than_current=atr_stop_is_tighter_than_current,
        profit_lock_due=favorable_move_pct >= config.CLERK_TACTICAL_PROFIT_LOCK_PCT,
        target_captured_pct=target_captured_pct,
        days_held=days_held,
        partial_profit_due=partial_profit_due,
        hard_exit_required=position.adverse_move_pct >= config.CLERK_TACTICAL_HARD_EXIT_PCT,
        h1_rsi=h1_rsi,
        h1_rsi_tier=h1_rsi_tier,
        h4_rsi=h4_rsi,
        h4_rsi_tier=h4_rsi_tier,
        nearest_resistance=nearest_resistance,
        nearest_support=nearest_support,
        backtest_favorability=backtest_favorability,
        favorable_excursion_median_r=favorable_excursion_median_r,
        trend_flip_against_count=trend_flip_against_count,
        quick_profit_lock_due=quick_profit_lock_due,
        nearest_structure_break=nearest_structure_break,
        nearest_liquidity_sweep=nearest_liquidity_sweep,
        m5_atr=m5_atr,
        atr_timeframe=atr_timeframe,
        structure_timeframe=structure_timeframe,
        trend_flip_basis=trend_flip_basis,
        velocity_timeframe=velocity_timeframe,
        m5_rsi=m5_rsi,
        m5_rsi_tier=classify_rsi_tier(m5_rsi),
        nearest_m5_resistance=nearest_m5_resistance,
        nearest_m5_support=nearest_m5_support,
        structure_trail_candidate=structure_trail_candidate,
        structure_trail_is_tighter=structure_trail_is_tighter,
        event_note=event_note,
        profit_trail_r0=profit_trail_r0,
        profit_trail_r=profit_trail_r,
        profit_trail_stop=profit_trail_stop,
    )


def _build_tactical_prompt(
    symbol: str,
    entry: AllocationEntry,
    position: Position,
    technical_context: str,
    account_equity: float,
    prior_tactical: dict | None,
    signals: TacticalSignals,
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
    if prior_tactical and (prior_tactical.get("defend_count") or prior_tactical.get("last_action_utc")):  # a Sentinel-only record is not a prior DEFEND
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

    atr_stop_line = (
        f"{signals.atr_stop_candidate:.5f} "
        f"({'tighter' if signals.atr_stop_is_tighter_than_current else 'NOT tighter'} "
        "than the current stop above)"
        if signals.atr_stop_candidate is not None
        else "not available (insufficient data for a live ATR reading)"
    )
    # Added 2026-09-10 — real, repeated incident: the local tactical model
    # proposed a SELL's NEW_STOP_LOSS below the live market price six
    # times across one real session (confusing "tighter" with "a lower
    # number" without checking which side of price that number needs to
    # sit on for THIS position's own direction) — the same failure class
    # RSI-tier/velocity-tier mislabeling already needed fixing at the
    # source for: a geometric fact, not a judgment call, left to free-
    # text derivation instead of being stated explicitly. Every one of
    # those six proposals was correctly rejected by _validate_and_apply_
    # tactical_verdict's own guardrails, but the whole DEFEND (including
    # the model's own correct underlying judgment that this position
    # needed defending) was discarded each time. Stating the actual valid
    # NUMERIC WINDOW up front, the same "already-resolved fact, don't
    # re-derive it" treatment already given to RSI tiers, aims to reduce
    # how often this happens in the first place; _validate_and_apply_
    # tactical_verdict's own new ATR-candidate fallback (2026-09-10)
    # covers the case where it still doesn't.
    if position.sl is not None:
        if position.side == "buy":
            valid_stop_range_line = (
                f"strictly BELOW the current live price ({position.price_current:.5f}) "
                f"AND ABOVE the current stop ({position.sl:.5f}) — i.e. somewhere in the "
                f"open interval ({position.sl:.5f}, {position.price_current:.5f})"
            )
        else:
            valid_stop_range_line = (
                f"strictly ABOVE the current live price ({position.price_current:.5f}) "
                f"AND BELOW the current stop ({position.sl:.5f}) — i.e. somewhere in the "
                f"open interval ({position.price_current:.5f}, {position.sl:.5f})"
            )
    elif position.side == "buy":
        valid_stop_range_line = f"strictly BELOW the current live price ({position.price_current:.5f})"
    else:
        valid_stop_range_line = f"strictly ABOVE the current live price ({position.price_current:.5f})"
    target_captured_line = (
        f"{signals.target_captured_pct:.1f}%"
        if signals.target_captured_pct is not None
        else "not available (no take-profit set, or it equals the entry price)"
    )
    days_held_line = f"{signals.days_held:.2f}" if signals.days_held is not None else "unknown"
    velocity_tf = signals.velocity_timeframe
    standard_multiple = (
        config.CLERK_TACTICAL_M5_ATR_STOP_MULTIPLE
        if signals.atr_timeframe == "M5"
        else config.CLERK_TACTICAL_ATR_STOP_MULTIPLE
    )
    if signals.velocity_tier is None:
        velocity_line = (
            "no live ATR% baseline is available to classify this instrument's velocity "
            "— the standard ATR multiple applies"
        )
    elif signals.velocity_tier == "fast":
        velocity_line = (
            f"this instrument's own typical {velocity_tf} ATR classifies it as a FAST-tier mover "
            "— a normal pullback here covers ground faster than a slow FX cross's own noise would, "
            f"which is why the ATR multiple below is wider than the usual "
            f"{standard_multiple}x default"
        )
    else:
        velocity_line = (
            f"this instrument's own typical {velocity_tf} ATR classifies it as a "
            f"{signals.velocity_tier.upper()}-tier mover — the standard ATR multiple applies"
        )
    def _rsi_line(label: str, rsi: float | None, tier: str | None) -> str:
        if rsi is None or tier is None:
            return f"{label} RSI: not available"
        return f"{label} RSI: {rsi:.0f} — ALREADY CLASSIFIED AS **{tier.upper()}** (>70 overbought / <30 oversold; do not re-derive this)"

    def _sr_line(level: SRLevel | None) -> str:
        # Added 2026-09-09 — see TacticalSignals.nearest_resistance/
        # nearest_support's own field comment. The SAME data already sits
        # somewhere in the full technical_context prose below (via
        # ai.ftmo_suggest.format_chart_structure) — this line exists so
        # the single most tactically relevant real structural level can't
        # be missed among everything else there.
        if level is None:
            return "not available (no confirmed swing structure yet)"
        pool = " — LIQUIDITY POOL (tight cluster of real equal highs/lows, not just a single touch)" if level.is_liquidity_pool else ""
        # Real band, not a single point — added 2026-09-20, direct user
        # challenge. None low/high (a pre-upgrade SRLevel) falls back to
        # the old single-point rendering.
        if level.low is not None and level.high is not None:
            band = f"{level.low:.5f}-{level.high:.5f} (mid {level.price:.5f})"
        else:
            band = f"{level.price:.5f}"
        return f"{band} ({level.touches}x real confirmed touches, {level.distance_pct:+.2f}% away){pool}"

    def _structure_break_line(
        break_: StructureBreak | None, sweep: LiquiditySweepEvent | None
    ) -> str:
        # Added 2026-09-21, Phase 5c of the charting-expert technical-
        # analysis upgrade — see TacticalSignals.nearest_structure_
        # break's own field comment for why this is a genuinely
        # DIFFERENT signal from trend_flip_against_count (surfaced
        # separately, as a deterministic circuit breaker, in _run_clerk_
        # tactical_check below) rather than a duplicate of it: a CHOCH is
        # a one-shot structural EVENT, trend_flip_against_count tracks
        # SUSTAINED contradiction building up poll over poll — both real,
        # neither redundant with the other.
        bits = []
        if break_ is not None:
            bits.append(
                f"{break_.kind} {break_.bars_ago} bars ago ({break_.direction}, broke "
                f"{break_.broken_level:.5f})"
            )
        if sweep is not None:
            bits.append(
                f"liquidity sweep {sweep.bars_ago} bars ago ({sweep.direction}, "
                f"{sweep.wick_penetration_pct:.2f}% wick penetration)"
            )
        if not bits:
            return "none confirmed recently"
        return "; ".join(bits)

    def _backtest_favorability_line(favorability: str | None, excursion_median_r: float | None) -> str | None:
        # Added 2026-09-12 — real gap this closes: this position's own
        # real historical backtest evidence used to reach the mega-
        # session's Claude prompt only, with extensive caveats (see
        # ai/ftmo_suggest.py's own "CRITICAL, easy to misread" favorable-
        # excursion paragraph) that never travel with the raw data into
        # this file's own technical_context. See analysis.backtest.
        # classify_backtest_favorability's own docstring for the full
        # SUPPORTED/CONTRADICTED/MIXED rule and the excursion-figure
        # gate. Silence on "mixed"/None is deliberate — unlike an absent
        # RSI/S-R reading (a real data gap worth flagging), a MIXED or
        # unavailable backtest read is genuinely "no clean signal," and
        # narrating that would add noise without a clear steer.
        if favorability == "supported":
            excursion_clause = (
                f", and its historical favorable-excursion magnitude ({excursion_median_r:.2f}R median, "
                "measured over a MUCH LONGER window than this account's own same-session holding horizon "
                "— background evidence for how far this move CAN go, never a same-session TP price itself) "
                "suggests real room to continue"
                if excursion_median_r is not None
                else ""
            )
            return (
                "This setup's own real historical backtest evidence on this position's side SUPPORTS its "
                "underlying thesis (positive average realized R-multiple across resolved historical trades "
                f"for this setup){excursion_clause} — lean toward HOLD or a smaller partial-close fraction "
                "over an aggressive one, all else equal."
            )
        if favorability == "contradicted":
            return (
                "This setup's own real historical backtest evidence on this position's side CONTRADICTS its "
                "underlying thesis (negative average realized R-multiple across resolved historical trades "
                "for this setup) — a real reason to lean toward tightening/taking profit sooner, not a "
                "reason for optimism regardless of how the position is currently performing."
            )
        return None

    backtest_favorability_line = _backtest_favorability_line(
        signals.backtest_favorability, signals.favorable_excursion_median_r
    )

    # Decision-tier lines (M5) — only when a real M5 read exists, so a caller with H1-only signals
    # renders exactly as before. The nearest-level / structure-break lines below follow the same
    # timeframe (signals.structure_timeframe), the H1/H4 RSI lines are context.
    decision_lines: list[str] = []
    if signals.m5_rsi is not None:
        decision_lines.append(f"- {_rsi_line('M5', signals.m5_rsi, signals.m5_rsi_tier)}")
    if signals.structure_trail_candidate is not None:
        decision_lines.append(
            "- M5 structure-trail stop candidate (just beyond the nearest M5 "
            f"{'support' if position.side == 'buy' else 'resistance'}, 0.4 of an M5 ATR of room): "
            f"{signals.structure_trail_candidate:.5f} "
            f"({'tighter' if signals.structure_trail_is_tighter else 'NOT tighter'} than the current stop above)"
        )
    if signals.event_note:
        decision_lines.append(f"- Event risk: {signals.event_note}")
    decision_tier_lines = "".join(line + "\n" for line in decision_lines)

    pre_screen = (
        "Deterministic pre-screen (computed directly from live data — use "
        "these numbers rather than re-deriving your own arithmetic; you "
        "still decide which tier applies):\n"
        f"- Velocity: {velocity_line}.\n"
        f"{decision_tier_lines}"
        f"- {_rsi_line('H1', signals.h1_rsi, signals.h1_rsi_tier)} (context only)\n"
        f"- {_rsi_line('H4', signals.h4_rsi, signals.h4_rsi_tier)} (context only)\n"
        f"- Nearest {signals.structure_timeframe} structural resistance (real swing-point clustering, not a guess): "
        f"{_sr_line(signals.nearest_resistance)}\n"
        f"- Nearest {signals.structure_timeframe} structural support: {_sr_line(signals.nearest_support)}\n"
        f"- Most recent {signals.structure_timeframe} structure break/liquidity sweep: "
        f"{_structure_break_line(signals.nearest_structure_break, signals.nearest_liquidity_sweep)}\n"
        f"- Favorable move so far: {signals.favorable_move_pct:.2f}% "
        f"(profit-lock threshold: {config.CLERK_TACTICAL_PROFIT_LOCK_PCT}% "
        f"— {'REACHED' if signals.profit_lock_due else 'not reached'})\n"
        f"- {signals.atr_timeframe} ATR-based stop candidate "
        f"({signals.atr_stop_multiple_used}x {signals.atr_timeframe} ATR from the current "
        f"price): {atr_stop_line}\n"
        f"- Valid NEW_STOP_LOSS range for this {position.side.upper()} position: "
        f"{valid_stop_range_line}. A number outside this range is not a real "
        "protective stop at all (this account's own history: a local model "
        "proposed one on the wrong side of price six times in one real session) "
        "and will be replaced automatically with the ATR-based stop candidate "
        "above instead of being applied as given.\n"
        f"- Take-profit target captured so far: {target_captured_line} "
        f"(partial-profit threshold: {config.CLERK_TACTICAL_PARTIAL_PROFIT_TARGET_PCT}% "
        f"within {config.CLERK_TACTICAL_PARTIAL_PROFIT_MAX_DAYS:.0f} days — "
        f"{'DUE' if signals.partial_profit_due else 'not due'})\n"
        f"- Days held: {days_held_line}\n"
        + (f"- {backtest_favorability_line}\n" if backtest_favorability_line is not None else "")
        + "\n"
        "If you choose DEFEND under the ATR-stop rule, use the ATR-based "
        "stop candidate above verbatim as NEW_STOP_LOSS rather than "
        "deriving your own number. The RSI tier labels above (OVERBOUGHT/"
        "OVERSOLD/NEUTRAL) are already correctly computed by Python — use "
        "them exactly as given in your own reasoning; do not describe an "
        "OVERSOLD reading as overbought or vice versa. When placing or "
        "judging a stop, weigh the nearest structural resistance/support "
        "above: a stop sitting exactly ON a real level (especially one "
        "flagged LIQUIDITY POOL — a tight cluster of genuine equal highs/"
        "lows, the classic signature of where other traders' own stops "
        "concentrate) is more likely to be caught by ordinary noise or a "
        "brief stop-hunt than one placed a real distance beyond it.\n"
    )

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
        f"{pre_screen}\n"
        f"{format_trend_wisdom(brief=config.CLERK_COMPACT_CONTEXT)}\n\n"
        "Below is this symbol's REAL, LIVE technical picture right now (the "
        "same data source that powers this account's own Asset Health "
        "analysis):\n\n"
        f"{_TIMEFRAME_ROLES_NOTE}\n\n{technical_context}\n\n"
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
        "Keep your reasoning short - at most three sentences (for HOLD, one sentence is enough) - then end with EXACTLY "
        "this block and nothing after it (omit a line entirely if it doesn't apply):\n\n"
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


def _run_clerk_tactical_check(
    symbol: str,
    entry: AllocationEntry,
    position: Position,
    technical_context: str,
    account_equity: float,
    prior_tactical: dict | None,
    signals: TacticalSignals,
    deterministic_only: bool = False,
) -> tuple[str, TacticalVerdict, str]:
    """The LLM-bound half of a tactical-defense check — pure local HTTP
    call (with a same-server backup model, see
    _run_clerk_prompt), no MT5 involvement, safe to run from inside a
    thread pool (see run_clerk_execution_check's own parallel round).
    Returns (symbol, verdict, raw_text) — a bare symbol string in
    position 0 like _run_clerk_invalidation_check, but a TacticalVerdict
    (not a bool) in position 1, which is exactly how the merge loop tells
    the two kinds of bare-symbol result apart.

    O'Neil's hard stop-loss ceiling ("cut the loss ... with no
    exceptions") is a genuine circuit-breaker, not a judgment call — when
    signals.hard_exit_required, this returns a synthetic EXIT verdict
    WITHOUT calling the LLM at all, and the merge loop applies it even
    while tactical-defense is otherwise in shadow mode (see
    TacticalVerdict.hard_exit's own docstring).

    A second, equally deterministic circuit-breaker (added 2026-09-16,
    real NVDA incident — see TacticalSignals.trend_flip_against_count's
    own comment) checks signals.trend_flip_against_count next, once
    hard_exit_required has already been ruled out: at/past
    config.CLERK_TREND_FLIP_EXIT_AFTER_POLLS, a synthetic EXIT fires
    every poll from there on (same `>=`, keep-insisting-until-it-actually-
    closes philosophy as hard_exit_required's own check) — checked
    FIRST, so a position that's already past both thresholds always
    exits rather than merely being re-reduced. Exactly AT config.
    CLERK_TREND_FLIP_PARTIAL_AFTER_POLLS (an exact `==`, not `>=`, so
    this fires ONCE per contradicting streak, not on every poll the
    count happens to sit at or above it) — a synthetic DEFEND with
    partial_close_fraction only (no forced stop change) cuts the
    CURRENT volume by config.CLERK_TREND_FLIP_PARTIAL_REDUCE_PCT. Both
    set hard_exit=True — same bypass-shadow-mode reasoning as the O'Neil
    ceiling: a sustained, self-acknowledged trend contradiction is a
    genuine circuit-breaker, not a discretionary judgment call waiting
    on the tactical-defense rollout toggle."""
    if signals.hard_exit_required:
        raw = (
            _DETERMINISTIC_CIRCUIT_BREAKER_PREFIX +
            f"Adverse move {position.adverse_move_pct:.2f}% at/past the "
            f"{config.CLERK_TACTICAL_HARD_EXIT_PCT}% hard-stop ceiling."
        )
        verdict = TacticalVerdict(
            tier="exit",
            rule_citation="William O'Neil (How to Make Money in Stocks)",
            numbers_citation=(
                f"Adverse move {position.adverse_move_pct:.2f}% at/past the "
                f"{config.CLERK_TACTICAL_HARD_EXIT_PCT}% hard-stop ceiling — no exceptions."
            ),
            raw_text=raw,
            hard_exit=True,
        )
        return symbol, verdict, raw

    if signals.quick_profit_lock_due:
        raw = (
            _DETERMINISTIC_CIRCUIT_BREAKER_PREFIX +
            f"Position has moved {signals.favorable_move_pct:.2f}% in its favor, "
            f">= {config.CLERK_QUICK_PROFIT_LOCK_COST_MULTIPLE:g}x real round-trip "
            "cost with margin — locking a small, real profit."
        )
        verdict = TacticalVerdict(
            tier="defend",
            partial_close_fraction=config.CLERK_QUICK_PROFIT_LOCK_REDUCE_PCT / 100.0,
            rule_citation=(
                "Schwager's partial-profit-taking discipline, applied early/small for real, "
                "frequent realized gains rather than waiting for the full 50%-of-target milestone"
            ),
            numbers_citation=(
                f"Favorable move {signals.favorable_move_pct:.2f}% covers "
                f"{config.CLERK_QUICK_PROFIT_LOCK_COST_MULTIPLE:g}x real round-trip cost — "
                "banking a small, genuinely real gain."
            ),
            raw_text=raw,
            hard_exit=True,
        )
        return symbol, verdict, raw

    if signals.trend_flip_against_count >= config.CLERK_TREND_FLIP_EXIT_AFTER_POLLS:
        raw = (
            _DETERMINISTIC_CIRCUIT_BREAKER_PREFIX +
            f"{signals.trend_flip_against_count} consecutive polls of a confirmed {signals.trend_flip_basis} trend "
            "flip against this position's own side."
        )
        verdict = TacticalVerdict(
            tier="exit",
            rule_citation="Sustained self-acknowledged trend flip (real NVDA incident, 2026-09-10/11)",
            numbers_citation=(
                f"{signals.trend_flip_basis} trend has read opposite this position's side for "
                f"{signals.trend_flip_against_count} consecutive polls, at/past the "
                f"{config.CLERK_TREND_FLIP_EXIT_AFTER_POLLS}-poll exit ceiling."
            ),
            raw_text=raw,
            hard_exit=True,
        )
        return symbol, verdict, raw

    if signals.trend_flip_against_count == config.CLERK_TREND_FLIP_PARTIAL_AFTER_POLLS:
        raw = (
            _DETERMINISTIC_CIRCUIT_BREAKER_PREFIX +
            f"{signals.trend_flip_against_count} consecutive polls of a confirmed {signals.trend_flip_basis} trend "
            "flip against this position's own side — reducing size."
        )
        verdict = TacticalVerdict(
            tier="defend",
            partial_close_fraction=config.CLERK_TREND_FLIP_PARTIAL_REDUCE_PCT / 100.0,
            rule_citation="Sustained self-acknowledged trend flip (real NVDA incident, 2026-09-10/11)",
            numbers_citation=(
                f"{signals.trend_flip_basis} trend has read opposite this position's side for "
                f"{signals.trend_flip_against_count} consecutive polls, at the "
                f"{config.CLERK_TREND_FLIP_PARTIAL_AFTER_POLLS}-poll partial-reduction threshold — "
                f"cutting current volume by {config.CLERK_TREND_FLIP_PARTIAL_REDUCE_PCT:.0f}%."
            ),
            raw_text=raw,
            hard_exit=True,
        )
        return symbol, verdict, raw

    if signals.profit_trail_stop is not None:
        raw = (
            _DETERMINISTIC_CIRCUIT_BREAKER_PREFIX +
            f"Position is {signals.profit_trail_r:.2f}R in profit (>= {config.CLERK_PROFIT_TRAIL_START_R:g}R): "
            f"ratcheting the stop to {signals.profit_trail_stop:.5g}, {config.CLERK_PROFIT_TRAIL_ATR:g} M5 ATR behind the price."
        )
        verdict = TacticalVerdict(
            tier="defend",
            new_stop_loss=signals.profit_trail_stop,
            rule_citation=(
                "Schwager/Murphy trailing stop — let profits run, never give a winner back; measured on this "
                "account's own M5 data (trail 1.5 ATR after +1R beat breakeven and partial rules)"
            ),
            numbers_citation=(
                f"{signals.profit_trail_r:.2f}R progress on an original risk of {signals.profit_trail_r0:.5g}; "
                f"new stop {signals.profit_trail_stop:.5g} vs current {position.sl:.5g}."
            ),
            raw_text=raw,
            hard_exit=True,
        )
        return symbol, verdict, raw

    if deterministic_only:
        # Fast lane: every deterministic circuit-breaker above has been ruled out; the discretionary model verdict is the full poll's.
        return symbol, TacticalVerdict(tier="hold"), _FAST_LANE_RAW

    prompt = _build_tactical_prompt(
        symbol, entry, position, technical_context, account_equity, prior_tactical, signals,
    )
    raw = _run_clerk_prompt(prompt, timeout=config.CLERK_LLM_TIMEOUT_SECONDS)
    return symbol, parse_tactical_verdict(raw), raw


_FAST_LANE_RAW = "[fast lane - no model call]"
_REUSED_RAW = "[unchanged since the last thinking pass - no model call]"


def _tactical_fingerprint(position: Position, signals: "TacticalSignals") -> str:
    """What a tactical review actually depends on, coarsely: the position's own terms, price in half-ATR buckets, the RSI tier, the
    trend-flip count, whether a profit lock or a sweep is in play. If none of it moved since the last HOLD, asking the model again
    (30-90 s) can only repeat that HOLD."""
    atr = signals.m5_atr or 0.0
    bucket = round(position.price_current / (0.5 * atr)) if atr > 0 else round(position.price_current, 4)
    return json.dumps([
        position.side, round(position.volume, 4), position.sl, position.tp, bucket, signals.m5_rsi_tier,
        signals.trend_flip_against_count, bool(signals.profit_lock_due), bool(signals.quick_profit_lock_due),
        signals.nearest_liquidity_sweep is not None, getattr(signals.nearest_structure_break, "kind", None),
    ])


def _tactical_unchanged(cache: dict, symbol: str, fingerprint: str, now: datetime) -> bool:
    """True when the last stored tactical verdict for `symbol` was a HOLD on this same fingerprint and is younger than
    config.CLERK_THINK_REUSE_MAX_MINUTES (a hard ceiling so a long quiet spell is still re-read)."""
    item = cache.get("items", {}).get(f"tactical:{symbol}")
    if not item or item.get("fingerprint") != fingerprint or (item.get("verdict") or {}).get("tier") != "hold":
        return False
    try:
        age = now - datetime.fromisoformat(item["checked_utc"])
    except (KeyError, TypeError, ValueError):
        return False
    return age <= timedelta(minutes=config.CLERK_THINK_REUSE_MAX_MINUTES)
_MAX_PARTIAL_ATTEMPTS = 3


def _tactical_verdict_to_dict(v: "TacticalVerdict") -> dict:
    return {
        "tier": v.tier, "new_stop_loss": v.new_stop_loss, "partial_close_fraction": v.partial_close_fraction,
        "rule_citation": v.rule_citation, "numbers_citation": v.numbers_citation, "raw_text": v.raw_text, "hard_exit": v.hard_exit,
    }


def _tactical_verdict_from_dict(d: dict) -> "TacticalVerdict":
    return TacticalVerdict(
        tier=d.get("tier", "hold"), new_stop_loss=d.get("new_stop_loss"), partial_close_fraction=d.get("partial_close_fraction"),
        rule_citation=d.get("rule_citation", ""), numbers_citation=d.get("numbers_citation", ""), raw_text=d.get("raw_text", ""),
        hard_exit=bool(d.get("hard_exit", False)),
    )


def _store_thinking_results(results: list, generated_utc: str | None, tactical_fingerprints: dict | None = None) -> None:
    """The thinking pass's output: every model verdict of this pass, written for the acting passes (ai/clerk_thinking.py).
    Deterministic results (structured triggers, circuit-breakers) are not stored - acting passes recompute those themselves."""
    now = datetime.now(timezone.utc)
    cache = clerk_thinking.for_suggestion(clerk_thinking.load_cache(), generated_utc)
    for result in results:
        if isinstance(result[0], PendingSetup):
            setup, confirmed, raw = result
            if str(raw).startswith("[structured trigger]"):
                continue
            clerk_thinking.store(cache, "pending", setup.symbol, {
                "confirmed": bool(confirmed), "raw_text": raw, "side": setup.side, "trigger_condition": setup.trigger_condition,
            }, now)
        elif isinstance(result[1], TacticalVerdict):
            symbol, verdict, raw = result
            if raw.startswith(_DETERMINISTIC_CIRCUIT_BREAKER_PREFIX) or raw in (_FAST_LANE_RAW, _REUSED_RAW):
                continue  # (a reused HOLD keeps its earlier stored item and fingerprint)
            clerk_thinking.store(cache, "tactical", symbol, {
                "verdict": _tactical_verdict_to_dict(verdict), "raw_text": raw,
                "fingerprint": (tactical_fingerprints or {}).get(symbol),
            }, now)
        else:
            symbol, confirmed, raw = result
            if str(raw).startswith(_DETERMINISTIC_CIRCUIT_BREAKER_PREFIX):
                continue  # a mechanical invalidation check: acting passes recompute it themselves every pass
            clerk_thinking.store(cache, "invalidation", symbol, {"confirmed": bool(confirmed), "raw_text": raw}, now)
    clerk_thinking.mark_think_completed(cache, now)
    clerk_thinking.save_cache(cache)


def _cached_thinking_results(checkable, checkable_watched, checkable_tactical, account_equity, generated_utc) -> tuple[list, int]:
    """What an acting pass uses instead of calling the model: for each item the freshest UNUSED stored verdict (marked used), in the
    exact result shapes the merge loop already consumes. A tactical item first runs its deterministic circuit-breakers (they always
    win); only when none fires is the stored model verdict considered."""
    now = datetime.now(timezone.utc)
    cache = clerk_thinking.for_suggestion(clerk_thinking.load_cache(), generated_utc)
    out: list = []
    applied = 0
    for setup, _context, _description in checkable:
        hit = clerk_thinking.take_fresh(cache, "pending", setup.symbol, now)
        if hit and hit.get("side") == setup.side and hit.get("trigger_condition") == setup.trigger_condition:
            out.append((setup, bool(hit["confirmed"]), hit.get("raw_text", "")))
            applied += 1
    for symbol, _raw, _cond, _state, _context in checkable_watched:
        hit = clerk_thinking.take_fresh(cache, "invalidation", symbol, now)
        if hit:
            out.append((symbol, bool(hit["confirmed"]), hit.get("raw_text", "")))
            applied += 1
    for symbol, entry, position, context, prior_tactical, signals in checkable_tactical:
        checked = _run_clerk_tactical_check(
            symbol, entry, position, context, account_equity, prior_tactical, signals, deterministic_only=True
        )
        if checked[2] != _FAST_LANE_RAW:
            out.append(checked)  # a deterministic circuit-breaker fired
            continue
        hit = clerk_thinking.take_fresh(cache, "tactical", symbol, now)
        if hit:
            out.append((symbol, _tactical_verdict_from_dict(hit["verdict"]), hit.get("raw_text", "")))
            applied += 1
        else:
            out.append(checked)  # nothing fired, nothing new thought: skipped by the merge loop
    clerk_thinking.save_cache(cache)
    return out, applied


def _pending_partial_symbols(
    settled: dict, positions_by_symbol: dict[str, Position], merged_allocation: dict[str, AllocationEntry]
) -> set[str]:
    """Symbols whose tactical partial close (DEFEND with a fraction - the 20% quick profit lock, a trend-flip partial or a
    model-proposed one) still has to reach the broker. The reduce is planned with the held-size rounding tolerance switched OFF
    for them, and re-attempted on later polls (up to _MAX_PARTIAL_ATTEMPTS) until the held volume actually drops below the
    volume recorded when the partial was decided. A retry also re-pins the target to the persisted, already-reduced risk and
    stop, so it cannot be undone by a carried-forward baseline that still holds the original size (pending-setup origin)."""
    out: set[str] = set()
    for symbol, rec in settled.items():
        tactical = rec.get("tactical") or {}
        from_volume = tactical.get("partial_pending_from_volume")
        position = positions_by_symbol.get(symbol)
        if not from_volume or position is None:
            continue
        if position.volume < float(from_volume) - 1e-9:
            tactical["partial_pending_from_volume"] = None  # it went through
            continue
        attempts = int(tactical.get("partial_attempts", 0))
        if attempts >= _MAX_PARTIAL_ATTEMPTS:
            logger.warning("%s: the tactical partial close did not execute after %d attempts - giving up on it.", symbol, attempts)
            tactical["partial_pending_from_volume"] = None
            continue
        tactical["partial_attempts"] = attempts + 1
        entry = merged_allocation.get(symbol)
        persisted_pct = tactical.get("persisted_pct")
        if entry is not None and persisted_pct is not None and persisted_pct < entry.pct:
            merged_allocation[symbol] = _with_changes(
                entry, pct=persisted_pct,
                stop_loss=tactical.get("persisted_stop_loss", entry.stop_loss),
                take_profit=tactical.get("persisted_take_profit", entry.take_profit),
            )
        out.add(symbol)
    return out


def _validate_and_apply_tactical_verdict(
    symbol: str,
    verdict: TacticalVerdict,
    position: Position,
    existing_entry: AllocationEntry | None,
    prior_tactical: dict | None,
    account_equity: float,
    get_spec: Callable,
    signals: TacticalSignals | None = None,
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

    Found on audit: without this third value, the caller (run_clerk_
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

    HOLD -> (None, tactical_state_or_None, ""): the allocation is NEVER
    touched on HOLD, but (added 2026-09-16) if `signals` carries a
    trend_flip_against_count that's changed since prior_tactical's own
    last-saved value, a tactical_state update persisting JUST that field
    is still returned — otherwise the persisted counter would only ever
    advance on a DEFEND/EXIT poll, never during the ordinary HOLD-heavy
    streak this whole mechanism exists to catch (the real NVDA incident
    was dozens of consecutive HOLD verdicts, not DEFEND ones). (None,
    None, "") when the count hasn't changed (nothing new to persist) or
    `signals` isn't given.

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
       cooldown window) — config.CLERK_TACTICAL_DEFEND_COOLDOWN_
       MINUTES / _MIN_RETRIGGER_PCT. SKIPPED ENTIRELY when verdict.
       hard_exit is True (added 2026-09-16, real bug found on self-audit
       before this ever ran live) — this cooldown exists to stop a
       DISCRETIONARY, LLM-driven DEFEND from thrashing, and was never
       meant to gate a deterministic circuit-breaker (the trend-flip
       partial-reduction verdict, same category as the O'Neil hard-exit
       ceiling) that _run_clerk_tactical_check constructs directly:
       without this, an unrelated real DEFEND minutes earlier could
       silently suppress the reduction AND, since a suppressed DEFEND
       persists nothing back to settlement, freeze TacticalSignals.
       trend_flip_against_count's own persistence too — preventing the
       counter from ever reaching the exit threshold for as long as the
       position doesn't independently worsen enough to clear the
       cooldown on its own.
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

       `signals` (added 2026-09-10, optional/defaults to None so any
       existing caller keeps working unchanged) — real, repeated
       incident: the local tactical model correctly judged a position
       needed defending but proposed a geometrically invalid stop (wrong
       side of live price, or less protective than the current stop) six
       times in one real session, discarding its own correct underlying
       judgment along with the bad number every time. When checks 2/3
       fail for the model's own proposed stop, this now falls back to
       signals.atr_stop_candidate — already guaranteed valid by
       construction (see _compute_tactical_signals) — INSTEAD of
       discarding the whole DEFEND, the same "the model decides WHETHER,
       Python decides the NUMBER" split already used for the pre-screen
       inputs (RSI/velocity tiers), now applied to the output too. Only
       ever substitutes a value that independently passes both checks
       itself; if even that fails (or `signals` isn't given), rejects
       exactly as before.
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
        if signals is not None:
            prior_count = (prior_tactical or {}).get("trend_flip_against_count", 0)
            if signals.trend_flip_against_count != prior_count:
                tactical_state = {
                    **(prior_tactical or {}),
                    "trend_flip_against_count": signals.trend_flip_against_count,
                }
                return None, tactical_state, ""
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
            "trend_flip_against_count": (
                signals.trend_flip_against_count if signals is not None
                else (prior_tactical or {}).get("trend_flip_against_count", 0)
            ),
        }
        return new_entry, tactical_state, ""

    # tier == "defend"
    now = datetime.now(timezone.utc)
    # Real bug found on self-audit (2026-09-16), before the trend-flip
    # partial-reduction verdict (see _run_clerk_tactical_check's own
    # docstring) ever ran live: this cooldown exists to stop a
    # DISCRETIONARY, LLM-driven DEFEND from thrashing/over-firing — it
    # was never meant to gate a DETERMINISTIC circuit-breaker verdict
    # this function itself constructs directly (verdict.hard_exit=True,
    # same flag the O'Neil hard-exit ceiling already uses to bypass the
    # shadow-mode toggle elsewhere). Without this guard, a real, unrelated
    # DEFEND minutes earlier could silently suppress the partial
    # reduction (return None, None, ...) — and since a suppressed DEFEND
    # persists NOTHING back to settlement, trend_flip_against_count's own
    # already-incremented value would never be saved either, freezing the
    # counter and preventing it from ever reaching the exit threshold at
    # all for as long as the position's adverse move doesn't independently
    # worsen enough to clear the cooldown on its own — exactly the kind
    # of slow, gradual bleed (real NVDA incident) this mechanism exists
    # to catch.
    if not verdict.hard_exit and prior_tactical and prior_tactical.get("last_action_utc"):
        try:
            last_action_dt = datetime.fromisoformat(prior_tactical["last_action_utc"])
        except ValueError:
            last_action_dt = None
        if last_action_dt is not None:
            minutes_since = (now - last_action_dt).total_seconds() / 60
            worsened_by = position.adverse_move_pct - prior_tactical.get("adverse_move_pct_at_last_action", 0.0)
            if (
                minutes_since < config.CLERK_TACTICAL_DEFEND_COOLDOWN_MINUTES
                and worsened_by < config.CLERK_TACTICAL_DEFEND_MIN_RETRIGGER_PCT
            ):
                reason = (
                    f"suppressed by cooldown (only {minutes_since:.1f}m since the last "
                    f"action, cooldown is {config.CLERK_TACTICAL_DEFEND_COOLDOWN_MINUTES}m; "
                    f"adverse move only worsened by {worsened_by:.2f}%, needs "
                    f">= {config.CLERK_TACTICAL_DEFEND_MIN_RETRIGGER_PCT:.2f}%)"
                )
                logger.info("%s: tactical DEFEND %s.", symbol, reason)
                return None, None, reason

    # This MUST be the position's own live price for the "sane side of
    # the live price" check just below — a stop needs to be validated
    # against where the market actually is right now, not a stale figure.
    entry_price = position.price_current
    # sizing_entry_price is DIFFERENT on purpose (real coupling bug found
    # live 2026-09-08): compute_rebalance_plan's own sizing now anchors
    # an ALREADY-HELD position to its real, fixed price_open, not a live-
    # tracking price (see that function's own "sizing_price" comment for
    # the full incident — 9 separate EURUSD tickets opened/torn down via
    # dozens of tiny partial fills/closes inside 3.5 hours, because a
    # live-tracking anchor made the implied lot size thrash every poll
    # from ordinary tick noise alone). pct_for_target_lots's own round-
    # trip math only reproduces the SAME target_lots if it's solved
    # against the exact price compute_rebalance_plan will itself use —
    # this used to be position.price_current (back when compute_
    # rebalance_plan's own clamp tracked live price too), and must now
    # be position.price_open to match the fix, or a tactical stop-tighten
    # would silently imply a different lot count than intended and get
    # misread as a real reallocation instead of an in-place stop amend.
    sizing_entry_price = position.price_open
    current_stop = position.sl
    new_stop = verdict.new_stop_loss

    def _is_valid_stop(candidate: float) -> bool:
        """Both deterministic guardrails (never-widen-stop + sane-side-
        of-live-price) at once — used for the model's own proposed stop,
        and (added 2026-09-10) for the ATR-based fallback candidate
        substituted in when the model's own number fails either one.
        Requires current_stop to be known (checked by the caller)."""
        more_protective = candidate > current_stop if position.side == "buy" else candidate < current_stop
        sane_side = candidate < entry_price if position.side == "buy" else candidate > entry_price
        return more_protective and sane_side

    if new_stop is not None:
        if current_stop is None:
            reason = "rejected: proposed a new stop but the position has no current stop to compare against"
            logger.warning("%s: tactical DEFEND %s.", symbol, reason)
            return None, None, reason
        if _is_valid_stop(new_stop):
            resolved_stop = new_stop
        else:
            # Real, repeated incident found live 2026-09-09/10: the local
            # tactical model correctly judged this position needed
            # defending, but proposed a geometrically invalid stop (wrong
            # side of the live price, or less protective than the
            # current stop) six times in one real session — discarding
            # the whole DEFEND, including its own correct underlying
            # judgment, along with the bad number every time. Rather than
            # trust the model's own arithmetic a second time (same class
            # of failure RSI-tier/velocity-tier mislabeling already
            # needed a deterministic fix for), substitute the already-
            # computed, always-valid-by-construction ATR stop candidate
            # (see TacticalSignals.atr_stop_candidate / _compute_
            # tactical_signals) instead of discarding the DEFEND outright
            # — only when it independently clears both guardrails itself;
            # otherwise this falls through to the exact same rejection as
            # before `signals` existed.
            fallback = signals.atr_stop_candidate if signals is not None else None
            if fallback is not None and _is_valid_stop(fallback):
                logger.warning(
                    "%s: tactical DEFEND proposed an invalid stop %.5f (current stop "
                    "%.5f, live price %.5f, side %s) — substituting the deterministic "
                    "ATR-based stop candidate %.5f instead of discarding the DEFEND.",
                    symbol, new_stop, current_stop, entry_price, position.side, fallback,
                )
                resolved_stop = fallback
            else:
                more_protective = new_stop > current_stop if position.side == "buy" else new_stop < current_stop
                if not more_protective:
                    reason = (
                        f"rejected: proposed stop {new_stop:.5f} is not more protective than "
                        f"the current stop {current_stop:.5f} (never-widen-stop rule), and no "
                        "valid ATR-based fallback stop was available either"
                    )
                else:
                    # Found on audit: "more protective than the OLD stop"
                    # alone doesn't rule out a stop placed past the
                    # CURRENT live price — e.g. an over-eager tighten
                    # proposing a buy's stop ABOVE the current price,
                    # which is not a valid protective stop at all (it
                    # would trigger immediately, or be rejected by the
                    # broker). Rejected explicitly here, at the guardrail
                    # layer, rather than relying on compute_rebalance_
                    # plan's own incidental wrong-side-of-entry check
                    # downstream (which would silently no-op this as
                    # "infeasible" — safe, but with no clear reason
                    # surfaced here, and last_tactical_verdicts would
                    # misleadingly show "applied").
                    reason = (
                        f"rejected: proposed stop {new_stop:.5f} is on the wrong side of "
                        f"the current live price {entry_price:.5f}, and no valid ATR-based "
                        "fallback stop was available either"
                    )
                logger.warning("%s: tactical DEFEND %s.", symbol, reason)
                return None, None, reason
    else:
        resolved_stop = current_stop

    if resolved_stop is None:
        reason = "rejected: no usable stop (current or proposed) to size against"
        logger.warning("%s: tactical DEFEND %s.", symbol, reason)
        return None, None, reason

    fraction = verdict.partial_close_fraction
    if fraction is not None and not verdict.hard_exit and position.adverse_move_pct >= 0 and config.CLERK_PARTIAL_CLOSE_REQUIRES_PROFIT:
        # Found on the audit of the real MSFT trade (2026-09-22): the backup model (phi4-mini, qwen3 was down)
        # applied a DEFEND with a 25% partial close justified by "a 15% gain was already achieved" while the
        # position was ~1.4% UNDERWATER — the partial close banked a loss on a premise that did not exist. A
        # DEFEND's partial close is defined as taking PROFIT; a position that is not in profit gets its stop
        # tightened (if the model proposed one) but never a discretionary partial close. The deterministic
        # circuit-breakers (hard_exit: trend-flip cut, hard-stop ceiling) are exempt — reducing a losing
        # position is exactly their job.
        if new_stop is None:
            reason = (
                f"rejected: proposed a partial close on a position that is not in profit (adverse move "
                f"{position.adverse_move_pct:.2f}%) — a DEFEND partial close takes profit"
            )
            logger.warning("%s: tactical DEFEND %s.", symbol, reason)
            return None, None, reason
        logger.warning(
            "%s: tactical DEFEND proposed a partial close on a position that is not in profit (adverse move "
            "%.2f%%) — dropping the partial close, keeping the stop change.",
            symbol, position.adverse_move_pct,
        )
        fraction = None
    if fraction is not None:
        # A position at (or near) the minimum lot cannot be partially closed: 0.01 lot x (1 - 50%) is below the broker's minimum
        # volume, so the plan came back "infeasible" on every poll and - because the DEFEND is persisted as a whole - the stop
        # tightening that came with it was never sent either (XAGUSD, 2026-09-26). Drop the impossible partial, keep the stop.
        _spec = get_spec(symbol)
        if _spec is not None and _spec.volume_step > 0:
            remaining = math.floor(position.volume * (1.0 - fraction) / _spec.volume_step + 1e-9) * _spec.volume_step
            if remaining < _spec.volume_min - 1e-12:
                if new_stop is None and resolved_stop == current_stop:
                    logger.info(
                        "%s: tactical DEFEND wanted a %.0f%% partial close but %.4g lot cannot be reduced below the %.4g minimum - "
                        "nothing to apply.", symbol, fraction * 100, position.volume, _spec.volume_min,
                    )
                    lock_state = (
                        {**(prior_tactical or {}), "quick_profit_lock_done": True}
                        if signals is not None and signals.quick_profit_lock_due else None
                    )  # a lock that can never be banked must not re-fire every poll
                    return None, lock_state, ""
                logger.info(
                    "%s: tactical DEFEND partial close (%.0f%%) is below the minimum lot (%.4g of %.4g) - dropping it, keeping the stop change.",
                    symbol, fraction * 100, _spec.volume_min, position.volume,
                )
                fraction = None
    target_lots = position.volume * (1.0 - fraction) if fraction is not None else position.volume

    new_pct = pct_for_target_lots(symbol, target_lots, sizing_entry_price, resolved_stop, account_equity, get_spec)
    if new_pct is None:
        reason = "rejected: could not be sized (no usable contract spec or stop distance)"
        logger.warning("%s: tactical DEFEND %s.", symbol, reason)
        return None, None, reason

    new_entry = AllocationEntry(
        pct=new_pct,
        price=sizing_entry_price,
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
        # Added 2026-09-16 — this dict does NOT spread prior_tactical
        # (unlike the EXIT branch above), so without this explicit line a
        # real DEFEND would silently drop any already-accumulating
        # trend-flip streak back to invisible (read as 0 next poll).
        "trend_flip_against_count": (
            signals.trend_flip_against_count if signals is not None
            else (prior_tactical or {}).get("trend_flip_against_count", 0)
        ),
        # Added 2026-09-18 — same reasoning as trend_flip_against_count
        # immediately above: this dict does NOT spread prior_tactical, so
        # ANY DEFEND (whatever triggered it — quick-profit-lock, a
        # trend-flip partial, or a normal LLM-driven DEFEND) must
        # explicitly re-list this too, or a DEFEND for a completely
        # different reason would silently erase an already-banked
        # quick-lock flag, letting it incorrectly re-fire later.
        "quick_profit_lock_done": (
            signals.quick_profit_lock_due if signals is not None and signals.quick_profit_lock_due
            else (prior_tactical or {}).get("quick_profit_lock_done", False)
        ),
        # Real bug found live 2026-09-03 (USDCHF): without persisting the
        # REDUCED size here, the very next poll's _build_carried_forward_
        # allocation rebuilds its baseline straight from the mega
        # session's own unchanged, original pct — silently buying the
        # tactically-reduced amount right back via a fresh "increase"
        # order, since nothing about the DEFEND's protective action was
        # remembered past this one poll. Persisting the resolved size/
        # stop/target here lets the carried-forward baseline honor the
        # reduction until a genuinely fresh mega session (which resets
        # this whole tactical dict — see _reset_settlement_for_new_
        # session) explicitly re-proposes a different one.
        "persisted_pct": new_pct,
        "persisted_stop_loss": resolved_stop,
        "persisted_take_profit": new_entry.take_profit,
        # A partial close was requested: remember the volume it must come DOWN from, so the plan can insist on the resize
        # (bypassing the held-size rounding tolerance) and retry it if the order did not go through (see _pending_partial_symbols).
        "partial_pending_from_volume": position.volume if fraction is not None else None,
        "partial_attempts": 0,
        # The trade's ORIGINAL risk distance (see TacticalSignals.profit_trail_r0): carried through every
        # DEFEND so a tightened stop never shrinks the yardstick the profit trail measures progress in.
        "profit_trail_r0": (
            signals.profit_trail_r0 if signals is not None and signals.profit_trail_r0
            else (prior_tactical or {}).get("profit_trail_r0")
        ),
    }
    return new_entry, tactical_state, ""


# _mega_session_is_live used to be its own local copy of the same
# freshness heuristic app.py's live-progress display also needs — see
# ai.mega_analysis.mega_session_is_live's own docstring for why keeping
# two independent copies was itself a real risk (a fix applied to only
# one, as almost happened here) and for the age-ceiling bug found and
# fixed in both at once, 2026-08-27. Now a single shared implementation,
# imported above under the original name so this module's own call site
# and the tests that patch "ai.clerk_execution._mega_session_is_live"
# keep working unchanged.


_CORRELATION_SIZE_REDUCTION_FACTOR = config.CLERK_CORRELATION_SIZE_FACTOR
# Comfortably above _MIN_CORRELATION_OBSERVATIONS (30) once the last bar
# is dropped by .pct_change() — a generous buffer, not the bare minimum.
_CORRELATION_CHECK_D1_BARS = 90


def _fetch_correlation_closes(symbol: str) -> pd.Series | None:
    """Cheap, single-symbol D1 close series for _apply_correlation_guard
    below — deliberately NOT ai.ftmo_suggest.compute_ftmo_correlation_
    pairs' own full analyze_ftmo_assets() pipeline (multi-timeframe
    fetches + backtests across all 18 instruments), which is fine once a
    day for the mega session but would turn every 5-minute Clerk poll
    into an 18-instrument multi-timeframe pass. None on any failure or
    an empty/too-short result, same degrade-honestly convention as
    every other MT5 fetch in this app — a correlation check this can't
    run is skipped, never guessed at or allowed to block a trade."""
    closes = fetch_mt5_price_history(symbol, "D1", count=_CORRELATION_CHECK_D1_BARS)["Close"].dropna()
    return closes if len(closes) > _MIN_CORRELATION_OBSERVATIONS else None


def _apply_correlation_guard(
    merged_allocation: dict[str, AllocationEntry], positions: list[Position]
) -> dict[str, AllocationEntry]:
    """Real gap found on audit (2026-09-02): the mega session's own
    reports had correctly identified correlated risk in writing more
    than once (e.g. "BTC and ETH correlate 0.82 with each other, so
    both firing the same day would compound rather than diversify
    risk... later triggers must be resized down") but nothing in this
    execution pipeline ever actually enforced it — compute_aggregate_
    heat_pct (below) is a flat, correlation-blind sum of nominal risk
    %, only ever checked against the FTMO daily-loss gate. Confirmed
    live against this account's real MT5 history: ETHUSD, BTCUSD, and
    XAGUSD — all "risk-asset" positions — hit their OWN independently,
    correctly-sized stop-losses within 2 minutes of each other on
    2026-08-28, compounding into a single ~$116 drawdown (71% of this
    account's entire realized loss over the audited period) purely
    because nothing stopped three correlated bets from stacking on the
    same day.

    Called once, right before compute_aggregate_heat_pct/compute_
    rebalance_plan — on the SAME merged_allocation every execution path
    (immediate opens, confirmed pending setups, tactical entries) has
    already funneled into by this point, so one guard covers all of
    them rather than needing a copy per call site.

    For every symbol about to become NEW exposure this poll (pct > 0,
    not already held), checks pairwise correlation of daily returns
    against every symbol ALREADY exposed — starting from currently-held
    positions, then growing to include each earlier candidate already
    processed this SAME poll (in a stable, sorted order), so two
    brand-new correlated candidates confirming in the very same poll
    are caught too, not just a new one stacking on an old one. On a
    real match (|r| >= _HIGH_CORRELATION_THRESHOLD — the identical 0.7
    threshold the mega session's own correlation feature already uses,
    imported from there rather than a second, possibly-drifting copy),
    HALVES the new position's size rather than blocking it outright —
    Schwager's own framing (already quoted throughout this account's
    book-wisdom) is that correlated positions are "similar to one
    larger position," so this keeps the COMBINED bet closer to what a
    single position's own risk budget was meant to allow, without
    assuming the correlation reading is wrong or the trade idea itself
    is bad enough to refuse outright.

    Degrades honestly on missing/insufficient price data for either
    side of a pair (skips just that one comparison, proceeds at full
    size) — a data gap must never silently block a legitimate trade,
    matching this project's own standing convention everywhere else.

    Direction-aware, not just |correlation| — a real correctness bug
    caught on self-review before this ever shipped: two POSITIVELY
    correlated instruments traded in OPPOSITE directions (long one,
    short the other), or two NEGATIVELY correlated instruments traded
    in the SAME direction, are a genuine HEDGE — their P&L tends to
    move apart, not together — and reducing either one in that case
    would be exactly backwards, penalizing real diversification instead
    of real risk-stacking. What actually matters is the pair's EFFECTIVE
    P&L correlation: same side (both buy or both sell) with a positive
    price correlation, or opposite sides with a negative one, both mean
    the two POSITIONS' own results move together — that combination,
    not raw |r| alone, is what triggers a reduction."""
    exposed_symbols = {p.symbol for p in positions}
    side_by_symbol: dict[str, str] = {p.symbol: p.side for p in positions}
    for symbol, entry in merged_allocation.items():
        if symbol != "CASH" and entry.pct > 0:
            side_by_symbol[symbol] = entry.side
    candidates = sorted(
        symbol for symbol, entry in merged_allocation.items()
        if symbol != "CASH" and entry.pct > 0 and symbol not in exposed_symbols
    )
    if not candidates:
        return merged_allocation

    closes_cache: dict[str, pd.Series | None] = {}

    def _closes(symbol: str) -> pd.Series | None:
        if symbol not in closes_cache:
            closes_cache[symbol] = _fetch_correlation_closes(symbol)
        return closes_cache[symbol]

    for candidate in candidates:
        candidate_closes = _closes(candidate)
        if candidate_closes is None:
            exposed_symbols.add(candidate)
            continue

        match: tuple[str, float] | None = None
        for other in sorted(exposed_symbols):
            other_closes = _closes(other)
            if other_closes is None:
                continue
            aligned = pd.DataFrame(
                {"a": candidate_closes.pct_change(), "b": other_closes.pct_change()}
            ).dropna()
            if len(aligned) < _MIN_CORRELATION_OBSERVATIONS:
                continue
            corr = aligned["a"].corr(aligned["b"])
            if corr is None or pd.isna(corr) or abs(corr) < config.CLERK_CORRELATION_GUARD_THRESHOLD:
                continue
            same_side = side_by_symbol.get(candidate) == side_by_symbol.get(other)
            effective_pnl_correlation_is_positive = (same_side and corr > 0) or (not same_side and corr < 0)
            if effective_pnl_correlation_is_positive:
                match = (other, float(corr))
                break

        if match is not None:
            other, corr = match
            entry = merged_allocation[candidate]
            reduced_pct = entry.pct * _CORRELATION_SIZE_REDUCTION_FACTOR
            logger.info(
                "Correlation guard: %s (r=%+.2f vs already-exposed %s) — sizing reduced from "
                "%.2f%% to %.2f%%.",
                candidate, corr, other, entry.pct, reduced_pct,
            )
            merged_allocation[candidate] = AllocationEntry(
                pct=reduced_pct,
                price=entry.price,
                stop_loss=entry.stop_loss,
                take_profit=entry.take_profit,
                side=entry.side,
                reason=(
                    f"{entry.reason} [Correlation guard: r={corr:+.2f} vs already-exposed "
                    f"{other} — size reduced from {entry.pct:.2f}% to {reduced_pct:.2f}%.]"
                ),
                invalidation_condition=entry.invalidation_condition,
                entry_mode=entry.entry_mode,
            )
        exposed_symbols.add(candidate)

    return merged_allocation


# Deliberately permissive — the stale-entry re-anchor below uses construct_trade_zone
# ONLY to find a real, validated M5 ENTRY ZONE (and to confirm at least one real
# M5-structure level exists beyond it in the profitable direction, a basic sanity
# check); it discards that function's own take_profits/reward_risk output entirely
# and keeps the ORIGINAL take_profit (Claude's own real target). The real reward:risk
# check against that kept target is _apply_reward_risk_floor_guard, which runs after.
_M5_ENTRY_MIN_REWARD_RISK = 0.0


def _broker_clock_offset(symbols) -> timedelta:
    """How far ahead of true UTC the broker's server clock runs, rounded to the
    nearest quarter hour (real FTMO offsets are whole/half hours). MT5 stamps
    an order's `time_setup` on that server clock while this process's own "now"
    is real UTC, so an order's age computed naively is off by the whole offset
    (a +3h server made a brand-new order look 3 hours YOUNGER — invisible under
    the old 24h age ceiling, material under the 3h intraday one). The offset is
    read from a live tick; a symbol whose tick is stale (a closed market)
    produces an offset that is not within seconds of a quarter hour and is
    skipped. 0 when no symbol gives a trustworthy reading."""
    for symbol in symbols:
        try:
            offset = get_server_time_offset(symbol)
        except Exception:
            continue
        if offset is None:
            continue
        quarter = 900.0
        rounded = round(offset.total_seconds() / quarter) * quarter
        if abs(offset.total_seconds() - rounded) <= 10.0 and abs(rounded) <= 14 * 3600:
            return timedelta(seconds=rounded)
    return timedelta(0)


_GUARD_NOTE_RE = re.compile(
    r"\[(?:Stale-entry|ATR stop-floor guard|Intraday size scalar|Reward:risk|Correlation guard|Event blackout)[^\]]*\]",
    re.IGNORECASE,  # the reward:risk guard writes its note lowercase
)


_ANY_GUARD_NOTE_RE = re.compile(r"\[([^\]]*(?:guard|Event blackout|Reward:risk|Stale-entry)[^\]]*)\]", re.IGNORECASE)


def _last_guard_note(reason: str | None) -> str | None:
    """The text of the LAST guard note ("[Entry-mode guard: ...]", "[Reward:risk ...]", ...) a guard appended to an
    entry's reason, or None - what the Clerk desk shows as WHY a target was held back."""
    matches = _ANY_GUARD_NOTE_RE.findall(reason or "")
    return matches[-1].strip() if matches else None


def _guard_blocks_this_poll(
    pct_before: dict[str, float],
    merged_allocation: dict[str, AllocationEntry],
    positions: list[Position],
    cooldowns: dict,
) -> dict[str, str]:
    """{symbol: why} for every UNFILLED candidate that had a live target going into the guard pipeline and came out at
    0% - so the desk can say "held back by <guard>" instead of the meaningless "on target, no change needed"."""
    held = {p.symbol for p in positions}
    blocks: dict[str, str] = {}
    for symbol, before in pct_before.items():
        entry = merged_allocation.get(symbol)
        if symbol == "CASH" or before <= 0 or entry is None or entry.pct > 0 or symbol in held:
            continue
        cooldown = cooldowns.get(symbol)
        note = _last_guard_note(entry.reason)
        if cooldown is not None and not note:
            try:
                until = datetime.fromisoformat(cooldown["until_utc"]).strftime("%H:%M")
            except (KeyError, TypeError, ValueError):
                until = "?"
            note = f"cooldown until {until} UTC after: {cooldown.get('reason') or 'an earlier guard rejection'}"
        blocks[symbol] = note or "a guard held the target at 0%"
    return blocks


def _candidate_symbols(merged_allocation: dict[str, AllocationEntry], positions: list[Position]) -> list[str]:
    """New-exposure candidates: pct > 0, not CASH, not already an open position —
    the same framing every guard in this pipeline uses."""
    exposed_symbols = {p.symbol for p in positions}
    return sorted(
        symbol for symbol, entry in merged_allocation.items()
        if symbol != "CASH" and entry.pct > 0 and symbol not in exposed_symbols
    )


def _with_changes(entry: AllocationEntry, **changes) -> AllocationEntry:
    return dataclasses.replace(entry, **changes)


_MARKET_CLOSED_RE = re.compile(r"market\s+closed", re.IGNORECASE)


def _order_backoff_until(backoff: dict, symbol: str, now_utc: datetime) -> datetime | None:
    """When a "Market closed" rejection put this symbol's order actions on hold until (None when not held).
    Expired or corrupt records are dropped in place."""
    record = backoff.get(symbol)
    if not record:
        return None
    try:
        until = datetime.fromisoformat(record["until_utc"])
    except (KeyError, TypeError, ValueError):
        backoff.pop(symbol, None)
        return None
    if until <= now_utc:
        backoff.pop(symbol, None)
        return None
    return until


def _learn_order_outcome(backoff: dict, symbol: str, result, now_utc: datetime) -> None:
    """A success clears the symbol's backoff; a "Market closed" rejection starts one, so the next polls do not
    hammer the broker with an order action that cannot work until the market reopens (56 failed cancels in one
    night before this existed)."""
    if result.success:
        backoff.pop(symbol, None)
        return
    if _MARKET_CLOSED_RE.search(result.comment or ""):
        until = now_utc + timedelta(minutes=config.MARKET_CLOSED_BACKOFF_MINUTES)
        if symbol not in backoff:
            logger.info(
                "%s: the market is closed — order actions for it are deferred until %s UTC.",
                symbol, until.strftime("%H:%M"),
            )
        backoff[symbol] = {"since_utc": now_utc.isoformat(), "until_utc": until.isoformat()}


def _minutes_to_session_close(history, now_utc: datetime, broker_offset: timedelta) -> float | None:
    """Minutes until the instrument's usual session end, learned from where its last sessions actually ended
    (the bar before each overnight gap of 3-30 hours; a 1-hour maintenance break or a weekend does not count).
    None when the instrument shows no daily close (24h markets), has too little history, or is not trading right
    now (last bar older than 20 minutes) — a missing estimate never blocks anything."""
    import pandas as pd

    if history is None or len(history) < 60:
        return None
    times = pd.DatetimeIndex(history.index) - pd.Timedelta(broker_offset)
    gaps = times[1:] - times[:-1]
    ends = [times[i] for i, gap in enumerate(gaps) if pd.Timedelta(hours=3) <= gap <= pd.Timedelta(hours=30)]
    if len(ends) < 2:
        return None
    close_tods = sorted(((t + pd.Timedelta(minutes=5)).hour * 60 + (t + pd.Timedelta(minutes=5)).minute) for t in ends[-3:])
    close_tod = float(close_tods[len(close_tods) // 2])
    now = pd.Timestamp(now_utc).tz_localize(None) if pd.Timestamp(now_utc).tzinfo is None else pd.Timestamp(now_utc).tz_convert("UTC").tz_localize(None)
    if now - times[-1] > pd.Timedelta(minutes=20):
        return None
    minutes = close_tod - (now.hour * 60 + now.minute + now.second / 60)
    if minutes < -1:
        if close_tod < 180:  # a session that ends just after midnight UTC
            minutes += 1440
        else:
            return None
    return minutes if 0 <= minutes <= 1440 else None


def _resting_order_expiry_hours(symbol: str, now_utc: datetime, broker_offset: timedelta, history_fn=None) -> float | None:
    """Hours until a NEW resting order should expire on the broker's side: the instrument's learned session close minus
    config.RESTING_ORDER_EXPIRY_BEFORE_CLOSE_MINUTES, so it can never sit overnight unmanaged (the NVDA case). None for a
    24h market, an unknown close or when it is too close to be useful (the pre-close guard already blocks those)."""
    if config.RESTING_ORDER_EXPIRY_BEFORE_CLOSE_MINUTES < 0:
        return None
    history_fn = history_fn or (lambda s: fetch_mt5_price_history(s, "M5", count=1500))
    try:
        minutes = _minutes_to_session_close(history_fn(symbol), now_utc, broker_offset)
    except Exception:  # noqa: BLE001 - no estimate = plain GTC, exactly as before
        return None
    if minutes is None:
        return None
    hours = (minutes - config.RESTING_ORDER_EXPIRY_BEFORE_CLOSE_MINUTES) / 60
    return hours if hours >= 0.25 else None


def _apply_pre_close_guard(
    merged_allocation: dict[str, AllocationEntry],
    positions: list[Position],
    pending_orders: list,
    now_utc: datetime,
    broker_offset: timedelta,
    history_fn=None,
) -> dict[str, AllocationEntry]:
    """No NEW resting order in the last config.NO_NEW_ORDER_MINUTES_BEFORE_CLOSE minutes of a non-24h
    instrument's session (see _minutes_to_session_close): it could not be managed or cancelled overnight. A
    symbol that already has an order resting or a position is never touched here, and the hold is temporary (no
    cooldown latch) — the setup is re-evaluated at the next open."""
    if config.NO_NEW_ORDER_MINUTES_BEFORE_CLOSE <= 0:
        return merged_allocation
    history_fn = history_fn or (lambda symbol: fetch_mt5_price_history(symbol, "M5", count=1500))
    held = {p.symbol for p in positions}
    resting = {o.symbol for o in pending_orders}
    for symbol in sorted(merged_allocation):
        entry = merged_allocation[symbol]
        if symbol == "CASH" or entry.pct <= 0 or symbol in held or symbol in resting:
            continue
        try:
            if get_symbol_category(symbol).startswith("Crypto"):
                continue
            minutes = _minutes_to_session_close(history_fn(symbol), now_utc, broker_offset)
        except Exception:
            logger.debug("Pre-close guard: no session estimate for %s (not blocking).", symbol, exc_info=True)
            continue
        if minutes is not None and minutes < config.NO_NEW_ORDER_MINUTES_BEFORE_CLOSE:
            logger.info(
                "Pre-close guard: %s's session ends in ~%.0f min (< %g) — not placing a new resting order that "
                "could not be managed overnight.",
                symbol, minutes, config.NO_NEW_ORDER_MINUTES_BEFORE_CLOSE,
            )
            merged_allocation[symbol] = _with_changes(entry, pct=0.0)
    return merged_allocation


def _apply_guard_cooldown(
    merged_allocation: dict[str, AllocationEntry],
    positions: list[Position],
    cooldowns: dict,
    now_utc: datetime,
) -> dict[str, AllocationEntry]:
    """Holds a symbol at pct 0 while its guard-rejection cooldown is active (see _record_guard_rejections).
    Only ever touches an UNFILLED candidate — a held position's management is never gated here. Expired
    cooldowns are dropped from `cooldowns` in place. The cooldown lives in the settlement file, which is
    rebuilt from scratch when a new Mega session supersedes the old one, so a fresh proposal always gets a
    fresh evaluation."""
    held = {p.symbol for p in positions}
    for symbol in list(cooldowns):
        try:
            until = datetime.fromisoformat(cooldowns[symbol]["until_utc"])
        except (KeyError, TypeError, ValueError):
            del cooldowns[symbol]
            continue
        if until <= now_utc:
            del cooldowns[symbol]
            continue
        entry = merged_allocation.get(symbol)
        if entry is None or entry.pct <= 0 or symbol in held:
            continue
        logger.info(
            "Guard-rejection cooldown: %s was rejected by a guard while unfilled — holding it at 0%% until "
            "%s UTC (no place/cancel churn).",
            symbol, until.strftime("%H:%M"),
        )
        merged_allocation[symbol] = _with_changes(entry, pct=0.0)
    return merged_allocation


def _record_guard_rejections(
    pct_before: dict[str, float],
    merged_allocation: dict[str, AllocationEntry],
    positions: list[Position],
    cooldowns: dict,
    now_utc: datetime,
    skip: set[str] | frozenset[str] = frozenset(),
) -> list[str]:
    """After the guard pipeline: every UNFILLED symbol that had a live pct going in and came out at 0 was
    rejected by a guard (stale-entry / reward:risk) — start its cooldown so the next polls do not resurrect
    it, place it, and cancel it again. Returns the newly latched symbols.

    `skip`: symbols rejected by the ENTRY-MODE guard (breakout already extended, market entry not chasable, setup dead):
    nothing was ever placed for them, so there is no place/cancel churn to prevent, and the condition is TRANSIENT - a
    breakout that ran 0.99 ATR past its trigger may be back inside the cap two polls later. Latching a 60-minute cooldown
    there (2026-09-25, XAGUSD) only delayed a trade the Mega session had just approved; they are re-judged every poll."""
    held = {p.symbol for p in positions}
    latched = []
    for symbol, before in pct_before.items():
        entry = merged_allocation.get(symbol)
        if symbol == "CASH" or before <= 0 or entry is None or entry.pct > 0 or symbol in held or symbol in cooldowns:
            continue
        if symbol in skip:
            continue
        until = now_utc + timedelta(minutes=config.GUARD_REJECTION_COOLDOWN_MINUTES)
        cooldowns[symbol] = {
            "since_utc": now_utc.isoformat(), "until_utc": until.isoformat(),
            "reason": _last_guard_note(entry.reason) or "a guard rejection",
        }
        latched.append(symbol)
        logger.info(
            "Guard-rejection cooldown started: %s rejected by a guard while unfilled — not placing it again for "
            "%g min (until %s UTC) or until the next Mega session.",
            symbol, config.GUARD_REJECTION_COOLDOWN_MINUTES, until.strftime("%H:%M"),
        )
    return latched


def _apply_entry_mode_guard(
    merged_allocation: dict[str, AllocationEntry],
    positions: list[Position],
    pending_orders: list,
    technical_context_cache: dict[str, tuple[FtmoAssetAnalysis, str]],
    market_prices: dict,
    market_entries_today: set[str],
) -> tuple[dict[str, AllocationEntry], dict[str, float]]:
    """Re-verifies every fresh entry that asks for a non-limit entry_mode ("stop" breakout trigger or
    "market") against the LIVE quote and M5 ATR (analysis.entry_mode.resolve_entry_mode: slippage cap,
    spread share of the stop, breakout not extended, one market entry per symbol per day, kill switch).
    Whatever fails is downgraded to the plain limit it always was, or rejected (pct -> 0) when the setup is
    already dead (through the stop / at the target / breakout extended) — never sent as asked. A passing
    "market" entry is re-priced to the live ask/bid (so the stop-floor and reward:risk guards below judge the
    real fill) and gets a send-time deviation cap, returned as {symbol: max_deviation_price}.

    Symbols that already rest an order of the requested kind are left alone: re-resolving a resting breakout
    trigger every poll would nudge it and churn cancel/replace. Held symbols are never candidates."""
    caps: dict[str, float] = {}
    resting_kinds: dict[str, set[str]] = {}
    for order in pending_orders or []:
        resting_kinds.setdefault(order.symbol, set()).add("stop" if "stop" in order.order_type.lower() else "limit")
    for symbol in _candidate_symbols(merged_allocation, positions):
        entry = merged_allocation[symbol]
        mode = normalise_entry_mode(entry.entry_mode)
        asset = market_prices.get(symbol)
        if mode == "limit":
            # A limit that is ALREADY MARKETABLE - a sell limit at/below the live bid, a buy limit at/above the live ask - cannot
            # rest: the broker answers "Invalid price". 2026-09-25, EURUSD: a sell limit at 1.1386 with the market at 1.1389 failed
            # that way on nine consecutive polls (80 minutes) until it filled by accident. Its own price is BETTER than planned, so
            # treat it as the market entry it is, through exactly the same caps (slippage, spread, one market entry a day, dead
            # setup) - and the invalidation guard has already removed it if its thesis line is broken.
            if "limit" in resting_kinds.get(symbol, set()) or asset is None or entry.price is None:
                continue
            if not ((asset.bid >= entry.price) if entry.side == "sell" else (asset.ask <= entry.price)):
                continue
            mode = "market"
            logger.info("Entry-mode guard: %s limit %.5g is already marketable at the live %s %.5g - resolving it as a market entry.",
                        symbol, entry.price, "bid" if entry.side == "sell" else "ask", asset.bid if entry.side == "sell" else asset.ask)
        if mode in resting_kinds.get(symbol, set()):
            continue
        fetched = technical_context_cache.get(symbol)
        if fetched is None or asset is None or entry.price is None or entry.stop_loss is None:
            merged_allocation[symbol] = _with_changes(entry, entry_mode="limit")
            continue
        analysis, _ = fetched
        min_stop_pct = analysis.trade_cost.min_stop_distance_pct if analysis.trade_cost is not None else 0.0
        decision = resolve_entry_mode(
            requested=mode, side=entry.side, planned_price=entry.price, stop_loss=entry.stop_loss,
            take_profit=entry.take_profit, bid=asset.bid, ask=asset.ask, atr=analysis.m5_stats.atr,
            min_stop_distance_price=min_stop_pct / 100 * (asset.ask or 0.0),
            market_entry_already_used=symbol in market_entries_today,
        )
        if decision.mode == "reject":
            logger.info("Entry-mode guard: %s %s rejected — %s.", symbol, mode, decision.reason)
            rehunt.record_dead_entry(
                symbol, entry.side, entry.price, entry.stop_loss, entry.take_profit, f"Entry-mode guard: {decision.reason}"
            )
            merged_allocation[symbol] = _with_changes(
                entry, pct=0.0, entry_mode="limit",
                reason=f"{entry.reason} [Entry-mode guard: {decision.reason} — rejected.]",
            )
            continue
        note = f" [Entry-mode guard: {decision.reason}.]" if decision.reason else ""
        logger.info("Entry-mode guard: %s requested %s -> %s at %s. %s", symbol, mode, decision.mode, decision.price, decision.reason)
        merged_allocation[symbol] = _with_changes(
            entry, entry_mode=decision.mode, price=decision.price, reason=f"{entry.reason}{note}"
        )
        if decision.mode == "market" and decision.max_deviation_price is not None:
            caps[symbol] = decision.max_deviation_price
    return merged_allocation, caps


def _apply_stale_entry_reanchor(
    merged_allocation: dict[str, AllocationEntry],
    positions: list[Position],
    technical_context_cache: dict[str, tuple[FtmoAssetAnalysis, str]],
) -> dict[str, AllocationEntry]:
    """Intraday decision-tier upgrade (2026-09-24). Claude now proposes entry/stop/
    target from M5 structure, so this guard no longer rewrites every entry (the
    old 2026-09-22 behavior, which moved just the ENTRY to an M5 zone midpoint while
    keeping an H1-scale stop and the old target — the exact recipe behind the MSFT
    loss). It only acts when a proposal has gone STALE between Mega Session's
    analysis and this poll:

    1. TARGET ALREADY REACHED — live price is already at or beyond the take-profit
       without the entry ever filling: the trade is obsolete, so it is rejected
       (pct -> 0) rather than chased. (Real prior incident: a pending buy sat
       unfilled while price ran through its own target.)
    2. ENTRY TOO FAR FROM MARKET — the planned limit sits further from live price
       than config.STALE_ENTRY_M5_ATR_MULTIPLE x the M5 ATR (a limit that far from
       market will not fill any time soon). If a real, current M5 entry zone exists
       in the same direction and is genuinely closer to live price, the entry moves
       to that zone's midpoint and the stop is re-derived at Claude's own original
       stop DISTANCE from it; the target is kept. No M5 zone -> untouched.

    Uses the M5 reads already carried on the cached analysis (no extra MT5 fetch).
    A refinement, never a new hard gate beyond case 1."""
    for symbol in _candidate_symbols(merged_allocation, positions):
        entry = merged_allocation[symbol]
        if entry.price is None or entry.stop_loss is None or entry.take_profit is None:
            continue
        if normalise_entry_mode(entry.entry_mode) != "limit":
            continue  # a verified stop/market entry is intentionally at/beyond the market: not "stale"
        fetched = technical_context_cache.get(symbol)
        if fetched is None:
            continue
        analysis, _ = fetched
        m5_atr = analysis.m5_stats.atr
        if m5_atr is None or m5_atr <= 0:
            continue
        live = analysis.base.ask if entry.side == "buy" else analysis.base.bid
        if not live:
            continue

        target_reached = live >= entry.take_profit if entry.side == "buy" else live <= entry.take_profit
        if target_reached:
            logger.info(
                "Stale-entry guard: %s live price %.5f is already at/through its own target %.5f without "
                "the entry ever filling — rejecting (pct %.4f%% -> 0), not chasing.",
                symbol, live, entry.take_profit, entry.pct,
            )
            rehunt.record_dead_entry(
                symbol, entry.side, entry.price, entry.stop_loss, entry.take_profit,
                f"live price {live:.5g} reached the target {entry.take_profit:.5g} without the entry ever filling",
            )
            merged_allocation[symbol] = _with_changes(
                entry, pct=0.0,
                reason=f"{entry.reason} [Stale-entry guard: price already reached the target without a fill — rejected.]",
            )
            continue

        distance_from_market = (live - entry.price) if entry.side == "buy" else (entry.price - live)
        if distance_from_market <= config.STALE_ENTRY_M5_ATR_MULTIPLE * m5_atr:
            continue

        m5_signals = classify_setups(
            analysis.m5_stats, analysis.m5_structure, analysis.m5_divergence, profile=M5_PROFILE
        )
        zone = construct_trade_zone(
            entry.side, analysis.m5_stats, analysis.m5_structure, m5_signals,
            min_reward_risk=_M5_ENTRY_MIN_REWARD_RISK, stop_atr=m5_atr,
        )
        if zone is None:
            continue
        new_entry_price = (zone.entry_low + zone.entry_high) / 2
        new_distance = (live - new_entry_price) if entry.side == "buy" else (new_entry_price - live)
        if new_distance < 0 or new_distance >= distance_from_market:
            continue  # not genuinely closer / not on the fillable side of live price

        old_stop_distance = abs(entry.price - entry.stop_loss)
        new_stop_loss = (
            new_entry_price - old_stop_distance if entry.side == "buy" else new_entry_price + old_stop_distance
        )
        logger.info(
            "Stale-entry re-anchor: %s planned entry %.5f is %.2fx M5 ATR from live price %.5f (limit %.2fx) "
            "— moved to the real M5 %s zone %.5f (%.5f-%.5f), stop distance kept at %.5f, target kept.",
            symbol, entry.price, distance_from_market / m5_atr, live, config.STALE_ENTRY_M5_ATR_MULTIPLE,
            zone.basis, new_entry_price, zone.entry_low, zone.entry_high, old_stop_distance,
        )
        merged_allocation[symbol] = _with_changes(
            entry, price=new_entry_price, stop_loss=new_stop_loss,
            reason=(
                f"{entry.reason} [Stale-entry re-anchor: the planned entry {entry.price:.5f} had drifted "
                f"{distance_from_market / m5_atr:.1f}x M5 ATR from live price; moved to a real, current M5 "
                f"{zone.basis} zone at {new_entry_price:.5f} — same stop distance, same target.]"
            ),
        )
    return merged_allocation


_INVALIDATION_LEVEL_RE = re.compile(r"close[sd]?\s+(?:back\s+)?(above|below)\s+\$?(\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)


def invalidation_already_true(side: str, condition: str | None, bid: float | None, ask: float | None) -> str | None:
    """A sentence saying WHY a not-yet-open entry is already invalid, or None. Reads the mechanical part of the Mega session's
    invalidation_condition ("M5 closes above 1.1397" for a sell, "closes below X" for a buy) and compares the live mid price:
    when the market is already beyond that line, the thesis is dead BEFORE the trade exists. Conditions it cannot parse, or that
    point the other way for the side, never block anything."""
    if not condition or not bid or not ask or side not in ("buy", "sell"):
        return None
    match = _INVALIDATION_LEVEL_RE.search(condition)
    if match is None:
        return None
    direction, level = match.group(1).lower(), float(match.group(2).replace(",", ""))
    mid = (bid + ask) / 2
    if side == "sell" and direction == "above" and mid > level:
        return f"live price {mid:.5g} is already above the invalidation level {level:.5g}"
    if side == "buy" and direction == "below" and mid < level:
        return f"live price {mid:.5g} is already below the invalidation level {level:.5g}"
    return None


def _apply_invalidation_guard(
    merged_allocation: dict[str, AllocationEntry], positions: list[Position], market_prices: dict
) -> dict[str, AllocationEntry]:
    """No NEW entry whose own invalidation condition is already true at the live price. Real case, 2026-09-25 EURUSD: a sell
    limit drafted at 1.1386 with the invalidation "M5 closes above 1.1397" kept failing "Invalid price" while the market ran up,
    finally filled at 1.1400 - already ABOVE its invalidation line - and was closed 13 minutes later as invalidated for +$1.30
    on a $25 risk. Re-judged every poll (nothing is placed, so no cooldown); the entry comes back if price returns inside."""
    for symbol in _candidate_symbols(merged_allocation, positions):
        entry = merged_allocation[symbol]
        asset = market_prices.get(symbol)
        if asset is None:
            continue
        why = invalidation_already_true(entry.side, entry.invalidation_condition, asset.bid, asset.ask)
        if why is None:
            continue
        logger.info("Invalidation guard: %s not entered - %s (\"%s\").", symbol, why, entry.invalidation_condition)
        if entry.price is not None and entry.stop_loss is not None:
            rehunt.record_dead_entry(symbol, entry.side, entry.price, entry.stop_loss, entry.take_profit, f"Invalidation guard: {why}")
        merged_allocation[symbol] = _with_changes(
            entry, pct=0.0, reason=f"{entry.reason} [Invalidation guard: {why} - the thesis is dead before entry.]"
        )
    return merged_allocation


def _apply_event_blackout_guard(
    merged_allocation: dict[str, AllocationEntry],
    positions: list[Position],
    now_utc: datetime,
    calendar_events: list | None,
) -> dict[str, AllocationEntry]:
    """No NEW or RESTING entry orders from config.EVENT_BLACKOUT_BEFORE_MINUTES before
    to EVENT_BLACKOUT_AFTER_MINUTES after a High-impact economic event for the
    symbol's own currencies (data/economic_calendar.py). Sets pct to 0 — for a
    symbol that already has a resting order the plan cancels it, and once the window
    passes the entry is re-evaluated fresh. Open positions are never touched (their
    event handling is advisory, via the tactical prompt). No calendar data ->
    nothing blocked (fail-open: an unofficial feed outage must not freeze trading)."""
    if not config.EVENT_BLACKOUT_ENABLED or not calendar_events:
        return merged_allocation
    for symbol in _candidate_symbols(merged_allocation, positions):
        status = economic_calendar.blackout_status(symbol, now_utc, calendar_events)
        if not status.active:
            continue
        entry = merged_allocation[symbol]
        logger.info(
            "Event blackout: %s entry blocked this poll (%s) — pct %.4f%% -> 0.", symbol, status.reason, entry.pct
        )
        merged_allocation[symbol] = _with_changes(
            entry, pct=0.0, reason=f"{entry.reason} [Event blackout: {status.reason} — no new/resting entry.]"
        )
    return merged_allocation


def _tactical_event_note(symbol: str, now_utc: datetime, calendar_events: list | None) -> str | None:
    """One advisory line for the tactical prompt: High-impact events for this symbol's
    currencies within the next 3 hours (or a very recent one)."""
    if not calendar_events:
        return None
    hits = economic_calendar.upcoming_events(
        symbol, now_utc, hours_ahead=3.0, events=calendar_events, hours_back=0.25
    )
    if not hits:
        return None
    parts = []
    for e in hits[:3]:
        minutes = (e.time_utc - now_utc).total_seconds() / 60
        when = f"in {minutes:.0f} min" if minutes >= 0 else f"{abs(minutes):.0f} min ago"
        parts.append(f"{e.currency} {e.title} {when} ({e.time_utc:%H:%M} UTC)")
    return "High-impact event(s) for this instrument's currencies: " + "; ".join(parts) + (
        " — consider tightening the stop or trimming before it lands; volatility and spread can spike."
    )


def _weekly_close_note(symbol: str, now_utc: datetime) -> str | None:
    """Advisory only (the user chose an advisory, not forced, same-day exit): on a Friday, within
    3 hours of this instrument's weekly close, say so — a position held through it is held over
    the weekend gap. Crypto trades through the weekend and gets nothing."""
    if now_utc.weekday() != 4:
        return None
    try:
        category = get_symbol_category(symbol)
    except Exception:
        return None
    if category.startswith("Crypto"):
        return None
    close_hour = (
        config.WEEKEND_FOREX_CLOSE_HOUR_UTC if category in ("Forex", "Exotics") else config.WEEKEND_OTHER_CLOSE_HOUR_UTC
    )
    minutes = close_hour * 60 - (now_utc.hour * 60 + now_utc.minute)
    if not 0 < minutes <= 180:
        return None
    return (
        f"the market closes for the weekend in about {minutes} min ({close_hour:02d}:00 UTC) — a position "
        "held past that is held over the weekend gap, consider closing or tightening now."
    )


def _apply_intraday_size_scalar(
    merged_allocation: dict[str, AllocationEntry],
    positions: list[Position],
    technical_context_cache: dict[str, tuple[FtmoAssetAnalysis, str]],
    now_utc: datetime,
    calendar_events: list | None,
    account_equity: float | None = None,
) -> dict[str, AllocationEntry]:
    """Deterministic, M5-based risk-% scaling for NEW entries (lot sizing on the
    decision tier; Python computes it, the pct Claude chose is the ceiling):

    - Volatility scalar = median M5 ATR% / current M5 ATR%, clamped to
      [config.INTRADAY_VOLATILITY_SCALAR_MIN, 1.0] — in an elevated-volatility tape
      the same stop distance is less safe, so risk % shrinks; a calm tape never
      sizes UP past what Claude chose.
    - Event run-up scalar = config.EVENT_RUNUP_SIZE_SCALAR when a High-impact event
      for the symbol's currencies lands within config.EVENT_RUNUP_HOURS (outside the
      hard blackout window, which already zeroed such entries).

    The scalar multiplies pct BEFORE compute_rebalance_plan turns pct into lots at the
    final stop distance.

    MINIMUM-LOT FLOOR (2026-09-25, XAGUSD on the ~$10k account): a scaled risk below what the broker's
    minimum lot needs at the entry's own stop distance does not produce a smaller position, it produces NO
    position ("can't afford even the minimum 0.01-lot"), silently killing a trade the Mega session and every
    guard had just approved. When the scaled pct falls under the minimum-lot risk but that minimum-lot risk
    still fits inside the pct Claude chose (its ceiling), the risk is raised to the minimum lot instead, and the
    reason says so. If even the minimum lot exceeds Claude's ceiling the trade stays infeasible, as before
    (config.SIZE_SCALAR_MIN_LOT_FLOOR=0 disables this)."""
    for symbol in _candidate_symbols(merged_allocation, positions):
        entry = merged_allocation[symbol]
        fetched = technical_context_cache.get(symbol)
        scalar = 1.0
        notes: list[str] = []
        if fetched is not None:
            analysis, _ = fetched
            current, median = analysis.m5_stats.atr_pct, analysis.m5_atr_pct_median
            if current and median and current > 0 and median > 0:
                vol_scalar = min(1.0, max(config.INTRADAY_VOLATILITY_SCALAR_MIN, median / current))
                if vol_scalar < 0.999:
                    scalar *= vol_scalar
                    notes.append(f"M5 ATR% {current:.3f} vs its median {median:.3f} -> x{vol_scalar:.2f}")
        if calendar_events:
            upcoming = economic_calendar.upcoming_events(
                symbol, now_utc, hours_ahead=config.EVENT_RUNUP_HOURS, events=calendar_events
            )
            if upcoming:
                scalar *= config.EVENT_RUNUP_SIZE_SCALAR
                notes.append(
                    f"{upcoming[0].currency} {upcoming[0].title} within {config.EVENT_RUNUP_HOURS:g}h -> "
                    f"x{config.EVENT_RUNUP_SIZE_SCALAR:.2f}"
                )
        if scalar >= 0.999:
            continue
        new_pct = entry.pct * scalar
        floor_note = ""
        if config.SIZE_SCALAR_MIN_LOT_FLOOR and account_equity and account_equity > 0 and fetched is not None:
            spec = fetched[0].base.contract_spec
            if spec is not None and entry.price is not None and entry.stop_loss is not None:
                min_lot_pct = spec.volume_min * abs(entry.price - entry.stop_loss) * spec.risk_per_price_unit / account_equity * 100
                if new_pct < min_lot_pct * 1.001 and min_lot_pct <= entry.pct:
                    floor_note = f" raised to the minimum-lot risk {min(entry.pct, min_lot_pct * 1.02):.4f}% (the scaled {new_pct:.4f}% would buy no position; still within the chosen {entry.pct:.4f}%)"
                    new_pct = min(entry.pct, min_lot_pct * 1.02)
        logger.info(
            "Intraday size scalar: %s risk %.4f%% -> %.4f%% (%s).%s", symbol, entry.pct, new_pct, "; ".join(notes), floor_note,
        )
        merged_allocation[symbol] = _with_changes(
            entry, pct=new_pct,
            reason=f"{entry.reason} [Intraday size scalar x{scalar:.2f}: {'; '.join(notes)}.{floor_note}]",
        )
    return merged_allocation


def _stabilize_resting_orders(
    merged_allocation: dict[str, AllocationEntry],
    positions: list[Position],
    technical_context_cache: dict[str, tuple[FtmoAssetAnalysis, str]],
    settled: dict,
) -> dict[str, AllocationEntry]:
    """Order-churn hysteresis. Every guard above is recomputed from live M5 data each
    poll, so the freshly computed price/stop/target/risk drift by small amounts poll to
    poll; compute_rebalance_plan treats anything beyond AMEND_TOLERANCE_PCT (0.05%) as an
    amend and would cancel-and-replace the resting order every few minutes (and reset its
    age). When an order is already resting for this symbol and every new value is within
    config.RESTING_ORDER_STABILITY_M5_ATR_MULTIPLE x the M5 ATR (risk % within
    config.RESTING_ORDER_PCT_STABILITY_FRACTION) of what is resting, keep the resting
    order's exact terms so the plan sees it as unchanged. A materially different plan
    (or a rejected one, pct 0) still goes through untouched."""
    for symbol in _candidate_symbols(merged_allocation, positions):
        record = settled.get(symbol) or {}
        if record.get("state") != "order_placed":
            continue
        prior = record.get("entry") or {}
        entry = merged_allocation[symbol]
        if prior.get("side") != entry.side:
            continue
        if normalise_entry_mode(prior.get("entry_mode")) != normalise_entry_mode(entry.entry_mode):
            continue  # a different order kind is a real change, never smoothed away
        p_price, p_stop, p_tp, p_pct = prior.get("price"), prior.get("stop_loss"), prior.get("take_profit"), prior.get("pct")
        if None in (p_price, p_stop, p_pct) or not p_pct or entry.price is None or entry.stop_loss is None:
            continue
        fetched = technical_context_cache.get(symbol)
        if fetched is None:
            continue
        analysis, _ = fetched
        atr = analysis.m5_stats.atr or analysis.h1_stats.atr
        if not atr or atr <= 0:
            continue
        tolerance = config.RESTING_ORDER_STABILITY_M5_ATR_MULTIPLE * atr
        tp_same = (entry.take_profit is None and p_tp is None) or (
            entry.take_profit is not None and p_tp is not None and abs(entry.take_profit - p_tp) <= tolerance
        )
        if (
            abs(entry.price - p_price) <= tolerance
            and abs(entry.stop_loss - p_stop) <= tolerance
            and tp_same
            and abs(entry.pct - p_pct) / p_pct <= config.RESTING_ORDER_PCT_STABILITY_FRACTION
        ):
            merged_allocation[symbol] = _with_changes(
                entry, pct=p_pct, price=p_price, stop_loss=p_stop, take_profit=p_tp
            )
    return merged_allocation


def _append_stale_pending_setup_cancels(
    plan: list, pending_orders: list[PendingOrder], settled: dict, now_utc: datetime, max_age_hours: float
) -> list:
    """compute_rebalance_plan deliberately never sees a Pending-Setup order once Clerk
    has placed it (its symbol is absent from the allocation on purpose — see that
    function's own invariant), so its age ceiling never applied to those orders: an
    M5-scale limit could rest for days. Cancels any such resting order older than
    `max_age_hours`; the cancel handler clears the settlement record, so the setup is
    re-checked fresh next poll. Symbols the plan already covers are left to it."""
    covered = {o.symbol for o in plan}
    by_symbol: dict[str, list[PendingOrder]] = {}
    for order in pending_orders or []:
        by_symbol.setdefault(order.symbol, []).append(order)
    for symbol, record in settled.items():
        if record.get("origin") != "pending_setup" or record.get("state") != "order_placed" or symbol in covered:
            continue
        orders = by_symbol.get(symbol, [])
        setup_times = [o.time_setup for o in orders if o.time_setup is not None]
        if not setup_times:
            continue
        oldest = min(setup_times)
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=timezone.utc)
        age_hours = (now_utc - oldest).total_seconds() / 3600
        if age_hours <= max_age_hours:
            continue
        side = "buy" if orders[0].order_type.startswith("buy") else "sell"
        plan.append(
            PlannedOrder(
                symbol=symbol, action="cancel", side=side, volume=sum(o.volume for o in orders),
                order_type="none", price=None, stop_loss=None,
                pending_tickets_to_cancel=[o.ticket for o in orders],
                reason=(
                    f"Cancelling: this confirmed pending-setup entry has rested unfilled for {age_hours:.1f}h, past the "
                    f"{max_age_hours:g}h intraday ceiling — an M5-scale entry is stale by now; the setup is "
                    "re-checked fresh next poll."
                ),
            )
        )
    return plan


def _apply_atr_stop_floor_guard(
    merged_allocation: dict[str, AllocationEntry],
    positions: list[Position],
    technical_context_cache: dict[str, tuple[FtmoAssetAnalysis, str]],
) -> dict[str, AllocationEntry]:
    """Real gap found on audit (2026-09-16): the Bulkowski 1.5x-H1-ATR
    stop-floor check already exists as a NAMED audit-critique category
    (ai.ftmo_suggest's own AUDIT_INSTRUCTION "Stops Systematically <
    1.5x ATR" flaw) but has never been enforced in code — only ever
    checked in prompt text, which an already-flagged draft can still get
    through anyway. Real incident this closes: a real EURUSD entry
    (2026-09-07) was independently audited at 0.57x H1 ATR against the
    1.5x floor, explicitly predicted in that same audit to be a "high
    probability of noise stop-out," executed anyway, and stopped out on
    ordinary noise exactly as predicted
    (records/ftmo/portfolio_suggestion_2026-09-07_122716.md:683-684).

    Same architectural slot as _apply_correlation_guard immediately
    above (a deterministic filter over merged_allocation before compute_
    rebalance_plan runs, not inside that function — it must stay pure/
    exchange-agnostic and has no per-symbol ATR data available to it).
    Candidates are symbols about to become NEW exposure this poll
    (pct > 0, not already an open position), same framing _apply_
    correlation_guard itself already uses via `positions`.

    For each candidate with both a real proposed price/stop_loss and a
    real H1 ATR available (technical_context_cache, already guaranteed
    to cover every pct>0 symbol by the fetch-loop that runs right before
    this is called): if the proposed |price - stop_loss| is tighter than
    config.ENTRY_ATR_STOP_FLOOR_MULTIPLE x H1 ATR, WIDENS the stop (on
    the correct side for entry.side — further below price for a buy,
    further above for a sell) to exactly meet the floor. Deliberately a
    widen, not a reject — mirrors MIN_STOP_DISTANCE_PCT's own existing
    precedent in risk/apply_suggestion.py: a trade whose direction/entry
    is otherwise sound shouldn't be discarded over a fixable stop
    distance. A symbol missing from the cache, with no H1 ATR yet, or
    missing a proposed price/stop_loss is skipped, never guessed at.

    Returns the same dict object, mutated in place, mirroring _apply_
    correlation_guard's own return contract exactly."""
    exposed_symbols = {p.symbol for p in positions}
    candidates = sorted(
        symbol for symbol, entry in merged_allocation.items()
        if symbol != "CASH" and entry.pct > 0 and symbol not in exposed_symbols
    )
    for symbol in candidates:
        entry = merged_allocation[symbol]
        if entry.price is None or entry.stop_loss is None:
            continue
        fetched = technical_context_cache.get(symbol)
        if fetched is None:
            continue
        analysis, _ = fetched
        # Decision-tier basis (2026-09-24): M5 ATR (H1 only when no M5 read exists), widened
        # to the spread / broker-minimum / pipeline-minimum floors — one shared definition
        # (ai.ftmo_suggest.effective_stop_floor) that the Mega Session sizing sheet also uses,
        # so the numbers Claude sizes against are the numbers this guard enforces.
        stop_floor = effective_stop_floor(analysis, entry.price)
        if stop_floor is None:
            continue
        floor_distance, atr_label = stop_floor.distance, stop_floor.atr_label
        atr = stop_floor.atr
        current_distance = abs(entry.price - entry.stop_loss)
        if current_distance >= floor_distance:
            continue

        widened_stop = (
            entry.price - floor_distance if entry.side == "buy" else entry.price + floor_distance
        )
        actual_multiple = current_distance / atr
        logger.info(
            "ATR stop-floor guard: %s stop %.5f (%.2fx %s ATR) is tighter than the %.2gx floor "
            "(effective floor %.5f) — widening to %.5f.",
            symbol, entry.stop_loss, actual_multiple, atr_label, stop_floor.multiple,
            floor_distance, widened_stop,
        )
        merged_allocation[symbol] = AllocationEntry(
            pct=entry.pct,
            price=entry.price,
            stop_loss=widened_stop,
            take_profit=entry.take_profit,
            side=entry.side,
            reason=(
                f"{entry.reason} [ATR stop-floor guard: stop was {actual_multiple:.2f}x {atr_label} ATR, "
                f"below the {config.ENTRY_ATR_STOP_FLOOR_MULTIPLE:.2g}x floor (or the spread/broker minimum) — widened "
                f"{entry.stop_loss:.5f} -> {widened_stop:.5f}.]"
            ),
            invalidation_condition=entry.invalidation_condition,
            entry_mode=entry.entry_mode,
        )
    return merged_allocation


def _side_relevant_win_rate(side: str, intraday: IntradayBacktests) -> float | None:
    """Real M5 win rate (0-100) for the side-matching backtest(s), only
    when backed by a genuinely large-enough resolved sample
    (config.MIN_RESOLVED_TRADES_FOR_WIN_RATE_FLOOR) — never a thin-sample
    number. A buy checks the oversold-RSI-long read and the support-bounce
    read (the two setups that bet LONG); a sell checks overbought-RSI-
    short and resistance-rejection (the two that bet SHORT) — see
    analysis.backtest's own RSIReactionBacktest/SupportResistanceBacktest
    docstrings for why each condition maps to that side.

    Uses the WORSE (min) of the two when both qualify — the conservative
    choice when two real, independent pieces of evidence disagree, not an
    average that could let one optimistic read outvote a poor one.

    None (never fabricated) when nothing qualifies — three real layers of
    "unknown," not one: the whole backtest object can be None (thin
    sample at analysis.backtest's own internal gate), `win_rate_pct`
    itself can be None even when the object exists (every simulated trade
    timed out — zero resolved, genuinely different from "0% winners"),
    or `intraday` itself can be the all-None empty default (intraday
    backtesting disabled, or M5 history came back empty for this
    symbol)."""
    if side == "buy":
        rsi_bt = intraday.rsi_oversold_backtest
        sr = intraday.support_resistance_backtest
        sr_win = sr.support_win_rate_pct if sr is not None else None
        sr_n = sr.support_tests if sr is not None else 0
    else:
        rsi_bt = intraday.rsi_overbought_backtest
        sr = intraday.support_resistance_backtest
        sr_win = sr.resistance_win_rate_pct if sr is not None else None
        sr_n = sr.resistance_tests if sr is not None else 0

    # A win-rate-derived (lower) R:R floor may only be earned by a setup the significance test calls
    # SUPPORTED against the random-entry baseline (analysis.edge_stats): a raw ~27% win rate is what
    # random entries score too, so it must never lower the bar. No verdict/baseline -> no lowering.
    verdicts = intraday_edge_verdicts(intraday, side)

    def _supported(key: str) -> bool:
        verdict = verdicts.get(key)
        return verdict is not None and verdict.verdict == EDGE_SUPPORTED

    candidates = []
    if (
        rsi_bt is not None and rsi_bt.win_rate_pct is not None
        and rsi_bt.trades >= config.MIN_RESOLVED_TRADES_FOR_WIN_RATE_FLOOR
        and _supported("rsi")
    ):
        candidates.append(rsi_bt.win_rate_pct)
    if sr_win is not None and sr_n >= config.MIN_RESOLVED_TRADES_FOR_WIN_RATE_FLOOR and _supported("sr"):
        candidates.append(sr_win)
    return min(candidates) if candidates else None


def _apply_reward_risk_floor_guard(
    merged_allocation: dict[str, AllocationEntry],
    positions: list[Position],
    technical_context_cache: dict[str, tuple[FtmoAssetAnalysis, str]] | None = None,
) -> dict[str, AllocationEntry]:
    """Real incident this closes (2026-09-17): a real INTC pending order
    came out at exactly 1:1 net reward:risk — traced to two genuinely
    CORRECT mechanisms colliding (INTC's own H1 ATR forcing a stop ~2x
    that ATR to survive its own noise; the account's own same-session
    realism ceiling capping the target at ~1.5-2x that SAME ATR), not a
    bug in either one. When both land on the same ATR multiple, ~1:1 is
    the best MATHEMATICALLY achievable ratio for that instrument right
    then — no cleverer entry/stop/target choice fixes it. The account's
    own instruction text (ai/ftmo_suggest.py:587-623) already names this
    exact situation and offers "size it smaller, treat it as lower-
    conviction, or leave it out" as equally-valid options, with no
    threshold saying when "leave it out" stops being optional — see
    config.MIN_NET_REWARD_RISK_RATIO's own comment for the real expected-
    value math (a 38%-win-rate setup, genuinely SUPPORTED by this
    account's own backtest at a 2:1 target, is NEGATIVE expected value at
    1:1) showing why "size it smaller" doesn't fix the sign of the
    problem, only its magnitude.

    Same architectural slot as _apply_atr_stop_floor_guard immediately
    above, run AFTER it deliberately: that guard may have just WIDENED
    this candidate's stop (a real risk increase), and this guard must
    judge the FINAL stop distance, not a pre-widening one that would
    understate how bad the ratio really is. Same candidate framing too
    (pct > 0, not already an open position) — an existing position's own
    tactical management is a completely different decision (adjusting
    risk already taken), never touched here.

    Unlike the stop-floor guard, this one REJECTS rather than resizes:
    there is no safe, mechanical adjustment available (tightening the
    stop reintroduces the noise-stop-out risk the ATR floor exists to
    prevent; extending the target breaks the same-session realism
    ceiling) — the honest response to a genuinely unfixable ratio is to
    not take the trade, exactly as ai/ftmo_suggest.py's own "leave it
    out" option already names, just enforced instead of left optional.

    Cost-netting is deliberately spread-only for this pass (TradeCost.
    spread_pct_of_price), not spread+commission — commission would also
    need get_contract_spec, a second fetch not otherwise needed at this
    call site; spread is the cost component that always applies and is
    already on the same TradeCost object. A real, disclosed MVP scoping,
    not an oversight — commission-inclusion is a cheap, valuable later
    refinement, not a blocker.

    Skips (never fabricates a verdict) a symbol missing price/stop_loss/
    take_profit, one whose real TradeCost isn't available (get_trade_
    economics returns None — real cost data is a prerequisite for a
    genuinely cost-netted check, same posture used everywhere else real
    cost feeds this account's math), or one with a zero/invalid risk
    distance.

    `technical_context_cache` (added 2026-09-18, defaulted to None so no
    existing caller/test breaks) makes the floor win-rate-aware instead
    of one flat number for every candidate: when this account's own real
    M5 intraday backtest evidence (see _side_relevant_win_rate) gives a
    genuinely reliable win rate for this candidate's own side, the
    required ratio becomes that win rate's OWN real breakeven point
    (with a safety margin), which can be LOWER than the flat default for
    a well-supported small-target setup — see config.MIN_RESOLVED_
    TRADES_FOR_WIN_RATE_FLOOR's own comment. Falls back to the flat
    config.MIN_NET_REWARD_RISK_RATIO whenever no cache is given, the
    symbol isn't in it, no reliable win rate is available, OR (fixed
    2026-09-21, real incident: a genuine MSFT double-bottom setup with
    its own cited 44% D1 win rate was rejected here purely because this
    same symbol's UNRELATED M5 scalp win rate read 23%) the win-rate-
    derived number would RAISE the requirement above the flat default —
    this M5 evidence is about one specific generic RSI-reaction/
    support-touch strategy, not necessarily the candidate's own real
    entry thesis, so it's only ever trusted to make the bar EASIER, never
    harder, than the flat default."""
    exposed_symbols = {p.symbol for p in positions}
    candidates = sorted(
        symbol for symbol, entry in merged_allocation.items()
        if symbol != "CASH" and entry.pct > 0 and symbol not in exposed_symbols
    )
    for symbol in candidates:
        entry = merged_allocation[symbol]
        if entry.price is None or entry.stop_loss is None or entry.take_profit is None:
            continue
        trade_cost = get_trade_economics(symbol)
        if trade_cost is None:
            continue
        risk = abs(entry.price - entry.stop_loss)
        if risk <= 0:
            continue
        reward = abs(entry.take_profit - entry.price)
        cost_distance = trade_cost.spread_pct_of_price / 100 * entry.price
        net_rr = (reward - cost_distance) / risk

        required_rr = config.MIN_NET_REWARD_RISK_RATIO
        floor_basis = f"flat {config.MIN_NET_REWARD_RISK_RATIO:g}:1 default"
        fetched = technical_context_cache.get(symbol) if technical_context_cache else None
        if fetched is not None:
            analysis, _ = fetched
            win_rate_pct = _side_relevant_win_rate(entry.side, analysis.intraday_backtests)
            if win_rate_pct is not None and 0 < win_rate_pct < 100:
                win_rate = win_rate_pct / 100.0
                breakeven_rr = (1 - win_rate) / win_rate
                win_rate_derived_rr = max(
                    config.MIN_ABSOLUTE_NET_REWARD_RISK_RATIO,
                    breakeven_rr * config.WIN_RATE_FLOOR_SAFETY_MARGIN,
                )
                # Real incident found live 2026-09-21, direct user report:
                # this M5 win rate reflects one specific, generic RSI-
                # reaction/support-touch scalp strategy — for a candidate
                # whose own real entry thesis is unrelated to that (e.g. a
                # D1-pattern-triggered full-session swing with its own,
                # far more relevant backtest already cited in its reason —
                # a real MSFT double-bottom setup with a genuine 44% D1
                # win rate got rejected here on an unrelated 23% M5 scalp
                # reading), letting an unrelated M5 sample RAISE the bar
                # rejects a trade on evidence that was never actually
                # about it. Only ever let this LOWER the requirement below
                # the flat default (the case this was designed for — a
                # well-evidenced M5 setup earns an easier bar) — never
                # raise it above the flat default from a mismatched or
                # poor M5 sample.
                if win_rate_derived_rr < required_rr:
                    required_rr = win_rate_derived_rr
                    floor_basis = f"win-rate-derived ({win_rate_pct:.0f}% real M5 win rate)"

        if "Stale-entry re-anchor" in (entry.reason or ""):
            required_rr += config.REANCHORED_ENTRY_RR_MARGIN
            floor_basis += f" +{config.REANCHORED_ENTRY_RR_MARGIN:g} re-anchor margin"

        if net_rr >= required_rr:
            continue

        logger.info(
            "Reward:risk floor guard: %s net R:R %.2f:1 is below the %.2f:1 floor (%s) — "
            "rejecting (pct %.4f%% -> 0%%), not downsizing.",
            symbol, net_rr, required_rr, floor_basis, entry.pct,
        )
        merged_allocation[symbol] = AllocationEntry(
            pct=0.0,
            price=entry.price,
            stop_loss=entry.stop_loss,
            take_profit=entry.take_profit,
            side=entry.side,
            reason=(
                f"{entry.reason} [reward:risk floor guard: net R:R {net_rr:.2f}:1 is below the "
                f"{required_rr:.2f}:1 floor ({floor_basis}) — rejected, not downsized.]"
            ),
            invalidation_condition=entry.invalidation_condition,
            entry_mode=entry.entry_mode,
        )
    return merged_allocation


def _export_chart_overlay(
    technical_context_cache: dict[str, tuple[FtmoAssetAnalysis, str]],
    merged_allocation: dict[str, AllocationEntry],
    positions_by_symbol: dict[str, Position],
    pending_orders: list[PendingOrder],
) -> None:
    """Writes this poll's own already-computed chart structure/levels
    and this poll's own final allocation decision to the overlay file
    mql5/Quant2ChartOverlay.mq5 draws — direct user request 2026-09-04:
    "I usually not read long prompts and reasoning, I am comfortable to
    read lines, levels, regions and some notes on the mt5 terminal
    screen." Reuses technical_context_cache exactly as this poll built
    it for its own real decisions, EXCEPT one small additional H1 price-
    history fetch per symbol added 2026-09-04 to build real regime-
    segment history (accumulation/distribution/trending/sideways
    vertical markers) — TechnicalStats/ChartStructureSnapshot never
    retain the raw H1 closes this needs, only derived scalar stats, so
    there was nothing already-cached this poll to reuse for it. Guarded
    per-symbol below so one symbol's fetch failure only costs that
    symbol's vertical markers, not the whole overlay export.

    Deliberately swallows every exception: this is a pure cosmetic/
    debugging side effect (see ai/chart_overlay.py's own docstring) that
    must never be able to interrupt or fail the real execution check
    that called it, regardless of what goes wrong (a locked file, a
    disconnected terminal mid-poll, anything)."""
    try:
        pending_by_symbol: dict[str, PendingOrder] = {}
        for order in pending_orders:
            pending_by_symbol.setdefault(order.symbol, order)

        lines_by_symbol: dict[str, list[str]] = {}
        for symbol, (analysis, _technical_context) in technical_context_cache.items():
            entry = merged_allocation.get(symbol)
            position = positions_by_symbol.get(symbol)

            # The ONE genuinely new fetch this export adds (see this
            # function's own docstring): TechnicalStats/ChartStructureSnapshot
            # never retain the raw H1 closes, only derived scalar stats, so
            # there's nothing already-cached this poll to reuse for a real
            # regime-segment HISTORY (as opposed to market_regime's own
            # single latest-window snapshot). Guarded per-symbol so one
            # symbol's fetch hiccup only blanks that symbol's vertical
            # markers, not every symbol's whole overlay this poll.
            regime_segments = regime_bar_count = None
            try:
                h1_closes = fetch_mt5_price_history(symbol, "H1")["Close"].dropna()
                regime_segments = compute_regime_segments(h1_closes)
                regime_bar_count = len(h1_closes)
            except Exception:
                logger.debug(
                    "Chart overlay: H1 regime-segment fetch failed for %s (cosmetic only, skipping vlines for it).",
                    symbol, exc_info=True,
                )

            inputs = OverlaySymbolInputs(
                symbol=symbol,
                h4_structure=analysis.h4_structure,
                h1_structure=analysis.h1_structure,
                h1_stats=analysis.h1_stats,
                position=position,
                pending_order=pending_by_symbol.get(symbol) if position is None else None,
                entry_price=entry.price if entry is not None else None,
                stop_loss=entry.stop_loss if entry is not None else None,
                take_profit=entry.take_profit if entry is not None else None,
                side=entry.side if entry is not None else None,
                note=entry.reason if entry is not None and entry.reason else None,
                h1_regime_segments=regime_segments,
                h1_bar_count=regime_bar_count,
            )
            lines_by_symbol[symbol] = build_overlay_lines(inputs)

        commondata_path = get_terminal_commondata_path()
        if commondata_path is not None:
            write_chart_overlay(lines_by_symbol, commondata_path)
    except Exception:
        logger.warning("Chart overlay export failed this poll (cosmetic only, continuing).", exc_info=True)


def run_clerk_execution_check(on_stage: Callable[[str], None] | None = None, fast: bool = False, think: bool = False) -> None:
    """The full execution-check orchestration — see this module's own docstring
    for the design this implements. Never raises for an anticipated
    failure mode (a missing suggestion, a blocked safety gate, an
    unreachable MT5 terminal all resolve to a logged, recorded outcome);
    a genuinely unexpected exception is left to propagate so the caller
    (clerk_execution_job.py, wrapped in run_with_timeout) can record
    it distinctly, matching ai.mega_analysis.run_scheduled_mega_analysis's
    own "never silently swallow a real failure" convention.

    `fast=True` is the FAST LANE (clerk_fast_job.py, every minute): the identical deterministic pipeline - reconcile, every guard,
    structured triggers, the rebalance plan, order execution, the deterministic circuit-breakers and profit trail - with NO model
    call (free-text pending triggers, invalidation checks and discretionary tactical verdicts stay with the full poll). It never
    writes the live progress file, never advances the full poll's review-interval marker and keeps the desk's headline status, so a
    full poll is neither delayed nor confused by it. Why it exists: one full poll (interval + a minutes-long local-model round)
    is a whole candle or more, and nothing about placing or amending an order needs a model.

    THINKING vs ACTING (config.CLERK_THINK_SPLIT_ENABLED, default on; see ai/clerk_thinking.py): with the split on, NO acting pass
    (this function without `think`) ever calls the model - it acts in seconds, using the freshest unused verdict the separate thinking
    pass stored. `think=True` is that thinking pass (clerk_think_job.py, its own lock): it builds the same lists, runs the model calls,
    stores the verdicts and returns without touching the settlement file, the journal, the desk status or any order."""
    quiet = fast or think  # no live-progress line, no headline status
    act_only = (fast or config.CLERK_THINK_SPLIT_ENABLED) and not think  # this pass never calls the model
    thinker_applied = 0  # verdicts an acting pass took from the thinking cache

    def _notify(message: str) -> None:
        logger.info(message)
        if not quiet:
            _write_execution_progress(message)
        if on_stage:
            on_stage(message)

    if not read_clerk_execution_enabled():
        _notify("Execution Clerk is disabled via the app's toggle — skipping this check.")
        if not quiet:
            _write_execution_state("disabled")
        return

    if _mega_session_is_live(read_mega_progress(), read_mega_state()):
        _notify("A mega analysis session is currently running — skipping this check.")
        if not quiet:
            _write_execution_state("skipped_mega_live")
        return

    suggestion = read_latest_suggestion()
    if not suggestion:
        _notify("No mega-analysis suggestion on file yet — nothing to check.")
        if not quiet:
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
            reason=s.get("reason", ""), trigger=normalise_trigger(s.get("trigger")),
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
    # One snapshot reused for every weekend-tradability check this poll
    # makes (the not-yet-settled trigger gate below, and the pre-weekend
    # cleanup widening further down) — see is_symbol_tradable_now's own
    # docstring.
    now_utc = datetime.now(timezone.utc)
    try:
        calendar_events = economic_calendar.fetch_calendar_events(now_utc)
    except Exception:
        logger.warning("Economic calendar unavailable this poll (event awareness disabled).", exc_info=True)
        calendar_events = []
    # First position found per symbol — same accepted "compare only the
    # first ticket" simplification risk/apply_suggestion.py::compute_
    # rebalance_plan's own amend-in-place path already documents for a
    # rare multi-ticket-per-symbol position.
    positions_by_symbol: dict[str, Position] = {}
    for p in positions:
        positions_by_symbol.setdefault(p.symbol, p)

    old_settled = settlement.get("settled", {})
    settlement["settled"] = _reconcile_settlement(old_settled, positions, pending_orders)
    newly_closed_symbols = _detect_newly_closed_symbols(old_settled, settlement["settled"])
    if newly_closed_symbols and config.CLERK_VAULT_JOURNAL_ENABLED and not think:
        _export_closed_trade_notes(newly_closed_symbols, old_settled)
    # Trade Journal (direct user request 2026-09-19): purely additive,
    # best-effort side effects run alongside the existing vault-note
    # export above — neither call can affect `settlement`, `positions`,
    # or anything this function returns; see ai.trade_journal's own
    # module docstring for the full "never puncture existing logic"
    # contract.
    for filled_symbol, filled_rec in ([] if think else _detect_newly_filled_symbols(old_settled, settlement["settled"])):
        trade_journal.record_filled(
            filled_symbol, filled_rec.get("order_ticket"), filled_rec.get("entry", {}).get("price")
        )
    if not think:
        _record_trade_journal_closures(newly_closed_symbols, old_settled)
        _reconcile_journal_unknown_closures(settlement["settled"])
    settlement["settled"] = _backfill_settlement_for_held_positions(
        settlement["settled"], positions, immediate_allocation_raw, account.equity, get_contract_spec,
    )
    # The Sentinel's (ai/sentinel.py) trailed stops become the recorded stops BEFORE the carried-forward target and the
    # drift check are built, so neither treats them as external interference. Best-effort: never blocks the poll.
    try:
        settlement["settled"], sentinel_notes = ingest_sentinel_stops(
            settlement["settled"], {p.symbol: p for p in positions}, read_sentinel_state().get("stops", {}),
            account.equity, get_contract_spec, pct_for_target_lots,
        )
        for note in sentinel_notes:
            logger.info("Sentinel: %s", note)
    except Exception:  # noqa: BLE001
        logger.warning("Could not ingest the Sentinel's stops - continuing without them.", exc_info=True)
    if not think:  # the thinking pass reads the settlement, never writes it
        _save_settlement(settlement)

    carried_forward = _build_carried_forward_allocation(
        immediate_allocation_raw, settlement["settled"], held_symbols=frozenset(positions_by_symbol)
    )

    # Drift detection/restore needs carried_forward already resolved (see
    # _iter_stop_drift's own docstring for why: a mismatch this app's own
    # fresh target already knows about isn't drift, it's just pending,
    # normal execution work) — must run AFTER the line above, never
    # before. Auto-RESTORES (not just warns) since 2026-09-15/16 — direct
    # user decision after a real XAGUSD incident where an unrestored,
    # only-ever-warned-about drift converted a break-even-managed
    # position into a real loss: "do not wait for any human intervention
    # just act and save the equity." See _restore_external_stop_drift's
    # own docstring for the full reasoning.
    # The restore MODIFIES (or even closes) a live position - an action, so never in the thinking pass.
    # ...and never against a CLOSED market or a symbol already in "Market closed" backoff: every attempt (a stop modify, then the
    # market-close fallback) is rejected by the broker, once per pass - 2026-09-26, XAGUSD on a Saturday, dozens of rejected orders.
    _drift_backoff = settlement.get("order_backoff", {})
    _drift_candidates = {
        s: p for s, p in positions_by_symbol.items()
        if is_symbol_tradable_now(s, now_utc) and _order_backoff_until(_drift_backoff, s, now_utc) is None
    }
    for drift_outcome in ([] if think else _restore_external_stop_drift(_drift_candidates, settlement["settled"], carried_forward)):
        # "STILL OPEN" marks the one genuinely dangerous residual case
        # (see _close_after_failed_stop_restore's own docstring): both
        # the restore AND the market-close fallback failed, so the
        # position is left running with neither fix landed — elevated
        # to error so this can't get lost among routine warning noise.
        if "STILL OPEN" in drift_outcome:
            logger.error("External stop-loss drift: %s", drift_outcome)
            _notify(f"URGENT — {drift_outcome}")
        else:
            logger.warning("External stop-loss drift: %s", drift_outcome)
            _notify(f"WARNING — {drift_outcome}")

    not_yet_settled = [s for s in pending_setups if s.symbol not in settlement["settled"]]
    if len(not_yet_settled) > config.CLERK_EXECUTION_MAX_PENDING_SETUPS:
        logger.warning(
            "%d not-yet-settled Pending Setups exceeds the cap of %d — checking only the first %d.",
            len(not_yet_settled), config.CLERK_EXECUTION_MAX_PENDING_SETUPS,
            config.CLERK_EXECUTION_MAX_PENDING_SETUPS,
        )
        not_yet_settled = not_yet_settled[: config.CLERK_EXECUTION_MAX_PENDING_SETUPS]

    # Already-settled (order_placed/filled) immediate_allocation symbols
    # carrying their own invalidation_condition — added 2026-08-23,
    # direct user request: the Clerk should watch the mega session's own
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
    if len(watched_positions) > config.CLERK_EXECUTION_MAX_WATCHED_POSITIONS:
        logger.warning(
            "%d watched positions exceeds the cap of %d — checking only the first %d.",
            len(watched_positions), config.CLERK_EXECUTION_MAX_WATCHED_POSITIONS,
            config.CLERK_EXECUTION_MAX_WATCHED_POSITIONS,
        )
        watched_positions = watched_positions[: config.CLERK_EXECUTION_MAX_WATCHED_POSITIONS]

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
    #
    # Fixed 2026-09-09 — real incident: this used to additionally require
    # settlement["settled"][symbol]["state"] == "filled", trusting this
    # app's OWN order-lifecycle bookkeeping as a proxy for "is this
    # symbol currently held." settlement tracks exactly ONE record per
    # symbol, keyed off whichever order this app most recently placed on
    # it — so a real, already-filled USDCAD short sitting alongside a
    # separate, still-resting "top-up" pending order on the SAME symbol
    # had its one settlement record stuck in "order_placed" (the pending
    # order's own state) for hours, even though a real position needing
    # tactical defense unambiguously existed. Confirmed live 2026-09-08:
    # the position ran to 86.6% of its own target intraday — a clean hit
    # of the Schwager partial-profit rule below — and got ZERO tactical
    # checks the whole time, purely because of this state mismatch, then
    # gave back roughly half the gain before anything could act on it.
    # `positions_by_symbol` is fetched fresh from live MT5 moments ago in
    # THIS SAME poll and is unambiguous ground truth for "is there a real
    # position to defend" — settlement's own state machine tracks a
    # different concern (this app's own order lifecycle) that can
    # legitimately diverge from it, so it's dropped from this gate
    # entirely rather than trusted as a proxy for it. Any symbol this
    # over-includes with no real, usable entry data still safely no-ops
    # downstream via `entry = carried_forward.get(symbol); if entry is
    # None: continue`.
    tactical_enabled = read_tactical_defense_enabled()
    tactical_candidates = list(positions_by_symbol)
    if len(tactical_candidates) > config.CLERK_EXECUTION_MAX_TACTICAL_CANDIDATES:
        logger.warning(
            "%d tactical-defense candidates exceeds the cap of %d — checking only the first %d.",
            len(tactical_candidates), config.CLERK_EXECUTION_MAX_TACTICAL_CANDIDATES,
            config.CLERK_EXECUTION_MAX_TACTICAL_CANDIDATES,
        )
        tactical_candidates = tactical_candidates[: config.CLERK_EXECUTION_MAX_TACTICAL_CANDIDATES]

    last_verdicts: dict = {}
    last_tactical_verdicts: dict = {}
    # Real gap found live 2026-09-16, direct user request: the AutoTrading-
    # off gate already surfaces a clear, visible "Execution blocked this
    # poll: ..." status the moment it fires — but a local Ollama server
    # being unreachable (confirmed live the same day: every pending-
    # setup/invalidation/tactical check that poll silently fell back to
    # its own fail-safe NOT_CONFIRMED/HOLD verdict, with the real cause
    # buried inside each individual verdict's own raw text rather than
    # surfaced as a poll-level status) had no equivalent. Set below, once
    # this poll's real LLM-bound results are in; carried through to both
    # the live _notify feed and this poll's own persisted last_detail
    # (see the final _write_execution_state call at the end of this
    # function) so it's visible as an ongoing status, not just a log line
    # that scrolls away.
    ollama_health_note = ""
    merged_allocation = dict(carried_forward)
    # Declared here (not inside the `if` below) so it's always a real
    # dict — possibly empty — by the time the chart-overlay export near
    # the end of this function reads it, regardless of whether this poll
    # had anything to check.
    technical_context_cache: dict[str, tuple[FtmoAssetAnalysis, str]] = {}
    # Moved out of the `if` below (2026-09-05, direct user report: "the
    # EA was working before... now nothing on the chart") — a symbol that
    # is only an IMMEDIATE ALLOCATION target (a resting pending order
    # like NVDA's, or a fresh target still awaiting feasibility sizing
    # like XAUUSD's) is never itself a not_yet_settled pending SETUP, a
    # watched POSITION, or a tactical candidate — none of the three lists
    # below ever include it — so its technical context, and therefore its
    # chart-overlay lines, silently vanished the moment its own mega-
    # session classification stopped being "pending setup" and became
    # "immediate allocation" instead. See the unconditional pass over
    # merged_allocation further down, right before _export_chart_overlay,
    # which now covers exactly this gap.
    # A symbol the Clerk only has to DEFEND (an open position with no new entry / pending setup on it) gets the LEAN analysis:
    # measured 3.5 s -> ~1 s per symbol, and it is fetched once per acting pass, every minute.
    _new_entry_symbols = {s.symbol for s in pending_setups}
    _lean_symbols = (
        {s for s in positions_by_symbol if s not in _new_entry_symbols} if config.CLERK_LEAN_ANALYSIS else set()
    )

    def _cached_technical_context(symbol: str) -> tuple[FtmoAssetAnalysis, str] | None:
        if symbol not in technical_context_cache:
            fetched = (
                _fetch_technical_context(symbol, market_prices, account.equity, lean=True)
                if symbol in _lean_symbols else _fetch_technical_context(symbol, market_prices, account.equity)
            )
            if fetched is None:
                return None
            technical_context_cache[symbol] = fetched
        return technical_context_cache[symbol]

    quick_lane_notes: list[str] = []  # what the deterministic gates decided this poll (shown in the desk's status line)
    llm_lane_counts = {"pending setups": 0, "invalidation checks": 0, "tactical checks": 0}
    if not_yet_settled or watched_positions or tactical_candidates:
        if not_yet_settled:
            _notify(f"Checking {len(not_yet_settled)} pending setup(s) against live technicals...")
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

        checkable: list[tuple[PendingSetup, str, str]] = []
        # Structured triggers (analysis/triggers.py) are evaluated HERE in Python on the completed M5 bars - no LLM call -
        # and their (setup, fired, reason) results join the merge loop first, in the same shape as an LLM verdict.
        trigger_results: list[tuple[PendingSetup, bool, str]] = []
        deterministic_invalidations: list[tuple[str, bool, str]] = []  # mechanical M5 invalidation lines, decided in Python
        for setup in not_yet_settled:
            # Real gap this closes, caught on a self-recheck: this is the
            # one Clerk decision point that places a genuinely NEW order
            # (every other candidate list below only manages something
            # that already exists) — deliberately in scope even though
            # Clerk's existing-position management stays untouched. Its
            # tactical prompt deliberately excludes market-status (see
            # _fetch_technical_context's own include_market_status
            # docstring), so the local model triggering this off stale,
            # frozen weekend data would have zero way to know the market
            # is actually closed — it would just confirm the trigger
            # normally, place a real resting order, and rely on the
            # pre-weekend-cleanup backstop below to catch it, at best a
            # full poll late. A deterministic pre-check is cheap and
            # removes that whole window: skip the LLM call entirely and
            # recheck next poll, same reasoning as the hard-exit
            # circuit-breaker's "no exceptions, no model call" pattern.
            if not is_symbol_tradable_now(setup.symbol, now_utc):
                logger.info("%s market is closed right now — skipping pending-setup trigger check.", setup.symbol)
                continue
            fetched = _cached_technical_context(setup.symbol)
            if fetched is None:
                logger.warning("%s is not currently visible in Market Watch — skipped.", setup.symbol)
                continue
            trigger_analysis, technical_context = fetched
            technical_context = _context_for_condition(trigger_analysis, technical_context, setup.trigger_condition, account.equity)
            if setup.trigger is not None:
                quote = market_prices[setup.symbol]
                outcome = evaluate_trigger(
                    setup.trigger, setup.side, trigger_analysis.m5_recent, quote.bid, quote.ask,
                    trigger_analysis.m5_stats.atr if trigger_analysis.m5_stats is not None else None,
                )
                trigger_results.append((setup, outcome.fired, f"[structured trigger] {outcome.reason}"))
                continue
            checkable.append((setup, technical_context, market_prices[setup.symbol].description))

        checkable_watched: list[tuple[str, dict, str, str, str]] = []
        for symbol, raw, invalidation_condition, state in watched_positions:
            fetched = _cached_technical_context(symbol)
            if fetched is None:
                logger.warning("%s (watched) is not currently visible in Market Watch — skipped.", symbol)
                continue
            watched_analysis, technical_context = fetched
            mechanical = deterministic_invalidation(raw, invalidation_condition, watched_analysis)
            if mechanical is not None:
                deterministic_invalidations.append((symbol, mechanical[0], mechanical[1]))
                continue
            technical_context = _context_for_condition(watched_analysis, technical_context, invalidation_condition, account.equity)
            checkable_watched.append((symbol, raw, invalidation_condition, state, technical_context))

        checkable_tactical: list[tuple[str, AllocationEntry, Position, str, dict | None, TacticalSignals]] = []
        for symbol in tactical_candidates:
            fetched = _cached_technical_context(symbol)
            if fetched is None:
                logger.warning("%s (tactical) is not currently visible in Market Watch — skipped.", symbol)
                continue
            analysis, technical_context = fetched
            entry = carried_forward.get(symbol)
            if entry is None:
                continue
            prior_tactical = settlement["settled"].get(symbol, {}).get("tactical")
            position = positions_by_symbol[symbol]
            signals = _compute_tactical_signals(
                position, entry, analysis.h1_stats, analysis.h4_stats, analysis.h1_structure, analysis.base,
                prior_tactical, trade_cost=analysis.trade_cost,
                m5_stats=analysis.m5_stats, m5_structure=analysis.m5_structure,
                m5_atr_pct_median=analysis.m5_atr_pct_median,
                intraday_backtests=analysis.intraday_backtests,
                event_note=" ".join(
                    n for n in (_tactical_event_note(symbol, now_utc, calendar_events), _weekly_close_note(symbol, now_utc)) if n
                ) or None,
            )
            checkable_tactical.append((symbol, entry, position, technical_context, prior_tactical, signals))

        # Added 2026-09-10 — lets the merge loop below pass this
        # symbol's own already-computed TacticalSignals into _validate_
        # and_apply_tactical_verdict's new ATR-stop fallback (see that
        # function's own docstring) without changing _run_clerk_tactical_
        # check's own (symbol, verdict, raw_text) return contract, which
        # app.py's identical-shaped invalidation-check results already
        # share the same isinstance(result[1], TacticalVerdict) dispatch
        # against.
        tactical_signals_by_symbol = {s: sig for s, _e, _p, _tc, _pt, sig in checkable_tactical}
        llm_lane_counts["pending setups"] = len(checkable)
        llm_lane_counts["invalidation checks"] = len(checkable_watched)
        llm_lane_counts["tactical checks"] = len(checkable_tactical)
        for _setup, _fired, _text in trigger_results:
            quick_lane_notes.append(
                f"{_setup.symbol} {'trigger FIRED' if _fired else 'trigger not fired'} ({_text.replace('[structured trigger] ', '')})"
            )

        # Phase 2 (submitted in parallel, LLM-bound): pure local HTTP
        # calls, no MT5 involvement. Known limitation as of 2026-08-30
        # (see this module's own docstring for the Ollama swap): unlike
        # the old Copilot-CLI/OpenRouter chain, both local models run on
        # ONE shared Ollama server, which serializes generation requests
        # for the same loaded model — this ThreadPoolExecutor submission
        # pattern is kept for now (unchanged shape from before the
        # swap), but real wall-clock time for N candidates in one poll
        # scales roughly with N, not with the slowest single call, the
        # way it did against genuinely concurrent cloud calls. Revisit
        # if this ever bumps into CLERK_EXECUTION_RUN_TIMEOUT_SECONDS in
        # practice — see this module's own docstring, "capacity
        # building" is a deliberately separate, later phase.
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
        tactical_fingerprints: dict[str, str] = {}
        if checkable or checkable_watched or checkable_tactical or trigger_results or deterministic_invalidations:
            elapsed_description = _describe_elapsed(generated_utc, datetime.now(timezone.utc))
            if act_only:
                # An acting pass NEVER waits for the model: deterministic breakers fire now, and the freshest unused thinking verdict
                # (if any) is taken from the cache the thinking job keeps filled.
                cached_results, thinker_applied = _cached_thinking_results(
                    checkable, checkable_watched, checkable_tactical, account.equity, generated_utc
                )
                results = trigger_results + deterministic_invalidations + cached_results
            elif config.CLERK_LLM_SEQUENTIAL:
                # One model call at a time, most urgent first (an open position's tactical review, then invalidation checks, then
                # pending triggers). The local Ollama server runs ONE generation at a time anyway, so submitting them all at once
                # bought no speed and made every request wait in its queue against the same per-call timeout: the later ones timed
                # out (300 s) and fell back to the backup model - the multi-minute polls of 2026-09-25/26.
                think_cache = clerk_thinking.for_suggestion(clerk_thinking.load_cache(), generated_utc) if think else {}
                think_now = datetime.now(timezone.utc)
                tactical_done = []
                for symbol, entry, position, technical_context, prior_tactical, signals in checkable_tactical:
                    tactical_fingerprints[symbol] = _tactical_fingerprint(position, signals)
                    if think and config.CLERK_THINK_REUSE_MAX_MINUTES > 0 and _tactical_unchanged(
                        think_cache, symbol, tactical_fingerprints[symbol], think_now
                    ):
                        logger.info("%s: nothing moved since the last HOLD - not asking the model again.", symbol)
                        tactical_done.append((symbol, TacticalVerdict(tier="hold"), _REUSED_RAW))
                        continue
                    tactical_done.append(
                        _run_clerk_tactical_check(symbol, entry, position, technical_context, account.equity, prior_tactical, signals)
                    )
                invalidation_done = [
                    _run_clerk_invalidation_check(
                        symbol, raw, cond, technical_context, account.equity, elapsed_description, state
                    )
                    for symbol, raw, cond, state, technical_context in checkable_watched
                ]
                pending_done = [
                    _run_clerk_verdict(setup, technical_context, account.equity, elapsed_description, description)
                    for setup, technical_context, description in checkable
                ]
                results = trigger_results + deterministic_invalidations + pending_done + invalidation_done + tactical_done
            else:
                with ThreadPoolExecutor(
                    max_workers=max(1, len(checkable) + len(checkable_watched) + len(checkable_tactical))
                ) as pool:
                    futures = [
                        pool.submit(
                            _run_clerk_verdict, setup, technical_context, account.equity, elapsed_description, description
                        )
                        for setup, technical_context, description in checkable
                    ] + [
                        pool.submit(
                            _run_clerk_invalidation_check, symbol, raw, cond, technical_context,
                            account.equity, elapsed_description, state,
                        )
                        for symbol, raw, cond, state, technical_context in checkable_watched
                    ] + [
                        pool.submit(
                            _run_clerk_tactical_check, symbol, entry, position, technical_context,
                            account.equity, prior_tactical, signals,
                        )
                        for symbol, entry, position, technical_context, prior_tactical, signals in checkable_tactical
                    ]
                    results = trigger_results + deterministic_invalidations + [f.result() for f in futures]

            ollama_health_note = _detect_ollama_outage(results)
            if ollama_health_note:
                logger.warning(ollama_health_note)
                _notify(f"WARNING — {ollama_health_note}")
            if think:
                _store_thinking_results(results, generated_utc, tactical_fingerprints)
                _notify(f"Thinking pass finished: {len(results)} verdict(s) stored for the acting passes.")
                return
            # Merged on the main thread, in original list order, after
            # every future resolves — never inside a worker thread — so
            # completion order can never make this non-deterministic.
            # Discriminated by result[0]'s type: PendingSetup for a
            # trigger-check; for the remaining two kinds, both return a
            # bare symbol string in position 0 (see _run_clerk_
            # invalidation_check's and _run_clerk_tactical_check's own
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
                    # Trade Journal (direct user request 2026-09-19):
                    # purely additive, best-effort — see this call's own
                    # sibling below for the full rationale.
                    trade_journal.record_clerk_check(setup.symbol, "pending_setup_trigger", confirmed, raw)
                    # The full reasoning text is logged here too (direct
                    # user request 2026-08-23: "also write this in a log
                    # somewhere") — previously it only ever landed in
                    # clerk_execution_state.json's last_verdicts, never
                    # in the actual human-readable log file.
                    logger.info(
                        "Clerk verdict for %s: %s — %s", setup.symbol,
                        "CONFIRMED" if confirmed else "NOT_CONFIRMED",
                        raw[:1000] if raw else "(no response text)",
                    )
                    if confirmed:
                        merged_allocation[setup.symbol] = AllocationEntry(
                            # A fired trigger's plan price is the TRIGGER LEVEL (what the entry guard measures the market
                            # entry's slippage from), not the model's estimate of the price at trigger time.
                            pct=setup.pct, price=setup.trigger["level"] if setup.trigger is not None else setup.price,
                            stop_loss=setup.stop_loss,
                            take_profit=setup.take_profit, side=setup.side, reason=setup.reason,
                            # A fired structured trigger means "go now with the trend": the market kind still passes every
                            # entry-mode guard (caps, extension, stale re-anchor, kill switch) further down this poll.
                            entry_mode="market" if setup.trigger is not None else "limit",
                        )
                elif isinstance(result[1], TacticalVerdict):
                    symbol, verdict, raw = result
                    if raw == _FAST_LANE_RAW:
                        continue  # nothing fired and no thinking verdict yet: never overwrite the last real verdict with a placeholder
                    logger.info(
                        "Clerk tactical check for %s: %s — %s", symbol, verdict.tier.upper(),
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
                            signals=tactical_signals_by_symbol.get(symbol),
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
                        # verdict.hard_exit bypasses the shadow-mode toggle
                        # entirely — it's O'Neil's "no exceptions" hard
                        # stop-loss ceiling, a genuine circuit-breaker (see
                        # TacticalVerdict.hard_exit's own docstring), not a
                        # discretionary DEFEND/EXIT waiting on the same
                        # rollout gate as every other tactical verdict.
                        if tactical_enabled or verdict.hard_exit:
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
                        "shadow_mode": not tactical_enabled and not verdict.hard_exit,
                        "skipped_reason": skipped_reason,
                    }
                    # Trade Journal (direct user request 2026-09-19):
                    # purely additive, best-effort — records the SAME
                    # decision already reflected above, never influences
                    # `applied`/`merged_allocation`/`settlement` itself.
                    trade_journal.record_tactical_action(symbol, verdict.tier, applied, skipped_reason, raw)
                else:
                    symbol, confirmed, raw = result
                    last_verdicts[symbol] = {
                        "confirmed": confirmed, "raw_text": raw,
                        "checked_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    trade_journal.record_clerk_check(symbol, "invalidation", confirmed, raw)
                    logger.info(
                        "Clerk invalidation check for %s: %s — %s", symbol,
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

    if think:  # nothing needed a model this pass: it still counts as completed
        _store_thinking_results([], generated_utc)
        _notify("Thinking pass finished: nothing needed a model.")
        return

    merged_allocation = _apply_correlation_guard(merged_allocation, positions)

    # Fills the real gap _cached_technical_context's own comment above
    # describes: every symbol genuinely part of the FINAL decided mix
    # (a real, nonzero target — held, pending, or still just proposed and
    # awaiting a feasibility/sizing pass) gets its technical context
    # fetched for the chart overlay even when nothing about it needed an
    # LLM check this poll. A no-op for anything already cached above
    # (not_yet_settled/watched/tactical) — _cached_technical_context's
    # own cache check skips those. Swallows a missing symbol the same way
    # every caller above already does (fetched is None -> skip), since
    # this is purely cosmetic, same contract as _export_chart_overlay
    # itself.
    for _symbol, _entry in merged_allocation.items():
        if _entry.pct > 0:
            _cached_technical_context(_symbol)

    # Runs AFTER the fetch-loop above, never before: every pct>0 symbol's
    # fresh H1 ATR is only guaranteed to be in technical_context_cache
    # once that loop has run. Intraday entry refinement runs FIRST in
    # this trio — see its own docstring for why: the ATR-stop-floor
    # guard must validate the FINAL (possibly just-refined) stop
    # distance, not a pre-refinement one.
    pct_pre_guard = {symbol: entry.pct for symbol, entry in merged_allocation.items()}
    merged_allocation = _apply_invalidation_guard(merged_allocation, positions, market_prices)
    merged_allocation = _apply_event_blackout_guard(merged_allocation, positions, now_utc, calendar_events)
    guard_cooldowns = settlement.setdefault("guard_cooldowns", {})
    merged_allocation = _apply_guard_cooldown(merged_allocation, positions, guard_cooldowns, now_utc)
    merged_allocation = _apply_pre_close_guard(
        merged_allocation, positions, pending_orders, now_utc, _broker_clock_offset(sorted(market_prices))
    )
    pct_before_guards = {symbol: entry.pct for symbol, entry in merged_allocation.items()}
    today_key = now_utc.strftime("%Y-%m-%d")
    market_entries = settlement.setdefault("market_entries", {})
    merged_allocation, entry_deviation_caps = _apply_entry_mode_guard(
        merged_allocation, positions, pending_orders, technical_context_cache, market_prices,
        {symbol for symbol, day in market_entries.items() if day == today_key},
    )
    entry_mode_rejected = {
        symbol for symbol, before in pct_before_guards.items()
        if before > 0 and merged_allocation.get(symbol) is not None and merged_allocation[symbol].pct <= 0
    }
    merged_allocation = _apply_stale_entry_reanchor(merged_allocation, positions, technical_context_cache)
    merged_allocation = _apply_atr_stop_floor_guard(merged_allocation, positions, technical_context_cache)
    merged_allocation = _apply_reward_risk_floor_guard(merged_allocation, positions, technical_context_cache)
    if _record_guard_rejections(
        pct_before_guards, merged_allocation, positions, guard_cooldowns, now_utc, skip=entry_mode_rejected
    ):
        _save_settlement(settlement)
    guard_blocks = _guard_blocks_this_poll(pct_pre_guard, merged_allocation, positions, guard_cooldowns)
    merged_allocation = _apply_intraday_size_scalar(
        merged_allocation, positions, technical_context_cache, now_utc, calendar_events, account_equity=account.equity
    )
    merged_allocation = _stabilize_resting_orders(
        merged_allocation, positions, technical_context_cache, settlement["settled"]
    )

    _export_chart_overlay(technical_context_cache, merged_allocation, positions_by_symbol, pending_orders)

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
    # Real gap caught on a self-recheck (2026-09-12): checking the heat
    # gate here, together with the two fundamental ones below, meant a
    # heat-blocked poll returned BEFORE compute_rebalance_plan ever ran
    # — silently preventing the pre-weekend cleanup (or any other pure
    # risk-REDUCING cancel) from ever executing on exactly the days
    # aggregate heat is already a problem, which is precisely when
    # freeing that risk-budget matters most. The two fundamental gates
    # (is_demo/allow_live_execution, trading_permitted) still block
    # EVERYTHING unconditionally here — neither means it's even safe to
    # send AN order to this account at all, cancels included. The heat
    # gate is instead checked per-order in the execution loop below, so
    # a pure "cancel" (never "amend_pending", which re-opens a new order
    # right after) can still get through even while new-risk-taking
    # stays blocked.
    fundamental_ok, fundamental_block_reason = check_execution_safety_gates(
        is_demo=is_demo,
        allow_live_execution=config.ALLOW_LIVE_EXECUTION,
        trading_permitted=trading_permitted,
        trading_blocked_reason=trading_blocked_reason,
        ftmo_heat_blocked=False,
        ftmo_heat_blocked_reason="",
    )
    if not fundamental_ok:
        _notify(f"Execution blocked this poll: {fundamental_block_reason}")
        # Folds in ollama_health_note too (see its own comment above) —
        # a real, live-observed combination: AutoTrading being off in
        # the terminal and Ollama being unreachable are two entirely
        # independent problems, and this account hit both at once. Only
        # ever surfacing the execution-gate message in that case would
        # have hidden the second, equally real issue.
        blocked_detail = fundamental_block_reason
        if ollama_health_note:
            blocked_detail += f" | WARNING: {ollama_health_note}"
        if fast:
            return  # logged above; the full poll records the blocked status
        _write_execution_state(
            "blocked", blocked_detail, last_verdicts=last_verdicts, last_tactical_verdicts=last_tactical_verdicts,
        )
        _mark_interval_ran(datetime.now(timezone.utc))
        return

    # `friday_pre_weekend_due`: the original, proactive Friday-hour
    # trigger, unchanged. `is_weekend_now`: a small, additive backstop —
    # without it, a fresh non-crypto pending setup created mid-weekend
    # (e.g. the mega session re-run on a Saturday) wouldn't be caught
    # until the FOLLOWING Friday, the same reasoning already applied to
    # the hard-exit circuit-breaker and the age-ceiling cancel: don't
    # rely solely on correct upstream behavior (here, the mega session's
    # own new market-open awareness — see ai.ftmo_suggest's "CHECK
    # WHETHER THIS INSTRUMENT'S MARKET IS EVEN OPEN RIGHT NOW"
    # instruction) when a deterministic backstop is this cheap to add.
    friday_pre_weekend_due = is_pre_weekend_cleanup_due(now_utc)
    is_weekend_now = now_utc.weekday() in (5, 6)
    pre_weekend_due = friday_pre_weekend_due or is_weekend_now
    weekend_tradable_symbols: set[str] = set()
    if pre_weekend_due:
        pending_symbols = {o.symbol for o in (pending_orders or [])}
        if friday_pre_weekend_due:
            # Proactive: fires BEFORE the actual weekly close (a
            # deliberate buffer — see CLERK_PRE_WEEKEND_CLEANUP_HOUR_UTC's
            # own comment), so it can't check "is this tradable right
            # now" — forex/other categories are still technically open
            # at this hour. Deliberately comprehensive instead: every
            # non-crypto pending order gets cancelled regardless of its
            # own exact close time, matching this feature's original
            # design intent. Cheap, symbol_info()-only lookup (no tick
            # fetch) — see get_symbol_category's own docstring.
            weekend_tradable_symbols = {
                s for s in pending_symbols if get_symbol_category(s).startswith("Crypto")
            }
        else:
            # Reactive Sat/Sun backstop: must reflect what's genuinely
            # tradable RIGHT NOW, not a blunt crypto-only rule — forex
            # reopens Sunday evening UTC, before every other category.
            # Real bug this fixes, caught on a self-recheck: a
            # crypto-only rule here would keep cancelling a legitimate
            # forex pending order the mega session correctly proposes
            # AFTER forex has already reopened, directly contradicting
            # the "mega session should suggest based on available
            # assets" design goal this whole feature exists for.
            weekend_tradable_symbols = {
                s for s in pending_symbols if is_symbol_tradable_now(s, now_utc)
            }

    broker_now = now_utc + _broker_clock_offset(sorted(market_prices))
    deliberate_resize = _pending_partial_symbols(settlement["settled"], positions_by_symbol, merged_allocation)
    plan = compute_rebalance_plan(
        positions, account, merged_allocation, get_contract_spec, market_prices,
        price_sanity_band_pct=config.PRICE_SANITY_BAND_PCT,
        pending_orders=pending_orders,
        amend_tolerance_pct=config.AMEND_TOLERANCE_PCT,
        max_pending_order_age_hours=config.INTRADAY_PENDING_ORDER_MAX_AGE_HOURS,
        min_stop_distance_pct=config.MIN_STOP_DISTANCE_PCT,
        held_position_size_tolerance_pct=config.HELD_POSITION_SIZE_TOLERANCE_PCT,
        is_pre_weekend=pre_weekend_due,
        weekend_tradable_symbols=weekend_tradable_symbols,
        deliberate_resize_symbols=deliberate_resize,
        max_position_margin_pct_of_equity=config.MAX_POSITION_MARGIN_PCT_OF_EQUITY,
        now=broker_now,
    )
    plan = _append_stale_pending_setup_cancels(
        plan, pending_orders, settlement["settled"], broker_now, config.INTRADAY_PENDING_ORDER_MAX_AGE_HOURS
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
        if action == "hold" and symbol in guard_blocks:
            # Nothing was sent because a guard held the target at 0% - say so, instead of "on target, no change needed".
            results_this_poll[symbol]["detail"] = f"held back by a guard: {guard_blocks[symbol]}"
            results_this_poll[symbol]["guard_blocked"] = True

    order_backoff = settlement.setdefault("order_backoff", {})
    for o in plan:
        deferred_until = _order_backoff_until(order_backoff, o.symbol, now_utc)
        if deferred_until is not None and o.action != "infeasible":
            logger.info(
                "%s (%s): deferred — the market was closed on the last attempt; not retrying before %s UTC.",
                o.symbol, o.action, deferred_until.strftime("%H:%M"),
            )
            _record_result(o.symbol, o.action, False, f"deferred until {deferred_until.strftime('%H:%M')} UTC (market closed)")
            continue
        if ftmo_heat_blocked and o.action != "cancel":
            # See the fundamental/heat gate split above — only a pure
            # cancel (real risk reduction, never new/maintained exposure)
            # is allowed through while heat-blocked.
            logger.info(
                "%s (%s): skipped — heat-blocked this poll (%s); only a pure cancel can execute while blocked.",
                o.symbol, o.action, ftmo_heat_blocked_reason,
            )
            _record_result(o.symbol, o.action, False, f"heat-blocked: {ftmo_heat_blocked_reason}")
            continue
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
                result = open_position(
                    o.symbol, o.side, o.volume, o.price, o.stop_loss, o.take_profit,
                    kind=o.order_type, max_deviation_price=entry_deviation_caps.get(o.symbol),
                    expiration_hours=(
                        _resting_order_expiry_hours(o.symbol, now_utc, _broker_clock_offset([o.symbol]))
                        if o.order_type in ("limit", "stop") else None
                    ),
                )
            except MT5ConnectionError as e:
                result = OrderResult(False, None, str(e), None)
            if result.success and o.order_type == "market":
                market_entries[o.symbol] = today_key
                _save_settlement(settlement)
            logger.info(
                "%s (%s %s): %s", o.symbol, o.action, o.order_type, "placed" if result.success else f"failed ({result.comment})"
            )
            _learn_order_outcome(order_backoff, o.symbol, result, now_utc)
            _record_result(o.symbol, o.action, result.success, "placed" if result.success else result.comment)
            # Trade Journal (direct user request 2026-09-19): purely
            # additive, best-effort — record_order_result swallows its
            # own exceptions and never raises, and only ever appends to
            # a story ai.ftmo_suggest.py already created; it can't
            # affect `result`, `settlement`, or anything below.
            trade_journal.record_order_result(
                o.symbol, o.action, result.success, "placed" if result.success else (result.comment or ""),
                terms={
                    "price": o.price, "stop_loss": o.stop_loss, "take_profit": o.take_profit, "volume": o.volume,
                    # Which guards changed Claude's numbers before this order went out (re-anchor, stop
                    # floor, size scalar, ...) — the journal note then explains why the numbers are what
                    # they are, not just what they are.
                    "guards": _GUARD_NOTE_RE.findall(merged_allocation[o.symbol].reason or ""),
                },
            )
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
                            "entry_mode": merged_allocation[o.symbol].entry_mode,
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
                _learn_order_outcome(order_backoff, o.symbol, result, now_utc)
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
                _learn_order_outcome(order_backoff, o.symbol, result, now_utc)
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
                _learn_order_outcome(order_backoff, o.symbol, result, now_utc)
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
                    result = open_position(
                        o.symbol, o.side, o.volume, o.price, o.stop_loss, o.take_profit,
                        kind=o.order_type, max_deviation_price=entry_deviation_caps.get(o.symbol),
                        expiration_hours=(
                            _resting_order_expiry_hours(o.symbol, now_utc, _broker_clock_offset([o.symbol]))
                            if o.order_type in ("limit", "stop") else None
                        ),
                    )
                except MT5ConnectionError as e:
                    result = OrderResult(False, None, str(e), None)
                if result.success and o.order_type == "market":
                    market_entries[o.symbol] = today_key
                    _save_settlement(settlement)
                logger.info(
                    "%s (amend_pending %s): %s", o.symbol, o.order_type, "replaced" if result.success else f"failed ({result.comment})"
                )
                _learn_order_outcome(order_backoff, o.symbol, result, now_utc)
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
                            "entry_mode": merged_allocation[o.symbol].entry_mode,
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
                _learn_order_outcome(order_backoff, o.symbol, result, now_utc)  # a "Market closed" answer defers the next attempts
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
            zero_target = merged_allocation.get(o.symbol)
            if o.symbol not in guard_blocks and zero_target is not None and zero_target.pct <= 0 and o.symbol != "CASH":
                _record_result(o.symbol, o.action, True, "target is 0% - nothing to place or hold")
            else:
                _record_result(o.symbol, o.action, True, "on target, no change needed")

    _save_settlement(settlement)
    _notify(f"Execution-check complete — {executed_count} order(s) sent this poll.")
    # ollama_health_note (set earlier this poll, see its own comment) is
    # folded into the PERSISTED status here, not just the transient
    # notify feed above — otherwise it would be invisible the moment this
    # poll's own "success" detail below overwrote it, exactly the gap
    # that made an unreachable Ollama server look like ordinary
    # NOT_CONFIRMED/HOLD activity instead of a real, ongoing outage.
    success_detail = f"{executed_count} order(s) sent"
    for _symbol, _why in sorted(guard_blocks.items()):
        quick_lane_notes.append(f"{_symbol} held back ({_why})")
    if quick_lane_notes:
        success_detail += " | quick gates: " + "; ".join(quick_lane_notes)[:900]
    llm_total = sum(llm_lane_counts.values())
    if act_only:
        success_detail += (
            f" | thinking runs separately: {thinker_applied} thinker verdict(s) applied this pass"
            if thinker_applied else " | thinking runs separately: no new thinker verdict to apply"
        )
    elif llm_total:
        success_detail += " | LLM: " + ", ".join(f"{n} {label}" for label, n in llm_lane_counts.items() if n)
    else:
        success_detail += " | LLM: nothing needed a model this poll (no free-text pending setup, no open position to review)"
    if ollama_health_note:
        success_detail += f" — WARNING: {ollama_health_note}"
    if fast:
        # Keep the desk's headline (the last FULL poll's status/detail/time) and its review countdown; only merge this pass's
        # per-symbol results, which is what shows an order placed / a guard hold between full polls.
        _write_execution_state(
            "success", success_detail, last_verdicts=last_verdicts, last_execution_results=results_this_poll,
            last_tactical_verdicts=last_tactical_verdicts, preserve_headline=True,
        )
        return
    _write_execution_state(
        "success", success_detail,
        last_verdicts=last_verdicts, last_execution_results=results_this_poll,
        last_tactical_verdicts=last_tactical_verdicts,
    )
    _mark_interval_ran(datetime.now(timezone.utc))
