"""The "curiosity function" — a mandatory, every-FTMO-mega-session
retrospective self-critique of this account's own recent CLOSED trades.

Direct user request 2026-09-05: an evolution of the existing past-
lessons mechanism (ai/portfolio_suggest.py's build_past_audit_lessons/
build_past_outcome_lessons). Those two only ever hand Claude either a
past AUDIT's own critique, or a bare price-delta since a past
suggestion — neither interrogates a SPECIFIC settled trade the way the
user described: looking at the real price action a few hours before/
after it, comparing that against the original thesis, and asking a
genuine "why did we do X instead of Y." The user explicitly wants the
heavy forensic reasoning delegated to a free OpenRouter model (not
Claude), so Claude only ever reads a short "report card" — where the
assumption failed and why — rather than being handed raw data to wade
through itself.

Priority order, per direct user instruction: (1) the worst real losing
closed trade(s), then (2) trades with an "abnormal" price move in the
hours around them that diverged from the setup's own thesis (a
mismatch, not necessarily a loss — a stop-out right before a reversal
matters here too). Also explicitly "recursive": a later session's own
report should be able to say "this is the same mistake as before," not
re-derive a fresh critique from nothing each time — see
build_recursion_context.

Compulsory, not a toggle: build_curiosity_report is always attempted by
ai.ftmo_suggest.suggest_ftmo_portfolio, and degrades to "" on ANY
failure at any stage (no qualifying trades, MT5 unreachable, every
fallback model exhausted) — this is a cosmetic/advisory addition to the
prompt, and must never be able to block or crash the real mega session.

FTMO-only: this module is never imported by ai/portfolio_suggest.py's
own shared build_past_lessons (which PMEX/PSX also use) — PMEX/PSX have
no MT5 closed-trade-history mechanics to build this from.

Every function up through build_candidate_dossier is pure or takes
injected I/O callables (fetch_range/get_history_deals_fn), matching
this codebase's own test style for MT5-touching code (mock the MT5
boundary, test the real logic around it) — only build_curiosity_report
itself performs real I/O by default.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

import config
from ai.openrouter_client import FAILED_MESSAGE as OPENROUTER_FAILED_MESSAGE
from ai.openrouter_client import MISSING_KEY_MESSAGE as OPENROUTER_MISSING_KEY_MESSAGE
from ai.openrouter_client import run_openrouter
from ai.portfolio_suggest import (
    AUDIT_MODEL_FALLBACKS,
    parse_final_allocation,
    parse_pending_setups,
)
from analysis.technical import classify_velocity_tier, compute_atr
from data.mt5_source import (
    CancelledPendingOrder,
    ClosedTrade,
    HistoricalDeal,
    fetch_mt5_price_history_range,
    get_cancelled_pending_orders,
    get_history_deals,
    group_closed_trades,
)

logger = logging.getLogger(__name__)


@dataclass
class CuriosityCandidate:
    trade: ClosedTrade | None
    reason_category: str  # "worst_loss" | "abnormal_move" | "missed_opportunity"
    thesis: str | None
    invalidation_condition: str | None
    pre_entry_path: pd.DataFrame  # H1 bars before entry/placement, for ATR + narrative context
    post_exit_path: pd.DataFrame  # H1 bars after exit — the forensic evidence for a CLOSED trade; empty for "missed_opportunity" (see resting_price_path instead)
    pre_entry_atr: float | None
    post_exit_move_atr_multiple: float | None = None  # "abnormal_move": post-exit move / pre-entry ATR. "missed_opportunity": the miss distance / pre-placement ATR (same "how many ATRs of real, unignorable movement" meaning, just measured against a different reference point)
    # --- added 2026-09-09: velocity tagging (every category) + missed-opportunity-specific fields ---
    # Real, motivating incident: INTC and AMD buy-limit orders, both
    # waiting for a technical pullback that never came on a genuinely
    # fast-moving market — both cancelled with price already well past
    # their own original take-profit, having never come close to
    # filling. The existing worst-loss/abnormal-move categories only
    # ever examine trades that actually opened and closed; a pending
    # order the market left behind was invisible to this whole function
    # before this addition.
    velocity_tier: str | None = None  # "fast" / "slow" / None — see analysis.technical.classify_velocity_tier, applied to this instrument's own H1 atr_pct at the relevant time, for EVERY category (not just missed_opportunity)
    cancelled_order: CancelledPendingOrder | None = None  # set only for "missed_opportunity"; trade is None in that case
    resting_price_path: pd.DataFrame = field(default_factory=pd.DataFrame)  # "missed_opportunity" only: real H1 price path for the window this order actually rested unfilled
    closest_approach_distance: float | None = None  # "missed_opportunity" only: signed distance from the entry to the closest the market ever came (positive = never reached the entry; negative would mean price crossed it, an anomaly worth reporting factually rather than assuming away)
    target_exceeded_before_fill: bool | None = None  # "missed_opportunity" only: did real price action move past the order's own original take-profit before the entry ever filled — the exact INTC/AMD signature


def select_worst_losers(
    trades: list[ClosedTrade], max_count: int = 1, min_loss_usd: float | None = None
) -> list[ClosedTrade]:
    """Priority 1. Real losses only (profit < 0), floor-filtered by
    min_loss_usd (default config.CURIOSITY_MIN_LOSS_USD) so a spread/
    commission-level "loss" is never crowned "the worst loser" on an
    unusually clean week, sorted worst-first, capped at max_count. Pure
    function, no I/O."""
    floor = config.CURIOSITY_MIN_LOSS_USD if min_loss_usd is None else min_loss_usd
    losers = [t for t in trades if t.profit < 0 and abs(t.profit) >= floor]
    losers.sort(key=lambda t: t.profit)  # ascending -> most negative (worst) first
    return losers[:max_count]


def _fetch_candidate_windows(
    trade: ClosedTrade,
    fetch_range: Callable[[str, str, datetime, datetime], pd.DataFrame],
    atr_context_hours: int,
    post_exit_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float | None]:
    """Pre-entry window (H1 bars, used both for the pre-entry ATR
    baseline and as narrative context) and post-exit window (the real
    forensic evidence — what actually happened after this trade closed).
    Shared by score_abnormal_moves and select_curiosity_candidates so
    both ever use IDENTICAL windows for the same trade — no risk of the
    scoring pass's own numbers silently disagreeing with what the final
    dossier shows."""
    pre_entry = fetch_range(
        trade.symbol, "H1",
        trade.opened_at - timedelta(hours=atr_context_hours), trade.opened_at,
    )
    atr = compute_atr(pre_entry) if not pre_entry.empty else None
    post_exit = fetch_range(
        trade.symbol, "H1",
        trade.closed_at, trade.closed_at + timedelta(hours=post_exit_hours),
    )
    return pre_entry, post_exit, atr


def score_abnormal_moves(
    trades: list[ClosedTrade],
    fetch_range: Callable[[str, str, datetime, datetime], pd.DataFrame],
    pool_size: int | None = None,
    atr_context_hours: int | None = None,
    post_exit_hours: int | None = None,
) -> list[tuple[ClosedTrade, float, float | None, pd.DataFrame, pd.DataFrame, float | None]]:
    """Priority 2. Scores up to `pool_size` most-recent closed trades
    (any P&L — this is about a price-action MISMATCH, not necessarily a
    loss) for how large a real post-exit price move was, relative to
    that trade's own pre-entry ATR baseline. Returns tuples of (trade,
    move_size_in_price_units, atr_multiple_or_None, pre_entry_df,
    post_exit_df, pre_entry_atr), sorted by atr_multiple descending, with
    None-multiple entries last — ATR/price data genuinely unavailable
    (e.g. a weekend-adjacent trade with too little pre-entry history) is
    a real, disclosed "can't score this one" outcome, never a fabricated
    0. The two DataFrames and the ATR value travel with each entry so
    select_curiosity_candidates can reuse them for the final dossier
    without a second, redundant MT5 fetch for the same trade."""
    pool_size = config.CURIOSITY_ABNORMAL_POOL_SIZE if pool_size is None else pool_size
    atr_context_hours = (
        config.CURIOSITY_ATR_CONTEXT_LOOKBACK_HOURS if atr_context_hours is None else atr_context_hours
    )
    post_exit_hours = config.CURIOSITY_POST_EXIT_WINDOW_HOURS if post_exit_hours is None else post_exit_hours

    scored: list[tuple[ClosedTrade, float, float | None, pd.DataFrame, pd.DataFrame, float | None]] = []
    for trade in trades[:pool_size]:
        pre_entry, post_exit, atr = _fetch_candidate_windows(trade, fetch_range, atr_context_hours, post_exit_hours)
        if post_exit.empty:
            scored.append((trade, 0.0, None, pre_entry, post_exit, atr))
            continue
        # Furthest CLOSE from the trade's own exit price, in EITHER
        # direction — a real, unignorable follow-through matters here
        # regardless of whether it "would have helped" or "confirmed"
        # the original side; that judgment is exactly what the model
        # call is for, not this deterministic scoring step.
        furthest = float(post_exit["Close"].sub(trade.close_price).abs().max())
        atr_multiple = (furthest / atr) if atr and atr > 0 else None
        scored.append((trade, furthest, atr_multiple, pre_entry, post_exit, atr))

    scored.sort(key=lambda item: (item[2] is None, -(item[2] or 0.0)))
    return scored


def _velocity_tier_for(atr: float | None, reference_price: float | None) -> str | None:
    """Shared by every candidate category — converts a raw ATR (price
    units) plus a reference price into this instrument's own H1 atr_pct,
    then classifies it via analysis.technical.classify_velocity_tier.
    None (never guessed) when either input is unavailable — same "don't
    fabricate a stat from missing data" convention as everything else in
    this file."""
    if atr is None or not reference_price:
        return None
    return classify_velocity_tier(atr / reference_price * 100)


def score_missed_opportunities(
    cancelled_orders: list[CancelledPendingOrder],
    fetch_range: Callable[[str, str, datetime, datetime], pd.DataFrame],
    pool_size: int | None = None,
    atr_context_hours: int | None = None,
    min_atr_multiple: float | None = None,
) -> list[tuple[CancelledPendingOrder, float | None, pd.DataFrame, pd.DataFrame, float | None, float | None, bool | None]]:
    """Priority 3 (added 2026-09-09) — real, motivating incident: INTC
    and AMD buy-limit orders, both waiting for a technical pullback that
    never came on a genuinely fast-moving market, both eventually
    cancelled with price already well past their own original take-
    profit, having never come anywhere close to filling. Neither
    select_worst_losers nor score_abnormal_moves can ever see this: both
    operate purely on ClosedTrade, and a pending order the market left
    behind produces no P&L and no exit price — it's structurally
    invisible to a "trades that happened" universe, no matter how often
    it recurs.

    For each cancelled/expired order (most recent `pool_size` first),
    fetches the real H1 price path for the window it actually rested
    (time_setup -> time_done) plus a pre-placement window for the ATR
    baseline (same _fetch_candidate_windows-style reasoning, just a
    genuinely different pair of reference points than a closed trade's
    own pre-entry/post-exit split). Computes:
    - closest_approach: the closest the market ever came to the entry
      price during the resting window, side-aware (a buy limit resting
      BELOW market is approached from above, via the window's own LOW;
      a sell limit resting ABOVE market is approached from below, via
      the window's own HIGH). Positive = never reached the entry at all;
      negative would mean price genuinely crossed the level without the
      order filling — a real anomaly (e.g. a broker-side minimum-
      distance rule, or a data gap), reported factually rather than
      assumed impossible.
    - target_exceeded: whether real price action moved past this
      order's own original take-profit before/without the entry ever
      filling — the exact, unambiguous INTC/AMD signature: even a fill
      now would offer no realistic remaining reward.
    - atr_multiple: |closest_approach| / pre-placement ATR — the same
      "how many ATRs of real, unignorable distance" scale score_
      abnormal_moves already uses, just measured against a different
      reference point (the never-reached entry, not a post-exit move).

    Only orders clearing `min_atr_multiple` (default config.
    CURIOSITY_MISSED_OPPORTUNITY_MIN_ATR_MULTIPLE) are genuinely
    "missed" in a way worth forensic attention — a cancellation for an
    unrelated, benign reason (a fresh mega session simply changed its
    mind, FTMO headroom forced it) with the market never having moved
    meaningfully is real information too, just not a story. An order
    with no usable ATR baseline is excluded outright (never scored as
    if it cleared an unverifiable threshold), same "never fabricate"
    discipline as every other stat in this file.

    Returns (order, atr_multiple_or_None, pre_placement_df, resting_df,
    atr, closest_approach_distance, target_exceeded), sorted by
    atr_multiple descending, already filtered to the threshold."""
    pool_size = config.CURIOSITY_MISSED_OPPORTUNITY_POOL_SIZE if pool_size is None else pool_size
    atr_context_hours = (
        config.CURIOSITY_ATR_CONTEXT_LOOKBACK_HOURS if atr_context_hours is None else atr_context_hours
    )
    min_atr_multiple = (
        config.CURIOSITY_MISSED_OPPORTUNITY_MIN_ATR_MULTIPLE if min_atr_multiple is None else min_atr_multiple
    )

    scored: list[tuple[CancelledPendingOrder, float | None, pd.DataFrame, pd.DataFrame, float | None, float | None, bool | None]] = []
    for order in cancelled_orders[:pool_size]:
        pre_placement = fetch_range(
            order.symbol, "H1", order.time_setup - timedelta(hours=atr_context_hours), order.time_setup,
        )
        atr = compute_atr(pre_placement) if not pre_placement.empty else None
        resting = fetch_range(order.symbol, "H1", order.time_setup, order.time_done)
        if resting.empty or atr is None or atr <= 0:
            continue  # no real basis to score this one — never fabricate a threshold comparison

        if order.side == "buy":
            closest_reached = float(resting["Low"].min())
            closest_approach = closest_reached - order.price_open
        else:
            closest_reached = float(resting["High"].max())
            closest_approach = order.price_open - closest_reached

        target_exceeded: bool | None = None
        if order.tp is not None:
            target_exceeded = bool(
                (resting["High"] >= order.tp).any() if order.side == "buy" else (resting["Low"] <= order.tp).any()
            )

        atr_multiple = abs(closest_approach) / atr
        if atr_multiple < min_atr_multiple:
            continue
        scored.append((order, atr_multiple, pre_placement, resting, atr, closest_approach, target_exceeded))

    scored.sort(key=lambda item: -(item[1] or 0.0))
    return scored


def select_curiosity_candidates(
    trades: list[ClosedTrade],
    fetch_range: Callable[[str, str, datetime, datetime], pd.DataFrame],
    max_losers: int = 1,
    max_abnormal: int = 1,
    min_loss_usd: float | None = None,
    pool_size: int | None = None,
    atr_context_hours: int | None = None,
    post_exit_hours: int | None = None,
    abnormal_atr_multiple: float | None = None,
    cancelled_orders: list[CancelledPendingOrder] | None = None,
    max_missed: int = 1,
    missed_pool_size: int | None = None,
    missed_min_atr_multiple: float | None = None,
) -> list[CuriosityCandidate]:
    """Orchestrates all three priorities: worst losers first, then
    abnormal-move trades (deduplicated by position_id against the worst
    losers already picked) whose post-exit move clears
    abnormal_atr_multiple, then (added 2026-09-09) missed-opportunity
    cancelled/expired pending orders whose miss clears missed_min_atr_
    multiple — see score_missed_opportunities' own docstring for why
    this third category exists at all (a pending order the market left
    behind is invisible to the first two, which only ever examine
    trades that actually opened and closed).

    `cancelled_orders` defaults to None (-> no missed-opportunity
    candidates at all) rather than requiring every existing caller to
    pass an empty list — keeps this function's own signature backward
    compatible for anything that hasn't been updated to fetch this new
    data yet.

    Every candidate, across all three categories, gets tagged with its
    own velocity_tier (analysis.technical.classify_velocity_tier applied
    to this instrument's own H1 atr_pct at the relevant time) — real,
    aggregatable context for reasoning about whether a given entry
    style (e.g. waiting for a pullback trigger) is well- or poorly-
    suited to THIS instrument's own typical speed, not just a bare fact
    about one isolated trade.

    Total output deliberately small (max_losers + max_abnormal +
    max_missed, default 3) so the eventual model prompt/report stays
    focused on real signal, not diluted across many trades.
    thesis/invalidation_condition are left None here — find_thesis_for_
    symbol fills them in separately (this function has no records-dir
    I/O of its own)."""
    atr_context_hours = (
        config.CURIOSITY_ATR_CONTEXT_LOOKBACK_HOURS if atr_context_hours is None else atr_context_hours
    )
    post_exit_hours = config.CURIOSITY_POST_EXIT_WINDOW_HOURS if post_exit_hours is None else post_exit_hours
    abnormal_atr_multiple = (
        config.CURIOSITY_ABNORMAL_ATR_MULTIPLE if abnormal_atr_multiple is None else abnormal_atr_multiple
    )

    worst_losers = select_worst_losers(trades, max_count=max_losers, min_loss_usd=min_loss_usd)
    selected_ids = {t.position_id for t in worst_losers}

    scored = score_abnormal_moves(
        trades, fetch_range, pool_size=pool_size,
        atr_context_hours=atr_context_hours, post_exit_hours=post_exit_hours,
    )
    abnormal_picks: list[tuple[ClosedTrade, float, float | None, pd.DataFrame, pd.DataFrame, float | None]] = []
    for entry in scored:
        if len(abnormal_picks) >= max_abnormal:
            break
        trade, _move_size, atr_multiple = entry[0], entry[1], entry[2]
        if trade.position_id in selected_ids:
            continue
        if atr_multiple is None or atr_multiple < abnormal_atr_multiple:
            continue
        abnormal_picks.append(entry)
        selected_ids.add(trade.position_id)

    missed_picks = score_missed_opportunities(
        cancelled_orders or [], fetch_range,
        pool_size=missed_pool_size, atr_context_hours=atr_context_hours,
        min_atr_multiple=missed_min_atr_multiple,
    )[:max_missed]

    candidates: list[CuriosityCandidate] = []
    for trade in worst_losers:
        pre_entry, post_exit, atr = _fetch_candidate_windows(trade, fetch_range, atr_context_hours, post_exit_hours)
        candidates.append(
            CuriosityCandidate(
                trade=trade, reason_category="worst_loss", thesis=None,
                invalidation_condition=None, pre_entry_path=pre_entry,
                post_exit_path=post_exit, pre_entry_atr=atr,
                velocity_tier=_velocity_tier_for(atr, trade.open_price),
            )
        )
    for trade, _move_size, atr_multiple, pre_entry, post_exit, atr in abnormal_picks:
        candidates.append(
            CuriosityCandidate(
                trade=trade, reason_category="abnormal_move", thesis=None,
                invalidation_condition=None, pre_entry_path=pre_entry,
                post_exit_path=post_exit, pre_entry_atr=atr,
                post_exit_move_atr_multiple=atr_multiple,
                velocity_tier=_velocity_tier_for(atr, trade.open_price),
            )
        )
    for order, atr_multiple, pre_placement, resting, atr, closest_approach, target_exceeded in missed_picks:
        candidates.append(
            CuriosityCandidate(
                trade=None, reason_category="missed_opportunity", thesis=None,
                invalidation_condition=None, pre_entry_path=pre_placement,
                post_exit_path=pd.DataFrame(), pre_entry_atr=atr,
                post_exit_move_atr_multiple=atr_multiple,
                velocity_tier=_velocity_tier_for(atr, order.price_open),
                cancelled_order=order, resting_price_path=resting,
                closest_approach_distance=closest_approach,
                target_exceeded_before_fill=target_exceeded,
            )
        )
    return candidates


_STAGE3_MARKER = "## Stage 3 — Claude's final revised suggestion\n\n"


def find_thesis_for_symbol(
    symbol: str, before: datetime, records_dir: Path | None = None
) -> tuple[str | None, str | None]:
    """Scans records_dir's saved portfolio_suggestion_*.md files, NEWEST
    FIRST, for the most recent one generated before `before` (a trade's
    own opened_at) that mentions `symbol` — checking BOTH
    parse_final_allocation (an immediate-allocation target) AND
    parse_pending_setups (the trade may have originated from a
    TRIGGERED pending setup instead, whose reason/trigger_condition
    live only there) — same technique build_past_outcome_lessons and
    app.py's _load_latest_saved_suggestion already use for this exact
    file format.

    Timezone bridge: a saved record's own filename timestamp is naive
    LOCAL machine time (ai/session_record.py's save_portfolio_session
    uses plain datetime.now()), while `before` (a ClosedTrade's own
    opened_at) is tz-aware UTC (data/mt5_source.py's get_history_deals).
    generated_at.astimezone(timezone.utc) is documented Python behavior
    for a naive datetime — interpreted as THIS system's own local zone,
    converted correctly — the same fix already applied in app.py's own
    "Latest Suggestion" panel earlier this session.

    Returns (reason, invalidation_condition), both None if nothing
    recoverable. Never raises."""
    records_dir = Path(config.FTMO_RECORDS_DIR) if records_dir is None else records_dir
    if not records_dir.exists():
        return None, None

    before_utc = before if before.tzinfo is not None else before.replace(tzinfo=timezone.utc)

    for path in sorted(records_dir.glob("portfolio_suggestion_*.md"), reverse=True):
        try:
            generated_at = datetime.strptime(
                path.stem.removeprefix("portfolio_suggestion_"), "%Y-%m-%d_%H%M%S"
            )
        except ValueError:
            continue
        if generated_at.astimezone(timezone.utc) > before_utc:
            continue  # this session came AFTER the trade opened -- can't be its origin

        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if _STAGE3_MARKER not in content:
            continue
        final_answer = content.split(_STAGE3_MARKER, 1)[1]

        # Only a REAL nonzero allocation counts as this symbol's own
        # originating thesis here -- a pct=0 entry is a CANCEL with no
        # real reason of its own to attribute to a later trade, and (per
        # the comment below) this account's own real history has
        # legitimately re-listed a just-cancelled symbol as a fresh
        # Pending Setup the SAME session. Returning early on a pct=0
        # match would shadow that pending setup's own real reason/
        # trigger_condition -- the actual origin of a trade that opens
        # once the setup later triggers -- with the cancel's own, usually
        # empty, reason instead (real bug found on self-review).
        allocation = parse_final_allocation(final_answer)
        if allocation and symbol in allocation and allocation[symbol].pct > 0:
            entry = allocation[symbol]
            return (entry.reason or None), entry.invalidation_condition

        # Only symbols with REAL nonzero exposure are off-limits for
        # parse_pending_setups (it fails the WHOLE parse if a pending
        # symbol collides with one here) — a pct=0 entry is a CANCEL,
        # and this account's own real history has legitimately re-listed
        # a just-cancelled symbol as a fresh Pending Setup the same
        # session (see _write_latest_suggestion's own comment for the
        # real BTCUSD incident this exact filter was first fixed for;
        # using frozenset(allocation.keys()) here would silently
        # reintroduce that same bug for thesis lookup).
        immediate_symbols = (
            frozenset(sym for sym, entry in allocation.items() if entry.pct > 0) if allocation else frozenset()
        )
        pending = parse_pending_setups(final_answer, immediate_symbols=immediate_symbols)
        if pending:
            for setup in pending:
                if setup.symbol == symbol:
                    return (setup.reason or None), setup.trigger_condition

    return None, None


def _build_missed_opportunity_dossier(candidate: CuriosityCandidate) -> str:
    """Sibling to the closed-trade dossier below, for a cancelled/
    expired pending order that never filled — see CuriosityCandidate's
    own field comments and score_missed_opportunities' own docstring
    for the real INTC/AMD incident this exists to surface. Same "pure
    factual text, no judgment baked in" discipline as the closed-trade
    version."""
    o = candidate.cancelled_order
    assert o is not None  # only ever called for reason_category == "missed_opportunity"
    resting_hours = (o.time_done - o.time_setup).total_seconds() / 3600
    lines = [
        f"### {o.symbol} — flagged as: missed opportunity (pending order never filled)",
        f"Side: {o.side} | Volume: {o.volume:g} lots | Order type: {o.order_type}",
        f"Placed: {o.time_setup:%Y-%m-%d %H:%M} UTC @ entry {o.price_open:g}",
        f"Cancelled/expired: {o.time_done:%Y-%m-%d %H:%M} UTC (rested {resting_hours:.1f}h, unfilled)",
        f"Original stop: {o.sl:g}" if o.sl is not None else "Original stop: none set",
        f"Original target: {o.tp:g}" if o.tp is not None else "Original target: none set",
    ]
    lines.append(f"Original thesis: {candidate.thesis}" if candidate.thesis else "Original thesis: not recoverable from saved records.")
    if candidate.invalidation_condition:
        lines.append(f"Stated trigger/invalidation condition: {candidate.invalidation_condition}")
    if candidate.velocity_tier is not None:
        lines.append(
            f"Velocity tier: {candidate.velocity_tier.upper()} (this instrument's own H1 ATR% at the time "
            "it was placed — see analysis.technical.classify_velocity_tier)"
        )
    if candidate.pre_entry_atr is not None:
        lines.append(f"Pre-placement H1 ATR baseline: {candidate.pre_entry_atr:.5g}")
    if candidate.closest_approach_distance is not None:
        if candidate.closest_approach_distance > 0:
            lines.append(
                f"Closest the market ever came to the entry while this order rested: "
                f"{candidate.closest_approach_distance:.5g} short of it — price never reached the entry level at all."
            )
        else:
            lines.append(
                f"Real price action moved {abs(candidate.closest_approach_distance):.5g} PAST the entry level "
                "without the order filling — a genuine anomaly (e.g. a broker minimum-distance rule), worth noting as-is."
            )
    if candidate.post_exit_move_atr_multiple is not None:
        lines.append(f"That distance = {candidate.post_exit_move_atr_multiple:.2f}x the pre-placement ATR.")
    if candidate.target_exceeded_before_fill:
        lines.append(
            "**TARGET ALREADY EXCEEDED**: real price action moved past this order's own original "
            "take-profit BEFORE the entry ever filled — even a fill now would offer no realistic reward left."
        )
    if not candidate.resting_price_path.empty:
        closes = ", ".join(f"{c:g}" for c in candidate.resting_price_path["Close"].head(12))
        lines.append(f"H1 closes for the real window this order rested: {closes}")
    return "\n".join(lines)


def build_candidate_dossier(candidate: CuriosityCandidate) -> str:
    """Pure factual text, no judgment baked in — symbol/side/volume,
    entry/exit price+time, realized P&L, the recovered thesis (or an
    explicit "not recoverable" note), the pre-entry/post-exit price
    paths, the pre-entry ATR, the velocity tier, and (for an abnormal-
    move candidate) the ATR-multiple of the post-exit move. Matches this
    codebase's "compute the real number, let the model reason"
    convention used everywhere else (backtests, feasibility %,
    build_past_outcome_lessons). Delegates to _build_missed_opportunity_
    dossier for the third category (added 2026-09-09), which has no
    ClosedTrade to read from at all."""
    if candidate.reason_category == "missed_opportunity":
        return _build_missed_opportunity_dossier(candidate)

    t = candidate.trade
    lines = [
        f"### {t.symbol} — flagged as: {candidate.reason_category.replace('_', ' ')}",
        f"Side: {t.side} | Volume: {t.volume:g} lots",
        f"Opened: {t.opened_at:%Y-%m-%d %H:%M} UTC @ {t.open_price:g}",
        f"Closed: {t.closed_at:%Y-%m-%d %H:%M} UTC @ {t.close_price:g}",
        f"Realized P&L: {t.profit:+.2f}",
    ]
    lines.append(f"Original thesis: {candidate.thesis}" if candidate.thesis else "Original thesis: not recoverable from saved records.")
    if candidate.invalidation_condition:
        lines.append(f"Stated invalidation condition: {candidate.invalidation_condition}")
    if candidate.velocity_tier is not None:
        lines.append(
            f"Velocity tier: {candidate.velocity_tier.upper()} (this instrument's own H1 ATR% at entry time "
            "— see analysis.technical.classify_velocity_tier)"
        )
    if candidate.pre_entry_atr is not None:
        lines.append(f"Pre-entry H1 ATR baseline: {candidate.pre_entry_atr:.5g}")
    if candidate.post_exit_move_atr_multiple is not None:
        lines.append(
            f"Post-exit price move: {candidate.post_exit_move_atr_multiple:.2f}x the pre-entry ATR "
            "within the forensic window -- a real, sizeable follow-through the setup didn't anticipate."
        )
    if not candidate.pre_entry_path.empty:
        closes = ", ".join(f"{c:g}" for c in candidate.pre_entry_path["Close"].tail(6))
        lines.append(f"Pre-entry H1 closes (last few bars before entry): {closes}")
    if not candidate.post_exit_path.empty:
        closes = ", ".join(f"{c:g}" for c in candidate.post_exit_path["Close"].head(12))
        lines.append(f"Post-exit H1 closes (bars after exit): {closes}")
    return "\n".join(lines)


def build_recursion_context(curiosity_records_dir: Path | None = None) -> str:
    """"" when no prior curiosity report exists (first-ever run — nothing
    to learn from yet, not a failure); otherwise the single most recent
    saved report's own text, framed as an explicit ask to check today's
    candidate(s) against it for a recurring pattern rather than
    re-deriving a fresh critique from nothing each session — the
    "recursive" half of this feature, direct user request."""
    curiosity_records_dir = (
        Path(config.CURIOSITY_RECORDS_DIR) if curiosity_records_dir is None else curiosity_records_dir
    )
    if not curiosity_records_dir.exists():
        return ""
    files = sorted(curiosity_records_dir.glob("curiosity_report_*.md"), reverse=True)
    if not files:
        return ""
    try:
        prior = files[0].read_text(encoding="utf-8")
    except OSError:
        return ""
    return (
        "Below is your own most recent retrospective report card from a past "
        "session. Explicitly say whether today's candidate trade(s) below "
        "resemble a pattern you already identified there -- a RECURRING "
        "mistake matters more than a fresh, isolated one -- rather than "
        "re-deriving a brand-new critique from nothing each time:\n\n" + prior
    )


_CURIOSITY_PROMPT_HEADER = (
    "You are conducting a genuine, honest retrospective self-critique for a "
    "real FTMO prop-firm trading account. Below is one or more real candidates "
    "-- either a CLOSED trade (with its original stated thesis if recoverable, "
    "and the real price action before entry and after exit), or (added "
    "2026-09-09) a MISSED OPPORTUNITY: a real pending order that was cancelled "
    "or expired without ever filling, because the market moved on before its "
    "own entry trigger was ever reached. For EACH closed trade, answer "
    "concretely: was the original reasoning actually right or wrong given "
    "what happened? Where specifically did the assumption fail, and why? "
    "What should have been done instead (a different entry, a different "
    "stop/target, sitting out entirely, or nothing at all -- the original "
    "call may genuinely have been sound even though the trade lost)?\n\n"
    "For EACH missed-opportunity candidate, answer just as concretely: was "
    "waiting for the stated trigger condition the right call given what "
    "actually happened, or would a nearer entry, a looser trigger, or a "
    "smaller immediate stake have captured a real move while still "
    "respecting risk? If the instrument's own velocity tier is given "
    "(FAST vs SLOW -- how much of its typical range arrives per hour), "
    "weigh explicitly whether a pullback-and-wait entry style is well- or "
    "poorly-suited to THIS instrument's own real speed, and say so plainly "
    "if you think it's a recurring mismatch rather than a one-off. Do not "
    "prescribe a fixed rule (e.g. \"always chase fast instruments\") -- state "
    "what this specific real evidence shows, and let the mega session weigh "
    "it against everything else it knows.\n\n"
    "Be specific and concrete, citing the real numbers given below -- this is "
    "a report card another AI will read and act on, not a general lecture. "
    "Keep the whole response focused, under roughly 400 words."
)


def _run_curiosity_model_with_retry(prompt: str) -> str:
    """Tries AUDIT_MODEL_FALLBACKS in order, starting from the one
    reasoning-tuned model in that bench (Nvidia Nemotron-Nano-Omni-30B-
    Reasoning) -- direct user instruction to delegate this to "another
    free openrouter model best fit for this type of task." Tries each
    model ONCE (not a per-model retry loop the way _run_audit_with_retry
    does — with 7 fallbacks already providing real redundancy, and a
    much shorter overall budget appropriate for one sequential call
    gating the mega session's own draft-prompt build, not a parallel
    background audit pool), moving to the next on any failure, bounded
    by a wall-clock deadline (config.CURIOSITY_RETRY_TIMEOUT_SECONDS)
    rather than a fixed attempt count, sleeping config.CURIOSITY_
    RETRY_INTERVAL_SECONDS between attempts. Each individual call uses
    its own short config.CURIOSITY_MODEL_TIMEOUT_SECONDS, deliberately
    NOT the audit pool's own config.OPENROUTER_TIMEOUT_SECONDS (90s) --
    the deadline check below only ever stops a NEW attempt from
    starting, never bounds one already in flight, so reusing that
    longer per-call budget across up to 7 fallbacks could let a run
    where several genuinely hang take several minutes despite this
    function's own "give up far sooner" design (real bug found on
    self-review, fixed before this ever ran live). A missing-key
    failure stops immediately (a config problem, not a transient one --
    no model in the list will succeed either). "" on total exhaustion --
    never raises."""
    deadline = time.monotonic() + config.CURIOSITY_RETRY_TIMEOUT_SECONDS
    for label, model_id, _profile in AUDIT_MODEL_FALLBACKS:
        if time.monotonic() >= deadline:
            logger.info("Curiosity model cascade: retry budget exhausted before trying %s.", label)
            break
        try:
            result = run_openrouter(prompt, model=model_id, timeout=config.CURIOSITY_MODEL_TIMEOUT_SECONDS)
        except Exception:
            result = OPENROUTER_FAILED_MESSAGE

        if result == OPENROUTER_MISSING_KEY_MESSAGE:
            return ""
        if result != OPENROUTER_FAILED_MESSAGE:
            return result

        logger.info("Curiosity model %s unavailable -- trying next fallback.", label)
        if time.monotonic() < deadline:
            time.sleep(config.CURIOSITY_RETRY_INTERVAL_SECONDS)
    return ""


def _candidate_symbol_and_reference_time(candidate: CuriosityCandidate) -> tuple[str, datetime]:
    """A missed-opportunity candidate has no ClosedTrade at all (see
    CuriosityCandidate's own field comments — trade is None, cancelled_
    order is set instead) — every place that needs "this candidate's
    symbol" and "the time to search saved records before" (find_thesis_
    for_symbol) or just "the symbol" (save_curiosity_report) must read
    from whichever of the two is actually populated, never assume trade
    unconditionally the way this code did before this category existed
    (a real bug caught while wiring this in: the old code crashed with
    AttributeError on a None trade, silently swallowed by build_
    curiosity_report's own outer try/except, degrading the ENTIRE report
    to empty rather than just skipping the one new category)."""
    if candidate.trade is not None:
        return candidate.trade.symbol, candidate.trade.opened_at
    assert candidate.cancelled_order is not None  # every candidate has exactly one of the two set
    return candidate.cancelled_order.symbol, candidate.cancelled_order.time_setup


def save_curiosity_report(
    report_text: str, candidates: list[CuriosityCandidate], curiosity_records_dir: Path | None = None
) -> Path | None:
    """Mirrors ai/session_record.py::save_portfolio_session's own
    pattern (a timestamped, human-readable .md transcript) but is
    self-contained here rather than touching the shared SessionRecord
    dataclass PMEX/PSX also use. Never raises: an OSError (disk full,
    permissions) is swallowed and None returned, same "cosmetic side
    effect, must never break the real feature" contract as its model."""
    curiosity_records_dir = (
        Path(config.CURIOSITY_RECORDS_DIR) if curiosity_records_dir is None else curiosity_records_dir
    )
    timestamp = datetime.now()
    symbols = ", ".join(sorted({_candidate_symbol_and_reference_time(c)[0] for c in candidates})) or "(none)"
    content = (
        f"# Curiosity Report — {timestamp:%Y-%m-%d %H:%M:%S}\n\n"
        f"**Candidates reviewed:** {symbols}\n\n"
        "## Report card\n\n"
        f"{report_text}\n"
    )
    try:
        curiosity_records_dir.mkdir(parents=True, exist_ok=True)
        path = curiosity_records_dir / f"curiosity_report_{timestamp:%Y-%m-%d_%H%M%S}.md"
        path.write_text(content, encoding="utf-8")
        return path
    except OSError:
        return None


def build_curiosity_report(
    records_dir: Path | None = None,
    curiosity_records_dir: Path | None = None,
    get_history_deals_fn: Callable[[datetime], list[HistoricalDeal]] = get_history_deals,
    fetch_range_fn: Callable[[str, str, datetime, datetime], pd.DataFrame] = fetch_mt5_price_history_range,
    get_cancelled_pending_orders_fn: Callable[[datetime], list[CancelledPendingOrder]] = get_cancelled_pending_orders,
) -> str:
    """Public orchestrator, called unconditionally (compulsory, not a
    toggle) from ai.ftmo_suggest.suggest_ftmo_portfolio: fetches the
    last config.CURIOSITY_LOOKBACK_DAYS of real closed trades AND (added
    2026-09-09) real cancelled/expired pending orders, selects up to ~3
    candidates (worst loser + abnormal-move + missed-opportunity, see
    select_curiosity_candidates), recovers each one's original thesis,
    builds a factual dossier per candidate, prepends the recursion
    context from the last saved report, sends the whole prompt to the
    curiosity model cascade, saves this run's own report, and returns
    the model's own condensed report-card text framed for Claude to
    read directly.

    A trades-fetch failure is not automatically a missed-opportunities
    failure (or vice versa) — get_cancelled_pending_orders_fn is called
    independently, inside the same outer try/except, so ANY genuinely
    unexpected exception anywhere still degrades the WHOLE report to ""
    (this stays a cosmetic/advisory addition, never allowed to block or
    crash the real mega session), but neither data source's own normal
    empty result (a quiet week with nothing cancelled) blocks the other
    from still producing a real report.

    "" on ANY failure at any stage — no closed trades AND no cancelled
    orders in the lookback window, no candidate cleared any selection
    bar, MT5 unreachable, every fallback model exhausted, or any
    genuinely unexpected exception."""
    try:
        lookback_start = datetime.now(timezone.utc) - timedelta(days=config.CURIOSITY_LOOKBACK_DAYS)
        deals = get_history_deals_fn(lookback_start.replace(tzinfo=None))
        trades = group_closed_trades(deals)
        cancelled_orders = get_cancelled_pending_orders_fn(lookback_start.replace(tzinfo=None))
        if not trades and not cancelled_orders:
            return ""

        candidates = select_curiosity_candidates(trades, fetch_range_fn, cancelled_orders=cancelled_orders)
        if not candidates:
            return ""

        for candidate in candidates:
            symbol, before = _candidate_symbol_and_reference_time(candidate)
            reason, invalidation = find_thesis_for_symbol(symbol, before, records_dir)
            candidate.thesis = reason
            candidate.invalidation_condition = invalidation

        dossiers = "\n\n".join(build_candidate_dossier(c) for c in candidates)
        recursion_context = build_recursion_context(curiosity_records_dir)
        prompt = f"{_CURIOSITY_PROMPT_HEADER}\n\n{dossiers}"
        if recursion_context:
            prompt = f"{recursion_context}\n\n{prompt}"

        report = _run_curiosity_model_with_retry(prompt)
        if not report:
            return ""

        save_curiosity_report(report, candidates, curiosity_records_dir)
        return (
            "Compulsory retrospective self-critique (curiosity function) -- an "
            "independent model's own report card on this account's recent real "
            "closed trades and/or cancelled pending orders, focused on its worst "
            "loss, any abnormal post-exit price move, and/or a pending order the "
            "market left behind before it ever filled. Check whether the CURRENT "
            "draft repeats any of the same mistakes flagged below:\n\n" + report
        )
    except Exception:
        logger.exception("build_curiosity_report failed -- degrading to empty (cosmetic/advisory only).")
        return ""
