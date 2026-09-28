"""Final live re-check of the Mega Session's entries (2026-09-25).

Found on the first real M5-only session: Mega analyses all instruments at the start of its run
(18:03-18:06 UTC) but its final answer arrives ~25 minutes later (18:28). In that time SOLUSD moved
+2.9 M5 ATR against its planned pullback entry and NVDA +1 ATR; a resting limit that started 1.6-2.3 ATR
below the market (55-67% fill odds, measured) was 3.7-4.5 ATR away (30-37%) by the time the Clerk placed
it, and both trades' targets were later reached without a fill.

This module closes that gap in two halves, both grounded in REAL numbers only:

1. build_live_recheck(draft, summary): right after the free-model audit, Python measures — from the live
   MT5 feed, not from Claude — the current bid/ask, a fresh M5 ATR/structure/session-level read and a
   fresh M5 trade-zone candidate for every symbol the draft proposes, flags each entry that drifted more
   than config.RECHECK_DRIFT_ATR M5 ATRs (or whose stop/target is already passed), and returns a prompt
   block for Claude's final revision. The block forbids web-searched or self-computed prices: the printed
   numbers are the only valid basis for any adjustment.
2. apply_recheck_validation(payload, ...): after Claude's final answer, every CHANGED entry is verified
   deterministically against a fresh quote and the same printed levels (distance from the price, each level
   within tolerance of a real level, stop no tighter than the effective floor, reach limit, net reward:risk,
   pct never raised). A re-issue that fails is discarded — the draft's entry is kept (or the symbol is
   dropped if its levels are already dead) — so a fabricated level can never reach the Clerk.

Everything degrades open: any failure returns None / an empty note list and the pipeline behaves exactly as
it did before this step existed."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import config
from analysis.entry_mode import normalise_entry_mode

logger = logging.getLogger(__name__)

# Measured 2026-09-24 on 20 symbols x 6000 M5 bars: the chance a buy limit d M5 ATRs below the price is
# touched within 3 hours (one direction), used verbatim in the prompt block.
_FILL_ODDS_TEXT = (
    "measured fill odds for a resting limit N M5 ATRs from the price within 3 hours: 1.0x 79%, 1.6x 67%, "
    "2.3x 55%, 3.0x 46%, 4.0x 34%, 4.5x 30% (and at 4x the target level is reached WITHOUT a fill ~63% of the time)"
)


@dataclass
class SymbolRecheck:
    symbol: str
    kind: str  # "immediate" | "pending"
    side: str
    entry: float
    stop: float
    target: float
    pct: float
    status: str  # ok | DRIFTED | MARKETABLE | TARGET_PASSED | STOP_BREACHED | HELD | NO_LIVE_DATA
    bid: float | None = None
    ask: float | None = None
    atr: float | None = None
    floor_distance: float | None = None
    snapshot_price: float | None = None
    drift_atr: float | None = None  # signed: > 0 = entry is BELOW the ask for a buy / ABOVE the bid for a sell (a pullback is needed)
    anchors: list[tuple[str, float, float]] = field(default_factory=list)  # (label, low, high) real levels
    playbook_text: str = ""  # analysis.playbook advice for this side from the FRESH read (printed in the entry-mode menu)
    text: str = ""

    @property
    def spread(self) -> float:
        return (self.ask - self.bid) if self.ask is not None and self.bid is not None else 0.0


@dataclass
class RecheckSheet:
    generated_utc: str
    symbols: dict[str, SymbolRecheck]
    prompt_block: str

    def flagged(self) -> list[str]:
        return [s for s, r in self.symbols.items() if r.status in ("DRIFTED", "TARGET_PASSED", "STOP_BREACHED")]


# --- helpers ------------------------------------------------------------------------------------------


def _snapshot_prices(summary: str) -> dict[str, float]:
    """The bid/ask mid each symbol had in the data snapshot Claude was given, parsed from the summary's own
    per-instrument header lines ('- MSFT (Microsoft, Spot CFD): bid 494.58, ask 494.93')."""
    prices: dict[str, float] = {}
    for match in re.finditer(r"^- (\S+) \([^)\n]*\): bid ([0-9.]+), ask ([0-9.]+)", summary, re.MULTILINE):
        try:
            prices[match.group(1)] = (float(match.group(2)) + float(match.group(3))) / 2
        except ValueError:
            continue
    return prices


def _anchors_from_analysis(analysis, bid: float, ask: float) -> list[tuple[str, float, float]]:
    """Every REAL price level the fresh M5 read offers: S/R bands, Fibonacci levels, session levels, and the
    live quote itself. A re-issued entry/target must land on one of these (within tolerance)."""
    anchors: list[tuple[str, float, float]] = []
    structure = analysis.m5_structure
    if structure is not None and structure.sr_levels is not None:
        for kind, levels in (("support", structure.sr_levels.support_levels), ("resistance", structure.sr_levels.resistance_levels)):
            for lvl in levels:
                lo = lvl.low if lvl.low is not None else lvl.price
                hi = lvl.high if lvl.high is not None else lvl.price
                anchors.append((f"M5 {kind} band {lo:.5g}-{hi:.5g}", min(lo, hi), max(lo, hi)))
    if structure is not None and structure.fibonacci is not None:
        for name, price in structure.fibonacci.levels.items():
            anchors.append((f"M5 Fibonacci {name} {price:.5g}", price, price))
    levels = analysis.intraday_levels
    if levels is not None:
        for name, price in (
            ("prev-day high", levels.prev_day_high), ("prev-day low", levels.prev_day_low),
            ("prev-day close", levels.prev_day_close), ("session open", levels.day_open),
            ("session high", levels.day_high), ("session low", levels.day_low), ("session VWAP", levels.vwap),
        ):
            if price:
                anchors.append((f"{name} {price:.5g}", price, price))
    # The trade-zone targets Claude was shown, including the measure-rule projections (half/full leg) — a
    # re-issued target may legitimately sit on one of these.
    try:
        from ai.ftmo_suggest import m5_trade_zone_candidates
        from analysis.setup_classifier import classify_setups
        from analysis.timeframe_profiles import M5_PROFILE

        signals = classify_setups(analysis.m5_stats, analysis.m5_structure, analysis.m5_divergence, profile=M5_PROFILE)
        for side, zone in m5_trade_zone_candidates(analysis, signals).items():
            if zone is None:
                continue
            for price in zone.take_profits:
                label = zone.target_labels.get(price, "zone target")
                anchors.append((f"M5 {side} {label} {price:.5g}", price, price))
    except Exception:  # noqa: BLE001 - the re-check degrades open; missing zone anchors only make it stricter
        logger.debug("live re-check: zone anchors unavailable", exc_info=True)
    m5_range = getattr(analysis, "m5_range", None)
    if m5_range is not None:  # the breakout trigger / structure-stop levels of the playbook are real levels too
        anchors.append((f"M5 {m5_range.bars}-bar range high {m5_range.high:.5g}", m5_range.high, m5_range.high))
        anchors.append((f"M5 {m5_range.bars}-bar range low {m5_range.low:.5g}", m5_range.low, m5_range.low))
    anchors.append((f"live bid {bid:.5g}", bid, bid))
    anchors.append((f"live ask {ask:.5g}", ask, ask))
    return anchors


def _signed_drift(side: str, entry: float, bid: float, ask: float) -> float:
    """Distance from the live quote to the entry (price units): > 0 when the entry is on the pullback side
    (below the ask for a buy, above the bid for a sell), < 0 when the market is already THROUGH it."""
    return (ask - entry) if side == "buy" else (entry - bid)


def _classify(side: str, entry: float, stop: float, target: float, bid: float, ask: float, atr: float | None) -> tuple[str, float | None]:
    if side == "buy":
        if bid <= stop:
            return "STOP_BREACHED", None
        if bid >= target:
            return "TARGET_PASSED", None
    else:
        if ask >= stop:
            return "STOP_BREACHED", None
        if ask <= target:
            return "TARGET_PASSED", None
    if not atr or atr <= 0:
        return "NO_LIVE_DATA", None
    drift = _signed_drift(side, entry, bid, ask) / atr
    if abs(drift) > config.RECHECK_DRIFT_ATR:
        return "DRIFTED", drift
    if drift <= 0:
        return "MARKETABLE", drift
    return "ok", drift


def _status_sentence(r: SymbolRecheck) -> str:
    if r.status == "STOP_BREACHED":
        return f"STATUS: STOP ALREADY BREACHED by the live price ({r.stop:.5g}) — the plan's stop is dead."
    if r.status == "TARGET_PASSED":
        return f"STATUS: TARGET ALREADY PASSED by the live price ({r.target:.5g}) — the move this entry was for has happened."
    if r.status == "NO_LIVE_DATA":
        return "STATUS: no fresh M5 ATR available — leave this entry exactly as drafted."
    where = "below the live ask" if r.side == "buy" else "above the live bid"
    if r.status == "DRIFTED":
        if r.drift_atr is not None and r.drift_atr > 0:
            return (
                f"STATUS: DRIFTED — the entry sits {r.drift_atr:.1f} M5 ATR {where} (limit {config.RECHECK_DRIFT_ATR:g}); "
                "a pullback that far is now needed to fill it."
            )
        return (
            f"STATUS: DRIFTED — the live price is already {abs(r.drift_atr or 0):.1f} M5 ATR THROUGH the entry on the "
            "wrong side (the level the thesis relied on has failed)."
        )
    if r.status == "MARKETABLE":
        return "STATUS: marketable — the live price is already at/through the entry, it would fill immediately at the market."
    return f"STATUS: ok — the entry sits {r.drift_atr:.1f} M5 ATR {where}; leave as drafted unless the audit independently required a change."


def _zone_lines(analysis, side: str) -> str:
    """Only the fresh M5 trade-zone candidate for THIS side, produced by the same advisory function the main
    context uses (real S/R + session extremes, stop floor respected, targets flagged past the reach limit)."""
    from ai.ftmo_suggest import format_trade_zone
    from analysis.setup_classifier import classify_setups
    from analysis.timeframe_profiles import M5_PROFILE

    signals = classify_setups(analysis.m5_stats, analysis.m5_structure, analysis.m5_divergence, profile=M5_PROFILE)
    text = format_trade_zone(analysis, signals)
    keep = [line.strip() for line in text.splitlines() if line.strip().startswith(f"{side} ")]
    return keep[0] if keep else f"{side}: no real structure/reward-risk currently clears the bar"


def _entry_mode_menu(r: SymbolRecheck) -> list[str]:
    """What each entry_mode would actually do for this symbol RIGHT NOW, decided by the same deterministic
    function the Clerk applies before sending (analysis.entry_mode.resolve_entry_mode) — so Claude chooses from
    verified options instead of guessing whether a market/breakout entry would be accepted."""
    from analysis.entry_mode import fill_odds_pct, resolve_entry_mode

    if r.kind != "immediate" or r.bid is None or r.ask is None or not r.atr or r.status in ("STOP_BREACHED", "TARGET_PASSED", "NO_LIVE_DATA"):
        return []
    is_buy = r.side == "buy"
    live = r.ask if is_buy else r.bid
    lines = ["  ENTRY MODES available now (Python-checked against the live quote; pick one with the \"entry_mode\" field of this symbol's JSON object):"]
    if r.drift_atr is not None:
        odds = fill_odds_pct(max(r.drift_atr, 0.0))
        lines.append(
            f"   - \"limit\" at your drafted {r.entry:.5g}: {max(r.drift_atr, 0.0):.1f} M5 ATR from the market -> ~{odds:.0f}% chance it is touched within 3h."
        )
    floor = r.floor_distance or 0.0
    if floor > 0:
        stop_probe = live - floor if is_buy else live + floor
        market = resolve_entry_mode(
            requested="market", side=r.side, planned_price=live, stop_loss=stop_probe, take_profit=None,
            bid=r.bid, ask=r.ask, atr=r.atr,
        )
        verdict = "available" if market.mode == "market" else f"NOT available ({market.reason})"
        lines.append(
            f"   - \"market\" at the live {'ask' if is_buy else 'bid'} {live:.5g}: {verdict}. Needs a stop beyond a printed level, a "
            f"target on a printed level, net reward:risk >= {config.MIN_NET_REWARD_RISK_RATIO:g} recomputed FROM the live price, and the "
            f"live price within {config.MARKET_ENTRY_MAX_SLIPPAGE_ATR:g} M5 ATR worse than the entry you state."
        )
    want = "resistance" if is_buy else "support"
    ahead = []
    for label, lo, hi in r.anchors:
        if f"M5 {want} band" not in label:
            continue
        trigger = hi if is_buy else lo
        ahead_by = (trigger - r.ask) if is_buy else (r.bid - trigger)
        if ahead_by > 0:
            ahead.append((ahead_by, label, trigger))
    if ahead:
        ahead_by, label, trigger = min(ahead)
        probe = resolve_entry_mode(
            requested="stop", side=r.side, planned_price=trigger, stop_loss=(live - floor if is_buy else live + floor) if floor > 0 else (live * 0.99 if is_buy else live * 1.01),
            take_profit=None, bid=r.bid, ask=r.ask, atr=r.atr,
        )
        verdict = f"accepted as a {r.side} stop" if probe.mode == "stop" else f"NOT accepted ({probe.reason})"
        lines.append(
            f"   - \"stop\" (breakout trigger) at the nearest printed {want} edge {trigger:.5g} ({label}), {ahead_by / r.atr:.1f} M5 ATR from the "
            f"market: {verdict}. Use it only when the thesis is a breakout THROUGH that level; the stop then goes back inside it."
        )
    else:
        lines.append(f"   - \"stop\": no printed {want} band ahead of the market to break through.")
    if r.playbook_text:
        lines.append(f"   - PLAYBOOK (with-the-trend breakout, Python-measured from this fresh read): {r.playbook_text}")
    return lines


def _symbol_text(r: SymbolRecheck, analysis) -> str:
    from ai.ftmo_suggest import format_chart_structure, format_intraday_levels

    lines = [
        f"- {r.symbol} ({r.kind} {r.side}, pct {r.pct:g}): drafted entry {r.entry:.5g} / stop {r.stop:.5g} / target {r.target:.5g}.",
    ]
    if r.bid is not None and r.ask is not None:
        moved = ""
        if r.snapshot_price:
            mid = (r.bid + r.ask) / 2
            change = (mid - r.snapshot_price) / r.snapshot_price * 100
            atr_moves = f" = {(mid - r.snapshot_price) / r.atr:+.1f} M5 ATR" if r.atr else ""
            moved = f"; the price in your data snapshot was ~{r.snapshot_price:.5g}, so it has moved {change:+.2f}%{atr_moves} since"
        atr_txt = f"M5 ATR {r.atr:.5g} ({r.atr / ((r.bid + r.ask) / 2) * 100:.3f}% of price)" if r.atr else "M5 ATR n/a"
        floor_txt = f"; minimum stop distance (M5-ATR/spread/broker floor) {r.floor_distance:.5g}" if r.floor_distance else ""
        lines.append(f"  LIVE: bid {r.bid:.5g} / ask {r.ask:.5g}; {atr_txt}{floor_txt}{moved}.")
    lines.append(f"  {_status_sentence(r)}")
    if analysis is not None and r.status not in ("NO_LIVE_DATA",):
        lines.append(f"  {format_chart_structure('FRESH M5', analysis.m5_structure, compact=True).strip()}")
        lines.append(f"  {format_intraday_levels(r.symbol, analysis.intraday_levels).strip()}")
        lines.append(f"  Fresh M5 trade-zone candidate: {_zone_lines(analysis, r.side)}")
    lines.extend(_entry_mode_menu(r))
    return "\n".join(lines)


_RULES = (
    "FINAL LIVE RE-CHECK OF YOUR ENTRIES (measured by Python from the live MT5 feed at {when} UTC, about {age} "
    "minutes after the data snapshot your draft was built from — a real-time comparison of each drafted entry "
    "with where price actually is now).\n"
    "WHY: a resting limit order fills only if price comes back to it, and the data behind your draft ages while "
    "you research and revise. {odds}.\n"
    "STRICT RULES for this step:\n"
    "1. The numbers below are the ONLY valid source for the current price, ATR, levels and distances in this "
    "step. Do NOT use WebSearch/WebFetch, memory, or your own arithmetic on older numbers to state a current "
    "price or level — web prices are delayed and unreliable, and a level you compute yourself is not a level "
    "the chart has. If a number you want is not printed here, you do not have it: leave that entry exactly as "
    "drafted, or drop it.\n"
    "2. Only entries marked DRIFTED, STOP ALREADY BREACHED or TARGET ALREADY PASSED need a decision. Leave every "
    "other entry exactly as drafted unless the audit independently required a change.\n"
    "3. For a DRIFTED immediate entry pick exactly ONE and say which in that symbol's reason: (A) RE-ISSUE — a "
    "fresh entry within ~1.6 M5 ATR of the live price sitting on a level printed below (a fresh S/R band edge, a "
    "Fibonacci level, a session level, or the live price itself); a stop beyond the next printed level and never "
    "tighter than the printed minimum stop distance; a target on a printed level within ~6 M5 ATR; then "
    "RECOMPUTE the reward:risk from those printed numbers — do NOT just move the entry and keep the old target "
    "(the ratio collapses and the Clerk's floor rejects it); (B) MOVE it to Pending Setups with a mechanical M5 "
    "trigger at a printed level; (C) DROP it (pct 0) if the thesis you wrote is no longer supported by the "
    "fresh structure. Never raise pct.\n"
    "4. If the live price has already passed the stop or the target, re-issue per (A) or drop per (C); never "
    "carry the dead levels forward.\n"
    "5. Every re-issued entry is mechanically re-verified by Python against a NEW live quote and these same "
    "printed levels: distance from the price at most {reissue:g} M5 ATR (for a market or breakout-stop entry: "
    "the market-entry / breakout rules of its ENTRY MODES menu), entry and target each within a "
    "tolerance of a printed level, stop no tighter than the printed minimum, target within the reach limit, net "
    "reward:risk at least {rr:g}, pct not raised. A re-issue that fails is discarded and your drafted entry is "
    "kept (or dropped if its levels are dead) — so cite the printed level behind each number.\n"
    "5b. ENTRY MODES (optional \"entry_mode\" field in an immediate entry's JSON object, default \"limit\"): a "
    "resting pullback limit is the WRONG order when the market may not come back (the measured fill odds above; "
    "books: Murphy 'dynamic markets don't give a second chance', O'Neil 'buy the pivot, not the cheapest price'). "
    "When a thesis is a continuation or a breakout and the ENTRY MODES menu below shows \"market\" or \"stop\" "
    "as available, you MAY choose it — the Clerk re-verifies it against the live quote at send time and falls "
    "back to a limit (or drops a dead setup) if it no longer holds. Never invent a trigger: a \"stop\" trigger "
    "must be a printed level.\n"
    "6. Write \"re-checked live at {when} UTC\" in the reason of every symbol you change.\n"
    "Keep the exact output format required above (the same headers and JSON blocks)."
)


# --- the two public halves --------------------------------------------------------------------------


_SHORTLIST_LINE = re.compile(r"^\s+\d+\.\s+(\S+)\s+(BUY|SELL)\s+\[tier\s+(\w)\]\s+-\s+(.*)$", re.MULTILINE)


def _shortlist_from_summary(summary: str) -> list[tuple[str, str, str, str]]:
    """(symbol, side, tier, notes) for every numbered line of the POSITION HUNT block in the data summary."""
    start = summary.find("POSITION HUNT")
    if start < 0:
        return []
    block = summary[start:]
    end = block.find("Vetoed (hard")
    if end >= 0:
        block = block[:end]
    return [(m.group(1), m.group(2).lower(), m.group(3), m.group(4)) for m in _SHORTLIST_LINE.finditer(block)]


def _shortlist_gaps_block(gaps: list[tuple[str, str, str, str]]) -> str:
    """The prompt paragraph naming shortlisted candidates the draft left out entirely (deterministic - no
    reasoning is parsed): Claude must include each or keep it out with a checkable reason."""
    lines = [
        "SHORTLIST GAPS (Python cross-check of your draft against the POSITION HUNT shortlist): these candidates are "
        "on the shortlist but appear nowhere in your draft's allocation or Pending Setups:"
    ]
    for symbol, side, tier, notes in gaps:
        lines.append(f"- {symbol} {side.upper()} [tier {tier}] - {notes}")
    lines.append(
        "For each one either ADD it (levels ONLY from the printed structure in your data; choose its entry_mode from where "
        "the market is; the same numeric rules as any entry) or keep it out with ONE specific, checkable reason in the "
        "Asset-Class Outlook - a veto id (V1-V7) that applies to it right now, or a printed fact (an event time, a level, a "
        "number). 'Weak setup', 'no edge' or silence is not a reason, and a NO-INFORMATION backtest never is. Do not invent a "
        "number to justify either choice."
    )
    return "\n".join(lines)


def build_live_recheck(
    draft: str, summary: str, now_utc: datetime | None = None, snapshot_utc: datetime | None = None
) -> RecheckSheet | None:
    """`snapshot_utc`: when the data behind the draft was gathered (the caller's own start time), so the
    prompt can state the real age instead of a guess. None whenever this step cannot be done honestly (feature off, unparseable draft, no live data at all)
    — the caller then revises exactly as it always did."""
    if not config.LIVE_RECHECK_ENABLED:
        return None
    from ai.ftmo_suggest import analyze_ftmo_asset_live, effective_stop_floor
    from ai.portfolio_suggest import parse_final_allocation, parse_pending_setups
    from data.mt5_source import get_market_watch, get_open_positions

    allocation = parse_final_allocation(draft, require_side=True)
    if not allocation:
        return None
    immediate = {
        s: e for s, e in allocation.items()
        if s != "CASH" and e.pct > 0 and e.price and e.stop_loss and e.take_profit
    }
    pending = parse_pending_setups(draft, immediate_symbols=frozenset(immediate)) or []
    pending = [p for p in pending if p.price and p.stop_loss and p.take_profit]

    now_utc = now_utc or datetime.now(timezone.utc)
    held = {p.symbol for p in get_open_positions()}
    covered = set(immediate) | {p.symbol for p in pending} | held | {
        s for s, e in allocation.items() if s != "CASH" and e.pct > 0
    }
    gaps = [item for item in _shortlist_from_summary(summary) if item[0] not in covered]
    gaps_block = _shortlist_gaps_block(gaps) if gaps else ""
    if not immediate and not pending:
        if not gaps_block:
            return None
        return RecheckSheet(now_utc.isoformat(), {}, gaps_block)
    quotes = {a.symbol: a for a in get_market_watch()}
    snapshot = _snapshot_prices(summary)

    symbols: dict[str, SymbolRecheck] = {}
    blocks: list[str] = []
    entries = [(s, "immediate", e.side, e.price, e.stop_loss, e.take_profit, e.pct) for s, e in immediate.items()] + [
        (p.symbol, "pending", p.side, p.price, p.stop_loss, p.take_profit, p.pct) for p in pending
    ]
    for symbol, kind, side, entry, stop, target, pct in entries:
        asset = quotes.get(symbol)
        rec = SymbolRecheck(symbol, kind, side, entry, stop, target, pct, "NO_LIVE_DATA", snapshot_price=snapshot.get(symbol))
        analysis = None
        if symbol in held and kind == "immediate":
            rec.status = "HELD"
            symbols[symbol] = rec
            continue
        if asset is not None:
            try:
                analysis = analyze_ftmo_asset_live(symbol, asset.bid, asset.ask, asset.description)
                rec.bid, rec.ask = asset.bid, asset.ask
                rec.atr = analysis.m5_stats.atr
                floor = effective_stop_floor(analysis, entry)
                rec.floor_distance = floor.distance if floor is not None else None
                rec.anchors = _anchors_from_analysis(analysis, asset.bid, asset.ask)
                if config.PLAYBOOK_ENABLED and getattr(analysis, "m5_range", None) is not None:
                    from analysis.playbook import advise as playbook_advise

                    rec.playbook_text = playbook_advise(
                        side, asset.bid, asset.ask, analysis.m5_stats.atr, getattr(analysis.m5_stats, "adx", None),
                        getattr(analysis.m5_stats, "adx_change_12", None), analysis.m5_range,
                    ).text()
                if kind == "immediate":
                    rec.status, rec.drift_atr = _classify(side, entry, stop, target, asset.bid, asset.ask, rec.atr)
                else:
                    # A pending setup waits for its own trigger, so being far from the price is its normal state; only
                    # levels that are already dead matter.
                    rec.status, rec.drift_atr = _classify(side, entry, stop, target, asset.bid, asset.ask, rec.atr)
                    if rec.status in ("DRIFTED", "MARKETABLE"):
                        rec.status = "ok"
            except Exception:
                logger.warning("Live re-check: could not read fresh data for %s (leaving it as drafted).", symbol, exc_info=True)
                rec.status = "NO_LIVE_DATA"
                analysis = None
        rec.text = _symbol_text(rec, analysis)
        symbols[symbol] = rec
        blocks.append(rec.text)

    if not blocks:
        return RecheckSheet(now_utc.isoformat(), symbols, gaps_block) if gaps_block else None
    age = f"{max((now_utc - snapshot_utc).total_seconds() / 60, 0):.0f}" if snapshot_utc else "20-30"
    when = now_utc.strftime("%H:%M")
    header = _RULES.format(
        when=when, age=age, odds=_FILL_ODDS_TEXT, reissue=config.RECHECK_REISSUE_MAX_ATR,
        rr=config.MIN_NET_REWARD_RISK_RATIO,
    )
    flagged = [s for s, r in symbols.items() if r.status in ("DRIFTED", "TARGET_PASSED", "STOP_BREACHED")]
    tail = (
        f"\nSummary: {len(flagged)} entr{'y' if len(flagged) == 1 else 'ies'} need a decision ({', '.join(flagged)})."
        if flagged else "\nSummary: no entry has drifted beyond the limit — leave your entries as drafted."
    )
    body = header + "\n\n" + "\n".join(blocks) + tail
    if gaps_block:
        body += "\n\n" + gaps_block
    return RecheckSheet(now_utc.isoformat(), symbols, body)


def _near_anchor(price: float, anchors, tol: float) -> str | None:
    for label, lo, hi in anchors:
        if lo - tol <= price <= hi + tol:
            return label
    return None


def _check_levels(
    side: str, entry: float, stop: float, target: float, rec: SymbolRecheck, bid: float, ask: float,
    draft_levels: tuple | None, require_near_market: bool, entry_mode: str = "limit",
) -> tuple[list[str], list[str]]:
    """(violations, verified-notes) for one candidate entry against fresh real data."""
    problems: list[str] = []
    notes: list[str] = []
    atr = rec.atr or 0.0
    spread = max(ask - bid, 0.0)
    if side == "buy":
        if not stop < entry < target:
            problems.append("stop/entry/target are not ordered stop < entry < target for a buy")
    elif not target < entry < stop:
        problems.append("stop/entry/target are not ordered target < entry < stop for a sell")
    if problems or atr <= 0:
        return problems or ["no fresh ATR to verify against"], notes

    tol = max(config.RECHECK_ANCHOR_TOLERANCE_ATR * atr, 2 * spread)
    dist_atr = _signed_drift(side, entry, bid, ask) / atr
    if require_near_market and entry_mode == "stop":
        # A breakout trigger sits BEYOND the market (dist < 0), or just through it (the market entry rule applies).
        if not (-config.STOP_ENTRY_MAX_DISTANCE_ATR <= dist_atr <= config.STOP_ENTRY_MAX_EXTENSION_ATR):
            problems.append(
                f"stop-order trigger sits {dist_atr:+.1f} M5 ATR from the live price (allowed -{config.STOP_ENTRY_MAX_DISTANCE_ATR:g} to "
                f"+{config.STOP_ENTRY_MAX_EXTENSION_ATR:g})"
            )
    elif require_near_market and entry_mode == "market":
        if abs(dist_atr) > config.MARKET_ENTRY_MAX_SLIPPAGE_ATR:
            problems.append(f"market entry states {dist_atr:+.1f} M5 ATR from the live price (allowed +/-{config.MARKET_ENTRY_MAX_SLIPPAGE_ATR:g})")
    elif require_near_market and not (-0.25 <= dist_atr <= config.RECHECK_REISSUE_MAX_ATR):
        problems.append(f"entry sits {dist_atr:+.1f} M5 ATR from the live price (allowed -0.25 to {config.RECHECK_REISSUE_MAX_ATR:g})")

    risk = abs(entry - stop)
    if rec.floor_distance and risk < rec.floor_distance * 0.999:
        problems.append(f"stop distance {risk:.5g} is tighter than the minimum {rec.floor_distance:.5g}")
    if risk > config.RECHECK_MAX_STOP_ATR * atr:
        problems.append(f"stop distance {risk / atr:.1f} ATR is wider than {config.RECHECK_MAX_STOP_ATR:g} ATR")

    reward = abs(target - entry)
    if reward / atr > config.M5_TARGET_REACH_ATR_LIMIT + 1e-9:
        problems.append(f"target is {reward / atr:.1f} M5 ATR away (reach limit {config.M5_TARGET_REACH_ATR_LIMIT:g})")
    net_rr = (reward - spread) / risk if risk > 0 else 0.0
    if net_rr < config.MIN_NET_REWARD_RISK_RATIO:
        problems.append(f"net reward:risk {net_rr:.2f} is below {config.MIN_NET_REWARD_RISK_RATIO:g}")

    for name, price, draft_price in (("entry", entry, draft_levels[0] if draft_levels else None), ("target", target, draft_levels[2] if draft_levels else None)):
        if draft_price is not None and abs(price - draft_price) <= 1e-9:
            notes.append(f"{name} unchanged from the draft")
            continue
        label = _near_anchor(price, rec.anchors, tol)
        if label is None:
            problems.append(f"{name} {price:.5g} is not within tolerance ({tol:.4g}) of any real level in the fresh read")
        else:
            notes.append(f"{name} anchored to {label}")
    return problems, notes


def apply_recheck_validation(
    payload: dict,
    draft_allocation: dict,
    draft_pending: list,
    sheet: RecheckSheet,
    fresh_quotes: dict[str, tuple[float, float]] | None = None,
) -> list[str]:
    """Verifies, in place on the latest-suggestion `payload`, every entry Claude CHANGED versus the draft for a
    symbol covered by the re-check sheet. A change that fails is reverted to the draft's levels (or dropped when
    those are already dead); a change that passes is annotated with what verified it. Returns human-readable
    notes (also logged)."""
    notes: list[str] = []
    fresh_quotes = fresh_quotes or {}

    def _quote(rec: SymbolRecheck) -> tuple[float, float] | None:
        if rec.symbol in fresh_quotes:
            return fresh_quotes[rec.symbol]
        if rec.bid is not None and rec.ask is not None:
            return rec.bid, rec.ask
        return None

    def _draft_levels(symbol: str, kind: str):
        if kind == "immediate":
            e = draft_allocation.get(symbol)
            return (e.price, e.stop_loss, e.take_profit, e.pct, e.side, normalise_entry_mode(e.entry_mode)) if e is not None else None
        p = next((x for x in draft_pending if x.symbol == symbol), None)
        return (p.price, p.stop_loss, p.take_profit, p.pct, p.side, "limit") if p is not None else None

    def _annotate(item: dict, text: str) -> None:
        item["reason"] = f"{item.get('reason', '')} [Live re-check: {text}]".strip()

    for symbol, rec in sheet.symbols.items():
        if rec.status in ("HELD",):
            continue
        quote = _quote(rec)
        draft = _draft_levels(symbol, rec.kind)
        if draft is None or quote is None:
            continue
        d_price, d_stop, d_tp, d_pct, d_side, d_mode = draft
        candidates = []  # (where, item)
        immediate_item = payload.get("immediate_allocation", {}).get(symbol)
        if immediate_item and (immediate_item.get("pct") or 0) > 0:
            candidates.append(("immediate", immediate_item))
        for item in payload.get("pending_setups", []):
            if item.get("symbol") == symbol:
                candidates.append(("pending", item))
        if not candidates:
            if rec.status in ("DRIFTED", "TARGET_PASSED", "STOP_BREACHED"):
                notes.append(f"{symbol}: was {rec.status} and Claude dropped it — nothing to verify.")
            continue

        for where, item in candidates:
            f_price, f_stop, f_tp = item.get("price"), item.get("stop_loss"), item.get("take_profit")
            f_pct, f_side = item.get("pct") or 0.0, item.get("side")
            pct_clamped_note = ""
            if f_pct > d_pct + 1e-9 and d_pct > 0 and f_side == d_side:
                # The re-check may never RAISE risk - but throwing the whole re-issue away for that alone lost a good
                # META pullback entry (2026-09-25: 0.4% -> 0.55% rejected, the drafted levels were dead, so the symbol
                # ended at 0% with no size at all). Clamp to the drafted risk and judge the levels on their merits.
                pct_clamped_note = f"pct clamped from {f_pct:g} to the drafted {d_pct:g}"
                item["pct"] = d_pct
                f_pct = d_pct
            f_mode = normalise_entry_mode(item.get("entry_mode"))
            changed = (
                f_mode != d_mode
                or f_side != d_side
                or not all(isinstance(x, (int, float)) for x in (f_price, f_stop, f_tp))
                or abs(f_price - d_price) > 1e-9 or abs(f_stop - d_stop) > 1e-9 or abs(f_tp - d_tp) > 1e-9
                or f_pct > d_pct + 1e-9 or where != rec.kind
            )
            if not changed:
                if rec.status in ("DRIFTED", "TARGET_PASSED", "STOP_BREACHED"):
                    notes.append(f"{symbol}: still {rec.status} — Claude kept the drafted levels unchanged.")
                continue

            problems: list[str] = []
            if where == "pending" and f_mode != "limit":
                problems.append("entry_mode applies to immediate entries only")
            if f_side != d_side:
                problems.append(f"side changed from {d_side} to {f_side}")
            if f_pct > d_pct + 1e-9:
                problems.append(f"pct raised from {d_pct:g} to {f_pct:g}")  # only when there is no drafted risk to clamp to
            if not problems and all(isinstance(x, (int, float)) for x in (f_price, f_stop, f_tp)):
                # Fresh quote, not the (few-minutes-old) sheet quote: the distance check must reflect NOW.
                bid, ask = quote
                more, verified = _check_levels(
                    f_side, f_price, f_stop, f_tp, rec, bid, ask, (d_price, d_stop, d_tp),
                    require_near_market=(where == "immediate"), entry_mode=f_mode,
                )
                problems += more
                if not more and f_mode != "limit" and where == "immediate":
                    from analysis.entry_mode import resolve_entry_mode

                    decision = resolve_entry_mode(
                        requested=f_mode, side=f_side, planned_price=f_price, stop_loss=f_stop, take_profit=f_tp,
                        bid=bid, ask=ask, atr=rec.atr,
                    )
                    if decision.mode != f_mode:
                        problems.append(f"entry_mode {f_mode!r} cannot be honoured on the live quote ({decision.reason})")
                    else:
                        verified.append(f"entry_mode {f_mode} confirmed on the live quote")
            elif not problems:
                problems.append("price/stop_loss/take_profit are not all numbers")
                verified = []
            else:
                verified = []

            if not problems:
                text = "re-issue verified against a fresh quote — " + "; ".join(verified) if verified else "verified"
                if pct_clamped_note:
                    text += f" ({pct_clamped_note})"
                _annotate(item, text)
                notes.append(
                    f"{symbol} ({where}): re-issue accepted ({'; '.join(verified)})"
                    + (f"; {pct_clamped_note}" if pct_clamped_note else "") + "."
                )
                continue

            dead = rec.status in ("STOP_BREACHED", "TARGET_PASSED")
            why = "; ".join(problems)
            if where != rec.kind:
                # A failed move between sections: undo it — the pending copy is removed and the drafted immediate
                # entry restored (or dropped when its levels are dead).
                if where == "pending":
                    payload["pending_setups"] = [p for p in payload["pending_setups"] if p is not item]
                    src = draft_allocation.get(symbol)
                    if src is not None:
                        restored = payload["immediate_allocation"].setdefault(symbol, {})
                        restored.update(
                            pct=0.0 if dead else d_pct, price=d_price, stop_loss=d_stop, take_profit=d_tp, side=d_side,
                            reason=src.reason, invalidation_condition=src.invalidation_condition, entry_mode=d_mode,
                        )
                        _annotate(restored, f"move to Pending Setups REJECTED ({why}) — " + ("dropped, drafted levels dead" if dead else "kept the drafted entry"))
                else:
                    # A failed move to an immediate entry: the immediate copy goes away ENTIRELY (a leftover 0% row would
                    # read as a close/cancel instruction and shows as a sizeless "target"), and the drafted Pending Setup
                    # is restored when its levels are still alive - otherwise the trade would silently vanish.
                    payload["immediate_allocation"].pop(symbol, None)
                    src_pending = next((x for x in draft_pending if x.symbol == symbol), None)
                    if src_pending is not None and not dead:
                        restored_pending = {
                            "symbol": src_pending.symbol, "side": src_pending.side, "pct": src_pending.pct,
                            "trigger_condition": src_pending.trigger_condition, "price": src_pending.price,
                            "stop_loss": src_pending.stop_loss, "take_profit": src_pending.take_profit,
                            "reason": src_pending.reason, "trigger": getattr(src_pending, "trigger", None),
                        }
                        _annotate(restored_pending, f"move to an immediate entry REJECTED ({why}) - kept the drafted Pending Setup")
                        payload.setdefault("pending_setups", []).append(restored_pending)
                    else:
                        notes.append(f"{symbol}: the drafted levels are dead, so nothing is left to watch or place.")
                notes.append(f"{symbol}: move {rec.kind}->{where} rejected ({why}).")
            elif dead:
                item["pct"] = 0.0
                _annotate(item, f"re-issue REJECTED ({why}) and the drafted levels are already dead — dropped")
                notes.append(f"{symbol} ({where}): re-issue rejected ({why}); drafted levels dead -> dropped.")
            else:
                item["price"], item["stop_loss"], item["take_profit"] = d_price, d_stop, d_tp
                item["pct"] = d_pct
                item["side"] = d_side
                item["entry_mode"] = d_mode
                _annotate(item, f"re-issue REJECTED ({why}) — kept the drafted levels")
                notes.append(f"{symbol} ({where}): re-issue rejected ({why}); kept the drafted levels.")

    for note in notes:
        logger.info("Live re-check: %s", note)
    return notes
