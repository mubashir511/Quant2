"""The Sentinel: a fast, LLM-free manager for what the Clerk's 15-minute poll cannot watch closely enough.

Why (2026-09-25 plan W5, user-approved): the Clerk reviews the account every few minutes and needs a local LLM for most
decisions; a profit trail that only ratchets once per poll gives back a fast move. The Sentinel runs every minute, needs no
model, and v1 does exactly ONE deterministic thing - the profit-trail ratchet (analysis/trail.py, the SAME helper the Clerk's
tactical signals use): once a position is at least CLERK_PROFIT_TRAIL_START_R of its original risk in profit, its stop
follows the price at CLERK_PROFIT_TRAIL_ATR x M5 ATR, only ever tighter, only by a real step. Measured on deep M5 history
(99,480 paired trades): the trail beat a fixed exit (-0.154R vs -0.176R, 18/20 symbols); breakeven-only, soft stops, time
stops and a TP1 partial did not help, so none of those are here.

How it coexists with the Clerk (which owns the settlement file and reverts any stop it did not record - see
ai/clerk_execution.py `_restore_external_stop_drift`):
  * The Sentinel holds the Clerk's own execution lock while it runs, so a Clerk poll and a Sentinel tick can never overlap.
  * After every stop it moves it records {ticket, stop, r0, sizing pct} in ITS OWN state file (never the settlement file).
  * At the start of every Clerk poll, `ingest_sentinel_stops` folds those stops into the settlement record as
    `tactical.sentinel_stop` / `sentinel_pct` / `profit_trail_r0`; the Clerk's carried-forward target and its drift check
    then treat the trailed stop as the recorded one (a tighter stop with the SAME lot count needs a smaller sizing pct, which
    is why the pct is recomputed with the Clerk's own `pct_for_target_lots`). A fresh mega session wipes tactical state, as
    it does for every other tactical action.
  * config.SENTINEL_LOG_ONLY (default True: the approved log-only phase) computes and logs what it WOULD do without touching
    any stop. Set SENTINEL_LOG_ONLY=0 to let it act.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import config
from analysis.trail import original_risk, trail_decision

logger = logging.getLogger(__name__)


@dataclass
class TrailAction:
    symbol: str
    ticket: int
    side: str
    old_stop: float
    new_stop: float
    take_profit: float | None
    r0: float
    progress_r: float
    sizing_pct: float


def _state_path() -> Path:
    return Path(config.SENTINEL_STATE_FILE)


def read_sentinel_state() -> dict:
    path = _state_path()
    if not path.exists():
        return {"stops": {}}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"stops": {}}
    if not isinstance(data, dict) or not isinstance(data.get("stops"), dict):
        return {"stops": {}}
    return data


def write_sentinel_state(state: dict) -> None:
    path = _state_path()
    try:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, path)
    except OSError as e:
        logger.warning("Could not write sentinel state %s: %s", path, e)


def _recorded_stop(rec: dict) -> float | None:
    tactical = rec.get("tactical") or {}
    for value in (tactical.get("sentinel_stop"), tactical.get("persisted_stop_loss"), (rec.get("entry") or {}).get("stop_loss")):
        if value is not None:
            return float(value)
    return None


def plan_trail_actions(
    positions: list,
    settled: dict,
    atr_by_symbol: dict[str, float | None],
    min_stop_pct_by_symbol: dict[str, float],
    state_stops: dict,
    equity: float,
    get_spec: Callable[[str], object | None],
    pct_for_target_lots: Callable,
) -> list[TrailAction]:
    """One TrailAction per Clerk-managed position whose trail says the stop should move. A position with no settlement
    record is not this app's to manage (the Clerk backfills real positions on its next poll); a position whose new stop
    could not be sized would be reverted by the Clerk's drift restore, so it is skipped rather than fought over."""
    actions: list[TrailAction] = []
    for position in positions:
        rec = settled.get(position.symbol)
        if rec is None or position.sl is None:
            continue
        tactical = rec.get("tactical") or {}
        prior = state_stops.get(position.symbol) or {}
        persisted_r0 = tactical.get("profit_trail_r0") or (prior.get("r0") if prior.get("ticket") == position.ticket else None)
        recorded = _recorded_stop(rec)
        r0 = original_risk(position.price_open, (rec.get("entry") or {}).get("stop_loss"), position.sl, persisted_r0)
        decision = trail_decision(
            position.side, position.price_open, position.price_current, position.sl, r0,
            atr_by_symbol.get(position.symbol), min_stop_pct_by_symbol.get(position.symbol, 0.0),
        )
        if decision.new_stop is None:
            continue
        pct = pct_for_target_lots(
            position.symbol, position.volume, position.price_open, decision.new_stop, equity, get_spec
        )
        if pct is None:
            logger.info("%s: trail to %.5f skipped - the new stop cannot be sized for the Clerk's ledger.", position.symbol, decision.new_stop)
            continue
        own_stop = prior.get("stop") if prior.get("ticket") == position.ticket else None  # not yet ingested by the Clerk
        known_stops = [v for v in (recorded, own_stop) if v is not None]
        if known_stops and all(
            abs(position.sl - v) / max(abs(v), 1e-12) * 100 > config.AMEND_TOLERANCE_PCT for v in known_stops
        ):
            # The live stop is not what this app last recorded (an outside change, or a Clerk amend still pending):
            # leave it to the Clerk's own drift logic rather than stacking a second change on top.
            logger.info("%s: live stop %.5f differs from the recorded %.5f - Sentinel leaves it to the Clerk.", position.symbol, position.sl, recorded)
            continue
        actions.append(TrailAction(
            symbol=position.symbol, ticket=position.ticket, side=position.side, old_stop=position.sl,
            new_stop=decision.new_stop, take_profit=position.tp, r0=decision.r0 or r0 or 0.0,
            progress_r=decision.progress_r or 0.0, sizing_pct=pct,
        ))
    return actions


def ingest_sentinel_stops(
    settled: dict,
    positions_by_symbol: dict,
    state_stops: dict,
    equity: float,
    get_spec: Callable[[str], object | None],
    pct_for_target_lots: Callable,
) -> tuple[dict, list[str]]:
    """Folds the Sentinel's trailed stops into the settlement records (called at the start of a Clerk poll, before the
    carried-forward target and the drift check). Only a stop that the live position really carries (same ticket, same
    price within the amend tolerance) and that is MORE protective than the recorded one is ingested; the sizing pct is
    recomputed for the position's own lot count at that stop so the Clerk's rebalance sees "already at target"."""
    notes: list[str] = []
    for symbol, item in (state_stops or {}).items():
        position = positions_by_symbol.get(symbol)
        rec = settled.get(symbol)
        if position is None or rec is None or position.sl is None:
            continue
        try:
            stop = float(item["stop"])
            ticket = int(item["ticket"])
        except (KeyError, TypeError, ValueError):
            continue
        if ticket != position.ticket:
            continue
        if abs(position.sl - stop) / max(abs(stop), 1e-12) * 100 > config.AMEND_TOLERANCE_PCT:
            continue
        recorded = _recorded_stop(rec)
        sign = 1.0 if position.side == "buy" else -1.0
        if recorded is not None and sign * (stop - recorded) <= 0:
            continue  # already recorded (or the Clerk has since moved it tighter)
        pct = pct_for_target_lots(symbol, position.volume, position.price_open, stop, equity, get_spec)
        if pct is None:
            notes.append(f"{symbol}: Sentinel stop {stop:.5f} could not be sized - left for the drift check")
            continue
        tactical = dict(rec.get("tactical") or {})
        tactical.update({
            "sentinel_stop": stop,
            "sentinel_pct": pct,
            "sentinel_utc": item.get("utc"),
            "profit_trail_r0": item.get("r0") or tactical.get("profit_trail_r0"),
        })
        rec["tactical"] = tactical
        notes.append(f"{symbol}: ingested the Sentinel's trailed stop {stop:.5f} (sizing {pct:.3f}%)")
    return settled, notes


def apply_sentinel_ratchet(entry, tactical: dict | None):
    """The carried-forward target with the Sentinel's trailed stop and its matching sizing pct, when one is recorded and
    it is more protective than the entry's own stop; otherwise the entry unchanged. Used at every place the Clerk builds
    a target from a settlement record."""
    tactical = tactical or {}
    stop, pct = tactical.get("sentinel_stop"), tactical.get("sentinel_pct")
    if stop is None or pct is None or entry is None:
        return entry
    if entry.stop_loss is not None:
        sign = 1.0 if entry.side == "buy" else -1.0
        if sign * (stop - entry.stop_loss) <= 0:
            return entry
    from dataclasses import replace

    return replace(entry, stop_loss=float(stop), pct=float(pct))


def _real_collaborators() -> dict:
    """The live MT5-backed collaborators (imported lazily so importing this module never needs the terminal)."""
    from analysis.technical import compute_atr
    from ai.clerk_execution import _load_settlement
    from data.mt5_execution import modify_position_sltp
    from data.mt5_source import (
        fetch_mt5_price_history, get_account_summary, get_contract_spec, get_open_positions, get_trade_economics,
    )

    def min_stop_pct(symbol: str) -> float:
        cost = get_trade_economics(symbol)
        return cost.min_stop_distance_pct if cost is not None else 0.0

    return {
        "positions_fn": get_open_positions,
        "settlement_fn": _load_settlement,
        "atr_fn": lambda symbol: compute_atr(fetch_mt5_price_history(symbol, "M5", count=120)),
        "min_stop_pct_fn": min_stop_pct,
        "equity_fn": lambda: get_account_summary().equity,
        "modify_fn": modify_position_sltp,
        "spec_fn": get_contract_spec,
    }


def run_sentinel_check(
    now_utc: datetime | None = None, *, log_only: bool | None = None, **collaborators
) -> dict:
    """One Sentinel tick. `collaborators` (positions_fn, settlement_fn, atr_fn, min_stop_pct_fn, equity_fn, modify_fn,
    spec_fn) are injectable for tests; anything not given is the real MT5-backed one. Returns a summary."""
    from risk.apply_suggestion import pct_for_target_lots

    now_utc = now_utc or datetime.now(timezone.utc)
    log_only = config.SENTINEL_LOG_ONLY if log_only is None else log_only
    deps = {**_real_collaborators(), **collaborators} if len(collaborators) < 7 else collaborators

    summary = {"checked_utc": now_utc.isoformat(), "log_only": log_only, "positions": 0, "actions": []}
    settled = (deps["settlement_fn"]() or {}).get("settled", {})
    all_positions = deps["positions_fn"]()
    positions = [p for p in all_positions if p.symbol in settled]
    summary["positions"] = len(positions)
    state = read_sentinel_state()
    held = {p.symbol for p in all_positions}
    state["stops"] = {sym: item for sym, item in state["stops"].items() if sym in held}  # forget closed positions
    if not positions:
        write_sentinel_state(state)
        return summary

    atr_by_symbol, min_stop_by_symbol = {}, {}
    for p in positions:
        try:
            atr_by_symbol[p.symbol] = deps["atr_fn"](p.symbol)
            min_stop_by_symbol[p.symbol] = deps["min_stop_pct_fn"](p.symbol)
        except Exception:  # noqa: BLE001 - one bad symbol never blocks the others
            logger.warning("%s: Sentinel could not read its M5 data.", p.symbol, exc_info=True)
    equity = deps["equity_fn"]()
    actions = plan_trail_actions(
        positions, settled, atr_by_symbol, min_stop_by_symbol, state["stops"], equity, deps["spec_fn"], pct_for_target_lots
    )
    for a in actions:
        line = (
            f"{a.symbol} {a.side} +{a.progress_r:.2f}R: trail stop {a.old_stop:.5f} -> {a.new_stop:.5f} "
            f"(r0 {a.r0:.5f}, sizing {a.sizing_pct:.3f}%)"
        )
        if log_only:
            logger.info("Sentinel [log-only] would %s", line)
            summary["actions"].append({"symbol": a.symbol, "applied": False, "text": line})
            continue
        position = next(p for p in positions if p.symbol == a.symbol)
        result = deps["modify_fn"](position, a.new_stop, a.take_profit)
        if result.success:
            state["stops"][a.symbol] = {"ticket": a.ticket, "stop": a.new_stop, "r0": a.r0, "utc": now_utc.isoformat()}
            write_sentinel_state(state)  # right after the modify - the Clerk ingests it next poll
            logger.info("Sentinel applied %s", line)
        else:
            logger.warning("Sentinel could not apply %s - broker said %s (%s)", line, result.retcode, result.comment)
        summary["actions"].append({"symbol": a.symbol, "applied": bool(result.success), "text": line})
    state["last_run_utc"] = now_utc.isoformat()
    state["last_actions"] = summary["actions"]
    write_sentinel_state(state)
    return summary
