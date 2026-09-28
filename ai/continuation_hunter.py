"""Continuation Watch, PHASE 2: the orchestration around the phase-1 filter (analysis.continuation_watch).

Full design (2026-09-28, direct user plan, all seven points):
  1. Trigger: only a WINNING close (ai.clerk_execution registers a watch right where it already detects one
     — never a loser; chasing continuation after a loss is exactly how discipline breaks).
  2. Wait 2-3 real M5 candles (config.CONTINUATION_MAX_WAIT_MINUTES) before judging anything — real evidence,
     not the first noisy tick after the exit. This module's own job re-checks every minute; the waiting and
     the "give up" cutoff both live in analysis.continuation_watch.evaluate_continuation.
  3. A deterministic checklist decides whether it's even worth asking a model — the phase-1 filter, PLUS the
     same objective cost-drag veto (V1) the Position Hunter already uses, reused directly (not
     re-implemented) via ai.ftmo_suggest.cost_drag_r against a standard 2x-M5-ATR stop. A candidate that
     fails either gate is dropped without ever spending a model call.
  4. The local model judges the checklist's own facts plus a fresh, LEAN (M5-only) technical read of the
     symbol RIGHT NOW — never the original trade's old levels. If it's unavailable or its answer doesn't
     parse, escalate to one free OpenRouter model from ai.portfolio_suggest.AUDIT_MODEL_FALLBACKS (the exact
     cascade ai.curiosity.py already uses for a similar "escalate a single judgement call" need).
  5. The new trade this decision would open is never sent by a second, independently-coded order path — see
     point 6.
  6. LOG-ONLY today (config.CONTINUATION_WATCH_LOG_ONLY, default True): every decision (propose or skip) is
     written to the trade journal and this module's own log, in full, with the checklist facts and which
     model answered — but no order is ever placed. Wiring a "propose" decision into the SAME allocation/
     compute_rebalance_plan pipeline every other entry already goes through (so it automatically inherits
     the correlation guard, the ATR/R:R floors, sizing, and the kill switch) is PHASE 3 — deliberately not
     built yet; log-only observation comes first, exactly as it did for the Sentinel.
  7. Every event (registered, still-waiting exhausted, cost-vetoed, model decision) is written to the trade
     journal via ai.trade_journal.record_continuation_watch, attached to the SAME story that just closed.

One watch per symbol at a time — a fresh winning close on a symbol already being watched simply replaces the
old watch (same "a fresh event resets this symbol's tactical state" convention used everywhere else in this
project, e.g. a new Mega suggestion resetting the settlement file).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
from ai import trade_journal
from ai.ftmo_suggest import analyze_ftmo_asset_live, cost_drag_r, format_clerk_context
from ai.ollama_client import FAILED_MESSAGE as OLLAMA_FAILED_MESSAGE
from ai.ollama_client import run_ollama
from ai.openrouter_client import FAILED_MESSAGE as OPENROUTER_FAILED_MESSAGE
from ai.openrouter_client import MISSING_KEY_MESSAGE as OPENROUTER_MISSING_KEY_MESSAGE
from ai.openrouter_client import run_openrouter
from ai.portfolio_suggest import AUDIT_MODEL_FALLBACKS
from analysis.continuation_watch import ContinuationVerdict, evaluate_continuation
from analysis.technical import compute_atr
from data.mt5_source import fetch_mt5_price_history_range

logger = logging.getLogger(__name__)

_LEADING_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
_STANDARD_STOP_ATR_MULTIPLE = 2.0  # the same "realistic stop" multiple the hunter's own min-viable-size line uses


def _state_path() -> Path:
    return Path(config.CONTINUATION_WATCH_STATE_FILE)


def load_state() -> dict:
    try:
        data = json.loads(_state_path().read_text())
    except (OSError, json.JSONDecodeError):
        return {"watching": {}}
    if not isinstance(data, dict) or not isinstance(data.get("watching"), dict):
        return {"watching": {}}
    return data


def save_state(state: dict) -> None:
    path = _state_path()
    try:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, path)
    except OSError:
        pass  # a missing state file only means a watch is silently dropped, never a crash


def register_watch(
    symbol: str, story_id: str, side: str, exit_price: float, exit_time: datetime, atr_at_close: float | None,
) -> None:
    """Called once, right where ai.clerk_execution already detects a WINNING close — never for a loss (the
    caller's own responsibility, not checked again here)."""
    if atr_at_close is None or atr_at_close <= 0:
        logger.info("%s: no usable M5 ATR at the close — not registering a continuation watch.", symbol)
        return
    state = load_state()
    state["watching"][symbol] = {
        "story_id": story_id, "side": side, "exit_price": exit_price,
        "exit_time_utc": exit_time.isoformat(), "atr_at_close": atr_at_close,
        "registered_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_state(state)
    logger.info("%s: continuation watch registered (%s exit at %.6g, ATR %.6g).", symbol, side, exit_price, atr_at_close)


@dataclass
class ContinuationDecision:
    propose: bool
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    reason: str = ""
    raw_text: str = ""


def _parse_continuation_decision(raw: str) -> ContinuationDecision:
    """Tolerant of loose formatting, same convention as parse_tactical_verdict: a model that answers CONTINUE:
    no (or fails to give a parseable yes) is always read as "skip" — never the reverse. A "yes" without a
    usable entry/stop/target also degrades to "skip", since there is nothing safe to log as a proposal."""
    match = re.search(r"CONTINUE\s*:\s*(YES|NO)", raw, re.IGNORECASE)
    if not match or match.group(1).upper() != "YES":
        return ContinuationDecision(False, raw_text=raw)

    def _num(label: str) -> float | None:
        m = re.search(rf"{label}\s*:\s*([^\n]+)", raw, re.IGNORECASE)
        if not m:
            return None
        n = _LEADING_NUMBER_PATTERN.search(m.group(1))
        return float(n.group()) if n else None

    entry, stop, target = _num("ENTRY"), _num("STOP"), _num("TARGET")
    reason_match = re.search(r"REASON\s*:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else ""
    if entry is None or stop is None or target is None:
        return ContinuationDecision(False, reason="model said yes but gave no usable entry/stop/target", raw_text=raw)
    return ContinuationDecision(True, entry=entry, stop=stop, target=target, reason=reason, raw_text=raw)


def _build_prompt(symbol: str, watch: dict, verdict: ContinuationVerdict, drag_r: float, technical_context: str) -> str:
    side_word = "long (bought)" if watch["side"] == "buy" else "short (sold)"
    return (
        f"CONTINUATION WATCH — not a normal new-trade review. {symbol} was just closed in profit on a {side_word} trade "
        f"(exit {watch['exit_price']:g}). Since that exit, price has moved a further {verdict.move_atr:+.2f}x the 5-minute "
        f"ATR in the SAME direction within {verdict.minutes_since_close:.0f} minutes — a real, measured continuation, "
        "already confirmed by Python, not your judgement to re-derive. The round-trip trading cost at a standard "
        f"{_STANDARD_STOP_ATR_MULTIPLE:g}x-ATR stop is {drag_r:.2f}R (limit {config.COST_DRAG_VETO_R:g}R) — already checked, "
        "this candidate clears it.\n\n"
        f"{technical_context}\n\n"
        f"TASK: decide whether to open a FRESH, separate {watch['side']} continuation trade on {symbol} right now, using "
        "TODAY's structure above — never the original trade's old levels. This is a lower-conviction, opportunistic add, "
        "not a core position: only propose it if the evidence is genuinely still there at THIS moment, with a real "
        "structural stop and target, not a guess.\n\n"
        "Answer in exactly this format:\n"
        "CONTINUE: yes or no\n"
        "ENTRY: <price, only if yes>\n"
        "STOP: <price, only if yes -- a real structural level>\n"
        "TARGET: <price, only if yes>\n"
        "REASON: <one or two sentences, citing real numbers above>"
    )


def _ask_local_then_openrouter(prompt: str) -> tuple[str, str]:
    """(raw_text, model_used_label). Tries the Clerk's own local model first; escalates to one free
    OpenRouter model from AUDIT_MODEL_FALLBACKS only when the local model is down or its answer doesn't
    parse as a clean yes/no at all -- the same "local first, cloud escalation on a genuine failure" shape
    ai.curiosity.py already uses for a comparable single-judgement-call need."""
    local_raw = run_ollama(
        prompt, model=config.CLERK_PRIMARY_MODEL, timeout=config.CLERK_LLM_TIMEOUT_SECONDS,
        keep_alive=config.CLERK_LLM_KEEP_ALIVE, max_tokens=config.CLERK_LLM_MAX_TOKENS,
    )
    if local_raw != OLLAMA_FAILED_MESSAGE and re.search(r"CONTINUE\s*:\s*(YES|NO)", local_raw, re.IGNORECASE):
        return local_raw, config.CLERK_PRIMARY_MODEL

    logger.info("Continuation watch: local model unavailable or unclear -- escalating to OpenRouter.")
    deadline = time.monotonic() + config.CONTINUATION_WATCH_RETRY_TIMEOUT_SECONDS
    for label, model_id, _profile in AUDIT_MODEL_FALLBACKS:
        if time.monotonic() >= deadline:
            break
        try:
            result = run_openrouter(prompt, model=model_id, timeout=config.CONTINUATION_WATCH_MODEL_TIMEOUT_SECONDS)
        except Exception:
            result = OPENROUTER_FAILED_MESSAGE
        if result == OPENROUTER_MISSING_KEY_MESSAGE:
            break
        if result != OPENROUTER_FAILED_MESSAGE and re.search(r"CONTINUE\s*:\s*(YES|NO)", result, re.IGNORECASE):
            return result, f"openrouter:{label}"
    return "", "none (local and every OpenRouter fallback failed)"


def _resolve(symbol: str, watch: dict, state: dict, outcome: str, journal_data: dict) -> None:
    trade_journal.record_continuation_watch(watch["story_id"], outcome, journal_data)
    state["watching"].pop(symbol, None)


def run_continuation_watch_check(now: datetime | None = None) -> dict:
    """Cheap when nothing is being watched (no MT5 call, no model call). Returns a small summary dict for
    logging by the caller job; never raises."""
    now = now or datetime.now(timezone.utc)
    state = load_state()
    watching = dict(state.get("watching", {}))
    summary = {"checked": 0, "still_waiting": 0, "expired": 0, "vetoed_cost": 0, "proposed": 0, "skipped": 0}
    for symbol, watch in watching.items():
        summary["checked"] += 1
        try:
            exit_time = datetime.fromisoformat(watch["exit_time_utc"])
            bars_since = fetch_mt5_price_history_range(symbol, "M5", exit_time, now)
            verdict = evaluate_continuation(watch["side"], watch["exit_price"], exit_time, watch["atr_at_close"], bars_since, now)

            if not verdict.passed:
                if not verdict.deadline_passed:
                    summary["still_waiting"] += 1
                    continue
                summary["expired"] += 1
                logger.info("%s: continuation watch expired -- %s", symbol, verdict.reason)
                _resolve(symbol, watch, state, "no_continuation", {
                    "checklist_reason": verdict.reason, "bars_seen": verdict.bars_seen, "move_atr": verdict.move_atr,
                })
                continue

            from data.mt5_source import get_market_watch

            live = next((a for a in get_market_watch() if a.symbol == symbol), None)
            if live is None:
                logger.info("%s: continuation watch -- no live quote right now, will retry.", symbol)
                continue
            fresh = analyze_ftmo_asset_live(symbol, live.bid, live.ask, live.description, lean=True)
            drag = cost_drag_r(fresh, _STANDARD_STOP_ATR_MULTIPLE * (fresh.m5_stats.atr or 0), live.ask) if fresh.m5_stats and fresh.m5_stats.atr else None

            if drag is not None and drag > config.COST_DRAG_VETO_R:
                summary["vetoed_cost"] += 1
                logger.info("%s: continuation watch -- vetoed on cost drag (%.2fR > %.2fR limit).", symbol, drag, config.COST_DRAG_VETO_R)
                _resolve(symbol, watch, state, "vetoed_cost", {"cost_drag_r": drag, "move_atr": verdict.move_atr, "minutes_since_close": verdict.minutes_since_close})
                continue

            technical_context = format_clerk_context(fresh)
            prompt = _build_prompt(symbol, watch, verdict, drag if drag is not None else 0.0, technical_context)
            raw, model_used = _ask_local_then_openrouter(prompt)
            decision = _parse_continuation_decision(raw) if raw else ContinuationDecision(False, reason="every model failed", raw_text="")

            journal_data = {
                "move_atr": verdict.move_atr, "minutes_since_close": verdict.minutes_since_close,
                "cost_drag_r": drag, "model_used": model_used, "raw_text": decision.raw_text,
                "propose": decision.propose, "entry": decision.entry, "stop": decision.stop,
                "target": decision.target, "reason": decision.reason,
                "log_only": config.CONTINUATION_WATCH_LOG_ONLY,
            }
            if decision.propose:
                summary["proposed"] += 1
                suffix = " (LOG-ONLY, no order sent)" if config.CONTINUATION_WATCH_LOG_ONLY else ""
                logger.info(
                    "%s: continuation watch PROPOSES a fresh %s at %.6g / stop %.6g / target %.6g -- %s%s",
                    symbol, watch["side"], decision.entry, decision.stop, decision.target, decision.reason, suffix,
                )
            else:
                summary["skipped"] += 1
                logger.info("%s: continuation watch -- model declined: %s", symbol, decision.reason)
            _resolve(symbol, watch, state, "proposed" if decision.propose else "skipped", journal_data)
        except Exception:
            logger.warning("%s: continuation watch model stage failed this poll (will retry).", symbol, exc_info=True)
            continue

    save_state(state)
    return summary
