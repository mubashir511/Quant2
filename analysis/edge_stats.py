"""Significance-aware reading of a backtest result.

Why this exists (2026-09-25, position-hunting review): a pooled M5 backtest that reads "27% win rate,
-0.05R" was being treated as evidence AGAINST a trade. Measured on this account's real M5 bars, a
RANDOM-entry simulation with the same stop/target/cost already lands at about that (a 2x-ATR stop with a
2:1 target wins ~27% at roughly -0.05R gross, more negative after spread) — i.e. that line is the null
result, not information about the thesis. A raw avg-R sign or win-rate cut therefore vetoes good
candidates on noise.

This module gives every simulated setup a fair yardstick — the same trade simulation (`analysis.backtest.
_simulate_trades`: identical stop, target, holding cap, real cost, broker stop floor) started from
sub-sampled RANDOM bars of the same history, both sides — and turns "avg R over n trades" into one of
three honest verdicts:

- SUPPORTED     the setup beat the null baseline by a real margin AND is net-positive.
- CONTRADICTED  the setup was clearly worse than the null baseline on a real sample.
- NO_INFORMATION everything else. This is NOT a reason to exclude anything.

Deterministic, never fabricates: no usable baseline -> NO_INFORMATION, never SUPPORTED.

Real limitation (disclosed, not hidden): the z-score uses the baseline's per-trade R standard deviation
and assumes independent trades. Episodes that overlap in time are positively correlated, so the true
uncertainty is larger than the z suggests — the thresholds are deliberately not tight (z>=+1.5 to
support, z<=-2 to contradict, and an n floor on each).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

import config
from analysis.backtest import _rolling_atr, _simulate_trades

SUPPORTED = "supported"
CONTRADICTED = "contradicted"
NO_INFORMATION = "no_information"


@dataclass(frozen=True)
class EdgeBaseline:
    """Random-entry result on the same bars/stop/target/cost: the yardstick a setup must beat."""

    side: str  # "buy" / "sell"
    mean_r: float  # net realised R per trade (cost netted, timeouts marked to market)
    std_r: float  # per-trade standard deviation of that R
    trades: int
    win_rate_pct: float | None  # wins / (wins + losses), like every other win rate in the backtests


@dataclass(frozen=True)
class EdgeVerdict:
    verdict: str  # SUPPORTED / CONTRADICTED / NO_INFORMATION
    z: float | None  # (avg_r - baseline mean) / (baseline std / sqrt(n)); None when not computable
    trades: int
    avg_r: float
    baseline_r: float | None

    @property
    def label(self) -> str:
        return {SUPPORTED: "SUPPORTED", CONTRADICTED: "CONTRADICTED", NO_INFORMATION: "NO-INFORMATION"}[self.verdict]


def compute_null_baseline(
    ohlc: pd.DataFrame,
    side: str,
    *,
    stop_atr_multiple: float,
    target_atr_multiple: float,
    max_holding_bars: int,
    min_stop_distance_pct: float = 0.0,
    round_trip_cost_pct: float = 0.0,
    step: int | None = None,
) -> EdgeBaseline | None:
    """Simulate `side` from every `step`-th bar of `ohlc` (default config.EDGE_BASELINE_ENTRY_STEP_BARS,
    at most config.EDGE_BASELINE_MAX_ENTRIES entries) with the SAME trade engine and parameters as the
    setup backtests. None (never fabricated) when the history is too thin for at least
    config.EDGE_BASELINE_MIN_TRADES resolved simulations or High/Low are absent."""
    if ohlc is None or ohlc.empty or side not in ("buy", "sell"):
        return None
    atr = _rolling_atr(ohlc)
    if atr is None:
        return None
    step = step or config.EDGE_BASELINE_ENTRY_STEP_BARS
    positions = list(range(config.EDGE_BASELINE_WARMUP_BARS, len(ohlc) - 1, step))
    if len(positions) > config.EDGE_BASELINE_MAX_ENTRIES:
        # Evenly thinned, never the most recent slice only — a regime-biased baseline would flatter or
        # punish setups for the wrong reason.
        stride = math.ceil(len(positions) / config.EDGE_BASELINE_MAX_ENTRIES)
        positions = positions[::stride]
    results = _simulate_trades(
        ohlc,
        positions,
        side,
        atr,
        stop_atr_multiple,
        target_atr_multiple,
        max_holding_bars,
        min_stop_distance_pct=min_stop_distance_pct,
        round_trip_cost_pct=round_trip_cost_pct,
    )
    if len(results) < config.EDGE_BASELINE_MIN_TRADES:
        return None
    rs = [r for r, _ in results]
    n = len(rs)
    mean = sum(rs) / n
    variance = sum((r - mean) ** 2 for r in rs) / (n - 1) if n > 1 else 0.0
    wins = sum(1 for _, o in results if o == "win")
    losses = sum(1 for _, o in results if o == "loss")
    return EdgeBaseline(
        side=side,
        mean_r=float(mean),
        std_r=float(math.sqrt(variance)),
        trades=n,
        win_rate_pct=(wins / (wins + losses) * 100) if (wins + losses) else None,
    )


def classify_edge(avg_r: float | None, trades: int, baseline: EdgeBaseline | None) -> EdgeVerdict:
    """SUPPORTED needs: n >= config.EDGE_MIN_TRADES_SUPPORTED, net avg_r > 0 and z >= config.
    EDGE_SUPPORTED_Z. CONTRADICTED needs: n >= config.EDGE_MIN_TRADES_CONTRADICTED and z <= config.
    EDGE_CONTRADICTED_Z. Otherwise NO_INFORMATION. No baseline or a zero-variance baseline -> z is None
    and the verdict is NO_INFORMATION (unknown is never promoted to a claim)."""
    if avg_r is None or baseline is None or trades <= 0 or baseline.std_r <= 0:
        return EdgeVerdict(NO_INFORMATION, None, max(int(trades or 0), 0), float(avg_r or 0.0), baseline.mean_r if baseline else None)
    z = (avg_r - baseline.mean_r) / (baseline.std_r / math.sqrt(trades))
    verdict = NO_INFORMATION
    if trades >= config.EDGE_MIN_TRADES_SUPPORTED and avg_r > 0 and z >= config.EDGE_SUPPORTED_Z:
        verdict = SUPPORTED
    elif trades >= config.EDGE_MIN_TRADES_CONTRADICTED and z <= config.EDGE_CONTRADICTED_Z:
        verdict = CONTRADICTED
    return EdgeVerdict(verdict, float(z), int(trades), float(avg_r), baseline.mean_r)
