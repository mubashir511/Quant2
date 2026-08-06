from dataclasses import dataclass

import config
from data.mt5_source import Position


@dataclass
class Suggestion:
    symbol: str
    action: str
    reason: str


def evaluate_positions(
    positions: list[Position],
    max_loss_pct: float | None = None,
    max_position_count: int | None = None,
    max_symbol_exposure_pct: float | None = None,
) -> list[Suggestion]:
    """Deterministic, auditable rebalance suggestions. No AI involved here."""
    max_loss_pct = config.MAX_LOSS_PCT if max_loss_pct is None else max_loss_pct
    max_position_count = (
        config.MAX_POSITION_COUNT if max_position_count is None else max_position_count
    )
    max_symbol_exposure_pct = (
        config.MAX_SYMBOL_EXPOSURE_PCT
        if max_symbol_exposure_pct is None
        else max_symbol_exposure_pct
    )

    suggestions: list[Suggestion] = []
    suggestions.extend(_cut_loss(positions, max_loss_pct))
    suggestions.extend(_no_stop(positions))
    suggestions.extend(_trim_excess_count(positions, max_position_count))
    suggestions.extend(_reduce_concentration(positions, max_symbol_exposure_pct))
    return suggestions


def _cut_loss(positions: list[Position], max_loss_pct: float) -> list[Suggestion]:
    out = []
    for p in positions:
        if p.adverse_move_pct >= max_loss_pct:
            out.append(
                Suggestion(
                    p.symbol,
                    "cut_loss",
                    f"Price has moved {p.adverse_move_pct:.1f}% against this "
                    f"{p.side} position, beyond the {max_loss_pct:.0f}% limit.",
                )
            )
    return out


def _no_stop(positions: list[Position]) -> list[Suggestion]:
    out = []
    for p in positions:
        if p.sl is None:
            out.append(
                Suggestion(p.symbol, "no_stop", "No stop-loss is set on this position.")
            )
    return out


def _trim_excess_count(
    positions: list[Position], max_position_count: int
) -> list[Suggestion]:
    excess = len(positions) - max_position_count
    if excess <= 0:
        return []
    worst_first = sorted(positions, key=lambda p: p.profit)
    out = []
    for p in worst_first[:excess]:
        out.append(
            Suggestion(
                p.symbol,
                "trim_excess_count",
                f"Open position count ({len(positions)}) exceeds the limit of "
                f"{max_position_count}; this is one of the weakest by P&L.",
            )
        )
    return out


def _reduce_concentration(
    positions: list[Position], max_symbol_exposure_pct: float
) -> list[Suggestion]:
    total_notional = sum(p.notional for p in positions)
    if total_notional <= 0:
        return []

    by_symbol: dict[str, float] = {}
    for p in positions:
        by_symbol[p.symbol] = by_symbol.get(p.symbol, 0.0) + p.notional

    out = []
    for symbol, notional in by_symbol.items():
        exposure_pct = notional / total_notional * 100
        if exposure_pct > max_symbol_exposure_pct:
            out.append(
                Suggestion(
                    symbol,
                    "reduce_concentration",
                    f"{symbol} is {exposure_pct:.0f}% of total exposure, above the "
                    f"{max_symbol_exposure_pct:.0f}% limit.",
                )
            )
    return out
