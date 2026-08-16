from dataclasses import dataclass, field
from typing import Callable

from ai.portfolio_suggest import AllocationEntry
from data.mt5_source import AccountSummary, ContractSpec, MarketAsset, Position


@dataclass
class PlannedOrder:
    symbol: str
    action: str  # "open" | "increase" | "reduce" | "close" | "hold" | "infeasible"
    side: str  # "buy" or "sell"
    volume: float
    order_type: str  # "limit" | "market" | "none"
    price: float | None
    stop_loss: float | None
    take_profit: float | None = None
    tickets_to_close: list[tuple[int, float]] = field(default_factory=list)
    reason: str = ""


def _round_down_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    steps = int(value / step + 1e-9)
    return round(steps * step, 8)


def compute_rebalance_plan(
    positions: list[Position],
    account: AccountSummary,
    allocation: dict[str, AllocationEntry],
    get_spec: Callable[[str], ContractSpec | None],
    market_prices: dict[str, MarketAsset],
    price_sanity_band_pct: float = 5.0,
) -> list[PlannedOrder]:
    """Pure, deterministic diff between what's currently held and the
    AI's target allocation — no network calls, no AI reasoning here, only
    arithmetic (mirrors risk/rebalance.py's "pure function" pattern).

    Long-only: a symbol with any existing short exposure (or a mix of
    both sides — never expected in practice, this system's suggestions
    are always long-oriented) is flagged for a full close rather than
    netted against a long target, so a short position is never silently
    left open or partially resized by mistake.

    A symbol held but missing from `allocation` entirely is treated as a
    0% target (close it) — matches the prompt's explicit instruction that
    every held symbol must appear in the final JSON even at 0, with this
    as the safety-net default in case that instruction is ever missed.

    Any proposed limit-entry price is clamped to within
    `price_sanity_band_pct` of the live ask, and a stop-loss OR take-
    profit on the wrong side of the (possibly-clamped) entry price is
    dropped rather than sent to the broker — both noted in `reason`,
    never silently substituted without a trace.
    """
    plans: list[PlannedOrder] = []

    by_symbol: dict[str, list[Position]] = {}
    for p in positions:
        by_symbol.setdefault(p.symbol, []).append(p)

    all_symbols = set(by_symbol) | {s for s in allocation if s.upper() != "CASH"}

    for symbol in sorted(all_symbols):
        symbol_positions = by_symbol.get(symbol, [])
        entry = allocation.get(symbol)
        pct = entry.pct if entry is not None else 0.0

        sell_positions = [p for p in symbol_positions if p.side == "sell"]
        buy_positions = [p for p in symbol_positions if p.side == "buy"]

        if sell_positions:
            tickets = [(p.ticket, p.volume) for p in symbol_positions]
            total_volume = sum(p.volume for p in symbol_positions)
            plans.append(
                PlannedOrder(
                    symbol=symbol,
                    action="close",
                    side="buy" if not buy_positions else "sell",
                    volume=total_volume,
                    order_type="market",
                    price=None,
                    stop_loss=None,
                    tickets_to_close=tickets,
                    reason=(
                        "Existing short or mixed-direction position on a "
                        "long-only system — closing fully; a new long "
                        "target can be applied once flat."
                    ),
                )
            )
            continue

        current_lots = sum(p.volume for p in buy_positions)

        spec = get_spec(symbol)
        if spec is None or spec.margin_initial <= 0:
            if current_lots > 0 or pct > 0:
                plans.append(
                    PlannedOrder(
                        symbol=symbol, action="infeasible", side="buy", volume=0.0,
                        order_type="none", price=None, stop_loss=None,
                        reason="No contract spec available for this symbol.",
                    )
                )
            continue

        target_margin = pct / 100 * account.equity
        target_lots = _round_down_to_step(target_margin / spec.margin_initial, spec.volume_step)

        if pct > 0 and target_lots < spec.volume_min:
            plans.append(
                PlannedOrder(
                    symbol=symbol, action="infeasible", side="buy", volume=0.0,
                    order_type="none", price=None, stop_loss=None,
                    reason=(
                        f"Target {pct:.1f}% of equity can't afford even the "
                        f"minimum {spec.volume_min:g}-lot for this instrument."
                    ),
                )
            )
            continue

        delta = target_lots - current_lots

        if abs(delta) < spec.volume_step / 2:
            plans.append(
                PlannedOrder(
                    symbol=symbol, action="hold", side="buy", volume=current_lots,
                    order_type="none", price=None, stop_loss=None,
                    reason="Already at target allocation.",
                )
            )
            continue

        if delta > 0:
            asset = market_prices.get(symbol)
            proposed_price = entry.price if entry is not None else None
            if asset is None or proposed_price is None:
                plans.append(
                    PlannedOrder(
                        symbol=symbol, action="infeasible", side="buy", volume=delta,
                        order_type="limit", price=None,
                        stop_loss=entry.stop_loss if entry is not None else None,
                        reason="No live quote or suggested price available to place a limit order.",
                    )
                )
                continue

            band = price_sanity_band_pct / 100
            lo, hi = asset.ask * (1 - band), asset.ask * (1 + band)
            clamped_price = min(max(proposed_price, lo), hi)
            note = ""
            if clamped_price != proposed_price:
                note = (
                    f" (clamped from {proposed_price:.4f} to stay within "
                    f"{price_sanity_band_pct:.0f}% of the live ask {asset.ask:.4f})"
                )

            stop_loss = entry.stop_loss if entry is not None else None
            if stop_loss is not None and stop_loss >= clamped_price:
                stop_loss = None
                note += " (suggested stop was on the wrong side of entry, dropped)"

            take_profit = entry.take_profit if entry is not None else None
            if take_profit is not None and take_profit <= clamped_price:
                take_profit = None
                note += " (suggested take-profit was on the wrong side of entry, dropped)"

            action = "open" if current_lots == 0 else "increase"
            plans.append(
                PlannedOrder(
                    symbol=symbol, action=action, side="buy", volume=delta,
                    order_type="limit", price=clamped_price, stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"Target {pct:.1f}% of equity.{note}",
                )
            )
        else:
            reduce_volume = -delta
            tickets: list[tuple[int, float]] = []
            remaining = reduce_volume
            for p in buy_positions:
                if remaining <= 1e-9:
                    break
                take = min(p.volume, remaining)
                tickets.append((p.ticket, take))
                remaining -= take

            action = "close" if target_lots == 0 else "reduce"
            plans.append(
                PlannedOrder(
                    symbol=symbol, action=action, side="sell", volume=reduce_volume,
                    order_type="market", price=None, stop_loss=None,
                    tickets_to_close=tickets,
                    reason=f"Target {pct:.1f}% of equity, reducing from {current_lots:g} lots.",
                )
            )

    return plans
