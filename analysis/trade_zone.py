"""Phase 5b of the charting-expert technical-analysis upgrade
(2026-09-21) — the first place in this codebase that actually PICKS a
concrete entry zone + stop-loss + take-profit(s), rather than only
validating a number the model already chose on its own. Pure
composition of data every earlier phase of this same upgrade already
computes: analysis.chart_structure's real S/R bands (Phase 1) and
Fibonacci/trendline reads, analysis.setup_classifier's SetupSignal list
(Phase 3), and analysis.backtest's per-level reliability (Phase 4) —
this module adds no new price-history computation of its own.

Per the user's own explicit, confirmed decision: this is ADVISORY ONLY.
Nothing here clamps a live order — the caller (ai/ftmo_suggest.py's
format_trade_zone) presents this as a strong, named candidate the model
must adopt or explicitly explain deviating from, mirroring this
project's existing "Python computes the number, model decides" split
already used for ai/ftmo_suggest.py's own minimum-viable-size line."""

from dataclasses import dataclass, field

from analysis.backtest import LevelReliabilityBacktest
from analysis.chart_structure import ChartStructureSnapshot, SRLevel
from analysis.setup_classifier import SetupSignal
from analysis.technical import TechnicalStats

# Same 1.5x ATR stop-buffer convention ai/ftmo_suggest.py's own
# MIN_VIABLE_SIZE_ATR_MULTIPLE already uses (Bulkowski, via data/book_
# wisdom.py's "Set a protective stop no closer than roughly 1.5x the
# recent average daily trading range" entry). Duplicated rather than
# imported — ai/ftmo_suggest.py imports FROM analysis/*, so importing
# back here would be a real circular import — but tied explicitly to
# that constant's own name in this comment so the two are never allowed
# to silently drift apart, same discipline as compute_rolling_rsi's own
# promotion elsewhere in this same upgrade.
TRADE_ZONE_STOP_ATR_MULTIPLE = 1.5  # mirrors ai/ftmo_suggest.py::MIN_VIABLE_SIZE_ATR_MULTIPLE

# The existing 2:1 reward-to-risk convention already in data/book_wisdom.py
# ("Require at least a 2:1 reward-to-risk ratio before sizing a
# position." — Bulkowski/Rockefeller) — reused as this function's own
# default floor, not a new number.
DEFAULT_MIN_REWARD_RISK = 2.0

_CONFIDENCE_ORDER = ("weak", "moderate", "strong")

# Real hold-rate thresholds (Phase 4's own LevelReliabilityBacktest.
# hold_rate_pct) that nudge confidence one notch up/down — disclosed,
# unvalidated-until-recalibrated first cuts, same treatment as this same
# upgrade's other placeholder thresholds (SR_RECENCY_HALF_LIFE_BARS,
# CHOCH_GATE_LOOKBACK_BARS).
TRADE_ZONE_STRONG_HOLD_RATE_PCT = 70.0
TRADE_ZONE_WEAK_HOLD_RATE_PCT = 40.0

# Structure stop (Bulkowski: "stop just beyond the minor low/high"; Schwager: where the technical picture
# changes) sits this many ATRs beyond the last swing extreme of the leg the trade rides.
STRUCTURE_STOP_BUFFER_ATR = 0.3
# Measure rule (Bulkowski): target = pattern height projected from the entry; conservative = half height.
MEASURED_MOVE_FRACTIONS = ((0.5, "measured move, half the leg"), (1.0, "measured move, full leg"))


@dataclass
class TradeZoneSuggestion:
    side: str  # "buy" / "sell"
    entry_low: float
    entry_high: float
    stop_loss: float
    take_profits: list[float]  # ascending distance from entry; each already clears min_reward_risk on its own
    reward_risk: float  # reward:risk of the NEAREST take-profit (the conservative case)
    confidence: str  # weak / moderate / strong
    basis: str  # which SetupSignal (or fallback rule) this zone was anchored to
    anchor_level: SRLevel | None  # the real S/R band the entry zone was anchored to, if any
    # Book-grounded additions (2026-09-25), all ADDITIVE to the calibrated floor-based stop above:
    # `structure_stop` = just beyond the swing extreme of the leg being ridden (None when that extreme is not
    # beyond the entry zone); `target_labels` names the take-profits that are measure-rule projections
    # rather than real S/R bands or session extremes (price -> label).
    structure_stop: float | None = None
    target_labels: dict[float, str] = field(default_factory=dict)


def _golden_zone(fib) -> tuple[float, float] | None:
    low = fib.levels.get("38.2%")
    high = fib.levels.get("61.8%")
    if low is None or high is None:
        return None
    return (low, high) if low <= high else (high, low)


def _nearest_same_side_level(levels: list[SRLevel]) -> SRLevel | None:
    candidates = [lvl for lvl in levels if lvl.low is not None and lvl.high is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda lvl: abs(lvl.distance_pct))


def _entry_zone_and_anchor(
    is_buy: bool, structure: ChartStructureSnapshot, signal_names: set[str]
) -> tuple[float, float, SRLevel | None, str] | None:
    same_side_levels = (
        (structure.sr_levels.support_levels if is_buy else structure.sr_levels.resistance_levels)
        if structure.sr_levels is not None
        else []
    )
    nearest_band = _nearest_same_side_level(same_side_levels)

    # 1. Pullback/continuation — Fibonacci golden zone intersected with
    # the nearest same-side S/R band, if the two genuinely overlap.
    if "pullback_continuation" in signal_names and structure.fibonacci is not None:
        golden = _golden_zone(structure.fibonacci)
        if golden is not None:
            glow, ghigh = golden
            if nearest_band is not None and nearest_band.low <= ghigh and nearest_band.high >= glow:
                return max(nearest_band.low, glow), min(nearest_band.high, ghigh), nearest_band, "pullback_continuation"
            return glow, ghigh, None, "pullback_continuation"

    # 2. Trend-following — price on its own trendline, ± one ATR buffer.
    if "trend_following" in signal_names and structure.trendlines is not None:
        relevant_line = structure.trendlines.support_trendline if is_buy else structure.trendlines.resistance_trendline
        if relevant_line is not None:
            return None, None, relevant_line, "trend_following"  # placeholder, ATR filled in by caller

    # 3. Fallback — the nearest same-side real S/R band.
    if nearest_band is not None:
        return nearest_band.low, nearest_band.high, nearest_band, "nearest_sr_level"

    # 4. Last resort — the bare Fibonacci golden zone, no S/R band anchor.
    if structure.fibonacci is not None:
        golden = _golden_zone(structure.fibonacci)
        if golden is not None:
            return golden[0], golden[1], None, "fibonacci_golden_zone"

    return None


def _adjust_confidence(
    base: str,
    reliability: LevelReliabilityBacktest | None,
    strong_hold_rate_pct: float = TRADE_ZONE_STRONG_HOLD_RATE_PCT,
    weak_hold_rate_pct: float = TRADE_ZONE_WEAK_HOLD_RATE_PCT,
) -> str:
    if reliability is None or reliability.hold_rate_pct is None:
        return base
    idx = _CONFIDENCE_ORDER.index(base)
    if reliability.hold_rate_pct >= strong_hold_rate_pct:
        idx = min(idx + 1, len(_CONFIDENCE_ORDER) - 1)
    elif reliability.hold_rate_pct <= weak_hold_rate_pct:
        idx = max(idx - 1, 0)
    return _CONFIDENCE_ORDER[idx]


def construct_trade_zone(
    side: str,
    stats: TechnicalStats,
    structure: ChartStructureSnapshot,
    setup_signals: list[SetupSignal],
    level_reliability: dict[float, LevelReliabilityBacktest] | None = None,
    min_reward_risk: float = DEFAULT_MIN_REWARD_RISK,
    stop_atr: float | None = None,
    extra_target_prices: list[float] | None = None,
    strong_hold_rate_pct: float = TRADE_ZONE_STRONG_HOLD_RATE_PCT,
    weak_hold_rate_pct: float = TRADE_ZONE_WEAK_HOLD_RATE_PCT,
    measured_moves: bool = False,
) -> TradeZoneSuggestion | None:
    """Picks one concrete entry zone + stop-loss + take-profit ladder for
    `side` ("buy"/"sell" — matches this codebase's own backtest-function
    convention, not "long"/"short"). None whenever there isn't enough
    real structure to anchor a genuine zone on, or the best available
    setup doesn't clear `min_reward_risk` — never a fabricated weak
    suggestion, matching analysis/backtest.py's own established
    "never fabricate" convention throughout this project.

    Basis priority: a fired `pullback_continuation` signal anchors to the
    Fibonacci golden zone intersected with the nearest same-side S/R
    band (or the bare golden zone if no band is close); a fired `trend_
    following` signal anchors to price's own trendline ± one ATR buffer;
    otherwise falls back to the single nearest same-side real S/R band,
    or the bare Fibonacci golden zone as a last resort. Returns None if
    none of these have enough real structure to build from.

    Take-profits are every real opposing-side S/R band beyond the entry
    zone that INDIVIDUALLY clears `min_reward_risk` against the entry
    zone's own worst-case edge, ascending by distance (nearest-that-
    qualifies first) — `reward_risk` reports that nearest qualifying
    target's own ratio. `level_reliability` (Phase 4's per-level real
    hold-rate), when the chosen anchor is a real SRLevel present in that
    dict, nudges confidence one notch up for a strong real hold-rate or
    one notch down for a weak one — never more than one notch, and never
    below "weak" or above "strong".

    `stop_atr` (added 2026-09-22, direct user challenge: this account's
    own H1-scale entries were taking hours-to-days to fill, structurally
    mismatched with its own same-day holding design) — when given, used
    ONLY for the stop-loss buffer (`TRADE_ZONE_STOP_ATR_MULTIPLE *
    stop_atr`) instead of `stats.atr`. Lets a caller feed a TIGHTER,
    faster-timeframe `stats` (e.g. M5) for genuinely precise, fast-to-
    fill entry-zone construction while still sizing the stop off a
    WIDER, calmer timeframe's own ATR (e.g. H1) — real room for the
    trade to develop over the rest of the session without either
    getting clipped by ordinary fast-timeframe noise or colliding with
    the broker's own minimum-stop-distance floor. The trend-following
    entry-zone WIDTH still always uses `stats.atr` (deliberately stays
    tight/precise on whatever timeframe `stats` itself represents) —
    only the stop-loss distance switches. None (the default) reproduces
    the exact prior behavior — every existing caller keeps working
    unchanged.

    `extra_target_prices` (added 2026-09-24, M5-only decision tier): additional real opposing price
    levels that count as take-profit candidates alongside the opposing S/R bands — the session
    magnets (prev-day high/low, today's high/low) the M5 read knows about. Measured on real broker
    bars, an M5 structure's own S/R bands cleared the 2:1 bar on only ~10-20% of symbol/side
    slots (H1's cleared ~60%) because M5 bands sit only a few M5 ATRs apart; these session extremes
    are the natural intraday targets. Only prices strictly beyond the entry zone on the profit side
    are used, and each must clear `min_reward_risk` on its own exactly like an S/R band.

    `strong_hold_rate_pct` / `weak_hold_rate_pct`: the hold-rate cutoffs of the one-notch confidence
    nudge. The defaults are the H1-era 70 / 40; the M5 caller passes cutoffs measured on M5 bands
    (config.M5_ZONE_STRONG_HOLD_RATE_PCT / M5_ZONE_WEAK_HOLD_RATE_PCT), where 70 / 40 are degenerate.

    `measured_moves` (2026-09-25, Bulkowski's measure rule): also offers the height of the last swing leg
    (Fibonacci swing high-low) projected from the entry reference in the trade's direction — half the leg
    (the conservative case) and the full leg — but only when the last leg ran WITH `side` (a buy after an
    up-leg, a sell after a down-leg: a pullback continuation), and each must clear `min_reward_risk` like
    any other target. `structure_stop` on the result is the stop just beyond that same leg's far extreme
    (swing low for a buy, swing high for a sell) by STRUCTURE_STOP_BUFFER_ATR x `stats.atr`, reported only
    when it sits beyond the calibrated floor stop; the returned `stop_loss` itself is unchanged."""
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    if stats.last_price is None or stats.atr is None:
        return None
    effective_stop_atr = stop_atr if stop_atr is not None else stats.atr

    is_buy = side == "buy"
    signal_names = {s.name for s in setup_signals}
    signal_by_name = {s.name: s for s in setup_signals}

    zone = _entry_zone_and_anchor(is_buy, structure, signal_names)
    if zone is None:
        return None
    entry_low, entry_high, anchor, basis = zone
    if basis == "trend_following":
        # anchor here is the Trendline itself (see _entry_zone_and_anchor's
        # own comment) — resolve the real ± ATR buffer band now that
        # `stats.atr` is in scope, and drop the trendline as the anchor_
        # level since it isn't a real SRLevel (no price to key level_
        # reliability by).
        center = anchor.current_price_on_line
        entry_low, entry_high = center - stats.atr, center + stats.atr
        anchor = None

    if entry_low is None or entry_high is None or entry_low > entry_high:
        return None

    if is_buy:
        stop_loss = entry_low - TRADE_ZONE_STOP_ATR_MULTIPLE * effective_stop_atr
        entry_reference = entry_high  # worst-case: paying the top of the zone
        opposing_levels = structure.sr_levels.resistance_levels if structure.sr_levels is not None else []
        beyond_entry = [lvl for lvl in opposing_levels if lvl.price > entry_high]
    else:
        stop_loss = entry_high + TRADE_ZONE_STOP_ATR_MULTIPLE * effective_stop_atr
        entry_reference = entry_low  # worst-case: selling at the bottom of the zone
        opposing_levels = structure.sr_levels.support_levels if structure.sr_levels is not None else []
        beyond_entry = [lvl for lvl in opposing_levels if lvl.price < entry_low]

    risk = abs(entry_reference - stop_loss)
    candidate_prices = [lvl.price for lvl in beyond_entry]
    for price in extra_target_prices or []:
        if price is not None and ((price > entry_high) if is_buy else (price < entry_low)):
            candidate_prices.append(price)
    if risk == 0 or not candidate_prices:
        return None

    target_labels: dict[float, str] = {}
    structure_stop: float | None = None
    fib = structure.fibonacci
    if fib is not None and fib.swing_high > fib.swing_low:
        leg = fib.swing_high - fib.swing_low
        leg_runs_with_side = fib.high_is_more_recent == is_buy
        if leg_runs_with_side:
            far_extreme = fib.swing_low if is_buy else fib.swing_high
            candidate_stop = (
                far_extreme - STRUCTURE_STOP_BUFFER_ATR * stats.atr
                if is_buy
                else far_extreme + STRUCTURE_STOP_BUFFER_ATR * stats.atr
            )
            if (candidate_stop < stop_loss) if is_buy else (candidate_stop > stop_loss):
                structure_stop = candidate_stop
            if measured_moves:
                for fraction, label in MEASURED_MOVE_FRACTIONS:
                    projected = entry_reference + fraction * leg if is_buy else entry_reference - fraction * leg
                    candidate_prices.append(projected)
                    target_labels[projected] = label

    candidate_prices.sort(key=lambda price: abs(price - entry_reference))
    qualifying = []
    for price in candidate_prices:
        reward = abs(price - entry_reference)
        rr = reward / risk
        if rr >= min_reward_risk and all(abs(price - kept) > 1e-12 for kept, _rr in qualifying):
            qualifying.append((price, rr))
    if not qualifying:
        return None

    base_confidence = signal_by_name[basis].confidence if basis in signal_by_name else "moderate"
    reliability = level_reliability.get(anchor.price) if level_reliability is not None and anchor is not None else None
    confidence = _adjust_confidence(base_confidence, reliability, strong_hold_rate_pct, weak_hold_rate_pct)

    return TradeZoneSuggestion(
        side=side,
        entry_low=entry_low,
        entry_high=entry_high,
        stop_loss=stop_loss,
        take_profits=[price for price, _rr in qualifying],
        reward_risk=qualifying[0][1],
        confidence=confidence,
        basis=basis,
        anchor_level=anchor,
        structure_stop=structure_stop,
        target_labels={price: label for price, label in target_labels.items() if any(price == kept for kept, _ in qualifying)},
    )
