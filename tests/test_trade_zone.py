import pytest

from analysis.backtest import LevelReliabilityBacktest
from analysis.chart_structure import (
    FIB_RATIOS,
    ChartStructureSnapshot,
    FibonacciLevels,
    SRLevel,
    SRLevelsResult,
    Trendline,
    TrendlineAnalysis,
)
from analysis.setup_classifier import SetupSignal
from analysis.technical import TechnicalStats
from analysis.trade_zone import construct_trade_zone


def _stats(**overrides) -> TechnicalStats:
    defaults = dict(
        last_price=100.0, sma20=None, pct_vs_sma20=None, trend=None,
        change_1m_pct=None, change_3m_pct=None, change_6m_pct=None,
        volatility_annualized_pct=None, support=None, resistance=None,
        range_width_pct=None, market_regime=None, atr=2.0, atr_pct=None,
        rsi=None, volume_trend_pct=None, momentum_acceleration=None,
    )
    return TechnicalStats(**{**defaults, **overrides})


def _fib(**overrides) -> FibonacciLevels:
    defaults = dict(
        swing_high=110.0, swing_low=90.0, high_is_more_recent=True,
        levels={}, current_price=100.0, nearest_level_name="50.0%",
        nearest_level_price=100.0, distance_to_nearest_pct=0.0,
    )
    f = FibonacciLevels(**{**defaults, **overrides})
    if not f.levels:
        # Real computation, mirroring compute_fibonacci_levels' own
        # formula exactly, so this fixture's golden zone is genuine
        # rather than a hand-picked approximation.
        span = f.swing_high - f.swing_low
        levels = {}
        for ratio in FIB_RATIOS:
            name = f"{ratio * 100:.1f}%"
            levels[name] = f.swing_high - ratio * span if f.high_is_more_recent else f.swing_low + ratio * span
        f.levels = levels
    return f


def _trendline(**overrides) -> Trendline:
    defaults = dict(
        slope_per_bar=0.0, direction="flat", current_price_on_line=100.0,
        price_vs_line="at", distance_pct=0.0, bars_since_last_point=0,
    )
    return Trendline(**{**defaults, **overrides})


# A real bullish-pullback fixture: uptrend swing 90 -> 110 (fresh high,
# retracing down), a 3-touch support band sitting inside the golden zone
# (97.5-98.5, golden zone is 97.64-102.36 -- genuine overlap, empirically
# confirmed), and a real resistance target far enough away to clear the
# default 2:1 floor.
_SUPPORT = SRLevel(price=98.0, touches=3, distance_pct=-2.0, low=97.5, high=98.5)
_RESISTANCE = SRLevel(price=108.0, touches=2, distance_pct=8.0, low=107.5, high=108.5)


def _structure(**overrides) -> ChartStructureSnapshot:
    defaults = dict(
        fibonacci=_fib(),
        sr_levels=SRLevelsResult(resistance_levels=[_RESISTANCE], support_levels=[_SUPPORT], tolerance_pct=0.5),
        trendlines=None,
        patterns=[],
    )
    return ChartStructureSnapshot(**{**defaults, **overrides})


def _reliability(hold_rate_pct: float, price: float = 98.0) -> dict:
    return {
        price: LevelReliabilityBacktest(
            level_price=price, level_low=97.5, level_high=98.5, side="support",
            tests=10, holds=8, breaks=2, hold_rate_pct=hold_rate_pct,
            trades=8, wins=6, losses=2, timeouts=0, win_rate_pct=75.0,
            avg_r_multiple=1.5, stop_atr_multiple=1.5, target_atr_multiple=3.0, max_holding_bars=10,
        )
    }


def test_construct_trade_zone_anchors_to_matching_signals_band():
    signals = [SetupSignal(name="pullback_continuation", detail="x")]
    result = construct_trade_zone("buy", _stats(), _structure(), signals)
    assert result is not None
    assert result.basis == "pullback_continuation"
    # Real golden-zone/SR-band intersection (empirically verified): golden
    # zone 97.64-102.36 intersected with the 97.5-98.5 support band.
    assert result.entry_low == pytest.approx(97.64)
    assert result.entry_high == pytest.approx(98.5)
    assert result.anchor_level == _SUPPORT
    assert result.stop_loss < result.entry_low  # buy: stop sits below the zone
    assert result.take_profits == [108.0]
    assert result.reward_risk >= 2.0
    assert result.confidence == "moderate"  # no level_reliability passed


def test_construct_trade_zone_stop_atr_override_widens_stop_without_touching_entry_zone():
    # Real use case (2026-09-22): a fast-timeframe `stats` (e.g. M5) gives
    # a tight, precise entry zone, while `stop_atr` supplies a WIDER,
    # calmer timeframe's own ATR (e.g. H1) purely for the stop-loss
    # buffer — the entry zone itself must stay exactly as tight/precise
    # as `stats.atr` on its own would produce.
    # min_reward_risk relaxed to 1.0 here specifically to isolate the
    # stop-distance behavior from the (separately, already-tested)
    # reward:risk floor gate — a genuinely wider stop naturally worsens
    # the ratio, which is real and correct, just not what this test is
    # about.
    signals = [SetupSignal(name="pullback_continuation", detail="x")]
    baseline = construct_trade_zone("buy", _stats(atr=2.0), _structure(), signals, min_reward_risk=1.0)
    widened = construct_trade_zone("buy", _stats(atr=2.0), _structure(), signals, min_reward_risk=1.0, stop_atr=4.0)
    assert baseline is not None and widened is not None
    assert widened.entry_low == pytest.approx(baseline.entry_low)
    assert widened.entry_high == pytest.approx(baseline.entry_high)
    assert widened.stop_loss < baseline.stop_loss  # a buy: wider stop sits further below
    assert (baseline.entry_low - baseline.stop_loss) == pytest.approx(1.5 * 2.0)
    assert (widened.entry_low - widened.stop_loss) == pytest.approx(1.5 * 4.0)


def test_construct_trade_zone_returns_none_below_min_reward_risk():
    signals = [SetupSignal(name="pullback_continuation", detail="x")]
    close_resistance = SRLevel(price=103.0, touches=2, distance_pct=3.0, low=102.5, high=103.5)
    structure = _structure(
        sr_levels=SRLevelsResult(resistance_levels=[close_resistance], support_levels=[_SUPPORT], tolerance_pct=0.5)
    )
    assert construct_trade_zone("buy", _stats(), structure, signals) is None


def test_construct_trade_zone_upgrades_confidence_with_strong_level_reliability():
    signals = [SetupSignal(name="pullback_continuation", detail="x")]
    result = construct_trade_zone("buy", _stats(), _structure(), signals, level_reliability=_reliability(90.0))
    assert result is not None
    assert result.confidence == "strong"


def test_construct_trade_zone_downgrades_confidence_with_weak_level_reliability():
    signals = [SetupSignal(name="pullback_continuation", detail="x")]
    result = construct_trade_zone("buy", _stats(), _structure(), signals, level_reliability=_reliability(20.0))
    assert result is not None
    assert result.confidence == "weak"


def test_construct_trade_zone_falls_back_to_fibonacci_without_a_close_band():
    # No support levels at all -> the pullback signal's own entry zone
    # falls back to the bare Fibonacci golden zone (no anchor_level), but
    # a real resistance target still far enough away to clear 2:1.
    signals = [SetupSignal(name="pullback_continuation", detail="x")]
    far_resistance = SRLevel(price=130.0, touches=2, distance_pct=30.0, low=129.0, high=131.0)
    structure = _structure(sr_levels=SRLevelsResult(resistance_levels=[far_resistance], support_levels=[]))
    result = construct_trade_zone("buy", _stats(), structure, signals)
    assert result is not None
    assert result.anchor_level is None
    assert result.entry_low == pytest.approx(97.64)
    assert result.entry_high == pytest.approx(102.36)
    assert result.take_profits == [130.0]


def test_construct_trade_zone_trend_following_uses_trendline_plus_atr_buffer():
    signals = [SetupSignal(name="trend_following", detail="x")]
    trendlines = TrendlineAnalysis(support_trendline=_trendline(current_price_on_line=99.0), resistance_trendline=None)
    structure = _structure(fibonacci=None, trendlines=trendlines)
    result = construct_trade_zone("buy", _stats(atr=1.0), structure, signals)
    assert result is not None
    assert result.basis == "trend_following"
    assert result.entry_low == pytest.approx(98.0)
    assert result.entry_high == pytest.approx(100.0)
    assert result.anchor_level is None  # a Trendline, not a real SRLevel — never fabricated as one


def test_construct_trade_zone_falls_back_to_nearest_sr_level_without_a_matching_signal():
    result = construct_trade_zone("buy", _stats(), _structure(), [])
    assert result is not None
    assert result.basis == "nearest_sr_level"
    assert result.entry_low == pytest.approx(97.5)
    assert result.entry_high == pytest.approx(98.5)
    assert result.anchor_level == _SUPPORT


def test_construct_trade_zone_sell_side_mirrors_buy_direction():
    support = SRLevel(price=92.0, touches=2, distance_pct=-8.0, low=91.5, high=92.5)
    resistance = SRLevel(price=102.0, touches=3, distance_pct=2.0, low=101.5, high=102.5)
    structure = _structure(sr_levels=SRLevelsResult(resistance_levels=[resistance], support_levels=[support]))
    result = construct_trade_zone("sell", _stats(), structure, [])
    assert result is not None
    assert result.stop_loss > result.entry_high  # sell: stop sits above the zone
    assert result.take_profits == [92.0]


def test_construct_trade_zone_raises_on_invalid_side():
    with pytest.raises(ValueError):
        construct_trade_zone("sideways", _stats(), _structure(), [])


def test_construct_trade_zone_none_without_last_price_or_atr():
    signals = [SetupSignal(name="pullback_continuation", detail="x")]
    assert construct_trade_zone("buy", _stats(last_price=None), _structure(), signals) is None
    assert construct_trade_zone("buy", _stats(atr=None), _structure(), signals) is None


def test_construct_trade_zone_none_without_any_real_structure():
    empty_structure = ChartStructureSnapshot(fibonacci=None, sr_levels=None, trendlines=None, patterns=[])
    assert construct_trade_zone("buy", _stats(), empty_structure, []) is None


# --- extra_target_prices (M5-only decision tier, 2026-09-24): session extremes as extra targets ---


def _support_only_structure() -> ChartStructureSnapshot:
    return _structure(sr_levels=SRLevelsResult(resistance_levels=[], support_levels=[_SUPPORT], tolerance_pct=0.5))


def test_extra_target_prices_supply_targets_when_no_opposing_sr_band_exists():
    signals = [SetupSignal(name="pullback_continuation", detail="x")]
    assert construct_trade_zone("buy", _stats(), _support_only_structure(), signals) is None
    result = construct_trade_zone("buy", _stats(), _support_only_structure(), signals, extra_target_prices=[110.0])
    assert result is not None and result.take_profits == [110.0]


def test_extra_target_prices_only_count_on_the_profit_side_and_must_clear_the_reward_risk_bar():
    signals = [SetupSignal(name="pullback_continuation", detail="x")]
    result = construct_trade_zone(
        "buy", _stats(), _structure(), signals,
        # 90 is on the wrong side of the entry, 99 clears nothing (reward 0.5 vs risk ~4+), 108.0 duplicates
        # the real S/R band target, 111 is a real further target.
        extra_target_prices=[90.0, 99.0, 108.0, 111.0, None],
    )
    assert result.take_profits == [108.0, 111.0]


def test_extra_target_prices_work_for_a_sell_and_the_default_is_unchanged():
    signals = [SetupSignal(name="pullback_continuation", detail="x")]
    with_extra = construct_trade_zone("sell", _stats(), _structure(), signals, extra_target_prices=[80.0])
    assert with_extra is not None and 80.0 in with_extra.take_profits
    assert construct_trade_zone("buy", _stats(), _structure(), signals, extra_target_prices=None) == construct_trade_zone(
        "buy", _stats(), _structure(), signals
    )


# --- Audit fixes 2026-09-24 (M5-only): hold-rate cutoffs are per-timeframe, entry distance is stated ---


def test_hold_rate_cutoffs_are_overridable_so_m5_can_use_its_own_measured_quartiles():
    signals = [SetupSignal(name="pullback_continuation", detail="x")]
    # A 20% hold rate: below the H1-era weak cutoff (40) -> downgraded by default...
    default = construct_trade_zone("buy", _stats(), _structure(), signals, level_reliability=_reliability(20.0))
    assert default.confidence == "weak"
    # ...but mid-pack against M5-measured cutoffs (weak <= 6, strong >= 25) -> unchanged.
    m5 = construct_trade_zone(
        "buy", _stats(), _structure(), signals, level_reliability=_reliability(20.0),
        strong_hold_rate_pct=25.0, weak_hold_rate_pct=6.0,
    )
    assert m5.confidence == "moderate"
    up = construct_trade_zone(
        "buy", _stats(), _structure(), signals, level_reliability=_reliability(26.0),
        strong_hold_rate_pct=25.0, weak_hold_rate_pct=6.0,
    )
    assert up.confidence == "strong"
    down = construct_trade_zone(
        "buy", _stats(), _structure(), signals, level_reliability=_reliability(5.0),
        strong_hold_rate_pct=25.0, weak_hold_rate_pct=6.0,
    )
    assert down.confidence == "weak"


# --- Book-grounded structure stop and measure-rule targets (2026-09-25) ---------------------------------

def test_measured_move_targets_are_off_by_default_and_the_stop_is_unchanged():
    signals = [SetupSignal(name="pullback_continuation", detail="x")]
    plain = construct_trade_zone("buy", _stats(), _structure(), signals)
    assert plain.take_profits == [108.0] and plain.target_labels == {}
    with_moves = construct_trade_zone("buy", _stats(), _structure(), signals, measured_moves=True)
    assert with_moves.stop_loss == pytest.approx(plain.stop_loss)  # the calibrated floor stop never moves


def test_buy_measured_move_projects_half_and_full_leg_from_the_entry_reference():
    signals = [SetupSignal(name="pullback_continuation", detail="x")]
    zone = construct_trade_zone("buy", _stats(), _structure(), signals, measured_moves=True)
    # leg 90 -> 110 = 20; entry reference = entry_high 98.5; half = 108.5, full = 118.5.
    assert zone.take_profits == pytest.approx([108.0, 108.5, 118.5])
    labels = {round(price, 6): label for price, label in zone.target_labels.items()}
    assert labels == {108.5: "measured move, half the leg", 118.5: "measured move, full leg"}


def test_measured_move_targets_must_still_clear_the_reward_risk_bar():
    signals = [SetupSignal(name="pullback_continuation", detail="x")]
    small_leg = _structure(fibonacci=_fib(swing_high=103.0, swing_low=97.0))  # leg 6: half = +3 -> below 2R
    zone = construct_trade_zone("buy", _stats(), small_leg, signals, measured_moves=True, extra_target_prices=[110.0])
    assert zone is not None
    assert all(price not in zone.target_labels for price in zone.take_profits)  # neither projection qualified


def test_structure_stop_sits_just_beyond_the_swing_extreme_and_only_when_wider_than_the_floor_stop():
    signals = [SetupSignal(name="pullback_continuation", detail="x")]
    zone = construct_trade_zone("buy", _stats(), _structure(), signals)
    assert zone.structure_stop == pytest.approx(90.0 - 0.3 * 2.0)
    assert zone.structure_stop < zone.stop_loss
    # A swing extreme sitting ABOVE the floor stop (a shallow leg) is not "beyond" it: no alternative reported.
    shallow = _structure(fibonacci=_fib(swing_high=104.0, swing_low=100.0))
    zone = construct_trade_zone("buy", _stats(), shallow, signals, extra_target_prices=[112.0])
    assert zone is not None and zone.structure_stop is None


def test_sell_mirrors_structure_stop_and_measured_moves_and_ignores_a_counter_leg():
    support = SRLevel(price=92.0, touches=2, distance_pct=-8.0, low=91.5, high=92.5)
    resistance = SRLevel(price=102.0, touches=3, distance_pct=2.0, low=101.5, high=102.5)
    down_leg = _structure(
        fibonacci=_fib(swing_high=110.0, swing_low=90.0, high_is_more_recent=False),
        sr_levels=SRLevelsResult(resistance_levels=[resistance], support_levels=[support]),
    )
    zone = construct_trade_zone("sell", _stats(), down_leg, [], measured_moves=True)
    assert zone.structure_stop == pytest.approx(110.0 + 0.3 * 2.0) and zone.structure_stop > zone.stop_loss
    assert min(zone.take_profits) < zone.entry_low
    # The last leg ran UP: a sell is counter-leg, so no measure-rule targets and no structure stop.
    up_leg = _structure(
        fibonacci=_fib(swing_high=110.0, swing_low=90.0, high_is_more_recent=True),
        sr_levels=SRLevelsResult(resistance_levels=[resistance], support_levels=[support]),
    )
    counter = construct_trade_zone("sell", _stats(), up_leg, [], measured_moves=True)
    assert counter.structure_stop is None and counter.target_labels == {}
