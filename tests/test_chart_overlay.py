from datetime import datetime

from ai.chart_overlay import OverlaySymbolInputs, build_overlay_lines, write_chart_overlay
from analysis.chart_structure import (
    ChartPattern,
    ChartStructureSnapshot,
    FibonacciLevels,
    SRLevel,
    SRLevelsResult,
    Trendline,
    TrendlineAnalysis,
)
from analysis.technical import RegimeSegment, TechnicalStats
from data.mt5_source import PendingOrder, Position


def _ts(atr=None, last_price=None) -> TechnicalStats:
    return TechnicalStats(
        last_price=last_price, sma20=None, pct_vs_sma20=None, trend=None,
        change_1m_pct=None, change_3m_pct=None, change_6m_pct=None,
        volatility_annualized_pct=None, support=None, resistance=None,
        range_width_pct=None, market_regime=None, atr=atr,
        atr_pct=None, rsi=None, volume_trend_pct=None, momentum_acceleration=None,
    )


def _empty_structure() -> ChartStructureSnapshot:
    return ChartStructureSnapshot(fibonacci=None, sr_levels=None, trendlines=None, patterns=[])


def _position(symbol="XAUUSD", side="buy", sl=1980.0, tp=2100.0, price_open=2000.0):
    return Position(
        symbol=symbol, volume=1.0, side=side, price_open=price_open, price_current=price_open,
        sl=sl, profit=0.0, opened_at=datetime(2026, 1, 1), ticket=1, tp=tp,
    )


def _pending_order(symbol="XAUUSD", order_type="buy limit", sl=1980.0, tp=2100.0, price_open=2000.0):
    return PendingOrder(symbol=symbol, volume=1.0, order_type=order_type, price_open=price_open, sl=sl, tp=tp, ticket=1)


def test_build_overlay_lines_empty_structure_and_no_position_produces_nothing():
    inputs = OverlaySymbolInputs(symbol="XAUUSD", h4_structure=_empty_structure(), h1_structure=_empty_structure())
    assert build_overlay_lines(inputs) == []


def test_build_overlay_lines_sr_levels_labeled_by_timeframe_and_kind():
    structure = ChartStructureSnapshot(
        fibonacci=None,
        sr_levels=SRLevelsResult(
            resistance_levels=[SRLevel(price=2050.0, touches=3, distance_pct=2.5)],
            support_levels=[SRLevel(price=1980.0, touches=5, distance_pct=-1.0)],
        ),
        trendlines=None,
        patterns=[],
    )
    inputs = OverlaySymbolInputs(symbol="XAUUSD", h4_structure=structure)
    lines = build_overlay_lines(inputs)
    resistance = next(l for l in lines if l.startswith("LEVEL|XAUUSD|2050|resistance"))
    support = next(l for l in lines if l.startswith("LEVEL|XAUUSD|1980|support"))
    assert resistance.endswith("|0.3")
    assert "H4" in resistance and "3x" in resistance
    assert support.endswith("|0.3")
    assert "H4" in support and "5x" in support


def test_build_overlay_lines_uses_the_real_tolerance_actually_used_not_the_flat_default():
    # Real bug found on independent audit, fixed 2026-09-10: this used to
    # hardcode the flat SR_CLUSTER_TOLERANCE_PCT module constant as every
    # level's own exported zone-width, even though analysis.chart_
    # structure.compute_chart_structure scales this tolerance by the
    # instrument's own ATR% for a fast mover (confirmed live: up to 0.9%
    # on INTC) — the terminal overlay was silently drawing a NARROWER
    # zone than what Python actually used to decide these were "the same
    # level." SRLevelsResult.tolerance_pct now carries the REAL value.
    structure = ChartStructureSnapshot(
        fibonacci=None,
        sr_levels=SRLevelsResult(
            resistance_levels=[SRLevel(price=2050.0, touches=3, distance_pct=2.5)],
            support_levels=[],
            tolerance_pct=0.9,
        ),
        trendlines=None,
        patterns=[],
    )
    inputs = OverlaySymbolInputs(symbol="XAUUSD", h4_structure=structure)
    lines = build_overlay_lines(inputs)
    resistance = next(l for l in lines if l.startswith("LEVEL|XAUUSD|2050|resistance"))
    assert resistance.endswith("|0.9")
    assert not resistance.endswith("|0.3")


def test_build_overlay_lines_flags_liquidity_pool_levels_in_the_exported_label():
    structure = ChartStructureSnapshot(
        fibonacci=None,
        sr_levels=SRLevelsResult(
            resistance_levels=[SRLevel(price=2050.0, touches=3, distance_pct=2.5, is_liquidity_pool=True)],
            support_levels=[SRLevel(price=1980.0, touches=1, distance_pct=-1.0, is_liquidity_pool=False)],
        ),
        trendlines=None,
        patterns=[],
    )
    inputs = OverlaySymbolInputs(symbol="XAUUSD", h4_structure=structure)
    lines = build_overlay_lines(inputs)
    resistance = next(l for l in lines if l.startswith("LEVEL|XAUUSD|2050|resistance"))
    support = next(l for l in lines if l.startswith("LEVEL|XAUUSD|1980|support"))
    assert "[LIQUIDITY POOL]" in resistance
    assert "[LIQUIDITY POOL]" not in support


def test_build_overlay_lines_caps_sr_levels_per_side():
    levels = [SRLevel(price=2000.0 + i, touches=1, distance_pct=0.0) for i in range(5)]
    structure = ChartStructureSnapshot(
        fibonacci=None, sr_levels=SRLevelsResult(resistance_levels=levels, support_levels=[]),
        trendlines=None, patterns=[],
    )
    inputs = OverlaySymbolInputs(symbol="XAUUSD", h4_structure=structure)
    lines = build_overlay_lines(inputs)
    assert len([l for l in lines if l.startswith("LEVEL")]) == 3


def test_build_overlay_lines_fibo_uses_real_time_anchors():
    structure = ChartStructureSnapshot(
        fibonacci=FibonacciLevels(
            swing_high=2100.0, swing_low=1900.0, high_is_more_recent=True,
            levels={"38.2%": 2023.6, "61.8%": 1976.4}, current_price=1980.0,
            nearest_level_name="61.8%", nearest_level_price=1976.4, distance_to_nearest_pct=0.18,
            swing_high_bars_ago=5, swing_low_bars_ago=30,
        ),
        sr_levels=None, trendlines=None, patterns=[],
    )
    inputs = OverlaySymbolInputs(symbol="XAUUSD", h1_structure=structure)
    lines = build_overlay_lines(inputs)
    # H1 bar = 3600s; high 5 bars ago = 18000s, low 30 bars ago = 108000s
    assert any(l.startswith("FIBO|XAUUSD|2100|1900|1|18000|108000|H1") for l in lines)


def test_build_overlay_lines_no_standalone_goldenzone_line():
    # v5, direct user feedback: a separate GOLDENZONE rectangle drew the
    # exact same price range as two of the EA's own labeled Fibonacci
    # bands, stacking two overlapping, ambiguously-labeled tints and
    # producing "multiple hatched regions... don't know what represents
    # what." The EA now fills/labels each fib band itself; no GOLDENZONE
    # line should ever be emitted.
    structure = ChartStructureSnapshot(
        fibonacci=FibonacciLevels(
            swing_high=2100.0, swing_low=1900.0, high_is_more_recent=True,
            levels={"38.2%": 2023.6, "61.8%": 1976.4}, current_price=1980.0,
            nearest_level_name="61.8%", nearest_level_price=1976.4, distance_to_nearest_pct=0.18,
        ),
        sr_levels=None, trendlines=None, patterns=[],
    )
    inputs = OverlaySymbolInputs(symbol="XAUUSD", h1_structure=structure)
    lines = build_overlay_lines(inputs)
    assert not any(l.startswith("GOLDENZONE") for l in lines)


def test_build_overlay_lines_only_h1_fibo_shown_not_h4():
    # Direct user feedback 2026-09-04: showing BOTH H4's and H1's own
    # independent Fibonacci ladder at once looked like "2 fibonacci
    # level systems" and was confusing, not additive the way two
    # independent S/R levels are. Only H1 (closer to current price
    # action) should ever produce a FIBO line; H4's own S/R zones and
    # trendline are unaffected.
    fib = FibonacciLevels(
        swing_high=2100.0, swing_low=1900.0, high_is_more_recent=True,
        levels={"38.2%": 2023.6, "61.8%": 1976.4}, current_price=1980.0,
        nearest_level_name="61.8%", nearest_level_price=1976.4, distance_to_nearest_pct=0.18,
    )
    h4_structure = ChartStructureSnapshot(fibonacci=fib, sr_levels=None, trendlines=None, patterns=[])
    h1_structure = ChartStructureSnapshot(fibonacci=fib, sr_levels=None, trendlines=None, patterns=[])
    inputs = OverlaySymbolInputs(symbol="XAUUSD", h4_structure=h4_structure, h1_structure=h1_structure)
    lines = build_overlay_lines(inputs)
    assert any(l.startswith("FIBO|XAUUSD") and "|H1" in l for l in lines)
    assert not any(l.startswith("FIBO|XAUUSD") and "|H4" in l for l in lines)


def test_build_overlay_lines_trendline_uses_real_seconds_not_chart_bar_offset():
    # Real bug found live 2026-09-04: v1 anchored trendlines using the
    # CHART's own bar offset (iTime(_Symbol, PERIOD_CURRENT, 20)) while
    # the price math assumed the ANALYSIS timeframe's bar spacing (H4/H1)
    # -- a real unit mismatch that drew "completely random" diagonal
    # lines whenever the chart's period didn't match the analysis
    # timeframe (e.g. an M5 chart with an H4-computed trendline).
    structure = ChartStructureSnapshot(
        fibonacci=None, sr_levels=None,
        trendlines=TrendlineAnalysis(
            resistance_trendline=None,
            support_trendline=Trendline(
                slope_per_bar=0.5, direction="rising", current_price_on_line=2000.0,
                price_vs_line="above", distance_pct=1.0, bars_since_last_point=2,
            ),
        ),
        patterns=[],
    )
    inputs = OverlaySymbolInputs(symbol="XAUUSD", h4_structure=structure)
    lines = build_overlay_lines(inputs)
    # price 20 H4-bars back = 2000.0 - 0.5*20 = 1990.0; 20 H4 bars = 20*14400 = 288000s
    assert any(l.startswith("TREND|XAUUSD|1990|2000|288000|support|H4") for l in lines)


def test_build_overlay_lines_pattern_becomes_a_note():
    structure = ChartStructureSnapshot(
        fibonacci=None, sr_levels=None, trendlines=None,
        patterns=[ChartPattern(name="double_top", detail="Two peaks near 2050.")],
    )
    inputs = OverlaySymbolInputs(symbol="XAUUSD", h4_structure=structure)
    lines = build_overlay_lines(inputs)
    assert "NOTE|XAUUSD|H4 pattern: double_top" in lines


def test_build_overlay_lines_atr_zone_from_h1_stats():
    inputs = OverlaySymbolInputs(
        symbol="XAUUSD", h1_stats=_ts(atr=10.0, last_price=2000.0),
    )
    lines = build_overlay_lines(inputs)
    # 1.75 * 10 = 17.5 -> low=1982.5, high=2017.5
    assert any(l.startswith("ZONE|XAUUSD|1982.5|2017.5|") and "1.75x H1 ATR" in l for l in lines)


def test_build_overlay_lines_atr_zone_skipped_without_atr():
    inputs = OverlaySymbolInputs(symbol="XAUUSD", h1_stats=_ts(atr=None, last_price=2000.0))
    lines = build_overlay_lines(inputs)
    assert not any(l.startswith("ZONE") for l in lines)


def test_build_overlay_lines_live_position_draws_entry_stop_target():
    inputs = OverlaySymbolInputs(symbol="XAUUSD", position=_position(side="sell", sl=2020.0, tp=1950.0, price_open=2000.0))
    lines = build_overlay_lines(inputs)
    assert "STOP|XAUUSD|2020" in lines
    assert "TARGET|XAUUSD|1950" in lines
    assert "ENTRY|XAUUSD|2000|sell" in lines


def test_build_overlay_lines_pending_order_used_when_no_live_position():
    inputs = OverlaySymbolInputs(symbol="XAUUSD", pending_order=_pending_order(order_type="sell limit", sl=2020.0, tp=1950.0))
    lines = build_overlay_lines(inputs)
    assert "ENTRY|XAUUSD|2000|sell (pending)" in lines


def test_build_overlay_lines_live_position_takes_priority_over_pending_order():
    # A symbol can carry a leftover pending-order record alongside an
    # already-filled position (e.g. an unrelated top-up order) — the
    # live position is what actually matters to draw.
    inputs = OverlaySymbolInputs(
        symbol="XAUUSD",
        position=_position(price_open=2000.0),
        pending_order=_pending_order(price_open=1900.0),
    )
    lines = build_overlay_lines(inputs)
    assert "ENTRY|XAUUSD|2000|buy" in lines
    assert not any("1900" in l for l in lines)


def test_build_overlay_lines_pending_setup_with_no_real_mt5_order():
    inputs = OverlaySymbolInputs(symbol="XAUUSD", entry_price=2000.0, stop_loss=1980.0, take_profit=2050.0, side="buy")
    lines = build_overlay_lines(inputs)
    assert "ENTRY|XAUUSD|2000|buy (watching)" in lines
    assert "STOP|XAUUSD|1980" in lines
    assert "TARGET|XAUUSD|2050" in lines


def test_build_overlay_lines_note_is_cleaned_and_truncated():
    inputs = OverlaySymbolInputs(symbol="XAUUSD", note="Line one\nwith a | pipe and\nnewlines" + "x" * 300)
    lines = build_overlay_lines(inputs)
    note_lines = [l for l in lines if l.startswith("NOTE|XAUUSD|")]
    assert len(note_lines) == 1
    note_text = note_lines[0].split("|", 2)[2]
    assert "\n" not in note_text
    assert len(note_text) <= 180


def test_build_overlay_lines_log_line_summarizes_live_position():
    inputs = OverlaySymbolInputs(
        symbol="XAUUSD", position=_position(side="buy", sl=1980.0, tp=2100.0, price_open=2000.0),
        note="Structurally-backed pullback within an aligned uptrend.",
    )
    lines = build_overlay_lines(inputs)
    log_line = next(l for l in lines if l.startswith("LOG|XAUUSD|"))
    assert "HOLDING BUY @ 2000" in log_line
    assert "stop 1980" in log_line
    assert "target 2100" in log_line
    assert "Structurally-backed pullback" in log_line


def test_build_overlay_lines_log_line_summarizes_pending_order():
    inputs = OverlaySymbolInputs(
        symbol="NVDA", pending_order=_pending_order(order_type="buy limit", price_open=220.0),
        note="Waiting for a pullback into H1 support.",
    )
    lines = build_overlay_lines(inputs)
    log_line = next(l for l in lines if l.startswith("LOG|NVDA|"))
    assert "WATCHING resting BUY limit @ 220" in log_line
    assert "Waiting for a pullback" in log_line


def test_build_overlay_lines_log_line_absent_without_any_context():
    inputs = OverlaySymbolInputs(symbol="XAUUSD")
    lines = build_overlay_lines(inputs)
    assert not any(l.startswith("LOG|") for l in lines)


def test_build_overlay_lines_no_vlines_without_regime_segments():
    inputs = OverlaySymbolInputs(symbol="XAUUSD")
    lines = build_overlay_lines(inputs)
    assert not any(l.startswith("VLINE|") for l in lines)


def test_build_overlay_lines_vline_uses_real_seconds_ago_from_bar_count():
    # bar_count=100, start_index=79 -> bars_ago = 99 - 79 = 20 -> 20*3600s
    segments = [RegimeSegment(start_index=79, regime="trending_up")]
    inputs = OverlaySymbolInputs(symbol="XAUUSD", h1_regime_segments=segments, h1_bar_count=100)
    lines = build_overlay_lines(inputs)
    vlines = [l for l in lines if l.startswith("VLINE|")]
    assert vlines == ["VLINE|XAUUSD|72000|uptrend|Uptrend begins"]


def test_build_overlay_lines_vline_categories_collapse_choppy_into_its_own_direction():
    segments = [
        RegimeSegment(start_index=0, regime="choppy_down"),
        RegimeSegment(start_index=10, regime="sideways"),
        RegimeSegment(start_index=20, regime="accumulation"),
        RegimeSegment(start_index=30, regime="trending_up"),
        RegimeSegment(start_index=40, regime="distribution"),
    ]
    inputs = OverlaySymbolInputs(symbol="XAUUSD", h1_regime_segments=segments, h1_bar_count=50)
    lines = build_overlay_lines(inputs)
    vlines = [l for l in lines if l.startswith("VLINE|")]
    categories = [l.split("|")[3] for l in vlines]
    assert categories == ["downtrend", "sideways", "accumulation", "uptrend", "distribution"]


def test_build_overlay_lines_vline_caps_to_most_recent_markers():
    segments = [RegimeSegment(start_index=i * 10, regime="sideways") for i in range(12)]
    inputs = OverlaySymbolInputs(symbol="XAUUSD", h1_regime_segments=segments, h1_bar_count=200)
    lines = build_overlay_lines(inputs)
    vlines = [l for l in lines if l.startswith("VLINE|")]
    assert len(vlines) == 8
    # Kept the most RECENT transitions (highest start_index), not the oldest:
    # start_index=110 of 200 bars -> bars_ago = 199-110 = 89 -> 89*3600s.
    assert vlines[-1] == "VLINE|XAUUSD|320400|sideways|Sideways range begins"


def test_write_chart_overlay_writes_combined_file(tmp_path):
    ok = write_chart_overlay(
        {"XAUUSD": ["LEVEL|XAUUSD|2000|support|test"], "EURUSD": ["LEVEL|EURUSD|1.1|support|test"]},
        tmp_path,
    )
    assert ok is True
    written = (tmp_path / "Files" / "quant2_overlay.txt").read_text(encoding="utf-8")
    assert "XAUUSD" in written
    assert "EURUSD" in written


def test_write_chart_overlay_fully_replaces_prior_content(tmp_path):
    write_chart_overlay({"XAUUSD": ["LEVEL|XAUUSD|2000|support|old"]}, tmp_path)
    write_chart_overlay({"EURUSD": ["LEVEL|EURUSD|1.1|support|new"]}, tmp_path)
    written = (tmp_path / "Files" / "quant2_overlay.txt").read_text(encoding="utf-8")
    assert "XAUUSD" not in written
    assert "EURUSD" in written


def test_write_chart_overlay_empty_input_writes_empty_file(tmp_path):
    ok = write_chart_overlay({}, tmp_path)
    assert ok is True
    written = (tmp_path / "Files" / "quant2_overlay.txt").read_text(encoding="utf-8")
    assert written == ""


def test_write_chart_overlay_returns_false_on_failure(tmp_path):
    # Point at a path that can't be created as a directory (a file
    # already sits where a directory needs to go).
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    ok = write_chart_overlay({"XAUUSD": ["NOTE|XAUUSD|x"]}, blocker)
    assert ok is False
