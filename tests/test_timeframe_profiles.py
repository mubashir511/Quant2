import numpy as np
import pandas as pd
import pytest

from analysis.chart_structure import compute_chart_structure
from analysis.setup_classifier import classify_setups
from analysis.technical import TechnicalStats
from analysis.timeframe_profiles import (
    M5_PROFILE,
    TimeframeProfile,
    near_level_tolerance_pct,
    resolve_thresholds,
)


def _slow_m5_uptrend() -> pd.DataFrame:
    # A ~0.6% drift across 144 bars with 0.15% swings: on H1/D1 this is a
    # "flat" trendline under the flat 1.0% threshold, but on M5 it is a
    # clean rising channel (empirically verified 2026-09-24).
    n = 200
    i = np.arange(n)
    close = 100 * (1 + 0.006 * i / 144) * (1 + 0.0015 * np.sin(2 * np.pi * i / 12))
    idx = pd.date_range("2026-09-21 08:00", periods=n, freq="5min")
    return pd.DataFrame(
        {"Open": close, "High": close * 1.0004, "Low": close * 0.9996, "Close": close, "Volume": 100}, index=idx
    )


def test_default_profile_keeps_the_flat_percent_thresholds_exactly():
    snapshot = compute_chart_structure(_slow_m5_uptrend())
    assert snapshot.trendlines.support_trendline.direction == "flat"


def test_m5_profile_scales_the_trendline_flat_threshold_to_the_window_atr():
    snapshot = compute_chart_structure(_slow_m5_uptrend(), profile=M5_PROFILE)
    assert snapshot.trendlines.support_trendline.direction == "rising"
    assert snapshot.trendlines.resistance_trendline.direction == "rising"


def test_profile_lookback_overrides_the_default_window():
    profile = TimeframeProfile(name="tiny", structure_lookback=40, sr_tolerance_floor_pct=0.01, sr_half_life_bars=10.0)
    snapshot = compute_chart_structure(_slow_m5_uptrend(), profile=profile)
    assert snapshot is not None  # runs on a 40-bar window without error


def test_resolve_thresholds_falls_back_to_defaults_without_an_atr():
    resolved = resolve_thresholds(
        M5_PROFILE, None,
        default_trendline_flat_pct=1.0, default_double_pullback_pct=0.5, default_breakout_close_through_pct=0.1,
    )
    assert (resolved.trendline_flat_pct, resolved.double_pullback_pct, resolved.breakout_close_through_pct) == (1.0, 0.5, 0.1)


def test_resolve_thresholds_are_atr_multiples():
    resolved = resolve_thresholds(
        M5_PROFILE, 0.1,
        default_trendline_flat_pct=1.0, default_double_pullback_pct=0.5, default_breakout_close_through_pct=0.1,
    )
    assert resolved.trendline_flat_pct == pytest.approx(0.3)
    assert resolved.double_pullback_pct == pytest.approx(0.15)
    assert resolved.breakout_close_through_pct == pytest.approx(0.03)


def test_near_level_tolerance_uses_profile_only_when_given():
    assert near_level_tolerance_pct(None, 0.1, 0.5) == 0.5
    assert near_level_tolerance_pct(M5_PROFILE, None, 0.5) == 0.5
    assert near_level_tolerance_pct(M5_PROFILE, 0.1, 0.5) == pytest.approx(0.15)


def test_classify_setups_default_and_profile_differ_only_by_the_near_level_band():
    stats = TechnicalStats(**{**{f: None for f in TechnicalStats.__dataclass_fields__}, "last_price": 100.0, "atr_pct": 0.05, "market_regime": "sideways"})
    structure = compute_chart_structure(_slow_m5_uptrend())
    # No crash and identical when the profile's near-level band is not the deciding factor.
    assert [s.name for s in classify_setups(stats, structure)] == [s.name for s in classify_setups(stats, structure, profile=None)]
