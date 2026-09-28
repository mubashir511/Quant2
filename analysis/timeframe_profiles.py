"""Per-timeframe threshold profiles for analysis.chart_structure and
analysis.setup_classifier (2026-09-24, intraday decision-tier upgrade).

The structure/setup engines are timeframe-agnostic by BAR COUNT, but several
of their flat-percentage thresholds were tuned on D1/H1 prices: a 1.0%
"flat trendline" band or a 0.5% double-top pullback is a normal move on H1
and an entire day's range on M5, so on M5/M15 bars those reads would always
say "flat" and never find a pattern. An intraday profile expresses those
thresholds as MULTIPLES OF THE WINDOW'S OWN ATR% instead, so they scale to
whatever instrument/timeframe they are handed with no per-symbol calibration.

`profile=None` everywhere means "exactly the pre-existing constants" — the
D1/H4/H1/MN1 reads are untouched by this module.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeframeProfile:
    name: str
    structure_lookback: int
    sr_tolerance_floor_pct: float
    sr_half_life_bars: float
    # ATR%-relative multiples (None -> keep the flat default constant).
    trendline_flat_atr_multiple: float | None = None
    double_pullback_atr_multiple: float | None = None
    breakout_close_through_atr_multiple: float | None = None
    near_level_atr_multiple: float | None = None


@dataclass(frozen=True)
class ResolvedThresholds:
    trendline_flat_pct: float
    double_pullback_pct: float
    breakout_close_through_pct: float


# Multiples chosen so that on a typical H1 read (ATR ~0.3%) they land on
# roughly today's flat constants (1.0% / 0.5% / 0.1% / 0.5%): 3.0x, 1.5x,
# 0.3x, 1.5x. Verified against real M5 broker bars before shipping. (An M15
# profile existed briefly on 2026-09-24 and was dropped: M5 + H1 split its duties.)
M5_PROFILE = TimeframeProfile(
    name="M5",
    structure_lookback=144,  # 12 hours of bars
    sr_tolerance_floor_pct=0.01,
    sr_half_life_bars=48.0,
    trendline_flat_atr_multiple=3.0,
    double_pullback_atr_multiple=1.5,
    breakout_close_through_atr_multiple=0.3,
    near_level_atr_multiple=1.5,
)

INTRADAY_PROFILES = {"M5": M5_PROFILE}


def resolve_thresholds(
    profile: TimeframeProfile,
    atr_pct: float | None,
    *,
    default_trendline_flat_pct: float,
    default_double_pullback_pct: float,
    default_breakout_close_through_pct: float,
) -> ResolvedThresholds:
    """Concrete percent thresholds for one window. An unavailable ATR% (too
    little history) falls back to the flat defaults, same "never fabricate"
    rule as every other ATR-derived read in this codebase."""

    def _pick(multiple: float | None, default: float) -> float:
        if multiple is None or atr_pct is None or atr_pct <= 0:
            return default
        return multiple * atr_pct

    return ResolvedThresholds(
        trendline_flat_pct=_pick(profile.trendline_flat_atr_multiple, default_trendline_flat_pct),
        double_pullback_pct=_pick(profile.double_pullback_atr_multiple, default_double_pullback_pct),
        breakout_close_through_pct=_pick(
            profile.breakout_close_through_atr_multiple, default_breakout_close_through_pct
        ),
    )


def near_level_tolerance_pct(profile: TimeframeProfile | None, atr_pct: float | None, default: float) -> float:
    if profile is None or profile.near_level_atr_multiple is None or atr_pct is None or atr_pct <= 0:
        return default
    return profile.near_level_atr_multiple * atr_pct
