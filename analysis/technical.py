from dataclasses import dataclass

import pandas as pd

SMA_WINDOW = 20
VOLATILITY_WINDOW = 20
TRADING_DAYS_1M = 21
TRADING_DAYS_3M = 63
TRADING_DAYS_6M = 126
TRADING_DAYS_PER_YEAR = 252

# Short-term trend (vs the 20-bar SMA) is now gated on how many standard
# deviations of the SAME 20-bar window's own prices the last price sits
# from that window's mean, not a flat percentage. Real bug found live
# 2026-08-21/22: a flat TREND_BAND_PCT=1.0% applied unscaled to H4/H1 bars
# (fed the same SMA_WINDOW=20 by ai/ftmo_suggest.py) made FX read "flat"
# almost everywhere — a real saved session showed EURUSD/GBPUSD H4 reads
# of "+0.1% vs 20-bar SMA" labeled flat, when that's actually a normal
# H4 move for FX, not a genuinely quiet one. A z-score band scales
# automatically with each instrument's OWN recent volatility and with
# timeframe (H1 bars have smaller absolute swings but proportionally
# similar relative behavior to D1), fixing the bias without needing a
# separate hardcoded band per timeframe. 0.5 standard deviations is a
# real, if modest, divergence from the window's own recent average — not
# an extreme outlier threshold, since "trend" here is meant to be a
# sensitive short-term read, not a rare-event flag.
TREND_BAND_Z = 0.5

# Medium-term "market pattern" window (~3 months of trading days) used for
# support/resistance and market-regime classification — deliberately
# longer than SMA_WINDOW's short-term trend read, so the two give
# complementary short- vs medium-term views rather than duplicating one
# another.
RANGE_WINDOW = 60

# market_regime is built from TWO genuinely different, complementary
# questions about the RANGE_WINDOW price path, not one:
#   (1) DIRECTION — has price made a real net move over the window at
#       all, scaled by the window's own volatility (same z-score idea as
#       TREND_BAND_Z above, just over the longer window)?
#   (2) EFFICIENCY — how directly did it get there (Kaufman's Efficiency
#       Ratio: net move / total path length; close to 1 = a clean,
#       straight-line move, close to 0 = a lot of back-and-forth churn
#       along the way)?
# Real bug found live 2026-08-21/22, found by testing against fresh MT5
# data rather than by reasoning about the thresholds alone: using
# efficiency ratio ALONE to answer BOTH questions at once labeled real,
# live FTMO instruments "sideways" even when they'd made a genuine net
# directional move over the window — e.g. AUDUSD's real D1 data showed
# efficiency_ratio=0.007 (i.e. almost pure back-and-forth noise by that
# measure alone) despite a real net move over the same 60 bars that a
# DIFFERENT, independently-computed detector in this codebase
# (chart_structure.py's swing-based higher-highs/higher-lows structure)
# also confirmed as a genuine uptrend on the identical bars. A "sideways"
# label was actively hiding that real direction rather than describing
# a genuine lack of one — a materially worse read for anyone deciding
# whether there's a trade here than a HONEST "yes it's moved, but
# choppily" would be. Raising the threshold instead of fixing the design
# (tried and empirically checked first, against this same real data)
# barely moved the mislabeling rate at all, confirming the two-question
# conflation was the actual defect, not just a mistuned number.
RANGE_DIRECTION_Z = 0.5
# Calibrated against real, live FTMO Market Watch D1 data (18 real
# instruments, fetched fresh) rather than picked from theory alone: since
# efficiency here is measured off the SAME half-window mean_shift as the
# direction check above (see that computation's own comment for why),
# its real achievable range on actual market data is much smaller than
# Kaufman's classic raw-endpoint-to-endpoint version — the live sample's
# own values topped out around 0.15, with most real "has genuine
# direction" instruments landing well under 0.10. 0.08 sits at roughly
# that live sample's own 75th percentile: the top quarter of genuinely-
# directional real instruments (the cleanest-moving ones) read as
# "trending", the rest as "choppy" — both are DIRECTIONAL calls now
# (see RANGE_DIRECTION_Z above), this threshold only decides how cleanly.
TRENDING_EFFICIENCY_RATIO = 0.08

# Momentum-acceleration: a SHORT-window sibling to market_regime's own
# direction check above, added 2026-08-26 direct user request after a
# real, recurring complaint (misreading pump/dump swings and sideways-
# vs-breakout moments) was traced to a genuine gap: RANGE_WINDOW=60 is
# wide enough that 3-5 pump/dump swings inside it can cancel out in the
# window's own first-half/second-half average, reading "sideways" even
# while a real, sharp new leg is actively underway right now — there was
# no shorter-window signal layered on top to catch that in-progress
# move. Deliberately reuses the EXACT same first-half-mean/second-half-
# mean-shift-over-std design as RANGE_DIRECTION_Z above (same statistical
# idea, just over a much shorter window) rather than a different
# approach, so the two stay conceptually consistent.
#
# MOMENTUM_WINDOW=12 is short enough to catch a fresh leg while still
# giving a clean 6/6 half-split. MOMENTUM_DIRECTION_Z is a principled
# placeholder, NOT yet validated against real live data the way
# RANGE_DIRECTION_Z/TRENDING_EFFICIENCY_RATIO were (see their own
# comments above for that process) — standard error of a mean scales as
# sigma/sqrt(n), so a threshold carrying roughly the same statistical
# stringency as RANGE_DIRECTION_Z=0.5 at n=30-per-half, scaled down to
# this window's n=6-per-half, is 0.5 * sqrt(30/6) ~= 1.1. Recalibrate
# this against real recent H1/H4 data for the specific instrument(s)
# this was built for before fully trusting it live — same "ship a
# principled first value, then check it against real data" process this
# file's other thresholds already went through.
MOMENTUM_WINDOW = 12
MOMENTUM_DIRECTION_Z = 1.1

# Average True Range window — the standard 14-period convention (Wilder's
# original), used here as a simple rolling mean rather than Wilder's
# smoothing for consistency with this file's other stats (plain windowed
# calculations, no exotic smoothing).
ATR_WINDOW = 14

# Velocity tiering — "fast" vs "slow" instruments, based purely on an
# instrument's OWN current H1 atr_pct (already computed above, on every
# existing caller — this needs no new data source at all). The real
# problem this answers: ATR is a bar-RANGE statistic (how far price
# typically travels across a whole H1 bar) and says nothing about the
# TIME PROFILE of that travel — a "normal" pullback on a fast/bursty
# instrument can cover a meaningful fraction of its own hourly range
# within minutes, while a slow/grinding instrument spreads the same
# relative range smoothly across the whole hour. A stop-distance
# discipline calibrated as if every instrument were the slow, diffusive
# case gets clipped by the fast case's own ordinary noise far more often
# — confirmed live 2026-09-08: a real XAUUSD trade's stop sat at well
# under half the instrument's own typical H1 range and was clipped
# within 93 minutes by perfectly ordinary movement, with the stated
# technical thesis (its own invalidation condition) never actually
# broken. The same day, a USDCAD short with roughly a third of gold's
# velocity took eighteen hours to cover a comparable relative distance.
#
# VELOCITY_FAST_THRESHOLD_PCT=0.25 is a first, principled cut from a
# real live comparison across this account's own Market Watch (2026-09-
# 08/09): FX majors/crosses typically ran 0.03-0.20%/hour (USDCNH lowest
# at 0.03%, most majors 0.10-0.20%), metals 0.31-0.85%/hour, equities
# 0.85-1.5%/hour — a real gap of 3x or more between the FX cluster and
# everything else, with this threshold sitting cleanly in the gap on
# both sides. NOT yet validated across many more days/instruments the
# way RANGE_DIRECTION_Z/TRENDING_EFFICIENCY_RATIO were (see those
# constants' own comments for that calibration process) — recalibrate
# against a wider real sample before fully trusting it right at the
# boundary, same "ship a principled first value, then check it against
# real data" process this file's other thresholds already went through.
VELOCITY_FAST_THRESHOLD_PCT = 0.25


def classify_velocity_tier(h1_atr_pct: float | None) -> str | None:
    """"fast" (gold/crypto/equities-style bursty movers, where a
    meaningful share of the hourly range can arrive in minutes) or "slow"
    (most FX majors/crosses, whose noise stays genuinely spread out) —
    see VELOCITY_FAST_THRESHOLD_PCT's own comment for the real data this
    threshold was cut from. None (never a guessed default, same
    convention as every other stat in this file) when h1_atr_pct itself
    isn't available — a caller with no real H1 ATR reading has no sound
    basis to assume either tier."""
    if h1_atr_pct is None:
        return None
    return "fast" if h1_atr_pct >= VELOCITY_FAST_THRESHOLD_PCT else "slow"

# RSI window — the standard 14-period convention, same simple-rolling-mean
# treatment as ATR above (not Wilder's smoothing).
RSI_WINDOW = 14

# Conventional RSI extremes — matches TechnicalStats.rsi's own documented
# convention (">70 overbought / <30 oversold") everywhere else in this
# codebase already uses. Kept as named constants (not inlined into
# classify_rsi_tier below) so every caller reads the exact same numbers
# this file's other RSI-consuming code already assumes.
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0


def classify_rsi_tier(rsi: float | None) -> str | None:
    """"overbought" / "oversold" / "neutral" from a plain, deterministic
    threshold check — added 2026-09-09 after a real, observed failure:
    a local model (qwen3:8b), reasoning freely over raw RSI numbers
    already labeled in the surrounding prose ("RSI 11 (>70 overbought/
    <30 oversold)"), repeatedly and consistently mislabeled a deeply
    OVERSOLD reading (11, then 15, then 15 again across consecutive
    polls) as "overbought" in its own written reasoning for a real,
    live USDCAD position. This is exactly the same failure class this
    codebase already fixed once before for stop-tightening direction
    (see TacticalSignals' own module docstring: "the backup model once
    proposed LOOSENING a stop while calling it 'tightening'") — a plain
    threshold comparison is arithmetic, not judgment, and has no reason
    to ever be left to a free-text model call when it can be computed
    once, deterministically, in Python and handed over as an
    already-resolved label instead. This function is that fix's shared
    building block; see ai.clerk_execution.TacticalSignals' own h1_rsi_
    tier/h4_rsi_tier fields for where it's actually wired into a live
    decision path. None (never a guessed default) when rsi itself is
    unavailable."""
    if rsi is None:
        return None
    if rsi > RSI_OVERBOUGHT:
        return "overbought"
    if rsi < RSI_OVERSOLD:
        return "oversold"
    return "neutral"


# Volume trend: recent average vs. a longer baseline average, expressed as
# a % difference — deliberately not a single "is volume high" verdict, so
# the model weighs it against price action itself rather than trusting a
# pre-made judgement call.
VOLUME_TREND_RECENT_WINDOW = 5
VOLUME_TREND_BASELINE_WINDOW = 20


@dataclass
class TechnicalStats:
    last_price: float | None
    sma20: float | None
    pct_vs_sma20: float | None
    trend: str | None  # "uptrend" / "downtrend" / "flat" (short-term, vs 20d SMA)
    change_1m_pct: float | None
    change_3m_pct: float | None
    change_6m_pct: float | None
    volatility_annualized_pct: float | None
    support: float | None  # rolling low over RANGE_WINDOW
    resistance: float | None  # rolling high over RANGE_WINDOW
    range_width_pct: float | None  # (resistance - support) / last_price
    market_regime: str | None  # "trending_up" / "trending_down" / "choppy_up" / "choppy_down" / "sideways" — see the market_regime computation's own comment for what each means
    atr: float | None  # Average True Range over ATR_WINDOW, in price units
    atr_pct: float | None  # atr / last_price * 100 — comparable across instruments
    rsi: float | None  # 0-100, momentum — conventionally overbought >70, oversold <30
    volume_trend_pct: float | None  # recent-vs-baseline average volume, %
    momentum_acceleration: str | None  # "accelerating_up" / "accelerating_down" / "stable" — a SHORT-window (MOMENTUM_WINDOW) sibling to market_regime, NOT a 4th market_regime value (see MOMENTUM_WINDOW's own comment); None only means insufficient history


def _compute_rsi(prices: pd.Series) -> float | None:
    """Classic Relative Strength Index over RSI_WINDOW days: average gain
    vs. average loss on up/down days, scaled to 0-100. None without a
    real window's worth of day-to-day changes."""
    if len(prices) < RSI_WINDOW + 1:
        return None

    delta = prices.diff().dropna().tail(RSI_WINDOW)
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _compute_volume_trend_pct(volume: pd.Series) -> float | None:
    """% difference between the recent-window average volume and a longer
    baseline-window average — positive means participation has picked up
    recently relative to its own normal level. None without a full
    baseline window of real (non-NaN) volume data."""
    volume = volume.dropna()
    if len(volume) < VOLUME_TREND_BASELINE_WINDOW:
        return None

    baseline = volume.tail(VOLUME_TREND_BASELINE_WINDOW).mean()
    if not baseline:
        return None
    recent = volume.tail(VOLUME_TREND_RECENT_WINDOW).mean()
    return float((recent - baseline) / baseline * 100)


def compute_atr(ohlc: pd.DataFrame) -> float | None:
    """Classic True Range (max of the day's own high-low range and its
    gap from the prior close) averaged over ATR_WINDOW days. None if
    there isn't enough history for a real window's worth of data —
    matches this file's rule of never fabricating a stat from a window
    shorter than it claims. Also None (rather than raising) if the given
    frame doesn't even have High/Low columns at all — some sources feeding
    this (e.g. data/psx_source.py's EOD history, which only has Open/
    Close/Volume) genuinely can't provide them, so this degrades exactly
    like "not enough rows" rather than crashing the whole stats computation
    for a source that just doesn't carry High/Low.

    Public (was _compute_atr) as of 2026-09-05 — ai/curiosity.py needs
    this exact same ATR baseline to score how "abnormal" a closed
    trade's post-exit price move was, and duplicating this logic in a
    second module risked the two definitions of ATR quietly drifting
    apart."""
    if not {"High", "Low", "Close"}.issubset(ohlc.columns):
        return None
    ohlc = ohlc.dropna(subset=["High", "Low", "Close"])
    if len(ohlc) < ATR_WINDOW + 1:
        return None

    high, low, close = ohlc["High"], ohlc["Low"], ohlc["Close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr = true_range.tail(ATR_WINDOW).mean()
    return float(atr) if pd.notna(atr) else None


def _classify_regime(window: pd.Series) -> str:
    """Classifies ONE window's own regime — DIRECTION first, then
    EFFICIENCY. Shared by compute_technical_stats (a single snapshot over
    the latest RANGE_WINDOW bars) and compute_regime_segments (this exact
    same classification walked across a ROLLING window to build a genuine
    history of when each regime began) so the two can never drift apart.

    DIRECTION: has this window's average LEVEL genuinely shifted,
    relative to the window's own volatility? Compares the first-half mean
    against the second-half mean rather than the two raw endpoint prices
    — a pure back-and-forth oscillation between two fixed levels can, by
    pure chance of exactly which day a window happens to start/end on,
    have its two single endpoints land on opposite extremes with no real
    underlying shift at all (confirmed while fixing this: a hand-built
    pure +5/-5 alternation with zero real trend read as "choppy_down"
    under a raw endpoint-to-endpoint z-score, purely from which of the two
    phases the window's first and last samples happened to land on).
    Averaging both halves smooths that sampling-boundary luck out far
    better than comparing two single days ever could — the same reason
    `trend` in compute_technical_stats compares against a mean, not a
    lone earlier price. Below RANGE_DIRECTION_Z, there's genuinely no
    real level shift to characterize further — "sideways" — regardless of
    how the path in between looked. See RANGE_DIRECTION_Z's own
    module-level comment for the real bug this two-step design fixes.

    EFFICIENCY (only reached once DIRECTION confirms a real move):
    decides whether it's been a clean, direct move ("trending") or a
    genuine net move reached via a lot of back-and-forth churn
    ("choppy"). A Kaufman's-Efficiency-Ratio-style measure: net move /
    total path length — close to 1 means the path length is barely more
    than the net move itself (clean); close to 0 means a lot of the path
    was retraced along the way. Uses the SAME mean_shift as the direction
    check (rather than Kaufman's own raw endpoint-to-endpoint delta) so
    both questions are answered off one consistent, sampling-boundary-
    robust measure of "how far did this window's level really move"
    instead of two different deltas that could, in principle, even
    disagree on sign at the edges."""
    half = len(window) // 2
    first_half_mean = float(window.iloc[:half].mean())
    second_half_mean = float(window.iloc[half:].mean())
    mean_shift = second_half_mean - first_half_mean
    window_std = float(window.std())
    direction_z = mean_shift / window_std if window_std > 0 else 0.0

    if abs(direction_z) < RANGE_DIRECTION_Z:
        return "sideways"

    daily_moves = window.diff().dropna().abs()
    total_path = float(daily_moves.sum())
    efficiency_ratio = abs(mean_shift) / total_path if total_path > 0 else 0.0
    is_efficient = efficiency_ratio >= TRENDING_EFFICIENCY_RATIO
    if mean_shift > 0:
        return "trending_up" if is_efficient else "choppy_up"
    return "trending_down" if is_efficient else "choppy_down"


@dataclass
class RegimeSegment:
    start_index: int  # bar index (0-based, oldest-first) in the ORIGINAL price series where this regime began
    regime: str  # "trending_up" / "trending_down" / "choppy_up" / "choppy_down" / "sideways" / "accumulation" / "distribution"


def compute_regime_segments(prices: pd.Series, window: int = RANGE_WINDOW) -> list["RegimeSegment"]:
    """Walks _classify_regime across a ROLLING window ending at every bar
    (rather than just the latest one, which is all market_regime in
    TechnicalStats ever gives you) to build a genuine history of when
    each regime began. Consecutive bars sharing the same label collapse
    into one segment; only real transitions are returned — added
    2026-09-03 direct user request to mark regime changes on the chart
    ("top to bottom straight lines to indicate accumulation, distribution
    regions, trending and sideways regimes").

    A "sideways" segment is further refined into "accumulation" (follows
    a down-leg: trending_down/choppy_down — Wyckoff-style, a range after
    selling suggests supply is being absorbed) or "distribution" (follows
    an up-leg: trending_up/choppy_up — a range after buying suggests
    supply is being distributed into it). This is inferred purely from
    which regime immediately preceded the range in THIS SAME data; a
    range with no preceding trend available (e.g. right at the start of
    the given history) stays labeled plain "sideways" rather than
    guessing a bias with nothing to infer it from.

    Needs at least 2*window bars to report anything — one full window to
    seed the very first classification, plus room for a genuine second
    reading alongside it — returns [] rather than guessing from less."""
    prices = prices.dropna()
    if len(prices) < window * 2:
        return []

    raw_labels = [
        _classify_regime(prices.iloc[end - window : end])
        for end in range(window, len(prices) + 1)
    ]

    # raw_labels[i] is the regime classification as of bar index (window - 1 + i)
    # in the original series — collapse consecutive repeats into segments.
    segments: list[RegimeSegment] = []
    for i, label in enumerate(raw_labels):
        bar_index = window - 1 + i
        if not segments or segments[-1].regime != label:
            segments.append(RegimeSegment(start_index=bar_index, regime=label))

    # Second pass: refine bare "sideways" segments using whatever bare
    # regime immediately preceded them (never a range's own label, since
    # consecutive identical labels are already collapsed above).
    for i, segment in enumerate(segments):
        if segment.regime != "sideways" or i == 0:
            continue
        preceding = segments[i - 1].regime
        if preceding in ("trending_down", "choppy_down"):
            segment.regime = "accumulation"
        elif preceding in ("trending_up", "choppy_up"):
            segment.regime = "distribution"

    return segments


def compute_technical_stats(
    prices: pd.Series, history: pd.DataFrame | None = None, periods_per_year: float = TRADING_DAYS_PER_YEAR
) -> TechnicalStats:
    """Pure computation over a close-price series, oldest first. `history`
    (High/Low/Close/Volume, same source/period) is optional and only
    feeds Average True Range (needs High/Low) and the volume-trend figure
    (needs Volume) — every other stat here, including RSI, only ever
    needed Close.

    Any stat that needs more history than is actually available (e.g. a
    3-month change on a contract that only just started trading) is left
    as None rather than computed from a shorter, misleading window.

    `periods_per_year` defaults to TRADING_DAYS_PER_YEAR (252) — correct
    for every existing caller, which all feed daily bars. It ONLY scales
    volatility_annualized_pct — the Kaufman efficiency ratio, RSI, and ATR
    are pure bar-count windows, not tied to a calendar-time assumption, so
    timeframe doesn't distort them. `trend` is NOT bar-count-only the same
    way: it's gated on a volatility-normalized z-score (TREND_BAND_Z), not
    a flat percentage, specifically because a flat percentage WAS
    distorted by timeframe (confirmed live: FTMO's own H4 reads showed
    ordinary ~0.1% moves labeled "flat" against a 1%-flat band sized for
    daily bars) — see TREND_BAND_Z's own module-level comment.
    A caller feeding a DIFFERENT bar frequency (confirmed live: FTMO's
    own H4/H1 reads, previously always scaled by sqrt(252) regardless of
    being fed 4-hour or 1-hour bars, understated real annualized
    volatility by roughly 2.5x-6x — H1 EURUSD read as LESS volatile than
    its own daily figure, the opposite of a sanity check) must pass the
    real number of bars per year for its own timeframe here."""
    prices = prices.dropna()
    if prices.empty:
        return TechnicalStats(*([None] * 17))

    last_price = float(prices.iloc[-1])

    sma20 = pct_vs_sma20 = trend = None
    if len(prices) >= SMA_WINDOW:
        window = prices.tail(SMA_WINDOW)
        sma20 = float(window.mean())
        pct_vs_sma20 = (last_price - sma20) / sma20 * 100
        window_std = float(window.std())
        if window_std > 0:
            trend_z = (last_price - sma20) / window_std
            if trend_z > TREND_BAND_Z:
                trend = "uptrend"
            elif trend_z < -TREND_BAND_Z:
                trend = "downtrend"
            else:
                trend = "flat"
        else:
            # A window with zero variance (every price identical) is
            # genuinely flat, not an undefined z-score.
            trend = "flat"

    def change_over(n_trading_days: int) -> float | None:
        if len(prices) <= n_trading_days:
            return None
        past_price = float(prices.iloc[-(n_trading_days + 1)])
        if past_price == 0:
            return None
        return (last_price - past_price) / past_price * 100

    change_1m_pct = change_over(TRADING_DAYS_1M)
    change_3m_pct = change_over(TRADING_DAYS_3M)
    change_6m_pct = change_over(TRADING_DAYS_6M)

    volatility_annualized_pct = None
    if len(prices) > VOLATILITY_WINDOW:
        period_returns = prices.pct_change().dropna().tail(VOLATILITY_WINDOW)
        if len(period_returns) >= 2:
            volatility_annualized_pct = float(
                period_returns.std() * (periods_per_year**0.5) * 100
            )

    support = resistance = range_width_pct = market_regime = None
    if len(prices) >= RANGE_WINDOW:
        window = prices.tail(RANGE_WINDOW)
        support = float(window.min())
        resistance = float(window.max())
        if last_price != 0:
            range_width_pct = (resistance - support) / last_price * 100
        market_regime = _classify_regime(window)

    # Direct fix for the "wakes up late in a sideways market" complaint
    # — see MOMENTUM_WINDOW's own module-level comment for the full
    # rationale. Independent of RANGE_WINDOW's own gate above (this can
    # compute fine even when the 60-bar window can't).
    momentum_acceleration = None
    if len(prices) >= MOMENTUM_WINDOW:
        momentum_window = prices.tail(MOMENTUM_WINDOW)
        momentum_half = MOMENTUM_WINDOW // 2
        momentum_shift = float(momentum_window.iloc[momentum_half:].mean()) - float(
            momentum_window.iloc[:momentum_half].mean()
        )
        momentum_std = float(momentum_window.std())
        momentum_z = momentum_shift / momentum_std if momentum_std > 0 else 0.0
        if momentum_z > MOMENTUM_DIRECTION_Z:
            momentum_acceleration = "accelerating_up"
        elif momentum_z < -MOMENTUM_DIRECTION_Z:
            momentum_acceleration = "accelerating_down"
        else:
            momentum_acceleration = "stable"

    atr = atr_pct = None
    if history is not None and not history.empty:
        atr = compute_atr(history)
        if atr is not None and last_price:
            atr_pct = atr / last_price * 100

    rsi = _compute_rsi(prices)

    volume_trend_pct = None
    if history is not None and "Volume" in history.columns:
        volume_trend_pct = _compute_volume_trend_pct(history["Volume"])

    return TechnicalStats(
        last_price=last_price,
        sma20=sma20,
        pct_vs_sma20=pct_vs_sma20,
        trend=trend,
        change_1m_pct=change_1m_pct,
        change_3m_pct=change_3m_pct,
        change_6m_pct=change_6m_pct,
        volatility_annualized_pct=volatility_annualized_pct,
        support=support,
        resistance=resistance,
        range_width_pct=range_width_pct,
        market_regime=market_regime,
        atr=atr,
        atr_pct=atr_pct,
        rsi=rsi,
        volume_trend_pct=volume_trend_pct,
        momentum_acceleration=momentum_acceleration,
    )
