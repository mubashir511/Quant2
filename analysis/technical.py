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

# Average True Range window — the standard 14-period convention (Wilder's
# original), used here as a simple rolling mean rather than Wilder's
# smoothing for consistency with this file's other stats (plain windowed
# calculations, no exotic smoothing).
ATR_WINDOW = 14

# RSI window — the standard 14-period convention, same simple-rolling-mean
# treatment as ATR above (not Wilder's smoothing).
RSI_WINDOW = 14

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


def _compute_atr(ohlc: pd.DataFrame) -> float | None:
    """Classic True Range (max of the day's own high-low range and its
    gap from the prior close) averaged over ATR_WINDOW days. None if
    there isn't enough history for a real window's worth of data —
    matches this file's rule of never fabricating a stat from a window
    shorter than it claims. Also None (rather than raising) if the given
    frame doesn't even have High/Low columns at all — some sources feeding
    this (e.g. data/psx_source.py's EOD history, which only has Open/
    Close/Volume) genuinely can't provide them, so this degrades exactly
    like "not enough rows" rather than crashing the whole stats computation
    for a source that just doesn't carry High/Low."""
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
        return TechnicalStats(*([None] * 16))

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

        # DIRECTION first: has this window's average LEVEL genuinely
        # shifted, relative to the window's own volatility? Compares the
        # first-half mean against the second-half mean rather than the two
        # raw endpoint prices — a pure back-and-forth oscillation between
        # two fixed levels can, by pure chance of exactly which day a
        # window happens to start/end on, have its two single endpoints
        # land on opposite extremes with no real underlying shift at all
        # (confirmed while fixing this: a hand-built pure +5/-5 alternation
        # with zero real trend read as "choppy_down" under a raw endpoint-
        # to-endpoint z-score, purely from which of the two phases the
        # window's first and last samples happened to land on). Averaging
        # 30 days on each side smooths that sampling-boundary luck out far
        # better than comparing two single days ever could — the same
        # reason `trend` above compares against a mean, not a lone
        # earlier price. Below RANGE_DIRECTION_Z, there's genuinely no
        # real level shift to characterize further — "sideways" —
        # regardless of how the path in between looked. See this
        # section's own module-level comment (above RANGE_DIRECTION_Z)
        # for the real bug this two-step design fixes.
        half = RANGE_WINDOW // 2
        first_half_mean = float(window.iloc[:half].mean())
        second_half_mean = float(window.iloc[half:].mean())
        mean_shift = second_half_mean - first_half_mean
        window_std = float(window.std())
        direction_z = mean_shift / window_std if window_std > 0 else 0.0

        if abs(direction_z) < RANGE_DIRECTION_Z:
            market_regime = "sideways"
        else:
            # There IS a real direction — EFFICIENCY now decides whether
            # it's been a clean, direct move ("trending") or a genuine net
            # move reached via a lot of back-and-forth churn ("choppy").
            # A Kaufman's-Efficiency-Ratio-style measure: net move / total
            # path length — close to 1 means the path length is barely
            # more than the net move itself (clean); close to 0 means a
            # lot of the path was retraced along the way. Uses the SAME
            # mean_shift as the direction check above (rather than
            # Kaufman's own raw endpoint-to-endpoint delta) so both
            # questions are answered off one consistent, sampling-
            # boundary-robust measure of "how far did this window's level
            # really move" instead of two different deltas that could, in
            # principle, even disagree on sign at the edges.
            daily_moves = window.diff().dropna().abs()
            total_path = float(daily_moves.sum())
            efficiency_ratio = abs(mean_shift) / total_path if total_path > 0 else 0.0
            is_efficient = efficiency_ratio >= TRENDING_EFFICIENCY_RATIO
            if mean_shift > 0:
                market_regime = "trending_up" if is_efficient else "choppy_up"
            else:
                market_regime = "trending_down" if is_efficient else "choppy_down"

    atr = atr_pct = None
    if history is not None and not history.empty:
        atr = _compute_atr(history)
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
    )
