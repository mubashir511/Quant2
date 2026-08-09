from dataclasses import dataclass

import pandas as pd

SMA_WINDOW = 20
VOLATILITY_WINDOW = 20
TRADING_DAYS_1M = 21
TRADING_DAYS_3M = 63
TRADING_DAYS_6M = 126
TRADING_DAYS_PER_YEAR = 252
TREND_BAND_PCT = 1.0

# Medium-term "market pattern" window (~3 months of trading days) used for
# support/resistance and market-regime classification — deliberately
# longer than SMA_WINDOW's short-term trend read, so the two give
# complementary short- vs medium-term views rather than duplicating one
# another.
RANGE_WINDOW = 60
TRENDING_EFFICIENCY_RATIO = 0.5
SIDEWAYS_EFFICIENCY_RATIO = 0.25

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
    market_regime: str | None  # "trending_up" / "trending_down" / "sideways" / "mixed"
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
    shorter than it claims."""
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
    prices: pd.Series, history: pd.DataFrame | None = None
) -> TechnicalStats:
    """Pure computation over a close-price series, oldest first. `history`
    (High/Low/Close/Volume, same source/period) is optional and only
    feeds Average True Range (needs High/Low) and the volume-trend figure
    (needs Volume) — every other stat here, including RSI, only ever
    needed Close.

    Any stat that needs more history than is actually available (e.g. a
    3-month change on a contract that only just started trading) is left
    as None rather than computed from a shorter, misleading window.
    """
    prices = prices.dropna()
    if prices.empty:
        return TechnicalStats(*([None] * 16))

    last_price = float(prices.iloc[-1])

    sma20 = pct_vs_sma20 = trend = None
    if len(prices) >= SMA_WINDOW:
        sma20 = float(prices.tail(SMA_WINDOW).mean())
        pct_vs_sma20 = (last_price - sma20) / sma20 * 100
        if pct_vs_sma20 > TREND_BAND_PCT:
            trend = "uptrend"
        elif pct_vs_sma20 < -TREND_BAND_PCT:
            trend = "downtrend"
        else:
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
        daily_returns = prices.pct_change().dropna().tail(VOLATILITY_WINDOW)
        if len(daily_returns) >= 2:
            volatility_annualized_pct = float(
                daily_returns.std() * (TRADING_DAYS_PER_YEAR**0.5) * 100
            )

    support = resistance = range_width_pct = market_regime = None
    if len(prices) >= RANGE_WINDOW:
        window = prices.tail(RANGE_WINDOW)
        support = float(window.min())
        resistance = float(window.max())
        if last_price != 0:
            range_width_pct = (resistance - support) / last_price * 100

        # Kaufman's Efficiency Ratio: net directional move over the window
        # divided by the total path length (sum of absolute day-to-day
        # moves). Close to 1 = strongly directional ("trending"); close to
        # 0 = choppy back-and-forth with little net progress ("sideways").
        net_change = last_price - float(window.iloc[0])
        daily_moves = window.diff().dropna().abs()
        total_path = float(daily_moves.sum())
        if total_path > 0:
            efficiency_ratio = abs(net_change) / total_path
            if efficiency_ratio >= TRENDING_EFFICIENCY_RATIO:
                market_regime = "trending_up" if net_change > 0 else "trending_down"
            elif efficiency_ratio < SIDEWAYS_EFFICIENCY_RATIO:
                market_regime = "sideways"
            else:
                market_regime = "mixed"

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
