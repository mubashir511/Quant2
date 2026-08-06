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


def compute_technical_stats(prices: pd.Series) -> TechnicalStats:
    """Pure computation over a close-price series, oldest first.

    Any stat that needs more history than is actually available (e.g. a
    3-month change on a contract that only just started trading) is left
    as None rather than computed from a shorter, misleading window.
    """
    prices = prices.dropna()
    if prices.empty:
        return TechnicalStats(*([None] * 12))

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
    )
