"""Historical validation of the *kinds* of implicit claims a market
narrative naturally makes (an RSI extreme implies reversal, a stock's
beta implies a structural sensitivity to the index, positive momentum
implies continuation) — checked against that instrument's own real
price history, not a generic textbook assumption.

This deliberately does NOT attempt to backtest specific macro-event
claims (e.g. "a rate cut should push this sector up") — no free, clean,
historical record of past State Bank of Pakistan rate decisions exists
(confirmed live: Trading Economics/CEIC have this but are paid services;
SBP's own site scatters individual decisions across undated-summary PDF
press releases with no table to scrape). What's built here instead is
the same *spirit* — does a claimed relationship actually hold up against
real historical evidence, not just plausible-sounding reasoning — using
data this project can reliably and freely get: an instrument's own full
price history (PSX's EOD feed reliably returns ~5 years).

Every function here is pure (no I/O, no network) and returns None rather
than a fabricated result when there isn't enough real history to trust —
same "never compute a stat from too short a window" rule as
analysis/technical.py.
"""

from dataclasses import dataclass

import pandas as pd

_MIN_BETA_OBSERVATIONS = 60  # ~3 months of trading days — a shorter window is too noisy to trust


def compute_beta(
    stock_prices: pd.Series, index_prices: pd.Series, min_observations: int = _MIN_BETA_OBSERVATIONS
) -> float | None:
    """Beta vs a benchmark index: covariance of daily returns over the
    benchmark's own return variance. None without a real window's worth
    of aligned trading days."""
    if stock_prices.empty or index_prices.empty:
        return None
    aligned = pd.DataFrame({"stock": stock_prices, "index": index_prices}).dropna()
    if len(aligned) < min_observations:
        return None
    stock_returns = aligned["stock"].pct_change().dropna()
    index_returns = aligned["index"].pct_change().dropna()
    index_variance = index_returns.var()
    if not index_variance:
        return None
    return float(stock_returns.cov(index_returns) / index_variance)


def _episode_start_positions(prices: pd.Series, mask: pd.Series) -> list[int]:
    """Positional indices of the FIRST day of each consecutive True
    streak in `mask` — shared by every backtest below that scans for
    "days matching some condition", so a condition that holds for many
    consecutive days (RSI staying overbought, price hugging support)
    counts as one real episode, not one per day. Counting every day
    separately would inflate the sample with highly-correlated,
    non-independent observations rather than genuinely distinct
    historical instances."""
    mask = mask.fillna(False)
    first_of_streak = mask & ~mask.shift(1, fill_value=False)
    return [prices.index.get_loc(idx) for idx in prices.index[first_of_streak]]


def _forward_returns_pct(prices: pd.Series, positions: list[int], forward_days: int) -> list[float]:
    """% return from each given position to `forward_days` later —
    dropping any position too close to the end of the series to have a
    real forward observation, and any position at a zero price (can't
    compute a % change from nothing)."""
    returns = []
    for pos in positions:
        forward_pos = pos + forward_days
        if forward_pos >= len(prices):
            continue
        start_price = prices.iloc[pos]
        if not start_price:
            continue
        returns.append(float((prices.iloc[forward_pos] - start_price) / start_price * 100))
    return returns


@dataclass
class RSIReactionBacktest:
    condition: str  # "overbought" or "oversold"
    threshold: float
    occurrences: int
    avg_forward_return_pct: float
    reversal_rate_pct: float  # % of occurrences where price moved the "expected" reversal direction
    forward_days: int


_RSI_WINDOW = 14


def _rolling_rsi(prices: pd.Series) -> pd.Series:
    """Classic 14-day RSI computed at every point in the series (not
    just the latest, unlike analysis.technical._compute_rsi) — needed
    here to find every historical day the series crossed an extreme
    level, not just today's reading."""
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(_RSI_WINDOW).mean()
    avg_loss = losses.rolling(_RSI_WINDOW).mean()
    # float("nan") rather than pd.NA — keeps the series plain float64
    # instead of upcasting to a nullable/object dtype for the division.
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    # Where avg_loss is exactly 0, the formula above is NaN by
    # construction (division guarded above) — fill with the correct
    # boundary value: 100 if there were real gains in the window, 50 if
    # the window was genuinely flat (both averages zero).
    zero_loss = avg_loss == 0
    return rsi.mask(zero_loss, (avg_gain > 0).astype(float) * 50.0 + 50.0)


def backtest_rsi_reaction(
    prices: pd.Series,
    forward_days: int = 10,
    overbought: float = 70.0,
    oversold: float = 30.0,
    min_occurrences: int = 5,
) -> tuple[RSIReactionBacktest | None, RSIReactionBacktest | None]:
    """For every historical day this instrument's own RSI crossed into
    overbought/oversold territory, checks what its price actually did
    over the following `forward_days` — real evidence for or against
    treating the *current* RSI extreme as a reversal signal, rather than
    assuming the textbook convention applies here.

    Consecutive extreme days are collapsed into one "episode" (only the
    first day of each streak counts) — RSI often stays extreme for
    several days in a row, and counting each of those separately would
    inflate the occurrence count with highly-correlated, non-independent
    samples rather than genuinely distinct historical instances.

    Returns (overbought_result, oversold_result), either None if there
    aren't at least `min_occurrences` real historical episodes to
    average over — a backtest from 1-2 instances isn't trustworthy
    evidence either way."""
    prices = prices.dropna()
    if len(prices) < _RSI_WINDOW + forward_days + 1:
        return None, None

    rsi = _rolling_rsi(prices)

    def _episode_returns(mask: pd.Series) -> list[float] | None:
        positions = _episode_start_positions(prices, mask)
        returns = _forward_returns_pct(prices, positions, forward_days)
        return returns if len(returns) >= min_occurrences else None

    overbought_returns = _episode_returns(rsi >= overbought)
    oversold_returns = _episode_returns(rsi <= oversold)

    overbought_result = None
    if overbought_returns is not None:
        reversal_rate = sum(1 for r in overbought_returns if r < 0) / len(overbought_returns) * 100
        overbought_result = RSIReactionBacktest(
            condition="overbought",
            threshold=overbought,
            occurrences=len(overbought_returns),
            avg_forward_return_pct=float(sum(overbought_returns) / len(overbought_returns)),
            reversal_rate_pct=float(reversal_rate),
            forward_days=forward_days,
        )

    oversold_result = None
    if oversold_returns is not None:
        reversal_rate = sum(1 for r in oversold_returns if r > 0) / len(oversold_returns) * 100
        oversold_result = RSIReactionBacktest(
            condition="oversold",
            threshold=oversold,
            occurrences=len(oversold_returns),
            avg_forward_return_pct=float(sum(oversold_returns) / len(oversold_returns)),
            reversal_rate_pct=float(reversal_rate),
            forward_days=forward_days,
        )

    return overbought_result, oversold_result


@dataclass
class BetaStabilityBacktest:
    beta_3m: float | None
    beta_6m: float | None
    beta_1y: float | None
    beta_full_history: float | None
    stable: bool | None  # None if too few windows computed to judge


_BETA_WINDOW_DAYS = {"beta_3m": 63, "beta_6m": 126, "beta_1y": 252}


def backtest_beta_stability(stock_prices: pd.Series, index_prices: pd.Series) -> BetaStabilityBacktest:
    """Computes beta over several distinct historical windows instead of
    just one — if they disagree substantially, the current single-window
    beta is more likely a coincidence of the recent period than a real
    structural constant, which matters for any claim that leans on beta
    as if it were a stable, "this is just how the stock behaves"
    property."""
    aligned = pd.DataFrame({"stock": stock_prices, "index": index_prices}).dropna()

    windows: dict[str, float | None] = {}
    for key, days in _BETA_WINDOW_DAYS.items():
        tail = aligned.tail(days)
        windows[key] = compute_beta(tail["stock"], tail["index"], min_observations=min(days, 60))
    windows["beta_full_history"] = compute_beta(aligned["stock"], aligned["index"])

    computed = [v for v in windows.values() if v is not None]
    stable = None
    if len(computed) >= 2:
        spread = max(computed) - min(computed)
        # A swing of more than 0.5 beta points across windows means the
        # instrument has behaved meaningfully differently relative to the
        # index at different times — not a stable structural constant.
        stable = spread <= 0.5

    return BetaStabilityBacktest(
        beta_3m=windows["beta_3m"],
        beta_6m=windows["beta_6m"],
        beta_1y=windows["beta_1y"],
        beta_full_history=windows["beta_full_history"],
        stable=stable,
    )


@dataclass
class MomentumPersistenceBacktest:
    correlation: float | None
    sample_size: int
    interpretation: str  # "persistent" / "mean_reverting" / "no_clear_pattern"


_MOMENTUM_WINDOW_DAYS = 21  # ~1 trading month
_MIN_MOMENTUM_SAMPLES = 8


def backtest_momentum_persistence(prices: pd.Series) -> MomentumPersistenceBacktest | None:
    """Across an instrument's own full history, checks whether a strong
    trailing 1-month return has historically been followed by a strong
    (persistent) or weak/reversed (mean-reverting) NEXT 1-month return —
    real evidence for whether "it's currently showing positive momentum"
    is actually a bullish signal for this specific instrument, or
    whether this instrument's own history says the opposite.

    Sampled at non-overlapping, spaced points (not a rolling window
    checked every single day) so the observations are reasonably
    independent of each other rather than nearly-duplicate overlapping
    windows inflating the apparent sample size."""
    prices = prices.dropna()
    step = _MOMENTUM_WINDOW_DAYS
    trailing_returns = []
    forward_returns = []
    # Walk in non-overlapping steps: need a full trailing window before
    # position i, and a full forward window after it.
    for i in range(step, len(prices) - step, step):
        before, at, after = prices.iloc[i - step], prices.iloc[i], prices.iloc[i + step]
        if not before or not at:
            continue
        trailing_returns.append((at - before) / before)
        forward_returns.append((after - at) / at)

    if len(trailing_returns) < _MIN_MOMENTUM_SAMPLES:
        return None

    trailing = pd.Series(trailing_returns)
    forward = pd.Series(forward_returns)
    if trailing.std() == 0 or forward.std() == 0:
        return None
    correlation = float(trailing.corr(forward))

    if correlation >= 0.15:
        interpretation = "persistent"
    elif correlation <= -0.15:
        interpretation = "mean_reverting"
    else:
        interpretation = "no_clear_pattern"

    return MomentumPersistenceBacktest(
        correlation=correlation, sample_size=len(trailing_returns), interpretation=interpretation
    )


@dataclass
class VolatilityRegimeBacktest:
    low_vol_avg_abs_move_pct: float
    low_vol_episodes: int
    high_vol_avg_abs_move_pct: float
    high_vol_episodes: int
    forward_days: int


_VOL_REGIME_WINDOW = 20  # matches analysis.technical.VOLATILITY_WINDOW, for the same rolling-vol convention
_VOL_REGIME_LOW_QUANTILE = 0.25
_VOL_REGIME_HIGH_QUANTILE = 0.75
_MIN_VOL_REGIME_EPISODES = 10


def backtest_volatility_regime(
    prices: pd.Series, forward_days: int = 10, min_episodes: int = _MIN_VOL_REGIME_EPISODES
) -> VolatilityRegimeBacktest | None:
    """Tests a specific, named claim already cited in this project's book
    wisdom (Pring's "coiled spring" — volatility contraction tends to
    precede a bigger move) against this instrument's own real history,
    rather than assuming the principle transfers: compares the average
    ABSOLUTE forward move following the instrument's own historically
    LOW-volatility episodes against its own HIGH-volatility episodes. If
    low-vol periods are actually followed by *smaller* subsequent moves
    (volatility clustering — low vol tends to stay low), that directly
    contradicts the "coiled spring" reading for this instrument; if
    followed by *larger* moves, that supports it.

    "Low"/"high" are this instrument's own historical bottom/top quartile
    of rolling volatility, not an absolute threshold — what counts as low
    volatility for one stock can be ordinary for another."""
    prices = prices.dropna()
    if len(prices) < _VOL_REGIME_WINDOW + forward_days + 1:
        return None

    rolling_vol = prices.pct_change().rolling(_VOL_REGIME_WINDOW).std()
    valid_vol = rolling_vol.dropna()
    if len(valid_vol) < _VOL_REGIME_WINDOW:
        return None

    low_threshold = valid_vol.quantile(_VOL_REGIME_LOW_QUANTILE)
    high_threshold = valid_vol.quantile(_VOL_REGIME_HIGH_QUANTILE)

    low_positions = _episode_start_positions(prices, rolling_vol <= low_threshold)
    high_positions = _episode_start_positions(prices, rolling_vol >= high_threshold)

    low_moves = [abs(r) for r in _forward_returns_pct(prices, low_positions, forward_days)]
    high_moves = [abs(r) for r in _forward_returns_pct(prices, high_positions, forward_days)]

    if len(low_moves) < min_episodes or len(high_moves) < min_episodes:
        return None

    return VolatilityRegimeBacktest(
        low_vol_avg_abs_move_pct=float(sum(low_moves) / len(low_moves)),
        low_vol_episodes=len(low_moves),
        high_vol_avg_abs_move_pct=float(sum(high_moves) / len(high_moves)),
        high_vol_episodes=len(high_moves),
        forward_days=forward_days,
    )


@dataclass
class SupportResistanceBacktest:
    support_tests: int
    support_hold_rate_pct: float  # % of tests where price was higher forward_days later (held, didn't break down)
    resistance_tests: int
    resistance_reject_rate_pct: float  # % of tests where price was lower forward_days later (rejected, didn't break out)
    forward_days: int


_SR_WINDOW = 60  # matches analysis.technical.RANGE_WINDOW, for the same support/resistance convention
_SR_PROXIMITY_PCT = 2.0
_MIN_SR_TESTS = 5


def backtest_support_resistance_reaction(
    prices: pd.Series,
    forward_days: int = 10,
    proximity_pct: float = _SR_PROXIMITY_PCT,
    min_tests: int = _MIN_SR_TESTS,
) -> SupportResistanceBacktest | None:
    """The support/resistance lines shown elsewhere (and used to justify
    entries and stops) are computed from a rolling window — this checks
    how often price has actually held or broken through those SAME
    levels historically, rather than assuming a support/resistance line
    is reliable just because it's a well-known charting concept.

    Support/resistance for day t is computed from the PRECEDING window
    only (`shift(1)` before the rolling min/max) — using a window that
    includes today would make "near the resistance" trivially true
    whenever today happens to be a new high, which isn't a real test of
    whether a PRE-EXISTING level held."""
    prices = prices.dropna()
    if len(prices) < _SR_WINDOW + forward_days + 2:
        return None

    prior = prices.shift(1)
    rolling_support = prior.rolling(_SR_WINDOW).min()
    rolling_resistance = prior.rolling(_SR_WINDOW).max()

    # abs() matters here: without it, a price far BELOW support (already
    # broken down) or far ABOVE resistance (already broken out) would
    # also satisfy "not too far above support" / "not too far below
    # resistance" trivially, since the raw (unsigned) difference can be
    # a large negative number — this must mean genuinely CLOSE to the
    # level, from either side, not merely on the expected side of it.
    near_support = (prices - rolling_support).abs() / rolling_support * 100 <= proximity_pct
    near_resistance = (rolling_resistance - prices).abs() / rolling_resistance * 100 <= proximity_pct

    support_positions = _episode_start_positions(prices, near_support)
    resistance_positions = _episode_start_positions(prices, near_resistance)

    support_returns = _forward_returns_pct(prices, support_positions, forward_days)
    resistance_returns = _forward_returns_pct(prices, resistance_positions, forward_days)

    if len(support_returns) < min_tests or len(resistance_returns) < min_tests:
        return None

    support_hold_rate = sum(1 for r in support_returns if r > 0) / len(support_returns) * 100
    resistance_reject_rate = sum(1 for r in resistance_returns if r < 0) / len(resistance_returns) * 100

    return SupportResistanceBacktest(
        support_tests=len(support_returns),
        support_hold_rate_pct=float(support_hold_rate),
        resistance_tests=len(resistance_returns),
        resistance_reject_rate_pct=float(resistance_reject_rate),
        forward_days=forward_days,
    )
