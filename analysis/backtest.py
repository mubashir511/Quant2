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

from analysis.technical import ATR_WINDOW

_MIN_BETA_OBSERVATIONS = 60  # ~3 months of trading days — a shorter window is too noisy to trust

# Real trade-simulation convention shared by every directional backtest
# below (RSI reversal, support/resistance bounce) — added 2026-08-22
# replacing the old "average return N days later" style, which couldn't
# tell a clean winner from a trade that crashed 8% then drifted back to
# +2% by day 10 (both counted as identical "positive returns"). A stop/
# target this simple can't capture everything a real, instrument-
# specific stop/target debate would (see ai/ftmo_suggest.py's own
# DEBATE THE STOP AND TARGET instruction, which deliberately does NOT
# apply one flat ATR multiple uniformly) — it's a necessary, disclosed
# simplification to get a CONSISTENT, comparable historical rule at all;
# the 1.5x-stop/2:1-reward:risk combination isn't arbitrary, it matches
# this project's own already-cited "2:1 Bulkowski/Rockefeller" reward:
# risk convention and a common ATR-stop multiple, not a new invention.
TRADE_SIM_STOP_ATR_MULTIPLE = 1.5
TRADE_SIM_REWARD_RISK_RATIO = 2.0
TRADE_SIM_TARGET_ATR_MULTIPLE = TRADE_SIM_STOP_ATR_MULTIPLE * TRADE_SIM_REWARD_RISK_RATIO
TRADE_SIM_MAX_HOLDING_BARS = 10


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


def _rolling_atr(ohlc: pd.DataFrame) -> pd.Series | None:
    """Same True Range definition as analysis.technical._compute_atr (max
    of the bar's own high-low range and its gap from the prior close),
    computed as a rolling series across the WHOLE history rather than
    just the latest value — a historical trade simulation has to use the
    ATR that was actually available AT each past entry point, not
    today's current ATR applied retroactively to trades years earlier.

    None if the frame has no High/Low at all — PSX's own EOD feed (see
    data/psx_source.py's own docstring) genuinely can't provide them,
    same "explicitly disclosed as unavailable, never fabricated"
    convention analysis.technical._compute_atr already uses for exactly
    this same gap."""
    if not {"High", "Low", "Close"}.issubset(ohlc.columns):
        return None
    high, low, close = ohlc["High"], ohlc["Low"], ohlc["Close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(ATR_WINDOW).mean()


def _simulate_trades(
    ohlc: pd.DataFrame,
    entry_positions: list[int],
    side: str,
    atr: pd.Series,
    stop_atr_multiple: float,
    target_atr_multiple: float,
    max_holding_bars: int,
    min_stop_distance_pct: float = 0.0,
    round_trip_cost_pct: float = 0.0,
    swap_pct_per_day: float = 0.0,
) -> list[tuple[float, str]]:
    """The real engine behind every directional backtest below — replaces
    the old "average return N bars later" measurement (which couldn't
    tell a clean winner from a trade that crashed 8% then drifted back
    to +2% by the measurement day) with an actual simulated trade: a
    stop `stop_atr_multiple` x that entry's OWN historical ATR away, a
    target `target_atr_multiple` x ATR away, then a real bar-by-bar walk
    forward checking each bar's real High/Low for a touch.

    Real execution realism added 2026-08-22 (direct user request: "review
    the allowed lot size, trade cost, and other execution related
    features and broker's allowed guard rails... such results are more
    realistic and dependable"):
    - `min_stop_distance_pct` (from data/mt5_source.py::TradeCost, MT5's
      own SYMBOL_TRADE_STOPS_LEVEL): if the ATR-based stop would sit
      CLOSER to entry than this broker-enforced floor, both stop and
      target are widened to respect it — proportionally, so the target
      stays the same reward:risk multiple of the (now wider) stop — since
      a stop tighter than the floor could never actually have been
      placed as a real order in the first place.
    - `round_trip_cost_pct`/`swap_pct_per_day` (real spread+commission,
      and per-bar-held swap — see TradeCost) are netted out of every
      trade's realized R, win/loss/timeout alike, since a real trade
      pays them regardless of outcome. This only ever moves the realized
      R-multiple; it never changes whether a trade is classified a win,
      loss, or timeout — that's still decided purely by whether price
      actually touched the (possibly widened) stop/target level.

    Real, disclosed limitation: these are TODAY's live spread/commission/
    swap rate, applied uniformly to every historical trade — there is no
    historical spread/swap time series available to this project (MT5
    doesn't expose one, and FTMO's own commission schedule is a single
    current published rate, not a historized one), so this assumes
    today's %-of-price cost is a reasonable stand-in for what it was at
    each past entry. The ATR-based stop/target itself has no such gap
    (it uses the REAL historical ATR at each entry point, not today's) —
    only the cost/swap side of the adjustment is a current-rate proxy.

    If a single bar's range would have touched BOTH stop and target —
    a genuine, unresolvable ambiguity without real intrabar tick/order
    data — the stop counts as hit first. This is a deliberately
    conservative convention (never overstates the win rate by assuming
    the more favorable outcome), a standard practice in bar-based
    backtesting.

    A trade that touches neither within `max_holding_bars` is NOT
    discarded — it's marked to market at its exit close and returned as
    a "timeout", so the overall picture reflects every trade actually
    taken, not just the ones that neatly resolved one way or the other.

    Returns one (r_multiple, outcome) pair per real simulated trade,
    outcome in {"win", "loss", "timeout"} — kept as an explicit tag
    rather than inferred from the R value's sign, since a timeout
    sitting at a slightly positive mark-to-market R is genuinely
    undecided, not the same claim as a real target hit."""
    high, low, close = ohlc["High"], ohlc["Low"], ohlc["Close"]
    sign = 1.0 if side == "buy" else -1.0
    results: list[tuple[float, str]] = []
    for pos in entry_positions:
        if pos + 1 >= len(ohlc):
            continue
        entry_atr = atr.iloc[pos]
        if pd.isna(entry_atr) or entry_atr <= 0:
            continue
        entry_price = close.iloc[pos]
        if not entry_price:
            continue
        stop_distance = stop_atr_multiple * entry_atr
        # Real broker guard rail: a stop tighter than the symbol's own
        # enforced minimum could never actually have been placed — widen
        # both legs together so the target keeps the same R:R multiple
        # of the (now realistic) stop distance.
        min_stop_distance = min_stop_distance_pct / 100 * entry_price
        if min_stop_distance > stop_distance:
            stop_distance = min_stop_distance
        target_distance = stop_distance * (target_atr_multiple / stop_atr_multiple)
        stop_price = entry_price - sign * stop_distance
        target_price = entry_price + sign * target_distance

        outcome: str | None = None
        outcome_r = 0.0
        mark_to_market_r = 0.0
        bars_held = 0
        for fwd in range(1, max_holding_bars + 1):
            idx = pos + fwd
            if idx >= len(ohlc):
                break
            bars_held = fwd
            bar_high, bar_low, bar_close = high.iloc[idx], low.iloc[idx], close.iloc[idx]
            mark_to_market_r = sign * (bar_close - entry_price) / stop_distance
            stop_hit = (bar_low <= stop_price) if side == "buy" else (bar_high >= stop_price)
            target_hit = (bar_high >= target_price) if side == "buy" else (bar_low <= target_price)
            if stop_hit:
                outcome, outcome_r = "loss", -1.0
                break
            if target_hit:
                outcome, outcome_r = "win", target_atr_multiple / stop_atr_multiple
                break
        if outcome is None:
            outcome, outcome_r = "timeout", mark_to_market_r

        # Real cost, netted into the realized R AFTER the win/loss/
        # timeout verdict — a real trade pays spread/commission
        # regardless of outcome, and swap for every bar actually held
        # (already correctly signed: a negative swap_pct_per_day shrinks
        # the result, a positive one grows it).
        cost_r = (round_trip_cost_pct / 100 * entry_price) / stop_distance
        swap_r = (swap_pct_per_day / 100 * bars_held * entry_price) / stop_distance
        outcome_r = outcome_r - cost_r + swap_r
        results.append((float(outcome_r), outcome))
    return results


@dataclass
class _TradeSimSummary:
    trades: int
    wins: int
    losses: int
    timeouts: int
    win_rate_pct: float | None  # wins / (wins + losses), among definitively RESOLVED trades only; None if none resolved
    avg_r_multiple: float  # mean realized R across every trade taken, timeouts included at their mark-to-market value


def _summarize_trade_results(results: list[tuple[float, str]]) -> _TradeSimSummary:
    wins = sum(1 for _, outcome in results if outcome == "win")
    losses = sum(1 for _, outcome in results if outcome == "loss")
    timeouts = sum(1 for _, outcome in results if outcome == "timeout")
    resolved = wins + losses
    return _TradeSimSummary(
        trades=len(results),
        wins=wins,
        losses=losses,
        timeouts=timeouts,
        win_rate_pct=(wins / resolved * 100) if resolved > 0 else None,
        avg_r_multiple=float(sum(r for r, _ in results) / len(results)) if results else 0.0,
    )


@dataclass
class RSIReactionBacktest:
    """Real trade-simulation result, replacing the old avg-forward-return
    style (see this module's own top-of-file note and _simulate_trades'
    own docstring for why): "overbought" bets SHORT (a reversal down is
    the textbook claim being tested), "oversold" bets LONG. win_rate_pct
    is None (not 0.0) when zero trades ever definitively resolved —
    genuinely different from "0% winners". `min_stop_distance_pct`/
    `round_trip_cost_pct`/`swap_pct_per_day_used` (added 2026-08-22)
    disclose exactly what real broker constraints/costs were baked into
    avg_r_multiple — all default to 0.0 when a caller has no real
    execution data for this symbol (e.g. PSX, no live broker feed at
    all), which is honestly "not simulated with real costs", not
    "simulated with zero real-world friction and that's fine"."""
    condition: str  # "overbought" or "oversold"
    threshold: float
    trades: int
    wins: int
    losses: int
    timeouts: int
    win_rate_pct: float | None
    avg_r_multiple: float
    stop_atr_multiple: float
    target_atr_multiple: float
    max_holding_bars: int
    min_stop_distance_pct: float = 0.0
    round_trip_cost_pct: float = 0.0
    swap_pct_per_day_used: float = 0.0


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
    ohlc: pd.DataFrame,
    max_holding_bars: int = TRADE_SIM_MAX_HOLDING_BARS,
    overbought: float = 70.0,
    oversold: float = 30.0,
    min_occurrences: int = 5,
    stop_atr_multiple: float = TRADE_SIM_STOP_ATR_MULTIPLE,
    target_atr_multiple: float = TRADE_SIM_TARGET_ATR_MULTIPLE,
    min_stop_distance_pct: float = 0.0,
    round_trip_cost_pct: float = 0.0,
    long_swap_pct_per_day: float = 0.0,
    short_swap_pct_per_day: float = 0.0,
) -> tuple[RSIReactionBacktest | None, RSIReactionBacktest | None]:
    """For every historical bar this instrument's own RSI crossed into
    overbought/oversold territory, simulates the REAL trade the textbook
    reversal claim implies (short on overbought, long on oversold) with
    an ATR-based stop/target (see this module's own top-of-file note),
    walked forward bar-by-bar for a genuine win/loss verdict — see
    _simulate_trades' own docstring for exactly how a win/loss/timeout is
    decided. Replaces the old "average return `forward_days` later"
    measurement, which couldn't distinguish a clean winner from a trade
    that dropped hard and only recovered to positive by the measurement
    day.

    `min_stop_distance_pct`/`round_trip_cost_pct`/`long_swap_pct_per_day`/
    `short_swap_pct_per_day` (added 2026-08-22, all default to 0.0 — no
    broker constraint, no cost) let a caller with real live broker data
    (see data/mt5_source.py::TradeCost) make the simulation respect that
    broker's own minimum stop distance and net real spread/commission/
    swap out of every trade — see _simulate_trades' own docstring. The
    RIGHT swap rate for each side is picked automatically: overbought
    bets short (uses `short_swap_pct_per_day`), oversold bets long (uses
    `long_swap_pct_per_day`).

    Consecutive extreme bars are collapsed into one "episode" (only the
    first bar of each streak counts) — RSI often stays extreme for
    several bars in a row, and counting each of those separately would
    inflate the occurrence count with highly-correlated, non-independent
    samples rather than genuinely distinct historical instances.

    Returns (overbought_result, oversold_result), either None if there
    aren't at least `min_occurrences` real historical episodes, or if
    `ohlc` has no High/Low at all (ATR needs it — see _rolling_atr) — a
    backtest from 1-2 instances, or with no real stop/target basis at
    all, isn't trustworthy evidence either way."""
    ohlc = ohlc.dropna(subset=[c for c in ("High", "Low", "Close") if c in ohlc.columns])
    prices = ohlc["Close"] if "Close" in ohlc.columns else pd.Series(dtype=float)
    if len(prices) < _RSI_WINDOW + max_holding_bars + 1:
        return None, None

    atr = _rolling_atr(ohlc)
    if atr is None:
        return None, None

    rsi = _rolling_rsi(prices)

    def _episode_result(mask: pd.Series, side: str, swap_pct_per_day: float) -> RSIReactionBacktest | None:
        positions = _episode_start_positions(prices, mask)
        if len(positions) < min_occurrences:
            return None
        trade_results = _simulate_trades(
            ohlc, positions, side, atr, stop_atr_multiple, target_atr_multiple, max_holding_bars,
            min_stop_distance_pct=min_stop_distance_pct, round_trip_cost_pct=round_trip_cost_pct,
            swap_pct_per_day=swap_pct_per_day,
        )
        if len(trade_results) < min_occurrences:
            return None
        summary = _summarize_trade_results(trade_results)
        return RSIReactionBacktest(
            condition="overbought" if side == "sell" else "oversold",
            threshold=overbought if side == "sell" else oversold,
            trades=summary.trades,
            wins=summary.wins,
            losses=summary.losses,
            timeouts=summary.timeouts,
            win_rate_pct=summary.win_rate_pct,
            avg_r_multiple=summary.avg_r_multiple,
            stop_atr_multiple=stop_atr_multiple,
            target_atr_multiple=target_atr_multiple,
            max_holding_bars=max_holding_bars,
            min_stop_distance_pct=min_stop_distance_pct,
            round_trip_cost_pct=round_trip_cost_pct,
            swap_pct_per_day_used=swap_pct_per_day,
        )

    overbought_result = _episode_result(rsi >= overbought, side="sell", swap_pct_per_day=short_swap_pct_per_day)
    oversold_result = _episode_result(rsi <= oversold, side="buy", swap_pct_per_day=long_swap_pct_per_day)
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
    """Real trade-simulation result (see this module's own top-of-file
    note and _simulate_trades' own docstring), replacing the old "% of
    tests where price was higher/lower forward_days later" measurement:
    a support test bets LONG (the level holds), a resistance test bets
    SHORT (the level rejects). Each side's win_rate_pct is None (not
    0.0) when zero of that side's trades ever definitively resolved.
    `min_stop_distance_pct`/`round_trip_cost_pct`/swap fields (added
    2026-08-22) disclose exactly what real broker constraints/costs were
    baked into each side's avg_r_multiple — see RSIReactionBacktest's
    own equivalent fields for the same reasoning."""
    support_tests: int
    support_wins: int
    support_losses: int
    support_timeouts: int
    support_win_rate_pct: float | None
    support_avg_r_multiple: float
    resistance_tests: int
    resistance_wins: int
    resistance_losses: int
    resistance_timeouts: int
    resistance_win_rate_pct: float | None
    resistance_avg_r_multiple: float
    stop_atr_multiple: float
    target_atr_multiple: float
    max_holding_bars: int
    min_stop_distance_pct: float = 0.0
    round_trip_cost_pct: float = 0.0
    support_swap_pct_per_day_used: float = 0.0
    resistance_swap_pct_per_day_used: float = 0.0


_SR_WINDOW = 60  # matches analysis.technical.RANGE_WINDOW, for the same support/resistance convention
_SR_PROXIMITY_PCT = 2.0
_MIN_SR_TESTS = 5


def backtest_support_resistance_reaction(
    ohlc: pd.DataFrame,
    max_holding_bars: int = TRADE_SIM_MAX_HOLDING_BARS,
    proximity_pct: float = _SR_PROXIMITY_PCT,
    min_tests: int = _MIN_SR_TESTS,
    stop_atr_multiple: float = TRADE_SIM_STOP_ATR_MULTIPLE,
    target_atr_multiple: float = TRADE_SIM_TARGET_ATR_MULTIPLE,
    min_stop_distance_pct: float = 0.0,
    round_trip_cost_pct: float = 0.0,
    long_swap_pct_per_day: float = 0.0,
    short_swap_pct_per_day: float = 0.0,
) -> SupportResistanceBacktest | None:
    """The support/resistance lines shown elsewhere (and used to justify
    entries and stops) are computed from a rolling window — this
    simulates the REAL trade each level's own convention implies (long
    off support, short off resistance) with a genuine ATR-based stop/
    target, walked forward for a real win/loss verdict (see
    _simulate_trades), rather than the old "% of tests where price was
    merely higher/lower N bars later" measurement, which couldn't tell
    a level that held cleanly from one that broke down hard and only
    recovered above the test price by the measurement bar.

    `min_stop_distance_pct`/`round_trip_cost_pct`/`long_swap_pct_per_day`/
    `short_swap_pct_per_day` (added 2026-08-22, all default to 0.0) —
    same real broker-execution realism as backtest_rsi_reaction's own
    equivalent parameters: the support (long) leg uses
    `long_swap_pct_per_day`, the resistance (short) leg uses
    `short_swap_pct_per_day`.

    Support/resistance for bar t is computed from the PRECEDING window
    only (`shift(1)` before the rolling min/max) — using a window that
    includes today would make "near the resistance" trivially true
    whenever today happens to be a new high, which isn't a real test of
    whether a PRE-EXISTING level held.

    None if there aren't at least `min_tests` real historical tests on
    EACH side, or if `ohlc` has no High/Low at all (ATR needs it)."""
    ohlc = ohlc.dropna(subset=[c for c in ("High", "Low", "Close") if c in ohlc.columns])
    prices = ohlc["Close"] if "Close" in ohlc.columns else pd.Series(dtype=float)
    if len(prices) < _SR_WINDOW + max_holding_bars + 2:
        return None

    atr = _rolling_atr(ohlc)
    if atr is None:
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

    if len(support_positions) < min_tests or len(resistance_positions) < min_tests:
        return None

    support_results = _simulate_trades(
        ohlc, support_positions, "buy", atr, stop_atr_multiple, target_atr_multiple, max_holding_bars,
        min_stop_distance_pct=min_stop_distance_pct, round_trip_cost_pct=round_trip_cost_pct,
        swap_pct_per_day=long_swap_pct_per_day,
    )
    resistance_results = _simulate_trades(
        ohlc, resistance_positions, "sell", atr, stop_atr_multiple, target_atr_multiple, max_holding_bars,
        min_stop_distance_pct=min_stop_distance_pct, round_trip_cost_pct=round_trip_cost_pct,
        swap_pct_per_day=short_swap_pct_per_day,
    )
    if len(support_results) < min_tests or len(resistance_results) < min_tests:
        return None

    support_summary = _summarize_trade_results(support_results)
    resistance_summary = _summarize_trade_results(resistance_results)

    return SupportResistanceBacktest(
        support_tests=support_summary.trades,
        support_wins=support_summary.wins,
        support_losses=support_summary.losses,
        support_timeouts=support_summary.timeouts,
        support_win_rate_pct=support_summary.win_rate_pct,
        support_avg_r_multiple=support_summary.avg_r_multiple,
        resistance_tests=resistance_summary.trades,
        resistance_wins=resistance_summary.wins,
        resistance_losses=resistance_summary.losses,
        resistance_timeouts=resistance_summary.timeouts,
        resistance_win_rate_pct=resistance_summary.win_rate_pct,
        resistance_avg_r_multiple=resistance_summary.avg_r_multiple,
        stop_atr_multiple=stop_atr_multiple,
        target_atr_multiple=target_atr_multiple,
        max_holding_bars=max_holding_bars,
        min_stop_distance_pct=min_stop_distance_pct,
        round_trip_cost_pct=round_trip_cost_pct,
        support_swap_pct_per_day_used=long_swap_pct_per_day,
        resistance_swap_pct_per_day_used=short_swap_pct_per_day,
    )
