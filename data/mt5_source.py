import csv
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

import config

logger = logging.getLogger(__name__)

# A symbol visible via symbols_get() can still fail to select on the
# first attempt — confirmed live (MT5 terminal's own Symbols window
# showing complete real spec data for SP500-SE26 while our own API calls
# couldn't read it) that this is a real, if usually brief, propagation
# delay in the API session's own symbol cache, not a permanent failure.
# A short bounded retry trades a fraction of a second for a real chance
# of catching a symbol that would otherwise be wrongly written off —
# same trade-off already made for OpenRouter audit retries elsewhere in
# this project, just at a much shorter timescale appropriate to a local
# API call instead of a network request.
_SYMBOL_SELECT_RETRY_ATTEMPTS = 3
_SYMBOL_SELECT_RETRY_DELAY_SECONDS = 0.3


def _ensure_symbol_selected(mt5, symbol: str) -> bool:
    """Selects a symbol for this API session, retrying briefly on
    failure. Never raises. Returns whether it ultimately succeeded —
    callers should still attempt their real read either way, since a
    select failure doesn't guarantee the read will fail too, and a
    select success doesn't guarantee it will succeed either; this is a
    best-effort nudge, not a hard precondition."""
    for attempt in range(_SYMBOL_SELECT_RETRY_ATTEMPTS):
        if mt5.symbol_select(symbol, True):
            return True
        if attempt < _SYMBOL_SELECT_RETRY_ATTEMPTS - 1:
            time.sleep(_SYMBOL_SELECT_RETRY_DELAY_SECONDS)

    logger.warning(
        "mt5.symbol_select(%s, True) failed after %d attempts (last_error=%s)",
        symbol, _SYMBOL_SELECT_RETRY_ATTEMPTS, mt5.last_error(),
    )
    return False


@dataclass
class Position:
    symbol: str
    volume: float
    side: str  # "buy" or "sell"
    price_open: float
    price_current: float
    sl: float | None
    profit: float
    opened_at: datetime
    ticket: int = 0  # MT5 position ticket — needed to close/reduce this specific position
    tp: float | None = None

    @property
    def adverse_move_pct(self) -> float:
        """Positive when price has moved against the position."""
        if self.price_open == 0:
            return 0.0
        if self.side == "buy":
            return (self.price_open - self.price_current) / self.price_open * 100
        return (self.price_current - self.price_open) / self.price_open * 100

    @property
    def notional(self) -> float:
        return self.volume * self.price_current


@dataclass
class PendingOrder:
    symbol: str
    volume: float
    order_type: str  # e.g. "buy limit", "sell stop"
    price_open: float  # the trigger price (limit/stop level)
    sl: float | None
    tp: float | None
    ticket: int = 0
    time_setup: datetime | None = None  # when this order was placed — lets callers show/reason about its age


@dataclass
class AccountSummary:
    balance: float
    equity: float
    free_margin: float
    currency: str


@dataclass
class MarketAsset:
    symbol: str
    description: str
    bid: float
    ask: float


@dataclass
class ContractSpec:
    volume_min: float
    volume_step: float
    volume_max: float
    trade_contract_size: float
    currency_margin: str
    margin_initial: float  # margin required for 1.0 lot, in currency_margin


class MT5ConnectionError(RuntimeError):
    pass


def connect(
    login: str | int | None = None,
    password: str | None = None,
    server: str | None = None,
    path: str | None = None,
) -> None:
    """Connects to an MT5 terminal/account. Each param overrides the
    matching `config.MT5_*` global when given, and falls back to it
    otherwise — every existing call site (which calls `connect()` with
    no args, for the original PMEX account) keeps working unchanged.

    The MetaTrader5 Python package supports exactly one active terminal
    connection per process (confirmed against official MQL5 docs before
    this was written) — there is no way to hold two accounts connected
    simultaneously. Switching accounts within one process means this
    function gets called again with different credentials, which
    `mt5.initialize()` documents as supported but real-world reports
    show can silently leave stale state from the previous connection
    behind. To catch that instead of trusting it blindly, this
    explicitly re-reads `mt5.account_info()` after a successful
    `initialize()` and verifies it actually matches the account that
    was just requested — a mismatch raises the same MT5ConnectionError
    a real connection failure would, rather than silently proceeding
    against the wrong account."""
    import MetaTrader5 as mt5

    path = path if path is not None else config.MT5_PATH
    login = login if login is not None else config.MT5_LOGIN
    password = password if password is not None else config.MT5_PASSWORD
    server = server if server is not None else config.MT5_SERVER

    kwargs = {}
    if path:
        kwargs["path"] = path
    if login:
        kwargs["login"] = int(login)
    if password:
        kwargs["password"] = password
    if server:
        kwargs["server"] = server

    if not mt5.initialize(**kwargs):
        error = mt5.last_error()
        raise MT5ConnectionError(
            f"Could not connect to the MT5 terminal ({error}). "
            "Make sure the terminal is running and logged in."
        )

    if login:
        info = mt5.account_info()
        if info is None:
            raise MT5ConnectionError(
                f"Connected to the MT5 terminal, but could not verify it's the "
                f"requested account (login {login}) — account_info() returned "
                f"nothing right after a successful initialize()."
            )
        if info.login != int(login):
            raise MT5ConnectionError(
                f"MT5 terminal is connected to a DIFFERENT account than "
                f"requested: asked for login {login}, but the terminal is "
                f"showing login {info.login} ({info.server}). This looks like "
                f"stale state left over from a previous connection — try "
                f"again, or check the terminal itself."
            )
        if server and server.lower() != info.server.lower():
            raise MT5ConnectionError(
                f"MT5 terminal is connected to login {login}, but on server "
                f"{info.server!r} instead of the requested {server!r} — "
                f"refusing to proceed against a mismatched account."
            )


def is_trading_permitted() -> tuple[bool, str]:
    """Whether the CURRENTLY connected terminal/account can actually
    place a real order right now — distinct from whether the ACCOUNT is
    configured for algo trading at all (account_info().trade_expert can
    be True while this is False). Confirmed live: this project's own
    real FTMO account had terminal_info().trade_allowed=False (the
    "AutoTrading" toggle in the MT5 terminal's own toolbar was off) —
    every order_send() call would have silently failed with
    TRADE_RETCODE_CLIENT_DISABLES_AT (10027) regardless of how correct
    the order itself was, which is exactly the kind of failure that
    looks like "Apply Suggestion doesn't work" without a clear enough
    reason to explain why. Returns (permitted, reason) so a caller can
    show something actionable instead of waiting to decode a bare MT5
    retcode after the fact — check this BEFORE offering "Confirm and
    Execute", not just after a rejection."""
    import MetaTrader5 as mt5

    terminal = mt5.terminal_info()
    account = mt5.account_info()
    if terminal is None or account is None:
        return False, "Could not read the terminal/account trading-permission status."
    if not terminal.trade_allowed:
        return False, (
            "AutoTrading is turned OFF in the MT5 terminal itself (the "
            "AutoTrading button in the terminal's own toolbar) — real "
            "orders cannot be sent until it's enabled there."
        )
    if not account.trade_allowed:
        return False, (
            "This account is not currently permitted to trade (check "
            "Tools > Options > Expert Advisors in the terminal, or "
            "contact your broker if this is unexpected)."
        )
    return True, ""


def get_open_positions() -> list[Position]:
    import MetaTrader5 as mt5

    raw = mt5.positions_get()
    if raw is None:
        error = mt5.last_error()
        raise MT5ConnectionError(f"Failed to fetch open positions ({error}).")

    positions = []
    for p in raw:
        positions.append(
            Position(
                symbol=p.symbol,
                volume=p.volume,
                side="buy" if p.type == mt5.POSITION_TYPE_BUY else "sell",
                price_open=p.price_open,
                price_current=p.price_current,
                sl=p.sl if p.sl else None,
                profit=p.profit,
                opened_at=datetime.fromtimestamp(p.time),
                ticket=p.ticket,
                tp=p.tp if p.tp else None,
            )
        )
    return positions


_ORDER_TYPE_LABELS = {
    2: "buy limit",
    3: "sell limit",
    4: "buy stop",
    5: "sell stop",
    6: "buy stop limit",
    7: "sell stop limit",
}


def get_pending_orders() -> list[PendingOrder]:
    """Working orders not yet filled (limit/stop orders) — distinct from
    get_open_positions(), which only returns already-filled positions.
    MT5 keeps these as two separate concepts; a symbol can have a real,
    live pending order sitting on the account with zero effect on
    positions_get() until price actually reaches it."""
    import MetaTrader5 as mt5

    raw = mt5.orders_get()
    if raw is None:
        error = mt5.last_error()
        raise MT5ConnectionError(f"Failed to fetch pending orders ({error}).")

    orders = []
    for o in raw:
        orders.append(
            PendingOrder(
                symbol=o.symbol,
                volume=o.volume_current,
                order_type=_ORDER_TYPE_LABELS.get(o.type, f"type {o.type}"),
                price_open=o.price_open,
                sl=o.sl if o.sl else None,
                tp=o.tp if o.tp else None,
                ticket=o.ticket,
                time_setup=datetime.fromtimestamp(o.time_setup) if o.time_setup else None,
            )
        )
    return orders


def get_current_price(symbol: str) -> float | None:
    """Live mid price for a single symbol — used to show a pending
    order's current market price alongside its trigger price, without
    the cost of fetching the entire Market Watch for just one symbol.
    Calls symbol_select() first, same reasoning as every other
    symbol-keyed read in this file (see the update #18 lesson in this
    project's own history: a symbol can show visible=True yet still fail
    symbol_info_tick() until explicitly selected for this API session).
    Returns None on any failure rather than raising — this is best-effort
    display data, not core account/position data."""
    import MetaTrader5 as mt5

    _ensure_symbol_selected(mt5, symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or (tick.bid == 0 and tick.ask == 0):
        return None
    return (tick.bid + tick.ask) / 2


def get_current_bid_ask(symbol: str) -> tuple[float, float] | None:
    """Live (bid, ask) for a single symbol — get_current_price's sibling
    for a caller that needs the two sides separately (e.g. a spread
    figure), same "one cheap tick fetch, not the whole Market Watch"
    reasoning. Built for the watchlist's per-asset detail popup, which
    was re-fetching all ~17-21 Market Watch symbols on every auto-
    refresh tick just to find the one it needed — confirmed live as a
    real, meaningful contributor to that popup's reported update lag."""
    import MetaTrader5 as mt5

    _ensure_symbol_selected(mt5, symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or (tick.bid == 0 and tick.ask == 0):
        return None
    return tick.bid, tick.ask


def get_server_time_offset(symbol: str = "EURUSD") -> timedelta | None:
    """How far ahead of (or behind) this machine's true UTC clock the
    connected MT5 broker's own server clock currently is — computed
    fresh from a live tick's own timestamp every call, never hardcoded,
    since this is confirmed live to run several hours off local time and
    to shift with the broker's own DST rules, independent of this
    machine's (see get_history_deals' own docstring for the original
    live-verified finding). `time.time()` (this machine's epoch seconds,
    always UTC regardless of local timezone/DST display settings) is
    subtracted from the tick's own epoch seconds to get the real offset.

    Purely informational/display — nothing in this project schedules
    against this value, since a broker's server-time DST transition can
    itself be a source of drift; scheduling stays anchored to plain UTC
    (see config.MEGA_ANALYSIS_TRIGGER_HOUR_UTC). Returns None if `symbol`
    has no live tick on this account (e.g. not present in Market Watch)
    rather than raising — callers should treat that as "broker time
    unknown," not a hard failure."""
    import MetaTrader5 as mt5

    _ensure_symbol_selected(mt5, symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or tick.time <= 0:
        return None
    return timedelta(seconds=tick.time - time.time())


def get_account_summary() -> AccountSummary:
    import MetaTrader5 as mt5

    info = mt5.account_info()
    if info is None:
        error = mt5.last_error()
        raise MT5ConnectionError(f"Failed to fetch account info ({error}).")

    return AccountSummary(
        balance=info.balance,
        equity=info.equity,
        free_margin=info.margin_free,
        currency=info.currency,
    )


def get_market_watch() -> list[MarketAsset]:
    """Tradable instruments the user has added to their MT5 Market Watch.

    Calls symbol_select() before reading each tick, same reasoning as
    get_contract_spec: a symbol showing as "visible" via symbols_get()
    doesn't guarantee it's "selected" for THIS API session's own tick
    cache, which is what symbol_info_tick() actually depends on — this
    matters more here than in get_contract_spec, since a dropped symbol
    here means the AI never sees it at all, not just a missing feasibility
    line for something it does see. Every visible symbol that still gets
    dropped (no tick, or a zero bid/ask) is now logged — previously a
    silent `continue`, indistinguishable from "not visible" or "correctly
    excluded," with no way to tell how many symbols this was actually
    happening to."""
    import MetaTrader5 as mt5

    symbols = mt5.symbols_get()
    if symbols is None:
        error = mt5.last_error()
        raise MT5ConnectionError(f"Failed to fetch symbols ({error}).")

    assets = []
    dropped = []
    for s in symbols:
        if not s.visible:
            continue
        _ensure_symbol_selected(mt5, s.name)
        tick = mt5.symbol_info_tick(s.name)
        if tick is None or (tick.bid == 0 and tick.ask == 0):
            dropped.append(s.name)
            continue  # no live quote right now
        assets.append(
            MarketAsset(symbol=s.name, description=s.description, bid=tick.bid, ask=tick.ask)
        )

    if dropped:
        logger.warning(
            "get_market_watch: %d visible symbol(s) dropped for missing/zero tick: %s",
            len(dropped), ", ".join(dropped),
        )
    return assets


def _load_symbol_spec_from_csv(symbol: str, csv_path: str) -> ContractSpec | None:
    """Fallback contract-spec source: a CSV dumped by mql5/DumpSymbolSpecs.mq5,
    a script that runs *inside* the MT5 terminal and reads its own internal
    symbol database directly — so it isn't subject to the external Python
    API session's symbol-selection quirk at all. Only ever as fresh as
    whenever that script was last run, so it's tried only after the live
    API path has already come back empty, never in place of it."""
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("symbol") != symbol:
                    continue
                return ContractSpec(
                    volume_min=float(row["volume_min"]),
                    volume_step=float(row["volume_step"]),
                    volume_max=float(row["volume_max"]),
                    trade_contract_size=float(row["trade_contract_size"]),
                    currency_margin=row["currency_margin"],
                    margin_initial=float(row["margin_initial"]),
                )
    except (OSError, ValueError, KeyError) as e:
        logger.warning(
            "Failed to read symbol spec CSV fallback (%s) for %s: %s: %s",
            csv_path, symbol, type(e).__name__, e,
        )
    return None


def _compute_margin_via_order_calc(symbol: str) -> float:
    """Real per-1.0-lot margin via mt5.order_calc_margin() — MT5's own
    calc-mode-agnostic margin calculation (works for FOREX/CFD calc
    modes, unlike the static symbol_info().margin_initial field). Needs
    a live ask price, so re-reads the tick rather than assuming the
    caller already has a fresh one. 0.0 (not None — matches
    ContractSpec.margin_initial's own type, and compute_rebalance_plan's
    existing margin_initial<=0 check treats it as "no usable spec"
    either way) if the tick or the calc call itself comes back empty."""
    import MetaTrader5 as mt5

    tick = mt5.symbol_info_tick(symbol)
    if tick is None or tick.ask <= 0:
        logger.warning(
            "_compute_margin_via_order_calc(%s): no usable ask price to calc margin from",
            symbol,
        )
        return 0.0

    margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, 1.0, tick.ask)
    if margin is None:
        error = mt5.last_error()
        logger.warning(
            "_compute_margin_via_order_calc(%s): order_calc_margin returned None (last_error=%s)",
            symbol, error,
        )
        return 0.0
    return margin


def get_contract_spec(symbol: str) -> ContractSpec | None:
    """Real order-size/margin constraints for a symbol — e.g. a minimum
    lot can require far more margin than a small account holds at all,
    not just "a lot" of it. None on any failure (also the expected result
    under USE_MOCK_DATA, where there's no live MT5 connection) — but logs
    the actual reason first (confirmed live: this silently returned None
    for SP500-SE26 across every single Portfolio Suggestion run for a
    full day, with no way to tell "symbol truly has no margin data on
    this broker" apart from "transient failure" apart from "a bug" — same
    diagnosability gap ai/openrouter_client.py's run_openrouter() had
    before its own error-visibility fix).

    Calls symbol_select() first — confirmed via the MT5 terminal's own
    Symbols window that SP500-SE26 has complete, real spec data on the
    broker side (Initial margin 140500 PKR, etc.), so symbol_info()
    returning None for it wasn't a broker-side data gap. This is a known
    MT5 Python API quirk: a symbol can show as "visible" via symbols_get()
    (reflecting the terminal's Market Watch display) while still not
    being "selected" for THIS API session's own internal cache, which is
    what symbol_info() actually depends on — the two are tracked
    separately. Explicitly selecting it first is the documented fix.

    If the live API still has nothing after that, falls back to
    config.MT5_SYMBOL_SPECS_CSV_PATH when configured (see
    _load_symbol_spec_from_csv) — a genuinely independent second data
    path, not just another retry of the same one.

    `margin_initial` specifically also falls back to mt5.order_calc_margin()
    when symbol_info() reports it as 0 — confirmed live on a real FTMO
    account that FOREX-calc-mode symbols (SYMBOL_CALC_MODE_FOREX, e.g.
    EURUSD) always report margin_initial=0 via symbol_info(); that field
    only ever carries a real number for futures-style calc modes like
    PMEX's own contracts. risk/apply_suggestion.py's compute_rebalance_plan
    treats margin_initial<=0 as "no spec available" and marks the symbol
    infeasible — without this fallback, EVERY forex/CFD-calc-mode symbol
    (i.e. most of FTMO's own tradable universe) would silently be
    unactionable via Apply Suggestion. order_calc_margin() is MT5's own
    calc-mode-agnostic margin calculation, the same one the terminal
    itself uses — confirmed live to return the correct number (matched
    contract_size * price / leverage by hand for FTMO's EURUSD)."""
    spec = None
    try:
        import MetaTrader5 as mt5

        _ensure_symbol_selected(mt5, symbol)
        info = mt5.symbol_info(symbol)
    except Exception as e:
        logger.warning(
            "get_contract_spec raised for %s: %s: %s", symbol, type(e).__name__, e
        )
        info = None
    else:
        if info is None:
            error = mt5.last_error()
            logger.warning("mt5.symbol_info(%s) returned None (last_error=%s)", symbol, error)

    if info is not None:
        try:
            margin_initial = info.margin_initial
            if margin_initial <= 0:
                margin_initial = _compute_margin_via_order_calc(symbol)
            spec = ContractSpec(
                volume_min=info.volume_min,
                volume_step=info.volume_step,
                volume_max=info.volume_max,
                trade_contract_size=info.trade_contract_size,
                currency_margin=info.currency_margin,
                margin_initial=margin_initial,
            )
        except AttributeError as e:
            logger.warning("mt5.symbol_info(%s) returned an incomplete object: %s", symbol, e)

    if spec is None and config.MT5_SYMBOL_SPECS_CSV_PATH:
        spec = _load_symbol_spec_from_csv(symbol, config.MT5_SYMBOL_SPECS_CSV_PATH)
        if spec is not None:
            logger.info("get_contract_spec: used CSV fallback for %s (live API had nothing)", symbol)

    return spec


@dataclass
class HistoricalDeal:
    ticket: int
    time: datetime
    symbol: str
    profit: float  # total realized effect on balance: profit + swap + commission
    volume: float
    position_id: int = 0  # groups a position's open/close legs into one round-trip trade
    price: float = 0.0
    side: str = ""  # "buy"/"sell" for a real trade leg, "" for a non-trade balance deal


def get_history_deals(date_from: datetime, date_to: datetime | None = None) -> list[HistoricalDeal]:
    """Real closed-trade history from the currently-connected account —
    the raw material risk/ftmo_rules.py needs to reconstruct real daily
    P&L and the account's own historical equity curve, since MT5 exposes
    no direct "history of daily EOD balances" query. `profit` sums
    MT5's own `profit`/`swap`/`commission` deal fields, since all three
    genuinely affect the account's real balance, not just the headline
    trade P&L. Deliberately includes every deal (including zero-profit
    position-opening entries), not just closes — callers decide their
    own aggregation rather than this function making that choice.

    Empty list on any failure, same "don't fabricate, don't raise for a
    normal empty result" contract as every other read-only fetch in this
    file — a genuinely fresh account with zero trade history is a real,
    expected case (e.g. a brand-new FTMO Challenge before its first
    trade), not an error.

    Deal timestamps come from MT5.history_deals_get() in the broker's own
    SERVER clock, not this machine's local clock — confirmed live against
    the real FTMO account that the server clock can run meaningfully
    ahead of local time (~2 hours observed). A default `date_to=datetime.
    now()` (local) silently excluded deals that had already happened in
    server time but whose timestamp still looked "in the future" next to
    the too-early local bound — a real trade's closing deal went missing
    from every FTMO compliance computation until this was found. Default
    now pushes `date_to` a full day into local-future to comfortably
    absorb any such clock skew; `date_from` already uses the same
    "far enough to not matter" approach at its own call site."""
    import MetaTrader5 as mt5

    date_to = date_to if date_to is not None else datetime.now() + timedelta(days=1)
    raw = mt5.history_deals_get(date_from, date_to)
    if raw is None:
        error = mt5.last_error()
        logger.warning("get_history_deals: history_deals_get returned nothing (last_error=%s)", error)
        return []

    return [
        HistoricalDeal(
            ticket=d.ticket,
            time=datetime.fromtimestamp(d.time),
            symbol=d.symbol,
            profit=d.profit + d.swap + d.commission,
            volume=d.volume,
            position_id=d.position_id,
            price=d.price,
            side=(
                ("buy" if d.type == mt5.DEAL_TYPE_BUY else "sell")
                if d.symbol and d.type in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL)
                else ""
            ),
        )
        for d in raw
    ]


@dataclass
class ClosedTrade:
    position_id: int
    symbol: str
    side: str  # "buy"/"sell", from the position's own opening leg
    volume: float
    opened_at: datetime
    closed_at: datetime
    open_price: float
    close_price: float
    profit: float  # net realized P&L across every leg of this position

    @property
    def duration(self) -> timedelta:
        return self.closed_at - self.opened_at


def group_closed_trades(deals: list[HistoricalDeal]) -> list[ClosedTrade]:
    """Groups a flat deal list (open + close legs, commission-only rows,
    and non-trade balance operations all mixed together, exactly what
    get_history_deals() returns) into one row per position that has
    actually finished — real win/loss trade history, not raw deal rows.

    Sums every leg's already-fully-loaded profit (profit+swap+commission)
    onto its position_id — confirmed live this is the only way to get
    the true net result: a real closed XAUUSD trade's balance impact
    only matched exactly when both the opening leg's commission and the
    closing leg's profit+swap+commission were summed together, not
    either alone. A position with only one deal on record is still
    open (or, rarely, a non-trade balance adjustment) and is excluded —
    this list is specifically "trades that finished." side/open_price
    come from the earliest deal (the position's own opening leg) and
    close_price from the latest (its closing leg) — a position can have
    more than 2 deals (partial closes), so this always takes the FIRST
    and LAST by time rather than assuming exactly two. Newest first."""
    by_position: dict[int, list[HistoricalDeal]] = {}
    for d in deals:
        if not d.symbol or not d.position_id:
            continue
        by_position.setdefault(d.position_id, []).append(d)

    trades = []
    for position_id, position_deals in by_position.items():
        if len(position_deals) < 2:
            continue
        position_deals.sort(key=lambda d: d.time)
        opening_leg = position_deals[0]
        closing_leg = position_deals[-1]
        trades.append(
            ClosedTrade(
                position_id=position_id,
                symbol=closing_leg.symbol,
                side=opening_leg.side,
                volume=closing_leg.volume,
                opened_at=opening_leg.time,
                closed_at=closing_leg.time,
                open_price=opening_leg.price,
                close_price=closing_leg.price,
                profit=sum(d.profit for d in position_deals),
            )
        )
    trades.sort(key=lambda t: t.closed_at, reverse=True)
    return trades


_MT5_TIMEFRAMES = ("H1", "H4", "D1", "MN1")


def fetch_mt5_price_history(symbol: str, timeframe: str, count: int = 300) -> pd.DataFrame:
    """Real OHLCV bars for `symbol` at `timeframe` ("H1"/"H4"/"D1"/"MN1"),
    fetched directly from the currently-connected MT5 terminal's own
    price feed — the broker's real data for the exact symbol, not a
    best-effort external ticker match the way PMEX's Yahoo-based
    enrichment needs (see data/underlying.py). Shaped into the same
    Open/High/Low/Close/Volume, oldest-first DataFrame shape
    analysis/technical.py already expects, so compute_technical_stats()
    works on it unchanged — no new analysis code needed for a new
    timeframe, only this fetch. Volume here is MT5's tick_volume (real
    tick counts), not a settled trade-volume figure — same caveat this
    project already states for Yahoo's own volume data.

    Empty (but correctly-typed) DataFrame on any failure — no data for
    this symbol/timeframe combination is a normal, expected outcome (a
    newly-listed symbol, a broker that doesn't retain intraday history
    that far back — bar availability is bounded by the broker's own
    server-side history retention, not something this function
    controls), not treated as a connection failure. Matches every other
    price-history fetch function's contract in this codebase
    (data/market_history.py, data/psx_source.py)."""
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"], dtype=float)
    if timeframe not in _MT5_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {timeframe!r} — expected one of {_MT5_TIMEFRAMES}")

    import MetaTrader5 as mt5

    mt5_timeframe = {
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "MN1": mt5.TIMEFRAME_MN1,
    }[timeframe]

    _ensure_symbol_selected(mt5, symbol)
    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        logger.warning(
            "fetch_mt5_price_history(%s, %s): copy_rates_from_pos returned nothing (last_error=%s)",
            symbol, timeframe, error,
        )
        return empty

    raw = pd.DataFrame(rates)
    # .values, not the bare Series — raw's default integer index and the
    # new DatetimeIndex below share no labels, so building this dict from
    # the Series themselves would align-by-label against the `index=`
    # param and silently fill every value with NaN (confirmed live via a
    # failing test) rather than raising anything.
    return pd.DataFrame(
        {
            "Open": raw["open"].astype(float).values,
            "High": raw["high"].astype(float).values,
            "Low": raw["low"].astype(float).values,
            "Close": raw["close"].astype(float).values,
            "Volume": raw["tick_volume"].astype(float).values,
        },
        index=pd.to_datetime(raw["time"], unit="s"),
    )


@dataclass
class TradeCost:
    """Real, MT5-native cost-of-trade figures for a symbol — deliberately
    separate from ContractSpec (which risk/apply_suggestion.py depends on
    for order sizing): this is informational data for the AI prompt, not
    an execution input, and keeping it out of ContractSpec avoids
    widening that dataclass's actual purpose. Commission is NOT included
    here — MT5's API has no field for it at all (confirmed live: not on
    symbol_info(), not on account_info() beyond an unrelated margin-calc
    "commission_blocked" figure) — callers that know their own broker's
    real commission schedule (see ai/ftmo_suggest.py) layer it on top of
    this using `category` to look up the right rate."""

    category: str  # broker's own top-level symbol-path category, e.g. "Forex", "Metals CFD"
    spread_pct_of_price: float  # round-trip cost from crossing the spread once, as a % of price
    swap_long_pct_per_day: float | None  # holding a BUY overnight, %-of-notional/day (negative = a cost)
    swap_short_pct_per_day: float | None  # same, for a SELL
    # Real broker-enforced minimum stop/target distance from the current
    # price (MT5's own SYMBOL_TRADE_STOPS_LEVEL, converted from points to
    # a %-of-price figure) — 0.0 (not None) means this broker/symbol has
    # no such restriction, a real and common value, not a missing one.
    # Added 2026-08-22 for analysis/backtest.py's trade-simulation engine
    # to respect: an ATR-based stop tighter than this floor could never
    # actually have been placed as a real order, so a backtest that
    # ignores it is quietly more favorable than reality.
    min_stop_distance_pct: float = 0.0


# MT5's own swap-calculation-mode constants this project has verified a
# real conversion formula for (confirmed live: a single FTMO account can
# mix both — EURUSD/XAUUSD use POINTS, BTCUSD uses INTEREST_CURRENT).
# Values match MetaTrader5.SYMBOL_SWAP_MODE_POINTS / _INTEREST_CURRENT /
# _INTEREST_OPEN — hardcoded rather than imported from the mt5 module so
# this stays evaluable without a live MT5 import at module load time.
_SWAP_MODE_POINTS = 1
_SWAP_MODE_INTEREST_CURRENT = 5
_SWAP_MODE_INTEREST_OPEN = 6
# Standard forex/CFD market convention for accruing an annual swap rate
# daily — not this project's own choice, matches how brokers themselves
# compute it.
_SWAP_DAYS_PER_YEAR = 360


def _compute_swap_pct_per_day(info, price: float) -> tuple[float | None, float | None]:
    """Real swap cost as a %-of-notional-per-day figure, handling the
    two swap_mode conventions this project has verified live (see the
    constants above). SYMBOL_SWAP_MODE_POINTS: swap_long/short are a raw
    point value, converted to money via trade_tick_value scaled by
    point/trade_tick_size (equal for most symbols, but not guaranteed —
    confirmed live this broker's own point==tick_size for every symbol
    checked, so this scaling is a no-op here, but it's not assumed).
    SYMBOL_SWAP_MODE_INTEREST_CURRENT/_OPEN: swap_long/short are already
    annual % rates of notional, accrued daily on the conventional
    360-day basis. Returns (None, None) for any other, rarer swap_mode —
    this project hasn't verified a formula for those, and a silently
    wrong cost figure is worse than an honest gap."""
    if price <= 0:
        return None, None

    if info.swap_mode == _SWAP_MODE_POINTS:
        if info.trade_tick_size <= 0 or info.trade_contract_size <= 0:
            return None, None
        point_value = info.trade_tick_value * (info.point / info.trade_tick_size)
        notional = info.trade_contract_size * price
        if notional <= 0:
            return None, None
        return (
            info.swap_long * point_value / notional * 100,
            info.swap_short * point_value / notional * 100,
        )

    if info.swap_mode in (_SWAP_MODE_INTEREST_CURRENT, _SWAP_MODE_INTEREST_OPEN):
        return info.swap_long / _SWAP_DAYS_PER_YEAR, info.swap_short / _SWAP_DAYS_PER_YEAR

    return None, None


def get_symbol_category(symbol: str) -> str:
    """The broker's own top-level symbol-path category (e.g. "Forex",
    "Metals CFD", "Crypto", "Agriculture", "Cash CFD") — pulled out of
    get_trade_economics as its own cheap call (symbol_info() only, no
    tick fetch, no spread/swap math) since a symbol's category never
    changes mid-session, so a caller that just wants to group/label
    symbols (e.g. the watchlist grid) shouldn't pay for a live tick +
    swap computation it doesn't need, repeated every fast refresh tick.
    "Uncategorized" if symbol_info() fails or has no path — never
    fabricated, matching every other real-data fetch in this file."""
    try:
        import MetaTrader5 as mt5

        _ensure_symbol_selected(mt5, symbol)
        info = mt5.symbol_info(symbol)
    except Exception as e:
        logger.warning("get_symbol_category raised for %s: %s: %s", symbol, type(e).__name__, e)
        return "Uncategorized"
    if info is None or not getattr(info, "path", ""):
        return "Uncategorized"
    return info.path.split("\\")[0]


def get_trade_economics(symbol: str) -> TradeCost | None:
    """Real spread + swap cost for `symbol`, straight from the live MT5
    feed — the deterministic, "don't make the model guess a number
    Python can compute exactly" counterpart to letting the AI reason
    about whether a trade's edge actually survives real trading costs
    (matters most on a short holding horizon, where a modest edge can be
    mostly or entirely eaten by round-trip spread plus a night or two of
    swap). None on any failure (no live quote, no symbol info) — never
    fabricated, matching every other real-data fetch in this file."""
    try:
        import MetaTrader5 as mt5

        _ensure_symbol_selected(mt5, symbol)
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
    except Exception as e:
        logger.warning("get_trade_economics raised for %s: %s: %s", symbol, type(e).__name__, e)
        return None

    if info is None or tick is None or tick.ask <= 0 or tick.bid <= 0:
        return None

    spread_pct = (tick.ask - tick.bid) / tick.ask * 100
    swap_long_pct, swap_short_pct = _compute_swap_pct_per_day(info, tick.ask)
    category = info.path.split("\\")[0] if getattr(info, "path", "") else "Uncategorized"

    # trade_stops_level is a POINT count (0 = broker imposes no minimum
    # distance at all, a common real value on ECN-style accounts — not
    # every symbol/broker enforces one). Guarded with `getattr`/`or 0`
    # since a stubbed or unusually old symbol_info() might not carry
    # this field at all — degrades to "no known restriction" rather than
    # raising, same convention as every other optional field here.
    stops_level_points = getattr(info, "trade_stops_level", 0) or 0
    min_stop_distance_pct = stops_level_points * info.point / tick.ask * 100 if stops_level_points > 0 else 0.0

    return TradeCost(
        category=category,  # kept inline (not get_symbol_category) — info is already fetched here
        spread_pct_of_price=spread_pct,
        swap_long_pct_per_day=swap_long_pct,
        swap_short_pct_per_day=swap_short_pct,
        min_stop_distance_pct=min_stop_distance_pct,
    )
