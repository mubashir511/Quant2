import csv
import logging
import time
from dataclasses import dataclass
from datetime import datetime

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


def connect() -> None:
    import MetaTrader5 as mt5

    kwargs = {}
    if config.MT5_PATH:
        kwargs["path"] = config.MT5_PATH
    if config.MT5_LOGIN:
        kwargs["login"] = int(config.MT5_LOGIN)
    if config.MT5_PASSWORD:
        kwargs["password"] = config.MT5_PASSWORD
    if config.MT5_SERVER:
        kwargs["server"] = config.MT5_SERVER

    if not mt5.initialize(**kwargs):
        error = mt5.last_error()
        raise MT5ConnectionError(
            f"Could not connect to the MT5 terminal ({error}). "
            "Make sure the terminal is running and logged in."
        )


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
            )
        )
    return positions


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
    path, not just another retry of the same one."""
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
            spec = ContractSpec(
                volume_min=info.volume_min,
                volume_step=info.volume_step,
                volume_max=info.volume_max,
                trade_contract_size=info.trade_contract_size,
                currency_margin=info.currency_margin,
                margin_initial=info.margin_initial,
            )
        except AttributeError as e:
            logger.warning("mt5.symbol_info(%s) returned an incomplete object: %s", symbol, e)

    if spec is None and config.MT5_SYMBOL_SPECS_CSV_PATH:
        spec = _load_symbol_spec_from_csv(symbol, config.MT5_SYMBOL_SPECS_CSV_PATH)
        if spec is not None:
            logger.info("get_contract_spec: used CSV fallback for %s (live API had nothing)", symbol)

    return spec
