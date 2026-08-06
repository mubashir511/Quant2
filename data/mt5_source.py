from dataclasses import dataclass
from datetime import datetime

import config


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
    """Tradable instruments the user has added to their MT5 Market Watch."""
    import MetaTrader5 as mt5

    symbols = mt5.symbols_get()
    if symbols is None:
        error = mt5.last_error()
        raise MT5ConnectionError(f"Failed to fetch symbols ({error}).")

    assets = []
    for s in symbols:
        if not s.visible:
            continue
        tick = mt5.symbol_info_tick(s.name)
        if tick is None or (tick.bid == 0 and tick.ask == 0):
            continue  # no live quote right now
        assets.append(
            MarketAsset(symbol=s.name, description=s.description, bid=tick.bid, ask=tick.ask)
        )
    return assets


def get_contract_spec(symbol: str) -> ContractSpec | None:
    """Real order-size/margin constraints for a symbol — e.g. a minimum
    lot can require far more margin than a small account holds at all,
    not just "a lot" of it. None on any failure (also the expected result
    under USE_MOCK_DATA, where there's no live MT5 connection)."""
    try:
        import MetaTrader5 as mt5

        info = mt5.symbol_info(symbol)
    except Exception:
        return None
    if info is None:
        return None

    return ContractSpec(
        volume_min=info.volume_min,
        volume_step=info.volume_step,
        volume_max=info.volume_max,
        trade_contract_size=info.trade_contract_size,
        currency_margin=info.currency_margin,
        margin_initial=info.margin_initial,
    )
