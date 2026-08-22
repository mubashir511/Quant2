from datetime import datetime, timedelta

from data.mt5_source import AccountSummary, MarketAsset, PendingOrder, Position


def get_open_positions() -> list[Position]:
    """Synthetic positions covering each rebalance rule for offline dev/testing."""
    now = datetime.now()
    return [
        # Breaches MAX_LOSS_PCT, has a stop set.
        Position("GO10OZ", 1.0, "buy", 2000.0, 1750.0, sl=1900.0, profit=-250.0,
                 opened_at=now - timedelta(days=3), ticket=100001),
        # No stop loss set at all.
        Position("SV5OZ", 2.0, "sell", 30.0, 30.5, sl=None, profit=-100.0,
                  opened_at=now - timedelta(days=1), ticket=100002),
        # Healthy position, no rule triggered.
        Position("CL100BBL", 1.0, "buy", 80.0, 82.0, sl=78.0, profit=200.0,
                  opened_at=now - timedelta(hours=6), ticket=100003),
        # Concentrated symbol exposure (large notional vs. the rest).
        Position("GO10OZ", 5.0, "buy", 2000.0, 2010.0, sl=1950.0, profit=50.0,
                  opened_at=now - timedelta(hours=2), ticket=100004),
        # Pushes total position count over MAX_POSITION_COUNT.
        Position("NG10K", 1.0, "sell", 3.0, 3.05, sl=3.2, profit=-15.0,
                  opened_at=now - timedelta(hours=1), ticket=100005),
        Position("PL50OZ", 1.0, "buy", 1000.0, 1010.0, sl=980.0, profit=10.0,
                  opened_at=now - timedelta(minutes=30), ticket=100006),
        Position("HG25K", 1.0, "buy", 4.0, 4.02, sl=3.9, profit=8.0,
                  opened_at=now - timedelta(minutes=10), ticket=100007),
    ]


def get_pending_orders() -> list[PendingOrder]:
    """No synthetic pending orders are modeled — an empty list matches
    get_history_deals()'s own "no mock implementation, empty is a normal
    case" contract below."""
    return []


def get_current_price(symbol: str) -> float | None:
    """No synthetic tick feed is modeled — mirrors get_pending_orders()'s
    own "no mock implementation" contract, since mock mode has no
    pending orders to show a current price next to anyway."""
    return None


def get_current_bid_ask(symbol: str) -> tuple[float, float] | None:
    """Unlike get_current_price, this one IS backed by the same
    synthetic Market Watch data get_market_watch() returns (rather than
    a bare None) — the watchlist's per-asset detail popup needs a real
    (bid, ask) pair to render anything meaningful. Reads the shared
    _MOCK_MARKET_WATCH list directly rather than calling get_market_
    watch() — a real single-symbol lookup shouldn't masquerade as (or,
    in a test counting calls, get confused with) a full-list fetch; the
    real data/mt5_source.py version is genuinely a single cheap tick
    read, and this should cost-model the same way."""
    for asset in _MOCK_MARKET_WATCH:
        if asset.symbol == symbol:
            return asset.bid, asset.ask
    return None


def get_account_summary() -> AccountSummary:
    return AccountSummary(balance=10000.0, equity=9903.0, free_margin=8500.0, currency="USD")


def get_history_deals(date_from: datetime, date_to: datetime | None = None) -> list:
    """No synthetic trade history is modeled here — an empty list is
    exactly data/mt5_source.py's own real "no history" contract too (a
    fresh account with zero trades is a normal, expected case there), so
    FTMO's compliance-status reconstruction (risk/ftmo_rules.py) degrades
    cleanly to its documented zero-history case under mock data instead
    of crashing for lack of a mock implementation."""
    return []


_MOCK_MARKET_WATCH = [
    MarketAsset("GO10OZ", "Gold 10oz", bid=2000.0, ask=2000.5),
    MarketAsset("SV5OZ", "Silver 5oz", bid=30.0, ask=30.05),
    MarketAsset("CL100BBL", "Crude Oil 100bbl", bid=80.0, ask=80.10),
    MarketAsset("NG10K", "Natural Gas 10k MMBtu", bid=3.0, ask=3.02),
    MarketAsset("HG25K", "Copper 25k lb", bid=4.0, ask=4.02),
]


def get_market_watch() -> list[MarketAsset]:
    """Synthetic Market Watch instruments for offline dev/testing."""
    return list(_MOCK_MARKET_WATCH)


def get_symbol_category(symbol: str) -> str:
    """Synthetic category matching the real broker's own top-level
    symbol-path grouping convention (see data/mt5_source.py's own
    get_symbol_category) — enough to exercise the watchlist's category
    grouping offline, without a live MT5 connection."""
    return {
        "GO10OZ": "Metals CFD",
        "SV5OZ": "Metals CFD",
        "HG25K": "Metals CFD",
        "CL100BBL": "Commodities",
        "NG10K": "Commodities",
    }.get(symbol, "Uncategorized")
