from datetime import datetime, timedelta

from data.mt5_source import AccountSummary, MarketAsset, Position


def get_open_positions() -> list[Position]:
    """Synthetic positions covering each rebalance rule for offline dev/testing."""
    now = datetime.now()
    return [
        # Breaches MAX_LOSS_PCT, has a stop set.
        Position("GO10OZ", 1.0, "buy", 2000.0, 1750.0, sl=1900.0, profit=-250.0,
                 opened_at=now - timedelta(days=3)),
        # No stop loss set at all.
        Position("SV5OZ", 2.0, "sell", 30.0, 30.5, sl=None, profit=-100.0,
                  opened_at=now - timedelta(days=1)),
        # Healthy position, no rule triggered.
        Position("CL100BBL", 1.0, "buy", 80.0, 82.0, sl=78.0, profit=200.0,
                  opened_at=now - timedelta(hours=6)),
        # Concentrated symbol exposure (large notional vs. the rest).
        Position("GO10OZ", 5.0, "buy", 2000.0, 2010.0, sl=1950.0, profit=50.0,
                  opened_at=now - timedelta(hours=2)),
        # Pushes total position count over MAX_POSITION_COUNT.
        Position("NG10K", 1.0, "sell", 3.0, 3.05, sl=3.2, profit=-15.0,
                  opened_at=now - timedelta(hours=1)),
        Position("PL50OZ", 1.0, "buy", 1000.0, 1010.0, sl=980.0, profit=10.0,
                  opened_at=now - timedelta(minutes=30)),
        Position("HG25K", 1.0, "buy", 4.0, 4.02, sl=3.9, profit=8.0,
                  opened_at=now - timedelta(minutes=10)),
    ]


def get_account_summary() -> AccountSummary:
    return AccountSummary(balance=10000.0, equity=9903.0, free_margin=8500.0, currency="USD")


def get_market_watch() -> list[MarketAsset]:
    """Synthetic Market Watch instruments for offline dev/testing."""
    return [
        MarketAsset("GO10OZ", "Gold 10oz", bid=2000.0, ask=2000.5),
        MarketAsset("SV5OZ", "Silver 5oz", bid=30.0, ask=30.05),
        MarketAsset("CL100BBL", "Crude Oil 100bbl", bid=80.0, ask=80.10),
        MarketAsset("NG10K", "Natural Gas 10k MMBtu", bid=3.0, ask=3.02),
        MarketAsset("HG25K", "Copper 25k lb", bid=4.0, ask=4.02),
    ]
