from datetime import datetime, timedelta, timezone

import config
from ai import rehunt

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


def test_a_dead_entry_is_recorded_once_per_symbol_per_utc_day():
    assert rehunt.record_dead_entry("SOLUSD", "buy", 82.5, 81.0, 86.0, "target reached without a fill", NOW) is True
    assert rehunt.record_dead_entry("SOLUSD", "buy", 83.0, 81.5, 87.0, "a second cause the same day", NOW + timedelta(hours=2)) is False
    records = rehunt.recent_dead_entries(NOW + timedelta(hours=3))
    assert len(records) == 1 and records[0]["reason"] == "target reached without a fill"  # the first cause wins
    # A new UTC day is a new chance.
    assert rehunt.record_dead_entry("SOLUSD", "buy", 83.0, 81.5, 87.0, "next day", NOW + timedelta(days=1)) is True


def test_old_records_age_out_of_the_prompt_and_the_file():
    rehunt.record_dead_entry("NVDA", "buy", 180.0, 178.0, 186.0, "old", NOW - timedelta(hours=48))
    assert rehunt.recent_dead_entries(NOW) == []
    rehunt.record_dead_entry("AMD", "sell", 150.0, 152.0, 144.0, "fresh", NOW)
    rehunt.record_dead_entry("META", "buy", 600.0, 596.0, 612.0, "ancient", NOW - timedelta(days=config.REHUNT_LEDGER_KEEP_DAYS + 2))
    rehunt.record_dead_entry("TSLA", "buy", 300.0, 296.0, 312.0, "today", NOW)
    symbols = {r["symbol"] for r in rehunt.recent_dead_entries(NOW)}
    assert symbols == {"AMD", "TSLA"}
    import json
    from pathlib import Path

    stored = json.loads(Path(config.REHUNT_LEDGER_FILE).read_text())
    assert not any("META" in key for key in stored)  # pruned beyond the keep window


def test_the_prompt_block_lists_history_as_history_and_is_empty_when_nothing_died():
    assert rehunt.format_rehunt_block(NOW) == ""
    rehunt.record_dead_entry("SOLUSD", "buy", 82.5, 81.0, 86.0, "live price 86.4 reached the target 86 without the entry ever filling", NOW)
    block = rehunt.format_rehunt_block(NOW + timedelta(hours=1))
    assert "RE-HUNT CANDIDATES" in block and "ONCE" in block and "history, not a level to reuse" in block
    assert "SOLUSD BUY" in block and "drafted entry 82.5 / stop 81 / target 86" in block and "reached the target 86" in block


def test_the_ledger_never_raises_even_on_a_corrupt_or_unwritable_file(tmp_path, monkeypatch):
    from pathlib import Path

    Path(config.REHUNT_LEDGER_FILE).write_text("{not json")
    assert rehunt.recent_dead_entries(NOW) == []
    assert rehunt.record_dead_entry("SOLUSD", "buy", 1.0, 0.9, 1.2, "x", NOW) is True  # corrupt file is replaced
    monkeypatch.setattr(config, "REHUNT_LEDGER_FILE", str(tmp_path / "missing_dir" / "ledger.json"))
    assert rehunt.record_dead_entry("AMD", "buy", 1.0, 0.9, 1.2, "x", NOW) is False  # cannot write: swallowed, not raised


def test_the_clerk_records_a_target_reached_entry_and_an_entry_mode_reject():
    from ai.clerk_execution import _apply_entry_mode_guard, _apply_stale_entry_reanchor
    from ai.ftmo_suggest import FtmoAssetAnalysis
    from ai.portfolio_suggest import AllocationEntry, AssetAnalysis
    from analysis.chart_structure import ChartStructureSnapshot
    from analysis.technical import TechnicalStats
    from data.mt5_source import MarketAsset

    def stats(atr):
        return TechnicalStats(**{**{f: None for f in TechnicalStats.__dataclass_fields__}, "atr": atr})

    empty = ChartStructureSnapshot(fibonacci=None, sr_levels=None, trendlines=None, patterns=[])
    analysis = FtmoAssetAnalysis(
        base=AssetAnalysis(symbol="EURUSD", description="x", bid=101.0, ask=101.02, display_name=None),
        h4_stats=stats(None), h1_stats=stats(None), h4_structure=empty, h1_structure=empty, trade_cost=None,
        m5_stats=stats(0.2),
    )
    cache = {"EURUSD": (analysis, "")}
    reached = AllocationEntry(pct=1.0, price=99.0, stop_loss=98.0, take_profit=101.0, side="buy", reason="thesis")
    assert _apply_stale_entry_reanchor({"EURUSD": reached}, [], cache)["EURUSD"].pct == 0.0
    dead = AllocationEntry(pct=1.0, price=100.0, stop_loss=101.5, take_profit=104.0, side="buy", reason="thesis", entry_mode="market")
    prices = {"EURUSD": MarketAsset(symbol="EURUSD", description="x", bid=101.0, ask=101.02)}
    _apply_entry_mode_guard({"EURUSD": dead}, [], [], cache, prices, set())
    records = rehunt.recent_dead_entries()
    assert [r["symbol"] for r in records] == ["EURUSD"]  # once per symbol per day, the first cause kept
    assert "reached the target" in records[0]["reason"]
