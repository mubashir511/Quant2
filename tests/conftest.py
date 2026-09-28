"""Test-suite-wide isolation of the REAL, live data locations.

Found on audit (2026-09-24): tests that drive run_clerk_execution_check / the tactical pipeline with fixture
symbols like "XAUUSD" were appending fake events (hundreds of `tactical_action` entries from stub verdicts) to the
REAL trade journal under records/ftmo_trade_journal — and through it into the REAL Obsidian "Trades/Lifecycle"
notes — every time the suite ran, because only some test modules redirected config.TRADE_JOURNAL_DIR. Redirecting
these locations for EVERY test, by default, means a new test can no longer forget to. A test that needs its own
location still overrides them (its own patch is applied inside this one)."""

import pytest

import config


@pytest.fixture(autouse=True)
def _isolate_real_journal_and_vault(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRADE_JOURNAL_DIR", str(tmp_path / "_isolated_trade_journal"))
    monkeypatch.setattr(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "_isolated_obsidian_vault"))
    # The shared news archive/caches: a test that runs the Clerk check for a fixture symbol ("TESTSYM") used
    # to fetch real headlines and append them to records/ftmo_symbol_news/TESTSYM.json (found on audit).
    monkeypatch.setattr(config, "SYMBOL_NEWS_DIR", str(tmp_path / "_isolated_symbol_news"))
    monkeypatch.setattr(config, "CATEGORY_NEWS_DIR", str(tmp_path / "_isolated_category_news"))
    monkeypatch.setattr(config, "NEWS_FETCH_CACHE_FILE", str(tmp_path / "_isolated_news_fetch_cache.json"))
    monkeypatch.setattr(config, "ECONOMIC_CALENDAR_CACHE_FILE", str(tmp_path / "_isolated_calendar_cache.json"))
    # The Mega final live re-check reads the live MT5 feed; a test that wants it turns it on explicitly.
    monkeypatch.setattr(config, "LIVE_RECHECK_ENABLED", False)
    # The re-hunt ledger (ai/rehunt.py) is a real file the Clerk writes; never let a test touch the live one.
    monkeypatch.setattr(config, "REHUNT_LEDGER_FILE", str(tmp_path / "_isolated_rehunt_ledger.json"))
    monkeypatch.setattr(config, "PRICE_CACHE_DIR", str(tmp_path / "_isolated_price_cache"))
    # Broker-side order expiry needs a live M5 fetch per order; a test that wants it turns it on explicitly.
    monkeypatch.setattr(config, "RESTING_ORDER_EXPIRY_BEFORE_CLOSE_MINUTES", -1.0)
    # The Sentinel's state file is what the Clerk ingests stops from; never read the live one in a test.
    monkeypatch.setattr(config, "SENTINEL_STATE_FILE", str(tmp_path / "_isolated_sentinel_state.json"))
    monkeypatch.setattr(config, "SYMBOL_CARD_FILE", str(tmp_path / "_isolated_symbol_cards.json"))
    # The Mega tradable-now filter reads live ticks + the wall-clock calendar; a test that wants it turns it on explicitly.
    monkeypatch.setattr(config, "MEGA_TRADABLE_FILTER_ENABLED", False)
    # Existing Clerk tests exercise the original single poll that thinks and acts; the split has its own tests.
    monkeypatch.setattr(config, "CLERK_THINK_SPLIT_ENABLED", False)
    # The compact model context / mechanical invalidation have their own tests; the older tests exercise the original full-context path.
    monkeypatch.setattr(config, "CLERK_COMPACT_CONTEXT", False)
    monkeypatch.setattr(config, "CLERK_DETERMINISTIC_INVALIDATION", False)
    monkeypatch.setattr(config, "CLERK_THINK_CACHE_FILE", str(tmp_path / "_isolated_thinking_cache.json"))
    # Continuation Watch (ai/continuation_hunter.py): its own state file, never the live one.
    monkeypatch.setattr(config, "CONTINUATION_WATCH_STATE_FILE", str(tmp_path / "_isolated_continuation_watch_state.json"))
    yield
