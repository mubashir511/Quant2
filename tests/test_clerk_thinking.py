from datetime import datetime, timedelta, timezone

import config
from ai import clerk_thinking as ct

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)


def test_a_verdict_is_used_once_only_while_fresh_and_only_for_its_own_suggestion():
    cache = ct.for_suggestion(ct.load_cache(), "gen-1")
    ct.store(cache, "invalidation", "EURUSD", {"confirmed": True, "raw_text": "r"}, NOW)
    assert ct.take_fresh(cache, "invalidation", "EURUSD", NOW + timedelta(minutes=1))["confirmed"] is True
    assert ct.take_fresh(cache, "invalidation", "EURUSD", NOW + timedelta(minutes=2)) is None  # used once
    ct.store(cache, "pending", "TSLA", {"confirmed": True}, NOW)
    assert ct.take_fresh(cache, "pending", "TSLA", NOW + timedelta(minutes=config.CLERK_THINK_CACHE_TTL_MINUTES + 1)) is None  # stale
    assert ct.take_fresh(cache, "pending", "NOPE", NOW) is None
    # a new Mega suggestion never inherits the old session's verdicts
    ct.store(cache, "tactical", "SOL", {"verdict": {}}, NOW)
    fresh = ct.for_suggestion(cache, "gen-2")
    assert fresh["items"] == {} and fresh["generated_utc"] == "gen-2"
    assert ct.for_suggestion(cache, "gen-1") is cache


def test_round_trip_and_is_think_due():
    cache = ct.for_suggestion(ct.load_cache(), "gen-1")
    assert ct.is_think_due(cache, "gen-1", NOW, 5)  # never thought yet
    ct.mark_think_completed(cache, NOW)
    ct.save_cache(cache)
    loaded = ct.load_cache()
    assert not ct.is_think_due(loaded, "gen-1", NOW + timedelta(minutes=4), 5)
    assert ct.is_think_due(loaded, "gen-1", NOW + timedelta(minutes=5), 5)
    assert ct.is_think_due(loaded, "gen-2", NOW + timedelta(minutes=1), 5)  # a NEW suggestion is thought about at once
