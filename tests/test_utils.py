import time

import pytest

from utils import run_with_timeout


def test_run_with_timeout_returns_real_result_when_fast_enough():
    assert run_with_timeout(lambda: 42, timeout_seconds=2, default=-1) == 42


def test_run_with_timeout_returns_default_on_timeout():
    def _slow():
        time.sleep(2)
        return "too late"

    assert run_with_timeout(_slow, timeout_seconds=0.1, default="fallback") == "fallback"


def test_run_with_timeout_catches_real_exception_by_default():
    def _boom():
        raise ValueError("real failure, not a timeout")

    assert run_with_timeout(_boom, timeout_seconds=2, default="fallback") == "fallback"


def test_run_with_timeout_propagates_real_exception_when_not_catching():
    def _boom():
        raise ValueError("real failure, not a timeout")

    with pytest.raises(ValueError, match="real failure"):
        run_with_timeout(_boom, timeout_seconds=2, default="fallback", catch_exceptions=False)


def test_run_with_timeout_still_swallows_the_timeout_itself_when_not_catching_exceptions():
    # catch_exceptions=False only affects genuine exceptions raised BY
    # the wrapped call — the timeout itself must always resolve to
    # `default`, never propagate as a raised TimeoutError, regardless of
    # that flag (a caller passing catch_exceptions=False specifically
    # wants to distinguish "timed out" from "raised" via the return
    # value, not have the timeout itself become a third, unhandled case).
    def _slow():
        time.sleep(2)
        return "too late"

    result = run_with_timeout(_slow, timeout_seconds=0.1, default="fallback", catch_exceptions=False)
    assert result == "fallback"
