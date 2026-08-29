import pandas as pd

from analysis.candlestick_patterns import (
    CANDLESTICK_MIN_BARS,
    _prior_trend,
    detect_candlestick_patterns,
)


def _bar(open_, high, low, close):
    return {"Open": open_, "High": high, "Low": low, "Close": close}


def _history(bars: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(bars)


def _names(patterns) -> list[str]:
    return [p.name for p in patterns]


def _filler_downtrend(n: int, start: float = 100.0) -> list[dict]:
    return [_bar(start - i * 0.5, start - i * 0.5 + 0.1, start - i * 0.5 - 0.1, start - i * 0.5 - 0.4) for i in range(n)]


def _filler_uptrend(n: int, start: float = 90.0) -> list[dict]:
    return [_bar(start + i * 0.5, start + i * 0.5 + 0.4, start + i * 0.5 - 0.1, start + i * 0.5 + 0.3) for i in range(n)]


def test_detect_candlestick_patterns_returns_empty_below_min_bars():
    bars = _filler_downtrend(CANDLESTICK_MIN_BARS - 1)
    assert detect_candlestick_patterns(_history(bars)) == []


def test_detect_candlestick_patterns_returns_empty_without_ohlc_columns():
    df = pd.DataFrame({"Close": [1.0] * CANDLESTICK_MIN_BARS})
    assert detect_candlestick_patterns(df) == []


def test_prior_trend_none_without_enough_history():
    closes = pd.Series([1.0, 2.0, 3.0])
    assert _prior_trend(closes, idx=1) is None


def test_prior_trend_up_and_down():
    closes = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 105.0, 106.0])
    assert _prior_trend(closes, idx=7) == "up"
    closes_down = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 95.0, 94.0])
    assert _prior_trend(closes_down, idx=7) == "down"


def test_prior_trend_none_when_flat():
    closes = pd.Series([100.0] * 8)
    assert _prior_trend(closes, idx=7) is None


def test_doji_detected():
    bars = _filler_downtrend(7) + [_bar(97.0, 97.5, 96.5, 97.02)]
    patterns = detect_candlestick_patterns(_history(bars))
    assert "doji" in _names(patterns)


def test_bullish_engulfing_detected():
    bars = _filler_downtrend(6) + [
        _bar(100.4, 100.5, 99.9, 100.0),  # bearish
        _bar(99.9, 100.7, 99.8, 100.6),  # bullish, engulfs prior body
    ]
    patterns = detect_candlestick_patterns(_history(bars))
    assert "bullish_engulfing" in _names(patterns)


def test_bearish_engulfing_detected():
    bars = _filler_downtrend(6) + [
        _bar(100.0, 100.5, 99.9, 100.4),  # bullish
        _bar(100.6, 100.7, 99.5, 99.6),  # bearish, engulfs prior body
    ]
    patterns = detect_candlestick_patterns(_history(bars))
    assert "bearish_engulfing" in _names(patterns)


def test_engulfing_detected_with_no_gap_between_bars():
    # Real bug found on self-audit: continuously-traded FTMO instruments
    # never gap between consecutive bars (confirmed live — median open-
    # vs-prior-close difference is exactly 0.0 across real H1 data) — an
    # earlier version required a strict gap beyond the prior body and
    # missed 20-40% of real engulfing shapes as a result. The current
    # bar opening EXACTLY at the prior bar's close (no gap at all) must
    # still count as engulfing as long as the body itself fully
    # contains the prior one.
    bars = _filler_downtrend(6) + [
        _bar(100.4, 100.5, 99.9, 100.0),  # bearish, closes at 100.0
        _bar(100.0, 100.7, 99.8, 100.6),  # bullish, opens exactly at 100.0 — no gap
    ]
    patterns = detect_candlestick_patterns(_history(bars))
    assert "bullish_engulfing" in _names(patterns)


def test_hammer_detected_after_downtrend():
    bars = _filler_downtrend(7) + [_bar(96.6, 97.05, 95.4, 97.0)]  # body 0.4, lower shadow 1.2 (3x)
    patterns = detect_candlestick_patterns(_history(bars))
    assert "hammer" in _names(patterns)
    assert "hanging_man" not in _names(patterns)


def test_hanging_man_detected_after_uptrend_same_shape_as_hammer():
    bars = _filler_uptrend(7) + [_bar(93.0, 93.05, 91.6, 92.6)]  # identical shape, uptrend context
    patterns = detect_candlestick_patterns(_history(bars))
    assert "hanging_man" in _names(patterns)
    assert "hammer" not in _names(patterns)


def test_shooting_star_detected_after_uptrend():
    bars = _filler_uptrend(7) + [_bar(92.6, 94.0, 92.5, 93.0)]  # body 0.4, upper shadow 1.0 (2.5x)
    patterns = detect_candlestick_patterns(_history(bars))
    assert "shooting_star" in _names(patterns)
    assert "inverted_hammer" not in _names(patterns)


def test_inverted_hammer_detected_after_downtrend_same_shape_as_shooting_star():
    bars = _filler_downtrend(7) + [_bar(97.0, 98.4, 96.9, 97.4)]  # identical shape, downtrend context
    patterns = detect_candlestick_patterns(_history(bars))
    assert "inverted_hammer" in _names(patterns)
    assert "shooting_star" not in _names(patterns)


def test_no_hammer_family_pattern_without_prior_trend_context():
    # A textbook hammer shape sitting on a perfectly flat run has no
    # real prior trend to resolve hammer-vs-hanging_man from — must not
    # emit either name rather than guessing.
    bars = [_bar(100.0, 100.1, 99.9, 100.0)] * 7 + [_bar(96.6, 97.05, 95.4, 97.0)]
    patterns = detect_candlestick_patterns(_history(bars))
    assert "hammer" not in _names(patterns)
    assert "hanging_man" not in _names(patterns)


def test_morning_star_detected():
    bars = _filler_downtrend(6) + [
        _bar(97.6, 97.7, 96.5, 96.6),  # long bearish, body 1.0, midpoint 97.1
        _bar(96.5, 96.65, 96.4, 96.55),  # small star, sits below midpoint
        _bar(96.6, 97.6, 96.5, 97.5),  # bullish closing back above midpoint
    ]
    patterns = detect_candlestick_patterns(_history(bars))
    assert "morning_star" in _names(patterns)


def test_evening_star_detected():
    bars = _filler_downtrend(6) + [
        _bar(96.6, 97.7, 96.5, 97.6),  # long bullish, body 1.0, midpoint 97.1
        _bar(97.6, 97.75, 97.5, 97.65),  # small star, sits above midpoint
        _bar(97.6, 97.7, 96.5, 96.6),  # bearish closing back below midpoint
    ]
    patterns = detect_candlestick_patterns(_history(bars))
    assert "evening_star" in _names(patterns)


def test_morning_star_rejected_when_flanking_candles_are_not_actually_long():
    # Real bug found on self-audit: an earlier version never checked
    # that the first/third candles were genuinely LONG relative to the
    # star — three tiny, nearly-flat bodies (the star only barely
    # smaller than its neighbors) must NOT read as a strong 3-bar
    # reversal just because the direction/midpoint arithmetic happens
    # to line up.
    bars = _filler_downtrend(6) + [
        _bar(97.11, 97.15, 96.95, 97.10),  # barely bearish, body 0.01
        _bar(97.10, 97.13, 97.02, 97.08),  # "star" body 0.02 — not meaningfully smaller
        _bar(97.09, 97.16, 97.00, 97.11),  # barely bullish, body 0.02
    ]
    patterns = detect_candlestick_patterns(_history(bars))
    assert "morning_star" not in _names(patterns)


def test_multiple_patterns_can_fire_at_once():
    # A doji-shaped final bar that ALSO genuinely engulfs the prior
    # bearish body — both signals are real and independently true, so
    # both should appear, matching setup_classifier.py's own "more than
    # one signal can legitimately fire" convention.
    bars = _filler_downtrend(6) + [
        _bar(100.4, 100.5, 99.9, 100.0),  # bearish, body 0.4
        _bar(99.9, 104.0, 96.0, 100.5),  # bullish, engulfs prior body, but body/range=0.075 (doji)
    ]
    patterns = detect_candlestick_patterns(_history(bars))
    names = _names(patterns)
    assert "doji" in names
    assert "bullish_engulfing" in names
