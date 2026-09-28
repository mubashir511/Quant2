import pandas as pd
import pytest

from analysis.intraday_context import compute_intraday_levels


def _d1(rows):
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {"Open": [r[1] for r in rows], "High": [r[2] for r in rows], "Low": [r[3] for r in rows], "Close": [r[4] for r in rows]},
        index=idx,
    )


def _m5(start, closes, volume=100):
    idx = pd.date_range(start, periods=len(closes), freq="5min")
    return pd.DataFrame(
        {"Open": closes, "High": [c + 0.1 for c in closes], "Low": [c - 0.1 for c in closes], "Close": closes, "Volume": volume},
        index=idx,
    )


def test_levels_use_prior_completed_day_and_todays_forming_bar():
    d1 = _d1([
        ("2026-09-18", 98, 101, 97, 100),
        ("2026-09-21", 100, 104, 99, 103),   # prior completed day
        ("2026-09-22", 103, 105, 102, 104),  # today's forming bar
    ])
    m5 = _m5("2026-09-22 00:00", [103, 104, 105, 104])
    lv = compute_intraday_levels(m5, d1)
    assert (lv.prev_day_high, lv.prev_day_low, lv.prev_day_close) == (104, 99, 103)
    assert (lv.day_open, lv.day_high, lv.day_low) == (103, 105, 102)
    assert lv.last_price == 104
    # ADR over the two completed days: (101-97 + 104-99)/2 = 4.5; today's range 3 => 66.7% used
    assert lv.adr == pytest.approx(4.5)
    assert lv.range_used_pct == pytest.approx(3 / 4.5 * 100)


def test_vwap_only_covers_todays_session_bars_and_weights_by_volume():
    d1 = _d1([("2026-09-21", 100, 104, 99, 103), ("2026-09-22", 103, 105, 102, 104)])
    yesterday = _m5("2026-09-21 20:00", [50, 50])  # must be excluded
    today = _m5("2026-09-22 00:00", [100, 110])
    today["Volume"] = [300, 100]
    lv = compute_intraday_levels(pd.concat([yesterday, today]), d1)
    assert lv.session_bars == 2
    expected = (100 * 300 + 110 * 100) / 400  # typical price == close here (High/Low symmetric)
    assert lv.vwap == pytest.approx(expected)


def test_vwap_is_none_without_volume():
    d1 = _d1([("2026-09-21", 100, 104, 99, 103), ("2026-09-22", 103, 105, 102, 104)])
    m5 = _m5("2026-09-22 00:00", [100, 101], volume=0)
    assert compute_intraday_levels(m5, d1).vwap is None


def test_none_without_enough_real_data():
    d1 = _d1([("2026-09-22", 103, 105, 102, 104)])
    assert compute_intraday_levels(_m5("2026-09-22 00:00", [1, 2]), d1) is None
    assert compute_intraday_levels(pd.DataFrame(), d1) is None
