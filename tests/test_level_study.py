import numpy as np
import pandas as pd
import pytest

from tools.studies import levels


def test_outcome_touch_hold_and_break_for_a_support():
    hi = np.array([100, 100.2, 100.1, 101.5, 102.5], float)
    lo = np.array([100, 99.05, 99.2, 100.0, 101.0], float)
    touched, held, bounce = levels.outcome(hi, lo, 0, 99.0, "sup", 1.0)  # bar 1 low 99.05 <= 99.0 + 0.1
    assert touched and held and bounce >= 1.0
    broke = levels.outcome(np.array([100, 99.5, 99.4, 99.3], float), np.array([100, 98.9, 98.0, 97.0], float), 0, 99.0, "sup", 1.0)
    assert broke[0] and not broke[1]  # touched, then traded 0.75 ATR through before bouncing 1 ATR
    assert levels.outcome(np.array([100, 100.5, 100.4], float), np.array([100, 99.6, 99.7], float), 0, 98.0, "sup", 1.0)[0] is False


def test_outcome_mirrors_for_a_resistance():
    hi = np.array([100, 100.9, 100.4, 99.5], float)
    lo = np.array([100, 100.0, 99.0, 98.9], float)
    touched, held, _ = levels.outcome(hi, lo, 0, 101.0, "res", 1.0)
    assert touched and held


def test_round_numbers_scale_with_the_price():
    assert levels.round_step(116.0) == 10 and levels.round_step(60000.0) == 1000 and levels.round_step(1.08) == pytest.approx(0.1)
    assert levels.is_round(120.05, 0.5, 116.0) and not levels.is_round(117.3, 0.5, 116.0)
    assert levels.is_round(60003.0, 30.0, 61000.0)


def _wavy_frame(n=4200, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = 100 + 4 * np.sin(t / 60.0) + 2 * np.sin(t / 17.0) + rng.normal(0, 0.15, n)
    high, low = close + np.abs(rng.normal(0.15, 0.05, n)), close - np.abs(rng.normal(0.15, 0.05, n))
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": rng.uniform(50, 150, n)}, index=idx)


@pytest.fixture(scope="module")
def rows():
    return levels.collect_level_rows({"A": _wavy_frame(seed=1), "B": _wavy_frame(seed=2)}, step=250, warmup=1300)


def test_rows_cover_every_candidate_kind_with_their_features(rows):
    assert {"zone", "reaction", "range", "random"} <= set(rows.kind)
    for column in ("sym", "time", "kind", "side", "dist", "touched", "held", "bounce", "round", "h1_conf"):
        assert column in rows.columns
    assert rows.dist.between(0.3, 5.0).all()
    zones = rows[rows.kind == "zone"]
    assert zones.touches.notna().all() and (zones.touches >= 1).all()
    assert rows[rows.kind == "reaction"].bars_ago.notna().all()
    assert (rows[~rows.touched].held == False).all() and (rows[~rows.touched].bounce == 0).all()  # noqa: E712


def test_the_sides_are_on_the_right_side_of_the_price_by_construction(rows):
    assert set(rows.side) == {"sup", "res"}
    assert rows.groupby(["sym", "time"]).ngroups > 10


def test_summaries_have_the_documented_shape(rows):
    kinds = levels.kind_summary(rows)
    assert {"kind", "n", "touch", "hold", "fill_hold", "mean_dist"} <= set(kinds.columns) and kinds.n.sum() == len(rows)
    table = levels.hold_table(rows, "touches", kinds=("zone",), min_n=1)
    assert set(table.columns) >= {"group", "n", "touch", "hold", "fill_hold", "rand_hold", "rand_fill_hold"}
    assert levels.hold_table(rows, "dist", [0.3, 1.5, 3.0, 5.0], min_n=1).n.sum() <= (rows.kind != "random").sum()


def test_real_minus_random_feeds_the_gate(rows):
    edge = levels.real_minus_random_by_half(rows, "reaction")
    assert set(edge) == {"per_symbol", "first_half", "second_half"} and set(edge["per_symbol"]) <= {"A", "B"}


def test_no_look_ahead_in_the_sampled_structure(rows):
    """Tampering with bars after the sampled moment plus the forward window cannot change earlier rows' features."""
    frame = _wavy_frame(seed=1)
    tampered = frame.copy()
    tampered.iloc[-300:, tampered.columns.get_loc("High")] += 50
    tampered.iloc[-300:, tampered.columns.get_loc("Low")] -= 50
    a = levels.collect_level_rows({"A": frame}, step=250, warmup=1300)
    b = levels.collect_level_rows({"A": tampered}, step=250, warmup=1300)
    cutoff = frame.index[len(frame) - 300 - levels.FWD - 60]
    a, b = a[a.time < cutoff].reset_index(drop=True), b[b.time < cutoff].reset_index(drop=True)
    assert len(a) > 20 and a[["time", "kind", "side", "dist", "touches", "touched", "held"]].equals(b[["time", "kind", "side", "dist", "touches", "touched", "held"]])
