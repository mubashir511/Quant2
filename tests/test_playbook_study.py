import numpy as np
import pandas as pd
import pytest

from tools.studies import playbook
from tools.studies.oos import GateResult, oos_gate, split_halves


def _trend_frame(n=14000, seed=1, drift=0.004, noise=0.12, start="2025-01-01"):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(drift + rng.normal(0, noise, n))
    high = close + np.abs(rng.normal(0.05, 0.03, n))
    low = close - np.abs(rng.normal(0.05, 0.03, n))
    idx = pd.date_range(start, periods=n, freq="5min")
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": rng.uniform(50, 150, n)}, index=idx)


@pytest.fixture(scope="module")
def rows():
    frames = {"UP1": _trend_frame(seed=1), "UP2": _trend_frame(seed=2, noise=0.15)}
    return playbook.collect_rows(frames, cost_by_symbol={"UP1": 0.002, "UP2": 0.02}, step=24)


def test_rows_have_every_entry_type_and_the_documented_columns(rows):
    assert {"market", "limit", "stop", "stop_struct"} <= set(rows.typ)
    for column in ("sym", "time", "regime", "adx", "adx_up", "typ", "filled", "gross", "net"):
        assert column in rows.columns
    assert set(rows.regime) <= {"ADX<20", "ADX20-30", "ADX>=30"}
    stops = rows[rows.typ == "stop_struct"]
    assert {"tight", "ahead", "vr"} <= set(stops.columns) and (stops.ahead.dropna().between(0, 1.5)).all()


def test_unfilled_opportunities_count_as_zero_and_market_always_fills(rows):
    assert (rows[~rows.filled].net == 0).all() and (rows[~rows.filled].gross == 0).all()
    assert rows[rows.typ == "market"].filled.all()
    assert (rows[rows.typ == "limit"].filled.mean() < 1.0)  # some limits never fill


def test_spread_is_charged_per_symbol_in_r(rows):
    m = rows[(rows.typ == "market") & rows.filled]
    drag = (m.gross - m.net).groupby(m.sym).mean()
    assert drag["UP2"] > drag["UP1"] * 5 > 0  # 10x the cost % -> ~10x the drag in R
    assert playbook.cheap_symbols(rows) == {"UP1"}


def test_in_a_persistent_uptrend_going_with_the_trend_pays_gross(rows):
    market = rows[(rows.typ == "market") & rows.filled]
    assert market.sign.eq(1).all()  # every aligned opportunity is a long
    assert market.gross.mean() > 0


def test_summarise_and_breakout_table_shapes(rows):
    table = playbook.summarise(rows)
    assert {"regime", "typ", "opps", "fill_rate", "gross_per_fill", "net_per_fill", "net_per_opp", "se"} <= set(table.columns)
    assert table.opps.sum() == len(rows)
    cheap_only = playbook.summarise(rows, {"UP1"})
    assert set(cheap_only.regime) <= set(table.regime) and cheap_only.opps.sum() < table.opps.sum()
    refine = playbook.breakout_table(rows, "ahead", [0, 0.5, 1.0, 1.5])
    assert set(refine.columns) == {"bin", "n", "gross", "cheap_n", "cheap_net"} and refine.n.sum() > 0
    assert playbook.breakout_table(rows, "not_a_column", [0, 1]).empty


def test_no_look_ahead_altering_late_bars_does_not_change_earlier_opportunities():
    frame = _trend_frame(n=14000, seed=5)
    tampered = frame.copy()
    rng = np.random.default_rng(99)
    tail = slice(len(frame) - 300, len(frame))
    tampered.iloc[tail, tampered.columns.get_loc("Close")] += rng.normal(0, 5, 300)
    tampered.iloc[tail, tampered.columns.get_loc("High")] += 8
    tampered.iloc[tail, tampered.columns.get_loc("Low")] -= 8
    a = playbook.collect_rows({"S": frame}, step=24)
    b = playbook.collect_rows({"S": tampered}, step=24)
    cutoff = frame.index[len(frame) - 300 - playbook.VALID - playbook.HOLD - 5]
    a, b = a[a.time < cutoff].reset_index(drop=True), b[b.time < cutoff].reset_index(drop=True)
    assert len(a) > 50 and a[["time", "typ", "filled", "gross"]].equals(b[["time", "typ", "filled", "gross"]])


def test_edge_by_symbol_and_half_feeds_the_gate(rows):
    edge = playbook.edge_by_symbol_and_half(rows, better="market", worse="limit", column="gross")
    assert set(edge) == {"per_symbol", "first_half", "second_half"} and set(edge["per_symbol"]) <= {"UP1", "UP2"}
    verdict = oos_gate(edge["first_half"], edge["second_half"], list(edge["per_symbol"].values()), min_symbol_share=0.5)
    assert isinstance(verdict, GateResult)


# ---- the gate itself ------------------------------------------------------------------------------------------

def test_gate_needs_both_halves_and_enough_symbols():
    good = oos_gate(0.05, 0.03, [0.1, 0.2, -0.05, 0.3, 0.1])
    assert good.passed and good.symbol_share == pytest.approx(0.8)
    flipped = oos_gate(0.05, -0.01, [0.1] * 5)
    assert not flipped.passed and any("second half" in r for r in flipped.reasons)
    thin = oos_gate(0.05, 0.03, [0.1, -0.1, -0.2, 0.1, -0.3])
    assert not thin.passed and any("40%" in r for r in thin.reasons)


def test_gate_fails_on_missing_input_and_supports_lower_is_better():
    assert not oos_gate(None, 0.1, [0.1]).passed
    assert not oos_gate(0.1, 0.1, None).passed
    assert not oos_gate(0.1, 0.1, []).passed
    assert oos_gate(-0.2, -0.1, [-0.1, -0.3], positive=False).passed
    assert not oos_gate(0.02, 0.03, [0.1, 0.2], min_effect=0.05).passed


def test_split_halves_uses_the_median_time():
    pairs = [(i, float(i)) for i in range(8)]
    assert split_halves(pairs) == (1.5, 5.5)
    assert split_halves(pairs[:3]) == (None, None)
