"""Re-run the position-hunting studies on the account's real broker data and write a dated report.

    .venv\\Scripts\\python.exe -m tools.studies.run_all            # writes records/studies/YYYY-MM-DD.md

Meant to be run monthly (Lo's Adaptive Markets caveat: a rule that earned its place last quarter can decay).
A rule whose effect flips sign or shrinks to noise should be down-weighted - the report prints the numbers next
to the ones the rules were built on so that is visible at a glance, and section 4 runs the out-of-sample gate.
Uses the deep price cache (data/price_cache.py, refreshed first) so every statistic rests on up to 120k M5 / 60k H1
bars per symbol. Needs the MT5 terminal (like every other script that reads broker history); it never sends an order."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from data.mt5_source import connect, get_market_watch, get_trade_economics  # noqa: E402
from data.price_cache import deep_history  # noqa: E402
from tools.studies import playbook  # noqa: E402
from tools.studies.oos import oos_gate  # noqa: E402
from tools.studies.studies import cost_drag_by_stop_study, exit_rule_study, htf_alignment_study  # noqa: E402

# What the rules were built on (2026-09-25), printed beside today's numbers.
BASELINE = {
    "alignment": "2007-2026 (667,701 H1 bars): aligned +0.021R vs counter -0.009R vs mixed -0.004R gross; aligned > counter on 14/20 symbols; positive every year since 2019, reversed 2008-2015 (2017-2026 subset: +0.050R vs +0.003R)",
    "exit": "net mean R: base -0.174, be@1R -0.163, partial@1R -0.155, trail@1R -0.148 (trail better than base on 17/20 symbols)",
    "cost": "mean round-trip cost 0.19R at a 2x M5-ATR stop, 0.09R at 4x, 0.06R at 6x",
    "playbook": "gross R per fill: limit -0.03..-0.05, market +0.02..+0.03, breakout stop +0.04..+0.07 (best at ADX>=30); structure-stop breakout: gross +0.086, cheaper-half net +0.077",
}
EXIT_STUDY_BARS = 40000  # the exit study is 4 rules x every entry: cap the depth it walks


def _fmt(x, digits=3):
    return "n/a" if x is None or x != x else f"{x:+.{digits}f}"


def main() -> Path:
    connect(login=config.FTMO_MT5_LOGIN, password=config.FTMO_MT5_PASSWORD, server=config.FTMO_MT5_SERVER)
    assets = get_market_watch()
    m5, h1, cost = {}, {}, {}
    for asset in assets:
        m5_frame = deep_history(asset.symbol, "M5", refresh=True).dropna(subset=["High", "Low", "Close"])
        h1_frame = deep_history(asset.symbol, "H1", refresh=True).dropna(subset=["High", "Low", "Close"])
        economics = get_trade_economics(asset.symbol)
        if len(m5_frame) > 1000:
            m5[asset.symbol] = m5_frame
        if len(h1_frame) > 3000:
            h1[asset.symbol] = h1_frame
        cost[asset.symbol] = economics.spread_pct_of_price if economics else 0.0

    from tools.studies.symbol_cards import rebuild as rebuild_symbol_cards

    rebuild_symbol_cards(m5)  # the Symbol Behaviour Cards ride on the same refreshed deep cache

    alignment = htf_alignment_study(h1, buckets=("4h", "1D"), hold=24, step=6, warmup=600)
    exits = exit_rule_study({s: f.tail(EXIT_STUDY_BARS) for s, f in m5.items()}, cost_by_symbol=cost)
    drag = cost_drag_by_stop_study(m5, cost)
    rows = playbook.collect_rows(m5, cost_by_symbol=cost, step=12)
    cheap = playbook.cheap_symbols(rows)

    lines = [f"# Position-hunting studies - {date.today().isoformat()}", "",
             f"Real broker data; deep cache ({sum(len(f) for f in m5.values()):,} M5 bars over {len(m5)} symbols, "
             f"{sum(len(f) for f in h1.values()):,} H1 bars); random entries; higher-timeframe flags from CLOSED bars only.", ""]
    lines += ["## 1. Higher-timeframe alignment (H1 entries, closed H4 + D1 trend)", f"Built on: {BASELINE['alignment']}", ""]
    for group, stats in alignment["pooled"].items():
        lines.append(f"- {group}: n={stats['n']}, gross mean {_fmt(stats['mean_r'])}R (se {stats['se']:.3f})")
    lines.append(f"- aligned beats counter on {alignment['symbols_aligned_beats_counter']} of {alignment['symbols_tested']} symbols")
    for year, row in alignment["by_year"].items():
        lines.append(f"  - {year}: aligned {_fmt(row['aligned'])}R vs counter {_fmt(row['counter'])}R (n aligned {row['n_aligned']})")
    lines += ["", f"## 2. Exit rules (M5, last {EXIT_STUDY_BARS:,} bars, stop 2 ATR, target 2R, hold 96 bars, spread charged)", f"Built on: {BASELINE['exit']}", ""]
    for rule, stats in exits["rules"].items():
        beats = exits["beats_base_on_symbols"].get(rule)
        suffix = f", better than base on {beats}/{exits['symbols']} symbols" if beats is not None else ""
        lines.append(f"- {rule}: net mean {_fmt(stats['mean_net_r'])}R (se {stats['se']:.3f}, n={stats['n']}){suffix}")
    lines += ["", "## 3. Cost drag by stop size", f"Built on: {BASELINE['cost']}", ""]
    for k, value in drag["pooled"].items():
        lines.append(f"- stop {k:g}x M5 ATR: mean round-trip cost {_fmt(value, 2)}R")
    worst = sorted(drag["per_symbol"].items(), key=lambda kv: kv[1][2.0], reverse=True)[:5]
    lines.append("- costliest at 2x: " + ", ".join(f"{s} {v[2.0]:.2f}R" for s, v in worst))

    lines += ["", "## 4. Entry playbook (with the closed H4+D1 trend; stop 2 ATR / target 2R; deep M5)", f"Built on: {BASELINE['playbook']}", ""]
    table = playbook.summarise(rows)
    lines.append("All symbols (net R per opportunity; unfilled = 0):")
    for r in table.itertuples():
        lines.append(f"- {r.regime:9s} {r.typ:12s} opps {r.opps:6d} fill {r.fill_rate:5.1%} gross/fill {_fmt(r.gross_per_fill)} net/fill {_fmt(r.net_per_fill)} net/opp {_fmt(r.net_per_opp)} (se {r.se:.3f})")
    lines.append(f"\nCheaper half of symbols ({', '.join(sorted(cheap))}):")
    for r in playbook.summarise(rows, cheap).itertuples():
        lines.append(f"- {r.regime:9s} {r.typ:12s} opps {r.opps:6d} fill {r.fill_rate:5.1%} net/fill {_fmt(r.net_per_fill)} net/opp {_fmt(r.net_per_opp)} (se {r.se:.3f})")
    lines += ["", "Breakout structure-stop trades, cheaper half (bin: n, gross, cheap net):"]
    for feature, bins in (("vr", [0, 0.7, 1.0, 1.5, 3, 1e9]), ("adx", [20, 25, 30, 40, 100]), ("ahead", [0, 0.5, 1.0, 1.5]), ("tight", [0, 3, 5, 8, 14, 999])):
        t = playbook.breakout_table(rows[rows.adx >= 20], feature, bins, cheap=cheap)
        lines.append(f"- {feature}: " + "; ".join(f"{r.bin}: n={r.n}, gross {_fmt(r.gross)}, net {_fmt(r.cheap_net)}" for r in t.itertuples()))

    lines += ["", "### Out-of-sample gates (both time halves and >= 70% of symbols must agree)"]
    for label, better, worse, column in (
        ("breakout (structure stop) vs limit, net R per opportunity", "stop_struct", "limit", "net"),
        ("breakout (structure stop) vs market, net R per opportunity", "stop_struct", "market", "net"),
        ("structure stop vs fixed 2 ATR stop on breakouts, net R per opportunity", "stop_struct", "stop", "net"),
        ("market vs limit, gross R per opportunity", "market", "limit", "gross"),
    ):
        edge = playbook.edge_by_symbol_and_half(rows, better, worse, column)
        verdict = oos_gate(edge["first_half"], edge["second_half"], list(edge["per_symbol"].values()))
        share = "n/a" if verdict.symbol_share is None else format(verdict.symbol_share, ".0%")
        why = "; " + "; ".join(verdict.reasons) if verdict.reasons else ""
        lines.append(f"- {label}: {'PASS' if verdict.passed else 'FAIL'} (halves {_fmt(verdict.first_half)} / {_fmt(verdict.second_half)}; symbols agreeing {share}{why})")

    out_dir = ROOT / "records" / "studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{date.today().isoformat()}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {path}")
    return path


if __name__ == "__main__":
    main()
