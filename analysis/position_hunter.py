"""Position Hunter — the deterministic shortlist that starts the Mega Session's search for trades.

Why (2026-09-25 position-hunting review): the last session excluded 16 of 20 symbols, largely on pooled
backtest/ATR-filter grounds that, when measured, carry almost no information (a random-entry simulation
scores the same "27% win / slightly negative R"). Meanwhile trading WITH the closed higher-timeframe trend
(H4 and D1 close vs SMA20) is the books' oldest rule ("trade with the intermediate trend", Murphy; Schwager) and it
measured small but positive on the deep history: 2007-2026, 667,701 H1 bars, 20 symbols - aligned +0.021R gross per
trade vs -0.009R counter-trend (14 of 20 symbols). It is REGIME-DEPENDENT, though: positive in every year since 2019
(+0.05R in 2020/2024/2026) but reversed in 2008-2015 (aligned -0.03..-0.18R) - Lo's adaptive-markets warning made
concrete - so it is a ranking factor to monitor monthly, never a hard filter. And the biggest self-inflicted leak is cost: the
broker's round-trip spread is 0.19R at a 2x M5-ATR stop and above 1R on WHEAT/COCOA.

So the hunter does two things only, both objective:

1. RANK every (symbol, side) — aligned-with-D1/H4 first, then M5 agreement, a SUPPORTED edge verdict, a
   structurally-clean zone, and low cost drag. Ranking is advice.
2. HARD-VETO only what is objectively untradable right now, each with a stable id the model must cite to
   leave a shortlisted candidate out (everything else stays the model's judgement, per the user's decision
   "hard vetoes only, everything else advisory"):

   V1 cost drag above config.COST_DRAG_VETO_R at the tightest valid stop
   V2 no real structure at all (no S/R band and no swing leg)
   V3 High-impact event blackout window for the symbol's currencies
   V4 minimum-viable size exceeds the risk budget the account can still spend
   V5 correlated with a higher-ranked selection already carrying the same directional exposure
   V6 backtest CONTRADICTED vs a random-entry baseline (analysis.edge_stats) with nothing supporting it
   V7 market closed, or inside the final config.NO_NEW_ORDER_MINUTES_BEFORE_CLOSE minutes of the session

   (V8 - "stop or target already dead" - needs a drafted entry, so it lives in ai.live_recheck.)

Pure: no I/O, no MT5. The caller (ai.ftmo_suggest.build_hunt_facts) reads real analysis objects into
`HuntFacts`; nothing here invents a number, and a missing input never creates a veto or a claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import config

_DIRECTION = {"buy": "up", "sell": "down"}

VETO_TEXT = {
    "V1": "round-trip cost above the objective limit at the stop the playbook would use (measured: net negative above ~0.10R at every target multiple)",
    "V2": "no real structure (no support/resistance band and no swing leg) to anchor a stop or target",
    "V3": "High-impact economic event blackout window",
    "V4": "minimum viable size exceeds the risk budget the account can still spend",
    "V7": "market closed / inside the final minutes before the session close",
}


@dataclass
class HuntFacts:
    """Everything the hunter needs about ONE symbol, already measured by the caller."""

    symbol: str
    d1_dir: str | None = None  # "up" / "down" / "flat" / None (context)
    h4_dir: str | None = None  # same (context)
    m5_dir: str | None = None  # M5 trend AND regime agree, else None (decision tier)
    cost_drag_r: float | None = None  # round-trip cost / tightest valid stop
    has_structure: bool | None = None  # None = unknown (never vetoes)
    zone_rr: dict[str, float | None] = field(default_factory=dict)  # side -> nearest-target R:R of the M5 zone
    edge_verdicts: dict[str, dict[str, str]] = field(default_factory=dict)  # side -> {"rsi": verdict, "sr": verdict}
    blackout_reason: str | None = None
    market_open: bool | None = None  # None = unknown
    minutes_to_close: float | None = None
    min_viable_pct: float | None = None
    # side -> analysis.playbook.PlaybookAdvice (the with-the-trend breakout proposal); {} when the playbook is off
    playbook: dict = field(default_factory=dict)


@dataclass
class HuntCandidate:
    symbol: str
    side: str  # "buy" / "sell"
    tier: str  # "A" aligned D1+H4, "B" one agrees and the other is flat/unknown, "C" counter-trend or mixed
    score: float
    m5_agrees: bool | None  # None = no clear M5 direction
    vetoes: list[tuple[str, str]] = field(default_factory=list)  # (id, specific text)
    notes: list[str] = field(default_factory=list)
    drag: float | None = None  # cost drag in R at the tightest valid stop, for compact listings
    advice: object | None = None  # analysis.playbook.PlaybookAdvice for this side (None when the playbook is off)


@dataclass
class HuntResult:
    shortlist: list[HuntCandidate]
    vetoed: list[HuntCandidate]
    no_read: list[str]  # symbols with no directional read at all (no D1/H4/M5 direction)
    overflow: list[HuntCandidate] = field(default_factory=list)  # eligible (no veto) but ranked below the cap


def _tier(side: str, d1: str | None, h4: str | None) -> str:
    """A = D1 and H4 both with the side; B = one with it, the other flat/unknown; M = neither has a direction
    (M5-only); C = at least one AGAINST it (counter-trend or mixed - measured ~0R, the same as random)."""
    want = _DIRECTION[side]
    against = "down" if want == "up" else "up"
    if against in (d1, h4):
        return "C"
    agree = [d == want for d in (d1, h4)]
    if all(agree):
        return "A"
    if any(agree):
        return "B"
    return "M"


def _edge_summary(verdicts: dict[str, str]) -> str | None:
    values = set(verdicts.values())
    if "supported" in values and "contradicted" in values:
        return "mixed"
    if "supported" in values:
        return "supported"
    if "contradicted" in values:
        return "contradicted"
    return None


def _score(f: HuntFacts, side: str, tier: str, m5_agrees: bool | None) -> float:
    # Trend alignment is a NUDGE, sized to what was measured (deep M5, 30,790 / 28,486 breakout fills, both halves): a
    # breakout WITH the closed D1+H4 trend grossed +0.074R vs +0.038R against it - about +0.04R, and counter-trend breakouts
    # were still gross-positive. It used to dominate the ranking (A 3.0 vs C 0.0) and push good counter-trend breakouts out.
    # The backtest verdict no longer moves the score at all: ~90% of reads are NO-INFORMATION and the rest is noise-level.
    score = {"A": 1.0, "B": 0.6, "M": 0.5, "C": 0.3}[tier]
    if m5_agrees:
        score += 0.5
    elif m5_agrees is False:
        score -= 0.25
    advice = f.playbook.get(side)
    if advice is not None and getattr(advice, "status", None) == "ACTIVE":
        score += 0.5  # the breakout trigger is live right now on this side - the M5 event the playbook waits for
    if f.zone_rr.get(side):
        score += 0.5
    if f.cost_drag_r is not None:
        score -= min(f.cost_drag_r, 1.0)  # cheaper first; capped so one outlier never dominates
    return score


def _vetoes(f: HuntFacts, side: str, headroom_budget_pct: float | None, tier: str | None = None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if f.cost_drag_r is not None and f.cost_drag_r > config.COST_DRAG_VETO_R:
        out.append(("V1", f"round-trip cost is {f.cost_drag_r:.2f}R at the stop the playbook would use (limit {config.COST_DRAG_VETO_R:g}R)"))
    if f.has_structure is False:
        out.append(("V2", VETO_TEXT["V2"]))
    if f.blackout_reason:
        out.append(("V3", f.blackout_reason))
    if (
        headroom_budget_pct is not None
        and f.min_viable_pct is not None
        and f.min_viable_pct > headroom_budget_pct
    ):
        out.append(("V4", f"needs at least {f.min_viable_pct:.2f}% risk to clear the broker minimum lot; only {max(headroom_budget_pct, 0):.2f}% may be risked"))
    if f.market_open is False:
        out.append(("V7", "market is closed right now"))
    elif (
        f.minutes_to_close is not None
        and config.NO_NEW_ORDER_MINUTES_BEFORE_CLOSE > 0
        and f.minutes_to_close < config.NO_NEW_ORDER_MINUTES_BEFORE_CLOSE
    ):
        out.append(("V7", f"only ~{max(f.minutes_to_close, 0):.0f} min to the session close (no new resting orders in the last {config.NO_NEW_ORDER_MINUTES_BEFORE_CLOSE:g})"))
    return out


def _correlated_same_exposure(
    a: tuple[str, str], b: tuple[str, str], pairs: list[tuple[str, str, float]]
) -> float | None:
    """The correlation r when (symbol, side) a and b carry the SAME directional exposure (positively
    correlated and the same side, or negatively correlated and opposite sides), else None."""
    for sym_a, sym_b, r in pairs:
        if {sym_a, sym_b} != {a[0], b[0]}:
            continue
        same_exposure = (r > 0) == (a[1] == b[1])
        return r if same_exposure else None
    return None


def hunt(
    facts: list[HuntFacts],
    correlation_pairs: list[tuple[str, str, float]] | None = None,
    risk_budget_pct: float | None = None,
    max_candidates: int | None = None,
) -> HuntResult:
    """Rank every symbol's best side, apply the objective vetoes, then greedily keep the top
    `max_candidates` (default config.POSITION_HUNT_MAX_CANDIDATES) while skipping candidates correlated
    with a higher-ranked kept one (V5)."""
    max_candidates = max_candidates or config.POSITION_HUNT_MAX_CANDIDATES
    pairs = correlation_pairs or []
    candidates: list[HuntCandidate] = []
    no_read: list[str] = []
    for f in facts:
        directions = [d for d in (f.d1_dir, f.h4_dir, f.m5_dir) if d in ("up", "down")]
        if not directions:
            no_read.append(f.symbol)
            continue
        best: HuntCandidate | None = None
        for side in ("buy", "sell"):
            tier = _tier(side, f.d1_dir, f.h4_dir)
            m5_agrees = None if f.m5_dir is None else f.m5_dir == _DIRECTION[side]
            cand = HuntCandidate(f.symbol, side, tier, _score(f, side, tier, m5_agrees), m5_agrees)
            if best is None or cand.score > best.score:
                best = cand
        assert best is not None
        best.drag = f.cost_drag_r
        best.advice = f.playbook.get(best.side)
        # A side with nothing pointing at it (tier C and M5 silent or against) is not a candidate - UNLESS the breakout
        # playbook is ACTIVE for it: a range break through the trigger is the M5 event itself, and counter-trend breakouts
        # measured gross-positive (+0.038R), so the HTF label alone is no reason to drop it.
        if best.tier == "C" and not best.m5_agrees and not (best.advice is not None and getattr(best.advice, "status", None) == "ACTIVE"):
            no_read.append(f.symbol)
            continue
        best.vetoes = _vetoes(f, best.side, risk_budget_pct, best.tier)
        best.notes = _notes(f, best)
        candidates.append(best)

    candidates.sort(key=lambda c: c.score, reverse=True)
    shortlist: list[HuntCandidate] = []
    vetoed: list[HuntCandidate] = []
    overflow: list[HuntCandidate] = []
    for cand in candidates:
        if cand.vetoes:
            vetoed.append(cand)
            continue
        clash = next(
            (
                (kept, r)
                for kept in shortlist
                if (r := _correlated_same_exposure((cand.symbol, cand.side), (kept.symbol, kept.side), pairs)) is not None
            ),
            None,
        )
        if clash is not None:
            kept, r = clash
            # Correlation is INFORMATION, never a reason to leave a good setup out: the aggregate-heat ceiling already sums
            # every stop as if all were hit together (the worst case), so correlated positions cannot breach it.
            cand.notes.append(
                f"correlated (r={r:+.2f}) with shortlisted {kept.symbol} {kept.side} - informational only, the heat ceiling already counts both"
            )
        if len(shortlist) < max_candidates:
            shortlist.append(cand)
        else:
            overflow.append(cand)  # beyond the cap is not a veto: still eligible, just lower-ranked
    return HuntResult(shortlist=shortlist, vetoed=vetoed, no_read=no_read, overflow=overflow)


def _notes(f: HuntFacts, c: HuntCandidate) -> list[str]:
    notes: list[str] = []
    want = _DIRECTION[c.side]
    notes.append(
        {
            "A": "WITH the D1 and H4 trend (deep history 2007-2026: +0.021R vs -0.009R counter-trend gross; positive every year since 2019, reversed 2008-2015)",
            "B": f"partly with the higher-timeframe trend (D1 {f.d1_dir or 'n/a'}, H4 {f.h4_dir or 'n/a'})",
            "M": "no D1/H4 direction (flat/unknown): an M5-only setup, neither tailwind nor headwind",
            "C": f"AGAINST the higher-timeframe trend (D1 {f.d1_dir or 'n/a'}, H4 {f.h4_dir or 'n/a'}): measured breakouts against it grossed +0.038R vs +0.074R with it - tradable, no extra bar required",
        }[c.tier]
    )
    if c.m5_agrees is True:
        notes.append(f"M5 trend+regime agree ({want})")
    elif c.m5_agrees is False:
        notes.append(f"M5 reads {f.m5_dir} - against this side")
    else:
        notes.append("M5 has no clear direction")
    edge = _edge_summary(f.edge_verdicts.get(c.side, {}))
    if edge:
        notes.append(f"M5 backtest vs random entries: {edge.upper()} (context for sizing only - never a veto)")
    else:
        notes.append("M5 backtest: no information (not a reason to exclude)")
    rr = f.zone_rr.get(c.side)
    notes.append(f"M5 zone nearest target {rr:.1f}R" if rr else "no M5 zone target clears 2R (partial+trail or a nearer entry)")
    if f.cost_drag_r is not None:
        notes.append(f"cost drag {f.cost_drag_r:.2f}R at the playbook's structure stop")
    return notes


def format_position_hunt(result: HuntResult) -> str:
    """The prompt block: shortlist first (ranked), then the vetoed list with veto ids. The model must give
    a veto id (or a specific, checkable reason tied to printed numbers) for any shortlisted candidate it
    leaves out."""
    lines = [
        "POSITION HUNT (deterministic, Python - ranked from this run's own data; the ORDER is advice, the "
        "VETOES are objective and the only hard exclusions). Your Asset-Class Outlook must address EVERY "
        "shortlisted candidate individually: include it (even small) or give ONE specific, checkable reason "
        "(a veto id that applies to it now, or a printed fact) - 'weak', 'quiet' or silence is not a reason:"
    ]
    if not result.shortlist:
        lines.append("  shortlist: none - every directional candidate is vetoed below or has no directional read.")
    for i, c in enumerate(result.shortlist, 1):
        lines.append(f"  {i}. {c.symbol} {c.side.upper()} [tier {c.tier}] - " + "; ".join(c.notes))
        if c.advice is not None:
            caveat = "" if c.tier == "A" else " (measured with the trend: gross +0.074R; against it or mixed: +0.017..+0.038R - smaller, still positive)"
            lines.append(f"       {c.advice.text()}{caveat}")
    if result.vetoed:
        lines.append("  Vetoed (hard, objective - cite the id to leave one out; you may still trade one only by addressing the veto):")
        for c in result.vetoed:
            lines.append(f"    - {c.symbol} {c.side.upper()} [tier {c.tier}]: " + "; ".join(f"{vid}: {text}" for vid, text in c.vetoes))
    if result.overflow:
        lines.append(
            "  Also eligible (no veto, ranked below the shortlist - still yours to trade if your analysis prefers): "
            + "; ".join(f"{c.symbol} {c.side.upper()} [tier {c.tier}, cost drag "
                        f"{'n/a' if c.drag is None else format(c.drag, '.2f') + 'R'}]" for c in result.overflow)
        )
    if result.no_read:
        lines.append("  No directional read (D1, H4 and M5 all flat/unclear, or only a counter-trend M5-silent side): " + ", ".join(result.no_read))
    lines.append(
        "  Veto ids: " + "; ".join(f"{vid} {text}" for vid, text in VETO_TEXT.items()) + "."
    )
    return "\n".join(lines)
