from dataclasses import dataclass


@dataclass
class BookPrinciple:
    principle: str
    author: str
    source: str
    rationale: str


# Reintroduced from this project's earlier NotebookLM-driven book-rules
# pipeline (previously encoded as hard deterministic rules in a since-
# rebuilt risk/rebalance.py — see memory: project_book_rules_pipeline).
# Per explicit request, these come back here as *advisory* reasoning
# context only: nothing in risk/rebalance.py enforces any of this, and the
# AI is free to reason its way to a different conclusion — this feeds the
# already-discretionary Portfolio Suggestion feature, not the strict
# rebalance narrator.
#
# 2026-08-08: expanded by querying NotebookLM directly against the user's
# actual owned books (Bulkowski's Encyclopedia of Chart Patterns, Schwager's
# Getting Started in Technical Analysis, O'Neil's How to Make Money in
# Stocks, Nison's Japanese Candlestick Charting Techniques, Pring's
# Technical Analysis Explained, Rockefeller's Technical Analysis For
# Dummies, Murphy's Technical Analysis of the Financial Markets) rather
# than from secondhand summaries — see memory: feedback_ai_reasoning_scope_
# carveout. Deliberately excluded from this set (same triage logic as the
# original pipeline): chart-pattern/breakout signal-confirmation rules
# (busted-pattern reversals, tick-level stops, breakout-penetration
# filters, industry-group confirmation) — these are stock-picking signal
# logic for a future `signals/rules.py`, not portfolio-mix advisory;
# options/OTC allocation caps — not applicable, PMEX here trades futures
# only; pyramid/scale-in sizing rules — not applicable, this feature
# produces a one-shot % allocation mix, not a sequence of scaled-in
# entries over time; a few near-duplicate rules across authors — kept only
# the best-reasoned version of each to avoid prompt bloat.
BOOK_PRINCIPLES: list[BookPrinciple] = [
    BookPrinciple(
        principle="Require at least a 2:1 reward-to-risk ratio before sizing a position.",
        author="Thomas Bulkowski / James Rockefeller",
        source="chart-pattern and position-sizing literature",
        rationale=(
            "At a 2:1 payoff, a strategy can stay profitable even below a "
            "50% win rate — it shifts the burden from being right most of "
            "the time to sizing winners and losers asymmetrically."
        ),
    ),
    BookPrinciple(
        principle=(
            "Cap total portfolio 'heat' — the aggregate capital at risk "
            "across all open positions if every stop were hit — to "
            "roughly 25-35% of total equity, not sized position-by-"
            "position in isolation."
        ),
        author="Jack Schwager",
        source="Getting Started in Technical Analysis",
        rationale=(
            "Summing risk across every open position shows total exposure "
            "and whether capital remains for new trades; without this "
            "cap, individually reasonable position sizes can still "
            "combine into a drawdown that 'steamrollers' into disaster."
        ),
    ),
    BookPrinciple(
        principle=(
            "Limit any single market to roughly 10-15% of total equity, "
            "and any single correlated market/sector group (e.g. precious "
            "metals, energy) to roughly 20-25% — not just avoiding "
            "concentration in one instrument."
        ),
        author="John Murphy",
        source="Technical Analysis of the Financial Markets",
        rationale=(
            "Instruments in the same group (e.g. precious metals, energy) "
            "often move on the same underlying driver; diversification "
            "measured only by instrument count can still leave a "
            "portfolio concentrated in one macro risk."
        ),
    ),
    BookPrinciple(
        principle=(
            "Tighten the stop-loss as a position builds a solid gain (e.g. "
            "around +15%), rather than leaving the original stop in place "
            "indefinitely."
        ),
        author="William O'Neil",
        source="How to Make Money in Stocks",
        rationale=(
            "In the spirit of O'Neil's capital-preservation approach: "
            "protecting a gain already earned matters more than giving a "
            "winning position unlimited room, since a meaningful reversal "
            "can erase much of the edge already won."
        ),
    ),
    BookPrinciple(
        principle=(
            "Consider taking profit around a 20-25% gain, with an "
            "exception — roughly an 8-week grace window — for a "
            "genuinely strong, fast-moving trend to run further first."
        ),
        author="William O'Neil",
        source="How to Make Money in Stocks (CANSLIM method)",
        rationale=(
            "O'Neil's own study of historical big winners found gains "
            "tend to concentrate in a limited window; a flat profit-take "
            "target with a grace-period exception balances locking in "
            "solid gains against cutting a genuinely strong trend short "
            "too early."
        ),
    ),
    BookPrinciple(
        principle=(
            "Risk no more than 1-2% of total equity on any single trade, "
            "and never more than an absolute 3% — reconsider trading "
            "altogether if regularly forced to risk more than that."
        ),
        author="Jack Schwager",
        source="Getting Started in Technical Analysis",
        rationale=(
            "Money management matters more than the trading method "
            "itself; keeping any one trade's downside small prevents a "
            "string of ordinary selection mistakes from destroying the "
            "account."
        ),
    ),
    BookPrinciple(
        principle=(
            "Size positions by volatility, not by a flat lot count — "
            "trade fewer units in more volatile markets so the dollar-"
            "risk per position stays roughly consistent."
        ),
        author="Jack Schwager",
        source="Getting Started in Technical Analysis",
        rationale=(
            "Without this adjustment, an equally-sized position in a calm "
            "market and a volatile one carries very different real risk, "
            "even though the position size looks the same on paper."
        ),
    ),
    BookPrinciple(
        principle=(
            "Reduce leverage in proportion to equity drawdown (e.g. cut "
            "exposure by roughly 20% after a 20% equity decline), and cut "
            "position size sharply — or stop trading entirely — during a "
            "losing streak."
        ),
        author="Jack Schwager",
        source="Getting Started in Technical Analysis",
        rationale=(
            "This keeps a losing phase from compounding into a "
            "disastrous retracement, rather than trading a shrinking "
            "account at the same size as a full one."
        ),
    ),
    BookPrinciple(
        principle=(
            "If a position captures a large share of its expected move "
            "unusually fast (e.g. 50-60% of the target within a week), "
            "take partial profits immediately rather than waiting for the "
            "full target."
        ),
        author="Jack Schwager",
        source="Getting Started in Technical Analysis",
        rationale=(
            "Gains realized that quickly often invite a sharp retracement "
            "and 'nervous liquidation' on the first pullback; banking "
            "part of the gain lets the rest of the position ride more "
            "comfortably."
        ),
    ),
    BookPrinciple(
        principle=(
            "Cap total invested funds at roughly 50% of account equity, "
            "holding the remainder in reserve — size the cash reserve as "
            "a deliberate floor, not just whatever percentage is left "
            "over."
        ),
        author="John Murphy",
        source="Technical Analysis of the Financial Markets",
        rationale=(
            "An uncommitted reserve absorbs periods of adversity and "
            "drawdown that a fully-invested account has no buffer "
            "against."
        ),
    ),
    BookPrinciple(
        principle=(
            "Reason through the intermarket chain when assessing cross-"
            "asset risk: a stronger dollar typically pressures "
            "commodities; falling commodities ease inflation pressure and "
            "support bonds; rising bond prices (falling yields) are "
            "typically supportive for equities — and the chain runs in "
            "reverse too."
        ),
        author="John Murphy",
        source="Technical Analysis of the Financial Markets",
        rationale=(
            "These four asset classes don't move independently — a shock "
            "in one has a fairly predictable directional ripple through "
            "the others, which matters directly when the mix spans "
            "currencies, commodities, and equity indices at once."
        ),
    ),
    BookPrinciple(
        principle=(
            "A strengthening dollar tends to hurt large multinational/"
            "large-cap-heavy equities more than smaller, domestically-"
            "oriented ones, since it makes their goods more expensive "
            "abroad — weigh this when choosing which equity index, if "
            "any, to include."
        ),
        author="John Murphy",
        source="Technical Analysis of the Financial Markets",
        rationale=(
            "Large-cap indices carry disproportionate exposure to "
            "multinational exporters, so a DXY move isn't a neutral macro "
            "fact for them — it's a direct headwind or tailwind worth "
            "naming explicitly."
        ),
    ),
    BookPrinciple(
        principle=(
            "Prefer trades offering at least a 3-to-1 profit-to-loss "
            "ratio, stricter than a bare 2:1 floor."
        ),
        author="John Murphy",
        source="Technical Analysis of the Financial Markets",
        rationale=(
            "Since even skilled traders often win well under half their "
            "trades, winners need to be meaningfully larger than losers "
            "on average to stay profitable over time."
        ),
    ),
    BookPrinciple(
        principle=(
            "Cut the loss on any position at an absolute maximum of 7-8% "
            "below the entry price, with no exceptions."
        ),
        author="William O'Neil",
        source="How to Make Money in Stocks",
        rationale=(
            "Every catastrophic loss starts out as a small one; treating "
            "7-8% as a hard ceiling, not a guideline, is what keeps a bad "
            "trade from ever becoming an account-threatening one."
        ),
    ),
    BookPrinciple(
        principle=(
            "Favor a small number of well-researched positions, scaled to "
            "account size, over broad diversification for its own sake."
        ),
        author="William O'Neil",
        source="How to Make Money in Stocks",
        rationale=(
            "Broad diversification can act as a 'hedge for lack of "
            "knowledge' — concentrating in fewer, better-understood "
            "positions allows for more meaningful position sizes and "
            "closer tracking of each one."
        ),
    ),
    BookPrinciple(
        principle=(
            "Set a protective stop no closer than roughly 1.5x the recent "
            "average daily trading range (ATR), not an arbitrary fixed "
            "distance."
        ),
        author="Thomas Bulkowski",
        source="Encyclopedia of Chart Patterns",
        rationale=(
            "A stop placed without regard to an instrument's normal "
            "daily movement risks being triggered by ordinary noise "
            "rather than an actual change in the trade's thesis."
        ),
    ),
    BookPrinciple(
        principle=(
            "Expect bonds, stocks, and commodities to lead and lag each "
            "other in a fairly consistent rotation across the business "
            "cycle (bonds typically turn first, then stocks, then "
            "commodities, both at bottoms and at tops) rather than "
            "treating their moves as unrelated."
        ),
        author="Martin Pring",
        source="Technical Analysis Explained",
        rationale=(
            "Recognizing which stage of this rotation the current macro "
            "data suggests helps frame whether a given asset class move "
            "looks early, on-time, or already-late relative to the "
            "others."
        ),
    ),
    BookPrinciple(
        principle=(
            "Favor holding equity index exposure only when both the S&P "
            "500 and the (inverted) 10-year Treasury yield are above "
            "their own 12-month moving averages — treat this as a "
            "condition for including equity exposure at all, not just an "
            "input to size it."
        ),
        author="Martin Pring",
        source="Technical Analysis Explained",
        rationale=(
            "This filter is meant to let a trader participate in genuine "
            "bull markets while sidestepping the higher volatility and "
            "losses that tend to come with a rising-rate environment."
        ),
    ),
    BookPrinciple(
        principle=(
            "Treat an RSI reading above 70 or below 30 as a signal a move "
            "may be overstretched and due for a pause — especially when "
            "it coincides with price at a known support or resistance "
            "level — but wait for RSI to actually cross back through that "
            "70/30 line, not just touch it, before treating a reversal as "
            "confirmed."
        ),
        author="Martin Pring / John Murphy",
        source="Technical Analysis Explained; Technical Analysis of the Financial Markets",
        rationale=(
            "An extreme reading alone only shows a move is statistically "
            "overstretched, not that it has already turned; a strong "
            "trend can keep an oscillator pinned at an extreme for a long "
            "time, so acting on the cross-back rather than the extreme "
            "itself avoids exiting or fading a move too early."
        ),
    ),
    BookPrinciple(
        principle=(
            "Require a genuine increase in volume to treat a breakout "
            "from a range or pattern as valid — if the move isn't "
            "accompanied by a real pickup in participation, question the "
            "pattern rather than trust it."
        ),
        author="John Murphy",
        source="Technical Analysis of the Financial Markets",
        rationale=(
            "Rising volume in the breakout direction reflects real "
            "conviction; a breakout on unchanged or falling volume more "
            "often reflects an absence of sellers than genuine new "
            "demand, making it more prone to failing or whipsawing back."
        ),
    ),
    BookPrinciple(
        principle=(
            "Weigh a breakout more heavily the more its volume exceeds "
            "the instrument's own recent average — breakouts on "
            "above-average volume tend to travel farther than those on "
            "light volume."
        ),
        author="Thomas Bulkowski",
        source="Encyclopedia of Chart Patterns",
        rationale=(
            "This isn't just a qualitative preference — Bulkowski's own "
            "pattern statistics show breakout-day volume above the "
            "recent average is associated with measurably stronger "
            "post-breakout performance."
        ),
    ),
    BookPrinciple(
        principle=(
            "Treat an unusually narrow, low-volatility trading range "
            "(shrinking ATR, drying-up volume) as a coiled-spring setup "
            "that often precedes a significant move once it resolves, "
            "not as a sign the instrument has simply gone quiet for good."
        ),
        author="Martin Pring",
        source="Technical Analysis Explained",
        rationale=(
            "A narrow range with light volume reflects a temporary "
            "stalemate between buyers and sellers; when that balance "
            "eventually breaks, the resulting move tends to be larger "
            "precisely because it built up over the quiet period rather "
            "than releasing gradually."
        ),
    ),
    BookPrinciple(
        principle=(
            "Weigh a wider recent trading range (higher ATR or volatility "
            "relative to price) as suggesting more potential energy "
            "behind whatever move follows it, once the range resolves in "
            "either direction."
        ),
        author="John Murphy / Thomas Bulkowski",
        source="Technical Analysis of the Financial Markets; Encyclopedia of Chart Patterns",
        rationale=(
            "Both authors independently find that the size of the range "
            "a price builds during consolidation scales with the size of "
            "the move that follows it — a bigger base reflects more "
            "accumulated volatility and energy, not just noise."
        ),
    ),
]


def format_book_wisdom() -> str:
    lines = ["Investing-literature principles (advisory context, not enforced rules):"]
    for p in BOOK_PRINCIPLES:
        lines.append(f"- {p.principle} — {p.author} ({p.source}).")
        lines.append(f"  Why: {p.rationale}")
    return "\n".join(lines)
