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
# rebalance narrator. Only principles with a clear author attribution from
# that earlier pipeline are included here (a position-count preference
# from that same pipeline was the user's own choice, not book-attributed,
# so it's deliberately left out of this "author guidance" set).
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
            "across all open positions if every stop were hit — rather "
            "than sizing each position in isolation."
        ),
        author="Jack Schwager",
        source="the Market Wizards series",
        rationale=(
            "Individually reasonable position sizes can still combine "
            "into an unacceptable simultaneous drawdown if too many are "
            "risked at once; capping aggregate heat guards against that "
            "correlated-loss tail risk, not just single-trade risk."
        ),
    ),
    BookPrinciple(
        principle=(
            "Limit exposure to any single correlated market/sector group, "
            "not just any single instrument."
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
]


def format_book_wisdom() -> str:
    lines = ["Investing-literature principles (advisory context, not enforced rules):"]
    for p in BOOK_PRINCIPLES:
        lines.append(f"- {p.principle} — {p.author} ({p.source}).")
        lines.append(f"  Why: {p.rationale}")
    return "\n".join(lines)
