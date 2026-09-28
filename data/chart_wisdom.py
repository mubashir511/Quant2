from data.book_wisdom import BookPrinciple

# Phase 5a of the charting-expert technical-analysis upgrade (2026-09-21,
# direct user challenge: "current technical analysis is still weak...
# i want my core technical analysis architecture to be a charting
# expert"). Reuses BookPrinciple verbatim (see that dataclass's own
# docstring) rather than redefining an identical dataclass here — this is
# the same advisory-reasoning-context pattern data/book_wisdom.py already
# established for portfolio construction, extended to chart-reading.
#
# Every entry below grounds a specific deterministic rule this same
# upgrade added or already relies on (see the analysis/*.py module cited
# in each rationale), so the model reading this alongside that rule's own
# computed output understands WHY the rule exists, not just what number
# it produced. tier follows BookPrinciple's own convention: "trend"
# (H4/H1-tactical, also fed to the Execution Clerk) vs. the default
# "regime" (bigger-picture, Mega Session only).
CHART_PRINCIPLES: list[BookPrinciple] = [
    BookPrinciple(
        principle=(
            "A trendline only earns real validity once price has touched "
            "and respected it at least three times — two points merely "
            "define a line, a third real touch is what confirms the "
            "market itself is actually respecting that slope."
        ),
        author="John Murphy",
        source="Technical Analysis of the Financial Markets",
        rationale=(
            "Grounds this codebase's own touch-count-gated support/"
            "resistance ranking (analysis/chart_structure.py's SRLevel."
            "touches, and setup_classifier.py's STRONG_TOUCH_COUNT floor) "
            "— a line drawn through only two points is a guess about "
            "the market's structure, not yet evidence of it."
        ),
        tier="trend",
    ),
    BookPrinciple(
        principle=(
            "Dow theory's own reversal criteria: an established trend is "
            "still intact as long as each successive rally and reaction "
            "keeps making higher highs and higher lows (or the mirror for "
            "a downtrend) — the trend is only genuinely broken once price "
            "makes a lower low in an uptrend, or a higher high in a "
            "downtrend, undercutting the prior swing."
        ),
        author="John Murphy",
        source="Technical Analysis of the Financial Markets",
        rationale=(
            "Grounds analysis/chart_structure.py's detect_structure_"
            "breaks — a BOS (break of structure) confirms the prevailing "
            "trend by extending it past its own prior swing; a CHOCH "
            "(change of character) is exactly this Dow-theory reversal "
            "criteria being met for the first time against the "
            "established trend, mechanized directly from real swing-"
            "point data rather than eyeballed off a chart."
        ),
        tier="trend",
    ),
    BookPrinciple(
        principle=(
            "A breakout from a chart pattern or trading range needs a "
            "genuine pickup in volume to be trusted — a move through a "
            "level on light, unchanged participation is far more likely "
            "to be a false start than a breakout on volume clearly above "
            "the instrument's own recent average."
        ),
        author="Thomas Bulkowski",
        source="Encyclopedia of Chart Patterns",
        rationale=(
            "Grounds analysis/chart_structure.py's detect_breakouts and "
            "its BREAKOUT_VOLUME_CONFIRM_RATIO threshold — Bulkowski's "
            "own pattern statistics tie above-average breakout volume to "
            "measurably stronger post-breakout follow-through, which is "
            "exactly the real/fake distinction that function's "
            "volume_confirmed flag is built to surface. Since this "
            "account trades FX/CFDs, MT5's own volume is real tick-count "
            "activity, not literal traded share volume — a real, "
            "disclosed proxy for participation, not a claim of exact "
            "equivalence."
        ),
        tier="trend",
    ),
    BookPrinciple(
        principle=(
            "A 'busted' pattern — one that breaks out convincingly in one "
            "direction and then reverses hard through its own starting "
            "range — tends to travel unusually far once it reverses, "
            "often farther than the pattern's own original target would "
            "have implied. Treat a clean failure of a recent breakout as "
            "a real, actionable reversal signal in its own right, not "
            "just a canceled trade."
        ),
        author="Thomas Bulkowski",
        source="Encyclopedia of Chart Patterns",
        rationale=(
            "Grounds setup_classifier.py's existing busted_pattern_"
            "reversal rule directly — a failed breakout isn't neutral "
            "information, it's Bulkowski's own documented statistical "
            "edge that a busted pattern's reversal move tends to "
            "outperform the pattern's original, now-invalidated thesis."
        ),
    ),
    BookPrinciple(
        principle=(
            "A liquidity sweep — a sharp wick through a well-known "
            "support or resistance level that closes back inside the "
            "range within a few bars — is the classic stop-hunt "
            "signature: the level was touched only to trigger the stop "
            "orders clustered just beyond it, not because the market "
            "genuinely intends to hold beyond that level. Treat a swift "
            "close-back-inside as evidence AGAINST the breakout, not "
            "confirmation of one."
        ),
        author="Thomas Bulkowski",
        source="Encyclopedia of Chart Patterns",
        rationale=(
            "Grounds analysis/chart_structure.py's detect_liquidity_"
            "sweeps and its SWEEP_MAX_BARS_TO_CLOSE_BACK window — the "
            "same real/fake-move distinction Bulkowski's own busted-"
            "pattern research documents, applied to a raw S/R level "
            "rather than a fully-formed chart pattern."
        ),
        tier="trend",
    ),
    BookPrinciple(
        principle=(
            "A pullback that retraces into the 38.2-61.8% Fibonacci zone "
            "of the prior impulse leg, without breaking the structure "
            "that impulse established, is a genuine continuation "
            "candidate — but a pullback is only a pullback for as long as "
            "the underlying trend structure stays intact. The moment a "
            "counter-trend swing breaks that structure (a change of "
            "character), what looked like a pullback has to be "
            "re-classified as the start of a real reversal instead."
        ),
        author="John Murphy",
        source="Technical Analysis of the Financial Markets",
        rationale=(
            "Grounds setup_classifier.py's own arbitration between "
            "pullback_continuation and reversal_candidate — the CHOCH-"
            "gate (pullback_continuation suppressed once a counter-trend "
            "change of character has occurred within CHOCH_GATE_"
            "LOOKBACK_BARS) is a direct mechanization of this exact "
            "distinction, closing the gap where a deep, trend-ending "
            "retracement that happened to land in the golden zone used "
            "to be called 'just a pullback' with no tiebreak at all."
        ),
        tier="trend",
    ),
    BookPrinciple(
        principle=(
            "Divergence between price and a momentum oscillator like "
            "RSI — price making a new high while RSI makes a LOWER high "
            "(bearish), or price making a new low while RSI makes a "
            "HIGHER low (bullish) — is one of the earliest, most reliable "
            "warnings that a trend's own underlying momentum is fading "
            "even while price itself is still extending. It is a warning "
            "sign to watch for confirmation, not by itself a signal to "
            "act on immediately."
        ),
        author="Martin Pring",
        source="Technical Analysis Explained",
        rationale=(
            "Grounds analysis/technical.py's detect_rsi_divergence "
            "directly — Pring's own treatment of divergence as an early-"
            "warning tool (not a standalone trigger) is exactly why "
            "setup_classifier.py only upgrades reversal_candidate to "
            "confidence='strong' when divergence lines up WITH another "
            "independent confirmation (a CHOCH or a liquidity sweep), "
            "rather than acting on divergence alone."
        ),
        tier="trend",
    ),
    BookPrinciple(
        principle=(
            "A single candlestick reversal pattern (e.g. a hammer, "
            "engulfing candle, or doji) carries far more weight when it "
            "forms at a level with independent significance — a prior "
            "swing high/low, a round number, a well-tested support or "
            "resistance zone — than the identical-looking candle forming "
            "in the middle of open space with no structural context "
            "behind it."
        ),
        author="Steve Nison",
        source="Japanese Candlestick Charting Techniques",
        rationale=(
            "Grounds setup_classifier.py's existing candlestick_"
            "reversal_confirmed rule, which specifically requires the "
            "candlestick signal to align with a real, independently-"
            "computed support/resistance level rather than firing on the "
            "candle shape in isolation — Nison's own core teaching that "
            "candlestick signals are confirmed by their CONTEXT, not "
            "their shape alone."
        ),
    ),
    BookPrinciple(
        principle=(
            "A support or resistance level is a zone with real width, "
            "not a single exact price — respect the band the level's own "
            "historical touches actually clustered into rather than "
            "treating a level as invalidated the instant price ticks a "
            "fraction beyond its nominal center."
        ),
        author="John Murphy / Thomas Bulkowski",
        source="Technical Analysis of the Financial Markets; Encyclopedia of Chart Patterns",
        rationale=(
            "Grounds analysis/chart_structure.py's SRLevel.low/.high band "
            "(Phase 1 of this same upgrade) directly — a level was "
            "previously exposed as only a single cluster-mean price even "
            "though the clustering tolerance that produces it already "
            "implies a real band width; treating that band as the level, "
            "not a point, is what lets an entry/stop/target be built "
            "around a real zone instead of one fragile exact number."
        ),
        tier="trend",
    ),
    # Added 2026-09-22, sourced live from this project's own NotebookLM
    # notebook (all 9 owned books already uploaded there) — direct user
    # request following a real, live challenge: "in conditions are
    # extremely overbought then why not look for SHORT condition?" On
    # investigation, the deterministic layer had no rule at all for
    # "RSI extreme, no confirmed reversal pattern yet" — the closest
    # existing rule (reversal_candidate) requires an actual double
    # top/bottom to already exist. These two entries ground that real,
    # named gap in the books' own actual guidance (not paraphrased —
    # each quote below is the book's own real text, cited chapter/page)
    # for the day a rule is built to close it.
    BookPrinciple(
        principle=(
            "An extreme RSI reading (e.g. above 85 or below 15) with no "
            "confirmed reversal pattern yet is a 'trading alert,' never "
            "itself a signal — in a strong trend the reading can stay "
            "extreme for a long time, and the FIRST push into the extreme "
            "zone is usually just a warning, not the moment to act. "
            "Before treating it as an actual entry, wait for at least one "
            "real confirmation: the RSI actually crossing back through "
            "its own 70/30 line (not just touching it), a real reversal "
            "candlestick forming at the extreme, or price itself breaking "
            "its most recent minor swing point or trendline."
        ),
        author="Martin Pring / John Murphy / Jack Schwager / Steve Nison",
        source=(
            "Technical Analysis Explained; Technical Analysis of the "
            "Financial Markets; Getting Started in Technical Analysis; "
            "Japanese Candlestick Charting Techniques"
        ),
        rationale=(
            "Murphy's own words: 'Any strong trend, either up or down, "
            "usually produces an extreme oscillator reading before too "
            "long. In such cases, claims that a market is overbought or "
            "oversold are usually premature... The first move into the "
            "overbought or oversold region is usually just a warning' "
            "(Technical Analysis of the Financial Markets, ch. 10, "
            "p.233-234); Schwager independently calls trading a bare "
            "extreme against a trend 'a recipe for disaster,' preferring "
            "oscillator readings used only as 'trading alerts... "
            "positions are established only when price confirms a "
            "reversal' (Getting Started in Technical Analysis, ch. 6/10); "
            "Pring's own 'Major Technical Principle': 'divergences only "
            "warn of a weakening or strengthening market condition and do "
            "not represent actual buy and sell signals... it is essential "
            "to wait for a confirmation from the price itself' (Technical "
            "Analysis Explained, ch. 13). Directly names the real gap "
            "this codebase's own setup-classifier had no rule for: an "
            "extended, unconfirmed move (no double top/bottom yet) "
            "genuinely deserves a WATCH-for-confirmation framing, not "
            "silence and not an immediate fade."
        ),
        tier="trend",
    ),
    BookPrinciple(
        principle=(
            "Flags and pennants are genuinely reliable, well-documented "
            "continuation patterns, even though a specific one is too "
            "geometrically subjective for this codebase's own detector to "
            "flag automatically — Bulkowski's own real statistics show "
            "upward-breakout flags hit their measured-move target 64% of "
            "the time in bull markets, pennants 60%, and a 'high and "
            "tight flag' (a quick 90-100%+ advance in under 2 months, "
            "then a tight consolidation) has a 0% breakeven-failure rate "
            "across 307 real historical samples, averaging a further "
            "42-69% gain. Counter-intuitively, don't assume heavy "
            "breakout volume always means a stronger move: for a bull-"
            "market upward flag breakout specifically, LIGHT breakout "
            "volume has historically outperformed heavy volume (26% "
            "average gain vs. 19%), because everyone buying right at the "
            "breakout leaves no one left to buy on the following days, "
            "inviting an immediate throwback."
        ),
        author="Thomas Bulkowski",
        source="Encyclopedia of Chart Patterns",
        rationale=(
            "Real, cited numbers, not a paraphrase: regular flags reach "
            "their price target 64%/55% of the time (bull/bear, upward "
            "breakout) with a 4%/3% breakeven-failure rate (ch. 21, "
            "p.335-349); high-and-tight flags hit their target 90%/91% of "
            "the time with 0 of 307 samples failing to move at least 5% "
            "(ch. 22, p.350-359); pennants reach target 60%/63% of the "
            "time (ch. 34, p.522-535). The light-breakout-volume nuance "
            "(ch. 21, p.346, Table 21.7) is a real, disclosed complication "
            "for this codebase's own BREAKOUT_VOLUME_CONFIRM_RATIO logic "
            "(analysis/chart_structure.py's detect_breakouts): that rule "
            "is about a close beyond a raw S/R level, not specifically a "
            "flag/pennant breakout, so the two aren't in direct conflict "
            "— but 'more volume always means a stronger breakout' is not "
            "a universal truth even in Bulkowski's own numbers, and is "
            "worth remembering before overweighting volume confirmation "
            "on what might actually be a flag/pennant setup."
        ),
    ),
]


def format_chart_wisdom() -> str:
    lines = [
        "Charting/price-action literature principles (advisory context "
        "for reading the deterministic structure signals above, not "
        "enforced rules):"
    ]
    for p in CHART_PRINCIPLES:
        lines.append(f"- {p.principle} — {p.author} ({p.source}).")
        lines.append(f"  Why: {p.rationale}")
    return "\n".join(lines)
