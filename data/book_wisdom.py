from dataclasses import dataclass


@dataclass
class BookPrinciple:
    principle: str
    author: str
    source: str
    rationale: str
    # "regime" (default) = big-picture/structural, Claude-mega-session-only
    # wisdom. "trend" = shorter-term/tactical (H4/H1-level) wisdom also fed
    # to the Execution Clerk's tactical-defense check (see format_trend_
    # wisdom below) — added 2026-08-27 per direct user request after a real
    # gold position went from +$28 to -$61 while the mega session hadn't
    # run in days and the Clerk had zero authority to react. Existing
    # entries are tagged in place, never removed/relocated, so
    # format_book_wisdom() (still used by PMEX/PSX) stays unchanged.
    tier: str = "regime"


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
        tier="trend",
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
        tier="trend",
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
        tier="trend",
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
        tier="trend",
    ),
    BookPrinciple(
        # Sharpened 2026-08-27 (live-verified via NotebookLM, see memory:
        # project_notebooklm_scholar) with O'Neil's own real relative
        # position-count tiers — previously only the qualitative framing
        # below. Kept relative/scale-based rather than his literal 1990s
        # USD dollar bands, since this app spans FTMO/USD and PMEX+PSX/
        # PKR accounts of very different absolute scale — the load-
        # bearing, currency-independent insight is that position count
        # should grow with capital only up to a firm ceiling, not
        # open-endedly.
        principle=(
            "Favor a small number of well-researched positions over broad "
            "diversification for its own sake, scaled to account size: "
            "O'Neil's own tiers cap even a large account at roughly 6-7 "
            "positions, a mid-sized account at 4-5, and a small account "
            "at 3 — position count should grow with capital, but only up "
            "to a point, not open-ended."
        ),
        author="William O'Neil",
        source="How to Make Money in Stocks",
        rationale=(
            "O'Neil's own reasoning: 'the more you diversify, the less "
            "you know about any one area' — broad diversification can "
            "act as a 'hedge for lack of knowledge, or ignorance,' where "
            "concentrating in fewer, better-understood positions allows "
            "for more meaningful position sizes and closer tracking of "
            "each one."
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
        tier="trend",
    ),
    BookPrinciple(
        # Sharpened 2026-08-27, "Phase 3" of the book-wisdom initiative —
        # previously only a one-sentence summary of this same model. Now
        # carries the full stage-by-stage detail, Murphy's own
        # deflationary-decoupling exception, and O'Neil's "three steps
        # and stumble"/late-cycle sector clue, sourced from the same
        # live, cited NotebookLM queries as the geopolitical-shock
        # entries below. data/macro_source.py::fetch_macro_cycle_
        # diagnostic() (added this same phase) computes a REAL numeric
        # version of exactly this 12-month-MA read for the actual
        # current data — this entry is the "how to read it" companion,
        # that function is the "what the data says right now" companion.
        # Distinct from the separate 12-month-MA equity-inclusion filter
        # entry just below (a binary go/no-go rule using the same MA
        # mechanic, not a duplicate of this descriptive stage read).
        principle=(
            "Pring's 6-stage market cycle model gives bond/stock/"
            "commodity rotation real structure, not just 'they rotate': "
            "Stage I — bonds rise while stocks and commodities fall "
            "(bond bull begins in recession); Stage II — bonds and "
            "stocks rise while commodities still fall (equities look "
            "through still-falling corporate profits); Stage III — "
            "bonds, stocks, and commodities all rise (recovery "
            "underway, commodity prices bottom); Stage IV — stocks and "
            "commodities rise while bonds fall (rates rise, bond bear "
            "begins, but the equity uptrend continues on improving "
            "productivity); Stage V — commodities rise while bonds and "
            "stocks fall (economy overheats, equities top out as the "
            "profit outlook sours); Stage VI — bonds, stocks, and "
            "commodities all fall (slide into recession). Early-cycle "
            "stages (I-II) tend to favor homebuilders/retail/consumer "
            "finance/utilities/insurance; mid-cycle (III) favors "
            "manufacturing; late-cycle (IV-V) favors capital-spending/"
            "steel/chemicals/mining, with gold/metals/energy turning "
            "from laggards into leaders — O'Neil's own practical tell: "
            "when machinery and capital-goods groups start running up, "
            "that's a sign the cycle is near its tail end. Two real, "
            "named exceptions keep this honest rather than mechanical: "
            "in a deflationary environment bonds and stocks can decouple "
            "entirely (bonds rise while stocks fall, since deflation is "
            "bad for both commodities and stocks); and the Fed's own "
            "historical 'three steps and stumble' pattern — three "
            "consecutive discount-rate hikes has frequently marked the "
            "start of a bear market or recession. Treat this as a "
            "descriptive framework for reading where the cycle currently "
            "sits, not a precise or reliably-timed predictive algorithm "
            "— in Murphy's own words, 'the leads and lags vary from "
            "cycle to cycle and have little forecasting value.'"
        ),
        author="Martin Pring / John Murphy / William O'Neil",
        source=(
            "Technical Analysis Explained; Technical Analysis of the "
            "Financial Markets; How to Make Money in Stocks"
        ),
        rationale=(
            "Recognizing which stage of this rotation the current macro "
            "data suggests helps frame whether a given asset class move "
            "looks early, on-time, or already-late relative to the "
            "others — and the honest caveats matter as much as the "
            "model itself: a trader who treats this as a precise timing "
            "tool, or who forgets the deflationary exception, will "
            "misread the very data this model is meant to clarify."
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
    # Added 2026-08-27, "Phase 2" of the book-wisdom initiative — direct
    # user request: "search the books for the perspective of risk
    # management and diverse portfolio creation (within the broker
    # limitations)." Sourced from live, cited NotebookLM queries against
    # the user's own 9-book library (see memory: project_notebooklm_
    # scholar), specifically correlation-aware sizing and diversification
    # under a constrained/broker-limited instrument universe. All five
    # entries are tier="regime" (the default, left unset below) — this is
    # portfolio-construction-time wisdom, not the short-term tactical
    # content Phase 1's tier="trend" covers.
    BookPrinciple(
        principle=(
            "When multiple open or candidate positions sit in highly "
            "correlated markets, treat them as functionally one larger "
            "position and size/leverage the group down accordingly — a "
            "portfolio of correlated markets should run at lower "
            "aggregate leverage than an equally-sized but genuinely "
            "diversified one."
        ),
        author="Jack Schwager",
        source="Getting Started in Technical Analysis",
        rationale=(
            "In Schwager's own words, 'positions in strongly correlated "
            "markets are similar to one larger position,' and the "
            "leverage of a homogenous, correlated portfolio 'should be "
            "adjusted downward' versus a more genuinely diversified one "
            "of equivalent size — a correlation-conditional adjustment, "
            "distinct from a flat aggregate-heat or per-group percentage "
            "cap."
        ),
    ),
    BookPrinciple(
        principle=(
            "True diversification requires spreading exposure across "
            "market groups with genuinely low correlation to each "
            "other — but treat any correlation reading as a snapshot, "
            "not a guarantee: correlations can decouple sharply exactly "
            "during a crisis. In the October 1997 Asian crisis, the "
            "normally-positive bond/stock correlation broke down "
            "entirely as investors bought bonds and sold stocks at once."
        ),
        author="John Murphy",
        source="Technical Analysis of the Financial Markets",
        rationale=(
            "A portfolio built on today's low correlation can quietly "
            "lose its diversification benefit the moment it's needed "
            "most, when a shared shock makes previously-independent "
            "markets move together — a real, named historical example, "
            "not a hypothetical caveat."
        ),
    ),
    BookPrinciple(
        principle=(
            "Multiple positions don't achieve real diversification just "
            "because they carry different symbols — they need genuinely "
            "different underlying drivers. Four foreign-currency "
            "positions that all trade against the US dollar move "
            "together, not independently; likewise, a second technology "
            "stock adds no diversification to an existing technology "
            "position, while a position in an unrelated sector (e.g. "
            "agriculture or finance, if already trading metals) does."
        ),
        author="John Murphy / James Rockefeller",
        source="Technical Analysis of the Financial Markets; Technical Analysis For Dummies",
        rationale=(
            "Both authors independently warn against this exact 'false "
            "diversification' trap — a wider symbol count with the same "
            "underlying driver is concentration wearing a diversified "
            "costume, not real risk reduction."
        ),
    ),
    BookPrinciple(
        principle=(
            "When the tradeable instrument universe itself is limited "
            "(e.g. by which symbols a broker's Market Watch exposes), "
            "diversification doesn't have to stop there — it can still "
            "be pursued across multiple systems or approaches, multiple "
            "parameter sets, and multiple time frames applied to that "
            "same limited set of markets, not only across a larger "
            "instrument count."
        ),
        author="Jack Schwager / John Murphy",
        source="Getting Started in Technical Analysis; Technical Analysis of the Financial Markets",
        rationale=(
            "Schwager's own framing: diversification 'applies... not "
            "only [to] multiple markets but also multiple systems... and "
            "multiple system variations... for each market'; Murphy "
            "independently: 'diversify among markets, systems, "
            "parameters, and time frames' — directly answers a broker-"
            "constrained instrument universe, where adding more markets "
            "isn't always an option but diversifying HOW an existing "
            "market is traded still is."
        ),
    ),
    BookPrinciple(
        principle=(
            "A low historical correlation between positions is not a "
            "permanent safety guarantee: independently-built portfolios "
            "can end up highly correlated simply by adapting to the same "
            "market environment ('crowded trades'), and new connections "
            "between previously-unrelated assets ('tight coupling') can "
            "make the whole system more fragile during a crisis. More "
            "broadly, the boundaries between asset classes have grown "
            "blurred enough that diversifying purely by asset allocation "
            "is no longer as reliable a risk control as it once was."
        ),
        author="Andrew Lo",
        source="Adaptive Markets",
        rationale=(
            "Lo's own account of the 2007 quant-fund unwind: independent, "
            "secret proprietary strategies had all converged on similar "
            "portfolios, which 'reduced the effective liquidity of those "
            "components, and it was impossible for everyone to unwind "
            "their positions at once' — plus his own explicit 'Principle "
            "4A: Asset Allocation' caveat that this blurring has made "
            "allocation-based diversification less effective than during "
            "calmer periods."
        ),
    ),
    # Added 2026-08-27, "Phase 3" of the book-wisdom initiative — direct
    # user request: "search the books for the perspective of geopolitics
    # shocks and macro patterns (within the broker limitations)." Sourced
    # from live, cited NotebookLM queries against the user's own 9-book
    # library (see memory: project_notebooklm_scholar). All four entries
    # are tier="regime" (the default, left unset below). This account's
    # real, broker-limited 17-symbol universe (forex pairs, MSFT/NVDA/
    # AMD/INTC CFDs, XAUUSD/XAGUSD, BTCUSD/ETHUSD) has no bond/rate
    # instrument at all — this content is leading-indicator CONTEXT for
    # timing the FX/gold/equity-CFD positions that ARE actually
    # tradeable here, not a literal "go trade bonds/rates" instruction.
    BookPrinciple(
        # Flagged as the single most actionable, least obvious point of
        # this whole set.
        principle=(
            "When a geopolitical shock or unexpected headline hits, "
            "watch how the market actually reacts to it rather than "
            "reasoning from the headline in isolation: if a price fails "
            "to move the way the news 'should' move it, the news is "
            "most likely already priced in — or the market is already "
            "in the process of turning. Also weigh which direction a "
            "surprise breaks: surprises tend to occur in the direction "
            "of the existing primary trend (upside surprises in a bull "
            "market, downside surprises in a bear market), not against "
            "it."
        ),
        author="Martin Pring",
        source="Technical Analysis Explained",
        rationale=(
            "Pring's own words: 'observe the reaction of any market to "
            "news events, especially unexpected ones... If a news event "
            "that would normally be expected to move the price does not "
            "do so, the likelihood is that all the news—good or bad—is "
            "already reflected in the price,' and 'if a price does not "
            "respond to news in the expected way, it is probably in the "
            "process of turning.' This lines up with Schwager's own "
            "account of Marty Schwartz reading a market's refusal to "
            "sell off on scary headlines as a bullish tell — the absence "
            "of follow-through itself is the signal, not the headline."
        ),
    ),
    BookPrinciple(
        principle=(
            "Treat live price action itself as already incorporating a "
            "geopolitical shock — don't wait on a slower news cycle or "
            "official investigation to 'confirm' what happened before "
            "trusting what the chart is already showing. Price tends to "
            "lead the known fundamentals, not follow them."
        ),
        author="John Murphy",
        source="Technical Analysis of the Financial Markets",
        rationale=(
            "Murphy's own principle: 'Market action discounts "
            "everything... anything that can possibly affect the "
            "price—fundamentally, politically, psychologically, or "
            "otherwise—is actually reflected in the price,' extending "
            "even to 'acts of God' (natural disasters) being 'almost "
            "instantaneously' assimilated; and 'market price tends to "
            "lead the known fundamentals... acts as a leading "
            "indicator.' Andrew Lo's own account of the 1986 Challenger "
            "disaster is a vivid, concrete illustration of this: the "
            "market identified and punished the responsible supplier "
            "(Morton Thiokol) within minutes — what took the official "
            "Rogers Commission five months to establish."
        ),
    ),
    BookPrinciple(
        principle=(
            "Don't try to trade a geopolitical shock by guessing what "
            "happens next — build and hold a portfolio structured to "
            "survive a wide range of outcomes, rather than one that "
            "only works if a specific crisis prediction proves right. "
            "Don't let a panicky, shock-driven market mood dictate your "
            "own actions: never buy immediately after a sharp spike, "
            "and never sell immediately after a sharp drop."
        ),
        author="Benjamin Graham (with Jason Zweig commentary)",
        source="The Intelligent Investor",
        rationale=(
            "'The only certainty you can derive from past financial "
            "data is that the future is always surprising. And it will "
            "most brutally surprise those who are the most certain they "
            "know what is about to happen... Instead of trying to build "
            "a portfolio that would thrive if what you think will "
            "happen does happen, strive to build a portfolio that "
            "should thrive no matter what happens. Stop fruitlessly "
            "trying to predict the unknowable.' Graham's own explicit "
            "rule reinforces the same discipline in the moment: 'Never "
            "buy a stock immediately after a substantial rise or sell "
            "one immediately after a substantial drop' — the 'Mr. "
            "Market' framing is a reminder that a panicky counterparty's "
            "mood is not a reason to panic yourself."
        ),
    ),
    BookPrinciple(
        principle=(
            "Recognize the 'fight-or-flight' panic-selling reflex a "
            "geopolitical shock triggers for what it is — a hard-wired "
            "survival instinct, not a reasoned trading decision — and "
            "be deliberate about not acting on it directly: it drives "
            "indiscriminate selling of risk assets and indiscriminate "
            "buying of perceived-safe ones, regardless of whether that "
            "specific asset was actually affected."
        ),
        author="Andrew Lo",
        source="Adaptive Markets",
        rationale=(
            "Lo's own account: 'Sudden increases in equity volatility "
            "cause a significant portion of investors to rapidly reduce "
            "their holdings through a fight-or-flight response... This "
            "panic selling puts downward pressure on equity prices, and "
            "upward pressure on the prices of safer assets,' and his "
            "own warning that 'while our fear reflexes may protect us "
            "from injury, they do little to prevent us from losing "
            "large sums of money.' Notably, Lo's own Challenger example "
            "(see the Murphy entry above) shows the flip side of this "
            "same reflex machinery: collective price discovery can be "
            "both extremely fast AND accurate under stress, even as the "
            "same reflex system also drives indiscriminate overreaction "
            "elsewhere — it cuts both ways."
        ),
    ),
]


def format_book_wisdom() -> str:
    lines = ["Investing-literature principles (advisory context, not enforced rules):"]
    for p in BOOK_PRINCIPLES:
        lines.append(f"- {p.principle} — {p.author} ({p.source}).")
        lines.append(f"  Why: {p.rationale}")
    return "\n".join(lines)


# New as of 2026-08-27, sourced from live, cited NotebookLM queries against
# the user's own 9-book library (not secondhand paraphrasing — see memory:
# project_notebooklm_scholar) specifically for the "trend"/tactical tier:
# holding-period framing and short-selling asymmetric risk. This is
# genuinely new content, not a re-tag of an existing BOOK_PRINCIPLES entry —
# format_trend_wisdom() below combines both.
#
# Every entry below has been individually re-verified against a fresh,
# separately-cited NotebookLM query per author (audit pass, 2026-08-27) —
# not just the original broader scope-check query these entries were
# first drafted from. The Rockefeller and Graham/Lo entries matched the
# real book text exactly; the O'Neil/Murphy entry's rationale was found
# to over-attribute the "unlimited downside" framing to Murphy (his real,
# cited content here is the squeeze's own mechanics, not that framing)
# and was corrected in place — see that entry's own comment.
TREND_PRINCIPLES: list[BookPrinciple] = [
    BookPrinciple(
        principle=(
            "Match trade management to the holding period actually being "
            "run: Rockefeller's own framework distinguishes position "
            "traders (weeks/months/years), swing traders (roughly 3-10 "
            "days), day traders (a single trading day, watching 5-min/"
            "15-min/1-hour indicators and 2-3 hour micro-trends within "
            "it), and scalpers (seconds/minutes). This account's own "
            "intraday-to-at-most-one-trading-day design maps to "
            "Rockefeller's 'day trader' category — judge whether a "
            "position still fits a single-session thesis accordingly, "
            "not against a swing- or position-trader's much longer "
            "patience."
        ),
        author="James Rockefeller",
        source="Technical Analysis For Dummies",
        rationale=(
            "A tactical read that would be perfectly reasonable patience "
            "for a swing trader (holding through a multi-day pullback) "
            "can be exactly the wrong instinct for a day trader's own "
            "position, since the two are operating on entirely different "
            "clocks with different confirmation windows."
        ),
        tier="trend",
    ),
    BookPrinciple(
        principle=(
            "Treat a losing short position with more urgency than an "
            "equivalent losing long: a long's downside is capped at zero, "
            "but a short's downside is theoretically unlimited ('the sky "
            "is the limit' if it isn't cut), and a short run against a "
            "thin, illiquid, or already-popular/crowded instrument "
            "carries real short-squeeze risk (forced buying-to-cover "
            "accelerating the very move working against it) — cut a "
            "deteriorating short faster than the same deterioration "
            "would justify on a long."
        ),
        author="William O'Neil / John Murphy",
        source="How to Make Money in Stocks; Technical Analysis of the Financial Markets",
        rationale=(
            # Live-verified 2026-08-27 via a real, cited NotebookLM query
            # (see memory: project_notebooklm_scholar): O'Neil's own book
            # explicitly covers BOTH halves ("otherwise, the sky is the
            # limit, and a short-selling mistake could cause sickening
            # losses"; and his own hard rule, "never sell short a thinly
            # capitalized... stock; it's too easy for the stock to be run
            # up on you [a short squeeze]"). Murphy's real, cited content
            # in this search covers the SQUEEZE'S OWN MECHANICS (a wave
            # of short positions all showing a loss at once, scrambling
            # to cover, driving the very panic-buying that deepens the
            # squeeze) rather than the unlimited-downside framing itself
            # — corrected here from an earlier draft that over-attributed
            # that specific framing to Murphy too.
            "O'Neil's own explicit warnings — 'the sky is the limit' on an "
            "uncut short, and his hard rule against ever shorting a "
            "thinly-capitalized stock because of short-squeeze risk — "
            "both underscore that a short's risk profile is structurally "
            "worse than a long's, where the downside is at least capped "
            "at zero. Murphy's own account of a squeeze's actual "
            "mechanics (a wave of short positions all showing a loss at "
            "once, forced to cover, driving the very panic-buying that "
            "deepens it) explains why that risk compounds so quickly "
            "once triggered."
        ),
        tier="trend",
    ),
    BookPrinciple(
        principle=(
            "An irrational or crowded move against a short (or a highly "
            "leveraged long/short pair) can persist far longer than the "
            "position's own thesis would suggest is rational — leveraged "
            "exposure 'cuts both ways,' amplifying an adverse move exactly "
            "as much as it would have amplified a favorable one — so "
            "don't treat 'the market is wrong' as a reason to hold through "
            "continued deterioration without tightening risk."
        ),
        author="Benjamin Graham / Andrew Lo",
        source="The Intelligent Investor; Adaptive Markets",
        rationale=(
            "Graham's own point about shorting popular names testing "
            "'courage, stamina, and depth of pocketbook,' and Lo's point "
            "that leverage amplifies both sides of a position, both argue "
            "for defending capital while a move is running against a "
            "position rather than waiting for eventual vindication."
        ),
        tier="trend",
    ),
]


def format_trend_wisdom() -> str:
    trend = [p for p in BOOK_PRINCIPLES if p.tier == "trend"] + TREND_PRINCIPLES
    lines = [
        "Short-term/tactical investing-literature principles (advisory "
        "context for defending or exiting an already-open position between "
        "full sessions, not enforced rules):"
    ]
    for p in trend:
        lines.append(f"- {p.principle} — {p.author} ({p.source}).")
        lines.append(f"  Why: {p.rationale}")
    return "\n".join(lines)
