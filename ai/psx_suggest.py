from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

import config
from ai.claude_cli import CLI_FAILED_PREFIX, CLI_MISSING_MESSAGE, run_claude
from ai.portfolio_suggest import AUDIT_MODELS, AllocationEntry, build_audit_block
from ai.session_record import SessionRecord, save_portfolio_session
from analysis.technical import TechnicalStats, compute_technical_stats
from data.book_wisdom import format_book_wisdom
from data.macro_source import fetch_country_indicators, fetch_fx_rate_to_usd, fetch_pakistan_rates
from data.psx_source import (
    PSX_INDICES,
    PSXAnnouncement,
    PSXAsset,
    PSXCompanyProfile,
    PSXFinancials,
    PSXFundamentals,
    get_psx_company_data,
    get_psx_history,
    get_sector_name,
)

DEFAULT_INDEX = "KSE100"


def _fetch_index_stats(index_tag: str = DEFAULT_INDEX) -> TechnicalStats:
    # PSX's own EOD timeseries endpoint works for any index tag the same
    # way it does for an individual symbol (confirmed live for every tag
    # in data.psx_source.PSX_INDICES) — used both to give the model a
    # market-wide technical read and to compute each enriched stock's
    # relative strength against whichever index it was drawn from.
    history = get_psx_history(index_tag)
    prices = history["Close"] if not history.empty else pd.Series(dtype=float)
    return compute_technical_stats(prices, history=history if not history.empty else None)


_MIN_BETA_OBSERVATIONS = 60  # ~3 months of trading days — a shorter window is too noisy to trust


def _compute_beta(stock_prices: pd.Series, index_prices: pd.Series) -> float | None:
    """Beta vs the benchmark index: how sensitive this stock's own daily
    moves structurally are to the index's moves (>1 more volatile than
    the index, <1 less, negative moves opposite it) — a genuinely
    different concept from relative strength, which only measures how
    much return actually differed, not how tightly the two co-move.
    None without a real window's worth of aligned trading days, matching
    this project's rule of never fabricating a stat from too little
    history."""
    if stock_prices.empty or index_prices.empty:
        return None
    aligned = pd.DataFrame({"stock": stock_prices, "index": index_prices}).dropna()
    if len(aligned) < _MIN_BETA_OBSERVATIONS:
        return None
    stock_returns = aligned["stock"].pct_change().dropna()
    index_returns = aligned["index"].pct_change().dropna()
    index_variance = index_returns.var()
    if not index_variance:
        return None
    return float(stock_returns.cov(index_returns) / index_variance)


def _compute_week52_position_pct(
    current: float, week52_low: float | None, week52_high: float | None
) -> float | None:
    """Where the current price sits within its own 52-week range, as a
    %: 0 = at the 52-week low, 100 = at the 52-week high. A complementary,
    longer-horizon read to the existing ~3-month support/resistance
    range — a stock can be "trending up" over 3 months while still sitting
    well below its own 52-week high, which this surfaces explicitly."""
    if week52_low is None or week52_high is None:
        return None
    span = week52_high - week52_low
    if span <= 0:
        return None
    return (current - week52_low) / span * 100

# app.py imports parse_final_allocation/strip_allocation_block/
# AllocationEntry from ai.portfolio_suggest directly for PSX suggestions
# too — those are pure string/dict parsing with no PMEX-specific content
# (the same ```json {"SYMBOL": {"pct", "price", "stop_loss"}, "CASH": pct}
# schema works equally well for equities), so there's nothing PSX-specific
# to re-export here.


_INSTRUCTION_HEAD = (
    "You are a portfolio-construction assistant researching the Pakistan "
    "Stock Exchange (PSX) for a HYPOTHETICAL portfolio — there is no real "
    "brokerage account behind this (the user's broker, K-Trade, has no "
    "programmatic API), so there are no real positions to reconcile "
    "against and nothing will be executed automatically. Treat the "
    "hypothetical capital figure given below as a clean starting slate. "
    "Your job is to produce the best-reasoned illustrative allocation you "
    "can, which the user will consider and execute manually (or not) "
    "through their own broker. You have WebSearch and WebFetch tools "
    "available."
    "\n\n"
    "EXPLICIT INVESTMENT OBJECTIVE (stated directly by the user, treat "
    "this as the goal every sizing/risk decision below serves): beat "
    "Pakistan's average inflation rate (the real current figure is given "
    "in the macro snapshot below) by a meaningful margin, and achieve "
    "high long-term wealth growth, while taking the MINIMUM risk "
    "necessary to do so — not risk avoidance for its own sake. The user "
    "has explicitly said they do NOT want an overly conservative "
    "portfolio that fails to beat inflation, since a mix that can't even "
    "outpace inflation defeats the purpose of investing at all (it loses "
    "real purchasing power even while looking 'safe' in nominal terms). "
    "This does not mean maximize risk or return — it means don't default "
    "to an unnecessarily defensive mix out of caution alone; justify the "
    "actual risk taken as what's needed to plausibly beat inflation and "
    "grow real wealth over the long term, and say explicitly, near the "
    "start of your visible answer, what the mix's expected return "
    "profile implies relative to that real inflation figure."
    "\n\n"
    "Source quality matters — prioritize in this order:\n"
    "1. Official/primary sources: the State Bank of Pakistan (SBP), the "
    "Pakistan Bureau of Statistics (PBS), the Ministry of Finance, PSX's "
    "own regulatory filings/announcements, and the IMF/World Bank for "
    "Pakistan-specific data.\n"
    "2. Major investment banks' and rating agencies' public research on "
    "Pakistan when actually available (e.g. JPMorgan, Moody's, S&P, "
    "Fitch) — read what they've actually published, don't guess at their "
    "view.\n"
    "3. Major, reputable Pakistani and international financial/general "
    "news outlets: Dawn, Business Recorder, The News International, "
    "Bloomberg, Reuters, Financial Times.\n"
    "4. Social media/forums (X/Twitter, Reddit) — useful for catching "
    "real-time sentiment shifts before slower outlets report on it, but "
    "treat it as lower-confidence than tiers 1-3: verify a notable claim "
    "against a primary/news source before treating it as fact.\n"
    "Do not cite Wikipedia or low-quality/unverified blogs as a source. If "
    "a genuinely credible source can't be found for something, say so "
    "rather than falling back to a weak one or inventing a citation."
    "\n\n"
    "This suggestion's scope has been deliberately narrowed by the user "
    "to a specific PSX index (named in the data below) rather than the "
    "full ~490-symbol market — build your mix ONLY from the constituents "
    "given below that index, do not reach for a symbol outside it even "
    "if it appears in the broader market-mover context. You're also "
    "given that selected index's own technical read (a market-wide "
    "trend/momentum signal, separate from any individual stock) — weigh "
    "individual-stock signals against the direction of the index they "
    "sit in, not in isolation."
    "\n\n"
    "You're given, per symbol: technical context (a short-term trend vs. "
    "20-day moving average, annualized volatility, 14-day RSI — "
    "conventionally overbought above 70, oversold below 30 — a volume "
    "trend, a medium-term support/resistance range with a market-type "
    "classification, relative strength vs. the selected index over 1 and "
    "3 months — a positive figure means this stock is outperforming that "
    "index over that window, which matters for sector-rotation reasoning "
    "independent of the stock's own absolute trend — AND beta vs. that "
    "same index, a genuinely different concept from relative strength: "
    "beta measures how tightly this stock's day-to-day moves have "
    "co-moved with the index structurally (>1 historically amplifies the "
    "index's moves, <1 dampens them), while relative strength only "
    "measures how much total return actually differed; use beta when "
    "reasoning about how a position would behave in a broad market swing, "
    "and relative strength when reasoning about whether it's currently in "
    "or out of favor); fundamental context (P/E ratio, market "
    "capitalization, shares outstanding, free-float %) where available; "
    "PSX's own published circuit-breaker band and 52-week range PLUS "
    "where the current price sits within that 52-week range as a % (0 = "
    "at the 52-week low, 100 = at the 52-week high — a longer-horizon "
    "complement to the ~3-month support/resistance range, since a stock "
    "can be trending up over 3 months while still well below its own "
    "52-week high); REAL sector classification (PSX's own official "
    "industry category — e.g. Commercial Banks, Cement, Oil & Gas "
    "Exploration Companies, Technology & Communication, Real Estate "
    "Investment Trust — not a bare numeric code) and whether it's a "
    "current member of the PSX Dividend 20 Index (the exchange's own "
    "classification of its top 20 dividend-paying stocks — real, "
    "structured evidence of an established dividend-paying track record, "
    "though not a yield figure; a non-member may still pay dividends, "
    "just isn't currently in that top-20 group); NOTE: Return on Equity, "
    "Return on Assets, Debt-to-Equity, and the actual Dividend Yield % "
    "are NOT available from this data source at all (no balance-sheet or "
    "payout data) — verify these via WebSearch if they matter for a "
    "specific decision rather than assuming a figure or estimating one "
    "from what IS given; REAL "
    "REPORTED financial-statement figures straight from the company's own "
    "filings on the PSX Data Portal — annual and trailing-quarter Sales, "
    "Profit after Taxation, and EPS (not estimates), plus Gross/Net Profit "
    "Margin, EPS Growth, and PEG ratio where available; a business-profile "
    "summary (what the company actually does, its CEO/Chairperson, its "
    "auditor, and fiscal year end) — use the auditor's identity as a "
    "light credibility signal (a well-known major audit firm vs. an "
    "unfamiliar small one is worth noting, not decisive on its own); and "
    "a handful of the company's own most recent official disclosures "
    "(financial-result submissions, board-meeting notices, and other "
    "regulatory filings, each with a real date) — treat these as a "
    "genuine primary source for company-specific news, not a substitute "
    "for your own broader WebSearch, but a strong starting point. A "
    "company with a recent 'Board Meeting and Closed Period' disclosure "
    "is often about to announce a dividend or corporate action — check "
    "for this explicitly "
    "per symbol rather than only from a general news search. Average True "
    "Range is NOT available for PSX symbols (the data source has no "
    "intraday high/low series) — when a stop distance is needed, derive "
    "it from annualized volatility or the shown support/resistance range "
    "instead, and say so explicitly rather than presenting a substitute "
    "as if it were ATR."
    "\n\n"
    "Use each candidate's real sector classification AND its business-"
    "profile description together to drive TARGETED research, not just "
    "generic company-name-plus-financial-news searches: a Commercial "
    "Bank's relevant research angle is interest-rate policy and asset "
    "quality; an Oil & Gas Exploration or Refinery company's is global "
    "crude prices, exploration/import policy, and refining margins; a "
    "Cement or Textile company's is construction demand or export demand "
    "and input-cost pressure (energy, cotton); a Fertilizer company's is "
    "gas supply and subsidy policy; a Technology & Communication "
    "company's is IT-export policy and currency effects — reason out the "
    "right angle from what the business description actually says the "
    "company does, don't rely on the sector label alone if the "
    "description reveals something more specific.\n\n"
    "You're also given real sector-performance data (today's average % "
    "change across every sector in the WHOLE PSX market, not just the "
    "selected index) — use this to inform genuine diversification "
    "reasoning: which sectors are currently in favor, out of favor, or "
    "look potentially undervalued after recent underperformance. This is "
    "NOT an instruction to force equal weighting across sectors, avoid "
    "any sector, or hit a specific sector-balance target — it's context "
    "for you to reason with, the same way the other data here is. State "
    "explicitly, for whichever sectors are represented in your mix, why "
    "their current sector-level backdrop supports or doesn't materially "
    "change your view on that specific holding.\n\n"
    "On dividends specifically: a stock with a genuinely stable, "
    "established dividend-paying history can act as a partial hedge "
    "against market volatility or geopolitical/macro uncertainty — "
    "investors often rotate toward reliable dividend payers during "
    "turbulent periods since the dividend itself provides a return "
    "component independent of price movement. Use PSX Dividend 20 Index "
    "membership (shown per symbol) as a real starting signal for this, "
    "then verify via WebSearch whether a candidate has an actual "
    "multi-year track record of consistent (not just recent or one-off) "
    "dividend payments before relying on this characteristic in your "
    "reasoning — don't assume stability from index membership alone."
    "\n\n"
    "You're also given a real Pakistan macro snapshot (GDP growth, "
    "inflation, unemployment from the World Bank; the current PKR/USD "
    "rate; and REAL current KIBOR and government-bond — Market Treasury "
    "Bill and Pakistan Investment Bond — cut-off yields straight from the "
    "State Bank of Pakistan's own rates page, a genuine short-to-long "
    "Pakistani yield curve) — use this as your starting point for any "
    "risk-free-rate or cash-opportunity-cost reasoning, then use "
    "WebSearch specifically for what it doesn't cover: the literal SBP "
    "policy rate figure and its recent trajectory, current IMF program "
    "status, and foreign reserves, since those aren't available as "
    "structured data here."
    "\n\n"
    "PSX-specific mechanics you should verify via WebSearch rather than "
    "assume, since exact rules can change and vary by symbol/margin "
    "class: T+2 settlement, the circuit-breaker/price-limit band shown "
    "per symbol above, minimum lot/board-lot sizes, and Capital Gains Tax "
    "treatment for the account's holding period. Do not assert a specific "
    "numeric trading rule as fact unless you've verified it live or it's "
    "given in the data above."
    "\n\n"
    "You're also given a set of investing-literature principles (position "
    "sizing, portfolio heat, group-exposure limits, stop/profit "
    "discipline), each attributed to the named author who argues for it. "
    "These are advisory, not enforced rules. For each one that's actually "
    "relevant here, briefly explain *why* that author argues for it and "
    "use that reasoning to inform your suggested mix, sizing, and cash "
    "reserve. You may reason your way to a different conclusion than a "
    "principle would suggest, but say so and explain why."
)


_STAGE1_DRAFT_INSTRUCTION = (
    "You are the primary research analyst producing the initial draft in "
    "a multi-stage portfolio-suggestion pipeline. Use WebSearch and "
    "WebFetch extensively, not sparingly. Actively research: Pakistan's "
    "current macroeconomic situation (SBP policy rate and its recent "
    "trajectory, inflation/CPI, PKR exchange-rate stability and any IMF "
    "program status, foreign reserves), political stability, and any "
    "sector-specific or company-specific news (earnings, dividends, "
    "regulatory action) for the symbols under consideration."
    "\n\n"
    "Research depth is a priority, not an afterthought: run a separate, "
    "specific search for the macro picture and for each symbol/sector "
    "under serious consideration — do not settle for one or two broad "
    "queries covering everything at once. Where an initial search only "
    "turns up a weak or generic result, search again with a more specific "
    "query rather than moving on."
    "\n\n"
    "Draft a diversified hypothetical mix (which symbols, roughly what "
    "proportion of capital each) and a cash reserve to keep unallocated. "
    "This draft will be sent to several independent free models for a "
    "critical audit, and you'll then get a chance to revise it based on "
    "their feedback before your final answer goes to the user — so be "
    "thorough and explicit about your reasoning now, not just your "
    "conclusion."
)


_ROLE_STAGE2_SYNTHESIZE = (
    "Below is your own draft suggestion and reasoning from the first "
    "pass, plus independent audit reports from other models that "
    "reviewed your specific draft for flaws, gaps, and disagreements. "
    "Your job now is chiefly revision: read the audit reports, weigh "
    "their specific criticisms against your own original reasoning and "
    "the raw data yourself, and revise your draft into a final "
    "recommendation — for each point you accept, say what changed and "
    "why; for each point you reject, say why you're sticking with your "
    "original call. You remain the final decision-maker, not a rubber "
    "stamp for the audits: use WebSearch/WebFetch yourself to spot-check "
    "any specific claim the audits flagged as contested, stale, or "
    "consequential enough to double-check."
)


_ROLE_STAGE2_SELF_REVIEW = (
    "Below is your own draft suggestion and reasoning from the first "
    "pass. The independent audit that normally reviews it was not "
    "available this run (the free audit models were unreachable or out "
    "of quota), so nobody has checked your reasoning for flaws — that job "
    "now falls back to you. Re-examine your own draft critically: verify "
    "its key claims via WebSearch/WebFetch where practical, and run it "
    "through the failure-mode checklist below the way an independent "
    "auditor would."
    "\n\n"
    "Because the independent audit was unavailable this run, state so "
    "plainly and near the start of your visible answer."
)


_INSTRUCTION_TAIL = (
    "Work through this reasoning internally, in order, before you write "
    "your visible answer — but do NOT print it as separate labeled "
    "stages:\n"
    "1. Run the mix under review through these failure modes "
    "specifically, using real numbers already given above (and WebSearch "
    "where noted):\n"
    "   a. Sector/group concentration and diversification — group the "
    "mix's holdings by sector and state each sector's total % "
    "explicitly. If any sector exceeds roughly 30-40% of capital, "
    "justify why the concentration is warranted or resize it down. "
    "Cross-check this against the sector-performance data given above: "
    "if your mix ended up concentrated in one or two sectors without "
    "having actually reasoned about their current sector-level backdrop, "
    "that's a gap to fix now, not a coincidence to leave unexplained. "
    "Also check the pairwise-correlation data given below the per-symbol "
    "data: if the mix includes both sides of a shown highly-correlated "
    "pair, that's concentrated risk even if the two symbols are in "
    "different sectors — sector grouping alone can miss a cross-sector "
    "correlation driven by a shared macro factor.\n"
    "   b. Liquidity/free-float risk — a low free-float % or thin volume "
    "relative to the position size means the position may be hard to "
    "exit at the quoted price; flag any holding where this applies and "
    "size it more conservatively.\n"
    "   c. Circuit-breaker awareness — the shown circuit-breaker band "
    "limits how far a symbol can move in a single session; factor this "
    "into any stop-loss placed near that boundary (a stop just outside "
    "the band may not fill the day it's triggered).\n"
    "   d. Currency/political risk — construct one adverse scenario "
    "(e.g. a PKR devaluation, a delayed IMF tranche, political "
    "instability) and assess which holdings would be hit hardest and "
    "whether the mix is overexposed to that single scenario.\n"
    "   e. Dividend/ex-date risk — if a holding has an upcoming "
    "dividend or ex-date within a normal holding horizon, note the "
    "expected price adjustment so it isn't mistaken for an adverse "
    "move.\n"
    "   f. Aggregate heat — for every position in the mix, multiply its "
    "% allocation by its stop-loss distance (%) to get that position's "
    "contribution to capital at risk, then sum across the whole mix. "
    "State this total explicitly and check it against a ~6-8% cap.\n"
    "   g. Idle cash — if the mix leaves a meaningful cash reserve, "
    "define specific conditional triggers for deploying it (tied to the "
    "support/resistance levels given above), each with an explicit "
    "time-based fallback.\n"
    "   h. Sector inclusion/exclusion reasoning — explicitly decide, and "
    "be ready to state, WHY each sector you allocated to was chosen "
    "(tie it to that sector's current performance/valuation backdrop "
    "from the sector-performance data above, and to the specific "
    "candidates' own technical/fundamental picture) — not just that it "
    "was chosen. Separately, identify at least the largest 2-3 sectors "
    "that are well-represented among the enriched constituents below but "
    "that your mix allocates NOTHING to, and state why each was left "
    "out (e.g. weak recent/current performance, no attractive candidate "
    "within it after checking, overlaps too much with a sector already "
    "held). An unexplained total absence of a major available sector is "
    "as much a gap as an unexplained concentration.\n"
    "2. Revise the mix based on what step 1 found, incorporating the "
    "strongest points from this checklist pass and from the audit/self-"
    "review findings above.\n"
    "\n"
    "Now write your visible answer as ONE cohesive final recommendation. "
    "For each symbol and for the cash reserve, explain the reasoning "
    "behind that decision inline. Include the sector inclusion/exclusion "
    "reasoning from checklist item (h) above as its own clearly-"
    "identifiable part of the answer (a short section or clearly-signaled "
    "paragraph is fine — it does not need a rigid header), not folded "
    "invisibly into individual symbol justifications where the user would "
    "have to reconstruct it themselves. State clearly, near the start, "
    "that this is a hypothetical, discretionary illustration with no real "
    "account behind it and nothing will be executed automatically — the "
    "user would place any trades manually through their own broker."
    "\n\n"
    "Prioritize thoroughness and rigor over brevity — there is no strict "
    "length limit on this response."
    "\n\n"
    "Write all of your reasoning and explanation as plain prose/markdown "
    "— do not put any of it inside a fenced code block. The ONLY fenced "
    "code block in your entire response must be a single one at the very "
    "end, exactly like this (replace the example values with your actual "
    "final numbers, one key per symbol plus one \"CASH\" key, pct values "
    "summing to 100, no comments or extra text inside the block). Every "
    "non-CASH key must be an object with three numbers: \"pct\" (the "
    "target allocation), \"price\" (a specific, realistic entry price "
    "given the symbol's current price shown above), and \"stop_loss\" (a "
    "specific stop price, derived from volatility or support/resistance "
    "as instructed above since ATR isn't available here):\n"
    "```json\n"
    '{"EXAMPLE_SYMBOL": {"pct": 15, "price": 82.50, "stop_loss": 74.00}, "CASH": 25}\n'
    "```"
)


AUDIT_INSTRUCTION = (
    "You are an independent audit reviewer in a multi-stage PSX (Pakistan "
    "Stock Exchange) portfolio-suggestion pipeline. This is a HYPOTHETICAL "
    "portfolio with no real account behind it. You do NOT have web search "
    "or any live data access — reason only from the market/fundamental "
    "data given below (already gathered by another system), plus Claude's "
    "own first-pass suggested mix and reasoning, produced from that same "
    "data using live web research."
    "\n\n"
    "Your job is NOT to produce your own independent competing mix — it "
    "is to critically audit that draft's specific suggestion and "
    "reasoning:\n"
    "- Check its logic and math: do the stated percentages actually sum "
    "correctly, does the sizing look sound given the volatility/liquidity "
    "data given.\n"
    "- Check whether it adequately addressed: sector concentration, "
    "liquidity/free-float risk, circuit-breaker awareness for any stop "
    "near the band, currency/political risk, dividend/ex-date timing, and "
    "aggregate stop-based risk — flag any of these it skipped or handled "
    "weakly.\n"
    "- Identify anything it got wrong, missed, or reasoned poorly about, "
    "based on the data you both were given.\n"
    "- Note where you would weigh something differently, and why.\n"
    "\n"
    "Produce a structured audit report — agreements, flaws, gaps, and "
    "specific suggested improvements — not a rewritten competing "
    "allocation. Since you have no live data access, don't claim to "
    "fact-check anything beyond what's in the data given here. Keep your "
    "response focused — under 400 words."
)


def build_psx_stage1_instruction() -> str:
    return f"{_INSTRUCTION_HEAD}\n\n{_STAGE1_DRAFT_INSTRUCTION}\n\n{_INSTRUCTION_TAIL}"


def build_psx_stage2_instruction(audit_available: bool) -> str:
    role = _ROLE_STAGE2_SYNTHESIZE if audit_available else _ROLE_STAGE2_SELF_REVIEW
    return f"{_INSTRUCTION_HEAD}\n\n{role}\n\n{_INSTRUCTION_TAIL}"


@dataclass
class PSXAssetAnalysis:
    """Field names deliberately match ai.portfolio_suggest.AssetAnalysis's
    symbol/display_name/prices/stats shape so app.py's existing
    _render_instrument_chart works unchanged for PSX symbols too."""

    symbol: str
    display_name: str | None
    company_name: str | None
    sector_code: str
    sector_name: str
    current: float
    change_pct: float
    volume: int
    prices: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    stats: TechnicalStats = field(
        default_factory=lambda: compute_technical_stats(pd.Series(dtype=float))
    )
    fundamentals: PSXFundamentals | None = None
    financials: PSXFinancials | None = None
    announcements: list[PSXAnnouncement] = field(default_factory=list)
    profile: PSXCompanyProfile | None = None
    benchmark_index: str = DEFAULT_INDEX
    relative_strength_1m_pct: float | None = None
    relative_strength_3m_pct: float | None = None
    beta_vs_index: float | None = None
    week52_position_pct: float | None = None
    is_dividend20_member: bool = False


def analyze_psx_assets(
    assets: list[PSXAsset], index_tag: str = DEFAULT_INDEX
) -> list[PSXAssetAnalysis]:
    """Picks the top config.MAX_PSX_ENRICHED_ASSETS constituents of the
    given index (by volume) for full technical + fundamental + financial-
    statement + company-disclosure + relative-strength enrichment —
    narrowing the pool to a real PSX index (see data.psx_source.
    PSX_INDICES), by explicit user request, rather than always drawing
    from the full ~490-symbol listing. Used here the way a user's own
    curated MT5 Market Watch list sizes the pool on the PMEX side, since
    there's no equivalent per-user curation for PSX."""
    members = [a for a in assets if index_tag in a.listed_in]
    pool = sorted(members, key=lambda a: a.volume, reverse=True)[: config.MAX_PSX_ENRICHED_ASSETS]

    # Fetched once for the whole pool, not per-symbol — relative strength
    # and beta are each stock's own numbers measured against the *same*
    # benchmark series, not something that varies per symbol to fetch.
    # Benchmarked against the same index the pool was drawn from, so both
    # always mean "vs. the index you actually selected", not always vs.
    # KSE-100 regardless of choice.
    index_history = get_psx_history(index_tag)
    index_prices = index_history["Close"] if not index_history.empty else pd.Series(dtype=float)
    index_stats = compute_technical_stats(
        index_prices, history=index_history if not index_history.empty else None
    )

    def _relative(stock_change: float | None, index_change: float | None) -> float | None:
        if stock_change is None or index_change is None:
            return None
        return stock_change - index_change

    analyses = []
    for asset in pool:
        history = get_psx_history(asset.symbol)
        prices = history["Close"] if not history.empty else pd.Series(dtype=float)
        stats = compute_technical_stats(prices, history=history if not history.empty else None)
        company_data = get_psx_company_data(asset.symbol)
        fundamentals = company_data.fundamentals
        week52_low = fundamentals.week52_low if fundamentals else None
        week52_high = fundamentals.week52_high if fundamentals else None
        analyses.append(
            PSXAssetAnalysis(
                symbol=asset.symbol,
                display_name=asset.company_name or asset.symbol,
                company_name=asset.company_name,
                sector_code=asset.sector_code,
                sector_name=get_sector_name(asset.sector_code),
                current=asset.current,
                change_pct=asset.change_pct,
                volume=asset.volume,
                prices=prices,
                stats=stats,
                fundamentals=fundamentals,
                financials=company_data.financials,
                announcements=company_data.announcements,
                profile=company_data.profile,
                benchmark_index=index_tag,
                relative_strength_1m_pct=_relative(stats.change_1m_pct, index_stats.change_1m_pct),
                relative_strength_3m_pct=_relative(stats.change_3m_pct, index_stats.change_3m_pct),
                beta_vs_index=_compute_beta(prices, index_prices),
                week52_position_pct=_compute_week52_position_pct(
                    asset.current, week52_low, week52_high
                ),
                is_dividend20_member="PSXDIV20" in asset.listed_in,
            )
        )
    return analyses


def _fmt(value: float | None, suffix: str = "") -> str:
    return f"{value:.2f}{suffix}" if value is not None else "not available"


def _fmt_series(periods: list[str], values: list[float | None]) -> str:
    return ", ".join(
        f"{period}={value:,.2f}" if value is not None else f"{period}=not available"
        for period, value in zip(periods, values)
    )


def _format_financials(financials: PSXFinancials) -> list[str]:
    lines = []
    if financials.annual_periods:
        lines.append("  financials (annual, real reported figures):")
        for metric, values in financials.annual.items():
            lines.append(f"    {metric}: {_fmt_series(financials.annual_periods, values)}")
    if financials.quarterly_periods:
        lines.append("  financials (trailing quarters, real reported figures):")
        for metric, values in financials.quarterly.items():
            lines.append(f"    {metric}: {_fmt_series(financials.quarterly_periods, values)}")
    if financials.ratio_periods:
        lines.append("  ratios (annual):")
        for metric, values in financials.ratios.items():
            lines.append(f"    {metric}: {_fmt_series(financials.ratio_periods, values)}")
    if not lines:
        lines.append("  financials: not available")
    return lines


def format_psx_asset_context(analyses: list[PSXAssetAnalysis]) -> str:
    lines = []
    for a in analyses:
        s = a.stats
        lines.append(f"- {a.symbol} ({a.display_name}), sector: {a.sector_name}")
        lines.append(
            f"  price: {a.current:.2f} PKR, today's change: {a.change_pct:.2f}%, volume: {a.volume:,}"
        )
        lines.append(
            f"  PSX Dividend 20 Index member: {'yes' if a.is_dividend20_member else 'no'} "
            "(a real, structured signal this is one of PSX's top 20 dividend-paying "
            "stocks by the exchange's own classification — not a yield figure, but "
            "genuine evidence of an established dividend-paying track record; a "
            "'no' does not mean the stock never pays a dividend, just that it isn't "
            "currently one of the top 20)"
        )
        lines.append(
            f"  technical: trend={s.trend or 'not available'}, "
            f"volatility_annualized={_fmt(s.volatility_annualized_pct, '%')}, "
            f"RSI={_fmt(s.rsi)}, volume_trend={_fmt(s.volume_trend_pct, '%')}, "
            f"ATR: not available (no intraday high/low in this data source)"
        )
        lines.append(
            f"  relative strength vs {a.benchmark_index} index: "
            f"1-month={_fmt(a.relative_strength_1m_pct, '%')}, "
            f"3-month={_fmt(a.relative_strength_3m_pct, '%')} "
            "(positive = outperforming that index over that window); "
            f"beta vs {a.benchmark_index}={_fmt(a.beta_vs_index)} "
            "(>1 historically more volatile than the index, <1 less, "
            "negative moves opposite it — not available without ~3 "
            "months of aligned trading history)"
        )
        if s.support is not None and s.resistance is not None:
            lines.append(
                f"  pattern: support={s.support:.2f}, resistance={s.resistance:.2f}, "
                f"market_regime={s.market_regime}"
            )
        f_ = a.fundamentals
        if f_ is not None:
            lines.append(
                f"  fundamentals: P/E={_fmt(f_.pe_ratio)}, "
                f"market_cap={_fmt(f_.market_cap_pkr_000s)} (PKR '000s), "
                f"shares_outstanding={_fmt(f_.shares_outstanding)}, "
                f"free_float={_fmt(f_.free_float_pct, '%')}"
            )
            lines.append(
                "  ROE, ROA, Debt-to-Equity, and Dividend Yield are NOT available from "
                "this data source (no balance-sheet/payout data) — verify via WebSearch "
                "if these matter for a specific decision, don't assume a value."
            )
            if f_.circuit_breaker_low is not None:
                lines.append(
                    f"  circuit breaker band: {f_.circuit_breaker_low:.2f} — {f_.circuit_breaker_high:.2f}"
                )
            if f_.week52_low is not None:
                lines.append(
                    f"  52-week range: {f_.week52_low:.2f} — {f_.week52_high:.2f} "
                    f"(current price is {_fmt(a.week52_position_pct, '%')} of the way "
                    "through this range, 0=52-week low, 100=52-week high)"
                )
        else:
            lines.append("  fundamentals: not available — verify P/E, market cap via WebSearch")

        if a.financials is not None:
            lines += _format_financials(a.financials)
        else:
            lines.append("  financials: not available — verify via WebSearch")

        if a.announcements:
            lines.append("  recent official PSX disclosures:")
            for ann in a.announcements:
                lines.append(f"    [{ann.category}] {ann.date}: {ann.title}")
        else:
            lines.append("  recent official PSX disclosures: none found")

        p_ = a.profile
        if p_ is not None:
            if p_.description:
                lines.append(f"  business: {p_.description}")
            if p_.key_people:
                people = ", ".join(f"{name} ({role})" for name, role in p_.key_people)
                lines.append(f"  key people: {people}")
            if p_.auditor:
                lines.append(f"  auditor: {p_.auditor}")
            if p_.fiscal_year_end:
                lines.append(f"  fiscal year end: {p_.fiscal_year_end}")
        else:
            lines.append("  business profile: not available")
    return "\n".join(lines)


def build_psx_macro_context() -> str:
    """Real Pakistan macro data — GDP growth/inflation/unemployment from
    the World Bank's free keyless API (same source and function already
    used for the PMEX side's 10-country snapshot, just pointed at
    Pakistan), the current PKR/USD rate via yfinance (the same
    fetch_fx_rate_to_usd already live-verified for the PMEX FX-context
    feature), and real KIBOR/government-bond yields straight from the
    State Bank of Pakistan's own rates page — a genuine Pakistani yield
    curve, the closest free structured proxy to the policy rate itself
    (KIBOR tracks it closely in practice). The literal policy rate figure,
    current IMF program status, and FX reserves still aren't available as
    a free structured feed — the prompt explicitly directs WebSearch at
    those instead of asserting a stale/guessed number here."""
    indicators = fetch_country_indicators(countries=("PK",))[0]
    pkr_usd = fetch_fx_rate_to_usd("PKR")
    rates = fetch_pakistan_rates()

    lines = [
        "Pakistan macro snapshot (World Bank, latest available year):",
        f"- GDP growth: {_fmt(indicators.gdp_growth_pct, '%')}",
        f"- Inflation (CPI): {_fmt(indicators.inflation_pct, '%')}",
        f"- Unemployment: {_fmt(indicators.unemployment_pct, '%')}",
        f"- PKR/USD exchange rate: {_fmt(pkr_usd)}",
    ]
    if rates is not None:
        as_of = f" (as of {rates.as_of})" if rates.as_of else ""
        lines.append(f"Pakistan rates from the State Bank of Pakistan{as_of}:")
        if rates.kibor_pct:
            lines.append(
                "- KIBOR (interbank lending benchmark): "
                + ", ".join(f"{tenor}={v:.2f}%" for tenor, v in rates.kibor_pct.items())
            )
        if rates.mtb_yield_pct:
            lines.append(
                "- Market Treasury Bill cut-off yields: "
                + ", ".join(f"{tenor}={v:.2f}%" for tenor, v in rates.mtb_yield_pct.items())
            )
        if rates.pib_yield_pct:
            lines.append(
                "- Pakistan Investment Bond (fixed-rate) cut-off yields: "
                + ", ".join(f"{tenor}={v:.2f}%" for tenor, v in rates.pib_yield_pct.items())
            )
    else:
        lines.append("Pakistan KIBOR/bond yields: not available — verify via WebSearch.")
    lines.append(
        "The literal SBP policy rate figure, current IMF program status, and FX reserves "
        "are NOT available as structured data here — verify current values via WebSearch "
        "rather than assuming a figure."
    )
    return "\n".join(lines)


def build_psx_index_context(index_tag: str = DEFAULT_INDEX) -> str:
    """The selected index's own technical read — a market-wide 'is the
    tape itself bullish, bearish, or choppy right now' signal, distinct
    from any individual stock's own technicals, and the benchmark every
    enriched stock's relative-strength figure is measured against."""
    label = PSX_INDICES.get(index_tag, index_tag)
    s = _fetch_index_stats(index_tag)
    return (
        f"{label} index (the benchmark this suggestion's stocks are drawn from):\n"
        f"- trend={s.trend or 'not available'}, "
        f"1-month change={_fmt(s.change_1m_pct, '%')}, "
        f"3-month change={_fmt(s.change_3m_pct, '%')}, "
        f"RSI={_fmt(s.rsi)}, market_regime={s.market_regime or 'not available'}"
    )


_MIN_SECTOR_MEMBERS_FOR_AVERAGE = 3


def compute_sector_performance(all_assets: list[PSXAsset]) -> list[tuple[str, float, int]]:
    """Average today's % change per sector, across ALL listed PSX symbols
    (not just the narrowed index pool below) — a market-wide sector-
    rotation signal built entirely from data already fetched for every
    symbol (today's change_pct from the market-watch table), no new
    network calls or per-symbol history needed. Deliberately scoped to
    the whole market rather than just the selected index's constituents:
    sector rotation is a market-wide phenomenon, and a narrow index might
    contain only a handful of names in a given sector, understating or
    overstating that sector's real movement today. Sectors with fewer
    than _MIN_SECTOR_MEMBERS_FOR_AVERAGE listed symbols are excluded — an
    "average" of 1-2 thinly-traded names isn't a meaningful sector read.
    Sorted best-performing first."""
    changes_by_sector: dict[str, list[float]] = {}
    for a in all_assets:
        changes_by_sector.setdefault(get_sector_name(a.sector_code), []).append(a.change_pct)

    performance = [
        (sector, sum(changes) / len(changes), len(changes))
        for sector, changes in changes_by_sector.items()
        if len(changes) >= _MIN_SECTOR_MEMBERS_FOR_AVERAGE
    ]
    performance.sort(key=lambda p: p[1], reverse=True)
    return performance


def build_psx_sector_context(all_assets: list[PSXAsset]) -> str:
    performance = compute_sector_performance(all_assets)
    if not performance:
        return "Sector performance: not available."

    lines = [
        "Sector performance today, across the WHOLE PSX market (not just the "
        "selected index's constituents below) — average % change per sector, best "
        "to worst. Use this as a starting point for identifying which sectors are "
        "currently in favor, out of favor, or potentially undervalued after recent "
        "underperformance — this is diversification INPUT, not an instruction to "
        "allocate toward or away from any specific sector, and the actual buildable "
        "mix is still limited to the constituents enriched below regardless of "
        "which sector they're in:"
    ]
    for sector, avg_change, count in performance:
        lines.append(f"- {sector}: {avg_change:+.2f}% avg ({count} symbols)")
    return "\n".join(lines)


def compute_sector_allocation(
    allocation: dict[str, AllocationEntry], analyses: list[PSXAssetAnalysis]
) -> dict[str, float]:
    """Aggregates a final allocation's per-symbol % by real sector name —
    the sector-level view of the same decision the per-symbol allocation
    chart already shows, for a dedicated sector-allocation chart (app.py).
    CASH is its own "Cash" bucket. A symbol the model allocated to that
    isn't in the enriched pool (shouldn't happen given the prompt's
    explicit "only choose from these constituents" scope constraint, but
    not provably impossible) is bucketed as "Unclassified" rather than
    silently dropped or raising — same "disclose, don't hide" convention
    as everywhere else in this file."""
    sector_by_symbol = {a.symbol: a.sector_name for a in analyses}
    totals: dict[str, float] = {}
    for symbol, entry in allocation.items():
        bucket = "Cash" if symbol == "CASH" else sector_by_symbol.get(symbol, "Unclassified")
        totals[bucket] = totals.get(bucket, 0.0) + entry.pct
    return totals


_HIGH_CORRELATION_THRESHOLD = 0.7
_MIN_CORRELATION_OBSERVATIONS = 30  # ~6 trading weeks — same "real window" rule as elsewhere here
_MAX_CORRELATION_PAIRS_SHOWN = 15


def compute_correlation_pairs(analyses: list[PSXAssetAnalysis]) -> list[tuple[str, str, float]]:
    """Pairwise correlation of daily returns among the enriched pool's
    own price histories — a genuinely portfolio-level signal distinct
    from anything computed per-symbol: Modern Portfolio Theory's core
    point is that a stock's own mean return and volatility aren't enough
    to size a *portfolio*, since two individually-reasonable picks that
    move together add concentrated risk rather than real diversification.
    Computed entirely from price series already fetched for each
    symbol's own technical stats — no new network calls.

    Returns only pairs with |correlation| >= _HIGH_CORRELATION_THRESHOLD,
    sorted by magnitude descending and capped — a full N-choose-2 matrix
    for a ~20-symbol pool (190 pairs) is unwieldy prompt content, and the
    actual decision only needs to know which specific pairs move closely
    enough together (or apart) to matter."""
    returns = {}
    for a in analyses:
        prices = a.prices.dropna()
        if len(prices) < _MIN_CORRELATION_OBSERVATIONS:
            continue
        returns[a.symbol] = prices.pct_change().dropna()

    pairs = []
    symbols = list(returns.keys())
    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i + 1 :]:
            aligned = pd.DataFrame({"a": returns[sym_a], "b": returns[sym_b]}).dropna()
            if len(aligned) < _MIN_CORRELATION_OBSERVATIONS:
                continue
            corr = aligned["a"].corr(aligned["b"])
            if corr is not None and not pd.isna(corr) and abs(corr) >= _HIGH_CORRELATION_THRESHOLD:
                pairs.append((sym_a, sym_b, float(corr)))

    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    return pairs[:_MAX_CORRELATION_PAIRS_SHOWN]


def format_correlation_context(analyses: list[PSXAssetAnalysis]) -> str:
    pairs = compute_correlation_pairs(analyses)
    if not pairs:
        return (
            "Pairwise correlation among the enriched pool below: no pair currently "
            f"has |correlation| >= {_HIGH_CORRELATION_THRESHOLD} — no strong "
            "diversification-limiting pairs detected from this data alone (this does "
            "not rule out correlation building during a shared market-wide stress "
            "scenario; reason about that separately)."
        )
    lines = [
        "Pairwise correlation among the enriched pool below (daily returns, only pairs "
        f"with |r| >= {_HIGH_CORRELATION_THRESHOLD} shown) — holding both sides of a "
        "strongly positive pair adds concentrated risk rather than real "
        "diversification; a strongly negative pair can offset risk if held together:"
    ]
    for sym_a, sym_b, corr in pairs:
        direction = "move together" if corr > 0 else "move opposite each other"
        lines.append(f"- {sym_a} & {sym_b}: r={corr:.2f} ({direction})")
    return "\n".join(lines)


def build_psx_summary(
    hypothetical_capital_pkr: float,
    all_assets: list[PSXAsset],
    analyses: list[PSXAssetAnalysis],
    index_tag: str = DEFAULT_INDEX,
) -> str:
    label = PSX_INDICES.get(index_tag, index_tag)
    index_members = [a for a in all_assets if index_tag in a.listed_in]
    movers = sorted(index_members, key=lambda a: abs(a.change_pct), reverse=True)[:10]
    mover_lines = [
        f"- {a.symbol}: {a.current:.2f} PKR ({a.change_pct:+.2f}%), volume {a.volume:,}"
        for a in movers
    ]

    lines = [
        f"Hypothetical capital: {hypothetical_capital_pkr:,.2f} PKR — this is NOT a real "
        "account; no positions exist and nothing will be executed automatically.",
        "",
        build_psx_index_context(index_tag),
        "",
        build_psx_sector_context(all_assets),
        "",
        build_psx_macro_context(),
        "",
        format_book_wisdom(),
        "",
        f"Scope for this suggestion: the {label} index ({len(index_members)} of the "
        f"{len(all_assets)} total listed PSX symbols), narrowed by explicit user choice — "
        f"build the mix ONLY from the {len(analyses)} constituents enriched below (the "
        "largest by volume, up to the enrichment cap), not from the broader ~"
        f"{len(all_assets)}-symbol market.",
        "",
        f"Today's biggest movers within the {label} index (by absolute % change):",
        *mover_lines,
        "",
        f"{label} constituents (technical + fundamental context):",
        format_psx_asset_context(analyses),
        "",
        format_correlation_context(analyses),
    ]
    return "\n".join(lines)


def suggest_psx_portfolio(
    summary: str,
    timeout: int | None = None,
    revision_timeout: int | None = None,
    on_stage: Callable[[str], None] | None = None,
    on_audit_progress: Callable[[str], None] | None = None,
    model: str | None = None,
    save_record: bool = False,
) -> str:
    """Mirrors ai.portfolio_suggest.suggest_portfolio's three-stage draft/
    audit/revise flow exactly, but with PSX-appropriate instructions and
    audit checklist, and its own records directory (config.PSX_RECORDS_DIR)
    so PSX equity audit lessons never mix with PMEX futures ones — the two
    markets' failure modes don't meaningfully transfer between each other."""
    timeout = config.PORTFOLIO_SUGGESTION_TIMEOUT_SECONDS if timeout is None else timeout
    revision_timeout = (
        config.PORTFOLIO_REVISION_TIMEOUT_SECONDS if revision_timeout is None else revision_timeout
    )
    model = config.PORTFOLIO_SUGGESTION_MODEL if model is None else model
    records_dir = Path(config.PSX_RECORDS_DIR)

    def _notify(message: str) -> None:
        if on_stage:
            on_stage(message)

    _notify("Claude is researching PSX and drafting an initial hypothetical mix (live web search)...")
    draft_prompt = f"{build_psx_stage1_instruction()}\n\n{summary}"
    draft = run_claude(
        draft_prompt,
        timeout=timeout,
        allowed_tools=["WebSearch", "WebFetch"],
        model=model,
    )
    if draft == CLI_MISSING_MESSAGE or draft.startswith(CLI_FAILED_PREFIX):
        return draft

    _notify(f"Sending the draft to {len(AUDIT_MODELS)} independent free models for audit...")
    audit = build_audit_block(
        summary,
        draft,
        on_progress=on_audit_progress,
        audit_instruction=AUDIT_INSTRUCTION,
        records_dir=records_dir,
    )
    _notify(
        "Audit received — Claude is revising its suggestion..."
        if audit.audit_available
        else "Independent audit wasn't available this run — Claude is re-checking its own draft instead..."
    )

    revise_prompt = (
        f"{build_psx_stage2_instruction(audit.audit_available)}\n\n{summary}\n\n"
        f"Your own draft from the first pass:\n{draft}\n\n"
        f"Independent audit reports on that draft:\n{audit.block}"
    )
    final_answer = run_claude(
        revise_prompt,
        timeout=revision_timeout,
        allowed_tools=["WebSearch", "WebFetch"],
        model=model,
    )

    if save_record:
        save_portfolio_session(
            SessionRecord(
                summary=summary,
                model=model,
                draft=draft,
                audit_block=audit.block,
                audit_available=audit.audit_available,
                final_answer=final_answer,
            ),
            records_dir=records_dir,
        )

    return final_answer
