import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

import config
from ai.claude_cli import CLI_FAILED_PREFIX, CLI_MISSING_MESSAGE, run_claude
from ai.portfolio_suggest import AUDIT_MODELS, AllocationEntry, build_audit_block, build_past_lessons
from ai.session_record import SessionRecord, save_portfolio_session
from analysis.backtest import (
    BetaStabilityBacktest,
    MomentumPersistenceBacktest,
    RSIReactionBacktest,
    SupportResistanceBacktest,
    VolatilityRegimeBacktest,
    backtest_beta_stability,
    backtest_momentum_persistence,
    backtest_rsi_reaction,
    backtest_support_resistance_reaction,
    backtest_volatility_regime,
    compute_beta,
)
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

logger = logging.getLogger(__name__)

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


@dataclass
class EPSGrowthTrend:
    """A fundamental-momentum read distinct from the raw annual/quarterly
    EPS figures already shown in the financials block: how many
    consecutive REPORTED annual periods (newest first, per
    PSXFinancials' own column order) show EPS growing over the prior
    one, plus the latest annual and quarterly year-over-year % moves.
    Complements (not duplicates) the company's own reported "EPS
    Growth" ratio, if present, by surfacing a multi-year streak the
    single latest-period ratio figure can't show on its own."""

    latest_annual_eps: float | None
    prior_annual_eps: float | None
    annual_yoy_growth_pct: float | None
    consecutive_growth_years: int
    latest_quarterly_eps: float | None
    year_ago_quarterly_eps: float | None
    quarterly_yoy_growth_pct: float | None


def _pct_growth(latest: float | None, prior: float | None) -> float | None:
    # A % growth figure is only meaningful with a positive base — EPS
    # swinging from a loss to a profit (or vice versa) makes a % change
    # figure nonsensical (e.g. -5000 -> 3000 isn't a "+160% loss"), so
    # that case is left as None and the raw EPS values speak for
    # themselves instead.
    if latest is None or prior is None or prior <= 0:
        return None
    return (latest - prior) / prior * 100


_QUARTERS_PER_YEAR = 4


def compute_eps_growth_trend(financials: PSXFinancials | None) -> EPSGrowthTrend | None:
    """Real reported EPS only (never estimated) — None when the company
    page's financial-statement tables don't include an EPS row at all,
    same "don't fabricate from too little" convention as everywhere
    else in this file."""
    if financials is None:
        return None
    annual_eps = financials.annual.get("EPS", [])
    if len(annual_eps) < 2:
        return None

    latest_annual, prior_annual = annual_eps[0], annual_eps[1]

    consecutive_growth_years = 0
    for cur, prev in zip(annual_eps, annual_eps[1:]):
        if cur is None or prev is None or cur <= prev:
            break
        consecutive_growth_years += 1

    quarterly_eps = financials.quarterly.get("EPS", [])
    latest_quarterly = quarterly_eps[0] if quarterly_eps else None
    year_ago_quarterly = (
        quarterly_eps[_QUARTERS_PER_YEAR] if len(quarterly_eps) > _QUARTERS_PER_YEAR else None
    )

    return EPSGrowthTrend(
        latest_annual_eps=latest_annual,
        prior_annual_eps=prior_annual,
        annual_yoy_growth_pct=_pct_growth(latest_annual, prior_annual),
        consecutive_growth_years=consecutive_growth_years,
        latest_quarterly_eps=latest_quarterly,
        year_ago_quarterly_eps=year_ago_quarterly,
        quarterly_yoy_growth_pct=_pct_growth(latest_quarterly, year_ago_quarterly),
    )


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
    "high long-term wealth growth, by taking genuinely CALCULATED risk — "
    "not the minimum risk that technically clears the inflation bar, and "
    "not reckless/unresearched risk either. The user has been explicit, "
    "across multiple rounds of feedback, that they want a more aggressive "
    "posture than earlier runs of this tool actually produced (specifically "
    "flagging a run that ended up roughly half in cash as too "
    "conservative) — but 'more aggressive' means larger, more deliberate "
    "positions backed by real conviction from the data/research above, "
    "not larger positions for their own sake. What makes risk 'calculated' "
    "rather than 'gambling' is unchanged and still fully required: a real "
    "stop-loss on every position, sizing that reflects each position's own "
    "volatility and how strong its supporting evidence actually is (a "
    "high-conviction, well-evidenced case earns a meaningfully larger "
    "size than a marginal one — don't spread capital evenly across "
    "positions regardless of conviction), genuine sector/correlation "
    "diversification (not concentration into one story), and every claim "
    "grounded in the real data/backtests/research given, not a hunch. "
    "Within those real constraints, resolve genuine uncertainty toward "
    "being MORE invested, not less — a mix that ends up mostly cash or "
    "mostly ultra-defensive low-beta names 'to be safe' is failing this "
    "objective just as much as an unresearched gamble would, just less "
    "visibly, since it quietly fails to beat inflation instead of failing "
    "loudly. One specific, common failure mode to actively resist: the "
    "book-wisdom principles below include a generic '~50% of equity in "
    "reserve' cash-reserve default (Murphy) — that is a reasonable "
    "starting anchor for a generic account, but it is NOT this user's "
    "objective, and following it reflexively is exactly the over-"
    "conservative outcome this paragraph is correcting. Absent a "
    "specific, genuinely elevated near-term risk you've identified in "
    "your own research (not a generic 'markets can always fall'), a cash "
    "reserve in roughly the 10-25% range — sized down from that generic "
    "anchor because it's now backed by real per-position conviction, not "
    "abandoned — is a more appropriate default under this objective; if "
    "you do keep more in cash than that, say explicitly why the specific "
    "evidence in front of you warrants it, the same way the checklist "
    "below already asks you to justify any book-wisdom deviation. Say "
    "explicitly, near the start of your visible answer, what the mix's "
    "expected return profile implies relative to that real inflation "
    "figure."
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
    "or out of favor); REAL HISTORICAL BACKTESTS of this specific "
    "instrument's own past behavior — not a generic textbook assumption "
    "— covering exactly the kinds of claims a technical read tempts you "
    "to make: how often its own past RSI overbought/oversold episodes "
    "actually reversed as the textbook convention predicts (with the "
    "real average forward return and reversal rate), whether its beta "
    "has been a stable structural property across different historical "
    "windows or swings substantially (in which case treat the single "
    "current-window beta shown as circumstantial, not reliable), and "
    "whether its own price history shows real momentum persistence or "
    "mean-reversion; whether its own historically LOW-volatility episodes "
    "have actually been followed by BIGGER subsequent moves (supporting "
    "the textbook 'coiled spring' idea that a volatility contraction "
    "precedes a bigger move) or by smaller ones (volatility clustering — "
    "quiet periods tend to stay quiet for this instrument, contradicting "
    "that idea); and how often price has actually held at the shown "
    "support level or been rejected at the shown resistance level "
    "historically, rather than assuming a support/resistance line is "
    "reliable just because it's a well-known charting concept. Ground any "
    "claim you make about RSI, beta, momentum, volatility regime, or "
    "support/resistance reliability for a specific stock in ITS OWN "
    "backtest evidence rather than the generic convention when the two "
    "disagree — say so explicitly when a stock's real history contradicts "
    "the textbook assumption, since that's a genuinely more informed read "
    "than assuming the convention holds everywhere; a real EPS growth "
    "trend (from the same reported annual/quarterly financial-statement "
    "figures shown below, not an estimate) — the latest annual and "
    "quarterly year-over-year EPS % change plus how many consecutive "
    "reported annual periods showed EPS growth immediately before the "
    "latest one; a multi-year growth streak is a materially stronger "
    "fundamental signal than one good year, and a streak that just broke "
    "is worth noting explicitly even if the latest single year still "
    "looks fine in isolation; fundamental context "
    "(P/E ratio, market "
    "capitalization, shares outstanding, free-float %) where available; "
    "PSX's own published circuit-breaker band — a DAILY limit only: it "
    "resets every session around that day's own previous close, so it "
    "is NOT a multi-day or multi-week price ceiling. A multi-week price "
    "target sitting outside today's circuit-breaker band is completely "
    "normal and not a constraint on that target at all — the stock "
    "simply needs more than one session to get there. Never describe a "
    "target as 'within' or 'against' the circuit-breaker band, or size a "
    "target down to fit inside it; the band is only ever relevant to "
    "near-term execution timing (e.g. whether a stop placed close to "
    "today's boundary might not fill the specific day it's triggered) — "
    "and 52-week range PLUS "
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
    "\n\n"
    "On the reward:risk principles specifically (2:1 Bulkowski/"
    "Rockefeller, 3:1 Murphy): the ratio must be an HONEST OUTPUT of a "
    "genuinely-derived entry, stop, and target — never work backward "
    "from the ratio to pick a target. Derive the stop from real "
    "volatility/support first, derive the target from a real technical "
    "level (actual resistance, a real prior high) or a real fundamental "
    "projection — then report whatever ratio actually results, even if "
    "it's below 2:1. A stock that genuinely doesn't offer 2:1 right now "
    "is real information (size it smaller, treat it as lower-conviction, "
    "or leave it out) — stretching its target to an optimistic level "
    "just to clear the ratio manufactures a number instead of reporting "
    "one, and is a worse outcome than honestly saying the ratio falls "
    "short."
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
    "\n\n"
    "End this draft with a compact \"External Research Notes\" list: one "
    "short bullet per genuinely WebSearch/WebFetch-sourced fact you're "
    "relying on (the claim plus its source, e.g. \"SBP held the policy "
    "rate at 11% per the Aug 2026 MPC statement — Business Recorder\"), "
    "not a repeat of your prose. The audit models below have no web "
    "access of their own and can't tell a real researched fact from an "
    "invented one just by reading your prose — this list is what lets "
    "them see exactly which claims are genuine live research (to weigh "
    "and reason over) versus unstated assumptions (to actually flag). "
    "Keep it to the facts that materially influenced a sizing/inclusion "
    "decision, not every search result you looked at."
)


_ROLE_STAGE2_SYNTHESIZE = (
    "Below is your own draft suggestion and reasoning from the first "
    "pass, plus independent audit reports from other models that "
    "reviewed your specific draft for flaws, gaps, and disagreements. "
    "Your job now is chiefly revision: read the audit reports, weigh "
    "their specific criticisms against your own original reasoning and "
    "the raw data yourself, and revise your draft into a final "
    "recommendation. This weighing is INTERNAL working process, not "
    "content for the visible report: decide privately for each critique "
    "whether it changes your view, and let that decision simply improve "
    "the final analysis — do not narrate the back-and-forth (no 'the "
    "audit flagged...', 'upon review, I now think...', 'other models "
    "noted...', 'my initial draft said...' anywhere in the visible "
    "answer). You remain the final decision-maker, not a rubber stamp "
    "for the audits: use WebSearch/WebFetch yourself to spot-check any "
    "specific claim the audits flagged as contested, stale, or "
    "consequential enough to double-check."
    "\n\n"
    "WEIGH EACH AUDIT BY ITS SOURCE MODEL, not equally by default. Each "
    "audit below is labeled with its source model's own approximate "
    "scale and specialization (e.g. '550B params, general-purpose "
    "reasoning' vs. '30B total/3B active MoE, CODING-AGENT-specialized — "
    "weigh its open-ended judgment calls with more caution'). Use this "
    "as a starting prior, not a hard rule: a larger, general-purpose "
    "reasoning model's open-ended judgment call (e.g. 'this sector "
    "allocation looks too concentrated given the macro backdrop') "
    "deserves more default trust than the same kind of judgment call "
    "from a smaller or coding-specialized model, since the latter's own "
    "training targeted a different kind of task. But this prior does "
    "NOT apply to concrete, verifiable points — a real arithmetic error, "
    "a real contradiction against the data given above, a real gap "
    "against the failure-mode checklist — those stand on their own "
    "merits regardless of which model raised them, and a smaller/"
    "specialized model catching something every larger model missed is "
    "still a genuine catch, not a fluke to discount. In short: verify "
    "concrete claims by checking them against the real data (or "
    "WebSearch) rather than by which model said it; use the scale/"
    "specialization label mainly to decide how much extra scrutiny an "
    "open-ended judgment call from a smaller/specialized model deserves "
    "before you accept it. One audit is a different kind of source "
    "entirely: the GitHub Copilot CLI review, if present, has real live "
    "web access and was asked to independently verify your own draft's "
    "External Research Notes claims — treat what it reports as SUPPORTED "
    "or CONTRADICTED with real authority (it's the only reviewer that can "
    "actually confirm or deny a fact, not just judge whether your "
    "reasoning about it sounds internally consistent), and if it flags a "
    "claim as CONTRADICTED, correct or drop that claim rather than "
    "weighing it against the OpenRouter models' opinions. This weighing "
    "is internal analysis, same as the rest of this paragraph — don't "
    "narrate it in the visible answer either (no 'the larger model said "
    "X, the smaller one said Y', no naming which specific model's "
    "critique swayed a decision)."
)


_ROLE_STAGE2_SELF_REVIEW = (
    "Below is your own draft suggestion and reasoning from the first "
    "pass. The independent audit that normally reviews it was not "
    "available this run (the free audit models were unreachable or out "
    "of quota), so nobody has checked your reasoning for flaws — that job "
    "now falls back to you. Re-examine your own draft critically: verify "
    "its key claims via WebSearch/WebFetch where practical, and run it "
    "through the failure-mode checklist below the way an independent "
    "auditor would — internally; this self-check is process, not "
    "content, and shouldn't be narrated in the visible report either."
    "\n\n"
    "Because the independent audit was unavailable this run, disclose "
    "that fact plainly as one sentence within the Executive Summary "
    "section of your visible answer (defined below) — this is the one "
    "piece of process information the user genuinely needs to know, "
    "since it means this run had one less layer of independent review; "
    "everything else about how the answer was produced stays out of the "
    "report."
)


# Session-continuation variants — see ai/portfolio_suggest.py's own copy
# of this comment for the full rationale (identical here): used when
# stage 2 resumes stage 1's own `claude -p --session-id` conversation
# instead of a fresh, fully self-contained call, so _INSTRUCTION_HEAD,
# `summary`, and the stage-1 draft (all unchanged since stage 1, already
# in the model's own conversation history) don't need to be resent.
# Derived via .replace(), not hand-duplicated, so they can't silently
# drift from the originals above.
_ROLE_STAGE2_SYNTHESIZE_CONTINUED = _ROLE_STAGE2_SYNTHESIZE.replace(
    "Below is your own draft suggestion and reasoning from the first "
    "pass, plus independent audit reports from other models that "
    "reviewed your specific draft for flaws, gaps, and disagreements.",
    "Your own draft suggestion and reasoning from the first pass is "
    "already above in this conversation — no need to repeat it. Below "
    "are independent audit reports from other models that reviewed your "
    "specific draft for flaws, gaps, and disagreements.",
)

_ROLE_STAGE2_SELF_REVIEW_CONTINUED = _ROLE_STAGE2_SELF_REVIEW.replace(
    "Below is your own draft suggestion and reasoning from the first "
    "pass. The independent audit that normally reviews it was not "
    "available this run",
    "Your own draft suggestion and reasoning from the first pass is "
    "already above in this conversation. The independent audit that "
    "normally reviews it was not available this run",
)


_INSTRUCTION_TAIL = (
    "Work through this reasoning internally, in order, before you write "
    "your visible answer — but do NOT print it as separate labeled "
    "stages. Do not use headers like 'Draft', 'Internal Review', "
    "'Revision Process', 'Stress-Test', or 'Revised' anywhere in your "
    "response — there is no such thing as a visible internal-review "
    "section; every one of the mandatory headers below is the only "
    "structure this answer has:\n"
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
    "size it more conservatively. Also say so explicitly in that "
    "holding's own thesis section: the entry/stop prices given are clean "
    "theoretical levels, and on a genuinely thin/low-free-float name, "
    "real fills can slip meaningfully away from them — note that a "
    "limit order (not a market order) and some tolerance for a worse-"
    "than-planned fill are appropriate for that specific holding, rather "
    "than presenting the stated prices as if they're guaranteed.\n"
    "   c. Circuit-breaker awareness — the shown circuit-breaker band "
    "limits how far a symbol can move in a SINGLE SESSION only (it "
    "resets daily); factor this into any stop-loss placed near that "
    "boundary (a stop just outside the band may not fill the day it's "
    "triggered) — but never into a multi-day/week price TARGET. If a "
    "target sits outside today's band, that is not a squeeze, a "
    "constraint, or a reason to resize the target down — it just means "
    "the move plays out over more than one session, which is normal for "
    "a fundamentals/macro-driven multi-week thesis.\n"
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
    "State this total explicitly and check it against a ~10-15% cap "
    "(raised from this tool's earlier, more conservative 6-8% cap, per "
    "the calculated-risk objective above — still a real, hard ceiling, "
    "just no longer artificially tighter than the risk this mix's own "
    "diversification and stop discipline can actually support).\n"
    "   g. Idle cash — per the investment objective above, treat any cash "
    "reserve beyond roughly 10-25% of capital as needing its own "
    "explicit justification (a specific, genuinely elevated near-term "
    "risk you identified, not generic caution), the same as an "
    "unusually large single-position bet would need justifying. Whatever "
    "reserve you do leave, define specific conditional triggers for "
    "deploying it (tied to the support/resistance levels given above), "
    "each with an explicit time-based fallback — don't leave it idle "
    "indefinitely.\n"
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
    "Now write your visible answer as a STRUCTURED INVESTMENT REPORT — "
    "the way a professional equity-research or wealth-management note "
    "reads (clear sections, a real narrative arc), not a raw stream of "
    "reasoning and not a transcript of the draft/audit/revise process "
    "above. Use exactly these markdown section headers, in this order; "
    "if a section is genuinely thin for this run, keep the header and "
    "write one honest sentence under it rather than omitting the "
    "section entirely — the structure itself is part of what makes this "
    "readable:\n"
    "## Executive Summary\n"
    "3-5 sentences: your overall market stance, what this mix's expected "
    "return profile implies relative to the real Pakistan inflation "
    "figure given above (per the investment objective stated earlier), "
    "the headline allocation idea, and the single biggest risk to watch "
    "this cycle. State here too, in one sentence, that this is a "
    "hypothetical, discretionary illustration with no real account "
    "behind it and nothing will be executed automatically (and, if "
    "applicable this run, that the independent audit layer wasn't "
    "available — see above).\n"
    "## Macro & Market Backdrop\n"
    "A flowing narrative — interpretation, not a restated bullet list of "
    "the macro/rate numbers given above — on what the real inflation, "
    "KIBOR/bond-yield, and PKR/FX data actually imply for equities right "
    "now, plus whatever political/IMF/policy context you found via "
    "WebSearch.\n"
    "## Sector Outlook\n"
    "The sector inclusion/exclusion reasoning from checklist item (h) "
    "above belongs HERE, as this section's actual content — which "
    "sectors you're constructive on and why, which you're avoiding and "
    "why, tied to the real sector-performance data and to what you found "
    "via research.\n"
    "## Investment Thesis by Position\n"
    "One short subsection per included holding — lead each with the "
    "symbol in bold (e.g. \"**MCB (Commercial Banks):**\") — covering in "
    "flowing prose what the company does, why it earns its place now "
    "(technical + fundamental + business-specific catalyst), and the "
    "sizing/entry/stop rationale. This is where per-position points from "
    "the checklist (liquidity, circuit-breaker awareness, dividend/ex-"
    "date timing, relative strength/beta) belong — applied to that "
    "specific holding, not listed separately. If you're relying on an "
    "RSI, beta, or momentum-based argument for a holding, ground it "
    "explicitly in that instrument's own real historical backtest "
    "evidence given above rather than the generic textbook convention — "
    "and if the audit's backtest scorecard flagged that argument as "
    "CONTRADICTED by this instrument's own history, either drop that "
    "specific argument for this holding (using a different, supported "
    "rationale instead) or explain concretely why you're keeping it "
    "despite the historical evidence against it.\n"
    "## Portfolio Construction & Risk Management\n"
    "Aggregate heat, pairwise correlation, and currency/political "
    "stress-test findings, synthesized as your own risk-management "
    "conclusions about the mix as a whole — not a checklist recitation.\n"
    "## Recommended Allocation\n"
    "A short closing summary of the final numbers (prose or a simple "
    "markdown table), immediately before the required trailing JSON "
    "block. The cash % stated here MUST exactly equal the JSON block's "
    "\"CASH\" value below — if a position is conditional/not-yet-"
    "triggered (e.g. a buy-on-pullback target not hit yet), it is NOT "
    "in the JSON and its % must still count as cash in both places; "
    "don't state a lower cash % in prose than the JSON actually shows "
    "just because a pending target is discussed elsewhere in the report.\n"
    "## Outlook & Triggers to Revisit\n"
    "Idle-cash deployment triggers (with their required time-based "
    "fallback) and what would change this view going forward.\n"
    "\n"
    "Throughout, write in ONE confident, single-voice analyst register — "
    "never reference the multi-stage or multi-model process that "
    "produced this answer (no 'the audit flagged...', 'upon revision...', "
    "'other models noted...', 'my draft said...'). Every conclusion, "
    "whichever pass it originated in, is presented simply as this "
    "report's own analysis; a reader should not be able to tell this was "
    "a multi-stage process at all — that's process, and process isn't "
    "content."
    "\n\n"
    "Prioritize thoroughness and rigor over brevity — there is no strict "
    "length limit on this response."
    "\n\n"
    "Write all of your reasoning and explanation as plain prose/markdown "
    "using the section headers specified above — do not put any of it "
    "inside a fenced code block. The ONLY fenced code block in your "
    "entire response must be a single one at the very end (after the "
    "'Recommended Allocation' section), exactly like this (replace the "
    "example values with your actual final numbers, one key per symbol "
    "plus one \"CASH\" key, pct values summing to 100, no comments or "
    "extra text inside the block). Every non-CASH key must be an object "
    "with four numbers: \"pct\" (the target allocation), \"price\" (a "
    "specific, realistic entry price given the symbol's current price "
    "shown above — weigh missed-fill risk against price improvement: a "
    "price shaded meaningfully below the current quote in the hope of a "
    "pullback can mean the position is simply never entered if the stock "
    "instead runs, which is a real cost for your highest-conviction "
    "ideas specifically, not a free option), \"stop_loss\" (a "
    "specific stop price, derived from volatility or support/resistance "
    "as instructed above since ATR isn't available here), and "
    "\"take_profit\" (a specific target price on the correct side of "
    "your entry — above it for a long — matching the exact target level "
    "your own reward:risk reasoning above already derives):\n"
    "```json\n"
    '{"EXAMPLE_SYMBOL": {"pct": 15, "price": 82.50, "stop_loss": 74.00, "take_profit": 96.00}, "CASH": 25}\n'
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
    "The draft ends with an \"External Research Notes\" list — each line "
    "there is a fact the draft-writer found via a real live web search "
    "you cannot independently repeat or verify. Not being able to verify "
    "one of these is NOT the same as it being fabricated or "
    "unreliable: treat it as a real, reasonably trustworthy input and "
    "focus your scrutiny on how it was WEIGHTED or APPLIED to a sizing/"
    "inclusion decision, not on the mere fact that you can't check it "
    "yourself. Reserve genuine skepticism for numeric claims that appear "
    "in the draft's prose but nowhere in this Notes list or the "
    "structured data given below — an unlisted, oddly-precise figure "
    "(e.g. a specific yield % or growth rate cited without a matching "
    "Notes entry) is a real gap worth flagging.\n\n"
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
    "BACKTEST THE DRAFT'S UNDERLYING LOGIC, not just its arithmetic. The "
    "draft doesn't just state numbers — it implies a small system of "
    "cause-and-effect rules about how these instruments behave (e.g. "
    "'this stock's positive relative strength means it should keep "
    "outperforming', 'this overbought reading means a pullback is likely', "
    "'this beta means it will move roughly twice as much as the index'). "
    "Treat the draft as implicitly claiming a function — given a "
    "condition X (a technical reading, a regime), it asserts an expected "
    "market response Y — and you have real historical evidence below to "
    "test specific values of X against, the same way you'd test any "
    "claimed function by plugging in inputs and checking the outputs it "
    "actually produced in the past:\n"
    "1. For each holding, identify the specific technical/behavioral "
    "claim(s) the draft is relying on to justify it (momentum "
    "continuing, a reversal being likely, beta implying a certain risk "
    "level, etc.).\n"
    "2. Cross-check EACH such claim against that exact instrument's own "
    "historical backtest evidence given below (its real past RSI-"
    "reaction rate and average forward return, whether its beta has been "
    "stable or has swung across different historical windows, whether its "
    "own history shows real momentum persistence or mean-reversion, "
    "whether its own low-volatility episodes have historically been "
    "followed by bigger or smaller moves — the 'coiled spring' question — "
    "and how often price has actually held at support or been rejected at "
    "resistance historically) — this is genuine historical evidence for "
    "THIS instrument specifically, not a generic textbook assumption. If "
    "the draft leans on an EPS growth story for a holding, cross-check it "
    "against the real reported EPS growth trend given below too (the "
    "consecutive-growth-year count and latest annual/quarterly % changes) "
    "— a draft claiming 'strong earnings momentum' when the streak just "
    "broke, or citing one good quarter while ignoring a longer decline, is "
    "a real gap to flag.\n"
    "3. Score each claim you checked: SUPPORTED (the historical evidence "
    "agrees with the draft's implied logic), CONTRADICTED (the "
    "instrument's own history shows the opposite — e.g. the draft treats "
    "an overbought reading as bearish but this instrument's own reversal "
    "rate after past overbought episodes is well under 50%, the draft "
    "leans on beta as a stable risk measure but the backtest shows it "
    "swinging widely across windows, the draft treats low volatility as a "
    "'coiled spring' setup but this instrument's low-vol episodes have "
    "historically been followed by SMALLER moves, or the draft leans on a "
    "support/resistance level for its stop/entry logic despite this "
    "instrument's own history showing that level rarely holds), or "
    "UNTESTABLE (not enough real historical episodes were available to "
    "judge either way — say so rather than guessing). Cite the actual "
    "numbers you're basing this on.\n"
    "4. A CONTRADICTED score is a real, concrete flaw to raise — treat it "
    "with the same weight as a math error, not a minor stylistic note.\n"
    "\n"
    "Produce a structured audit report — agreements, flaws, gaps, the "
    "backtest scorecard from above, and specific suggested improvements "
    "— not a rewritten competing allocation. Since you have no live data "
    "access, don't claim to fact-check anything beyond what's in the "
    "data given here (the historical backtests ARE data given here, not "
    "something you're fetching yourself). Keep your response focused — "
    "under 550 words."
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
    rsi_overbought_backtest: RSIReactionBacktest | None = None
    rsi_oversold_backtest: RSIReactionBacktest | None = None
    beta_stability_backtest: BetaStabilityBacktest | None = None
    momentum_persistence_backtest: MomentumPersistenceBacktest | None = None
    volatility_regime_backtest: VolatilityRegimeBacktest | None = None
    support_resistance_backtest: SupportResistanceBacktest | None = None
    eps_growth_trend: EPSGrowthTrend | None = None


def analyze_psx_assets(
    assets: list[PSXAsset],
    index_tag: str = DEFAULT_INDEX,
    on_progress: Callable[[str], None] | None = None,
) -> list[PSXAssetAnalysis]:
    """Picks the top config.MAX_PSX_ENRICHED_ASSETS constituents of the
    given index (by volume) for full technical + fundamental + financial-
    statement + company-disclosure + relative-strength enrichment —
    narrowing the pool to a real PSX index (see data.psx_source.
    PSX_INDICES), by explicit user request, rather than always drawing
    from the full ~490-symbol listing. Used here the way a user's own
    curated MT5 Market Watch list sizes the pool on the PMEX side, since
    there's no equivalent per-user curation for PSX.

    `on_progress`, if given, is called once per pool constituent with a
    single updating message — same contract as
    ai.portfolio_suggest.analyze_assets' own param."""
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
    total = len(pool)
    for i, asset in enumerate(pool, start=1):
        if on_progress is not None:
            on_progress(f"Analyzing PSX constituents: {i}/{total} — {asset.symbol}")
        history = get_psx_history(asset.symbol)
        prices = history["Close"] if not history.empty else pd.Series(dtype=float)
        stats = compute_technical_stats(prices, history=history if not history.empty else None)
        company_data = get_psx_company_data(asset.symbol)
        fundamentals = company_data.fundamentals
        week52_low = fundamentals.week52_low if fundamentals else None
        week52_high = fundamentals.week52_high if fundamentals else None
        rsi_overbought_bt, rsi_oversold_bt = backtest_rsi_reaction(prices)
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
                beta_vs_index=compute_beta(prices, index_prices),
                week52_position_pct=_compute_week52_position_pct(
                    asset.current, week52_low, week52_high
                ),
                is_dividend20_member="PSXDIV20" in asset.listed_in,
                rsi_overbought_backtest=rsi_overbought_bt,
                rsi_oversold_backtest=rsi_oversold_bt,
                beta_stability_backtest=backtest_beta_stability(prices, index_prices),
                momentum_persistence_backtest=backtest_momentum_persistence(prices),
                volatility_regime_backtest=backtest_volatility_regime(prices),
                support_resistance_backtest=backtest_support_resistance_reaction(prices),
                eps_growth_trend=compute_eps_growth_trend(company_data.financials),
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


def _format_rsi_backtest(bt: RSIReactionBacktest | None, condition: str) -> str:
    if bt is None:
        return (
            f"  historical {condition} RSI reaction: not enough real historical episodes "
            "in this instrument's own history to compute — treat any RSI-reversal claim "
            "for it as unverified assumption, not evidence."
        )
    return (
        f"  historical {condition} RSI reaction (real, this instrument's own past): "
        f"{bt.occurrences} distinct past episodes where RSI reached {bt.threshold:.0f}, "
        f"average {bt.forward_days}-trading-day return afterward = "
        f"{bt.avg_forward_return_pct:+.2f}%, reversed as the textbook convention would "
        f"predict {bt.reversal_rate_pct:.0f}% of the time"
    )


def _format_backtests(a: PSXAssetAnalysis) -> list[str]:
    lines = [_format_rsi_backtest(a.rsi_overbought_backtest, "overbought")]
    lines.append(_format_rsi_backtest(a.rsi_oversold_backtest, "oversold"))

    bs = a.beta_stability_backtest
    if bs is not None and bs.stable is not None:
        windows = ", ".join(
            f"{label}={_fmt(value)}"
            for label, value in (
                ("3m", bs.beta_3m),
                ("6m", bs.beta_6m),
                ("1y", bs.beta_1y),
                ("full history", bs.beta_full_history),
            )
            if value is not None
        )
        verdict = (
            "STABLE — beta looks like a genuine structural property, not a fluke of one window"
            if bs.stable
            else "UNSTABLE — beta swings substantially across windows, so treat the single "
            "current-window beta shown above as circumstantial, not a reliable constant"
        )
        lines.append(f"  historical beta stability across windows ({windows}): {verdict}")
    else:
        lines.append("  historical beta stability: not enough aligned history to compute")

    mp = a.momentum_persistence_backtest
    if mp is not None:
        lines.append(
            f"  historical momentum pattern (this instrument's own history, "
            f"{mp.sample_size} independent ~1-month periods): correlation between a "
            f"period's own return and the NEXT period's return = {mp.correlation:+.2f} "
            f"-> {mp.interpretation.replace('_', ' ')} "
            "(persistent = past winners tended to keep winning; mean_reverting = past "
            "winners tended to give it back; no_clear_pattern = neither reliably)"
        )
    else:
        lines.append("  historical momentum pattern: not enough history to compute")

    vr = a.volatility_regime_backtest
    if vr is not None:
        lines.append(
            f"  historical volatility-regime reaction (this instrument's own past "
            f"{vr.low_vol_episodes} low-volatility and {vr.high_vol_episodes} "
            f"high-volatility episodes, {vr.forward_days}-trading-day forward move): "
            f"avg move after LOW-vol episodes = {vr.low_vol_avg_abs_move_pct:.2f}%, "
            f"avg move after HIGH-vol episodes = {vr.high_vol_avg_abs_move_pct:.2f}% -> "
            + (
                "supports the 'coiled spring' reading (quiet periods historically precede "
                "bigger moves for this instrument)"
                if vr.low_vol_avg_abs_move_pct > vr.high_vol_avg_abs_move_pct
                else "CONTRADICTS the 'coiled spring' reading (this instrument's own low-"
                "volatility periods have historically been followed by SMALLER moves, not "
                "bigger ones — volatility has clustered/persisted instead)"
            )
        )
    else:
        lines.append("  historical volatility-regime reaction: not enough history to compute")

    sr = a.support_resistance_backtest
    if sr is not None:
        lines.append(
            f"  historical support/resistance reliability (this instrument's own past, "
            f"{sr.forward_days}-trading-day forward check): support held (price higher "
            f"afterward) {sr.support_hold_rate_pct:.0f}% of {sr.support_tests} real past "
            f"tests; resistance rejected (price lower afterward) "
            f"{sr.resistance_reject_rate_pct:.0f}% of {sr.resistance_tests} real past "
            "tests — use this to judge how much weight the support/resistance range shown "
            "above deserves for THIS instrument specifically, rather than assuming "
            "support/resistance lines are reliable just because they're a well-known "
            "charting concept."
        )
    else:
        lines.append(
            "  historical support/resistance reliability: not enough real historical "
            "tests of these levels to compute"
        )

    eg = a.eps_growth_trend
    if eg is not None:
        annual_growth = (
            f"{eg.annual_yoy_growth_pct:+.1f}%"
            if eg.annual_yoy_growth_pct is not None
            else "not meaningful (base period EPS was zero/negative)"
        )
        quarterly_growth = (
            f"{eg.quarterly_yoy_growth_pct:+.1f}%"
            if eg.quarterly_yoy_growth_pct is not None
            else "not available or not meaningful"
        )
        lines.append(
            f"  EPS growth trend (real reported figures): latest annual EPS="
            f"{_fmt(eg.latest_annual_eps)} vs prior year {_fmt(eg.prior_annual_eps)} "
            f"({annual_growth}); {eg.consecutive_growth_years} consecutive reported annual "
            f"period(s) of EPS growth immediately preceding the latest; latest quarterly "
            f"EPS={_fmt(eg.latest_quarterly_eps)} vs year-ago quarter "
            f"{_fmt(eg.year_ago_quarterly_eps)} ({quarterly_growth})"
        )
    else:
        lines.append("  EPS growth trend: not available — verify via WebSearch")
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
        lines += _format_backtests(a)
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


def _fetch_current_psx_price(symbol: str) -> float | None:
    """The PSX-specific `fetch_current_price` callable for
    build_past_outcome_lessons — PSX symbols are already native (no
    ticker-resolution step needed, unlike PMEX), so this is just the
    latest close from the same endpoint every other PSX price read in
    this codebase uses. None on an empty/failed fetch, silently skipped
    by the caller rather than fabricated."""
    history = get_psx_history(symbol)
    if history.empty:
        return None
    return float(history["Close"].iloc[-1])


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

    # A caller-generated session ID lets stage 2 RESUME stage 1's own
    # conversation (see below) instead of paying to resend _INSTRUCTION_
    # HEAD, the whole `summary` market/macro dump, and the stage-1 draft
    # a second time — none of that changed between the two calls, and
    # live-verified `claude -p --resume` correctly retains it (including
    # specific WebSearch-found figures) without needing it repeated.
    session_id = str(uuid.uuid4())
    # Computed ONCE, before stage 1 even runs — see ai/portfolio_suggest.py
    # ::suggest_portfolio's own copy of this comment for the full
    # rationale (identical here, just pointed at PSX's own records dir and
    # its own native-symbol price fetcher).
    _notify(
        "Building past-session context (real market outcomes and audit "
        "lessons from previous PSX runs)..."
    )
    past_lessons = build_past_lessons(fetch_current_price=_fetch_current_psx_price, records_dir=records_dir)
    _notify("Claude is researching PSX and drafting an initial hypothetical mix (live web search)...")
    draft_prompt = f"{build_psx_stage1_instruction()}\n\n{summary}"
    if past_lessons:
        draft_prompt += f"\n\n{past_lessons}"
    draft = run_claude(
        draft_prompt,
        timeout=timeout,
        allowed_tools=["WebSearch", "WebFetch"],
        model=model,
        session_id=session_id,
    )
    if draft == CLI_MISSING_MESSAGE or draft.startswith(CLI_FAILED_PREFIX):
        return draft

    _notify(f"Sending the draft to {len(AUDIT_MODELS)} independent free models for audit...")
    audit = build_audit_block(
        summary,
        draft,
        on_progress=on_audit_progress,
        audit_instruction=AUDIT_INSTRUCTION,
        past_lessons=past_lessons,
    )
    _notify(
        "Audit received — Claude is revising its suggestion..."
        if audit.audit_available
        else "Independent audit wasn't available this run — Claude is re-checking its own draft instead..."
    )

    role_continued = (
        _ROLE_STAGE2_SYNTHESIZE_CONTINUED if audit.audit_available else _ROLE_STAGE2_SELF_REVIEW_CONTINUED
    )
    lean_revise_prompt = (
        f"{role_continued}\n\n{_INSTRUCTION_TAIL}\n\n"
        f"Independent audit reports on your draft above:\n{audit.block}"
    )
    final_answer = run_claude(
        lean_revise_prompt,
        timeout=revision_timeout,
        allowed_tools=["WebSearch", "WebFetch"],
        model=model,
        resume_session_id=session_id,
    )

    # Never let an infrastructure hiccup in the lean path (session
    # expired/evicted between calls, a CLI version without --resume
    # support, etc.) degrade the actual answer — fall back to the fully
    # self-contained prompt, which needs no session at all, so the worst
    # case is exactly the pre-existing behavior, not a worse one.
    if final_answer == CLI_MISSING_MESSAGE or final_answer.startswith(CLI_FAILED_PREFIX):
        logger.warning(
            "suggest_psx_portfolio: resumed stage-2 call failed (%s), falling back to a full-context retry",
            final_answer[:200],
        )
        _notify(
            "The quick revision attempt didn't respond — retrying with a "
            "fresh, fully self-contained request (takes a bit longer, but "
            "doesn't rely on the earlier session still being live)..."
        )
        revise_prompt = (
            f"{build_psx_stage2_instruction(audit.audit_available)}\n\n{summary}\n\n"
            f"Your own draft from the first pass:\n{draft}\n\n"
            f"Independent audit reports on that draft:\n{audit.block}"
        )
        if past_lessons:
            revise_prompt += f"\n\n{past_lessons}"
        final_answer = run_claude(
            revise_prompt,
            timeout=revision_timeout,
            allowed_tools=["WebSearch", "WebFetch"],
            model=model,
        )

    if save_record:
        _notify("Saving session record for future reference...")
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
