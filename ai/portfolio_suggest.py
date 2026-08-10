import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

import config
from ai.claude_cli import CLI_FAILED_PREFIX, CLI_MISSING_MESSAGE, run_claude
from ai.openrouter_client import FAILED_MESSAGE as OPENROUTER_FAILED_MESSAGE
from ai.openrouter_client import MISSING_KEY_MESSAGE as OPENROUTER_MISSING_KEY_MESSAGE
from ai.openrouter_client import run_openrouter
from ai.session_record import SessionRecord, save_portfolio_session
from analysis.technical import TechnicalStats, compute_technical_stats
from data.book_wisdom import format_book_wisdom
from data.commodity_geography import INDEX_LINKED_COUNTRIES, METAL_LINKED_COUNTRIES
from data.crop_context import CROP_TRADE_PROFILES, fetch_crop_supply_demand_context
from data.macro_source import fetch_country_indicators, fetch_fx_rate_to_usd, fetch_market_indicators
from data.market_history import fetch_price_history_ohlcv
from data.mt5_source import AccountSummary, ContractSpec, MarketAsset, Position, get_contract_spec
from data.news_source import fetch_recent_headlines
from data.underlying import resolve_yahoo_ticker

logger = logging.getLogger(__name__)

ENERGY_COMMODITIES = {"Crude Oil", "Natural Gas"}

_INSTRUCTION_HEAD = (
    "You are a portfolio-construction assistant for a personal PMEX futures "
    "trading account that currently holds no open positions. You have "
    "WebSearch and WebFetch tools available."
    "\n\n"
    "Source quality matters — prioritize in this order:\n"
    "1. Official/primary sources: the IMF, World Bank, central banks (US "
    "Federal Reserve, ECB, Bank of England, Bank of Japan, People's Bank "
    "of China, Reserve Bank of India, Bank of Korea, Saudi Central Bank, "
    "UAE Central Bank), national statistics/finance-ministry portals (e.g. "
    "US BLS/BEA, UK ONS, Germany's Destatis, Japan's Cabinet Office, "
    "China's NBS, India's MOSPI, Korea's KOSTAT, Taiwan's DGBAS), and "
    "recognized data providers (e.g. ADP's employment report).\n"
    "2. Major investment banks' and monetary agencies' public research/"
    "commentary when actually available (e.g. JPMorgan, Goldman Sachs, "
    "Morgan Stanley, Citigroup, HSBC, UBS, Deutsche Bank, BIS) — read what "
    "they've actually published, don't guess at their view.\n"
    "3. Major, reputable financial/general news outlets: Bloomberg, "
    "Reuters, Financial Times, The Economist, CNBC, BBC, CNN, CBC, The New "
    "York Times, The Washington Post.\n"
    "4. Social media/forums (X/Twitter, Reddit, financial forums) — useful "
    "for catching real-time sentiment shifts and influential commentary "
    "before slower outlets report on it, but treat it as lower-confidence "
    "than tiers 1-3: verify a notable claim against a primary/news source "
    "before treating it as fact, and when citing a post, name the actual "
    "account/subreddit and platform rather than a vague 'social media is "
    "saying...'.\n"
    "Do not cite Wikipedia or low-quality/unverified blogs as a source. If "
    "a genuinely credible source can't be found for something, say so "
    "rather than falling back to a weak one or inventing a citation."
    "\n\n"
    "If a 'Current Open Positions' section appears below, the account is "
    "not starting from empty — treat those as real, already-committed "
    "capital, not a hypothetical. Your final mix must explicitly reconcile "
    "against every one of them (keep at current size, trim, add to, or "
    "close), not propose a fresh allocation as if they didn't exist. When "
    "no such section appears, the account currently holds nothing and "
    "you're proposing a mix from a clean slate."
    "\n\n"
    "You're also given: a macro snapshot (US Treasury yield curve, dollar "
    "index, VIX, and GDP growth/inflation/unemployment for 10 major "
    "economies — US, UK, France, Germany, Japan, China, India, South "
    "Korea, Saudi Arabia, UAE), per-instrument technical context — a "
    "short-term trend vs. 20-day moving average, annualized volatility, "
    "Average True Range (a real, computed volatility-in-price-units "
    "figure — use it, not an assumed flat percentage, for stop distances), "
    "14-day RSI (momentum — conventionally overbought above 70, oversold "
    "below 30), a volume trend (recent vs. this instrument's own 20-day "
    "average — note this reflects the broader global market this data "
    "comes from, not PMEX's own order flow specifically), AND a "
    "medium-term (~3-month, daily-resolution) pattern read: a support/"
    "resistance range and a market-type classification "
    "('trending_up'/'trending_down' when price has moved fairly directly "
    "in one direction over that window, 'sideways' when it's mostly "
    "chopped in a range, 'mixed' in between). These price/support/"
    "resistance levels come from a third-party market-data source and may "
    "use a different quote convention than this account's own instrument "
    "(e.g. US grain futures like corn are commonly quoted in cents per "
    "bushel, so a level shown as 402 there means $4.02/bushel) — before "
    "using any such level as a trigger price or citing it in your final "
    "answer, sanity-check its scale/units against a quick WebSearch for "
    "that instrument's current real-world price, and convert if needed; "
    "plus recent headlines "
    "where available, and a feasibility line: the REAL minimum-lot margin "
    "requirement (in the account's own margin currency) as a % of "
    "account equity, flagged 'NOT AFFORDABLE' past 100%. This is a hard "
    "constraint, not a preference — on a small account, a single "
    "instrument's minimum tradable lot can require several times the "
    "entire account's equity, making any % allocation to it literally "
    "impossible to execute, not just impractical. And — for any crops currently "
    "traded — a crop supply/demand section with real recent rainfall and "
    "short-term forecast data for that crop's major exporting and "
    "importing countries (a simplified single-region-per-country proxy, "
    "not precise agronomic modeling); reason about whether that pattern "
    "suggests a supply surplus/shortfall (e.g. a major exporter showing "
    "unusually low rainfall going into a forecast dry spell is a bullish "
    "supply signal). There is also a 'Research directives' section — "
    "follow those using WebSearch/WebFetch specifically (e.g. oil "
    "chokepoint conditions, named commodity companies' outlook, and the "
    "specific producer/consumer countries linked to whichever metals or "
    "indices are traded — search each named country for anything recent "
    "and relevant: mining strikes, sanctions, power/supply disruptions, "
    "trade policy, demand trends), applying the same source-quality and "
    "honesty rules above. If an instrument "
    "relates to a major economy not already in the macro snapshot (e.g. "
    "Taiwan), dig into that country's own government/statistics portal "
    "too. Use all of this — the macro data, the per-instrument data, the "
    "crop weather data, and what you find via search — as real reasoning "
    "input, not decoration."
    "\n\n"
    "For each instrument, also cross-reference its price pattern (the "
    "short-term trend and the medium-term support/resistance/market-type) "
    "against the macro snapshot, crop weather data where relevant, and the "
    "news/government-policy developments you find for its linked countries "
    "— over that same multi-month window. Use this to explain how the "
    "asset appears to have been reacting to these kinds of events or "
    "policy moves, and what that implies for forward-looking risk (e.g. "
    "'this has tended to move sharply on this country's rate decisions; "
    "given the current policy stance, watch for X'). This is qualitative "
    "pattern-matching from what you actually find, not a formal backtest — "
    "say so plainly if a connection you're drawing is speculative rather "
    "than clearly evidenced."
    "\n\n"
    "You're also given a set of investing-literature principles (position "
    "sizing, portfolio heat, group-exposure limits, stop/profit "
    "discipline), each attributed to the named author who argues for it. "
    "These are advisory, not enforced rules — nothing in this codebase "
    "requires following them. For each one that's actually relevant to "
    "the instruments/account here, briefly explain *why* that author "
    "argues for it (the reasoning given alongside each principle below) "
    "and use that reasoning — not just the bare rule — to inform your "
    "suggested mix, sizing, and cash reserve. You may reason your way to "
    "a different conclusion than a principle would suggest, but say so "
    "and explain why, rather than silently ignoring it."
)


_STAGE1_DRAFT_INSTRUCTION = (
    "You are the primary research analyst producing the initial draft in "
    "a multi-stage portfolio-suggestion pipeline. Use WebSearch and "
    "WebFetch extensively, not sparingly. Actively research the last ~6 "
    "months of geopolitical developments (wars, floods, droughts, "
    "tariffs, trade-route disruption, tax policy changes) that could "
    "affect the instruments below."
    "\n\n"
    "Research depth is a priority, not an afterthought: run a separate, "
    "specific search for each instrument, each linked country, and each "
    "named institution/company below — do not settle for one or two broad "
    "queries covering everything at once. Where an initial search only "
    "turns up a weak or generic result, search again with a more specific "
    "query (e.g. the country/topic name plus a date range or the specific "
    "policy/event) rather than moving on. Aim to ground the final answer "
    "in noticeably more distinct credible sources than a shallow pass "
    "would produce — breadth across topics matters as much as depth on "
    "any one of them. Include a specific search for social-media/forum "
    "commentary (X/Twitter, Reddit — e.g. r/wallstreetbets, r/investing, "
    "r/Forex, or a relevant commodity-specific subreddit) on the "
    "instruments and countries involved: a single widely-shared post from "
    "a known trader/analyst account can move markets faster than "
    "traditional news picks it up, so this is a genuine research target, "
    "not an afterthought."
    "\n\n"
    "Draft a diversified starting mix (which instruments, roughly what "
    "proportion of equity each) and a cash reserve to keep unallocated. "
    "Before including any instrument, check its feasibility line: only "
    "include it if the minimum lot is actually affordable. Where the lot "
    "granularity doesn't allow a smooth percentage (common here — whole "
    "lots only, no fractional sizing), state the position as a realistic "
    "whole-lot count and the actual resulting percentage, not an "
    "arbitrary smooth target that doesn't correspond to any valid order. "
    "If no combination of minimum lots fits the account at all, say so "
    "explicitly instead of drafting an impossible allocation. The mix "
    "under review below (for the failure-mode checklist) is this draft "
    "you're producing now."
    "\n\n"
    "This draft will be sent to several independent free models for a "
    "critical audit, and you'll then get a chance to revise it based on "
    "their feedback before your final answer goes to the user — so be "
    "thorough and explicit about your reasoning now, not just your "
    "conclusion, to give the audit something substantial to check."
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
    "original call rather than silently ignoring it. You do not need to "
    "repeat your original research from scratch, but you remain the "
    "final decision-maker, not a rubber stamp for the audits: use "
    "WebSearch/WebFetch yourself to spot-check any specific claim the "
    "audits flagged as contested, stale, or consequential enough to "
    "double-check before you rely on it. The mix under review below is "
    "your own draft from the first pass — revise it, don't restart from "
    "a blank page."
)


_ROLE_STAGE2_SELF_REVIEW = (
    "Below is your own draft suggestion and reasoning from the first "
    "pass. The independent audit that normally reviews it was not "
    "available this run (the free audit models were unreachable or out "
    "of quota), so nobody has checked your reasoning for flaws — that job "
    "now falls back to you. Re-examine your own draft critically: verify "
    "its key claims via WebSearch/WebFetch where practical, and run it "
    "through the failure-mode checklist below the way an independent "
    "auditor would, rather than simply restating it. The mix under "
    "review below is your own draft from the first pass."
    "\n\n"
    "Because the independent audit was unavailable this run, state so "
    "plainly and near the start of your visible answer (for example: "
    "\"the independent audit of my draft was not available for this run, "
    "so I re-checked it myself\") so the user understands this run had "
    "one less layer of independent review."
)


_INSTRUCTION_TAIL = (
    "Work through this reasoning internally, in order, before you write "
    "your visible answer — but do NOT print it as separate labeled "
    "stages. Do not use headers like 'Draft', 'Stress-Test', or 'Revised' "
    "anywhere in your response:\n"
    "1. Run the mix under review through these failure modes "
    "specifically, using real numbers already given above (and WebSearch "
    "where noted) rather than generic caveats. Skip a point only if it's "
    "genuinely not applicable (e.g. no FX section means the account is "
    "already USD-denominated):\n"
    "   a. FX/base-currency mismatch — if an FX section is present above, "
    "the account's own currency differs from the USD pricing/settlement "
    "of most instruments; consider how a currency move could distort "
    "returns or trigger stops independent of the instrument's own price "
    "action.\n"
    "   b. Futures roll yield — every instrument here is a *dated* "
    "futures contract (visible in its symbol/expiry), not a spot "
    "buy-and-hold. For EVERY instrument you're including, not just the "
    "largest or most-discussed one, use WebSearch to check whether it's "
    "currently in contango or backwardation and estimate the implied "
    "roll cost/benefit (as an approximate %/month) — a mix spanning "
    "several asset classes (e.g. energy and precious metals) can have "
    "opposite roll dynamics per class, so checking only one class and "
    "assuming the rest are the same is a real gap, not a shortcut. If a "
    "'Multi-expiry contracts detected' section appears above, compute the "
    "real roll spread from those two actual live quotes for that specific "
    "pair instead of relying only on a generic web-searched assumption — "
    "genuine broker data takes priority over an inferred proxy wherever "
    "both exist.\n"
    "   c. Correlation under stress — construct one adverse macro "
    "scenario from the yield/DXY/VIX data given above (e.g. a real-yield "
    "spike or a DXY surge) and assess whether the mix's positions "
    "would move together more than their individual weights suggest.\n"
    "   d. Execution/liquidity risk — each instrument above has a "
    "bid-ask spread% shown; identify any wide-spread instrument in the "
    "mix and factor in recommending limit/stop-limit orders over market "
    "orders where that matters. That spread is a single snapshot, not a "
    "guarantee — spreads, especially on micro-lot contracts, can widen "
    "further during off-peak/lower-liquidity trading hours, so factor "
    "that into your order-type and timing guidance rather than treating "
    "the shown number as a ceiling.\n"
    "   e. Idle cash — if the mix leaves a meaningful cash reserve, "
    "define specific conditional triggers for deploying it, tied to the "
    "support/resistance levels already given per instrument above (e.g. "
    "'deploy X% if [instrument] pulls back to its support level' or "
    "'deploy X% on a confirmed breakout above [instrument]'s resistance'). "
    "Every such trigger must also have an explicit time-based fallback "
    "(e.g. 'if not triggered within N trading days, do X instead') — an "
    "open-ended trigger that could leave cash idle indefinitely is itself "
    "a weakness. If the FX section above is present, also research the "
    "account country's current short-term risk-free/policy rate via "
    "WebSearch and explicitly weigh idle cash's opportunity cost against "
    "it (and against staying invested) when sizing the reserve — don't "
    "size cash as an arbitrary leftover percentage.\n"
    "   f. Position sizing vs. volatility, and hedge feasibility — size "
    "positions smaller, relative to equal-risk peers, for instruments "
    "showing higher annualized volatility% above. Only propose a hedge or "
    "alternative instrument that's actually listed among the Market Watch "
    "instruments below — don't suggest options, ETFs, or any other "
    "product that isn't shown as tradable on this account.\n"
    "   g. Aggregate heat — for every position in the mix, multiply its "
    "% allocation by its stop-loss distance (%) to get that position's "
    "contribution to capital at risk, then sum these across the whole "
    "mix. State this total explicitly (e.g. 'aggregate heat: 5.2% of "
    "equity') and check it against a ~6-8% cap: if a simultaneous "
    "worst-case stop-out across every position would erase more than "
    "that, resize down until it doesn't, rather than sizing positions "
    "independently of each other.\n"
    "   h. Sector/group concentration — group the mix's positions by "
    "correlated market/sector (e.g. precious metals, energy, grains, "
    "equity indices) and sum the % allocation within each group. State "
    "each group's total explicitly (e.g. 'metals + energy: 48% of "
    "equity') rather than leaving it implicit in the per-instrument "
    "numbers. If any group exceeds roughly 30-40% of equity, you may "
    "still keep it that way, but you must explicitly justify why the "
    "concentration is warranted here (not just note that it exists) or "
    "resize it down.\n"
    "   i. Asset-class exclusion — if an entire asset class is present "
    "and tradable in the Market Watch instruments below (e.g. equity "
    "indices, when the mix is otherwise all commodities) but ends up at "
    "0% in your final mix, state that exclusion explicitly and justify "
    "it, the same way you'd justify an inclusion — don't let an absence "
    "go unexplained just because there's no number to attach it to.\n"
    "2. Revise the mix based on what step 1 found, incorporating whatever "
    "the strongest points were from this checklist pass, and from the "
    "audit/self-review findings above, whichever applied this run.\n"
    "\n"
    "Now write your visible answer as ONE cohesive final recommendation — "
    "not the three steps above shown separately. For each instrument and "
    "for the cash reserve, explain the reasoning behind that decision "
    "inline (weaving in whichever of the considerations above are "
    "relevant to it) as part of justifying the number, the way an analyst "
    "would write a finished recommendation rather than show their scratch "
    "work. No deterministic sizing or risk rules are wired up yet for "
    "this feature, so clearly state that the final answer is still an "
    "early-stage, discretionary starting point for discussion, not a "
    "precise or final recommendation."
    "\n\n"
    "Prioritize thoroughness and rigor over brevity — there is no strict "
    "length limit on this response. Take the time you need to research "
    "and reason properly; a longer, well-evidenced answer is preferred "
    "over a shorter, shallower one. Stay organized (clear paragraphs or "
    "bullet points per instrument) and avoid needless repetition."
    "\n\n"
    "Write all of your reasoning and explanation as plain prose/markdown "
    "(paragraphs, bullet points, bold text) — do not put any of it inside "
    "a fenced code block. The ONLY fenced code block in your entire "
    "response must be a single one at the very end, exactly like this "
    "(replace the example values with your actual final numbers, one key "
    "per instrument symbol traded above plus one \"CASH\" key, pct values "
    "summing to 100, no comments or extra text inside the block). Every "
    "non-CASH key must be an object with three numbers: \"pct\" (the "
    "target allocation), \"price\" (a specific limit-order entry price — "
    "it must be realistic and achievable given the instrument's current "
    "bid/ask shown above, not a distant support/resistance level or an "
    "arbitrary round number), and \"stop_loss\" (a specific stop price). "
    "When a real ATR figure is shown for that instrument above, derive "
    "the stop distance from it (e.g. price minus roughly 1.5x ATR for a "
    "long, per Bulkowski's principle above) rather than an assumed flat "
    "percentage — a stop distance that ignores an instrument's own actual "
    "daily volatility risks being either needlessly tight or far looser "
    "than intended. Only fall back to a flat-percentage stop when ATR is "
    "explicitly marked not available for that instrument, and say so. If "
    "an instrument is currently held (see Current Open Positions above) "
    "but you're not including it in the final mix, it must still appear "
    "here with \"pct\": 0 — never omit a currently-held instrument just "
    "because you're recommending closing it; an omission cannot be "
    "distinguished from an oversight, and this pipeline treats an "
    "omitted held instrument as being closed:\n"
    "```json\n"
    '{"EXAMPLE_SYMBOL": {"pct": 15, "price": 82.50, "stop_loss": 78.00}, "CASH": 25}\n'
    "```"
)


def build_stage1_instruction() -> str:
    return f"{_INSTRUCTION_HEAD}\n\n{_STAGE1_DRAFT_INSTRUCTION}\n\n{_INSTRUCTION_TAIL}"


def build_stage2_instruction(audit_available: bool) -> str:
    role = _ROLE_STAGE2_SYNTHESIZE if audit_available else _ROLE_STAGE2_SELF_REVIEW
    return f"{_INSTRUCTION_HEAD}\n\n{role}\n\n{_INSTRUCTION_TAIL}"


def _spread_pct(a: MarketAsset) -> float | None:
    """Bid-ask spread as a % of price — real, already-available execution-
    risk data (wide spreads mean market orders can fill far from quote)."""
    if not a.ask:
        return None
    return (a.ask - a.bid) / a.ask * 100


@dataclass
class AssetAnalysis:
    """Everything computed for one Market Watch instrument — single source
    of truth shared by the prompt text builder and the chart renderer, so
    charting doesn't re-fetch what the text-building already fetched."""

    symbol: str
    description: str
    bid: float
    ask: float
    display_name: str | None
    prices: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    stats: TechnicalStats = field(
        default_factory=lambda: compute_technical_stats(pd.Series(dtype=float))
    )
    headlines: list[str] = field(default_factory=list)
    contract_spec: ContractSpec | None = None


def analyze_assets(assets: list[MarketAsset]) -> list[AssetAnalysis]:
    """Resolve, fetch, and compute technical stats for the first assets (up
    to config.MAX_ENRICHED_ASSETS) that map to a Yahoo ticker. Instruments
    without a mapping, or beyond the cap, still get an entry (display_name
    = None) so callers can list them plainly without a network round-trip.
    Contract spec (real order-size/margin constraints) is fetched for
    every asset regardless of Yahoo mapping — it only needs the PMEX
    symbol via the already-open MT5 connection, no extra network call."""
    results = []
    enriched_count = 0
    for a in assets:
        contract_spec = get_contract_spec(a.symbol)

        resolved = None
        if enriched_count < config.MAX_ENRICHED_ASSETS:
            resolved = resolve_yahoo_ticker(a.symbol, a.description)

        if resolved is None:
            results.append(
                AssetAnalysis(a.symbol, a.description, a.bid, a.ask, None, contract_spec=contract_spec)
            )
            continue

        display_name, yahoo_ticker = resolved
        enriched_count += 1

        history = fetch_price_history_ohlcv(yahoo_ticker)
        prices = history["Close"]
        stats = compute_technical_stats(prices, history=history)
        headlines = fetch_recent_headlines(yahoo_ticker, limit=config.NEWS_HEADLINES_PER_ASSET)

        results.append(
            AssetAnalysis(
                a.symbol, a.description, a.bid, a.ask, display_name,
                prices, stats, headlines, contract_spec,
            )
        )

    return results


def _feasibility_line(r: AssetAnalysis, account_equity: float | None) -> str | None:
    """Whether the minimum tradable lot is actually affordable — a real
    constraint (order-size/margin), not a preference. On a small account,
    a single instrument's minimum lot can require several times the
    entire account's equity, making any % allocation to it impossible,
    not just impractical (confirmed live: PALDIUM100-SE26's minimum lot
    needed ~3.7x a 1,000,000 PKR account).

    Also states margin_initial (the raw per-1.0-lot rate) and volume_min/
    volume_step directly, not just the pre-multiplied min-lot figure —
    confirmed live that giving only the multiplied number forces the
    model to reverse-engineer a per-lot rate to size any position beyond
    the literal minimum lot, which produced a 3x margin error in one run
    (it can now just multiply margin_initial by however many lots it
    wants, respecting volume_step).

    When account_equity is known but the contract spec itself couldn't be
    fetched (confirmed live: happened for SP500-SE26 specifically, every
    single run for a full day, while every other instrument's spec
    resolved fine — see get_contract_spec's own logging for the real
    reason), this returns an explicit "not available" line rather than
    None — a silently omitted line here previously meant this instrument
    got a free pass on the affordability check no other instrument got,
    for no better reason than a data gap the model was never told about."""
    if account_equity is None or account_equity <= 0:
        return None
    if not r.contract_spec:
        return (
            "  feasibility: not available (contract spec could not be fetched for "
            "this symbol) — its minimum-lot affordability has NOT been verified; "
            "do not assume it's affordable without checking some other way before "
            "including it"
        )
    spec = r.contract_spec
    min_lot_margin = spec.margin_initial * spec.volume_min
    pct = min_lot_margin / account_equity * 100
    flag = "NOT AFFORDABLE — " if pct > 100 else ""
    return (
        f"  feasibility: {flag}min lot margin ~{min_lot_margin:,.0f} "
        f"{spec.currency_margin} ({pct:.1f}% of equity); "
        f"margin per 1.0 lot ~{spec.margin_initial:,.0f} {spec.currency_margin} "
        f"(multiply this directly for any lot count); "
        f"min lot size {spec.volume_min:g}, lot step {spec.volume_step:g}"
    )


def format_enriched_asset_context(
    analyses: list[AssetAnalysis], account_equity: float | None = None
) -> str:
    """Pure formatting over already-computed AssetAnalysis objects — same
    text shape as before this was split out of a single fetch+format loop,
    plus an optional feasibility line when account_equity is supplied."""
    lines = []
    for r in analyses:
        spread_pct = _spread_pct(r)
        feasibility_line = _feasibility_line(r, account_equity)

        if r.display_name is None:
            lines.append(f"- {r.symbol} ({r.description}): bid {r.bid}, ask {r.ask}")
            if spread_pct is not None:
                lines.append(f"  execution: bid-ask spread {spread_pct:.2f}% of price")
            if feasibility_line is not None:
                lines.append(feasibility_line)
            lines.append(
                "  technical: not available (no price-history source mapped for this "
                "symbol) — do not assume a volatility or ATR figure for it; verify via "
                "WebSearch before using it in position sizing or setting its stop"
            )
            continue

        lines.append(f"- {r.symbol} ({r.display_name}): bid {r.bid}, ask {r.ask}")
        if spread_pct is not None:
            lines.append(f"  execution: bid-ask spread {spread_pct:.2f}% of price")
        if feasibility_line is not None:
            lines.append(feasibility_line)

        stats = r.stats
        if stats.last_price is not None:
            stat_bits = []
            if stats.trend is not None:
                stat_bits.append(f"{stats.trend} ({stats.pct_vs_sma20:+.1f}% vs 20d SMA)")
            if stats.change_1m_pct is not None:
                stat_bits.append(f"1M {stats.change_1m_pct:+.1f}%")
            if stats.change_3m_pct is not None:
                stat_bits.append(f"3M {stats.change_3m_pct:+.1f}%")
            if stats.change_6m_pct is not None:
                stat_bits.append(f"6M {stats.change_6m_pct:+.1f}%")
            if stats.volatility_annualized_pct is not None:
                stat_bits.append(
                    f"volatility {stats.volatility_annualized_pct:.1f}% (annualized)"
                )
            else:
                stat_bits.append(
                    "volatility not available (insufficient price history) — verify "
                    "via WebSearch before sizing"
                )
            if stats.atr_pct is not None:
                stat_bits.append(
                    f"ATR {stats.atr_pct:.2f}% of price (14-day) — use this, not a "
                    "flat percentage, as the basis for this instrument's stop distance"
                )
            else:
                stat_bits.append(
                    "ATR not available (insufficient price history) — verify a real "
                    "ATR via WebSearch before setting this instrument's stop"
                )
            if stats.rsi is not None:
                stat_bits.append(f"RSI {stats.rsi:.0f} (14-day, >70 overbought/<30 oversold)")
            else:
                stat_bits.append("RSI not available (insufficient price history)")
            if stats.volume_trend_pct is not None:
                stat_bits.append(
                    f"volume {stats.volume_trend_pct:+.0f}% vs its own 20-day average "
                    "(global market volume, not PMEX-specific)"
                )
            else:
                stat_bits.append("volume trend not available (no volume data for this source)")
            if stat_bits:
                lines.append(f"  technical: {', '.join(stat_bits)}")

            pattern_bits = []
            if stats.support is not None and stats.resistance is not None:
                pattern_bits.append(
                    f"support {stats.support:.2f} / resistance {stats.resistance:.2f} "
                    f"(range {stats.range_width_pct:.1f}% of price, over ~3 months)"
                )
            if stats.market_regime is not None:
                pattern_bits.append(f"market type: {stats.market_regime}")
            if pattern_bits:
                lines.append(f"  pattern: {', '.join(pattern_bits)}")
        else:
            lines.append(
                "  technical: not available (price-history fetch failed) — do not "
                "assume a volatility or ATR figure for it; verify via WebSearch "
                "before using it in position sizing or setting its stop"
            )
        for headline in r.headlines:
            lines.append(f"  news: {headline}")

    return "\n".join(lines)


def build_enriched_asset_context(assets: list[MarketAsset]) -> str:
    """Convenience wrapper kept for existing callers/tests — prefer calling
    analyze_assets() once and format_enriched_asset_context() when the raw
    analyses are also needed elsewhere (e.g. charting), to avoid re-fetching."""
    return format_enriched_asset_context(analyze_assets(assets))


def build_macro_snapshot() -> str:
    """US yield curve/DXY/VIX plus GDP growth/inflation/unemployment for
    the US, UK, Japan, and China. Missing values are simply omitted rather
    than faked — see data/macro_source.py for why a given field can be None."""
    market = fetch_market_indicators()
    countries = fetch_country_indicators()

    lines = ["Macro snapshot:"]

    yield_bits = []
    if market.yield_3m_pct is not None:
        yield_bits.append(f"3M {market.yield_3m_pct:.2f}%")
    if market.yield_10y_pct is not None:
        yield_bits.append(f"10Y {market.yield_10y_pct:.2f}%")
    if market.yield_30y_pct is not None:
        yield_bits.append(f"30Y {market.yield_30y_pct:.2f}%")
    if yield_bits:
        line = f"- US Treasury yields: {', '.join(yield_bits)}"
        if market.yield_curve_10y_3m_spread is not None:
            line += f" (10Y-3M spread {market.yield_curve_10y_3m_spread:+.2f}pp)"
        lines.append(line)
    if market.dxy is not None:
        lines.append(f"- US Dollar Index (DXY): {market.dxy:.2f}")
    if market.vix is not None:
        lines.append(f"- VIX (volatility/fear index): {market.vix:.2f}")

    for c in countries:
        bits = []
        if c.gdp_growth_pct is not None:
            bits.append(f"GDP growth {c.gdp_growth_pct:.1f}%")
        if c.inflation_pct is not None:
            bits.append(f"inflation {c.inflation_pct:.1f}%")
        if c.unemployment_pct is not None:
            bits.append(f"unemployment {c.unemployment_pct:.1f}%")
        if bits:
            lines.append(f"- {c.country}: {', '.join(bits)}")

    if len(lines) == 1:
        lines.append("- macro data unavailable right now")

    return "\n".join(lines)


def _detect_present_commodities(assets: list[MarketAsset]) -> set[str]:
    """Distinct resolved display names (e.g. {"Wheat", "Crude Oil", "Gold"})
    actually present in the live Market Watch. Single detection point that
    both the crop weather context and the research directives key off of,
    so coverage always matches what the broker actually offers rather than
    a fixed list."""
    names = set()
    for a in assets:
        resolved = resolve_yahoo_ticker(a.symbol, a.description)
        if resolved is not None:
            names.add(resolved[0])
    return names


def build_crop_supply_demand_context(assets: list[MarketAsset]) -> str:
    """Weather-based supply/demand context for whichever crops (from
    CROP_TRADE_PROFILES) the broker's Market Watch actually contains right
    now. Empty string if none are currently traded."""
    present = _detect_present_commodities(assets)
    crops_present = present & CROP_TRADE_PROFILES.keys()
    if not crops_present:
        return ""
    return fetch_crop_supply_demand_context(crops_present)


def build_research_directives(assets: list[MarketAsset]) -> str:
    """Pure text, no fetching — points the already-granted WebSearch/
    WebFetch access at the specific things asked about, scoped to what's
    actually traded rather than always searching for everything."""
    present = _detect_present_commodities(assets)
    directives = []

    if present & ENERGY_COMMODITIES:
        directives.append(
            "- Energy is traded here: research current conditions, safety, "
            "and traffic trends (increasing/decreasing/normal) at major oil "
            "chokepoints (Strait of Hormuz, Strait of Malacca, Suez Canal, "
            "Panama Canal), and recent geopolitics among major producers "
            "(OPEC+ members, Russia, US shale)."
        )
        directives.append(
            "- Look for recent public outlook/commentary from major energy "
            "companies (e.g. ExxonMobil, Shell, BP, Saudi Aramco)."
        )

    if present & CROP_TRADE_PROFILES.keys():
        directives.append(
            "- Crops are traded here: look for recent public outlook/"
            "commentary from major agricultural traders/processors (e.g. "
            "ADM, Bunge, Cargill, Louis Dreyfus Company) and any current "
            "drought/flood/export-ban news for the exporter/importer "
            "countries listed in the crop supply/demand section below."
        )

    linked_countries_by_asset = {**METAL_LINKED_COUNTRIES, **INDEX_LINKED_COUNTRIES}
    for asset_name in sorted(present & linked_countries_by_asset.keys()):
        countries = linked_countries_by_asset[asset_name]
        directives.append(
            f"- {asset_name} is traded here: it's most linked to "
            f"{', '.join(countries)} — research recent developments there "
            "specifically (mining strikes, sanctions, power/supply "
            "disruptions, trade policy, demand trends) rather than only "
            "generic global commentary."
        )

    if not directives:
        return ""

    return "Research directives:\n" + "\n".join(directives)


_EXPIRY_SYMBOL_PATTERN = re.compile(r"^(?P<base>.+)-(?P<expiry>[A-Z]{2}\d{2})$")


def build_multi_expiry_context(assets: list[MarketAsset]) -> str:
    """Detects instruments where the same underlying trades at more than
    one contract expiry in this account's own Market Watch (PMEX's own
    symbol convention appears to be "{BASE}-{2-letter-month}{2-digit-
    year}", e.g. MAIZELD-AU26 vs MAIZELD-JY26 — confirmed live in this
    account's own data) and surfaces every expiry's real live quote side
    by side.

    This is genuine, broker-verified data for computing an actual roll
    yield/contango-backwardation spread for this specific pair — unlike
    a generic web-searched commodity-wide assumption, it's this account's
    own tradable contracts. Deliberately does NOT guess which expiry is
    nearer/farther from the month code alone — PMEX's own code-to-
    calendar-month convention isn't something this codebase has verified,
    and a wrong guess here would silently flip the contango/backwardation
    sign. The instruction using this asks the model to confirm the actual
    expiry order via WebSearch/PMEX's own contract calendar first."""
    groups: dict[str, list[MarketAsset]] = {}
    for asset in assets:
        match = _EXPIRY_SYMBOL_PATTERN.match(asset.symbol)
        if match:
            groups.setdefault(match.group("base"), []).append(asset)

    multi_expiry = {base: group for base, group in groups.items() if len(group) > 1}
    if not multi_expiry:
        return ""

    lines = [
        "Multi-expiry contracts detected (same underlying, different "
        "contract months, both real live quotes from this account's own "
        "Market Watch — use these to compute an actual roll yield/"
        "contango-backwardation spread for this specific pair rather than "
        "a generic commodity-wide web-searched assumption; confirm via "
        "WebSearch/PMEX's own contract calendar which expiry is nearer "
        "before concluding the direction, since the symbol's own month "
        "code isn't verified against PMEX's calendar here):"
    ]
    for base, group in multi_expiry.items():
        parts = ", ".join(f"{a.symbol} (bid {a.bid}, ask {a.ask})" for a in group)
        lines.append(f"- {base}: {parts}")
    return "\n".join(lines)


def build_fx_context(account: AccountSummary) -> str:
    """Only present when the account's own currency isn't USD — states the
    raw exchange rate as a fact for the model to reason about (base-
    currency vs. USD-priced-instrument mismatch), not a computed risk
    score. Empty string for a USD account (nothing to flag)."""
    if account.currency.upper() == "USD":
        return ""
    rate = fetch_fx_rate_to_usd(account.currency)
    if rate is None:
        return (
            f"FX: account is denominated in {account.currency}, but most "
            "instruments below are USD-priced/settled — live exchange "
            "rate unavailable right now, but the mismatch itself still "
            "applies."
        )
    return (
        f"FX: account is denominated in {account.currency} "
        f"(1 USD ~ {rate:.2f} {account.currency} right now), but most "
        "instruments below are USD-priced/settled — a currency move can "
        f"distort {account.currency}-denominated returns independent of "
        "the instrument's own price action."
    )


def build_positions_context(positions: list[Position]) -> str:
    """Current open positions, if any — feeds the AI's target mix so it
    reasons about keeping/trimming/closing what's actually held instead
    of proposing a fresh mix as if the account were empty. Empty string
    when there are no open positions (nothing to reconcile against)."""
    if not positions:
        return ""
    lines = ["Current Open Positions (real, already held — reconcile the "
             "final mix against these, don't treat the account as empty):"]
    for p in positions:
        pnl_pct = -p.adverse_move_pct
        lines.append(
            f"- {p.symbol}: {p.side} {p.volume:g} lots, opened at "
            f"{p.price_open:.2f}, now {p.price_current:.2f} "
            f"({pnl_pct:+.1f}%, {p.profit:+.2f} unrealized)"
            + (f", stop at {p.sl:.2f}" if p.sl is not None else ", no stop set")
        )
    return "\n".join(lines)


def build_portfolio_summary(
    account: AccountSummary,
    assets: list[MarketAsset],
    positions: list[Position] | None = None,
    analyses: list[AssetAnalysis] | None = None,
) -> str:
    """`analyses` lets a caller (e.g. app.py, for charting) compute
    analyze_assets(assets) once and reuse it here, instead of this
    function re-fetching the same data internally. `positions` (if
    supplied) lets the final mix reconcile against what's actually held
    — see build_positions_context."""
    lines = [
        f"Account: balance {account.balance:.2f} {account.currency}, "
        f"equity {account.equity:.2f}, free margin {account.free_margin:.2f}",
        "",
        format_book_wisdom(),
        "",
        build_macro_snapshot(),
    ]

    positions_context = build_positions_context(positions or [])
    if positions_context:
        lines += ["", positions_context]

    fx_context = build_fx_context(account)
    if fx_context:
        lines += ["", fx_context]

    crop_context = build_crop_supply_demand_context(assets)
    if crop_context:
        lines += ["", crop_context]

    directives = build_research_directives(assets)
    if directives:
        lines += ["", directives]

    multi_expiry_context = build_multi_expiry_context(assets)
    if multi_expiry_context:
        lines += ["", multi_expiry_context]

    lines += ["", "Tradable instruments (Market Watch):"]
    if not assets:
        lines.append("- none visible in Market Watch")
    else:
        resolved_analyses = analyze_assets(assets) if analyses is None else analyses
        lines.append(format_enriched_asset_context(resolved_analyses, account_equity=account.equity))

    return "\n".join(lines)


_ALLOCATION_BLOCK_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _last_allocation_match(response_text: str) -> re.Match | None:
    """The one ```json block that actually represents the final allocation
    — always the *last* fenced json block in the response (the prompt
    asks for it at the very end). Shared by parse/strip below so they
    never disagree about which block is "the" allocation block."""
    matches = list(_ALLOCATION_BLOCK_PATTERN.finditer(response_text))
    return matches[-1] if matches else None


@dataclass
class AllocationEntry:
    pct: float
    price: float | None = None
    stop_loss: float | None = None


def parse_final_allocation(response_text: str) -> dict[str, AllocationEntry] | None:
    """Extract the trailing ```json {...}``` allocation block the prompt
    asks for (see build_stage1_instruction/build_stage2_instruction).
    Each non-CASH value is an object with pct/price/stop_loss (see the
    JSON-block spec in _INSTRUCTION_TAIL); CASH is a bare number. Also
    tolerates a bare number for a non-CASH key (price/stop_loss come back
    None) in case the model reverts to the old shape — a missing price is
    a real limitation for execution, not a parse failure, so this still
    parses rather than dropping the whole allocation. Returns None on
    anything unexpected — missing block, malformed JSON, wrong shape —
    rather than raising; the text response still displays fine even if
    this fails, it just means no allocation chart/execution plan."""
    match = _last_allocation_match(response_text)
    if match is None:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not data:
        return None

    result: dict[str, AllocationEntry] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            return None
        if isinstance(value, (int, float)):
            result[key] = AllocationEntry(pct=float(value))
            continue
        if not isinstance(value, dict) or not isinstance(value.get("pct"), (int, float)):
            return None
        price = value.get("price")
        stop_loss = value.get("stop_loss")
        if price is not None and not isinstance(price, (int, float)):
            return None
        if stop_loss is not None and not isinstance(stop_loss, (int, float)):
            return None
        result[key] = AllocationEntry(
            pct=float(value["pct"]),
            price=float(price) if price is not None else None,
            stop_loss=float(stop_loss) if stop_loss is not None else None,
        )
    return result


def strip_allocation_block(response_text: str) -> str:
    """Removes only that same last ```json allocation block so the
    displayed prose doesn't show a trailing raw JSON dump. Deliberately
    does NOT remove every ```json-fenced block in the text — if the model
    used a code fence earlier for other legitimate content, that content
    stays visible. (Previously used .sub() to strip every match, which
    could silently delete large chunks of the actual explanation if the
    model used more than one fenced block — fixed here.)"""
    match = _last_allocation_match(response_text)
    if match is None:
        return response_text.strip()
    return (response_text[: match.start()] + response_text[match.end() :]).strip()


AUDIT_INSTRUCTION = (
    "You are an independent audit reviewer in a multi-stage "
    "portfolio-suggestion pipeline. You do NOT have web search or any "
    "live data access — reason only from the account, market, macro, "
    "contract-feasibility, and research data given below (already "
    "gathered by another system), plus Claude's own first-pass suggested "
    "portfolio mix and reasoning, produced from that same data using live "
    "web research (WebSearch/WebFetch)."
    "\n\n"
    "Your job is NOT to produce your own independent competing mix — it "
    "is to critically audit that draft's specific suggestion and "
    "reasoning:\n"
    "- Check its logic and math: do the stated percentages actually sum "
    "correctly, does every included instrument respect its feasibility "
    "line, does the sizing look sound given the volatility/technical data "
    "given.\n"
    "- Check whether it adequately addressed each of these: FX/"
    "base-currency mismatch, futures roll yield (contango/backwardation), "
    "correlation under stress, execution/liquidity risk, idle-cash "
    "deployment triggers with time-based fallbacks, and "
    "position-sizing-vs-volatility/hedge-feasibility — flag any of these "
    "it skipped or handled weakly.\n"
    "- Identify anything it got wrong, missed, or reasoned poorly about, "
    "based on the data you both were given.\n"
    "- Note where you would weigh something differently, and why.\n"
    "\n"
    "Produce a structured audit report — agreements, flaws, gaps, and "
    "specific suggested improvements — not a rewritten competing "
    "allocation. Since you have no live data access, don't claim to "
    "fact-check anything beyond what's in the data given here; audit its "
    "reasoning and internal consistency, not facts you can't verify. Keep "
    "your response focused — under 400 words."
)


def get_openrouter_audit(
    summary: str,
    draft_suggestion: str,
    model: str,
    timeout: int | None = None,
    past_lessons: str = "",
    audit_instruction: str = AUDIT_INSTRUCTION,
) -> str:
    # audit_instruction defaults to the PMEX-futures-specific checklist
    # above but is overridable — ai/psx_suggest.py passes its own
    # equity-appropriate checklist through the same retry/pooling
    # machinery below rather than duplicating it.
    timeout = config.OPENROUTER_TIMEOUT_SECONDS if timeout is None else timeout
    prompt = (
        f"{audit_instruction}\n\n{summary}\n\n"
        f"Draft suggestion to audit:\n{draft_suggestion}"
    )
    if past_lessons:
        prompt += f"\n\n{past_lessons}"
    return run_openrouter(prompt, model=model, timeout=timeout)


AUDIT_MODELS: list[tuple[str, str]] = [
    # The largest/most-capable general-purpose text models among
    # OpenRouter's currently ~14 :free-tagged models (confirmed live via
    # /api/v1/models — this roster churns often and third-party "top
    # free model" lists proved stale within days, so verify live before
    # changing this again, not from memory or search results). Ranked by
    # parameter count as the best available proxy for audit strength
    # (bigger/reasoning-tuned models catch more) since OpenRouter doesn't
    # expose real usage/popularity data via API and their rankings page
    # is JS-rendered, not scrapable. Excluded from consideration:
    # nvidia/nemotron-3.5-content-safety (a moderation classifier, wrong
    # task) and nvidia/nemotron-nano-12b-v2-vl (vision-language, unneeded
    # here) — both smoke-tested-irrelevant rather than tested.
    ("Nvidia Nemotron-Ultra-550B", "nvidia/nemotron-3-ultra-550b-a55b:free"),  # 550B params
    ("Nvidia Nemotron-Super-120B", "nvidia/nemotron-3-super-120b-a12b:free"),  # 120B params
    ("Google Gemma 4 31B", "google/gemma-4-31b-it:free"),  # 31B params
    ("Nvidia Nemotron-Nano-Omni-30B-Reasoning", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"),  # 30B, reasoning-tuned
    ("OpenAI gpt-oss-20b", "openai/gpt-oss-20b:free"),  # 20B params
    # A 6th model, deliberately not because it's stronger than the Gemma
    # above — confirmed live via /api/v1/models that OpenRouter currently
    # has exactly one other free Google model, and it's a 26B MoE with
    # only 4B *active* params, weaker than the 31B dense model already in
    # this list, not a genuine capability upgrade. Added anyway, by
    # explicit user request, purely for redundancy: one more independent
    # model in the pool is still one more chance of surviving a shared-
    # pool rate-limit that happens to hit several models simultaneously
    # (confirmed live earlier this session that this does happen).
    ("Google Gemma 4 26B A4B", "google/gemma-4-26b-a4b-it:free"),  # 26B total / 4B active (MoE)
]


def build_past_audit_lessons(
    records_dir: Path | None = None, max_sessions: int = 2, max_chars_per_session: int = 3000
) -> str:
    """Pulls the actual audit findings (Stage 2) out of the most recent
    saved Portfolio Suggestion transcripts (see ai/session_record.py) so
    the CURRENT audit models can check whether the new draft repeats a
    mistake a past audit already flagged — by explicit user request,
    after noticing runs just hours apart had markedly different quality.

    Deliberately hands over the past audits' own raw findings, not a
    from-scratch summary of them — building a separate summarizer would
    be its own new AI call with its own cost/failure modes, and this
    project's established pattern everywhere else (macro data, book
    wisdom, positions) is to give real source data and let the model
    reason over it, not pre-digest it first. Truncates per session and
    caps how many sessions are included so this can't balloon an already
    token-heavy prompt — free audit models in particular may have
    tighter context budgets than Claude itself.

    "" when there's no records directory yet (first-ever run) or it's
    empty — nothing to learn from, not a failure."""
    records_dir = Path(config.PORTFOLIO_RECORDS_DIR) if records_dir is None else records_dir
    if not records_dir.exists():
        return ""

    files = sorted(records_dir.glob("portfolio_suggestion_*.md"), reverse=True)[:max_sessions]
    if not files:
        return ""

    sections = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue

        parts = text.split("## Stage 2 — Independent audits\n\n", 1)
        if len(parts) < 2:
            continue
        audit_block = parts[1].split("\n\n## Stage 3", 1)[0].strip()
        if not audit_block:
            continue

        excerpt = audit_block[:max_chars_per_session]
        sections.append(f"--- From a past session ({path.stem}) ---\n{excerpt}")

    if not sections:
        return ""

    return (
        "Audit findings from this account's most recent past Portfolio "
        "Suggestion session(s) — check whether the CURRENT draft repeats "
        "any of the same mistakes flagged below, and call it out "
        "explicitly if so, rather than re-discovering (or re-missing) "
        "the same issue independently:\n\n" + "\n\n".join(sections)
    )


@dataclass
class AuditResult:
    block: str
    audit_available: bool


@dataclass
class ModelAuditStatus:
    """One model's live progress through _run_audit_with_retry, for a UI
    to poll and display — see build_audit_block's on_progress param."""
    state: str  # "waiting" | "in_progress" | "retrying" | "succeeded" | "gave_up"
    attempt: int
    max_attempts: int
    retry_at: float | None = None  # time.monotonic() timestamp of next retry attempt


def _run_audit_with_retry(
    label: str,
    model: str,
    summary: str,
    draft: str,
    status_map: dict[str, ModelAuditStatus] | None = None,
    past_lessons: str = "",
    audit_instruction: str = AUDIT_INSTRUCTION,
) -> tuple[str, str]:
    """Retries a failed/rate-limited audit model every
    config.AUDIT_RETRY_INTERVAL_SECONDS, up to config.AUDIT_RETRY_TIMEOUT_SECONDS
    total, before giving up on it. OpenRouter's free-tier "shared pool"
    rate-limiting has been confirmed live to be transient — the same model
    has failed then succeeded minutes later — so retrying trades wall-clock
    time for a real chance at recovering a model that would otherwise be
    wrongly written off for the whole run, by explicit user request
    (strongest possible audit takes priority over speed here).

    Never retries a missing-key failure — that's a config problem, not a
    transient one, and can't resolve itself within the wait.

    Uses a fixed attempt count rather than tracking wall-clock elapsed time,
    so this is testable by mocking time.sleep alone (no fake clock needed).

    `status_map`, if given, gets a live-updated ModelAuditStatus under
    `label` at every state transition — runs inside a worker thread (see
    build_audit_block), so it only ever *replaces* the value at an
    already-existing key (never inserts/deletes one), which is what makes
    a concurrent .items() read from the polling thread in build_audit_block
    safe without an explicit lock."""
    max_attempts = max(1, config.AUDIT_RETRY_TIMEOUT_SECONDS // config.AUDIT_RETRY_INTERVAL_SECONDS)
    result = OPENROUTER_FAILED_MESSAGE

    def _set_status(state: str, attempt: int, retry_at: float | None = None) -> None:
        if status_map is not None:
            status_map[label] = ModelAuditStatus(
                state=state, attempt=attempt, max_attempts=max_attempts, retry_at=retry_at
            )

    for attempt in range(max_attempts):
        _set_status("in_progress", attempt + 1)
        try:
            result = get_openrouter_audit(
                summary,
                draft,
                model=model,
                past_lessons=past_lessons,
                audit_instruction=audit_instruction,
            )
        except Exception:
            result = OPENROUTER_FAILED_MESSAGE

        if result != OPENROUTER_FAILED_MESSAGE:
            _set_status("succeeded", attempt + 1)
            return label, result

        if attempt < max_attempts - 1:
            logger.info(
                "Audit model %s unavailable (attempt %d/%d) — retrying in %ds",
                model, attempt + 1, max_attempts, config.AUDIT_RETRY_INTERVAL_SECONDS,
            )
            _set_status(
                "retrying", attempt + 1, retry_at=time.monotonic() + config.AUDIT_RETRY_INTERVAL_SECONDS
            )
            time.sleep(config.AUDIT_RETRY_INTERVAL_SECONDS)

    _set_status("gave_up", max_attempts)
    return label, result


def _format_audit_progress(status_map: dict[str, ModelAuditStatus]) -> str:
    """Human-readable snapshot of every audit model's live status, for a
    UI to show as a sub-status under "Sending the draft to N independent
    free models for audit..." — see build_audit_block's on_progress."""
    lines = []
    succeeded = retrying = in_progress = gave_up = 0
    for label, status in status_map.items():
        if status.state == "succeeded":
            succeeded += 1
            lines.append(f"✓ {label} — audit received")
        elif status.state == "gave_up":
            gave_up += 1
            lines.append(f"✗ {label} — unavailable after {status.max_attempts} attempts, giving up")
        elif status.state == "retrying":
            retrying += 1
            wait_left = max(0, round(status.retry_at - time.monotonic())) if status.retry_at else 0
            lines.append(
                f"⏳ {label} — unavailable, retrying in {wait_left}s "
                f"(attempt {status.attempt}/{status.max_attempts})"
            )
        elif status.state == "waiting":
            in_progress += 1
            lines.append(f"… {label} — waiting to start")
        else:  # "in_progress"
            in_progress += 1
            lines.append(
                f"… {label} — waiting for response (attempt {status.attempt}/{status.max_attempts})"
            )

    total = len(status_map)
    summary_line = (
        f"{succeeded}/{total} models completed, {in_progress} in progress, "
        f"{retrying} retrying, {gave_up} gave up"
    )
    return "\n".join([summary_line, *(f"- {line}" for line in lines)])


def build_audit_block(
    summary: str,
    draft: str,
    on_progress: Callable[[str], None] | None = None,
    audit_instruction: str = AUDIT_INSTRUCTION,
    records_dir: Path | None = None,
) -> AuditResult:
    """Runs every model in AUDIT_MODELS in parallel against Claude's own
    stage-1 draft, retrying each one individually (see
    _run_audit_with_retry) if it's unavailable. Always attempted — once
    stage 1 produces a draft, there's always something for the audits to
    review, unlike the old Gemini-gated design. audit_available is true if
    *any* model responds, so the pool tolerates several being down at once
    (see AUDIT_MODELS for why it's spread across multiple providers).

    `audit_instruction` and `records_dir` let a different market's
    suggestion pipeline (see ai/psx_suggest.py) reuse this exact retry/
    pooling/progress machinery with its own failure-mode checklist and its
    own past-session lessons, without duplicating any of the threading
    logic below — defaults preserve the original PMEX-futures behavior
    exactly for every existing caller.

    `on_progress`, if given, is called from THIS function's own thread
    (not from the worker threads themselves — Streamlit commands aren't
    safe to call from a background thread) with a live per-model status
    summary every second, so a UI can show which models have responded,
    which are retrying and when, and which have given up, instead of one
    static line for the whole (up to ~10-minutes-per-model) audit phase."""
    status_map: dict[str, ModelAuditStatus] = {
        label: ModelAuditStatus(state="waiting", attempt=0, max_attempts=0) for label, _ in AUDIT_MODELS
    }
    # Computed once here, not per-model — it's the same file-read result
    # for every model in the pool, so no reason to redo the I/O 6 times.
    past_lessons = build_past_audit_lessons(records_dir=records_dir)

    with ThreadPoolExecutor(max_workers=len(AUDIT_MODELS)) as pool:
        # Submitted as individual futures (not list(pool.map(...))) so one
        # audit's exception can't abort iteration before a sibling's
        # already-completed result is collected.
        futures = [
            pool.submit(
                _run_audit_with_retry,
                label,
                model,
                summary,
                draft,
                status_map,
                past_lessons,
                audit_instruction,
            )
            for label, model in AUDIT_MODELS
        ]

        last_rendered = None
        while not all(f.done() for f in futures):
            if on_progress is not None:
                text = _format_audit_progress(status_map)
                if text != last_rendered:
                    on_progress(text)
                    last_rendered = text
            time.sleep(1)
        if on_progress is not None:
            on_progress(_format_audit_progress(status_map))

        audit_results = [f.result() for f in futures]

    audit_sections = []
    audit_available = False
    for label, text in audit_results:
        if text in (OPENROUTER_FAILED_MESSAGE, OPENROUTER_MISSING_KEY_MESSAGE):
            audit_sections.append(f"{label} audit: not available this time.")
        else:
            audit_sections.append(f"{label} audit:\n{text}")
            audit_available = True

    return AuditResult(block="\n\n".join(audit_sections), audit_available=audit_available)


def suggest_portfolio(
    summary: str,
    timeout: int | None = None,
    revision_timeout: int | None = None,
    on_stage: Callable[[str], None] | None = None,
    on_audit_progress: Callable[[str], None] | None = None,
    model: str | None = None,
    save_record: bool = False,
) -> str:
    # Resolved at call time (not as default-arg values) so a config/env
    # change is picked up without needing to reload this module.
    timeout = config.PORTFOLIO_SUGGESTION_TIMEOUT_SECONDS if timeout is None else timeout
    revision_timeout = (
        config.PORTFOLIO_REVISION_TIMEOUT_SECONDS if revision_timeout is None else revision_timeout
    )
    model = config.PORTFOLIO_SUGGESTION_MODEL if model is None else model

    def _notify(message: str) -> None:
        if on_stage:
            on_stage(message)

    _notify("Claude is researching the market and drafting an initial suggestion (live web search)...")
    draft_prompt = f"{build_stage1_instruction()}\n\n{summary}"
    draft = run_claude(
        draft_prompt,
        timeout=timeout,
        allowed_tools=["WebSearch", "WebFetch"],
        model=model,
    )
    if draft == CLI_MISSING_MESSAGE or draft.startswith(CLI_FAILED_PREFIX):
        # Nothing meaningful to record — the draft itself never happened.
        return draft

    _notify(f"Sending the draft to {len(AUDIT_MODELS)} independent free models for audit...")
    audit = build_audit_block(summary, draft, on_progress=on_audit_progress)
    _notify(
        "Audit received — Claude is revising its suggestion..."
        if audit.audit_available
        else "Independent audit wasn't available this run — Claude is re-checking its own draft instead..."
    )

    revise_prompt = (
        f"{build_stage2_instruction(audit.audit_available)}\n\n{summary}\n\n"
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
            )
        )

    return final_answer
