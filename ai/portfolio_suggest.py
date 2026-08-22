import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

import config
from ai.claude_cli import CLI_FAILED_PREFIX, CLI_MISSING_MESSAGE, run_claude
from ai.copilot_cli import CLI_FAILED_PREFIX as COPILOT_FAILED_PREFIX
from ai.copilot_cli import CLI_MISSING_MESSAGE as COPILOT_MISSING_MESSAGE
from ai.copilot_cli import run_copilot
from ai.openrouter_client import FAILED_MESSAGE as OPENROUTER_FAILED_MESSAGE
from ai.openrouter_client import MISSING_KEY_MESSAGE as OPENROUTER_MISSING_KEY_MESSAGE
from ai.openrouter_client import run_openrouter
from ai.session_record import SessionRecord, save_portfolio_session
from analysis.backtest import (
    MomentumPersistenceBacktest,
    RSIReactionBacktest,
    SupportResistanceBacktest,
    VolatilityRegimeBacktest,
    backtest_momentum_persistence,
    backtest_rsi_reaction,
    backtest_support_resistance_reaction,
    backtest_volatility_regime,
)
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
    "EXPLICIT RISK CALIBRATION (stated directly by the user, treat this "
    "as the goal every sizing/risk decision below serves): the user wants "
    "genuinely CALCULATED risk-taking, not the most conservative mix that "
    "technically diversifies, and not reckless/unresearched risk either — "
    "they've been explicit that they don't want 'a gambling board' but "
    "also don't want an overly cautious mix, specifically flagging that "
    "an earlier run of this tool ended up too conservative (holding an "
    "unnecessarily large, mostly-idle cash reserve). What makes risk "
    "'calculated' rather than 'gambling' is unchanged and still fully "
    "required: a real stop-loss on every position, sizing that reflects "
    "each instrument's own volatility and how strong its supporting "
    "evidence actually is (a high-conviction, well-evidenced case earns a "
    "meaningfully larger size than a marginal one — don't spread equity "
    "evenly across instruments regardless of conviction), genuine group/"
    "correlation diversification, and every claim grounded in the real "
    "macro/technical/research data given, not a hunch. Within those real "
    "constraints, resolve genuine uncertainty toward being MORE invested, "
    "not less. One specific failure mode to actively resist: the book-"
    "wisdom principles below include a generic '~50% of equity in "
    "reserve' cash-reserve default (Murphy) — that's a reasonable "
    "starting anchor for a generic account, but following it reflexively "
    "is exactly the over-conservative outcome this paragraph is "
    "correcting. IMPORTANT — this no longer reads as a raw notional-"
    "capital split: \"pct\" (see the JSON schema below) is the % of "
    "equity actually at risk if a position's stop is hit, not how much "
    "capital is committed to it, so CASH being very high (85%+ is normal) "
    "is NOT itself evidence of over-conservatism the way it would be "
    "under a notional-allocation reading — real lot sizes for even a 1-3% "
    "risk figure can represent meaningful, genuinely-invested exposure "
    "once leverage is accounted for. Judge conviction by whether the "
    "real, necessarily-small risk budget (see the aggregate-heat cap "
    "below) is actually being spent on genuinely good setups from your "
    "own research, not by how large CASH looks in this accounting. "
    "Absent a specific, genuinely elevated near-term risk you've "
    "identified in your own research (not a generic 'markets can always "
    "fall'), avoid leaving the aggregate-heat budget meaningfully "
    "underused; if you do, say explicitly why the specific evidence in "
    "front of you warrants it."
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
    "resistance range and a market-type classification: 'sideways' when "
    "the window's own price level genuinely hasn't shifted (scaled "
    "against its own volatility, not a flat percentage); otherwise there "
    "IS a real net directional move, and 'trending_up'/'trending_down' "
    "vs. 'choppy_up'/'choppy_down' further distinguishes whether that "
    "move was reached cleanly or via a lot of back-and-forth churn along "
    "the way — a choppy read is still real, actionable directional "
    "evidence, not a weaker cousin of sideways; treat it as a genuine "
    "(if noisier) directional signal, not as an absence of one. These price/support/"
    "resistance levels come from a third-party market-data source and may "
    "use a different quote convention than this account's own instrument "
    "(e.g. US grain futures like corn are commonly quoted in cents per "
    "bushel, so a level shown as 402 there means $4.02/bushel) — before "
    "using any such level as a trigger price or citing it in your final "
    "answer, sanity-check its scale/units against a quick WebSearch for "
    "that instrument's current real-world price, and convert if needed; "
    "REAL HISTORICAL BACKTESTS of this specific instrument's own past "
    "behavior over its real ~5-year price history — not a generic "
    "textbook assumption — covering exactly the kinds of claims a "
    "technical read tempts you to make: how often its own past RSI "
    "overbought/oversold episodes actually reversed as the textbook "
    "convention predicts (with the real average forward return and "
    "reversal rate), whether its own price history shows real momentum "
    "persistence or mean-reversion, whether its own historically LOW-"
    "volatility episodes have actually been followed by BIGGER "
    "subsequent moves (supporting the textbook 'coiled spring' idea) or "
    "by smaller ones (volatility clustering — quiet periods tend to stay "
    "quiet for this instrument, contradicting that idea), and how often "
    "price has actually held at the shown support level or been rejected "
    "at the shown resistance level historically, rather than assuming a "
    "support/resistance line is reliable just because it's a well-known "
    "charting concept. Ground any claim you make about RSI, momentum, "
    "volatility regime, or support/resistance reliability for a specific "
    "instrument in ITS OWN backtest evidence rather than the generic "
    "convention when the two disagree — say so explicitly when an "
    "instrument's real history contradicts the textbook assumption. NOTE: "
    "unlike PSX's equivalent equity backtests, there is deliberately NO "
    "beta-vs-benchmark backtest here — PMEX instruments span metals, "
    "energy, grains, currencies, and equity indices with no single "
    "natural per-instrument benchmark the way PSX's own index "
    "constituents have, and fabricating one (e.g. a generic S&P 500 beta "
    "for corn futures) would be closer to noise than evidence, so it's "
    "left out rather than invented; "
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
    "\n\n"
    "On the reward:risk principles specifically (2:1 Bulkowski/"
    "Rockefeller, 3:1 Murphy): the ratio must be an HONEST OUTPUT of a "
    "genuinely-derived entry, stop, and target — never work backward "
    "from the ratio to pick a target. Derive the stop from real ATR/"
    "volatility first, derive the target from a real technical level "
    "(actual resistance, a real prior high) — then report whatever ratio "
    "actually results, even if it's below 2:1. An instrument that "
    "genuinely doesn't offer 2:1 right now is real information (size it "
    "smaller, treat it as lower-conviction, or leave it out) — "
    "stretching its target to an optimistic level just to clear the "
    "ratio manufactures a number instead of reporting one, and is a "
    "worse outcome than honestly saying the ratio falls short."
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
    "\n\n"
    "End this draft with a compact \"External Research Notes\" list: one "
    "short bullet per genuinely WebSearch/WebFetch-sourced fact you're "
    "relying on (the claim plus its source, e.g. \"Corn export forecast "
    "cut 3% per the Aug 2026 USDA WASDE report\"), not a repeat of your "
    "prose. The audit models below have no web access of their own and "
    "can't tell a real researched fact from an invented one just by "
    "reading your prose — this list is what lets them see exactly which "
    "claims are genuine live research (to weigh and reason over) versus "
    "unstated assumptions (to actually flag). Keep it to the facts that "
    "materially influenced a sizing/inclusion decision, not every search "
    "result you looked at."
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
    "answer). You do not need to repeat your original research from "
    "scratch, but you remain the final decision-maker, not a rubber "
    "stamp for the audits: use WebSearch/WebFetch yourself to spot-check "
    "any specific claim the audits flagged as contested, stale, or "
    "consequential enough to double-check before you rely on it. The mix "
    "under review below is your own draft from the first pass — revise "
    "it, don't restart from a blank page."
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
    "auditor would, rather than simply restating it — internally; this "
    "self-check is process, not content, and shouldn't be narrated in "
    "the visible report either. The mix under review below is your own "
    "draft from the first pass."
    "\n\n"
    "Because the independent audit was unavailable this run, disclose "
    "that fact plainly as one sentence within the Executive Summary "
    "section of your visible answer (defined below) — this is the one "
    "piece of process information the user genuinely needs to know, "
    "since it means this run had one less layer of independent review; "
    "everything else about how the answer was produced stays out of the "
    "report."
)


# Session-continuation variants of the two roles above, used when stage 2
# resumes stage 1's own `claude -p --session-id` conversation (see
# suggest_portfolio) instead of a fresh, fully self-contained call. In
# that mode _INSTRUCTION_HEAD, `summary` (the market/macro data dump),
# and the stage-1 draft are ALL already in the model's own conversation
# history — repeating them verbatim in stage 2's prompt would be pure
# duplicate token cost with zero new information, since nothing about
# them changed between stage 1 and stage 2. Derived from the originals
# via .replace() (not hand-duplicated) so the two variants can never
# silently drift apart when the source text above is edited — the
# accompanying tests assert the swapped phrase is actually gone, so a
# future wording change that breaks the .replace() match fails loudly
# instead of leaving a stale "below" reference in the continued variant.
_ROLE_STAGE2_SYNTHESIZE_CONTINUED = _ROLE_STAGE2_SYNTHESIZE.replace(
    "Below is your own draft suggestion and reasoning from the first "
    "pass, plus independent audit reports from other models that "
    "reviewed your specific draft for flaws, gaps, and disagreements.",
    "Your own draft suggestion and reasoning from the first pass is "
    "already above in this conversation — no need to repeat it. Below "
    "are independent audit reports from other models that reviewed your "
    "specific draft for flaws, gaps, and disagreements.",
).replace(
    "The mix under review below is your own draft from the first pass — "
    "revise it, don't restart from a blank page.",
    "The mix under review is your own draft from the first pass, already "
    "shown above in this conversation — revise it, don't restart from a "
    "blank page.",
)

_ROLE_STAGE2_SELF_REVIEW_CONTINUED = _ROLE_STAGE2_SELF_REVIEW.replace(
    "Below is your own draft suggestion and reasoning from the first "
    "pass. The independent audit that normally reviews it was not "
    "available this run",
    "Your own draft suggestion and reasoning from the first pass is "
    "already above in this conversation. The independent audit that "
    "normally reviews it was not available this run",
).replace(
    "The mix under review below is your own draft from the first pass.",
    "The mix under review is your own draft from the first pass, already "
    "shown above in this conversation.",
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
    "   e. Idle cash — per the risk-calibration objective above, CASH now "
    "means equity not currently placed at risk (not unused notional "
    "capital), so it will typically sit very high, 85%+ is normal here "
    "and is not itself evidence of excessive conservatism. Judge this by "
    "whether the aggregate-heat risk budget (item g) is being meaningfully "
    "used on genuinely good setups, not by CASH's raw size — leaving that "
    "risk budget substantially unused without a specific, genuinely "
    "elevated near-term risk you identified (not generic caution) needs "
    "its own explicit justification. Whatever reserve "
    "you do leave, define specific conditional triggers for deploying it, "
    "tied to the support/resistance levels already given per instrument "
    "above (e.g. 'deploy X% if [instrument] pulls back to its support "
    "level' or 'deploy X% on a confirmed breakout above [instrument]'s "
    "resistance'). Every such trigger must also have an explicit time-"
    "based fallback (e.g. 'if not triggered within N trading days, do X "
    "instead') — an open-ended trigger that could leave cash idle "
    "indefinitely is itself a weakness. If the FX section above is "
    "present, also research the account country's current short-term "
    "risk-free/policy rate via WebSearch and explicitly weigh idle "
    "cash's opportunity cost against it (and against staying invested) "
    "when sizing the reserve — don't size cash as an arbitrary leftover "
    "percentage.\n"
    "   f. Position sizing vs. volatility, and hedge feasibility — size "
    "positions smaller, relative to equal-risk peers, for instruments "
    "showing higher annualized volatility% above. Only propose a hedge or "
    "alternative instrument that's actually listed among the Market Watch "
    "instruments below — don't suggest options, ETFs, or any other "
    "product that isn't shown as tradable on this account.\n"
    "   g. Aggregate heat — every position's \"pct\" already directly "
    "equals its own contribution to capital at risk (real lot size is "
    "computed downstream from this risk% and your stop distance, not "
    "from pct as notional capital to commit) — simply sum the pct values "
    "across the whole mix. State this total explicitly (e.g. 'aggregate "
    "heat: 5.2% of equity') and check it against a ~10-15% cap (raised "
    "from this tool's earlier, more conservative 6-8% cap, per the "
    "calculated-risk objective above): if a simultaneous worst-case "
    "stop-out across every position would erase more than that, resize "
    "down until it doesn't, rather than sizing positions independently "
    "of each other.\n"
    "   h. Sector/group concentration — group the mix's positions by "
    "correlated market/sector (e.g. precious metals, energy, grains, "
    "equity indices) and sum the % allocation (i.e. risk%) within each "
    "group. Since total risk is already capped around 10-15% by item g, "
    "judge concentration as a SHARE of that risk budget, not of raw "
    "equity. State "
    "each group's total risk% explicitly (e.g. 'metals + energy: 4.8% of "
    "equity at risk') rather than leaving it implicit in the per-"
    "instrument numbers. If any group eats more than roughly a third to "
    "half of the mix's total aggregate heat, you may still keep it that "
    "way, but you must explicitly justify why the "
    "concentration is warranted here (not just note that it exists) or "
    "resize it down.\n"
    "   i. Asset-class exclusion — if an entire asset class is present "
    "and tradable in the Market Watch instruments below (e.g. equity "
    "indices, when the mix is otherwise all commodities) but ends up at "
    "0% in your final mix, state that exclusion explicitly and justify "
    "it, the same way you'd justify an inclusion — don't let an absence "
    "go unexplained just because there's no number to attach it to.\n"
    "   j. Backtest grounding — for any instrument where you're leaning "
    "on an RSI, momentum, volatility-regime, or support/resistance "
    "argument, cross-check it against that exact instrument's own "
    "backtest evidence given above. If the real historical evidence "
    "contradicts the textbook convention you were about to lean on (e.g. "
    "an overbought reading treated as bearish but this instrument's own "
    "reversal rate after past overbought episodes is well under 50%), "
    "either drop that specific argument in favor of a supported one or "
    "explain concretely why you're keeping it despite the historical "
    "evidence against it — don't silently keep the generic-convention "
    "framing once contradicted.\n"
    "2. Revise the mix based on what step 1 found, incorporating whatever "
    "the strongest points were from this checklist pass, and from the "
    "audit/self-review findings above, whichever applied this run.\n"
    "\n"
    "Now write your visible answer as a STRUCTURED INVESTMENT REPORT — "
    "the way a professional futures/commodities research note reads "
    "(clear sections, a real narrative arc), not a raw stream of "
    "reasoning and not a transcript of the draft/audit/revise process "
    "above. Use exactly these markdown section headers, in this order; "
    "if a section is genuinely thin for this run, keep the header and "
    "write one honest sentence under it rather than omitting the "
    "section entirely — the structure itself is part of what makes this "
    "readable:\n"
    "## Executive Summary\n"
    "3-5 sentences: your overall market stance, the headline allocation "
    "idea, and the single biggest risk to watch this cycle. State here "
    "too that no deterministic sizing/risk rules are wired up for this "
    "feature yet, so this is an early-stage, discretionary starting "
    "point for discussion, not a precise or final recommendation (and, "
    "if applicable this run, that the independent audit layer wasn't "
    "available — see above).\n"
    "## Macro & Market Backdrop\n"
    "A flowing narrative — interpretation, not a restated bullet list of "
    "the yield-curve/DXY/VIX/FX numbers given above — on what the real "
    "macro data actually implies for these instruments right now, plus "
    "whatever geopolitical/policy context you found via WebSearch.\n"
    "## Asset-Class Outlook\n"
    "Group the instruments by asset class/correlated market (e.g. "
    "precious metals, energy, grains, equity indices) and give your view "
    "per group — including roll-yield/contango-backwardation dynamics "
    "for that class, and explicit justification for any entire class "
    "that's present and tradable but excluded from the mix (checklist "
    "items b, h, i belong here).\n"
    "## Investment Thesis by Position\n"
    "One short subsection per included instrument — lead each with the "
    "symbol in bold (e.g. \"**GOLD-DE26:**\") — covering in flowing "
    "prose why it earns its place now (technical + fundamental + "
    "research-based catalyst), and the sizing/entry/stop rationale, "
    "including FX exposure, spread/liquidity, and volatility-based "
    "sizing where relevant (checklist items a, d, f belong here, applied "
    "per position rather than listed separately). If you're relying on "
    "an RSI, momentum, volatility-regime, or support/resistance-based "
    "argument for a holding, ground it explicitly in that instrument's "
    "own real historical backtest evidence given above rather than the "
    "generic textbook convention when the two disagree — checklist item "
    "j belongs here too, applied per position.\n"
    "## Portfolio Construction & Risk Management\n"
    "Aggregate heat and correlation-under-stress findings (checklist "
    "items c, g), synthesized as your own risk-management conclusions "
    "about the mix as a whole — not a checklist recitation.\n"
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
    "fallback, checklist item e) and what would change this view going "
    "forward.\n"
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
    "length limit on this response. Take the time you need to research "
    "and reason properly; a longer, well-evidenced answer is preferred "
    "over a shorter, shallower one. Avoid needless repetition."
    "\n\n"
    "Write all of your reasoning and explanation as plain prose/markdown "
    "using the section headers specified above — do not put any of it "
    "inside a fenced code block. The ONLY fenced code block in your "
    "entire response must be a single one at the very end (after the "
    "'Recommended Allocation' section), exactly like this (replace the "
    "example values with your actual final numbers, one key per "
    "instrument symbol traded above plus one \"CASH\" key, pct values "
    "summing to 100, no comments or extra text inside the block). Every "
    "non-CASH key must be an object with five fields: \"side\" (\"buy\" "
    "for a long or \"sell\" for a short — this pipeline can now execute "
    "REAL short positions on this account, not just longs, and a short "
    "idea should be held to the exact same evidence bar as a long: real "
    "backtest support for that direction, aligned multi-timeframe "
    "signals, a genuine setup — not proposed as an afterthought just "
    "because one is now technically possible. A short-biased finding "
    "your own analysis already sometimes surfaces in prose — a negative "
    "momentum-persistence correlation, a range_fade_candidate setup "
    "sitting at resistance, a resistance-rejection backtest — is "
    "exactly the kind of evidence that can now justify \"side\": \"sell\" "
    "directly instead of being discarded for lack of a schema slot), "
    "\"pct\" (the % of "
    "equity you are willing to LOSE if this position's stop-loss is hit "
    "— NOT notional capital or margin to commit. Real lot size is "
    "computed automatically downstream from this risk% and your actual "
    "stop distance: a wider stop needs fewer lots to risk the same %, a "
    "tighter stop needs more — so this should be a small number, "
    "typically similar to a single-trade risk cap [e.g. 1-3%], not a "
    "large \"how much of my capital goes here\" figure. CASH is what's "
    "left after every position's risk%, i.e. equity not currently placed "
    "at risk — expect it to typically be very high, 85%+ is normal and "
    "not itself a sign of excessive conservatism), \"price\" (a specific limit-order entry price — "
    "it must be realistic and achievable given the instrument's current "
    "bid/ask shown above, not a distant support/resistance level or an "
    "arbitrary round number; also weigh missed-fill risk against price "
    "improvement — a price shaded meaningfully below the current quote "
    "in the hope of a pullback can mean the position is simply never "
    "entered if the instrument instead runs, which is a real cost for "
    "your highest-conviction ideas specifically, not a free option), "
    "\"stop_loss\" (a specific stop price), and \"take_profit\" (a "
    "specific target price on the CORRECT side of your entry FOR THE "
    "DIRECTION YOU CHOSE — above entry for a long, below entry for a "
    "short — matching the exact target level your own reward:risk "
    "reasoning above already derives; this is the field that actually "
    "gets sent to the broker as the position's real take-profit order, "
    "not just prose, so it must be the same real number your analysis "
    "used, not a rounded-off or re-guessed one). "
    "When a real ATR figure is shown for that instrument above, derive "
    "the stop distance from it (e.g. price minus roughly 1.5x ATR for a "
    "long, or price PLUS roughly 1.5x ATR for a short, per Bulkowski's "
    "principle above) rather than an assumed flat "
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
    '{"EXAMPLE_LONG": {"side": "buy", "pct": 1.5, "price": 82.50, "stop_loss": 78.00, "take_profit": 94.00}, '
    '"EXAMPLE_SHORT": {"side": "sell", "pct": 1.5, "price": 145.00, "stop_loss": 149.50, "take_profit": 133.00}, '
    '"CASH": 97.0}\n'
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
    rsi_overbought_backtest: RSIReactionBacktest | None = None
    rsi_oversold_backtest: RSIReactionBacktest | None = None
    momentum_persistence_backtest: MomentumPersistenceBacktest | None = None
    volatility_regime_backtest: VolatilityRegimeBacktest | None = None
    support_resistance_backtest: SupportResistanceBacktest | None = None
    # "yahoo" (default — PMEX/PSX always, FTMO where a Yahoo mapping
    # exists) or "mt5" (FTMO's own native-D1 backfill — see
    # ai/ftmo_suggest.py::_enrich_with_native_d1). Placed last, and
    # always passed by keyword at every real call site, specifically so
    # adding it here can't silently shift any existing positional
    # AssetAnalysis(...) call's arguments — see format_enriched_asset_
    # context's volume-caveat branch for the one place this is read.
    data_source: str = "yahoo"


def analyze_assets(
    assets: list[MarketAsset], on_progress: Callable[[str], None] | None = None
) -> list[AssetAnalysis]:
    """Resolve, fetch, and compute technical stats for the first assets (up
    to config.MAX_ENRICHED_ASSETS) that map to a Yahoo ticker. Instruments
    without a mapping, or beyond the cap, still get an entry (display_name
    = None) so callers can list them plainly without a network round-trip.
    Contract spec (real order-size/margin constraints) is fetched for
    every asset regardless of Yahoo mapping — it only needs the PMEX
    symbol via the already-open MT5 connection, no extra network call.

    `on_progress`, if given, is called once per instrument with a single
    updating message (not one line per symbol — a caller wires this to
    an in-place-updating UI element, the same pattern build_audit_block's
    own on_progress already uses) — this loop can genuinely take a while
    across a couple dozen instruments (a real network round-trip per
    enriched one), and previously had no progress visibility at all."""
    results = []
    enriched_count = 0
    total = len(assets)
    for i, a in enumerate(assets, start=1):
        if on_progress is not None:
            on_progress(f"Analyzing instruments: {i}/{total} — {a.symbol}")
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

        history = fetch_price_history_ohlcv(yahoo_ticker, period="5y")
        prices = history["Close"]
        stats = compute_technical_stats(prices, history=history)
        headlines = fetch_recent_headlines(yahoo_ticker, limit=config.NEWS_HEADLINES_PER_ASSET)
        rsi_overbought_bt, rsi_oversold_bt = backtest_rsi_reaction(history)

        results.append(
            AssetAnalysis(
                a.symbol, a.description, a.bid, a.ask, display_name,
                prices, stats, headlines, contract_spec,
                rsi_overbought_backtest=rsi_overbought_bt,
                rsi_oversold_backtest=rsi_oversold_bt,
                momentum_persistence_backtest=backtest_momentum_persistence(prices),
                volatility_regime_backtest=backtest_volatility_regime(prices),
                support_resistance_backtest=backtest_support_resistance_reaction(history),
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


def _format_rsi_backtest(bt: RSIReactionBacktest | None, condition: str) -> str:
    """Real trade-simulation result, not a raw forward-return average (see
    analysis/backtest.py's own top-of-file note) — reports what actually
    simulating the textbook reversal trade (short on overbought, long on
    oversold) with a real ATR-based stop/target would have won or lost,
    bar by bar, against this instrument's own history."""
    if bt is None:
        return (
            f"  historical {condition} RSI reaction: not enough real historical episodes "
            "(or no High/Low data to derive a real stop/target from) in this instrument's "
            "own history to compute — treat any RSI-reversal claim for it as unverified "
            "assumption, not evidence."
        )
    side = "short" if condition == "overbought" else "long"
    win_rate = f"{bt.win_rate_pct:.0f}%" if bt.win_rate_pct is not None else "n/a (none resolved yet)"
    return (
        f"  historical {condition} RSI reaction (real trade simulation, this instrument's "
        f"own past): {bt.trades} distinct past episodes where RSI reached {bt.threshold:.0f}, "
        f"simulating the textbook {side} reversal trade with a {bt.stop_atr_multiple:g}x-ATR "
        f"stop / {bt.target_atr_multiple:g}x-ATR target (max {bt.max_holding_bars} bars held) "
        f"-> {bt.wins} wins, {bt.losses} losses, {bt.timeouts} timed out; win rate {win_rate} "
        f"of resolved trades, average realized {bt.avg_r_multiple:+.2f}R across all of them"
        f"{_real_execution_note(bt.round_trip_cost_pct, bt.swap_pct_per_day_used, bt.min_stop_distance_pct)}"
    )


def _real_execution_note(round_trip_cost_pct: float, swap_pct_per_day: float, min_stop_distance_pct: float) -> str:
    """Discloses whenever a trade-simulation backtest (RSIReactionBacktest
    or one side of SupportResistanceBacktest) was actually netted against
    real broker cost/guard-rail data — added 2026-08-22, direct user
    request to make these results "more realistic and dependable" by
    accounting for real trade cost and broker execution constraints, not
    just an idealized ATR stop/target. Empty string (no note at all) when
    every one of these stayed at the engine's own zero-cost/zero-
    restriction default — true for PMEX/PSX today, since neither has this
    real data wired in yet (see analysis/backtest.py's own top-of-file
    note) — so the base sentence above reads as a pure idealized
    simulation for them, honestly, rather than implying a cost check that
    didn't happen.

    Cost/swap and the guard-rail note are joined with "; ", not folded
    into one "net of real X, Y, Z" list — "net of real" only fits the
    cost/swap numbers grammatically; the guard-rail clause describes a
    structural change to the stop, not something "netted" (a real
    wording bug caught live: the joined-list version literally read
    "net of real ... stop widened to...", broken grammar, not just
    redundant "real real" phrasing)."""
    cost_parts = []
    if round_trip_cost_pct > 0:
        cost_parts.append(f"{round_trip_cost_pct:.4f}% round-trip cost")
    if swap_pct_per_day:
        cost_parts.append(f"{swap_pct_per_day:+.4f}%/day swap")
    clauses = []
    if cost_parts:
        clauses.append("net of real " + " and ".join(cost_parts))
    if min_stop_distance_pct > 0:
        # Widening OVERRIDES the stated ATR multiple for any trade whose
        # ATR-based stop would've sat tighter than this floor — the "Xx-
        # ATR stop" figure quoted alongside this note is what trades NOT
        # hitting the floor used, not a guarantee every trade did.
        clauses.append(
            f"stop widened to the broker's own {min_stop_distance_pct:.3f}% minimum on any trade "
            "where the ATR-based stop would've been tighter than that"
        )
    if not clauses:
        return ""
    return " [" + "; ".join(clauses) + "]"


def format_backtests(r: AssetAnalysis) -> list[str]:
    """Real multi-year backtests of this specific instrument's own price
    history (see analysis/backtest.py) — mirrors ai/psx_suggest.py's own
    `_format_backtests` in spirit and wording, kept as an independent
    copy rather than a shared import since the two files' report
    templates are each maintained standalone. Public (unlike its PSX
    counterpart) since it's shared by both PMEX's own prompt and FTMO's
    (via format_enriched_asset_context, which ai/ftmo_suggest.py's own
    format_ftmo_asset_context calls per-instrument) — same reasoning as
    ai/ftmo_suggest.py::fetch_ftmo_status's own promotion to public. Not
    currently used by app.py's own watchlist popup, which renders the
    same underlying dataclasses directly as charts/metrics instead (see
    app.py::_render_backtest_metrics). Deliberately does NOT
    include a beta-stability backtest the way the PSX version does: that
    needs a per-instrument benchmark index, and PMEX spans metals,
    energy, grains, currencies, and equity indices with no single
    natural benchmark the way PSX's own index constituents have — rather
    than fabricate one (e.g. a generic S&P 500 beta for corn futures
    would be close to meaningless), it's left out here and disclosed as
    out of scope in the prompt text instead."""
    lines = [_format_rsi_backtest(r.rsi_overbought_backtest, "overbought")]
    lines.append(_format_rsi_backtest(r.rsi_oversold_backtest, "oversold"))

    mp = r.momentum_persistence_backtest
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

    vr = r.volatility_regime_backtest
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

    sr = r.support_resistance_backtest
    if sr is not None:
        support_rate = f"{sr.support_win_rate_pct:.0f}%" if sr.support_win_rate_pct is not None else "n/a"
        resistance_rate = (
            f"{sr.resistance_win_rate_pct:.0f}%" if sr.resistance_win_rate_pct is not None else "n/a"
        )
        support_note = _real_execution_note(
            sr.round_trip_cost_pct, sr.support_swap_pct_per_day_used, sr.min_stop_distance_pct
        )
        resistance_note = _real_execution_note(
            sr.round_trip_cost_pct, sr.resistance_swap_pct_per_day_used, sr.min_stop_distance_pct
        )
        lines.append(
            f"  historical support/resistance reliability (real trade simulation, this "
            f"instrument's own past, {sr.stop_atr_multiple:g}x-ATR stop / "
            f"{sr.target_atr_multiple:g}x-ATR target, max {sr.max_holding_bars} bars held): "
            f"buying off support -> {sr.support_wins}W/{sr.support_losses}L/{sr.support_timeouts} "
            f"timed out across {sr.support_tests} real tests, win rate {support_rate}, avg "
            f"{sr.support_avg_r_multiple:+.2f}R{support_note}; shorting off resistance -> "
            f"{sr.resistance_wins}W/{sr.resistance_losses}L/{sr.resistance_timeouts} timed out "
            f"across {sr.resistance_tests} real tests, win rate {resistance_rate}, avg "
            f"{sr.resistance_avg_r_multiple:+.2f}R{resistance_note} — use this to judge how much "
            "weight the support/resistance range shown above deserves for THIS instrument "
            "specifically, rather than assuming support/resistance lines are reliable just "
            "because they're a well-known charting concept."
        )
    else:
        lines.append(
            "  historical support/resistance reliability: not enough real historical tests "
            "of these levels (or no High/Low data to derive a real stop/target from) to compute"
        )
    return lines


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
                # Yahoo-sourced volume is a broader-market proxy (this
                # data provider's own feed, not this specific broker's
                # order flow); MT5-native volume (FTMO's own D1 backfill
                # for symbols with no Yahoo mapping — see ai/
                # ftmo_suggest.py::_enrich_with_native_d1) genuinely IS
                # this account's own broker feed, so the caveat only
                # applies to the Yahoo case.
                volume_note = (
                    "global market volume, not this account's own order flow"
                    if r.data_source == "yahoo"
                    else "this account's own MT5 feed"
                )
                stat_bits.append(
                    f"volume {stats.volume_trend_pct:+.0f}% vs its own 20-day average "
                    f"({volume_note})"
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
            lines += format_backtests(r)
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
    take_profit: float | None = None
    side: str = "buy"  # "buy" or "sell" — see risk/apply_suggestion.py::compute_rebalance_plan


def parse_final_allocation(
    response_text: str, require_side: bool = False
) -> dict[str, AllocationEntry] | None:
    """Extract the trailing ```json {...}``` allocation block the prompt
    asks for (see build_stage1_instruction/build_stage2_instruction).
    Each non-CASH value is an object with pct/price/stop_loss/side (see
    the JSON-block spec in _INSTRUCTION_TAIL); CASH is a bare number. Also
    tolerates a bare number for a non-CASH key (price/stop_loss come back
    None) in case the model reverts to the old shape — a missing price is
    a real limitation for execution, not a parse failure, so this still
    parses rather than dropping the whole allocation. `side` defaults to
    "buy" when absent from an object-shape entry, UNLESS `require_side`
    is True (pass this for FTMO/PMEX, whose schema always includes
    `side`) — there, an object-shape entry silently missing `side` fails
    the whole parse instead, the same as an invalid value, because for
    those two exchanges an absent side on a symbol currently held SHORT
    would otherwise default to "buy" and get treated as a genuine
    direction flip: a real position force-closed and reversed on a
    silently-guessed field. PSX (never emits `side` at all) and the old
    bare-number shape are unaffected by `require_side` — bare numbers
    still default to "buy" regardless, since PSX has no execution path
    for this default to ever mislead. An explicit but INVALID side
    (anything other than "buy"/"sell", case-insensitive) always fails the
    whole parse — a malformed direction is a structural schema violation,
    not a "missing, tolerate it" case, and silently guessing a direction
    here is exactly the kind of substitution this pipeline's execution
    path has already been burned by once. Returns None on anything
    unexpected — missing block, malformed JSON, wrong shape — rather than
    raising; the text response still displays fine even if this fails, it
    just means no allocation chart/execution plan."""
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
        take_profit = value.get("take_profit")
        if price is not None and not isinstance(price, (int, float)):
            return None
        if stop_loss is not None and not isinstance(stop_loss, (int, float)):
            return None
        if take_profit is not None and not isinstance(take_profit, (int, float)):
            return None
        side = value.get("side")
        if side is None:
            if require_side:
                return None
            side_norm = "buy"
        elif isinstance(side, str) and side.lower() in ("buy", "sell"):
            side_norm = side.lower()
        else:
            return None
        result[key] = AllocationEntry(
            pct=float(value["pct"]),
            price=float(price) if price is not None else None,
            stop_loss=float(stop_loss) if stop_loss is not None else None,
            take_profit=float(take_profit) if take_profit is not None else None,
            side=side_norm,
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


def strip_leading_process_narration(response_text: str) -> str:
    """A prompt-only defense against a model narrating its internal
    draft/audit/revision process (e.g. a leading "## Internal Review &
    Revision Process" section listing "Key audit findings I'm accepting")
    isn't fully reliable — confirmed live: a real haiku run did this
    despite the explicit instruction not to. This is a zero-token, purely
    mechanical backstop: every report from this pipeline is contractually
    required to open with "## Executive Summary" as its first section, so
    anything before that exact header is process scaffolding the model
    wasn't supposed to print, not real report content — strip it rather
    than trusting prompt wording alone across every model in the picker.
    A no-op if the response already starts there (idx == 0) or doesn't
    contain the marker at all (idx == -1, can't safely guess what's
    leading content in that case)."""
    marker = "## Executive Summary"
    idx = response_text.find(marker)
    if idx <= 0:
        return response_text
    return response_text[idx:].strip()


AUDIT_INSTRUCTION = (
    "You are an independent audit reviewer in a multi-stage "
    "portfolio-suggestion pipeline. You do NOT have web search or any "
    "live data access — reason only from the account, market, macro, "
    "contract-feasibility, and research data given below (already "
    "gathered by another system), plus Claude's own first-pass suggested "
    "portfolio mix and reasoning, produced from that same data using live "
    "web research (WebSearch/WebFetch)."
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
    "BACKTEST THE DRAFT'S UNDERLYING LOGIC, not just its arithmetic. The "
    "draft doesn't just state numbers — it implies a small system of "
    "cause-and-effect rules about how these instruments behave (e.g. "
    "'this overbought reading means a pullback is likely', 'this "
    "instrument's positive momentum means it should keep outperforming'). "
    "Treat the draft as implicitly claiming a function — given a "
    "condition X (a technical reading, a regime), it asserts an expected "
    "market response Y — and you have real historical evidence below to "
    "test specific values of X against:\n"
    "1. For each holding, identify the specific technical/behavioral "
    "claim(s) the draft is relying on to justify it (momentum "
    "continuing, a reversal being likely, a volatility-contraction "
    "'coiled spring' setup, a support/resistance level holding).\n"
    "2. Cross-check EACH such claim against that exact instrument's own "
    "historical backtest evidence given below (its real simulated RSI-"
    "reversal and support/resistance-bounce trade win rates and average "
    "realized R-multiple — an actual ATR-based stop/target walked "
    "forward bar by bar to a genuine win/loss, not a bare average return "
    "N bars later — plus whether its own history shows real momentum "
    "persistence or mean-reversion, and whether its own low-volatility "
    "episodes have historically been followed by bigger or smaller "
    "moves) — this is genuine historical "
    "evidence for THIS instrument specifically, not a generic textbook "
    "assumption. Note: there is no beta-vs-benchmark backtest here (PMEX "
    "spans too many uncorrelated asset classes for one natural "
    "benchmark) — don't fault the draft for lacking one.\n"
    "3. Score each claim you checked: SUPPORTED (the historical evidence "
    "agrees with the draft's implied logic), CONTRADICTED (the "
    "instrument's own history shows the opposite), or UNTESTABLE (not "
    "enough real historical episodes were available to judge either way "
    "— say so rather than guessing). Cite the actual numbers you're "
    "basing this on.\n"
    "4. A CONTRADICTED score is a real, concrete flaw to raise — treat it "
    "with the same weight as a math error, not a minor stylistic note.\n"
    "\n"
    "Produce a structured audit report — agreements, flaws, gaps, the "
    "backtest scorecard from above, and specific suggested improvements "
    "— not a rewritten competing allocation. Since you have no live data "
    "access, don't claim to fact-check anything beyond what's in the "
    "data given here (the historical backtests ARE data given here, not "
    "something you're fetching yourself); audit its reasoning and "
    "internal consistency, not facts you can't verify. Keep your "
    "response focused — under 550 words."
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


# (label, OpenRouter model id, capability/specialization profile) — the
# third element is shown to Claude alongside each model's own audit text
# in build_audit_block's output specifically so the stage-2 synthesis can
# weigh each critique by how much confidence its source model's scale/
# specialization actually warrants (see _ROLE_STAGE2_SYNTHESIZE), rather
# than treating a 550B general-reasoning model and a 30B coding-agent
# model as equally authoritative by default just because both produced a
# paragraph of audit text.
AUDIT_MODELS: list[tuple[str, str, str]] = [
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
    (
        "Nvidia Nemotron-Ultra-550B",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "550B params, general-purpose reasoning — the largest model in this pool",
    ),
    (
        "Nvidia Nemotron-Super-120B",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "120B params, general-purpose reasoning",
    ),
    # Google Gemma 4 31B occupied this slot originally, but was swapped
    # out 2026-08-16 after being confirmed persistently unavailable, not
    # just transiently rate-limited: it failed EVERY real session across
    # multiple days (user-reported), and a direct live smoke test with a
    # trivial one-word prompt (not even a real audit-sized one) still
    # got a 429 from "Google AI Studio" with `limit_source: upstream_
    # provider_shared_pool` — that shared pool is saturated at a level
    # this pipeline's existing 10-attempt/600s retry loop can't route
    # around, so this is a genuine dead slot, not noise. Confirmed the
    # OTHER Gemma already in this pool (26B A4B, below) does NOT share
    # this problem — smoke-tested clean — so only this one entry needed
    # replacing, not "avoid Gemma/Google AI Studio entirely." Re-checked
    # the live /api/v1/models roster rather than reusing the fallback
    # candidate named in an earlier round's comment (inclusionai/
    # ling-3.0-tiny:free) — it had disappeared from the roster entirely
    # since then, confirming this list really does churn and must be
    # re-verified live each time, not assumed stable.
    (
        "Dots Studio Dots3-Note Preview",
        "dots-studio/dots-3-note-preview:free",
        "280B total/16B active MoE, general-purpose reasoning — the lightest model in "
        "its own family but still comfortably larger than most of this pool",
    ),
    (
        "Nvidia Nemotron-Nano-Omni-30B-Reasoning",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "30B total/3B active MoE, general-purpose, reasoning-tuned",
    ),
    ("OpenAI gpt-oss-20b", "openai/gpt-oss-20b:free", "20B params, general-purpose reasoning"),
    # A 6th model, deliberately not because it's stronger than the Gemma
    # above — confirmed live via /api/v1/models that OpenRouter currently
    # has exactly one other free Google model, and it's a 26B MoE with
    # only 4B *active* params, weaker than the 31B dense model already in
    # this list, not a genuine capability upgrade. Added anyway, by
    # explicit user request, purely for redundancy: one more independent
    # model in the pool is still one more chance of surviving a shared-
    # pool rate-limit that happens to hit several models simultaneously
    # (confirmed live earlier this session that this does happen).
    (
        "Google Gemma 4 26B A4B",
        "google/gemma-4-26b-a4b-it:free",
        "26B total/4B active MoE, general-purpose reasoning — the smallest ACTIVE-parameter "
        "count among the pool's general-purpose models, kept for provider redundancy over "
        "raw capability",
    ),
    # Expanded 6 -> 10 (2026-08-11), re-checking /api/v1/models live rather
    # than assuming the roster above was still current — it wasn't: 8 new
    # free models had appeared since the 6th-model decision. After the same
    # two task-fit exclusions as above (still present: nvidia/nemotron-3.5-
    # content-safety, nvidia/nemotron-nano-12b-v2-vl), 6 genuinely new
    # candidates remained. All 4 added below were smoke-tested live
    # (a real, successful chat-completions round-trip) before being added,
    # matching this list's established practice of never trusting an entry
    # from the models listing alone.
    #
    # Honest caveat, not silently glossed over: 3 of these 4 (everything
    # below except the Nemotron) are marketed as CODING-AGENT models
    # (Poolside's own copy cites Terminal-Bench, a coding benchmark;
    # Cohere's North family debuts as "its first agentic coding model") —
    # a genuinely different post-training specialization than the general
    # instruction-following/reasoning models that made up the first 6, even
    # though they're still general chat-completion-capable LLMs that can
    # produce a written audit. This is a real, deliberate size-over-fit
    # tradeoff to reach 10 total: after excluding the content-safety/
    # vision-language models, only 2 of the 6 remaining free candidates
    # (the ones NOT added here — inclusionai/ling-3.0-tiny at 7.9B and
    # nvidia/nemotron-nano-9b-v2 at 9B) are both general-purpose AND
    # smaller than every model added below — there was no way to reach 10
    # using only general-purpose models without going smaller than what's
    # already in the pool. If a future audit run's OpenRouter transcripts
    # show these three producing noticeably shallower/more code-flavored
    # critiques than the general-purpose models, that's the first thing to
    # revisit — swap one or more back out for ling-3.0-tiny/nemotron-nano-
    # 9b-v2 despite their smaller size, rather than assuming the pooling/
    # retry machinery itself is at fault. Per-model profile strings below
    # are what let stage 2 actually act on this caveat per audit, rather
    # than this comment being the only place it's recorded.
    (
        "Poolside Laguna S 2.1",
        "poolside/laguna-s-2.1:free",
        "118B total/8B active MoE, CODING-AGENT-specialized (tuned for coding-agent "
        "benchmarks, not general financial/textual reasoning) — weigh its open-ended "
        "judgment calls with more caution than the general-purpose models above, though "
        "any concrete, verifiable point it raises still stands on its own merits",
    ),
    (
        "Poolside Laguna XS 2.1",
        "poolside/laguna-xs-2.1:free",
        "33B total/3B active MoE, CODING-AGENT-specialized (tuned for coding-agent "
        "benchmarks, not general financial/textual reasoning) — same lower-default-trust "
        "caveat as Laguna S above",
    ),
    (
        "Cohere North Mini Code",
        "cohere/north-mini-code:free",
        "30B total/3B active MoE, CODING-AGENT-specialized (Cohere's own debut agentic "
        "CODING model) — same lower-default-trust caveat as the Poolside models above",
    ),
    (
        "Nvidia Nemotron 3 Nano 30B A3B",
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "30B total/3B active MoE, general-purpose agentic reasoning (not coding-specialized)",
    ),
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


def build_past_outcome_lessons(
    fetch_current_price: Callable[[str], float | None],
    records_dir: Path | None = None,
    max_sessions: int = 2,
    max_symbols_per_session: int = 10,
) -> str:
    """Real, computed evidence of how recently-suggested positions have
    actually performed since — the half of "learning from past mistakes"
    build_past_audit_lessons doesn't cover. That function carries forward
    what a past AUDIT criticized about the REASONING (a subjective
    judgment call); this carries forward what the REAL MARKET actually
    did afterward (an objective fact) — a genuinely different, more
    direct way to catch an actual market misunderstanding rather than
    just a methodology slip an audit happened to notice.

    Pure computation, no LLM call: parses each of the last `max_sessions`
    saved sessions' own final (Stage 3) allocation JSON via
    parse_final_allocation, then for every non-CASH symbol with both a
    suggested price and stop_loss, fetches ONE current price via the
    caller-supplied `fetch_current_price` (exchange-specific — PSX uses
    its own symbols directly, PMEX needs Yahoo-ticker resolution first;
    see each module's own small wrapper) and reports the real % move
    since the suggestion and whether the stop would already have been
    breached. Deliberately does NOT claim a limit entry actually filled
    (PSX's feed has no intraday High/Low to know that for certain) —
    states the entry/current/stop numbers and lets the reasoning stage
    draw its own conclusion, the same "compute the real number, let the
    model reason about what it means" split this project uses everywhere
    else (backtests, feasibility %, aggregate heat).

    A symbol `fetch_current_price` can't resolve/fetch returns None and
    is silently skipped, not fabricated — same disclosure convention as
    everywhere else in this codebase. "" when there's nothing to report
    (no past sessions, or none of them had a re-priceable symbol)."""
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

        parts = text.split("## Stage 3 — Claude's final revised suggestion\n\n", 1)
        if len(parts) < 2:
            continue
        allocation = parse_final_allocation(parts[1])
        if not allocation:
            continue

        lines = []
        checked = 0
        for symbol, entry in allocation.items():
            if symbol == "CASH" or not entry.price or entry.stop_loss is None:
                continue
            if checked >= max_symbols_per_session:
                break
            checked += 1
            current = fetch_current_price(symbol)
            if current is None:
                continue
            pct_change = (current - entry.price) / entry.price * 100
            stop_status = (
                "STOP WOULD HAVE BEEN HIT" if current <= entry.stop_loss else "stop not hit"
            )
            lines.append(
                f"- {symbol}: suggested entry {entry.price:g}, stop {entry.stop_loss:g} -> "
                f"current price {current:g} ({pct_change:+.1f}%), {stop_status}"
            )

        if lines:
            sections.append(f"--- From a past session ({path.stem}) ---\n" + "\n".join(lines))

    if not sections:
        return ""

    return (
        "REAL, computed outcomes of positions suggested in this account's "
        "most recent past Portfolio Suggestion session(s), re-priced just "
        "now — use this as genuine evidence for whether that past thesis "
        "has been validated or contradicted by actual subsequent market "
        "behavior, not just whether the reasoning sounded sound at the "
        "time. A position still well above its entry and away from its "
        "stop is a real confirmation, worth noting briefly; a position "
        "that has already breached its stop or moved sharply against the "
        "original thesis is real evidence the underlying market read may "
        "have been wrong — say so explicitly if the current draft is "
        "making a similar case for the same symbol or sector again, "
        "rather than silently repeating an already-contradicted read:\n\n"
        + "\n\n".join(sections)
    )


def build_past_lessons(
    fetch_current_price: Callable[[str], float | None],
    records_dir: Path | None = None,
) -> str:
    """Combines both kinds of "learning from past sessions" this pipeline
    has — build_past_audit_lessons (what a past AUDIT criticized about
    the reasoning) and build_past_outcome_lessons (what the REAL MARKET
    actually did to past suggestions since) — into the single block
    computed ONCE per run and given to stage 1's draft, the resumed
    stage-2 session (for free, via session continuity — see
    suggest_portfolio), and every audit model. Each half already carries
    its own usage guidance inline in its own returned text, so nothing
    further needs to be said about it in the static instruction prompts
    — this only ever appears in a run's actual token cost when there's
    real past-session data to show, unlike a permanent instruction
    paragraph that would cost tokens on every run regardless."""
    parts = [
        build_past_audit_lessons(records_dir=records_dir),
        build_past_outcome_lessons(fetch_current_price=fetch_current_price, records_dir=records_dir),
    ]
    return "\n\n".join(p for p in parts if p)


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
    # The real failure reason, when known (confirmed live: this was
    # previously discarded everywhere on a "gave_up" — the UI showed a
    # bare "unavailable, giving up" and Claude's own audit block showed
    # "not available this time", with the actual cause — a timeout, an
    # auth failure, a non-zero exit code with real stderr — visible
    # nowhere, forcing a manual re-run just to find out why).
    detail: str | None = None


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
            reason = f" — {status.detail}" if status.detail else ""
            lines.append(
                f"✗ {label} — unavailable after {status.max_attempts} attempts, giving up{reason}"
            )
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


_EXTERNAL_NOTES_HEADING = "External Research Notes"
_NO_EXTERNAL_NOTES_MESSAGE = (
    f'No "{_EXTERNAL_NOTES_HEADING}" section found in the draft — nothing to '
    "independently verify this run."
)


def build_copilot_verification(draft: str, timeout: int | None = None) -> str:
    """One independent live-web fact-check of the draft's own "External
    Research Notes" list, via GitHub Copilot CLI — the only reviewer in
    this pipeline with real internet access (the 10 OpenRouter audit
    models have none, and can only judge a claim's internal consistency,
    never confirm or deny it against the real world).

    Deliberately scoped to a bounded, already-extracted list of claims
    rather than open-ended research: a live smoke test confirmed Copilot
    CLI has no dedicated search-engine tool (only web_fetch/curl), so
    open-ended "go find relevant news" is slow, trial-and-error fetching
    against guessed URLs — a bounded "check these specific claims" task is
    the mode that actually worked reliably in that same test.

    A free, zero-token, zero-subprocess guard: if the draft has no
    External Research Notes section at all, there's nothing to check —
    skip spawning Copilot entirely rather than paying for a call that
    would just report that back.

    Also deliberately sends Copilot ONLY that Notes section, not the
    entire draft: its actual job is bounded to those specific listed
    claims (stage 1 is instructed to put this section LAST, so slicing
    from its first occurrence to the end of the draft captures exactly
    that), and the surrounding investment-thesis prose is both unrelated
    to what it needs to check and meaningfully larger — cheaper for
    Copilot's own request budget, and a tighter task with less to get
    distracted by, not just a smaller prompt for its own sake."""
    if _EXTERNAL_NOTES_HEADING not in draft:
        return _NO_EXTERNAL_NOTES_MESSAGE
    notes_section = draft[draft.index(_EXTERNAL_NOTES_HEADING) :]
    timeout = config.COPILOT_VERIFICATION_TIMEOUT_SECONDS if timeout is None else timeout
    prompt = (
        "You are an independent fact-checker for a portfolio-suggestion "
        "pipeline. Below is an \"External Research Notes\" list — specific "
        "claims (each with a claimed source) that another AI's draft "
        "relies on. You have real, live web access — USE IT via your "
        "web_fetch/curl tools: for each claim listed, try to independently "
        "find and fetch corroborating or contradicting evidence, then "
        "report, per claim: SUPPORTED (you found matching evidence), "
        "CONTRADICTED (you found evidence against it), or COULDN'T VERIFY "
        "(no clear evidence within reasonable effort) — with a one-line "
        "reason and the URL you checked.\n\n"
        f"{notes_section}"
    )
    return run_copilot(prompt, timeout=timeout)


# The label Copilot's own live progress is tracked/rendered under in
# status_map — deliberately short (unlike copilot_label below, the long
# descriptive one used in the final audit_sections text for Claude),
# since this one appears in the live per-model UI ticker alongside the
# 10 OpenRouter models' own short labels.
_COPILOT_STATUS_LABEL = "Copilot CLI"


def _run_copilot_with_status(draft: str, status_map: dict[str, ModelAuditStatus] | None = None) -> str:
    """Wraps build_copilot_verification with the same live status_map
    tracking _run_audit_with_retry gives the 10 OpenRouter models —
    without this, Copilot's real progress was invisible to on_progress
    (confirmed: it never appeared in the live UI ticker at all), and
    since build_audit_block's polling loop only watched the OpenRouter
    futures, a slow Copilot call left the whole audit phase looking
    frozen at its last OpenRouter-only render while Copilot kept running
    silently underneath it — a real, confirmed cause of "the page looks
    stuck." Copilot has no retry loop (see build_copilot_verification's
    own docstring), so this only ever has one real state transition:
    waiting -> in_progress -> succeeded/gave_up."""
    if status_map is not None:
        status_map[_COPILOT_STATUS_LABEL] = ModelAuditStatus(state="in_progress", attempt=1, max_attempts=1)

    result = build_copilot_verification(draft)

    if status_map is not None:
        failed = result == COPILOT_MISSING_MESSAGE or result.startswith(COPILOT_FAILED_PREFIX)
        status_map[_COPILOT_STATUS_LABEL] = ModelAuditStatus(
            state="gave_up" if failed else "succeeded",
            attempt=1,
            max_attempts=1,
            detail=result if failed else None,
        )
    return result


def _fetch_current_pmex_price(symbol: str) -> float | None:
    """The PMEX-specific `fetch_current_price` callable for
    build_past_outcome_lessons — a past suggestion's saved allocation only
    has the raw PMEX symbol (e.g. "GO10OZ"), not a resolved Yahoo ticker,
    so this re-resolves it the same way analyze_assets does. Deliberately
    fetches only "5d" of history (not the 5y default used for backtests)
    since only the latest close is actually needed here — a much smaller,
    faster request for a task that doesn't need years of context. None if
    the symbol can't be resolved or the fetch comes back empty — silently
    skipped by the caller, not fabricated."""
    resolved = resolve_yahoo_ticker(symbol)
    if resolved is None:
        return None
    _, yahoo_ticker = resolved
    history = fetch_price_history_ohlcv(yahoo_ticker, period="5d")
    if history.empty:
        return None
    return float(history["Close"].iloc[-1])


def build_audit_block(
    summary: str,
    draft: str,
    on_progress: Callable[[str], None] | None = None,
    audit_instruction: str = AUDIT_INSTRUCTION,
    past_lessons: str = "",
) -> AuditResult:
    """Runs every model in AUDIT_MODELS in parallel against Claude's own
    stage-1 draft, retrying each one individually (see
    _run_audit_with_retry) if it's unavailable. Always attempted — once
    stage 1 produces a draft, there's always something for the audits to
    review, unlike the old Gemini-gated design. audit_available is true if
    *any* model responds, so the pool tolerates several being down at once
    (see AUDIT_MODELS for why it's spread across multiple providers).

    `audit_instruction` lets a different market's suggestion pipeline (see
    ai/psx_suggest.py) reuse this exact retry/pooling/progress machinery
    with its own failure-mode checklist, without duplicating any of the
    threading logic below — the default preserves the original PMEX-
    futures behavior exactly for every existing caller. `past_lessons` is
    computed ONCE by the caller (build_past_lessons) before stage 1 even
    runs — not recomputed here — specifically so stage 1's draft prompt,
    the resumed stage-2 session, and this audit pool all reason over the
    exact same past-session evidence without paying for the real price-
    fetching it involves more than once per run.

    `on_progress`, if given, is called from THIS function's own thread
    (not from the worker threads themselves — Streamlit commands aren't
    safe to call from a background thread) with a live per-model status
    summary every second, so a UI can show which models have responded,
    which are retrying and when, and which have given up, instead of one
    static line for the whole (up to ~10-minutes-per-model) audit phase."""
    status_map: dict[str, ModelAuditStatus] = {
        label: ModelAuditStatus(state="waiting", attempt=0, max_attempts=0)
        for label, _, _ in AUDIT_MODELS
    }
    # Copilot gets its own entry too — see _run_copilot_with_status's own
    # docstring for why this matters: without it, a slow Copilot call was
    # invisible to on_progress and could leave the UI looking frozen
    # after every OpenRouter model had already finished.
    status_map[_COPILOT_STATUS_LABEL] = ModelAuditStatus(state="waiting", attempt=0, max_attempts=1)
    # Looked up when composing audit_sections below, so each model's audit
    # text can be shown to Claude alongside its own capability/
    # specialization profile — see AUDIT_MODELS' own comment for why.
    profile_by_label = {label: profile for label, _, profile in AUDIT_MODELS}

    with ThreadPoolExecutor(max_workers=len(AUDIT_MODELS) + 1) as pool:
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
            for label, model, _ in AUDIT_MODELS
        ]
        # Runs alongside the OpenRouter pool, not after it, so it doesn't
        # add net wall-clock time to every run — no retry loop (see
        # build_copilot_verification's own docstring for why that's a
        # deliberate simplification, not an oversight), but its live
        # status IS now tracked (_run_copilot_with_status) and included
        # in the loop's own completion check below, so it can't finish
        # invisibly after every OpenRouter model already has.
        copilot_future = pool.submit(_run_copilot_with_status, draft, status_map)

        last_rendered = None
        while not (all(f.done() for f in futures) and copilot_future.done()):
            if on_progress is not None:
                text = _format_audit_progress(status_map)
                if text != last_rendered:
                    on_progress(text)
                    last_rendered = text
            time.sleep(1)
        if on_progress is not None:
            on_progress(_format_audit_progress(status_map))

        audit_results = [f.result() for f in futures]
        copilot_result = copilot_future.result()

    audit_sections = []
    audit_available = False
    for label, text in audit_results:
        profile = profile_by_label[label]
        if text in (OPENROUTER_FAILED_MESSAGE, OPENROUTER_MISSING_KEY_MESSAGE):
            audit_sections.append(f"{label} ({profile}) audit: not available this time.")
        else:
            audit_sections.append(f"{label} ({profile}) audit:\n{text}")
            audit_available = True

    copilot_label = "GitHub Copilot CLI (live web access — the only reviewer that can independently verify a claim, not just judge its internal consistency)"
    if copilot_result == COPILOT_MISSING_MESSAGE or copilot_result.startswith(COPILOT_FAILED_PREFIX):
        # The real reason (a timeout, an auth failure, a non-zero exit
        # code with its actual stderr) travels with the failure, not
        # just a generic "not available" — this is what gets saved into
        # the session record, the only place a real cause survives past
        # the live UI ticker's own lifetime.
        audit_sections.append(f"{copilot_label} audit: not available this time ({copilot_result}).")
    elif copilot_result == _NO_EXTERNAL_NOTES_MESSAGE:
        # Nothing was actually verified — show the note but don't count it
        # as a contributing audit voice (audit_available stays whatever the
        # 10 OpenRouter models already decided).
        audit_sections.append(f"{copilot_label} audit: {copilot_result}")
    else:
        audit_sections.append(f"{copilot_label} audit:\n{copilot_result}")
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

    # A caller-generated session ID lets stage 2 RESUME stage 1's own
    # conversation (see below) instead of paying to resend _INSTRUCTION_
    # HEAD, the whole `summary` market/macro dump, and the stage-1 draft
    # a second time — none of that changed between the two calls, and
    # live-verified `claude -p --resume` correctly retains it (including
    # specific WebSearch-found figures) without needing it repeated.
    session_id = str(uuid.uuid4())
    # Computed ONCE, before stage 1 even runs, so both what a past AUDIT
    # criticized and what the REAL MARKET actually did to past
    # suggestions since reach stage 1's draft, the resumed stage-2
    # session (for free, via session continuity), and every audit model
    # — without re-reading past session files or re-fetching real prices
    # a second time for the audit stage's own benefit.
    _notify(
        "Building past-session context (real market outcomes and audit "
        "lessons from previous runs)..."
    )
    past_lessons = build_past_lessons(fetch_current_price=_fetch_current_pmex_price)
    _notify("Claude is researching the market and drafting an initial suggestion (live web search)...")
    draft_prompt = f"{build_stage1_instruction()}\n\n{summary}"
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
        # Nothing meaningful to record — the draft itself never happened.
        return draft

    _notify(f"Sending the draft to {len(AUDIT_MODELS)} independent free models for audit...")
    audit = build_audit_block(summary, draft, on_progress=on_audit_progress, past_lessons=past_lessons)
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
    # support, etc.) degrade the actual answer — fall back to today's
    # fully self-contained prompt, which needs no session at all, so the
    # worst case is exactly the pre-existing behavior, not a worse one.
    if final_answer == CLI_MISSING_MESSAGE or final_answer.startswith(CLI_FAILED_PREFIX):
        logger.warning(
            "suggest_portfolio: resumed stage-2 call failed (%s), falling back to a full-context retry",
            final_answer[:200],
        )
        _notify(
            "The quick revision attempt didn't respond — retrying with a "
            "fresh, fully self-contained request (takes a bit longer, but "
            "doesn't rely on the earlier session still being live)..."
        )
        revise_prompt = (
            f"{build_stage2_instruction(audit.audit_available)}\n\n{summary}\n\n"
            f"Your own draft from the first pass:\n{draft}\n\n"
            f"Independent audit reports on that draft:\n{audit.block}"
        )
        if past_lessons:
            # The resumed session (where past_lessons originally arrived,
            # inside stage 1's prompt) is exactly what just failed — this
            # brand-new, non-resumed fallback call has no memory of it at
            # all unless it's included here explicitly.
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
            )
        )

    return final_answer
