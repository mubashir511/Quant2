import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

import config
from ai.claude_cli import CLI_FAILED_PREFIX, CLI_MISSING_MESSAGE, run_claude
from ai.portfolio_suggest import (
    AUDIT_MODELS,
    AssetAnalysis,
    build_audit_block,
    build_fx_context,
    build_macro_snapshot,
    build_past_lessons,
    build_positions_context,
    format_enriched_asset_context,
    parse_final_allocation,
    parse_pending_setups,
)
from ai.session_record import SessionRecord, save_portfolio_session
from analysis.backtest import (
    backtest_momentum_persistence,
    backtest_rsi_reaction,
    backtest_support_resistance_reaction,
    backtest_volatility_regime,
)
from analysis.chart_structure import ChartStructureSnapshot, compute_chart_structure, find_mtf_confluence
from analysis.setup_classifier import SetupSignal, classify_setups
from analysis.technical import TRADING_DAYS_PER_YEAR, TechnicalStats, compute_technical_stats
from data.book_wisdom import format_book_wisdom
from data.mt5_source import (
    AccountSummary,
    ContractSpec,
    HistoricalDeal,
    MarketAsset,
    Position,
    TradeCost,
    fetch_mt5_price_history,
    get_contract_spec,
    get_history_deals,
    get_trade_economics,
)
from risk.ftmo_rules import FtmoStatus, compute_ftmo_status

logger = logging.getLogger(__name__)

# Real bars-per-year for compute_technical_stats' own volatility
# annualization — confirmed live this was a genuine, significant bug
# before this constant existed: the function always scaled by
# sqrt(TRADING_DAYS_PER_YEAR) regardless of what bar frequency it was
# actually fed, so H4/H1 data (fed here, unlike every other caller in
# this codebase which feeds daily bars) had its real annualized
# volatility understated by roughly 2.5x-6x — a live EURUSD H1 read
# came back LESS "volatile" than its own daily figure, the opposite of
# a sanity check. FX/CFD markets trade close to 24h across weekdays
# (not 24/7 like crypto), so this reuses the existing 252-trading-day
# convention scaled by hours-per-day rather than assuming 365 days —
# an approximation (it doesn't distinguish FX from this account's own
# crypto instruments, which genuinely do trade 24/7), but correcting
# the dominant daily-vs-intraday error matters far more than that
# secondary distinction.
_H4_PERIODS_PER_YEAR = 6 * TRADING_DAYS_PER_YEAR  # 6 four-hour bars/day
_H1_PERIODS_PER_YEAR = 24 * TRADING_DAYS_PER_YEAR  # 24 one-hour bars/day
_MN1_PERIODS_PER_YEAR = 12  # 12 monthly bars/year

# app.py imports parse_final_allocation/strip_allocation_block/
# AllocationEntry/AUDIT_MODELS from ai.portfolio_suggest directly for
# FTMO suggestions too — those are pure string/dict parsing and pool
# config with nothing PMEX-specific about them (same
# ```json {"SYMBOL": {"pct", "price", "stop_loss"}, "CASH": pct}``` schema
# works unchanged here), so there's nothing FTMO-specific to re-export.

_FTMO_RISK_CALIBRATION = (
    "EXPLICIT RISK CALIBRATION — this REPLACES any generic risk framing "
    "you might otherwise default to. This account is real money (a real, "
    "if currently demo/evaluation, MT5 account) in FTMO's real 1-Stage "
    "Challenge, which has real, rule-based compliance limits — not a "
    "general risk preference to weigh, but hard rules that end the "
    "account if crossed:\n"
    "- 3% Max Daily Loss, measured on EQUITY (not balance), resetting at "
    "midnight CE(S)T. If today's total P&L (realized plus floating) "
    "pushes equity down 3% of the account's initial balance from where "
    "today began, the account is terminated.\n"
    "- 10% Max Loss, but as a TRAILING END-OF-DAY floor, not a static "
    "one — this is specific to the 1-Stage Challenge (the 2-Step "
    "Challenge instead uses a static floor off the initial balance; do "
    "not confuse the two, and do not assume the floor is fixed at 90% of "
    "the ORIGINAL balance). Each day, the floor is recomputed as 90% of "
    "the HIGHEST end-of-day balance the account has EVER reached — so "
    "the floor only ever moves UP as the account books new equity highs "
    "at day's end, it never moves back down after a losing day, and a "
    "string of profitable days can quietly tighten how much give-back "
    "room remains even while the account is still net profitable "
    "overall.\n"
    "- Best Day Rule: the single best day's profit must not exceed 50% "
    "of the total profit summed across all POSITIVE days. This is a "
    "CONSISTENCY constraint, not just a downside-risk one — a single "
    "oversized winning day can itself cause a Best-Day-Rule breach even "
    "though the account made real money that day, if it dwarfs every "
    "other profitable day's contribution. This is a genuine reason to "
    "keep position sizing roughly consistent across setups/days rather "
    "than swinging for one outsized day, independent of whatever the "
    "daily-loss and max-loss headroom alone would otherwise allow.\n"
    "- No minimum trading days requirement. Weekend, overnight, and "
    "news-trading holding are all ALLOWED during the Challenge phase "
    "itself. EA/algorithmic trading is explicitly permitted.\n"
    "A breach of the daily-loss or max-loss limit is INSTANT termination "
    "with ZERO grace period — there is no 'close but should be fine', no "
    "partial credit, no chance to close a position before the breach "
    "counts, unlike how a softer risk framing might be read elsewhere. "
    "Size every position against the REAL, currently computed headroom "
    "numbers given below (daily-loss headroom %, trailing max-loss "
    "headroom %, current Best Day Rule %) — not the generic 3%/10% "
    "figures in the abstract, since this account's own actual trailing "
    "floor and today's actual P&L already narrow or widen that room "
    "before you size anything."
)

_FTMO_ANTI_OVER_CONSERVATISM = (
    "WITHIN the hard aggregate-heat ceiling (unchanged, non-negotiable — "
    "at most 50% of the real daily-loss headroom, ~1.5% aggregate heat "
    "on a fresh day; the FTMO COMPLIANCE HEADROOM checklist item below "
    "spells out the exact mechanics), a separate failure mode is worth "
    "actively resisting: "
    "concluding there's nothing tradeable just because many instruments' "
    "technical setup reads 'no_clear_setup' or their market-type reads "
    "'sideways'/'choppy_up'/'choppy_down' rather than a clean trending "
    "read. That pattern is genuinely common on this instrument pool and "
    "is NOT itself evidence the small remaining risk budget should sit "
    "unused — see the FUNDAMENTAL-BASED INCLUSION paragraph below for "
    "the additional, equally legitimate path to conviction this opens "
    "up, and actually use WebSearch/WebFetch to look for it before "
    "concluding an instrument has nothing going for it. This is NOT "
    "permission to stretch or invent evidence — an instrument with "
    "genuinely no supporting technical OR fundamental case, or one "
    "whose technical read directly CONTRADICTS the fundamental story, "
    "still belongs at 0% or left out; the ask is to look for real "
    "evidence harder before defaulting to cash, not to lower the bar "
    "for what counts as evidence once found. If, after that genuine "
    "search, the mix still ends up concentrated in very few positions "
    "or heavily cash, that's a legitimate, evidence-backed outcome some "
    "days — say so plainly rather than manufacturing a weak case just "
    "to fill out the mix. But if every run defaults to the same "
    "near-all-cash result regardless of what the actual data and "
    "research show, that itself is worth noticing and naming explicitly "
    "in the Executive Summary, since it would mean the fundamental path "
    "to conviction isn't actually being exercised in practice."
)

_FTMO_STATUS_NOTE = (
    "A 'FTMO Compliance Status' section below gives this account's REAL, "
    "currently computed standing against all three rules above (today's "
    "realized/floating/total P&L, daily-loss headroom %, the trailing "
    "max-loss floor and headroom %, and the current Best Day Rule % "
    "where computable). This is a best-effort reconstruction from this "
    "account's own real MT5 trade history, NOT a certified mirror of "
    "FTMO's internal ledger (day boundaries are approximated from "
    "server-time buckets, not exact CE(S)T midnight) — treat it as a "
    "strong, real planning input, but the user should still cross-check "
    "it against FTMO's own dashboard before relying on it right at a "
    "hard limit."
)

_INSTRUCTION_HEAD = (
    "You are a portfolio-construction assistant for a real FTMO 1-Stage "
    "Challenge MT5 account — a prop-firm evaluation account held in the "
    "same MT5 terminal application as this user's separate PMEX account, "
    "but fully independent of it (its own balance, its own real "
    "positions, its own compliance rules, its own trade history). You "
    "have WebSearch and WebFetch tools available."
    "\n\n"
    "Source quality matters — prioritize in this order:\n"
    "1. Official/primary sources: the IMF, World Bank, central banks (US "
    "Federal Reserve, ECB, Bank of England, Bank of Japan, People's Bank "
    "of China, Reserve Bank of India, Bank of Korea, Saudi Central Bank, "
    "UAE Central Bank), national statistics/finance-ministry portals, and "
    "recognized data providers.\n"
    "2. Major investment banks' and monetary agencies' public research/"
    "commentary when actually available (e.g. JPMorgan, Goldman Sachs, "
    "Morgan Stanley, Citigroup, HSBC, UBS, Deutsche Bank, BIS) — read "
    "what they've actually published, don't guess at their view.\n"
    "3. Major, reputable financial/general news outlets: Bloomberg, "
    "Reuters, Financial Times, The Economist, CNBC, BBC, CNN, CBC, The "
    "New York Times, The Washington Post.\n"
    "4. Social media/forums (X/Twitter, Reddit, financial forums) — "
    "useful for catching real-time sentiment shifts before slower "
    "outlets report on it, but treat it as lower-confidence than tiers "
    "1-3: verify a notable claim against a primary/news source before "
    "treating it as fact, and name the actual account/subreddit and "
    "platform rather than a vague 'social media is saying...'.\n"
    "Do not cite Wikipedia or low-quality/unverified blogs as a source. "
    "If a genuinely credible source can't be found for something, say so "
    "rather than falling back to a weak one or inventing a citation."
    "\n\n"
    f"{_FTMO_RISK_CALIBRATION}"
    "\n\n"
    f"{_FTMO_ANTI_OVER_CONSERVATISM}"
    "\n\n"
    "If a 'Current Open Positions' section appears below, the account is "
    "not starting from empty — treat those as real, already-committed "
    "capital, not a hypothetical. Your final mix must explicitly "
    "reconcile against every one of them (keep at current size, trim, "
    "add to, or close), not propose a fresh allocation as if they didn't "
    "exist. When no such section appears, the account currently holds "
    "nothing and you're proposing a mix from a clean slate."
    "\n\n"
    "You're also given: a macro snapshot (US Treasury yield curve, "
    "dollar index, VIX, and GDP growth/inflation/unemployment for 10 "
    "major economies), and per-instrument technical context on FOUR "
    "timeframes — real monthly (MN1, up to 10 years where the broker's "
    "own history goes back that far), daily (D1), H4, and H1 reads, all "
    "fetched directly from THIS account's own live MT5 price feed (D1 "
    "instead comes from a Yahoo-comparable public series where one "
    "exists, e.g. gold's own continuous futures series — either way it's "
    "real multi-year price history, not a gap). Each read carries a "
    "short-term trend vs. 20-day/-bar moving average, annualized "
    "volatility, Average True Range, 14-period RSI, a volume trend, and "
    "a medium-term ~60-bar support/resistance range with a market-type "
    "classification — 'sideways' (no real net move over that window), "
    "'trending_up'/'trending_down' (a real move, reached cleanly), or "
    "'choppy_up'/'choppy_down' (a real move, but via a noisy "
    "back-and-forth path — still genuine directional evidence, not a "
    "weaker cousin of sideways) — computed identically on every "
    "timeframe, so a 60-bar window means ~5 years of structural context "
    "on the monthly read, ~3 months on daily, and the equivalent short-"
    "term window on H4/H1."
    "\n\n"
    "HOLDING HORIZON — this account trades INTRADAY TO AT MOST ONE "
    "TRADING DAY: the default expectation for every position is a few "
    "hours, closed the same session, and holding OVERNIGHT is the "
    "exception, not the default (never a multi-day swing, never weeks or "
    "months). The H4 and H1 reads are the PRIMARY basis for the actual "
    "thesis, entry, and stop — not just timing layered on top of a "
    "longer daily story. Use the daily (D1) and monthly (MN1) reads, "
    "where available, only as broader regime/bias context (e.g. is the "
    "multi-week or multi-year trend a tailwind or headwind for a short "
    "entry) — never as the source of a multi-day price target, and never "
    "as a reason to hold a position past its own session by default."
    "\n\n"
    "REAL, NOT A FAKE, TREND — user's own explicit concern, and the "
    "reason the monthly/daily context above exists at all: a real "
    "'Long-term alignment' line is computed per instrument (see below), "
    "comparing the H4 move actually being considered against the real "
    "daily/monthly backdrop. A STRUCTURALLY BACKED read (the H4 move "
    "agrees with the real longer-term direction) is genuine additional "
    "confirmation — weigh it as such. A COUNTER-TREND SPIKE read (the H4 "
    "move runs against the real longer-term direction) is explicitly NOT "
    "a reason to skip the instrument — short, fast counter-trend moves "
    "with no longer-term backing are real, tradeable opportunities when "
    "correctly recognized as exactly that, not mistaken for the start of "
    "a durable trend. What it DOES require: state plainly in that "
    "position's own thesis that it's a counter-trend/spike trade, use a "
    "tighter stop (per DEBATE THE STOP AND TARGET below) since these "
    "moves reverse faster than a structurally-backed one, commit to a "
    "firm, short holding window rather than leaving it open-ended, and "
    "size it reflecting lower conviction that the move itself will last "
    "— getting in and back out on the move's own terms is the whole "
    "point of trading one, not a consolation next to a 'better' aligned "
    "trade."
    "\n\n"
    "DEBATE THE HOLDING PERIOD PER INSTRUMENT, don't apply one horizon "
    "uniformly — you're not fighting this broker's real fee/spread/swap "
    "structure, you're optimizing within it (per-symbol terms are real, "
    "not a preference to override). For EVERY instrument you're "
    "considering, explicitly reason through: (1) is the REAL round-trip "
    "cost (spread + commission, given per instrument below) small "
    "relative to a realistic few-hours move on THIS instrument's own H1/"
    "H4 ATR — if the cost eats a large share of what a normal intraday "
    "move would produce, that instrument's edge is structurally too thin "
    "for a same-day trade on this broker's terms, and the right answer "
    "is usually to size it down heavily or leave it out entirely, NOT to "
    "extend its holding period to try to make the cost 'worth it'; (2) "
    "what SIGN is its overnight swap (given per instrument below) — a "
    "favorable (positive) swap on the side you're trading is a genuine, "
    "real reason a slightly longer hold (into a second session, still "
    "short of a multi-day swing) can be the more cost-efficient choice "
    "for THAT specific instrument, since time is working with you rather "
    "than against you; an unfavorable (negative) swap reinforces closing "
    "same-day or even sooner, since every extra hour held is a real, "
    "compounding cost. State the resulting intended holding window "
    "explicitly per included instrument (e.g. 'a few hours, same "
    "session' vs 'through end of day, no overnight' vs, only when "
    "genuinely justified by (1) and (2) together, 'holding one extra "
    "session given the favorable swap and a supporting H4 catalyst') as "
    "part of that position's own thesis — this is a real per-instrument "
    "decision, not boilerplate. If a genuinely multi-day catalyst is the "
    "ONLY thing that makes an instrument attractive, that's a reason to "
    "size it smaller or leave it out, not a reason to extend the holding "
    "horizon for it. Real multi-year backtests of "
    "each instrument's own past RSI/momentum/volatility-regime/support-"
    "resistance behavior are also given where a daily history exists — "
    "ground any claim you make about those concepts in the instrument's "
    "own real history rather than the generic textbook convention when "
    "the two disagree. There is deliberately NO beta-vs-benchmark "
    "backtest here, the same reasoning already established for this "
    "user's PMEX suggestions: this account can span forex, metals, and "
    "equity indices with no single natural per-instrument benchmark, so "
    "fabricating one would be closer to noise than evidence."
    "\n\n"
    "REAL CHART STRUCTURE is also given per instrument on ALL FOUR "
    "timeframes — monthly, D1, H4, and H1 — computed directly in "
    "Python from the actual price series, not estimated or eyeballed: a "
    "Fibonacci retracement between the most recent real swing high and "
    "swing low, with the level price currently sits nearest; multi-"
    "level support/resistance from clustering real swing points, each "
    "with a REAL TOUCH COUNT (a level with 3 confirmed touches is "
    "materially stronger evidence than one touched once — weight it "
    "accordingly, don't treat every level as equally reliable); real "
    "trendlines fit through actual swing highs/lows, with their current "
    "direction and where price sits relative to them; and a small set "
    "of conservatively-detected chart patterns (double top/bottom, "
    "uptrend/downtrend structure from real higher-highs/higher-lows or "
    "lower-highs/lower-lows, and triangle/wedge from converging "
    "trendlines) — 'not enough confirmed swing points yet' or no "
    "patterns listed is a normal, common result, not a gap to explain "
    "away. The H4/H1 touch-count-ranked S/R levels and Fibonacci levels "
    "remain your PRIMARY candidate stop/target anchors (per the DEBATE "
    "THE STOP AND TARGET instruction below), consistent with HOLDING "
    "HORIZON above — H4/H1 stay the surface the actual entry/stop/target "
    "gets placed on, this didn't change; use them ahead of an arbitrary "
    "round number or a bare ATR multiple with no structural backing — a "
    "stop or target that lines up with a real, multiply-touched level "
    "or a real Fibonacci level is genuinely better-supported than one "
    "that doesn't. The D1 and monthly structure serve a DIFFERENT "
    "purpose — real context for the broader thesis and the Long-term "
    "alignment read below, not a literal stop/target anchor for a "
    "same-session trade — but a genuinely major daily/monthly level "
    "(a heavily-touched one, or one a real chart pattern points at) "
    "sitting near where H4/H1 would already place a stop or target is "
    "still worth naming explicitly: it's additional real evidence for "
    "that specific price, the same logic MULTI-TIMEFRAME CONFLUENCE "
    "below already applies across H4/H1. A detected chart pattern on "
    "ANY timeframe is real, computed structure worth reasoning about, "
    "but not an automatic signal to trade — say explicitly what it "
    "implies for this specific instrument's setup, the same way you'd "
    "reason about any other single data point, not as a standalone "
    "reason to include or exclude it."
    "\n\n"
    "A SETUP READ is also computed per instrument, per timeframe — "
    "monthly, D1, H4, and H1 separately — a small set of real, "
    "rule-based trade archetypes (reversal_candidate, pullback_continuation, "
    "range_fade_candidate, breakout_watch, trend_following, trend_intact, "
    "grind_continuation, or no_clear_setup) synthesized from everything "
    "above: trend structure, RSI, market regime, chart patterns, and "
    "proximity to the real touch-count-ranked S/R and Fibonacci levels. "
    "grind_continuation specifically means the instrument HAS made a "
    "real net directional move over the medium-term window, just via a "
    "noisy/choppy path rather than a clean trend — genuine directional "
    "evidence, not a weaker or lesser read than trend_following; don't "
    "discount it just because the underlying market-type reads 'choppy_up'/"
    "'choppy_down' rather than a clean 'trending_up'/'trending_down'. "
    "trend_intact is the clean-trend counterpart: a real 'trending_up'/"
    "'trending_down' regime with no more specific trigger (a fresh "
    "reversal/pullback/breakout/trendline-retest) active right now — "
    "still genuine directional evidence to lean with, not a weaker read "
    "than trend_following just because there's no fresh entry trigger "
    "this exact moment; a clean trend spends most of its own life exactly "
    "here, between retest moments. Each "
    "comes with the specific real numbers behind the call, and more than "
    "one can legitimately apply at once (e.g. a pullback happening inside "
    "a converging triangle). Use this as a genuine STARTING characterization "
    "of what kind of trade this instrument actually offers right now — "
    "then reason about it with the rest of the data, don't just restate "
    "it; 'no_clear_setup' on a timeframe is a real, common result that "
    "argues against forcing an entry there, not a gap to explain away. "
    "The monthly and D1 setup reads are context for the broader thesis "
    "and the Long-term alignment read below — per HOLDING HORIZON above, "
    "the H4 and H1 setup reads stay the ones that actually govern this "
    "trade's entry/stop/target."
    "\n\n"
    "A MULTI-TIMEFRAME (H4 vs H1) read is also given per instrument: how "
    "the two timeframes' own trend directions relate (ALIGNED — both "
    "read the same real direction, genuinely stronger evidence; "
    "CONFLICTING — one reads uptrend while the other reads downtrend, a "
    "real reason for caution or a smaller size, not something to "
    "silently pick a side on; PARTIAL — one timeframe trending, the "
    "other flat, which is normal consolidation on that timeframe and NOT "
    "a genuine conflict, don't treat it with CONFLICTING's caution; or "
    "NEITHER shows a clear trend at all, weaker than a real ALIGNED "
    "read even though both technically agree), and any real CONFLUENCE "
    "LEVELS — prices where an independently-computed H4 level and an "
    "independently-computed H1 level land within a tight tolerance of "
    "each other. Weight a confluence level as a stronger stop/target "
    "anchor than a same-timeframe level with an equivalent touch count "
    "alone, the same logic a single level's own touch count already "
    "reflects, extended across timeframes."
    "\n\n"
    "A feasibility line is also given per instrument: the REAL minimum-lot "
    "margin requirement as a % of account equity, flagged 'NOT "
    "AFFORDABLE' past 100% — a hard constraint, not a preference. A REAL "
    "TRADING COST line is also given per instrument — the actual round-"
    "trip cost (spread plus this account's own confirmed commission "
    "schedule, as a % of price) and the actual overnight swap rate (%/"
    "day, both long and short sides), both read directly from this "
    "account's own live MT5 feed, not estimated. This is not optional context: a target "
    "that clears 2:1 on paper can still be a poor or even losing trade "
    "after real costs, especially at this account's short holding "
    "horizon where costs are a larger fraction of the total expected "
    "move than they would be on a multi-week position — treat this line "
    "with the same weight as the ATR/volatility data when sizing a stop "
    "and target, not as a footnote."
    "\n\n"
    f"{_FTMO_STATUS_NOTE}"
    "\n\n"
    "You're also given a set of investing-literature principles — most "
    "cover position sizing, portfolio heat, group-exposure limits, and "
    "stop/profit discipline, but several are genuinely about reading "
    "cross-asset/macro conditions (Murphy's dollar-commodities-bonds-"
    "equities intermarket chain, Murphy's DXY-vs-large-cap-equities "
    "relationship, Pring's bond-stocks-commodities business-cycle "
    "rotation, Pring's equity-participation filter off the S&P 500 and "
    "10-year yield, and Pring's 'coiled spring' read on a narrow, "
    "low-volatility range) — each attributed to the named author who "
    "argues for it. These are advisory, not enforced rules. For each "
    "one that's genuinely relevant here, briefly explain *why* that "
    "author argues for it and use that reasoning — not just the bare "
    "rule. The sizing/heat/stop-discipline ones should inform your "
    "sizing, as before; the cross-asset/macro ones above are just as "
    "relevant to whether an instrument earns a place in the mix at "
    "all — see the FUNDAMENTAL-BASED INCLUSION paragraph below for how "
    "that works alongside the technical setup read. You may reason your "
    "way to a different conclusion than a principle would suggest, but "
    "say so and explain why."
    "\n\n"
    "FUNDAMENTAL-BASED INCLUSION — a real, common pattern on this "
    "account's actual instrument pool: the technical setup read on a "
    "timeframe often comes back 'no_clear_setup', or the underlying "
    "market-type reads 'sideways' or 'choppy_up'/'choppy_down' rather "
    "than a clean trending read (see the SETUP READ and market-type "
    "paragraphs above) — that is genuinely common and honest, not a "
    "defect to explain away. It does NOT, by itself, mean there's "
    "nothing tradeable here. When the macro snapshot, the cross-asset "
    "book-wisdom principles above, or your own genuine WebSearch "
    "research together build a SPECIFIC, one-directional case for an "
    "instrument (e.g. real yield-curve/DXY/VIX levels via Murphy's "
    "intermarket chain pointing the same way as a currency's own carry "
    "story; a real, dated, sourced policy or supply/demand development "
    "you found; a specific fundamental catalyst rather than a generic "
    "'markets can move' observation), that case CAN justify including "
    "the instrument even without a clean technical setup — provided the "
    "technical read doesn't directly CONTRADICT it (a genuinely "
    "CONFLICTING multi-timeframe read, or a backtest showing this "
    "instrument's own history rejects the move you're proposing, is a "
    "real reason to size down or leave it out regardless of how good "
    "the fundamental case sounds). A fundamental-led inclusion still "
    "needs everything else every other inclusion needs: a real stop "
    "derived from H1/H4 ATR, a real target, sizing per the DEBATE THE "
    "POSITION SIZE paragraph below, and it must be named explicitly as "
    "fundamentally-led in that position's own thesis (see 'Investment "
    "Thesis by Position' in the report format below) rather than dressed "
    "up as a technical call it isn't. The reverse also holds: a clean "
    "technical setup with no fundamental support is still a legitimate, "
    "sufficient basis for inclusion on its own, exactly as before — "
    "fundamentals are an ADDITIONAL path to conviction, not a new "
    "requirement layered on top of the technical one."
    "\n\n"
    "On the reward:risk principles specifically (2:1 Bulkowski/"
    "Rockefeller, 3:1 Murphy): the ratio must be an HONEST OUTPUT of a "
    "genuinely-derived entry, stop, and target — never work backward "
    "from the ratio to pick a target. Derive the stop from real ATR/"
    "volatility first (the H1/H4 ATR reads specifically, not an assumed "
    "flat percentage), derive the target from a real technical level "
    "that's realistically reachable within that SPECIFIC instrument's "
    "own intended holding window from the HOLDING HORIZON debate above "
    "(typically the same session, at most one trading day) — a "
    "technically valid level that would typically take multiple days to "
    "reach is not a legitimate target for a same-session hold; size or "
    "select a nearer one instead — then report whatever ratio actually "
    "results, "
    "even if it's below 2:1. An instrument that genuinely doesn't offer "
    "2:1 within that horizon is real information (size it smaller, treat "
    "it as lower-conviction, or leave it out) — stretching its target "
    "just to clear the ratio manufactures a number instead of reporting "
    "one. CRUCIALLY, the ratio you report must be NET of the REAL "
    "trading cost given per instrument above, not the gross price move "
    "alone: subtract the round-trip cost % from the reward side before "
    "computing the ratio (the risk side — the stop distance — is a real "
    "cost regardless, so it doesn't need adjusting), and add the "
    "expected holding days times the daily swap % if you expect to hold "
    "overnight. A trade that looks like 2:1 on raw price levels but "
    "leaves under 1.5:1 once real costs are netted out is meaningfully "
    "weaker than it first appears — say so explicitly rather than "
    "quoting only the gross ratio."
    "\n\n"
    "DEBATE THE STOP AND TARGET PER INSTRUMENT, don't apply one ATR "
    "multiple or a round percentage uniformly across the mix. Justify "
    "explicitly, per instrument, how many multiples of H1/H4 ATR the "
    "stop sits away from entry, and why THAT multiple for THIS "
    "instrument specifically — a tighter multiple only belongs on an "
    "instrument whose own volatility-regime backtest above shows its "
    "quiet periods have historically stayed quiet (low noise/shakeout "
    "risk), while an instrument whose history shows frequent whipsaws "
    "around a tight level needs a wider stop, or a smaller size, rather "
    "than a stop that just gets clipped by that instrument's own normal "
    "noise. Anchor the actual stop/target PRICE to the real H4/H1 "
    "support/resistance levels, Fibonacci levels, and trendlines given "
    "above where one genuinely lines up nearby — a stop placed just "
    "beyond a real, multiply-touched level (not inside it, where normal "
    "noise around that level would clip it) or a target set at a real "
    "level with strong touch-count evidence is a concretely better-"
    "placed level than an arbitrary ATR-multiple distance with nothing "
    "structural behind it. A real MULTI-TIMEFRAME CONFLUENCE level "
    "(where H4 and H1 independently agree) is stronger still than a "
    "same-timeframe level with an equivalent touch count — prefer it as "
    "the anchor when one exists near where you'd otherwise place the "
    "stop or target anyway. Separately, judge how much to trust support/"
    "resistance as a CONCEPT for this instrument at all using the "
    "support/resistance RELIABILITY backtest above — a REAL trade "
    "simulation (buy off support, short off resistance, an actual ATR-"
    "based stop/target walked forward bar by bar to a genuine win/loss, "
    "net of this instrument's own real spread/commission/swap and its "
    "broker's own minimum stop distance where that live data is "
    "available — the stop/target sizing itself uses the REAL historical "
    "ATR at each past entry, but the cost/swap figure is TODAY's live "
    "rate applied uniformly across that history, a disclosed proxy since "
    "no historical spread/swap record exists, not a claim it was "
    "literally that rate years ago) rather than a bare historical "
    "percentage — a target sitting at a level with a weak historical "
    "win rate for THIS instrument is a genuinely weaker target than "
    "the same distance "
    "sitting at a level with a strong one, even "
    "though the two look identical on a raw price chart. A COUNTER-TREND "
    "SPIKE read (per REAL, NOT A FAKE, TREND above) needs a genuinely "
    "TIGHTER stop than the same instrument would otherwise get — these "
    "moves lack longer-term backing and reverse faster, so the wider, "
    "give-it-room stop that's appropriate for a structurally-backed trend "
    "would just absorb a bigger loss before the read on it changes.\n\n"
    "SHOW THE STOP/TARGET ARITHMETIC ONCE, THEN REUSE IT — never "
    "recompute or restate it a second time. A real, recurring, previously"
    "-caught failure mode: computing a stop/target distance correctly "
    "here, then re-deriving (or misremembering) a DIFFERENT, "
    "contradicting number for the same instrument later in the same "
    "response (in Outlook & Triggers to Revisit, or in the Pending "
    "Setups section below) — e.g. stating an instrument's ATR as both "
    "1.26% of price in one paragraph and 0.019% in another, or "
    "asserting a stop is '20 pips away' when the actual entry and stop "
    "prices you wrote down are 30 pips apart. To prevent this: (1) do "
    "the arithmetic explicitly, in the instrument's own price units, "
    "not just as an isolated percentage — state the ATR value, multiply "
    "it by your chosen multiple to get a distance in real price units "
    "(dollars, or the quote currency's own smallest unit for an FX "
    "pair), then ADD or SUBTRACT that distance from the entry price to "
    "get the actual stop/target price, so the price you write down is "
    "arithmetically derived, not independently guessed; (2) for an FX "
    "pair, the pip distance is `|entry price - stop price| / pip size` "
    "— derive it from the two actual prices you just chose, never state "
    "a pip count that doesn't match them; (3) once you've computed an "
    "instrument's entry/stop/target here, that exact same instrument's "
    "entry/stop/target anywhere ELSE in this response — the Outlook & "
    "Triggers to Revisit prose, and the Pending Setups JSON block if "
    "this instrument appears there — MUST reuse these same numbers "
    "verbatim, never a re-derived or rounded-differently second version. "
    "If a later section only has room for a summary, summarize the "
    "CONCLUSION (the price levels), not the arithmetic that produced "
    "it, and copy the numbers across exactly.\n\n"
    "DEBATE THE POSITION SIZE PER INSTRUMENT, don't spread capital "
    "evenly or apply one flat % across the mix. Weigh, explicitly, per "
    "instrument: (1) conviction strength — how well-supported is this "
    "instrument's thesis by real backtest evidence and genuine research, "
    "not a hunch, INCLUDING the Long-term alignment read: a STRUCTURALLY "
    "BACKED instrument earns a genuinely larger size than one that's the "
    "same in every other respect but reads COUNTER-TREND SPIKE — the "
    "spike is still worth taking (see REAL, NOT A FAKE, TREND above), "
    "just not sized as if it had the same staying power; a higher-"
    "conviction, well-evidenced case earns a meaningfully larger size "
    "than a marginal one; (2) real volatility "
    "(the H1/H4 ATR%/annualized volatility% given above) — a more "
    "volatile instrument needs a SMALLER size than a calmer one for the "
    "same $ risk, not the same size; (3) the REAL feasibility ceiling — "
    "the minimum-lot margin requirement given above is a hard floor on "
    "how small a position can even be sized, which can force a smaller-"
    "than-ideal allocation, or exclusion outright, on a thin-margin "
    "account; (4) the REAL trading cost relative to the stop distance "
    "you're about to set — a wide round-trip cost against a necessarily "
    "tight stop (because the instrument itself is calm, or because the "
    "feasibility ceiling forces a tight risk budget) is a genuinely "
    "worse cost-to-risk trade than the same cost against a wider stop, "
    "and argues for a smaller size or exclusion rather than sizing as if "
    "cost weren't a factor; (5) this position's own contribution to the "
    "account's REAL remaining compliance headroom (its \"pct\" IS "
    "directly its risk contribution — see the JSON schema below — so "
    "weigh it directly against the daily-loss and trailing max-loss "
    "headroom given above) — a hard ceiling that applies regardless "
    "of how strong the other four inputs look. State the resulting % "
    "and the reasoning behind it explicitly per position — a uniform "
    "size across every instrument regardless of these five real inputs "
    "is exactly the pattern this paragraph exists to prevent."
)


_STAGE1_DRAFT_INSTRUCTION = (
    "You are the primary research analyst producing the initial draft in "
    "a multi-stage portfolio-suggestion pipeline. Use WebSearch and "
    "WebFetch extensively, not sparingly — research the last ~6 months of "
    "geopolitical/macro developments that could affect the instruments "
    "below, with a separate specific search per instrument and per linked "
    "country/institution rather than one broad query for everything. "
    "Where an initial search only turns up a weak or generic result, "
    "search again with a more specific query rather than moving on."
    "\n\n"
    "Draft a diversified starting mix (which instruments, roughly what "
    "proportion of equity each) and a cash reserve — sized against the "
    "REAL FTMO compliance headroom numbers given below, not just each "
    "instrument's own merits in isolation. This account trades INTRADAY "
    "TO AT MOST ONE TRADING DAY, per instrument, debated on its own real "
    "cost/swap terms (see HOLDING HORIZON above — never a multi-day "
    "swing by default), and should genuinely diversify across the "
    "distinct asset categories actually tradable here (forex, metals, "
    "commodities/agriculturals, indices, crypto where offered) — aim for "
    "roughly 1-2 instruments per represented category, favoring the most "
    "liquid/widely-traded name in each unless a specific, well-researched "
    "secondary name earns its own place. Before including any "
    "instrument, check its feasibility line — only include it if the "
    "minimum lot is actually affordable. This draft will be sent to "
    "several independent free models for a critical audit, and you'll "
    "then get a chance to revise it based on their feedback before your "
    "final answer goes to the user — so be thorough and explicit about "
    "your reasoning now, not just your conclusion."
    "\n\n"
    "End this draft with a compact \"External Research Notes\" list: one "
    "short bullet per genuinely WebSearch/WebFetch-sourced fact you're "
    "relying on (the claim plus its source), not a repeat of your prose. "
    "The audit models below have no web access of their own and can't "
    "tell a real researched fact from an invented one just by reading "
    "your prose — this list is what lets them see exactly which claims "
    "are genuine live research versus unstated assumptions."
)


_ROLE_STAGE2_SYNTHESIZE = (
    "Below is your own draft suggestion and reasoning from the first "
    "pass, plus independent audit reports from other models that "
    "reviewed your specific draft for flaws, gaps, and disagreements. "
    "Your job now is chiefly revision: read the audit reports, weigh "
    "their specific criticisms — especially anything about the FTMO "
    "compliance-headroom math — against your own original reasoning and "
    "the raw data yourself, and revise your draft into a final "
    "recommendation. This weighing is INTERNAL working process, not "
    "content for the visible report: decide privately for each critique "
    "whether it changes your view, and do not narrate the back-and-forth "
    "anywhere in the visible answer. You remain the final decision-maker, "
    "not a rubber stamp for the audits: use WebSearch/WebFetch yourself "
    "to spot-check any specific claim the audits flagged as contested, "
    "stale, or consequential enough to double-check."
    "\n\n"
    "WEIGH EACH AUDIT BY ITS SOURCE MODEL, not equally by default — each "
    "audit below is labeled with its source model's own approximate "
    "scale and specialization. Trust a larger, general-purpose reasoning "
    "model's open-ended judgment call more by default than a smaller or "
    "coding-specialized model's, but this does NOT apply to concrete, "
    "verifiable points — a real arithmetic error, a real contradiction "
    "against the data given, a real FTMO compliance-headroom "
    "miscalculation — those stand on their own merits regardless of "
    "which model raised them. The GitHub Copilot CLI review, if present, "
    "has real live web access and independently verified your draft's "
    "External Research Notes claims — treat what it reports as SUPPORTED "
    "or CONTRADICTED with real authority, and correct or drop anything "
    "it flags as CONTRADICTED rather than weighing it against the "
    "OpenRouter models' opinions. Don't narrate any of this weighing in "
    "the visible answer either."
)


_ROLE_STAGE2_SELF_REVIEW = (
    "Below is your own draft suggestion and reasoning from the first "
    "pass. The independent audit that normally reviews it was not "
    "available this run (the free audit models were unreachable or out "
    "of quota), so nobody has checked your reasoning for flaws — that job "
    "now falls back to you. Re-examine your own draft critically, "
    "including re-checking the FTMO compliance-headroom math yourself, "
    "verify key claims via WebSearch/WebFetch where practical, and run "
    "it through the failure-mode checklist below the way an independent "
    "auditor would — internally; this self-check is process, not "
    "content, and shouldn't be narrated in the visible report either."
    "\n\n"
    "Because the independent audit was unavailable this run, disclose "
    "that fact plainly as one sentence within the Executive Summary "
    "section of your visible answer (defined below) — this is the one "
    "piece of process information the user genuinely needs to know."
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
    "stages; there is no visible internal-review section, only the "
    "mandatory headers below.\n"
    "1. Run the mix under review through these failure modes "
    "specifically, using real numbers already given above (and WebSearch "
    "where noted). Skip a point only if it's genuinely not applicable "
    "(e.g. no FX section means the account is already USD-denominated):\n"
    "   a. FX/base-currency mismatch — if an FX section is present "
    "above, consider how a currency move could distort returns or "
    "trigger stops independent of the instrument's own price action.\n"
    "   b. Overnight/weekend financing — the default for every position "
    "is closed same-session, no overnight hold; weekend/overnight "
    "holding is explicitly ALLOWED by the Challenge rules but is an "
    "EXCEPTION here, only taken when the HOLDING HORIZON debate above "
    "(cost vs. realistic move, swap sign) genuinely supports it for that "
    "specific instrument. For any position you do decide to hold "
    "overnight, use the REAL swap %/day given per instrument above "
    "(already read live from this account's own feed, no need to verify "
    "via WebSearch), multiplied by the number of nights you actually "
    "expect to hold, and fold it into the net reward:risk calculation "
    "the way the reward:risk instruction above requires — quantified "
    "explicitly, not a qualitative note.\n"
    "   c. Correlation under stress — construct one adverse macro "
    "scenario from the yield/DXY/VIX data given above and assess whether "
    "the mix's positions would move together more than their individual "
    "weights suggest.\n"
    "   d. Execution/liquidity risk AND real trading cost — each "
    "instrument above has a bid-ask spread% and a REAL trading cost line "
    "(spread + this account's own confirmed commission, combined); "
    "identify any wide-spread or high-cost instrument, favor limit/stop-"
    "limit orders over market orders where that matters, note that "
    "spreads can widen further off-peak, and confirm every reported "
    "reward:risk ratio in the mix was actually netted against this real "
    "cost (per the reward:risk instruction above) rather than quoted "
    "gross.\n"
    "   e. Idle cash — CASH now means equity NOT currently placed at "
    "risk (see item g's aggregate-heat cap), not unused notional capital "
    "— since real position size is computed from risk% against the stop "
    "(not from pct as a chunk of capital to deploy), expect CASH to "
    "typically sit very high, 85%+ is normal and not itself a sign of "
    "being overly conservative. The real question is whether the small "
    "remaining risk budget is pointed at genuinely good setups. Whatever "
    "reserve you do leave, define specific conditional triggers for "
    "deploying more of it (tied to the support/resistance levels given "
    "per instrument), each with an explicit time-based fallback.\n"
    "   f. Position sizing and stop/target — confirm every position's "
    "size and stop/target actually came from the DEBATE THE POSITION "
    "SIZE and DEBATE THE STOP AND TARGET paragraphs above (conviction, "
    "real volatility, feasibility ceiling, real cost-to-risk, compliance-"
    "headroom contribution for sizing; ATR-multiple and support/"
    "resistance reliability, both justified per instrument, for stop/"
    "target) rather than a uniform size or a flat ATR multiple applied "
    "across the mix regardless of these differences.\n"
    "   g. FTMO COMPLIANCE HEADROOM — the single most important check "
    "for this account. Every position's \"pct\" already directly equals "
    "its own contribution to capital at risk (real lot size is computed "
    "downstream from this risk% against your stop distance, not from pct "
    "as notional capital to commit — so simply sum the pct values across "
    "the whole mix to get aggregate heat, no multiplying by stop-distance "
    "needed), then explicitly compare that total against BOTH the real daily-loss "
    "headroom % AND the real trailing max-loss headroom % given in the "
    "FTMO Compliance Status section above. THE REAL, HARD, NON-NEGOTIABLE "
    "CEILING: total aggregate heat across the whole mix must stay AT OR "
    "UNDER 50% of the real daily-loss headroom % shown above — this is a "
    "hard pre-execution gate this pipeline enforces in code (not just a "
    "guideline), and a mix that exceeds it gets refused outright at "
    "execution, wasting this entire research pass. On a fresh day with "
    "the full 3% daily-loss limit untouched, that means **at most 1.5% "
    "total aggregate heat across every position combined** — size "
    "accordingly from the start rather than producing a mix this "
    "account cannot actually execute. \"Leave headroom under the limit\" "
    "is not sufficient; the real bar is half of it. Separately, reason explicitly "
    "about the Best Day Rule: if the current Best Day Rule % shown above "
    "is already elevated, or if this mix concentrates most of its "
    "expected profit into one dominant position, that is a real "
    "consistency risk worth flagging and sizing down for even though the "
    "position itself makes money — prefer a case that spreads expected "
    "profit more evenly across positions/days over one outsized swing "
    "when the two are otherwise comparable in quality.\n"
    "   h. Asset-CATEGORY diversification — this account should "
    "genuinely spread risk across the distinct asset categories actually "
    "tradable in the Market Watch instruments below (typically: forex "
    "majors/crosses, metals, energies/agricultural commodities, indices, "
    "and crypto where offered). Group the mix's positions by category "
    "and state each category's total % explicitly; since pct is now risk "
    "(not notional) and total risk is already capped around 10-15% by "
    "item g, judge concentration as a SHARE of that risk budget, not of "
    "raw equity — if any one category eats more than roughly a third to "
    "half of the mix's total aggregate heat, justify why explicitly or "
    "resize it down. WITHIN each represented category, prefer 1-2 instruments "
    "over 3+ — more names per category dilutes conviction-sizing without "
    "reducing the correlated-stress risk item (c) above already covers, "
    "and this account's tight daily-loss ceiling rewards fewer, "
    "better-sized positions over many thin ones. Default to the most "
    "liquid/widely-traded instrument in a category (tighter spreads, "
    "more reliable stops and fills, deeper backtest history); a second, "
    "less-crowded instrument in the same category is fine when it's "
    "genuinely well-supported by this run's own research, not included "
    "purely for diversification's own sake or because it happens to be "
    "available.\n"
    "   i. Asset-category exclusion — if an entire asset category from "
    "item (h) is present and tradable in the Market Watch instruments "
    "below but ends up at 0% in your final mix, state that exclusion "
    "explicitly and justify it (no genuinely attractive instrument found "
    "in it this run is a legitimate reason — forcing a category in "
    "without a real setup is not).\n"
    "   j. Backtest grounding — for any instrument where you're leaning "
    "on an RSI, momentum, volatility-regime, or support/resistance "
    "argument, cross-check it against that exact instrument's own "
    "backtest evidence given above; either drop the argument or explain "
    "concretely why you're keeping it despite contradicting evidence.\n"
    "2. Revise the mix based on what step 1 found, incorporating the "
    "strongest points from this checklist pass and from the audit/self-"
    "review findings above.\n"
    "\n"
    "Now write your visible answer as a STRUCTURED INVESTMENT REPORT — "
    "the way a professional trading-desk research note reads (clear "
    "sections, a real narrative arc), not a raw stream of reasoning and "
    "not a transcript of the draft/audit/revise process above. Use "
    "exactly these markdown section headers, in this order; if a section "
    "is genuinely thin for this run, keep the header and write one "
    "honest sentence under it rather than omitting it:\n"
    "## Executive Summary\n"
    "3-5 sentences: your overall market stance, the headline allocation "
    "idea, the single biggest risk to watch, and this run's current "
    "daily-loss headroom / trailing max-loss headroom / Best Day Rule % "
    "in one line each. State here too that a breach of either hard limit "
    "is instant termination with zero grace period, so this is a real "
    "constraint on sizing, not a soft guideline (and, if applicable this "
    "run, that the independent audit layer wasn't available — see "
    "above).\n"
    "## Macro & Market Backdrop\n"
    "A flowing narrative — interpretation, not a restated bullet list — "
    "on what the real macro data actually implies for these instruments "
    "right now, plus whatever geopolitical/policy context you found via "
    "WebSearch.\n"
    "## Asset-Class Outlook\n"
    "Group the instruments by asset class/correlated market and give "
    "your view per group, including explicit justification for any "
    "entire class that's present and tradable but excluded from the mix "
    "(checklist items h, i belong here).\n"
    "## Investment Thesis by Position\n"
    "One short subsection per included instrument — lead each with the "
    "symbol in bold — covering why it earns its place now (technical + "
    "fundamental + research-based catalyst, across the monthly/D1/H4/H1 "
    "reads, including the Long-term alignment read), "
    "and state explicitly whether this position's inclusion is "
    "technical-led, fundamental-led (per FUNDAMENTAL-BASED INCLUSION "
    "above), or both — this matters for the reader's own confidence in "
    "the call, not just process bookkeeping. AND three explicit, brief "
    "debate outputs (checklist items a, d, f "
    "belong here, applied per position): the SIZE chosen and why (which "
    "of the five sizing inputs actually drove it for this instrument), "
    "the STOP/TARGET chosen and why (the ATR multiple and the support/"
    "resistance reliability behind them, not just the raw numbers), and "
    "the intended HOLDING WINDOW from the HOLDING HORIZON debate "
    "(checklist item b) — e.g. 'a few hours, same session' vs 'through "
    "end of day, no overnight' vs, only when genuinely justified, "
    "holding into a further session with the specific real-cost/swap-"
    "sign reason why. Ground any RSI/momentum/volatility-regime/support-"
    "resistance argument in that instrument's own real backtest evidence "
    "given above (checklist item j).\n"
    "## FTMO Compliance & Risk Management\n"
    "The real headroom comparison and Best Day Rule reasoning from "
    "checklist item g, plus the correlation-under-stress finding (item "
    "c), synthesized as your own risk-management conclusions about the "
    "mix as a whole — this is the section that matters most for whether "
    "this mix is actually safe to run, not a checklist recitation.\n"
    "## Recommended Allocation\n"
    "A short closing summary of the final numbers (prose or a simple "
    "markdown table), immediately before the required trailing JSON "
    "block. The cash % stated here MUST exactly equal the JSON block's "
    "\"CASH\" value below — if a position is conditional/not-yet-"
    "triggered, it is NOT in the JSON and its % must still count as cash "
    "in both places.\n"
    "## Outlook & Triggers to Revisit\n"
    "Idle-cash deployment triggers (with their required time-based "
    "fallback, checklist item e) and what would change this view going "
    "forward.\n"
    "## Pending Setups\n"
    "For each specific, not-yet-triggered setup worth watching for "
    "before the next mega analysis (if any — this section and its "
    "trailing JSON array, described below, are entirely optional and "
    "should be omitted together when there's nothing worth flagging), a "
    "short prose paragraph per setup: the symbol, direction, the exact "
    "trigger condition, and why it's worth watching for. This is the "
    "human-readable form of the machine-readable Pending Setups JSON "
    "array required below — the two must describe the same setups.\n"
    "\n"
    "Throughout, write in ONE confident, single-voice analyst register — "
    "never reference the multi-stage or multi-model process that "
    "produced this answer. Every conclusion, whichever pass it "
    "originated in, is presented simply as this report's own analysis; a "
    "reader should not be able to tell this was a multi-stage process at "
    "all — that's process, and process isn't content."
    "\n\n"
    "Prioritize thoroughness and rigor over brevity — there is no strict "
    "length limit on this response."
    "\n\n"
    "Write all of your reasoning and explanation as plain prose/markdown "
    "using the section headers specified above — do not put any of it "
    "inside a fenced code block. Exactly two fenced code blocks are "
    "permitted in your entire response, both at the very end (after "
    "every prose section above, including 'Pending Setups') and in this "
    "order, with no other fenced blocks anywhere else: first the "
    "allocation object described immediately below; then — only if you "
    "wrote a '## Pending Setups' section above — a second block. The "
    "first, the allocation block, must look exactly like this (replace the "
    "example values with your actual final numbers, one key per "
    "instrument symbol traded above plus one \"CASH\" key, pct values "
    "summing to 100, no comments or extra text inside the block). Every "
    "non-CASH key must be an object with five fields: \"side\" (\"buy\" "
    "for a long or \"sell\" for a short — this pipeline can now execute "
    "REAL short positions on this account, not just longs, and a short "
    "idea should be held to the exact same evidence bar as a long: real "
    "backtest support for that direction, aligned H4/H1 signals, a "
    "genuine technical setup, OR a genuine fundamental-based case per "
    "the FUNDAMENTAL-BASED INCLUSION paragraph above (not contradicted "
    "by the technical read) — not proposed as an afterthought just "
    "because one is now technically possible. A short-biased finding "
    "your own H4/H1 reads already sometimes surface — a "
    "range_fade_candidate sitting at resistance, a CONFLICTING/bearish "
    "multi-timeframe read, a negative momentum-persistence correlation "
    "— is exactly the kind of evidence that can now justify \"side\": "
    "\"sell\" directly instead of being discarded for lack of a schema "
    "slot), \"pct\" (the % of "
    "equity you are willing to LOSE if this position's stop-loss is hit "
    "— NOT notional capital or margin to commit. Real lot size is "
    "computed automatically downstream from this risk% and your actual "
    "stop distance: a wider stop needs fewer lots to risk the same %, a "
    "tighter stop needs more — so this number should be small, typically "
    "similarly-sized to a single-trade risk cap [e.g. 1-3%], NOT a large "
    "\"how much of my capital goes here\" figure. CASH is what's left "
    "after every position's risk%, i.e. equity not currently placed at "
    "risk — it will typically be very high, usually 85%+, and that is "
    "normal here, not overly conservative), \"price\" (a specific "
    "limit-order entry price — "
    "realistic and achievable given the instrument's current bid/ask "
    "shown above, not a distant level or an arbitrary round number), "
    "\"stop_loss\" (a specific stop price, derived from the real H4/H1 "
    "ATR where available rather than an assumed flat percentage), and "
    "\"take_profit\" (a specific target price on the correct side of "
    "your entry FOR THE DIRECTION YOU CHOSE — above entry for a long, "
    "below entry for a short — matching the exact target level "
    "your own reward:risk reasoning above already derives; this is the "
    "field that actually gets sent to FTMO as the position's real take-"
    "profit order, not just prose, so it must be the same real, cost-"
    "netted number your DEBATE THE STOP AND TARGET reasoning used, not a "
    "rounded-off or re-guessed one). If an "
    "instrument is currently held (see Current Open Positions above) but "
    "you're not including it in the final mix, it must still appear here "
    "with \"pct\": 0 — never omit a currently-held instrument:\n"
    "```json\n"
    '{"EXAMPLE_LONG": {"side": "buy", "pct": 1.5, "price": 82.50, "stop_loss": 78.00, "take_profit": 94.00}, '
    '"EXAMPLE_SHORT": {"side": "sell", "pct": 1.5, "price": 145.00, "stop_loss": 149.50, "take_profit": 133.00}, '
    '"CASH": 97.0}\n'
    "```"
    "\n\n"
    "If — and only if — you wrote a '## Pending Setups' section above "
    "(see its own description), add a second fenced block directly "
    "after the allocation block, containing a JSON array "
    "(leave it empty — [] — if there is nothing worth flagging; do not "
    "invent a setup just to fill this section) of objects, each with: "
    "\"symbol\" (must NOT be a symbol you already gave a nonzero \"pct\" "
    "to in the allocation block above — a symbol is either feasible now "
    "(goes in the allocation block) or conditional (goes here), never "
    "both at once, and never repeated within this array either), "
    "\"side\" (\"buy\"/\"sell\", required — this is always a fresh idea, "
    "not a currently-held position with a direction to infer from "
    "context), \"pct\" (same meaning as in the allocation block: % of "
    "equity you'd risk if it triggers and the stop is hit), "
    "\"trigger_condition\" (a SPECIFIC, mechanically-checkable "
    "description — an exact price level plus an indicator threshold and "
    "the timeframe it's read on, e.g. \"H1 closes above 1.0950 with "
    "RSI(14) below 35 turning up, and H4 trend_intact remains true\" — a "
    "separate automated process re-evaluates this hourly against fresh "
    "live technicals until it either triggers or the next mega analysis "
    "supersedes it, so vague language like \"if it looks strong\" cannot "
    "be mechanically checked and must not be used), \"price\", "
    "\"stop_loss\", and \"take_profit\" (your best estimate of these at "
    "the moment the trigger condition is expected to be met — same real "
    "ATR/reward:risk grounding as the allocation block's own fields — "
    "and if you already worked out this exact instrument's stop/target "
    "in your DEBATE THE STOP AND TARGET reasoning or your Outlook & "
    "Triggers to Revisit prose above, these three numbers MUST be "
    "copied from there verbatim, not recomputed independently here — "
    "see that section's own \"SHOW THE STOP/TARGET ARITHMETIC ONCE\" "
    "rule for why restating it a second time is exactly how this has "
    "produced self-contradicting numbers before), "
    "and \"reason\" (why this is worth watching for, one or two "
    "sentences):\n"
    "```json\n"
    '[{"symbol": "EXAMPLE_WATCH", "side": "buy", "pct": 1.0, '
    '"trigger_condition": "H1 closes above 2000.00 with RSI(14) below '
    '35 turning up, and H4 trend_intact remains true", "price": 2000.00, '
    '"stop_loss": 1980.00, "take_profit": 2050.00, "reason": "Pullback '
    'into a well-tested H4 support zone within an intact uptrend."}]\n'
    "```"
)


AUDIT_INSTRUCTION = (
    "You are an independent audit reviewer in a multi-stage FTMO "
    "1-Stage-Challenge portfolio-suggestion pipeline. This is a REAL "
    "prop-firm evaluation account with real, rule-based compliance "
    "limits — a daily-loss or trailing max-loss breach is INSTANT "
    "termination with ZERO grace period, not a soft guideline, so weigh "
    "sizing scrutiny accordingly. You do NOT have web search or any live "
    "data access — reason only from the account, market, macro, "
    "contract-feasibility, FTMO compliance-headroom, and research data "
    "given below (already gathered by another system), plus Claude's own "
    "first-pass suggested mix and reasoning, produced from that same "
    "data using live web research."
    "\n\n"
    "The draft ends with an \"External Research Notes\" list — each line "
    "there is a fact the draft-writer found via a real live web search "
    "you cannot independently repeat or verify. Not being able to verify "
    "one of these is NOT the same as it being fabricated: treat it as a "
    "real, reasonably trustworthy input and focus your scrutiny on how "
    "it was WEIGHTED or APPLIED to a sizing/inclusion decision, not on "
    "the mere fact you can't check it yourself. Reserve genuine "
    "skepticism for numeric claims that appear in the draft's prose but "
    "nowhere in this Notes list or the structured data given below.\n\n"
    "Your job is NOT to produce your own competing mix — it is to "
    "critically audit this draft's specific suggestion and reasoning:\n"
    "- Check its math: do the stated percentages sum correctly, does "
    "every included instrument respect its feasibility line, does the "
    "sizing look sound given the D1/H4/H1 volatility/ATR data given.\n"
    "- Check the FTMO compliance math specifically, and treat this as "
    "the single most important thing to get right: does the draft's own "
    "stated aggregate heat leave REAL headroom under BOTH the real "
    "daily-loss headroom % and the real trailing max-loss headroom % "
    "given below (not just avoid crossing zero exactly)? Did it "
    "correctly treat the 10% max-loss floor as TRAILING (based on the "
    "account's highest-ever end-of-day balance, only ever moving up) "
    "rather than static? Did it reason about the Best Day Rule as a "
    "CONSISTENCY constraint (a single oversized winning day can itself "
    "cause a breach even though it made money), not just a downside-"
    "risk one?\n"
    "- Check whether it adequately addressed: correlation under stress, "
    "execution/liquidity risk, overnight/weekend financing cost for "
    "multi-day holds, idle-cash deployment triggers with time-based "
    "fallbacks, and position-sizing-vs-volatility — flag any it skipped "
    "or handled weakly.\n"
    "- Check HOLDING-HORIZON REALISM: this account trades intraday to at "
    "most one trading day, per instrument, debated on its own real cost/"
    "swap terms — flag any target price that implicitly assumes a multi-"
    "day hold to reach, any thesis that leans mainly on the daily (D1) "
    "read rather than the H4/H1 reads that should be doing the actual "
    "entry/stop/target work, and any position held/implied overnight "
    "without an explicit per-instrument justification (real cost vs. "
    "realistic move, and swap SIGN specifically) — an overnight hold "
    "with no stated reason, or one that ignores an unfavorable swap "
    "sign, is a real gap to flag.\n"
    "- Check ASSET-CATEGORY DIVERSIFICATION: does the mix genuinely "
    "spread across the distinct categories actually tradable (forex, "
    "metals, commodities/agriculturals, indices, crypto where offered), "
    "roughly 1-2 instruments per represented category, favoring liquid "
    "names by default? Flag over-concentration in one category, an "
    "unjustified total absence of an available category, or 3+ thinly-"
    "justified instruments crowded into a single category.\n"
    "- Check REAL TRADING COST awareness: was every reported reward:risk "
    "ratio actually netted against the REAL trading cost line given per "
    "instrument (spread + this account's own confirmed commission), not "
    "just the gross price move? Flag any position whose reward:risk "
    "would fall meaningfully short of what's claimed once that real cost "
    "is subtracted, and any multi-day hold that didn't fold in the real "
    "swap %/day times the expected holding days.\n"
    "- Check POSITION-SIZE AND STOP/TARGET DEBATE: did every position's "
    "size actually reflect the five real inputs it should (conviction "
    "strength backed by real evidence, real H1/H4 volatility, the "
    "feasibility ceiling, real cost-to-risk, and its own contribution to "
    "the real compliance headroom) rather than a uniform size regardless "
    "of these differences? Did every stop/target come with an explicit "
    "ATR-multiple and support/resistance-reliability justification "
    "specific to that instrument, rather than a flat multiple or a round "
    "number applied the same way across every position? Flag any "
    "position sized or stopped identically to its peers despite "
    "genuinely different volatility, conviction, or cost profiles.\n"
    "- Check SETUP READ and MULTI-TIMEFRAME use: setup reads are now "
    "computed on all four timeframes (monthly, D1, H4, H1) — did the "
    "draft actually engage with each position's computed setup read "
    "(reversal, pullback, range-fade, breakout-watch, trend-following, "
    "trend-intact, or grind-continuation — trend-intact means a real "
    "'trending_up'/'trending_down' regime with no more specific trigger "
    "active right now, grind-continuation means a real net directional "
    "move over the medium-term window reached via a noisy/choppy path "
    "rather than a clean trend; BOTH are genuine directional evidence, "
    "treat a draft dismissing either as 'no real setup' as UNDER-"
    "weighing real evidence, not a correct caution) rather than ignore it, and does "
    "its stated thesis genuinely match that characterization rather "
    "than contradict it (e.g. calling something a trend-following entry "
    "when the H4/H1 setup read says range_fade_candidate)? The monthly/"
    "D1 setup reads are context (feeding the thesis and the Long-term "
    "alignment read below) — flag a draft that anchors its actual entry/"
    "stop/target reasoning on a monthly or D1 setup read instead of H4/H1, "
    "not a draft that merely mentions the longer-term read as supporting "
    "context. Did it use a real H4/H1 CONFLICTING trend read as a reason "
    "for caution, or silently pick whichever timeframe agreed with its "
    "own thesis? Flag any stop/target that ignored a real multi-"
    "timeframe confluence level sitting right where a same-timeframe "
    "level alone was used instead.\n"
    "- Check LONG-TERM ALIGNMENT use: each position also has a real "
    "'Long-term alignment' read comparing its H4 move against the real "
    "daily/monthly backdrop — STRUCTURALLY BACKED, COUNTER-TREND SPIKE, "
    "MIXED, or no real backdrop available. A COUNTER-TREND SPIKE read is "
    "NOT itself a flaw to raise — the pipeline explicitly treats a short, "
    "fast counter-trend move as a real, legitimate trade when correctly "
    "labeled and managed, not something to avoid; do NOT flag a position "
    "as unsound merely for being counter-trend. DO flag: a draft that "
    "calls something a durable/structural trend when its own Long-term "
    "alignment read says COUNTER-TREND SPIKE (a real mischaracterization, "
    "not a labeling nitpick); a COUNTER-TREND SPIKE position sized or "
    "stopped identically to a STRUCTURALLY BACKED one, rather than with "
    "the tighter stop and smaller size that read calls for; or a draft "
    "that silently ignores this read entirely.\n"
    "- Check FUNDAMENTAL-BASED INCLUSION use: this pipeline explicitly "
    "allows a position to be included on a genuine, specific, sourced "
    "macro/fundamental case (from the macro snapshot, the cross-asset "
    "book-wisdom principles, or the draft's own web research) even when "
    "its technical setup read is 'no_clear_setup' or its market-type "
    "reads 'sideways'/'choppy_up'/'choppy_down' — do NOT flag a position "
    "as lacking justification merely because it lacks a clean technical "
    "setup; check instead whether the fundamental case actually given is "
    "genuinely specific and sourced (real numbers/dates/sources, not a "
    "vague macro platitude) and whether the technical read actually "
    "CONTRADICTS it (a real CONFLICTING multi-timeframe read or a "
    "backtest showing the instrument's own history rejects the proposed "
    "direction) — THAT combination is the real flaw to flag, not the "
    "absence of a technical setup by itself.\n"
    "- Identify anything it got wrong, missed, or reasoned poorly about, "
    "based on the data you both were given.\n"
    "- Note where you would weigh something differently, and why.\n"
    "\n"
    "BACKTEST THE DRAFT'S UNDERLYING LOGIC, not just its arithmetic. The "
    "draft implies a small system of cause-and-effect claims about how "
    "these instruments behave (e.g. 'this overbought reading means a "
    "pullback is likely'). You have real historical evidence below to "
    "test specific values against:\n"
    "1. For each holding, identify the specific technical/behavioral "
    "claim(s) the draft relies on to justify it.\n"
    "2. Cross-check EACH claim against that exact instrument's own "
    "historical backtest evidence given below (its real simulated RSI-"
    "reversal and support/resistance-bounce trade win rates and average "
    "realized R-multiple — an actual ATR-based stop/target walked "
    "forward bar by bar to a genuine win/loss, not a bare average return "
    "N bars later, and — wherever this instrument's own live spread/"
    "commission/swap/broker-minimum-stop-distance data is available — "
    "already netted against those real execution costs and constraints, "
    "not an idealized zero-friction result (the stop/target sizing uses "
    "the real historical ATR at each past entry; the cost/swap figure "
    "netted in is today's live rate applied uniformly across that "
    "history, a disclosed proxy, not a historical record — don't flag "
    "this as a flaw, no historical spread/swap record exists to do "
    "better) — plus whether its own "
    "history shows real momentum "
    "persistence or mean-reversion, and whether its own low-volatility "
    "episodes have historically been followed by bigger or smaller "
    "moves). Note: there is no "
    "beta-vs-benchmark backtest here (this account can span forex, "
    "metals, and equity indices with no single natural benchmark) — "
    "don't fault the draft for lacking one.\n"
    "3. Score each claim: SUPPORTED (the historical evidence agrees with "
    "the draft's implied logic), CONTRADICTED (the instrument's own "
    "history shows the opposite), or UNTESTABLE (not enough real "
    "historical episodes to judge). Cite the actual numbers you're "
    "basing this on.\n"
    "4. A CONTRADICTED score is a real, concrete flaw — treat it with "
    "the same weight as a math or compliance-headroom error.\n"
    "\n"
    "If the draft includes a 'Pending Setups' section and its trailing "
    "JSON array, give those the same real-numbers scrutiny as the "
    "allocation block, PLUS one check specific to this section: for "
    "each entry, actually RECOMPUTE the stop distance (|price - "
    "stop_loss|), the reward distance (|take_profit - price|), and the "
    "gross R:R (reward/risk) directly from that entry's own three JSON "
    "numbers — a real, recurring failure mode caught before is the "
    "prose claiming one R:R (or one ATR%, or one pip count) while the "
    "JSON's own price/stop_loss/take_profit numbers imply a different "
    "one (e.g. a stop 30 pips from entry labeled '20 pips', or an ATR-"
    "based distance computed correctly in prose but a different, "
    "uncorrected price written into the JSON). Flag ANY mismatch "
    "between the prose's claimed numbers and what the JSON's own three "
    "fields actually compute as a required-fix arithmetic error, not a "
    "stylistic note — same weight as a compliance-headroom error. Also "
    "check that each trigger_condition is specific and mechanically "
    "checkable (a real price level plus an indicator threshold and "
    "timeframe), not vague language an automated hourly check couldn't "
    "evaluate.\n"
    "\n"
    "Produce a structured audit report — agreements, flaws, gaps, the "
    "backtest scorecard from above, and specific suggested improvements "
    "— not a rewritten competing allocation. Keep your response focused "
    "— under 550 words."
)


def build_ftmo_stage1_instruction() -> str:
    return f"{_INSTRUCTION_HEAD}\n\n{_STAGE1_DRAFT_INSTRUCTION}\n\n{_INSTRUCTION_TAIL}"


def build_ftmo_stage2_instruction(audit_available: bool) -> str:
    role = _ROLE_STAGE2_SYNTHESIZE if audit_available else _ROLE_STAGE2_SELF_REVIEW
    return f"{_INSTRUCTION_HEAD}\n\n{role}\n\n{_INSTRUCTION_TAIL}"


@dataclass
class FtmoAssetAnalysis:
    """Wraps a PMEX-shape AssetAnalysis (`base`) with the two extra
    intraday reads that are genuinely unique to FTMO's own tighter,
    faster-paced compliance timeline: real H4/H1 technical stats fetched
    directly from this account's own live MT5 price feed via
    data.mt5_source.fetch_mt5_price_history — a real broker-native fetch
    that works for every FTMO symbol. Unlike PMEX/PSX, `base` is built
    entirely natively too (see _build_bare_base_analysis/_enrich_with_
    native_d1) rather than via ai.portfolio_suggest.analyze_assets()'s
    Yahoo-based enrichment — FTMO deliberately doesn't depend on Yahoo
    for anything, not even for the one symbol (XAUUSD) that technically
    has a Yahoo mapping; see _build_bare_base_analysis's own docstring
    for the real incident that motivated this.

    h4_structure/h1_structure/d1_structure/mn1_structure carry the real
    Fibonacci/support-resistance/trendline/chart-pattern reads for all
    FOUR timeframes — real feature added live 2026-08-22, user's own
    direct request after the monthly-timeframe addition above only gave
    D1/monthly a bare TechnicalStats read: "i want you to redesign the
    charts and its decision with all H1, H4, D1 and monthly... same with
    the sub section setup read" — chart structure and setup
    classification (see classify_setups) previously stopped at H4/H1
    only, on the theory that a monthly/daily S/R level or pattern isn't
    actionable for an account holding intraday-to-1-day. That theory
    held for what the AI uses to place an actual entry/stop/target (H4/H1
    still are, and remain, the PRIMARY basis per HOLDING HORIZON below)
    but not for what a human — or the model's own broader thesis-building
    — benefits from SEEING: the same real net-direction-vs-noise question
    market_regime already answers per timeframe has a structural/pattern
    analogue too, and withholding it wasn't a considered tradeoff, just
    an earlier, narrower reading of "not actionable." Computed via the
    exact same compute_chart_structure() call as H4/H1, which itself
    already caps its own swing-point scan to the last STRUCTURE_LOOKBACK
    (90) bars regardless of input size — so a D1 read is a real ~3-month
    structural window and a monthly read is a real ~7.5-year one (or
    whatever's actually available for a newer listing), not an
    unbounded, ever-growing scan.

    `mn1_stats`/`d1_stats`-via-`base.stats` (real bug found live: this
    account's own analysis never read a daily-or-longer timeframe
    visibly, only H4/H1 — user's own words: "it can tend to throw us in
    a fake trend unguarded" — a genuinely correct concern, since a real
    short-term move with no longer-term backing IS meaningfully
    different from one that has it) add real monthly (MN1) TechnicalStats
    alongside D1's own (already on `base.stats`) as long-term structural
    CONTEXT — H4/H1 remain the PRIMARY basis for entry/stop/target per
    HOLDING HORIZON below regardless of how much structure D1/monthly now
    also carry. Same "always a real TechnicalStats/ChartStructureSnapshot,
    mostly-empty fields rather than a bare None" treatment as h4_stats/
    h1_stats when a symbol genuinely lacks enough MT5 history (a newer
    listing) — degrades the same way every other "not enough real
    history" case in this codebase already does, not a crash. mn1_stats/
    d1_structure/mn1_structure are defaulted (unlike h4_stats/h1_stats/
    h4_structure/h1_structure) purely so existing test fixtures built
    before these fields existed don't need updating just to keep
    constructing this dataclass."""

    base: AssetAnalysis
    h4_stats: TechnicalStats
    h1_stats: TechnicalStats
    h4_structure: ChartStructureSnapshot
    h1_structure: ChartStructureSnapshot
    trade_cost: TradeCost | None
    mn1_stats: TechnicalStats = field(default_factory=lambda: TechnicalStats(*([None] * 16)))
    d1_structure: ChartStructureSnapshot = field(
        default_factory=lambda: ChartStructureSnapshot(fibonacci=None, sr_levels=None, trendlines=None, patterns=[])
    )
    mn1_structure: ChartStructureSnapshot = field(
        default_factory=lambda: ChartStructureSnapshot(fibonacci=None, sr_levels=None, trendlines=None, patterns=[])
    )


# ~6 years of daily bars — enough real history for the backtest functions'
# own episode-count thresholds to have real statistical power, without an
# excessive per-symbol fetch. Confirmed live against the real FTMO
# terminal: EURUSD/GBPUSD/MSFT go back to 2018, NVDA to 2020, BTCUSD to
# 2021 — MT5's own native feed comfortably covers this depth for the
# instruments this account actually trades.
_D1_BACKTEST_BARS = 1500

# 10 years of monthly bars — comfortably above RANGE_WINDOW=60 (the
# minimum needed for market_regime/support-resistance to populate at
# all on the monthly timeframe), with real margin for symbols with a
# shorter broker-side history (crypto, newer CFDs) to still get whatever
# depth genuinely exists rather than being truncated short of it.
_MN1_BACKTEST_BARS = 120


def _build_bare_base_analysis(a: MarketAsset) -> AssetAnalysis:
    """FTMO's own base-analysis builder — deliberately does NOT call
    ai.portfolio_suggest.analyze_assets()/resolve_yahoo_ticker() the way
    PMEX/PSX do, not even for a symbol (XAUUSD, via its GC=F mapping)
    that technically resolves. Real incident, not a hypothetical: the
    unattended daily mega-analysis job hung for 24+ hours straight,
    entirely because XAUUSD was the ONE FTMO symbol that ever touched
    Yahoo at all, and a fresh cold-started process negotiating Yahoo's
    session from scratch (unlike this project's own long-running
    Streamlit server, which only ever does that negotiation once and
    reuses it for the rest of its life) hit a real hang with no timeout
    protecting it at the time. A hard timeout now guards every Yahoo
    call project-wide regardless (see utils.run_with_timeout), but for
    FTMO specifically there's a strictly better fix available: it simply
    doesn't need Yahoo for ANYTHING price/technical, since MT5 already
    holds real multi-year native history for every FTMO instrument
    (confirmed live, see _enrich_with_native_d1's own docstring) — so
    the whole exchange's analysis pipeline can be made to not depend on
    an external, occasionally-flaky network service at all, not just
    tolerate it timing out gracefully. The one real cost: XAUUSD loses
    its own couple of real Yahoo headlines, since MT5 has no news feed
    of its own — but every OTHER FTMO symbol already has that same gap
    (headlines=[] is `_enrich_with_native_d1`'s own long-standing,
    disclosed contract), so this just makes XAUUSD consistent with the
    rest of the pool instead of a one-off exception."""
    return AssetAnalysis(
        symbol=a.symbol,
        description=a.description,
        bid=a.bid,
        ask=a.ask,
        display_name=None,
        contract_spec=get_contract_spec(a.symbol),
    )


def _real_backtest_execution_kwargs(
    trade_cost: TradeCost | None, contract_spec: ContractSpec | None, current_ask: float | None
) -> dict:
    """Combines this instrument's own REAL spread, FTMO commission, swap,
    and broker-enforced minimum stop distance into the kwargs
    analysis/backtest.py's trade-simulation engine uses to net real
    execution cost/constraints into its win-rate/R-multiple results —
    direct user request 2026-08-22: "review the allowed lot size, trade
    cost, and other execution related features and broker's allowed
    guard rails and then apply your backtest trade... such results are
    more realistic and dependable." FTMO-specific for now (the one
    exchange with both a live MT5 feed for real spread/swap/stop-level
    data AND a published commission schedule to layer on top — PMEX
    shares the MT5 feed but has no commission plumbing of its own yet;
    PSX has no live broker connection at all) — returns an empty dict
    (the engine's own existing zero-cost/zero-restriction default,
    unchanged from before this feature existed) for any missing real
    input rather than guessing or partially applying one.

    Commission is added into round-trip cost as a %, converted from its
    own $/lot or %-based schedule via `contract_spec`/`current_ask` (see
    `_ftmo_commission_pct_round_turn`) — silently left out (not defaulted
    to 0) when either is unavailable, since 0% commission would itself
    be a fabricated number, not a genuine "confirmed zero" the way
    Agriculture's real zero-commission rate is.

    `current_ask` deliberately means the CURRENT live ask, not this
    instrument's D1 close — must match format_ftmo_trade_cost's own
    price source exactly (real inconsistency caught on a self-recheck:
    an earlier version used the D1 series' last close here while
    format_ftmo_trade_cost used the live ask, so the two could silently
    disagree on Forex/Exotics' price-dependent commission % purely from
    using different prices, not from any real difference in cost).

    `trade_cost`/`contract_spec` are always TODAY's live figures (this
    project has no historical spread/swap time series to draw from —
    see analysis/backtest.py's own top-of-file/_simulate_trades note),
    applied uniformly to every historical trade the simulation replays —
    a reasonable, disclosed proxy for "what this instrument's own
    round-trip friction has generally looked like," not a claim that
    the spread/commission/swap were literally identical years ago."""
    if trade_cost is None:
        return {}
    round_trip_cost_pct = trade_cost.spread_pct_of_price
    if contract_spec is not None and current_ask:
        commission_pct, _detail = _ftmo_commission_pct_round_turn(
            trade_cost.category, contract_spec.trade_contract_size, current_ask
        )
        if commission_pct is not None:
            round_trip_cost_pct += commission_pct
    return dict(
        round_trip_cost_pct=round_trip_cost_pct,
        long_swap_pct_per_day=trade_cost.swap_long_pct_per_day or 0.0,
        short_swap_pct_per_day=trade_cost.swap_short_pct_per_day or 0.0,
        min_stop_distance_pct=trade_cost.min_stop_distance_pct,
    )


def _enrich_with_native_d1(
    base: AssetAnalysis, d1_history: pd.DataFrame | None = None, trade_cost: TradeCost | None = None
) -> AssetAnalysis:
    """Backfills real D1 technical stats + backtests from MT5's own
    native price feed for a bare (display_name is None) base analysis —
    every FTMO symbol, always, since _build_bare_base_analysis above
    never attempts Yahoo resolution in the first place (this project's
    own history: FTMO used to only backfill whatever Yahoo left bare,
    confirmed live at the time to be MOST of a real FTMO Market Watch —
    16 of 17 real symbols on one account — since data/underlying.py's
    keyword map targets PMEX's commodity-futures naming, not forex/CFD/
    crypto symbols; the one Yahoo-covered exception, XAUUSD, has since
    been made to skip Yahoo too, for reasons that function's own
    docstring covers). Unlike PMEX (whose dated, expiring futures
    contracts genuinely lack a long-running single MT5 history, which is
    why IT still needs Yahoo's continuous underlying series), FTMO
    trades non-expiring CFD/forex instruments that MT5 itself already
    holds real multi-year daily history for — a strictly better,
    broker-native source for this account specifically, needing no
    keyword-map maintenance at all.

    The `display_name is not None` early return is now a defensive no-op
    for FTMO's own callers (always None going in) rather than the
    load-bearing branch it used to be — kept rather than removed, since
    it costs nothing and keeps this function honest/safe if anything
    ever calls it with an already-enriched base again. `headlines` stays
    empty for a native-D1 entry (no MT5-native news feed) — an honest,
    disclosed gap, same treatment as any other missing-data field in
    this pipeline, not a reason to withhold the real technical/backtest
    data that IS available.

    `d1_history`, if the caller already fetched it (analyze_ftmo_assets
    now also needs the same D1 bars for chart-structure — see
    FtmoAssetAnalysis's own docstring for why), skips this function's
    own redundant fetch of the identical (symbol, "D1", count) call —
    real, if cheap, waste avoided the same way this project already
    avoids it elsewhere (compute_chart_structure's own shared-swing-scan
    fix). None (the default) preserves the exact original fetch-it-
    yourself behavior for every existing caller/test.

    `trade_cost`, if the caller already fetched it (analyze_ftmo_assets/
    analyze_ftmo_asset_live both need it anyway for their own "REAL
    trading cost" line), is what lets the RSI/S-R backtests below
    simulate with real spread/commission/swap/broker-stop-distance
    instead of the engine's own zero-friction default — see
    _real_backtest_execution_kwargs. None skips this entirely (same
    zero-cost default as before this feature existed), not an error."""
    if base.display_name is not None:
        return base

    if d1_history is None:
        d1_history = fetch_mt5_price_history(base.symbol, "D1", count=_D1_BACKTEST_BARS)
    if d1_history.empty:
        return base

    prices = d1_history["Close"]
    # base.ask, not the D1 series' last close — matches EXACTLY what
    # format_ftmo_trade_cost uses for its own commission conversion (see
    # that function's own `price = analysis.base.ask`), so the "REAL
    # trading cost" line and this backtest's own netted cost can never
    # silently disagree on Forex/Exotics' price-dependent commission %
    # just because they happened to compute it from two different prices
    # (a real, if tiny, inconsistency caught on a self-recheck).
    exec_kwargs = _real_backtest_execution_kwargs(trade_cost, base.contract_spec, base.ask)
    rsi_overbought_bt, rsi_oversold_bt = backtest_rsi_reaction(d1_history, **exec_kwargs)
    return AssetAnalysis(
        symbol=base.symbol,
        description=base.description,
        bid=base.bid,
        ask=base.ask,
        display_name=base.description,
        data_source="mt5",
        prices=prices,
        stats=compute_technical_stats(prices, history=d1_history),
        headlines=[],
        contract_spec=base.contract_spec,
        rsi_overbought_backtest=rsi_overbought_bt,
        rsi_oversold_backtest=rsi_oversold_bt,
        momentum_persistence_backtest=backtest_momentum_persistence(prices),
        volatility_regime_backtest=backtest_volatility_regime(prices),
        support_resistance_backtest=backtest_support_resistance_reaction(d1_history, **exec_kwargs),
    )


def analyze_ftmo_assets(
    assets: list[MarketAsset], on_progress: Callable[[str], None] | None = None
) -> list[FtmoAssetAnalysis]:
    """Builds each symbol's own bare base analysis (contract spec only —
    see _build_bare_base_analysis's own docstring for why this
    deliberately never touches Yahoo, not even for XAUUSD), backfills
    real D1 technical/backtest coverage natively from MT5 (see
    _enrich_with_native_d1), then extends it with real H4/H1 reads per
    symbol — see FtmoAssetAnalysis's own docstring for why this
    wrap-rather-than-duplicate shape was chosen. One single progress
    pass now (previously two — a base Yahoo-resolution pass, then this
    function's own extension pass — before FTMO stopped needing Yahoo
    at all)."""
    results = []
    total = len(assets)
    for i, a in enumerate(assets, start=1):
        if on_progress is not None:
            on_progress(f"Analyzing {i}/{total} — {a.symbol} (D1/H4/H1/monthly + chart structure + trading cost)")
        bare_base = _build_bare_base_analysis(a)
        # Both fetched once, up front, and threaded into
        # _enrich_with_native_d1 below (rather than letting it do its own
        # identical fetches) so the SAME D1 bars also feed d1_structure's
        # chart-structure scan, and the SAME trade_cost feeds the RSI/S-R
        # backtests' real cost/guard-rail simulation AND the analysis'
        # own "REAL trading cost" line below — without a second,
        # redundant MT5 round-trip for either.
        d1_history = fetch_mt5_price_history(bare_base.symbol, "D1", count=_D1_BACKTEST_BARS)
        trade_cost = get_trade_economics(bare_base.symbol)
        base = _enrich_with_native_d1(bare_base, d1_history, trade_cost)
        h4_history = fetch_mt5_price_history(base.symbol, "H4")
        h1_history = fetch_mt5_price_history(base.symbol, "H1")
        mn1_history = fetch_mt5_price_history(base.symbol, "MN1", count=_MN1_BACKTEST_BARS)
        results.append(
            FtmoAssetAnalysis(
                base=base,
                h4_stats=compute_technical_stats(
                    h4_history["Close"], history=h4_history, periods_per_year=_H4_PERIODS_PER_YEAR
                ),
                h1_stats=compute_technical_stats(
                    h1_history["Close"], history=h1_history, periods_per_year=_H1_PERIODS_PER_YEAR
                ),
                h4_structure=compute_chart_structure(h4_history),
                h1_structure=compute_chart_structure(h1_history),
                trade_cost=trade_cost,
                mn1_stats=compute_technical_stats(
                    mn1_history["Close"], history=mn1_history, periods_per_year=_MN1_PERIODS_PER_YEAR
                ),
                d1_structure=compute_chart_structure(d1_history),
                mn1_structure=compute_chart_structure(mn1_history),
            )
        )
    return results


def analyze_ftmo_asset_live(symbol: str, bid: float, ask: float, description: str) -> FtmoAssetAnalysis:
    """The same monthly/D1/H4/H1 technical stats + chart structure +
    backtests + trade cost analyze_ftmo_assets() computes in its batch
    pipeline, but for exactly one symbol, on demand — built for the Watchlist's
    per-asset detail popup (app.py), which needs a fast, genuinely live
    single-symbol read rather than the full batch pipeline's per-symbol
    Yahoo/news lookups. Deliberately skips analyze_assets()'s Yahoo-based
    enrichment entirely and always builds the D1 base natively from MT5
    (same construction as _enrich_with_native_d1, just unconditional
    rather than gated on "Yahoo didn't already cover this symbol") —
    MT5's own feed is both faster and more genuinely "live" than Yahoo's
    daily bars anyway, which is exactly what a real-time dashboard needs.

    Backtests are skipped (left None) only if D1 history itself came
    back empty — matches _enrich_with_native_d1's own guard, since none
    of the four backtest functions have been verified safe to call on a
    truly empty price series."""
    d1_history = fetch_mt5_price_history(symbol, "D1", count=_D1_BACKTEST_BARS)
    d1_prices = d1_history["Close"]
    # Both fetched once, up front, and reused below for the RSI/S-R
    # backtests' real cost/guard-rail simulation AND the analysis' own
    # base/trade_cost fields — same redundant-fetch avoidance as
    # analyze_ftmo_assets' own identical pattern.
    contract_spec = get_contract_spec(symbol)
    trade_cost = get_trade_economics(symbol)
    if d1_history.empty:
        rsi_overbought_bt = rsi_oversold_bt = None
        momentum_bt = volatility_regime_bt = support_resistance_bt = None
        d1_stats = compute_technical_stats(d1_prices)
    else:
        # `ask` (the function's own parameter), not the D1 series' last
        # close — see _enrich_with_native_d1's own identical fix/comment
        # for why: matches format_ftmo_trade_cost's own price source
        # exactly, so the two can never disagree on Forex/Exotics'
        # price-dependent commission % just from using different prices.
        exec_kwargs = _real_backtest_execution_kwargs(trade_cost, contract_spec, ask)
        rsi_overbought_bt, rsi_oversold_bt = backtest_rsi_reaction(d1_history, **exec_kwargs)
        momentum_bt = backtest_momentum_persistence(d1_prices)
        volatility_regime_bt = backtest_volatility_regime(d1_prices)
        support_resistance_bt = backtest_support_resistance_reaction(d1_history, **exec_kwargs)
        d1_stats = compute_technical_stats(d1_prices, history=d1_history)

    base = AssetAnalysis(
        symbol=symbol,
        description=description,
        bid=bid,
        ask=ask,
        display_name=description,
        data_source="mt5",
        prices=d1_prices,
        stats=d1_stats,
        headlines=[],
        contract_spec=contract_spec,
        rsi_overbought_backtest=rsi_overbought_bt,
        rsi_oversold_backtest=rsi_oversold_bt,
        momentum_persistence_backtest=momentum_bt,
        volatility_regime_backtest=volatility_regime_bt,
        support_resistance_backtest=support_resistance_bt,
    )
    h4_history = fetch_mt5_price_history(symbol, "H4")
    h1_history = fetch_mt5_price_history(symbol, "H1")
    mn1_history = fetch_mt5_price_history(symbol, "MN1", count=_MN1_BACKTEST_BARS)
    return FtmoAssetAnalysis(
        base=base,
        h4_stats=compute_technical_stats(
            h4_history["Close"], history=h4_history, periods_per_year=_H4_PERIODS_PER_YEAR
        ),
        h1_stats=compute_technical_stats(
            h1_history["Close"], history=h1_history, periods_per_year=_H1_PERIODS_PER_YEAR
        ),
        h4_structure=compute_chart_structure(h4_history),
        h1_structure=compute_chart_structure(h1_history),
        trade_cost=trade_cost,
        mn1_stats=compute_technical_stats(
            mn1_history["Close"], history=mn1_history, periods_per_year=_MN1_PERIODS_PER_YEAR
        ),
        d1_structure=compute_chart_structure(d1_history),
        mn1_structure=compute_chart_structure(mn1_history),
    )


# Real, round-turn commission by MT5's own symbol-path category — see
# config.py's own commentary for exactly how each rate was sourced/
# confirmed. A category not covered here returns (None, ...) rather than
# silently defaulting to 0 — an unconfirmed gap must read as a gap to the
# model, not as "this instrument is free to trade."
def _ftmo_commission_pct_round_turn(
    category: str, contract_size: float, price: float
) -> tuple[float | None, str]:
    if category in ("Forex", "Exotics"):
        if contract_size <= 0 or price <= 0:
            return None, "commission unknown (no contract size/price to convert the $/lot rate)"
        pct = config.FTMO_COMMISSION_FX_USD_PER_LOT_ROUND_TURN / (contract_size * price) * 100
        return pct, f"${config.FTMO_COMMISSION_FX_USD_PER_LOT_ROUND_TURN:.2f}/lot round-turn, FTMO-confirmed"
    if category in ("Metals CFD", "Commodities", "Cash III CFD"):
        pct = config.FTMO_COMMISSION_METALS_COMMODITIES_PCT_ROUND_TURN
        return pct, f"{pct:.4f}% round-turn, FTMO-confirmed"
    if category.startswith("Crypto"):
        pct = config.FTMO_COMMISSION_CRYPTO_PCT_ROUND_TURN
        return pct, f"{pct:.4f}% round-turn, user-reported (NOT independently confirmed)"
    if category == "Agriculture":
        return 0.0, "zero commission, confirmed"
    if category in ("Cash CFD", "Cash II CFD"):
        return 0.0, "zero commission on indices, FTMO-confirmed policy"
    return None, f"commission unknown for category {category!r} (not in this account's confirmed fee schedule)"


def format_ftmo_trade_cost(analysis: FtmoAssetAnalysis) -> str:
    """The REAL round-trip cost of trading this instrument, combining
    MT5-native spread (get_trade_economics) with FTMO's own confirmed
    commission schedule — the single number the reward:risk instruction
    below requires the model to net against its target before reporting
    a ratio, since neither piece alone tells you whether an edge
    actually survives real costs."""
    cost = analysis.trade_cost
    if cost is None:
        return "  REAL trading cost: not available (no live spread/swap data for this symbol)"

    spec = analysis.base.contract_spec
    contract_size = spec.trade_contract_size if spec is not None else 0.0
    price = analysis.base.ask
    commission_pct, commission_note = _ftmo_commission_pct_round_turn(cost.category, contract_size, price)

    if commission_pct is None:
        round_trip_pct = cost.spread_pct_of_price
        commission_display = f"unknown ({commission_note})"
        round_trip_note = " (excludes unknown commission — treat as a floor, not the full cost)"
    else:
        round_trip_pct = cost.spread_pct_of_price + commission_pct
        commission_display = f"{commission_pct:.4f}% ({commission_note})"
        round_trip_note = ""

    # Both directions reported independently — a short recommendation
    # needs its own real overnight cost to reason about, not just the
    # long side (which is all this used to surface).
    swap_parts = []
    if cost.swap_long_pct_per_day is not None:
        swap_parts.append(f"{cost.swap_long_pct_per_day:+.4f}%/day held long")
    if cost.swap_short_pct_per_day is not None:
        swap_parts.append(f"{cost.swap_short_pct_per_day:+.4f}%/day held short")
    swap_display = ", ".join(swap_parts) if swap_parts else "not available for this swap calculation mode"

    return (
        f"  REAL trading cost ({cost.category}): spread {cost.spread_pct_of_price:.4f}% + "
        f"commission {commission_display} = {round_trip_pct:.4f}% round-trip{round_trip_note} "
        f"(paid once, entry+exit combined, before any move in price); overnight swap "
        f"{swap_display} — multiply by planned holding days for a multi-day hold"
    )


def format_timeframe_stats(symbol: str, label: str, stats: TechnicalStats, *, long_term: bool = False) -> str:
    """Generic over any (symbol, label, TechnicalStats) — was named
    format_intraday_stats and hardcoded "(near-term entry timing/stop
    placement)" for every label until the real monthly (MN1) timeframe
    was added and reused this function verbatim: that phrasing is
    actively wrong for a monthly read (it's long-term structural CONTEXT,
    never a near-term entry/stop reference — see FtmoAssetAnalysis's own
    docstring on mn1_stats), so `long_term` now switches both the header
    phrase and the ATR line's own "use for near-term stop distance"
    suggestion, which was equally wrong for a monthly ATR (enormous
    relative to an actual intraday stop, not a real reference for one)."""
    purpose = "long-term structural/bias context, not for entry timing" if long_term else "near-term entry timing/stop placement"
    if stats.last_price is None:
        return f"  {label} technical ({purpose}): not available (insufficient MT5 {label} history for {symbol})"
    bits = [f"last {stats.last_price:.4f}"]
    if stats.trend is not None and stats.pct_vs_sma20 is not None:
        bits.append(f"{stats.trend} ({stats.pct_vs_sma20:+.1f}% vs 20-bar SMA)")
    if stats.market_regime is not None:
        # Real gap found on a self-audit re-check: this line was missing
        # entirely, so an H4/H1 market_regime of "trending_up"/"trending_
        # down"/"choppy_up"/"choppy_down" only ever reached the model
        # indirectly, through the setup-classifier's own text (rule 3 or
        # 6 below) — and ONLY when one of those specific rules happened
        # to fire. A clean trending_up/trending_down regime with no
        # matching chart-structure pattern (a real, common case — the two
        # detectors measure genuinely different things, see analysis/
        # technical.py's own market_regime comment) silently never
        # reached the model at all. Shown here directly, unconditionally,
        # the same way the D1 read's "market type" line already works via
        # ai/portfolio_suggest.py::format_enriched_asset_context.
        bits.append(f"market type: {stats.market_regime}")
    if stats.rsi is not None:
        bits.append(f"RSI {stats.rsi:.0f}")
    if stats.atr_pct is not None:
        atr_note = "reference only, NOT a near-term stop distance" if long_term else "use for near-term stop distance"
        bits.append(f"ATR {stats.atr_pct:.2f}% of price ({atr_note})")
    if stats.volatility_annualized_pct is not None:
        bits.append(f"volatility {stats.volatility_annualized_pct:.1f}% (annualized)")
    return f"  {label} technical ({purpose}): {', '.join(bits)}"


def format_chart_structure(label: str, snapshot: ChartStructureSnapshot) -> str:
    """Real, computed chart structure — Fibonacci retracement, multi-
    level support/resistance with real touch counts, trendlines, and
    conservatively-detected chart patterns (see analysis/chart_structure.py's
    own module docstring for exactly which patterns are attempted and
    why others deliberately aren't) — pure Python, zero extra tokens
    spent computing it, so the model reasons over real numbers instead
    of eyeballing a description of a chart it can't actually see."""
    parts = []

    fib = snapshot.fibonacci
    if fib is not None:
        direction = "retracing DOWN from a fresh high" if fib.high_is_more_recent else "retracing UP from a fresh low"
        parts.append(
            f"Fibonacci ({direction}, swing {fib.swing_low:.4f}-{fib.swing_high:.4f}): price "
            f"nearest {fib.nearest_level_name} ({fib.nearest_level_price:.4f}, "
            f"{fib.distance_to_nearest_pct:+.2f}% away)"
        )

    sr = snapshot.sr_levels
    if sr is not None and (sr.resistance_levels or sr.support_levels):
        res = "; ".join(f"{lvl.price:.4f} ({lvl.touches}x, {lvl.distance_pct:+.2f}%)" for lvl in sr.resistance_levels)
        sup = "; ".join(f"{lvl.price:.4f} ({lvl.touches}x, {lvl.distance_pct:+.2f}%)" for lvl in sr.support_levels)
        parts.append(
            f"S/R levels (price, real touch count, distance) — resistance: {res or 'none'}; "
            f"support: {sup or 'none'}"
        )

    tl = snapshot.trendlines
    if tl is not None and (tl.resistance_trendline or tl.support_trendline):
        bits = []
        if tl.resistance_trendline:
            r = tl.resistance_trendline
            bits.append(f"resistance trendline {r.direction} (price {r.price_vs_line} it, {r.distance_pct:+.2f}%)")
        if tl.support_trendline:
            s = tl.support_trendline
            bits.append(f"support trendline {s.direction} (price {s.price_vs_line} it, {s.distance_pct:+.2f}%)")
        parts.append("trendlines: " + "; ".join(bits))

    if snapshot.patterns:
        parts.append(
            "chart patterns: " + "; ".join(f"{p.name} ({p.detail})" for p in snapshot.patterns)
        )

    if not parts:
        return f"  {label} chart structure: not enough confirmed swing points yet"
    return f"  {label} chart structure: " + " | ".join(parts)


def _candidate_levels(structure: ChartStructureSnapshot) -> list[float]:
    """Every real price level a timeframe's own chart structure offers —
    S/R (both sides) plus every Fibonacci level — combined into one list
    for find_mtf_confluence to cross-check against another timeframe's
    own list. A level doesn't need to be labeled the same way (a
    Fibonacci level on H4 lining up with a plain S/R level on H1 is
    still real agreement) — the label is just how it was found, not
    part of what makes it worth trusting."""
    levels: list[float] = []
    if structure.sr_levels is not None:
        levels += [lvl.price for lvl in structure.sr_levels.resistance_levels]
        levels += [lvl.price for lvl in structure.sr_levels.support_levels]
    if structure.fibonacci is not None:
        levels += list(structure.fibonacci.levels.values())
    return levels


def format_mtf_confluence(analysis: FtmoAssetAnalysis) -> str:
    """Cross-timeframe read: does H4 agree with H1 on trend direction,
    and do any of their independently-computed levels line up at the
    same real price? Two timeframes agreeing on both counts is
    genuinely stronger evidence than either alone — this line exists so
    the model doesn't have to manually cross-reference the two H4/H1
    blocks above itself every time."""
    h4_trend = analysis.h4_stats.trend
    h1_trend = analysis.h1_stats.trend
    if h4_trend is None or h1_trend is None:
        alignment = "not available (insufficient history on at least one timeframe)"
    elif h4_trend == h1_trend == "flat":
        # Distinct from a real ALIGNED read — "we agree there's no clear
        # trend" is weaker evidence than "we agree on a real direction,"
        # and shouldn't read the same way to the model.
        alignment = "NEITHER TIMEFRAME SHOWS A CLEAR TREND — both H4 and H1 read flat"
    elif h4_trend == h1_trend:
        alignment = f"ALIGNED — both H4 and H1 read {h4_trend}"
    elif "flat" in (h4_trend, h1_trend):
        # One timeframe trending, the other flat is normal consolidation
        # on the shorter/longer read — NOT the same thing as one reading
        # uptrend while the other reads downtrend, so it shouldn't be
        # lumped into "CONFLICTING" and treated with the same caution.
        trending_tf, trending_dir = ("H4", h4_trend) if h4_trend != "flat" else ("H1", h1_trend)
        flat_tf = "H1" if trending_tf == "H4" else "H4"
        alignment = (
            f"PARTIAL — {trending_tf} reads {trending_dir}, {flat_tf} reads flat "
            "(not a genuine conflict, just unconfirmed on that timeframe)"
        )
    else:
        alignment = f"CONFLICTING — H4 reads {h4_trend}, H1 reads {h1_trend}"

    current_price = analysis.base.ask
    zones = find_mtf_confluence(
        higher_tf_levels=_candidate_levels(analysis.h4_structure),
        lower_tf_levels=_candidate_levels(analysis.h1_structure),
        current_price=current_price,
    )
    if zones:
        # find_mtf_confluence itself sorts by raw price (for its own
        # de-duplication pass), not relevance — truncating THAT order
        # would show whichever zones happen to have the lowest price,
        # not the ones nearest current price, when there are more than
        # 3. Re-sort by distance from current price before truncating.
        nearest_first = sorted(zones, key=lambda z: abs(z.distance_pct))
        zones_text = "; ".join(
            f"{z.avg_price:.4f} ({z.kind}, {z.distance_pct:+.2f}%)" for z in nearest_first[:3]
        )
    else:
        zones_text = "none found"

    return f"  Multi-timeframe (H4 vs H1) trend alignment: {alignment}; confluence levels: {zones_text}"


def _regime_direction(regime: str | None) -> str | None:
    """Collapses market_regime's 5 real states down to the 3 that matter
    for a direction-vs-direction comparison — "up"/"down" fold trending_*
    and choppy_* together (both are real net direction, see analysis/
    technical.py's own market_regime comment for why efficiency/
    cleanliness is a separate question from direction), "flat" covers
    sideways, None means not enough history to say anything."""
    if regime is None:
        return None
    if regime in ("trending_up", "choppy_up"):
        return "up"
    if regime in ("trending_down", "choppy_down"):
        return "down"
    return "flat"


def classify_long_term_alignment(
    analysis: FtmoAssetAnalysis,
) -> tuple[str, str | None, list[str], list[str]]:
    """Shared classification step behind BOTH format_long_term_alignment
    (the long, reasoning-heavy version fed to the AI prompt) and
    format_long_term_alignment_short (a one-line version for the UI
    popup) — real bug found live: the popup was originally showing the
    AI-facing text verbatim, a multi-sentence paragraph that's correct
    reasoning depth for a model but real noise for a human glancing at a
    dashboard; splitting the classification out once means both texts
    can never silently disagree with each other the way maintaining two
    independent copies of this same logic eventually would.

    Returns (state, short_direction, agreeing_backdrop_names,
    opposing_backdrop_names) where state is one of "not_available",
    "flat", "no_backdrop", "structurally_backed", "counter_trend_spike",
    "mixed"."""
    short = _regime_direction(analysis.h4_stats.market_regime)
    d1_dir = _regime_direction(analysis.base.stats.market_regime)
    mn1_dir = _regime_direction(analysis.mn1_stats.market_regime)

    if short is None:
        return "not_available", short, [], []
    if short == "flat":
        return "flat", short, [], []

    backdrop = [(name, d) for name, d in (("daily", d1_dir), ("monthly", mn1_dir)) if d not in (None, "flat")]
    if not backdrop:
        return "no_backdrop", short, [], []

    agreeing = [name for name, d in backdrop if d == short]
    opposing = [name for name, d in backdrop if d != short]
    if not opposing:
        return "structurally_backed", short, agreeing, opposing
    if not agreeing:
        return "counter_trend_spike", short, agreeing, opposing
    return "mixed", short, agreeing, opposing


def format_long_term_alignment(analysis: FtmoAssetAnalysis) -> str:
    """Real bug found live 2026-08-22 (user's own words: this account's
    analysis "reads, displays and analyzes H1, H4 but not the daily or
    monthly charts... this way the analysis is missing the long/medium
    term trends and can throw us in a fake trend unguarded") — H4/H1 were
    the only timeframes ever compared against EACH OTHER
    (format_mtf_confluence above), never against the genuinely longer-
    term backdrop this account's own D1 and monthly (MN1) reads already
    compute. This answers a real, different question than that H4-vs-H1
    comparison: does the CURRENT short-term (H4) move have real backing
    from the longer-term structure, or is it running against it —
    exactly the "fake trend" risk flagged above.

    Deliberately does NOT treat a counter-trend read as disqualifying —
    the user's own stated preference is the opposite: short-term spikes
    with no longer-term backing are explicitly worth trading for the
    quick move they offer, PROVIDED they're correctly recognized as that
    and exited on their own terms rather than mistaken for a durable
    trend. This function's whole job is making that recognition explicit
    and automatic rather than left to be inferred, not gatekeeping which
    trades are allowed."""
    state, short, agreeing, opposing = classify_long_term_alignment(analysis)

    if state == "not_available":
        return "  Long-term alignment: H4 market-type not available yet (insufficient history)."
    if state == "flat":
        return (
            "  Long-term alignment: H4 shows no real short-term direction right now — "
            "nothing yet to compare against the daily/monthly backdrop."
        )
    if state == "no_backdrop":
        return (
            "  Long-term alignment: no real daily or monthly directional backdrop available "
            "(insufficient history, or both read sideways) — this H4 move has no longer-term "
            "structure to either confirm or contradict it."
        )
    if state == "structurally_backed":
        named = " and ".join(agreeing)
        return (
            f"  Long-term alignment: STRUCTURALLY BACKED — the H4 {short}ward move agrees "
            f"with the real {named} backdrop, not just a short-term read in isolation — "
            "genuinely stronger conviction evidence for sizing (per DEBATE THE POSITION SIZE "
            "above). This is NOT, on its own, a reason to extend the holding window past this "
            "account's own intraday default — that decision still runs entirely through the "
            "HOLDING HORIZON/DEBATE THE HOLDING PERIOD debate above (real cost vs. move, swap "
            "sign); a structurally-backed move that also clears THAT bar is a stronger case for "
            "the overnight exception than an unbacked one would be, nothing more."
        )
    if state == "counter_trend_spike":
        named = " and ".join(opposing)
        return (
            f"  Long-term alignment: COUNTER-TREND SPIKE — the H4 {short}ward move runs "
            f"AGAINST the real {named} backdrop. This is NOT a reason to discard it outright "
            "— a real, fast counter-trend move can be genuinely tradeable on its own terms — "
            "but treat it explicitly as a short-lived move to capture and exit deliberately, "
            "not as the start of a durable trend: a tighter stop, a firmer holding-window "
            "commitment, and sizing that reflects lower conviction in it lasting are all "
            "warranted specifically BECAUSE of this read, not despite it."
        )
    return (
        f"  Long-term alignment: MIXED — the H4 {short}ward move agrees with the "
        f"{' and '.join(agreeing)} backdrop but runs against the {' and '.join(opposing)} "
        "one; partial, not full, longer-term confirmation."
    )


# Real UI messaging standard, set 2026-08-22 after direct user feedback
# on this exact message ("this is very long and complex... give simple,
# decisive message... like 'ball will drop down because gravity is
# present so do not stand under the range of ball otherwise it will hit
# you'"): every user-facing (not AI-prompt-facing) message on this page
# should name the cause, the consequence, and the action in one short
# sentence — not the full reasoning depth a model needs. This function
# is the template other UI messages on this page should follow the same
# way; see app.py's own use of it for a real example.
LONG_TERM_ALIGNMENT_SHORT_MESSAGES = {
    "not_available": "Not enough history yet to read this instrument's short-term trend.",
    "flat": "No real short-term move right now — nothing to size up yet.",
    "no_backdrop": "No long-term trend to check against yet — this move stands alone.",
    "structurally_backed": "Move matches the bigger trend, so it's likely real — size with confidence.",
    "counter_trend_spike": "Move fights the bigger trend, so it's likely a short spike — use a tight stop and exit fast, otherwise it may reverse on you.",
    "mixed": "Daily and monthly trends disagree, so this is only half-confirmed — size smaller.",
}


def format_long_term_alignment_short(analysis: FtmoAssetAnalysis) -> str:
    """One-line, human-facing version of format_long_term_alignment — for
    a UI display, not the AI prompt (which uses the long version above;
    a model needs the reasoning depth, a person reading a dashboard
    doesn't). app.py's own Asset Health popup calls classify_long_term_
    alignment directly instead of this wrapper (it separately needs the
    raw `state` for color-coding, so calling this too would classify the
    same analysis twice for no reason) — this convenience form is for
    any OTHER caller that just wants the one-line message and doesn't
    separately need the state. See LONG_TERM_ALIGNMENT_SHORT_MESSAGES'
    own comment for why these are worded the way they are."""
    state, _short, _agreeing, _opposing = classify_long_term_alignment(analysis)
    return LONG_TERM_ALIGNMENT_SHORT_MESSAGES[state]


def format_setup_signals(label: str, signals: list[SetupSignal]) -> str:
    if not signals:
        return f"  {label} setup read: none computed"
    bits = "; ".join(f"{s.name} — {s.detail}" for s in signals)
    return f"  {label} setup read: {bits}"


def format_ftmo_asset_context(
    analyses: list[FtmoAssetAnalysis], account_equity: float | None = None
) -> str:
    """Reuses format_enriched_asset_context (called once per single-asset
    slice, unchanged) for each symbol's existing daily/feasibility/
    backtest block, then appends chart-structure + setup-classification
    for ALL FOUR timeframes (real feature added live 2026-08-22, user's
    own direct request — see FtmoAssetAnalysis's own docstring for why
    D1/monthly previously stopped at a bare stats line), the multi-
    timeframe confluence/alignment read, and the real trading-cost line
    right after it — keeps every symbol's full read together rather than
    grouping all base reads first and all intraday reads after."""
    lines = []
    for a in analyses:
        lines.append(format_enriched_asset_context([a.base], account_equity=account_equity))
        lines.append(format_chart_structure("D1", a.d1_structure))
        lines.append(format_setup_signals("D1", classify_setups(a.base.stats, a.d1_structure)))
        lines.append(format_timeframe_stats(a.base.symbol, "Monthly", a.mn1_stats, long_term=True))
        lines.append(format_chart_structure("Monthly", a.mn1_structure))
        lines.append(format_setup_signals("Monthly", classify_setups(a.mn1_stats, a.mn1_structure)))
        lines.append(format_timeframe_stats(a.base.symbol, "H4", a.h4_stats))
        lines.append(format_chart_structure("H4", a.h4_structure))
        lines.append(format_setup_signals("H4", classify_setups(a.h4_stats, a.h4_structure)))
        lines.append(format_timeframe_stats(a.base.symbol, "H1", a.h1_stats))
        lines.append(format_chart_structure("H1", a.h1_structure))
        lines.append(format_setup_signals("H1", classify_setups(a.h1_stats, a.h1_structure)))
        lines.append(format_mtf_confluence(a))
        lines.append(format_long_term_alignment(a))
        lines.append(format_ftmo_trade_cost(a))
    return "\n".join(lines)


def fetch_ftmo_status(
    account_summary: AccountSummary, deals: list[HistoricalDeal] | None = None
) -> FtmoStatus:
    """Real, freshly-computed FTMO compliance status for the given (also
    freshly-fetched) account summary. Public/shared so every caller that
    needs a status matching its own exact account snapshot — app.py's
    top-of-page compliance panel, a Portfolio Suggestion click (which
    fetches its own newer account summary a moment later), and the
    unattended daily mega-analysis job (ai/mega_analysis.py) — gets one,
    rather than each silently reasoning over a slightly stale snapshot
    from earlier in its own run, a real (if usually small) correctness
    gap for a compliance-critical number.

    `deals` lets a caller that already fetched the full history this run
    (app.py's live-positions fragment, for its own Trade History table)
    pass it straight through instead of triggering a second identical
    unbounded MT5 history fetch; omit it to fetch fresh."""
    ftmo_deals = deals if deals is not None else get_history_deals(datetime(2000, 1, 1))
    # Excludes deals with no symbol (MT5's own deposit/withdrawal/credit
    # "balance" operations, e.g. the Challenge's initial funding itself)
    # — those aren't trading P&L, and folding them in here would badly
    # understate initial_balance (see risk/ftmo_rules.py's own matching
    # filter for the full reasoning).
    initial_balance = account_summary.balance - sum(d.profit for d in ftmo_deals if d.symbol)
    return compute_ftmo_status(
        ftmo_deals,
        initial_balance=initial_balance,
        current_equity=account_summary.equity,
        current_balance=account_summary.balance,
    )


def format_ftmo_status_context(status: FtmoStatus) -> str:
    """Formats the real, computed FtmoStatus into the exact block the
    prompt text above (_FTMO_STATUS_NOTE) tells the model to expect,
    repeating the same best-effort-reconstruction caveat inline — the
    model never reads risk/ftmo_rules.py's own module docstring, so the
    caveat has to travel with the data itself to actually reach it."""
    lines = [
        "FTMO Compliance Status (best-effort reconstruction from this "
        "account's own real MT5 trade history, NOT a certified mirror of "
        "FTMO's internal ledger — cross-check against FTMO's own "
        "dashboard before relying on this near a hard limit):",
        f"- Today's P&L: realized {status.today_realized_pl:+.2f}, "
        f"floating {status.today_floating_pl:+.2f}, total "
        f"{status.today_total_pl:+.2f}",
        f"- Daily-loss headroom: {status.daily_loss_headroom_pct:.2f}% of "
        f"initial balance remaining before the "
        f"{status.daily_loss_limit_pct:.0f}% daily limit is breached (0 "
        "or negative = already at/over the limit)",
        f"- Trailing max-loss floor (90% of the account's highest-ever "
        f"end-of-day balance so far): {status.trailing_max_loss_floor:,.2f}; "
        f"current equity sits {status.max_loss_headroom_pct:.2f}% of "
        "initial balance above that floor",
    ]
    if status.best_day_rule_pct is not None:
        lines.append(
            f"- Best Day Rule: the current best single day's profit is "
            f"{status.best_day_rule_pct:.1f}% of total profit summed "
            "across all positive days so far (must stay under 50%)"
        )
    else:
        lines.append(
            "- Best Day Rule: not yet computable (no positive trading "
            "day on record yet)"
        )
    return "\n".join(lines)


def build_ftmo_summary(
    account: AccountSummary,
    assets: list[MarketAsset],
    ftmo_status: FtmoStatus,
    positions: list[Position] | None = None,
    analyses: list[FtmoAssetAnalysis] | None = None,
) -> str:
    """`analyses` lets a caller (app.py, for charting) compute
    analyze_ftmo_assets(assets) once and reuse it here, instead of this
    function re-fetching the same H4/H1/contract-spec data internally."""
    lines = [
        f"Account: balance {account.balance:.2f} {account.currency}, "
        f"equity {account.equity:.2f}, free margin {account.free_margin:.2f}",
        "",
        format_ftmo_status_context(ftmo_status),
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

    lines += ["", "Tradable instruments (Market Watch):"]
    if not assets:
        lines.append("- none visible in Market Watch")
    else:
        resolved_analyses = analyze_ftmo_assets(assets) if analyses is None else analyses
        lines.append(format_ftmo_asset_context(resolved_analyses, account_equity=account.equity))

    return "\n".join(lines)


def _fetch_current_ftmo_price(symbol: str) -> float | None:
    """The FTMO-specific `fetch_current_price` callable for
    build_past_outcome_lessons — FTMO symbols are native MT5 symbols with
    a real live connection already open when this runs (unlike PMEX's
    Yahoo-ticker-resolution detour), so this just reads the latest daily
    close straight from the broker's own feed. None on an empty/failed
    fetch, silently skipped by the caller rather than fabricated."""
    history = fetch_mt5_price_history(symbol, "D1", count=1)
    if history.empty:
        return None
    return float(history["Close"].iloc[-1])


def read_latest_suggestion() -> dict:
    """Best-effort read of the FTMO pipeline's own derived, machine-
    readable artifact (see config.MEGA_ANALYSIS_LATEST_SUGGESTION_FILE's
    own comment) — `{}` on anything missing/unreadable, same safe-
    default convention as read_state/read_progress. The sole consumer
    (ai.copilot_execution) treats an empty dict as "nothing to check
    yet," never as an error.

    Despite the "MEGA_ANALYSIS" name (kept for backward compatibility
    with the existing config constant and file on disk — renaming would
    just churn every deployed instance for no behavioral gain), this is
    written by ANY successful suggest_ftmo_portfolio() call now, not
    only the scheduled mega session — see that function's own docstring
    for why: a real bug found live, the manual "Suggest Portfolio Mix"
    button and the scheduled run were built as two separate mechanisms
    when they should always have been the same one, differing only in
    what triggers them."""
    path = Path(config.MEGA_ANALYSIS_LATEST_SUGGESTION_FILE)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_latest_suggestion(final_answer: str) -> None:
    """Parses this FTMO suggestion's own final answer (the exact same
    text the .md session record already embeds) into real JSON, so
    Copilot's execution job never has to guess which saved .md record is
    the relevant one — both a manual click and a scheduled run land in
    the same FTMO_RECORDS_DIR today, indistinguishably. Called
    unconditionally from suggest_ftmo_portfolio() on every real success,
    regardless of caller — see that function's own docstring.

    On a parse failure (a malformed response — rare, but possible), logs
    a warning and leaves whatever was written by the PRIOR successful
    call in place rather than overwriting it with nothing — a stale-but-
    valid suggestion is safer for the execution job to keep reading than
    an empty one, and this must never wipe out a still-valid Pending
    Setups list.

    Written atomically (temp file + os.replace) since — unlike a purely
    cosmetic progress file — this file is read by a SEPARATE OS process
    (copilot_execution_job.py) on its own independent poll cycle and
    directly drives real order placement; a torn read from a non-atomic
    write is a real, if narrow, possibility worth closing off given what
    depends on this file being well-formed."""
    allocation = parse_final_allocation(final_answer, require_side=True)
    if allocation is None:
        logger.warning(
            "Could not parse this FTMO suggestion's own allocation block "
            "— leaving the prior latest-suggestion file untouched."
        )
        return
    immediate_symbols = frozenset(allocation.keys())
    pending_setups = parse_pending_setups(final_answer, immediate_symbols=immediate_symbols)
    if pending_setups is None:
        logger.warning(
            "Could not parse this FTMO suggestion's own Pending Setups "
            "block — leaving the prior latest-suggestion file untouched."
        )
        return

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "immediate_allocation": {
            symbol: {
                "pct": entry.pct,
                "price": entry.price,
                "stop_loss": entry.stop_loss,
                "take_profit": entry.take_profit,
                "side": entry.side,
            }
            for symbol, entry in allocation.items()
        },
        "pending_setups": [
            {
                "symbol": s.symbol,
                "side": s.side,
                "pct": s.pct,
                "trigger_condition": s.trigger_condition,
                "price": s.price,
                "stop_loss": s.stop_loss,
                "take_profit": s.take_profit,
                "reason": s.reason,
            }
            for s in pending_setups
        ],
    }
    path = Path(config.MEGA_ANALYSIS_LATEST_SUGGESTION_FILE)
    try:
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2))
        os.replace(tmp_path, path)
    except OSError as e:
        logger.warning("Could not write latest-suggestion file %s: %s", path, e)


def suggest_ftmo_portfolio(
    summary: str,
    timeout: int | None = None,
    revision_timeout: int | None = None,
    on_stage: Callable[[str], None] | None = None,
    on_audit_progress: Callable[[str], None] | None = None,
    model: str | None = None,
    save_record: bool = False,
    include_copilot: bool = False,
) -> str:
    """Mirrors ai.portfolio_suggest.suggest_portfolio's three-stage draft/
    audit/revise flow exactly, but with FTMO-appropriate instructions and
    audit checklist, and its own records directory
    (config.FTMO_RECORDS_DIR) so FTMO's daily-loss/trailing-max-loss/
    Best-Day-Rule audit lessons never mix with PMEX's or PSX's — the two
    other markets' failure modes don't meaningfully transfer here.

    `include_copilot` defaults to False — direct user request: Copilot
    has a separate, dedicated role for FTMO now (the hourly clerk/
    executioner, see ai/copilot_execution.py) and must not also
    double as an FTMO auditor, whether this is called from the manual
    "Suggest Portfolio Mix" button (app.py) or the unattended scheduled
    run (ai.mega_analysis.run_mega_analysis) — an earlier fix only
    scoped this to the scheduled path, which was too narrow; both FTMO
    call sites now agree. This is FTMO-specific: PMEX's own
    ai.portfolio_suggest.suggest_portfolio and PSX's own suggestion
    pipeline are untouched and still include Copilot in their audit pool
    by default via build_audit_block's own (unchanged) default of True
    — only FTMO's relationship with Copilot has changed.

    On any genuine (non-CLI-failure) result, ALSO parses it into the
    derived MEGA_ANALYSIS_LATEST_SUGGESTION_FILE artifact (see
    _write_latest_suggestion) — the sole source of truth
    ai.copilot_execution's clerk reads to know what's feasible right now
    vs. worth watching for until the next FTMO suggestion. Deliberately
    unconditional on the caller: a real design bug found live had this
    write happening only inside ai.mega_analysis.run_mega_analysis, so
    the manual "Suggest Portfolio Mix" button's own successful runs
    never armed Copilot at all, contradicting the user's own explicit
    intent that manual and scheduled runs differ only in what triggers
    them, never in what they produce or feed downstream."""
    timeout = config.PORTFOLIO_SUGGESTION_TIMEOUT_SECONDS if timeout is None else timeout
    revision_timeout = (
        config.PORTFOLIO_REVISION_TIMEOUT_SECONDS if revision_timeout is None else revision_timeout
    )
    model = config.PORTFOLIO_SUGGESTION_MODEL if model is None else model
    records_dir = Path(config.FTMO_RECORDS_DIR)

    def _notify(message: str) -> None:
        if on_stage:
            on_stage(message)

    session_id = str(uuid.uuid4())
    _notify(
        "Building past-session context (real market outcomes and audit "
        "lessons from previous FTMO runs)..."
    )
    past_lessons = build_past_lessons(
        fetch_current_price=_fetch_current_ftmo_price, records_dir=records_dir
    )
    _notify(
        "Claude is researching the market and drafting an initial FTMO "
        "suggestion (live web search)..."
    )
    draft_prompt = f"{build_ftmo_stage1_instruction()}\n\n{summary}"
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
        include_copilot=include_copilot,
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

    if final_answer == CLI_MISSING_MESSAGE or final_answer.startswith(CLI_FAILED_PREFIX):
        logger.warning(
            "suggest_ftmo_portfolio: resumed stage-2 call failed (%s), falling back to a full-context retry",
            final_answer[:200],
        )
        _notify(
            "The quick revision attempt didn't respond — retrying with a "
            "fresh, fully self-contained request (takes a bit longer, but "
            "doesn't rely on the earlier session still being live)..."
        )
        revise_prompt = (
            f"{build_ftmo_stage2_instruction(audit.audit_available)}\n\n{summary}\n\n"
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

    if not (final_answer == CLI_MISSING_MESSAGE or final_answer.startswith(CLI_FAILED_PREFIX)):
        _write_latest_suggestion(final_answer)

    return final_answer
