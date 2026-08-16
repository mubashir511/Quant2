import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import config
from ai.claude_cli import CLI_FAILED_PREFIX, CLI_MISSING_MESSAGE, run_claude
from ai.portfolio_suggest import (
    AUDIT_MODELS,
    AssetAnalysis,
    analyze_assets,
    build_audit_block,
    build_fx_context,
    build_macro_snapshot,
    build_past_lessons,
    build_positions_context,
    format_enriched_asset_context,
)
from ai.session_record import SessionRecord, save_portfolio_session
from analysis.chart_structure import ChartStructureSnapshot, compute_chart_structure, find_mtf_confluence
from analysis.setup_classifier import SetupSignal, classify_setups
from analysis.technical import TRADING_DAYS_PER_YEAR, TechnicalStats, compute_technical_stats
from data.book_wisdom import format_book_wisdom
from data.mt5_source import (
    AccountSummary,
    MarketAsset,
    Position,
    TradeCost,
    fetch_mt5_price_history,
    get_trade_economics,
)
from risk.ftmo_rules import FtmoStatus

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
    "major economies), and per-instrument technical context on THREE "
    "timeframes — a daily (D1) read where a Yahoo-comparable public price "
    "series exists for the symbol (short-term trend vs. 20-day moving "
    "average, annualized volatility, Average True Range, 14-day RSI, a "
    "volume trend, and a medium-term ~3-month support/resistance range "
    "with a market-type classification), PLUS real H4 and H1 reads "
    "fetched directly from THIS account's own live MT5 price feed for "
    "every symbol regardless of Yahoo availability (the same kind of "
    "stats, computed fresh on each shorter window)."
    "\n\n"
    "HOLDING HORIZON — this account trades INTRADAY TO AT MOST ONE "
    "TRADING DAY: the default expectation for every position is a few "
    "hours, closed the same session, and holding OVERNIGHT is the "
    "exception, not the default (never a multi-day swing, never weeks or "
    "months). The H4 and H1 reads are the PRIMARY basis for the actual "
    "thesis, entry, and stop — not just timing layered on top of a "
    "longer daily story. Use the daily (D1) read, where available, only "
    "as broader regime/bias context (e.g. is the multi-week trend a "
    "tailwind or headwind for a short entry) — never as the source of a "
    "multi-day price target, and never as a reason to hold a position "
    "past its own session by default."
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
    "REAL CHART STRUCTURE is also given per instrument on BOTH H4 and "
    "H1 (deliberately not on the daily read, which is regime/bias "
    "context only per HOLDING HORIZON above) — computed directly in "
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
    "away. Use the touch-count-ranked S/R levels and the Fibonacci "
    "levels as your PRIMARY candidate stop/target anchors (per the "
    "DEBATE THE STOP AND TARGET instruction below) ahead of an arbitrary "
    "round number or a bare ATR multiple with no structural backing — a "
    "stop or target that lines up with a real, multiply-touched level "
    "or a real Fibonacci level is genuinely better-supported than one "
    "that doesn't. A detected chart pattern is real, computed structure "
    "worth reasoning about, but not an automatic signal to trade — say "
    "explicitly what it implies for this specific instrument's setup, "
    "the same way you'd reason about any other single data point, not "
    "as a standalone reason to include or exclude it."
    "\n\n"
    "A SETUP READ is also computed per instrument, per timeframe (H4 "
    "and H1 separately) — a small set of real, rule-based trade "
    "archetypes (reversal_candidate, pullback_continuation, "
    "range_fade_candidate, breakout_watch, trend_following, or "
    "no_clear_setup) synthesized from everything above: trend "
    "structure, RSI, market regime, chart patterns, and proximity to "
    "the real touch-count-ranked S/R and Fibonacci levels. Each comes "
    "with the specific real numbers behind the call, and more than one "
    "can legitimately apply at once (e.g. a pullback happening inside a "
    "converging triangle). Use this as a genuine STARTING characterization "
    "of what kind of trade this instrument actually offers right now — "
    "then reason about it with the rest of the data, don't just restate "
    "it; 'no_clear_setup' on a timeframe is a real, common result that "
    "argues against forcing an entry there, not a gap to explain away."
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
    "day, long side), both read directly from this account's own live "
    "MT5 feed, not estimated. This is not optional context: a target "
    "that clears 2:1 on paper can still be a poor or even losing trade "
    "after real costs, especially at this account's short holding "
    "horizon where costs are a larger fraction of the total expected "
    "move than they would be on a multi-week position — treat this line "
    "with the same weight as the ATR/volatility data when sizing a stop "
    "and target, not as a footnote."
    "\n\n"
    f"{_FTMO_STATUS_NOTE}"
    "\n\n"
    "You're also given a set of investing-literature principles (position "
    "sizing, portfolio heat, group-exposure limits, stop/profit "
    "discipline), each attributed to the named author who argues for it. "
    "These are advisory, not enforced rules. For each one that's "
    "genuinely relevant here, briefly explain *why* that author argues "
    "for it and use that reasoning — not just the bare rule — to inform "
    "your sizing. You may reason your way to a different conclusion than "
    "a principle would suggest, but say so and explain why."
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
    "support/resistance RELIABILITY backtest above (the real % of that "
    "instrument's own past tests where a level actually held or was "
    "rejected) — a target sitting at a level with a weak historical "
    "hold/reject rate for THIS instrument is a genuinely weaker target "
    "than the same distance sitting at a level with a strong one, even "
    "though the two look identical on a raw price chart.\n\n"
    "DEBATE THE POSITION SIZE PER INSTRUMENT, don't spread capital "
    "evenly or apply one flat % across the mix. Weigh, explicitly, per "
    "instrument: (1) conviction strength — how well-supported is this "
    "instrument's thesis by real backtest evidence and genuine research, "
    "not a hunch; a higher-conviction, well-evidenced case earns a "
    "meaningfully larger size than a marginal one; (2) real volatility "
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
    "account's REAL remaining compliance headroom (its % allocation "
    "times its stop distance, against the daily-loss and trailing max-"
    "loss headroom given above) — a hard ceiling that applies regardless "
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
    "   e. Idle cash — treat any cash reserve beyond roughly 10-25% of "
    "equity as needing its own explicit justification. Whatever reserve "
    "you do leave, define specific conditional triggers for deploying "
    "it (tied to the support/resistance levels given per instrument), "
    "each with an explicit time-based fallback.\n"
    "   f. Position sizing and stop/target — confirm every position's "
    "size and stop/target actually came from the DEBATE THE POSITION "
    "SIZE and DEBATE THE STOP AND TARGET paragraphs above (conviction, "
    "real volatility, feasibility ceiling, real cost-to-risk, compliance-"
    "headroom contribution for sizing; ATR-multiple and support/"
    "resistance reliability, both justified per instrument, for stop/"
    "target) rather than a uniform size or a flat ATR multiple applied "
    "across the mix regardless of these differences.\n"
    "   g. FTMO COMPLIANCE HEADROOM — the single most important check "
    "for this account. For every position in the mix, multiply its % "
    "allocation by its stop-loss distance (%) to get that position's "
    "contribution to capital at risk, sum these across the whole mix "
    "(the same 'aggregate heat' figure other markets use), then "
    "explicitly compare that total against BOTH the real daily-loss "
    "headroom % AND the real trailing max-loss headroom % given in the "
    "FTMO Compliance Status section above — a worst-case simultaneous "
    "stop-out must leave REAL headroom under both, not just avoid "
    "crossing zero exactly, since a breach of either is instant "
    "termination with zero grace period. Separately, reason explicitly "
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
    "and state each category's total % explicitly; if any one category "
    "exceeds roughly 30-40% of equity, justify why explicitly or resize "
    "it down. WITHIN each represented category, prefer 1-2 instruments "
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
    "fundamental + research-based catalyst, across the D1/H4/H1 reads), "
    "AND three explicit, brief debate outputs (checklist items a, d, f "
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
    "inside a fenced code block. The ONLY fenced code block in your "
    "entire response must be a single one at the very end (after the "
    "'Recommended Allocation' section), exactly like this (replace the "
    "example values with your actual final numbers, one key per "
    "instrument symbol traded above plus one \"CASH\" key, pct values "
    "summing to 100, no comments or extra text inside the block). Every "
    "non-CASH key must be an object with four numbers: \"pct\" (the "
    "target allocation), \"price\" (a specific limit-order entry price — "
    "realistic and achievable given the instrument's current bid/ask "
    "shown above, not a distant level or an arbitrary round number), "
    "\"stop_loss\" (a specific stop price, derived from the real H4/H1 "
    "ATR where available rather than an assumed flat percentage), and "
    "\"take_profit\" (a specific target price on the correct side of "
    "your entry — above it for a long — matching the exact target level "
    "your own reward:risk reasoning above already derives; this is the "
    "field that actually gets sent to FTMO as the position's real take-"
    "profit order, not just prose, so it must be the same real, cost-"
    "netted number your DEBATE THE STOP AND TARGET reasoning used, not a "
    "rounded-off or re-guessed one). If an "
    "instrument is currently held (see Current Open Positions above) but "
    "you're not including it in the final mix, it must still appear here "
    "with \"pct\": 0 — never omit a currently-held instrument:\n"
    "```json\n"
    '{"EXAMPLE_SYMBOL": {"pct": 15, "price": 82.50, "stop_loss": 78.00, "take_profit": 94.00}, "CASH": 25}\n'
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
    "- Check SETUP READ and MULTI-TIMEFRAME use: did the draft actually "
    "engage with each position's computed setup read (reversal, "
    "pullback, range-fade, breakout-watch, trend-following) rather than "
    "ignore it, and does its stated thesis genuinely match that "
    "characterization rather than contradict it (e.g. calling something "
    "a trend-following entry when the setup read says range_fade_"
    "candidate)? Did it use a real H4/H1 CONFLICTING trend read as a "
    "reason for caution, or silently pick whichever timeframe agreed "
    "with its own thesis? Flag any stop/target that ignored a real "
    "multi-timeframe confluence level sitting right where a same-"
    "timeframe level alone was used instead.\n"
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
    "historical backtest evidence given below (its real past RSI-"
    "reaction rate and average forward return, whether its own history "
    "shows real momentum persistence or mean-reversion, whether its own "
    "low-volatility episodes have historically been followed by bigger "
    "or smaller moves, and how often its shown support/resistance "
    "levels have actually held or been rejected). Note: there is no "
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
    """Wraps a PMEX-shape AssetAnalysis (`base`, reused UNCHANGED from
    ai.portfolio_suggest.analyze_assets — its contract-spec fetch is
    already generic native-MT5, and its Yahoo-ticker daily technical/
    backtest enrichment works exactly the same way for whatever's in
    this FTMO account's own Market Watch) with the two extra intraday
    reads that are genuinely unique to FTMO's own tighter, faster-paced
    compliance timeline: real H4/H1 technical stats fetched directly
    from this account's own live MT5 price feed via
    data.mt5_source.fetch_mt5_price_history — a real broker-native fetch
    that works for every FTMO symbol regardless of whether it also has a
    Yahoo mapping (most won't, since PMEX's own keyword map targets
    commodity-futures naming, not forex/CFD tickers). h4_structure/
    h1_structure carry the real Fibonacci/support-resistance/trendline/
    chart-pattern reads for those same two timeframes — deliberately NOT
    computed for the daily (D1) read, since the HOLDING HORIZON debate
    already demotes D1 to broader regime context rather than the
    surface entry/stop/target decisions are actually made on."""

    base: AssetAnalysis
    h4_stats: TechnicalStats
    h1_stats: TechnicalStats
    h4_structure: ChartStructureSnapshot
    h1_structure: ChartStructureSnapshot
    trade_cost: TradeCost | None


def analyze_ftmo_assets(
    assets: list[MarketAsset], on_progress: Callable[[str], None] | None = None
) -> list[FtmoAssetAnalysis]:
    """Reuses ai.portfolio_suggest.analyze_assets() directly for the base
    (contract-spec + best-effort Yahoo daily) read, then extends it with
    real H4/H1 reads per symbol — see FtmoAssetAnalysis's own docstring
    for why this wrap-rather-than-duplicate shape was chosen.

    `on_progress` (same single-updating-message contract as
    analyze_assets' own param) covers BOTH passes — the base analyze_assets
    call first, then this function's own extension loop (H4/H1 fetch,
    chart structure ×2, real trading cost) — so a caller sees continuous
    progress across the full, now genuinely multi-step analysis instead
    of it going quiet again right as the FTMO-specific work starts."""
    base_analyses = analyze_assets(assets, on_progress=on_progress)
    results = []
    total = len(base_analyses)
    for i, base in enumerate(base_analyses, start=1):
        if on_progress is not None:
            on_progress(
                f"Multi-timeframe technical + chart structure + real trading cost: "
                f"{i}/{total} — {base.symbol}"
            )
        h4_history = fetch_mt5_price_history(base.symbol, "H4")
        h1_history = fetch_mt5_price_history(base.symbol, "H1")
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
                trade_cost=get_trade_economics(base.symbol),
            )
        )
    return results


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

    swap_display = (
        f"{cost.swap_long_pct_per_day:+.4f}%/day held long"
        if cost.swap_long_pct_per_day is not None
        else "not available for this swap calculation mode"
    )

    return (
        f"  REAL trading cost ({cost.category}): spread {cost.spread_pct_of_price:.4f}% + "
        f"commission {commission_display} = {round_trip_pct:.4f}% round-trip{round_trip_note} "
        f"(paid once, entry+exit combined, before any move in price); overnight swap "
        f"{swap_display} — multiply by planned holding days for a multi-day hold"
    )


def _format_intraday_stats(symbol: str, label: str, stats: TechnicalStats) -> str:
    if stats.last_price is None:
        return (
            f"  {label} technical (near-term entry timing/stop "
            f"placement): not available (insufficient MT5 {label} "
            f"history for {symbol})"
        )
    bits = [f"last {stats.last_price:.4f}"]
    if stats.trend is not None and stats.pct_vs_sma20 is not None:
        bits.append(f"{stats.trend} ({stats.pct_vs_sma20:+.1f}% vs 20-bar SMA)")
    if stats.rsi is not None:
        bits.append(f"RSI {stats.rsi:.0f}")
    if stats.atr_pct is not None:
        bits.append(f"ATR {stats.atr_pct:.2f}% of price (use for near-term stop distance)")
    if stats.volatility_annualized_pct is not None:
        bits.append(f"volatility {stats.volatility_annualized_pct:.1f}% (annualized)")
    return f"  {label} technical (near-term entry timing/stop placement): {', '.join(bits)}"


def _format_chart_structure(label: str, snapshot: ChartStructureSnapshot) -> str:
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


def _format_mtf_confluence(analysis: FtmoAssetAnalysis) -> str:
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


def _format_setup_signals(label: str, signals: list[SetupSignal]) -> str:
    if not signals:
        return f"  {label} setup read: none computed"
    bits = "; ".join(f"{s.name} — {s.detail}" for s in signals)
    return f"  {label} setup read: {bits}"


def format_ftmo_asset_context(
    analyses: list[FtmoAssetAnalysis], account_equity: float | None = None
) -> str:
    """Reuses format_enriched_asset_context (called once per single-asset
    slice, unchanged) for each symbol's existing daily/feasibility/
    backtest block, then appends the H4/H1 technical + chart-structure +
    setup-classification lines, the multi-timeframe confluence/alignment
    read, and the real trading-cost line right after it — keeps every
    symbol's full read together rather than grouping all base reads
    first and all intraday reads after."""
    lines = []
    for a in analyses:
        lines.append(format_enriched_asset_context([a.base], account_equity=account_equity))
        lines.append(_format_intraday_stats(a.base.symbol, "H4", a.h4_stats))
        lines.append(_format_chart_structure("H4", a.h4_structure))
        lines.append(_format_setup_signals("H4", classify_setups(a.h4_stats, a.h4_structure)))
        lines.append(_format_intraday_stats(a.base.symbol, "H1", a.h1_stats))
        lines.append(_format_chart_structure("H1", a.h1_structure))
        lines.append(_format_setup_signals("H1", classify_setups(a.h1_stats, a.h1_structure)))
        lines.append(_format_mtf_confluence(a))
        lines.append(format_ftmo_trade_cost(a))
    return "\n".join(lines)


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


def suggest_ftmo_portfolio(
    summary: str,
    timeout: int | None = None,
    revision_timeout: int | None = None,
    on_stage: Callable[[str], None] | None = None,
    on_audit_progress: Callable[[str], None] | None = None,
    model: str | None = None,
    save_record: bool = False,
) -> str:
    """Mirrors ai.portfolio_suggest.suggest_portfolio's three-stage draft/
    audit/revise flow exactly, but with FTMO-appropriate instructions and
    audit checklist, and its own records directory
    (config.FTMO_RECORDS_DIR) so FTMO's daily-loss/trailing-max-loss/
    Best-Day-Rule audit lessons never mix with PMEX's or PSX's — the two
    other markets' failure modes don't meaningfully transfer here."""
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

    return final_answer
