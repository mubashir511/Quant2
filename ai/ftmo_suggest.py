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
from ai import live_recheck, rehunt, trade_journal
from ai.curiosity import build_curiosity_report
from ai.portfolio_suggest import (
    AUDIT_MODELS,
    AssetAnalysis,
    _real_execution_note,
    build_audit_block,
    build_fx_context,
    build_macro_snapshot,
    build_past_lessons,
    build_pending_orders_context,
    build_positions_context,
    format_enriched_asset_context,
    parse_final_allocation,
    parse_pending_setups,
)
from ai.session_record import SessionRecord, save_portfolio_session
from analysis.backtest import (
    LevelReliabilityBacktest,
    RSIReactionBacktest,
    SupportResistanceBacktest,
    backtest_chart_pattern_reaction,
    backtest_level_reliability,
    backtest_momentum_persistence,
    backtest_rsi_reaction,
    backtest_support_resistance_reaction,
    backtest_volatility_regime,
)
from analysis.playbook import RangeRead, advise as playbook_advise, range_read
from analysis.position_hunter import HuntFacts, format_position_hunt, hunt
from analysis.edge_stats import (
    CONTRADICTED,
    NO_INFORMATION,
    SUPPORTED,
    EdgeBaseline,
    EdgeVerdict,
    classify_edge,
    compute_null_baseline,
)
from analysis.chart_structure import (
    ChartStructureSnapshot,
    SRLevelsResult,
    compute_chart_structure,
    find_mtf_confluence,
    find_swing_points,
)
from analysis.intraday_context import IntradayLevels, compute_intraday_levels
from analysis.htf_flags import closed_htf_directions
from analysis.level_map import build_level_map
from analysis.symbol_card import format_card as format_symbol_card
from analysis.setup_classifier import SetupSignal, classify_setups
from analysis.technical import (
    TRADING_DAYS_PER_YEAR,
    DivergenceSignal,
    TechnicalStats,
    compute_technical_stats,
    detect_rsi_divergence,
)
from analysis.timeframe_profiles import M5_PROFILE
from analysis.trade_zone import TRADE_ZONE_STOP_ATR_MULTIPLE, construct_trade_zone
from data.book_wisdom import format_book_wisdom, format_trend_wisdom
from data.chart_wisdom import format_chart_wisdom
from data.mt5_source import (
    AccountSummary,
    ContractSpec,
    HistoricalDeal,
    MarketAsset,
    PendingOrder,
    Position,
    TradeCost,
    fetch_mt5_price_history,
    get_contract_spec,
    get_history_deals,
    get_symbol_category,
    get_trade_economics,
    is_symbol_tradable_now,
)
from data import economic_calendar, symbol_news
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
_M5_PERIODS_PER_YEAR = 288 * TRADING_DAYS_PER_YEAR  # 288 five-minute bars/day

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

_POSITION_HUNTING_PROTOCOL = (
    "POSITION-HUNTING PROTOCOL (how to search for trades; this replaces filtering on backtests and ATR). "
    "The account is an intraday hunter, and measured on its own real data the biggest leaks are NOT a lack of "
    "conviction: (1) a pooled backtest of ~27% wins at a slightly negative average R is exactly what RANDOM entries "
    "with the same stop/target score, so it carries almost no information about a structural thesis — the M5 "
    "backtest lines therefore print a verdict against a random-entry baseline, and only CONTRADICTED (clearly "
    "worse than random on a real sample) is a genuine warning; NO-INFORMATION is never a reason to exclude or to "
    "shrink a trade. (2) An ATR reading, a 'quiet' or 'choppy' M5 tape, or a low win rate are not reasons either "
    "unless you quote the printed number that makes the trade unworkable. (3) The broker's round-trip cost is the "
    "one measured leak that is objective: the sizing sheet prints it in R for each instrument (mean 0.19R at a "
    "2x M5-ATR stop, 0.09R at 4x, above 1R on a few thin instruments) — that is what the cost veto is for.\n"
    "1. START from the POSITION HUNT shortlist above: it is ranked mainly by structure and cost, with only a LIGHT "
    "preference for trades with the closed D1/H4 trend (measured on 30,790 / 28,486 M5 breakout fills, both time "
    "halves: gross +0.074R with the trend vs +0.038R against it - a nudge of ~0.04R that reversed in 2008-2015, "
    "not a rule). A breakout AGAINST the higher-timeframe trend is a fully tradable candidate and needs no higher bar "
    "or special size cut just for that label; judge every candidate on its own entry, stop, target, cost and structure. "
    "Backtest lines are CONTEXT for sizing only - never a reason to exclude (they are mostly NO-INFORMATION and noisy). "
    "Correlation between symbols is information, not a reason: the aggregate-heat ceiling already counts every stop "
    "as if all were hit together, so if several assets of the same class are individually good, take them.\n"
    "2. EXCLUDE only with a reason a reviewer can check against the printed numbers: a veto id (V1-V4, V7) that "
    "applies to that candidate right now, or a specific printed fact (an event time, a level, a number). 'Weak "
    "setup', 'no clear edge' or silence for a shortlisted candidate is not sufficient — say what would have to "
    "be true for you to take it. A vetoed candidate may still be traded only if you address the veto in its reason.\n"
    "3. CHOOSE THE ENTRY ORDER for each trade from where the market is. MEASURED on 34,407 with-the-trend breakout moments "
    "(20 symbols, up to 200k M5 bars each, both time halves): a resting limit at the nearest reaction level is the WORST "
    "entry (-0.03..-0.05R gross per fill, it often never fills), a market order is +0.02..+0.03R, and a BREAKOUT STOP at the "
    "recent range extreme is +0.04..+0.07R - so the with-the-trend default is the PLAYBOOK line printed under each shortlisted "
    "candidate (order type, trigger, structure stop, target). Take it unless you can name a printed fact against it; a "
    "PLAYBOOK status of WAIT/EXTENDED means do not chase (O'Neil; Schwager Rule 8) - wait for a secondary consolidation or "
    "list it as a Pending Setup with a structured trigger. Fill odds for a limit N M5 ATRs from the price (touched within 3 "
    "hours): 1.0x 79%, 1.6x 67%, 2.3x 55%, 3x 46%, 4x 34%, 4.5x 30%. Use a limit ONLY for a deliberate counter-move entry "
    "within ~1.6 ATR of the price at a REACTION POINT from the LEVEL MAP; \"entry_mode\": \"stop\" (a buy stop above / sell "
    "stop below the market) for a breakout THROUGH a printed level, still not extended; \"entry_mode\": \"market\" for a "
    "continuation already in motion. The final live re-check prints which modes are actually available and verifies your "
    "choice; the Clerk re-verifies it at send time.\n"
    "3b. LEVEL CHOICE (the LEVEL MAP under each symbol): the map lists every real candidate nearest-first - zones, unbroken "
    "REACTION POINTS the price actually turned from, range/session extremes, Fibonacci - with measured fill odds. MEASURED: no "
    "kind of level (touches, liquidity pool, round number, higher-timeframe confluence, volume, Fibonacci) beat a random "
    "price at the same distance, so do NOT call a level 'the strongest' as if that were established; distance and what the "
    "price has DONE are the evidence. If you pick a level that is not the nearest real candidate, name the nearer ones and "
    "say why they fail (a recent reaction point that price turned from and did not undercut is the usual reason not to "
    "skip it - the SOLUSD case: price bounced from the earlier low, never reaching the deeper zone chosen). Place the stop "
    "BEYOND the level by the printed penetration buffer (levels that held were still pierced by a median 0.5, P75 1.3, "
    "P90 2.7 ATR).\n"
    "3c. STRUCTURED TRIGGERS: a resting limit may never fill and a free-text trigger is judged by a small model. For a Pending "
    "Setup that is a break or a reclaim of a specific price, add the optional \"trigger\" object (range_break / reclaim / "
    "close_beyond + level): Python then checks it on the completed M5 bars every poll and fires a market entry when it is "
    "met and not extended. On M5, a touch beyond the level is enough - a close filter measured worse.\n"
    "4. THE STOP IS THE INVALIDATION LEVEL, NOT AN ATR HABIT: place it beyond the swing extreme or level that "
    "would prove the thesis wrong (the trade-zone line prints a structure-stop alternative and each stop's cost "
    "drag; for a breakout the PLAYBOOK prints the opposite side of the 24-bar range, clipped 1.5-4 ATR). MEASURED: the "
    "structure stop alone was worth +0.07..+0.11R per opportunity over a fixed 2 ATR stop (fewer wick-outs, lower spread "
    "drag) and it helps any entry type. A wider stop with a proportionally smaller size is fine — the pct you give is the "
    "risk, so a wide stop does not add risk, and it lowers the cost drag. Do not add soft (close-based) stops, time stops "
    "or a TP1 partial: all measured no better than the plain hard stop plus the trail. A target beyond the ~6 M5 ATR reach line is acceptable as the "
    "RUNNER of a plan that also has a nearer partial target: the Clerk trails the stop 1.5 M5 ATR behind the price "
    "once a trade is +1R (measured: better than breakeven-only or partial-only rules), so say in the reason where "
    "the partial is and that the rest is trailed.\n"
    "5. RE-HUNT: an entry from an earlier session that died unfilled is listed under RE-HUNT CANDIDATES when there "
    "is one; re-issue it once with fresh levels if its thesis holds today, otherwise leave it out and say why.\n"
    "SIZE BY MEASURED QUALITY, not by trend label: base 0.4% risk per trade; up to 0.75% when EVERY playbook filter is OK "
    "(ADX >= 30 and rising, tick volume >= 1x), the cost drag is <= 0.05R and the stop is a real structure stop; 0.25-0.3% when "
    "filters are WEAK. Never above 0.75% per trade; the 1.5% aggregate ceiling is unchanged. Why the cap: simulated with the "
    "measured R distribution (edge +0.03..+0.06R), the 3% daily-loss limit is never the binding constraint under the "
    "heat ceiling - the 10% max-loss line is, and the odds of eventually blowing it rise from ~13% at 0.25% per trade to "
    "~25% at 0.4%, ~34% at 0.75% and ~43% at 1.5% while the expected return stays ~+2% per quarter, so extra risk buys speed, "
    "not profit.\n"
    "None of this asks you to trade more for its own sake: a session with two excellent, well-sized entries is a "
    "good session — the protocol is about not discarding a workable aligned candidate for a reason the data does "
    "not support."
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
    "If an 'Outstanding Pending Orders' section appears below, this "
    "account also has one or more real, not-yet-filled GTC limit orders "
    "resting in MT5 alongside (or instead of) any open positions — each "
    "already committed to entering at a specific level if price reaches "
    "it, not a hypothetical you're free to ignore. Every one of them "
    "must be reconciled explicitly in your final mix, using the exact "
    "same rule as an open position: keep it (restate the same side and "
    "\"pct\", and the same price/stop_loss/take_profit if nothing about "
    "it should change), update it (restate it with a NEW price, "
    "stop_loss, and/or take_profit — same side, same symbol — if "
    "today's data now calls for different terms while the underlying "
    "idea still holds), or cancel it (give it \"pct\": 0 in the final "
    "mix, exactly like closing a currently-held instrument) — never "
    "omit a symbol with an outstanding pending order from the final "
    "allocation block. A pending order that's been resting a long time "
    "(its own age is given below) without filling is not automatically "
    "wrong, but IS worth a genuine, explicit judgment call: has enough "
    "changed since it was placed that its entry level, stop, or target "
    "no longer reflect the current real technical picture, or is the "
    "original thesis still intact and simply still waiting for price to "
    "arrive?"
    "\n\n"
    "You're also given: a macro snapshot (US Treasury yield curve, dollar "
    "index, VIX, and GDP growth/inflation/unemployment for 10 major "
    "economies), an economic calendar of upcoming High-impact events "
    "(central-bank decisions, CPI/NFP-type releases, speeches — with UTC "
    "times, plus a per-instrument line for the currencies that move it), "
    "and per-instrument technical context on FOUR timeframes in two "
    "tiers: the DECISION TIER — the M5 read, the ONLY basis for every "
    "entry, stop, target, trigger, invalidation and size — and the "
    "CONTEXT TIER — daily (D1), H4 and H1 reads, for regime and trend "
    "context only. All are fetched directly from THIS account's own live MT5 "
    "price feed (D1 instead comes from a Yahoo-comparable public series "
    "where one exists, e.g. gold's own continuous futures series — either "
    "way it's real multi-year price history, not a gap). Each read "
    "carries a short-term trend vs. 20-day/-bar moving average, "
    "annualized volatility, Average True Range, 14-period RSI, a volume "
    "trend, and a medium-term ~60-bar support/resistance range with a "
    "market-type classification — 'sideways' (no real net move over that "
    "window), 'trending_up'/'trending_down' (a real move, reached "
    "cleanly), or 'choppy_up'/'choppy_down' (a real move, but via a noisy "
    "back-and-forth path — still genuine directional evidence, not a "
    "weaker cousin of sideways) — computed identically on every "
    "timeframe, so a 60-bar window means ~3 months on daily, ~10 days on "
    "H4, ~2.5 days on H1, and only ~5 hours on M5 (exactly the window an "
    "intraday entry needs)."
    "\n\n"
    "HOLDING HORIZON AND TIMEFRAME ROLES — this account trades INTRADAY "
    "TO AT MOST ONE TRADING DAY: the default expectation for every "
    "position is a few hours, closed the same session, and holding "
    "OVERNIGHT is the exception, not the default (never a multi-day "
    "swing, never weeks or months). Each timeframe has ONE job. M5 is the "
    "ONLY DECISION timeframe and the PRIMARY (and only) basis for the actual "
    "entry, stop, target, trigger condition, invalidation condition and "
    "position size. M5 supplies the STRUCTURE — the touch-count-ranked "
    "support/resistance, Fibonacci, trendlines and patterns that say WHERE "
    "a stop or target sits, and the setup archetype — the TIMING — the "
    "precise entry zone, the confirming trigger, the first sign a setup is "
    "failing — and the ATR that sizes the stop DISTANCE (at least 2x the M5 "
    "ATR, and never inside the spread/broker minimums) and says how far a "
    "target can realistically travel in a few-hour hold (about 6x the M5 "
    "ATR). The 'Session levels' line "
    "(prior-day high/low/close, the session's range, session VWAP, the "
    "average daily range and how much of it is already used) tells you "
    "where an intraday stop or target realistically sits and how much "
    "movement is left. The D1, H4 and H1 reads are the CONTEXT TIER: use "
    "them ONLY to decide the regime and the permission — is the larger "
    "trend a tailwind or a headwind for the M5 setup, which direction is "
    "allowed, where are the big magnet levels a trade should not run into "
    "— and NEVER as the source of an entry, stop, target or size (a "
    "higher-timeframe level may be named as extra confluence when it "
    "lines up with an M5 level, nothing more). A setup that runs "
    "against the context tier is a counter-trend trade and must be "
    "labeled and sized as one (see the next paragraph); a strong M5 "
    "setup inside a context tier that shows no clear trend is a "
    "legitimate range trade, sized accordingly. Never treat a daily read "
    "as a reason to hold a position past its own session by default."
    "\n\n"
    "REAL, NOT A FAKE, TREND — user's own explicit concern, and the "
    "reason the daily context above exists at all: a real "
    "'Long-term alignment' line is computed per instrument (see below), "
    "comparing the M5 move actually being considered against the real "
    "daily backdrop. A STRUCTURALLY BACKED read (the M5 move "
    "agrees with the real longer-term direction) is genuine additional "
    "confirmation — weigh it as such. A COUNTER-TREND SPIKE read (the M5 "
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
    "relative to a realistic few-hours move on THIS instrument's own M5 "
    "ATR (a few hours of travel is on the order of 6x the M5 ATR) — if the cost eats a large share of what a normal intraday "
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
    "CHECK WHETHER THIS INSTRUMENT'S MARKET IS EVEN OPEN RIGHT NOW — each "
    "instrument's own data section above states its live market status "
    "('market CLOSED (weekend)' when closed, nothing extra when open). "
    "Crypto (BTCUSD/ETHUSD) trades 24/7 and is always open. Forex/exotics "
    "close for the weekend and reopen Sunday evening UTC. Metals, "
    "commodities, indices, and equities stay closed the entire weekend, "
    "until Monday. Do NOT propose a NEW immediate allocation or a NEW "
    "pending setup on an instrument marked CLOSED — it literally cannot "
    "fill right now. Prioritize your new proposals among instruments that "
    "are open right now; a currently-closed instrument can still be "
    "discussed as a Monday-forward watchlist idea, but not proposed as "
    "something expected to fill today. This does NOT change how you "
    "evaluate or hold an EXISTING position on an instrument that happens "
    "to be closed right now — that stays governed entirely by the "
    "holding-period/swap guidance above, unaffected by this note."
    "\n\n"
    "HEADING INTO A WEEKEND, DON'T PROPOSE A NEW PENDING SETUP THAT CAN'T "
    "ACTUALLY BE MANAGED BEFORE THE MARKET CLOSES — every pending setup "
    "you propose is placed as a real, resting GTC limit order; if this "
    "instrument's market closes for the weekend before that order would "
    "realistically trigger, it just sits unmanaged for 2+ closed days "
    "with nobody able to react to it, and this account's own execution "
    "layer will cancel it outright rather than let it ride into Monday's "
    "reopen at a possibly-gapped price. This does NOT apply to crypto "
    "(BTCUSD/ETHUSD), which genuinely trades through the weekend — treat "
    "it exactly like any other instrument. For every OTHER instrument, "
    "if you're proposing a fresh pending setup late in the trading week, "
    "either size it with a realistic chance of triggering before the "
    "close, or explicitly note the timing risk in its own reason if you "
    "propose it anyway. A freed-up risk-budget from a weekend cancellation "
    "is a real, legitimate reason to look harder at a weekend-tradable "
    "instrument (crypto) if one genuinely earns it on its own technical/"
    "fundamental case — never size one up just because idle risk-budget "
    "exists."
    "\n\n"
    "REAL CHART STRUCTURE is also given per instrument on ALL FOUR "
    "timeframes — M5, H1, H4 and D1 — computed directly in "
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
    "lower-highs/lower-lows, triangle/wedge from converging trendlines, "
    "and Nison's own candlestick shapes on the LATEST closed bar — doji "
    "(indecision, never directional on its own), bullish/bearish "
    "engulfing, hammer/hanging man, shooting star/inverted hammer, and "
    "morning/evening star, each only surfaced when the exact mechanical "
    "shape/context Nison defines is genuinely met, never a loose visual "
    "impression) — 'not enough confirmed swing points yet' or no "
    "patterns listed is a normal, common result, not a gap to explain "
    "away. The M5 touch-count-ranked S/R levels, Fibonacci levels and "
    "trendlines are your ONLY candidate stop/target anchors (per the "
    "DEBATE THE STOP AND TARGET instruction below), consistent with "
    "HOLDING HORIZON above. Use them ahead of an arbitrary round "
    "number or a bare ATR multiple with no structural backing — a stop or "
    "target that lines up with a real, multiply-touched level or a real "
    "Fibonacci level is genuinely better-supported than one that doesn't. "
    "The D1, H4 and H1 structure serve a DIFFERENT purpose — real context for "
    "the broader thesis and the Long-term alignment read below, not a "
    "literal stop/target anchor for a same-session trade — but a "
    "genuinely major higher-timeframe level (a heavily-touched one, or "
    "one a real chart pattern points at) sitting near where M5 would "
    "already place a stop or target is still worth naming explicitly: "
    "it's additional real evidence for that specific price (the 'M5 "
    "levels backed by H1/H4' list in the Intraday trend alignment line "
    "already computes this), the same logic MULTI-TIMEFRAME CONFLUENCE "
    "below applies. A detected chart pattern on "
    "ANY timeframe is real, computed structure worth reasoning about, "
    "but not an automatic signal to trade — say explicitly what it "
    "implies for this specific instrument's setup, the same way you'd "
    "reason about any other single data point, not as a standalone "
    "reason to include or exclude it."
    "\n\n"
    "A SETUP READ is also computed per instrument, per timeframe — "
    "D1, H4, H1 and M5 separately — a small set of real, "
    "rule-based trade archetypes (reversal_candidate, pullback_continuation, "
    "range_fade_candidate, breakout_watch, trend_following, trend_intact, "
    "grind_continuation, in_progress_move, busted_pattern_reversal, "
    "candlestick_reversal_confirmed, or no_clear_setup) synthesized from "
    "everything above: trend structure, RSI, market regime, momentum "
    "acceleration, chart/candlestick patterns, and proximity to the real "
    "touch-count-ranked S/R and Fibonacci levels. "
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
    "here, between retest moments. in_progress_move means the medium-term "
    "market type reads 'sideways' (several recent swings cancelling out in "
    "that window's own average) but a much shorter window shows a real, "
    "efficient push already underway right now — don't dismiss this as "
    "noise just because the medium-term regime hasn't caught up yet; a "
    "fresh move can be genuinely underway inside what still looks like a "
    "flat window. busted_pattern_reversal means a double-top/double-bottom "
    "implied one breakout direction but real subsequent price action "
    "instead confirms the OPPOSITE direction — per Bulkowski's own "
    "documented statistics, a busted pattern often travels FURTHER than "
    "the original pattern's own target (traders positioned for the "
    "expected breakout are forced to reverse into the move), so treat "
    "this as stronger directional evidence in the busted direction, not a "
    "reason to distrust the underlying data. candlestick_reversal_"
    "confirmed means a real candlestick reversal shape (e.g. a hammer, "
    "engulfing, or star pattern) appeared in the specific context that "
    "makes it meaningful (a bullish shape after a downtrend or at a "
    "double-bottom; bearish after an uptrend or at a double-top) — added "
    "weight behind that reversal thesis, not a standalone trigger on its "
    "own; a bare 'doji' appearing in the chart-structure list below (not "
    "a setup read on its own) means genuine indecision, not a directional "
    "signal either way. Each "
    "comes with the specific real numbers behind the call, and more than "
    "one can legitimately apply at once (e.g. a pullback happening inside "
    "a converging triangle). Use this as a genuine STARTING characterization "
    "of what kind of trade this instrument actually offers right now — "
    "then reason about it with the rest of the data, don't just restate "
    "it; 'no_clear_setup' on a timeframe is a real, common result that "
    "argues against forcing an entry there, not a gap to explain away. "
    "The D1, H4 and H1 setup reads are context for the broader thesis and the "
    "Long-term alignment read below — per HOLDING HORIZON above, the M5 "
    "setup read is the one that actually governs this trade's "
    "entry/stop/target: it names the trade archetype and its structure, "
    "and its trade-zone candidate says where the entry, stop and target "
    "sit and whether the entry is ready now."
    "\n\n"
    "TWO MULTI-TIMEFRAME READS are given per instrument. The primary one "
    "is the Intraday trend alignment line (the M5 decision read vs the H1 "
    "context, with H4 beside it): "
    "ALIGNED (all three read the same real direction — genuinely stronger "
    "evidence), an M5 vs H1 CONFLICT (M5 pulling against the H1 context — a "
    "headwind to weigh in conviction, size and target distance, or treat it "
    "explicitly as a pullback entry, never a silent pick of a side), MIXED, or none showing a clear trend "
    "— plus the M5 levels that coincide with an independently-computed "
    "H1/H4 level. Weight an M5 level with H1/H4 backing as a stronger "
    "stop/target anchor than an M5 level with an equivalent touch count "
    "alone. The secondary, context-tier one is the H4 vs H1 line (ALIGNED "
    "/ CONFLICTING / PARTIAL / neither shows a clear trend, with H4/H1 "
    "confluence levels): it describes the regime the M5 trade sits "
    "inside, and a CONFLICTING H4 vs H1 read is still a real reason for "
    "caution or a smaller size on a trade that depends on that regime, "
    "not something to silently pick a side on; PARTIAL (one trending, the "
    "other flat) is normal consolidation, not a genuine conflict."
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
    "say so and explain why. One principle specifically needs this "
    "account's own context applied rather than taken at face value: "
    "O'Neil's position-count tiers (a 'small account' capped around 3 "
    "positions) describe a personal, open-ended brokerage portfolio with "
    "no daily kill-switch, compounded over years — a different context "
    "from this account's own daily-reset compliance limits, worth "
    "weighing rather than taken at face value either direction. Earlier "
    "guidance here told you to override this tier outright and favor "
    "spreading into many small positions; that instruction is now "
    "REVERSED (direct user instruction, 2026-09-10, after two real, "
    "observed problems it caused: (a) a mega session bought BTCUSD while "
    "the account's own data explicitly read 'H4 and H1 ALIGNED on a "
    "downtrend' — a real, specific conflicting signal that never got "
    "addressed in the thesis, while a weaker, genuinely-present bullish "
    "H4 pattern did; (b) several sessions in a row where padding the mix "
    "with enough positions to feel diversified, against a small fixed "
    "total-risk ceiling, produced stops too tight for the instrument's "
    "own real oscillation and take-profits that rarely got hit before "
    "the trend reversed — several positions simply too small to clear "
    "the broker's own minimum lot size, sitting dead in the pending-"
    "order table). This account's real risk control is still the hard "
    "aggregate-heat ceiling (checklist item g below) — but that ceiling "
    "is a MAXIMUM you must never exceed, not a budget you're obligated "
    "to spend down to zero by adding more names. Choose the best real "
    "trend(s) today's data actually supports and size each one "
    "genuinely — real velocity, real structure, a realistic stop/target, "
    "and whether the resulting risk actually clears this instrument's "
    "own minimum-lot floor — one position or five, whichever the "
    "evidence honestly supports that day. Padding the count to look "
    "diversified, or shrinking a real position's risk below what its "
    "own minimum lot can express just to fit more names in, are both "
    "mistakes now, not the goal."
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
    "derived from M5 ATR (or beyond real structure), a real target, sizing per the DEBATE THE "
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
    "volatility first (the M5 ATR read specifically, not an assumed "
    "flat percentage), derive the target from a real technical level "
    "that's realistically reachable within that SPECIFIC instrument's "
    "own intended holding window from the HOLDING HORIZON debate above "
    "(typically the same session, at most one trading day) — a "
    "technically valid level that would typically take multiple days to "
    "reach is not a legitimate target for a same-session hold; size or "
    "select a nearer one instead. Make this a NUMERIC self-check, not a "
    "judgment call: state explicitly, per instrument, 'reward distance = "
    "<price/pips> = <N>x the M5 ATR' (the M5 trade-zone candidate already "
    "lists this multiple for each of its targets and stars any beyond the limit) — if that multiple exceeds roughly "
    "6x the M5 ATR (about the median furthest excursion an instrument reaches from its "
    "starting point in a 4-hour holding window — and in ONE chosen direction price "
    "touches a level N M5 ATRs away within 3 hours only about 60% of the time at 2x, "
    "45% at 3x and 26% at 5x, so a target at 6x is a long shot), or the target sits beyond the average daily "
    "range still remaining on the 'Session levels' line, the target is "
    "a long shot (more often a multi-day level) wearing a same-session label; "
    "either select a nearer level that actually sits within ~6x the M5 "
    "ATR (the same odds apply to an ENTRY: the M5 trade-zone line states how many M5 ATRs "
    "its entry zone sits from the last price, and a resting limit 3.5x away fills less than "
    "2 times in 5 inside an order's 3-hour life — prefer a nearer zone or a trigger-based "
    "entry over a far limit), or explicitly widen the stated "
    "holding horizon for that instrument (with its own cost/swap "
    "justification) rather than leaving a same-session horizon paired "
    "with a target that ATR math shows cannot realistically arrive that "
    "fast — then report whatever ratio actually "
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
    "quoting only the gross ratio. A real, deterministic backstop now "
    "enforces the low end of this: if the net (cost-adjusted) ratio comes "
    "out below roughly 1.8:1, the position is automatically REJECTED "
    "downstream — not sized down, not carried at reduced conviction — "
    "regardless of how compelling the underlying story is. Don't spend a "
    "full thesis on a candidate you can already tell will fail this once "
    "netted; either find a genuinely better entry/stop/target combination "
    "for it, or leave it out from the start. Conversely, the same "
    "backstop works in your favor too: when a specific instrument's own "
    "real, well-sampled M5 win rate (shown in its own Same-session "
    "(M5) backtest evidence below, with a SUPPORTED verdict against its random-entry baseline) is genuinely strong, that setup's "
    "own real breakeven ratio is mathematically lower than 2:1 — a "
    "modest-target, high-probability trade backed by that real evidence "
    "is a legitimate, valuable inclusion, not something to discard just "
    "because it doesn't hit the usual 2:1 headline number. Don't "
    "manufacture a stretched target just to look like a 'proper' swing "
    "trade when the real, honest small-target version is what the "
    "evidence actually supports."
    "\n\n"
    "DON'T LET CHASING A BETTER RATIO BECOME AN EXCUSE TO MISS THE "
    "TREND. Real pattern confirmed live: a resting pending buy limit for "
    "one instrument sat unfilled for days while price ran straight past "
    "it — up over 11% from where the setup was first proposed — with "
    "each session simply re-stating 'the thesis hasn't been invalidated, "
    "it simply hasn't triggered' and carrying the SAME old entry/stop/"
    "target forward unchanged. By the time this was caught, price had "
    "already run past the ORIGINAL TAKE-PROFIT too — the trade's own "
    "exit level was behind current price and the position had never "
    "opened. The instinct behind waiting (get a bigger, textbook reward:"
    "risk ratio by entering at a deeper, more favorable price) is real "
    "and usually correct — but on a genuinely ALIGNED M5 trend (with the "
    "H1/H4 context not against it), "
    "particularly one this account already trades on the M5 "
    "timeframe rather than swinging for a multi-day move, that instinct "
    "can quietly become GREED: holding out for the ideal entry that "
    "maximizes the ratio, while the trend itself keeps moving further "
    "away and the whole idea goes untraded. Whenever you carry forward "
    "an existing pending/resting order rather than proposing a fresh "
    "one, explicitly state (1) how long it's been resting, (2) how far "
    "current price has moved from BOTH its entry and its target since it "
    "was first proposed, and (3) reason honestly about whether the "
    "multi-timeframe trend read still supports waiting, or whether "
    "current price now offers a smaller but real, achievable reward "
    "that's worth taking over continuing to hold out for the original, "
    "larger one. If price has already reached or passed the ORIGINAL "
    "TARGET without the entry ever filling, treat that as a hard signal "
    "the level needs fresh reconsideration — either re-derive the entry/"
    "target from the nearest CURRENT support/resistance and re-justify "
    "the new ATR-multiples and R:R from scratch, or drop it — simply "
    "re-asserting the old thesis as 'not yet invalidated' is not an "
    "acceptable response to that specific signal. A smaller, honestly-"
    "reported reward:risk from a real, current entry that's actually IN "
    "the trend beats a textbook ratio sitting unfilled at a price the "
    "market has already left behind."
    "\n\n"
    "DEBATE THE STOP AND TARGET PER INSTRUMENT, don't apply one ATR "
    "multiple or a round percentage uniformly across the mix. Justify "
    "explicitly, per instrument, how many multiples of M5 ATR the "
    "stop sits away from entry, and why THAT multiple for THIS "
    "instrument specifically — a tighter multiple only belongs on an "
    "instrument whose own volatility-regime backtest above shows its "
    "quiet periods have historically stayed quiet (low noise/shakeout "
    "risk), while an instrument whose history shows frequent whipsaws "
    "around a tight level needs a wider stop, or a smaller size, rather "
    "than a stop that just gets clipped by that instrument's own normal "
    "noise. Anchor the actual stop/target PRICE to the real M5 "
    "support/resistance levels, Fibonacci levels, trendlines, prior-day high/low and session VWAP given "
    "above where one genuinely lines up nearby — a stop placed just "
    "beyond a real, multiply-touched level (not inside it, where normal "
    "noise around that level would clip it) or a target set at a real "
    "level with strong touch-count evidence is a concretely better-"
    "placed level than an arbitrary ATR-multiple distance with nothing "
    "structural behind it. A real MULTI-TIMEFRAME CONFLUENCE level "
    "(an M5 level an independent H1/H4 level also lands on) is stronger still than a "
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
    "USE THE REAL HISTORICAL FAVORABLE-EXCURSION MAGNITUDE FOR TP SIZING "
    "WHEN IT'S GIVEN — don't just default to a flat 2:1 ATR-multiple "
    "target. Each backtest above can report a SEPARATE figure alongside "
    "its win rate: the real median/mean distance (same R-multiple units "
    "as the win-rate figures) this exact setup's underlying price has "
    "historically continued traveling in its favorable direction. "
    "CRITICAL, easy to misread: this figure is measured across EVERY "
    "historical entry (winners, losers, and timeouts alike) and is "
    "floored at zero, so it comes back POSITIVE for nearly every setup on "
    "nearly every instrument REGARDLESS of whether that setup actually "
    "wins — a setup that mostly LOSES can still show a real positive "
    "median excursion simply because price often wobbles favorably for a "
    "while before reversing into the loss that actually closes the "
    "trade. A positive number here is NOT independent evidence the setup "
    "works and does NOT offset a poor win rate/negative avg-R on the "
    "SAME line above — always read this figure TOGETHER WITH that "
    "win-rate figure, never as a substitute for it or a reason to "
    "reconsider a setup its own win rate already argues against; a "
    "setup with a strong win rate AND a large favorable-excursion figure "
    "is genuinely better evidence than either fact alone, but a large "
    "favorable-excursion figure on a setup with a poor win rate just "
    "describes how far price tends to drift before typically failing, "
    "not a reason for confidence in it. It's also "
    "measured on DAILY bars over a MUCH LONGER look-forward window than "
    "this account's own HOLDING HORIZON above — this can mean weeks to "
    "several months of real price travel, not the few-hours-to-one-day "
    "window this account actually holds a position for. Treat it EXACTLY "
    "like the D1 regime context per HOLDING HORIZON above: real "
    "background evidence for how big this setup's underlying move CAN "
    "get, NEVER the source of the actual same-session TP price itself — "
    "the same mistake the D1 'never a multi-day price target' rule "
    "above already exists to prevent. This figure is also DIFFERENT from "
    "the fixed target the win-rate/avg-R figures above are artificially "
    "capped at, and is NOT netted against real trading cost/swap the way "
    "the win-rate figure's own avg-R is (a pure price-magnitude reading, "
    "not a realized-trade P&L figure). Its own line explicitly says which "
    "side/setup it belongs to and is drawn from THAT SAME side's own "
    "historical entries above it — its sample size will usually be "
    "SMALLER than that side's own trade count, since it requires a "
    "longer, undisturbed look-forward window to count an entry at all; "
    "that gap is expected, not a data inconsistency. Prefer the MEDIAN "
    "over the mean "
    "as the typical reading — a real move-size distribution is "
    "right-skewed, so a few outsized trend runs can pull the mean up "
    "hard without being the representative case. Make this a NUMERIC "
    "self-check too, same discipline as "
    "the reward:risk check above: when this figure is available for a "
    "setup you're using, state explicitly whether your chosen target "
    "sits NEAR, WELL SHORT of, or WELL PAST that setup's own historical "
    "median magnitude, and justify which — a target set well past the "
    "historical median is a bet this specific instance will outperform "
    "most of this setup's own history, which needs a real, stated reason "
    "(e.g. genuinely stronger multi-timeframe confluence than the median "
    "case), not silence. Do NOT simply set every target to this figure "
    "by rote — it's a distribution's median/mean over real but nonuniform "
    "history, not a promise this specific trade travels that far; a real, "
    "disclosed sample size is given alongside it, and a small one "
    "deserves the same skepticism you already give a thin win-rate "
    "sample. Do NOT stretch or shrink your own read of a real technical "
    "level (S/R, Fibonacci, trendline) just to make your target match "
    "this number either — anchor the target to a real structural level "
    "first (per DEBATE THE STOP AND TARGET above), then use this figure "
    "to judge whether that level's implied distance is conservative, in "
    "line with, or aggressive relative to this setup's own real history, "
    "not the other way around. This figure can also inform CONVICTION "
    "and sizing (see DEBATE THE POSITION SIZE below) — but ONLY for a "
    "setup whose win rate/avg-R above already supports it on its own "
    "merits (per the CRITICAL warning near the start of this paragraph): "
    "for such a setup, a genuinely larger typical favorable move, at a "
    "real sample size, is one additional real input toward higher "
    "conviction, never a substitute for that win rate and never "
    "sufficient on its own. When "
    "this figure isn't available (too few real historical entries kept a "
    "full look-forward window, or no High/Low data at all), treat the "
    "flat 2:1 ATR target as the only real anchor available and say so, "
    "rather than inventing a magnitude claim with nothing behind it."
    "\n\n"
    "WEIGH THIS INSTRUMENT'S OWN VELOCITY, NOT JUST ITS ATR MAGNITUDE — "
    "real incident: a real trade on this account was stopped out by an "
    "ordinary ~9-point wiggle on gold within 93 minutes, even though its "
    "own stated invalidation condition never actually fired (RSI stayed "
    "nowhere near oversold, price never closed through the stated level) "
    "— the stop was simply narrower than that instrument's own normal, "
    "fast noise. The M5 ATR% given per instrument above isn't just a "
    "distance to multiply — it also tells you how much of that "
    "instrument's own hourly range typically arrives in a single burst "
    "rather than spread evenly across the hour: an instrument whose "
    "typical (median over the last ~2 days) M5 ATR% is roughly 0.06% or higher (gold and copper "
    "sit near 0.10%, crypto 0.13-0.40%, equities 0.19-0.45%; most FX majors/crosses instead "
    "run 0.02-0.04%) can cover a meaningful fraction of its own typical "
    "hourly range in minutes, not hours — a stop sized as if it were a "
    "slow, diffusive mover gets clipped by that instrument's own ordinary "
    "noise far more often than the ATR multiple alone would suggest. For "
    "an instrument at or above that threshold, either widen the ATR "
    "multiple genuinely (not just nudge it) relative to what a slow FX "
    "cross would get, or explicitly size the position smaller to hold the "
    "same wider stop at the same dollar risk — state which choice you "
    "made and why, the same way COUNTER-TREND SPIKE above already asks "
    "you to justify a tighter multiple, just in the opposite direction "
    "here.\n\n"
    "WEIGH HOW DEEP A PULLBACK ENTRY SHOULD REALISTICALLY WAIT FOR — real "
    "incident: real INTC and AMD buy-limit orders both sat below a "
    "genuinely fast-moving market waiting for a technical pullback that "
    "never came, both eventually cancelled with price already well past "
    "their own original take-profit, having never come close to filling "
    "— the wait itself was the mistake, not the direction. A deep "
    "retracement (a 50-61.8% Fibonacci retrace, or a wait for price to "
    "return all the way to a support/resistance zone) is a reasonable "
    "entry style for a genuinely range-bound or newly-reversing "
    "instrument, but is fighting the evidence on one already showing a "
    "real uptrend/downtrend structure (see each timeframe's own chart "
    "structure above — 'uptrend_structure'/'downtrend_structure' means "
    "the two most recent swing highs AND the two most recent swing lows "
    "both agree on direction, a genuinely stronger claim than price "
    "merely sitting above a moving average) AND reading fast-tier on its "
    "own M5 ATR% (see the velocity discussion above): a strong, fast "
    "trend's own pullbacks are typically shallow and brief by "
    "construction — the strength of the move is largely the absence of a "
    "deep retest. For a fast-tier instrument already in confirmed trend "
    "structure, weigh a shallower continuation entry (nearer the current "
    "price, or triggered off a smaller real retracement) or a "
    "breakout/momentum entry with a reduced size against the standard "
    "deep-retracement wait, and state explicitly which you chose and "
    "why — this is a real, per-instrument judgment call, not a rule to "
    "apply uniformly (a genuinely range-bound or slow-tier instrument "
    "gets no such adjustment: the deep-retracement wait remains "
    "perfectly reasonable there). Also weigh any real structural level "
    "flagged as a LIQUIDITY POOL (a tight cluster of genuine equal highs/"
    "lows — the classic signature of where other traders' own stops or "
    "pending orders concentrate): a pullback ENTRY sitting exactly at "
    "such a level risks the same fate as a STOP sitting there (see "
    "the S/R discussion in each timeframe's own chart structure above) "
    "— price may tag it only briefly on a stop-hunt before continuing, "
    "never truly settling there long enough to fill or hold a passive "
    "order.\n\n"
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
    "(the M5 ATR% and annualized volatility% given above) — a more "
    "volatile instrument needs a SMALLER size than a calmer one for the "
    "same $ risk, not the same size; (3) the REAL feasibility ceiling — "
    "TWO separate hard floors, both genuine, not the same constraint: the "
    "minimum-lot MARGIN requirement given above (can the account afford "
    "the min lot's margin at all — rarely the real constraint on a "
    "funded account), and separately the 'minimum viable size' line "
    "given per instrument below (can the pct this instrument would "
    "realistically get, against a REAL stop distance, actually clear "
    "the broker's own minimum lot — this is the one that has actually "
    "been biting: real positions sized below it don't trade smaller, "
    "they round to zero and die in the pending-order table unfilled). "
    "If the risk budget this instrument would realistically get falls "
    "under its own minimum-viable-size floor, drop it entirely rather "
    "than sizing it below that floor and hoping it fills anyway. For an "
    "ALREADY-HELD position whose stop you're revising, use the exact "
    "'Held-position sizing rates' number given for it (below the main "
    "per-instrument detail) verbatim — pct = that rate x your chosen "
    "stop distance — rather than estimating a small round pct yourself: "
    "real incident, 2026-09-11, a held NVDA position's own revised stop "
    "was reported at a pct just under what that exact distance actually "
    "needed, silently rounding to LESS than one tradable share and "
    "blocking the whole revision from ever reaching the real account for "
    "hours, with no error visible anywhere in that session's own report; "
    "(4) the "
    "REAL trading cost relative to the stop distance "
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
    "of how strong the other four inputs look. Also read the 'Intraday sizing sheet' given per instrument (lots and margin share at a 2x M5 ATR stop — tight M5-scale stops buy MORE lots per risk % than an H1-scale stop would, so check the margin share is sane) and any upcoming High-impact event line for that instrument's currencies: a position expected to be open across such an event should be sized down or skipped (Clerk also refuses new resting entries from 30 minutes before to 10 minutes after one). State the resulting % "
    "and the reasoning behind it explicitly per position — a uniform "
    "size across every instrument regardless of these five real inputs "
    "is exactly the pattern this paragraph exists to prevent."
    "\n\n" + _POSITION_HUNTING_PROTOCOL
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
    "swing by default), and should look across the "
    "distinct asset categories actually tradable here (forex, metals, "
    "commodities/agriculturals, indices, equities, crypto where offered) "
    "rather than defaulting to whichever is cheapest/easiest to check — "
    "but how many positions end up in any one category, or in the mix "
    "overall, must come from how many REAL, independently-supported "
    "setups actually exist today, never from a target headcount (direct "
    "user instruction, reversing an earlier 2026-09-05 instruction that "
    "pushed the opposite way, after it produced stops too tight for real "
    "instrument oscillation and positions too small to clear the "
    "broker's own minimum lot across several sessions). A category with "
    "one genuinely strong setup and nothing else is a one-name category "
    "today, not a gap to fill. The aggregate-heat ceiling below is the "
    "one hard limit on total risk — treat it as a maximum, never a "
    "budget you're obligated to spend down to zero. Still favor the most "
    "liquid/widely-traded name in a category when two candidates are "
    "otherwise comparable. Before including any "
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
    "\n\n"
    "ONE CLASS OF AUDIT OBJECTION IS NOT OPTIONAL TO WEIGH AWAY, "
    "regardless of which model raised it: an objection anchored to a "
    "concrete, checkable fact already sitting in the data given below — "
    "an audit correctly pointing out that the M5 (or H1/H4/D1 context) structure "
    "read you used (trend/setup classification) doesn't match what the "
    "actual computed data says, or that this SPECIFIC instrument's own "
    "backtest for the setup type you're proposing (e.g. buying an "
    "oversold bounce) is CONTRADICTED by its own backtest — clearly worse than the random-entry baseline printed with it on a real sample (a win rate near 27% / a slightly negative average R is what RANDOM entries score, so on its own it is NO-INFORMATION, not an objection). "
    "Confirmed live: a past run's H1 was actually reading downtrend_"
    "structure while the draft called it an 'aligned uptrend' with H4, "
    "and this instrument's own oversold-RSI-long backtest showed a 14-17% "
    "win rate with a negative average R — two independent audits caught "
    "both, and the final revision kept the same long thesis nearly "
    "verbatim anyway; the position went on to lose in exactly the way "
    "the ignored data predicted. Silently keeping a conclusion that a "
    "hard, data-backed objection like this directly contradicts is not "
    "an acceptable outcome of the internal weighing above — either the "
    "position/direction/size genuinely changes to account for it, or "
    "your visible thesis for that instrument explicitly states the real "
    "conflicting number (the actual trend read, or the actual backtest "
    "win rate/avg R) and gives a specific, genuine reason it's being "
    "taken anyway (e.g. a fundamental catalyst strong enough to override "
    "a middling technical backtest). A thesis that never mentions an "
    "instrument's own unfavorable backtest or a contradicted trend "
    "classification, while still asserting the read it contradicts, is "
    "a defect — this is the one place where reflecting the resolution "
    "in the visible answer is required, not merely internal process."
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
    "   c. Correlation under stress (LOW weight) — construct one adverse macro "
    "scenario from the yield/DXY/VIX data given above and assess whether "
    "the mix's positions would move together more than their individual "
    "weights suggest. Never flag a position merely for being correlated with another: the aggregate-heat "
    "ceiling already sums every stop as if all were hit together, and same-class setups that are each good "
    "may all be taken.\n"
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
    "   h. Asset-CATEGORY perspective — this account trades across "
    "several distinct asset categories actually tradable in the Market "
    "Watch instruments below (typically: forex majors/crosses, metals, "
    "energies/agricultural commodities, indices, equities, and crypto "
    "where offered). Group the mix's positions by category and state "
    "each category's total % explicitly; pct is risk, not notional, and "
    "the hard ceiling is item g's real number above (1.5% on a fresh "
    "day, not a wider figure) — judge concentration as a SHARE of THAT "
    "budget, not of raw equity; if any one category eats more than "
    "roughly a third to half of the mix's total aggregate heat, justify "
    "why explicitly or resize it down. REVERSED direct user instruction "
    "(originally 2026-09-05, reversed 2026-09-10 after two real problems "
    "it caused: a session bought BTCUSD while the account's own data "
    "explicitly read 'H4 and H1 ALIGNED on a downtrend', unaddressed in "
    "the thesis; and several sessions in a row where padding a category "
    "to avoid looking 'thin' produced stops too tight for real "
    "oscillation and positions too small to clear the broker's own "
    "minimum lot, sitting dead in the pending-order table): do NOT pad a "
    "category with additional positions just to avoid looking thin or "
    "'1-2 names' — a category holding ONE genuinely strong, well-"
    "evidenced setup and nothing else is complete as-is; a category with "
    "three real, independently-supported setups is also fine. The "
    "number of names in a category is a RESULT of how much real evidence "
    "exists today, never a target to hit either direction. A real, clean "
    "directional read on its OWN technical merit (trend_intact/"
    "trend_following/grind_continuation, or better, a genuinely ALIGNED "
    "multi-timeframe read) is necessary but NOT sufficient on its own — "
    "actually weigh it against any CONFLICTING read on another relevant "
    "timeframe before including it (the exact BTCUSD mistake above: a "
    "genuine H4 bullish pattern existed alongside an equally explicit "
    "'H4 and H1 ALIGNED on a downtrend' line that the thesis never "
    "engaged with at all). Stating a real technical reason for an "
    "exclusion is still good practice when you have one, but excluding "
    "an instrument simply because today's evidence for it is weaker than "
    "what you did include is now a completely legitimate reason on its "
    "own — you do not owe every technically-qualifying instrument a "
    "position. Still prefer the most liquid/widely-traded instrument in "
    "a category when two candidates are otherwise genuinely comparable "
    "(tighter spreads, more reliable stops/fills, deeper backtest "
    "history), and never include a same-category instrument purely "
    "to look diversified without its OWN independent technical case — the "
    "bar is genuine evidence per instrument, not a headcount target "
    "either direction.\n"
    "   i. Asset-category exclusion — if an entire asset category from "
    "item (h) is present and tradable in the Market Watch instruments "
    "below but ends up at 0% in your final mix, state that exclusion "
    "explicitly and justify it (no genuinely attractive instrument found "
    "in it this run is a legitimate reason — forcing a category in "
    "without a real setup is not). Apply EQUAL research depth across "
    "categories before concluding one has nothing — a real pattern "
    "across this account's own past sessions is forex ending up "
    "included far more often than metals/indices/equities/crypto "
    "combined, which is at least as likely to reflect forex simply "
    "getting checked first and more thoroughly (it's cheaper and "
    "always liquid, so it's an easy default) as it is to reflect forex "
    "genuinely having the best setups every single day. Before writing "
    "off a category, confirm you actually looked at ITS OWN M5 "
    "structure, backtest, and any live catalyst with the same effort "
    "spent on forex — not just a quick glance because a forex idea "
    "already looked good enough to stop searching.\n"
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
    "fundamental + research-based catalyst, across the M5 decision tier and the D1/H4/H1 context tier "
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
    "idea should be held to the exact same evidence bar as a long: an M5 backtest verdict for that direction "
    "that is not CONTRADICTED (SUPPORTED is better, NO-INFORMATION is neutral), aligned M5 signals (with a context tier that doesn't argue against it), a "
    "genuine technical setup, OR a genuine fundamental-based case per "
    "the FUNDAMENTAL-BASED INCLUSION paragraph above (not contradicted "
    "by the technical read) — not proposed as an afterthought just "
    "because one is now technically possible. A short-biased finding "
    "your own M5 reads already sometimes surface — a "
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
    "\"stop_loss\" (a specific stop price, derived from the real M5 "
    "ATR where available rather than an assumed flat percentage), and "
    "\"take_profit\" (a specific target price on the correct side of "
    "your entry FOR THE DIRECTION YOU CHOSE — above entry for a long, "
    "below entry for a short — matching the exact target level "
    "your own reward:risk reasoning above already derives; this is the "
    "field that actually gets sent to FTMO as the position's real take-"
    "profit order, not just prose, so it must be the same real, cost-"
    "netted number your DEBATE THE STOP AND TARGET reasoning used, not a "
    "rounded-off or re-guessed one), \"reason\" (one or two sentences: "
    "the specific thesis for THIS target right now — required whenever "
    "\"pct\" is above 0; may be a short, honest one-liner like \"No "
    "change from yesterday's thesis\" when a still-held or still-pending "
    "position's own case genuinely hasn't changed, but must still be a "
    "real, current statement, not a placeholder), and "
    "\"invalidation_condition\" (a SPECIFIC, mechanically-checkable "
    "condition — the exact same style already required for a Pending "
    "Setup's own \"trigger_condition\" below: an exact price level plus "
    "an indicator threshold and the timeframe it's read on (M5 — the decision tier; a higher-timeframe regime clause may be added but never replaces it), e.g. \"M5 "
    "closes below 1.0900\" or \"RSI(14) breaks above 75 on M5\" — a "
    "separate automated process re-checks this every few minutes "
    "against fresh live technicals for as long as this position stays "
    "open or pending, and treats it firing as a signal to exit/cancel "
    "this position immediately, without waiting for the next mega "
    "analysis; vague language like \"if it stops looking good\" cannot "
    "be mechanically checked and must not be used. Required whenever "
    "\"pct\" is above 0; omit it, or set it to null, for \"pct\": 0/CASH, "
    "since there's nothing left to invalidate). Optionally, for an IMMEDIATE entry, add "
    "\"entry_mode\": \"limit\" (the default when omitted — a resting pullback order at \"price\"), "
    "\"stop\" (a buy stop above / sell stop below the market with \"price\" as the breakout trigger, "
    "which must be a real level) or \"market\" (enter now at the live ask/bid; \"price\" is then the "
    "entry you assumed and must be within about half an M5 ATR of it) — see the POSITION-HUNTING "
    "PROTOCOL. \"stop_loss\" and \"take_profit\" are attached to the order in every mode. The Clerk "
    "re-verifies the mode against the live quote and falls back to a limit (or drops a dead setup) "
    "if it no longer holds; never use it for a Pending Setup. To close an already-held "
    "position or cancel an already-outstanding pending order RIGHT NOW "
    "rather than wait for its invalidation_condition, simply give it "
    "\"pct\": 0 here and explain why in \"reason\" — no different from "
    "dropping any other instrument from the mix. To update the "
    "stop-loss and/or take-profit of an already-held position or an "
    "already-outstanding pending order while keeping the same side and "
    "the same size, restate it here with the SAME \"pct\" and \"side\" "
    "but the NEW \"stop_loss\"/\"take_profit\" values — the account's "
    "own automated execution layer applies the change directly to the "
    "existing position/order rather than closing and reopening it, so "
    "this is the correct way to revise a stop or target without "
    "disturbing an already-committed entry. If an "
    "instrument is currently held (see Current Open Positions above) but "
    "you're not including it in the final mix, it must still appear here "
    "with \"pct\": 0 — never omit a currently-held instrument:\n"
    "```json\n"
    '{"EXAMPLE_LONG": {"side": "buy", "pct": 1.5, "price": 82.50, "stop_loss": 78.00, "take_profit": 94.00, '
    '"reason": "Pullback into a well-tested M5 support zone within an intact uptrend.", '
    '"invalidation_condition": "M5 closes below 76.00"}, '
    '"EXAMPLE_SHORT": {"side": "sell", "pct": 1.5, "price": 145.00, "stop_loss": 149.50, "take_profit": 133.00, '
    '"reason": "Fading a rejection at a heavily-touched M5 resistance level.", '
    '"invalidation_condition": "M5 RSI(14) breaks above 70"}, '
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
    "the timeframe it's read on (M5 — the decision tier), e.g. \"M5 closes above 1.0950 with "
    "M5 RSI(14) below 35 turning up, and H4 trend_intact remains true\" — a "
    "separate automated process re-evaluates this every few minutes against fresh "
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
    "sentences), and OPTIONALLY \"trigger\": a STRUCTURED trigger that Python "
    "checks exactly on the completed M5 bars (no model opinion, no waiting for the next "
    "mega session) and turns into a MARKET entry the moment it fires: "
    "{\"kind\": \"range_break\" (price trades through \"level\" in the trade's "
    "direction and is not extended past it - the with-the-trend playbook), or \"reclaim\" (a "
    "bar pierced \"level\" against the trade and a later bar closed back inside - a "
    "spring/upthrust), or \"close_beyond\" (the last completed M5 bar closed beyond "
    "\"level\"), \"level\": a real price from the charts/Level Map, \"within_bars\": "
    "optional 1-24}. Use it whenever the setup is a break or a reclaim of a specific "
    "price - it triggers reliably where a resting limit may never fill; keep the free-text "
    "\"trigger_condition\" too (it is what the reader sees and the fallback when no "
    "trigger object is given):\n"
    "```json\n"
    '[{"symbol": "EXAMPLE_WATCH", "side": "buy", "pct": 1.0, '
    '"trigger_condition": "M5 closes above 2000.00 with M5 RSI(14) below '
    '35 turning up, and H4 trend_intact remains true", "price": 2000.00, '
    '"stop_loss": 1980.00, "take_profit": 2050.00, "reason": "Pullback '
    'into a well-tested M5 support zone within an intact uptrend."}]\n'
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
    "sizing look sound given the M5 volatility/ATR data given (D1/H4/H1 are context).\n"
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
    "read rather than the M5 read that should be doing the actual "
    "entry/stop/target work, and any position held/implied overnight "
    "without an explicit per-instrument justification (real cost vs. "
    "realistic move, and swap SIGN specifically) — an overnight hold "
    "with no stated reason, or one that ignores an unfavorable swap "
    "sign, is a real gap to flag. Do this one NUMERICALLY, not "
    "impressionistically: for each same-session position, take its own "
    "stated reward distance and its own stated M5 ATR (both given in "
    "the draft), and divide — a target that comes out to roughly more "
    "than roughly 6x that same M5 ATR reading (or beyond the remaining average daily range) is a concrete achievability "
    "violation regardless of how the draft narrates it or what ratio it "
    "claims; name the instrument and the actual multiple you computed "
    "rather than only asserting the target 'looks far'.\n"
    "- Check STALE CARRIED-FORWARD LEVELS, real pattern confirmed live: "
    "a pending order sat unfilled for days while price ran over 11% past "
    "it and past its own take-profit, each session just repeating 'not "
    "yet invalidated' rather than re-examining anything. For any "
    "position carried forward unchanged from a prior session, check "
    "whether the draft actually disclosed how far price has moved from "
    "the entry AND the target since it was first proposed — if it "
    "didn't, that's a real gap to flag. If price has already reached or "
    "passed the ORIGINAL TARGET without the entry ever filling and the "
    "draft still just reasserts the old thesis unchanged, that's a "
    "concrete instance of chasing a bigger reward:risk ratio at the cost "
    "of missing a trend that's already confirmed and moving — name it "
    "explicitly, the same weight as a math error.\n"
    "- Check ASSET-CATEGORY PERSPECTIVE: does the mix look across "
    "the distinct categories actually tradable (forex, "
    "metals, commodities/agriculturals, indices, equities, crypto where offered), "
    "favoring liquid names by default when two candidates are otherwise "
    "comparable? Flag over-concentration in one category (measured as its "
    "own share of the mix's total aggregate heat — see the FTMO "
    "compliance-math check above — not a bare name count), an unjustified total "
    "absence of an available category, or any same-category instrument "
    "included without its OWN independent technical/fundamental case. "
    "REVERSED per direct user instruction (2026-09-10): a category "
    "holding just one genuinely strong position, or an overall mix of "
    "just one or two positions total, is NOT itself something to flag — "
    "challenge the QUALITY of each position's own case, never the raw "
    "count. Conversely, DO flag: (a) any position sized so small "
    "(relative to the heat ceiling and how many other positions are "
    "competing for it) that its risk distance can't clear the "
    "instrument's own minimum-lot floor, or ends up tighter than its own "
    "real ATR-based stop distance warrants — a sign the mix tried to fit "
    "too many positions into too little risk budget, not a sign of good "
    "diversification; (b) any position whose thesis cites one "
    "timeframe's technical pattern while a DIFFERENT computed read on "
    "another relevant timeframe (a multi-timeframe alignment line, a "
    "setup read, a chart pattern) points the opposite direction and "
    "never gets addressed — a real incident this specifically guards "
    "against: a BTCUSD buy cited a bullish H4 pattern while the same "
    "data explicitly read 'H4 and H1 ALIGNED on a downtrend,' left "
    "unaddressed in the thesis.\n"
    "- Check REAL TRADING COST awareness: was every reported reward:risk "
    "ratio actually netted against the REAL trading cost line given per "
    "instrument (spread + this account's own confirmed commission), not "
    "just the gross price move? Flag any position whose reward:risk "
    "would fall meaningfully short of what's claimed once that real cost "
    "is subtracted, and any multi-day hold that didn't fold in the real "
    "swap %/day times the expected holding days.\n"
    "- Check POSITION-SIZE AND STOP/TARGET DEBATE: did every position's "
    "size actually reflect the five real inputs it should (conviction "
    "strength backed by real evidence, real M5 volatility, the "
    "feasibility ceiling, real cost-to-risk, and its own contribution to "
    "the real compliance headroom) rather than a uniform size regardless "
    "of these differences? Did every stop/target come with an explicit "
    "ATR-multiple and support/resistance-reliability justification "
    "specific to that instrument, rather than a flat multiple or a round "
    "number applied the same way across every position? Flag any "
    "position sized or stopped identically to its peers despite "
    "genuinely different volatility, conviction, or cost profiles.\n"
    "- Check SETUP READ and MULTI-TIMEFRAME use: setup reads are now "
    "computed on all four timeframes (D1, H4, H1, M5) — did the "
    "draft actually engage with each position's computed setup read "
    "(reversal, pullback, range-fade, breakout-watch, trend-following, "
    "trend-intact, grind-continuation, in-progress-move, busted-pattern-"
    "reversal, or candlestick-reversal-confirmed — trend-intact means a "
    "real 'trending_up'/'trending_down' regime with no more specific "
    "trigger active right now, grind-continuation means a real net "
    "directional move over the medium-term window reached via a noisy/"
    "choppy path rather than a clean trend, in-progress-move means the "
    "medium-term regime reads sideways but a shorter window shows a real "
    "push already underway, busted-pattern-reversal means a double-top/"
    "bottom's implied direction got contradicted by real subsequent "
    "price action (Bulkowski's own statistics say these often travel "
    "FURTHER than the original target); ALL are genuine directional "
    "evidence, treat a draft dismissing any of them as 'no real setup' "
    "as UNDER-weighing real evidence, not a correct caution) rather than ignore it, and does "
    "its stated thesis genuinely match that characterization rather "
    "than contradict it (e.g. calling something a trend-following entry "
    "when the M5 setup read says range_fade_candidate)? The D1/H4/H1 setup "
    "reads are context (feeding the thesis and the Long-term alignment "
    "read below) — flag a draft that anchors its actual entry/stop/target "
    "reasoning on a D1, H4 or H1 setup read instead of the M5 decision "
    "tier, not a draft that merely mentions the higher-timeframe read as "
    "supporting context. Did it use a real M5-vs-H1 CONFLICT, or an H4/H1 "
    "CONFLICTING trend read, as a reason for caution, or silently pick "
    "whichever timeframe agreed with its own thesis? Flag any stop/target "
    "that used an M5 level alone where a real M5 level that "
    "an independent H1/H4 level also lands on was sitting right there.\n"
    "- Check LONG-TERM ALIGNMENT use: each position also has a real "
    "'Long-term alignment' read comparing its M5 regime against the real "
    "daily backdrop — STRUCTURALLY BACKED, COUNTER-TREND SPIKE, "
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
    "- Check ENTRY / LEVEL CHOICE against the measured playbook: (a) for each with-the-trend trade, does the entry order "
    "match the PLAYBOOK line under that candidate (breakout stop / market / wait), or did the draft use a resting limit "
    "at a level without a printed reason (a limit at the nearest reaction level measured the worst entry)? (b) did every "
    "chosen support/resistance appear in that symbol's LEVEL MAP, and when it is not the nearest real candidate, did the "
    "draft name the nearer ones (especially an unbroken REACTION POINT) and say why they fail? Flag any claim that a level "
    "is 'the strongest' - that is not what the data shows. (c) is the stop beyond the level by the printed penetration "
    "buffer rather than on it, and is it a structure stop rather than a fixed ATR habit? (d) did a break/reclaim "
    "Pending Setup use a structured trigger when one fits?\n"
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
    "reversal, support/resistance-bounce, AND double-top/double-bottom "
    "chart-pattern trade win rates and average realized R-multiple — an "
    "actual ATR-based stop/target walked forward bar by bar to a genuine "
    "win/loss, not a bare average return N bars later, and — wherever "
    "this instrument's own live spread/commission/swap/broker-minimum-"
    "stop-distance data is available — already netted against those real "
    "execution costs and constraints, not an idealized zero-friction "
    "result (the stop/target sizing uses the real historical ATR at each "
    "past entry; the cost/swap figure netted in is today's live rate "
    "applied uniformly across that history, a disclosed proxy, not a "
    "historical record — don't flag this as a flaw, no historical "
    "spread/swap record exists to do better) — plus whether its own "
    "history shows real momentum "
    "persistence or mean-reversion, and whether its own low-volatility "
    "episodes have historically been followed by bigger or smaller "
    "moves). Note: there is no "
    "beta-vs-benchmark backtest here (this account can span forex, "
    "metals, and equity indices with no single natural benchmark) — "
    "don't fault the draft for lacking one.\n"
    "2b. A SEPARATE favorable-excursion magnitude figure (real median/"
    "mean R-distance a setup's price has continued traveling favorably, "
    "over a longer, UNCAPPED window, unrelated to the fixed-target win/"
    "loss verdict) may also be given alongside some of these backtests. "
    "This figure is measured across EVERY historical entry including "
    "losers and is floored at zero, so it comes back POSITIVE for nearly "
    "every setup on nearly every instrument REGARDLESS of whether that "
    "setup actually wins — treat a positive favorable-excursion figure "
    "as NEVER sufficient on its own to soften, offset, or excuse a poor "
    "win rate/negative avg-R finding on that SAME setup. A draft that "
    "cites a favorable-excursion figure as a reason to keep or upsize a "
    "position its own win-rate/avg-R backtest already CONTRADICTS is "
    "making exactly the same mistake as the real, documented incident "
    "above (the 14-17%-win-rate oversold-RSI-long thesis kept nearly "
    "verbatim despite two independent audits flagging it, which then "
    "lost in exactly the way the ignored data predicted) — flag this "
    "with the same weight, not as a softer or more excusable version of "
    "it just because a real number was cited.\n"
    "2c. Each instrument's own data section states whether its market is "
    "open right now. Flag as a hard, real mistake — not a judgment call — "
    "any NEW immediate allocation or NEW pending setup the draft proposes "
    "on an instrument whose data section says its market is CLOSED right "
    "now: it cannot fill. This does not apply to holding/managing an "
    "EXISTING position on a currently-closed instrument, which is a "
    "separate, legitimate decision governed by the holding-period/swap "
    "guidance elsewhere.\n"
    "3. Score each claim: SUPPORTED (the historical evidence agrees with "
    "the draft's implied logic), CONTRADICTED (the instrument's own "
    "history shows the opposite), or UNTESTABLE (not enough real "
    "historical episodes to judge). Read every M5 backtest line against the "
    "RANDOM-ENTRY BASELINE printed with it: use the verdict printed on the line "
    "(SUPPORTED / NO-INFORMATION / CONTRADICTED) — a raw win rate or a slightly negative "
    "average R with a NO-INFORMATION verdict is UNTESTABLE, never CONTRADICTED. "
    "Also check the draft against the POSITION HUNT block: flag every "
    "shortlisted candidate the draft omitted without a veto id or a specific, checkable fact from the printed data. "
    "Cite the actual numbers you're "
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
    "timeframe), not vague language an automated check every few minutes couldn't "
    "evaluate.\n"
    "\n"
    "Every non-CASH entry in the main allocation block above should also "
    "carry its own \"reason\" and, whenever \"pct\" is above 0, its own "
    "\"invalidation_condition\" — check both the same way you already "
    "check a Pending Setup's trigger_condition: is invalidation_condition "
    "genuinely specific and mechanically checkable (a real price level "
    "plus an indicator threshold and timeframe), not vague language like "
    "\"if it stops looking good\" that a separate automated process "
    "checking every few minutes couldn't actually evaluate? Is "
    "\"reason\" a real, current statement about THIS target rather than "
    "an empty string or an obviously copy-pasted placeholder? A missing "
    "or unusable invalidation_condition on a real position is a genuine "
    "gap worth flagging — it means nothing will catch a real disaster on "
    "that position before the next mega session.\n"
    "\n"
    "Produce a structured audit report — agreements, flaws, gaps, the "
    "backtest scorecard from above, and specific suggested improvements "
    "— not a rewritten competing allocation. Keep your response focused "
    "— under 550 words."
)


# One directive per AUDIT_MODELS slot (order matters — see build_audit_
# block's own `focus_directives` param), added 2026-09-03 alongside the
# 10->3 audit-model trim, direct user request: 3 models each reviewing
# every single one of AUDIT_INSTRUCTION's ~10 bullets meant stage 2 often
# had to wade through the SAME finding (a math error, a stale S/R level)
# flagged nearly verbatim 3 times, while a genuinely narrow finding one
# model caught got no more attention than one everybody already agreed
# on. Splitting the "judgment-quality" bullets three ways (not the
# universal ones — see the note appended to every group below) means
# each reviewer spends its own ~550-word budget on real depth in its own
# lane instead of shallow coverage of all ten. The split is BY BULLET,
# not by INSTRUMENT — every model still reviews the whole draft, just
# through a different lens, so a genuinely cross-cutting problem (an
# instrument with both a bad stop AND a bad cost calc) still gets caught
# by whichever group happens to own each half.
_AUDIT_FOCUS_UNIVERSAL_NOTE = (
    "\n\nRegardless of your assigned focus above, ALWAYS still run: the "
    "basic arithmetic check (\"Check its math\"), the full BACKTEST THE "
    "DRAFT'S UNDERLYING LOGIC section (SUPPORTED/CONTRADICTED scoring per "
    "claim, treating a CONTRADICTED score with the same weight as a math "
    "or compliance-headroom error), and the Pending Setups recomputation "
    "when that section is present — these are cheap to verify and "
    "specifically benefit from independent, redundant cross-checking "
    "across all three reviewers, unlike the more open-ended judgment "
    "bullets above, which is exactly why they stay universal rather than "
    "being split like the rest."
)

AUDIT_FOCUS_GROUPS: list[str] = [
    (
        "YOUR PRIMARY FOCUS THIS REVIEW (Group A — Compliance & Risk "
        "Structure): three independent models are auditing this same "
        "draft in parallel, each with a different primary focus, "
        "specifically so the same finding doesn't get flagged three "
        "times over while a narrower one goes unnoticed. Concentrate your "
        "written review on: the FTMO compliance math (TRAILING max-loss "
        "floor vs. static, Best Day Rule as a consistency constraint, "
        "real daily-loss headroom, zero-grace-period termination risk); "
        "correlation-under-stress, execution/liquidity risk, overnight/"
        "weekend financing cost, idle-cash deployment, and position-"
        "sizing-vs-volatility; and the POSITION-SIZE AND STOP/TARGET "
        "DEBATE bullet (does every position's size and stop/target "
        "actually reflect its own real inputs, not a uniform template). "
        "You do not need to also write out full analysis for the "
        "technical-setup-read or cost/diversification/fundamentals "
        "bullets (the other two reviewers own those) unless you spot "
        "something unambiguous and urgent there."
        + _AUDIT_FOCUS_UNIVERSAL_NOTE
    ),
    (
        "YOUR PRIMARY FOCUS THIS REVIEW (Group B — Technical & Timing "
        "Read): three independent models are auditing this same draft in "
        "parallel, each with a different primary focus, specifically so "
        "the same finding doesn't get flagged three times over while a "
        "narrower one goes unnoticed. Concentrate your written review on: "
        "HOLDING-HORIZON REALISM (including the numeric reward-distance-"
        "vs-ATR self-check the draft is required to show its work on); "
        "SETUP READ and MULTI-TIMEFRAME use (does the draft's stated "
        "trend/setup classification for each timeframe actually match "
        "the real computed data); and LONG-TERM ALIGNMENT use. You do "
        "not need to also write out full analysis for the compliance-"
        "math or cost/diversification/fundamentals bullets (the other "
        "two reviewers own those) unless you spot something unambiguous "
        "and urgent there."
        + _AUDIT_FOCUS_UNIVERSAL_NOTE
    ),
    (
        "YOUR PRIMARY FOCUS THIS REVIEW (Group C — Cost, Diversification "
        "& Fundamentals): three independent models are auditing this "
        "same draft in parallel, each with a different primary focus, "
        "specifically so the same finding doesn't get flagged three "
        "times over while a narrower one goes unnoticed. Concentrate "
        "your written review on: REAL TRADING COST awareness (reward:"
        "risk actually netted against real spread/commission/swap, not "
        "just the gross price move); ASSET-CATEGORY DIVERSIFICATION; and "
        "FUNDAMENTAL-BASED INCLUSION use. You do not need to also write "
        "out full analysis for the compliance-math or technical-setup-"
        "read bullets (the other two reviewers own those) unless you "
        "spot something unambiguous and urgent there."
        + _AUDIT_FOCUS_UNIVERSAL_NOTE
    ),
]


def build_ftmo_stage1_instruction() -> str:
    return f"{_INSTRUCTION_HEAD}\n\n{_STAGE1_DRAFT_INSTRUCTION}\n\n{_INSTRUCTION_TAIL}"


def build_ftmo_stage2_instruction(audit_available: bool) -> str:
    role = _ROLE_STAGE2_SYNTHESIZE if audit_available else _ROLE_STAGE2_SELF_REVIEW
    return f"{_INSTRUCTION_HEAD}\n\n{role}\n\n{_INSTRUCTION_TAIL}"


@dataclass
class IntradayBacktests:
    """The M15-bar analogue of the D1 backtest fields already on
    AssetAnalysis — real win-rate/avg-R/favorable-excursion evidence
    computed on the SAME timeframe (M15) and holding-window scale
    (config.TRADE_SIM_INTRADAY_MAX_HOLDING_BARS, ~4 real hours by
    default) this account actually holds a position for, instead of the
    D1 backtests' own 10-bar/~2-week one. See config.py's own
    TRADE_SIM_INTRADAY_MAX_HOLDING_BARS comment for the full incident
    this closes and the real, empirical basis (this account's own closed
    -trade history) behind the default. All fields None when intraday
    backtesting is disabled (config.INTRADAY_BACKTEST_ENABLED) or M15
    history came back too thin to backtest — same honest-degrade
    convention every other backtest field in this codebase already
    follows, never a fabricated result.

    Deliberately no double_bottom_backtest/double_top_backtest fields —
    unlike D1's own AssetAnalysis, which does carry them. Real cost
    measured live while shipping this: backtest_chart_pattern_reaction
    needed a genuinely large window (40,000 M15 bars) to reliably clear
    its own ≥5-independent-occurrence gate on real EURUSD M15 data, and
    at that scale ITS OWN compute cost (not the fetch — that stayed
    trivial) ran 16.65s at 15,000 bars and 29.33s at 25,000 bars, PER
    SYMBOL. Multiplied across a ~20-symbol Market Watch every mega
    session — and, once Clerk's own execution layer also consumes this
    data (a later, not-yet-built phase), potentially recomputed on every
    Clerk poll too unless that phase adds its own cache — this would add
    many real minutes for a figure of secondary relevance to TP/SL sizing
    next to RSI-reaction/S-R (see _compute_intraday_backtests' own
    docstring). Explicitly deferred, not silently dropped — a smarter,
    incremental/cached pattern-detection approach could revisit this
    later."""

    rsi_overbought_backtest: RSIReactionBacktest | None = None
    rsi_oversold_backtest: RSIReactionBacktest | None = None
    support_resistance_backtest: SupportResistanceBacktest | None = None
    # Random-entry yardsticks on the same bars/stop/target/cost (analysis/edge_stats.py): a setup's
    # avg R only means something relative to these, never relative to zero.
    null_baseline_buy: EdgeBaseline | None = None
    null_baseline_sell: EdgeBaseline | None = None


_EMPTY_INTRADAY_BACKTESTS = IntradayBacktests()


def intraday_edge_verdicts(intraday: IntradayBacktests, side: str) -> dict[str, EdgeVerdict]:
    """Significance-aware verdicts (analysis.edge_stats) for the M5 setups that bet on `side`
    ("buy"/"sell"): key "rsi" = the RSI-reaction setup, key "sr" = the support (buy) / resistance (sell)
    bounce. A setup with no backtest is simply absent; a setup whose baseline is unavailable is
    NO_INFORMATION - never SUPPORTED by default."""
    baseline = intraday.null_baseline_buy if side == "buy" else intraday.null_baseline_sell
    out: dict[str, EdgeVerdict] = {}
    rsi = intraday.rsi_oversold_backtest if side == "buy" else intraday.rsi_overbought_backtest
    if rsi is not None:
        out["rsi"] = classify_edge(rsi.avg_r_multiple, rsi.trades, baseline)
    sr = intraday.support_resistance_backtest
    if sr is not None:
        if side == "buy":
            out["sr"] = classify_edge(sr.support_avg_r_multiple, sr.support_tests, baseline)
        else:
            out["sr"] = classify_edge(sr.resistance_avg_r_multiple, sr.resistance_tests, baseline)
    return out


@dataclass
class FtmoAssetAnalysis:
    """CURRENT TIER MODEL (2026-09-24): M5 (the m5_* fields) is the ONLY decision timeframe — entry,
    stop, target, trigger, invalidation and sizing all come from it. D1 (`base.stats`/d1_*), H4 (h4_*)
    and H1 (h1_*) are regime CONTEXT only. Older paragraphs below that call H4/H1 the "PRIMARY basis"
    describe the pre-2026-09-24 design and are kept for their incident history.

    Wraps a PMEX-shape AssetAnalysis (`base`) with the two extra
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
    d1_structure: ChartStructureSnapshot = field(
        default_factory=lambda: ChartStructureSnapshot(fibonacci=None, sr_levels=None, trendlines=None, patterns=[])
    )
    # Real, intraday-holding-window-calibrated backtest evidence (M15
    # bars, config.TRADE_SIM_INTRADAY_MAX_HOLDING_BARS-scale holding cap)
    # — see IntradayBacktests' and _compute_intraday_backtests' own
    # docstrings for the full incident this closes (the existing backtest
    # fields on `base` are computed on DAILY bars, weeks-to-months
    # holding/excursion windows, mismatched with this account's real
    # intraday-only holding constraint). Defaulted (unlike h4_stats/
    # h1_stats) purely so existing test fixtures built before this field
    # existed don't need updating just to keep constructing this
    # dataclass — same precedent as mn1_stats/d1_structure/mn1_structure.
    intraday_backtests: IntradayBacktests = field(default_factory=IntradayBacktests)
    # Real trend-vs-reversal early warning — added 2026-09-20, direct
    # user challenge. Computed once per timeframe (_compute_divergence_
    # for_history) and reused wherever classify_setups is called for
    # that same timeframe, rather than recomputed at each call site.
    # Defaulted None so existing test fixtures built before these fields
    # existed don't need updating — same precedent as mn1_stats/
    # d1_structure/mn1_structure above.
    h4_divergence: DivergenceSignal | None = None
    h1_divergence: DivergenceSignal | None = None
    d1_divergence: DivergenceSignal | None = None
    # Real, per-level "has THIS exact zone specifically been tested N
    # times and held M?" evidence (Phase 4, added 2026-09-21; moved from H1 to M5 on 2026-09-24
    # when M5 became the only decision timeframe) — keyed by SRLevel.price, backtested on the
    # M5 bars with the same intraday holding cap / stop / target the M5 backtests use.
    # Distinct from support_resistance_backtest on `base` (a POOLED stat across every
    # rolling-window S/R touch on D1) — this backtests each individual m5_structure.sr_levels
    # band on its own. Empty dict (never None) when compute_level_reliability=False on the
    # call that built this analysis (see analyze_ftmo_asset_live's own docstring for the real
    # per-symbol-cost reason it's opt-in) or when no level cleared its own min-tests floor.
    m5_level_reliability: dict[float, LevelReliabilityBacktest] = field(default_factory=dict)
    # DECISION-TIER read (2026-09-24, M5 only): entry timing, trigger, the stop-distance ATR,
    # short-range structure (S/R, Fibonacci, trendlines, patterns -> where stops/targets sit) and
    # the reachability ATR. D1/H4/H1 (the stats/structure/divergence fields above) are regime
    # CONTEXT only. (M15 and Monthly were dropped: D1 covers Monthly's role, M5 carries the
    # decisions.) Same defaulted-field precedent as every other late addition here.
    m5_stats: TechnicalStats = field(default_factory=lambda: TechnicalStats(*([None] * 17)))
    m5_structure: ChartStructureSnapshot = field(
        default_factory=lambda: ChartStructureSnapshot(fibonacci=None, sr_levels=None, trendlines=None, patterns=[])
    )
    m5_divergence: DivergenceSignal | None = None
    # Median M5 ATR% over the last ~2 days of bars — the baseline Clerk's volatility size scalar
    # compares the CURRENT M5 ATR% against (only ever downsizes).
    m5_atr_pct_median: float | None = None
    intraday_levels: IntradayLevels | None = None
    # The last config.PLAYBOOK_RANGE_BARS COMPLETED M5 bars' extremes + tick-volume ratio (analysis.playbook.range_read):
    # the breakout trigger and structure stop of the with-the-trend playbook.
    m5_range: RangeRead | None = None
    # The last config.TRIGGER_RECENT_BARS COMPLETED M5 bars (High/Low/Close, oldest first) - what analysis.triggers evaluates
    # a structured Pending Setup trigger on (the forming bar is excluded, no look-ahead).
    m5_recent: pd.DataFrame | None = None
    # analysis.level_map.LevelMap per side ("buy": supports below the price, "sell": resistances above): every plausible
    # reversal level near the price, nearest first, with measured fill odds (the SOLUSD fix, plan W4).
    m5_levels: dict = field(default_factory=dict)
    # {"h4": "up"/"down"/None, "d1": ...} from CLOSED higher-timeframe buckets only (analysis.htf_flags) - the flags the
    # alignment numbers were measured on; the Position Hunter prefers them to the still-forming-bar TechnicalStats trend.
    htf_closed: dict = field(default_factory=dict)


# ~6 years of daily bars — enough real history for the backtest functions'
# own episode-count thresholds to have real statistical power, without an
# excessive per-symbol fetch. Confirmed live against the real FTMO
# terminal: EURUSD/GBPUSD/MSFT go back to 2018, NVDA to 2020, BTCUSD to
# 2021 — MT5's own native feed comfortably covers this depth for the
# instruments this account actually trades.
_D1_BACKTEST_BARS = 1500



def _build_bare_base_analysis(a: MarketAsset, now_utc: datetime | None = None) -> AssetAnalysis:
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
    rest of the pool instead of a one-off exception.

    `now_utc`, if the caller already has a single snapshot for this run
    (analyze_ftmo_assets does, so every symbol in one run judges "is it
    open right now" against the same instant), is used to compute
    market_open via data.mt5_source.is_symbol_tradable_now; None (the
    default) takes its own fresh snapshot."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    return AssetAnalysis(
        symbol=a.symbol,
        description=a.description,
        bid=a.bid,
        ask=a.ask,
        display_name=None,
        contract_spec=get_contract_spec(a.symbol),
        market_open=is_symbol_tradable_now(a.symbol, now_utc),
    )


def _fetch_ftmo_headlines(symbol: str, description: str) -> list[str]:
    """Real, live headlines for this FTMO asset's own context line —
    added 2026-09-20, direct user request to give Mega Session real news
    for the first time (previously always `headlines=[]`, a long-
    standing, disclosed gap — see _enrich_with_native_d1's own original
    docstring: FTMO deliberately never depended on Yahoo for anything,
    after a real 24+-hour hang incident on a cold-started process
    talking to Yahoo with no timeout protecting it at the time). That
    original hang risk no longer applies the same way: every real fetch
    now goes through data.symbol_news's own shared, already-timeout-
    protected primitives (utils.run_with_timeout, see data.news_source's
    own _NEWS_TIMEOUT_SECONDS), and — the actual point of this change —
    the SAME shared, 20-minute cache Clerk/Researcher/the webapp already
    read from, so a mega-session run occurring soon after either of
    those has already fetched a symbol's news costs nothing extra, not
    a fresh cold-started network round-trip.

    Includes Researcher's per-symbol AND category-specialty layers —
    real gap closed 2026-09-20 on direct user challenge ("news quantity
    and sources are very less... where those all sources gone?"): the
    first version of this function only carried the per-symbol Yahoo/
    Google layer, silently missing the category-specialty RSS (FXStreet/
    CoinDesk/Investing.com) layer Researcher's own daily report already
    includes. Mega Session runs about as often as Researcher (once/day),
    so there's no reason to withhold that fuller picture the way Clerk
    deliberately still does (Clerk's own _fetch_clerk_news_block stays
    per-symbol-only on purpose — its much higher poll frequency is
    exactly why it was kept lightweight from the start; that reasoning
    hasn't changed). Uses config.RESEARCHER_HEADLINES_PER_SYMBOL (not
    the lighter NEWS_HEADLINES_PER_ASSET) to match Researcher's own
    depth, since matching its richness is the whole point of this
    change. Category items are tagged ("[category] ...") so they stay
    identifiable once flattened into this plain list[str] field.

    Deliberately does NOT also include the macro/geopolitical layer
    (index/oil/gold proxies + CNBC) here, unlike Researcher's own per-
    symbol report — Researcher builds one SEPARATE model prompt per
    symbol, so repeating the same macro block in each is free; Mega
    Session instead builds ONE combined prompt covering every symbol
    (format_enriched_asset_context loops all assets into it), so doing
    the same here would duplicate the identical ~15-20 macro lines once
    per symbol (~20x over) for no benefit. Macro news is added exactly
    ONCE instead, at the real top level — see build_ftmo_summary's own
    call to _format_ftmo_macro_news_context.

    [] (never fabricated) on any failure — this function must never be
    able to withhold the real technical/backtest data the rest of this
    pipeline computes, only ever add or omit "news:" lines on top of it,
    matching the exact same "an honest, disclosed gap, not a reason to
    withhold what IS available" posture this file already documents for
    every other missing-data field.

    `description` also now feeds get_symbol_news_block's own
    "intelligent search" fallback (added 2026-09-20) — a symbol whose
    category matches no known ticker convention (e.g. Agriculture/Cash
    CFD commodities) still gets real news via a Google search built
    from this same real MT5 description, instead of nothing."""
    try:
        category = get_symbol_category(symbol)
        per_symbol = symbol_news.get_symbol_news_block(
            symbol, category, limit=config.RESEARCHER_HEADLINES_PER_SYMBOL, description=description
        )
        now = datetime.now(timezone.utc)
        headlines = [symbol_news.headline_with_age(item, now) for item in per_symbol]

        category_items = symbol_news.get_category_news_items(category, limit=config.RESEARCHER_HEADLINES_PER_SYMBOL)
        headlines += [f"[{category}] {symbol_news.headline_with_age(item, now)}" for item in category_items]

        return headlines
    except Exception:
        logger.warning("Could not fetch real headlines for %s this run (degrading to none).", symbol, exc_info=True)
        return []


def _format_ftmo_macro_news_context() -> str:
    """The market-wide/geopolitical news layer (index/oil/gold proxies +
    CNBC's own top-news feed) — added ONCE to the overall mega-session
    prompt (build_ftmo_summary), not per-symbol; see _fetch_ftmo_
    headlines's own docstring for exactly why this one stays out of the
    per-asset `headlines` field. [] (never fabricated) on any failure,
    same posture as _fetch_ftmo_headlines."""
    try:
        items = symbol_news.get_macro_news_items()
    except Exception:
        logger.warning("Could not fetch macro/geopolitical news this run (degrading to none).", exc_info=True)
        return "Real market-wide/geopolitical headlines: (unavailable this run)"
    if not items:
        return "Real market-wide/geopolitical headlines: (none found this run)"
    lines = ["Real market-wide/geopolitical headlines (shared context, not symbol-specific):"]
    lines += [f"- {item['title']}" for item in items]
    return "\n".join(lines)


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
            trade_cost.category, contract_spec.risk_per_price_unit, current_ask
        )
        if commission_pct is not None:
            round_trip_cost_pct += commission_pct
    return dict(
        round_trip_cost_pct=round_trip_cost_pct,
        long_swap_pct_per_day=trade_cost.swap_long_pct_per_day or 0.0,
        short_swap_pct_per_day=trade_cost.swap_short_pct_per_day or 0.0,
        min_stop_distance_pct=trade_cost.min_stop_distance_pct,
    )


def _intraday_backtest_kwargs(
    trade_cost: TradeCost | None, contract_spec: ContractSpec | None, ask: float | None
) -> dict:
    """The M5 simulation parameters shared by the pooled M5 backtests and the per-level reliability
    backtests: real cost/broker-stop realism, swap zeroed (the backtest's swap math assumes 1 bar =
    1 calendar day, ~288x wrong on M5, and a ~4h cap never crosses a swap point anyway), holding
    capped at config.TRADE_SIM_INTRADAY_MAX_HOLDING_BARS, and the simulated stop/target set at
    config.M5_ATR_STOP_MULTIPLE x / 2x that multiple of M5 ATR so the simulated trade matches the
    stop distance Clerk actually enforces (2:1 reward:risk)."""
    exec_kwargs = _real_backtest_execution_kwargs(trade_cost, contract_spec, ask)
    exec_kwargs["long_swap_pct_per_day"] = 0.0
    exec_kwargs["short_swap_pct_per_day"] = 0.0
    exec_kwargs["max_holding_bars"] = config.TRADE_SIM_INTRADAY_MAX_HOLDING_BARS
    exec_kwargs["excursion_horizon_bars"] = config.TRADE_SIM_INTRADAY_EXCURSION_HORIZON_BARS
    exec_kwargs["stop_atr_multiple"] = config.M5_ATR_STOP_MULTIPLE
    exec_kwargs["target_atr_multiple"] = 2 * config.M5_ATR_STOP_MULTIPLE
    return exec_kwargs


def _compute_intraday_backtests(
    symbol: str,
    contract_spec: ContractSpec | None,
    ask: float | None,
    trade_cost: TradeCost | None,
    m5_history: pd.DataFrame | None = None,
) -> IntradayBacktests:
    """The M5-bar counterpart to _enrich_with_native_d1's own D1 backtest calls (re-based from M15 on
    2026-09-24 when the M15 tier was dropped) — backtest_rsi_reaction/backtest_support_resistance_
    reaction completely unmodified (purely bar-count-indexed), fed M5 bars and intraday-scaled
    parameters: holding capped at config.TRADE_SIM_INTRADAY_MAX_HOLDING_BARS (48 M5 bars ~ 4h), and the
    simulated stop/target set at config.M5_ATR_STOP_MULTIPLE x / 2x that multiple of M5 ATR so the
    simulated trade matches the stop distance Clerk actually enforces (2:1 reward:risk).
    backtest_chart_pattern_reaction is deliberately NOT called (compute cost, see IntradayBacktests).

    `proximity_pct` is ATR-relative (config.M5_SR_PROXIMITY_ATR_MULTIPLE x the instrument's own median
    M5 ATR%, clamped): measured on real bars, any flat value degenerates on some instrument (0.5% gave
    EURUSD 23/5 tests but MSFT 525/531).

    Swap is zeroed (the backtest's swap math assumes 1 bar = 1 calendar day, ~288x wrong on M5, and a
    ~4h cap never crosses a swap point anyway); real spread/commission and the broker's minimum stop
    distance still apply from the same TradeCost the D1 backtests use.

    Returns _EMPTY_INTRADAY_BACKTESTS (never raises, never fabricates) if intraday backtesting is
    disabled, M5 history comes back empty, or `trade_cost` is unavailable. `m5_history` (the caller's
    already-fetched M5_BACKTEST_BARS-deep bars, see _compute_intraday_reads) skips this function's own
    fetch."""
    if not config.INTRADAY_BACKTEST_ENABLED or trade_cost is None:
        return _EMPTY_INTRADAY_BACKTESTS
    if m5_history is None:
        m5_history = fetch_mt5_price_history(symbol, "M5", count=config.M5_BACKTEST_BARS)
    if m5_history.empty:
        return _EMPTY_INTRADAY_BACKTESTS
    exec_kwargs = _intraday_backtest_kwargs(trade_cost, contract_spec, ask)
    median_pct = median_atr_pct(m5_history)
    proximity_pct = (
        min(config.M5_SR_PROXIMITY_MAX_PCT, max(config.M5_SR_PROXIMITY_MIN_PCT, config.M5_SR_PROXIMITY_ATR_MULTIPLE * median_pct))
        if median_pct
        else config.M5_SR_PROXIMITY_MAX_PCT
    )
    rsi_overbought_bt, rsi_oversold_bt = backtest_rsi_reaction(m5_history, **exec_kwargs)
    support_resistance_bt = backtest_support_resistance_reaction(m5_history, proximity_pct=proximity_pct, **exec_kwargs)
    baseline_kwargs = dict(
        stop_atr_multiple=exec_kwargs["stop_atr_multiple"],
        target_atr_multiple=exec_kwargs["target_atr_multiple"],
        max_holding_bars=exec_kwargs["max_holding_bars"],
        min_stop_distance_pct=exec_kwargs.get("min_stop_distance_pct", 0.0),
        round_trip_cost_pct=exec_kwargs.get("round_trip_cost_pct", 0.0),
    )
    return IntradayBacktests(
        rsi_overbought_backtest=rsi_overbought_bt,
        rsi_oversold_backtest=rsi_oversold_bt,
        support_resistance_backtest=support_resistance_bt,
        null_baseline_buy=compute_null_baseline(m5_history, "buy", **baseline_kwargs),
        null_baseline_sell=compute_null_baseline(m5_history, "sell", **baseline_kwargs),
    )


def _verdict_suffix(verdict: EdgeVerdict | None) -> str:
    """' -> SUPPORTED (z +2.1 vs random +0.01R)' style tail for a backtest line; '' with no verdict."""
    if verdict is None:
        return ""
    if verdict.z is None or verdict.baseline_r is None:
        return f" -> {verdict.label} (no random-entry baseline available)"
    return f" -> {verdict.label} (z {verdict.z:+.1f} vs random {verdict.baseline_r:+.2f}R, n={verdict.trades})"


def _format_intraday_backtests(a: "FtmoAssetAnalysis") -> str:
    """The M5/intraday-holding-window counterpart to ai.portfolio_
    suggest.format_backtests' own D1 backtest lines (embedded just above
    this, for the same symbol, via format_enriched_asset_context) —
    deliberately a SEPARATE, FTMO-local block rather than a change to
    that shared function, since format_backtests is also used by PMEX/
    PSX's own D1-only prompts, which have no M5 data at all.

    Every line states the real bars-to-hours conversion explicitly
    (config.TRADE_SIM_INTRADAY_MAX_HOLDING_BARS M5 bars = N real
    minutes of M5 DATA, not a calendar-time guarantee — this account's
    instruments don't all trade continuously, so N real hours of M5
    bars can span more than N hours of wall-clock time for anything that
    closes overnight/weekends) — the same real-time-horizon labeling the
    existing D1 lines are missing today (they state '(max N bars held)'
    with no calendar-time conversion at all). A short header sentence
    makes the real scale difference between the two blocks impossible to
    miss: this evidence is calibrated to this account's own real holding
    window; the D1 block just above spans weeks to months instead (see
    config.TRADE_SIM_INTRADAY_MAX_HOLDING_BARS's own comment for the
    full incident this closes).

    Deliberately does NOT yet state that this evidence should be treated
    as PRIMARY over the D1 block — that's a real judgment call about how
    a model should weigh two different evidence sources, held back
    pending a live comparison of real mega-session output with/without
    it (this is a prompt-content change, not deterministic code; see this
    account's own shadow-mode precedent in ai.clerk_execution for why
    that verification step matters before flipping a new signal to
    primary)."""
    bt = a.intraday_backtests
    max_bars = config.TRADE_SIM_INTRADAY_MAX_HOLDING_BARS
    max_minutes = max_bars * 5
    max_hours = max_minutes / 60
    lines = [
        f"Same-session (M5) backtest evidence — real trade simulation on this instrument's "
        f"own M5 bars, holding capped at {max_bars} M5 bars (~{max_hours:.1f}h of M5 data, "
        "not a calendar-time guarantee — actual elapsed wall-clock time depends on this "
        "instrument's own trading-hours pattern). Calibrated to this account's own real "
        "intraday holding window, unlike the D1 backtest block above (which spans weeks to "
        "months — background context only, see HOLDING HORIZON). A genuinely strong, "
        "well-sampled win rate below can justify a smaller, more modest target than the "
        "usual 2:1 aim — a real, deterministic backstop evaluates this specifically, so a "
        "well-evidenced small win is worth proposing here, not discarding. HOW TO READ THESE "
        "LINES: a win rate near 27% / avg slightly below 0R is what RANDOM entries with the same "
        "stop/target/cost score - that is the null result, not evidence against a trade. Each "
        "line therefore carries a verdict against that random-entry baseline: SUPPORTED (beat it "
        "by a real margin), NO-INFORMATION (indistinguishable from random - NOT a reason to "
        "exclude or shrink a trade whose thesis rests on structure/trend/catalyst) or "
        "CONTRADICTED (clearly worse than random on a real sample - a genuine warning):"
    ]
    for baseline_side, baseline_name in ((bt.null_baseline_buy, "long"), (bt.null_baseline_sell, "short")):
        if baseline_side is not None:
            win = f"{baseline_side.win_rate_pct:.0f}%" if baseline_side.win_rate_pct is not None else "n/a"
            lines.append(
                f"  random-entry baseline ({baseline_name}, {baseline_side.trades} sampled M5 entries, same "
                f"stop/target/cost): win rate {win}, avg {baseline_side.mean_r:+.2f}R"
            )
    if bt.rsi_overbought_backtest is None and bt.rsi_oversold_backtest is None:
        lines.append(
            "  intraday RSI reaction: not enough real M5 history/episodes for this "
            "instrument to compute — treat as unavailable, not as a neutral result."
        )
    else:
        for condition_bt, condition, side in (
            (bt.rsi_overbought_backtest, "overbought", "short"),
            (bt.rsi_oversold_backtest, "oversold", "long"),
        ):
            if condition_bt is None:
                lines.append(f"  intraday {condition} RSI reaction: not enough real M5 episodes to compute")
                continue
            rsi_verdict = intraday_edge_verdicts(bt, "sell" if side == "short" else "buy").get("rsi")
            win_rate = (
                f"{condition_bt.win_rate_pct:.0f}%" if condition_bt.win_rate_pct is not None else "n/a"
            )
            lines.append(
                f"  intraday {condition} RSI reaction ({side} simulated): {condition_bt.trades} real M5 "
                f"episodes -> {condition_bt.wins}W/{condition_bt.losses}L/{condition_bt.timeouts} timed "
                f"out, win rate {win_rate}, avg {condition_bt.avg_r_multiple:+.2f}R"
                f"{_real_execution_note(condition_bt.round_trip_cost_pct, condition_bt.swap_pct_per_day_used, condition_bt.min_stop_distance_pct)}"
                f"{_verdict_suffix(rsi_verdict)}"
            )
    sr = bt.support_resistance_backtest
    if sr is None:
        lines.append(
            "  intraday support/resistance reliability: not enough real M5 tests of these "
            "levels to compute"
        )
    else:
        support_rate = f"{sr.support_win_rate_pct:.0f}%" if sr.support_win_rate_pct is not None else "n/a"
        resistance_rate = (
            f"{sr.resistance_win_rate_pct:.0f}%" if sr.resistance_win_rate_pct is not None else "n/a"
        )
        lines.append(
            f"  intraday support/resistance reliability: buying off support -> "
            f"{sr.support_wins}W/{sr.support_losses}L/{sr.support_timeouts} timed out across "
            f"{sr.support_tests} real M5 tests, win rate {support_rate}, avg "
            f"{sr.support_avg_r_multiple:+.2f}R; shorting off resistance -> "
            f"{sr.resistance_wins}W/{sr.resistance_losses}L/{sr.resistance_timeouts} timed out "
            f"across {sr.resistance_tests} real M5 tests, win rate {resistance_rate}, avg "
            f"{sr.resistance_avg_r_multiple:+.2f}R"
        )
        buy_verdict = intraday_edge_verdicts(bt, "buy").get("sr")
        sell_verdict = intraday_edge_verdicts(bt, "sell").get("sr")
        lines.append(
            f"  ... verdict vs random baseline: support bounce (long){_verdict_suffix(buy_verdict) or ' n/a'}; "
            f"resistance rejection (short){_verdict_suffix(sell_verdict) or ' n/a'}"
        )
    return "\n".join(lines)


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
    ever calls it with an already-enriched base again. `headlines` was
    long `[]` here (no MT5-native news feed) — as of 2026-09-20 it's real
    news via _fetch_ftmo_headlines/data.symbol_news, the same shared
    fetch/cache Clerk and Researcher use (see that function's own
    docstring for why the original Yahoo-hang concern no longer applies
    the same way).

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
    double_bottom_bt, double_top_bt = backtest_chart_pattern_reaction(d1_history, **exec_kwargs)
    return AssetAnalysis(
        symbol=base.symbol,
        description=base.description,
        bid=base.bid,
        ask=base.ask,
        display_name=base.description,
        data_source="mt5",
        market_open=base.market_open,
        prices=prices,
        stats=compute_technical_stats(prices, history=d1_history),
        headlines=_fetch_ftmo_headlines(base.symbol, base.description),
        contract_spec=base.contract_spec,
        rsi_overbought_backtest=rsi_overbought_bt,
        rsi_oversold_backtest=rsi_oversold_bt,
        momentum_persistence_backtest=backtest_momentum_persistence(prices),
        volatility_regime_backtest=backtest_volatility_regime(prices),
        support_resistance_backtest=backtest_support_resistance_reaction(d1_history, **exec_kwargs),
        double_bottom_backtest=double_bottom_bt,
        double_top_backtest=double_top_bt,
    )


def _compute_divergence_for_history(history: pd.DataFrame) -> DivergenceSignal | None:
    """Thin composition helper — added 2026-09-20, direct user challenge
    ("differential between trend and reversal") — so each of the 4
    timeframes' own divergence read is one line at each analyze_ftmo_
    asset_live/analyze_ftmo_assets call site, not a repeated 2-line
    find_swing_points+detect_rsi_divergence pair. Deliberately a SEPARATE
    fractal scan from compute_chart_structure's own internal one (which
    doesn't expose its swing_points on ChartStructureSnapshot) — a real,
    disclosed small redundancy, not the multiplied-many-times-over one
    compute_chart_structure's own docstring already fixed elsewhere."""
    if history.empty or "Close" not in history.columns:
        return None
    swing_highs, swing_lows = find_swing_points(history)
    return detect_rsi_divergence(history["Close"], swing_highs=swing_highs, swing_lows=swing_lows)


def _backtest_all_sr_levels(
    h1_history: pd.DataFrame,
    sr_levels: SRLevelsResult | None,
    trade_cost: TradeCost | None,
    contract_spec: ContractSpec | None,
    ask: float | None,
    max_holding_bars: int | None = None,
    *,
    intraday: bool = False,
) -> dict[float, LevelReliabilityBacktest]:
    """Phase 4 orchestrator (added 2026-09-21) — runs analysis.backtest.
    backtest_level_reliability once per REAL structure.sr_levels band. `intraday=True` (the M5
    decision tier) simulates each touch exactly like the M5 backtests do: the intraday holding
    cap and excursion horizon, the M5-ATR stop and 2x target, swap zeroed (see
    _intraday_backtest_kwargs).
    (support bets long via long_swap_pct_per_day, resistance bets short
    via short_swap_pct_per_day — same directional convention as every
    other backtest in this file), keyed by SRLevel.price so a caller can
    look up "what's this exact level's own real react rate" the same way
    it already looks up format_chart_structure's own band text.

    Reuses _real_backtest_execution_kwargs verbatim for cost/broker-stop
    realism (same real spread/commission/min-stop-distance every other
    D1/M15 backtest in this file already applies) — only the long/short
    swap split needs adapting, since backtest_level_reliability takes a
    single `swap_pct_per_day` (one simulated side per call) rather than
    the pooled function's own both-sides-at-once shape.

    Skips any level still missing a real band (`low`/`high` is None —
    never happens for h1_structure post-Phase-1, but a defensive no-op
    for any pre-Phase-1-shaped SRLevel a caller might still construct
    directly, e.g. in tests) and any level that doesn't independently
    clear backtest_level_reliability's own min_tests floor (that
    function returns None rather than a fabricated weak result — this
    orchestrator just doesn't add those to the returned dict).

    Empty dict for h1_history.empty or sr_levels is None — never raises,
    matches this file's own "degrade, don't crash" convention throughout."""
    if sr_levels is None or h1_history.empty:
        return {}
    exec_kwargs = (
        _intraday_backtest_kwargs(trade_cost, contract_spec, ask)
        if intraday
        else _real_backtest_execution_kwargs(trade_cost, contract_spec, ask)
    )
    common_kwargs = {
        k: v for k, v in exec_kwargs.items() if k not in ("long_swap_pct_per_day", "short_swap_pct_per_day")
    }
    if max_holding_bars is not None:
        # Intraday callers pass the holding cap (~4h of M5 bars) instead of the D1-scale
        # default of 10 bars.
        common_kwargs["max_holding_bars"] = max_holding_bars
    results: dict[float, LevelReliabilityBacktest] = {}
    for side, levels, swap_key in (
        ("support", sr_levels.support_levels, "long_swap_pct_per_day"),
        ("resistance", sr_levels.resistance_levels, "short_swap_pct_per_day"),
    ):
        swap_pct_per_day = exec_kwargs.get(swap_key, 0.0)
        for level in levels:
            if level.low is None or level.high is None:
                continue
            result = backtest_level_reliability(
                h1_history, level.price, level.low, level.high, side,
                swap_pct_per_day=swap_pct_per_day, **common_kwargs,
            )
            if result is not None:
                results[level.price] = result
    return results


def median_atr_pct(history: pd.DataFrame, window: int = 14, lookback: int = 576) -> float | None:
    """Median rolling-ATR% (ATR / close * 100) over the last `lookback` bars
    — the "normal" volatility level for this instrument on this timeframe.
    None (never fabricated) without High/Low/Close or enough bars."""
    if history is None or history.empty or not {"High", "Low", "Close"}.issubset(history.columns):
        return None
    prev_close = history["Close"].shift(1)
    true_range = pd.concat(
        [history["High"] - history["Low"], (history["High"] - prev_close).abs(), (history["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_pct = (true_range.rolling(window).mean() / history["Close"] * 100).dropna().tail(lookback)
    return float(atr_pct.median()) if len(atr_pct) >= 20 else None


def _build_level_maps(m5_history, m5_stats, m5_structure, intraday_levels, m5_range, ask) -> dict:
    """The Level Map (analysis.level_map) for both sides; {} when disabled or unreadable - never blocks an analysis."""
    if not config.LEVEL_MAP_ENABLED or m5_stats is None or not m5_stats.atr:
        return {}
    price = ask or float(m5_history["Close"].iloc[-1])
    try:
        return {
            side: build_level_map(
                side, price, m5_stats.atr, bars=m5_history, sr_levels=m5_structure.sr_levels,
                fibonacci=m5_structure.fibonacci, intraday_levels=intraday_levels, range_read=m5_range,
            )
            for side in ("buy", "sell")
        }
    except Exception:  # noqa: BLE001
        logger.warning("Level Map could not be built.", exc_info=True)
        return {}


def _recent_completed_bars(history: pd.DataFrame | None) -> pd.DataFrame | None:
    """The last config.TRIGGER_RECENT_BARS completed M5 bars (High/Low/Close), dropping the still-forming last row."""
    if history is None or not {"High", "Low", "Close"}.issubset(history.columns) or len(history) < 2:
        return None
    return history[["High", "Low", "Close"]].iloc[:-1].tail(config.TRIGGER_RECENT_BARS).copy()


def _compute_intraday_reads(
    symbol: str,
    d1_history: pd.DataFrame,
    trade_cost: TradeCost | None,
    contract_spec: ContractSpec | None,
    ask: float | None,
    compute_level_reliability: bool = True,
    lean: bool = False,
) -> dict:
    """`lean=True` (the Execution Clerk, for a symbol it only has to DEFEND): the M5 reads only - no deep M5 history fetch, no
    pooled intraday backtests, no level reliability, no closed higher-timeframe flags (only the Position Hunter reads those).

    The decision-tier M5 reads as FtmoAssetAnalysis keyword arguments — ONE helper shared by the batch
    Mega Session pipeline (analyze_ftmo_assets) and the on-demand live path (analyze_ftmo_asset_live,
    which is also what Clerk's technical context is built from), so Clerk automatically sees exactly the
    same M5 block Claude did. M5 is the ONLY decision timeframe (2026-09-24), so ONE M5 fetch
    (config.M5_BACKTEST_BARS deep when intraday backtests are on) feeds everything here: the stats /
    structure / divergence / session levels on the latest config.INTRADAY_M5_BARS bars, the pooled
    intraday backtests on the full depth, and (when `compute_level_reliability`) the per-level
    reliability of each M5 S/R band on the full depth. M5 structure uses the ATR-relative intraday
    profile (analysis/timeframe_profiles.py) instead of the flat D1/H1-tuned thresholds.
    `intraday_levels` (prior-day H/L, session VWAP, ADR used) needs both M5 and D1 bars and is None
    without them."""
    depth = max(config.INTRADAY_M5_BARS, config.M5_BACKTEST_BARS) if (config.INTRADAY_BACKTEST_ENABLED and not lean) else config.INTRADAY_M5_BARS
    full_history = fetch_mt5_price_history(symbol, "M5", count=depth)
    m5_history = full_history.tail(config.INTRADAY_M5_BARS)
    m5_structure = compute_chart_structure(m5_history, profile=M5_PROFILE)
    m5_stats = compute_technical_stats(m5_history["Close"], history=m5_history, periods_per_year=_M5_PERIODS_PER_YEAR)
    intraday_levels = compute_intraday_levels(m5_history, d1_history)
    m5_range = range_read(full_history)
    if lean:
        return {
            "m5_stats": m5_stats,
            "m5_structure": m5_structure,
            "m5_levels": _build_level_maps(m5_history, m5_stats, m5_structure, intraday_levels, m5_range, ask),
            "m5_divergence": _compute_divergence_for_history(m5_history),
            "m5_atr_pct_median": median_atr_pct(m5_history),
            "intraday_levels": intraday_levels,
            "m5_range": m5_range,
            "m5_recent": _recent_completed_bars(full_history),
        }
    return {
        "m5_stats": m5_stats,
        "m5_structure": m5_structure,
        "htf_closed": closed_htf_directions(full_history["Close"]) if config.HUNTER_CLOSED_HTF_FLAGS else {},
        "m5_levels": _build_level_maps(m5_history, m5_stats, m5_structure, intraday_levels, m5_range, ask),
        "m5_divergence": _compute_divergence_for_history(m5_history),
        "m5_atr_pct_median": median_atr_pct(m5_history),
        "intraday_levels": intraday_levels,
        "m5_range": m5_range,
        "m5_recent": _recent_completed_bars(full_history),
        "intraday_backtests": _compute_intraday_backtests(
            symbol, contract_spec, ask, trade_cost, m5_history=full_history
        ),
        "m5_level_reliability": (
            _backtest_all_sr_levels(
                full_history, m5_structure.sr_levels, trade_cost, contract_spec, ask, intraday=True
            )
            if compute_level_reliability
            else {}
        ),
    }


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
    at all).

    One `now_utc` snapshot taken up front and reused for every symbol in
    this run, so market_open reflects a single consistent instant rather
    than drifting across a run that can take a while over a couple dozen
    instruments."""
    results = []
    total = len(assets)
    now_utc = datetime.now(timezone.utc)
    for i, a in enumerate(assets, start=1):
        if on_progress is not None:
            on_progress(
                f"Analyzing {i}/{total} — {a.symbol} (D1/H4/H1/M5 + chart structure + trading cost)"
            )
        bare_base = _build_bare_base_analysis(a, now_utc)
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
                d1_structure=compute_chart_structure(d1_history),
                h4_divergence=_compute_divergence_for_history(h4_history),
                h1_divergence=_compute_divergence_for_history(h1_history),
                d1_divergence=_compute_divergence_for_history(d1_history),
                # Batch Mega Session pipeline — already tolerates heavier per-symbol cost, unlike
                # the fast Watchlist-popup path (see analyze_ftmo_asset_live's own
                # compute_level_reliability flag docstring): per-level M5 reliability is on.
                **_compute_intraday_reads(
                    base.symbol, d1_history, trade_cost, base.contract_spec, base.ask, compute_level_reliability=True
                ),
            )
        )
    return results


def analyze_ftmo_asset_live(
    symbol: str, bid: float, ask: float, description: str, compute_level_reliability: bool = False, lean: bool = False
) -> FtmoAssetAnalysis:
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
    truly empty price series.

    `compute_level_reliability` (Phase 4, added 2026-09-21, default
    False): running a full trade simulation per M5 S/R level (up to ~6
    levels/symbol via _backtest_all_sr_levels) multiplies this call's own
    backtest cost — acceptable for the batch Mega Session
    pipeline (analyze_ftmo_assets, which always passes True), but not for
    this function's own fast single-symbol Watchlist-popup path, which
    callers should leave at the default False unless they specifically
    need per-level reliability for one symbol on demand."""
    d1_history = fetch_mt5_price_history(symbol, "D1", count=_D1_BACKTEST_BARS)
    d1_prices = d1_history["Close"]
    # Both fetched once, up front, and reused below for the RSI/S-R
    # backtests' real cost/guard-rail simulation AND the analysis' own
    # base/trade_cost fields — same redundant-fetch avoidance as
    # analyze_ftmo_assets' own identical pattern.
    contract_spec = get_contract_spec(symbol)
    trade_cost = get_trade_economics(symbol)
    if d1_history.empty or lean:
        # `lean` (the Execution Clerk defending an open position): the six D1 backtests - about 60% of this call's cost - feed
        # nothing a defend/exit decision uses, so they are skipped along with the headlines below.
        rsi_overbought_bt = rsi_oversold_bt = None
        momentum_bt = volatility_regime_bt = support_resistance_bt = None
        double_bottom_bt = double_top_bt = None
        d1_stats = compute_technical_stats(d1_prices) if d1_history.empty else compute_technical_stats(d1_prices, history=d1_history)
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
        double_bottom_bt, double_top_bt = backtest_chart_pattern_reaction(d1_history, **exec_kwargs)
        d1_stats = compute_technical_stats(d1_prices, history=d1_history)

    base = AssetAnalysis(
        symbol=symbol,
        description=description,
        bid=bid,
        ask=ask,
        display_name=description,
        data_source="mt5",
        market_open=is_symbol_tradable_now(symbol, datetime.now(timezone.utc)),
        prices=d1_prices,
        stats=d1_stats,
        headlines=[] if lean else _fetch_ftmo_headlines(symbol, description),
        contract_spec=contract_spec,
        rsi_overbought_backtest=rsi_overbought_bt,
        rsi_oversold_backtest=rsi_oversold_bt,
        momentum_persistence_backtest=momentum_bt,
        volatility_regime_backtest=volatility_regime_bt,
        support_resistance_backtest=support_resistance_bt,
        double_bottom_backtest=double_bottom_bt,
        double_top_backtest=double_top_bt,
    )
    h4_history = fetch_mt5_price_history(symbol, "H4")
    h1_history = fetch_mt5_price_history(symbol, "H1")
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
        d1_structure=compute_chart_structure(d1_history),
        h4_divergence=_compute_divergence_for_history(h4_history),
        h1_divergence=_compute_divergence_for_history(h1_history),
        d1_divergence=_compute_divergence_for_history(d1_history),
        **_compute_intraday_reads(
            symbol, d1_history, trade_cost, contract_spec, ask, compute_level_reliability=compute_level_reliability, lean=lean
        ),
    )


# Real, round-turn commission by MT5's own symbol-path category — see
# config.py's own commentary for exactly how each rate was sourced/
# confirmed. A category not covered here returns (None, ...) rather than
# silently defaulting to 0 — an unconfirmed gap must read as a gap to the
# model, not as "this instrument is free to trade."
def _ftmo_commission_pct_round_turn(
    category: str, contract_size: float, price: float
) -> tuple[float | None, str]:
    """`contract_size` here means the lot's money per one price unit IN ACCOUNT CURRENCY (ContractSpec.
    risk_per_price_unit), so contract_size * price is the notional in account currency. Passing the raw
    trade_contract_size overstated a USDJPY notional 159x (JPY-quoted) and so understated its commission %
    by the same factor (found 2026-09-25)."""
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
    if category.startswith("Equities"):
        # Direct user correction 2026-08-26: this account's equity
        # symbols (path category "Equities I CFD", confirmed live) had
        # no rate at all before this, which read to the model as a real
        # cost-data gap rather than a genuinely small, known cost — see
        # config.py's own comment on FTMO_COMMISSION_EQUITIES_PCT_
        # ROUND_TURN for the exact instruction. startswith (not an exact
        # match), mirroring Crypto above, in case this account is ever
        # given a second equities group (e.g. "Equities II CFD") the
        # same way it already has Crypto I/II and Cash II/III.
        pct = config.FTMO_COMMISSION_EQUITIES_PCT_ROUND_TURN
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
    contract_size = spec.risk_per_price_unit if spec is not None else 0.0
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


# Real incident, 2026-09-10: several real positions were sized so small
# (against a small aggregate-heat budget split across too many names)
# that their real risk distance couldn't clear the broker's own minimum
# lot at all — "Risking 0.1% of equity against this stop distance can't
# afford even the minimum 0.01-lot for this instrument," discovered only
# AFTER the mega session had already committed to including them, when
# risk/apply_suggestion.py::compute_rebalance_plan tried to actually size
# the order. The existing feasibility line (_feasibility_line, in
# portfolio_suggest.py) only checks MARGIN affordability (can the account
# even afford the min lot's margin at all — rarely the real constraint on
# a funded account) — a completely different question from RISK
# affordability (does the pct this instrument would realistically get,
# against a REAL stop distance, produce enough lots to clear volume_min).
# 1.5x H1 ATR is not an arbitrary number: it's the same Bulkowski stop-
# width floor this account's own audit checklist already cites elsewhere
# (see AUDIT_INSTRUCTION's "Stops Systematically < 1.5x ATR" flaw) —
# reusing it here means this number and that critique can never quietly
# disagree about what "a realistic stop" means for the same instrument.
MIN_VIABLE_SIZE_ATR_MULTIPLE = 1.5


@dataclass
class StopFloor:
    distance: float  # minimum sensible stop distance, in price units
    atr: float  # the ATR the multiple was applied to
    atr_label: str  # "M5" (decision tier) or "H1" (fallback when no M5 read exists)
    multiple: float  # the ATR multiple applied (config.M5_ATR_STOP_MULTIPLE, or the H1 fallback multiple)
    atr_distance: float  # multiple x atr, before the other floors
    binding: str  # which component set `distance`: "atr" / "spread" / "broker minimum" / "0.1% minimum"


def effective_stop_floor(analysis: "FtmoAssetAnalysis", entry_price: float | None = None) -> StopFloor | None:
    """The ONE definition of "the tightest stop this instrument can sensibly carry", shared by
    Clerk's ATR stop-floor guard (which enforces it) and the Mega Session's minimum-viable-size
    and sizing-sheet lines (which size against it). The widest of: config.M5_ATR_STOP_MULTIPLE x
    the M5 ATR (H1 ATR x config.ENTRY_ATR_STOP_FLOOR_MULTIPLE only when no M5 read exists);
    config.MIN_STOP_SPREAD_MULTIPLE x the live spread; the broker's own minimum stop distance; and
    this pipeline's own MIN_STOP_DISTANCE_PCT (below which an order is infeasible). None without an
    ATR or a price. The M5 multiple (2.0) is the same stop distance the earlier 1.5x M15 ATR gave —
    measured median M15/M5 ATR ratio 1.42 across 20 real symbols."""
    price = entry_price or analysis.base.ask
    m5 = analysis.m5_stats.atr
    use_m5 = m5 is not None and m5 > 0
    atr = m5 if use_m5 else analysis.h1_stats.atr
    if atr is None or atr <= 0 or not price:
        return None
    multiple = config.M5_ATR_STOP_MULTIPLE if use_m5 else config.ENTRY_ATR_STOP_FLOOR_MULTIPLE
    atr_distance = multiple * atr
    ask, bid = analysis.base.ask, analysis.base.bid
    spread = (ask - bid) if ask and bid else 0.0
    candidates = [("atr", atr_distance), ("spread", config.MIN_STOP_SPREAD_MULTIPLE * max(spread, 0.0))]
    if analysis.trade_cost is not None and analysis.trade_cost.min_stop_distance_pct > 0:
        candidates.append(("broker minimum", analysis.trade_cost.min_stop_distance_pct / 100 * price * 1.02))
    candidates.append(("0.1% minimum", config.MIN_STOP_DISTANCE_PCT / 100 * price * 1.02))
    binding, distance = max(candidates, key=lambda c: c[1])
    return StopFloor(
        distance=distance, atr=atr, atr_label="M5" if use_m5 else "H1", multiple=multiple,
        atr_distance=atr_distance, binding=binding,
    )


def min_viable_risk_pct(analysis: FtmoAssetAnalysis, account_equity: float | None) -> float | None:
    """The smallest risk % of equity at which the tightest valid stop (effective_stop_floor) still buys the
    broker's minimum lot — the numeric core of format_ftmo_min_viable_size. None without a spec, a floor
    or positive equity (never a guessed default)."""
    if account_equity is None or account_equity <= 0:
        return None
    spec = analysis.base.contract_spec
    floor = effective_stop_floor(analysis)
    if spec is None or floor is None:
        return None
    return (spec.volume_min * floor.distance * spec.risk_per_price_unit) / account_equity * 100


def format_ftmo_min_viable_size(analysis: FtmoAssetAnalysis, account_equity: float | None) -> str | None:
    """The minimum pct (risk %) this instrument needs to clear its own
    broker minimum lot at a REALISTIC stop distance — computed once,
    deterministically, in Python (the exact inverse of risk/apply_
    suggestion.py::compute_rebalance_plan's own risk formula: target_lots
    = (pct/100 * equity) / (stop_distance * contract_size), solved for
    the smallest pct that reaches volume_min) — the same "the model
    decides WHETHER, Python decides the NUMBER" split already used for
    RSI/velocity tiers and the entry-clamping stop fix, applied to the
    sizing-viability question this account's own real pending-order
    graveyard kept exposing. Sizing an instrument below this floor
    doesn't produce a smaller real position — MT5 rounds it to zero
    volume, so it doesn't trade at all; better to see that BEFORE writing
    the thesis than discover it only once execution tries and fails.

    None (never a guessed default) when the real inputs this needs
    (contract spec, a real H1 ATR reading) aren't both available — matches
    this file's own "never fabricate a stat from missing data"
    convention (see _feasibility_line's own analogous contract, and
    analysis.technical.compute_atr)."""
    if account_equity is None or account_equity <= 0:
        return None
    spec = analysis.base.contract_spec
    # Decision-tier basis (2026-09-24): the shared effective stop floor — M5 ATR (H1 only when
    # no M5 read exists), widened to the spread / broker / 0.1% minimums Clerk also enforces.
    floor = effective_stop_floor(analysis)
    if spec is None or floor is None:
        return None
    stop_distance = floor.distance
    stop_basis = (
        f"{floor.multiple:g}x {floor.atr_label} ATR"
        if floor.binding == "atr"
        else f"the {floor.binding} floor, wider than {floor.multiple:g}x {floor.atr_label} ATR"
    )
    min_pct = min_viable_risk_pct(analysis, account_equity)
    return (
        f"  minimum viable size: at a realistic stop ({stop_basis} "
        f"= {stop_distance:.5g} price units from entry), this instrument needs AT LEAST "
        f"{min_pct:.2f}% risk to clear its own {spec.volume_min:g}-lot broker minimum — sizing "
        "it below that doesn't produce a smaller real position, MT5 rounds it to zero volume and "
        "it never trades at all (real incident: 'Risking 0.1% of equity... can't afford even the "
        "minimum 0.01-lot' — several real positions died in the pending-order table exactly this "
        "way). If the risk budget remaining after higher-conviction positions can't cover this "
        "minimum, drop this instrument entirely rather than sizing it below its own viable floor."
    )


def format_ftmo_held_position_sizing_rates(
    positions: list[Position], analyses: list[FtmoAssetAnalysis], account_equity: float | None
) -> str:
    """Real incident, 2026-09-11: the mega session revised NVDA's own
    stop (a genuine improvement — widening an unsafe 0.6x-H1-ATR stop to
    a real 1.53x) but reported pct=0.02% for it — just under the 0.0242%
    that exact stop distance actually needed to size even ONE whole
    share, given NVDA's own real contract spec (volume_step=1.0, whole
    shares only). Every clerk poll since then failed with "can't afford
    even the minimum 1-lot," leaving the REAL position stuck on its old,
    wider stop and stale, stretched target for hours — a beneficial,
    clearly-intended stop-tightening blocked outright by a razor-thin
    arithmetic miss, not a deliberate reduction. (See risk/apply_
    suggestion.py::compute_rebalance_plan's own matching safety-net fix
    for the code-level half of this same incident.)

    For every currently-held FTMO position, hands over the exact,
    already-computed RATE — pct needed per ONE price-unit of stop
    distance to keep the CURRENTLY HELD lot count unchanged — the same
    "the model decides WHETHER, Python decides the NUMBER" split already
    used for RSI/velocity tiers, the ATR-stop candidate, and minimum-
    viable-size: revising a held position's stop becomes "multiply this
    rate by your chosen distance," not free-text margin/contract-size
    arithmetic this account has already gotten wrong once, live.

    "" when there's nothing to show (no positions, no equity, or no
    resolvable contract spec for a given symbol — never fabricates a
    rate from missing data, same convention as format_ftmo_min_viable_
    size)."""
    if not positions or not account_equity or account_equity <= 0:
        return ""
    analysis_by_symbol = {a.base.symbol: a for a in analyses}
    lines = []
    for p in positions:
        analysis = analysis_by_symbol.get(p.symbol)
        if analysis is None or analysis.base.contract_spec is None:
            continue
        spec = analysis.base.contract_spec
        rate = (p.volume * spec.risk_per_price_unit) / account_equity * 100
        lines.append(
            f"  {p.symbol} sizing rate: to keep the CURRENTLY HELD {p.volume:g} lot(s) unchanged "
            f"while revising this position's own stop, use pct = {rate:.6f}% x your chosen stop "
            f"distance (in this instrument's own price units) — e.g. a stop distance of 1.00 "
            f"needs {rate:.4f}% pct, a distance of 5.00 needs {rate * 5:.4f}% pct. Compute this "
            "directly rather than estimating a small round number: a pct that misses this rate, "
            "even slightly, can round DOWN to less than the minimum tradable lot and silently "
            "block the entire revision from ever being applied — the position keeps trading on "
            "its OLD stop/target indefinitely, with no error surfaced anywhere in this report."
        )
    if not lines:
        return ""
    return (
        "Held-position sizing rates (real, precomputed — use these verbatim when revising an "
        "already-held position's stop, do not re-derive this arithmetic yourself):\n" + "\n".join(lines)
    )


def format_timeframe_stats(
    symbol: str, label: str, stats: TechnicalStats, *, long_term: bool = False, context: bool = False,
    decision: bool = False,
) -> str:
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
    if long_term:
        purpose = "long-term structural/bias context, not for entry timing"
    elif decision:
        purpose = "DECISION tier — the basis for entry, stop, target, trigger and sizing"
    elif context:
        purpose = "regime/trend CONTEXT only — NOT the basis for entry/stop/target, those come from M5"
    else:
        purpose = "near-term entry timing/stop placement"
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
    if stats.momentum_acceleration is not None:
        # Direct fix for a real, recurring complaint: a medium-term
        # market_regime of "sideways" can hide several pump/dump swings
        # cancelling out in its own 60-bar average — this short-window
        # sibling (see analysis/technical.py's own MOMENTUM_WINDOW
        # comment) catches a fresh move already underway even when the
        # line above still reads sideways. Shown unconditionally,
        # including "stable" — itself useful confirmation, not noise.
        bits.append(f"short-term momentum (12-bar): {stats.momentum_acceleration}")
    if stats.rsi is not None:
        bits.append(f"RSI {stats.rsi:.0f}")
    if stats.atr_pct is not None:
        if decision:
            atr_note = "size the stop from this ATR and check a target's reachability against it"
        elif long_term or context:
            atr_note = "reference only, NOT a stop distance — size stops from M5 ATR"
        else:
            atr_note = "use for near-term stop distance"
        bits.append(f"ATR {stats.atr_pct:.2f}% of price ({atr_note})")
    if stats.volatility_annualized_pct is not None:
        bits.append(f"volatility {stats.volatility_annualized_pct:.1f}% (annualized)")
    return f"  {label} technical ({purpose}): {', '.join(bits)}"


def format_chart_structure(label: str, snapshot: ChartStructureSnapshot, *, compact: bool = False) -> str:
    """Real, computed chart structure — Fibonacci retracement, multi-
    level support/resistance with real touch counts, trendlines, and
    conservatively-detected chart patterns (see analysis/chart_structure.py's
    own module docstring for exactly which patterns are attempted and
    why others deliberately aren't) — pure Python, zero extra tokens
    spent computing it, so the model reasons over real numbers instead
    of eyeballing a description of a chart it can't actually see.

    `compact` (audit finding 2026-09-24: the M15/M5 decision tier grew the whole-symbol Mega
    context by ~120k characters, ~27%): a shorter rendering of the SAME numbers for the timeframes
    that do not need the full explanatory text — M5 (entry timing) and the H4/H1 context tier. Band
    edges, touch counts, distances, trendline direction, patterns and breakouts all stay; only the
    recency-weighted score, the mid price, the legend text and the long stop-hunt caveat are dropped."""
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
        def _sr_level_text(lvl) -> str:
            # LIQUIDITY POOL (added 2026-09-09): a genuinely tight
            # sub-cluster of real equal highs/lows within this level's
            # own zone — see analysis.chart_structure.SRLevel's own field
            # comment. A stronger, more specific claim than touch count
            # alone (an obvious resting-stop/pending-order concentration,
            # not just "price visited here repeatedly").
            tag = " [LIQUIDITY POOL]" if lvl.is_liquidity_pool else ""
            # Real band (low-high), not a single point — added 2026-09-20,
            # direct user challenge: the clustering tolerance that
            # produces a level always implied a real band width, but this
            # only ever surfaced the cluster's mean. `low`/`high` are None
            # for any SRLevel not built via the current clustering path
            # (a pre-upgrade direct construction, e.g. in an older test
            # fixture) — falls back to the old single-point rendering
            # rather than printing "None-None".
            if lvl.low is not None and lvl.high is not None:
                band = f"{lvl.low:.4f}-{lvl.high:.4f} (mid {lvl.price:.4f})"
            else:
                band = f"{lvl.price:.4f}"
            score = f"{lvl.weighted_score:.1f}" if lvl.weighted_score is not None else "n/a"
            if compact:
                short_band = f"{lvl.low:.4f}-{lvl.high:.4f}" if lvl.low is not None and lvl.high is not None else f"{lvl.price:.4f}"
                return f"{short_band} ({lvl.touches}x, {lvl.distance_pct:+.2f}%){tag}"
            return f"{band} ({lvl.touches}x raw / {score} recency-weighted, {lvl.distance_pct:+.2f}%){tag}"
        res = "; ".join(_sr_level_text(lvl) for lvl in sr.resistance_levels)
        sup = "; ".join(_sr_level_text(lvl) for lvl in sr.support_levels)
        if compact:
            parts.append(f"S/R (band, touches, distance) — resistance: {res or 'none'}; support: {sup or 'none'}")
        else:
            parts.append(
                f"S/R levels (band, raw touch count / recency-weighted strength, distance) — "
                f"resistance: {res or 'none'}; support: {sup or 'none'}"
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
        def _pattern_text(p) -> str:
            if not compact or len(p.detail) <= 120:
                return f"{p.name} ({p.detail})"
            return f"{p.name} ({p.detail[:117].rstrip()}...)"

        parts.append("chart patterns: " + "; ".join(_pattern_text(p) for p in snapshot.patterns))

    # Real vs. fake — added 2026-09-20, direct user challenge. "tick-
    # volume" (not "volume") is deliberate wording: MT5's own Volume
    # column for FX/CFDs is real tick-count/quote-frequency, not genuine
    # traded volume — see analysis.chart_structure's own module comment.
    if snapshot.breakouts:
        def _breakout_text(b) -> str:
            conf = "tick-volume-confirmed" if b.volume_confirmed else "NOT tick-volume-confirmed"
            ratio = f"{b.volume_ratio:.1f}x baseline" if b.volume_ratio is not None else "no volume data"
            return f"{b.direction} through {b.level_price:.4f} ({b.close_through_pct:+.2f}%, {conf}, {ratio}, {b.bars_ago} bar(s) ago)"
        parts.append("breakouts: " + "; ".join(_breakout_text(b) for b in snapshot.breakouts))

    if snapshot.liquidity_sweeps:
        def _sweep_text(s) -> str:
            if compact:
                return (
                    f"{s.direction} {s.level_price:.4f} (wicked {s.wick_penetration_pct:.2f}% through, "
                    f"closed back inside, {s.bars_ago} bar(s) ago; stop-hunt, not a break)"
                )
            return (
                f"{s.direction} {s.level_price:.4f} (wicked {s.wick_penetration_pct:.2f}% through, "
                f"closed back inside, {s.bars_ago} bar(s) ago) — possible stop-hunt/liquidity sweep, not a real break"
            )
        parts.append("liquidity sweeps: " + "; ".join(_sweep_text(s) for s in snapshot.liquidity_sweeps))

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

    `short_direction` is the M5 (decision timeframe) regime direction since 2026-09-24; it used to
    be H4's.

    Returns (state, short_direction, agreeing_backdrop_names,
    opposing_backdrop_names) where state is one of "not_available",
    "flat", "no_backdrop", "structurally_backed", "counter_trend_spike",
    "mixed"."""
    # The decision timeframe's own regime (M5, 2026-09-24) against the daily backdrop — H4/H1 are
    # context read elsewhere (format_intraday_alignment, format_mtf_confluence).
    short = _regime_direction(analysis.m5_stats.market_regime)
    d1_dir = _regime_direction(analysis.base.stats.market_regime)

    if short is None:
        return "not_available", short, [], []
    if short == "flat":
        return "flat", short, [], []

    backdrop = [(name, d) for name, d in (("daily", d1_dir),) if d not in (None, "flat")]
    if not backdrop:
        return "no_backdrop", short, [], []

    agreeing = [name for name, d in backdrop if d == short]
    opposing = [name for name, d in backdrop if d != short]
    if not opposing:
        return "structurally_backed", short, agreeing, opposing
    if not agreeing:
        return "counter_trend_spike", short, agreeing, opposing
    return "mixed", short, agreeing, opposing


def aligned_m5_trend_direction(stats: TechnicalStats | None) -> str | None:
    """The M5 (decision timeframe) counterpart of aligned_h1_h4_trend_direction, for risk management:
    'up'/'down' only when TWO independent M5 reads agree — the current price-vs-20-bar-SMA trend
    snapshot AND the 60-bar market_regime drift direction. Same dual-confirmation idea as the H1+H4
    version (a fast snapshot plus a slower drift must agree), on the timeframe the account's stops
    are actually sized in. None when either is missing, flat, or they disagree."""
    if stats is None or stats.trend in (None, "flat"):
        return None
    trend_dir = "up" if stats.trend == "uptrend" else "down"
    regime_dir = _regime_direction(stats.market_regime)
    return trend_dir if regime_dir == trend_dir else None


def aligned_h1_h4_trend_direction(h4_trend: str | None, h1_trend: str | None) -> str | None:
    """Real, un-hedged short-term directional read: 'up'/'down' when H1
    and H4's own TREND fields (TechnicalStats.trend — a simple, current
    price-vs-20-bar-SMA snapshot, NOT market_regime's medium-term drift
    classification) agree with EACH OTHER. None when they disagree, are
    unavailable, or read flat — there's nothing genuinely un-hedged to
    report in any of those cases.

    Takes bare trend strings rather than a whole FtmoAssetAnalysis (its
    original shape when first extracted 2026-09-16) so BOTH real callers
    can reuse it without reshaping their own already-unpacked data:
    _tactical_trend_vs_regime_conflict below (analysis.h4_stats.trend/
    analysis.h1_stats.trend) and ai.clerk_execution._compute_tactical_
    signals (bare h4_stats/h1_stats TechnicalStats objects, no
    FtmoAssetAnalysis wrapper available there at all) — one shared
    definition of "aligned," never two copies that could silently
    diverge on what counts as a real tactical trend read."""
    if h4_trend is None or h1_trend is None or h4_trend != h1_trend or h4_trend == "flat":
        return None
    return "up" if h4_trend == "uptrend" else "down"


def _tactical_trend_vs_regime_conflict(analysis: FtmoAssetAnalysis, regime_direction: str | None) -> str | None:
    """Real incident, 2026-09-10: a BTCUSD buy cited STRUCTURALLY BACKED
    (this function's own regime-based direction agreed with D1/Monthly),
    while the account's own H4-vs-H1 TREND read (format_mtf_confluence,
    a genuinely different metric — TechnicalStats.trend, a simple
    current-price-vs-20-bar-SMA snapshot, NOT market_regime's medium-term
    drift classification) explicitly read ALIGNED DOWNTREND on both
    timeframes — the exact opposite direction, never engaged with in the
    thesis. Both metrics were individually correct (a regime can drift
    net-upward while briefly sitting under its own 20-bar SMA) — the real
    gap was presenting two genuinely different signals as if they were
    restating the same fact, with nothing flagging that they'd actually
    diverged.

    Returns a real, concrete warning string when H1 AND H4's own `trend`
    fields agree with EACH OTHER (a real, un-hedged tactical read, not
    just noise on one timeframe) but disagree with `regime_direction` —
    this account trades intraday (hours, not weeks), so a real, agreeing
    SHORT-term signal is exactly the kind of thing a slower, more
    structural regime read can quietly overrule if nobody points out
    they've diverged. None when there's nothing to flag (no real
    tactical read, or the two genuinely agree)."""
    tactical_direction = aligned_h1_h4_trend_direction(analysis.h4_stats.trend, analysis.h1_stats.trend)
    if tactical_direction is None:
        return None
    if regime_direction in (None, "flat") or tactical_direction == regime_direction:
        # regime_direction is "flat"/None (the M5 regime shows no real
        # net direction, or there's not enough history) -- there's
        # nothing genuinely OPPOSITE for the tactical read to conflict
        # WITH; wording this as a conflict would be misleading, not just
        # unhelpful.
        return None
    return (
        f"  ** CONTEXT/DECISION CONFLICT: the H1 AND H4 context reads' own TREND readings (a simple, "
        f"current price-vs-20-bar-SMA snapshot — see the Multi-timeframe H4-vs-H1 line above) both "
        f"read {tactical_direction.upper()}ward, the OPPOSITE of the M5 regime direction "
        "below. This is NOT the same signal stated twice — the M5 regime is a short, "
        "decision-timeframe drift classification, H1/H4 TREND is the larger flow the trade sits "
        "inside — and a decision-tier M5 move running against an AGREEING H1+H4 flow is a "
        "counter-flow trade: label it as one in the thesis, size it smaller, and keep the target "
        "near, rather than reading the M5 move as a durable trend. Real incident this guards "
        "against: a BTCUSD buy cited a STRUCTURALLY BACKED regime read while H1 and H4 both "
        "explicitly read downtrend underneath it, unaddressed in the thesis — engage with this "
        "explicitly, don't treat it as noise to explain away. **"
    )


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
    trades are allowed.

    Deliberately says "the M5 REGIME direction" below (H4's until 2026-09-24, when M5 became the
    only decision timeframe), not just "the M5 move" — real incident, 2026-09-10: that vaguer wording
    read as flatly restating format_mtf_confluence's own H4-vs-H1 TREND
    line above it, when the two are genuinely different metrics
    (market_regime vs. TechnicalStats.trend) that can legitimately
    disagree — see _tactical_trend_vs_regime_conflict's own docstring for
    the real BTCUSD incident this caused and the explicit conflict
    warning appended below when it happens again."""
    state, short, agreeing, opposing = classify_long_term_alignment(analysis)
    conflict_warning = _tactical_trend_vs_regime_conflict(analysis, short)

    if state == "not_available":
        return "  Long-term alignment: M5 market-type not available yet (insufficient history)."
    if state == "flat":
        return (
            "  Long-term alignment: the M5 regime shows no real net direction "
            "right now — nothing yet to compare against the daily backdrop."
        )
    if state == "no_backdrop":
        return (
            "  Long-term alignment: no real daily directional backdrop available "
            "(insufficient history, or it reads sideways) — the M5 regime has no "
            "longer-term structure to either confirm or contradict it."
        )
    if state == "structurally_backed":
        named = " and ".join(agreeing)
        text = (
            f"  Long-term alignment: STRUCTURALLY BACKED — the M5 REGIME direction "
            f"({short}ward) agrees with the real {named} backdrop, not just a short-term read in "
            "isolation — genuinely stronger conviction evidence for sizing (per DEBATE THE "
            "POSITION SIZE above). This is NOT, on its own, a reason to extend the holding window "
            "past this account's own intraday default — that decision still runs entirely through "
            "the HOLDING HORIZON/DEBATE THE HOLDING PERIOD debate above (real cost vs. move, swap "
            "sign); a structurally-backed move that also clears THAT bar is a stronger case for "
            "the overnight exception than an unbacked one would be, nothing more."
        )
    elif state == "counter_trend_spike":
        named = " and ".join(opposing)
        text = (
            f"  Long-term alignment: COUNTER-TREND SPIKE — the M5 REGIME direction "
            f"({short}ward) runs AGAINST the real {named} backdrop. This is NOT a reason to "
            "discard it outright — a real, fast counter-trend move can be genuinely tradeable on "
            "its own terms — but treat it explicitly as a short-lived move to capture and exit "
            "deliberately, not as the start of a durable trend: a tighter stop, a firmer "
            "holding-window commitment, and sizing that reflects lower conviction in it lasting "
            "are all warranted specifically BECAUSE of this read, not despite it."
        )
    else:
        text = (
            f"  Long-term alignment: MIXED — the M5 REGIME direction ({short}ward) "
            f"agrees with the {' and '.join(agreeing)} backdrop but runs against the "
            f"{' and '.join(opposing)} one; partial, not full, longer-term confirmation."
        )
    return text if conflict_warning is None else f"{text}\n{conflict_warning}"


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
    "mixed": "The bigger trends disagree, so this is only half-confirmed — size smaller.",
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
    # Confidence tier — added 2026-09-20, direct user challenge. Every
    # existing rule (3-10) still reports "moderate" (SetupSignal's own
    # default) since only reversal_candidate has real confirmation logic
    # so far — shown for every signal regardless, so a reader always
    # sees the same field rather than only for the ones that vary.
    bits = "; ".join(f"{s.name} [{s.confidence}] — {s.detail}" for s in signals)
    return f"  {label} setup read: {bits}"


def format_trade_zone_for(
    label: str,
    stats: TechnicalStats,
    structure: ChartStructureSnapshot,
    signals: list[SetupSignal],
    level_reliability: dict | None = None,
    *,
    min_reward_risk: float | None = None,
    stop_atr: float | None = None,
    reach_atr: float | None = None,
    extra_target_prices: list[float] | None = None,
    strong_hold_rate_pct: float | None = None,
    weak_hold_rate_pct: float | None = None,
    reach_limit_atr: float | None = None,
    measured_moves: bool = False,
    drag_for_distance: Callable[[float], float | None] | None = None,
) -> str:
    """Advisory trade-zone candidates (buy AND sell) for ONE timeframe (the M5 decision read, via
    format_trade_zone). Same advisory-only framing as ever: a strong, named candidate to adopt or
    explicitly override, never a hard rule. `reach_atr` (the M5 ATR): when given, each target is also
    stated as a multiple of it — the exact number the prompt's reachability rule (a same-session
    target sits within ~6x the M5 ATR) needs, so the model does not have to derive it.
    `extra_target_prices`: real session levels (prev-day/today's high and low) offered as targets
    alongside the S/R bands — see analysis.trade_zone.construct_trade_zone."""
    lines = [
        f"  {label} trade-zone candidate (advisory — a strong candidate to adopt or explicitly "
        "override with your own reasoning, never a hard rule):"
    ]
    kwargs = {"level_reliability": level_reliability}
    if min_reward_risk is not None:
        kwargs["min_reward_risk"] = min_reward_risk
    if stop_atr is not None:
        kwargs["stop_atr"] = stop_atr
    if extra_target_prices:
        kwargs["extra_target_prices"] = extra_target_prices
    if strong_hold_rate_pct is not None:
        kwargs["strong_hold_rate_pct"] = strong_hold_rate_pct
    if weak_hold_rate_pct is not None:
        kwargs["weak_hold_rate_pct"] = weak_hold_rate_pct
    if measured_moves:
        kwargs["measured_moves"] = True
    any_candidate = False
    for side in ("buy", "sell"):
        zone = construct_trade_zone(side, stats, structure, signals, **kwargs)
        if zone is None:
            lines.append(f"    {side}: no real structure/reward-risk currently clears the bar")
            continue
        any_candidate = True
        anchor_text = f", anchored to a real {zone.anchor_level.touches}-touch level" if zone.anchor_level is not None else ""
        targets = ", ".join(
            f"{tp:.4f}" + (f" [{zone.target_labels[tp]}]" if tp in zone.target_labels else "") for tp in zone.take_profits
        )
        reach_text = ""
        if reach_atr and reach_atr > 0:
            entry_ref = zone.entry_high if side == "buy" else zone.entry_low
            ratios = [abs(tp - entry_ref) / reach_atr for tp in zone.take_profits]
            flag = "*" if reach_limit_atr else ""
            multiples = ", ".join(
                f"{ratio:.1f}x{flag if reach_limit_atr and ratio > reach_limit_atr else ''}" for ratio in ratios
            )
            reach_text = f"; distance in {label} ATRs: {multiples}"
            if reach_limit_atr and any(ratio > reach_limit_atr for ratio in ratios):
                reach_text += (
                    f" (* = beyond the ~{reach_limit_atr:g}x same-session reachability limit — only sensible "
                    "as the runner of a partial-and-trail plan: take part at a nearer target, trail the rest; "
                    "otherwise prefer a nearer target or justify the wider horizon)"
                )
        entry_gap_text = ""
        if reach_atr and reach_atr > 0 and stats.last_price:
            # How far the entry zone sits from the last price, in the same ATR units the fill odds
            # depend on: a limit that far away has to be TOUCHED first (measured: 60% within 3h at
            # 2 M5 ATRs, 39% at 3.5, 26% at 5) — the stated fact, not a rule.
            gap = (zone.entry_low - stats.last_price) if zone.entry_low > stats.last_price else (
                stats.last_price - zone.entry_high if zone.entry_high < stats.last_price else 0.0
            )
            entry_gap_text = f" [entry zone {abs(gap) / reach_atr:.1f}x the {label} ATR from the last price]"
        stop_text = f"stop {zone.stop_loss:.4f}"
        if drag_for_distance is not None:
            entry_ref = zone.entry_high if side == "buy" else zone.entry_low
            drag = drag_for_distance(abs(entry_ref - zone.stop_loss))
            if drag is not None:
                stop_text += f" (cost drag {drag:.2f}R)"
        if zone.structure_stop is not None:
            entry_ref = zone.entry_high if side == "buy" else zone.entry_low
            structure_distance = abs(entry_ref - zone.structure_stop)
            structure_atr = f", {structure_distance / reach_atr:.1f}x the {label} ATR from the entry" if reach_atr else ""
            structure_drag = drag_for_distance(structure_distance) if drag_for_distance is not None else None
            drag_part = f", cost drag {structure_drag:.2f}R" if structure_drag is not None else ""
            stop_text += (
                f"; structure-stop alternative {zone.structure_stop:.4f} (just beyond the swing extreme of the "
                f"leg being ridden{structure_atr}{drag_part} - the wider stop means a smaller size for the same "
                "risk; your call which the thesis needs)"
            )
        lines.append(
            f"    {side} [{zone.confidence}, basis={zone.basis}{anchor_text}]: entry "
            f"{zone.entry_low:.4f}-{zone.entry_high:.4f}{entry_gap_text}, {stop_text}, "
            f"target(s) {targets} (nearest clears {zone.reward_risk:.1f}R{reach_text})"
        )
    if not any_candidate:
        lines.append("    (context only — not enough real structure right now to construct either side)")
    return "\n".join(lines)


def format_intraday_levels(symbol: str, levels: IntradayLevels | None) -> str:
    """Session context for the decision tier: prior-day H/L/C, today's
    O/H/L, session VWAP (tick-volume proxy), and how much of the average
    daily range today has already used — a target beyond the day's likely
    remaining range is much less reachable in the same session."""
    if levels is None:
        return f"  Session levels for {symbol}: not available (insufficient M5/D1 history)"
    px = levels.last_price

    def _dist(v: float | None) -> str:
        return f" ({(v - px) / px * 100:+.2f}% from price)" if v else ""

    parts = [f"last {px:.4f}"]
    if levels.prev_day_high is not None:
        parts.append(
            f"prev-day H {levels.prev_day_high:.4f}{_dist(levels.prev_day_high)} / "
            f"L {levels.prev_day_low:.4f}{_dist(levels.prev_day_low)} / C {levels.prev_day_close:.4f}"
        )
    if levels.day_open is not None:
        session_label = f"session {levels.session_date}" if levels.session_date else "today"
        parts.append(f"{session_label} O {levels.day_open:.4f} H {levels.day_high:.4f} L {levels.day_low:.4f}")
    if levels.vwap is not None:
        parts.append(f"session VWAP {levels.vwap:.4f}{_dist(levels.vwap)} (tick-volume weighted)")
    if levels.adr:
        used = f", {levels.range_used_pct:.0f}% of it already used in that session" if levels.range_used_pct is not None else ""
        parts.append(f"avg daily range {levels.adr:.4f} ({levels.adr / px * 100:.2f}% of price){used}")
    return f"  Session levels ({symbol}, broker clock): " + "; ".join(parts)


def format_intraday_alignment(analysis: "FtmoAssetAnalysis") -> str:
    """The M5 decision read's trend vs the H1 and H4 CONTEXT reads, plus any M5 level that coincides
    with an H1/H4 level — a short-range level backed by a higher-timeframe level is stronger evidence
    than either alone. The confluence tolerance is one M5 ATR% (a flat 0.5% would be ~4 M5 ATRs on a
    stock and make nearly every M5 level look "backed")."""
    trends = {"M5": analysis.m5_stats.trend, "H1": analysis.h1_stats.trend, "H4": analysis.h4_stats.trend}
    if any(v is None for v in trends.values()):
        alignment = "not available (insufficient history on at least one of M5/H1/H4)"
    elif len(set(trends.values())) == 1:
        alignment = (
            "NONE OF M5/H1/H4 SHOWS A CLEAR TREND — all read flat"
            if trends["H1"] == "flat"
            else f"ALIGNED — M5, H1 and H4 all read {trends['H1']}"
        )
    elif trends["M5"] != trends["H1"] and "flat" not in (trends["M5"], trends["H1"]):
        alignment = (
            f"M5 vs H1 CONFLICT — M5 {trends['M5']}, H1 {trends['H1']} (H4 {trends['H4']}): the M5 setup runs "
            "against the H1 context, a headwind to weigh in conviction, size and target distance"
        )
    else:
        alignment = f"MIXED — M5 {trends['M5']}, H1 {trends['H1']}, H4 {trends['H4']}"
    higher_levels = _candidate_levels(analysis.h1_structure) + _candidate_levels(analysis.h4_structure)
    confluence_tolerance = max(0.03, analysis.m5_stats.atr_pct or 0.5)
    zones = find_mtf_confluence(
        higher_tf_levels=higher_levels,
        lower_tf_levels=_candidate_levels(analysis.m5_structure),
        current_price=analysis.base.ask,
        tolerance_pct=confluence_tolerance,
    )
    if zones:
        nearest = sorted(zones, key=lambda z: abs(z.distance_pct))[:3]
        zones_text = "; ".join(f"{z.avg_price:.4f} ({z.kind}, {z.distance_pct:+.2f}%)" for z in nearest)
    else:
        zones_text = "none found"
    return f"  Intraday trend alignment (M5 decision read vs H1/H4 context): {alignment}; M5 levels backed by H1/H4: {zones_text}"


def _round_trip_cost_pct(analysis: "FtmoAssetAnalysis") -> float | None:
    cost = analysis.trade_cost
    if cost is None:
        return None
    spec = analysis.base.contract_spec
    contract_size = spec.risk_per_price_unit if spec is not None else 0.0
    commission_pct, _note = _ftmo_commission_pct_round_turn(cost.category, contract_size, analysis.base.ask)
    return cost.spread_pct_of_price + (commission_pct or 0.0)


def cost_drag_r(analysis: "FtmoAssetAnalysis", stop_distance: float, price: float | None = None) -> float | None:
    """Round-trip spread + commission expressed in R for a stop `stop_distance` price units away:
    the fraction of the risked amount the broker takes before the trade has done anything. MEASURED
    2026-09-25 on real spreads: mean 0.19R at a 2x M5-ATR stop, 0.09R at 4x, 0.06R at 6x; per symbol
    from ~0.01R (BTC) to >1R (WHEAT, COCOA). None without a cost read, a price or a positive stop."""
    cost_pct = _round_trip_cost_pct(analysis)
    price = price or analysis.base.ask
    if cost_pct is None or not price or stop_distance <= 0:
        return None
    return (cost_pct / 100 * price) / stop_distance


def stop_distance_for_drag(analysis: "FtmoAssetAnalysis", max_drag_r: float, price: float | None = None) -> float | None:
    """The stop distance (price units) at which round-trip cost falls to `max_drag_r` R — the inverse of
    cost_drag_r. Information for the model, deliberately NOT a new stop floor: Claude picks the stop
    from structure and decides whether a wider one is justified."""
    cost_pct = _round_trip_cost_pct(analysis)
    price = price or analysis.base.ask
    if cost_pct is None or not price or max_drag_r <= 0:
        return None
    return (cost_pct / 100 * price) / max_drag_r


def format_intraday_sizing_sheet(analysis: "FtmoAssetAnalysis", account_equity: float | None) -> str | None:
    """Deterministic sizing sheet from the M5 decision tier (Python
    computes the numbers, the model decides the pct): at the effective M5
    stop floor (2x M5 ATR unless a spread/broker/0.1% minimum binds), what does
    risking 0.5%/1.0% of equity actually buy — lots, and
    approximate margin as a share of equity — and how big is the round-trip
    cost relative to that stop. Tight M5-scale stops mean MORE lots for
    the same risk pct than an H1-scale stop would, so the margin share
    is shown explicitly. None without a contract spec, M5 ATR or equity."""
    spec = analysis.base.contract_spec
    floor = effective_stop_floor(analysis)
    if account_equity is None or account_equity <= 0 or spec is None or floor is None or floor.atr_label != "M5":
        return None
    stop_distance = floor.distance
    price = analysis.base.ask
    basis = f"{floor.multiple:g}x M5 ATR" if floor.binding == "atr" else f"the {floor.binding} floor"
    parts = [f"stop = {stop_distance:.5g} price units ({stop_distance / price * 100:.3f}% of price; {basis})"]
    for risk_pct in (0.5, 1.0):
        raw_lots = (risk_pct / 100 * account_equity) / (stop_distance * spec.risk_per_price_unit)
        # Same epsilon-guarded round-down compute_rebalance_plan uses: a bare `//` returns
        # 15 for 0.16 // 0.01 (float repr), understating a whole lot step.
        lots = round(int(raw_lots / spec.volume_step + 1e-9) * spec.volume_step, 8) if spec.volume_step > 0 else raw_lots
        if lots < spec.volume_min:
            parts.append(f"{risk_pct:g}% risk -> below the {spec.volume_min:g}-lot minimum (would not trade)")
            continue
        margin_pct = lots * spec.margin_initial / account_equity * 100
        parts.append(f"{risk_pct:g}% risk -> {lots:g} lots (~{margin_pct:.1f}% of equity as margin)")
    cost_pct = _round_trip_cost_pct(analysis)
    drag = cost_drag_r(analysis, stop_distance, price)
    if cost_pct is not None and drag is not None:
        parts.append(f"round-trip cost {cost_pct:.4f}% = {drag:.2f}R against that stop")
        cheap_stop = stop_distance_for_drag(analysis, config.COST_DRAG_TARGET_R, price)
        if drag > config.COST_DRAG_TARGET_R and cheap_stop is not None and floor.atr > 0:
            parts.append(
                f"a stop of {cheap_stop:.5g} price units ({cheap_stop / floor.atr:.1f}x M5 ATR) would cap that "
                f"drag at {config.COST_DRAG_TARGET_R:g}R - information for choosing the structure stop, not a floor"
            )
        if drag > config.COST_DRAG_SHEET_VETO_R:
            parts.append(
                f"COST VETO: {drag:.2f}R > {config.COST_DRAG_SHEET_VETO_R:g}R at the tightest valid stop - the broker "
                "would keep more than a third of every risked unit; skip unless a structure stop this wide "
                "or wider is genuinely justified"
            )
    return "  Intraday sizing sheet (M5-based, precomputed): " + "; ".join(parts)


def _session_target_prices(levels: IntradayLevels | None) -> list[float]:
    """Prev-day and today's high/low — the real session extremes an intraday trade runs to."""
    if levels is None:
        return []
    return [
        price
        for price in (levels.prev_day_high, levels.prev_day_low, levels.day_high, levels.day_low)
        if price
    ]


def m5_trade_zone_candidates(a: "FtmoAssetAnalysis", m5_signals: list[SetupSignal]) -> dict:
    """{"buy": TradeZoneSuggestion | None, "sell": ...} built with EXACTLY the arguments format_trade_zone
    prints (same effective stop floor, session extremes, hold-rate cutoffs, measure-rule targets), so any
    other consumer — the final live re-check's level list — sees the same numbers Claude was shown."""
    floor = effective_stop_floor(a)
    stop_atr = (
        floor.distance / TRADE_ZONE_STOP_ATR_MULTIPLE if floor is not None and floor.atr_label == "M5" else None
    )
    kwargs = dict(
        level_reliability=a.m5_level_reliability,
        extra_target_prices=_session_target_prices(a.intraday_levels) or None,
        strong_hold_rate_pct=config.M5_ZONE_STRONG_HOLD_RATE_PCT,
        weak_hold_rate_pct=config.M5_ZONE_WEAK_HOLD_RATE_PCT,
        measured_moves=True,
    )
    if stop_atr is not None:
        kwargs["stop_atr"] = stop_atr
    return {side: construct_trade_zone(side, a.m5_stats, a.m5_structure, m5_signals, **kwargs) for side in ("buy", "sell")}


def format_trade_zone(a: "FtmoAssetAnalysis", m5_signals: list[SetupSignal]) -> str:
    """The M5 trade-zone candidate (the only decision timeframe since 2026-09-24): BOTH "buy" and
    "sell" (the model, not this function, decides which direction fits its thesis, or says plainly
    when neither clears the bar). The stop respects the same effective floor Clerk enforces
    (effective_stop_floor: M5 ATR / spread / broker / 0.1% minimum), or Claude would adopt a stop the
    guard then silently widens; targets are the real opposing M5 S/R bands plus the session
    extremes, each stated in M5 ATRs for the reachability rule.

    ADVISORY ONLY, per the user's own explicit decision: presented as a strong, named candidate the
    model must adopt or explicitly explain deviating from, never a hard clamp — mirrors
    format_ftmo_min_viable_size's own "Python computes the number, model decides" framing.
    `a.m5_level_reliability` is an empty dict (never None) on the fast Watchlist-popup path —
    construct_trade_zone treats that exactly like "no reliability data available"."""
    floor = effective_stop_floor(a)
    stop_atr = (
        floor.distance / TRADE_ZONE_STOP_ATR_MULTIPLE if floor is not None and floor.atr_label == "M5" else None
    )
    return format_trade_zone_for(
        "M5", a.m5_stats, a.m5_structure, m5_signals, a.m5_level_reliability,
        stop_atr=stop_atr, reach_atr=a.m5_stats.atr, extra_target_prices=_session_target_prices(a.intraday_levels),
        strong_hold_rate_pct=config.M5_ZONE_STRONG_HOLD_RATE_PCT, weak_hold_rate_pct=config.M5_ZONE_WEAK_HOLD_RATE_PCT,
        reach_limit_atr=config.M5_TARGET_REACH_ATR_LIMIT, measured_moves=True,
        drag_for_distance=lambda distance: cost_drag_r(a, distance),
    )


def format_ftmo_asset_context(
    analyses: list[FtmoAssetAnalysis],
    account_equity: float | None = None,
    include_favorable_excursion: bool = True,
    include_market_status: bool = True,
    calendar_events: list | None = None,
    now_utc: datetime | None = None,
) -> str:
    """Reuses format_enriched_asset_context (called once per single-asset
    slice, unchanged) for each symbol's existing daily/feasibility/
    backtest block, then appends chart-structure + setup-classification
    for ALL FOUR timeframes (real feature added live 2026-08-22, user's
    own direct request — see FtmoAssetAnalysis's own docstring for why
    D1/monthly previously stopped at a bare stats line), the multi-
    timeframe confluence/alignment read, the real trading-cost line, and
    (added 2026-09-11) the real minimum-viable-size line (see format_ftmo_
    min_viable_size's own docstring for the real incident it targets)
    right after it — keeps every symbol's full read together rather than
    grouping all base reads first and all intraday reads after.

    `include_favorable_excursion=False`: ai/clerk_execution.py's own
    _fetch_technical_context MUST pass this — its tactical-verdict
    prompts reuse this exact function's output as their technical_context
    but never include this file's own _INSTRUCTION_HEAD, which is the
    ONLY place the favorable-excursion figure's critical misread warning
    and HOLDING HORIZON scale-mismatch caveat actually live (see
    ai/portfolio_suggest.py::format_backtests' own docstring for the full
    reasoning).

    `include_market_status=False`: same caller, same reasoning — see
    format_enriched_asset_context's own docstring. Claude's own mega-
    session prompt always DOES include _INSTRUCTION_HEAD, so it omits
    both of these arguments and keeps their defaults."""
    lines = []
    for a in analyses:
        lines.append(
            format_enriched_asset_context(
                [a.base],
                account_equity=account_equity,
                include_favorable_excursion=include_favorable_excursion,
                include_market_status=include_market_status,
                daily_atr_is_context=True,
            )
        )
        symbol = a.base.symbol
        # ---- DECISION TIER: M5 ONLY (the PRIMARY basis for entry, stop, target, trigger,
        # invalidation and sizing) -------------------------------------------------------
        lines.append(
            f"  === DECISION TIER for {symbol} — M5: the ONLY basis for entry, stop, target, trigger, "
            "invalidation and sizing. M5 supplies the structure (support/resistance, Fibonacci, "
            "trendlines, patterns) that says WHERE stops and targets sit, the entry timing and trigger, "
            "the ATR that sizes the stop and checks a target's reachability, and the intraday backtest "
            "evidence. (The D1 daily read and backtest lines printed above, and the D1/H4/H1 reads "
            "further below, are CONTEXT only.) ==="
        )
        if config.INTRADAY_BACKTEST_ENABLED:
            lines.append(_format_intraday_backtests(a))
        lines.append(format_timeframe_stats(symbol, "M5", a.m5_stats, decision=True))
        lines.append(format_chart_structure("M5", a.m5_structure))
        m5_signals = classify_setups(a.m5_stats, a.m5_structure, a.m5_divergence, profile=M5_PROFILE)
        lines.append(format_setup_signals("M5", m5_signals))
        lines.append(format_trade_zone(a, m5_signals))
        for level_map in (a.m5_levels or {}).values():
            if level_map is not None:
                lines.append(level_map.text())
        lines.append(format_intraday_levels(symbol, a.intraday_levels))
        if config.SYMBOL_CARD_ENABLED:
            card_line = format_symbol_card(symbol)
            if card_line is not None:
                lines.append(card_line)
        lines.append(format_intraday_alignment(a))
        if calendar_events is not None and now_utc is not None:
            event_line = economic_calendar.format_symbol_events(symbol, now_utc, events=calendar_events)
            if event_line is not None:
                lines.append(event_line)
        lines.append(format_ftmo_trade_cost(a))
        min_viable_size_line = format_ftmo_min_viable_size(a, account_equity)
        if min_viable_size_line is not None:
            lines.append(min_viable_size_line)
        sizing_sheet = format_intraday_sizing_sheet(a, account_equity)
        if sizing_sheet is not None:
            lines.append(sizing_sheet)
        # ---- CONTEXT TIER: D1 / H4 / H1 (regime + trend only) ----------------------------
        lines.append(
            f"  === CONTEXT TIER for {symbol} — D1 / H4 / H1: regime, trend direction and big magnet levels "
            "ONLY. Use it to decide whether the M5 setup is WITH or AGAINST the larger trend (a tailwind "
            "or headwind, a permission to trade a direction or not); never source an entry, stop, target "
            "or size from these timeframes. ==="
        )
        lines.append(format_chart_structure("D1", a.d1_structure))
        lines.append(format_setup_signals("D1", classify_setups(a.base.stats, a.d1_structure, a.d1_divergence)))
        lines.append(format_timeframe_stats(symbol, "H4", a.h4_stats, context=True))
        lines.append(format_chart_structure("H4", a.h4_structure, compact=True))
        lines.append(format_setup_signals("H4", classify_setups(a.h4_stats, a.h4_structure, a.h4_divergence)))
        lines.append(format_timeframe_stats(symbol, "H1", a.h1_stats, context=True))
        lines.append(format_chart_structure("H1", a.h1_structure, compact=True))
        lines.append(format_setup_signals("H1", classify_setups(a.h1_stats, a.h1_structure, a.h1_divergence)))
        lines.append(format_mtf_confluence(a))
        lines.append(format_long_term_alignment(a))
    return "\n".join(lines)


def _compact_level_lines(a: "FtmoAssetAnalysis") -> list[str]:
    lines = []
    for side, label in (("buy", "Nearest supports below"), ("sell", "Nearest resistances above")):
        level_map = (a.m5_levels or {}).get(side)
        if level_map is not None and level_map.candidates:
            parts = [
                f"{c.price:.5g} ({c.distance_atr:.1f} ATR{', reaction point' if c.is_reaction else ''})"
                for c in level_map.candidates[:3]
            ]
            lines.append(f"  {label}: " + "; ".join(parts))
    return lines


def _htf_context_line(a: "FtmoAssetAnalysis") -> str:
    def _read(stats) -> str:
        if stats is None:
            return "n/a"
        return f"{stats.trend or 'n/a'}" + (f" ({stats.market_regime})" if getattr(stats, "market_regime", None) else "")

    return (
        "  Higher-timeframe context (regime only, never the basis for an M5 judgement): "
        f"D1 {_read(a.base.stats)}; H4 {_read(a.h4_stats)}; H1 {_read(a.h1_stats)}"
    )


def format_clerk_context(
    a: "FtmoAssetAnalysis", calendar_events: list | None = None, now_utc: datetime | None = None
) -> str:
    """The Execution Clerk's COMPACT view of one symbol, for its local-model prompts (about a quarter of
    format_ftmo_asset_context). The Clerk only judges an M5 trigger / invalidation line and defends an OPEN position, so it
    gets the M5 decision read (stats, structure, setup read, nearest levels, session levels, alignment, events) and a single
    line of higher-timeframe regime. Dropped, because no Clerk decision uses it: the D1 daily backtest blocks and headlines
    (the verdict prompt fetches its own news), the M5 backtest evidence and behaviour card, the entry trade-zone and sizing
    sheets (entry planning), and the full D1/H4/H1 structure reads. A condition the analyst wrote on H1/H4/D1 gets the full
    context instead (ai.clerk_execution._context_for_condition)."""
    symbol = a.base.symbol
    spread_pct = (a.base.ask - a.base.bid) / a.base.ask * 100 if a.base.ask else 0.0
    lines = [
        f"- {symbol} ({a.base.description}): bid {a.base.bid}, ask {a.base.ask} (spread {spread_pct:.3f}% of price)",
        f"  === M5 DECISION READ for {symbol} - compact Clerk view: the M5 read is the ONLY basis for judging a trigger, an "
        "invalidation condition, a stop or a defense ===",
        format_timeframe_stats(symbol, "M5", a.m5_stats, decision=True),
        format_chart_structure("M5", a.m5_structure, compact=True),
        format_setup_signals("M5", classify_setups(a.m5_stats, a.m5_structure, a.m5_divergence, profile=M5_PROFILE)),
    ]
    lines.extend(_compact_level_lines(a))
    lines.append(format_intraday_levels(symbol, a.intraday_levels))
    lines.append(format_intraday_alignment(a))
    if calendar_events is not None and now_utc is not None:
        event_line = economic_calendar.format_symbol_events(symbol, now_utc, events=calendar_events)
        if event_line is not None:
            lines.append(event_line)
    lines.append(_htf_context_line(a))
    return "\n".join(lines)


# The 4 classify_setups archetypes this account's own prompt text already
# describes as genuine, established directional evidence -- as opposed to
# a fresh single-moment trigger (reversal_candidate, breakout_watch,
# busted_pattern_reversal, candlestick_reversal_confirmed) that's harder
# to silently skim past precisely because it's a rarer, more dramatic
# single-pattern event. A clean, ongoing trend has the opposite problem:
# it spends most of its own life exactly here, with no fresh trigger of
# its own, which is exactly what made it easy to bury inside 17 separate
# ~20-line per-instrument blocks (real incident, 2026-09-05 — see
# build_trend_radar's own docstring).
_TREND_RADAR_SETUP_NAMES = frozenset({"trend_intact", "trend_following", "grind_continuation", "in_progress_move"})


_TREND_TO_DIRECTION = {"uptrend": "up", "downtrend": "down", "flat": "flat"}


def _default_minutes_to_close(symbol: str, now_utc: datetime) -> float | None:
    """Minutes to the instrument's learned session close (Clerk's own estimator, so Mega and Clerk agree on
    when a session is about to end); None on any failure or for a 24h market."""
    try:
        from ai.clerk_execution import _broker_clock_offset, _minutes_to_session_close

        history = fetch_mt5_price_history(symbol, "M5", count=1500)
        return _minutes_to_session_close(history, now_utc, _broker_clock_offset([symbol]))
    except Exception:  # noqa: BLE001 - a missing estimate never vetoes anything
        logger.debug("session-close estimate unavailable for %s", symbol, exc_info=True)
        return None


def _playbook_stop_distance(a: FtmoAssetAnalysis, floor) -> float:
    """The stop distance the with-the-trend playbook would actually use - the opposite side of the 24-bar range clipped to
    [PLAYBOOK_STOP_MIN_ATR, PLAYBOOK_STOP_MAX_ATR] M5 ATRs, never tighter than the broker/spread floor. The hunter used to
    charge cost against the TIGHTEST valid stop (2 ATR), which overstated it 2x for the trades it actually recommends
    (AMD showed 0.29R against ~0.06R at its 4 ATR structure stop) and fed a wrong cost into the R:R arithmetic."""
    atr = a.m5_stats.atr if a.m5_stats is not None else None
    rng = a.m5_range
    if not atr or atr <= 0 or rng is None:
        return floor.distance
    clipped = min(max(rng.high - rng.low, config.PLAYBOOK_STOP_MIN_ATR * atr), config.PLAYBOOK_STOP_MAX_ATR * atr)
    return max(clipped, floor.distance)


def build_hunt_facts(
    a: FtmoAssetAnalysis,
    account_equity: float | None = None,
    calendar_events: list | None = None,
    now_utc: datetime | None = None,
    market_open_fn: Callable[[str, datetime], bool] | None = None,
    minutes_to_close_fn: Callable[[str, datetime], float | None] | None = None,
) -> HuntFacts:
    """Reads one analysis into the plain facts analysis.position_hunter ranks and vetoes on. Every field is
    a real measurement or None: a failed read never becomes a veto or a claim."""
    symbol = a.base.symbol
    now_utc = now_utc or datetime.now(timezone.utc)
    floor = effective_stop_floor(a)
    drag = cost_drag_r(a, _playbook_stop_distance(a, floor)) if floor is not None else None
    structure = a.m5_structure
    has_structure: bool | None = None
    if structure is not None:
        sr = structure.sr_levels
        has_structure = bool(
            (sr is not None and (sr.support_levels or sr.resistance_levels)) or structure.fibonacci is not None
        )
    signals = classify_setups(a.m5_stats, a.m5_structure, a.m5_divergence, profile=M5_PROFILE)
    zones = m5_trade_zone_candidates(a, signals)
    zone_rr = {side: (zone.reward_risk if zone is not None else None) for side, zone in zones.items()}
    edge = {
        side: {key: verdict.verdict for key, verdict in intraday_edge_verdicts(a.intraday_backtests, side).items()}
        for side in ("buy", "sell")
    }
    blackout_reason = None
    if config.EVENT_BLACKOUT_ENABLED and calendar_events:
        status = economic_calendar.blackout_status(symbol, now_utc, calendar_events)
        if status.active:
            blackout_reason = status.reason
    try:
        market_open = (market_open_fn or is_symbol_tradable_now)(symbol, now_utc)
    except Exception:  # noqa: BLE001
        market_open = None
    playbook = {}
    if config.PLAYBOOK_ENABLED:
        for side in ("buy", "sell"):
            playbook[side] = playbook_advise(
                side, a.base.bid, a.base.ask, a.m5_stats.atr, a.m5_stats.adx, a.m5_stats.adx_change_12, a.m5_range
            )
    return HuntFacts(
        symbol=symbol,
        d1_dir=(a.htf_closed or {}).get("d1") or _TREND_TO_DIRECTION.get(a.base.stats.trend),
        h4_dir=(a.htf_closed or {}).get("h4") or _TREND_TO_DIRECTION.get(a.h4_stats.trend),
        m5_dir=aligned_m5_trend_direction(a.m5_stats),
        cost_drag_r=drag,
        has_structure=has_structure,
        zone_rr=zone_rr,
        edge_verdicts=edge,
        blackout_reason=blackout_reason,
        market_open=market_open,
        minutes_to_close=(minutes_to_close_fn or _default_minutes_to_close)(symbol, now_utc) if market_open else None,
        min_viable_pct=min_viable_risk_pct(a, account_equity),
        playbook=playbook,
    )


def build_position_hunt(
    analyses: list[FtmoAssetAnalysis],
    account_equity: float | None = None,
    ftmo_status: FtmoStatus | None = None,
    calendar_events: list | None = None,
    now_utc: datetime | None = None,
    market_open_fn: Callable[[str, datetime], bool] | None = None,
    minutes_to_close_fn: Callable[[str, datetime], float | None] | None = None,
) -> str:
    """The Position Hunt prompt block (analysis.position_hunter): ranked shortlist + hard, objective vetoes.
    Degrades open: any per-symbol failure skips that symbol, never the whole block."""
    from risk.ftmo_rules import DEFAULT_HEADROOM_FRACTION

    facts: list[HuntFacts] = []
    for a in analyses:
        try:
            facts.append(build_hunt_facts(a, account_equity, calendar_events, now_utc, market_open_fn, minutes_to_close_fn))
        except Exception:  # noqa: BLE001
            logger.warning("Position hunt skipped %s (analysis unreadable).", a.base.symbol, exc_info=True)
    if not facts:
        return ""
    budget = None
    if ftmo_status is not None:
        budget = max(ftmo_status.daily_loss_headroom_pct, 0.0) * DEFAULT_HEADROOM_FRACTION
    # UNCAPPED pairs: the prompt's correlation table shows only the 15 strongest, but the hunter's correlation notes must see every
    # |r| >= 0.7 pair (a US-tech + crypto pool easily has more than 15).
    return format_position_hunt(hunt(facts, compute_ftmo_correlation_pairs(analyses, limit=None), risk_budget_pct=budget))


def build_trend_radar(analyses: list[FtmoAssetAnalysis]) -> str:
    """Pure, ZERO-extra-cost summary scan flagging every instrument with a
    genuine M5 directional setup read (H1 or H4 until 2026-09-24, when M5 became the only
    decision timeframe; the H4/H1 alignment stays as a context label), as one compact list placed
    BEFORE the full per-instrument detail below (format_ftmo_asset_
    context) rather than requiring Claude to independently notice one
    inside 17 separate ~20-line blocks.

    Real incident this directly targets, 2026-09-05: a real mega session
    excluded EURUSD and GBPUSD from the mix (99.1% cash overall) despite
    each instrument's OWN already-computed data reading H1 setup read:
    trend_intact (real directional evidence per this account's own prompt
    text) — the final written reasoning for EURUSD cited one data point,
    called it "moot," and then gave no replacement reason at all. The
    underlying evidence was correctly computed; it just never got
    surfaced saliently enough among ~300 lines of per-instrument detail
    to be genuinely engaged with, rather than skimmed past while building
    a mix already anchored on 1-2 "best ideas."

    Deliberately reuses the EXACT SAME classify_setups/classify_long_
    term_alignment calls format_ftmo_asset_context already makes per
    instrument below — no new MT5 fetch, no new AI/network call, just a
    second (millisecond-cheap, pure-Python) pass over data already in
    memory this run. Answers the user's own "use existing architecture
    more efficiently, don't put extra usage on any one" instruction
    directly: this is a prompt-shaping change, not a new pipeline stage,
    new model call, or new OpenRouter/MT5 usage of any kind.

    "" when nothing qualifies (every instrument reads no_clear_setup/
    sideways/choppy this run) — a normal, legitimate outcome some days,
    not suppressed or padded to look non-empty."""
    rows = []
    for a in analyses:
        m5_names = {s.name for s in classify_setups(a.m5_stats, a.m5_structure, a.m5_divergence, profile=M5_PROFILE)}
        trend_setups = sorted(m5_names & _TREND_RADAR_SETUP_NAMES)
        if not trend_setups:
            continue

        h4_trend, h1_trend = a.h4_stats.trend, a.h1_stats.trend
        if h4_trend and h1_trend and h4_trend == h1_trend and h4_trend != "flat":
            mtf = "H4/H1 ALIGNED"
        elif h4_trend == "flat" or h1_trend == "flat":
            mtf = "H4/H1 partial"
        elif h4_trend and h1_trend:
            mtf = "H4/H1 CONFLICTING"
        else:
            mtf = "H4/H1 n/a"

        lt_state, _short, _agreeing, _opposing = classify_long_term_alignment(a)
        lt_label = {
            "structurally_backed": "long-term STRUCTURALLY BACKED",
            "counter_trend_spike": "long-term counter-trend spike",
            "mixed": "long-term mixed",
            "flat": "no long-term backdrop needed (flat short-term)",
            "no_backdrop": "no long-term backdrop available",
            "not_available": "long-term not available",
        }[lt_state]
        rows.append(f"- {a.base.symbol}: {'/'.join(trend_setups)} ({mtf}, {lt_label})")

    if not rows:
        return ""
    return (
        "Trend Radar (computed from this run's own data above, zero extra "
        "cost — every instrument listed here already shows a genuine "
        "directional/trend setup read on M5 (the decision timeframe; the H4/H1 label is its context); this list exists so "
        "none gets silently skimmed past while building the mix). For "
        "EVERY instrument named here, your Asset-Class Outlook section "
        "must address it individually — either include it, even small, "
        "or state the SPECIFIC technical reason it's excluded. A cited "
        "data point you then call irrelevant/moot without a replacement "
        "reason, or silence, is not sufficient:\n" + "\n".join(rows)
    )


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


# Real numeric pairwise correlation for FTMO's own candidate instruments —
# added 2026-08-27, "Phase 2" of the book-wisdom initiative, direct user
# request: risk management and diverse portfolio creation "within the
# broker limitations." Mirrors ai.psx_suggest.compute_correlation_pairs/
# format_correlation_context exactly (same algorithm, same thresholds) —
# deliberately an FTMO-local copy rather than a shared helper: when PSX's
# own version was built, it was written directly in ai/psx_suggest.py
# rather than added to the ai.portfolio_suggest kernel both FTMO and PSX
# already import shared protocol pieces from (build_audit_block,
# build_past_lessons, AUDIT_MODELS) — that kernel holds the shared
# 3-stage draft/audit/revise PROTOCOL, not per-market numeric/enrichment
# logic, which has consistently stayed local to each market's own module.
# Reads FtmoAssetAnalysis's nested `.base.symbol`/`.base.prices` (FTMO
# wraps a PMEX-shape AssetAnalysis under `.base` — see FtmoAssetAnalysis's
# own docstring — unlike PSX's flat `.symbol`/`.prices`). Needs ZERO new
# network calls: `.base.prices` already holds a real ~6-year D1 series
# (_D1_BACKTEST_BARS=1500), fetched once per symbol during
# analyze_ftmo_assets()'s own existing batch pass.
_HIGH_CORRELATION_THRESHOLD = 0.7
_MIN_CORRELATION_OBSERVATIONS = 30  # ~6 trading weeks — mirrors ai.psx_suggest's own threshold
_MAX_CORRELATION_PAIRS_SHOWN = 15


def compute_ftmo_correlation_pairs(
    analyses: list[FtmoAssetAnalysis], limit: int | None = _MAX_CORRELATION_PAIRS_SHOWN
) -> list[tuple[str, str, float]]:
    """Pairwise correlation of daily returns among the Market Watch pool
    analyzed this poll. Returns only pairs with |correlation| >=
    _HIGH_CORRELATION_THRESHOLD, sorted by magnitude descending and
    capped at _MAX_CORRELATION_PAIRS_SHOWN — see this module's own
    comment just above for the full rationale (mirrors ai.psx_suggest's
    identical function)."""
    returns = {}
    for a in analyses:
        prices = a.base.prices.dropna()
        if len(prices) < _MIN_CORRELATION_OBSERVATIONS:
            continue
        returns[a.base.symbol] = prices.pct_change().dropna()

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
    return pairs if limit is None else pairs[:limit]


def format_ftmo_correlation_context(analyses: list[FtmoAssetAnalysis]) -> str:
    pairs = compute_ftmo_correlation_pairs(analyses)
    if not pairs:
        return (
            "Pairwise correlation among the Market Watch instruments analyzed "
            f"above: no pair currently has |correlation| >= {_HIGH_CORRELATION_THRESHOLD} "
            "— no strong diversification-limiting pairs detected from this data "
            "alone (this does not rule out correlation building during a shared "
            "market-wide stress scenario; reason about that separately)."
        )
    lines = [
        "Pairwise correlation among the Market Watch instruments analyzed above "
        f"(daily returns, only pairs with |r| >= {_HIGH_CORRELATION_THRESHOLD} shown) "
        "— holding both sides of a strongly positive pair adds concentrated risk "
        "rather than real diversification; a strongly negative pair can offset "
        "risk if held together:"
    ]
    for sym_a, sym_b, corr in pairs:
        direction = "move together" if corr > 0 else "move opposite each other"
        lines.append(f"- {sym_a} & {sym_b}: r={corr:.2f} ({direction})")
    return "\n".join(lines)


def build_ftmo_summary(
    account: AccountSummary,
    assets: list[MarketAsset],
    ftmo_status: FtmoStatus,
    positions: list[Position] | None = None,
    analyses: list[FtmoAssetAnalysis] | None = None,
    pending_orders: list[PendingOrder] | None = None,
) -> str:
    """`analyses` lets a caller (app.py, for charting) compute
    analyze_ftmo_assets(assets) once and reuse it here, instead of this
    function re-fetching the same H4/H1/contract-spec data internally.

    `pending_orders` (added 2026-08-23, direct user request) surfaces
    outstanding, not-yet-filled GTC limit orders the same way `positions`
    already surfaces filled ones — previously invisible to this prompt
    entirely, even though this account's own suggested trades are placed
    as pending limit orders that can rest unfilled for hours or days."""
    lines = [
        f"Account: balance {account.balance:.2f} {account.currency}, "
        f"equity {account.equity:.2f}, free margin {account.free_margin:.2f}",
        "",
        format_ftmo_status_context(ftmo_status),
        "",
        format_book_wisdom(),
        "",
        format_trend_wisdom(),
        "",
        format_chart_wisdom(),
        "",
        build_macro_snapshot(),
        "",
        _format_ftmo_macro_news_context(),
    ]

    # Economic calendar (data/economic_calendar.py): fetched ONCE here (a
    # disk-cached free feed) and threaded into every symbol's own context
    # block for its currency-specific upcoming-event line. Fail-soft: an
    # unavailable feed yields an explicit "event risk UNKNOWN" line, never
    # a silent omission.
    now_utc = datetime.now(timezone.utc)
    try:
        calendar_events = economic_calendar.fetch_calendar_events(now_utc)
    except Exception:
        logger.warning("Could not load the economic calendar this run (degrading to unknown).", exc_info=True)
        calendar_events = []
    lines += ["", economic_calendar.format_calendar_block(now_utc, events=calendar_events)]

    positions_context = build_positions_context(positions or [])
    if positions_context:
        lines += ["", positions_context]

    pending_orders_context = build_pending_orders_context(pending_orders or [])
    if pending_orders_context:
        lines += ["", pending_orders_context]

    fx_context = build_fx_context(account)
    if fx_context:
        lines += ["", fx_context]

    lines += ["", "Tradable instruments (Market Watch):"]
    if not assets:
        lines.append("- none visible in Market Watch")
    else:
        resolved_analyses = analyze_ftmo_assets(assets) if analyses is None else analyses
        # Placed BEFORE the full per-instrument detail below, not after —
        # see build_trend_radar's own docstring for the real incident
        # this directly targets. "" (nothing genuinely trending this run)
        # omits the block entirely rather than padding it.
        position_hunt = ""
        if config.POSITION_HUNT_ENABLED:
            try:
                position_hunt = build_position_hunt(
                    resolved_analyses, account.equity, ftmo_status, calendar_events, now_utc
                )
            except Exception:  # noqa: BLE001 - degrade to the trend radar, never break the session
                logger.warning("Position hunt failed; falling back to the trend radar.", exc_info=True)
        if position_hunt:
            lines += ["", position_hunt]
        else:
            trend_radar = build_trend_radar(resolved_analyses)
            if trend_radar:
                lines += ["", trend_radar]
        rehunt_block = rehunt.format_rehunt_block(now_utc)
        if rehunt_block:
            lines += ["", rehunt_block]
        lines.append("")
        lines.append(
            format_ftmo_asset_context(
                resolved_analyses, account_equity=account.equity, calendar_events=calendar_events, now_utc=now_utc
            )
        )
        lines.append("")
        lines.append(format_ftmo_correlation_context(resolved_analyses))
        held_sizing_rates = format_ftmo_held_position_sizing_rates(
            positions or [], resolved_analyses, account.equity
        )
        if held_sizing_rates:
            lines += ["", held_sizing_rates]

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


def _write_latest_suggestion(final_answer: str, recheck: tuple | None = None) -> None:
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

    `recheck` = (draft_text, RecheckSheet) from the final live re-check (ai/live_recheck.py): every entry Claude
    CHANGED for a re-checked symbol is verified against a fresh live quote and the real levels the sheet printed,
    and reverted to the draft's levels (or dropped when those are dead) when it does not hold up — so a fabricated
    or off-market level from the last revision can never reach the Clerk. Any failure here leaves the payload as
    parsed (the behaviour before the re-check existed).

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
    # Real bug found live 2026-08-25: this used to be frozenset(allocation.
    # keys()) — ALL symbols, regardless of pct. That's too strict now that
    # a symbol can legitimately get "pct": 0 specifically to CANCEL/close
    # it while being separately re-listed in Pending Setups as a fresh
    # watch idea (e.g. "cancel this stale pending order, replace it with a
    # cleaner trigger once conditions cool") — Claude did exactly this on
    # a real run the same day for BTCUSD, and the pre-existing "already in
    # the allocation block at all" exclusion rejected it as a false
    # duplicate, failing the WHOLE Pending Setups parse and silently
    # discarding that day's entire suggestion (including every other
    # symbol's real reason/invalidation_condition). Only symbols still
    # carrying REAL, nonzero exposure in the allocation block are still
    # off-limits for Pending Setups — a pct: 0 entry has nothing left to
    # conflict with a fresh idea for the same symbol.
    immediate_symbols = frozenset(sym for sym, entry in allocation.items() if entry.pct > 0)
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
                "reason": entry.reason,
                "invalidation_condition": entry.invalidation_condition,
                "entry_mode": entry.entry_mode,
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
                "trigger": s.trigger,
            }
            for s in pending_setups
        ],
    }
    if recheck is not None and recheck[1] is not None:
        try:
            from data.mt5_source import get_market_watch

            draft_text, sheet = recheck
            draft_allocation = parse_final_allocation(draft_text, require_side=True) or {}
            draft_pending = parse_pending_setups(
                draft_text,
                immediate_symbols=frozenset(sym for sym, e in draft_allocation.items() if e.pct > 0),
            ) or []
            fresh_quotes = {a.symbol: (a.bid, a.ask) for a in get_market_watch()}
            notes = live_recheck.apply_recheck_validation(payload, draft_allocation, draft_pending, sheet, fresh_quotes)
            payload["live_recheck"] = {
                "checked_utc": sheet.generated_utc,
                "statuses": {sym: rec.status for sym, rec in sheet.symbols.items()},
                "notes": notes,
            }
        except Exception:
            logger.warning("Live re-check validation failed — writing the suggestion as parsed.", exc_info=True)

    path = Path(config.MEGA_ANALYSIS_LATEST_SUGGESTION_FILE)
    try:
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2))
        os.replace(tmp_path, path)
    except OSError as e:
        logger.warning("Could not write latest-suggestion file %s: %s", path, e)

    # Trade Journal (direct user request 2026-09-19): a purely additive,
    # best-effort side effect — ai.trade_journal.record_proposals already
    # swallows every exception itself and never raises, so this can never
    # affect whether the real suggestion file above got written, and
    # never touches `payload`, `allocation`, or anything this function
    # returns. This is the ONLY place a trade's `proposed` event is ever
    # created (see that module's own docstring for why it can't live
    # inside Clerk's execution loop instead — a rejected/never-triggered
    # suggestion would never reach that point at all).
    trade_journal.record_proposals(payload)


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
    has a separate, dedicated role for FTMO now (the 15-minute clerk/
    executioner, see ai/copilot_execution.py) and must not also
    double as an FTMO auditor, whether this is called from the manual
    "Suggest Portfolio Mix" button (app.py) or the unattended scheduled
    run (ai.mega_analysis.run_mega_analysis) — an earlier fix only
    scoped this to the scheduled path, which was too narrow; both FTMO
    call sites now agree. As of 2026-08-25 this is no longer FTMO-
    specific: PMEX's ai.portfolio_suggest.suggest_portfolio and PSX's
    suggest_psx_portfolio now also pass include_copilot=False at their
    own build_audit_block call sites, direct user request — Copilot has
    no auditor role anywhere in this app for now. build_audit_block's
    own default parameter is still True (see its own docstring); only
    every real caller's explicit override changed.

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
    snapshot_started_utc = datetime.now(timezone.utc)
    _notify(
        "Building past-session context (real market outcomes and audit "
        "lessons from previous FTMO runs)..."
    )
    past_lessons = build_past_lessons(
        fetch_current_price=_fetch_current_ftmo_price, records_dir=records_dir
    )
    # Compulsory, not a toggle — direct user request 2026-09-05: an
    # independent model's own retrospective self-critique of this
    # account's real closed trades (worst loser + any abnormal post-exit
    # price move), concatenated onto the SAME past_lessons text so it
    # reaches the draft prompt, the audit pool, and the resumed-session
    # fallback below exactly the way past-lessons content already does —
    # see ai/curiosity.py's own module docstring for the full design.
    # build_curiosity_report degrades to "" on any failure at any stage,
    # so this can never block or crash a real mega session.
    _notify("Running the compulsory FTMO retrospective self-critique (curiosity function)...")
    curiosity_report = build_curiosity_report(records_dir=records_dir)
    if curiosity_report:
        past_lessons = "\n\n".join(p for p in (past_lessons, curiosity_report) if p)
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
        focus_directives=AUDIT_FOCUS_GROUPS,
    )
    _notify(
        "Audit received — Claude is revising its suggestion..."
        if audit.audit_available
        else "Independent audit wasn't available this run — Claude is re-checking its own draft instead..."
    )

    # Final live re-check (ai/live_recheck.py): real bid/ask, fresh M5 levels and a drift verdict for every drafted
    # entry, measured NOW from the MT5 feed — the draft's own numbers are ~25 minutes old by this point. Degrades
    # open: any failure and the revision below is exactly what it was before this step existed.
    recheck_sheet = None
    try:
        _notify("Re-checking every drafted entry against the live price...")
        recheck_sheet = live_recheck.build_live_recheck(draft, summary, snapshot_utc=snapshot_started_utc)
    except Exception:
        logger.warning("Live re-check unavailable this run (revising without it).", exc_info=True)
    recheck_block = recheck_sheet.prompt_block if recheck_sheet is not None else ""

    role_continued = (
        _ROLE_STAGE2_SYNTHESIZE_CONTINUED if audit.audit_available else _ROLE_STAGE2_SELF_REVIEW_CONTINUED
    )
    lean_revise_prompt = (
        f"{role_continued}\n\n{_INSTRUCTION_TAIL}\n\n"
        f"Independent audit reports on your draft above:\n{audit.block}"
    )
    if recheck_block:
        lean_revise_prompt += f"\n\n{recheck_block}"
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
        if recheck_block:
            revise_prompt += f"\n\n{recheck_block}"
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
        _write_latest_suggestion(final_answer, recheck=(draft, recheck_sheet) if recheck_sheet is not None else None)

    return final_answer
