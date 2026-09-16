import os

from dotenv import load_dotenv

load_dotenv()

USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "0") == "1"

MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_PATH = os.getenv("MT5_PATH")

# Gates app.py's own login screen (direct user request 2026-09-02,
# alongside setting up Tailscale-based remote/mobile access) — Streamlit
# has no built-in authentication, and this app can place real orders on
# a live funded FTMO account, so reachability alone (even over a private
# Tailscale network) must never be the only thing standing between a
# stranger and the account. Empty/unset (the default) means "no password
# configured yet" -> the gate fails OPEN, same as this app's original
# local-only-no-auth behavior, never a silent bypass of a REAL configured
# one. Set a real value before actually exposing this beyond localhost.
QUANT2_WEBAPP_PASSWORD = os.getenv("QUANT2_WEBAPP_PASSWORD", "")

MAX_LOSS_PCT = float(os.getenv("MAX_LOSS_PCT", "10"))
MAX_POSITION_COUNT = int(os.getenv("MAX_POSITION_COUNT", "6"))
MAX_SYMBOL_EXPOSURE_PCT = float(os.getenv("MAX_SYMBOL_EXPOSURE_PCT", "30"))

# How many Market Watch instruments get full technical+news enrichment in
# the Portfolio Suggestion feature (each one costs 2 network round-trips).
# get_market_watch() already drops expired/no-quote symbols before this
# cap is ever applied, so raising it only means more of the *live* symbols
# get full chart treatment, not that stale contracts start counting.
MAX_ENRICHED_ASSETS = int(os.getenv("MAX_ENRICHED_ASSETS", "20"))
NEWS_HEADLINES_PER_ASSET = int(os.getenv("NEWS_HEADLINES_PER_ASSET", "2"))

# Portfolio Suggestion now lets Claude do real multi-step web research
# (WebSearch/WebFetch), so it needs much more time than a plain narration
# call. Raised from 900s 2026-09-01: a real live run (the first full FTMO
# draft after this machine's Claude Code CLI was freshly (re)installed)
# used the entire 900s budget without completing and was cleanly
# terminated: ai/claude_cli.py's own timeout/process-tree-kill handling
# worked correctly, not a hang that needed manual cleanup. NOTE (found
# 2026-09-07, correcting this comment's own prior claim): this constant
# governs PMEX's ai.portfolio_suggest.suggest_portfolio only, which
# defaults to PORTFOLIO_SUGGESTION_MODEL="opus" — FTMO's own mega session
# is a SEPARATE code path (ai.mega_analysis.run_mega_analysis passes
# model=config.MEGA_ANALYSIS_MODEL, default "sonnet") that merely reuses
# this SAME timeout constant for its own Stage-1 draft call. A real
# 2026-09-07 FTMO timeout at the full 1500s (see that constant's own
# comment below) happened running Sonnet, not Opus — so "a slower/
# deeper-reasoning model" is NOT a reliable explanation for a timeout on
# either pipeline; the real driver is more likely the sheer volume of
# sequential WebSearch/WebFetch calls Stage 1 is instructed to make (one
# per instrument, per linked country/institution) than raw model
# reasoning speed. Mega analysis runs once a DAY (not on a tight poll
# cycle like the Clerk), so there's no real cost to a generous budget
# here — see ai.mega_analysis._RUN_TIMEOUT_SECONDS, raised alongside this
# to give the extra room to actually be usable.
PORTFOLIO_SUGGESTION_TIMEOUT_SECONDS = int(
    os.getenv("PORTFOLIO_SUGGESTION_TIMEOUT_SECONDS", "1500")
)
# Model used for Portfolio Suggestion's deep research pass. Deliberately
# separate from ai/narrate.py, which stays on the CLI's default model —
# the narrator only explains fixed deterministic output, so it doesn't
# need heavier reasoning; this feature does real open-ended research.
PORTFOLIO_SUGGESTION_MODEL = os.getenv("PORTFOLIO_SUGGESTION_MODEL", "opus")

# Revision pass (stage 2) is narrower-scoped than the stage-1 draft — it
# re-runs the failure-mode checklist's targeted searches (FX rate,
# contango/backwardation, risk-free rate) but not stage 1's full
# per-instrument/per-country/per-institution research sweep — so it gets
# its own, smaller timeout budget. Raised proportionally alongside
# PORTFOLIO_SUGGESTION_TIMEOUT_SECONDS 2026-09-01 (see that constant's
# own comment for why) — same underlying model/tool-use latency applies
# here too, just against a narrower research scope.
PORTFOLIO_REVISION_TIMEOUT_SECONDS = int(
    os.getenv("PORTFOLIO_REVISION_TIMEOUT_SECONDS", "900")
)

# OpenRouter (free tier) — plain API, no agentic tools/search, used for the
# pool of free models (see AUDIT_MODELS in ai/portfolio_suggest.py) that
# audit Claude's own draft suggestion after stage 1.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_TIMEOUT_SECONDS = int(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "90"))

# Free-tier OpenRouter models can 429 from a shared pool contended by
# other users (confirmed live, transient — the same model has failed then
# succeeded minutes later, repeatedly, this session). Rather than writing
# an unavailable model off immediately, each one gets rechecked every
# AUDIT_RETRY_INTERVAL_SECONDS, up to a total of AUDIT_RETRY_TIMEOUT_SECONDS,
# before giving up on it — trading worst-case wall-clock time for the
# strongest possible audit (as many of the 5 models weighing in as
# possible), by explicit user request.
AUDIT_RETRY_INTERVAL_SECONDS = int(os.getenv("AUDIT_RETRY_INTERVAL_SECONDS", "60"))
AUDIT_RETRY_TIMEOUT_SECONDS = int(os.getenv("AUDIT_RETRY_TIMEOUT_SECONDS", "600"))

# GitHub Copilot CLI runs one extra audit-pool voice alongside the 10
# OpenRouter models — the only one with real live web access, used to
# independently fact-check the draft's own "External Research Notes"
# list (see ai/copilot_cli.py, build_copilot_verification). Live-verified
# it has no dedicated search-engine tool (only web_fetch/curl), so a
# bounded few-claim check can take a couple of minutes of trial-and-error
# fetching — no retry loop here (unlike AUDIT_RETRY_*): this runs against
# the user's own authenticated account, not OpenRouter's shared free-tier
# pool, so a failure is more likely a genuine timeout than transient
# contention worth retrying.
COPILOT_VERIFICATION_TIMEOUT_SECONDS = int(os.getenv("COPILOT_VERIFICATION_TIMEOUT_SECONDS", "240"))

# Local folder where each Portfolio Suggestion run's full transcript (input
# data, Claude's draft, the audit reports, Claude's final answer) gets
# saved as a Markdown file, like minutes of a meeting, for future reference.
PORTFOLIO_RECORDS_DIR = os.getenv("PORTFOLIO_RECORDS_DIR", "records")

# "Apply Suggestion" places real MT5 orders (see data/mt5_execution.py,
# risk/apply_suggestion.py) — refuses to run unless MT5_SERVER looks like
# a demo account, or this is explicitly set. Deliberately off by default:
# this is the first order-execution code in the app, and it shouldn't
# silently start working the first time someone points MT5_SERVER at a
# real account.
ALLOW_LIVE_EXECUTION = os.getenv("ALLOW_LIVE_EXECUTION", "0") == "1"

# How far (as a %) a suggestion's proposed limit-entry price is allowed to
# sit from the live ask before it gets clamped back to the boundary
# instead of being sent to the broker as-is — a backstop against the AI
# proposing an unreasonable/unachievable price, not a substitute for it
# proposing a sensible one in the first place.
PRICE_SANITY_BAND_PCT = float(os.getenv("PRICE_SANITY_BAND_PCT", "5"))

# Real incident, 2026-09-10: a suggested XAUUSD entry (4415.00, stop
# 4370.00, an intentional 1.38x-ATR/$45 risk) sat resting as gold fell all
# day. Each time it re-priced, PRICE_SANITY_BAND_PCT clamped the ENTRY
# down toward the live market (a buy limit can't price above the current
# ask) while the absolute stop price stayed fixed at 4370.00 — so the
# $45 risk silently shrank to $3.48, then $0.65, on an instrument moving
# $4-7 per MINUTE that hour. The second fill was stopped out in 5 seconds
# — not a real adverse move, just ordinary noise clipping a stop that had
# become accidentally, invisibly tiny. compute_rebalance_plan now re-
# derives the stop distance from the CLAMPED entry (see its own "entry
# clamped" comment) rather than reusing the stale absolute price, which
# fixes this case directly — this floor is the last-resort backstop for
# whatever still slips through (e.g. the ORIGINAL suggested distance was
# already this tight): a % of the sizing price that's comfortably below
# every real, legitimately-tight stop seen live so far (the tightest
# genuine one on record: EURUSD's own 0.15%-of-price stop) but clearly
# above the two degenerate distances that caused this incident (0.08%
# and 0.01%) — a principled first cut, not yet validated across many
# more days/instruments, same "ship a value, then recalibrate against
# real data" process this file's other thresholds went through.
MIN_STOP_DISTANCE_PCT = float(os.getenv("MIN_STOP_DISTANCE_PCT", "0.1"))

# Real incident, 2026-09-11: the mega session revised an already-held
# NVDA position's own stop (a genuine improvement, widening an unsafe
# 0.6x-H1-ATR stop to a real 1.53x) but reported pct=0.02% for it — just
# under the 0.0242% that exact stop distance actually needed to size even
# ONE whole share, given NVDA's own real contract spec (whole shares
# only). Every clerk poll since then failed with "can't afford even the
# minimum 1-lot," leaving the REAL position stuck on its old, wider stop
# and stale, stretched target for HOURS — a beneficial, clearly-intended
# stop-tightening blocked outright by a razor-thin arithmetic miss, not a
# deliberate reduction. ai/ftmo_suggest.py::format_ftmo_held_position_
# sizing_rates now hands the mega session the exact rate to use instead
# of estimating (the real fix, at the source) — this tolerance is the
# code-level backstop for whatever still slips through: when an already-
# held position's own freshly-computed (pre-rounding) lot count is within
# this % of what's ALREADY held, treat it as the same rounding-precision
# issue rather than blocking the whole stop/target revision — a pure
# in-place amend that preserves the real current size is what was
# actually intended, not a silent, invisible-to-everyone failure to
# apply anything at all.
HELD_POSITION_SIZE_TOLERANCE_PCT = float(os.getenv("HELD_POSITION_SIZE_TOLERANCE_PCT", "25"))

# How close (as a %) an already-resting pending order's own price/stop/
# target must be to a fresh target's own numbers before compute_rebalance_
# plan treats them as "unchanged" (hold) rather than "amend" — added
# 2026-08-23 so trivial, insignificant differences (rounding, a slightly
# re-quoted live price) don't trigger a pointless cancel-and-reopen or
# SL/TP-modify every single poll.
AMEND_TOLERANCE_PCT = float(os.getenv("AMEND_TOLERANCE_PCT", "0.05"))

# How long (in hours) a still-unfilled GTC pending entry order is allowed
# to rest before compute_rebalance_plan cancels it unconditionally, even
# if the current mega-session target still matches its terms — added
# after a real incident: every entry is placed GTC (never expires on its
# own), and this account's mega session runs manually/on-demand rather
# than on a fixed schedule, so a session can go days without re-running.
# A silver entry from one evening's "same-session, no overnight hold"
# session sat resting for 23+ hours with nothing re-examining it, then
# filled the next day in the middle of an unrelated flash-crash. Default
# of 24h mirrors "at most one trading day", the same ceiling already used
# throughout the AI's own entry-timing instructions (ai/ftmo_suggest.py).
MAX_PENDING_ORDER_AGE_HOURS = float(os.getenv("MAX_PENDING_ORDER_AGE_HOURS", "24"))

# Pre-weekend pending-order cleanup (added 2026-09-12) — real incident:
# an INTC pending limit order and a USDCHF one both survived into a
# weekend because the only existing cancel trigger (a fresh mega session
# superseding an old one) never fired in time, and once it tried, the
# market was already closed and the cancel silently failed. This is a
# THIRD, independent cancel condition (see risk/apply_suggestion.py::
# compute_rebalance_plan's own docstring) — every non-crypto pending
# order gets cancelled once per Friday, freeing its share of aggregate
# heat for the mega session to redeploy into a weekend-tradable
# instrument if it wants to. Default ON, mirroring CLERK_EXECUTION_
# ENABLED_FILE's own opt-out-not-opt-in posture (this is a defensive
# cleanup, not new unattended trading authority — unlike CLERK_TACTICAL_
# DEFENSE_ENABLED_FILE's deliberately opt-in stance).
CLERK_PRE_WEEKEND_CLEANUP_ENABLED = os.getenv("CLERK_PRE_WEEKEND_CLEANUP_ENABLED", "true").lower() != "false"
# UTC hour on Fridays this fires at (once per Friday — see
# ai.clerk_execution.is_pre_weekend_cleanup_due). A few hours after this
# account's own 14:00 UTC daily mega-session trigger, giving Friday's
# genuine intraday setups a fair run, comfortably before the earliest-
# closing tradable category (US cash equities, ~20:00-21:00 UTC), and
# early enough to leave a real window for a follow-up mega session to
# actually use the freed risk-budget before the weekend is over.
CLERK_PRE_WEEKEND_CLEANUP_HOUR_UTC = int(os.getenv("CLERK_PRE_WEEKEND_CLEANUP_HOUR_UTC", "18"))

# Approximate, documented weekend trading-calendar boundaries used by
# data.mt5_source.is_symbol_tradable_now — there is no real per-symbol
# session-hours API in this MT5 Python package, so these are broker/DST-
# dependent estimates, not exact hours. Forex/exotics close Friday
# evening and reopen Sunday evening UTC; every other category (metals,
# commodities, indices, equities, agriculture) stays closed from Friday
# evening through Monday. Crypto is exempt entirely (always tradable).
WEEKEND_FOREX_CLOSE_HOUR_UTC = int(os.getenv("WEEKEND_FOREX_CLOSE_HOUR_UTC", "21"))
WEEKEND_FOREX_REOPEN_HOUR_UTC = int(os.getenv("WEEKEND_FOREX_REOPEN_HOUR_UTC", "22"))
WEEKEND_OTHER_CLOSE_HOUR_UTC = int(os.getenv("WEEKEND_OTHER_CLOSE_HOUR_UTC", "21"))

# Optional fallback contract-spec source: a CSV dumped *from inside* the
# MT5 terminal by mql5/DumpSymbolSpecs.mq5, read by get_contract_spec()
# only when the live mt5.symbol_info() API call still comes back empty
# after _ensure_symbol_selected's retries. That script has direct access
# to the terminal's own internal symbol database, so it isn't subject to
# the external Python API session's own selection-state quirk at all —
# a genuinely independent second path, not just another retry of the same
# one. Leave unset to disable (get_contract_spec then just returns None
# as it always has). Point it at wherever your terminal's Data Folder
# writes Files (MetaEditor: File > Open Data Folder > MQL5 > Files).
MT5_SYMBOL_SPECS_CSV_PATH = os.getenv("MT5_SYMBOL_SPECS_CSV_PATH", "")

# --- PSX (Pakistan Stock Exchange) research/suggestion avenue ---
# Unlike PMEX (live MT5 account), there's no broker API for this side (the
# user's K-Trade account has no programmatic access) — PSX suggestions are
# a hypothetical, research-only portfolio built from the exchange's own
# public Data Portal (dps.psx.com.pk) plus the same web-research pipeline,
# with no account/position awareness and no execution.
PSX_REQUEST_TIMEOUT_SECONDS = int(os.getenv("PSX_REQUEST_TIMEOUT_SECONDS", "15"))

# How many KSE-100 constituents (ranked by volume) get full technical +
# fundamental enrichment — mirrors MAX_ENRICHED_ASSETS' role for PMEX, just
# scoped to the flagship blue-chip index rather than a user-curated list,
# since PSX has ~490 listed symbols in total and there's no equivalent of
# "what the user put in their own Market Watch" to size the pool by.
MAX_PSX_ENRICHED_ASSETS = int(os.getenv("MAX_PSX_ENRICHED_ASSETS", "20"))

# Separate from PORTFOLIO_RECORDS_DIR so a PSX equity session's past-audit
# lessons never get fed into a PMEX futures audit or vice versa — the two
# markets' failure modes (roll yield/margin vs. dividend/circuit-breaker/
# free-float risk) don't meaningfully transfer between each other.
PSX_RECORDS_DIR = os.getenv("PSX_RECORDS_DIR", "records/psx")

# --- FTMO (prop-firm 1-Stage Challenge, demo account) ---
# A second real MT5 account, added inside the SAME terminal application as
# the original PMEX account (confirmed with the user) — so it reuses
# MT5_PATH (same terminal.exe) but needs its own login/password/server,
# since the MetaTrader5 Python package supports only one active connection
# per process at a time (see data/mt5_source.py::connect's own docstring
# for the live-verified detail). Deliberately unprefixed MT5_PATH/prefixed
# FTMO_MT5_LOGIN etc. mirrors this codebase's existing convention: PMEX is
# the implicit original unprefixed account, PSX/FTMO are the parallel
# `{NAME}_`-prefixed additions.
FTMO_MT5_LOGIN = os.getenv("FTMO_MT5_LOGIN")
FTMO_MT5_PASSWORD = os.getenv("FTMO_MT5_PASSWORD")
FTMO_MT5_SERVER = os.getenv("FTMO_MT5_SERVER")

# Separate from PORTFOLIO_RECORDS_DIR/PSX_RECORDS_DIR for the same reason
# PSX has its own — FTMO's failure modes (prop-firm daily-loss/trailing-
# max-loss/Best-Day-Rule compliance) don't meaningfully transfer to either
# PMEX or PSX's own past-audit lessons.
FTMO_RECORDS_DIR = os.getenv("FTMO_RECORDS_DIR", "records/ftmo")

# ---- Curiosity function (mandatory FTMO retrospective self-critique) ----
# Direct user request 2026-09-05: an evolution of the past-lessons
# mechanism above (build_past_audit_lessons/build_past_outcome_lessons)
# that interrogates a SPECIFIC settled trade's real price action before/
# after it, not just a bare price-delta or a past audit's own critique.
# See ai/curiosity.py's own module docstring for the full design.

# Own dir, separate from FTMO_RECORDS_DIR — this feature's own saved
# report cards must never collide with that dir's own
# portfolio_suggestion_*.md glob.
CURIOSITY_RECORDS_DIR = os.getenv("CURIOSITY_RECORDS_DIR", "records/ftmo_curiosity")

# How far back to pull closed trades each run — one FTMO trading week,
# wide enough to always have something real to critique on a quiet day
# without dredging up month-old trades whose market context has moved on.
CURIOSITY_LOOKBACK_DAYS = int(os.getenv("CURIOSITY_LOOKBACK_DAYS", "7"))

# Bounded pool of most-recent closed trades (win or loss) scored for an
# "abnormal move" — capped so the windowed MT5 fetches this needs (2 per
# trade scored) stay cheap, and so a quiet week's few real trades never
# forces scoring ancient ones just to fill the pool.
CURIOSITY_ABNORMAL_POOL_SIZE = int(os.getenv("CURIOSITY_ABNORMAL_POOL_SIZE", "15"))

# Losses smaller than this are spread/commission-level noise, not a real
# "losing trade" worth forensic attention — skip them entirely rather
# than ever crowning one "the worst loser" on an unusually clean week.
CURIOSITY_MIN_LOSS_USD = float(os.getenv("CURIOSITY_MIN_LOSS_USD", "10"))

# How many hours of PRE-ENTRY history to pull to establish a real ATR
# baseline — wide enough (7 days) to reliably clear analysis.technical's
# own ATR_WINDOW+1=15 H1 bars even across a weekend gap (a trade opened
# right after Sunday's reopen would otherwise come up short).
CURIOSITY_ATR_CONTEXT_LOOKBACK_HOURS = int(os.getenv("CURIOSITY_ATR_CONTEXT_LOOKBACK_HOURS", "168"))

# The forensic window: how many hours AFTER a trade's own close to check
# for a real price move that contradicts what the setup assumed would
# happen next — the same "what did the market actually do afterward"
# question build_past_outcome_lessons asks at session granularity,
# here at single-trade granularity.
CURIOSITY_POST_EXIT_WINDOW_HOURS = int(os.getenv("CURIOSITY_POST_EXIT_WINDOW_HOURS", "24"))

# A post-exit move is "abnormal" (diverged from the setup's own thesis)
# once it exceeds this many pre-entry ATRs — 2.0x is a real, unignorable
# follow-through move, not routine noise (ATR by definition already
# captures routine range). Not yet validated against real live data the
# way RANGE_DIRECTION_Z/TRENDING_EFFICIENCY_RATIO were (see analysis/
# technical.py's own comments on those) — recalibrate against this
# account's real trade history before fully trusting it.
CURIOSITY_ABNORMAL_ATR_MULTIPLE = float(os.getenv("CURIOSITY_ABNORMAL_ATR_MULTIPLE", "2.0"))

# Same resilience pattern as AUDIT_RETRY_INTERVAL_SECONDS/AUDIT_RETRY_
# TIMEOUT_SECONDS above, but a much shorter budget — this is ONE
# sequential model call gating the mega session's own draft-prompt
# build, not a parallel background audit pool, so it must give up and
# degrade gracefully far sooner.
CURIOSITY_RETRY_INTERVAL_SECONDS = int(os.getenv("CURIOSITY_RETRY_INTERVAL_SECONDS", "30"))
CURIOSITY_RETRY_TIMEOUT_SECONDS = int(os.getenv("CURIOSITY_RETRY_TIMEOUT_SECONDS", "150"))
# Deliberately its OWN, shorter timeout — NOT OPENROUTER_TIMEOUT_SECONDS
# (90s, the audit pool's own per-call budget). Real bug found on
# self-review: reusing that 90s value per call, across up to 7 fallback
# models, could let a run where several genuinely hang take 7*90s+ —
# directly defeating CURIOSITY_RETRY_TIMEOUT_SECONDS' own "give up far
# sooner" purpose, since the deadline check only ever stops a NEW
# attempt from STARTING, not one already in flight. A short per-call
# timeout keeps that gap small even in the worst case.
CURIOSITY_MODEL_TIMEOUT_SECONDS = int(os.getenv("CURIOSITY_MODEL_TIMEOUT_SECONDS", "30"))

# --- Missed-opportunity detection (2026-09-09) ---
# Real, motivating incident: INTC and AMD buy-limit orders, both waiting
# for a technical pullback that never came on a genuinely fast-moving
# market — both eventually cancelled with price already well past their
# own original take-profit, having never come anywhere close to filling.
# The existing curiosity function (worst-loser + abnormal-move) had NO
# visibility into this at all, since it only ever examines trades that
# actually opened and closed — a pending order the market left behind
# produces neither a P&L nor an exit price. These three constants mirror
# CURIOSITY_ABNORMAL_POOL_SIZE/_MIN_LOSS_USD/_ABNORMAL_ATR_MULTIPLE's own
# role, just for this third candidate category.
#
# Bounded pool of most-recent cancelled/expired pending orders scored —
# same "cheap, bounded MT5 fetch, not an ancient-history trawl" reasoning
# as CURIOSITY_ABNORMAL_POOL_SIZE.
CURIOSITY_MISSED_OPPORTUNITY_POOL_SIZE = int(os.getenv("CURIOSITY_MISSED_OPPORTUNITY_POOL_SIZE", "15"))
# A cancelled order only counts as a genuine "missed opportunity" worth
# forensic attention once price moved at least this many multiples of
# the pre-placement ATR away from the entry (or through the original
# target) during its resting window — filters out a cancellation that
# was genuinely benign (a fresh mega session simply changed its mind,
# or FTMO compliance headroom forced it) with the market never having
# moved meaningfully at all. Same 2.0x convention as CURIOSITY_ABNORMAL_
# ATR_MULTIPLE, for the same "a real, unignorable move" reasoning — not
# yet validated against a wide real sample, same caveat as that constant.
CURIOSITY_MISSED_OPPORTUNITY_MIN_ATR_MULTIPLE = float(
    os.getenv("CURIOSITY_MISSED_OPPORTUNITY_MIN_ATR_MULTIPLE", "2.0")
)
# How many missed-opportunity candidates get a full forensic dossier per
# curiosity report — same small-and-focused reasoning as max_losers/
# max_abnormal in ai.curiosity.select_curiosity_candidates (default 1
# each) so the eventual report stays readable, not a data dump.
CURIOSITY_MAX_MISSED_OPPORTUNITIES = int(os.getenv("CURIOSITY_MAX_MISSED_OPPORTUNITIES", "1"))

# --- Researcher: a third, independent agent (ai/researcher.py) ---
# Phase 1 build (standalone, not yet wired into the mega session or
# Clerk — see the "Researcher" plan for the full design): fetches real
# Yahoo Finance headlines per Market Watch symbol and has a model
# synthesize a short report + sentiment lean, on its own schedule,
# independent of both the Mega Session (Claude, real token cost, once a
# day) and Clerk (technical-only, no research at all today).
#
# Model backend switched from local Ollama (qwen3:8b/phi4-mini) to a
# free-tier OpenRouter cascade 2026-09-15/16 — direct user decision after
# a real, independently-audited head-to-head found qwen3:8b (with AND
# without thinking enabled) scoring 20-24/50 against Claude's 45-49/50 on
# identical real data, while the SAME free cascade this project's own
# Mega Session audit pass already trusts (ai.portfolio_suggest.
# AUDIT_MODELS) scored 40/50 on the same test — see ai.researcher.
# _run_researcher_model's own docstring for the full real evidence. No
# new config knobs needed for the model roster itself: it reuses
# AUDIT_MODELS directly rather than duplicating it here, so the two
# rosters can never silently drift apart.

# Opt-out, not opt-in — same posture as CLERK_EXECUTION_ENABLED_FILE.
RESEARCHER_ENABLED_FILE = os.getenv("RESEARCHER_ENABLED_FILE", "researcher_enabled.json")
# Once-daily trigger (direct user request 2026-09-15/16, "once per day
# mostly an hour before the USA day open" — replacing an hourly-interval
# schedule that made sense for a local model with no per-call cost, but
# not for a cloud-model cascade with a real per-call time/rate-limit
# budget worth spending once, not 24 times, a day) — same real "fixed
# UTC hour/minute, not local time" reasoning as ai.mega_analysis's own
# MEGA_ANALYSIS_TRIGGER_HOUR_UTC/MINUTE_UTC (see that constant's own
# comment for why: local-time scheduling drifts across DST, and the US/
# EU DST switchover dates don't even line up). 13:30 UTC is this
# account's own already-established real anchor for "NY cash-equity-
# index open" (see MEGA_ANALYSIS_TRIGGER_HOUR_UTC's own comment) during
# EDT (roughly mid-March-early November) — one hour before that is 12:30
# UTC. Like Mega Session's own trigger, this is intentionally NOT DST-
# adjusted (this codebase has no real per-symbol trading-session-
# timezone dependency elsewhere either), so it'll read as ~1h30m before
# open during EST (roughly November-mid-March) instead of exactly 1h —
# a known, accepted drift, not a bug.
RESEARCHER_TRIGGER_FILE = os.getenv("RESEARCHER_TRIGGER_FILE", "researcher_trigger.json")
RESEARCHER_TRIGGER_HOUR_UTC = int(os.getenv("RESEARCHER_TRIGGER_HOUR_UTC", "12"))
RESEARCHER_TRIGGER_MINUTE_UTC = int(os.getenv("RESEARCHER_TRIGGER_MINUTE_UTC", "30"))
# Widened from an hourly job's own 15 min (2026-09-14) now that a missed
# window means missing the WHOLE day, not just one of 24 hourly chances
# — same reasoning as MEGA_ANALYSIS_GRACE_MINUTES, sized larger here
# since Researcher's own in-app trigger (see app.py's own fragment) only
# fires while a real browser tab is connected, so a wider window gives a
# real, meaningfully better chance of catching one within it.
RESEARCHER_GRACE_MINUTES = int(os.getenv("RESEARCHER_GRACE_MINUTES", "60"))
RESEARCHER_STATE_FILE = os.getenv("RESEARCHER_STATE_FILE", "researcher_state.json")
RESEARCHER_PROGRESS_FILE = os.getenv("RESEARCHER_PROGRESS_FILE", "researcher_progress.json")
# Own dir, separate from every other records dir — one saved report per
# symbol per run, globbed and read back by whatever eventually consumes it.
RESEARCHER_RECORDS_DIR = os.getenv("RESEARCHER_RECORDS_DIR", "records/researcher")
RESEARCHER_HEADLINES_PER_SYMBOL = int(os.getenv("RESEARCHER_HEADLINES_PER_SYMBOL", "5"))
# Per-model-call ceiling for the AUDIT_MODELS cascade (see this block's
# own top comment) — live-tested real free-tier response times ranged
# 33-90s per call; 120s gives real headroom for a slower moment without
# being the multi-hundred-second ceiling the old local-thinking-mode
# setup needed.
RESEARCHER_MODEL_TIMEOUT_SECONDS = int(os.getenv("RESEARCHER_MODEL_TIMEOUT_SECONDS", "120"))
# Ceiling on the whole run (all symbols) — sized for a 3-tier cloud-
# model cascade run once a day, not the old hourly local-model job: up
# to 3 * RESEARCHER_MODEL_TIMEOUT_SECONDS per symbol in the worst case
# (every tier failing before the last one answers) across ~17 symbols is
# a real, if unlikely, ~17-minute ceiling; 3600s (1 hour) gives ample
# headroom above that without being unbounded.
RESEARCHER_RUN_TIMEOUT_SECONDS = int(os.getenv("RESEARCHER_RUN_TIMEOUT_SECONDS", "3600"))
# Self-calibration ledger (direct user request 2026-09-15/16, "look for
# more dimensions which can further improve quality") — a small, bounded,
# per-symbol history of past sentiment calls + the real price at call
# time, so each new report can honestly tell the model (and a human
# reader) how its own recent directional calls on THIS symbol have
# actually played out, instead of every new call reading as equally
# trustworthy regardless of any track record. Own file, separate from
# RESEARCHER_STATE_FILE — that file holds only the last run's own
# outcome, not a rolling multi-entry history per symbol.
RESEARCHER_CALIBRATION_FILE = os.getenv("RESEARCHER_CALIBRATION_FILE", "researcher_calibration.json")
# Bounded so the ledger file can't grow forever — 20 entries per symbol
# at the once-daily cadence above is ~20 days of rolling history,
# comfortably enough to judge a real recent track record without an
# unbounded file.
RESEARCHER_CALIBRATION_MAX_ENTRIES_PER_SYMBOL = int(
    os.getenv("RESEARCHER_CALIBRATION_MAX_ENTRIES_PER_SYMBOL", "20")
)
# Minimum real time that must pass before a past calibration entry gets
# judged hit/miss (see ai.researcher._build_track_record_block) — a
# fixed hour count rather than "one interval" now that Researcher runs
# once a day, not hourly: a call needs genuine real market time to play
# out, but requiring a full 24h could make yesterday's call permanently
# un-judgeable if today's run lands even slightly under 24h later (grace-
# window timing variance). 6h is enough real time for a real directional
# move to show, well short of a full day.
RESEARCHER_CALIBRATION_MIN_AGE_HOURS = int(os.getenv("RESEARCHER_CALIBRATION_MIN_AGE_HOURS", "6"))

# Real, round-turn commission rates by asset category — MT5's API has no
# commission field at all (confirmed live: neither symbol_info() nor
# account_info() exposes one), so unlike everything else in
# data/mt5_source.py::get_trade_economics, this can't be fetched live and
# has to be entered here from what's actually visible in the terminal's
# own Symbol Specification window. FX and metals/commodities rates are
# independently confirmed against FTMO's own official trading-update
# blog post (both quoted PER SIDE there, doubled below to round-turn);
# crypto is the user's own terminal reading only, ASSUMED per-side like
# the rest of FTMO's schedule (not independently confirmed — flag this
# if it turns out to be wrong). Indices carry zero commission per FTMO's
# own "Zero Commissions on Indices" post — handled directly in
# ai/ftmo_suggest.py's category lookup, not as a config value here, since
# it's a structural fact of the fee schedule rather than a rate that
# might need per-deployment tuning.
FTMO_COMMISSION_FX_USD_PER_LOT_ROUND_TURN = float(
    os.getenv("FTMO_COMMISSION_FX_USD_PER_LOT_ROUND_TURN", "5.00")
)
FTMO_COMMISSION_METALS_COMMODITIES_PCT_ROUND_TURN = float(
    os.getenv("FTMO_COMMISSION_METALS_COMMODITIES_PCT_ROUND_TURN", "0.0014")
)
FTMO_COMMISSION_CRYPTO_PCT_ROUND_TURN = float(
    os.getenv("FTMO_COMMISSION_CRYPTO_PCT_ROUND_TURN", "0.0650")
)
# Equities ("Equities I CFD" in this account's own Market Watch path —
# confirmed live via records/ftmo/*.md's own "REAL trading cost" lines)
# had NO rate here at all until 2026-08-26 — direct user correction:
# every equity symbol's commission was showing as "unknown" (see
# _ftmo_commission_pct_round_turn's own fallback), which the mega
# session's own category-diversification instructions then read as a
# real cost-data gap, not a zero-cost asset class, and avoided trading
# equities as a result. User-provided rate: 0.002% of notional value
# per lot, round-turn — same percent-of-notional style as metals/
# commodities/crypto above, not a flat $/lot rate like FX.
FTMO_COMMISSION_EQUITIES_PCT_ROUND_TURN = float(
    os.getenv("FTMO_COMMISSION_EQUITIES_PCT_ROUND_TURN", "0.002")
)

# Overrides for risk/rebalance.py's generic MAX_POSITION_COUNT/
# MAX_SYMBOL_EXPOSURE_PCT, specific to FTMO (by explicit user request):
# unlike PMEX's single-market futures book, FTMO's account is meant to
# genuinely diversify across several distinct asset categories (forex,
# metals, commodities/agriculturals, indices, equities, crypto where offered) at
# roughly 1-2 instruments each — that alone can reach 8-10 positions,
# above PMEX's own MAX_POSITION_COUNT=6 default. The per-symbol exposure
# cap is tightened rather than reused as-is, since a wider, more-diversified
# book should also mean no single name dominates it as much as PMEX's
# narrower one might reasonably allow.
FTMO_MAX_POSITION_COUNT = int(os.getenv("FTMO_MAX_POSITION_COUNT", "10"))
FTMO_MAX_SYMBOL_EXPOSURE_PCT = float(os.getenv("FTMO_MAX_SYMBOL_EXPOSURE_PCT", "20"))

# --- FTMO daily "mega market analysis" (unattended, OS-scheduled) ---
# Runs the full FTMO Portfolio Suggestion pipeline (Claude Sonnet draft +
# the entire AUDIT_MODELS pool — Copilot is excluded from FTMO's audit
# pool entirely now, not just here, see
# ai.ftmo_suggest.suggest_ftmo_portfolio's own docstring: it has a
# separate, dedicated role for FTMO and shouldn't also spend its own
# budget auditing) once a day with nobody watching, via a standalone
# script (mega_analysis_job.py) triggered by a Windows
# Scheduled Task — deliberately NOT an in-process background thread,
# confirmed live that a Streamlit script's own top-level code never
# executes at all until a browser session connects, so a thread started
# from inside app.py can't reliably self-arm after an unattended restart.
#
# Trigger time is anchored in UTC, not local PC time or the MT5 broker's
# own server time — local-time scheduling would silently drift by an hour
# across DST transitions (Windows Task Scheduler's local-time triggers
# track wall-clock time, not a fixed UTC instant, and the US/EU DST
# switchover dates don't even line up with each other). 14:00 UTC sits
# right after the NY cash-equity-index open (13:30 UTC), inside the
# London/New York liquidity overlap, with the entire NY session still
# ahead for an intraday-to-1-day hold — the richest, most representative
# window across this account's actual mix (forex majors, metals, US cash
# indices, crypto, agriculturals/commodities).
MEGA_ANALYSIS_TRIGGER_HOUR_UTC = int(os.getenv("MEGA_ANALYSIS_TRIGGER_HOUR_UTC", "14"))
MEGA_ANALYSIS_TRIGGER_MINUTE_UTC = int(os.getenv("MEGA_ANALYSIS_TRIGGER_MINUTE_UTC", "0"))
# mega_analysis_job.py is invoked by a repeating OS-level poll (every 15
# minutes, set at the Task Scheduler level, not here), not a one-shot
# trigger — this window is how long after the trigger instant a poll may
# still treat today as due; it must exceed that 15-minute poll interval
# so at least one poll always lands inside it. A poll landing OUTSIDE
# this window with no successful run yet recorded for today means the PC
# was off/asleep through the whole window — today is treated as missed
# and skipped, not run late, by explicit design.
MEGA_ANALYSIS_GRACE_MINUTES = int(os.getenv("MEGA_ANALYSIS_GRACE_MINUTES", "20"))
# Minimum spacing between mega-analysis attempts within one grace window
# — real, live bug found 2026-08-31: is_due()'s own "not yet run today"
# check only clears once a run reaches status="success" (see _write_
# state above: last_run_date_utc is left unchanged on error/timeout/
# cli_failed, by design, so a poll ~20 min later gets one more real
# chance before the day's slot is treated as missed). That design
# assumed the caller polls at OS-Task-Scheduler-like intervals (many
# minutes apart); app.py's own in-app auto-trigger checks is_due() on a
# 1-SECOND st.fragment tick instead, so a genuinely persistent failure
# (confirmed live: the `claude` CLI unreachable from this process) was
# retried on nearly every single tick — a real retry storm doing a full
# MT5-connect-and-analyze cycle back to back for the whole grace window,
# which is what actually produced the reported "status messages
# dancing"/full-page sluggishness, not a hang. This cooldown restores
# the natural multi-minute spacing an OS-level poll would have given for
# free, without changing is_due()'s own semantics (still one real
# "missed today" outcome if the whole window elapses without success).
MEGA_ANALYSIS_RETRY_COOLDOWN_MINUTES = float(os.getenv("MEGA_ANALYSIS_RETRY_COOLDOWN_MINUTES", "5"))
# Idempotency/status marker shared between mega_analysis_job.py (writes
# it after every attempt) and app.py's countdown display (reads it to
# know whether today's run already happened, and to show the last
# outcome) — see ai/mega_analysis.py.
MEGA_ANALYSIS_STATE_FILE = os.getenv("MEGA_ANALYSIS_STATE_FILE", "mega_analysis_state.json")
# The unattended run's own LIVE, in-progress status — a separate file
# from MEGA_ANALYSIS_STATE_FILE (which only ever reflects the LAST
# completed attempt's outcome). Added 2026-08-22, direct user request:
# the automated run should show the same live step-by-step status the
# manual "Suggest Portfolio Mix" button already shows, the only real
# difference being WHAT triggers it (the scheduler vs. a click), not
# whether it's visible while running. Written by ai.mega_analysis's own
# progress callback on every stage/audit-progress update; read by
# app.py's countdown widget to detect and display a run actively
# happening right now, distinct from "here's how the last one went."
MEGA_ANALYSIS_PROGRESS_FILE = os.getenv("MEGA_ANALYSIS_PROGRESS_FILE", "mega_analysis_progress.json")
# User-controlled on/off switch for the unattended scheduled run, added
# 2026-08-23 direct user request (a toggle next to the manual "Suggest
# Portfolio Mix" button, mutually exclusive with it): defaults to
# ENABLED when the file is missing/corrupt — this is opt-out, not
# opt-in, so a fresh install or a deleted file never silently stops the
# daily run. mega_analysis_job.py checks this before is_due() on every
# poll; app.py's toggle writes it and reads it back as the toggle's
# source of truth (so a totally separate OS process can see the choice).
MEGA_ANALYSIS_ENABLED_FILE = os.getenv("MEGA_ANALYSIS_ENABLED_FILE", "mega_analysis_enabled.json")
# User-overridable daily trigger time (UTC), added 2026-08-23 direct user
# request (a time picker next to the enable/disable toggle): when this
# file is missing/corrupt/out of range, ai.mega_analysis.read_mega_
# analysis_trigger() falls back to MEGA_ANALYSIS_TRIGGER_HOUR_UTC/
# MINUTE_UTC above unchanged, so an env-var-configured default still
# works exactly as before until the user actually picks a new time.
MEGA_ANALYSIS_TRIGGER_FILE = os.getenv("MEGA_ANALYSIS_TRIGGER_FILE", "mega_analysis_trigger.json")
# The real, intended production model for the daily unattended run —
# "sonnet" by default, matching the manual "Suggest Portfolio Mix"
# button's own default. Overridable purely so a manual test run (e.g.
# confirming the scheduler/lock-file fix actually completes end to end)
# can use a cheap/fast model like "haiku" without editing code — remove
# the env override once such a test is done, this should stay "sonnet"
# for every real daily run.
MEGA_ANALYSIS_MODEL = os.getenv("MEGA_ANALYSIS_MODEL", "sonnet")
# The derived, machine-readable artifact ai.mega_analysis.run_mega_analysis
# writes after every successful run — parsed straight from the same
# final-answer text the human-readable session record (a pure-prose .md
# file) already contains, but as real JSON this time
# ({"generated_utc", "immediate_allocation", "pending_setups"}) so the
# Clerk execution job (ai/clerk_execution.py) never has to
# guess which saved .md record came from the mega session specifically
# vs. a manual "Suggest Portfolio Mix" button click (both land in the
# same FTMO_RECORDS_DIR, indistinguishably, today).
MEGA_ANALYSIS_LATEST_SUGGESTION_FILE = os.getenv(
    "MEGA_ANALYSIS_LATEST_SUGGESTION_FILE", "mega_analysis_latest_suggestion.json"
)

# --- FTMO Execution Clerk check (unattended, OS-scheduled) ---
# The "clerk/executioner" half of the boardroom architecture (see
# ai/clerk_execution.py's own module docstring): reads the mega
# session's Pending Setups, checks each one against fresh live MT5
# technicals via a local Ollama model, and executes a confirmed one with
# position sizing recomputed from live equity — direct user request
# 2026-08-22/23, explicitly with NO new safety cap beyond the three
# gates the manual "Apply Suggestion" dialog already enforces (see
# risk/apply_suggestion.py::check_execution_safety_gates).
#
# How often the execution-check actually does real work — direct user
# request 2026-08-23: originally hourly, but the check itself is cheap
# (a handful of MT5 fetches + a few local-model calls, not a 35-minute
# AI pipeline), so tightened to check for a triggered Pending Setup more
# often. clerk_execution_job.py's own OS-level Task Scheduler poll
# interval (currently 5 minutes) is unrelated and unchanged — this is
# purely how many of those polls actually turn into a real check versus
# an instant "not due yet" exit.
CLERK_EXECUTION_CHECK_INTERVAL_MINUTES = int(
    os.getenv("CLERK_EXECUTION_CHECK_INTERVAL_MINUTES", "15")
)
# Analogous to MEGA_ANALYSIS_GRACE_MINUTES above but keyed to each
# CLERK_EXECUTION_CHECK_INTERVAL_MINUTES-sized window instead of a
# fixed daily clock time; this window must exceed however often the
# Task Scheduler poll actually runs so at least one poll always lands
# inside it every interval. A window this generous relative to the
# interval is safe, not wasteful — the per-interval dedup marker (see
# is_execution_due) means a real check still only happens once per
# interval regardless of how many polls land inside this window; a wide
# window just gives more polls a chance to catch up if an earlier one
# in the same interval was skipped or blocked.
CLERK_EXECUTION_GRACE_MINUTES = int(os.getenv("CLERK_EXECUTION_GRACE_MINUTES", "10"))
# Idempotency/status marker for this job, mirroring
# MEGA_ANALYSIS_STATE_FILE's own role exactly, just for this second job.
CLERK_EXECUTION_STATE_FILE = os.getenv(
    "CLERK_EXECUTION_STATE_FILE", "clerk_execution_state.json"
)
# Per-symbol settlement tracking (order_placed/filled/closed_after_fill —
# see ai/clerk_execution.py's own module docstring for why a flat
# "already executed" set isn't enough given open_position places a GTC
# PENDING limit order, never a market order) — reset whenever the mega
# session's own generated_utc changes, i.e. a fresh mega session
# supersedes all prior pending-setup tracking.
CLERK_EXECUTION_SETTLEMENT_FILE = os.getenv(
    "CLERK_EXECUTION_SETTLEMENT_FILE", "clerk_execution_settlement.json"
)
# Where the Obsidian architecture/trade-journal vault lives — a plain
# relative path, same convention as every other path config here, resolved
# against this app's own working directory. See ai.clerk_execution's
# _export_closed_trade_notes: a real closed-trade note is written under
# <this>/Trades/ the moment a position's own settlement state transitions
# from "filled" to "closed_after_fill" — purely a best-effort side effect,
# never allowed to affect an execution decision or fail a poll.
OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "obsidian_vault")
# Opt-out, not opt-in (same posture as CLERK_PRE_WEEKEND_CLEANUP_ENABLED) --
# this is a pure journaling side effect, not new trading authority.
CLERK_VAULT_JOURNAL_ENABLED = os.getenv("CLERK_VAULT_JOURNAL_ENABLED", "true").lower() != "false"
# Live, in-progress status for this job — mirrors
# MEGA_ANALYSIS_PROGRESS_FILE's own role/rationale exactly (same standing
# "automated runs must be as visible as a manual button click" principle
# applies here too), just for this second job.
CLERK_EXECUTION_PROGRESS_FILE = os.getenv(
    "CLERK_EXECUTION_PROGRESS_FILE", "clerk_execution_progress.json"
)
# User-controlled on/off switch for the whole Execution Clerk role,
# added 2026-08-23 direct user request (a toggle in the panel's own
# heading): defaults to ENABLED when the file is missing/corrupt, same
# opt-out-not-opt-in posture as MEGA_ANALYSIS_ENABLED_FILE. Checked once,
# inside ai.clerk_execution.run_clerk_execution_check itself, so all
# three of its call sites (the standalone poll, mega_analysis_job.py's
# inline pass, app.py's manual-button inline pass) respect it uniformly.
CLERK_EXECUTION_ENABLED_FILE = os.getenv(
    "CLERK_EXECUTION_ENABLED_FILE", "clerk_execution_enabled.json"
)
# User-overridable review frequency (minutes), added 2026-08-23 direct
# user request (a picker in the same panel): falls back to
# CLERK_EXECUTION_CHECK_INTERVAL_MINUTES above on a missing/corrupt/
# non-positive value — see ai.clerk_execution.read_clerk_execution_
# interval_minutes.
CLERK_EXECUTION_INTERVAL_FILE = os.getenv(
    "CLERK_EXECUTION_INTERVAL_FILE", "clerk_execution_interval.json"
)
# A hard ceiling on one execution-check pass — both the standalone
# poll AND the inline pass mega_analysis_job.py fires right after
# a successful run (see its own docstring) use this. Originally kept
# under a whole CLERK_EXECUTION_CHECK_INTERVAL_MINUTES window (600s <
# 900s at the 15-minute default) so a hang could never make a run
# overrun into the next scheduled poll — raised to 1500s alongside
# CLERK_LLM_TIMEOUT_SECONDS's own 2026-08-30 increase (300s per call):
# with several real candidates checked in one poll, each potentially
# needing a full primary-timeout-then-backup-fallback pair on this
# machine's CPU-offload-heavy local models, 600s is no longer a
# realistic ceiling for a legitimately-still-working multi-candidate
# run. This CAN now exceed the 900s poll interval on a heavy poll — that
# is safe, not merely tolerated: EXECUTION_LOCK_PATH's PID-liveness-
# aware lock (see job_lock.py) already makes an overlapping next poll
# skip cleanly rather than double-run, exactly the "not this margin
# alone" property this comment used to only mention in passing.
CLERK_EXECUTION_RUN_TIMEOUT_SECONDS = int(
    os.getenv("CLERK_EXECUTION_RUN_TIMEOUT_SECONDS", "1500")
)
# Defensive cap on how many not-yet-settled Pending Setups get a live
# Clerk verdict call in a single poll — nothing in the AI's own output
# schema bounds how many it could emit, and each one fans out into a
# real local-model call; excess entries beyond this are logged and
# dropped rather than checked, not silently truncated without a trace.
CLERK_EXECUTION_MAX_PENDING_SETUPS = int(
    os.getenv("CLERK_EXECUTION_MAX_PENDING_SETUPS", "10")
)
# Same defensive cap, for the OTHER live-Clerk-call phase added
# 2026-08-23: already-settled (order_placed/filled) immediate_allocation
# symbols carrying their own invalidation_condition, re-checked every
# poll the same way a Pending Setup's trigger_condition is. A separate
# constant from CLERK_EXECUTION_MAX_PENDING_SETUPS (not a shared one)
# so the two mechanisms stay independently tunable and testable.
CLERK_EXECUTION_MAX_WATCHED_POSITIONS = int(
    os.getenv("CLERK_EXECUTION_MAX_WATCHED_POSITIONS", "10")
)
# Same defensive cap, for the tactical-defense candidate phase added
# 2026-08-27: every already-FILLED immediate_allocation/fired-Pending-
# Setup symbol gets a live tactical DEFEND/EXIT check every poll,
# independent of whether it carries an invalidation_condition (see
# ai.clerk_execution's own docstring for the real gold-trade incident
# that motivated this new, separate authority).
CLERK_EXECUTION_MAX_TACTICAL_CANDIDATES = int(
    os.getenv("CLERK_EXECUTION_MAX_TACTICAL_CANDIDATES", "10")
)
# User-controlled on/off switch for the Clerk's tactical-defense
# authority (DEFEND/EXIT on an already-filled position's short-term
# "trend" read), added 2026-08-27. Defaults to DISABLED when the file is
# missing/corrupt — a deliberate break from this module's own usual
# opt-out-not-opt-in convention (compare CLERK_EXECUTION_ENABLED_FILE
# above), since this is fresh unattended authority over real money, not
# yet proven live — see ai.clerk_execution.read_tactical_defense_
# enabled and this project's own staged shadow-mode rollout plan.
CLERK_TACTICAL_DEFENSE_ENABLED_FILE = os.getenv(
    "CLERK_TACTICAL_DEFENSE_ENABLED_FILE", "clerk_tactical_defense_enabled.json"
)
# Cooldown/no-thrash guardrail for a DEFEND action re-firing on the same
# symbol: a new DEFEND inside this many minutes of the last one is
# suppressed UNLESS the position's adverse move has also worsened by at
# least CLERK_TACTICAL_DEFEND_MIN_RETRIGGER_PCT since that last action
# (both required together) — see ai.clerk_execution._validate_and_
# apply_tactical_verdict's own docstring.
CLERK_TACTICAL_DEFEND_COOLDOWN_MINUTES = int(
    os.getenv("CLERK_TACTICAL_DEFEND_COOLDOWN_MINUTES", "60")
)
CLERK_TACTICAL_DEFEND_MIN_RETRIGGER_PCT = float(
    os.getenv("CLERK_TACTICAL_DEFEND_MIN_RETRIGGER_PCT", "0.3")
)
# Sane bounds on a DEFEND verdict's own proposed partial-close fraction —
# parse_tactical_verdict REJECTS the whole verdict (never silently
# clamps) when the model proposes a fraction outside this range.
CLERK_TACTICAL_MIN_PARTIAL_CLOSE_FRACTION = float(
    os.getenv("CLERK_TACTICAL_MIN_PARTIAL_CLOSE_FRACTION", "0.10")
)
CLERK_TACTICAL_MAX_PARTIAL_CLOSE_FRACTION = float(
    os.getenv("CLERK_TACTICAL_MAX_PARTIAL_CLOSE_FRACTION", "0.75")
)
# Deterministic tactical pre-screen thresholds, added 2026-08-30 —
# Python-side numbers computed every poll and handed to the Clerk's LLM
# so it weighs already-computed candidates instead of deriving arithmetic
# itself (a real, observed failure mode: the backup model once proposed
# LOOSENING a stop while calling it "tightening"). See ai.clerk_execution.
# TacticalSignals/_compute_tactical_signals's own docstrings.
#
# O'Neil's hard stop-loss ceiling ("no exceptions") — the one rule here
# that bypasses the LLM call entirely as a genuine circuit-breaker, same
# category as risk.apply_suggestion.check_execution_safety_gates.
CLERK_TACTICAL_HARD_EXIT_PCT = float(os.getenv("CLERK_TACTICAL_HARD_EXIT_PCT", "7.0"))
# O'Neil's profit-lock trigger: tighten the stop once a position's
# favorable move reaches this percentage.
CLERK_TACTICAL_PROFIT_LOCK_PCT = float(os.getenv("CLERK_TACTICAL_PROFIT_LOCK_PCT", "15.0"))
# Bulkowski's ATR-multiple stop distance — the "slow"-tier (most FX
# majors/crosses) default; see CLERK_TACTICAL_ATR_STOP_MULTIPLE_FAST
# below for the "fast" tier (gold/crypto/equities-style bursty movers).
CLERK_TACTICAL_ATR_STOP_MULTIPLE = float(os.getenv("CLERK_TACTICAL_ATR_STOP_MULTIPLE", "1.5"))
# Added 2026-09-09, real incident: a real XAUUSD trade's stop sat at
# well under half the instrument's own typical H1 range and was clipped
# within 93 minutes by perfectly ordinary noise, with its own stated
# invalidation condition never actually broken — see analysis.technical.
# classify_velocity_tier's own comment for the full real-data comparison
# this is built from. A wider multiple for an instrument classified
# "fast" by its own current H1 atr_pct gives ordinary noise the room a
# slower instrument's own noise never needed in the first place — this
# is a real, live-money adjustment (not yet tuned against a wide sample
# the way CLERK_TACTICAL_ATR_STOP_MULTIPLE's own 1.5x convention was),
# so recalibrate this value against real outcomes before fully trusting
# the exact number.
CLERK_TACTICAL_ATR_STOP_MULTIPLE_FAST = float(os.getenv("CLERK_TACTICAL_ATR_STOP_MULTIPLE_FAST", "2.5"))
# Schwager's partial-profit trigger: take partial profits when a position
# has captured at least this share of its expected move (take-profit
# distance) within CLERK_TACTICAL_PARTIAL_PROFIT_MAX_DAYS of being opened.
CLERK_TACTICAL_PARTIAL_PROFIT_TARGET_PCT = float(
    os.getenv("CLERK_TACTICAL_PARTIAL_PROFIT_TARGET_PCT", "50.0")
)
CLERK_TACTICAL_PARTIAL_PROFIT_MAX_DAYS = float(
    os.getenv("CLERK_TACTICAL_PARTIAL_PROFIT_MAX_DAYS", "7")
)

# Local Ollama models powering the Clerk's own LLM calls (verdict,
# invalidation, and tactical-defense checks) — direct user request
# 2026-08-30, replacing GitHub Copilot CLI + an OpenRouter backup chain
# entirely (see ai/clerk_execution.py's own module docstring for the two
# real, already-observed failure modes that motivated this: Copilot's
# monthly CLI quota exhausted 2026-08-25, and the whole 10-model
# OpenRouter pool going unavailable AT ONCE 2026-08-27). Both run via a
# local Ollama server (ai/ollama_client.py) — no API key, no shared rate
# limit, no per-token cost.
#
# qwen3:8b is primary as of 2026-08-31 (direct user request, after
# gemma4:12b proved unsuitable for this machine's 4GB-VRAM GPU — see
# CLERK_LLM_TIMEOUT_SECONDS's own history below for the real multi-
# candidate timeout incident that motivated re-testing). Live-tested
# with thinking DISABLED: 5.9s cold load (vs gemma4:12b's 11.5s), and
# 3/3 correct, consistently-cited DEFEND verdicts on a real profit-lock/
# partial-profit scenario (Schwager's rule, correct 0.10-0.75-bounded
# partial-close fractions every time) — a real improvement over both the
# original qwen2.5:7b A/B test (which once got a stop-tightening
# DIRECTION backwards) and gemma4:12b's own CPU-offload-driven latency
# on this specific GPU (only ~24-38% of either gemma4 variant fit in
# 4GB VRAM, vs qwen3:8b's ~39%, and gemma4's absolute model size means
# far more of it lands on the slow CPU path regardless of percentage).
#
# One real, honestly-noted nuance from the same testing (2/2 trials on a
# separate neutral scenario): qwen3:8b showed some tendency to DEFEND
# (tighten a stop) even short of the profit-lock threshold, citing
# Bulkowski's ATR-floor rule somewhat loosely rather than staying
# perfectly disciplined to the pre-screen's own due/not-due flags.
# Judged an acceptable tradeoff, not a safety issue — it never proposed
# widening a stop or acting on the wrong side of price, and
# _validate_and_apply_tactical_verdict's own independent never-widen-
# stop/sane-side-of-price checks (ai/clerk_execution.py) reject an
# unsafe DEFEND regardless of what any model proposes.
CLERK_PRIMARY_MODEL = os.getenv("CLERK_PRIMARY_MODEL", "qwen3:8b")
# phi4-mini is the backup as of 2026-08-31 (direct user request,
# replacing gemma4:12b) — direct user correction: a backup that's too
# slow to finish inside this job's own real timeframe defeats the whole
# point of having one, so "most accurate in isolation" isn't enough on
# its own; the backup specifically needs to also be FAST. Live-tested:
# 5.1s cold load (fastest of everything tried this session) and ~64% of
# it fits in this machine's 4GB VRAM (only ~1.3GB on the slow CPU path —
# also the best fit tested, well ahead of qwen3:8b's own ~39%). 2/3
# DEFEND-scenario trials parsed as correct, well-cited verdicts; the
# third had a formatting slip (echoed the prompt's own "-- or --"
# template text, producing two FINAL_VERDICT tokens) that
# parse_tactical_verdict's own "last token wins" rule resolved to a
# fail-safe HOLD — a missed action, never a wrong or dangerous one, so
# judged an acceptable trade for a fallback role: gemma4:12b was
# strictly more reliable in every test but too slow to serve as a
# backup at all (11.5s load alone, up to 76% CPU-offloaded) — removed
# from this machine's Ollama install entirely, no longer referenced
# anywhere in this roster.
CLERK_BACKUP_MODEL = os.getenv("CLERK_BACKUP_MODEL", "phi4-mini")
# Per-call timeout for a Clerk LLM call. Originally sized at 150s from
# isolated 30-70s single-call tests (2026-08-29/30), but a real multi-
# candidate poll (2026-08-30, 4 pending setups checked in one run) hit 3
# read-timeouts on gemma4:12b at 150s each. Root-caused via Ollama's own
# /api/ps: gemma4:12b (8.9GB loaded) only fits ~24% (2.1GB) on this RTX
# 500 Ada's 4GB VRAM, running the other ~76% (6.8GB) on CPU — real GPU
# capacity math, not a connectivity or quota issue. Two effects compound
# under this constraint: (1) a fully unloaded model pays an ~11.5s cold-
# load tax (measured live) before any inference starts, and (2) Ollama
# serializes generation on its one shared server, so with N candidates
# submitted concurrently via the ThreadPoolExecutor, candidate 2+'s own
# 150s timer was burning down while it sat queued behind candidate 1's
# still-running CPU-heavy call, before candidate 2's own inference had
# even begun. 300s gives real headroom for cold-load + queue-wait + the
# CPU-bound majority of the model actually generating a full verdict —
# see CLERK_EXECUTION_RUN_TIMEOUT_SECONDS above, raised alongside this so
# the outer per-run ceiling doesn't kill a check that's still making
# real progress within this larger per-call budget.
CLERK_LLM_TIMEOUT_SECONDS = int(os.getenv("CLERK_LLM_TIMEOUT_SECONDS", "300"))
