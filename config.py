import os

from dotenv import load_dotenv

load_dotenv()

USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "0") == "1"

MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_PATH = os.getenv("MT5_PATH")

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
# (WebSearch/WebFetch) on a stronger model, so it needs much more time than
# a plain narration call.
PORTFOLIO_SUGGESTION_TIMEOUT_SECONDS = int(
    os.getenv("PORTFOLIO_SUGGESTION_TIMEOUT_SECONDS", "900")
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
# its own, smaller timeout budget.
PORTFOLIO_REVISION_TIMEOUT_SECONDS = int(
    os.getenv("PORTFOLIO_REVISION_TIMEOUT_SECONDS", "600")
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

# Overrides for risk/rebalance.py's generic MAX_POSITION_COUNT/
# MAX_SYMBOL_EXPOSURE_PCT, specific to FTMO (by explicit user request):
# unlike PMEX's single-market futures book, FTMO's account is meant to
# genuinely diversify across several distinct asset categories (forex,
# metals, commodities/agriculturals, indices, crypto where offered) at
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
# Copilot execution job (ai/copilot_execution.py) never has to
# guess which saved .md record came from the mega session specifically
# vs. a manual "Suggest Portfolio Mix" button click (both land in the
# same FTMO_RECORDS_DIR, indistinguishably, today).
MEGA_ANALYSIS_LATEST_SUGGESTION_FILE = os.getenv(
    "MEGA_ANALYSIS_LATEST_SUGGESTION_FILE", "mega_analysis_latest_suggestion.json"
)

# --- FTMO Copilot execution check (unattended, OS-scheduled) ---
# The "clerk/executioner" half of the boardroom architecture (see
# ai/copilot_execution.py's own module docstring): reads the mega
# session's Pending Setups, checks each one against fresh live MT5
# technicals via GitHub Copilot CLI, and executes a confirmed one with
# position sizing recomputed from live equity — direct user request
# 2026-08-22/23, explicitly with NO new safety cap beyond the three
# gates the manual "Apply Suggestion" dialog already enforces (see
# risk/apply_suggestion.py::check_execution_safety_gates).
#
# How often the execution-check actually does real work — direct user
# request 2026-08-23: originally hourly, but the check itself is cheap
# (a handful of MT5 fetches + a few Copilot calls, not a 35-minute AI
# pipeline), so tightened to check for a triggered Pending Setup more
# often. copilot_execution_job.py's own OS-level Task Scheduler poll
# interval (currently 5 minutes) is unrelated and unchanged — this is
# purely how many of those polls actually turn into a real check versus
# an instant "not due yet" exit.
COPILOT_EXECUTION_CHECK_INTERVAL_MINUTES = int(
    os.getenv("COPILOT_EXECUTION_CHECK_INTERVAL_MINUTES", "15")
)
# Analogous to MEGA_ANALYSIS_GRACE_MINUTES above but keyed to each
# COPILOT_EXECUTION_CHECK_INTERVAL_MINUTES-sized window instead of a
# fixed daily clock time; this window must exceed however often the
# Task Scheduler poll actually runs so at least one poll always lands
# inside it every interval. A window this generous relative to the
# interval is safe, not wasteful — the per-interval dedup marker (see
# is_execution_due) means a real check still only happens once per
# interval regardless of how many polls land inside this window; a wide
# window just gives more polls a chance to catch up if an earlier one
# in the same interval was skipped or blocked.
COPILOT_EXECUTION_GRACE_MINUTES = int(os.getenv("COPILOT_EXECUTION_GRACE_MINUTES", "10"))
# Idempotency/status marker for this job, mirroring
# MEGA_ANALYSIS_STATE_FILE's own role exactly, just for this second job.
COPILOT_EXECUTION_STATE_FILE = os.getenv(
    "COPILOT_EXECUTION_STATE_FILE", "copilot_execution_state.json"
)
# Per-symbol settlement tracking (order_placed/filled/closed_after_fill —
# see ai/copilot_execution.py's own module docstring for why a flat
# "already executed" set isn't enough given open_position places a GTC
# PENDING limit order, never a market order) — reset whenever the mega
# session's own generated_utc changes, i.e. a fresh mega session
# supersedes all prior pending-setup tracking.
COPILOT_EXECUTION_SETTLEMENT_FILE = os.getenv(
    "COPILOT_EXECUTION_SETTLEMENT_FILE", "copilot_execution_settlement.json"
)
# Live, in-progress status for this job — mirrors
# MEGA_ANALYSIS_PROGRESS_FILE's own role/rationale exactly (same standing
# "automated runs must be as visible as a manual button click" principle
# applies here too), just for this second job.
COPILOT_EXECUTION_PROGRESS_FILE = os.getenv(
    "COPILOT_EXECUTION_PROGRESS_FILE", "copilot_execution_progress.json"
)
# User-controlled on/off switch for the whole Copilot Execution Clerk
# role, added 2026-08-23 direct user request (a toggle in the panel's own
# heading): defaults to ENABLED when the file is missing/corrupt, same
# opt-out-not-opt-in posture as MEGA_ANALYSIS_ENABLED_FILE. Checked once,
# inside ai.copilot_execution.run_copilot_execution_check itself, so all
# three of its call sites (the standalone poll, mega_analysis_job.py's
# inline pass, app.py's manual-button inline pass) respect it uniformly.
COPILOT_EXECUTION_ENABLED_FILE = os.getenv(
    "COPILOT_EXECUTION_ENABLED_FILE", "copilot_execution_enabled.json"
)
# User-overridable review frequency (minutes), added 2026-08-23 direct
# user request (a picker in the same panel): falls back to
# COPILOT_EXECUTION_CHECK_INTERVAL_MINUTES above on a missing/corrupt/
# non-positive value — see ai.copilot_execution.read_copilot_execution_
# interval_minutes.
COPILOT_EXECUTION_INTERVAL_FILE = os.getenv(
    "COPILOT_EXECUTION_INTERVAL_FILE", "copilot_execution_interval.json"
)
# A hard ceiling on one execution-check pass — both the standalone
# poll AND the inline pass mega_analysis_job.py fires right after
# a successful run (see its own docstring) use this. Sized generously
# above a realistic worst case (a handful of pending setups, sequential
# MT5 fetches, then PARALLEL Copilot verdict calls each up to
# COPILOT_VERIFICATION_TIMEOUT_SECONDS) but under a whole
# COPILOT_EXECUTION_CHECK_INTERVAL_MINUTES window (600s < 900s at the
# current 15-minute default — the shared EXECUTION_LOCK_PATH is what
# actually keeps an overlapping next poll safe, not this margin alone),
# so a hang here can never make either this job or the daily
# mega-analysis job overrun its own scheduled window.
COPILOT_EXECUTION_RUN_TIMEOUT_SECONDS = int(
    os.getenv("COPILOT_EXECUTION_RUN_TIMEOUT_SECONDS", "600")
)
# Defensive cap on how many not-yet-settled Pending Setups get a live
# Copilot verdict call in a single poll — nothing in the AI's own output
# schema bounds how many it could emit, and each one fans out into a
# real subprocess spawn; excess entries beyond this are logged and
# dropped rather than checked, not silently truncated without a trace.
COPILOT_EXECUTION_MAX_PENDING_SETUPS = int(
    os.getenv("COPILOT_EXECUTION_MAX_PENDING_SETUPS", "10")
)
