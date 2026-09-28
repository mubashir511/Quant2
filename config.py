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

# Real gap, 2026-09-17 (direct user request): the Clerk's pending-setup
# verdict prompt has invited "use your own live web access ... e.g.
# checking for any major news" since before the 2026-08-30 switch to
# local Ollama models — neither qwen3:8b nor phi4-mini can browse the
# web at all, so that instruction has been quietly unfulfillable for
# weeks (see ai.clerk_execution._run_clerk_prompt's own docstring, which
# already documented this as a known, accepted trade-off). Fixed by
# reusing the same lightweight fetch_recent_headlines() path the mega
# session itself already relies on (ai.portfolio_suggest), NOT the
# Researcher's heavier multi-source fetch — the Clerk's own configurable
# poll interval (read_clerk_execution_interval_minutes(), currently 5
# minutes but user-adjustable at any time) means a per-poll Yahoo round
# -trip per symbol would multiply real request volume (and the already-
# documented yfinance hang risk, see data.news_source's own
# _NEWS_TIMEOUT_SECONDS incident) far more than the once-daily
# Researcher ever does. This cache is deliberately a plain wall-clock
# TTL, not tied to poll count, specifically so real request volume stays
# capped at this same rate no matter how short the poll interval is set
# to — headlines don't meaningfully change minute to minute anyway, so
# the same fetch is reused across many consecutive
# polls instead of hitting Yahoo fresh every time.
CLERK_NEWS_CACHE_MINUTES = int(os.getenv("CLERK_NEWS_CACHE_MINUTES", "20"))

# --- data/symbol_news.py — the ONE shared per-symbol news fetch/cache,
# added 2026-09-20 direct user request ("save double calling of news"):
# Clerk, Researcher, Mega Session, and the webapp's own News section all
# now read through this single cache instead of each doing its own
# independent fetch. Real bug this closes, found live while building it:
# Clerk's OWN prior news fetch used data.underlying.resolve_yahoo_ticker
# (a PMEX-symbol keyword map) instead of ai.researcher's own already-
# correct, FTMO-native resolve_ftmo_yahoo_ticker — confirmed live against
# this account's real current mix: 17 of 22 symbols (77%) silently
# resolved to NO Yahoo ticker at all under the old resolver (every
# equity/forex/crypto symbol; only metals and a couple of commodities
# happened to work by coincidence, since their MT5 descriptions literally
# contain a PMEX keyword like "Gold"). Consolidating onto the shared,
# already-correct resolver fixes this for Clerk as a side effect, not
# just deduplicates the network calls.
#
# Same 20-minute default as CLERK_NEWS_CACHE_MINUTES above (that
# constant is now unused — kept only so its own historical tests/comment
# aren't disturbed — direct user confirmation this value is fine to
# reuse for the shared cache too: "I think we have already set it 20min
# last time for clerk... no issue with that").
SYMBOL_NEWS_CACHE_MINUTES = int(os.getenv("SYMBOL_NEWS_CACHE_MINUTES", "20"))

# Where the real, per-symbol news history lives — one JSON file per
# symbol (records/ftmo_symbol_news/{symbol}.json), the single source of
# truth; the Obsidian vault note for the same symbol (obsidian_vault/
# News/{symbol}.md) is a deterministic RENDER of it, regenerated on every
# new item — same local/gitignored/account-independent convention as
# TRADE_JOURNAL_DIR.
SYMBOL_NEWS_DIR = os.getenv("SYMBOL_NEWS_DIR", "records/ftmo_symbol_news")

# Same convention as SYMBOL_NEWS_DIR, for the category-specialty layer
# (FXStreet/CoinDesk/Investing.com) — added 2026-09-20, direct user
# report that category news had no vault presence at all before this
# (records/ftmo_category_news/{category}.json -> obsidian_vault/News/
# Category/{category}.md), unlike the per-symbol layer.
CATEGORY_NEWS_DIR = os.getenv("CATEGORY_NEWS_DIR", "records/ftmo_category_news")

# Rolling retention window, direct user request ("save... for one
# month... after one month the vault will be reset") — every time a
# symbol's news store is touched, entries older than this many days are
# pruned, so the note always shows a trailing month rather than growing
# forever or being wiped all at once on a fixed calendar date.
SYMBOL_NEWS_RETENTION_DAYS = int(os.getenv("SYMBOL_NEWS_RETENTION_DAYS", "30"))

# Real bug found 2026-09-20, direct user challenge ("i don't want to
# explode the usage quota... neither do i want any analysis to go
# without the news context"): Clerk/Researcher/Mega/Trade Audit each run
# as their OWN standalone spawned process per tray-timer tick (see
# quant_app_tray.ps1's fire-and-forget Process.Start() per job), so the
# in-memory _symbol_news_cache/_category_news_cache/_macro_news_cache
# dicts in data/symbol_news.py reset to empty on every single invocation
# -- they only ever deduped calls made within ONE process's lifetime
# (e.g. the long-running webapp), never across the 4 real consumers.
# This file is a small disk-backed mirror of those same caches so a real
# fetch made by any one process is visible to every other process for
# the rest of SYMBOL_NEWS_CACHE_MINUTES, without changing that TTL or
# ever serving stale-past-TTL data.
NEWS_FETCH_CACHE_FILE = os.getenv("NEWS_FETCH_CACHE_FILE", "records/news_fetch_cache.json")

# Economic calendar (data/economic_calendar.py, 2026-09-24 intraday
# decision-tier upgrade): the free weekly JSON feed is cached on disk and
# shared by every process. The blackout window keeps Clerk from placing or
# leaving RESTING entry orders from `BEFORE` minutes ahead of a High-impact
# event for the symbol's own currencies until `AFTER` minutes past it.
ECONOMIC_CALENDAR_CACHE_FILE = os.getenv("ECONOMIC_CALENDAR_CACHE_FILE", "records/economic_calendar_cache.json")
ECONOMIC_CALENDAR_CACHE_MINUTES = int(os.getenv("ECONOMIC_CALENDAR_CACHE_MINUTES", "60"))
EVENT_BLACKOUT_ENABLED = os.getenv("EVENT_BLACKOUT_ENABLED", "1") not in ("0", "false", "False")
EVENT_BLACKOUT_BEFORE_MINUTES = int(os.getenv("EVENT_BLACKOUT_BEFORE_MINUTES", "30"))
EVENT_BLACKOUT_AFTER_MINUTES = int(os.getenv("EVENT_BLACKOUT_AFTER_MINUTES", "10"))
EVENT_BLACKOUT_IMPACTS = tuple(
    s.strip().title() for s in os.getenv("EVENT_BLACKOUT_IMPACTS", "High").split(",") if s.strip()
)

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

# Real gap, 2026-09-16: Bulkowski's 1.5x-H1-ATR stop floor already exists
# as a named audit critique (ai/ftmo_suggest.py's own AUDIT_INSTRUCTION
# "Stops Systematically < 1.5x ATR" flaw) but was never enforced in code
# before this — see ai.clerk_execution._apply_atr_stop_floor_guard. Real
# incident this closes: a real EURUSD entry was independently audited at
# 0.57x H1 ATR against the 1.5x floor, explicitly predicted in that same
# audit to be a "high probability of noise stop-out," was executed
# anyway, and was stopped out on ordinary noise exactly as predicted
# (records/ftmo/portfolio_suggestion_2026-09-07_122716.md:683-684).
# Deliberately a WIDEN, not a reject — mirrors MIN_STOP_DISTANCE_PCT's
# own precedent immediately above: a trade whose direction/entry is
# otherwise sound shouldn't be discarded over a fixable stop distance.
ENTRY_ATR_STOP_FLOOR_MULTIPLE = float(os.getenv("ENTRY_ATR_STOP_FLOOR_MULTIPLE", "1.5"))

# Real gap, 2026-09-17 (direct user request): the mega session's own
# win-rate/avg-R/favorable-excursion backtest evidence — the numbers
# actually used to judge whether a take-profit is realistic — runs on
# DAILY bars with a 10-bar (~2 calendar week) holding cap and a 90-bar
# (~4.5 month) excursion horizon (see analysis/backtest.py's own
# TRADE_SIM_MAX_HOLDING_BARS/TRADE_SIM_EXCURSION_HORIZON_BARS). This
# account holds intraday only, at most one trading day (see
# ai/ftmo_suggest.py's own HOLDING HORIZON instruction) — the backtest
# evidence itself was never recalibrated to match, so every TP/SL
# realism check was effectively grounded in weeks-to-months of real price
# travel, not the few hours this account actually holds for.
#
# M15 (not H1) is the real intraday-evidence timeframe: originally
# assumed unavailable in this MT5 Python package (data/mt5_source.py's
# own `_MT5_TIMEFRAMES` only listed H1/H4/D1/MN1) — live-verified
# otherwise, on direct user push-back, that this was this codebase's own
# subset choice, not an MT5 limitation (both the installed and latest
# PyPI package builds genuinely expose TIMEFRAME_M1 through M30, and
# real M15 history goes back 1.5-5.8 years across every asset class this
# account trades — confirmed live via copy_rates_from_pos, not assumed).
#
# TRADE_SIM_INTRADAY_MAX_HOLDING_BARS default (16 M15 bars = ~4 hours) is
# empirically derived, not guessed: pulled this account's own real closed
# -trade history (data.mt5_source.group_closed_trades over the last 180
# days, 27 real trades) — median hold 2.31h (~9 M15 bars), P75 3.71h
# (~15 bars), only 2/27 (7.4%) ran past 24h, confirming "closed same
# session" really is the norm here. 16 sits just above the real P75,
# without drifting into the small, real overnight-exception tail (P90
# jumped to 23.4h) — that tail is exactly what the existing D1 evidence
# and the account's own explicit "overnight exception, only when
# justified" rule already cover separately.
# Re-based 2026-09-24 when the M15 tier was dropped (M5 + H1 split its duties): 48 M5 bars = the same
# ~4 hours the original 16 M15 bars covered.
TRADE_SIM_INTRADAY_MAX_HOLDING_BARS = int(os.getenv("TRADE_SIM_INTRADAY_MAX_HOLDING_BARS", "48"))
# Same 9x-the-holding-cap convention analysis/backtest.py's own existing
# 90/10 D1 pair already uses (TRADE_SIM_EXCURSION_HORIZON_BARS is 9x
# TRADE_SIM_MAX_HOLDING_BARS) — applied to the new intraday figure
# instead of an unrelated, separately-guessed number.
TRADE_SIM_INTRADAY_EXCURSION_HORIZON_BARS = int(
    os.getenv("TRADE_SIM_INTRADAY_EXCURSION_HORIZON_BARS", str(TRADE_SIM_INTRADAY_MAX_HOLDING_BARS * 9))
)
# 5,000 M15 bars — confirmed live to FETCH in well under 20ms per symbol
# and to comfortably span multiple real months of history, far more than
# the RSI/S-R sample-size gates in analysis/backtest.py actually need
# (real EURUSD check: 155/168 RSI-reaction episodes at this size). NOT
# increased further to chase chart-pattern coverage (see ai.ftmo_suggest.
# _compute_intraday_backtests' own docstring for why that backtest is
# skipped for the intraday variant entirely) — pattern detection's own
# compute cost, not the fetch, is what makes a larger window impractical
# (confirmed live: 16.65s at 15,000 bars, 29.33s at 25,000, PER SYMBOL,
# for backtest_chart_pattern_reaction alone — multiplied across ~20
# Market Watch symbols this would add many real minutes to a mega-session
# run for a figure that still needed 40,000 bars to reliably populate).
# Re-based to M5 on 2026-09-24 (the M15 tier was dropped): measured on real bars, 15,000 M5 bars gave
# 450-490 RSI episodes per side but cost ~0.7s per symbol (10x the old M15 figure); 6,000 bars still gives
# ~190 episodes per side (well above the 20-resolved-trade gate) at ~0.3s. Name kept as the historical
# "intraday backtest" count.
M5_BACKTEST_BARS = int(os.getenv("M5_BACKTEST_BARS", "6000"))
# Real gap found live shipping this feature: analysis/backtest.py's own
# _SR_PROXIMITY_PCT (2.0%) is implicitly calibrated for DAILY bars — a
# 60-bar rolling S/R window spans ~15 real hours on M15 vs. ~3 real
# months on D1, so a 2%-of-price proximity band that's a genuine "near
# the level" test on D1 is wide enough on M15 that price sits inside it
# almost continuously, collapsing what should be many distinct episodes
# into one — the backtest returned None (insufficient independent
# episodes) at 2.0% AND 1.0% on real EURUSD M15 data. Re-derived
# empirically rather than guessed: swept 2.0% down to 0.05% against real
# EURUSD M15 bars; 0.5% was the first value giving a healthy, clearly-
# distinct episode count on both sides (9 support/18 resistance tests at
# M15_BACKTEST_BARS=5000) without over-counting noise the way sub-0.1%
# values started to (150+ "tests" each side — no longer a meaningfully
# distinct "near the level" event). backtest_support_resistance_reaction's
# own `proximity_pct` parameter already supports this override — no code
# change needed there, only this value passed in for the intraday call.
# On M5 a FLAT proximity is instrument-dependent (measured on 15,000 real M5 bars: 0.5% left EURUSD with
# 23/5 tests but MSFT with 525/531, 0.1% gave EURUSD 317/328), so the intraday S/R backtest now uses an
# ATR-relative proximity: this many multiples of the instrument's own median M5 ATR%, clamped to a range.
M5_SR_PROXIMITY_ATR_MULTIPLE = float(os.getenv("M5_SR_PROXIMITY_ATR_MULTIPLE", "4.0"))
M5_SR_PROXIMITY_MIN_PCT = float(os.getenv("M5_SR_PROXIMITY_MIN_PCT", "0.1"))
M5_SR_PROXIMITY_MAX_PCT = float(os.getenv("M5_SR_PROXIMITY_MAX_PCT", "0.5"))
# Real, explicit off-switch for the whole intraday-evidence feature,
# mirroring this account's existing enable/disable toggle pattern (mega
# session/Clerk execution/Researcher each have one) — cheap insurance on
# a change to the core TP/SL evidence pipeline.
INTRADAY_BACKTEST_ENABLED = os.getenv("INTRADAY_BACKTEST_ENABLED", "true").lower() == "true"

# Real incident, 2026-09-17: a real INTC pending order came out at exactly
# 1:1 net reward:risk (risk 4.44, reward 4.44) — traced to two genuinely
# CORRECT mechanisms colliding, not a bug: INTC's own H1 ATR forced a
# stop ~2x that ATR (correct — tighter gets noise-stopped on a fast
# mover), and the account's own same-session realism ceiling caps the
# target at ~1.5-2x that SAME ATR (correct — farther isn't reachable in
# one session). When both land on the same ATR multiple, ~1:1 is the
# best MATHEMATICALLY achievable ratio — no cleverer entry/stop/target
# choice fixes it. The account's own cited backtest for this exact setup
# (support-bounce, 38% win rate) was measured at a 2:1 target; the SAME
# 38% win rate at 1:1 is NEGATIVE expected value (0.38*1 - 0.62*1 =
# -0.24R). The existing instruction text (ai/ftmo_suggest.py:587-623)
# already names this situation and offers "size it smaller, treat it as
# lower-conviction, or leave it out" as equally-valid options, with no
# threshold saying when "leave it out" stops being optional — advisory
# text a compelling fundamental story (US government stake, Apple
# foundry talks) can and did override. See ai.clerk_execution._apply_
# reward_risk_floor_guard for the deterministic backstop this enables:
# REJECT (pct=0), never downsize, a not-yet-held candidate below this
# floor.
#
# 1.8 is derived, not guessed: this account's own real backtests cluster
# in a 25-40% win-rate range across setups; at a representative 35% win
# rate, breakeven requires (1-0.35)/0.35 ~= 1.86:1 before any real cost
# is netted. 1.8 sits just under that — the account's own already-cited
# "2:1 Bulkowski/Rockefeller" literature floor, made a hard line instead
# of advisory, with a small tolerance band rather than a razor's-edge
# cutoff at the textbook number itself.
MIN_NET_REWARD_RISK_RATIO = float(os.getenv("MIN_NET_REWARD_RISK_RATIO", "1.8"))

# Real gap, 2026-09-18 (direct user request): the flat floor above is one
# number for every candidate, derived from a GENERIC, assumed win rate.
# But this account's own real M5 intraday backtest evidence (see
# ai.ftmo_suggest.IntradayBacktests) now computes a REAL, per-symbol,
# per-side win rate for many candidates — breakeven R:R is mathematically
# a function of win rate, (1-w)/w, so one flat number wastes real,
# already-computed evidence that could make the floor smarter in BOTH
# directions: looser for a setup whose own real win rate is genuinely
# good (the "small dagger" case — a high-win-rate, modest-target setup
# is legitimately profitable even below 1.8:1), tighter for one whose
# own real win rate reads worse than the generic assumption (this
# account's own REAL historical win rate has actually been 25.9% over
# the last 180 days — worse than the 35% this flat floor assumes).
# See ai.clerk_execution._side_relevant_win_rate's own docstring for how
# a per-symbol win rate is selected, and _apply_reward_risk_floor_guard
# for how it's applied.
#
# A real sample-size floor, separate from the backtest functions' own
# internal min_occurrences/min_tests gates (5) — trusting a win rate
# enough to LOWER a risk floor deserves a stricter bar than merely
# "enough to report a number at all."
MIN_RESOLVED_TRADES_FOR_WIN_RATE_FLOOR = int(os.getenv("MIN_RESOLVED_TRADES_FOR_WIN_RATE_FLOOR", "20"))
# A hard backstop under the win-rate-derived path: no computed win rate,
# however good it looks, should ever let a candidate through below a
# bare 1:1 — a backtested win rate is itself an estimate with real
# sampling error, not a guarantee.
MIN_ABSOLUTE_NET_REWARD_RISK_RATIO = float(os.getenv("MIN_ABSOLUTE_NET_REWARD_RISK_RATIO", "1.0"))
# Requires the setup to clear its own real breakeven by a real margin,
# not scrape by exactly at the theoretical breakeven line.
WIN_RATE_FLOOR_SAFETY_MARGIN = float(os.getenv("WIN_RATE_FLOOR_SAFETY_MARGIN", "1.3"))

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
# Where ai.trade_journal's own per-trade lifecycle records live — one
# JSON file per trade "story" (a symbol's own trade idea from Mega
# Session's original proposal through every Clerk touch to however it
# ultimately resolves), added 2026-09-19 direct user request. Same
# local/gitignored/account-independent convention as CURIOSITY_RECORDS_
# DIR above — this is the single source of truth; the Obsidian vault
# note for the same story is a deterministic RENDER of this JSON,
# regenerated on every append, never hand-parsed back for logic.
TRADE_JOURNAL_DIR = os.getenv("TRADE_JOURNAL_DIR", "records/ftmo_trade_journal")

# --- Trade Audit (ai/trade_audit.py) — the 4th agent-like automation
# role, added 2026-09-20 direct user request: a once-daily retrospective
# 3-model coaching review of every CLOSED trade in the journal above,
# reusing ai.portfolio_suggest.build_audit_block's existing concurrent
# audit pool unmodified, saved into that trade's own Obsidian vault node.
# Opt-out, not opt-in — same posture as CLERK_EXECUTION_ENABLED_FILE/
# RESEARCHER_ENABLED_FILE/MEGA_ANALYSIS_ENABLED_FILE.
TRADE_AUDIT_ENABLED_FILE = os.getenv("TRADE_AUDIT_ENABLED_FILE", "trade_audit_enabled.json")

# DELIBERATE DEPARTURE from this codebase's own established "fixed UTC
# hour/minute, not DST-adjusted" convention (see RESEARCHER_TRIGGER_HOUR_
# UTC's own comment) — every other job's trigger anchors loosely to
# "roughly before/after the US day," where a DST-driven hour of drift
# twice a year is an accepted, documented non-issue. This job is
# different: it exists SPECIFICALLY to track one real, precise external
# event — 5:00 PM America/New_York local time, the real daily close of
# the US/FTMO forex trading session — and an hour of undetected drift
# half the year would mean running while that session may still
# genuinely be open (during EST). So this job computes its own trigger
# instant FRESH EACH DAY via Python's stdlib zoneinfo (3.9+, no new
# language-level dependency — see ai.trade_audit._ny_close_trigger_utc)
# rather than reading a fixed HOUR_UTC/MINUTE_UTC pair the way every
# other job here does: 21:00 UTC during EDT (~mid-March to early
# November) vs 22:00 UTC during EST (~November to mid-March), correctly
# and automatically, every day — verified live via web search 2026-09-20
# and confirmed against both an EDT and an EST date on this machine.
# NOTE: zoneinfo needs the IANA tz database at runtime; Windows CPython
# does NOT ship it, so this depends on the third-party `tzdata` PyPI
# package (added explicitly to requirements.txt alongside this feature —
# previously present on this machine only as an incidental transitive
# dependency of an unrelated package, which a clean install elsewhere
# could not have relied on).
TRADE_AUDIT_TRIGGER_HOUR_LOCAL = int(os.getenv("TRADE_AUDIT_TRIGGER_HOUR_LOCAL", "17"))
TRADE_AUDIT_TRIGGER_MINUTE_LOCAL = int(os.getenv("TRADE_AUDIT_TRIGGER_MINUTE_LOCAL", "0"))
TRADE_AUDIT_TRIGGER_TZ = os.getenv("TRADE_AUDIT_TRIGGER_TZ", "America/New_York")

# Real MT5 deal-settlement/trade-closing data needs a little wall-clock
# time to finalize right at the moment of the exact session close — this
# buffer is ADDED to the real NY-close instant above before the job is
# ever considered due, so the "closed" events this job reads are for
# trades that genuinely finished settling, not a race against MT5's own
# end-of-day bookkeeping.
TRADE_AUDIT_SETTLEMENT_BUFFER_MINUTES = int(os.getenv("TRADE_AUDIT_SETTLEMENT_BUFFER_MINUTES", "45"))

# Same "missing this window loses the whole day" reasoning as
# RESEARCHER_GRACE_MINUTES — wider than that job's 60 minutes since this
# one may need to sequentially audit several stories (up to TRADE_AUDIT_
# MAX_STORIES_PER_RUN below) before the window closes.
TRADE_AUDIT_GRACE_MINUTES = int(os.getenv("TRADE_AUDIT_GRACE_MINUTES", "120"))

TRADE_AUDIT_STATE_FILE = os.getenv("TRADE_AUDIT_STATE_FILE", "trade_audit_state.json")

# A hard ceiling on one Trade Audit pass — sized against TRADE_AUDIT_MAX_
# STORIES_PER_RUN below, each of which runs a full 3-model concurrent
# audit (comparable per-story cost to one AUDIT_RETRY_TIMEOUT_SECONDS
# window), with real margin.
TRADE_AUDIT_RUN_TIMEOUT_SECONDS = int(os.getenv("TRADE_AUDIT_RUN_TIMEOUT_SECONDS", "1800"))

# Defensive cap on how many unaudited closed stories get a real 3-model
# review in a single daily run, same "excess is picked up on a LATER
# poll rather than let one poll run unbounded" reasoning as CLERK_
# EXECUTION_MAX_PENDING_SETUPS — a large opening backlog (e.g. right
# after this feature ships) drains gradually over several days by
# design, not all at once, so it never monopolizes the free model pool.
TRADE_AUDIT_MAX_STORIES_PER_RUN = int(os.getenv("TRADE_AUDIT_MAX_STORIES_PER_RUN", "3"))

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
# Explicit, persistent allowlist of FTMO symbols the architecture should
# actually consider — added 2026-09-19, direct user request after a real
# incident: MT5's own Market Watch "visible" flag turned out NOT to be a
# stable signal for "the symbols we've chosen to trade." The broker
# terminal silently re-populates old symbols on login/reconnect
# (confirmed live: the user manually removed old symbols and added new
# ones directly in the terminal, and the old ones came back after simply
# logging back in — independent of any runtime symbol_select() call,
# ours or theirs). Separately, data/mt5_source.py's own _ensure_symbol_
# selected runs defensively wherever a symbol's live data is read by
# name, which re-marks it visible too — confirmed live: a stray, already-
# open browser tab still rendering the OLD symbol list kept resurrecting
# them on every page refresh, with no code change involved at all.
# Market Watch visibility can therefore never be trusted to durably
# represent this account's real choice of symbols — this file is the
# actual source of truth instead, read once inside get_market_watch()
# itself so every consumer (Clerk/Mega Session/Researcher/app.py) is
# filtered uniformly, with zero per-caller changes needed.
#
# Missing/empty/corrupt file means NO filter at all — get_market_watch()
# returns everything currently visible, exactly its original behavior —
# so this is purely additive until the user actually sets a real list via
# app.py's own symbol-mix control.
FTMO_SYMBOL_MIX_FILE = os.getenv("FTMO_SYMBOL_MIX_FILE", "ftmo_symbol_mix.json")

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
# A second deterministic circuit-breaker, same category as the hard-exit
# ceiling above — added 2026-09-16 after a real incident: NVDA was held
# long for most of a trading day while Clerk's own tactical checks
# explicitly stated, across dozens of consecutive polls, that both H1
# and H4 read downtrend (sometimes self-contradicting in the same
# sentence) before finally exiting (clerk_execution_log.txt:22101-23099,
# 2026-09-10/11). See ai.clerk_execution.TacticalSignals.trend_flip_
# against_count's own comment for the persisted-counter mechanics.
# 3 consecutive polls of a confirmed trend flip against a held position
# forces a 50%-of-current-volume reduction; a further 3 consecutive
# polls after that (6 total) forces a full exit — direct user decision:
# not an instant full exit, and not stop-only, to give the trade room to
# genuinely reverse back before the bot commits to closing it.
CLERK_TREND_FLIP_PARTIAL_AFTER_POLLS = int(os.getenv("CLERK_TREND_FLIP_PARTIAL_AFTER_POLLS", "3"))
CLERK_TREND_FLIP_EXIT_AFTER_POLLS = int(os.getenv("CLERK_TREND_FLIP_EXIT_AFTER_POLLS", "6"))
CLERK_TREND_FLIP_PARTIAL_REDUCE_PCT = float(os.getenv("CLERK_TREND_FLIP_PARTIAL_REDUCE_PCT", "50.0"))
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

# Real gap, 2026-09-18 (direct user request): the 50%-of-target trigger
# above is a large, late milestone — real, observed consequence, this
# account's positions have mostly closed at breakeven or small loss, not
# small-green, this engagement. CLERK_TACTICAL_PARTIAL_PROFIT_TARGET_PCT
# is also only ever advisory prompt context (ai.clerk_execution's own
# TacticalSignals.partial_profit_due field is never checked in
# _run_clerk_tactical_check's own deterministic circuit breakers) — the
# same "advisory text loses to model discretion" failure class already
# fixed twice this session. This is a SECOND, much earlier, genuinely
# deterministic profit-lock: a small, real slice gets banked as soon as
# a position has moved favorably enough to safely cover its OWN real
# round-trip cost with margin — far earlier than 50% of target — for a
# new, aspirant trader's own stated need for occasional small, real,
# realized-profit events, without loosening any entry-side risk
# discipline. 3x real cost is a real, derived margin (not a guessed flat
# %): enough that the banked gain is genuinely real after cost, not a
# coin-flip-thin margin a single tick of noise could erase.
CLERK_QUICK_PROFIT_LOCK_COST_MULTIPLE = float(os.getenv("CLERK_QUICK_PROFIT_LOCK_COST_MULTIPLE", "3.0"))
# A genuinely small slice — vs. the trend-flip circuit breaker's own
# defensive 50% — leaving most of the position open to still reach its
# full target; this is meant to feel like a quick, low-stakes bonus, not
# a real de-risking action.
CLERK_QUICK_PROFIT_LOCK_REDUCE_PCT = float(os.getenv("CLERK_QUICK_PROFIT_LOCK_REDUCE_PCT", "20.0"))

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
CLERK_LLM_TIMEOUT_SECONDS = int(os.getenv("CLERK_LLM_TIMEOUT_SECONDS", "180"))  # was 300; calls now run one at a time, warm


# Decision-tier M5 bar count fetched per symbol for Mega Session and Clerk's technical context.
# 600 M5 bars ~= 2 trading days: the structure window is capped by analysis/timeframe_profiles.py's
# M5 lookback (144), the rest feeds the median-ATR% baseline the volatility size scalar compares
# against. (M15 was dropped 2026-09-24: M5 supplies entry timing / stop ATR / short-range structure,
# H1 supplies the structure anchors and the reachability ATR.)
INTRADAY_M5_BARS = int(os.getenv("INTRADAY_M5_BARS", "600"))
# The M5-ATR multiple used for stop floors and tactical stops wherever the H1-era 1.5x ATR used to
# apply. Derived from REAL data (20 symbols, 2026-09-24): median M15 ATR / M5 ATR = 1.42, so the
# old 1.5x M15 ATR stop is 2.1x M5 ATR; 2.0 keeps the same stop distance in M5 terms.
M5_ATR_STOP_MULTIPLE = float(os.getenv("M5_ATR_STOP_MULTIPLE", "2.0"))


# --- Intraday decision-tier guards (2026-09-24) -------------------------------
# Resting M5-scale limit orders must not sit for a day: the older 24h age
# ceiling (MAX_PENDING_ORDER_AGE_HOURS) stays as the outer bound, Clerk now
# uses this much shorter one for every resting entry, including Pending-Setup
# orders that the plan-level age rule never saw.
INTRADAY_PENDING_ORDER_MAX_AGE_HOURS = float(os.getenv("INTRADAY_PENDING_ORDER_MAX_AGE_HOURS", "3"))
# Stop floor extras: at least this many spreads wide, and never below the
# broker's own minimum stop distance (previously unenforced on the live path).
MIN_STOP_SPREAD_MULTIPLE = float(os.getenv("MIN_STOP_SPREAD_MULTIPLE", "4"))
# A planned entry is STALE when live price sits further than this many M5 ATRs
# from it (a limit that far from market will not fill soon). Chosen from the fill
# math, not guessed. MEASURED on 20 real symbols x 6000 M5 bars (audit 2026-09-24; per-bar
# close-to-close sigma = 0.72 ATR, matching the 0.7 assumed here): the chance price TOUCHES a
# level d M5 ATRs away, in one direction, inside the 3h (36-bar) resting-order ceiling is 79% at
# 1.0, 60% at 2.0, 45% at 3.0, 39% at 3.5, 26% at 5.0 (interpolated: 30% at ~4.4). 3.5 is kept
# (a level that is unlikely — under 2-in-5 — to be touched inside the order's own life); the
# earlier M15-derived "30% at 3.5 M5 ATRs" figure was a units slip, the real 30% point is ~4.4.
STALE_ENTRY_M5_ATR_MULTIPLE = float(os.getenv("STALE_ENTRY_M5_ATR_MULTIPLE", "3.5"))
# Once a guard (stale-entry rejection, reward:risk floor) has rejected an UNFILLED entry and the resting order
# was cancelled, the Clerk does not place that symbol again for this many minutes (or until the next Mega
# session, which resets the tracking). Found on the first real M5-only session (2026-09-24 18:28 UTC): the
# stale-entry re-anchor moves the entry to a fresh zone every poll while the target stays put, so the net
# reward:risk hovered around the 1.8 floor and SOLUSD was placed and cancelled 9 times (NVDA 3 and 2) in ~7 hours.
GUARD_REJECTION_COOLDOWN_MINUTES = float(os.getenv("GUARD_REJECTION_COOLDOWN_MINUTES", "60"))
# Found 2026-09-25: an NVDA limit was placed ~4 minutes before the US close, then could not be cancelled for the
# whole night ("Market closed", 56 failed attempts). No NEW resting order is placed for a non-24h instrument
# inside this many minutes of the session end the instrument's own recent bars show (fail-open: no estimate,
# no block), and after a "Market closed" rejection the same order action is not retried for the backoff below.
NO_NEW_ORDER_MINUTES_BEFORE_CLOSE = float(os.getenv("NO_NEW_ORDER_MINUTES_BEFORE_CLOSE", "20"))
MARKET_CLOSED_BACKOFF_MINUTES = float(os.getenv("MARKET_CLOSED_BACKOFF_MINUTES", "30"))
# A stale entry the Clerk had to re-anchor rests on a re-derived zone plus Claude's ORIGINAL target, so its net
# reward:risk is less trustworthy and hovers near the floor (SOLUSD was placed/cancelled 9 times on that edge);
# it must clear the floor by this margin before an order is placed.
REANCHORED_ENTRY_RR_MARGIN = float(os.getenv("REANCHORED_ENTRY_RR_MARGIN", "0.3"))
# A discretionary (LLM) DEFEND may take a partial close only while the position is in profit; see
# ai/clerk_execution._validate_and_apply_tactical_verdict for the real MSFT incident.
CLERK_PARTIAL_CLOSE_REQUIRES_PROFIT = os.getenv("CLERK_PARTIAL_CLOSE_REQUIRES_PROFIT", "1") not in ("0", "false", "False", "")
# Order-churn hysteresis: a resting order keeps its exact price/stop/target
# while the freshly computed values stay within this many M5 ATRs (0.5 M15 ATR = 0.7 M5 ATR)
# (and its risk % within this fraction) of what is already resting.
RESTING_ORDER_STABILITY_M5_ATR_MULTIPLE = float(os.getenv("RESTING_ORDER_STABILITY_M5_ATR_MULTIPLE", "0.7"))
RESTING_ORDER_PCT_STABILITY_FRACTION = float(os.getenv("RESTING_ORDER_PCT_STABILITY_FRACTION", "0.2"))
# Volatility scalar: risk % is scaled by median-M5-ATR% / current-M5-ATR%,
# clamped to [this, 1.0] — only ever downsizes in an elevated-volatility tape.
INTRADAY_VOLATILITY_SCALAR_MIN = float(os.getenv("INTRADAY_VOLATILITY_SCALAR_MIN", "0.5"))
# Entries inside this many hours BEFORE a High-impact event (but outside the
# hard blackout window) are downsized by this factor.
EVENT_RUNUP_HOURS = float(os.getenv("EVENT_RUNUP_HOURS", "2"))
EVENT_RUNUP_SIZE_SCALAR = float(os.getenv("EVENT_RUNUP_SIZE_SCALAR", "0.5"))
# One position's initial margin may not exceed this % of equity (0 disables).
MAX_POSITION_MARGIN_PCT_OF_EQUITY = float(os.getenv("MAX_POSITION_MARGIN_PCT_OF_EQUITY", "25"))

# Reachability limit for a same-session target, in M5 ATRs. MEASURED 2026-09-24 (20 symbols x 6000 M5
# bars): the median furthest excursion from an entry within the 48-bar (4h) holding cap is 5.7 M5 ATRs, and
# price touches a level in ONE chosen direction within 3h only 60% / 45% / 26% of the time at 2x / 3x / 5x.
# The M5 trade-zone line flags any target past this limit as a long shot.
M5_TARGET_REACH_ATR_LIMIT = float(os.getenv("M5_TARGET_REACH_ATR_LIMIT", "6.0"))
# Final live re-check of the Mega Session's entries (ai/live_recheck.py, 2026-09-25): after the free-model audit
# Python compares every drafted entry with the LIVE price; an entry more than RECHECK_DRIFT_ATR M5 ATRs from it
# (the measured fill odds are 46% at 3x and 30% at 4.5x) goes back to Claude with fresh real levels. A re-issue
# is then verified: within RECHECK_REISSUE_MAX_ATR of the price (67% fill odds at 1.6x, 60% at 2x), entry/target
# each within RECHECK_ANCHOR_TOLERANCE_ATR of a real level, stop within RECHECK_MAX_STOP_ATR.
LIVE_RECHECK_ENABLED = os.getenv("LIVE_RECHECK_ENABLED", "1") not in ("0", "false", "False", "")
RECHECK_DRIFT_ATR = float(os.getenv("RECHECK_DRIFT_ATR", "3.0"))
RECHECK_REISSUE_MAX_ATR = float(os.getenv("RECHECK_REISSUE_MAX_ATR", "2.0"))
RECHECK_ANCHOR_TOLERANCE_ATR = float(os.getenv("RECHECK_ANCHOR_TOLERANCE_ATR", "0.35"))
RECHECK_MAX_STOP_ATR = float(os.getenv("RECHECK_MAX_STOP_ATR", "8.0"))
# Zone-confidence hold-rate cutoffs for the M5 trade zone (analysis.trade_zone.construct_trade_zone
# nudges confidence one notch by the anchor level's backtested hold rate). The H1-era cutoffs (70 / 40)
# are degenerate on M5: MEASURED 2026-09-24 on 97 real M5 S/R bands (48-bar hold window), the hold-rate
# distribution is p10 0 / p25 6 / p50 15 / p75 25 / p90 44, so 89% of bands sat at or below 40 and 3%
# at or above 70 — nearly every zone was downgraded and none upgraded. Shifted control bands (same width,
# +/-1.5 and +/-3 M5 ATRs away) held 15.5% on average vs 20.2% for the real bands (win rate 35% vs 43%),
# i.e. real levels do carry signal, so the cutoffs are set at the real-band quartiles: p75 (25) and p25 (6).
M5_ZONE_STRONG_HOLD_RATE_PCT = float(os.getenv("M5_ZONE_STRONG_HOLD_RATE_PCT", "25"))
M5_ZONE_WEAK_HOLD_RATE_PCT = float(os.getenv("M5_ZONE_WEAK_HOLD_RATE_PCT", "6"))
# M5-based tactical stop candidates (the H1-era 1.5x / 2.5x-fast ATR multiples are kept for the H1
# fallback): 2.0x M5 ATR is the same distance the old 1.5x M15 ATR gave; fast-tier movers get 3.0x.
CLERK_TACTICAL_M5_ATR_STOP_MULTIPLE = float(os.getenv("CLERK_TACTICAL_M5_ATR_STOP_MULTIPLE", "2.0"))
CLERK_TACTICAL_M5_ATR_STOP_MULTIPLE_FAST = float(os.getenv("CLERK_TACTICAL_M5_ATR_STOP_MULTIPLE_FAST", "3.0"))

# ---- Significance-aware backtest evidence (analysis/edge_stats.py, 2026-09-25 position-hunting review) ----
# A pooled M5 backtest that reads "~27% win rate, slightly negative R" is the NULL result: MEASURED on real
# M5 bars, random 2x/4x-ATR entries with a 2:1 target win ~27% at about -0.05R gross. Setups are therefore
# judged against a random-entry baseline on the same bars/stop/target/cost instead of against zero.
EDGE_SUPPORTED_Z = float(os.getenv("EDGE_SUPPORTED_Z", "1.5"))
EDGE_CONTRADICTED_Z = float(os.getenv("EDGE_CONTRADICTED_Z", "-2.0"))
EDGE_MIN_TRADES_SUPPORTED = int(os.getenv("EDGE_MIN_TRADES_SUPPORTED", "20"))
EDGE_MIN_TRADES_CONTRADICTED = int(os.getenv("EDGE_MIN_TRADES_CONTRADICTED", "30"))
EDGE_BASELINE_ENTRY_STEP_BARS = int(os.getenv("EDGE_BASELINE_ENTRY_STEP_BARS", "6"))
EDGE_BASELINE_MAX_ENTRIES = int(os.getenv("EDGE_BASELINE_MAX_ENTRIES", "800"))
EDGE_BASELINE_MIN_TRADES = int(os.getenv("EDGE_BASELINE_MIN_TRADES", "100"))
EDGE_BASELINE_WARMUP_BARS = int(os.getenv("EDGE_BASELINE_WARMUP_BARS", "60"))

# ---- Cost drag as an explicit gate (ai.ftmo_suggest.cost_drag_r, 2026-09-25) ----
# Round-trip spread+commission as a share of the risked amount. MEASURED on real spreads: mean 0.19R at a 2x
# M5-ATR stop, 0.09R at 4x, 0.06R at 6x; ~0.01R (BTC) up to >1R (WHEAT, COCOA). COST_DRAG_TARGET_R is only the
# figure shown next to "the stop that would cap drag" (information, not a floor — Claude picks the stop);
# COST_DRAG_VETO_R is the objective hard veto (V1) at the tightest valid stop.
COST_DRAG_TARGET_R = float(os.getenv("COST_DRAG_TARGET_R", "0.10"))
# 2026-09-25 (deep M5, 30,790 with-the-trend breakout fills, structure stop): net R by round-trip cost in R at the stop -
# cost <= 0.10R: +0.03..+0.06R per fill at 2-4R targets; cost > 0.10R: -0.10..-0.23R at EVERY target multiple. V1 therefore
# vetoes above 0.10R measured at the playbook's structure stop (it was 0.35R at the tightest stop).
COST_DRAG_VETO_R = float(os.getenv("COST_DRAG_VETO_R", "0.10"))
# The sizing sheet's "COST VETO" wording is measured at the TIGHTEST valid stop, where the drag is ~2x the playbook-stop figure;
# it keeps its own, looser limit.
COST_DRAG_SHEET_VETO_R = float(os.getenv("COST_DRAG_SHEET_VETO_R", "0.35"))

# ---- Position Hunter shortlist (analysis/position_hunter.py, 2026-09-25) ----
POSITION_HUNT_MAX_CANDIDATES = int(os.getenv("POSITION_HUNT_MAX_CANDIDATES", "8"))
POSITION_HUNT_ENABLED = os.getenv("POSITION_HUNT_ENABLED", "1") not in ("0", "false", "False", "")

# ---- New entry order types (analysis/entry_mode.py, 2026-09-25; enabled immediately by user decision) ----
# Kill switch: 0 makes every entry a plain limit again. Caps are re-verified from a live quote by the Clerk
# guard and by the final live re-check; a failed check downgrades to a limit (or rejects a dead setup).
NEW_ENTRY_KINDS_ENABLED = os.getenv("NEW_ENTRY_KINDS_ENABLED", "1") not in ("0", "false", "False", "")
MARKET_ENTRY_MAX_SLIPPAGE_ATR = float(os.getenv("MARKET_ENTRY_MAX_SLIPPAGE_ATR", "0.5"))  # live worse than the plan, M5 ATRs
MARKET_ENTRY_MAX_SPREAD_STOP_FRACTION = float(os.getenv("MARKET_ENTRY_MAX_SPREAD_STOP_FRACTION", "0.25"))
MARKET_ENTRY_SEND_DEVIATION_ATR = float(os.getenv("MARKET_ENTRY_SEND_DEVIATION_ATR", "0.15"))  # tick movement allowed poll->send
STOP_ENTRY_MAX_DISTANCE_ATR = float(os.getenv("STOP_ENTRY_MAX_DISTANCE_ATR", "2.0"))  # breakout trigger from the market
STOP_ENTRY_MAX_EXTENSION_ATR = float(os.getenv("STOP_ENTRY_MAX_EXTENSION_ATR", "0.6"))  # O'Neil: do not chase past the pivot

# ---- Deterministic profit trail (ai/clerk_execution.py TacticalSignals.profit_trail_*, 2026-09-25) ----
# MEASURED (20 real symbols x 6000 M5 bars, 38,960 paired random trades, spread charged): trailing the stop 1.5 M5
# ATR behind the price once the trade is +1R lifts net expectancy from -0.174R to -0.148R (better on 17/20
# symbols) — better than breakeven-at-1R (-0.163R), partial-50%+breakeven (-0.155R) or breakeven-at-1.5R
# (-0.172R). Applied as a deterministic, shadow-mode-exempt DEFEND (stop only, never a partial close).
CLERK_PROFIT_TRAIL_ENABLED = os.getenv("CLERK_PROFIT_TRAIL_ENABLED", "1") not in ("0", "false", "False", "")
CLERK_PROFIT_TRAIL_START_R = float(os.getenv("CLERK_PROFIT_TRAIL_START_R", "1.0"))
CLERK_PROFIT_TRAIL_ATR = float(os.getenv("CLERK_PROFIT_TRAIL_ATR", "1.5"))
CLERK_PROFIT_TRAIL_MIN_STEP_ATR = float(os.getenv("CLERK_PROFIT_TRAIL_MIN_STEP_ATR", "0.25"))  # no micro-amends

# ---- Re-hunt ledger (ai/rehunt.py, 2026-09-25) ----
REHUNT_LEDGER_FILE = os.getenv("REHUNT_LEDGER_FILE", "rehunt_ledger.json")
REHUNT_LEDGER_KEEP_DAYS = int(os.getenv("REHUNT_LEDGER_KEEP_DAYS", "7"))
REHUNT_MAX_AGE_HOURS = float(os.getenv("REHUNT_MAX_AGE_HOURS", "36"))

# ---- Deep price-history cache (data/price_cache.py, 2026-09-25 plan W2) ----
# The broker serves far more than the 6,000 M5 bars the live pipeline fetches (measured: up to 200,000 M5 bars for
# FX/crypto, ~100,000 for stocks). Studies, Symbol Behaviour Cards and out-of-sample gates read this cache.
PRICE_CACHE_DIR = os.getenv("PRICE_CACHE_DIR", "records/price_cache")
DEEP_HISTORY_BARS_M5 = int(os.getenv("DEEP_HISTORY_BARS_M5", "120000"))
DEEP_HISTORY_BARS_H1 = int(os.getenv("DEEP_HISTORY_BARS_H1", "60000"))
DEEP_HISTORY_BARS_D1 = int(os.getenv("DEEP_HISTORY_BARS_D1", "6000"))

# ---- Playbook selector (analysis/playbook.py, 2026-09-25 plan W3) ----
# Measured on 205,758 aligned M5 opportunities (see analysis/playbook.py): breakout stop at the 24-bar range extreme with
# the stop at the opposite side of the range (clipped 1.5-4 ATR): +0.086R gross, +0.077R net on the cheaper half of symbols.
PLAYBOOK_ENABLED = os.getenv("PLAYBOOK_ENABLED", "1") not in ("0", "false", "False", "")
PLAYBOOK_RANGE_BARS = int(os.getenv("PLAYBOOK_RANGE_BARS", "24"))
# ADX is a quality dial, not a gate: deep history showed the structure-stop breakout net positive on the cheaper half of symbols in EVERY regime
# (ADX<20 +0.034R, 20-30 +0.053R, >=30 +0.093R per opportunity); raise this to make it a gate.
PLAYBOOK_MIN_ADX = float(os.getenv("PLAYBOOK_MIN_ADX", "0"))
PLAYBOOK_MAX_TRIGGER_AHEAD_ATR = float(os.getenv("PLAYBOOK_MAX_TRIGGER_AHEAD_ATR", "1.5"))
PLAYBOOK_STOP_MIN_ATR = float(os.getenv("PLAYBOOK_STOP_MIN_ATR", "1.5"))
PLAYBOOK_STOP_MAX_ATR = float(os.getenv("PLAYBOOK_STOP_MAX_ATR", "4.0"))
PLAYBOOK_MIN_VOLUME_RATIO = float(os.getenv("PLAYBOOK_MIN_VOLUME_RATIO", "0.7"))
PLAYBOOK_TARGET_RR = float(os.getenv("PLAYBOOK_TARGET_RR", "2.0"))
PLAYBOOK_MEASURED_GROSS = float(os.getenv("PLAYBOOK_MEASURED_GROSS", "0.086"))
PLAYBOOK_MEASURED_NET_CHEAP = float(os.getenv("PLAYBOOK_MEASURED_NET_CHEAP", "0.077"))

# ---- Broker-side expiry of resting orders (data/mt5_execution.py open_position(expiration_hours=), 2026-09-25) ----
# A NEW limit/stop order for a session-limited instrument expires this many minutes before the instrument's learned session
# close, so it can never sit overnight where nobody can manage it (the NVDA case). A negative value disables the expiry.
RESTING_ORDER_EXPIRY_BEFORE_CLOSE_MINUTES = float(os.getenv("RESTING_ORDER_EXPIRY_BEFORE_CLOSE_MINUTES", "5"))

# ---- Structured Pending-Setup triggers (analysis/triggers.py, 2026-09-25) ----
# How many completed M5 bars the Clerk keeps in FtmoAssetAnalysis.m5_recent for evaluating a structured trigger.
TRIGGER_RECENT_BARS = int(os.getenv("TRIGGER_RECENT_BARS", "30"))

# ---- The Sentinel (ai/sentinel.py, clerk_sentinel_job.py; 2026-09-25) ----
# A one-minute, LLM-free profit-trail ratchet. SENTINEL_LOG_ONLY=1 (default) only logs what it WOULD do; set 0 to let it
# move stops. SENTINEL_ENABLED=0 switches the job off entirely.
SENTINEL_ENABLED = os.getenv("SENTINEL_ENABLED", "1") not in ("0", "false", "False", "")
SENTINEL_LOG_ONLY = os.getenv("SENTINEL_LOG_ONLY", "1") not in ("0", "false", "False", "")
SENTINEL_STATE_FILE = os.getenv("SENTINEL_STATE_FILE", "sentinel_state.json")

# ---- Level Map (analysis/level_map.py, 2026-09-25) ----
LEVEL_MAP_ENABLED = os.getenv("LEVEL_MAP_ENABLED", "1") not in ("0", "false", "False", "")
LEVEL_MAP_MAX_DISTANCE_ATR = float(os.getenv("LEVEL_MAP_MAX_DISTANCE_ATR", "6.0"))  # beyond this a resting order is unlikely to fill in hours
LEVEL_MAP_MIN_DISTANCE_ATR = float(os.getenv("LEVEL_MAP_MIN_DISTANCE_ATR", "0.5"))  # closer than this is inside the noise/spread band
LEVEL_MAP_MAX_CANDIDATES = int(os.getenv("LEVEL_MAP_MAX_CANDIDATES", "6"))
LEVEL_MAP_MIN_BOUNCE_ATR = float(os.getenv("LEVEL_MAP_MIN_BOUNCE_ATR", "1.0"))  # a reaction point: price turned at least this far within 12 bars
# Measured 2026-09-25 (deep M5, 20 symbols): how far price pokes THROUGH a level that then held, in M5 ATRs.
LEVEL_PENETRATION_MEDIAN_ATR = 0.5
LEVEL_PENETRATION_P75_ATR = 1.3
LEVEL_PENETRATION_P90_ATR = 2.7

# ---- Symbol Behaviour Card (analysis/symbol_card.py, tools/studies/symbol_cards.py; 2026-09-25) ----
SYMBOL_CARD_ENABLED = os.getenv("SYMBOL_CARD_ENABLED", "1") not in ("0", "false", "False", "")
SYMBOL_CARD_FILE = os.getenv("SYMBOL_CARD_FILE", "records/symbol_cards.json")
SYMBOL_CARD_MIN_BARS = int(os.getenv("SYMBOL_CARD_MIN_BARS", "5000"))  # fewer completed M5 bars than this: no card
SYMBOL_CARD_STALE_DAYS = int(os.getenv("SYMBOL_CARD_STALE_DAYS", "45"))

# The Position Hunter reads D1/H4 direction from CLOSED higher-timeframe buckets (analysis/htf_flags.py) - the flags the
# alignment study measured - instead of the TechnicalStats trend that includes the still-forming bar. 0 = the old read.
HUNTER_CLOSED_HTF_FLAGS = os.getenv("HUNTER_CLOSED_HTF_FLAGS", "1") not in ("0", "false", "False", "")

# The Clerk's intraday size scalar never shrinks a NEW entry below the broker's minimum lot when that minimum lot still fits
# inside the risk % Claude chose (a scaled risk that buys no position silently kills an approved trade). 0 = old behaviour.
SIZE_SCALAR_MIN_LOT_FLOOR = os.getenv("SIZE_SCALAR_MIN_LOT_FLOOR", "1") not in ("0", "false", "False", "")


# Clerk correlation guard (ai/clerk_execution.py::_apply_correlation_guard). Correlation is LOW-weight information: the aggregate-heat
# ceiling already sums every stop as if all were hit together, so correlated positions cannot breach it. The guard remains only for
# near-duplicates (|r| >= 0.90; it was 0.70) and trims the newcomer by a quarter (it halved it). Set the threshold above 1 to switch off.
CLERK_CORRELATION_GUARD_THRESHOLD = float(os.getenv("CLERK_CORRELATION_GUARD_THRESHOLD", "0.90"))
CLERK_CORRELATION_SIZE_FACTOR = float(os.getenv("CLERK_CORRELATION_SIZE_FACTOR", "0.75"))

# ---- Mega session: analyse only instruments that can trade right now (ai/mega_analysis.py::select_tradable_assets) ----
# An instrument is skipped when its market calendar is closed, it has no tick, or its last tick is older than this many minutes
# behind the freshest tick in the Market Watch pool (a session that ended for the day). Held / resting-order symbols are always kept.
MEGA_TRADABLE_FILTER_ENABLED = os.getenv("MEGA_TRADABLE_FILTER_ENABLED", "1") not in ("0", "false", "False", "")
MEGA_TRADABLE_MAX_TICK_AGE_MINUTES = float(os.getenv("MEGA_TRADABLE_MAX_TICK_AGE_MINUTES", "30"))

# The Clerk FAST LANE (clerk_fast_job.py): the deterministic pipeline every minute, no model call. 0 = only the full poll runs.
CLERK_FAST_LANE_ENABLED = os.getenv("CLERK_FAST_LANE_ENABLED", "1") not in ("0", "false", "False", "")

# ---- Thinking vs acting (ai/clerk_thinking.py, clerk_think_job.py; 2026-09-26) ----
# ON: no Clerk pass that acts (fast lane, regular poll, post-Mega hand-off) ever waits for the local model; a separate thinking job
# keeps a cache of model verdicts that the acting passes use in seconds. OFF: the old single poll that thinks and then acts.
CLERK_THINK_SPLIT_ENABLED = os.getenv("CLERK_THINK_SPLIT_ENABLED", "1") not in ("0", "false", "False", "")
CLERK_THINK_CACHE_FILE = os.getenv("CLERK_THINK_CACHE_FILE", "clerk_thinking_cache.json")
# A stored model verdict older than this is ignored (the thinking job renews them every review interval).
CLERK_THINK_CACHE_TTL_MINUTES = float(os.getenv("CLERK_THINK_CACHE_TTL_MINUTES", "12"))

# When the MT5 terminal answers "Authorization failed" to a login, every process backs off this long before trying credentials again
# (data/mt5_source.py::connect). A terminal that is (or becomes) logged in is still attached to immediately, with no credentials.
MT5_AUTH_BACKOFF_FILE = os.getenv("MT5_AUTH_BACKOFF_FILE", "mt5_auth_backoff.json")
MT5_AUTH_BACKOFF_MINUTES = float(os.getenv("MT5_AUTH_BACKOFF_MINUTES", "5"))

# ---- Clerk thinking diet (2026-09-26): what makes the local-model rounds short ----
# Measured on this machine (qwen3:8b, 4GB VRAM, ~8 tokens/s): the cost of a call is almost entirely the tokens the model WRITES
# (prompt evaluation of a compact prompt is ~0.3 s), a reload after Ollama's default 5-minute idle unload costs 9-250 s, and requests
# submitted in parallel to one Ollama server just queue behind each other against the per-call timeout.
CLERK_LLM_KEEP_ALIVE = os.getenv("CLERK_LLM_KEEP_ALIVE", "30m")  # keep the Clerk's model resident between thinking passes
CLERK_COMPACT_CONTEXT = os.getenv("CLERK_COMPACT_CONTEXT", "1") not in ("0", "false", "False", "")  # ~1/4 of the old context
CLERK_DETERMINISTIC_INVALIDATION = os.getenv("CLERK_DETERMINISTIC_INVALIDATION", "1") not in ("0", "false", "False", "")
CLERK_LLM_SEQUENTIAL = os.getenv("CLERK_LLM_SEQUENTIAL", "1") not in ("0", "false", "False", "")

# The Clerk analyses a symbol it only has to DEFEND (an open position, no new entry on it) with the LEAN analysis: M5 reads only.
CLERK_LEAN_ANALYSIS = os.getenv("CLERK_LEAN_ANALYSIS", "1") not in ("0", "false", "False", "")

# Continuation Watch, phase 2 (2026-09-28): the job/orchestration around the phase-1 filter — see
# ai/continuation_hunter.py's own module docstring for the full seven-point design. LOG_ONLY=1 (the
# default) means every decision is computed and written to the trade journal/log but NO ORDER IS EVER SENT —
# wiring a "propose" into the real entry pipeline is phase 3, not built yet. ENABLED=1 lets the job run at
# all (in log-only mode by default, so it starts producing real observations right away, same rollout shape
# as the Sentinel).
CONTINUATION_WATCH_ENABLED = os.getenv("CONTINUATION_WATCH_ENABLED", "1") not in ("0", "false", "False", "")
CONTINUATION_WATCH_LOG_ONLY = os.getenv("CONTINUATION_WATCH_LOG_ONLY", "1") not in ("0", "false", "False", "")
CONTINUATION_WATCH_STATE_FILE = os.getenv("CONTINUATION_WATCH_STATE_FILE", "continuation_watch_state.json")
# Escalation to OpenRouter when the local model is down/unclear (same shape as ai.curiosity's own
# _run_curiosity_model_with_retry): a short per-call timeout, tried across a few fallbacks within one budget.
CONTINUATION_WATCH_MODEL_TIMEOUT_SECONDS = int(os.getenv("CONTINUATION_WATCH_MODEL_TIMEOUT_SECONDS", "30"))
CONTINUATION_WATCH_RETRY_TIMEOUT_SECONDS = int(os.getenv("CONTINUATION_WATCH_RETRY_TIMEOUT_SECONDS", "150"))

# Continuation Watch (phase 1, 2026-09-28): how long a just-closed WINNING trade gets to prove it's still
# running before this filter gives up on it, how many real M5 bars that needs, and how big the move (in ATR)
# must be. See analysis/continuation_watch.py's own docstring for the calibration these numbers came from —
# 15 minutes / 3 bars is exactly the "2-3 candles is a fair time to decide" the user asked for; 2.0x ATR is a
# conservative first cut (the two real trades that motivated this cleared 4.5x and 21x, with room to spare).
CONTINUATION_MIN_BARS = int(os.getenv("CONTINUATION_MIN_BARS", "3"))
CONTINUATION_MAX_WAIT_MINUTES = float(os.getenv("CONTINUATION_MAX_WAIT_MINUTES", "15"))
CONTINUATION_MIN_ATR_MOVE = float(os.getenv("CONTINUATION_MIN_ATR_MOVE", "2.0"))

# Thinking pass: a held position whose last tactical verdict was HOLD and whose relevant facts have not moved (see
# ai.clerk_execution._tactical_fingerprint) is not put to the model again, for at most this many minutes (0 = always ask).
CLERK_THINK_REUSE_MAX_MINUTES = float(os.getenv("CLERK_THINK_REUSE_MAX_MINUTES", "20"))
# Cap on the tokens the Clerk's local model may write per answer (the answers are 2-3 sentences plus a verdict block).
CLERK_LLM_MAX_TOKENS = int(os.getenv("CLERK_LLM_MAX_TOKENS", "450"))

# Time-critical Clerk jobs (fast lane, Sentinel, regular poll, post-Mega hand-off) wait up to this long for the execution lock instead
# of skipping when another short pass holds it (the minute timers all fire in the same second).
CLERK_LOCK_WAIT_SECONDS = float(os.getenv("CLERK_LOCK_WAIT_SECONDS", "30"))
