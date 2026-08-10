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
