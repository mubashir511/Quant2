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
MAX_ENRICHED_ASSETS = int(os.getenv("MAX_ENRICHED_ASSETS", "12"))
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
