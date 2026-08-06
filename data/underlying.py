import re

# Keyword-based mapping from a PMEX symbol/description to a comparable Yahoo
# Finance ticker, used purely to pull public price history and news for
# extra AI context — PMEX pricing/positions always come from MT5 itself.
#
# Order matters: checked top to bottom, first match wins. Keep more specific
# keywords (e.g. "PALDIUM" for Palladium) above more generic ones that could
# otherwise collide (e.g. a bare "PL"). Matching guards against adjacent
# *letters* (so "DJ" doesn't false-match inside "ADJUSTABLE") but allows
# adjacent digits, since these symbols pack letters directly against
# numbers with no separator (e.g. "NSDQ100", "GO10OZ").
_KEYWORD_MAP: list[tuple[str, str, str]] = [
    ("PALDIUM", "Palladium", "PA=F"),
    ("PALLADIUM", "Palladium", "PA=F"),
    ("PLATINUM", "Platinum", "PL=F"),
    ("SILVER", "Silver", "SI=F"),
    ("GOLD", "Gold", "GC=F"),
    ("COPPER", "Copper", "HG=F"),
    ("CRUDE", "Crude Oil", "CL=F"),
    ("NATGAS", "Natural Gas", "NG=F"),
    ("NATURAL GAS", "Natural Gas", "NG=F"),
    ("NSDQ", "Nasdaq 100", "^IXIC"),
    ("NASDAQ", "Nasdaq 100", "^IXIC"),
    ("SP500", "S&P 500", "^GSPC"),
    ("S&P", "S&P 500", "^GSPC"),
    ("DJ", "Dow Jones", "^DJI"),
    ("DOW", "Dow Jones", "^DJI"),
    ("WHEAT", "Wheat", "ZW=F"),
    ("MAIZE", "Corn", "ZC=F"),
    ("CORN", "Corn", "ZC=F"),
    ("RICE", "Rice", "ZR=F"),
    ("SUGAR", "Sugar", "SB=F"),
]


def resolve_yahoo_ticker(symbol: str, description: str = "") -> tuple[str, str] | None:
    """Best-effort match to a Yahoo ticker from a PMEX symbol/description.

    Returns (display_name, yahoo_ticker), or None if nothing matches —
    callers should degrade gracefully rather than treat that as an error.
    """
    haystack = f"{symbol} {description}".upper()
    for keyword, display_name, yahoo_ticker in _KEYWORD_MAP:
        pattern = r"(?<![A-Z])" + re.escape(keyword) + r"(?![A-Z])"
        if re.search(pattern, haystack):
            return display_name, yahoo_ticker
    return None
