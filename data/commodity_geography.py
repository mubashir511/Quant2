# Producer/consumer country linkage per metal, and the primary country
# linked to major equity indices — used purely to direct WebSearch/WebFetch
# at the countries most relevant to each traded instrument. Not a data
# fetch: there's no reliable free API for mine production/consumption
# stats at this granularity, so this stays a hardcoded reference table,
# same pattern as data/crop_context.py::CROP_TRADE_PROFILES. Keyed by the
# same display names data/underlying.py already produces.
METAL_LINKED_COUNTRIES: dict[str, list[str]] = {
    "Gold": ["China", "Australia", "Russia", "United States", "India"],
    "Silver": ["Mexico", "Peru", "China", "Russia"],
    "Platinum": ["South Africa", "Russia", "Zimbabwe"],
    "Palladium": ["Russia", "South Africa"],
    "Copper": ["Chile", "Peru", "China", "Democratic Republic of Congo"],
}

INDEX_LINKED_COUNTRIES: dict[str, list[str]] = {
    "S&P 500": ["United States"],
    "Nasdaq 100": ["United States"],
    "Dow Jones": ["United States"],
}
