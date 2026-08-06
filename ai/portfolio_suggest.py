import json
import re
from dataclasses import dataclass, field

import pandas as pd

import config
from ai.claude_cli import run_claude
from analysis.technical import TechnicalStats, compute_technical_stats
from data.book_wisdom import format_book_wisdom
from data.commodity_geography import INDEX_LINKED_COUNTRIES, METAL_LINKED_COUNTRIES
from data.crop_context import CROP_TRADE_PROFILES, fetch_crop_supply_demand_context
from data.macro_source import fetch_country_indicators, fetch_fx_rate_to_usd, fetch_market_indicators
from data.market_history import fetch_price_history
from data.mt5_source import AccountSummary, ContractSpec, MarketAsset, get_contract_spec
from data.news_source import fetch_recent_headlines
from data.underlying import resolve_yahoo_ticker

ENERGY_COMMODITIES = {"Crude Oil", "Natural Gas"}

SYSTEM_INSTRUCTION = (
    "You are a portfolio-construction assistant for a personal PMEX futures "
    "trading account that currently holds no open positions. You have "
    "WebSearch and WebFetch tools available — use them extensively, not "
    "sparingly. Actively research the last ~6 months of geopolitical "
    "developments (wars, floods, droughts, tariffs, trade-route "
    "disruption, tax policy changes) that could affect the instruments "
    "below."
    "\n\n"
    "Research depth is a priority, not an afterthought: run a separate, "
    "specific search for each instrument, each linked country, and each "
    "named institution/company below — do not settle for one or two broad "
    "queries covering everything at once. Where an initial search only "
    "turns up a weak or generic result, search again with a more specific "
    "query (e.g. the country/topic name plus a date range or the specific "
    "policy/event) rather than moving on. Aim to ground the final answer "
    "in noticeably more distinct credible sources than a shallow pass "
    "would produce — breadth across topics matters as much as depth on "
    "any one of them."
    "\n\n"
    "Source quality matters — prioritize in this order:\n"
    "1. Official/primary sources: the IMF, World Bank, central banks (US "
    "Federal Reserve, ECB, Bank of England, Bank of Japan, People's Bank "
    "of China, Reserve Bank of India, Bank of Korea, Saudi Central Bank, "
    "UAE Central Bank), national statistics/finance-ministry portals (e.g. "
    "US BLS/BEA, UK ONS, Germany's Destatis, Japan's Cabinet Office, "
    "China's NBS, India's MOSPI, Korea's KOSTAT, Taiwan's DGBAS), and "
    "recognized data providers (e.g. ADP's employment report).\n"
    "2. Major investment banks' and monetary agencies' public research/"
    "commentary when actually available (e.g. JPMorgan, Goldman Sachs, "
    "Morgan Stanley, Citigroup, HSBC, UBS, Deutsche Bank, BIS) — read what "
    "they've actually published, don't guess at their view.\n"
    "3. Major, reputable financial/general news outlets: Bloomberg, "
    "Reuters, Financial Times, The Economist, CNBC, BBC, CNN, CBC, The New "
    "York Times, The Washington Post.\n"
    "Do not cite Wikipedia or low-quality/unverified blogs as a source. If "
    "a genuinely credible source can't be found for something, say so "
    "rather than falling back to a weak one or inventing a citation."
    "\n\n"
    "You're also given: a macro snapshot (US Treasury yield curve, dollar "
    "index, VIX, and GDP growth/inflation/unemployment for 10 major "
    "economies — US, UK, France, Germany, Japan, China, India, South "
    "Korea, Saudi Arabia, UAE), per-instrument technical context — a "
    "short-term trend vs. 20-day moving average, momentum, annualized "
    "volatility, AND a medium-term (~3-month, daily-resolution) pattern "
    "read: a support/resistance range and a market-type classification "
    "('trending_up'/'trending_down' when price has moved fairly directly "
    "in one direction over that window, 'sideways' when it's mostly "
    "chopped in a range, 'mixed' in between) — plus recent headlines "
    "where available, and a feasibility line: the REAL minimum-lot margin "
    "requirement (in the account's own margin currency) as a % of "
    "account equity, flagged 'NOT AFFORDABLE' past 100%. This is a hard "
    "constraint, not a preference — on a small account, a single "
    "instrument's minimum tradable lot can require several times the "
    "entire account's equity, making any % allocation to it literally "
    "impossible to execute, not just impractical. And — for any crops currently "
    "traded — a crop supply/demand section with real recent rainfall and "
    "short-term forecast data for that crop's major exporting and "
    "importing countries (a simplified single-region-per-country proxy, "
    "not precise agronomic modeling); reason about whether that pattern "
    "suggests a supply surplus/shortfall (e.g. a major exporter showing "
    "unusually low rainfall going into a forecast dry spell is a bullish "
    "supply signal). There is also a 'Research directives' section — "
    "follow those using WebSearch/WebFetch specifically (e.g. oil "
    "chokepoint conditions, named commodity companies' outlook, and the "
    "specific producer/consumer countries linked to whichever metals or "
    "indices are traded — search each named country for anything recent "
    "and relevant: mining strikes, sanctions, power/supply disruptions, "
    "trade policy, demand trends), applying the same source-quality and "
    "honesty rules above. If an instrument "
    "relates to a major economy not already in the macro snapshot (e.g. "
    "Taiwan), dig into that country's own government/statistics portal "
    "too. Use all of this — the macro data, the per-instrument data, the "
    "crop weather data, and what you find via search — as real reasoning "
    "input, not decoration."
    "\n\n"
    "For each instrument, also cross-reference its price pattern (the "
    "short-term trend and the medium-term support/resistance/market-type) "
    "against the macro snapshot, crop weather data where relevant, and the "
    "news/government-policy developments you find for its linked countries "
    "— over that same multi-month window. Use this to explain how the "
    "asset appears to have been reacting to these kinds of events or "
    "policy moves, and what that implies for forward-looking risk (e.g. "
    "'this has tended to move sharply on this country's rate decisions; "
    "given the current policy stance, watch for X'). This is qualitative "
    "pattern-matching from what you actually find, not a formal backtest — "
    "say so plainly if a connection you're drawing is speculative rather "
    "than clearly evidenced."
    "\n\n"
    "You're also given a set of investing-literature principles (position "
    "sizing, portfolio heat, group-exposure limits, stop/profit "
    "discipline), each attributed to the named author who argues for it. "
    "These are advisory, not enforced rules — nothing in this codebase "
    "requires following them. For each one that's actually relevant to "
    "the instruments/account here, briefly explain *why* that author "
    "argues for it (the reasoning given alongside each principle below) "
    "and use that reasoning — not just the bare rule — to inform your "
    "suggested mix, sizing, and cash reserve. You may reason your way to "
    "a different conclusion than a principle would suggest, but say so "
    "and explain why, rather than silently ignoring it."
    "\n\n"
    "Work through this reasoning internally, in order, before you write "
    "your visible answer — but do NOT print it as separate labeled "
    "stages. Do not use headers like 'Draft', 'Stress-Test', or 'Revised' "
    "anywhere in your response:\n"
    "1. Draft a diversified starting mix (which instruments, roughly what "
    "proportion of equity each) and a cash reserve to keep unallocated. "
    "Before including any instrument, check its feasibility line: only "
    "include it if the minimum lot is actually affordable. Where the lot "
    "granularity doesn't allow a smooth percentage (common here — whole "
    "lots only, no fractional sizing), state the position as a realistic "
    "whole-lot count and the actual resulting percentage, not an "
    "arbitrary smooth target that doesn't correspond to any valid order. "
    "If no combination of minimum lots fits the account at all, say so "
    "explicitly instead of drafting an impossible allocation.\n"
    "2. Attack your own draft against these failure modes specifically, "
    "using real numbers already given above (and WebSearch where noted) "
    "rather than generic caveats. Skip a point only if it's genuinely not "
    "applicable (e.g. no FX section means the account is already "
    "USD-denominated):\n"
    "   a. FX/base-currency mismatch — if an FX section is present above, "
    "the account's own currency differs from the USD pricing/settlement "
    "of most instruments; consider how a currency move could distort "
    "returns or trigger stops independent of the instrument's own price "
    "action.\n"
    "   b. Futures roll yield — every instrument here is a *dated* "
    "futures contract (visible in its symbol/expiry), not a spot "
    "buy-and-hold. For contracts you're including, use WebSearch to check "
    "whether it's currently in contango or backwardation and factor the "
    "implied roll cost/benefit into your reasoning.\n"
    "   c. Correlation under stress — construct one adverse macro "
    "scenario from the yield/DXY/VIX data given above (e.g. a real-yield "
    "spike or a DXY surge) and assess whether your draft's positions "
    "would move together more than their individual weights suggest.\n"
    "   d. Execution/liquidity risk — each instrument above has a "
    "bid-ask spread% shown; identify any wide-spread instrument in your "
    "draft and factor in recommending limit/stop-limit orders over market "
    "orders where that matters.\n"
    "   e. Idle cash — if your draft leaves a meaningful cash reserve, "
    "define specific conditional triggers for deploying it, tied to the "
    "support/resistance levels already given per instrument above (e.g. "
    "'deploy X% if [instrument] pulls back to its support level' or "
    "'deploy X% on a confirmed breakout above [instrument]'s resistance'). "
    "Every such trigger must also have an explicit time-based fallback "
    "(e.g. 'if not triggered within N trading days, do X instead') — an "
    "open-ended trigger that could leave cash idle indefinitely is itself "
    "a weakness. If the FX section above is present, also research the "
    "account country's current short-term risk-free/policy rate via "
    "WebSearch and explicitly weigh idle cash's opportunity cost against "
    "it (and against staying invested) when sizing the reserve — don't "
    "size cash as an arbitrary leftover percentage.\n"
    "   f. Position sizing vs. volatility, and hedge feasibility — size "
    "positions smaller, relative to equal-risk peers, for instruments "
    "showing higher annualized volatility% above. Only propose a hedge or "
    "alternative instrument that's actually listed among the Market Watch "
    "instruments below — don't suggest options, ETFs, or any other "
    "product that isn't shown as tradable on this account.\n"
    "3. Revise the draft based on what step 2 found.\n"
    "\n"
    "Now write your visible answer as ONE cohesive final recommendation — "
    "not the three steps above shown separately. For each instrument and "
    "for the cash reserve, explain the reasoning behind that decision "
    "inline (weaving in whichever of the considerations above are "
    "relevant to it) as part of justifying the number, the way an analyst "
    "would write a finished recommendation rather than show their scratch "
    "work. No deterministic sizing or risk rules are wired up yet for "
    "this feature, so clearly state that the final answer is still an "
    "early-stage, discretionary starting point for discussion, not a "
    "precise or final recommendation."
    "\n\n"
    "Prioritize thoroughness and rigor over brevity — there is no strict "
    "length limit on this response. Take the time you need to research "
    "and reason properly; a longer, well-evidenced answer is preferred "
    "over a shorter, shallower one. Stay organized (clear paragraphs or "
    "bullet points per instrument) and avoid needless repetition."
    "\n\n"
    "Write all of your reasoning and explanation as plain prose/markdown "
    "(paragraphs, bullet points, bold text) — do not put any of it inside "
    "a fenced code block. The ONLY fenced code block in your entire "
    "response must be a single one at the very end, exactly like this "
    "(replace the example values with your actual final numbers, one key "
    "per instrument symbol traded above plus one \"CASH\" key, values "
    "summing to 100, no comments or extra text inside the block):\n"
    "```json\n"
    '{"EXAMPLE_SYMBOL": 15, "CASH": 25}\n'
    "```"
)


def _spread_pct(a: MarketAsset) -> float | None:
    """Bid-ask spread as a % of price — real, already-available execution-
    risk data (wide spreads mean market orders can fill far from quote)."""
    if not a.ask:
        return None
    return (a.ask - a.bid) / a.ask * 100


@dataclass
class AssetAnalysis:
    """Everything computed for one Market Watch instrument — single source
    of truth shared by the prompt text builder and the chart renderer, so
    charting doesn't re-fetch what the text-building already fetched."""

    symbol: str
    description: str
    bid: float
    ask: float
    display_name: str | None
    prices: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    stats: TechnicalStats = field(
        default_factory=lambda: compute_technical_stats(pd.Series(dtype=float))
    )
    headlines: list[str] = field(default_factory=list)
    contract_spec: ContractSpec | None = None


def analyze_assets(assets: list[MarketAsset]) -> list[AssetAnalysis]:
    """Resolve, fetch, and compute technical stats for the first assets (up
    to config.MAX_ENRICHED_ASSETS) that map to a Yahoo ticker. Instruments
    without a mapping, or beyond the cap, still get an entry (display_name
    = None) so callers can list them plainly without a network round-trip.
    Contract spec (real order-size/margin constraints) is fetched for
    every asset regardless of Yahoo mapping — it only needs the PMEX
    symbol via the already-open MT5 connection, no extra network call."""
    results = []
    enriched_count = 0
    for a in assets:
        contract_spec = get_contract_spec(a.symbol)

        resolved = None
        if enriched_count < config.MAX_ENRICHED_ASSETS:
            resolved = resolve_yahoo_ticker(a.symbol, a.description)

        if resolved is None:
            results.append(
                AssetAnalysis(a.symbol, a.description, a.bid, a.ask, None, contract_spec=contract_spec)
            )
            continue

        display_name, yahoo_ticker = resolved
        enriched_count += 1

        prices = fetch_price_history(yahoo_ticker)
        stats = compute_technical_stats(prices)
        headlines = fetch_recent_headlines(yahoo_ticker, limit=config.NEWS_HEADLINES_PER_ASSET)

        results.append(
            AssetAnalysis(
                a.symbol, a.description, a.bid, a.ask, display_name,
                prices, stats, headlines, contract_spec,
            )
        )

    return results


def _feasibility_line(r: AssetAnalysis, account_equity: float | None) -> str | None:
    """Whether the minimum tradable lot is actually affordable — a real
    constraint (order-size/margin), not a preference. On a small account,
    a single instrument's minimum lot can require several times the
    entire account's equity, making any % allocation to it impossible,
    not just impractical (confirmed live: PALDIUM100-SE26's minimum lot
    needed ~3.7x a 1,000,000 PKR account)."""
    if account_equity is None or not r.contract_spec or account_equity <= 0:
        return None
    spec = r.contract_spec
    min_lot_margin = spec.margin_initial * spec.volume_min
    pct = min_lot_margin / account_equity * 100
    flag = "NOT AFFORDABLE — " if pct > 100 else ""
    return (
        f"  feasibility: {flag}min lot margin ~{min_lot_margin:,.0f} "
        f"{spec.currency_margin} ({pct:.1f}% of equity)"
    )


def format_enriched_asset_context(
    analyses: list[AssetAnalysis], account_equity: float | None = None
) -> str:
    """Pure formatting over already-computed AssetAnalysis objects — same
    text shape as before this was split out of a single fetch+format loop,
    plus an optional feasibility line when account_equity is supplied."""
    lines = []
    for r in analyses:
        spread_pct = _spread_pct(r)
        feasibility_line = _feasibility_line(r, account_equity)

        if r.display_name is None:
            lines.append(f"- {r.symbol} ({r.description}): bid {r.bid}, ask {r.ask}")
            if spread_pct is not None:
                lines.append(f"  execution: bid-ask spread {spread_pct:.2f}% of price")
            if feasibility_line is not None:
                lines.append(feasibility_line)
            continue

        lines.append(f"- {r.symbol} ({r.display_name}): bid {r.bid}, ask {r.ask}")
        if spread_pct is not None:
            lines.append(f"  execution: bid-ask spread {spread_pct:.2f}% of price")
        if feasibility_line is not None:
            lines.append(feasibility_line)

        stats = r.stats
        if stats.last_price is not None:
            stat_bits = []
            if stats.trend is not None:
                stat_bits.append(f"{stats.trend} ({stats.pct_vs_sma20:+.1f}% vs 20d SMA)")
            if stats.change_1m_pct is not None:
                stat_bits.append(f"1M {stats.change_1m_pct:+.1f}%")
            if stats.change_3m_pct is not None:
                stat_bits.append(f"3M {stats.change_3m_pct:+.1f}%")
            if stats.change_6m_pct is not None:
                stat_bits.append(f"6M {stats.change_6m_pct:+.1f}%")
            if stats.volatility_annualized_pct is not None:
                stat_bits.append(
                    f"volatility {stats.volatility_annualized_pct:.1f}% (annualized)"
                )
            if stat_bits:
                lines.append(f"  technical: {', '.join(stat_bits)}")

            pattern_bits = []
            if stats.support is not None and stats.resistance is not None:
                pattern_bits.append(
                    f"support {stats.support:.2f} / resistance {stats.resistance:.2f} "
                    f"(range {stats.range_width_pct:.1f}% of price, over ~3 months)"
                )
            if stats.market_regime is not None:
                pattern_bits.append(f"market type: {stats.market_regime}")
            if pattern_bits:
                lines.append(f"  pattern: {', '.join(pattern_bits)}")
        for headline in r.headlines:
            lines.append(f"  news: {headline}")

    return "\n".join(lines)


def build_enriched_asset_context(assets: list[MarketAsset]) -> str:
    """Convenience wrapper kept for existing callers/tests — prefer calling
    analyze_assets() once and format_enriched_asset_context() when the raw
    analyses are also needed elsewhere (e.g. charting), to avoid re-fetching."""
    return format_enriched_asset_context(analyze_assets(assets))


def build_macro_snapshot() -> str:
    """US yield curve/DXY/VIX plus GDP growth/inflation/unemployment for
    the US, UK, Japan, and China. Missing values are simply omitted rather
    than faked — see data/macro_source.py for why a given field can be None."""
    market = fetch_market_indicators()
    countries = fetch_country_indicators()

    lines = ["Macro snapshot:"]

    yield_bits = []
    if market.yield_3m_pct is not None:
        yield_bits.append(f"3M {market.yield_3m_pct:.2f}%")
    if market.yield_10y_pct is not None:
        yield_bits.append(f"10Y {market.yield_10y_pct:.2f}%")
    if market.yield_30y_pct is not None:
        yield_bits.append(f"30Y {market.yield_30y_pct:.2f}%")
    if yield_bits:
        line = f"- US Treasury yields: {', '.join(yield_bits)}"
        if market.yield_curve_10y_3m_spread is not None:
            line += f" (10Y-3M spread {market.yield_curve_10y_3m_spread:+.2f}pp)"
        lines.append(line)
    if market.dxy is not None:
        lines.append(f"- US Dollar Index (DXY): {market.dxy:.2f}")
    if market.vix is not None:
        lines.append(f"- VIX (volatility/fear index): {market.vix:.2f}")

    for c in countries:
        bits = []
        if c.gdp_growth_pct is not None:
            bits.append(f"GDP growth {c.gdp_growth_pct:.1f}%")
        if c.inflation_pct is not None:
            bits.append(f"inflation {c.inflation_pct:.1f}%")
        if c.unemployment_pct is not None:
            bits.append(f"unemployment {c.unemployment_pct:.1f}%")
        if bits:
            lines.append(f"- {c.country}: {', '.join(bits)}")

    if len(lines) == 1:
        lines.append("- macro data unavailable right now")

    return "\n".join(lines)


def _detect_present_commodities(assets: list[MarketAsset]) -> set[str]:
    """Distinct resolved display names (e.g. {"Wheat", "Crude Oil", "Gold"})
    actually present in the live Market Watch. Single detection point that
    both the crop weather context and the research directives key off of,
    so coverage always matches what the broker actually offers rather than
    a fixed list."""
    names = set()
    for a in assets:
        resolved = resolve_yahoo_ticker(a.symbol, a.description)
        if resolved is not None:
            names.add(resolved[0])
    return names


def build_crop_supply_demand_context(assets: list[MarketAsset]) -> str:
    """Weather-based supply/demand context for whichever crops (from
    CROP_TRADE_PROFILES) the broker's Market Watch actually contains right
    now. Empty string if none are currently traded."""
    present = _detect_present_commodities(assets)
    crops_present = present & CROP_TRADE_PROFILES.keys()
    if not crops_present:
        return ""
    return fetch_crop_supply_demand_context(crops_present)


def build_research_directives(assets: list[MarketAsset]) -> str:
    """Pure text, no fetching — points the already-granted WebSearch/
    WebFetch access at the specific things asked about, scoped to what's
    actually traded rather than always searching for everything."""
    present = _detect_present_commodities(assets)
    directives = []

    if present & ENERGY_COMMODITIES:
        directives.append(
            "- Energy is traded here: research current conditions, safety, "
            "and traffic trends (increasing/decreasing/normal) at major oil "
            "chokepoints (Strait of Hormuz, Strait of Malacca, Suez Canal, "
            "Panama Canal), and recent geopolitics among major producers "
            "(OPEC+ members, Russia, US shale)."
        )
        directives.append(
            "- Look for recent public outlook/commentary from major energy "
            "companies (e.g. ExxonMobil, Shell, BP, Saudi Aramco)."
        )

    if present & CROP_TRADE_PROFILES.keys():
        directives.append(
            "- Crops are traded here: look for recent public outlook/"
            "commentary from major agricultural traders/processors (e.g. "
            "ADM, Bunge, Cargill, Louis Dreyfus Company) and any current "
            "drought/flood/export-ban news for the exporter/importer "
            "countries listed in the crop supply/demand section below."
        )

    linked_countries_by_asset = {**METAL_LINKED_COUNTRIES, **INDEX_LINKED_COUNTRIES}
    for asset_name in sorted(present & linked_countries_by_asset.keys()):
        countries = linked_countries_by_asset[asset_name]
        directives.append(
            f"- {asset_name} is traded here: it's most linked to "
            f"{', '.join(countries)} — research recent developments there "
            "specifically (mining strikes, sanctions, power/supply "
            "disruptions, trade policy, demand trends) rather than only "
            "generic global commentary."
        )

    if not directives:
        return ""

    return "Research directives:\n" + "\n".join(directives)


def build_fx_context(account: AccountSummary) -> str:
    """Only present when the account's own currency isn't USD — states the
    raw exchange rate as a fact for the model to reason about (base-
    currency vs. USD-priced-instrument mismatch), not a computed risk
    score. Empty string for a USD account (nothing to flag)."""
    if account.currency.upper() == "USD":
        return ""
    rate = fetch_fx_rate_to_usd(account.currency)
    if rate is None:
        return (
            f"FX: account is denominated in {account.currency}, but most "
            "instruments below are USD-priced/settled — live exchange "
            "rate unavailable right now, but the mismatch itself still "
            "applies."
        )
    return (
        f"FX: account is denominated in {account.currency} "
        f"(1 USD ~ {rate:.2f} {account.currency} right now), but most "
        "instruments below are USD-priced/settled — a currency move can "
        f"distort {account.currency}-denominated returns independent of "
        "the instrument's own price action."
    )


def build_portfolio_summary(
    account: AccountSummary,
    assets: list[MarketAsset],
    analyses: list[AssetAnalysis] | None = None,
) -> str:
    """`analyses` lets a caller (e.g. app.py, for charting) compute
    analyze_assets(assets) once and reuse it here, instead of this
    function re-fetching the same data internally."""
    lines = [
        f"Account: balance {account.balance:.2f} {account.currency}, "
        f"equity {account.equity:.2f}, free margin {account.free_margin:.2f}",
        "",
        format_book_wisdom(),
        "",
        build_macro_snapshot(),
    ]

    fx_context = build_fx_context(account)
    if fx_context:
        lines += ["", fx_context]

    crop_context = build_crop_supply_demand_context(assets)
    if crop_context:
        lines += ["", crop_context]

    directives = build_research_directives(assets)
    if directives:
        lines += ["", directives]

    lines += ["", "Tradable instruments (Market Watch):"]
    if not assets:
        lines.append("- none visible in Market Watch")
    else:
        resolved_analyses = analyze_assets(assets) if analyses is None else analyses
        lines.append(format_enriched_asset_context(resolved_analyses, account_equity=account.equity))

    return "\n".join(lines)


_ALLOCATION_BLOCK_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _last_allocation_match(response_text: str) -> re.Match | None:
    """The one ```json block that actually represents the final allocation
    — always the *last* fenced json block in the response (the prompt
    asks for it at the very end). Shared by parse/strip below so they
    never disagree about which block is "the" allocation block."""
    matches = list(_ALLOCATION_BLOCK_PATTERN.finditer(response_text))
    return matches[-1] if matches else None


def parse_final_allocation(response_text: str) -> dict[str, float] | None:
    """Extract the trailing ```json {...}``` allocation block the prompt
    asks for (see SYSTEM_INSTRUCTION). Returns None on anything unexpected
    — missing block, malformed JSON, wrong shape — rather than raising;
    the text response still displays fine even if this fails, it just
    means no allocation chart."""
    match = _last_allocation_match(response_text)
    if match is None:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not data:
        return None
    if not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in data.items()):
        return None
    return {k: float(v) for k, v in data.items()}


def strip_allocation_block(response_text: str) -> str:
    """Removes only that same last ```json allocation block so the
    displayed prose doesn't show a trailing raw JSON dump. Deliberately
    does NOT remove every ```json-fenced block in the text — if the model
    used a code fence earlier for other legitimate content, that content
    stays visible. (Previously used .sub() to strip every match, which
    could silently delete large chunks of the actual explanation if the
    model used more than one fenced block — fixed here.)"""
    match = _last_allocation_match(response_text)
    if match is None:
        return response_text.strip()
    return (response_text[: match.start()] + response_text[match.end() :]).strip()


def suggest_portfolio(summary: str, timeout: int | None = None) -> str:
    # Resolved at call time (not as default-arg values) so a config/env
    # change is picked up without needing to reload this module.
    timeout = config.PORTFOLIO_SUGGESTION_TIMEOUT_SECONDS if timeout is None else timeout
    prompt = f"{SYSTEM_INSTRUCTION}\n\n{summary}"
    return run_claude(
        prompt,
        timeout=timeout,
        allowed_tools=["WebSearch", "WebFetch"],
        model=config.PORTFOLIO_SUGGESTION_MODEL,
    )
