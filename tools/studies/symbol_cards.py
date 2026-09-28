r"""Rebuild the Symbol Behaviour Cards (analysis/symbol_card.py) from the deep price cache.

    .venv\Scripts\python.exe -m tools.studies.symbol_cards

Refreshes the M5 cache for every Market Watch symbol (needs the MT5 terminal, never sends an order), computes the raw metrics,
shrinks each toward the pool and writes config.SYMBOL_CARD_FILE. Meant monthly (tools.studies.run_all does the same at the end
of its own run); a card older than config.SYMBOL_CARD_STALE_DAYS is labelled stale in the prompt."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from analysis.symbol_card import build_cards, compute_symbol_metrics, save_cards  # noqa: E402
from data.mt5_source import connect, get_market_watch  # noqa: E402
from data.price_cache import deep_history  # noqa: E402


def rebuild(frames: dict) -> dict:
    cards = build_cards({symbol: compute_symbol_metrics(frame) for symbol, frame in frames.items()})
    save_cards(cards)
    return cards


def main() -> None:
    connect(login=config.FTMO_MT5_LOGIN, password=config.FTMO_MT5_PASSWORD, server=config.FTMO_MT5_SERVER)
    frames = {a.symbol: deep_history(a.symbol, "M5", refresh=True) for a in get_market_watch()}
    cards = rebuild(frames)
    print(f"wrote {config.SYMBOL_CARD_FILE}: {len(cards['symbols'])} symbols")


if __name__ == "__main__":
    main()
