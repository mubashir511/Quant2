from ai.claude_cli import run_claude
from data.mt5_source import Position
from risk.rebalance import Suggestion

SYSTEM_INSTRUCTION = (
    "You are a desk-officer narrator for a personal futures trading dashboard. "
    "Explain the positions and suggestions below in plain English. "
    "Never suggest an action that is not already listed below. "
    "Never perform or correct any arithmetic yourself. "
    "Keep the response under 150 words."
)


def build_summary(positions: list[Position], suggestions: list[Suggestion]) -> str:
    lines = ["Open positions:"]
    if not positions:
        lines.append("- none")
    for p in positions:
        lines.append(
            f"- {p.symbol} {p.side} {p.volume} lots, opened {p.price_open}, "
            f"now {p.price_current}, P&L {p.profit:.2f}, "
            f"stop {'none' if p.sl is None else p.sl}"
        )

    lines.append("")
    lines.append("Rebalance suggestions:")
    if not suggestions:
        lines.append("- none, all positions within rules")
    for s in suggestions:
        lines.append(f"- [{s.action}] {s.symbol}: {s.reason}")

    return "\n".join(lines)


def narrate(summary: str, timeout: int = 60) -> str:
    prompt = f"{SYSTEM_INSTRUCTION}\n\n{summary}"
    return run_claude(prompt, timeout=timeout)
