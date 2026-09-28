"""Out-of-sample gate: a rule may be adopted only if its effect survives being split in time and across symbols.

Books say it (Schwager: test on data not used to build the rule; Lo: rules decay) and our own numbers say it (the
per-symbol level hold-rate on 3 weeks of M5 had a split-half correlation of 0.36 - mostly noise). Every study that
proposes a default-on rule reports its result through `oos_gate`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    first_half: float | None = None
    second_half: float | None = None
    symbol_share: float | None = None  # share of symbols whose effect has the required sign


def oos_gate(
    first_half: float | None,
    second_half: float | None,
    per_symbol: list[float] | None = None,
    min_effect: float = 0.0,
    min_symbol_share: float = 0.7,
    positive: bool = True,
) -> GateResult:
    """PASS needs: BOTH time halves beyond `min_effect` in the wanted direction (positive = larger is better), and at
    least `min_symbol_share` of the per-symbol effects in that direction. Missing inputs FAIL (unknown is never a pass)."""
    sign = 1.0 if positive else -1.0
    reasons: list[str] = []
    for name, value in (("first half", first_half), ("second half", second_half)):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            reasons.append(f"{name} is unavailable")
        elif sign * value <= min_effect:
            reasons.append(f"{name} effect {value:+.3f} does not clear {sign * min_effect:+.3f}")
    share = None
    if per_symbol is not None:
        arr = np.array([x for x in per_symbol if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
        if len(arr) == 0:
            reasons.append("no per-symbol effects")
        else:
            share = float(np.mean(sign * arr > min_effect))
            if share < min_symbol_share:
                reasons.append(f"only {share:.0%} of symbols show the effect (need {min_symbol_share:.0%})")
    else:
        reasons.append("per-symbol effects were not supplied")
    return GateResult(passed=not reasons, reasons=reasons, first_half=first_half, second_half=second_half, symbol_share=share)


def split_halves(values_by_time: list[tuple[object, float]]) -> tuple[float | None, float | None]:
    """Mean of the earlier and the later half of (timestamp, value) pairs split at the median timestamp."""
    if len(values_by_time) < 4:
        return None, None
    ordered = sorted(values_by_time, key=lambda tv: tv[0])
    mid = len(ordered) // 2
    return float(np.mean([v for _, v in ordered[:mid]])), float(np.mean([v for _, v in ordered[mid:]]))
