"""Demo module used by the smoke test."""

import math
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    tokens: int


def estimate_tokens(text: str) -> int:
    """Rough token estimate for English text."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return num / den if den else 0.0
