"""De-duplication — drop exact and near-duplicate questions.

Synthetic generation occasionally repeats itself; near-dups inflate the corpus without
adding signal and can bias fine-tuning. We drop exact matches (normalized hash) and
near-dups (token-shingle Jaccard over a threshold), keeping the first occurrence.
"""
from __future__ import annotations

import re

from .schema import Example

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def _shingles(text: str, k: int = 5) -> set[str]:
    toks = _norm(text).split()
    if len(toks) < k:
        return {" ".join(toks)}
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedup(examples: list[Example], threshold: float = 0.8) -> list[Example]:
    """Keep the first of any exact- or near-duplicate questions."""
    seen_hashes: set[int] = set()
    kept: list[Example] = []
    kept_shingles: list[set[str]] = []
    for ex in examples:
        h = hash(_norm(ex.instruction))
        if h in seen_hashes:
            continue
        sh = _shingles(ex.instruction)
        if any(_jaccard(sh, prev) >= threshold for prev in kept_shingles):
            continue
        seen_hashes.add(h)
        kept.append(ex)
        kept_shingles.append(sh)
    return kept
