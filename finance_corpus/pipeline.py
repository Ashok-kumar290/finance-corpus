"""Orchestrate: specs -> batch requests -> generate -> Examples -> dedup -> JSONL."""
from __future__ import annotations

from pathlib import Path

import structlog

from .dedup import dedup
from .generate import DEFAULT_MODEL, SYSTEM_PROMPT, build_requests, generate
from .schema import Example, write_jsonl
from .taxonomy import iter_specs

log = structlog.get_logger()


def run(
    client,
    out_path: str | Path = "data/corpus.jsonl",
    limit: int | None = None,
    items_per_request: int = 4,
    model: str = DEFAULT_MODEL,
    dedup_threshold: float = 0.8,
) -> dict:
    """Generate the corpus and write SFT-format JSONL. `client` is anthropic.Anthropic()
    (or a MockBatchClient built from the same requests for offline runs)."""
    specs = list(iter_specs(limit))
    requests = build_requests(specs, model=model, items_per_request=items_per_request)
    log.info("corpus.requests_built", requests=len(requests), model=model,
             items_per_request=items_per_request)

    pairs = generate(client, requests)
    by_id = {r["custom_id"]: r["_spec"] for r in requests}

    examples: list[Example] = []
    for cid, item in pairs:
        spec = by_id.get(cid, {})
        if not (item.get("question") and item.get("answer")):
            continue
        examples.append(Example(
            instruction=item["question"], output=item["answer"],
            reasoning=item.get("reasoning", ""),
            topic=spec.get("topic", ""), subtopic=spec.get("subtopic", ""),
            difficulty=spec.get("difficulty", ""),
        ))

    deduped = dedup(examples, threshold=dedup_threshold)
    records = [ex.to_sft(SYSTEM_PROMPT) for ex in deduped]
    n = write_jsonl(records, out_path)
    stats = {"raw_items": len(examples), "after_dedup": len(deduped), "written": n,
             "out": str(out_path)}
    log.info("corpus.done", **stats)
    return stats
