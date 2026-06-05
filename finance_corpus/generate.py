"""Synthetic generation via the Anthropic Batch API.

Each request asks the model to write a few ORIGINAL finance items for one (topic, subtopic,
difficulty, angle) spec, returned as strict JSON (structured outputs). All requests share
one frozen system prompt carrying a `cache_control` breakpoint, so after the first write
the shared prefix is served from cache. Combined with the Batch API's 50% discount, bulk
generation is cheap.

Defaults to claude-opus-4-8 (highest quality). For high-volume bulk you may pass
--model claude-sonnet-4-6 (your call — it's cheaper, not auto-selected).

CACHING NOTE: prompt caching only engages when the cached prefix exceeds the model's
minimum (~4096 tokens for Opus 4.8, ~2048 for Sonnet 4.6). The SYSTEM_PROMPT below is
detailed but may fall under that; enrich it with a longer style guide / worked exemplars
to cross the threshold, and verify with usage.cache_read_input_tokens > 0.
"""
from __future__ import annotations

import json
import time

DEFAULT_MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = (
    "You are a meticulous finance, economics, and quantitative-methods educator writing "
    "ORIGINAL training material for a finance-reasoning model. For each request you produce "
    "a small set of self-contained study items.\n\n"
    "Hard rules:\n"
    "1. Write everything in your own words. Do NOT copy, paraphrase, or reconstruct any "
    "copyrighted textbook, exam-prep, or question-bank text. Generate fresh content from "
    "the underlying concepts (which are public knowledge).\n"
    "2. Every item must be factually correct and internally consistent. For numerical "
    "problems, do the arithmetic carefully and show each step.\n"
    "3. Each item has three parts: a clear QUESTION, the correct ANSWER, and REASONING — a "
    "step-by-step worked explanation a learner could follow.\n"
    "4. For multiple-choice items, embed the options in the question text and make the "
    "distractors plausible; put the correct option in ANSWER and explain why the others are "
    "wrong in REASONING.\n"
    "5. Be precise with definitions, formulas, and units. State assumptions explicitly.\n"
    "6. Vary phrasing and structure across items so they are not near-duplicates.\n\n"
    "Return ONLY the JSON object specified by the response schema — no preamble."
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["question", "answer", "reasoning"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def _user_prompt(topic: str, subtopic: str, difficulty: str, angle: str, n: int) -> str:
    return (
        f"Topic: {topic.replace('_', ' ')}\nSubtopic: {subtopic}\n"
        f"Difficulty: {difficulty}\nStyle for these items: {angle}\n\n"
        f"Write {n} distinct, original {difficulty}-level items on '{subtopic}'. "
        f"Each should reflect the requested style. Return them as the JSON 'items' array."
    )


def build_requests(specs, model: str = DEFAULT_MODEL, items_per_request: int = 4) -> list[dict]:
    """Build Batch API request dicts. Pure (no network) — safe to inspect/test/dry-run.

    The shared `system` block (with cache_control) is identical across every request, so
    the cached prefix is reused; only the per-spec user message varies.
    """
    shared_system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    requests = []
    for i, (topic, subtopic, difficulty, angle) in enumerate(specs):
        requests.append({
            "custom_id": f"{topic}.{subtopic}.{difficulty}.{i}".replace(" ", "_")[:64],
            "params": {
                "model": model,
                "max_tokens": 4000,
                "system": shared_system,
                "output_config": {"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
                "messages": [{
                    "role": "user",
                    "content": _user_prompt(topic, subtopic, difficulty, angle, items_per_request),
                }],
            },
            "_spec": {"topic": topic, "subtopic": subtopic, "difficulty": difficulty},
        })
    return requests


def _strip_internal(requests: list[dict]) -> list[dict]:
    """Remove our `_spec` annotation before sending to the API."""
    return [{"custom_id": r["custom_id"], "params": r["params"]} for r in requests]


def parse_message_text(text: str) -> list[dict]:
    """output_config.format guarantees the first text block is valid JSON for the schema."""
    try:
        return json.loads(text).get("items", [])
    except (json.JSONDecodeError, AttributeError):
        return []


def generate(client, requests: list[dict], poll_interval: int = 30) -> list[tuple[str, dict]]:
    """Submit a batch, poll to completion, return (custom_id, item) pairs.

    `client` is an anthropic.Anthropic() (or a compatible mock). Runs on the user's machine.
    """
    batch = client.messages.batches.create(requests=_strip_internal(requests))
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        time.sleep(poll_interval)

    out: list[tuple[str, dict]] = []
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            continue
        msg = result.result.message
        text = next((blk.text for blk in msg.content if blk.type == "text"), "")
        for item in parse_message_text(text):
            out.append((result.custom_id, item))
    return out


# --------------------------------------------------------------- offline mock
class MockBatchClient:
    """Minimal stand-in for anthropic.Anthropic().messages.batches — runs the whole
    pipeline offline (no key, no spend) with deterministic fake items, so the wiring can
    be tested and demoed. Real generation swaps this for anthropic.Anthropic()."""

    class _Msg:
        def __init__(self, text): self.content = [type("B", (), {"type": "text", "text": text})()]

    class _Res:
        def __init__(self, cid, text):
            self.custom_id = cid
            self.result = type("R", (), {"type": "succeeded", "message": MockBatchClient._Msg(text)})()

    def __init__(self, requests: list[dict]):
        self._requests = requests

    class _Batches:
        def __init__(self, outer): self.outer = outer
        def create(self, requests):
            self._reqs = requests
            return type("Batch", (), {"id": "mock_batch", "processing_status": "in_progress"})()
        def retrieve(self, _id):
            return type("Batch", (), {"id": _id, "processing_status": "ended"})()
        def results(self, _id):
            for r in self.outer._requests:
                items = {"items": [{
                    "question": f"[mock] Q on {r['_spec']['subtopic']} ({r['_spec']['difficulty']})",
                    "answer": "[mock] answer",
                    "reasoning": "[mock] step-by-step reasoning.",
                }]}
                yield MockBatchClient._Res(r["custom_id"], json.dumps(items))

    @property
    def messages(self):
        return type("M", (), {"batches": MockBatchClient._Batches(self)})()
