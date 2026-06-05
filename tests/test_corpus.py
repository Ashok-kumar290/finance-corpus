"""Corpus pipeline tests — all offline (no network, no API key)."""
from __future__ import annotations

from finance_corpus.dedup import dedup
from finance_corpus.generate import (
    DEFAULT_MODEL,
    MockBatchClient,
    build_requests,
    parse_message_text,
)
from finance_corpus.pipeline import run
from finance_corpus.schema import Example, read_jsonl
from finance_corpus.taxonomy import iter_specs, total_specs


def test_taxonomy_specs():
    assert total_specs() > 30
    specs = list(iter_specs(10))
    assert len(specs) == 10
    for topic, sub, diff, angle in specs:
        assert topic and sub and diff and angle


def test_build_requests_shape_and_caching():
    reqs = build_requests(list(iter_specs(5)), items_per_request=3)
    assert len(reqs) == 5
    ids = [r["custom_id"] for r in reqs]
    assert len(set(ids)) == 5                                  # unique custom_ids
    p = reqs[0]["params"]
    assert p["model"] == DEFAULT_MODEL
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"}   # cached prefix
    assert p["output_config"]["format"]["type"] == "json_schema"      # structured output


def test_parse_message_text():
    assert parse_message_text('{"items": [{"question": "q"}]}') == [{"question": "q"}]
    assert parse_message_text("not json") == []


def test_dedup_drops_exact_and_near():
    base = dict(output="a", reasoning="r", topic="t", subtopic="s", difficulty="d")
    ex = [
        Example(instruction="What is the time value of money in finance", **base),
        Example(instruction="What is the time value of money in finance", **base),   # exact dup
        Example(instruction="What is the time value of money in finance really", **base),  # near
        Example(instruction="Explain duration and convexity for a bond", **base),     # distinct
    ]
    kept = dedup(ex, threshold=0.8)
    assert len(kept) == 2


def test_pipeline_end_to_end_mock(tmp_path):
    reqs = build_requests(list(iter_specs(8)), items_per_request=2)
    client = MockBatchClient(reqs)
    out = tmp_path / "corpus.jsonl"
    stats = run(client, out_path=out, limit=8, items_per_request=2)
    assert stats["written"] > 0
    records = read_jsonl(out)
    rec = records[0]
    assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
    assert "topic" in rec["meta"]
