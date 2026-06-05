"""Validate + summarize a corpus JSONL before spending GPU on fine-tuning.

    python scripts/inspect_corpus.py --data data/corpus.jsonl

Checks every record is a well-formed system/user/assistant triple with non-empty content,
and reports size, a rough token estimate, and topic/difficulty coverage. Runs offline.
"""
from __future__ import annotations

import argparse
from collections import Counter

from finance_corpus.schema import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/corpus.jsonl")
    args = ap.parse_args()

    records = read_jsonl(args.data)
    bad = 0
    chars = 0
    topics: Counter = Counter()
    diffs: Counter = Counter()
    for r in records:
        msgs = r.get("messages", [])
        roles = [m.get("role") for m in msgs]
        ok = roles == ["system", "user", "assistant"] and all(
            (m.get("content") or "").strip() for m in msgs
        )
        bad += 0 if ok else 1
        chars += sum(len(m.get("content", "")) for m in msgs)
        meta = r.get("meta", {})
        topics[meta.get("topic", "?")] += 1
        diffs[meta.get("difficulty", "?")] += 1

    approx_tokens = chars // 4   # ~4 chars/token, rough
    print(f"\nrecords: {len(records)}  ·  malformed: {bad}  ·  ~{approx_tokens:,} tokens "
          f"(~{chars:,} chars)")
    print(f"avg ~{(approx_tokens // max(1, len(records))):,} tokens/example")
    print("\nby difficulty:", dict(diffs))
    print("top topics:", dict(topics.most_common(8)))
    if bad:
        print(f"\n⚠️  {bad} malformed records — fix or drop before training.")
    else:
        print("\n✅ all records well-formed.")
    # fine-tune sizing hint
    if len(records) < 500:
        print("note: <500 examples — fine for a smoke run; aim for a few-thousand for a "
              "real fine-tune (quality > volume).")


if __name__ == "__main__":
    main()
