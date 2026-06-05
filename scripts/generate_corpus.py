"""Generate the finance instruction corpus.

    python scripts/generate_corpus.py --mock --limit 10        # offline, no key, no spend
    python scripts/generate_corpus.py --dry-run --limit 50     # build+inspect requests only
    python scripts/generate_corpus.py --limit 200              # real (needs ANTHROPIC_API_KEY)

--mock runs the whole pipeline with deterministic fake items so you can validate the
wiring and the JSONL output offline. Drop --mock (and set ANTHROPIC_API_KEY) for real
generation via the Batch API.
"""
from __future__ import annotations

import argparse
import json

import structlog

from finance_corpus.generate import DEFAULT_MODEL, MockBatchClient, build_requests
from finance_corpus.pipeline import run
from finance_corpus.taxonomy import iter_specs, total_specs

log = structlog.get_logger()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="offline run with fake items (no key)")
    ap.add_argument("--dry-run", action="store_true", help="build + show requests, don't call API")
    ap.add_argument("--limit", type=int, default=None, help="cap number of generation specs")
    ap.add_argument("--items-per-request", type=int, default=4)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="data/corpus.jsonl")
    args = ap.parse_args()

    specs = list(iter_specs(args.limit))
    print(f"specs: {len(specs)} (of {total_specs()} total subtopics) · "
          f"{args.items_per_request} items/request · model={args.model}")

    if args.dry_run:
        reqs = build_requests(specs, model=args.model, items_per_request=args.items_per_request)
        print(f"would submit {len(reqs)} batch requests "
              f"(~{len(reqs) * args.items_per_request} items before dedup)")
        sample = {"custom_id": reqs[0]["custom_id"], "params": reqs[0]["params"]}
        print("\n--- sample request ---")
        print(json.dumps(sample, indent=2)[:1200])
        return

    if args.mock:
        reqs = build_requests(specs, model=args.model, items_per_request=args.items_per_request)
        client = MockBatchClient(reqs)
    else:
        import anthropic
        client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY

    stats = run(client, out_path=args.out, limit=args.limit,
                items_per_request=args.items_per_request, model=args.model)
    print(f"\nwrote {stats['written']} examples -> {stats['out']} "
          f"(raw {stats['raw_items']}, after dedup {stats['after_dedup']})")


if __name__ == "__main__":
    main()
