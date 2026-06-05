# finance-corpus

Generates an **original, owned** finance / economics / quant **instruction dataset** for
fine-tuning a finance-reasoning model (the brain behind the QuantCopilot analyst layer).

**Clean by construction:** every example is *synthetically generated* from a topic taxonomy
(generic subject labels — uncopyrightable facts), with an explicit instruction never to copy
any textbook, exam-prep, or question-bank text. No copyrighted material is ingested, so the
dataset is genuinely your IP and defensible in diligence.

## How it works
- **Topic taxonomy** (`taxonomy.py`) → (topic, subtopic, difficulty, angle) specs.
- **Batch API generation** (`generate.py`) → one request per spec, returning a few original
  Q&A+reasoning items as **structured JSON**. A **frozen, cached system prompt** is shared
  across all requests (prefix caching), and the **Batch API** runs at **50% cost**.
- **Dedup** (`dedup.py`) → drop exact + near-duplicate questions.
- **Output**: standard SFT **JSONL** (`messages` + `meta`) — fine-tunes any base model.

## Run
```bash
pip install -e ".[dev]"
python scripts/generate_corpus.py --mock --limit 10      # offline, no key, no spend
python scripts/generate_corpus.py --dry-run --limit 50   # inspect requests before paying
export ANTHROPIC_API_KEY=...                              # real generation
python scripts/generate_corpus.py --limit 200            # via Batch API
pytest                                                    # offline tests
```

## Cost & model notes
- Defaults to `claude-opus-4-8` (best quality). For high-volume bulk, pass
  `--model claude-sonnet-4-6` (cheaper — your choice, not auto-selected).
- **Caching only engages above the model's minimum prefix** (~4096 tokens Opus 4.8,
  ~2048 Sonnet 4.6). Enrich `SYSTEM_PROMPT` with a longer style guide / worked exemplars to
  cross it; verify with `usage.cache_read_input_tokens > 0`.
- Batch API: up to 100k requests/batch, ~1h typical, results kept 29 days.

## Then what
JSONL → fine-tune an open base model (RunPod/Lambda) → benchmark with a **self-generated,
CFA-style eval** (built from the public Learning Outcome topics, original questions) →
wire the model into QuantCopilot as the reasoning/analyst layer.
