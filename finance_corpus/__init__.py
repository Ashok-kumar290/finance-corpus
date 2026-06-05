"""finance-corpus — generate an ORIGINAL, owned finance/econ/quant instruction dataset
for fine-tuning a finance-reasoning model. Synthetic generation via the Anthropic Batch
API (cheap, cached); no copyrighted material is ingested."""
from .pipeline import run
from .schema import Example, read_jsonl, write_jsonl

# Shared system prompt — used both when authoring training data and at inference, so the
# model sees the same instruction it was trained under.
TRAIN_SYSTEM = (
    "You are a precise finance, economics, and quantitative-methods tutor. "
    "Explain with clear step-by-step reasoning, then state the final answer."
)

__all__ = ["run", "Example", "read_jsonl", "write_jsonl", "TRAIN_SYSTEM"]
