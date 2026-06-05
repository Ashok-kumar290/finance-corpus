"""finance-corpus — generate an ORIGINAL, owned finance/econ/quant instruction dataset
for fine-tuning a finance-reasoning model. Synthetic generation via the Anthropic Batch
API (cheap, cached); no copyrighted material is ingested."""
from .pipeline import run
from .schema import Example, read_jsonl, write_jsonl

__all__ = ["run", "Example", "read_jsonl", "write_jsonl"]
