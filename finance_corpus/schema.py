"""Fine-tune example schema + JSONL I/O.

An `Example` is one original finance/econ/quant instruction → answer-with-reasoning pair.
We serialize to the standard SFT messages format so it fine-tunes any base model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Example:
    instruction: str            # the question / task
    output: str                 # the correct answer
    reasoning: str              # the worked, step-by-step explanation
    topic: str
    subtopic: str
    difficulty: str
    source: str = "synthetic"
    meta: dict = field(default_factory=dict)

    def to_sft(self, system: str) -> dict:
        """Standard supervised-fine-tune record: a messages list + metadata."""
        answer = f"{self.reasoning}\n\nAnswer: {self.output}".strip()
        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": self.instruction},
                {"role": "assistant", "content": answer},
            ],
            "meta": {
                "topic": self.topic, "subtopic": self.subtopic,
                "difficulty": self.difficulty, "source": self.source, **self.meta,
            },
        }


def write_jsonl(records: list[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]
