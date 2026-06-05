#!/usr/bin/env bash
# One-time setup for QLoRA fine-tuning on a fresh Colab/RunPod GPU box (torch pre-installed).
set -euo pipefail

pip install -q -U transformers peft trl bitsandbytes accelerate datasets
pip install -q -e .   # finance_corpus (for the corpus inspector)

python - <<'PY'
import torch
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
echo "setup done."
echo "Faster alternative if versions drift: use the maintained Unsloth Qwen2.5 Colab"
echo "notebook and point it at data/corpus.jsonl (same data, maintained training loop)."
