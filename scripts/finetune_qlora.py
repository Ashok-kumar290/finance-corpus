"""QLoRA fine-tune of an open base model on the finance corpus (single 24GB+ GPU / Colab L4).

    # smoke run first — tiny base, few samples, prove the loop works
    python scripts/finetune_qlora.py --model Qwen/Qwen2.5-1.5B-Instruct --max-samples 200 \
        --data data/corpus.jsonl --out adapters/finance-smoke --epochs 1

    # real run once it works
    python scripts/finetune_qlora.py --model Qwen/Qwen2.5-7B-Instruct --data data/corpus.jsonl \
        --out adapters/finance-7b --epochs 3

4-bit (NF4) loads the base so a 7B fits on ~16GB; LoRA trains a small adapter on top. The
data is our `messages`-format JSONL — TRL applies the model's chat template automatically.

⚠️ VERSION NOTE: the fine-tune stack (transformers / peft / trl / bitsandbytes) moves fast.
This is a known-good *pattern*, but if Colab installs versions where an arg has shifted,
the maintained **Unsloth Qwen2.5 Colab notebook** is a drop-in alternative — point it at
this same JSONL. Don't fight version drift; use the maintained notebook and keep our data.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", default="data/corpus.jsonl")
    ap.add_argument("--out", default="adapters/finance")
    ap.add_argument("--max-samples", type=int, default=None, help="cap dataset (smoke runs)")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    args = ap.parse_args()

    # imported here so --help works without the heavy GPU stack installed
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    print(f"base={args.model} · data={args.data} · epochs={args.epochs} · "
          f"effective batch={args.batch * args.grad_accum}")

    use_bf16 = torch.cuda.is_bf16_supported()   # T4/Turing lack bf16 -> fall back to fp16
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"precision: {'bf16' if use_bf16 else 'fp16 (T4/Turing GPU)'}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto", torch_dtype=dtype,
    )

    ds = load_dataset("json", data_files=args.data, split="train")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"training on {len(ds)} examples")

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    cfg = SFTConfig(
        output_dir=args.out, per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=use_bf16, fp16=not use_bf16, logging_steps=10, save_strategy="epoch",
        max_length=args.max_seq_len, packing=False, report_to="none",   # TRL renamed max_seq_length->max_length
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora,
                         processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"\n✅ LoRA adapter saved → {args.out}")
    print("   merge for serving, or load base+adapter at inference time.")


if __name__ == "__main__":
    main()
