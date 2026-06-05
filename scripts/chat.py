"""Talk to the fine-tuned model: load the base + your LoRA adapter and answer a question.

    python scripts/chat.py --adapter adapters/finance-smoke \
        --question "A bond has a $40 coupon and trades at $950. What is its current yield?"

This closes the loop (train → load → use). Note: a *smoke* adapter (few examples) will
behave close to the base model — it proves serving works, not that the model got smart.
That needs a real corpus + real run.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="adapters/finance-smoke")
    ap.add_argument("--question", default="Explain duration and convexity for a bond.")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from finance_corpus import TRAIN_SYSTEM

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="auto", dtype=dtype)
    model = PeftModel.from_pretrained(model, args.adapter)   # apply the fine-tuned adapter
    model.eval()

    messages = [{"role": "system", "content": TRAIN_SYSTEM},
                {"role": "user", "content": args.question}]
    # return_dict=True -> a BatchEncoding (input_ids + attention_mask); pass with **enc.
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
    answer = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    print(f"\nQ: {args.question}\n")
    print(f"A: {answer}\n")


if __name__ == "__main__":
    main()
