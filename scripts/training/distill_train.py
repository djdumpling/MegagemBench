#!/usr/bin/env python3
"""ExIt distill (the 'fast'/generalize step): train a LoRA adapter toward the
piKL targets via in-image TRL SFTTrainer.

(Formerly ``scripts/phase3/exit_distill_train.py``.)

prime-rl is NOT in the phase3 Modal image (modal_common.py ignores it), so the
distill uses TRL's SFTTrainer — which, given {prompt, completion} rows, masks
the prompt and trains on the assistant completion. The piKL exporter that
originally fed this trainer was deleted; the corpus builder of record is now
``scripts/training/distill_export.py``. Reuses the SAME LoRA config as GRPO
(``megagem_lora_config``). Saves an HF-format adapter for vLLM hot-swap via
``megagem.training.adapter_sync.push_adapter_to_vllm``.

$0 Prime. Sparse distill data ⇒ low lr + few epochs + start-from-(merged)-policy
= the anchor. The lambda=inf data is the policy's own modal action, so its distill
should be a near no-op — the built-in forgetting / sanity check.

Heavy imports (torch/trl) are inside main() so the file imports/compiles without
a GPU; it RUNS on the box only.
"""

from __future__ import annotations

import argparse


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="TRL LoRA distill toward piKL targets (ExIt fast step).")
    ap.add_argument("--base-model", required=True,
                    help="merged current policy (iter0 = the SFT blueprint)")
    ap.add_argument("--data", required=True, help="distill_export {prompt,completion} JSONL")
    ap.add_argument("--out", required=True, help="adapter output dir")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--init-adapter", default="",
                    help="continue training an EXISTING LoRA adapter (round-2 "
                         "distill: learn the residual on top of round 1) "
                         "instead of a fresh peft_config")
    args = ap.parse_args(argv)

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from megagem.training.grpo_harness import megagem_lora_config  # same LoRA as GRPO

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2")
    # avoid the benign None-grad warning with gradient checkpointing + LoRA
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    ds = load_dataset("json", data_files=args.data, split="train")
    n = len(ds)
    if n == 0:
        print("[exit-distill-train] ABORT: 0 distill rows", flush=True)
        return 2

    peft_cfg = megagem_lora_config()
    if args.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter,
                                          is_trainable=True)
        peft_cfg = None   # trainer receives an already-wrapped PeftModel
        print(f"[exit-distill-train] continuing adapter {args.init_adapter}",
              flush=True)

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=args.max_seq_len,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=2,
        save_strategy="no",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds,
        processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.out)   # PEFT ⇒ adapter-only
    print(f"[exit-distill-train] n_examples={n} epochs={args.epochs} lr={args.lr} "
          f"-> adapter {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
