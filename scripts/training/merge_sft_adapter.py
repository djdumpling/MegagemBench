#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model with a clean tokenizer config,
then optionally push the result to the Hub.

The public ``djdumpling/qwen3-4b-instruct-megagem-sft-step1200`` checkpoint's
``tokenizer_config.json`` stores ``extra_special_tokens`` as a list rather
than a dict, which crashes ``AutoTokenizer.from_pretrained`` on transformers
4.55.4 (locked-decisions caveat, 2026-05-15). This script
re-merges from ``Qwen/Qwen3-4B-Instruct-2507`` + the public LoRA adapter,
**saves the tokenizer from the base** (not from the broken merged repo), and
verifies the resulting tokenizer loads before exiting.

  # Local-only merge (verifies tokenizer):
  python scripts/training/merge_sft_adapter.py \\
      --adapter djdumpling/qwen3-4b-instruct-megagem-sft-step1200-lora \\
      --output-dir checkpoints/megagem-sft-step1200-clean

  # Merge + push to Hub:
  python scripts/training/merge_sft_adapter.py \\
      --adapter djdumpling/qwen3-4b-instruct-megagem-sft-step1200-lora \\
      --output-dir checkpoints/megagem-sft-step1200-clean \\
      --push-to djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2 \\
      --private
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument(
        "--adapter",
        default="djdumpling/qwen3-4b-instruct-megagem-sft-step1200-lora",
        help="Hub repo id or local path to the LoRA adapter.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to save the merged checkpoint locally.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="cuda / cpu / mps. Defaults to cuda if available, else cpu.",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--push-to",
        default=None,
        help="If set, push merged model to this Hub repo id after saving locally.",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="When pushing, create the repo as private (no effect on existing repos).",
    )
    p.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the post-save AutoTokenizer.from_pretrained sanity load.",
    )
    return p.parse_args()


def pick_device(requested: str | None) -> str:
    import torch
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    args = parse_args()

    # Heavy imports inside main so --help is snappy and the script's intent is
    # obvious from its docstring without dragging in torch on import.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = pick_device(args.device)
    print(f"[merge] device={device} dtype={args.dtype}")
    print(f"[merge] base={args.base}")
    print(f"[merge] adapter={args.adapter}")

    # Pre-flight VRAM check: if a vLLM server from pre_rl_burst is still up,
    # cuda:0 will be mostly full and the model load will OOM with a less
    # obvious error. Surface it here instead.
    if device.startswith("cuda"):
        idx = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(idx)
        free_gb, total_gb = free / 1e9, total / 1e9
        print(f"[merge] cuda:{idx} VRAM free={free_gb:.1f} GB / total={total_gb:.1f} GB")
        # bf16 4B model needs ~8 GB; demand a 4 GB headroom margin.
        need_gb = 8.0 if args.dtype == "bfloat16" else (8.0 if args.dtype == "float16" else 16.0)
        if free_gb < need_gb + 4.0:
            print(
                f"[merge] WARN: only {free_gb:.1f} GB free, need ~{need_gb:.0f} GB + headroom. "
                f"Kill any vLLM server (or set CUDA_VISIBLE_DEVICES) before retrying.",
                file=sys.stderr,
            )

    # Tokenizer comes from the base — this is the whole point. The broken
    # public merged checkpoint has it as a list; the base has it as a dict.
    print("[merge] loading tokenizer from base…")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    # device_map=device skips the CPU materialization + copy that
    # ``.to(device)`` after ``from_pretrained`` causes — saves ~8 GB peak
    # host RAM and ~30 s of wall time for the 4 B base.
    print("[merge] loading base model weights…")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map=device, trust_remote_code=True
    )

    print("[merge] attaching adapter…")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)

    print("[merge] merging LoRA into base weights…")
    merged = peft_model.merge_and_unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = str(args.output_dir.resolve())
    print(f"[merge] saving model → {out}")
    merged.save_pretrained(out)
    print(f"[merge] saving tokenizer (from base) → {out}")
    tokenizer.save_pretrained(out)

    if not args.skip_verify:
        print("[merge] verifying tokenizer reloads from saved dir…")
        try:
            reloaded = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
        except Exception as e:
            print(f"[merge] FAIL: tokenizer reload raised {type(e).__name__}: {e}", file=sys.stderr)
            print("[merge] the merged checkpoint is on disk but the tokenizer is busted.", file=sys.stderr)
            return 2
        # The bug we're guarding against would crash at __init__; if we got
        # here, vocab access is a sanity ping.
        print(f"[merge] tokenizer reload OK (vocab={reloaded.vocab_size})")

    if args.push_to:
        print(f"[merge] pushing to Hub repo={args.push_to} private={args.private}")
        merged.push_to_hub(args.push_to, private=args.private)
        tokenizer.push_to_hub(args.push_to, private=args.private)
        print(f"[merge] pushed: https://huggingface.co/{args.push_to}")
    else:
        print("[merge] done. To push: --push-to <repo_id> [--private]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
