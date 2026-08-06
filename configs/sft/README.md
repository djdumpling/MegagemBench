# SFT config and seed splits

`megagem.toml` is the [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)
SFT config used to train the blueprint checkpoint
(`djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2`). See
[`docs/training.md`](../../docs/training.md) for the full runbook.

## Seed splits

Locked SFT v2 seed splits for chart A (ranges revised 2026-05-12, expanded from
50/10/10). They are contiguous ranges, so they live in code — see
`SPLIT_RANGES` in `src/megagem/training/seed_splits.py`:

| Split | Seeds | Count | Purpose |
| ----- | ----- | ----- | ------- |
| `train` | 1001–1150 | 150 | SFT training examples |
| `val` | 1151–1160 | 10 | checkpoint selection and early stopping |
| `test` | 1161–1170 | 10 | final reporting **only** |

The test split must not be used for training, checkpoint selection, prompt
tuning, or intermediate model debugging.

Anything that takes a seed selection accepts a split name, an inclusive range,
an explicit list, or a path to a seed file (`resolve_seeds` in the same module):

```bash
SEEDS=val               bash scripts/eval/eval_qwen_baseline.sh
SEEDS=30000-30059       bash scripts/eval/eval_qwen_baseline.sh
uv run python scripts/eval/eval_qwen_baseline.py --seeds 1151,1152
```

## Generated data (2026-05-12)

- Raw teacher trajectories (gitignored scratch, `results/sft_v2_train_only/`
  and `results/sft_v2_val_only/`; 150 + 10 games): 70/30 Gemini-3-Flash /
  Claude-Opus-4.6 single-teacher self-play.
- Extracted SFT examples (HuggingFace):
  [`djdumpling/megagem_sft`](https://huggingface.co/datasets/djdumpling/megagem_sft)
  — produced by `src/megagem/evals/prepare_sft_data.py --require-valid` with top-2-by-final-score
  filtering. Two splits: `sft_train.jsonl` and `sft_val.jsonl`.
