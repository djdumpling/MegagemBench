#!/usr/bin/env python3
"""
Preprocess MegaGem SFT data for Qwen3-4B-Instruct.

Loads the HuggingFace dataset of record (``djdumpling/megagem_sft``) or locally
exported SFT examples, formats assistant targets for Qwen3-4B-Instruct, and
saves seed-disjoint train/val splits.

Usage:
    python -m megagem.training.preprocess_sft            # pulls the HF dataset
    python -m megagem.training.preprocess_sft \
        --train-jsonl data/sft_v2_examples/sft_train.jsonl \
        --val-jsonl data/sft_v2_examples/sft_val.jsonl    # local export
    python -m megagem.training.preprocess_sft --output-dir data/sft_processed
"""

import argparse
import json
import re
from pathlib import Path

from datasets import Dataset, DatasetDict

from megagem.training.seed_splits import load_seed_file, load_split_seeds


# src/megagem/training/preprocess_sft.py -> repo root is four levels up.
REPO_ROOT = Path(__file__).resolve().parents[3]
# Local fallbacks are gitignored scratch (data/); the source of record is the
# HF dataset djdumpling/megagem_sft, used when no local export is present.
DEFAULT_INPUT_JSONL = REPO_ROOT / "data" / "sft_v2_examples" / "sft_trainval.jsonl"
DEFAULT_V2_TRAIN_JSONL = REPO_ROOT / "data" / "sft_v2_examples" / "sft_train.jsonl"
DEFAULT_V2_VAL_JSONL = REPO_ROOT / "data" / "sft_v2_examples" / "sft_val.jsonl"
DEFAULT_TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"


def unwrap_think_tags(assistant_content: str) -> str:
    """Remove literal think tags while preserving their reasoning text."""
    return re.sub(r"</?think>", "", assistant_content).strip()


def _decode_jsonish(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def process_example(example: dict) -> dict:
    """Process a single SFT example for Qwen3-4B-Instruct."""
    messages = example["messages"]
    # Handle messages stored as JSON string (v2 dataset) or list (v1)
    if isinstance(messages, str):
        messages = json.loads(messages)
    new_messages = []

    for msg in messages:
        if msg["role"] == "assistant":
            new_messages.append({"role": msg["role"], "content": unwrap_think_tags(msg["content"])})
        else:
            new_messages.append(msg)

    metadata = example.get("metadata")
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    if len(new_messages) < 2 or new_messages[-1]["role"] != "assistant":
        raise ValueError("SFT examples must end with an assistant message")

    return {
        "prompt": new_messages[:-1],
        "completion": [new_messages[-1]],
        "metadata": metadata,
    }


def print_chat_template_preview(example: dict, tokenizer_name: str) -> None:
    """Render and print one training example through the tokenizer chat template."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        messages = example.get("messages") or (example["prompt"] + example["completion"])
        rendered = tok.apply_chat_template(messages, tokenize=False)
    except Exception as exc:
        print(f"\nWARNING: Could not render {tokenizer_name} chat-template preview: {exc}")
        return

    print(f"\n--- Rendered {tokenizer_name} chat-template preview ---")
    print(rendered[:2000] + ("..." if len(rendered) > 2000 else ""))


def seed_from_metadata(metadata: dict | None) -> int | None:
    """Extract seed from v2 metadata, falling back to ``game_file`` names."""
    if not isinstance(metadata, dict):
        return None
    seed = metadata.get("seed")
    if seed is not None:
        try:
            return int(seed)
        except (TypeError, ValueError):
            return None
    game_file = metadata.get("game_file")
    if isinstance(game_file, str):
        match = re.search(r"(?:^|_)seed_(\d+)(?:\.json)?$", game_file)
        if match:
            return int(match.group(1))
    return None


def load_raw_dataset(
    input_jsonl: Path | None,
    dataset_name: str,
    no_local_fallback: bool = False,
) -> Dataset:
    """Load local JSONL when available, otherwise fall back to HuggingFace.

    Set ``no_local_fallback=True`` to skip the DEFAULT_INPUT_JSONL path even
    when it exists on disk — used by automated pipelines that must guarantee
    they're consuming the latest HF dataset, not a stale local cache.
    """
    from datasets import concatenate_datasets, load_dataset

    if input_jsonl is not None:
        print(f"Loading dataset from local JSONL: {input_jsonl}")
        return _load_json_split(input_jsonl, "train")
    if not no_local_fallback and DEFAULT_INPUT_JSONL.exists():
        print(f"Loading dataset from default local JSONL: {DEFAULT_INPUT_JSONL}")
        return _load_json_split(DEFAULT_INPUT_JSONL, "train")
    print(f"Loading dataset from HuggingFace: {dataset_name}")
    # Name data files explicitly so HF's auto-discovery doesn't also pull in
    # the `*_stats.json` sidecar files (different schema -> CastError). Pulls
    # both train and val, then concatenates so the seed-based split downstream
    # routes rows correctly (val seeds live only in sft_val.jsonl).
    splits = load_dataset(
        dataset_name,
        data_files={
            "train": "sft_train.jsonl",
            "validation": "sft_val.jsonl",
        },
    )
    return concatenate_datasets([splits["train"], splits["validation"]])


def _load_json_split(path: Path, split_name: str) -> Dataset:
    """Load one JSONL file as a HuggingFace Dataset split."""
    print(f"Loading {split_name} dataset from local JSONL: {path}")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return Dataset.from_list(rows)


def _seed_set(rows: Dataset, split_name: str) -> set[int]:
    """Return all seeds present in a split, raising on missing seed metadata."""
    seeds: set[int] = set()
    missing_seed = 0
    for row in rows:
        metadata = _decode_jsonish(row.get("metadata"))
        seed = seed_from_metadata(metadata)
        if seed is None:
            missing_seed += 1
        else:
            seeds.add(seed)
    if missing_seed:
        raise ValueError(
            f"{missing_seed} {split_name} examples have no seed metadata. "
            "Regenerate data with src/megagem/evals/prepare_sft_data.py."
        )
    return seeds


def load_presplit_dataset(
    train_jsonl: Path,
    val_jsonl: Path,
    train_seeds: set[int],
    val_seeds: set[int],
) -> tuple[DatasetDict, dict]:
    """Load already seed-disjoint train/validation JSONL files."""
    train_rows = _load_json_split(train_jsonl, "train")
    val_rows = _load_json_split(val_jsonl, "validation")

    train_seen = _seed_set(train_rows, "train")
    val_seen = _seed_set(val_rows, "validation")
    overlap = train_seen & val_seen
    if overlap:
        raise ValueError(f"Train and validation JSONLs overlap on seeds: {sorted(overlap)}")

    unexpected_train = train_seen - train_seeds
    unexpected_val = val_seen - val_seeds
    if unexpected_train:
        raise ValueError(
            "Train JSONL contains seeds outside the train split: "
            f"{sorted(unexpected_train)}"
        )
    if unexpected_val:
        raise ValueError(
            "Validation JSONL contains seeds outside the validation split: "
            f"{sorted(unexpected_val)}"
        )

    processed_train = train_rows.map(process_example, remove_columns=train_rows.column_names)
    processed_val = val_rows.map(process_example, remove_columns=val_rows.column_names)

    return (
        DatasetDict({"train": processed_train, "validation": processed_val}),
        {
            "mode": "files",
            "train_examples": len(processed_train),
            "validation_examples": len(processed_val),
            "train_input_jsonl": str(train_jsonl),
            "validation_input_jsonl": str(val_jsonl),
            "train_seed_count_in_data": len(train_seen),
            "validation_seed_count_in_data": len(val_seen),
        },
    )


def split_by_seed(
    rows: Dataset,
    train_seeds: set[int],
    val_seeds: set[int],
) -> tuple[DatasetDict, dict]:
    """Create train/validation datasets using committed seed assignments."""
    train_rows = []
    val_rows = []
    skipped_other_seed = 0
    missing_seed = 0

    for row in rows:
        metadata = _decode_jsonish(row.get("metadata"))
        seed = seed_from_metadata(metadata)
        if seed is None:
            missing_seed += 1
            continue
        if "messages" in row:
            example = {
                "messages": _decode_jsonish(row["messages"]),
                "metadata": metadata,
            }
        else:
            example = {
                "prompt": _decode_jsonish(row["prompt"]),
                "completion": _decode_jsonish(row["completion"]),
                "metadata": metadata,
            }
        if seed in train_seeds:
            train_rows.append(example)
        elif seed in val_seeds:
            val_rows.append(example)
        else:
            skipped_other_seed += 1

    if missing_seed:
        raise ValueError(
            f"{missing_seed} examples have no seed metadata. Regenerate data with "
            "src/megagem/evals/prepare_sft_data.py so train/val can be split by seed."
        )

    return (
        DatasetDict({
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(val_rows),
        }),
        {
            "train_examples": len(train_rows),
            "validation_examples": len(val_rows),
            "skipped_other_seed": skipped_other_seed,
            "missing_seed": missing_seed,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess MegaGem SFT data for Qwen3-4B-Instruct")
    parser.add_argument("--dataset", default="djdumpling/megagem_sft")
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", default="data/sft_processed")
    parser.add_argument("--train-seed-file", type=Path, default=None)
    parser.add_argument("--val-seed-file", type=Path, default=None)
    parser.add_argument("--template-tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument('--no-local-fallback', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_seeds = set(
        load_seed_file(args.train_seed_file)
        if args.train_seed_file
        else load_split_seeds("train")
    )
    val_seeds = set(
        load_seed_file(args.val_seed_file)
        if args.val_seed_file
        else load_split_seeds("val")
    )
    overlap = train_seeds & val_seeds
    if overlap:
        raise ValueError(f"Train and validation seed files overlap: {sorted(overlap)}")

    train_jsonl = args.train_jsonl
    val_jsonl = args.val_jsonl
    if train_jsonl is None and val_jsonl is None:
        if (
            not args.no_local_fallback
            and DEFAULT_V2_TRAIN_JSONL.exists()
            and DEFAULT_V2_VAL_JSONL.exists()
        ):
            train_jsonl = DEFAULT_V2_TRAIN_JSONL
            val_jsonl = DEFAULT_V2_VAL_JSONL
    elif train_jsonl is None or val_jsonl is None:
        raise ValueError("--train-jsonl and --val-jsonl must be provided together")

    if train_jsonl is not None and val_jsonl is not None:
        print("Formatting already split train/validation files for Qwen3-4B-Instruct...")
        dataset_dict, split_stats = load_presplit_dataset(
            train_jsonl,
            val_jsonl,
            train_seeds,
            val_seeds,
        )
    else:
        dataset = load_raw_dataset(
            args.input_jsonl,
            args.dataset,
            no_local_fallback=args.no_local_fallback,
        )
        print(f"Loaded {len(dataset)} examples")

        print("Formatting assistant turns for Qwen3-4B-Instruct...")
        processed = dataset.map(process_example, remove_columns=dataset.column_names)

        print(
            "Splitting by seed: "
            f"{len(train_seeds)} train seeds, {len(val_seeds)} validation seeds"
        )
        dataset_dict, split_stats = split_by_seed(processed, train_seeds, val_seeds)
        split_stats.update({"mode": "seed"})

    split_stats.update({
        "train_seeds": sorted(train_seeds),
        "validation_seeds": sorted(val_seeds),
    })

    print(f"Train: {len(dataset_dict['train'])} examples")
    print(f"Validation: {len(dataset_dict['validation'])} examples")
    if len(dataset_dict["train"]) == 0 or len(dataset_dict["validation"]) == 0:
        raise ValueError(
            "Seed split produced an empty train or validation set. Check that "
            "the raw SFT JSONL was generated from the committed train/val seeds."
        )

    # Save as JSONL (load_dataset auto-detects split names from filenames)
    for split_name, split_data in dataset_dict.items():
        jsonl_path = output_dir / f"{split_name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for example in split_data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"Saved {jsonl_path} ({len(split_data)} examples)")

    stats_path = output_dir / "preprocess_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "assistant_format": "instruct",
                "template_tokenizer": args.template_tokenizer,
                "output_dir": str(output_dir),
                **split_stats,
            },
            f,
            indent=2,
        )
    print(f"Saved {stats_path}")

    # Print a sample to verify
    print("\n--- Sample processed example ---")
    sample = dataset_dict["train"][0]
    sample_messages = sample.get("messages") or (sample["prompt"] + sample["completion"])
    for msg in sample_messages:
        role = msg["role"]
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        print(f"[{role}]: {content}")
    print_chat_template_preview(sample, args.template_tokenizer)


if __name__ == "__main__":
    main()
