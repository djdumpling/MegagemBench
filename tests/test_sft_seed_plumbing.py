from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from megagem.evals.prepare_sft_data import is_invalid_for_sft
from megagem.training.preprocess_sft import process_example, split_by_seed
from megagem.training.preprocess_sft import load_presplit_dataset
from megagem.training.seed_splits import (
    ensure_disjoint,
    load_seed_file,
    load_split_seeds,
    parse_seed_values,
    resolve_seeds,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _example(seed: int, assistant: str = 'why\n\n{"bid": 1}') -> dict:
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {"seed": seed, "game_file": f"megagem_3_4_6_seed_{seed}.json"},
    }


def test_locked_sft_seed_splits_are_disjoint():
    splits = {name: load_split_seeds(name)
              for name in ("train", "val", "test")}
    ensure_disjoint(splits)
    assert {name: len(seeds) for name, seeds in splits.items()} == {
        "train": 150,
        "val": 10,
        "test": 10,
    }


def test_locked_sft_seed_splits_are_the_documented_ranges():
    """Pin the exact values (configs/sft/README.md): silently shifting a
    boundary would change what "train" and "held-out test" mean for every
    published number."""
    assert load_split_seeds("train") == list(range(1001, 1151))
    assert load_split_seeds("val") == list(range(1151, 1161))
    assert load_split_seeds("test") == list(range(1161, 1171))
    assert load_split_seeds("validation") == load_split_seeds("val")


def test_resolve_seeds_accepts_names_ranges_lists_and_files(tmp_path):
    """The seed selection crosses into bash (SEEDS=... in the eval drivers), so
    every spec form the drivers can pass must resolve."""
    assert resolve_seeds("val") == list(range(1151, 1161))
    assert resolve_seeds("VAL") == list(range(1151, 1161))
    assert resolve_seeds("30000-30002") == [30000, 30001, 30002]
    assert resolve_seeds("1151,1152") == [1151, 1152]
    assert resolve_seeds("7") == [7]

    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("# comment\n41\n\n42\n")
    assert resolve_seeds(str(seed_file)) == [41, 42]
    assert load_seed_file(seed_file) == [41, 42]

    for bad in ("", "nope", "30002-30000"):
        try:
            resolve_seeds(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_parse_seed_values_accepts_spaces_and_commas():
    assert parse_seed_values(["1001,1002", "1003"]) == [1001, 1002, 1003]


def test_instruct_format_unwraps_think_tags():
    row = process_example(_example(1001, '<think>short reason</think>\n\n{"bid": 3}'))
    assert [msg["role"] for msg in row["prompt"]] == ["system", "user"]
    assert row["completion"] == [
        {"role": "assistant", "content": 'short reason\n\n{"bid": 3}'}
    ]


def test_split_by_seed_uses_metadata_seed():
    rows = Dataset.from_list([
        process_example(_example(1001)),
        process_example(_example(1051)),
        process_example(_example(1061)),
    ])
    dataset, stats = split_by_seed(rows, train_seeds={1001}, val_seeds={1051})
    assert len(dataset["train"]) == 1
    assert len(dataset["validation"]) == 1
    assert set(dataset["train"].column_names) == {"prompt", "completion", "metadata"}
    assert stats["skipped_other_seed"] == 1


def test_presplit_dataset_uses_committed_seed_sets(tmp_path):
    train_path = tmp_path / "sft_train.jsonl"
    val_path = tmp_path / "sft_val.jsonl"
    train_path.write_text(json.dumps(_example(1001)) + "\n", encoding="utf-8")
    val_path.write_text(json.dumps(_example(1151)) + "\n", encoding="utf-8")

    dataset, stats = load_presplit_dataset(
        train_path,
        val_path,
        train_seeds={1001},
        val_seeds={1151},
    )

    assert len(dataset["train"]) == 1
    assert len(dataset["validation"]) == 1
    assert set(dataset["train"].column_names) == {"prompt", "completion", "metadata"}
    assert stats["mode"] == "files"


def test_sft_valid_filter_rejects_prose_fallback():
    assert is_invalid_for_sft({
        "parse_valid": True,
        "legal_valid": True,
        "default_used": False,
        "parse_method": "prose_fallback",
    })
