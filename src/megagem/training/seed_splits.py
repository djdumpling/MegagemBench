"""The locked SFT seed splits, and helpers for resolving seed selections.

The splits are contiguous ranges, so they are defined here in code rather than
as data files. Treat them as frozen: they partition which games trained the
published model, which selected its checkpoint, and which were never looked at.
Changing a boundary silently changes what every published number means.
"""

from __future__ import annotations

from pathlib import Path

# Locked 2026-05-12. Disjoint by construction; `ensure_disjoint` re-checks.
SPLIT_RANGES: dict[str, range] = {
    "train": range(1001, 1151),       # 150 seeds — SFT training examples
    "val": range(1151, 1161),         # 10 seeds — checkpoint selection / early stop
    "validation": range(1151, 1161),  # alias
    "test": range(1161, 1171),        # 10 seeds — final reporting ONLY
}


def load_seed_file(path: Path) -> list[int]:
    """Load newline-delimited integer seeds, allowing comments and blanks."""
    seeds: list[int] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        try:
            seeds.append(int(text))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: expected integer seed, got {text!r}") from exc
    return seeds


def load_split_seeds(split: str) -> list[int]:
    """Seeds of one named split (``train`` / ``val`` / ``test``)."""
    key = split.lower()
    if key not in SPLIT_RANGES:
        valid = ", ".join(sorted(SPLIT_RANGES))
        raise ValueError(f"Unknown seed split {split!r}. Expected one of: {valid}")
    return list(SPLIT_RANGES[key])


def ensure_disjoint(split_to_seeds: dict[str, list[int]]) -> None:
    """Raise if any seed appears in more than one split."""
    seen: dict[int, str] = {}
    for split, seeds in split_to_seeds.items():
        for seed in seeds:
            if seed in seen:
                raise ValueError(
                    f"Seed {seed} appears in both {seen[seed]!r} and {split!r}"
                )
            seen[seed] = split


def parse_seed_values(values: list[str]) -> list[int]:
    """Parse CLI seed values like ``1 2 3`` or ``1,2,3``."""
    seeds: list[int] = []
    for value in values:
        for part in value.split(","):
            text = part.strip()
            if text:
                seeds.append(int(text))
    return seeds


def resolve_seeds(spec: str) -> list[int]:
    """Resolve a seed selection written as any of:

    * a split name — ``val``
    * an inclusive range — ``1151-1160``
    * an explicit list — ``1151,1152`` or ``"1151 1152"``
    * a path to a newline-delimited seed file (kept so callers holding a
      generated file, e.g. an ad-hoc eval seed set, keep working)

    Raises ValueError if the spec matches none of these.
    """
    text = str(spec).strip()
    if not text:
        raise ValueError("empty seed spec")

    key = text.lower()
    if key in SPLIT_RANGES:
        return load_split_seeds(key)

    if "-" in text and "," not in text:
        lo, _, hi = text.partition("-")
        if lo.strip().isdigit() and hi.strip().isdigit():
            start, end = int(lo), int(hi)
            if end < start:
                raise ValueError(f"seed range {text!r} ends before it starts")
            return list(range(start, end + 1))

    path = Path(text)
    if path.is_file():
        return load_seed_file(path)

    try:
        return parse_seed_values([text])
    except ValueError as exc:
        valid = ", ".join(sorted(SPLIT_RANGES))
        raise ValueError(
            f"Could not resolve seeds from {spec!r}: expected a split name "
            f"({valid}), a range like 1151-1160, a comma-separated list, or a "
            f"path to a seed file."
        ) from exc
