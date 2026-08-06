#!/usr/bin/env python3
"""BIBD schedule for the 13-model frontier eval (2026-06-13 v3).

13 IS Steiner-friendly (13 = 1 mod 6), so we use a proper STEINER TRIPLE SYSTEM
STS(13): 26 triples over 13 points with EVERY one of the 78 pairs in EXACTLY ONE
triple (lambda=1, uniform replication r=6) — the cleanest possible balanced design.

Construction: cyclic STS(13) on Z_13 from two base blocks developed mod 13:
  {0,1,4} (differences 1,3,4) and {0,2,7} (differences 2,5,6) — their differences
partition {1..6} exactly once. Points 0..12 map to BIBD_MODELS (registry numbers).

Run shape (driver scripts/eval/bibd_eval.py): each triple is played at NUM_SEEDS
distinct decks, each deck in 3 cyclic seat rotations. 26 x 8 x 3 = 624 games.
"""
from __future__ import annotations

from itertools import combinations

from megagem.evals.model_mapping import BIBD_MODELS

V = len(BIBD_MODELS)               # 13
BASE_BLOCKS = [(0, 1, 4), (0, 2, 7)]
NUM_SEEDS = 8
ROTATIONS = 3


def bibd_triplets() -> list[tuple[int, int, int]]:
    """26 triples over the 13 BIBD model numbers (cyclic STS(13); all 78 pairs once)."""
    assert V == 13, f"STS(13) needs exactly 13 models, got {V}"
    triples, seen = [], set()
    for block in BASE_BLOCKS:
        for k in range(V):
            t = tuple(sorted(BIBD_MODELS[(p + k) % V] for p in block))
            if t not in seen:
                seen.add(t)
                triples.append(t)
    return triples


def verify(triples) -> dict:
    pts = sorted(BIBD_MODELS)
    pair_count = {pr: 0 for pr in combinations(pts, 2)}
    for t in triples:
        for a, b in combinations(sorted(t), 2):
            pair_count[(a, b)] += 1
    counts = sorted(set(pair_count.values()))
    rep = {p: sum(1 for t in triples if p in t) for p in pts}
    return {
        "n_triples": len(triples),
        "n_points": len(pts),
        "all_pairs_exactly_once": counts == [1],
        "lambda_distribution": {c: sum(1 for v in pair_count.values() if v == c) for c in counts},
        "replication_per_model": sorted(set(rep.values())),
        "uncovered_pairs": [pr for pr, c in pair_count.items() if c == 0],
    }


def main():
    triples = bibd_triplets()
    info = verify(triples)
    print(f"STS(13): {info['n_triples']} triples over {info['n_points']} models {BIBD_MODELS}")
    print(f"  all 78 pairs covered exactly once: {info['all_pairs_exactly_once']}")
    print(f"  lambda distribution: {info['lambda_distribution']}")
    print(f"  replication per model: {info['replication_per_model']}  (uniform r=6)")
    print(f"  uncovered pairs: {info['uncovered_pairs']}")
    print(f"  run: {info['n_triples']} x {NUM_SEEDS} seeds x {ROTATIONS} rotations "
          f"= {info['n_triples'] * NUM_SEEDS * ROTATIONS} games")
    for pr in [(11, 12), (11, 13), (12, 13)]:
        hit = [t for t in triples if pr[0] in t and pr[1] in t]
        print(f"  local pair {pr} co-tabled in: {hit}")
    print("\ntriples (model numbers):")
    for i, t in enumerate(triples, 1):
        print(f"  {i:2}. {t}")
    assert info["all_pairs_exactly_once"], "NOT a valid STS — aborting"


if __name__ == "__main__":
    main()
