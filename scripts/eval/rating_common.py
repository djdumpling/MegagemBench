#!/usr/bin/env python3
"""Shared loader + cluster keys for the rating-analysis scripts
(transitivity_diagnostic.py, plackett_luce_eval.py).

Reads MegaGem game JSONs (the format written by run_game / run_benchmark):
  metadata.models      -> list of model_id strings, indexed by seat (player_id)
  metadata.seed        -> int; the seed FULLY determines the deck/deal (verified:
                          all games at a given seed share an identical deal), so
                          seed == deck identity. The BIBD uses only 3 seeds = 3
                          decks, i.e. the deck is a controlled fixed factor with
                          few levels, NOT a variance source sampled over.
  final_results.final_scores -> [{player_id, final_score, ...}]  (ties = equal score)

System python3 (numpy/scipy live there, not the numpy-only .venv).
"""
from __future__ import annotations

import glob
import json
from collections import namedtuple
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Retired dry-run corpus (regenerable scratch; not shipped with the package).
DEFAULT_SOURCES = [str(REPO_ROOT / "src" / "megagem" / "evals" / "results" / "former")]

# models/scores are tuples indexed by seat; seed is the deck id.
Game = namedtuple("Game", ["path", "seed", "models", "scores"])


def model_labels():
    """(id->number, id->display) from the repo registry; empty dicts if unavailable."""
    try:
        from megagem.evals.model_mapping import MODELS, NUM_MODELS
        id2num = {MODELS[n][0]: n for n in range(1, NUM_MODELS + 1)}
        id2disp = {MODELS[n][0]: MODELS[n][1] for n in range(1, NUM_MODELS + 1)}
        return id2num, id2disp
    except Exception:  # noqa: BLE001
        return {}, {}


def short(model_id: str, id2disp: dict | None = None) -> str:
    if id2disp and model_id in id2disp:
        return id2disp[model_id]
    return model_id.split("/")[-1]


def iter_paths(sources):
    paths = []
    for s in sources:
        p = Path(s)
        if p.is_dir():
            paths += [str(x) for x in p.glob("*.json")
                      if x.name.startswith(("megagem_", "benchmark_", "top3_"))]
        elif any(ch in s for ch in "*?["):
            paths += glob.glob(s)
        elif p.is_file():
            paths.append(str(p))
    return sorted(set(paths))


def resolve_restrict(restrict):
    """restrict may be a list of model_ids, registry numbers (as str/int), or
    substrings. Returns a set of full model_ids (best effort) or None."""
    if not restrict:
        return None
    id2num, _ = model_labels()
    num2id = {n: mid for mid, n in id2num.items()}
    out = set()
    for r in restrict:
        rs = str(r).strip()
        if rs.isdigit() and int(rs) in num2id:
            out.add(num2id[int(rs)])
        elif rs in id2num:
            out.add(rs)
        else:  # substring match against known ids
            hit = [mid for mid in id2num if rs in mid]
            out.update(hit if hit else {rs})
    return out or None


def load_games(sources, restrict=None):
    """Return (games, n_bad). If `restrict` (set of model_ids) is given, keep only
    games whose ALL seats are in the set (clean within-pool rankings)."""
    rset = restrict if isinstance(restrict, (set, frozenset)) or restrict is None \
        else resolve_restrict(restrict)
    games, bad = [], 0
    for path in iter_paths(sources):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            bad += 1
            continue
        meta = d.get("metadata") or {}
        models = meta.get("models")
        if not (isinstance(models, list) and len(models) >= 3):
            bad += 1
            continue
        fr = d.get("final_results") or {}
        fs = fr.get("final_scores") or []
        if len(fs) != len(models):
            bad += 1
            continue
        sc = [None] * len(models)
        ok = True
        for e in fs:
            pid, v = e.get("player_id"), e.get("final_score")
            if pid is None or v is None or pid >= len(models):
                ok = False
                break
            sc[pid] = float(v)
        if not ok or any(x is None for x in sc):
            bad += 1
            continue
        if rset is not None and not all(m in rset for m in models):
            continue
        games.append(Game(path=path, seed=meta.get("seed"),
                          models=tuple(models), scores=tuple(sc)))
    return games, bad


def cluster_key(g: Game, mode: str = "triplet_seed"):
    """The bootstrap resampling unit.
      triplet_seed : (this matchup, this deck) — the independent experimental unit
                     in the BIBD (the 6 seat-perms of one triplet+seed are near-
                     replicates and stay together). ~78 clusters in the 468 BIBD.
      seed         : the deck only (very few — degenerate for bootstrap).
      triplet      : the matchup, pooling decks.
      game         : every game independent (ignores CRN — undercovers).
    """
    if mode == "seed":
        return ("seed", g.seed)
    if mode == "triplet":
        return ("trip", frozenset(g.models))
    if mode == "game":
        return ("game", g.path)
    return ("ts", frozenset(g.models), g.seed)


def ordering(g: Game):
    """Seats sorted best->worst by final_score; returns (list_of_model_ids,
    list_of_scores) in that order. Stable, so ties keep seat order; tie handling
    is the caller's job (see plackett_luce_eval tie expansion)."""
    idx = sorted(range(len(g.models)), key=lambda i: -g.scores[i])
    return [g.models[i] for i in idx], [g.scores[i] for i in idx]
