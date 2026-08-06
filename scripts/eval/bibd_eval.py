#!/usr/bin/env python3
"""BIBD frontier eval driver — 13 models, STS(13) x 8 seeds x 3 cyclic rotations
= 624 games. Runs INSIDE the Modal container (modal_eval.py::bibd_eval), which
serves the 3 local 4B models (base #11, SFT #12, distilled #13) on vLLM and
passes their URLs; the 10 API models auto-route to the PRIME gateway (run_game
falls back to PRIME for any model not registered as a local endpoint).

Design (src/megagem/evals/generate_bibd_schedule.py): STS(13) covers every one of the 78
model pairs exactly once; each triple is played at 8 distinct fresh decks (seeds
770001-770008, verified distinct + disjoint from all training/eval seeds), each
deck in 3 cyclic seat rotations (every model occupies each seat once per deck).

Per-game JSON is written to <out-dir> with run_benchmark's naming
(megagem_<n0>_<n1>_<n2>_seed_<S>.json, seat order preserved so rotations are
distinct files) -> RESUME-SAFE: an existing JSON is skipped. These are the files
the analysis scripts (transitivity_diagnostic.py, plackett_luce_eval.py) read.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from megagem.evals.game_runner import register_vllm_endpoint
from megagem.rollout import run_game
from megagem.evals.model_mapping import get_model_for_number, get_number_for_model, BIBD_MODELS
from megagem.evals.generate_bibd_schedule import bibd_triplets

DEFAULT_SEEDS = list(range(770001, 770009))  # 8 fresh, verified-distinct decks
VALUE_CHART = "A"


def json_filename(models_seat_order: list[str], seed: int, prefix: str = "megagem") -> str:
    """<prefix>_<n0>_<n1>_<n2>_seed_<S>.json — seat order preserved (rotations are
    distinct files). prefix='megagem' for the BIBD; 'top3' for the focused run so
    those games never collide with / get mixed into the BIBD result set."""
    nums = [str(get_number_for_model(m)) for m in models_seat_order]
    return f"{prefix}_{'_'.join(nums)}_seed_{seed}.json"


def cyclic_rotations(triplet_nums):
    a, b, c = triplet_nums
    return [(a, b, c), (c, a, b), (b, c, a)]  # each model each seat once


def build_games(seeds, triple=None):
    """List of (seat_order_model_numbers, seed).
    triple=None  -> the STS(13) BIBD over all 26 triples (deterministically SHUFFLED
                    so a --max-games prefix is a uniform sample; resume is by filename).
    triple given -> a single fixed triple x seeds x 3 cyclic rotations (focused run;
                    homogeneous, so no shuffle)."""
    import random as _random
    if triple is not None:
        return [(rot, seed) for seed in seeds for rot in cyclic_rotations(tuple(triple))]
    games = []
    for trip in bibd_triplets():                # sorted model numbers
        for seed in seeds:
            for rot in cyclic_rotations(trip):
                games.append((rot, seed))
    _random.Random(20240613).shuffle(games)
    return games


def fingerprint(model_id, url):
    """Greedy probe to confirm the 3 local endpoints are distinct served models."""
    from openai import OpenAI
    client = OpenAI(api_key="EMPTY", base_url=url)
    prompt = ("You are bidding in a sealed-bid auction for a single Blue gem. You "
              "have 30 coins and need Blue for a mission. In one sentence, state your bid and why.")
    try:
        r = client.chat.completions.create(model=model_id, messages=[{"role": "user", "content": prompt}],
                                           temperature=0.0, max_completion_tokens=120)
        txt = (r.choices[0].message.content or "").strip()
        return {"model": model_id, "ok": True, "sha8": hashlib.sha256(txt.encode()).hexdigest()[:8],
                "head": txt[:120]}
    except Exception as e:  # noqa: BLE001
        return {"model": model_id, "ok": False, "error": str(e)[:200]}


def scores_from(g):
    try:
        fr = g.get("final_results") or g
        return {int(p["player_id"]): float(p["final_score"]) for p in fr["final_scores"]}
    except Exception:  # noqa: BLE001
        return None


async def play(seat_nums, seed, games_dir, logs_dir, sem, counters, prefix="megagem"):
    models = [get_model_for_number(n) for n in seat_nums]
    fname = json_filename(models, seed, prefix)
    fpath = games_dir / fname
    if fpath.exists():                          # resume: skip completed
        sc = scores_from(json.loads(fpath.read_text()))
        if sc is not None:
            counters["resumed"] += 1
            return {"seat_nums": seat_nums, "seed": seed, "scores": sc, "resumed": True}
    async with sem:
        try:
            await run_game(models=models, value_chart=VALUE_CHART, seed=seed, num_players=3,
                           silent=True, output_file=str(logs_dir / (fname[:-5] + ".txt")),
                           json_filename=fname, results_dir=games_dir)
        except Exception as e:  # noqa: BLE001
            counters["failed"] += 1
            print(f"[FAIL seeds{seed} {seat_nums}] {str(e)[:160]}", flush=True)
            return {"seat_nums": seat_nums, "seed": seed, "error": str(e)[:200]}
    sc = scores_from(json.loads(fpath.read_text())) if fpath.exists() else None
    if sc is None:
        counters["failed"] += 1
        return {"seat_nums": seat_nums, "seed": seed, "error": "no JSON / unparseable"}
    counters["ok"] += 1
    done = counters["ok"] + counters["resumed"]
    if done % 20 == 0 or done <= 5:
        print(f"[{done}/{counters['total']}] seeds{seed} "
              + " ".join(f"{get_model_for_number(n).split('/')[-1][:14]}={sc.get(i,0):.0f}"
                         for i, n in enumerate(seat_nums)), flush=True)
    return {"seat_nums": seat_nums, "seed": seed, "scores": sc}


async def amain(base_url, sft_url, dist_url, out_dir, seeds=None, max_parallel=32,
                commit_every=50, commit_cb=None, max_games=None, require_model=None,
                triple=None, fname_prefix="megagem"):
    """Run the BIBD (triple=None) or a focused single-triple round-robin (triple set).
    Only the LOCAL models whose url is non-None are served/registered (the BIBD serves
    base/SFT/distilled; the top-3 run serves only distilled, Geminis route to PRIME).
    commit_cb (modal Volume.commit) is called every commit_every games -> preemption-safe."""
    served = [(get_model_for_number(n), u) for n, u in
              [(11, base_url), (12, sft_url), (13, dist_url)] if u]
    for mid, url in served:
        register_vllm_endpoint(mid, mid, url, "EMPTY")

    # fingerprint each served local model (confirms it responds + they're distinct)
    fps = [fingerprint(m, u) for m, u in served]
    ok_shas = [f.get("sha8") for f in fps if f.get("ok")]
    distinct = len(set(ok_shas)) == len(ok_shas)
    print(f"[fingerprint] served locals={[m for m, _ in served]} all_ok="
          f"{all(f.get('ok') for f in fps)} distinct={distinct}", flush=True)
    for f in fps:
        print(f"  {f['model'].split('/')[-1]:42} sha8={f.get('sha8')} ok={f.get('ok')} :: {f.get('head','')[:70]!r}", flush=True)

    out_dir = Path(out_dir); games_dir = out_dir / "games"; logs_dir = out_dir / "logs"
    games_dir.mkdir(parents=True, exist_ok=True); logs_dir.mkdir(parents=True, exist_ok=True)

    seeds = seeds or DEFAULT_SEEDS
    games = build_games(seeds, triple=triple)
    if require_model:                   # focus a probe on one model's games (e.g. validate a swap)
        games = [(rot, sd) for rot, sd in games if require_model in rot]
    if max_games:                       # metered probe: run only the first N games
        games = games[:max_games]
    counters = {"ok": 0, "failed": 0, "resumed": 0, "total": len(games)}
    mode = f"fixed triple {tuple(triple)}" if triple else f"{len(bibd_triplets())} STS triples"
    print(f"[eval] {mode} x {len(seeds)} seeds x 3 rotations = {len(games)} games | "
          f"prefix={fname_prefix} | max_parallel={max_parallel}", flush=True)

    sem = asyncio.Semaphore(max_parallel)
    tasks = [asyncio.create_task(play(sn, sd, games_dir, logs_dir, sem, counters, fname_prefix))
             for sn, sd in games]
    rows = []
    for fut in asyncio.as_completed(tasks):
        rows.append(await fut)
        if commit_cb and len(rows) % commit_every == 0:
            try:
                commit_cb()
                print(f"[commit] {len(rows)}/{len(games)} persisted to volume", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[commit] failed (non-fatal): {str(e)[:120]}", flush=True)

    gpm = {}
    for r in rows:
        if "scores" in r:
            for n in r["seat_nums"]:
                gpm[n] = gpm.get(n, 0) + 1
    model_nums = list(dict.fromkeys(triple)) if triple else BIBD_MODELS
    summary = {"n_games": len(games), "ok": counters["ok"], "resumed": counters["resumed"],
               "failed": counters["failed"], "seeds": seeds, "triple": triple,
               "rotations": 3, "fname_prefix": fname_prefix,
               "games_per_model": {get_model_for_number(n): gpm.get(n, 0) for n in model_nums},
               "fingerprints": fps, "local_distinct": distinct, "games_dir": str(games_dir)}
    (out_dir / "bibd_manifest.json").write_text(json.dumps(summary, indent=2))
    if commit_cb:
        try:
            commit_cb()
        except Exception:  # noqa: BLE001
            pass
    print(f"[bibd] DONE ok={counters['ok']} resumed={counters['resumed']} "
          f"failed={counters['failed']} of {len(games)}", flush=True)
    print(f"[bibd] games_per_model: {json.dumps(summary['games_per_model'])}", flush=True)
    return 0 if counters["failed"] == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="")
    ap.add_argument("--sft-url", default="")
    ap.add_argument("--dist-url", default="")
    ap.add_argument("--seeds", default="", help="comma seeds; default 770001..770008")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-parallel", type=int, default=32)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--require-model", type=int, default=0)
    ap.add_argument("--triple", default="", help="comma model numbers for a fixed-triple focused run")
    ap.add_argument("--fname-prefix", default="megagem")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    triple = [int(x) for x in args.triple.split(",")] if args.triple else None
    return asyncio.run(amain(args.base_url or None, args.sft_url or None, args.dist_url or None,
                             args.out_dir, seeds=seeds, max_parallel=args.max_parallel,
                             max_games=(args.max_games or None),
                             require_model=(args.require_model or None),
                             triple=triple, fname_prefix=args.fname_prefix))


if __name__ == "__main__":
    raise SystemExit(main())
