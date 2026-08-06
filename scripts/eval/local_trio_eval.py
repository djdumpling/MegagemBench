#!/usr/bin/env python3
"""Head-to-head eval of the three LOCAL 4B models — base vs SFT vs distilled —
all at the same table, $0 API. Fully-crossed: N seeds × 3 cyclic seat rotations
(each model occupies each seat once per deck) = 3N games. Every PAIR is compared
in all 3N games, so "is distilled > SFT?" is an n=3N paired contrast — far
sharper than marginal ELO CIs.

Includes a MAPPING/FINGERPRINT check first: probes each served endpoint with a
fixed prompt and confirms the three are distinct models (so SFT and distilled
can't be silently swapped or the distill adapter dropped).

Run INSIDE the Modal container (modal_eval.py::local_trio_eval), which serves the
three models and passes their URLs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from itertools import combinations
from pathlib import Path

from megagem.evals.game_runner import register_vllm_endpoint
from megagem.rollout import run_game


def fingerprint(model_id: str, url: str) -> dict:
    """Greedy probe so identical weights => identical text. Used to confirm the
    three endpoints are distinct models (mapping not swapped, adapter applied)."""
    import hashlib

    from openai import OpenAI
    client = OpenAI(api_key="EMPTY", base_url=url)
    prompt = ("You are bidding in a sealed-bid auction for a single Blue gem. "
              "You have 30 coins and need Blue for a mission. In one sentence, "
              "state your bid and why.")
    try:
        r = client.chat.completions.create(
            model=model_id, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_completion_tokens=120)
        txt = (r.choices[0].message.content or "").strip()
        return {"model": model_id, "ok": True, "len": len(txt),
                "sha8": hashlib.sha256(txt.encode()).hexdigest()[:8],
                "head": txt[:160]}
    except Exception as e:  # noqa: BLE001
        return {"model": model_id, "ok": False, "error": str(e)[:200]}


def scores_from(g: dict) -> dict[int, float] | None:
    try:
        fr = g.get("final_results") or g
        return {int(p["player_id"]): float(p["final_score"]) for p in fr["final_scores"]}
    except Exception:  # noqa: BLE001
        return None


async def play(models: list[str], seed: int, rot: int, games_dir: Path,
               sem: asyncio.Semaphore) -> dict:
    fname = f"g_seed{seed}_rot{rot}.json"
    fpath = games_dir / fname
    if fpath.exists():
        sc = scores_from(json.loads(fpath.read_text()))
        if sc is not None:
            return {"seed": seed, "rot": rot, "models": models, "scores": sc, "resumed": True}
    async with sem:
        try:
            state = await run_game(models=models, seed=seed, num_players=3, silent=True,
                                   output_file=str(games_dir / (fname[:-5] + ".txt")),
                                   results_dir=games_dir, json_filename=fname)
        except Exception as e:  # noqa: BLE001
            return {"seed": seed, "rot": rot, "models": models, "error": str(e)[:200]}
    sc = None
    if fpath.exists():
        sc = scores_from(json.loads(fpath.read_text()))
    if sc is None and isinstance(state, dict):
        sc = scores_from(state)
    if sc is None:
        return {"seed": seed, "rot": rot, "models": models, "error": "unparseable"}
    print(f"[seed{seed} rot{rot}] " + "  ".join(
        f"{m.split('/')[-1][:18]}={sc.get(i,0):.0f}" for i, m in enumerate(models)), flush=True)
    return {"seed": seed, "rot": rot, "models": models, "scores": sc}


def trueskill_ratings(rows: list[dict], ids: list[str]) -> dict:
    try:
        import trueskill
    except ModuleNotFoundError:
        return {"note": "trueskill not installed in this env — pairwise records "
                "above are the primary read; compute TS locally if needed"}
    env = trueskill.TrueSkill(mu=25.0, sigma=25.0/3, beta=25.0/6, tau=25.0/300,
                              draw_probability=0.0)
    R = {m: env.create_rating() for m in ids}
    for r in rows:
        if "scores" not in r:
            continue
        order = sorted(range(len(r["models"])), key=lambda i: -r["scores"].get(i, 0))
        ranked_models = [r["models"][i] for i in order]
        groups = [[R[m]] for m in ranked_models]
        new = env.rate(groups)
        for i, m in enumerate(ranked_models):
            R[m] = new[i][0]
    return {m: {"mu_x40": round(R[m].mu*40, 1), "sigma_x40": round(R[m].sigma*40, 1),
                "score_x40": round((R[m].mu-3*R[m].sigma)*40, 1)} for m in ids}


def aggregate(rows: list[dict], ids: list[str]) -> dict:
    ok = [r for r in rows if "scores" in r]
    n_err = sum(1 for r in rows if "error" in r)
    per = {m: {"scores": [], "by_seat": {0: [], 1: [], 2: []}, "wins": 0} for m in ids}
    for r in ok:
        sc = r["scores"]
        win_i = max(range(3), key=lambda i: sc.get(i, 0))
        for i, m in enumerate(r["models"]):
            per[m]["scores"].append(sc.get(i, 0))
            per[m]["by_seat"][i].append(sc.get(i, 0))
            if i == win_i:
                per[m]["wins"] += 1

    def mean(xs):
        return round(sum(xs)/len(xs), 2) if xs else None

    summary = {"n_games": len(rows), "n_ok": len(ok), "n_error": n_err, "per_model": {}}
    for m in ids:
        s = per[m]["scores"]
        summary["per_model"][m] = {
            "games": len(s), "win_rate": round(per[m]["wins"]/len(s), 3) if s else None,
            "mean_score": mean(s),
            "mean_score_by_seat": {k: mean(v) for k, v in per[m]["by_seat"].items()}}

    # pairwise head-to-head: same-game score comparison (paired by deck+rotation)
    pair = {}
    for a, b in combinations(ids, 2):
        wins_a = wins_b = ties = 0
        diffs = []
        for r in ok:
            ia, ib = r["models"].index(a), r["models"].index(b)
            da = r["scores"].get(ia, 0); db = r["scores"].get(ib, 0)
            diffs.append(da-db)
            if da > db: wins_a += 1
            elif db > da: wins_b += 1
            else: ties += 1
        n = len(diffs); md = sum(diffs)/n if n else 0
        sd = (sum((x-md)**2 for x in diffs)/(n-1))**0.5 if n > 1 else 0
        se = sd/math.sqrt(n) if n else 0
        # paired sign test (exclude ties) on "a beats b"
        nd = wins_a + wins_b
        p = (min(1.0, 2*sum(math.comb(nd, i) for i in range(0, min(wins_a, wins_b)+1))*0.5**nd)
             if nd else 1.0)
        pair[f"{a.split('/')[-1]} vs {b.split('/')[-1]}"] = {
            "n": n, "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
            "a_winrate_vs_b": round(wins_a/n, 3) if n else None,
            "mean_score_diff_a_minus_b": round(md, 2),
            "95ci": [round(md-1.96*se, 2), round(md+1.96*se, 2)],
            "t": round(md/se, 2) if se > 0 else None, "sign_test_p": round(p, 4)}
    summary["pairwise_same_game"] = pair
    summary["trueskill"] = trueskill_ratings(ok, ids)
    return summary


async def amain(a) -> int:
    ids = [a.base_id, a.sft_id, a.dist_id]
    urls = {a.base_id: a.base_url, a.sft_id: a.sft_url, a.dist_id: a.dist_url}
    for m, u in urls.items():
        register_vllm_endpoint(m, m, u, "EMPTY")

    out_dir = Path(a.out_dir); games_dir = out_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)

    # --- mapping / fingerprint check (greedy; must be 3 distinct models) ---
    fps = [fingerprint(m, urls[m]) for m in ids]
    sha = {f["model"]: f.get("sha8") for f in fps}
    distinct = len({f.get("sha8") for f in fps if f.get("ok")}) == sum(1 for f in fps if f.get("ok"))
    sft_neq_dist = sha.get(a.sft_id) != sha.get(a.dist_id)
    print(f"[fingerprint] distinct={distinct}  SFT!=distilled={sft_neq_dist}", flush=True)
    for f in fps:
        print(f"  {f['model'].split('/')[-1]:40} sha8={f.get('sha8')} :: {f.get('head','')[:90]!r}", flush=True)

    # --- 3N games: N seeds x 3 cyclic rotations ---
    seeds = list(range(a.seed_start, a.seed_start + a.num_seeds))
    rotations = [[ids[0], ids[1], ids[2]], [ids[2], ids[0], ids[1]], [ids[1], ids[2], ids[0]]]
    sem = asyncio.Semaphore(a.max_parallel)
    tasks = [play(rotations[rot], seed, rot, games_dir, sem)
             for seed in seeds for rot in range(3)]
    rows = list(await asyncio.gather(*tasks))

    summary = aggregate(rows, ids)
    summary["mapping_check"] = {"distinct_models": distinct, "sft_neq_distilled": sft_neq_dist,
                                "fingerprints": fps}
    summary["ids"] = {"base": a.base_id, "sft": a.sft_id, "distilled": a.dist_id}
    (out_dir / "trio_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=1))
    print(f"[trio] DONE n_ok={summary['n_ok']}/{summary['n_games']}", flush=True)
    print(f"[trio] pairwise: {json.dumps(summary['pairwise_same_game'])}", flush=True)
    print(f"[trio] trueskill: {json.dumps(summary['trueskill'])}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True); ap.add_argument("--sft-url", required=True)
    ap.add_argument("--dist-url", required=True)
    ap.add_argument("--base-id", required=True); ap.add_argument("--sft-id", required=True)
    ap.add_argument("--dist-id", required=True)
    ap.add_argument("--num-seeds", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=100000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-parallel", type=int, default=16)
    return asyncio.run(amain(ap.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
