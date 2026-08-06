#!/usr/bin/env python3
"""Verification eval: confirm an uploaded HF model plays MegaGem end-to-end.

Serves the merged HF model (loaded from HF, via vLLM under SERVED_NAME) and
plays it vs 2x an API opponent across a FULLY-CROSSED design: each of N decks
(seeds) is played with the policy in EACH of the 3 seats => 3N games. No
selector, no adapter — the plain uploaded weights. Reports overall + per-seat
win-rate and margin, plus error/parse-failure counts, so a broken upload shows
up as errors or degenerate scores rather than a silent pass.

Run INSIDE the Modal container (modal_release.py::verify_hf_model). Resumable:
existing parseable game JSONs are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from megagem.evals.game_runner import register_vllm_endpoint
from megagem.rollout import run_game


def outcome_from(g: dict, seat: int) -> dict | None:
    """Score the policy seat from either record shape (written game_data file or
    run_game's returned env state)."""
    try:
        fr = g.get("final_results") or g
        fs = {int(p["player_id"]): float(p["final_score"]) for p in fr["final_scores"]}
        return {"win": bool(int(fr["winner_id"]) == seat),
                "policy_delta": fs[seat] - max(v for k, v in fs.items() if k != seat),
                "policy_score": fs[seat],
                "scores": {str(k): round(v, 1) for k, v in fs.items()}}
    except Exception:  # noqa: BLE001
        return None


def game_outcome(path: Path, seat: int) -> dict | None:
    try:
        return outcome_from(json.loads(path.read_text()), seat)
    except Exception:  # noqa: BLE001
        return None


async def play_one(policy: str, opponent: str, seed: int, seat: int,
                   games_dir: Path, sem: asyncio.Semaphore,
                   pikl_base: dict | None = None) -> dict:
    fname = f"g_seed{seed}_seat{seat}.json"
    fpath = games_dir / fname
    if fpath.exists():
        prior = game_outcome(fpath, seat)
        if prior is not None:
            return {"seed": seed, "seat": seat, **prior, "resumed": True}
    models = [opponent] * 3
    models[seat] = policy
    # When the EV selector is on, it must act at the POLICY's seat (which rotates).
    pikl = {**pikl_base, "seat": seat} if pikl_base is not None else None
    async with sem:
        try:
            state = await run_game(
                models=models, seed=seed, num_players=3, silent=True,
                output_file=str(games_dir / (fname[:-5] + ".txt")),
                results_dir=games_dir, json_filename=fname, pikl_config=pikl)
        except Exception as e:  # noqa: BLE001
            return {"seed": seed, "seat": seat, "error": str(e)[:300]}
    out = game_outcome(fpath, seat) or outcome_from(state, seat)
    if out is None:
        return {"seed": seed, "seat": seat, "error": "unparseable game"}
    print(f"[seed={seed} seat={seat}] win={out['win']} "
          f"delta={out['policy_delta']:+.1f} score={out['policy_score']:.1f}",
          flush=True)
    return {"seed": seed, "seat": seat, **out}


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    summary: dict = {"n_games": len(rows), "n_ok": len(ok), "n_error": len(errs),
                     "errors": errs[:10]}
    if ok:
        n = len(ok)
        summary["win_rate"] = round(sum(r["win"] for r in ok) / n, 4)
        summary["mean_margin"] = round(sum(r["policy_delta"] for r in ok) / n, 3)
        summary["mean_policy_score"] = round(sum(r["policy_score"] for r in ok) / n, 2)
        by_seat = {}
        for k in (0, 1, 2):
            ks = [r for r in ok if r["seat"] == k]
            if ks:
                by_seat[k] = {
                    "n": len(ks),
                    "win_rate": round(sum(r["win"] for r in ks) / len(ks), 3),
                    "mean_margin": round(sum(r["policy_delta"] for r in ks) / len(ks), 2)}
        summary["by_seat"] = by_seat
    return summary


async def amain(a) -> int:
    register_vllm_endpoint(a.policy_name, a.served_name, a.vllm_url, "EMPTY")
    out_dir = Path(a.out_dir)
    games_dir = out_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(a.seed_start, a.seed_start + a.num_seeds))
    sem = asyncio.Semaphore(a.max_parallel)

    pikl_base = None
    if a.selector:
        pikl_base = {"enabled": True, "target": "ev_dist",
                     "ev_model_path": a.ev_model_path,
                     "ev_value_head_path": a.ev_value_head_path,
                     "ev_gate_min": a.gate_min_ev, "ev_pacing_lam": a.pacing_lam,
                     "ev_vhat_debias": a.vhat_debias, "max_parallel": 4}
        print(f"[verify] EV selector ON: {json.dumps(pikl_base)}", flush=True)

    # Fail-fast: one game alone first, so a wrong opponent string / broken serve
    # aborts before burning the full fan-out.
    first = await play_one(a.policy_name, a.opponent, seeds[0], 0, games_dir, sem,
                           pikl_base=pikl_base)
    if "error" in first:
        (out_dir / "verify_summary.json").write_text(
            json.dumps({"aborted": "smoke game failed", "first": first}, indent=2))
        print(f"[verify] SMOKE FAILED: {first['error']}", flush=True)
        return 2

    tasks = [play_one(a.policy_name, a.opponent, seed, seat, games_dir, sem,
                      pikl_base=pikl_base)
             for seed in seeds for seat in (0, 1, 2)
             if not (seed == seeds[0] and seat == 0)]
    rows = [first] + list(await asyncio.gather(*tasks))
    summary = summarize(rows)
    (out_dir / "verify_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=1))
    print(f"[verify] DONE: {json.dumps(summary)}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm-url", required=True)
    ap.add_argument("--policy-name", default="distill-para")
    ap.add_argument("--served-name", default="qwen/qwen3-4b-instruct")
    ap.add_argument("--opponent", default="anthropic/claude-sonnet-4.6")
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=61000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-parallel", type=int, default=16)
    # Optional EV selector (the deployable tier = weights + analytic selector).
    ap.add_argument("--selector", action="store_true")
    ap.add_argument("--ev-model-path", default="")
    ap.add_argument("--ev-value-head-path", default="")
    ap.add_argument("--gate-min-ev", type=float, default=1.0)
    ap.add_argument("--pacing-lam", type=float, default=0.5)
    ap.add_argument("--vhat-debias", type=float, default=2.0)
    a = ap.parse_args(argv)
    return asyncio.run(amain(a))


if __name__ == "__main__":
    raise SystemExit(main())
