"""Modal app: MegaGem evaluation — API panels, local trios, and the BIBD frontier.

ENTRYPOINTS:
  modal run modal_eval.py::panel_eval_main --adapter-path ...
  modal run modal_eval.py::check_api_models_main         # API pre-flight ping
  modal run modal_eval.py::serve_local_smoke_main        # 3-model co-serve smoke
  modal run modal_eval.py::local_trio_eval_main          # base vs SFT vs distilled
  modal run modal_eval.py::bibd_eval_main                # STS(13) frontier benchmark
  modal run modal_eval.py::top3_eval_main                # top-cluster extension games

Shares one `modal.App` with modal_train.py / modal_release.py via modal_common.
"""

from __future__ import annotations

import os
import pathlib

from modal_common import (
    GPU,
    HF_CACHE,
    LOCAL_VLLM_PATH,
    RESULTS_DIR,
    VLLM_CACHE_DIR,
    _hf_token_ok,
    _serve_one,
    app,
    hf_cache,
    hf_secret,
    prime_secret,
    results_vol,
    vllm_cache,
)

@app.function(
    gpu=GPU,
    timeout=86400,
    volumes={HF_CACHE: hf_cache, RESULTS_DIR: results_vol,
             VLLM_CACHE_DIR: vllm_cache},
    secrets=[hf_secret, prime_secret],   # base is private; opponents via Prime
)
def panel_eval(
    *,
    adapter_path: str = ("/results/phase3_grpo_evidence_pool_evidence_01"
                         "/adapters/step_200"),
    base_model: str = "djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2",
    panels: str = "vs_flash",            # "" / "all" ⇒ full 6-opponent panel
    num_seeds: int = 0,                  # 0 ⇒ 10 SFT val seeds; N ⇒ 3N held-out
    seed_start: int = 30000,             # held-out base — disjoint from all
                                         # SFT/RL/§3.6 seed ranges
    max_parallel: int = 32,              # API-bound; eval is resumable
    vllm_ready_timeout_s: int = 1200,
    run_tag: str = "modal",
) -> dict:
    """RL-checkpoint win-rate eval vs a fixed opponent panel.

    Merges `adapter_path` onto its SFT base, serves the merged model as
    `qwen/qwen3-4b-instruct` on vLLM, runs eval_qwen_baseline (N seeds × 3
    seat-rotations) — the methodology behind the documented SFT step-1200
    baseline (vs_flash = 30.0%, n=30; repl_07 SFT baseline eval).
    `panels=""`/"all" ⇒ full 6-opponent panel + self-play; `num_seeds>0` ⇒ N
    held-out seeds (3N games/panel) for tighter win-rate CI.
    """
    import pathlib
    import subprocess
    import sys

    # HF-token preflight (SFT base is private).
    hf_err = _hf_token_ok(base_model)
    if hf_err:
        return {"panel_eval_rc": 2, "aborted": hf_err}

    # PRIME_API_KEY preflight — opponents route through Prime Inference; a
    # missing key would sys.exit mid-eval after the GPU+merge spend.
    if not os.environ.get("PRIME_API_KEY", "").strip():
        return {"panel_eval_rc": 2, "aborted": (
            "PRIME_API_KEY not in container env — pi-secret missing.")}

    # Merge adapter onto SFT base once per adapter; reused after.
    merged_dir = (f"{RESULTS_DIR}/merged/"
                  + adapter_path.strip("/").replace("/", "__"))
    if pathlib.Path(merged_dir, "config.json").exists():
        print(f"[panel_eval] reusing merged model at {merged_dir}")
    else:
        print(f"[panel_eval] merging {adapter_path} onto {base_model} …")
        m = subprocess.run(
            [sys.executable, "scripts/training/merge_sft_adapter.py",
             "--base", base_model, "--adapter", adapter_path,
             "--output-dir", merged_dir],
            cwd="/repo", env={**os.environ},
        )
        if m.returncode != 0:
            return {"panel_eval_rc": 2, "aborted": (
                f"adapter merge FAILED (rc={m.returncode}) — see logs")}
        results_vol.commit()

    # Panel eval: vLLM on merged model × seeds × 3-seat-rotation × panels.
    out_dir = f"{RESULTS_DIR}/panel_eval_{run_tag}"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Seed set: 0 ⇒ 10 SFT val seeds (matches doc'd baseline); N>0 ⇒ N held-out
    # seeds from seed_start, disjoint from train/val/test/RL/§3.6 ranges.
    if num_seeds > 0:
        seeds_spec = f"{seed_start}-{seed_start + num_seeds - 1}"
    else:
        seeds_spec = "val"
    extra = ("" if panels.strip().lower() in ("", "all")
             else f"--panels {panels.strip()}")
    env = {
        **os.environ,
        "MODEL_PATH": merged_dir,
        "SERVED_NAME": "qwen/qwen3-4b-instruct",
        "SEEDS": seeds_spec,
        "OUTPUT_DIR": out_dir,
        "MAX_PARALLEL": str(max_parallel),
        "VLLM_READY_TIMEOUT_S": str(vllm_ready_timeout_s),
        "YES": "1",
        "EXTRA_ARGS": extra,
        "PY": sys.executable,            # eval_qwen_baseline.sh's [[ -x "$PY" ]]
    }
    rc = subprocess.run(
        ["bash", "scripts/eval/eval_qwen_baseline.sh"], cwd="/repo", env=env,
    ).returncode

    # Aggregate per-game JSONs → per-panel win rate (same aggregator as SFT).
    win_rates = []
    try:
        from _aggregate_qwen_eval import aggregate  # on PYTHONPATH in-image
        win_rates = aggregate(pathlib.Path(out_dir))
    except Exception as e:  # noqa: BLE001
        print(f"[panel_eval] aggregation failed: {type(e).__name__}: {e}")

    results_vol.commit()
    vllm_cache.commit()
    return {
        "panel_eval_rc": rc,
        "adapter_path": adapter_path,
        "base_model": base_model,
        "merged_dir": merged_dir,
        "results_dir": out_dir,
        "panels": panels,
        "win_rates": win_rates,
        "sft_baseline_vs_flash": 0.30,   # repl_07 SFT baseline eval
    }


@app.local_entrypoint()
def panel_eval_main(
    adapter_path: str = ("/results/phase3_grpo_evidence_pool_evidence_01"
                         "/adapters/step_200"),
    base_model: str = "djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2",
    panels: str = "vs_flash",
    num_seeds: int = 0,
    seed_start: int = 30000,
    max_parallel: int = 32,              # API-bound; eval is resumable
    vllm_ready_timeout_s: int = 1200,
    run_tag: str = "modal",
):
    res = panel_eval.remote(
        adapter_path=adapter_path, base_model=base_model, panels=panels,
        num_seeds=num_seeds, seed_start=seed_start,
        max_parallel=max_parallel, vllm_ready_timeout_s=vllm_ready_timeout_s,
        run_tag=run_tag,
    )
    if res.get("aborted"):
        print(f"\nABORTED: {res['aborted']}")
        return
    print(f"\npanel_eval rc={res['panel_eval_rc']}  panels={res['panels']}")
    print(f"  adapter: {res['adapter_path']}")
    print(f"  results: {res['results_dir']}  (modal volume get megagem-results)")
    rows = res.get("win_rates") or []
    if not rows:
        print("  (no aggregated win rates — check the eval logs)")
    base = res.get("sft_baseline_vs_flash")
    for r in rows:
        wr = r.get("win_rate")
        wr_s = f"{wr:.1%}" if wr is not None else "--"
        line = (f"  {r['panel']:<16} n={r['n_games']:<4} "
                f"win_rate={wr_s:>7}  qwen_score={r.get('mean_qwen_score')}  "
                f"delta={r.get('mean_qwen_delta')}")
        if r.get("n_skipped"):
            # Unscorable games are excluded from the rate above — never let that
            # pass unremarked on a number we report.
            line += f"   [!] {r['n_skipped']} game(s) skipped: {r.get('skips')}"
        if r["panel"] == "vs_flash" and base is not None:
            line += f"   [vs SFT step-1200 baseline: {base:.0%}]"
        print(line)


# ============================================================================ #
# BIBD overnight pre-flight: (d) verify every API model string resolves at the #
# gateway, and (a) confirm the three LOCAL 4B models serve concurrently + play.#
# Both are validation gates BEFORE the full 468/936-game run (which is gated on #
# the schedule decision + a cost go).                                          #
# ============================================================================ #
@app.function(timeout=900, secrets=[prime_secret])   # no GPU — pure API ping
def check_api_models(models: str = "") -> dict:
    """(d) Ping each BIBD API model (registry indices 1-10 by default) with a
    trivial 1-token completion via the PRIME gateway; report which resolve. A
    dead model string here = a triplet that would error all night, so this is
    the cheap insurance run before committing."""
    import os
    from openai import OpenAI

    key = os.environ.get("PRIME_API_KEY", "").strip()
    if not key:
        return {"rc": 2, "aborted": "PRIME_API_KEY missing (pi-secret)"}
    from megagem.evals.model_mapping import get_model_for_number
    model_list = ([m.strip() for m in models.split(",") if m.strip()]
                  or [get_model_for_number(n) for n in range(1, 11)])
    client = OpenAI(api_key=key, base_url="https://api.pinference.ai/api/v1")
    out = {}
    for m in model_list:
        try:
            # Mirror the game's param handling: max_completion_tokens (not
            # max_tokens) + temperature=1 — required by GPT-5-class reasoning
            # models, accepted by the rest (multi_agent_env.py:261).
            r = client.chat.completions.create(
                model=m, messages=[{"role": "user", "content": "Reply with exactly: ok"}],
                max_completion_tokens=16, temperature=1.0)
            txt = (r.choices[0].message.content or "").strip()
            out[m] = {"ok": True, "reply": txt[:40]}
            print(f"  OK    {m}  -> {txt[:30]!r}", flush=True)
        except Exception as e:  # noqa: BLE001
            out[m] = {"ok": False, "error": str(e)[:200]}
            print(f"  FAIL  {m}  -> {str(e)[:120]}", flush=True)
    n_ok = sum(1 for v in out.values() if v["ok"])
    return {"rc": 0 if n_ok == len(model_list) else 1,
            "n_ok": n_ok, "n_total": len(model_list), "results": out}


@app.local_entrypoint()
def check_api_models_main(models: str = ""):
    res = check_api_models.remote(models=models)
    if res.get("aborted"):
        print(f"\nABORTED: {res['aborted']}")
        return
    print(f"\ncheck_api_models: {res.get('n_ok')}/{res.get('n_total')} resolved")
    for m, v in (res.get("results") or {}).items():
        mark = "OK  " if v["ok"] else "FAIL"
        print(f"  [{mark}] {m}{'' if v['ok'] else '  -- ' + v.get('error','')[:120]}")
    if res.get("rc") != 0:
        print("\n  ⚠ at least one API model did not resolve — fix before the overnight run.")



@app.function(
    gpu=GPU, timeout=7200,
    volumes={HF_CACHE: hf_cache, RESULTS_DIR: results_vol, VLLM_CACHE_DIR: vllm_cache},
    secrets=[hf_secret],   # private SFT/distilled repos; no API — local trio only
)
def serve_local_smoke(seed: int = 62000, gpu_frac: float = 0.3,
                      vllm_ready_timeout_s: int = 1800) -> dict:
    """(a) Validate the 3-concurrent-vLLM serving plan: serve base + SFT +
    distilled on one GPU (ports 8000/1/2), then play ONE local 3-player game
    (base vs SFT vs distilled) to confirm the full run_game pipeline + measure
    response coverage. No API. This is the serving gate for the overnight run."""
    import asyncio
    import contextlib

    from megagem.evals.model_mapping import get_model_for_number
    base = get_model_for_number(11); sft = get_model_for_number(12); dist = get_model_for_number(13)
    err = _hf_token_ok(sft) or _hf_token_ok(dist)
    if err:
        return {"rc": 2, "aborted": err}
    out_dir = f"{RESULTS_DIR}/bibd_serve_smoke"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    procs = []
    try:
        specs = [(base, "base", 8000), (sft, "sft", 8001), (dist, "dist", 8002)]
        urls = {}
        for reg_id, name, port in specs:
            vllm_path = LOCAL_VLLM_PATH.get(reg_id, reg_id)  # real HF repo for vLLM
            p, url = _serve_one(vllm_path, reg_id, port, gpu_frac,   # served-name = reg id
                                f"{out_dir}/vllm_{name}.log", vllm_ready_timeout_s)
            procs.append(p); urls[reg_id] = url

        from megagem.evals.game_runner import register_vllm_endpoint
        from megagem.rollout import run_game
        for mid, url in urls.items():
            register_vllm_endpoint(mid, mid, url, "EMPTY")

        fname = "smoke_local.json"
        state = asyncio.run(run_game(
            models=[base, sft, dist], seed=seed, num_players=3, silent=True,
            output_file=f"{out_dir}/smoke_local.txt",
            results_dir=pathlib.Path(out_dir), json_filename=fname))
        import json as _json
        try:
            g = _json.loads(pathlib.Path(f"{out_dir}/{fname}").read_text())
        except Exception:  # noqa: BLE001
            g = state if isinstance(state, dict) else {}
        fr = g.get("final_results") or g
        scores = {int(p["player_id"]): float(p["final_score"])
                  for p in (fr.get("final_scores") or [])} if fr.get("final_scores") else {}
        # per-seat response coverage (did each local model actually generate?)
        cov = {0: 0, 1: 0, 2: 0}
        for r in (g.get("rounds") or []):
            for p in (r.get("players") or []):
                pid = p.get("player_id")
                if pid in cov and isinstance(p.get("reasoning"), str) and p["reasoning"].strip():
                    cov[pid] += 1
        results_vol.commit(); vllm_cache.commit()
        return {"rc": 0, "served": list(urls), "scores": scores,
                "response_counts_by_seat": cov,
                "seats": {"0": base, "1": sft, "2": dist}, "results_dir": out_dir}
    finally:
        for p in procs:
            with contextlib.suppress(Exception):
                p.terminate(); p.wait(timeout=30)


@app.local_entrypoint()
def serve_local_smoke_main(seed: int = 62000, gpu_frac: float = 0.3):
    import json
    res = serve_local_smoke.remote(seed=seed, gpu_frac=gpu_frac)
    if res.get("aborted"):
        print(f"\nABORTED: {res['aborted']}")
        return
    print(f"\nserve_local_smoke rc={res.get('rc')}")
    print(f"  served (3 local vLLM): {json.dumps(res.get('served'))}")
    print(f"  seats: {json.dumps(res.get('seats'))}")
    print(f"  final scores: {json.dumps(res.get('scores'))}")
    print(f"  response counts by seat (each local model generated?): "
          f"{json.dumps(res.get('response_counts_by_seat'))}")


@app.function(
    gpu=GPU, timeout=28800,   # 8h ceiling; 300 local games run in ~1-1.5h
    volumes={HF_CACHE: hf_cache, RESULTS_DIR: results_vol, VLLM_CACHE_DIR: vllm_cache},
    secrets=[hf_secret],   # 3 LOCAL models only — ZERO API spend
)
def local_trio_eval(num_seeds: int = 100, seed_start: int = 100000,
                    gpu_frac: float = 0.3, max_parallel: int = 32,
                    vllm_ready_timeout_s: int = 1800, run_tag: str = "trio") -> dict:
    """Head-to-head base vs SFT vs distilled, all local ($0 API): num_seeds × 3
    cyclic seat rotations. Serves the three 4B models on one GPU, runs a mapping/
    fingerprint check, then the games. Driver: scripts/eval/local_trio_eval.py.
    Resumable (existing game JSONs are skipped)."""
    import contextlib
    import json
    import subprocess
    import sys as _sys

    from megagem.evals.model_mapping import get_model_for_number
    base = get_model_for_number(11); sft = get_model_for_number(12); dist = get_model_for_number(13)
    err = _hf_token_ok(sft) or _hf_token_ok(dist)
    if err:
        return {"rc": 2, "aborted": err}
    out_dir = f"{RESULTS_DIR}/local_trio_{run_tag}"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    procs = []
    try:
        for reg_id, name, port in [(base, "base", 8000), (sft, "sft", 8001), (dist, "dist", 8002)]:
            vllm_path = LOCAL_VLLM_PATH.get(reg_id, reg_id)
            p, _url = _serve_one(vllm_path, reg_id, port, gpu_frac,
                                 f"{out_dir}/vllm_{name}.log", vllm_ready_timeout_s)
            procs.append(p)
        rc = subprocess.run(
            [_sys.executable, "scripts/eval/local_trio_eval.py",
             "--base-url", "http://localhost:8000/v1", "--sft-url", "http://localhost:8001/v1",
             "--dist-url", "http://localhost:8002/v1",
             "--base-id", base, "--sft-id", sft, "--dist-id", dist,
             "--num-seeds", str(num_seeds), "--seed-start", str(seed_start),
             "--out-dir", out_dir, "--max-parallel", str(max_parallel)],
            cwd="/repo", env={**os.environ}).returncode
        results_vol.commit(); vllm_cache.commit()
        summary = None
        with contextlib.suppress(Exception):
            summary = json.loads(pathlib.Path(f"{out_dir}/trio_summary.json").read_text())
        return {"rc": rc, "results_dir": out_dir, "summary": summary}
    finally:
        for p in procs:
            with contextlib.suppress(Exception):
                p.terminate(); p.wait(timeout=30)


@app.local_entrypoint()
def local_trio_eval_main(num_seeds: int = 100, seed_start: int = 100000,
                         max_parallel: int = 32, gpu_frac: float = 0.3,
                         run_tag: str = "trio"):
    import json
    res = local_trio_eval.remote(num_seeds=num_seeds, seed_start=seed_start,
                                 max_parallel=max_parallel, gpu_frac=gpu_frac, run_tag=run_tag)
    if res.get("aborted"):
        print(f"\nABORTED: {res['aborted']}")
        return
    s = res.get("summary") or {}
    print(f"\nlocal_trio_eval rc={res.get('rc')}  results={res.get('results_dir')}")
    print(f"  games: {s.get('n_ok')} ok / {s.get('n_error')} err (of {s.get('n_games')})")
    mc = s.get("mapping_check") or {}
    print(f"  MAPPING: distinct_models={mc.get('distinct_models')} "
          f"sft_neq_distilled={mc.get('sft_neq_distilled')}")
    print(f"  per-model: {json.dumps(s.get('per_model'))}")
    print(f"  PAIRWISE (same-game): {json.dumps(s.get('pairwise_same_game'))}")
    print(f"  trueskill (x40): {json.dumps(s.get('trueskill'))}")


@app.function(
    gpu=GPU, timeout=86400,   # 24h ceiling (Modal max); full 624 @ parallel 16 ~3-5h.
    # Belt-and-suspenders: commit-every persists progress + resume-by-filename, so even
    # a 24h overrun loses nothing — just relaunch to finish.
    volumes={HF_CACHE: hf_cache, RESULTS_DIR: results_vol, VLLM_CACHE_DIR: vllm_cache},
    secrets=[hf_secret, prime_secret],   # hf: private SFT/distilled; prime: API models
)
def bibd_eval(seeds: str = "", gpu_frac: float = 0.3, max_parallel: int = 32,
              commit_every: int = 50, max_games: int = 0, require_model: int = 0,
              vllm_ready_timeout_s: int = 1800, run_tag: str = "frontier") -> dict:
    """13-model frontier BIBD: STS(13) x 8 seeds x 3 rotations = 624 games. Serves
    the 3 local 4B models (base #11 / SFT #12 / distilled #13) on one GPU; the 10
    API models route to PRIME. Driver runs IN-PROCESS so the volume is committed
    every `commit_every` games (preemption-safe + resumable). Existing JSONs skip."""
    import asyncio
    import contextlib

    from megagem.evals.model_mapping import get_model_for_number
    base = get_model_for_number(11); sft = get_model_for_number(12); dist = get_model_for_number(13)
    err = _hf_token_ok(sft) or _hf_token_ok(dist)
    if err:
        return {"rc": 2, "aborted": err}
    if not os.environ.get("PRIME_API_KEY"):
        return {"rc": 2, "aborted": "PRIME_API_KEY not set (needed for the 10 API models)"}

    out_dir = f"{RESULTS_DIR}/bibd_{run_tag}"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    seed_list = [int(s) for s in seeds.split(",")] if seeds else None

    procs = []
    try:
        for reg_id, name, port in [(base, "base", 8000), (sft, "sft", 8001), (dist, "dist", 8002)]:
            vllm_path = LOCAL_VLLM_PATH.get(reg_id, reg_id)
            p, _url = _serve_one(vllm_path, reg_id, port, gpu_frac,
                                 f"{out_dir}/vllm_{name}.log", vllm_ready_timeout_s)
            procs.append(p)

        import bibd_eval as driver  # scripts/eval on PYTHONPATH in-image
        rc = asyncio.run(driver.amain(
            "http://localhost:8000/v1", "http://localhost:8001/v1", "http://localhost:8002/v1",
            out_dir, seeds=seed_list, max_parallel=max_parallel,
            commit_every=commit_every, commit_cb=lambda: results_vol.commit(),
            max_games=(max_games or None), require_model=(require_model or None)))

        results_vol.commit(); vllm_cache.commit()
        import json as _json
        manifest = None
        with contextlib.suppress(Exception):
            manifest = _json.loads(pathlib.Path(f"{out_dir}/bibd_manifest.json").read_text())
        return {"rc": rc, "results_dir": out_dir, "manifest": manifest}
    finally:
        for p in procs:
            with contextlib.suppress(Exception):
                p.terminate(); p.wait(timeout=30)


@app.local_entrypoint()
def bibd_eval_main(seeds: str = "", max_parallel: int = 32, gpu_frac: float = 0.3,
                   commit_every: int = 50, max_games: int = 0, require_model: int = 0,
                   run_tag: str = "frontier"):
    import json
    res = bibd_eval.remote(seeds=seeds, max_parallel=max_parallel, gpu_frac=gpu_frac,
                           commit_every=commit_every, max_games=max_games,
                           require_model=require_model, run_tag=run_tag)
    if res.get("aborted"):
        print(f"\nABORTED: {res['aborted']}")
        return
    m = res.get("manifest") or {}
    print(f"\nbibd_eval rc={res.get('rc')}  results={res.get('results_dir')}")
    print(f"  games: ok={m.get('ok')} resumed={m.get('resumed')} failed={m.get('failed')} of {m.get('n_games')}")
    print(f"  local models distinct={m.get('local_distinct')}")
    print(f"  games_per_model: {json.dumps(m.get('games_per_model'))}")

@app.function(
    gpu=GPU, timeout=86400,
    volumes={HF_CACHE: hf_cache, RESULTS_DIR: results_vol, VLLM_CACHE_DIR: vllm_cache},
    secrets=[hf_secret, prime_secret],   # hf: private distilled; prime: the two Geminis
)
def top3_eval(num_seeds: int = 150, seed_start: int = 880001, gpu_frac: float = 0.45,
              max_parallel: int = 16, commit_every: int = 25, vllm_ready_timeout_s: int = 1800,
              run_tag: str = "topcluster") -> dict:
    """Focused top-3 round-robin to resolve the BIBD's top-tier tie: the single triple
    {distilled(13), gemini-3.1-pro(3), gemini-3-flash(4)} x num_seeds FRESH decks x 3
    cyclic rotations. Serves ONLY distilled locally; the two Geminis route to PRIME.
    Output files are 'top3_*.json' in a separate dir so they never mix with the BIBD."""
    import asyncio
    import contextlib

    from megagem.evals.model_mapping import get_model_for_number
    dist = get_model_for_number(13)
    err = _hf_token_ok(dist)
    if err:
        return {"rc": 2, "aborted": err}
    if not os.environ.get("PRIME_API_KEY"):
        return {"rc": 2, "aborted": "PRIME_API_KEY not set (needed for the Geminis)"}

    out_dir = f"{RESULTS_DIR}/top3_{run_tag}"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    seed_list = list(range(seed_start, seed_start + num_seeds))

    procs = []
    try:
        vllm_path = LOCAL_VLLM_PATH.get(dist, dist)
        p, _url = _serve_one(vllm_path, dist, 8002, gpu_frac,
                             f"{out_dir}/vllm_dist.log", vllm_ready_timeout_s)
        procs.append(p)

        import bibd_eval as driver  # scripts/eval on PYTHONPATH in-image
        rc = asyncio.run(driver.amain(
            None, None, "http://localhost:8002/v1", out_dir, seeds=seed_list,
            max_parallel=max_parallel, commit_every=commit_every,
            commit_cb=lambda: results_vol.commit(), triple=[13, 3, 4], fname_prefix="top3"))

        results_vol.commit(); vllm_cache.commit()
        import json as _json
        manifest = None
        with contextlib.suppress(Exception):
            manifest = _json.loads(pathlib.Path(f"{out_dir}/bibd_manifest.json").read_text())
        return {"rc": rc, "results_dir": out_dir, "manifest": manifest}
    finally:
        for p in procs:
            with contextlib.suppress(Exception):
                p.terminate(); p.wait(timeout=30)


@app.local_entrypoint()
def top3_eval_main(num_seeds: int = 150, seed_start: int = 880001, max_parallel: int = 16,
                   gpu_frac: float = 0.45, commit_every: int = 25, run_tag: str = "topcluster"):
    import json
    res = top3_eval.remote(num_seeds=num_seeds, seed_start=seed_start, max_parallel=max_parallel,
                           gpu_frac=gpu_frac, commit_every=commit_every, run_tag=run_tag)
    if res.get("aborted"):
        print(f"\nABORTED: {res['aborted']}")
        return
    m = res.get("manifest") or {}
    print(f"\ntop3_eval rc={res.get('rc')}  results={res.get('results_dir')}")
    print(f"  games: ok={m.get('ok')} resumed={m.get('resumed')} failed={m.get('failed')} of {m.get('n_games')}")
    print(f"  games_per_model: {json.dumps(m.get('games_per_model'))}")
