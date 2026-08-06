#!/usr/bin/env python3
"""Phase-3 GRPO self-play driver for MegaGem (the RL plan §3.1-3.5).

Wraps the Phase-2 seam (`megagem.training.grpo_harness`) into a single
persistent `MegaGemGRPOTrainer.train()` over `max_steps=--steps`, with a
dynamic rollout_func that syncs the live LoRA adapter to vLLM each generation
and rolls fresh games against a lagged-self opponent pool. Reward defaults
come from `H.reward_config_from_env()`; health gates are reported, not fatal;
the spend/no-spend decision is `phase3_eval.py`'s. wandb is on iff
WANDB_API_KEY is set (Modal's wandb-secret).

Usage:

  # GPU box (prime-rl venv; never `uv run`):
  prime-rl/.venv/bin/python scripts/training/phase3_grpo.py \
      --model djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2 \
      --vllm-url http://localhost:8000/v1 \
      --heuristic-url http://localhost:8100/v1 \
      --steps 50 --k 8 --num-seeds 6 --output results/phase3_grpo/run.json

  # CPU dry-run (no GPU/net):
  .venv/bin/python scripts/training/phase3_grpo.py --dry-run --output /tmp/d.json
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import copy
import dataclasses
import glob
import json
import math
import os
import statistics
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import megagem.training.adapter_sync as ADP
import megagem.training.grpo_harness as H
from megagem.training.opponent_pool import (
    OpponentPool, OpponentSpec, Snapshot, anneal_probability,
)
from megagem.training.run_telemetry import (
    install_from_env, install_health_monitor_from_env, RolloutHealthMonitor,
)

from megagem_steps import _post_tag_actor_mask  # sibling script (reuse 2.3)


def _dataclass_asdict(obj):
    """asdict with a __dict__ fallback for non-dataclass-y fields."""
    if dataclasses.is_dataclass(obj):
        try:
            return dataclasses.asdict(obj)
        except Exception:  # noqa: BLE001
            return {f.name: getattr(obj, f.name)
                    for f in dataclasses.fields(obj)}
    return dict(getattr(obj, "__dict__", {}))

HEURISTIC_MODEL = "megagem/heuristic-v1"
_HEURISTIC_SPEC = OpponentSpec(
    kind="heuristic", served_name=HEURISTIC_MODEL, actor_id="heuristic")


def _current_self_spec() -> OpponentSpec:
    """Greedy opponent seat backed by the current live trainable adapter."""
    return OpponentSpec(
        kind="current_self",
        served_name=ADP.ADAPTER_NAME,
        actor_id="current_self",
    )


def _needs_heuristic_endpoint(args) -> bool:
    """True only for legacy no-pool runs or explicit heuristic league mixing."""
    return (not bool(getattr(args, "opponent_pool", True))) or (
        float(getattr(args, "p_heuristic", 0.0) or 0.0) > 0.0
    )


# --------------------------------------------------------------------------- #
# On-policy roll: trainable seat = current adapter via vLLM; others = shim.    #
# --------------------------------------------------------------------------- #
# §A.7: the trainable seat MUST sample (T=1.0, top-p 0.95) or the K rollouts
# at one seed vs the deterministic heuristic collapse to identical games →
# σ=0 K-groups → no GRPO signal. run_game external rollouts are NOT governed
# by GRPOConfig.temperature/top_p (that is TRL-internal generation, unused
# here); they are governed by the chat-completions params, which run_game now
# forwards via `caller_api_params`. The heuristic shim ignores them (it is
# deterministic by construction) so only the trainable seat is sampled.
A7_SAMPLING = {"temperature": 1.0, "top_p": 0.95}

# §A.7: pooled opponents (snapshots / API models) play GREEDY so they are
# deterministic within a K-group — only the trainable seat varies across the K.
# run_game applies these per seat; the heuristic shim ignores sampling params,
# snapshot LoRAs honour them.
OPPONENT_SAMPLING = {"temperature": 0.0}


def _roll_onpolicy(
    *, trainable_model, seeds, k, num_players, value_chart,
    trainable_seat, tmp_dir, sampling=None, max_parallel=32,
    dump_dir=None, roll_index=0, opponent_for_seat=None, opponents_for_table=None,
    games_out=None, telemetry=None,
) -> list[dict]:
    """Roll seeds×k REAL games on-policy, post-tag the actor mask (bid +
    nested reveal), flatten through the **unmodified** pipeline → §A.7
    K-grouped rows. The trainable seat is served the *live* adapter and is
    sampled per §A.7 (T=1.0); opponents play greedy (T=0).

    `opponent_for_seat` (optional): `Callable[[int seed], OpponentSpec]` — the
    §3.3 opponent pool's per-seed draw. ONE opponent per seed, held constant
    across that seed's K rollouts (§A.7 group integrity), playing greedy so it
    contributes zero within-group variance. `None` ⇒ the legacy single
    stationary heuristic for every opponent seat.

    `opponents_for_table` (optional, opt-in): `Callable[[int seed], dict[int
    table_seat, OpponentSpec]]` — the PSRO/league HETEROGENEOUS-table draw: each
    NON-trainable seat gets its OWN opponent, so the trainee can face two
    DIFFERENT opponents at one table (every prior run faced two identical clone
    opponents — the structural defect). Loss-safe: `_post_tag_actor_mask` masks
    by seat, not by opponent identity, so masking is correct regardless. The map
    is held constant across that seed's K rollouts (§A.7). When provided it
    OVERRIDES `opponent_for_seat`; `None` ⇒ the legacy homogeneous behaviour.

    `dump_dir` (optional): persist this roll's actor-tagged schema-v3 games
    into `<dump_dir>/roll_<roll_index>/` BEFORE flattening. Phase-3 rollouts
    are otherwise lost with the run's TemporaryDirectory; dumping them is the
    only way to run the reward diagnostic
    (`scripts/training/reward_score_correlation.py`) on the TRUE phase-3
    distribution instead of the phase-1 self-play proxy. One subdir per roll
    so each is a clean K-group set (seeds×k games) the diagnostic can consume
    directly. Dumping is a pure side effect — it never touches the rows
    returned to the trainer.

    Games within a roll run **concurrently** (semaphore-bounded). vLLM
    batches concurrent generations on the GPU; the heuristic shim is
    uvicorn-async. With `max_parallel=32` rollout wall time drops ~5–8× at
    no extra GPU cost. The §A.7 standardization is computed AFTER all games
    complete, so concurrency cannot affect group boundaries or advantage
    values — it only reorders generation, not the dataset. `max_parallel=1`
    restores the old sequential behaviour as a clean fallback.
    """
    from megagem.rollout import run_game  # lazy: needs endpoints/network

    # Per-seat rollout sampling: the trainable seat samples (§A.7) so the K
    # rollouts diverge; opponent seats play greedy so they are deterministic
    # within a K-group. run_game applies these per seat.
    trainable_sampling = sampling or A7_SAMPLING
    caller_api_params: list[dict] = [OPPONENT_SAMPLING] * num_players
    caller_api_params[trainable_seat] = trainable_sampling

    # Opponents are drawn ONCE per seed and held constant across that seed's K
    # games (§A.7 group integrity). Per-table-seat map {seed: {seat: spec}}:
    #   - opponents_for_table (opt-in, HETEROGENEOUS): each non-trainable seat
    #     gets its OWN draw → the trainee can face two DIFFERENT opponents at one
    #     table (the league/PSRO case). Loss-safe (masking is by seat).
    #   - else opponent_for_seat (HOMOGENEOUS, legacy): one opponent on every
    #     non-trainable seat — byte-identical to the prior behaviour.
    #   - else (both None) the legacy stationary heuristic.
    _nontrainable = [p for p in range(num_players) if p != trainable_seat]
    if opponents_for_table is not None:
        opp_by_seed_seat = {s: dict(opponents_for_table(s)) for s in seeds}
    else:
        opp_by_seed_seat = {
            s: {seat: (opponent_for_seat(s) if opponent_for_seat is not None
                       else _HEURISTIC_SPEC)
                for seat in _nontrainable}
            for s in seeds
        }

    def _models_for(s: int) -> list[str]:
        m = [trainable_model] * num_players
        for seat in _nontrainable:
            m[seat] = opp_by_seed_seat[s][seat].served_name
        return m

    def _opp_actor_id(s: int) -> str:
        # Cosmetic label for the masked opponent turns; "+"-joined across the
        # table's opponents — identical to the single id when homogeneous.
        ids = sorted({opp_by_seed_seat[s][seat].actor_id for seat in _nontrainable})
        return "+".join(ids)

    total_games = len(seeds) * k
    done_counter = [0]  # asyncio is single-threaded → no lock needed
    game_durations: list[float] = []  # per-game wall times for tail analysis
    roll_tag = f"roll {int(roll_index):03d}"
    t_roll_start = time.perf_counter()
    # Telemetry window: vLLM /metrics + nvidia-smi + AsyncOpenAI status counts
    # are sampled in background daemon threads (see megagem.training.run_telemetry); we just
    # bracket the rollout so the summary reflects only this roll's window.
    tele_handle = telemetry.window_open() if telemetry is not None else None

    async def _play_one(s: int, ki: int, sem: asyncio.Semaphore) -> dict:
        async with sem:
            fname = f"p3_seed{s}_r{ki}.json"
            t0 = time.perf_counter()
            # silent=True drops the per-round "Round N" line each run_game
            # emits under quiet=True — with max_parallel=32 those lines from
            # 32 concurrent games interleave into illegible noise. We replace
            # it with a single line per *game finish* below.
            await run_game(
                models=_models_for(s), value_chart=value_chart, seed=s,
                num_players=num_players, output_file="trajectory",
                json_filename=fname, quiet=True, silent=True,
                results_dir=Path(tmp_dir),
                caller_api_params=caller_api_params,
            )
            g = json.loads((Path(tmp_dir) / fname).read_text())
            done_counter[0] += 1
            n_done = done_counter[0]
            n_rounds = len(g.get("rounds") or [])
            dur = time.perf_counter() - t0
            game_durations.append(dur)
            print(f"  [{roll_tag}] game {n_done:>3}/{total_games} done  "
                  f"seed={s} k={ki}  rounds={n_rounds}  {dur:.1f}s",
                  flush=True)
            return g

    async def _play_all() -> list[dict]:
        sem = asyncio.Semaphore(max(1, int(max_parallel)))
        coros = [_play_one(s, ki, sem) for s in seeds for ki in range(k)]
        return await asyncio.gather(*coros)

    print(f"  [{roll_tag}] launching {total_games} games "
          f"(max_parallel={max_parallel})", flush=True)
    games_raw = asyncio.run(_play_all())
    roll_wall_s = time.perf_counter() - t_roll_start
    print(f"  [{roll_tag}] {total_games} games complete in "
          f"{roll_wall_s:.1f}s", flush=True)
    if telemetry is not None and tele_handle is not None:
        try:
            stats = telemetry.window_close(tele_handle)
            block = telemetry.format_summary(
                int(roll_index), roll_wall_s, game_durations, stats)
            if block:
                print(block, flush=True)
            headroom = telemetry.format_headroom(
                int(roll_index), int(total_games), roll_wall_s,
                int(max_parallel), stats)
            if headroom:
                print(headroom, flush=True)
        except Exception as e:  # noqa: BLE001 — telemetry never aborts a roll
            print(f"  [{roll_tag}] telemetry error: "
                  f"{type(e).__name__}: {e}", flush=True)
    pairs = [(s, ki) for s in seeds for ki in range(k)]  # _play_all order
    # Defensive §A.7 invariant: every K rollout of a seed faced that seed's
    # single drawn opponent. A regression to a per-rollout draw would mix
    # opponents within a K-group and corrupt the within-group advantage.
    for (s, _ki), g in zip(pairs, games_raw):
        got = (g.get("metadata") or {}).get("models")
        if got != _models_for(s):
            raise AssertionError(
                f"§A.7 violation: seed {s} faced models {got}, "
                f"expected {_models_for(s)}")
    games = [_post_tag_actor_mask(
        g, trainable_seat=trainable_seat,
        opponent_actor_id=_opp_actor_id(s))
        for (s, _ki), g in zip(pairs, games_raw)]
    if dump_dir is not None:
        roll_out = Path(dump_dir) / f"roll_{int(roll_index):03d}"
        roll_out.mkdir(parents=True, exist_ok=True)
        for (s, ki), g in zip(pairs, games):
            (roll_out / f"p3_seed{s}_r{ki}.json").write_text(
                json.dumps(g, default=str))
    if games_out is not None:
        games_out.extend(games)
    # one standardization pass over all K games at this seed-set ⇒ §A.7 groups
    return H.flatten_training_rows(games, trainable_seat=trainable_seat)


# --------------------------------------------------------------------------- #
# Health gates — reported, not fatal.                                         #
# --------------------------------------------------------------------------- #
def _health(series: dict, rows, args) -> dict:
    kls = series.get("kl", [])
    losses = series.get("loss", [])
    grads = series.get("grad_norm", [])
    advs = [float(r["precomputed_advantage"]) for r in rows] if rows else []
    clen = [len(r.get("completion") or "") for r in rows] if rows else []
    kl_max = max(kls) if kls else 0.0
    adv_var = (sum((a - (sum(advs) / len(advs))) ** 2 for a in advs)
               / len(advs)) if advs else 0.0
    gates = {
        "kl_bounded": (not kls) or (math.isfinite(kl_max)
                                    and kl_max <= args.kl_max),
        "nan_free": all(math.isfinite(x) for x in losses)
        and all(math.isfinite(x) for x in grads),
        "advantage_variance_nondegenerate": adv_var > 1e-9,
    }
    return {
        "gates": gates,
        "all_pass": all(gates.values()),
        "kl_max_observed": kl_max,
        "kl_max_allowed": args.kl_max,
        "advantage_variance": adv_var,
        "kl_trajectory": H.summarize_series(kls),
        "loss_trajectory": H.summarize_series(losses),
        "grad_norm_trajectory": H.summarize_series(grads),
        "completion_len_chars": H.summarize_series(clen),
        # Telemetry (reported, NOT gated) — entropy + PPO clip fractions, so
        # the clip-higher / kl_beta call on the next run is data-driven.
        "entropy_trajectory": H.summarize_series(series.get("entropy", [])),
        "clip_fraction_trajectory": H.summarize_series(
            series.get("clip_ratio/region_mean", [])),
        "clip_fraction_low_trajectory": H.summarize_series(
            series.get("clip_ratio/low_mean", [])),
        "clip_fraction_high_trajectory": H.summarize_series(
            series.get("clip_ratio/high_mean", [])),
    }


# --------------------------------------------------------------------------- #
# On-policy shape — spg arithmetic shared by the GPU run and the dry-run gate. #
# --------------------------------------------------------------------------- #
def _spg_shape(rows_per_gen: int, num_generations: int, *,
               num_processes: int = 1, grad_accum=None,
               on_policy: bool = False) -> dict:
    """TRL micro-batch / steps-per-generation / gradient-accumulation arithmetic
    — pure integers, the SINGLE source of truth shared by `_gpu_run` and the
    `--dry-run` pre-check (so the gate can never drift from what trains).

    TRL identity (the knobs we set):
        generation_batch_size     = micro × num_processes × spg   (rows/generation)
        effective optimizer batch = micro × num_processes × ga    (rows/optimizer step)
        optimizer steps / generation = spg // ga
        ON-POLICY  ⟺  spg == ga   (one optimizer step per generation)

    Why this grew an `ga`/`num_processes`/`on_policy` axis (phase3-rl-resize-8xh200):
    the old shape pinned ga=1, so the ONLY way to consume a big generation batch
    was many small (micro) off-policy steps → at rows_per_gen=1024 that is spg=64,
    refresh-starved. `on_policy=True` sets ga=spg so the whole generation is one
    optimizer step via accumulation — large on-policy batch at ZERO extra
    activation memory. Explicit `grad_accum` amortises (spg//ga steps/generation).
    `num_processes` (DDP world size) multiplies the per-step batch.

    Defaults (grad_accum=None, on_policy=False, num_processes=1) ⇒ ga=1 ⇒
    byte-identical to the pre-ga harness. `micro` is bounded by `PHASE2_MICRO_CAP`
    (activation memory); raise it on a dedicated training GPU to shrink ga.
    """
    g = max(2, int(num_generations))
    np_ = max(1, int(num_processes))
    gen_batch = int(rows_per_gen) * g
    micro_cap = int(os.environ.get("PHASE2_MICRO_CAP", "8"))
    # micro: largest ≤cap (doubling from g) s.t. micro×num_processes tiles the
    # generation batch evenly (TRL needs gen_batch divisible by micro×np).
    micro = g
    while micro * 2 <= micro_cap and gen_batch % (micro * 2 * np_) == 0:
        micro *= 2
    spg = gen_batch // (micro * np_)          # total micro-steps per generation
    if on_policy:
        ga = spg                              # one optimizer step per generation
    elif grad_accum is not None:
        ga = max(1, int(grad_accum))
    else:
        ga = 1                                # legacy: micro-step == optimizer step
    return {
        "g": g, "gen_batch": gen_batch, "num_processes": np_,
        "micro_cap": micro_cap, "micro": micro, "ga": ga, "spg": spg,
        "opt_steps_per_gen": max(1, spg // ga),
        "effective_batch": micro * np_ * ga,
        "divisible": gen_batch % (micro * np_) == 0,
    }


def _rotated_seat(roll_index: int, *, base_seat: int, num_players: int,
                  rotate: bool) -> int:
    """Trainable seat for `roll_index`. Round-robin across all seats when
    `rotate` (constant within a roll ⇒ constant within each K-group, so the
    §A.7 within-group standardization is unaffected); else the fixed
    `base_seat`. Pure + module-level so it is unit-testable without standing up
    the GPU trainer (the `_gpu_run` closure delegates here)."""
    if not rotate:
        return int(base_seat)
    return (int(base_seat) + int(roll_index)) % int(num_players)


# --------------------------------------------------------------------------- #
# CPU dry-run — every new glue piece, no GPU/network.                         #
# --------------------------------------------------------------------------- #
def _dry_run(args) -> dict:
    """1) actor-mask post-tag (incl. nested reveal) + seat-parameterized
    export contract + §A.7 group_key — reuses the 2.3 wiring check.
    2) heuristic-shim legality on REAL stored corpus prompts through the
    real parser, only when the requested config actually uses the heuristic."""

    # ---- (1) post-tag + export + group key -------------------------------
    games = H.rollout_group(
        args.seed_start, trainable_policy_fn=H.make_stub_policy_fn("bestresp"),
        k=args.k)
    stripped = []
    for g in games:
        gg = copy.deepcopy(g)
        for rnd in gg["rounds"]:
            for rec in rnd["players"]:
                rec.pop("actor_id", None)
        gg["rounds"][0]["players"][1]["reveal"] = {"actor_id": "STALE"}
        stripped.append(gg)
    tagged = [_post_tag_actor_mask(
        g, trainable_seat=args.trainable_seat,
        opponent_actor_id="heuristic") for g in stripped]
    rows = H.flatten_training_rows(tagged, trainable_seat=args.trainable_seat)
    gkeys = {r["group_key"] for r in rows}
    wiring = {
        "post_tag_top_level": all(
            rec["actor_id"] in (H.TRAINABLE_ACTOR_ID, "heuristic")
            for g in tagged for rnd in g["rounds"] for rec in rnd["players"]),
        "post_tag_nested_reveal":
            tagged[0]["rounds"][0]["players"][1]["reveal"]["actor_id"]
            == "heuristic",
        "export_contract_ok": len(rows) == args.k * H.NUM_ROUNDS,
        "all_rows_trainable_seat": all(
            r["actor_id"] == H.TRAINABLE_ACTOR_ID
            and r["player_id"] == args.trainable_seat for r in rows),
        "a7_single_kgroup_for_one_seed": len(gkeys) == 1,  # K rollouts, 1 seed
    }

    # ---- (2) heuristic legality on real prompts --------------------------
    heuristic_required = _needs_heuristic_endpoint(args)
    legality = {"bid_checked": 0, "reveal_checked": 0,
                "illegal": [], "unparsed": [],
                "skipped": not heuristic_required}
    if heuristic_required:
        from megagem.training.heuristic_endpoint import decide
        from megagem.game.actions import parse_bid, parse_reveal

        corpus = sorted(glob.glob(
            "results/phase1_corpus/chart_A/**/*.json", recursive=True))[:6]
        for fpath in corpus:
            g = json.load(open(fpath))
            for rnd in g.get("rounds", []):
                for rec in rnd.get("players", []):
                    bp = rec.get("prompt")
                    if bp:
                        out = decide([{"role": "user", "content": bp}])
                        pb = parse_bid(out)
                        legality["bid_checked"] += 1
                        if not pb.valid:
                            legality["unparsed"].append(("bid", fpath))
                        elif pb.bid < 0:
                            legality["illegal"].append(("bid<0", pb.bid, fpath))
                    rv = rec.get("reveal")
                    if isinstance(rv, dict) and rv.get("prompt"):
                        rp = rv["prompt"]
                        out = decide([{"role": "user", "content": rp}])
                        pr = parse_reveal(out)
                        legality["reveal_checked"] += 1
                        if not pr.valid:
                            legality["unparsed"].append(("reveal", fpath))
                        else:
                            # in-hand legality vs the prompt's own hand section
                            from megagem.training.heuristic_endpoint import _extract_section
                            hand = (_extract_section(rp, "your_private_hand")
                                    or {}).get("cards") or []
                            if hand and pr.gem_color not in hand:
                                legality["illegal"].append(
                                    ("reveal_not_in_hand", pr.gem_color, fpath))

    legal_ok = (
        True if not heuristic_required else (
            not legality["illegal"] and not legality["unparsed"]
            and legality["bid_checked"] > 0
            and legality["reveal_checked"] > 0
        )
    )

    # ---- (3) §3.3 opponent pool — pinned anchor, draw determinism, ring
    # buffer eviction (pinned never evicted), PFSP weighting cold-start. -----
    # repl_08 design: pinned step_0 anchor replaces the scripted heuristic as
    # the always-available stationary opponent. With anneal=0/p_max=1.0 the
    # pool is active from step 0.
    anchor_snap = Snapshot(
        step=0, served_name=ADP.snapshot_served_name(0),
        adapter_path="snapshots/step_0")
    _dry_run_heur_spec = (
        _HEURISTIC_SPEC if args.p_heuristic > 0.0 else None)
    # Codex round-2 fix: dry-run must pass api_specs + p_api so the
    # construction-time `p_heuristic + p_api > 1` budget check fires on the
    # CPU path. Pre-fix, an invalid `p_heuristic=0.7 + p_api=0.4` would
    # only crash after the rollout vLLM was spun up — wasted minutes of
    # GPU time. Parse the same way the GPU path does (~line 1728).
    _dry_run_api_models = [
        m.strip() for m in (args.opp_api_models or "").split(",")
        if m.strip()
    ]
    _dry_run_api_specs = [
        OpponentSpec(
            kind="api", served_name=m,
            actor_id="api_" + m.replace("/", "_"))
        for m in _dry_run_api_models
    ]
    dpool = OpponentPool(
        pinned_snapshots=[anchor_snap],
        max_snapshots=args.max_snapshots,
        anneal_start=args.opp_anneal_start, anneal_end=args.opp_anneal_end,
        p_max=args.opp_anneal_pmax, rng_seed=args.opp_pool_seed,
        # Codex fix: dry-run must validate the same `p_anchor_floor` the GPU
        # path uses, so out-of-range values fail before spend.
        p_anchor_floor=args.opp_anchor_floor,
        current_self_spec=_current_self_spec(),
        p_current_self=args.p_current_self,
        # repl_08 v3: same for p_heuristic — dry-run must reject an invalid
        # value (e.g. > 1 or < 0) and the heuristic_spec=None mismatch.
        heuristic_spec=_dry_run_heur_spec,
        p_heuristic=args.p_heuristic,
        heuristic_anneal_end=args.heuristic_anneal_end,
        # repl_08 v3 (Codex round-2): API specs/prob in dry-run so the
        # cold-start budget check (p_heuristic + p_api ≤ 1) fires on CPU.
        api_specs=_dry_run_api_specs,
        p_api=args.opp_api_prob,
    )
    for i in range(args.max_snapshots + 1):  # +1 ⇒ forces exactly one UNPINNED eviction
        st = args.snapshot_every * (i + 1)
        dpool.add_snapshot(Snapshot(
            st, ADP.snapshot_served_name(st), f"snapshots/step_{st}"))
    far = args.opp_anneal_end + 1000
    draw_seeds = list(range(args.seed_start, args.seed_start + 4))
    # Post-tag with a SNAPSHOT actor_id: the §A.6 mask must still drop every
    # opponent token — it is opponent-agnostic (heuristic / snapshot alike).
    snap_rows = H.flatten_training_rows(
        [_post_tag_actor_mask(g, trainable_seat=args.trainable_seat,
                              opponent_actor_id="snapshot_75")
         for g in stripped],
        trainable_seat=args.trainable_seat)
    pool_checks = {
        "anneal_zero_at_start": anneal_probability(
            args.opp_anneal_start, anneal_start=args.opp_anneal_start,
            anneal_end=args.opp_anneal_end,
            p_max=args.opp_anneal_pmax) == 0.0,
        "anneal_pmax_after_end": abs(anneal_probability(
            far, anneal_start=args.opp_anneal_start,
            anneal_end=args.opp_anneal_end, p_max=args.opp_anneal_pmax)
            - args.opp_anneal_pmax) < 1e-9,
        # Unpinned ring buffer evicted oldest UNPINNED. Pinned anchor stays.
        "unpinned_ring_buffer_size":
            len(dpool.unpinned_snapshots()) == args.max_snapshots,
        "pinned_anchor_preserved":
            len(dpool.pinned_snapshots()) == 1
            and dpool.pinned_snapshots()[0].step == 0,
        "total_pool_size":
            len(dpool.snapshots()) == args.max_snapshots + 1,
        "draw_deterministic_per_seed": all(
            dpool.draw(far, s).served_name == dpool.draw(far, s).served_name
            for s in draw_seeds),
        # Cold-start PFSP: no WR observations yet ⇒ uniform regime.
        "pfsp_cold_start_uniform":
            dpool.telemetry(far).get("pfsp_regime") == "uniform",
        "snapshot_actor_mask_ok": (
            len(snap_rows) == args.k * H.NUM_ROUNDS
            and all(r["actor_id"] == H.TRAINABLE_ACTOR_ID
                    for r in snap_rows)),
        "no_heuristic_default_pool": (
            args.p_heuristic == 0.0
            and all(dpool.draw(far, s).kind != "heuristic"
                    for s in range(args.seed_start, args.seed_start + 32))
        ),
        "current_self_draw_possible": (
            args.p_current_self <= 0.0
            or any(dpool.draw(far, s).kind == "current_self"
                   for s in range(args.seed_start, args.seed_start + 128))
        ),
    }

    # Projected on-policy shape — pure arithmetic via the SAME _spg_shape the
    # GPU run uses. Catches a REFRESH-STARVED config (e.g. the old spg=24 →
    # ~9 fresh refreshes) BEFORE any GPU spend. With --require-onpolicy a
    # refresh-starved projection FAILS the dry-run; otherwise it is reported
    # only (the default seam PROFILE is legitimately seam-shaped).
    #
    # repl_08: spg=2 is the INTENTIONAL strict on-policy shape — distinguish
    # "low refresh count" (the failure we want to catch) from "spg>1" (which
    # is true even for the intentional spg=2 design). The starvation guard
    # bites when projected_n_refresh < args.steps // 5 (the "≤20% of steps
    # got fresh rollouts" definition). spg=2 over 200/400 steps gives
    # 100/200 refreshes — comfortably above the threshold.
    shape = _spg_shape(
        args.rows_per_gen, args.num_generations,
        num_processes=args.num_processes,
        grad_accum=args.gradient_accumulation_steps,
        on_policy=args.on_policy)
    # Fresh generations = optimizer-steps / (steps-per-generation). With ga the
    # cadence is opt_steps_per_gen = spg//ga, NOT raw spg: --on-policy (ga=spg)
    # ⟹ opt_steps_per_gen=1 ⟹ EVERY optimizer step gets a fresh roll.
    proj_n_refresh = args.steps // shape["opt_steps_per_gen"]
    # The starvation threshold: a run is refresh-starved when fewer than 20%
    # of optimizer steps got a fresh rollout, OR (legacy) when refreshes are
    # absolutely tiny. NOT marked starved purely on a high spg; ga can make a
    # high-spg run fully on-policy.
    refresh_floor = max(5, args.steps // 5)
    proj_starved = proj_n_refresh < refresh_floor
    proj_seam = (shape["opt_steps_per_gen"] > 1) or proj_starved
    onpolicy_shape = {
        **shape, "steps": args.steps,
        "projected_n_refresh": proj_n_refresh,
        "projected_is_seam_shape": proj_seam,
        "projected_is_refresh_starved": proj_starved,
        "refresh_floor": refresh_floor,
        "require_onpolicy": args.require_onpolicy,
    }

    passed = all(wiring.values()) and legal_ok and all(pool_checks.values())
    if args.require_onpolicy and proj_starved:
        passed = False  # refresh-starved evidence config — abort before $ spend
    return {
        "step": "3.x", "mode": "DRY-RUN (CPU — glue + heuristic legality)",
        "status": "PASS" if passed else "FAIL",
        "wiring_checks": wiring,
        "heuristic_legality": legality,
        "heuristic_legal_ok": legal_ok,
        "pool_checks": pool_checks,
        "onpolicy_shape": onpolicy_shape,
        "config": {
            "steps": args.steps, "k": args.k,
            "learning_rate": args.learning_rate, "kl_beta": args.kl_beta,
            "kl_max": args.kl_max,
            "epsilon": args.epsilon, "epsilon_high": args.epsilon_high,
        },
        "rows": len(rows), "group_keys": sorted(gkeys),
    }


def _balanced_select(rows: list[dict], n: int, *, seed: int = 0) -> list[dict]:
    """Pick exactly `n` rows spread EVENLY across K-groups AND, within each
    group, UNIFORMLY across its K rollouts × rounds (never a prefix).

    `build_training_rows` emits rows deterministically sorted by `group_key`
    (then game/round), so a naive `rows[:n]` is prefix-biased — with 6 seeds ×
    K=8 it would train almost entirely on seed-0's group. Round-robin one row
    per `group_key` per cycle so the kept batch covers every seed/group.

    Within a group the rows stay (game, round)-ordered, so popping the FRONT
    each cycle would keep only that group's EARLIEST rollout/rounds — then K=8
    improves advantage normalization but NOT gradient coverage: late rounds and
    rollouts k≥1 would never reach the loss. So each group is shuffled (seeded
    ⇒ reproducible; the caller passes the roll index so successive rolls draw
    differently) before the round-robin, making the kept batch a uniform sample
    of each group's rollouts × rounds. Each row keeps the within-group-
    standardized `precomputed_advantage` that `flatten` already finalised over
    the FULL groups (selection is a *subset*, never a recompute).
    """
    import random
    from collections import OrderedDict
    buckets: "OrderedDict[str, list]" = OrderedDict()
    for r in rows:
        buckets.setdefault(r["group_key"], []).append(r)
    rng = random.Random(seed)
    queues: list[list] = []
    for v in buckets.values():
        q = list(v)
        rng.shuffle(q)  # within-group uniform coverage — not a prefix
        queues.append(q)
    out: list[dict] = []
    while len(out) < n and any(queues):
        for q in queues:
            if q:
                out.append(q.pop())
                if len(out) == n:
                    break
    return out


def _train_seeds_for_roll(
    seed_start: int, num_seeds: int, roll_index: int, *, fixed: bool = False
) -> list[int]:
    """Training seed schedule.

    The old evidence run reused the same handful of seeds for every rollout,
    confounding policy learning with six game instances. The default is now a
    fresh contiguous seed block per rollout; ``fixed=True`` is retained only as
    an explicit reproducibility escape hatch.
    """
    offset = 0 if fixed else int(roll_index) * int(num_seeds)
    start = int(seed_start) + offset
    return list(range(start, start + int(num_seeds)))


# --------------------------------------------------------------------------- #
# DDP (rank-sharded training, phase3-rl-resize-8xh200). SINGLE-NODE only       #
# (torchrun --standalone ⇒ all ranks share the local fs). Design: RANK 0 runs  #
# the existing roll (drives the vLLM DP workers, computes advantages over the  #
# FULL K-groups), then hands the rows to the other ranks via a file; every     #
# rank trains its DistributedSampler shard and DDP all-reduces gradients. So   #
# generation happens ONCE (not per-rank) and K-groups are never split — the    #
# 2nd train GPU buys training throughput (ga halves), not 2× the bottleneck.   #
# All of this is inert when world_size==1 (the recommended 7gen+1train config).#
# --------------------------------------------------------------------------- #
def _ddp_rank_world() -> tuple[int, int]:
    """(rank, world_size) from the torchrun env; (0, 1) when not distributed."""
    return (int(os.environ.get("RANK", "0")),
            int(os.environ.get("WORLD_SIZE", "1")))


def _ddp_rows_paths(tmp_dir, ridx: int) -> tuple[str, str]:
    base = os.path.join(str(tmp_dir), f"ddp_rows_{int(ridx)}")
    return base + ".json", base + ".done"


def _ddp_send_rows(tmp_dir, ridx: int, rows: list) -> None:
    """Rank 0: publish this roll's FULL rows (advantages already finalised over
    the complete K-groups) for the other ranks. The `.done` marker is written
    LAST so a reader never sees a partial file."""
    jpath, dpath = _ddp_rows_paths(tmp_dir, ridx)
    with open(jpath, "w") as f:
        json.dump(rows, f, default=str)
    with open(dpath, "w") as f:
        f.write("ok")


def _ddp_recv_rows(tmp_dir, ridx: int, *, poll_s: float = 0.5) -> list:
    """Ranks>0: block until rank 0 has published roll `ridx`, then load it. The
    timeout (PHASE3_DDP_RECV_TIMEOUT_S, default 3600) must exceed a full roll's
    generation wall-time so a genuinely-slow roll isn't mistaken for a dead
    rank 0 — but a crashed rank 0 still fails LOUDLY instead of hanging forever."""
    timeout_s = float(os.environ.get("PHASE3_DDP_RECV_TIMEOUT_S", "3600"))
    jpath, dpath = _ddp_rows_paths(tmp_dir, ridx)
    waited = 0.0
    while not os.path.exists(dpath):
        time.sleep(poll_s)
        waited += poll_s
        if waited >= timeout_s:
            raise RuntimeError(
                f"[phase3][ddp] rank {_ddp_rank_world()[0]} timed out after "
                f"{timeout_s:.0f}s waiting for rank-0 roll {ridx} ({dpath}). "
                "Rank 0 likely crashed during generation/adapter-push.")
    with open(jpath) as f:
        return json.load(f)


def _safe_float(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _wandb_log_megagem(flat: dict, megagem_step: int) -> None:
    """wandb.log under a decoupled `megagem/step` x-axis.

    Why a separate axis: TRL fires multiple wandb events per train step
    (loss, kl, lr, entropy, clip_ratio, …) via report_to=['wandb'], so
    wandb's internal step counter races ahead of `trainer.state.global_step`
    by ~4-5× per step. Naive `wandb.log(payload, step=trainer.state.global_step)`
    from roll_fn's post-roll telemetry then tries to log to a step LESS than
    wandb's current step → wandb silently drops the payload (the bug seen in
    noapi_long_dapo_repl_07: megagem/* panels empty after roll 0). Defining
    `megagem/step` as a custom step_metric for all `megagem/*` keys decouples
    the panels from wandb's global counter; each call lands as one point on
    the roll-indexed curve. `define_metric` is idempotent so re-defining per
    call is fine.
    """
    if not flat:
        return
    try:
        import wandb
        if wandb.run is None:
            return
        wandb.define_metric("megagem/step", hidden=True)
        wandb.define_metric("megagem/*", step_metric="megagem/step")
        payload = dict(flat)
        payload["megagem/step"] = int(megagem_step)
        wandb.log(payload)
    except Exception:  # noqa: BLE001 — telemetry never aborts a roll
        pass


def _quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return ys[idx]


def _dist(xs: list[float]) -> dict:
    vals = [v for v in (_safe_float(x) for x in xs) if v is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "p10": _quantile(vals, 0.10),
        "median": _quantile(vals, 0.50),
        "p90": _quantile(vals, 0.90),
        "max": max(vals),
    }


def _parse_group_key(value):
    if not isinstance(value, str):
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None


def _row_seed(row: dict):
    gk = _parse_group_key(row.get("group_key"))
    if not isinstance(gk, tuple) or not gk:
        return None
    base = gk[0]
    if isinstance(base, tuple) and base:
        return base[0]
    return None


def _row_identity(row: dict) -> tuple:
    return (
        row.get("group_key"),
        row.get("game_id"),
        row.get("round_index"),
        row.get("player_id"),
        row.get("phase"),
    )


def _counter_json(counter: Counter) -> dict:
    return {str(k): int(v) for k, v in sorted(counter.items(),
                                             key=lambda kv: str(kv[0]))}


def _reward_component_totals(rows: list[dict]) -> dict:
    totals = Counter()
    for r in rows:
        comps = r.get("reward_components") or {}
        for k in ("legal", "shaping", "terminal_correction", "terminal"):
            v = _safe_float(comps.get(k))
            if v is not None:
                totals[k] += v
    totals["sum"] = sum(totals[k] for k in (
        "legal", "shaping", "terminal_correction", "terminal"))
    return {k: float(totals.get(k, 0.0)) for k in (
        "legal", "shaping", "terminal_correction", "terminal", "sum")}


# M2 — reward-component within-K-group variance decomposition.
_REWARD_COMPONENTS = ("legal", "shaping", "terminal", "terminal_correction")


def _reward_component_var_decomposition(rows: list[dict]) -> dict:
    """Of the WITHIN-K-group reward variance, what fraction does each reward
    component contribute?

    GRPO's gradient is driven by the within-group variance of the per-game
    reward (the K rollouts at one seed/seat). A component that is ~constant
    across those K rollouts contributes ~0 variance — it is *cosmetic to the
    gradient* no matter how large its absolute magnitude. This decomposes the
    within-group variance into terminal-margin / λ-competitive-shaping /
    legality / terminal-correction shares, computed over the SAME
    ``(group_key, game_id)`` per-game sums that ``_kgroup_reward_stats`` uses
    for ``group_std`` — so the shares are directly comparable to the headline
    std. Hypothesis (Phase-3 data): ``within_group_var_share["shaping"] ≈ 0``
    ⇒ λ=0.01 shaping is cosmetic; raising λ adds mass, not gradient.

    Pure / side-effect-free; safe to call inside the per-roll telemetry path.
    """
    comps = _REWARD_COMPONENTS
    per_game_comp: dict = defaultdict(lambda: {c: 0.0 for c in comps})
    seen: set = set()
    for r in rows:
        try:
            key = (r["group_key"], r["game_id"])
        except (KeyError, TypeError):
            continue
        seen.add(key)
        cg = per_game_comp[key]
        rc = r.get("reward_components") or {}
        for c in comps:
            v = _safe_float(rc.get(c))
            if v is not None:
                cg[c] += v
    by_group: dict = defaultdict(list)  # group_key -> [(total, {comp: val}), …]
    for key in seen:
        cg = per_game_comp[key]
        by_group[key[0]].append((sum(cg[c] for c in comps), cg))
    eps = 1e-9
    shares: dict = {c: [] for c in comps}
    sum_share: list = []
    var_totals: list = []
    n_usable = 0
    for members in by_group.values():
        if len(members) < 2:
            continue
        var_total = statistics.pvariance([m[0] for m in members])
        if var_total <= eps:
            continue
        n_usable += 1
        var_totals.append(var_total)
        cvars = {c: statistics.pvariance([m[1][c] for m in members])
                 for c in comps}
        for c in comps:
            shares[c].append(cvars[c] / var_total)
        # Σ component-vars / var_total > 1 ⇔ negative cross-covariance; this
        # ratio surfaces how much of the decomposition is cross-term (the
        # components are NOT independent, so the per-component shares do not
        # sum to 1 in general).
        sum_share.append(sum(cvars.values()) / var_total)

    def _m(xs: list) -> float | None:
        return (sum(xs) / len(xs)) if xs else None

    return {
        "n_groups_usable": n_usable,
        "n_games": len(seen),
        "within_group_var_share": {c: _m(shares[c]) for c in comps},
        "sum_of_component_vars_over_total": _m(sum_share),
        "mean_within_group_var_total": _m(var_totals),
        "mean_abs_per_game": {
            c: _m([abs(per_game_comp[k][c]) for k in per_game_comp])
            for c in comps
        },
    }


def _rows_diagnostics(rows: list[dict], *,
                      opponents_by_seed: dict[int, str] | None = None) -> dict:
    opponents_by_seed = opponents_by_seed or {}
    by_seed, by_opponent, by_round, by_phase = Counter(), Counter(), Counter(), Counter()
    terminal = Counter()
    for r in rows:
        seed = _row_seed(r)
        by_seed[seed if seed is not None else "unknown"] += 1
        by_opponent[opponents_by_seed.get(seed, "unknown")] += 1
        by_round[r.get("round_index", "unknown")] += 1
        by_phase[r.get("phase", "unknown")] += 1
        terminal["terminal" if r.get("is_terminal_turn") else "non_terminal"] += 1
    return {
        "total_rows": len(rows),
        "unique_rows": len({_row_identity(r) for r in rows}),
        "by_seed": _counter_json(by_seed),
        "by_opponent": _counter_json(by_opponent),
        "by_round": _counter_json(by_round),
        "by_phase": _counter_json(by_phase),
        "terminal_turns": _counter_json(terminal),
        "reward": _dist([r.get("precomputed_reward") for r in rows]),
        "advantage": _dist([r.get("precomputed_advantage") for r in rows]),
        "reward_components": _reward_component_totals(rows),
    }


def _mean_numeric(values) -> float | None:
    vals = [v for v in (_safe_float(x) for x in values) if v is not None]
    return statistics.fmean(vals) if vals else None


def _fraction_numeric(values, pred) -> float | None:
    vals = [v for v in (_safe_float(x) for x in values) if v is not None]
    if not vals:
        return None
    return sum(1 for v in vals if pred(v)) / len(vals)


def _row_completion_token_count(row: dict, tokenizer=None) -> int:
    text = str(row.get("completion") or "")
    if tokenizer is not None:
        try:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            return max(1, len(ids or []))
        except Exception:  # noqa: BLE001 — diagnostics only
            pass
    return max(1, len(text.split()))


def _weighted_row_means(rows: list[dict], keys, *, tokenizer=None) -> dict:
    nums = {key: 0.0 for key in keys}
    dens = {key: 0.0 for key in keys}
    for row in rows:
        w = float(_row_completion_token_count(row, tokenizer))
        for key in keys:
            val = _safe_float(row.get(key))
            if val is None:
                continue
            nums[key] += val * w
            dens[key] += w
    return {
        key: ((nums[key] / dens[key]) if dens[key] > 0 else None)
        for key in keys
    }


def _mean_delta(full_rows: list[dict], selected_rows: list[dict], key: str) -> dict:
    full_mean = _mean_numeric(row.get(key) for row in full_rows)
    selected_mean = _mean_numeric(row.get(key) for row in selected_rows)
    return {
        "full_mean": full_mean,
        "selected_mean": selected_mean,
        "selected_minus_full": (
            selected_mean - full_mean
            if full_mean is not None and selected_mean is not None else None
        ),
    }


def _bucket_selection_delta(full_rows: list[dict], selected_rows: list[dict],
                            key: str, bucket_fn) -> dict:
    buckets = sorted({
        str(bucket_fn(row)) for row in [*full_rows, *selected_rows]
        if bucket_fn(row) is not None
    })
    out: dict = {}
    for bucket in buckets:
        frows = [r for r in full_rows if str(bucket_fn(r)) == bucket]
        srows = [r for r in selected_rows if str(bucket_fn(r)) == bucket]
        fmean = _mean_numeric(r.get(key) for r in frows)
        smean = _mean_numeric(r.get(key) for r in srows)
        out[bucket] = {
            "full_n": len(frows),
            "selected_n": len(srows),
            "coverage": (len(srows) / len(frows)) if frows else None,
            "full_mean": fmean,
            "selected_mean": smean,
            "selected_minus_full": (
                smean - fmean
                if fmean is not None and smean is not None else None
            ),
        }
    return out


def _selection_bias_diagnostics(
    full_rows: list[dict], selected_rows: list[dict], *,
    opponents_by_seed: dict[int, str] | None = None, tokenizer=None,
) -> dict:
    """Compare the post-filter rollout rows to the rows the optimizer sees."""
    opponents_by_seed = opponents_by_seed or {}

    def _kind(row: dict):
        seed = _row_seed(row)
        return opponents_by_seed.get(seed, "unknown")

    def _round(row: dict):
        ri = row.get("round_index")
        return f"r{int(ri):02d}" if isinstance(ri, int) else "unknown"

    def _terminal(row: dict):
        return "terminal" if row.get("is_terminal_turn") else "non_terminal"

    adv = _mean_delta(full_rows, selected_rows, "precomputed_advantage")
    reward = _mean_delta(full_rows, selected_rows, "precomputed_reward")
    tok_keys = ("precomputed_advantage", "precomputed_reward")
    tok_full = _weighted_row_means(full_rows, tok_keys, tokenizer=tokenizer)
    tok_sel = _weighted_row_means(selected_rows, tok_keys, tokenizer=tokenizer)
    adv_tok_full = tok_full.get("precomputed_advantage")
    adv_tok_sel = tok_sel.get("precomputed_advantage")
    rwd_tok_full = tok_full.get("precomputed_reward")
    rwd_tok_sel = tok_sel.get("precomputed_reward")
    neg_full = _fraction_numeric(
        (r.get("precomputed_advantage") for r in full_rows), lambda v: v < 0)
    neg_sel = _fraction_numeric(
        (r.get("precomputed_advantage") for r in selected_rows), lambda v: v < 0)
    out = {
        "advantage": adv,
        "reward": reward,
        "token_weighted": {
            "advantage_full_mean": adv_tok_full,
            "advantage_selected_mean": adv_tok_sel,
            "advantage_selected_minus_full": (
                adv_tok_sel - adv_tok_full
                if adv_tok_full is not None and adv_tok_sel is not None
                else None
            ),
            "reward_full_mean": rwd_tok_full,
            "reward_selected_mean": rwd_tok_sel,
            "reward_selected_minus_full": (
                rwd_tok_sel - rwd_tok_full
                if rwd_tok_full is not None and rwd_tok_sel is not None
                else None
            ),
        },
        "negative_advantage_frac": {
            "full": neg_full,
            "selected": neg_sel,
            "selected_minus_full": (
                neg_sel - neg_full
                if neg_full is not None and neg_sel is not None else None
            ),
        },
        "by_opponent": _bucket_selection_delta(
            full_rows, selected_rows, "precomputed_advantage", _kind),
        "by_phase": _bucket_selection_delta(
            full_rows, selected_rows, "precomputed_advantage",
            lambda r: r.get("phase", "unknown")),
        "by_round": _bucket_selection_delta(
            full_rows, selected_rows, "precomputed_advantage", _round),
        "by_terminal": _bucket_selection_delta(
            full_rows, selected_rows, "precomputed_advantage", _terminal),
        "warnings": [],
    }
    adv_sel = _safe_float(adv.get("selected_mean"))
    adv_delta = _safe_float(adv.get("selected_minus_full"))
    adv_tok_delta = _safe_float(
        out["token_weighted"].get("advantage_selected_minus_full"))
    if adv_sel is not None and adv_sel < -0.01:
        out["warnings"].append(
            "selected advantage mean is negative; optimizer batch may be "
            "biased toward anti-update rows")
    if adv_delta is not None and adv_delta < -0.02:
        out["warnings"].append(
            "selected rows have materially lower advantage mean than the "
            "available rollout rows")
    if adv_tok_delta is not None and adv_tok_delta < -0.02:
        out["warnings"].append(
            "token-weighted selected advantage is materially lower than the "
            "available rollout rows")
    return out


def _fmt_metric(x, *, signed: bool = False, pct: bool = False,
                digits: int = 4) -> str:
    v = _safe_float(x)
    if v is None:
        return "n/a"
    if pct:
        return f"{100 * v:.1f}%"
    sign = "+" if signed else ""
    return f"{v:{sign}.{digits}f}"


def _format_selection_bias_block(roll_index: int, bias: dict) -> str:
    if not bias:
        return ""
    if "error" in bias:
        return (f"  [roll {int(roll_index):03d}] SELECT-BIAS  ERROR "
                f"{bias['error']}")
    adv = bias.get("advantage") or {}
    reward = bias.get("reward") or {}
    tok = bias.get("token_weighted") or {}
    neg = bias.get("negative_advantage_frac") or {}
    lines = [
        f"  [roll {int(roll_index):03d}] SELECT-BIAS  "
        f"adv full={_fmt_metric(adv.get('full_mean'), signed=True)} "
        f"selected={_fmt_metric(adv.get('selected_mean'), signed=True)} "
        f"d={_fmt_metric(adv.get('selected_minus_full'), signed=True)}  "
        f"neg={_fmt_metric(neg.get('full'), pct=True)}->"
        f"{_fmt_metric(neg.get('selected'), pct=True)}",
        f"  [roll {int(roll_index):03d}] SELECT-BIAS  "
        f"tok_adv full={_fmt_metric(tok.get('advantage_full_mean'), signed=True)} "
        f"selected={_fmt_metric(tok.get('advantage_selected_mean'), signed=True)} "
        f"d={_fmt_metric(tok.get('advantage_selected_minus_full'), signed=True)}  "
        f"reward_d={_fmt_metric(reward.get('selected_minus_full'), signed=True)} "
        f"tok_reward_d={_fmt_metric(tok.get('reward_selected_minus_full'), signed=True)}",
    ]
    kind_parts = []
    for kind in _OPPONENT_KIND_ORDER:
        stats = (bias.get("by_opponent") or {}).get(kind)
        if not stats:
            continue
        kind_parts.append(
            f"{kind}: d_adv="
            f"{_fmt_metric(stats.get('selected_minus_full'), signed=True)} "
            f"cov={_fmt_metric(stats.get('coverage'), pct=True)}")
    if kind_parts:
        lines.append(
            f"  [roll {int(roll_index):03d}] SELECT-BIAS by-kind  "
            + "  |  ".join(kind_parts))
    for warning in bias.get("warnings") or []:
        lines.append(
            f"  [roll {int(roll_index):03d}] SELECT-BIAS  WARN {warning}")
    return "\n".join(lines)


def _terminal_metrics_by_game(games: list[dict], trainable_seat: int) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for gid, g in enumerate(games):
        finals = ((g.get("final_results") or {}).get("final_scores") or [])
        scores = {
            int(x["player_id"]): float(x["final_score"])
            for x in finals
            if "player_id" in x and "final_score" in x
        }
        if trainable_seat not in scores:
            continue
        others = [v for p, v in scores.items() if p != trainable_seat]
        margin = scores[trainable_seat] - (
            statistics.fmean(others) if others else 0.0)
        out[gid] = {
            "final_score": scores[trainable_seat],
            "true_margin": margin,
        }
    return out


def _rankdata(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
    if sx <= 1e-12 or sy <= 1e-12:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    return statistics.fmean((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def _anchor_winrate_trend(rolls_meta: list, *, anchor_step: int = 0) -> dict:
    """Trainee win-rate vs the FIXED step-0 anchor, as a trend over training.

    This is the ONLY metric that detects ABSOLUTE improvement in self-play:
    win-rate against a *contemporaneous* equal-strength opponent is pinned at
    chance by construction (the opponent improves in lockstep), so a flat
    contemporaneous WR says nothing. The pinned step-0 SFT anchor is a fixed
    reference — if the policy is getting absolutely stronger, its win-rate vs
    that anchor must rise. A flat/zero slope here is the signature of "no
    durable improvement" (the v3_2 finding); a positive slope is the signal
    v3_1 showed. Reads ``per_snapshot_age.by_step[anchor_step].{win_rate,
    n_total}`` already produced per roll by ``_per_snapshot_age_stats``.

    Pure; tolerant of int- or str-keyed ``by_step`` (in-memory vs JSON-loaded).
    """
    pts: list = []  # (step, win_rate, n_total)
    for r in rolls_meta:
        if not isinstance(r, dict):
            continue
        step = r.get("step")
        by_step = ((r.get("per_snapshot_age") or {}).get("by_step") or {})
        anc = by_step.get(anchor_step)
        if anc is None:
            anc = by_step.get(str(anchor_step))
        if not isinstance(anc, dict):
            continue
        wr = _safe_float(anc.get("win_rate"))
        n = anc.get("n_total")
        if wr is None or step is None or not n:
            continue
        pts.append((float(step), wr, int(n)))
    pts.sort(key=lambda p: p[0])
    total_games = sum(p[2] for p in pts)
    base = {
        "anchor_step": anchor_step,
        "n_rolls_with_anchor": len(pts),
        "total_anchor_games": total_games,
    }
    if len(pts) < 2:
        return {**base, "slope": None, "delta_half": None,
                "spearman_wr_vs_step": None,
                "first_half_mean_wr": (pts[0][1] if pts else None),
                "second_half_mean_wr": (pts[0][1] if pts else None),
                "win_rate_series": [{"step": int(s), "win_rate": w, "n": n}
                                    for s, w, n in pts]}
    steps = [p[0] for p in pts]
    wrs = [p[1] for p in pts]
    # OLS slope = cov(step, wr) / var(step). Computed directly (not via pearson)
    # so a perfectly FLAT win-rate reports slope 0.0 — the null we want to flag —
    # rather than None (pearson is undefined when wr has zero variance).
    mx = statistics.fmean(steps)
    my = statistics.fmean(wrs)
    sxx = sum((x - mx) ** 2 for x in steps)
    sxy = sum((x - mx) * (y - my) for x, y in zip(steps, wrs))
    slope = (sxy / sxx) if sxx > 1e-12 else None
    # Standard error + t-stat of the slope, so the gate can test SIGNIFICANCE
    # (a positive-but-noisy slope must NOT read as "improving"). Needs ≥3 points
    # (df = n-2 ≥ 1); otherwise slope_t stays None and callers fall back to sign.
    slope_se = None
    slope_t = None
    if slope is not None and len(pts) > 2:
        sse = sum((y - (my + slope * (x - mx))) ** 2
                  for x, y in zip(steps, wrs))
        s2 = sse / (len(pts) - 2)
        if s2 <= 1e-18:                      # perfect linear fit
            slope_se = 0.0
            slope_t = (math.inf if slope > 0 else
                       (-math.inf if slope < 0 else 0.0))
        else:
            slope_se = math.sqrt(s2 / sxx)
            slope_t = slope / slope_se
    half = len(pts) // 2
    first_mean = statistics.fmean(wrs[:half] or wrs[:1])
    second_mean = statistics.fmean(wrs[half:] or wrs[-1:])
    return {
        **base,
        "slope": slope,
        "slope_se": slope_se,
        "slope_t": slope_t,
        "first_half_mean_wr": first_mean,
        "second_half_mean_wr": second_mean,
        "delta_half": second_mean - first_mean,
        "spearman_wr_vs_step": _spearman(steps, wrs),
        "win_rate_series": [{"step": int(s), "win_rate": w, "n": n}
                            for s, w, n in pts],
    }


def _anchor_winrate_improving(trend: dict, *, slope_min: float = 0.0,
                              t_min: float = 2.0) -> bool:
    """Informational gate: is the anchor-WR trend SIGNIFICANTLY upward?

    Tests slope *significance*, not just sign — a positive-but-noisy slope
    (the repl_09 false-positive: slope>0 at t≈1.8) must NOT read as improving.
    `slope=None` ⇒ too few anchor draws to judge ⇒ non-fatal pass (don't FAIL a
    run we can't assess). `slope_t=None` ⇒ <3 points, can't assess significance
    ⇒ fall back to sign.
    """
    if not isinstance(trend, dict):
        return True
    slope = trend.get("slope")
    if slope is None:
        return True
    t = trend.get("slope_t")
    if t is None:
        return slope > slope_min
    return (slope > slope_min) and (t >= t_min)


def _vec_l2(d: dict | None) -> float | None:
    return math.sqrt(sum(v * v for v in d.values())) if d else None


def _vec_cos(d1: dict | None, d2: dict | None) -> float | None:
    """Cosine over the SHARED keys of two bucket-vectors. None if no overlap or
    either shared-restricted vector is ~zero."""
    if not d1 or not d2:
        return None
    keys = sorted(set(d1) & set(d2))
    if not keys:
        return None
    u = [d1[k] for k in keys]
    v = [d2[k] for k in keys]
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    if nu < 1e-12 or nv < 1e-12:
        return None
    return sum(x * y for x, y in zip(u, v)) / (nu * nv)


def _m3_bucket_stats(rows: list[dict], rmax: int) -> dict:
    """Core M3 computation on a row subset: split the subset's K-groups into two
    disjoint halves, take the per-bucket (phase × round-third, from the SHARED
    ``rmax``) mean standardized advantage in each half, correlate them, and
    return the combined bucket-vector. Pure; reused for the full roll and for
    each seat / length-tercile slice (same ``rmax`` ⇒ comparable buckets across
    slices)."""
    def _bucket(r: dict):
        ri = r.get("round_index")
        if not isinstance(ri, int):
            return None
        third = 0 if rmax <= 0 else min(2, (3 * ri) // (rmax + 1))
        return f"{r.get('phase') or '?'}|t{third}"

    gks = sorted({str(r.get("group_key")) for r in rows
                  if r.get("group_key") is not None})
    half_of = {gk: (i % 2) for i, gk in enumerate(gks)}
    halves = [defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])]
    combined: dict = defaultdict(lambda: [0.0, 0])
    for r in rows:
        gk = r.get("group_key")
        a = _safe_float(r.get("precomputed_advantage"))
        if gk is None or a is None:
            continue
        h = half_of.get(str(gk))
        b = _bucket(r)
        if h is None or b is None:
            continue
        halves[h][b][0] += a
        halves[h][b][1] += 1
        combined[b][0] += a
        combined[b][1] += 1

    def _means(acc: dict) -> dict:
        return {b: s / n for b, (s, n) in acc.items() if n > 0}

    mA, mB = _means(halves[0]), _means(halves[1])
    cmean = _means(combined)
    shared = sorted(set(mA) & set(mB))
    shc = (_pearson([mA[b] for b in shared], [mB[b] for b in shared])
           if len(shared) >= 2 else None)
    return {
        "n_groups": len(gks),
        "n_buckets": len(cmean),
        "n_shared_buckets": len(shared),
        "split_half_corr": shc,
        "transferable_magnitude": _vec_l2(cmean),
        "magnitude_A": _vec_l2(mA),
        "magnitude_B": _vec_l2(mB),
        "vector": cmean,
    }


def _accumulate_seat_vector(state: dict, sig: dict) -> None:
    """Running-mean the roll's bucket-vector into a per-seat accumulator on
    ``state`` so the NEXT roll's cross_seat_cosine compares STABLE per-seat
    tilts (a single roll is one seat, since the trainable seat rotates per
    roll). Mutates ``state['seat_vectors']`` / ``state['seat_vectors_n']``."""
    seat = sig.get("seat")
    vec = sig.get("vector")
    if seat is None or not isinstance(vec, dict) or not vec:
        return
    sv = state.setdefault("seat_vectors", {})
    sn = state.setdefault("seat_vectors_n", {})
    prev = sv.get(seat)
    n = sn.get(seat, 0)
    if prev is None:
        sv[seat] = dict(vec)
    else:
        merged: dict = {}
        for k in set(prev) | set(vec):
            pv, cv = prev.get(k), vec.get(k)
            if pv is None:
                merged[k] = cv
            elif cv is None:
                merged[k] = pv
            else:
                merged[k] = (pv * n + cv) / (n + 1)
        sv[seat] = merged
    sn[seat] = n + 1


def _transferable_signal_probe(rows: list[dict], *,
                               prev_vector: dict | None = None,
                               seat_vectors: dict | None = None) -> dict:
    """Does the within-group advantage point a CONSISTENT direction across
    *disjoint* boards, or is it board-idiosyncratic noise that cancels?

    The within-group-standardized advantage is μ=0 per group, so any non-zero
    MEAN advantage for a board-agnostic action bucket (phase × round-third) is
    the systematic, board-independent tilt — the *transferable* component that
    survives averaging across fresh seeds. We split the roll's K-groups into
    two disjoint halves (different boards), take the per-bucket mean advantage
    in each half, and correlate them: high ⇒ a real transferable direction;
    near-zero ⇒ the gradient cancels across boards (the diagnosed failure mode).
    ``self_consistency_cosine`` compares this roll's bucket-vector to the
    previous roll's (``prev_vector``) — a flat-near-zero cosine across the run
    is the smoking gun for "no durable transferable signal."

    SEAT conditioning (2026-05-31): the trainable seat rotates per roll, so a
    single roll is one seat. ``seat`` tags the roll; ``cross_seat_cosine``
    compares this roll's tilt to the OTHER seats' ACCUMULATED tilts (passed as
    ``seat_vectors``, maintained across rolls by ``_accumulate_seat_vector``).
    A high cross-seat cosine ⇒ the tilt is SEAT-INVARIANT (genuine transferable
    structure); a low/negative one ⇒ a POSITIONAL artifact (uncancelled
    seat-value asymmetry masquerading as transferable signal) — the hypothesis
    the FIXED-seat-0 v3_1/v3_2 runs could not test. LENGTH conditioning
    (``by_length_tercile``) buckets by per-game length tercile to see whether
    the tilt concentrates in short/long games. Returns ``vector`` + ``seat`` so
    the caller threads/accumulates them. Pure / side-effect-free.
    """
    rounds = [r.get("round_index") for r in rows
              if isinstance(r.get("round_index"), int)]
    if not rounds:
        return {"error": "no round_index on rows"}
    rmax = max(rounds)

    full = _m3_bucket_stats(rows, rmax)
    cmean = full["vector"]
    self_cos = _vec_cos(cmean, prev_vector) if prev_vector else None

    # --- seat conditioning --- #
    seats_present = sorted({r.get("player_id") for r in rows
                            if r.get("player_id") is not None})
    seat = seats_present[0] if len(seats_present) == 1 else None
    by_seat: dict = {}
    seat_vec_local: dict = {}
    for s in seats_present:
        st = _m3_bucket_stats([r for r in rows if r.get("player_id") == s], rmax)
        by_seat[s] = {"split_half_corr": st["split_half_corr"],
                      "transferable_magnitude": st["transferable_magnitude"],
                      "n_groups": st["n_groups"]}
        seat_vec_local[s] = st["vector"]

    cross: dict = {}
    # in-call pairwise (only when rows span >1 seat — tests / multi-roll input)
    for i, a in enumerate(seats_present):
        for b in seats_present[i + 1:]:
            c = _vec_cos(seat_vec_local.get(a), seat_vec_local.get(b))
            if c is not None:
                cross[f"{a}|{b}"] = c
    # cross-roll: this roll's seat vs the OTHER seats' accumulated tilts
    if seat_vectors and seat is not None:
        for s, vec in seat_vectors.items():
            if s == seat:
                continue
            c = _vec_cos(cmean, vec)
            if c is not None:
                cross.setdefault(f"{seat}|{s}", c)
    cross_vals = list(cross.values())

    # --- game-length conditioning (per-game length = max round_index) --- #
    glen: dict = {}
    for r in rows:
        gid = r.get("game_id")
        ri = r.get("round_index")
        if gid is None or not isinstance(ri, int):
            continue
        glen[gid] = max(glen.get(gid, ri), ri)
    by_length: dict = {}
    if len(glen) >= 3 and len({*glen.values()}) >= 2:
        lens = sorted(glen.values())
        q1 = lens[len(lens) // 3]
        q2 = lens[(2 * len(lens)) // 3]

        def _terc(length: int) -> str:
            return "short" if length <= q1 else ("long" if length > q2 else "mid")

        for label in ("short", "mid", "long"):
            sub = [r for r in rows if r.get("game_id") in glen
                   and _terc(glen[r["game_id"]]) == label]
            if sub:
                st = _m3_bucket_stats(sub, rmax)
                by_length[label] = {
                    "transferable_magnitude": st["transferable_magnitude"],
                    "n_groups": st["n_groups"]}
    length_mags = [v["transferable_magnitude"] for v in by_length.values()
                   if v.get("transferable_magnitude") is not None]
    length_spread = (max(length_mags) - min(length_mags)
                     if len(length_mags) >= 2 else None)

    return {
        "n_groups": full["n_groups"],
        "n_buckets": full["n_buckets"],
        "n_shared_buckets": full["n_shared_buckets"],
        "split_half_corr": full["split_half_corr"],
        "transferable_magnitude": full["transferable_magnitude"],
        "magnitude_A": full["magnitude_A"],
        "magnitude_B": full["magnitude_B"],
        "self_consistency_cosine": self_cos,
        "seat": seat,
        "n_seats_in_roll": len(seats_present),
        "by_seat": by_seat or None,
        "cross_seat_cosine": cross or None,
        "cross_seat_cosine_min": (min(cross_vals) if cross_vals else None),
        "cross_seat_cosine_mean": (sum(cross_vals) / len(cross_vals)
                                   if cross_vals else None),
        "by_length_tercile": by_length or None,
        "length_magnitude_spread": length_spread,
        "vector": cmean,
    }


def _kgroup_alignment(rows: list[dict], games: list[dict],
                      trainable_seat: int) -> dict:
    game_metrics = _terminal_metrics_by_game(games, trainable_seat)
    per_game: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"reward_sum": 0.0, "advantage_sum": 0.0, "n_rows": 0})
    for r in rows:
        gid = r.get("game_id")
        if not isinstance(gid, int):
            continue
        key = (str(r.get("group_key")), gid)
        per_game[key]["reward_sum"] += float(r.get("precomputed_reward") or 0.0)
        per_game[key]["advantage_sum"] += float(
            r.get("precomputed_advantage") or 0.0)
        per_game[key]["n_rows"] += 1

    groups: dict[str, list[dict]] = defaultdict(list)
    for (gk, gid), vals in per_game.items():
        if gid not in game_metrics:
            continue
        groups[gk].append({"game_id": gid, **vals, **game_metrics[gid]})

    out_groups = []
    for gk, vals in sorted(groups.items()):
        vals.sort(key=lambda x: x["game_id"])
        margins = [v["true_margin"] for v in vals]
        rewards = [v["reward_sum"] for v in vals]
        advs = [v["advantage_sum"] for v in vals]
        out_groups.append({
            "group_key": gk,
            "seed": _row_seed({"group_key": gk}),
            "n_games": len(vals),
            "true_margin": _dist(margins),
            "final_score": _dist([v["final_score"] for v in vals]),
            "reward_sum": _dist(rewards),
            "advantage_sum": _dist(advs),
            "spearman_reward_vs_margin": _spearman(rewards, margins),
            "spearman_advantage_vs_margin": _spearman(advs, margins),
            "pearson_reward_vs_margin": _pearson(rewards, margins),
            "pearson_advantage_vs_margin": _pearson(advs, margins),
            "degenerate": {
                "true_margin_std_le_1e-6": (
                    (statistics.pstdev(margins) if len(margins) > 1 else 0.0)
                    <= 1e-6),
                "reward_sum_std_le_1e-6": (
                    (statistics.pstdev(rewards) if len(rewards) > 1 else 0.0)
                    <= 1e-6),
                "advantage_sum_std_le_1e-6": (
                    (statistics.pstdev(advs) if len(advs) > 1 else 0.0)
                    <= 1e-6),
            },
        })

    reward_corrs = [
        g["spearman_reward_vs_margin"] for g in out_groups
        if g["spearman_reward_vs_margin"] is not None
    ]
    adv_corrs = [
        g["spearman_advantage_vs_margin"] for g in out_groups
        if g["spearman_advantage_vs_margin"] is not None
    ]
    return {
        "n_groups": len(out_groups),
        "spearman_reward_vs_margin": _dist(reward_corrs),
        "spearman_advantage_vs_margin": _dist(adv_corrs),
        "frac_margin_degenerate": (
            sum(g["degenerate"]["true_margin_std_le_1e-6"] for g in out_groups)
            / len(out_groups)
            if out_groups else None
        ),
        "frac_reward_degenerate": (
            sum(g["degenerate"]["reward_sum_std_le_1e-6"] for g in out_groups)
            / len(out_groups)
            if out_groups else None
        ),
        "groups": out_groups,
    }


_OPPONENT_KIND_ORDER = (
    "current_self", "snapshot", "mixed", "api", "heuristic", "unknown")


def _composite_opponent_spec(seat_specs: dict[int, OpponentSpec]) -> OpponentSpec:
    """Collapse per-seat opponents into one table-level telemetry spec."""
    specs = list((seat_specs or {}).values())
    if not specs:
        return OpponentSpec("unknown", "unknown", "unknown")
    kinds = {getattr(s, "kind", "unknown") or "unknown" for s in specs}
    if len(kinds) == 1:
        only = specs[0]
        if only.kind != "snapshot":
            return only
        steps = {getattr(s, "step", None) for s in specs}
        names = {getattr(s, "served_name", None) for s in specs}
        if len(steps) == 1 and len(names) == 1:
            return only
    signature = "+".join(
        sorted(getattr(s, "actor_id", getattr(s, "kind", "unknown"))
               for s in specs)
    )
    return OpponentSpec(
        kind="mixed", served_name=f"mixed:{signature}",
        actor_id=f"mixed:{signature}")


def _iter_opponent_assignments(opp_specs: dict,
                               opp_table: dict | None = None):
    """Yield actual opponent-seat assignments for realized mix accounting."""
    if opp_table:
        for seat_map in opp_table.values():
            for spec in (seat_map or {}).values():
                yield spec
    else:
        for spec in (opp_specs or {}).values():
            yield spec


def _opponent_assignment_mix_stats(opp_specs: dict,
                                   opp_table: dict | None = None) -> dict:
    specs = list(_iter_opponent_assignments(opp_specs, opp_table))
    total = len(specs)
    counts = Counter()
    pinned = 0
    unpinned = 0
    for spec in specs:
        kind = getattr(spec, "kind", "unknown") or "unknown"
        counts[kind] += 1
        if kind == "snapshot":
            if getattr(spec, "step", None) == 0:
                pinned += 1
            else:
                unpinned += 1
    denom = max(1, total)
    fractions = {
        k: counts.get(k, 0) / denom
        for k in _OPPONENT_KIND_ORDER
    }
    fractions["checkpoint"] = fractions.get("snapshot", 0.0)
    return {
        "total_assignments": total,
        "counts": _counter_json(counts),
        "fractions": fractions,
        "snapshot_pinned_count": pinned,
        "snapshot_unpinned_count": unpinned,
        "snapshot_pinned_frac": pinned / denom,
        "snapshot_unpinned_frac": unpinned / denom,
    }


def _per_opponent_kind_stats(
    rows: list[dict], games: list[dict], pairs: list[tuple[int, int]],
    opp_specs: dict, trainable_seat: int,
) -> dict:
    """Per-opponent-kind reward + trainable-seat win rate.

    Apimix's post-mortem (phase3_rl_run_findings Finding 15) showed that
    aggregate per-rollout reward stats can't distinguish "policy lost to
    Opus" from "policy lost to everyone." Without the per-kind split, the
    apimix degradation only manifested as a slow drift in advantages/mean
    that we missed. This function makes the breakdown visible per rollout.

    Returns: {kind -> {n_seeds, n_games, reward_mean, reward_std,
                       group_std_mean, win_rate, n_wins, n_total, ...}}
    where `kind` is "current_self" | "snapshot" | "api" | "heuristic",
    "mixed" for heterogeneous tables, or "unknown".
    """
    from collections import defaultdict as _dd
    import statistics as _st

    # seed → opponent kind ("heuristic"/"snapshot"/"api")
    seed_to_kind: dict[int, str] = {}
    for s, spec in opp_specs.items():
        seed_to_kind[int(s)] = getattr(spec, "kind", "unknown") or "unknown"

    # Reward stats from rows (one per trainable turn). Aggregate to per-game
    # via (group_key, game_id), then per-kind via the seed → kind map.
    per_kind_per_game: dict[str, dict[tuple, float]] = _dd(lambda: _dd(float))
    n_seeds_by_kind: dict[str, set] = _dd(set)
    n_rows_by_kind: dict[str, int] = _dd(int)
    adv_by_kind: dict[str, list[float]] = _dd(list)
    for r in rows:
        seed = _row_seed(r)
        kind = seed_to_kind.get(int(seed) if seed is not None else -1,
                                "unknown")
        n_seeds_by_kind[kind].add(seed)
        n_rows_by_kind[kind] += 1
        adv = _safe_float(r.get("precomputed_advantage"))
        if adv is not None:
            adv_by_kind[kind].append(adv)
        gk = r.get("group_key")
        gid = r.get("game_id")
        try:
            per_kind_per_game[kind][(gk, gid)] += float(
                r.get("precomputed_reward") or 0.0)
        except (TypeError, ValueError):
            continue

    # Trainable-seat win rate + margin from full game outcomes. Tie ⇒ counted
    # as win (matches how the §3.6 eval scores positional outcomes —
    # final_score >= max(others) is a "win"). Margin = trainable - max(others);
    # "HOW BADLY" the policy is losing is more diagnostic than win/loss alone
    # — apimix may have lost MORE points per loss vs strong opponents than
    # repl_03 lost vs Flash, but binary win-rate hides that.
    win_by_kind: dict[str, list[int]] = _dd(lambda: [0, 0])
    margins_by_kind: dict[str, list[float]] = _dd(list)
    for (s, _ki), g in zip(pairs, games):
        kind = seed_to_kind.get(int(s), "unknown")
        finals = ((g.get("final_results") or {}).get("final_scores") or [])
        if not finals:
            continue
        try:
            scores = {int(x["player_id"]): float(x["final_score"])
                      for x in finals if "player_id" in x and "final_score" in x}
        except (TypeError, ValueError, KeyError):
            continue
        if trainable_seat not in scores:
            continue
        max_other = max((v for k, v in scores.items() if k != trainable_seat),
                        default=float("-inf"))
        win_by_kind[kind][1] += 1
        # Codex round-1 (repl_08 v3): use STRICT > to match `_per_snapshot_
        # age_stats` (Codex round-2 fix) and `scripts/eval/_aggregate_qwen_eval.py` panel
        # eval. Ties no longer count as wins — keeps the heuristic-decay WR
        # signal on the same scale as the dual-gate target.
        if scores[trainable_seat] > max_other + 1e-9:
            win_by_kind[kind][0] += 1
        if max_other > float("-inf"):
            margins_by_kind[kind].append(scores[trainable_seat] - max_other)

    out: dict = {}
    all_kinds = set(per_kind_per_game) | set(win_by_kind)
    for kind in all_kinds:
        per_game = per_kind_per_game.get(kind, {})
        rewards = list(per_game.values())
        per_gk: dict = _dd(list)
        for (gk, _), v in per_game.items():
            per_gk[gk].append(v)
        stds = [_st.pstdev(v) if len(v) > 1 else 0.0 for v in per_gk.values()]
        wins, total = win_by_kind.get(kind, [0, 0])
        margins = margins_by_kind.get(kind, [])
        advs = adv_by_kind.get(kind, [])
        adv_mean = _st.fmean(advs) if advs else None
        adv_std = _st.pstdev(advs) if len(advs) > 1 else 0.0
        out[kind] = {
            "n_seeds": len(n_seeds_by_kind.get(kind, set())),
            "n_games": len(per_game),
            "n_rows": n_rows_by_kind.get(kind, 0),
            "reward_mean": (_st.fmean(rewards) if rewards else None),
            "reward_std": (_st.pstdev(rewards) if len(rewards) > 1 else 0.0),
            "group_std_mean": (_st.fmean(stds) if stds else None),
            "win_rate": (wins / total if total else None),
            "n_wins": wins,
            "n_total": total,
            # Margin: signed point gap vs best other. Negative ⇒ losing by
            # this many points on average. Captures "lost by 2 vs lost by 25."
            "margin_mean": (_st.fmean(margins) if margins else None),
            "margin_std": (_st.pstdev(margins) if len(margins) > 1 else 0.0),
            "margin_p50": (_quantile(margins, 0.50) if margins else None),
            "advantage_mean": adv_mean,
            "advantage_std": adv_std,
            "advantage_abs_mean": (
                _st.fmean([abs(a) for a in advs]) if advs else None),
            "advantage_zero_frac": (
                sum(1 for a in advs if abs(a) <= 1e-9) / len(advs)
                if advs else None),
            "advantage_snr_abs_mean_over_std": (
                abs(adv_mean) / adv_std
                if isinstance(adv_mean, (int, float)) and adv_std > 1e-12
                else None),
        }
    return out


def _reward_component_var_by_kind(rows: list[dict], opp_specs: dict) -> dict:
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        seed = _row_seed(r)
        spec = opp_specs.get(int(seed) if seed is not None else -1)
        kind = getattr(spec, "kind", "unknown") if spec is not None else "unknown"
        by_kind[kind or "unknown"].append(r)
    return {
        kind: _reward_component_var_decomposition(rs)
        for kind, rs in by_kind.items()
        if rs
    }


# --------------------------------------------------------------------------- #
# Extra per-roll diagnostics — added 2026-05-26 in response to apimix         #
# post-mortem. Each function is robust to malformed input (telemetry never    #
# aborts a roll); each returns None / {} on failure.                          #
# --------------------------------------------------------------------------- #
def _format_compliance_stats(rows: list[dict]) -> dict:
    """Per-row format-compliance: did the completion parse as valid bid/reveal?

    Mode-collapse / structural regression often shows up here BEFORE it shows
    up in reward — the policy starts emitting malformed outputs that get
    parsed via the brace-match or prose fallback, eventually failing entirely.
    Watch for `valid_rate` dropping or `parse_method_strict_rate` falling
    (more reliance on fallbacks).
    """
    try:
        from megagem.game.actions import (
            PARSE_METHOD_BRACE_MATCH, PARSE_METHOD_NONE,
            parse_bid, parse_reveal,
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"import failed: {type(e).__name__}: {e}"}

    n_total = 0; n_valid = 0; n_strict = 0; n_brace = 0; n_fallback = 0
    by_phase: dict = {"bid": [0, 0, 0], "reveal": [0, 0, 0]}  # [n, valid, strict]
    for r in rows:
        comp = r.get("completion") or ""
        phase = (r.get("phase") or "").lower()
        if not comp or phase not in ("bid", "reveal"):
            continue
        try:
            p = parse_bid(comp) if phase == "bid" else parse_reveal(comp)
        except Exception:  # noqa: BLE001 — bad completion is the diagnostic
            n_total += 1
            by_phase[phase][0] += 1
            continue
        n_total += 1
        by_phase[phase][0] += 1
        if p.valid:
            n_valid += 1
            by_phase[phase][1] += 1
        # parse_method: "" / "json" / "brace_match" / "prose_fallback" / "none"
        pm = getattr(p, "parse_method", "") or ""
        if pm not in (PARSE_METHOD_BRACE_MATCH, PARSE_METHOD_NONE, ""):
            # Strict JSON parse — best path. Anything else means we needed
            # a fallback to extract the action.
            n_strict += 1
            by_phase[phase][2] += 1
        elif pm == PARSE_METHOD_BRACE_MATCH:
            n_brace += 1
        else:
            n_fallback += 1

    if n_total == 0:
        return {"n_total": 0}
    return {
        "n_total": n_total,
        "valid_rate": n_valid / n_total,
        "parse_method_strict_rate": n_strict / n_total,
        "parse_method_brace_rate": n_brace / n_total,
        "parse_method_fallback_rate": n_fallback / n_total,
        "by_phase": {
            ph: {
                "n": v[0],
                "valid_rate": (v[1] / v[0] if v[0] else None),
                "strict_rate": (v[2] / v[0] if v[0] else None),
            } for ph, v in by_phase.items() if v[0] > 0
        },
    }


def _completion_repetition_stats(rows: list[dict],
                                 n_gram: int = 3) -> dict:
    """Trigram-level repetition rate per completion.

    `repetition = 1 - unique_ngrams / total_ngrams`. Pure mode collapse →
    repetition → 1. Healthy policy → repetition < 0.3 ish for trigrams. Watch
    for sudden increases in `repetition_p95` — that's the tail collapsing
    before the median does.
    """
    if n_gram < 2:
        return {"error": "n_gram must be >= 2"}
    reps: list[float] = []
    for r in rows:
        comp = (r.get("completion") or "").split()
        if len(comp) < n_gram + 1:
            continue
        ngrams = [tuple(comp[i:i + n_gram])
                  for i in range(len(comp) - n_gram + 1)]
        if not ngrams:
            continue
        uniq = len(set(ngrams))
        reps.append(1.0 - uniq / len(ngrams))
    if not reps:
        return {"n_completions": 0}
    reps_sorted = sorted(reps)
    n = len(reps_sorted)
    return {
        "n_completions": n,
        "n_gram": n_gram,
        "repetition_mean": sum(reps) / n,
        "repetition_p50": reps_sorted[n // 2],
        "repetition_p95": reps_sorted[min(n - 1, int(0.95 * (n - 1)))],
        "repetition_max": max(reps_sorted),
        "frac_high_rep": sum(1 for v in reps if v > 0.5) / n,  # >50% repeated
    }


def _reward_margin_spearman_per_roll(rows: list[dict], games: list[dict],
                                     pairs: list[tuple[int, int]],
                                     trainable_seat: int) -> dict:
    """Per-K-group Spearman of (reward_sum, true_margin) across the 8 games.

    Adapted from `_kgroup_alignment` (which runs at end-of-run). repl_03's
    end-of-run Spearman was 0.987 — the reward function ranks games almost
    identically to the §3.6 true-margin signal. Per-roll Spearman that
    DROPS would mean the reward design has decoupled from actual winning at
    this slice of the training distribution — a 5-alarm fire that
    advantages/mean alone can't catch.
    """
    from collections import defaultdict as _dd

    # Per-game reward sum (across the trainable seat's turns)
    per_game_reward: dict[tuple, float] = _dd(float)
    for r in rows:
        gk = r.get("group_key"); gid = r.get("game_id")
        try:
            per_game_reward[(gk, gid)] += float(r.get("precomputed_reward") or 0.0)
        except (TypeError, ValueError):
            continue

    # Per-game true margin from finals; map (seed, ki) → game then to game_id
    # via the post_tag_actor_mask game_id convention. The games list is in
    # `pairs` order. game_id is the index in that flat list.
    per_game_margin: dict[int, float] = {}
    for gid, ((_s, _ki), g) in enumerate(zip(pairs, games)):
        finals = ((g.get("final_results") or {}).get("final_scores") or [])
        if not finals:
            continue
        try:
            scores = {int(x["player_id"]): float(x["final_score"])
                      for x in finals if "player_id" in x and "final_score" in x}
        except (TypeError, ValueError, KeyError):
            continue
        if trainable_seat not in scores:
            continue
        max_other = max((v for k, v in scores.items() if k != trainable_seat),
                        default=float("-inf"))
        if max_other > float("-inf"):
            per_game_margin[gid] = scores[trainable_seat] - max_other

    # Group by K-group, compute Spearman per group
    groups: dict = _dd(lambda: {"rewards": [], "margins": []})
    for (gk, gid), rwd in per_game_reward.items():
        if gid in per_game_margin:
            groups[gk]["rewards"].append(rwd)
            groups[gk]["margins"].append(per_game_margin[gid])

    spearmans: list[float] = []
    for gk, v in groups.items():
        if len(v["rewards"]) < 2:
            continue
        sp = _spearman(v["rewards"], v["margins"])
        if sp is not None:
            spearmans.append(sp)
    if not spearmans:
        return {"n_groups": 0}
    return {
        "n_groups": len(spearmans),
        "spearman_mean": sum(spearmans) / len(spearmans),
        "spearman_min": min(spearmans),
        "spearman_p50": _quantile(spearmans, 0.50),
        "spearman_max": max(spearmans),
        "frac_below_0.5": sum(1 for s in spearmans if s < 0.5) / len(spearmans),
    }


def _per_snapshot_age_stats(games: list[dict], pairs: list[tuple[int, int]],
                            opp_specs: dict, trainable_seat: int,
                            current_step: int) -> dict:
    """Win rate per snapshot (and its age in steps).

    Healthy curriculum: policy should beat OLDER snapshots, struggle with
    NEWEST. If the trend reverses (we lose to old snapshots we used to beat),
    the policy is forgetting. The pool pulls snapshots at known steps;
    `age = current_step - snap_step`.
    """
    from collections import defaultdict as _dd
    seed_to_spec = {int(s): spec for s, spec in opp_specs.items()}

    by_step: dict[int, list[int]] = _dd(lambda: [0, 0])  # [wins, total]
    for (s, _ki), g in zip(pairs, games):
        spec = seed_to_spec.get(int(s))
        if not spec or getattr(spec, "kind", None) != "snapshot":
            continue
        step_val = getattr(spec, "step", None)
        if step_val is None:
            continue
        finals = ((g.get("final_results") or {}).get("final_scores") or [])
        if not finals:
            continue
        try:
            scores = {int(x["player_id"]): float(x["final_score"])
                      for x in finals if "player_id" in x and "final_score" in x}
        except (TypeError, ValueError, KeyError):
            continue
        if trainable_seat not in scores:
            continue
        max_other = max((v for k, v in scores.items() if k != trainable_seat),
                        default=float("-inf"))
        by_step[int(step_val)][1] += 1
        # STRICT win (matches `scripts/eval/_aggregate_qwen_eval.py` / panel_eval semantics)
        # so the PFSP feedback win-rate is on the same definition as the
        # dual-gate panel eval. Pre-Codex used `>=` which counted ties as wins
        # ⇒ PFSP marked snapshots ~easier than the spend gate did.
        if scores[trainable_seat] > max_other + 1e-9:
            by_step[int(step_val)][0] += 1

    if not by_step:
        return {"n_snapshots": 0}
    out: dict = {"n_snapshots": len(by_step), "by_step": {}}
    for snap_step, (wins, total) in sorted(by_step.items()):
        age = max(0, int(current_step) - int(snap_step))
        out["by_step"][snap_step] = {
            "age_steps": age,
            "win_rate": (wins / total if total else None),
            "n_wins": wins,
            "n_total": total,
        }
    return out


def _parse_dapo_float(name: str, default: str, *,
                      ge: float | None = None,
                      gt: float | None = None,
                      le: float | None = None,
                      strict: bool = False) -> float:
    """Parse a DAPO env-var with strict range validation.

    Silent fallbacks let nan/inf/negative thresholds silently disable
    filtering or destabilize the EMA. When DAPO is enabled (`strict=True`)
    we abort EARLY on a bad value — better than letting a $50 run train
    with no filtering or a dirty EMA. When DAPO is OFF the env-var is
    parsed but never used, so a bad value gets the silent default (still
    flagged via WARN for visibility).
    """
    raw = os.environ.get(name, default)

    def _fail(reason: str) -> float:
        if strict:
            raise SystemExit(f"[phase3] DAPO config invalid: {reason}")
        print(f"[phase3] WARN DAPO disabled but {reason} ignored "
              f"(falling back to default {default}).", flush=True)
        return float(default)

    try:
        v = float(raw)
    except (TypeError, ValueError):
        return _fail(f"env {name}={raw!r} is not a number")
    if not math.isfinite(v):
        return _fail(f"env {name}={raw!r} is not finite (nan/inf rejected)")
    if ge is not None and v < ge:
        return _fail(f"env {name}={v} below allowed minimum {ge}")
    if gt is not None and v <= gt:
        return _fail(f"env {name}={v} must be > {gt}")
    if le is not None and v > le:
        return _fail(f"env {name}={v} above allowed maximum {le}")
    return v


def _dapo_filter_degenerate_kgroups(
    rows: list[dict], *, opp_specs: dict, opp_ema_std: dict[str, float],
    abs_threshold: float, opp_rel_threshold: float, ema_alpha: float,
    exempt_kinds: frozenset[str] | None = None,
) -> tuple[list[dict], dict]:
    """DAPO dynamic sampling: drop rows from K-groups with near-zero reward
    variance, optionally calibrated per-opponent.

    A K-group whose 8 rollouts' per-game terminal-reward std falls below
    `threshold` carries no useful gradient signal — within-group
    standardization just amplifies noise into ±z. We drop those rows entirely
    (whole K-groups; partial drops would break the standardization math
    upstream — `precomputed_advantage` was finalised over the full group).

    Threshold per group:
      max(abs_threshold, opp_rel_threshold * opp_ema_std[opponent_kind])

    `opp_rel_threshold == 0` ⇒ fixed absolute threshold for every kind (the
    DAPO-paper recipe). `opp_rel_threshold > 0` ⇒ opponent-aware: a flat
    group vs Opus (where flatness is typical) is held to a stricter bar than
    a flat group vs heuristic (where flatness is suspicious).

    `exempt_kinds` (default: {"heuristic"} when caller passes it; None ⇒
    no exemption) lists opponent kinds for which the *relative* threshold is
    skipped — only the absolute floor `abs_threshold` applies. Rationale: for
    stationary deterministic opponents (the scripted heuristic), a low
    within-group std means the policy has reliably *learned to exploit* the
    opponent (the +R signal we want), not a degenerate K-group. The relative
    EMA-based threshold was designed for variable-strength opponents
    (snapshots, APIs) where flatness signals noise; applying it to a
    stationary opponent drops the highest-signal groups in the run. The
    absolute floor still catches truly-zero-std groups (all-identical
    rewards, e.g. all format failures).

    `opp_ema_std` is updated in-place — one EMA tick per K-group's observed
    std. The dict persists across rolls in `_gpu_run`'s scope so the
    per-opponent reference accumulates as training proceeds. EMA seed: first
    observation per kind initialises the EMA at that std (no zero-anchored
    cold-start bias).
    """
    import statistics as _st
    from collections import defaultdict as _dd

    per_game: dict = _dd(float)
    for r in rows:
        try:
            per_game[(r["group_key"], r["game_id"])] += float(
                r.get("precomputed_reward") or 0.0)
        except (TypeError, ValueError):
            continue

    groups: dict = _dd(list)
    group_kind: dict = {}
    for (gk, _gid), v in per_game.items():
        groups[gk].append(v)
        if gk not in group_kind:
            seed = _row_seed({"group_key": gk})
            spec = opp_specs.get(int(seed) if seed is not None else -1)
            group_kind[gk] = getattr(spec, "kind", "unknown") or "unknown"

    group_std: dict = {}
    for gk, vs in groups.items():
        std = _st.pstdev(vs) if len(vs) >= 2 else 0.0
        group_std[gk] = std
        kind = group_kind[gk]
        prev = opp_ema_std.get(kind)
        if prev is None:
            opp_ema_std[kind] = std
        else:
            opp_ema_std[kind] = ema_alpha * prev + (1.0 - ema_alpha) * std

    exempt = exempt_kinds or frozenset()
    degenerate: set = set()
    by_kind_total: dict = _dd(int)
    by_kind_dropped: dict = _dd(int)
    for gk, std in group_std.items():
        kind = group_kind[gk]
        by_kind_total[kind] += 1
        threshold = abs_threshold
        if (
            opp_rel_threshold > 0.0
            and kind in opp_ema_std
            and kind not in exempt
        ):
            threshold = max(threshold, opp_rel_threshold * opp_ema_std[kind])
        if std < threshold:
            degenerate.add(gk)
            by_kind_dropped[kind] += 1

    n_rows_before = len(rows)
    filtered = [r for r in rows if r["group_key"] not in degenerate]
    return filtered, {
        "enabled": True,
        "n_groups_total": len(group_std),
        "n_groups_degenerate": len(degenerate),
        "n_rows_before": n_rows_before,
        "n_rows_after": len(filtered),
        "abs_threshold": abs_threshold,
        "opp_rel_threshold": opp_rel_threshold,
        "ema_alpha": ema_alpha,
        "exempt_kinds": sorted(exempt),
        "by_kind_total": dict(by_kind_total),
        "by_kind_dropped": dict(by_kind_dropped),
        "opp_ema_std": dict(opp_ema_std),
        "group_std_min": (min(group_std.values()) if group_std else None),
        "group_std_mean": (
            sum(group_std.values()) / len(group_std) if group_std else None),
    }


def _format_dapo_block(roll_index: int, dapo: dict) -> str:
    """One-line greppable summary of DAPO dynamic sampling for this roll."""
    if not dapo or not dapo.get("enabled"):
        return ""
    tag = f"roll {int(roll_index):03d}"
    # Surface errors prominently. The roll-level WARN already printed at the
    # call site, but a final summary line here keeps `grep DAPO` complete.
    if "error" in dapo:
        line = (f"  [{tag}] DAPO  ERROR {dapo['error']}  "
                f"fallback_used={dapo.get('fallback_used', False)}")
        if dapo.get("fallback_reason"):
            line += f"  reason={dapo['fallback_reason']}"
        return line
    parts = [
        f"  [{tag}] DAPO  dropped="
        f"{dapo['n_groups_degenerate']}/{dapo['n_groups_total']} groups  "
        f"rows: {dapo['n_rows_before']}→{dapo['n_rows_after']}  "
        f"thr_abs={dapo['abs_threshold']:.4f} "
        f"thr_rel={dapo['opp_rel_threshold']:.3f}"
    ]
    by_kind_total = dapo.get("by_kind_total") or {}
    by_kind_dropped = dapo.get("by_kind_dropped") or {}
    if by_kind_total:
        kparts = []
        for kind in _OPPONENT_KIND_ORDER:
            total = by_kind_total.get(kind, 0)
            if total == 0:
                continue
            dropped = by_kind_dropped.get(kind, 0)
            ema = (dapo.get("opp_ema_std") or {}).get(kind)
            ema_s = f"{ema:.3f}" if isinstance(ema, (int, float)) else "n/a"
            kparts.append(f"{kind}: {dropped}/{total} (ema_std={ema_s})")
        if kparts:
            parts.append(f"  [{tag}] DAPO by-kind  " + "  |  ".join(kparts))
    # Codex alert: with opp_rel_threshold raised (repl_08 default 0.4), the
    # threshold for API/Flash groups scales with the API ema_std — if Flash
    # at T=0 is more consistent than snapshots, API groups may get dropped
    # at a much higher rate, silently undermining the realism anchor. Surface
    # API drop rate > 50% as a WARN line. Threshold floor lowered to
    # `api_total >= 2` (was 4) per Codex's follow-up: at the planned p_api=0.10
    # × num_seeds=24 the expected API K-groups per roll is only ~2.4, so the
    # exact catastrophic case (2/2 or 3/3 dropped) was below the original
    # alert floor.
    api_total = by_kind_total.get("api", 0)
    api_dropped = by_kind_dropped.get("api", 0)
    if api_total >= 2 and api_dropped / api_total > 0.5:
        parts.append(
            f"  [{tag}] DAPO  WARN api_drop_rate="
            f"{api_dropped}/{api_total}="
            f"{100*api_dropped/api_total:.0f}% — opp_rel_threshold may be "
            f"chewing through the Flash realism anchor; consider lowering "
            f"PHASE3_DAPO_OPP_REL_THRESHOLD or raising p_api.")
    # repl_08 v3: same WARN for the heuristic kind. The heuristic gives a
    # +R signal precisely BECAUSE the trainee crushes it — group_std for
    # heuristic K-groups can collapse fast (repl_07 measured 0.46 → 0.09
    # over 400 steps). If `opp_rel_threshold · ema_std` drops below the
    # actual group std, DAPO will filter away the very signal we re-
    # introduced the heuristic to provide. Surface this loudly.
    heur_total = by_kind_total.get("heuristic", 0)
    heur_dropped = by_kind_dropped.get("heuristic", 0)
    if heur_total >= 2 and heur_dropped / heur_total > 0.5:
        parts.append(
            f"  [{tag}] DAPO  WARN heuristic_drop_rate="
            f"{heur_dropped}/{heur_total}="
            f"{100*heur_dropped/heur_total:.0f}% — DAPO is filtering away "
            f"the heuristic learning signal we re-introduced via "
            f"--p-heuristic. Either lower PHASE3_DAPO_OPP_REL_THRESHOLD or "
            f"raise --p-heuristic so K-groups with non-degenerate spread "
            f"survive.")
    if dapo.get("fallback_used"):
        parts.append(f"  [{tag}] DAPO  WARN fallback: "
                     f"{dapo.get('fallback_reason')}")
    return "\n".join(parts)


def _format_extra_diagnostics(roll_index: int, fmt: dict, rep: dict,
                              align: dict, snap: dict) -> str:
    """Compact greppable block summarizing all four post-apimix diagnostics."""
    tag = f"roll {int(roll_index):03d}"
    lines: list[str] = []
    if fmt and "error" not in fmt and fmt.get("n_total", 0) > 0:
        lines.append(
            f"  [{tag}] FORMAT  n={fmt['n_total']} "
            f"valid={fmt['valid_rate']*100:.1f}%  "
            f"strict={fmt['parse_method_strict_rate']*100:.1f}%  "
            f"brace={fmt['parse_method_brace_rate']*100:.1f}%  "
            f"fallback={fmt['parse_method_fallback_rate']*100:.1f}%")
    if rep and "error" not in rep and rep.get("n_completions", 0) > 0:
        lines.append(
            f"  [{tag}] REPETITION  n={rep['n_completions']} "
            f"mean={rep['repetition_mean']:.2f} "
            f"p95={rep['repetition_p95']:.2f} "
            f"frac>0.5={rep['frac_high_rep']*100:.0f}% "
            f"({rep['n_gram']}-gram)")
    if align and align.get("n_groups", 0) > 0:
        lines.append(
            f"  [{tag}] REWARD-MARGIN-ALIGN  K-groups={align['n_groups']} "
            f"spearman mean={align['spearman_mean']:+.3f} "
            f"min={align['spearman_min']:+.3f} "
            f"frac<0.5={align['frac_below_0.5']*100:.0f}%")
    if snap and snap.get("n_snapshots", 0) > 0:
        parts = []
        for step_val, s in snap.get("by_step", {}).items():
            wr = s.get("win_rate")
            wrs = f"{100*wr:.0f}%" if isinstance(wr, (int, float)) else "n/a"
            parts.append(
                f"step{step_val}(age{s['age_steps']}): {wrs} "
                f"({s['n_wins']}/{s['n_total']})")
        if parts:
            lines.append(f"  [{tag}] PER-SNAPSHOT  " + "  |  ".join(parts))
    return "\n".join(lines)


def _format_opp_breakdown(roll_index: int, opp_stats: dict) -> str:
    """One-line greppable summary of per-kind reward + win rate."""
    parts: list[str] = []
    for kind in _OPPONENT_KIND_ORDER:
        s = opp_stats.get(kind)
        if not s or s.get("n_games", 0) == 0:
            continue
        wr = s.get("win_rate")
        wrs = f"{100*wr:>3.0f}%" if isinstance(wr, (int, float)) else "n/a"
        rm = s.get("reward_mean")
        rms = f"{rm:+.3f}" if isinstance(rm, (int, float)) else "n/a"
        parts.append(
            f"{kind}: n_games={s['n_games']:<3} rwd={rms} "
            f"win={wrs} ({s['n_wins']}/{s['n_total']})")
    if not parts:
        return ""
    return (f"  [roll {int(roll_index):03d}] OPP-BREAKDOWN  "
            + "  |  ".join(parts))


def _rollout_diagnostics(full_rows: list[dict], selected_rows: list[dict],
                         games: list[dict], *,
                         trainable_seat: int,
                         opponents_by_seed: dict[int, str] | None = None,
                         tokenizer=None) -> dict:
    full = _rows_diagnostics(full_rows, opponents_by_seed=opponents_by_seed)
    selected = _rows_diagnostics(
        selected_rows, opponents_by_seed=opponents_by_seed)
    full_unique = max(1, int(full["unique_rows"]))
    return {
        "full_rows": full,
        "selected_rows": selected,
        "selection": {
            "selected_unique_fraction": selected["unique_rows"] / full_unique,
            "selected_total_fraction": (
                selected["total_rows"] / full["total_rows"]
                if full["total_rows"] else None
            ),
        },
        "kgroup_alignment": _kgroup_alignment(
            full_rows, games, trainable_seat),
        "selection_bias": _selection_bias_diagnostics(
            full_rows, selected_rows, opponents_by_seed=opponents_by_seed,
            tokenizer=tokenizer),
    }


def _annotate_train_log_with_update_pressure(
    log_history: list[dict], rolls_meta: list[dict], *, kl_beta: float
) -> None:
    """Add geometry/pressure telemetry to TRL's per-step log entries in-place."""
    roll_steps = sorted(
        (int(r.get("step", -1)), r) for r in rolls_meta
        if isinstance(r.get("step"), int))
    if not roll_steps:
        return

    def _roll_for_step(step: int) -> dict | None:
        chosen = None
        for roll_step, meta in roll_steps:
            if roll_step < step:
                chosen = meta
            else:
                break
        return chosen

    for entry in log_history:
        step = entry.get("step")
        if not isinstance(step, int) or "train_runtime" in entry:
            continue
        meta = _roll_for_step(step)
        pressure = (meta or {}).get("update_pressure") or {}
        if pressure:
            entry["megagem/update/full_rows"] = pressure.get("full_rows")
            entry["megagem/update/selected_rows"] = pressure.get("selected_rows")
            entry["megagem/update/unique_selected_rows"] = pressure.get(
                "unique_selected_rows")
            entry["megagem/update/selected_unique_fraction"] = pressure.get(
                "selected_unique_fraction")
            entry["megagem/update/selected_total_fraction"] = pressure.get(
                "selected_total_fraction")
            entry["megagem/update/duplicate_factor_from_num_generations"] = (
                pressure.get("duplicate_factor_from_num_generations"))
            comp_n = _safe_float(entry.get("megagem/completions/n"))
            uniq = _safe_float(pressure.get("unique_selected_rows"))
            if comp_n is not None and uniq and uniq > 0:
                entry["megagem/update/observed_duplicate_factor"] = comp_n / uniq
            bias = pressure.get("selection_bias") or {}
            adv = bias.get("advantage") or {}
            reward = bias.get("reward") or {}
            tok = bias.get("token_weighted") or {}
            neg = bias.get("negative_advantage_frac") or {}
            for key, val in {
                "advantage_full_mean": adv.get("full_mean"),
                "advantage_selected_mean": adv.get("selected_mean"),
                "advantage_selected_minus_full": (
                    adv.get("selected_minus_full")),
                "reward_full_mean": reward.get("full_mean"),
                "reward_selected_mean": reward.get("selected_mean"),
                "reward_selected_minus_full": (
                    reward.get("selected_minus_full")),
                "advantage_token_full_mean": (
                    tok.get("advantage_full_mean")),
                "advantage_token_selected_mean": (
                    tok.get("advantage_selected_mean")),
                "advantage_token_selected_minus_full": (
                    tok.get("advantage_selected_minus_full")),
                "reward_token_selected_minus_full": (
                    tok.get("reward_selected_minus_full")),
                "negative_advantage_frac_full": neg.get("full"),
                "negative_advantage_frac_selected": neg.get("selected"),
                "negative_advantage_frac_selected_minus_full": (
                    neg.get("selected_minus_full")),
                "balanced_select_s": pressure.get("balanced_select_s"),
                "selection_callback_s": pressure.get("selection_callback_s"),
            }.items():
                if isinstance(val, (int, float)):
                    entry[f"megagem/update/{key}"] = val
            for kind, stats in (bias.get("by_opponent") or {}).items():
                if not isinstance(stats, dict):
                    continue
                safe_kind = str(kind).replace("/", "_")
                delta = stats.get("selected_minus_full")
                coverage = stats.get("coverage")
                if isinstance(delta, (int, float)):
                    entry[
                        f"megagem/update/adv_delta_by_opponent/{safe_kind}"
                    ] = delta
                if isinstance(coverage, (int, float)):
                    entry[
                        f"megagem/update/coverage_by_opponent/{safe_kind}"
                    ] = coverage
        kl = _safe_float(entry.get("kl"))
        if kl is not None:
            entry["megagem/update/kl_loss_term_approx"] = float(kl_beta) * kl
        loss = _safe_float(entry.get("loss"))
        kl_term = _safe_float(entry.get("megagem/update/kl_loss_term_approx"))
        if loss is not None and kl_term is not None:
            entry["megagem/update/policy_loss_term_approx"] = loss - kl_term
        # Bucket C#7 (repl_08 v3): β·KL / |loss| — the metric the plan's
        # β=0.015 risk-monitor flagged. >50% means KL term dominates loss
        # (over-regularized); <5% means β is cosmetic for this step. The
        # plan's worry was repl_07's step-183 KL spike (β·KL/|loss|=2.4×);
        # logging the ratio per step gives an early warning before another
        # spike. Guard against division by zero on the rare loss≈0 step.
        if (loss is not None and kl_term is not None
                and abs(loss) > 1e-9):
            entry["megagem/update/beta_kl_ratio"] = kl_term / abs(loss)


def _annotate_train_log_with_timing(
    log_history: list[dict], rolls_meta: list[dict], step_timing: list[dict]
) -> None:
    """Add rollout/selection/trainer timing decomposition to log_history."""
    roll_steps = sorted(
        (int(r.get("step", -1)), r) for r in rolls_meta
        if isinstance(r.get("step"), int))
    step_timing_by_step = {
        int(r["step"]): r for r in step_timing
        if isinstance(r.get("step"), int)
    }

    def _roll_for_step(step: int) -> dict | None:
        chosen = None
        for roll_step, meta in roll_steps:
            if roll_step < step:
                chosen = meta
            else:
                break
        return chosen

    for entry in log_history:
        step = entry.get("step")
        if not isinstance(step, int) or "train_runtime" in entry:
            continue
        meta = _roll_for_step(step)
        timing = (meta or {}).get("timing") or {}
        for key, val in timing.items():
            if isinstance(val, (int, float)):
                entry[f"megagem/timing/{key}"] = val
        pressure = (meta or {}).get("update_pressure") or {}
        for key in ("balanced_select_s", "selection_callback_s"):
            val = pressure.get(key)
            if isinstance(val, (int, float)):
                entry[f"megagem/timing/{key}"] = val
        rec = step_timing_by_step.get(step)
        if rec:
            for key, val in rec.items():
                if key != "step" and isinstance(val, (int, float)):
                    entry[f"megagem/timing/{key}"] = val
        total = _safe_float(entry.get("step_time"))
        if total is not None:
            entry["megagem/timing/step_time_total_s"] = total
            accounted = 0.0
            accounted_any = False
            for key in ("megagem/timing/roll_total_s",
                        "megagem/timing/balanced_select_s",
                        "megagem/timing/trainer_step_s"):
                val = _safe_float(entry.get(key))
                if val is not None:
                    accounted += val
                    accounted_any = True
            if accounted_any:
                entry["megagem/timing/step_time_accounted_s"] = accounted
                entry["megagem/timing/step_time_unaccounted_s"] = (
                    total - accounted)


def _annotate_train_log_with_checkpoints(
    log_history: list[dict], checkpoints: list[dict]
) -> None:
    by_step = {
        int(c["step"]): c for c in checkpoints
        if isinstance(c.get("step"), int)
    }
    for entry in log_history:
        step = entry.get("step")
        if not isinstance(step, int) or step not in by_step:
            continue
        ckpt = by_step[step]
        lora = ckpt.get("lora") or {}
        for key, val in lora.items():
            entry[f"megagem/lora/{key}"] = val
        probe = ckpt.get("probe_logprob") or {}
        for key, val in probe.items():
            entry[f"megagem/probe_logprob/{key}"] = val


# --------------------------------------------------------------------------- #
# GPU run — persistent PEFT model, single trainer + on-policy rollout.        #
# --------------------------------------------------------------------------- #
def make_onpolicy_rollout_func(tok, roll_fn, n, *,
                               system_prompt=H.SYSTEM_PROMPT,
                               selection_callback=None):
    """Dynamic on-policy rollout (replaces 2.3's *static* re-emit).

    A SINGLE persistent trainer drives the whole run, so Adam / LR-scheduler /
    global-step are continuous (the #3 fix). On-policy freshness lives here:
    when TRL asks for a new generation, `roll_fn()` syncs the *live* adapter
    to the rollout vLLM and rolls fresh games from the current policy; within
    one generation the same roll is reused so the §A.7 within-group
    standardization is internally consistent regardless of TRL's micro-chunk
    splitting. Re-roll is triggered when an already-served prompt index
    reappears (a new pass), not by guessing TRL's chunk schedule.

    Exactly `len(prompts)` rows are returned (verified TRL contract); the
    fixed-N dataset is satisfied by rolling ≫N rows and taking the first N —
    each kept row's `precomputed_advantage` was finalised by the one
    `flatten` standardization pass over the FULL set of rolled K-groups
    *before* truncation, so truncation never corrupts a kept row.
    """
    state = {"rows": None, "served": set(), "rolls": 0}

    def _ids(text: str) -> list[int]:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        return ids or [tok.eos_token_id or 0]

    def rollout_func(*a, **kw):
        prompts = kw.get("prompts") or (a[0] if a else None)
        if prompts is None:
            raise RuntimeError("rollout_func could not find 'prompts'")
        idxs = [int(str(p).split("i=")[1]) for p in prompts]
        new_pass = state["rows"] is None or any(
            i in state["served"] for i in idxs)
        if new_pass:
            rows = roll_fn()
            if len(rows) < n:
                raise RuntimeError(
                    f"on-policy roll produced {len(rows)} trainable rows < "
                    f"rows_per_gen={n}; raise --num-seeds/--k or lower "
                    f"--rows-per-gen (never pad — fake rows pollute GRPO).")
            roll_index = state["rolls"]
            select_t0 = time.perf_counter()
            state["rows"] = _balanced_select(rows, n, seed=roll_index)
            selection_s = time.perf_counter() - select_t0
            if selection_callback is not None:
                selection_callback(
                    roll_index, rows, state["rows"],
                    selection_s=selection_s)
            state["served"] = set()
            state["rolls"] += 1
        rows = state["rows"]
        pid, cid, pr, pa = [], [], [], []
        for i in idxs:
            row = rows[i]
            msgs = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["prompt"]}]
            ptext = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            pid.append(_ids(ptext)); cid.append(_ids(row["completion"]))
            pr.append(float(row["precomputed_reward"]))
            pa.append(float(row["precomputed_advantage"]))
            state["served"].add(i)
        assert len(pid) == len(prompts)
        return {"prompt_ids": pid, "completion_ids": cid,
                "logprobs": H.rollout_logprobs(cid),
                "precomputed_reward": pr, "precomputed_advantage": pa}

    return rollout_func, state


def _resolved_vllm_urls(args) -> list[str]:
    """Resolve the vLLM URL list from CLI args. --vllm-urls (comma-sep) wins;
    fall back to --vllm-url (single) wrapped as a one-element list for the
    legacy 2-GPU layout. Always returns a non-empty list or empty if both
    flags are unset (caller validates)."""
    urls_raw = getattr(args, "vllm_urls", None)
    if urls_raw:
        parsed = [u.strip() for u in urls_raw.split(",") if u.strip()]
        if parsed:
            return parsed
    single = getattr(args, "vllm_url", None)
    return [single] if single else []


def _gpu_run(args, tmp: str) -> dict:
    H.print_trl_env()
    vllm_urls = _resolved_vllm_urls(args)
    if not vllm_urls:
        raise SystemExit("non-dry-run needs --vllm-url (or --vllm-urls)")
    if _needs_heuristic_endpoint(args) and not args.heuristic_url:
        raise SystemExit(
            "non-dry-run needs --heuristic-url only for --no-opponent-pool "
            "or explicit --p-heuristic > 0")
    # surface the DP fan-out so it's obvious in the log
    if len(vllm_urls) > 1:
        print(f"[phase3] vLLM data-parallel: {len(vllm_urls)} workers "
              f"({', '.join(vllm_urls)})", flush=True)
    os.environ.setdefault("PHASE3_TERMINAL_CORRECTION", "1")  # locked cut
    # Silence the HF `datasets` per-call tqdm bar ("Map: 100%|████|…") that TRL
    # triggers every roll via its placeholder-dataset map. Modal sets the env
    # var via the subprocess env; this setdefault covers direct (non-Modal)
    # invocations. Idempotent — does nothing if the var is already set or if
    # the user explicitly opted in by exporting it to "0".
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from megagem import endpoints
    from datasets import Dataset
    # Env-var setdefault above only takes effect if `datasets` hadn't been
    # imported yet (its progress-bar flag is read on import). `trl` pulls
    # `datasets` transitively, so by the time we hit this line the module is
    # already loaded and the env var was a no-op. Calling the runtime API
    # forces the flag off regardless of import order. Try the modern top-level
    # name first; fall back to the legacy submodule.
    try:
        import datasets as _ds
        if hasattr(_ds, "disable_progress_bars"):
            _ds.disable_progress_bars()
        else:
            from datasets.utils.logging import (
                disable_progress_bar as _dpb,
            )
            _dpb()
    except Exception:  # noqa: BLE001 — telemetry cleanup never aborts
        pass
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainerCallback,
    )
    from trl import GRPOConfig, GRPOTrainer  # noqa: F401

    from megagem.training.megagem_grpo import (
        MegaGemGRPOTrainer, precomputed_reward_func,
    )

    os.environ.setdefault("EMPTY", "EMPTY")
    if _needs_heuristic_endpoint(args):
        endpoints.ENDPOINTS[HEURISTIC_MODEL] = {
            "model": HEURISTIC_MODEL, "url": args.heuristic_url, "key": "EMPTY"}
    # DP layout: store the full URL list under "url". `pick_url` (used by
    # megagem.rollout) handles both str and list, round-robining the list per
    # request so concurrent rollouts spread across vLLM workers.
    _endpoint_url_value = vllm_urls if len(vllm_urls) > 1 else vllm_urls[0]
    endpoints.ENDPOINTS[ADP.ADAPTER_NAME] = {
        "model": ADP.ADAPTER_NAME, "url": _endpoint_url_value, "key": "EMPTY"}

    # Optional telemetry bundle (vLLM /metrics + nvidia-smi + API status
    # counts). Opt-in via PHASE3_TELEMETRY=1 (modal_train.py sets this on by
    # default). MUST be installed BEFORE any AsyncOpenAI client is constructed
    # — the API tracker monkey-patches `AsyncOpenAI.__init__`, and run_game
    # builds its client cache on first call.
    # Telemetry currently scrapes /metrics from a single vLLM. For DP we pass
    # urls[0] — the local-host detection in the API tracker still classifies
    # all DP URLs as local (both share "localhost"/"127.0.0.1" substring), so
    # the API tally remains accurate. Per-worker /metrics fan-out is a future
    # extension; today we get partial vLLM-server-health visibility (worker 0).
    telemetry = install_from_env(vllm_urls[0])
    if telemetry is not None:
        print("[phase3] telemetry: vLLM /metrics + nvidia-smi + API tally "
              "enabled (PHASE3_TELEMETRY=1)", flush=True)
    # DAPO dynamic sampling (Finding 17 mitigation) — opt-in via
    # PHASE3_DAPO_DYNAMIC_SAMPLING=1. Drops K-groups whose 8-rollout
    # reward std < threshold so the optimizer never sees degenerate-group
    # "least bad of all losers" gradients. `opp_rel_threshold` > 0 makes
    # the per-group bar opponent-aware via a running per-kind EMA of std.
    dapo_enabled = os.environ.get(
        "PHASE3_DAPO_DYNAMIC_SAMPLING", "0") == "1"
    # abs threshold: must be > 0 (0 would never drop anything; negative is
    # nonsense — drops would be empty by construction).
    dapo_abs_threshold = _parse_dapo_float(
        "PHASE3_DAPO_MIN_GROUP_STD", "1e-3",
        gt=0.0, strict=dapo_enabled)
    # opp_rel: 0 disables the per-opponent scaling; negative is nonsense.
    dapo_opp_rel = _parse_dapo_float(
        "PHASE3_DAPO_OPP_REL_THRESHOLD", "0.0",
        ge=0.0, strict=dapo_enabled)
    # ema_alpha in [0, 1). alpha=1 freezes the EMA at its seed value;
    # alpha=0 disables EMA memory (each roll = its own observation).
    dapo_ema_alpha = _parse_dapo_float(
        "PHASE3_DAPO_EMA_ALPHA", "0.9",
        ge=0.0, le=0.9999, strict=dapo_enabled)
    # Heuristic-exempt (default ON): stationary deterministic opponents
    # produce low within-group std *because the policy has learned to
    # exploit them* (the +R signal). The relative EMA threshold was designed
    # for variable-strength opponents; applying it to heuristic drops the
    # highest-signal groups in the run. Absolute floor still applies.
    dapo_heuristic_exempt = os.environ.get(
        "PHASE3_DAPO_HEURISTIC_EXEMPT", "1") == "1"
    dapo_exempt_kinds = (
        frozenset({"heuristic"}) if dapo_heuristic_exempt else frozenset()
    )
    # State that persists across rolls — per-opponent-kind EMA of within-K-group
    # reward std. Used both as DAPO threshold reference (when opp_rel>0) and
    # as standalone telemetry.
    opp_ema_std: dict[str, float] = {}
    if dapo_enabled:
        print(f"[phase3] DAPO dynamic sampling: ON  "
              f"abs_threshold={dapo_abs_threshold} "
              f"opp_rel_threshold={dapo_opp_rel} "
              f"ema_alpha={dapo_ema_alpha} "
              f"exempt_kinds={sorted(dapo_exempt_kinds)}", flush=True)
    # Per-rollout health monitor (Finding 15 mitigation) — WARN-only by
    # default; PHASE3_HEALTH_ABORT=1 enables auto-kill after CONSEC_BAD bad
    # rolls. See RolloutHealthMonitor docstring for tunable thresholds.
    health_monitor = install_health_monitor_from_env()
    if health_monitor is not None:
        ab = "ABORT-on-threshold" if health_monitor.abort_on_threshold else "WARN-only"
        print(f"[phase3] health-monitor: enabled ({ab}); thresholds: "
              f"adv_neg<{health_monitor.adv_neg_threshold} "
              f"reward_drop>{health_monitor.reward_drop_pct*100:.0f}% "
              f"neg_roll>{health_monitor.neg_roll_fraction*100:.0f}% "
              f"reward_floor<{health_monitor.min_reward_floor} "
              f"consec_bad={health_monitor.consec_bad_for_abort}",
              flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    initial_seeds = _train_seeds_for_roll(
        args.seed_start, args.num_seeds, 0, fixed=args.fixed_train_seeds)
    reward_cfg = H.reward_config_from_env()  # 1a ON via env
    n = int(args.rows_per_gen)
    _shape = _spg_shape(
        n, args.num_generations,
        num_processes=args.num_processes,
        grad_accum=args.gradient_accumulation_steps,
        on_policy=args.on_policy)
    g, micro, spg = _shape["g"], _shape["micro"], _shape["spg"]
    ga, np_ = _shape["ga"], _shape["num_processes"]
    if not _shape["divisible"]:
        raise SystemExit(
            f"[phase3] batch shape not divisible: gen_batch={_shape['gen_batch']} "
            f"is not a multiple of micro({micro})×num_processes({np_}). "
            f"Adjust --rows-per-gen / PHASE2_MICRO_CAP / --num-processes.")
    if np_ > 1 and os.environ.get("PHASE3_ALLOW_DDP") != "1":
        raise SystemExit(
            f"[phase3] --num-processes={np_} enables rank-sharded DDP "
            "(rank-0-generates + file handoff; single-node only). This path is "
            "BUILT but not yet validated on a multi-GPU box — set "
            "PHASE3_ALLOW_DDP=1 for the first validation smoke and confirm: "
            "(1) generation happens ONCE (rank-0 logs the roll; ranks>0 don't), "
            "(2) loss/KL curve start matches np=1, (3) effective_batch=micro×np×ga "
            "as reported. The recommended 7gen+1train config needs no DDP.")
    print(f"[phase3] batch shape: micro={micro} np={np_} ga={ga} spg={spg} "
          f"opt_steps/gen={_shape['opt_steps_per_gen']} "
          f"effective_batch={_shape['effective_batch']} "
          f"on_policy={_shape['opt_steps_per_gen'] == 1}", flush=True)

    # ---- §3.3 lagged-self opponent pool ---------------------------------- #
    # Snapshots are LoRA adapters over the SAME base, served by the one rollout
    # vLLM (--max-loras) — the pool costs ~no extra GPU. API opponents (empty
    # by default ⇒ lagged-self only) resolve through Prime Inference.
    api_models = [m.strip() for m in (args.opp_api_models or "").split(",")
                  if m.strip()]
    # Preflight the API-opponent credential BEFORE training — else a sampled
    # API opponent crashes the rollout mid-run (run_game's sys.exit) after the
    # GPU spend. Mirrors phase3_eval's non-heuristic preflight.
    if api_models and not os.getenv(endpoints.PRIME_KEY):
        raise SystemExit(
            f"--opp-api-models set ({api_models}) but ${endpoints.PRIME_KEY} "
            f"is not in the environment — a drawn API opponent would crash "
            f"the rollout mid-training. Export {endpoints.PRIME_KEY} or drop "
            f"--opp-api-models.")
    for m in api_models:
        endpoints.ENDPOINTS.setdefault(m, {
            "model": m, "url": endpoints.PRIME_URL, "key": endpoints.PRIME_KEY})
    # Optional --opp-api-weights: comma-sep ints aligned 1:1 with api_models.
    # Empty ⇒ uniform (back-compat with repl_03 etc.).
    api_weights_raw = (args.opp_api_weights or "").strip()
    if api_weights_raw:
        try:
            api_weights = [int(w.strip()) for w in api_weights_raw.split(",")]
        except ValueError as e:
            raise SystemExit(
                f"--opp-api-weights must be comma-separated ints, got "
                f"{api_weights_raw!r}: {e}")
        if len(api_weights) != len(api_models):
            raise SystemExit(
                f"--opp-api-weights has {len(api_weights)} entries but "
                f"--opp-api-models has {len(api_models)} — counts must match.")
    else:
        api_weights = None
    pool = None
    if args.opponent_pool:
        # repl_08: scripted heuristic removed from training pool BY DEFAULT.
        # Pinned step_0 anchor is added AFTER trainer.model exists (so we can
        # save its adapter as the snapshot) — see _seed_step0_anchor() below.
        # heuristic_spec stays None ⇒ pool raises loud if it somehow ends up
        # empty (instead of silently falling back to the very opponent we
        # removed).
        # repl_08 v3: `--p-heuristic > 0` re-introduces the heuristic at the
        # specified probability WITH internal `(1-WR)²` decay (see
        # `megagem.training.opponent_pool._p_heuristic_effective`). Diagnosis behind this
        # knob: seam_smoke_02 (heuristic OFF) gave null learning signal in
        # 70 steps; this is the design fix.
        _pool_heuristic_spec = (
            _HEURISTIC_SPEC if args.p_heuristic > 0.0 else None)
        pool = OpponentPool(
            pinned_snapshots=[],
            max_snapshots=args.max_snapshots,
            anneal_start=args.opp_anneal_start,
            anneal_end=args.opp_anneal_end,
            p_max=args.opp_anneal_pmax,
            api_specs=[OpponentSpec(
                kind="api", served_name=m,
                actor_id="api_" + m.replace("/", "_")) for m in api_models],
            api_weights=api_weights,
            p_api=args.opp_api_prob,
            current_self_spec=_current_self_spec(),
            p_current_self=args.p_current_self,
            rng_seed=args.opp_pool_seed,
            heuristic_spec=_pool_heuristic_spec,
            # Codex fix: without an explicit floor, once the trainee crushes
            # the step_0 anchor (WR≈0.95 ⇒ f_hard floored to 0.01), the anchor
            # gets ≈1% of draws — barely anchoring. 0.15 reserves 15% of
            # snapshot draws for the pinned set (uniform pick across pinned)
            # so the anchor keeps providing a stationary signal.
            p_anchor_floor=args.opp_anchor_floor,
            p_heuristic=args.p_heuristic,
            heuristic_anneal_end=args.heuristic_anneal_end,
        )
    snapshot_events: list[dict] = []

    # ---- on-policy roll closure: sync LIVE adapter → vLLM, then roll ------ #
    holder = {"model": None}  # set to trainer.model after construction
    rolls_meta: list[dict] = []
    roll_context: dict[int, dict] = {}
    _ddp_roll = [0]  # per-roll counter, incremented in lockstep on every rank
    # M3 — carries the previous roll's transferable bucket-vector across rolls
    # so each roll can report a roll-to-roll self-consistency cosine.
    transferable_state: dict = {"vector": None}
    # Plan §D.1: completion-length stats on roll 0 — at that point LoRA B≡0
    # ⇒ the adapter is behaviourally the SFT base ⇒ this IS the SFT-baseline
    # length distribution for *this* run, populated for free from the rollout
    # we already do. Compared against later steps' completions/mean_length so
    # "did RL collapse the length?" is a quantitative question, not a vibe.
    sft_baseline_lengths: dict = {}

    def _kgroup_reward_stats(rows: list[dict]) -> dict:
        """Honest §A.7 K-group reward std on the FULL roll output.

        Aggregates ``precomputed_reward`` per ``(group_key, game_id)`` — that
        per-rollout sum is the value GRPO is actually trying to differentiate
        between K rollouts at the same seed/seat. Then groups those per-game
        scalars by ``group_key`` (size ≈ K=8) and reports within-K-group std.
        Computed BEFORE ``_balanced_select`` so it sees every game, not the
        shuffled subset; computed at the roll level (not per-step) so the
        K-group definition is what §A.7 says it is, not whatever TRL
        reshapes by ``num_generations``.
        """
        import statistics as _st
        from collections import defaultdict as _dd
        per_game: dict = _dd(float)
        groups: dict = _dd(list)
        n_turns = 0
        for r in rows:
            try:
                gk = r["group_key"]
                gid = r["game_id"]
                rv = float(r["precomputed_reward"])
            except (KeyError, TypeError, ValueError):
                continue
            per_game[(gk, gid)] += rv
            n_turns += 1
        for (gk, _gid), val in per_game.items():
            groups[gk].append(val)
        sizes = [len(v) for v in groups.values()]
        stds = [
            _st.pstdev(v) if len(v) > 1 else 0.0
            for v in groups.values()
        ]
        zero_thr = 1e-6
        n_zero = sum(1 for s in stds if s < zero_thr)
        n_groups = len(groups)
        # M2 — decompose the within-group variance into reward-component shares
        # (does λ-shaping contribute to the gradient, or is it cosmetic?).
        # Defensive: a decomposition failure must NOT cost us the std stats.
        try:
            comp_decomp = _reward_component_var_decomposition(rows)
        except Exception as e:  # noqa: BLE001 — telemetry never aborts
            comp_decomp = {"error": f"{type(e).__name__}: {e}"}
        return {
            "n_groups": n_groups,
            "n_games": len(per_game),
            "n_trainable_turns": n_turns,
            "group_size_mean": (sum(sizes) / n_groups) if n_groups else None,
            "group_size_min": (min(sizes) if sizes else None),
            "group_size_max": (max(sizes) if sizes else None),
            "group_std_mean": (sum(stds) / n_groups) if n_groups else None,
            "group_std_min": (min(stds) if stds else None),
            "group_std_max": (max(stds) if stds else None),
            "frac_group_zero_std": (n_zero / n_groups) if n_groups else None,
            # Per-rollout reward distribution across ALL K-groups — gives a
            # sense of the dynamic range; std≈0 here would indicate the
            # reward signal collapsed altogether (not just within a group).
            "per_game_reward_mean": (sum(per_game.values()) / len(per_game))
                if per_game else None,
            "per_game_reward_std": (_st.pstdev(list(per_game.values()))
                                    if len(per_game) > 1 else 0.0),
            "component_var_decomposition": comp_decomp,
        }

    def _length_stats(rows: list[dict]) -> dict:
        import statistics as _st
        chars = [len(r["completion"]) for r in rows if r.get("completion")]
        tok_lens: list[int] = []
        for r in rows:
            c = r.get("completion") or ""
            if not c:
                continue
            try:
                tok_lens.append(len(tok(c, add_special_tokens=False)["input_ids"]))
            except Exception:  # noqa: BLE001
                continue
        def _p(xs: list[int], q: float) -> float | None:
            if not xs:
                return None
            xs2 = sorted(xs)
            return float(xs2[min(len(xs2) - 1, max(0, int(q * (len(xs2) - 1))))])
        return {
            "n_completions": len(chars),
            "chars": {
                "mean": (sum(chars) / len(chars)) if chars else None,
                "median": (float(_st.median(chars)) if chars else None),
                "p95": _p(chars, 0.95),
                "min": (min(chars) if chars else None),
                "max": (max(chars) if chars else None),
            },
            "tokens": {
                "mean": (sum(tok_lens) / len(tok_lens)) if tok_lens else None,
                "median": (float(_st.median(tok_lens)) if tok_lens else None),
                "p95": _p(tok_lens, 0.95),
                "min": (min(tok_lens) if tok_lens else None),
                "max": (max(tok_lens) if tok_lens else None),
            },
        }

    def _trainable_param_snapshot(model_obj) -> dict[str, object]:
        snap = {}
        try:
            for name, p in model_obj.named_parameters():
                if getattr(p, "requires_grad", False):
                    snap[name] = p.detach().float().cpu().clone()
        except Exception:  # noqa: BLE001 — diagnostics only
            return {}
        return snap

    def _lora_parameter_stats(model_obj, ref: dict[str, object]) -> dict:
        total_sq = torch.zeros((), device="cpu")
        delta_sq = torch.zeros((), device="cpu")
        n_params = 0
        n_tensors = 0
        try:
            for name, p in model_obj.named_parameters():
                if not getattr(p, "requires_grad", False):
                    continue
                cur = p.detach().float().cpu()
                total_sq = total_sq + (cur * cur).sum()
                n_params += int(cur.numel())
                n_tensors += 1
                if name in ref:
                    d = cur - ref[name]
                    delta_sq = delta_sq + (d * d).sum()
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}
        return {
            "trainable_tensors": n_tensors,
            "trainable_params": n_params,
            "norm": float(torch.sqrt(total_sq).item()),
            "delta_norm_from_step0": float(torch.sqrt(delta_sq).item()),
        }

    def _probe_logprob_mean(model_obj, rows: list[dict], *,
                            max_examples: int = 4,
                            max_prompt_tokens: int = 1024,
                            max_completion_tokens: int = 64) -> float | None:
        probe = [r for r in rows if r.get("prompt") and r.get("completion")]
        probe = probe[:max_examples]
        if not probe:
            return None
        was_training = bool(getattr(model_obj, "training", False))
        vals: list[float] = []
        try:
            model_obj.eval()
            with torch.no_grad():
                device = next(model_obj.parameters()).device
                for row in probe:
                    msgs = [
                        {"role": "system", "content": H.SYSTEM_PROMPT},
                        {"role": "user", "content": row["prompt"]},
                    ]
                    ptext = tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                    pids = tok(ptext, add_special_tokens=False)["input_ids"]
                    cids = tok(
                        row["completion"], add_special_tokens=False
                    )["input_ids"]
                    if not pids or not cids:
                        continue
                    pids = pids[-max_prompt_tokens:]
                    cids = cids[:max_completion_tokens]
                    ids = pids + cids
                    if len(ids) < 2:
                        continue
                    input_ids = torch.tensor(
                        [ids], device=device, dtype=torch.long)
                    logits = model_obj(input_ids=input_ids).logits
                    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
                    target = input_ids[:, 1:]
                    gathered = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                    start = max(0, len(pids) - 1)
                    comp_logp = gathered[0, start:start + len(cids)]
                    if comp_logp.numel():
                        vals.append(float(comp_logp.mean().item()))
        except Exception:  # noqa: BLE001 — diagnostics only
            return None
        finally:
            if was_training:
                model_obj.train()
        return statistics.fmean(vals) if vals else None

    probe_state = {
        "rows": None,
        "ref_mean_logprob": None,
    }

    def _probe_checkpoint_stats() -> dict | None:
        rows = probe_state.get("rows")
        ref = probe_state.get("ref_mean_logprob")
        if not rows or ref is None:
            return None
        cur = _probe_logprob_mean(holder["model"], rows)
        if cur is None:
            return None
        return {
            "n_examples": len(rows),
            "mean_logprob": cur,
            "delta_from_step0": cur - float(ref),
            "ref_step0_mean_logprob": float(ref),
        }

    def _seat_for_roll(roll_index: int) -> int:
        """The trainable seat for this roll (delegates to the pure module-level
        `_rotated_seat`). With --rotate-seats, round-robin across all seats —
        constant WITHIN the roll, hence constant within each K-group, so §A.7
        within-group standardization is unaffected; else the fixed
        --trainable-seat. Rotating trains the policy from every seat, which the
        TrueSkill/panel eval (seat0/1/2 rotation) requires — training only seat
        0 leaves seats 1/2 off-distribution."""
        return _rotated_seat(
            roll_index, base_seat=args.trainable_seat,
            num_players=args.num_players,
            rotate=bool(getattr(args, "rotate_seats", False)))

    def roll_fn() -> list[dict]:
        # DDP: counter incremented identically on ALL ranks (roll_fn is called in
        # lockstep). Ranks>0 do NOT generate or push adapters — they receive
        # rank 0's full rows (advantages finalised over the complete K-groups)
        # and train their DistributedSampler shard. Inert when world_size==1.
        _ridx = _ddp_roll[0]
        _ddp_roll[0] += 1
        _rank, _world = _ddp_rank_world()
        if _world > 1 and _rank != 0:
            return _ddp_recv_rows(tmp, _ridx)
        t0 = time.perf_counter()
        timing: dict[str, float] = {}
        step = int(getattr(trainer.state, "global_step", 0))
        roll_index = len(rolls_meta)
        roll_seat = _seat_for_roll(roll_index)
        roll_seeds = _train_seeds_for_roll(
            args.seed_start, args.num_seeds, roll_index,
            fixed=args.fixed_train_seeds)
        _t = time.perf_counter()
        apath = ADP.save_step_adapter(
            holder["model"], roll_index, args.adapter_root)
        timing["adapter_save_s"] = time.perf_counter() - _t
        _t = time.perf_counter()
        sync = ADP.push_adapter_to_vllm_all(
            vllm_urls, ADP.ADAPTER_NAME, apath)
        timing["adapter_push_s"] = time.perf_counter() - _t
        # HARD ABORT on a failed push. Continuing would roll against whatever
        # vLLM currently serves (stale/base weights) → off-policy data while
        # GRPO stats look healthy — a silent correctness hole. For this first
        # GPU pass an unsynced adapter is fatal, not a "reported condition".
        if not sync.get("ok"):
            # Surface per-worker diagnostics so the operator can see which DP
            # worker(s) failed; print before raising in case downstream logging
            # truncates the exception message.
            per_url = sync.get("per_url", [])
            print(
                "[phase3] adapter→vLLM sync FAILED. Per-worker results:",
                flush=True)
            for entry in per_url:
                print(
                    f"  url={entry.get('url')} ok={entry.get('ok')} "
                    f"load_status={entry.get('load_status')} "
                    f"load_body={(entry.get('load_body') or '')[:200]!r}",
                    flush=True)
            raise RuntimeError(
                f"adapter→vLLM sync FAILED (load_status="
                f"{sync.get('load_status')}, body={sync.get('load_body')!r}, "
                f"per_url_failures="
                f"{[e.get('url') for e in per_url if not e.get('ok')]}); "
                f"aborting so we never train on stale/base weights. "
                f"{sync.get('fallback_hint') or ''}")
        # HETEROGENEOUS-table mode (opt-in, --hetero-opponents): each
        # non-trainable seat draws its OWN opponent from the pool, so the trainee
        # faces two DIFFERENT opponents instead of two clones (the PSRO/league
        # lever; every prior run used homogeneous tables). The two draws use
        # seat-perturbed seeds (distinct primes) so they are independent yet
        # deterministic, and held constant across the seed's K-group (§A.7).
        # opp_specs stays a representative {seed: spec} for the telemetry-only
        # post-roll diagnostics (no PFSP feedback path consumes it).
        hetero = bool(getattr(args, "hetero_opponents", False)) and pool is not None
        opp_table: dict | None = None
        if pool is None:
            opp_specs = {s: _HEURISTIC_SPEC for s in roll_seeds}
        elif hetero:
            _nt = [p for p in range(args.num_players) if p != roll_seat]
            opp_table = {
                s: {seat: pool.draw(step, s * 7919 + seat * 104729) for seat in _nt}
                for s in roll_seeds
            }
            opp_specs = {
                s: _composite_opponent_spec(opp_table[s])
                for s in roll_seeds
            }
            # Visibility: confirm heterogeneous tables actually materialise (a
            # seed whose non-trainable seats drew DIFFERENT served models). Roll 0
            # only — directly verifies the lever in the log, no per-game dump.
            _het_n = sum(
                1 for s in roll_seeds
                if len({opp_table[s][seat].served_name for seat in _nt}) > 1)
            if roll_index == 0:
                _ex = {s: [opp_table[s][seat].actor_id for seat in _nt]
                       for s in list(roll_seeds)[:4]}
                print(f"  [hetero] roll0: {_het_n}/{len(roll_seeds)} tables have "
                      f"≥2 DISTINCT opponents; sample(seat opp ids)={_ex}", flush=True)
        else:
            opp_specs = {s: pool.draw(step, s) for s in roll_seeds}
        assignment_mix = _opponent_assignment_mix_stats(opp_specs, opp_table)
        if (pool is not None and args.p_heuristic <= 0.0
                and int((assignment_mix.get("counts") or {}).get(
                    "heuristic", 0)) > 0):
            raise RuntimeError(
                "no-heuristic invariant violated: heuristic opponent was "
                f"drawn with --p-heuristic={args.p_heuristic}. mix="
                f"{assignment_mix}")
        games_for_diag: list[dict] = []
        _t = time.perf_counter()
        rows = _roll_onpolicy(
            trainable_model=ADP.ADAPTER_NAME, seeds=roll_seeds, k=args.k,
            num_players=args.num_players, value_chart=args.value_chart,
            trainable_seat=roll_seat, tmp_dir=tmp,
            sampling=A7_SAMPLING, max_parallel=args.max_parallel,
            dump_dir=args.dump_rollouts or None, roll_index=roll_index,
            opponent_for_seat=(None if hetero else (lambda s: opp_specs[s])),
            opponents_for_table=((lambda s: opp_table[s]) if hetero else None),
            games_out=games_for_diag,
            telemetry=telemetry)
        timing["game_rollout_s"] = time.perf_counter() - _t
        post_t0 = time.perf_counter()
        # Capture SFT-baseline length stats on the FIRST roll, before any
        # optimizer step has fired (LoRA B≡0 ⇒ behaviourally the SFT base).
        # Costs nothing extra — the rollout already ran.
        if not sft_baseline_lengths:
            try:
                sft_baseline_lengths.update(_length_stats(rows))
                sft_baseline_lengths["step"] = step
                sft_baseline_lengths["note"] = (
                    "Roll 0 — LoRA B≡0 ⇒ SFT-base behaviour. Compare against "
                    "later steps' completions/mean_length in train_log.json "
                    "to detect length collapse / runaway.")
                # Wandb (Plan §A.5) — push the SFT-baseline token-length
                # stats as run-level `summary` scalars (wandb's idiom for
                # one-shot, captured-once-at-start values). Lives next to
                # eval/* summary keys for quick inspection.
                if os.environ.get("WANDB_API_KEY"):
                    try:
                        import wandb
                        if wandb.run is not None:
                            tok_stats = sft_baseline_lengths.get("tokens") or {}
                            for k, v in tok_stats.items():
                                if isinstance(v, (int, float)):
                                    wandb.summary[
                                        f"megagem/sft_baseline/tokens/{k}"
                                    ] = v
                    except Exception:  # noqa: BLE001 — never aborts a roll
                        pass
            except Exception as e:  # noqa: BLE001 — telemetry never aborts
                sft_baseline_lengths["error"] = (
                    f"{type(e).__name__}: {e}")
        # §A.7 K-group reward stats — the HONEST replacement for TRL's
        # frac_reward_zero_std (which reshapes by num_generations onto our
        # heterogeneous flattened rows and is uninterpretable). Computed on
        # the FULL roll output before _balanced_select, so each K-group sees
        # all K rollouts at one seed/seat as §A.7 defines.
        try:
            kstats = _kgroup_reward_stats(rows)
        except Exception as e:  # noqa: BLE001 — telemetry never aborts
            kstats = {"error": f"{type(e).__name__}: {e}"}
        if "error" not in kstats:
            print(f"  [roll {roll_index:03d}] K-group reward stats: "
                  f"n_groups={kstats.get('n_groups')} "
                  f"size={kstats.get('group_size_min')}-{kstats.get('group_size_max')} "
                  f"group_std mean={kstats.get('group_std_mean'):.4f} "
                  f"min={kstats.get('group_std_min'):.4f}  "
                  f"frac_zero={kstats.get('frac_group_zero_std'):.3f}",
                  flush=True)
            # Wandb (Plan §A.5) — push the scalar fields of kstats as a
            # per-step time series under the megagem/kgroup/ namespace. Keyed
            # to `step` (trainer.state.global_step at the time of the roll),
            # which aligns these panels with TRL's per-step metrics already
            # logged via report_to=["wandb"].
            if os.environ.get("WANDB_API_KEY"):
                scalar_kstats = {
                    f"megagem/kgroup/{k}": v
                    for k, v in kstats.items()
                    if isinstance(v, (int, float))
                }
                # M2 — nested var-share dict isn't a top-level scalar, so flatten
                # the per-component shares explicitly (the headline is `shaping`).
                _cvd = kstats.get("component_var_decomposition") or {}
                for _c, _v in (_cvd.get("within_group_var_share") or {}).items():
                    if isinstance(_v, (int, float)):
                        scalar_kstats[f"megagem/kgroup/var_share/{_c}"] = _v
                _wandb_log_megagem(scalar_kstats, step)
        # Per-opponent-kind breakdown (Finding 15 mitigation) — splits
        # reward + win rate + margin by heuristic/snapshot/api. Without this,
        # aggregate per_game_reward_mean can't tell "policy lost to Opus"
        # from "policy lost to everyone." Failure-cheap: telemetry only.
        try:
            pairs = [(s, ki) for s in roll_seeds for ki in range(args.k)]
            opp_stats = _per_opponent_kind_stats(
                rows, games_for_diag, pairs, opp_specs, roll_seat)
        except Exception as e:  # noqa: BLE001 — telemetry never aborts
            opp_stats = {"error": f"{type(e).__name__}: {e}"}
        if isinstance(opp_stats, dict) and "error" not in opp_stats:
            block = _format_opp_breakdown(roll_index, opp_stats)
            if block:
                print(block, flush=True)
            # Wandb: per-kind scalars for cross-roll panels.
            if os.environ.get("WANDB_API_KEY"):
                flat = {}
                for kind, s in opp_stats.items():
                    for k, v in s.items():
                        if isinstance(v, (int, float)):
                            flat[f"megagem/opp/{kind}/{k}"] = v
                _wandb_log_megagem(flat, step)
        try:
            reward_component_var_by_kind = _reward_component_var_by_kind(
                rows, opp_specs)
        except Exception as e:  # noqa: BLE001 — telemetry never aborts
            reward_component_var_by_kind = {"error": f"{type(e).__name__}: {e}"}
        if (isinstance(reward_component_var_by_kind, dict)
                and "error" not in reward_component_var_by_kind
                and os.environ.get("WANDB_API_KEY")):
            flat = {}
            for kind, stats in reward_component_var_by_kind.items():
                shares = (stats or {}).get("within_group_var_share") or {}
                for comp, v in shares.items():
                    if isinstance(v, (int, float)):
                        flat[
                            f"megagem/opp/{kind}/reward_var_share/{comp}"
                        ] = v
                v = (stats or {}).get("mean_within_group_var_total")
                if isinstance(v, (int, float)):
                    flat[
                        f"megagem/opp/{kind}/mean_within_group_var_total"
                    ] = v
            if flat:
                _wandb_log_megagem(flat, step)
        opponent_gap_stats: dict = {}
        if isinstance(opp_stats, dict) and "error" not in opp_stats:
            cs = opp_stats.get("current_self") or {}
            ckpt = opp_stats.get("snapshot") or {}
            cs_margin = cs.get("margin_mean")
            ckpt_margin = ckpt.get("margin_mean")
            cs_reward = cs.get("reward_mean")
            ckpt_reward = ckpt.get("reward_mean")
            if isinstance(cs_margin, (int, float)) and isinstance(
                    ckpt_margin, (int, float)):
                opponent_gap_stats[
                    "current_self_minus_checkpoint_margin_mean"
                ] = cs_margin - ckpt_margin
            if isinstance(cs_reward, (int, float)) and isinstance(
                    ckpt_reward, (int, float)):
                opponent_gap_stats[
                    "current_self_minus_checkpoint_reward_mean"
                ] = cs_reward - ckpt_reward
            if opponent_gap_stats and os.environ.get("WANDB_API_KEY"):
                _wandb_log_megagem({
                    f"megagem/opp_gap/{k}": v
                    for k, v in opponent_gap_stats.items()
                    if isinstance(v, (int, float))
                }, step)

        # Repl_08 v3 diagnostic metrics (Bucket A + B + C#8) — directly back
        # the seam's PASS criteria R1-R3. Failure-cheap: a missing field
        # never aborts a roll. All metrics share the per-roll `step` axis
        # used by the rest of the megagem/* namespace.
        if os.environ.get("WANDB_API_KEY"):
            mix_flat: dict = {}
            # ---- Bucket A: realized vs configured opponent mix --------- #
            # A#1: realized fractions. Count actual opponent-seat
            # assignments so heterogeneous tables report the true per-seat
            # 80/20 mix instead of only the table-level composite kind.
            try:
                fractions = assignment_mix.get("fractions") or {}
                for k, v in fractions.items():
                    if isinstance(v, (int, float)):
                        mix_flat[f"megagem/opp_mix/realized/{k}_frac"] = v
                for k in ("snapshot_pinned_frac", "snapshot_unpinned_frac"):
                    v = assignment_mix.get(k)
                    if isinstance(v, (int, float)):
                        mix_flat[f"megagem/opp_mix/realized/{k}"] = v
                mix_flat["megagem/opp_mix/realized/total_assignments"] = (
                    assignment_mix.get("total_assignments", 0))
            except Exception:  # noqa: BLE001
                pass
            # A#2 + B#3-5 + C#8: pool-side telemetry. `pool` is None when
            # --no-opponent-pool is set (legacy path); skip the whole
            # block in that case so no spurious None values appear.
            if pool is not None:
                try:
                    pt = pool.telemetry(int(step))
                    # A#2: configured marginal probabilities the pool's
                    # mix-gate intends to deliver. Side-by-side with the
                    # realized fractions above = the visual R1/R2 gate.
                    rmp = pt.get("realized_marginal_p") or {}
                    for k, v in rmp.items():
                        if isinstance(v, (int, float)):
                            mix_flat[f"megagem/opp_mix/configured/{k}_p"] = v
                            realized = (
                                assignment_mix.get("fractions") or {}).get(k)
                            if isinstance(realized, (int, float)):
                                mix_flat[
                                    f"megagem/opp_mix/error/{k}_frac_minus_p"
                                ] = realized - v
                    # B#3-5: heuristic decay state. Only emitted when
                    # p_heuristic > 0 (legacy/pure-self-play runs skip).
                    p_heur_eff = pt.get("p_heuristic_effective")
                    if isinstance(p_heur_eff, (int, float)):
                        mix_flat["megagem/pool/p_heuristic_effective"] = (
                            p_heur_eff)
                    p_heur_cfg = pt.get("p_heuristic")
                    if isinstance(p_heur_cfg, (int, float)):
                        mix_flat["megagem/pool/p_heuristic_configured"] = (
                            p_heur_cfg)
                    hw = pt.get("heuristic_winrate") or {}
                    if isinstance(hw.get("win_rate"), (int, float)):
                        mix_flat["megagem/pool/heuristic_winrate_ewma"] = (
                            hw["win_rate"])
                    if isinstance(hw.get("n_games"), int):
                        mix_flat["megagem/pool/cumulative_heur_games"] = (
                            hw["n_games"])
                    # C#8: anchor realized fraction. Count seeds whose
                    # opponent is a SNAPSHOT with step == 0 (the pinned
                    # anchor). Validates that p_anchor_floor (0.15) is
                    # actually firing once PFSP would otherwise drown the
                    # anchor. Denominator is snapshot-only seeds (not all)
                    # so the metric is interpretable as "fraction of
                    # snapshot draws that hit the anchor".
                    try:
                        snap_specs = [
                            sp for sp in _iter_opponent_assignments(
                                opp_specs, opp_table)
                            if getattr(sp, "kind", None) == "snapshot"]
                        n_snap = len(snap_specs)
                        n_anchor = sum(
                            1 for sp in snap_specs
                            if getattr(sp, "step", None) == 0)
                        if n_snap > 0:
                            mix_flat[
                                "megagem/pool/anchor_realized_frac"
                            ] = n_anchor / n_snap
                    except Exception:  # noqa: BLE001
                        pass
                except Exception:  # noqa: BLE001
                    pass
            if mix_flat:
                _wandb_log_megagem(mix_flat, step)

        # DAPO dynamic sampling — drop rows from degenerate K-groups BEFORE
        # they reach `make_onpolicy_rollout_func` / `_balanced_select`. The
        # rows that get filtered out are the ones whose K-group's 8 rollouts
        # produced near-identical terminal rewards (the "all-lost-the-same /
        # all-won-the-same" pathology that turns within-group standardization
        # into noise amplification). Filtering is a pure subset — the
        # surviving rows keep their original `precomputed_advantage` from the
        # full-group standardization pass; no recompute. Falls back to the
        # unfiltered rows if the filter would push us below `rows_per_gen`
        # (rather than crashing — better a noisy step than a dead run).
        dapo_stats: dict = {"enabled": False}
        if dapo_enabled:
            try:
                rows_filtered, dapo_stats = _dapo_filter_degenerate_kgroups(
                    rows, opp_specs=opp_specs, opp_ema_std=opp_ema_std,
                    abs_threshold=dapo_abs_threshold,
                    opp_rel_threshold=dapo_opp_rel,
                    ema_alpha=dapo_ema_alpha,
                    exempt_kinds=dapo_exempt_kinds)
            except Exception as e:  # noqa: BLE001 — never abort a roll
                # An exception inside the filter (parse error on a row, missing
                # opp_specs key, etc.) is a code-path bug, not a noisy data
                # condition. We still keep the run alive (this is opt-in
                # mitigation, not a correctness invariant), but we make damn
                # sure the operator sees it: clear WARN line + traceback +
                # fallback_used=True so the JSON record matches the console.
                tb = traceback.format_exc()
                print(f"  [roll {int(roll_index):03d}] DAPO  WARN filter "
                      f"raised {type(e).__name__}: {e} — using unfiltered "
                      f"rows ({len(rows)}) to keep training alive. This is "
                      f"a code bug, not a data condition; fix before next "
                      f"run.\n{tb}", flush=True)
                dapo_stats = {
                    "enabled": True,
                    "error": f"{type(e).__name__}: {e}",
                    "fallback_used": True,
                    "fallback_reason": (
                        f"filter raised {type(e).__name__}: {e}"),
                    "traceback": tb,
                }
                rows_filtered = rows
            if isinstance(dapo_stats, dict) and "error" not in dapo_stats:
                if len(rows_filtered) >= n:
                    rows = rows_filtered
                else:
                    dapo_stats["fallback_used"] = True
                    dapo_stats["fallback_reason"] = (
                        f"filtered_rows={len(rows_filtered)} < "
                        f"rows_per_gen={n} — using unfiltered "
                        f"({len(rows)} rows) to keep training alive.")
            dapo_block = _format_dapo_block(roll_index, dapo_stats)
            if dapo_block:
                print(dapo_block, flush=True)
            # Wandb push for DAPO scalars.
            if os.environ.get("WANDB_API_KEY"):
                flat: dict = {}
                for k, v in dapo_stats.items():
                    if isinstance(v, (int, float, bool)):
                        flat[f"megagem/dapo/{k}"] = v
                for kind, ema in (
                        dapo_stats.get("opp_ema_std") or {}).items():
                    if isinstance(ema, (int, float)):
                        flat[f"megagem/dapo/opp_ema_std/{kind}"] = ema
                for kind, n_drop in (
                        dapo_stats.get("by_kind_dropped") or {}
                        ).items():
                    if isinstance(n_drop, int):
                        flat[f"megagem/dapo/by_kind_dropped/{kind}"] = (
                            n_drop)
                # Bucket C#6 (repl_08 v3): per-kind drop RATIO. The DAPO
                # WARN line fires above 0.5; charting the ratio shows
                # trends approaching that floor before the print alert
                # crosses it. Computed from existing dropped+total counts.
                by_kind_total = dapo_stats.get("by_kind_total") or {}
                by_kind_dropped = dapo_stats.get("by_kind_dropped") or {}
                for kind in _OPPONENT_KIND_ORDER:
                    total = by_kind_total.get(kind, 0)
                    if isinstance(total, int) and total > 0:
                        dropped = by_kind_dropped.get(kind, 0) or 0
                        flat[
                            f"megagem/dapo/by_kind_drop_rate/{kind}"
                        ] = dropped / total
                _wandb_log_megagem(flat, step)

        # Extra diagnostics added 2026-05-26 (post-apimix). Each is failure-
        # cheap; pass any errors through to rolls_meta but never abort.
        try:
            fmt_stats = _format_compliance_stats(rows)
        except Exception as e:  # noqa: BLE001
            fmt_stats = {"error": f"{type(e).__name__}: {e}"}
        try:
            rep_stats = _completion_repetition_stats(rows, n_gram=3)
        except Exception as e:  # noqa: BLE001
            rep_stats = {"error": f"{type(e).__name__}: {e}"}
        try:
            pairs = pairs if 'pairs' in dir() else [
                (s, ki) for s in roll_seeds for ki in range(args.k)]
            align_stats = _reward_margin_spearman_per_roll(
                rows, games_for_diag, pairs, roll_seat)
        except Exception as e:  # noqa: BLE001
            align_stats = {"error": f"{type(e).__name__}: {e}"}
        # M3 — transferable-signal probe (does the advantage point a consistent
        # direction across disjoint boards?). Threads the bucket-vector across
        # rolls for the roll-to-roll self-consistency cosine. Failure-cheap.
        try:
            transferable_signal = _transferable_signal_probe(
                rows, prev_vector=transferable_state.get("vector"),
                seat_vectors=transferable_state.get("seat_vectors"))
            if "error" not in transferable_signal:
                transferable_state["vector"] = transferable_signal.get("vector")
                _accumulate_seat_vector(transferable_state, transferable_signal)
        except Exception as e:  # noqa: BLE001 — telemetry never aborts
            transferable_signal = {"error": f"{type(e).__name__}: {e}"}
        try:
            snap_stats = _per_snapshot_age_stats(
                games_for_diag, pairs, opp_specs,
                roll_seat, current_step=step)
        except Exception as e:  # noqa: BLE001
            snap_stats = {"error": f"{type(e).__name__}: {e}"}
        # repl_08: pipe per-snapshot WR back into the pool so PFSP weighting
        # uses fresh observations. Trainee's win rate vs each snapshot —
        # AlphaStar `f_hard(x)=(1-x)^2` weighting becomes active once any
        # snapshot has >=10 games observed. Failure-cheap: never aborts a roll.
        if (pool is not None and isinstance(snap_stats, dict)
                and "error" not in snap_stats):
            try:
                for snap_step_val, sv in (snap_stats.get("by_step") or {}).items():
                    if not isinstance(sv, dict):
                        continue
                    wr = sv.get("win_rate")
                    n_total = sv.get("n_total")
                    if isinstance(wr, (int, float)) and isinstance(n_total, int):
                        pool.update_snapshot_winrate(
                            int(snap_step_val), float(wr), int(n_total))
            except Exception as e:  # noqa: BLE001 — telemetry never aborts
                print(f"  [roll {roll_index:03d}] WARN pool WR pipe-back "
                      f"failed: {type(e).__name__}: {e}", flush=True)
        # repl_08 v3: pipe heuristic WR back into the pool so the
        # `(1-WR)²` decay knows how dominant the trainee is. Same failure-
        # cheap pattern as snapshots — telemetry must never abort a roll.
        if (pool is not None and pool.p_heuristic > 0.0
                and isinstance(opp_stats, dict)):
            try:
                heur_s = opp_stats.get("heuristic") or {}
                wr_h = heur_s.get("win_rate")
                n_h = heur_s.get("n_total")
                if (isinstance(wr_h, (int, float))
                        and isinstance(n_h, int) and n_h > 0):
                    pool.update_heuristic_winrate(float(wr_h), int(n_h))
            except Exception as e:  # noqa: BLE001
                print(f"  [roll {roll_index:03d}] WARN heuristic WR pipe-back "
                      f"failed: {type(e).__name__}: {e}", flush=True)
        extra_block = _format_extra_diagnostics(
            roll_index, fmt_stats, rep_stats, align_stats, snap_stats)
        if extra_block:
            print(extra_block, flush=True)
        # Wandb push for the extras.
        if os.environ.get("WANDB_API_KEY"):
            flat: dict = {}
            for k, v in (fmt_stats or {}).items():
                if isinstance(v, (int, float)):
                    flat[f"megagem/format/{k}"] = v
            for k, v in (rep_stats or {}).items():
                if isinstance(v, (int, float)):
                    flat[f"megagem/repetition/{k}"] = v
            for k, v in (align_stats or {}).items():
                if isinstance(v, (int, float)):
                    flat[f"megagem/reward_align/{k}"] = v
            for k, v in (transferable_signal or {}).items():
                if isinstance(v, (int, float)):
                    flat[f"megagem/transferable/{k}"] = v
            # Snapshot per-step win rate
            for snap_step_val, sv in (snap_stats.get("by_step")
                                      if isinstance(snap_stats, dict)
                                      else {}).items() or {}.items():
                wr = sv.get("win_rate") if isinstance(sv, dict) else None
                if isinstance(wr, (int, float)):
                    flat[f"megagem/snap_winrate/step_{snap_step_val}"] = wr
                    flat[f"megagem/snap_age/step_{snap_step_val}"] = (
                        sv.get("age_steps", 0))
            _wandb_log_megagem(flat, step)

        # Health monitor (Finding 15 mitigation) — reads the latest TRL log
        # entries to get the most recent advantages/mean + train reward,
        # combines with this roll's per_game_reward_mean, fires WARN messages
        # if any threshold trips. Auto-abort is opt-in via
        # PHASE3_HEALTH_ABORT=1 (default: WARN-only).
        if health_monitor is not None:
            adv_mean = None
            train_rwd = None
            try:
                recent = list(getattr(
                    trainer.state, "log_history", []) or [])[-6:]
                adv_vals = [e["megagem/advantages/mean"] for e in recent
                            if isinstance(e.get("megagem/advantages/mean"),
                                          (int, float))]
                rwd_vals = [e["rewards/precomputed_reward_func/mean"]
                            for e in recent
                            if isinstance(
                                e.get("rewards/precomputed_reward_func/mean"),
                                (int, float))]
                if adv_vals:
                    adv_mean = statistics.fmean(adv_vals)
                if rwd_vals:
                    train_rwd = statistics.fmean(rwd_vals)
            except Exception:  # noqa: BLE001 — telemetry never aborts
                pass
            per_game_rwd = (kstats.get("per_game_reward_mean")
                            if isinstance(kstats, dict) else None)
            alerts = health_monitor.record(
                roll_index, step,
                per_game_reward_mean=per_game_rwd,
                advantages_mean=adv_mean,
                train_reward_mean=train_rwd,
                extras={"opp_stats_keys": list(
                    opp_stats.keys() if isinstance(opp_stats, dict) else [])})
            health_block = RolloutHealthMonitor.format_block(roll_index, alerts)
            if health_block:
                print(health_block, flush=True)
            if alerts.get("should_abort"):
                # Persist what we know before raising so the run JSON is still
                # readable. rolls_meta gets the partial roll appended below.
                raise RuntimeError(
                    f"RolloutHealthMonitor: aborting at roll {roll_index} "
                    f"(step {step}); "
                    f"consec_alert_rolls={alerts['consec_alert_rolls']}; "
                    f"reasons: {alerts['abort']}")

        # Trainer-process GPU0 mem accounting: torch.cuda's view is the
        # authoritative OOM-trigger signal (nvidia-smi can miss bursts between
        # 1Hz ticks AND can't isolate the trainer's allocation from other
        # tenants — though under our --split-gpus layout GPU0 is trainer-only).
        # `max_memory_allocated` is the all-time peak since process start;
        # rising over rolls implies a leak. `mem_get_info` is the live driver-
        # level free/total. WARN threshold from env (default 92%, same as the
        # telemetry alert).
        try:
            # torch is already imported at the top of _gpu_run (closure scope)
            _peak_alloc_gib = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
            _cur_alloc_gib = torch.cuda.memory_allocated(0) / (1024 ** 3)
            _reserved_gib = torch.cuda.memory_reserved(0) / (1024 ** 3)
            _free_b, _total_b = torch.cuda.mem_get_info(0)
            _free_gib = _free_b / (1024 ** 3)
            _total_gib = _total_b / (1024 ** 3)
            _used_pct = (1 - _free_b / _total_b) * 100.0 if _total_b else 0.0
            trainer_mem_gib: dict = {
                "peak_allocated": round(_peak_alloc_gib, 2),
                "current_allocated": round(_cur_alloc_gib, 2),
                "reserved": round(_reserved_gib, 2),
                "free": round(_free_gib, 2),
                "total": round(_total_gib, 2),
                "used_pct": round(_used_pct, 1),
            }
            try:
                _alert_pct = float(os.environ.get(
                    "PHASE3_GPU_MEM_ALERT_PCT", "92"))
            except (TypeError, ValueError):
                _alert_pct = 92.0
            _alert = _used_pct >= _alert_pct
            trainer_mem_gib["alert"] = _alert
            _alert_tag = " ⚠ ALERT" if _alert else ""
            print(
                f"  [roll {roll_index:03d}] TRAINER-MEM  "
                f"peak_alloc={_peak_alloc_gib:.1f}G  "
                f"cur_alloc={_cur_alloc_gib:.1f}G  "
                f"reserved={_reserved_gib:.1f}G  "
                f"free={_free_gib:.1f}/{_total_gib:.1f}G "
                f"(used={_used_pct:.1f}%){_alert_tag}",
                flush=True)
            if _alert:
                print(
                    f"  [roll {roll_index:03d}] TRAINER-MEM  WARN: gpu0 used "
                    f"{_used_pct:.1f}% ≥ alert={_alert_pct:.0f}%. OOM risk "
                    f"climbing — reduce micro_cap / k / num_seeds, or accept "
                    f"the risk and raise PHASE3_GPU_MEM_ALERT_PCT.",
                    flush=True)
        except Exception as e:  # noqa: BLE001 — telemetry never aborts
            trainer_mem_gib = {"error": f"{type(e).__name__}: {e}"}

        timing["postprocess_s"] = time.perf_counter() - post_t0
        timing["roll_total_s"] = time.perf_counter() - t0
        timing = {k: round(v, 4) for k, v in timing.items()}

        meta = {
            "roll": roll_index, "rows": len(rows), "step": step,
            "seeds": roll_seeds, "trainable_seat": roll_seat,
            "roll_s": round(timing["roll_total_s"], 2),
            "timing": timing,
            "adapter_sync_ok": sync.get("ok"),
            "adapter_sync_per_url": sync.get("per_url"),
            "trainer_mem_gib": trainer_mem_gib,
            "opponent_assignment_mix": assignment_mix,
            "kgroup_reward_stats": kstats,
            "per_opponent_stats": opp_stats,
            "opponent_gap_stats": opponent_gap_stats,
            "reward_component_var_by_kind": reward_component_var_by_kind,
            "format_compliance": fmt_stats,
            "completion_repetition": rep_stats,
            "reward_margin_alignment": align_stats,
            "transferable_signal": transferable_signal,
            "per_snapshot_age": snap_stats,
            "dapo_dynamic_sampling": dapo_stats,
        }
        meta["opponents"] = {s: opp_specs[s].served_name for s in roll_seeds}
        meta["opponent_kinds"] = {s: opp_specs[s].kind for s in roll_seeds}
        if opp_table:
            meta["opponent_table"] = {
                s: {seat: spec.served_name for seat, spec in opp_table[s].items()}
                for s in roll_seeds
            }
            meta["opponent_table_kinds"] = {
                s: {seat: spec.kind for seat, spec in opp_table[s].items()}
                for s in roll_seeds
            }
        roll_context[roll_index] = {
            "games": games_for_diag,
            "opponents": meta["opponents"],
            "opponent_kinds": meta["opponent_kinds"],
        }
        if pool is not None:
            meta["pool"] = pool.telemetry(step)
        rolls_meta.append(meta)
        if _world > 1:  # rank 0: publish the full roll for the other ranks
            _ddp_send_rows(tmp, _ridx, rows)
        return rows

    def _selection_callback(roll_index: int, full_rows: list[dict],
                            selected_rows: list[dict], *,
                            selection_s: float | None = None) -> None:
        if _ddp_rank_world()[0] != 0:
            return  # DDP: diagnostics + probe-logprob only on rank 0 (ranks>0
            #         have a stub rolls_meta/roll_context; their output is unsaved)
        cb_t0 = time.perf_counter()
        try:
            ctx = roll_context.get(roll_index, {})
            diag = _rollout_diagnostics(
                full_rows, selected_rows, ctx.get("games") or [],
                trainable_seat=_seat_for_roll(roll_index),
                opponents_by_seed=ctx.get("opponent_kinds") or {},
                tokenizer=tok)
            selection_bias = diag.get("selection_bias") or {}
            update_pressure = {
                "full_rows": diag["full_rows"]["total_rows"],
                "selected_rows": diag["selected_rows"]["total_rows"],
                "unique_selected_rows": diag["selected_rows"]["unique_rows"],
                "selected_unique_fraction": (
                    diag["selection"]["selected_unique_fraction"]),
                "selected_total_fraction": (
                    diag["selection"]["selected_total_fraction"]),
                "num_generations": g,
                "expected_trainer_rows": (
                    diag["selected_rows"]["total_rows"] * g),
                "duplicate_factor_from_num_generations": g,
                "balanced_select_s": (
                    round(float(selection_s), 4)
                    if selection_s is not None else None),
                "selection_bias": selection_bias,
            }
            sel_frac = diag["selection"]["selected_total_fraction"]
            if isinstance(sel_frac, (int, float)) and sel_frac < 0.25:
                update_pressure["selected_fraction_warn"] = (
                    "selected_total_fraction < 0.25; rollout coverage is "
                    "being discarded aggressively before the optimizer.")
                print(
                    f"  [roll {roll_index:03d}] UPDATE-PRESSURE  WARN "
                    f"selected_total_fraction={sel_frac:.3f} < 0.25 "
                    f"(full_rows={diag['full_rows']['total_rows']} "
                    f"selected_rows={diag['selected_rows']['total_rows']})",
                    flush=True)
            bias_block = _format_selection_bias_block(
                roll_index, selection_bias)
            if bias_block:
                print(bias_block, flush=True)
            if os.environ.get("WANDB_API_KEY"):
                flat: dict = {}
                adv = selection_bias.get("advantage") or {}
                reward = selection_bias.get("reward") or {}
                tok_bias = selection_bias.get("token_weighted") or {}
                neg = selection_bias.get("negative_advantage_frac") or {}
                for key, val in {
                    "advantage_full_mean": adv.get("full_mean"),
                    "advantage_selected_mean": adv.get("selected_mean"),
                    "advantage_selected_minus_full": (
                        adv.get("selected_minus_full")),
                    "reward_selected_minus_full": (
                        reward.get("selected_minus_full")),
                    "advantage_token_selected_minus_full": (
                        tok_bias.get("advantage_selected_minus_full")),
                    "negative_advantage_frac_selected_minus_full": (
                        neg.get("selected_minus_full")),
                    "balanced_select_s": update_pressure.get(
                        "balanced_select_s"),
                }.items():
                    if isinstance(val, (int, float)):
                        flat[f"megagem/selection_bias/{key}"] = val
                for kind, stats in (
                        selection_bias.get("by_opponent") or {}).items():
                    if not isinstance(stats, dict):
                        continue
                    safe_kind = str(kind).replace("/", "_")
                    delta = stats.get("selected_minus_full")
                    coverage = stats.get("coverage")
                    if isinstance(delta, (int, float)):
                        flat[
                            f"megagem/selection_bias/by_opponent/"
                            f"{safe_kind}/adv_delta"
                        ] = delta
                    if isinstance(coverage, (int, float)):
                        flat[
                            f"megagem/selection_bias/by_opponent/"
                            f"{safe_kind}/coverage"
                        ] = coverage
                _wandb_log_megagem(flat, int(
                    rolls_meta[roll_index].get("step", roll_index)
                    if roll_index < len(rolls_meta) else roll_index))
            update_pressure["selection_callback_s"] = round(
                time.perf_counter() - cb_t0, 4)
            rolls_meta[roll_index]["rollout_diagnostics"] = diag
            rolls_meta[roll_index]["update_pressure"] = update_pressure
            if probe_state["rows"] is None:
                probe_rows = list(selected_rows[:4])
                probe_state["rows"] = probe_rows
                probe_state["ref_mean_logprob"] = _probe_logprob_mean(
                    holder["model"], probe_rows)
        except Exception as e:  # noqa: BLE001 — telemetry never aborts training
            if roll_index < len(rolls_meta):
                rolls_meta[roll_index]["rollout_diagnostics"] = {
                    "error": f"{type(e).__name__}: {e}"}
        finally:
            roll_context.pop(roll_index, None)

    rollout_func, roll_state = make_onpolicy_rollout_func(
        tok, roll_fn, n, selection_callback=_selection_callback)

    # spg / micro / g from the shared _spg_shape — single source of truth with
    # the --dry-run pre-check. spg is the on-policy freshness cadence (reported,
    # not hidden).
    dataset = Dataset.from_dict({"prompt": [f"[mg-row] i={i}"
                                            for i in range(n)]})
    # LR scheduler (Plan §C). The default `linear` decays to ~0 over `max_steps`
    # — for a 200-step run that means the back half trains at LR < 1e-7,
    # silently destroying most of the compute (obs_smoke_01 showed KL crawled
    # to 0.001 vs the 0.5 budget). Cosine half-cycle keeps LR ≈ peak/2 at the
    # midpoint and decays smoothly. A `min_lr` floor is desirable but the
    # kwarg is version-dependent on the installed `transformers`; we set
    # `num_cycles=0.5` (half-cosine peak→0) unconditionally and let TRL's
    # `_filter_kwargs` drop anything the dataclass rejects.
    _cfg_kwargs = dict(
        output_dir=tmp, per_device_train_batch_size=micro,
        num_generations=g, steps_per_generation=spg, num_iterations=1,
        gradient_accumulation_steps=ga,  # ga>1 ⇒ large on-policy batch (see _spg_shape)
        max_completion_length=args.max_completion_length,
        max_prompt_length=8192, max_steps=args.steps,  # ONE continuous run
        learning_rate=args.learning_rate, beta=args.kl_beta,
        # `cosine_with_min_lr` (NOT plain `cosine`) is the SchedulerType that
        # actually honors `min_lr_rate` — it routes to
        # `get_cosine_with_min_lr_schedule_with_warmup` which accepts the kwarg.
        # Plain `cosine` routes to `get_cosine_schedule_with_warmup` and silently
        # drops/rejects `min_lr_rate`, leaving us with the repl_01 LR=6.17e-10
        # end-state. `min_lr_rate=0.1` floors decay at 10% of peak LR — at
        # the 2e-5 default that's 2e-6, three orders above repl_01's floor.
        # Smoke acceptance #4 verifies last-step LR >= 1e-6 on a 12-step run.
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"num_cycles": 0.5, "min_lr_rate": 0.1},
        temperature=1.0, top_p=0.95, logging_steps=1, save_strategy="no",
        # wandb (Plan §A.4) — TRL hands `_metrics[mode][key]` straight to
        # wandb when "wandb" is in `report_to`. Conditional on the env so
        # non-wandb launches stay quiet (image-default WANDB_MODE=disabled).
        report_to=(["wandb"] if os.environ.get("WANDB_API_KEY") else []),
        use_vllm=False, bf16=True, fp16=False,
        seed=args.seed, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    # Clip-higher (DAPO) is OPT-IN — insert the epsilon kwargs ONLY when set so
    # an unset knob keeps TRL's dataclass defaults (symmetric 0.2). Passing
    # epsilon=None would corrupt TRL's symmetric-fallback resolution.
    if args.epsilon is not None:
        _cfg_kwargs["epsilon"] = args.epsilon
    if args.epsilon_high is not None:
        _cfg_kwargs["epsilon_high"] = args.epsilon_high
    cfg = GRPOConfig(**H._filter_kwargs(GRPOConfig, _cfg_kwargs))

    ckpts: list[dict] = []
    step_timing: list[dict] = []
    init_lora_state: dict[str, object] = {}

    class _Ckpt(TrainerCallback):
        """Adapter checkpoints DURING the single continuous run (the only
        place a mid-training policy exists — there are no segments)."""
        def on_step_end(self, a, st, ctrl, **kw):
            if _ddp_rank_world()[0] != 0:
                return  # DDP: rank 0 owns the canonical (DDP-synced) adapter
            s = st.global_step
            if s > 0 and s % args.checkpoint_every == 0:
                rec = {"step": s, "path": ADP.save_step_adapter(
                    holder["model"], s, args.adapter_root)}
                rec["lora"] = _lora_parameter_stats(
                    holder["model"], init_lora_state)
                probe = _probe_checkpoint_stats()
                if probe is not None:
                    rec["probe_logprob"] = probe
                ckpts.append(rec)

    class _Snapshot(TrainerCallback):
        """§3.3 lagged-self snapshots — every `--snapshot-every` steps freeze
        the live adapter, push it to the rollout vLLM under an IMMUTABLE name,
        register it in the pool + ENDPOINTS, and evict+unload the oldest.
        Distinct from _Ckpt (the §3.6 eval checkpoints) so the two cadences
        are independent. A failed push is NON-fatal (unlike the trainable-sync
        hard abort): a missing snapshot only shrinks the pool — no
        stale-weights correctness hole."""
        def on_step_end(self, a, st, ctrl, **kw):
            if _ddp_rank_world()[0] != 0:
                return  # DDP: only rank 0 saves/pushes snapshots (avoid races)
            s = st.global_step
            if pool is None or s <= 0 or s % args.snapshot_every != 0:
                return
            path = ADP.save_snapshot_adapter(
                holder["model"], s, args.adapter_root)
            name = ADP.snapshot_served_name(s)
            sync = ADP.push_adapter_to_vllm_all(vllm_urls, name, path)
            if not sync.get("ok"):
                # DP fan-out: push_adapter_to_vllm_all has already rolled back
                # any URLs that succeeded, so no half-loaded leak. Persist
                # per-worker diagnostics so the operator can see which DP
                # worker(s) refused the snapshot.
                snapshot_events.append({
                    "step": s, "ok": False,
                    "load_status": sync.get("load_status"),
                    "load_body": sync.get("load_body"),
                    "per_url": sync.get("per_url")})
                return
            endpoints.ENDPOINTS[name] = {
                "model": name, "url": _endpoint_url_value, "key": "EMPTY"}
            evicted = pool.add_snapshot(Snapshot(
                step=s, served_name=name, adapter_path=path))
            unload_result = None
            if evicted is not None:
                unload_result = ADP.unload_adapter_from_vllm_all(
                    vllm_urls, evicted.served_name)
                endpoints.ENDPOINTS.pop(evicted.served_name, None)
            snapshot_events.append({
                "step": s, "ok": True, "name": name,
                "evicted": evicted.served_name if evicted else None,
                "per_url": sync.get("per_url"),
                "evict_per_url": (unload_result.get("per_url")
                                  if unload_result else None)})

    class _StepTiming(TrainerCallback):
        """Decompose wall time into rollout, selection, and train-step pieces."""
        def __init__(self):
            self._t0: float | None = None
            self._printed: set[int] = set()

        def _roll_for_step(self, step: int) -> dict | None:
            chosen = None
            for meta in rolls_meta:
                roll_step = meta.get("step")
                if isinstance(roll_step, int) and roll_step < step:
                    chosen = meta
            return chosen

        def _step_rec(self, step: int) -> dict | None:
            for rec in reversed(step_timing):
                if rec.get("step") == step:
                    return rec
            return None

        def on_step_begin(self, a, st, ctrl, **kw):
            if _ddp_rank_world()[0] != 0:
                return
            self._t0 = time.perf_counter()

        def on_step_end(self, a, st, ctrl, **kw):
            if _ddp_rank_world()[0] != 0 or self._t0 is None:
                return
            step = int(st.global_step)
            step_timing.append({
                "step": step,
                "trainer_step_s": round(time.perf_counter() - self._t0, 4),
            })
            self._t0 = None

        def on_log(self, a, st, ctrl, logs=None, **kw):
            if _ddp_rank_world()[0] != 0 or not isinstance(logs, dict):
                return
            step = int(st.global_step)
            meta = self._roll_for_step(step)
            timing = (meta or {}).get("timing") or {}
            pressure = (meta or {}).get("update_pressure") or {}
            for key, val in timing.items():
                if isinstance(val, (int, float)):
                    logs[f"megagem/timing/{key}"] = val
            for key in ("balanced_select_s", "selection_callback_s"):
                val = pressure.get(key)
                if isinstance(val, (int, float)):
                    logs[f"megagem/timing/{key}"] = val
            rec = self._step_rec(step)
            if rec:
                for key, val in rec.items():
                    if key != "step" and isinstance(val, (int, float)):
                        logs[f"megagem/timing/{key}"] = val
            total = _safe_float(logs.get("step_time"))
            if total is not None:
                logs["megagem/timing/step_time_total_s"] = total
                accounted = 0.0
                accounted_any = False
                for key in ("megagem/timing/roll_total_s",
                            "megagem/timing/balanced_select_s",
                            "megagem/timing/trainer_step_s"):
                    val = _safe_float(logs.get(key))
                    if val is not None:
                        accounted += val
                        accounted_any = True
                if accounted_any:
                    logs["megagem/timing/step_time_accounted_s"] = accounted
                    logs["megagem/timing/step_time_unaccounted_s"] = (
                        total - accounted)
            if step in self._printed or total is None:
                return
            self._printed.add(step)
            roll_s = _safe_float(logs.get("megagem/timing/roll_total_s"))
            games_s = _safe_float(logs.get("megagem/timing/game_rollout_s"))
            push_s = _safe_float(logs.get("megagem/timing/adapter_push_s"))
            select_s = _safe_float(logs.get("megagem/timing/balanced_select_s"))
            train_s = _safe_float(logs.get("megagem/timing/trainer_step_s"))
            other_s = _safe_float(
                logs.get("megagem/timing/step_time_unaccounted_s"))
            print(
                f"  [step {step:03d}] STEP-TIME  "
                f"total={_fmt_metric(total, digits=1)}s  "
                f"rollout={_fmt_metric(roll_s, digits=1)}s  "
                f"games={_fmt_metric(games_s, digits=1)}s  "
                f"adapter_push={_fmt_metric(push_s, digits=1)}s  "
                f"select={_fmt_metric(select_s, digits=3)}s  "
                f"trainer={_fmt_metric(train_s, digits=1)}s  "
                f"other={_fmt_metric(other_s, signed=True, digits=1)}s",
                flush=True)

    # Probe instrument (opt-in, PHASE3_PROBE=1): clean per-optimizer-step train
    # time + vLLM prefix-cache deltas → probe_timing.jsonl. Inert + isolated so
    # it can never perturb a real run; see megagem.training.probe_instrument.
    _callbacks = [_StepTiming(), _Ckpt(), _Snapshot()]
    if os.environ.get("PHASE3_PROBE") == "1":
        try:
            from megagem.training.probe_instrument import ProbeStepTimer
            _pdir = os.environ.get("PHASE3_PROBE_DIR") or tmp
            _callbacks.append(ProbeStepTimer(_pdir, vllm_urls))
            print(f"[phase3] PROBE on: per-step train-time + prefix-cache → "
                  f"{_pdir}/probe_timing.jsonl", flush=True)
        except Exception as _e:  # noqa: BLE001 — probe must never break a run
            print(f"[phase3] PROBE attach failed (non-fatal): "
                  f"{type(_e).__name__}: {_e}", flush=True)
    trainer = MegaGemGRPOTrainer(**H._filter_kwargs(
        GRPOTrainer.__init__, dict(
            model=model, reward_funcs=[precomputed_reward_func], args=cfg,
            train_dataset=dataset, processing_class=tok,
            rollout_func=rollout_func, callbacks=_callbacks,
            peft_config=H.megagem_lora_config(),  # applied ONCE
        )))
    holder["model"] = trainer.model  # PEFT-wrapped; the live policy
    init_lora_state = _trainable_param_snapshot(trainer.model)

    # #2: TRUE pre-training baseline — save the initial (untrained) adapter
    # BEFORE trainer.train(). At init LoRA B≡0 ⇒ behaviourally == SFT1200-v2;
    # this is the genuine step-0 policy the informational baseline evals.
    init_ckpt = ADP.save_step_adapter(trainer.model, 0, args.adapter_root)

    # repl_08: scripted heuristic removed from training — pre-seed the pool
    # with a PINNED step_0 snapshot pointing at the (LoRA-B≡0) SFT-base adapter.
    # This is the stationary anchor the policy can always train against, and
    # replaces the role the heuristic used to play. Pinned ⇒ never evicted
    # from the ring buffer; PFSP weighting includes it just like any other
    # snapshot. A failed push is FATAL (unlike unpinned snapshots) because the
    # pool would otherwise be empty at step 0 and OpponentPool.draw raises.
    seeded_anchor: dict | None = None
    if pool is not None:
        anchor_path = ADP.save_snapshot_adapter(
            trainer.model, 0, args.adapter_root)
        anchor_name = ADP.snapshot_served_name(0)
        anchor_sync = ADP.push_adapter_to_vllm_all(
            vllm_urls, anchor_name, anchor_path)
        if not anchor_sync.get("ok"):
            raise RuntimeError(
                f"step_0 anchor adapter→vLLM sync FAILED (load_status="
                f"{anchor_sync.get('load_status')}, "
                f"body={anchor_sync.get('load_body')!r}); without a pinned "
                f"anchor the opponent pool would be empty at step 0 (no "
                f"heuristic fallback in repl_08+). Per-worker results: "
                f"{anchor_sync.get('per_url')}")
        endpoints.ENDPOINTS[anchor_name] = {
            "model": anchor_name, "url": _endpoint_url_value, "key": "EMPTY"}
        anchor_snap = Snapshot(
            step=0, served_name=anchor_name, adapter_path=anchor_path)
        pool.add_pinned_snapshot(anchor_snap)
        seeded_anchor = {
            "step": 0, "name": anchor_name, "adapter_path": anchor_path,
            "per_url": anchor_sync.get("per_url"),
        }
        print(f"[phase3] repl_08 anchor pre-seeded: pinned step_0 snapshot "
              f"({anchor_name}); pool active from step 0.", flush=True)
        snapshot_events.append({
            "step": 0, "ok": True, "name": anchor_name, "pinned": True,
            "evicted": None, "per_url": anchor_sync.get("per_url"),
            "evict_per_url": None})

    crashed, reason = False, None
    t_start = time.perf_counter()
    try:
        trainer.train()  # single continuous run: optimizer/scheduler/step
    except Exception as e:  # noqa: BLE001
        crashed, reason = True, f"{type(e).__name__}: {e}"

    final_step = int(getattr(trainer.state, "global_step", args.steps))
    final_ckpt = ADP.save_step_adapter(
        trainer.model, final_step, args.adapter_root)
    final_checkpoint_diagnostics = {
        "step": final_step,
        "path": final_ckpt,
        "lora": _lora_parameter_stats(trainer.model, init_lora_state),
    }
    final_probe = _probe_checkpoint_stats()
    if final_probe is not None:
        final_checkpoint_diagnostics["probe_logprob"] = final_probe

    lh = list(getattr(trainer.state, "log_history", []))
    _annotate_train_log_with_update_pressure(
        lh, rolls_meta, kl_beta=args.kl_beta)
    _annotate_train_log_with_timing(lh, rolls_meta, step_timing)
    _annotate_train_log_with_checkpoints(
        lh, [*ckpts, final_checkpoint_diagnostics])
    # entropy + clip fractions ride the same harvest as kl: TRL's standard
    # (non-Liger) loss path logs them per step — phase3 runs loss_type="dapo".
    # clip_ratio/region_mean is the fraction of tokens in the clipped region.
    series = {k: [e[k] for e in lh if k in e]
              for k in ("kl", "loss", "grad_norm", "entropy",
                        "clip_ratio/region_mean", "clip_ratio/low_mean",
                        "clip_ratio/high_mean")}
    # Persist the FULL per-step TRL log_history as a sidecar — overall_health
    # keeps only per-metric summaries, but a long run wants the per-step record
    # (loss, grad_norm, kl, entropy, lr, step_time, megagem/advantages/*,
    # completions/*, …). One JSON array, element i = step i; the trailing
    # entry is the final train summary. Non-fatal — telemetry never aborts a run.
    train_log_path = args.output.parent / "train_log.json"
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if _ddp_rank_world()[0] == 0:  # DDP: rank 0 owns the canonical train log
            train_log_path.write_text(json.dumps(lh, indent=2, default=str))
    except OSError:
        train_log_path = None
    last_rows = roll_state.get("rows") or []
    overall = _health(series, last_rows, args)
    # M1 — anchor-WR trend: the absolute-improvement signal (see
    # `_anchor_winrate_trend`). Reported always; gated only as informational
    # telemetry (a flat anchor slope is the EXPECTED null for a diagnostic
    # run, so it must NOT flip `status` to FAIL by default). Opt-in
    # PHASE3_ANCHOR_ABORT folds it into `all_pass`, mirroring PHASE3_HEALTH_ABORT.
    try:
        anchor_trend = _anchor_winrate_trend(rolls_meta)
    except Exception as e:  # noqa: BLE001 — telemetry never aborts
        anchor_trend = {"error": f"{type(e).__name__}: {e}"}
    overall["anchor_winrate_trend"] = anchor_trend
    _slope = anchor_trend.get("slope") if isinstance(anchor_trend, dict) else None
    _slope_t = anchor_trend.get("slope_t") if isinstance(anchor_trend, dict) else None
    try:
        _slope_min = float(os.environ.get("PHASE3_ANCHOR_SLOPE_MIN", "0.0"))
    except (TypeError, ValueError):
        _slope_min = 0.0
    try:
        _t_min = float(os.environ.get("PHASE3_ANCHOR_T_MIN", "2.0"))
    except (TypeError, ValueError):
        _t_min = 2.0
    # Gate on SIGNIFICANCE (slope_t ≥ t_min), not just sign — a noisy positive
    # slope is not improvement. None slope ⇒ too few anchor draws to judge.
    overall["gates"]["anchor_winrate_improving"] = _anchor_winrate_improving(
        anchor_trend, slope_min=_slope_min, t_min=_t_min)
    _anchor_fatal = os.environ.get("PHASE3_ANCHOR_ABORT", "0") == "1"
    overall["all_pass"] = all(
        v for k, v in overall["gates"].items()
        if _anchor_fatal or k != "anchor_winrate_improving"
    )
    if isinstance(anchor_trend, dict) and "error" not in anchor_trend:
        print(
            f"[phase3] ANCHOR-WR-TREND slope={_slope if _slope is None else round(_slope, 6)} "
            f"t={_slope_t if _slope_t is None else round(_slope_t, 2)} "
            f"(t_min={_t_min}) delta_half={anchor_trend.get('delta_half')} "
            f"n_rolls={anchor_trend.get('n_rolls_with_anchor')} "
            f"total_anchor_games={anchor_trend.get('total_anchor_games')} "
            f"(improving={overall['gates']['anchor_winrate_improving']}, "
            f"fatal={_anchor_fatal})",
            flush=True)
        if os.environ.get("WANDB_API_KEY"):
            try:
                import wandb
                if wandb.run is not None:
                    for _k in ("slope", "delta_half", "total_anchor_games",
                               "spearman_wr_vs_step"):
                        _v = anchor_trend.get(_k)
                        if isinstance(_v, (int, float)):
                            wandb.summary[f"megagem/anchor_trend/{_k}"] = _v
            except Exception:  # noqa: BLE001 — telemetry never aborts
                pass
    passed = (not crashed) and overall["all_pass"]
    # On-policy-ness is optimizer-steps-per-generation (spg//ga), NOT raw spg:
    # a high-spg run consumed in ONE step via accumulation (ga=spg) is on-policy.
    _opg = max(1, spg // ga)
    cadence = ("per-step on-policy (1 optimizer step / generation)" if _opg == 1
               else f"on-policy every {_opg} steps (amortised; --on-policy or "
                     f"raise PHASE2_MICRO_CAP / ga for 1)")
    n_refresh = roll_state.get("rolls") or 0
    is_seam_shape = (_opg > 1) or (n_refresh < max(5, args.steps // 5))
    return {
        "step": "3.1-3.5", "mode": "GPU",
        "status": "PASS" if passed else "FAIL",
        "crashed": crashed, "crash_reason": reason,
        "training_mode": ("SINGLE persistent trainer — continuous "
                          "optimizer/scheduler/global-step (NOT per-segment "
                          "reset)"),
        "on_policy_cadence": cadence,
        # Honesty flag: spg>1 (or very few refreshes) ⇒ this is a SEAM-TEST
        # SHAPE, not a strong on-policy evidence run. The run script banners
        # on it and the eval carries it into final_gate_caveats.
        "is_seam_shape": is_seam_shape,
        "n_onpolicy_refreshes": n_refresh,
        "evidence_run_note": (
            "SEAM-TEST SHAPE — only %d fresh on-policy refreshes over %d "
            "steps (spg=%d). Do NOT read as a strong on-policy evidence run; "
            "raise PHASE2_MICRO_CAP / lower --rows-per-gen for spg=1."
            % (n_refresh, args.steps, spg)
        ) if is_seam_shape else "strong on-policy shape (spg=1)",
        "steps": args.steps, "global_step":
            int(getattr(trainer.state, "global_step", -1)),
        "rows_per_gen": n, "micro": micro, "spg": spg, "ga": ga,
        "num_processes": np_, "opt_steps_per_gen": _opg,
        "effective_batch": _shape["effective_batch"],
        "num_generations": g, "n_rolls": roll_state.get("rolls"),
        "k": args.k, "seeds": initial_seeds,
        "seed_schedule": {
            "mode": "fixed_reused" if args.fixed_train_seeds else "fresh_per_roll",
            "seed_start": args.seed_start,
            "num_seeds_per_roll": args.num_seeds,
            "initial_seeds": initial_seeds,
            "last_roll_seeds": (
                rolls_meta[-1].get("seeds") if rolls_meta else initial_seeds),
            "total_distinct_train_seeds_seen": len({
                s for r in rolls_meta for s in (r.get("seeds") or [])
            }),
        },
        "value_chart": args.value_chart,
        "sampling": A7_SAMPLING,
        "max_parallel": args.max_parallel,
        "rollout_dump_dir": args.dump_rollouts or None,
        "train_log": str(train_log_path) if train_log_path else None,
        # Plan §D.1 — captured on roll 0 (LoRA B≡0); the per-run reference for
        # "did RL collapse / inflate completion length?". Train-time per-step
        # `completions/mean_length` lives in train_log.json — compare against
        # the `tokens.mean` here.
        "sft_baseline_lengths": (sft_baseline_lengths or None),
        "config": {
            "model": args.model,
            "steps": args.steps, "k": args.k, "num_seeds": args.num_seeds,
            "rows_per_gen": n, "num_generations": g, "micro": micro,
            "spg": spg, "max_completion_length": args.max_completion_length,
            "checkpoint_every": args.checkpoint_every,
            "fixed_train_seeds": args.fixed_train_seeds,
            # `trainable_seat` is the BASE seat; with rotate_seats the actual
            # per-roll seat round-robins (recorded per-roll in rolls_meta).
            "trainable_seat": args.trainable_seat,
            "rotate_seats": bool(getattr(args, "rotate_seats", False)),
            "p_current_self": args.p_current_self,
            "heuristic_anneal_end": args.heuristic_anneal_end,
            "value_chart": args.value_chart, "max_parallel": args.max_parallel,
            "seed": args.seed,
            "learning_rate": args.learning_rate, "kl_beta": args.kl_beta,
            "kl_max": args.kl_max,
            # epsilon / epsilon_high: requested (None ⇒ TRL-defaulted) and the
            # post-resolution values the clip actually used (TRL resolves the
            # symmetric fallback onto the trainer, not the config).
            "epsilon": args.epsilon, "epsilon_high": args.epsilon_high,
            "epsilon_low_effective": getattr(trainer, "epsilon_low", None),
            "epsilon_high_effective": getattr(trainer, "epsilon_high", None),
            "loss_type": getattr(cfg, "loss_type", None),
        },
        # Full RewardConfig dump so scripts/analysis/inspect_rollouts.py can re-derive the
        # exact reward decomposition this run trained on (scale, shape, λ,
        # reveal weight, illegal penalty, terminal correction). asdict over
        # the live cfg means a new field added to RewardConfig flows through
        # without touching this site.
        "reward_config": _dataclass_asdict(reward_cfg),
        "overall_health": overall,
        "rolls": rolls_meta,
        "opponent_pool": ({
            "enabled": True,
            "snapshot_every": args.snapshot_every,
            "max_snapshots": args.max_snapshots,
            "anneal": {"start": args.opp_anneal_start,
                       "end": args.opp_anneal_end,
                       "p_max": args.opp_anneal_pmax},
            # Codex telemetry-honesty: in repl_08 pure-pool mode the pool's
            # `heuristic_spec` is None, so `draw()` never enters the
            # heuristic-fallback branch — every draw is either an API
            # opponent (sub-probability p_api) or a snapshot (via PFSP). The
            # anneal_start/end/p_max are NOT consulted in this mode. We log
            # the literal CLI values above (for reproducibility) AND a
            # `pool_semantics` block that records what the pool ACTUALLY did.
            "pool_semantics": {
                "heuristic_in_training": (pool.heuristic_spec is not None),
                "anneal_consulted": (
                    pool.heuristic_spec is not None
                    and pool.p_heuristic <= 0.0),
                "heuristic_decay_active": (pool.p_heuristic > 0.0),
                "p_heuristic_configured": float(pool.p_heuristic),
                "p_current_self_configured": float(pool.p_current_self),
                "anchor_floor": args.opp_anchor_floor,
                "note": (
                    "no-heuristic 80/20 league mode: current live adapter "
                    "gets p_current_self={ps:.3f} of opponent-seat draws; "
                    "checkpoints get the remaining mass after API/heuristic "
                    "gates. Within the checkpoint bucket, anchor_floor={af:.3f} "
                    "is reserved for pinned and the rest is PFSP-weighted. "
                    "heuristic_spec=None, so heuristic draws are impossible "
                    "unless --p-heuristic is explicitly >0."
                    .format(ps=pool.p_current_self, af=args.opp_anchor_floor)
                    if pool.p_current_self > 0.0 and pool.p_heuristic <= 0.0
                    else
                    (
                    "repl_08 v3 heuristic-with-decay: single-tier mix-gate. "
                    "heuristic gets p_heuristic_effective = p_heuristic·"
                    "max(0.10, (1-WR_heur)²); API gets p_api={pa:.3f} (ABSOLUTE "
                    "share of all draws, not conditional); snapshot bucket "
                    "gets the remainder. Within the snapshot bucket, "
                    "anchor_floor={af:.3f} is reserved for pinned. The legacy "
                    "anneal path is SKIPPED in this mode (Codex round-1 fix)."
                    " Construction-time guard: p_heuristic({ph:.3f}) + p_api "
                    "must sum to ≤ 1."
                    .format(ph=pool.p_heuristic, pa=args.opp_api_prob,
                            af=args.opp_anchor_floor)
                    if pool.p_heuristic > 0.0
                    else (
                        "repl_08 pure-pool mode: heuristic_spec=None ⇒ draws "
                        "are p_api × API + (1-p_api) × (anchor_floor × pinned "
                        "+ (1-anchor_floor) × PFSP-weighted-snapshots). The "
                        "anneal_* params are NOT consulted; they are logged "
                        "for back-compat only."
                        if pool.heuristic_spec is None else
                        "repl_07-era mode: draws follow the anneal schedule "
                        "from heuristic to pooled (snapshots+API)."))),
            },
            "api_models": api_models, "p_api": args.opp_api_prob,
            "p_current_self": args.p_current_self,
            "api_weights": (api_weights if api_weights is not None
                            else [1] * len(api_models)),
            "rng_seed": args.opp_pool_seed,
            # repl_08: step_0 SFT-base anchor pre-seeded as a pinned snapshot
            # (never evicted). None ⇒ pre-seed step skipped or failed (latter
            # would raise; former only with --no-opponent-pool which sets
            # pool=None above and thus enabled=False).
            "anchor_snapshot": seeded_anchor,
            "final_state": pool.telemetry(int(getattr(
                trainer.state, "global_step", args.steps))),
        } if pool is not None else {"enabled": False}),
        "dapo_dynamic_sampling_config": {
            "enabled": dapo_enabled,
            "abs_threshold": dapo_abs_threshold,
            "opp_rel_threshold": dapo_opp_rel,
            "ema_alpha": dapo_ema_alpha,
            "final_opp_ema_std": dict(opp_ema_std),
        },
        "snapshot_events": snapshot_events,
        "step_timing": step_timing,
        "checkpoints": {"step_0_pretrain": init_ckpt,
                        "intermediate": ckpts, "final": final_ckpt},
        "final_checkpoint_diagnostics": final_checkpoint_diagnostics,
        "adapter_root": args.adapter_root,
        "wall_s": round(time.perf_counter() - t_start, 2),
        "spend_decision_note": (
            "Health-only. The §3.6 paired-bootstrap spend/no-spend number "
            "comes from scripts/training/phase3_eval.py: step_0 (informational "
            "baseline, ≈0) and the FINAL checkpoint (the gated number)."),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__)
    p.add_argument("--model",
                   default="djdumpling/qwen3-4b-instruct-megagem-sft-step1200-v2")
    p.add_argument("--served-model-name", default="qwen/qwen3-4b-instruct")
    p.add_argument("--vllm-url", default=None,
                   help="LoRA-enabled rollout vLLM (trainable seat). For DP "
                        "(>1 vLLM workers) pass --vllm-urls instead; this "
                        "single-URL flag stays for the legacy 2-GPU layout.")
    p.add_argument("--vllm-urls", default=None,
                   help="Comma-separated list of vLLM URLs for data-parallel "
                        "rollout (each worker on its own GPU; round-robin per "
                        "request). Overrides --vllm-url. Adapter sync fans "
                        "out to all URLs; partial-push failure rolls back.")
    p.add_argument("--heuristic-url", default=None,
                   help="megagem.training.heuristic_endpoint shim (opponent seats).")
    p.add_argument("--steps", type=int, default=50,
                   help="GRPO update steps (first pass: short ~50). ONE "
                        "continuous trainer — optimizer/scheduler persist.")
    p.add_argument("--rows-per-gen", type=int, default=96,
                   help="fixed dataset N. Each on-policy roll must yield "
                        "≥N trainable rows (asserted, never padded). On-policy "
                        "freshness cadence = spg (derived; reported, raise "
                        "PHASE2_MICRO_CAP for per-step spg=1).")
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--k", type=int, default=8, help="§A.7 K rollouts/seed.")
    p.add_argument("--num-seeds", type=int, default=6)
    p.add_argument("--seed-start", type=int, default=9000)
    p.add_argument("--fixed-train-seeds", action="store_true",
                   help="Reuse the same training seeds every rollout. Default "
                        "is fresh_per_roll: seed_start + roll*num_seeds + i.")
    p.add_argument("--num-players", type=int, default=3)
    p.add_argument("--value-chart", default="A", help="chart A only.")
    p.add_argument("--trainable-seat", type=int, default=0)
    p.add_argument("--num-generations", type=int, default=2)
    # --- batch shape: gradient accumulation + DDP (phase3-rl-resize-8xh200) --- #
    p.add_argument("--on-policy", action="store_true",
                   help="Consume the WHOLE generation in ONE optimizer step via "
                        "gradient accumulation (ga=spg). Large on-policy batch at "
                        "no extra activation memory. Overrides --gradient-accumulation-steps.")
    p.add_argument("--gradient-accumulation-steps", type=int, default=None,
                   help="Explicit ga. None ⇒ legacy ga=1 (off-policy reuse). "
                        "Amortises generation over spg//ga optimizer steps.")
    p.add_argument("--num-processes", type=int,
                   default=int(os.environ.get("WORLD_SIZE", "1")),
                   help="DDP world size (torchrun --nproc_per_node). >1 multiplies "
                        "the per-step batch; requires PHASE3_ALLOW_DDP=1 until the "
                        "rollout is rank-sharded (see _gpu_run guard).")
    p.add_argument("--kl-max", type=float, default=0.5)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--kl-beta", type=float, default=0.01,
                   help="GRPO KL-penalty weight (loss term β·KL(π‖π_ref)). "
                        "0.01 default is light-touch — obs_smoke_01 showed "
                        "KL=1e-3 vs the 0.5 budget, so β didn't bind anyway. "
                        "Raise to 0.05+ if KL crosses 0.1 mid-run.")
    p.add_argument("--epsilon", type=float, default=None,
                   help="symmetric PPO clip epsilon. Unset ⇒ TRL default 0.2.")
    p.add_argument("--epsilon-high", type=float, default=0.28,
                   help="DAPO clip-higher: asymmetric UPPER clip epsilon. "
                        "Default 0.28 (was opt-in pre-repl_02). With the new "
                        "evidence geometry (spg≈6, rollout reuse), symmetric "
                        "0.2 would gate out positive-advantage updates on "
                        "stale rollouts; clip-higher lets them through while "
                        "keeping the lower clip tight. Pass --epsilon-high 0.2 "
                        "to opt back into symmetric clipping.")
    p.add_argument("--max-completion-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--adapter-root", default="results/phase3_grpo/adapters")
    p.add_argument("--max-parallel", type=int, default=32,
                   help="max concurrent games per rollout (semaphore-bounded). "
                        "vLLM batches concurrent generations on the GPU; ~5–6× "
                        "wall-time speedup at no extra GPU cost. Set to 1 to "
                        "restore the old sequential behaviour (fallback if "
                        "vLLM preempts under concurrency — check vllm_server.log).")
    p.add_argument("--dump-rollouts", default=None,
                   help="if set, persist each on-policy roll's actor-tagged "
                        "schema-v3 games into <DIR>/roll_NNN/ so the reward "
                        "diagnostic (scripts/training/reward_score_correlation.py)"
                        " can run on the TRUE phase-3 rollout distribution. "
                        "Off by default — rollouts are otherwise discarded "
                        "with the run's temp dir. ~80MB for an evidence run.")
    # ---- §3.3 lagged-self opponent pool ---------------------------------- #
    p.add_argument("--opponent-pool", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="§3.3 lagged-self opponent pool. --no-opponent-pool ⇒ "
                        "the legacy single stationary heuristic opponent.")
    p.add_argument("--hetero-opponents", action="store_true",
                   help="PSRO/league HETEROGENEOUS tables: draw a SEPARATE "
                        "opponent from the pool for EACH non-trainable seat, so "
                        "the trainee faces two DIFFERENT opponents instead of two "
                        "clones (the one structurally-untested lever). §A.7-safe "
                        "(per-seat draw constant within a K-group) and loss-safe "
                        "(masking is by seat). Pair with a DIVERSE pool config "
                        "(e.g. --p-heuristic + anchor + snapshots).")
    p.add_argument("--snapshot-every", type=int, default=25,
                   help="§3.3 snapshot cadence (optimizer steps).")
    p.add_argument("--max-snapshots", type=int, default=5,
                   help="§3.3 ring-buffer size — keep the last N snapshots.")
    p.add_argument("--opp-anneal-start", type=int, default=50,
                   help="step at which P(pooled opponent) leaves 0.")
    p.add_argument("--opp-anneal-end", type=int, default=None,
                   help="step at which P(pooled opponent) reaches "
                        "--opp-anneal-pmax. Default: == --steps (anneal over "
                        "the whole run).")
    p.add_argument("--opp-anneal-pmax", type=float, default=0.7,
                   help="max P(pooled opponent); the remaining 1-pmax of "
                        "K-groups always face the heuristic (a floor).")
    p.add_argument("--opp-api-models", default="",
                   help="comma-separated API opponent model ids (resolved via "
                        "PRIME_API_KEY). Empty ⇒ lagged-self only.")
    p.add_argument("--opp-api-prob", type=float, default=0.0,
                   help="P(API opponent) of TOTAL draws in league-mix mode; "
                        "legacy mode treats it as the post-anneal API "
                        "sub-probability.")
    p.add_argument("--p-current-self", type=float, default=0.80,
                   help="P(draw the current live adapter as the greedy "
                        "opponent) of TOTAL opponent-seat draws in "
                        "--opponent-pool mode. The remaining mass after "
                        "--p-heuristic, --opp-api-prob, and this value goes "
                        "to previous checkpoints via the snapshot/PFSP "
                        "bucket. Default 0.80 implements the no-heuristic "
                        "AlphaStar-style current-self/checkpoint mix.")
    p.add_argument("--opp-api-weights", default="",
                   help="comma-sep non-negative ints aligned with "
                        "--opp-api-models; relative pick weights within the "
                        "API bucket. Empty ⇒ uniform (back-compat). "
                        "Example: '15,4,1' with models 'G,O,S' ⇒ 75/20/5.")
    p.add_argument("--opp-pool-seed", type=int, default=0,
                   help="deterministic opponent-draw RNG seed.")
    p.add_argument("--opp-anchor-floor", type=float, default=0.15,
                   help="repl_08: minimum draw PROBABILITY for the pinned "
                        "step_0 anchor among snapshot draws (default 0.15 = "
                        "15%%). Without it, once the trainee crushes the "
                        "anchor (WR≈0.95 ⇒ f_hard floored to 0.01) the anchor "
                        "gets ~1%% of draws and stops anchoring. 0.0 ⇒ pure "
                        "PFSP weighting (repl_07-era behaviour).")
    p.add_argument("--p-heuristic", type=float, default=0.0,
                   help="repl_08 v3: P(draw the scripted heuristic) of TOTAL "
                        "draws (absolute share — single-tier mix-gate with "
                        "p_api). Decays in megagem.training.opponent_pool as `p_heuristic · "
                        "max(0.10, (1-WR_heur)²)` once heuristic WR is "
                        "observed — AlphaStar 'main exploiter' role. Default "
                        "0.0 ⇒ heuristic OFF in training (the seam_smoke_02 / "
                        "pure-self-play configuration). Set to 0.20-0.30 to "
                        "reintroduce the easy-win signal that seam_smoke_02 "
                        "lacked. Requires the heuristic shim endpoint to be "
                        "registered when > 0. Construction-time guard: "
                        "p_heuristic + p_api ≤ 1.")
    p.add_argument("--heuristic-anneal-end", type=int, default=0,
                   help="repl_08 v3.2: step at which the heuristic draw "
                        "probability anneals to EXACTLY 0 (linear from step 0). "
                        "Decoupled from the (1-WR)² decay floor, so the "
                        "heuristic bootstraps cold-start symmetry-breaking then "
                        "exits — leaving the SFT step_0 anchor + lagged "
                        "snapshots as the signal source (AlphaStar/OpenAI-Five "
                        "pattern). 0 (default) ⇒ no step anneal (floor-only, "
                        "back-compat: heuristic persists at ≈0.10·p_heuristic "
                        "forever). Recommended ~2× snapshot_every.")
    p.add_argument("--rotate-seats", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="rotate the trainable seat round-robin per roll "
                        "(seat = (trainable_seat + roll_index) %% num_players), "
                        "held constant within each K-group so §A.7 within-group "
                        "standardization is unaffected. Trains the policy from "
                        "ALL seats — required because the TrueSkill/panel eval "
                        "rates the policy across seat0/1/2 (training only seat 0 "
                        "leaves seats 1/2 off-distribution). --no-rotate-seats "
                        "(default) pins every roll to --trainable-seat.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--require-onpolicy", action="store_true",
                   help="--dry-run only: FAIL if the projected run is "
                        "seam-shaped (spg>1 or too few on-policy refreshes). "
                        "run_phase3.sh sets it for PROFILE=evidence so a "
                        "refresh-starved config is caught before GPU spend.")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    # --opp-anneal-end defaults to the run length so a direct python invocation
    # with --steps N anneals over the whole run (run_phase3.sh passes it
    # explicitly from the PROFILE).
    if args.opp_anneal_end is None:
        args.opp_anneal_end = args.steps
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.dry_run:
            result = _dry_run(args)
        else:
            Path(args.adapter_root).mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory() as tmp:
                result = _gpu_run(args, tmp)
    except SystemExit as e:
        result = {"step": "3.x", "status": "FAIL", "reason": f"ABORT: {e}"}
    except Exception as e:  # noqa: BLE001
        result = {"step": "3.x", "status": "FAIL",
                  "reason": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # DDP: only rank 0 writes the canonical output + banners. Ranks>0 carry a
    # stub rolls_meta (rank 0 owns generation), so letting them write would cl
    # obber the real result. They still return their status code for torchrun.
    if _ddp_rank_world()[0] == 0:
        args.output.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n[{result.get('mode', '?')}] Phase 3 — {result['status']}")
        if result["status"] != "PASS":
            if "traceback" in result:
                print(result["traceback"])
            else:
                print(f"  {result.get('reason') or result.get('crash_reason')}")
        print(f"→ {args.output}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
