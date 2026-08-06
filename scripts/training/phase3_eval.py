#!/usr/bin/env python3
"""Phase-3 §3.6 — paired-bootstrap eval: RL checkpoint vs SFT1200-v2.

The spend/no-spend number. For each fixed eval condition `i` (a (seed,
trainable_seat) on chart A vs the **same stationary heuristic** opponents),
play the trainable seat **twice under identical conditions** — once with the
RL checkpoint, once with the frozen `…-step1200-v2` SFT — and pair:

    delta_i = score_RL_i − score_SFT_i      (engine terminal score)

Pairing is exact *because the opponent is deterministic*: at a fixed seed the
heuristic plays the identical game in both legs, so `delta_i` isolates the
trainable policy (this is precisely why heuristic, not self-mirror, was
chosen — no opponent variance to pollute the comparison).

`score` is the **engine terminal score** —
`final_results.final_scores[pid].final_score` (all private hands auto-revealed
at game end, loans repaid, investments returned), i.e. the real game outcome.
It is deliberately NOT `scorer.score_at`: that is the hand-free *mid-game*
proxy the RL reward uses (Problem B′ leakage gate) and it diverges from the
true terminal score on 131/144 Phase-1 trajectories — the reward proxy and the
§3.6 eval ground truth are different objects by design. The CI is the reused,
dependency-free `H.paired_bootstrap_ci`.

**Gate (the RL plan §3.6): `ci_low(mean delta) > +2.0` score points.** Note the
threshold is +2, not >0. Run once per checkpoint (the run script invokes it
for step 0 and the final checkpoint; an intermediate is optional/cheap).

  prime-rl/.venv/bin/python scripts/training/phase3_eval.py \
      --label final --rl-served phase3-trainable \
      --sft-served qwen/qwen3-4b-instruct \
      --vllm-url http://localhost:8000/v1 \
      --heuristic-url http://localhost:8100/v1 \
      --output results/phase3_grpo/eval_final.json

  .venv/bin/python scripts/training/phase3_eval.py --dry-run --output /tmp/e.json
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import sys
import tempfile
import traceback
from pathlib import Path

import megagem.training.grpo_harness as H  # (paired_bootstrap_ci)
from megagem.rl.scorer import parse_transcript

HEURISTIC_MODEL = "megagem/heuristic-v1"
GATE_THRESHOLD = 2.0  # §3.6: ci_low(delta) must exceed +2 score points


def _opponent_model(eval_opponent: str) -> str:
    """Resolve --eval-opponent to the served-model-name for the opponent
    seats. 'heuristic' (default) ⇒ the deterministic heuristic shim; any other
    value is taken verbatim as a model id resolvable via megagem.endpoints
    (e.g. google/gemini-3-flash-preview via PRIME_API_KEY)."""
    if eval_opponent in ("heuristic", HEURISTIC_MODEL):
        return HEURISTIC_MODEL
    return eval_opponent


def _opponent_is_deterministic(eval_opponent: str) -> bool:
    """True only for the heuristic shim — deterministic by construction, so
    greedy replays are byte-identical and the paired bootstrap is exact. A
    hosted API model is NOT bitwise-deterministic even at T=0."""
    return eval_opponent in ("heuristic", HEURISTIC_MODEL)


def _game_details(transcript: dict, seat: int) -> dict:
    """Per-game decomposition (Phase 3 §C.3). Saved alongside the scalar
    `_terminal_score` so a post-mortem can tell margin-driven wins from
    illegal-avoidance, and policy-sample variance from game-shape variance.

    Returns ``{score, true_margin, num_rounds, n_illegal_actions,
    n_legal_actions}``. The terminal score field duplicates `_terminal_score`
    intentionally — same number, kept inline so the caller doesn't need both
    helpers to reconstruct a row.
    """
    parsed = parse_transcript(transcript)
    scores = {
        int(e.get("player_id", -1)): int(e.get("final_score", 0))
        for e in parsed.final_scores
        if "player_id" in e and "final_score" in e
    }
    others = [v for p, v in scores.items() if p != seat]
    margin = float(
        scores.get(seat, 0) - (sum(others) / len(others) if others else 0)
    )
    n_legal = 0
    n_illegal = 0
    for rnd in parsed.rounds:
        for rec in rnd.get("players", []) or []:
            if int(rec.get("player_id", -1)) != seat:
                continue
            bid_legal = bool(rec.get("parse_valid")) and bool(
                rec.get("legal_valid"))
            if bid_legal:
                n_legal += 1
            else:
                n_illegal += 1
            reveal = rec.get("reveal")
            if isinstance(reveal, dict):
                rev_legal = bool(reveal.get("parse_valid")) and bool(
                    reveal.get("legal_valid"))
                if rev_legal:
                    n_legal += 1
                else:
                    n_illegal += 1
    return {
        "score": scores.get(seat, 0),
        "true_margin": margin,
        "num_rounds": int(parsed.num_rounds),
        "n_legal_actions": n_legal,
        "n_illegal_actions": n_illegal,
    }


def _terminal_score(transcript: dict, seat: int) -> int:
    """**Engine** terminal score for `seat` — `final_results.final_scores`
    (all remaining private hands auto-revealed at game end; loans repaid,
    investments returned). This is the real game outcome §3.6 judges.

    NOT `scorer.score_at`: that is the deliberately hand-free *mid-game*
    proxy the RL reward uses (Problem B′ leakage gate) and it diverges from
    the true terminal score on 131/144 Phase-1 trajectories — using it would
    silently invalidate the paired-bootstrap gate. The reward proxy and the
    eval ground truth are different objects by design.
    """
    parsed = parse_transcript(transcript)
    for e in parsed.final_scores:
        if int(e.get("player_id", -1)) == seat:
            return int(e["final_score"])
    raise KeyError(
        f"no final_scores entry for seat {seat} "
        f"(final_scores has {len(parsed.final_scores)} entries)")


async def _play_async(
    models, *, seed, value_chart, num_players, tag, tmp, semaphore,
    temperature=None, counter=None, total=None, eval_label="eval",
) -> dict:
    """One paired-eval game, run inside a SHARED event loop under a semaphore.

    `temperature` (Lever C): if set, passed to BOTH arms symmetrically via
    `caller_api_params`. T=0.0 selects greedy decoding — both arms become
    deterministic given state, the per-game policy-sampling variance that
    dominates our CI collapses, and the run becomes a low-noise corroborating
    signal (at the cost of measuring the policy MODE not the full distribution
    the policy was trained for). `None` ⇒ no override; vLLM serves at its
    default sampling.

    `counter`/`total`/`eval_label` (UX): per-game completion logging, mirroring
    phase3_grpo's _play_one. ``silent=True`` drops run_game's per-round noise;
    we replace it with ONE line per game finish, with a running index so a
    concurrent eval is auditable in real time. ``counter`` is a single-element
    list to make async-safe incrementing trivial (asyncio is single-threaded).
    """
    import time as _time
    from megagem.rollout import run_game  # lazy
    async with semaphore:
        fname = f"eval_{tag}_seed{seed}.json"
        api_params = None
        if temperature is not None:
            api_params = {"temperature": float(temperature)}
            if float(temperature) == 0.0:
                # greedy ignores top_p but be explicit so a stray default
                # doesn't accidentally enable nucleus sampling.
                api_params["top_p"] = 1.0
        t0 = _time.perf_counter()
        await run_game(
            models=models, value_chart=value_chart, seed=seed,
            num_players=num_players, output_file="trajectory",
            json_filename=fname, quiet=True, silent=True,
            results_dir=Path(tmp), caller_api_params=api_params)
        g = json.loads((Path(tmp) / fname).read_text())
        if counter is not None and total is not None:
            counter[0] += 1
            n_rounds = len(g.get("rounds") or [])
            dur = _time.perf_counter() - t0
            print(f"  [{eval_label}] game {counter[0]:>3}/{total} done  "
                  f"seed={seed} tag={tag}  rounds={n_rounds}  {dur:.1f}s",
                  flush=True)
        return g


def _eval_grid(args) -> list[int]:
    return list(range(args.seed_start, args.seed_start + args.eval_seeds))


def _run_eval(args, tmp: str) -> dict:
    import os

    opp_model = _opponent_model(args.eval_opponent)
    opp_is_det = _opponent_is_deterministic(args.eval_opponent)
    if not args.vllm_url:
        raise SystemExit("non-dry-run needs --vllm-url")
    if opp_is_det and not args.heuristic_url:
        raise SystemExit("heuristic eval opponent needs --heuristic-url")

    from megagem import endpoints
    os.environ.setdefault("EMPTY", "EMPTY")
    if args.heuristic_url:
        endpoints.ENDPOINTS[HEURISTIC_MODEL] = {
            "model": HEURISTIC_MODEL, "url": args.heuristic_url,
            "key": "EMPTY"}
    for served in (args.rl_served, args.sft_served):
        endpoints.ENDPOINTS[served] = {
            "model": served, "url": args.vllm_url, "key": "EMPTY"}

    # Wandb (Plan §A.6) — resume the trainer's wandb run so the §3.6 eval
    # results land on the same page as the training time-series. modal_train.py
    # pre-generates WANDB_RUN_ID before launching the trainer subprocess, and
    # run_phase3.sh propagates the same env to this subprocess, so both
    # processes call wandb.init with the same id. `resume="allow"` no-ops if
    # the run already exists; with `reinit=True` we tolerate the prior trainer
    # process already having owned the run within this container's lifetime.
    _wandb_run = None
    if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_RUN_ID"):
        try:
            import wandb
            _wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "megagem-phase3"),
                id=os.environ["WANDB_RUN_ID"],
                resume="allow",
                reinit=True,
            )
        except Exception:  # noqa: BLE001 — telemetry never aborts an eval
            _wandb_run = None
    # A non-heuristic eval opponent must resolve through megagem.endpoints
    # (e.g. an API model via PRIME_API_KEY). Preflight so a misconfig fails
    # fast, not after burning eval-game budget on auth errors.
    if not opp_is_det:
        ep = endpoints.ENDPOINTS.get(opp_model)
        if ep is None:
            raise SystemExit(
                f"--eval-opponent {opp_model!r} not in megagem.endpoints "
                f"— add it there or use 'heuristic'.")
        key_env = ep.get("key")
        if key_env and key_env != "EMPTY" and not os.getenv(key_env):
            raise SystemExit(
                f"--eval-opponent {opp_model!r} needs the ${key_env} "
                f"environment variable set.")
        # The heuristic-calibrated +2 gate is meaningless against a different
        # opponent (different score scale) — refuse to emit a green PASS/SPEND
        # (status PASS + rc 0) on the stale default. --informational runs (the
        # re-baselining measurement itself) never gate, so they are exempt.
        if (not args.informational
                and abs(args.threshold - GATE_THRESHOLD) < 1e-9):
            raise SystemExit(
                f"--eval-opponent {opp_model!r} with the default "
                f"+{GATE_THRESHOLD} threshold: that gate is heuristic-"
                f"calibrated and invalid for this opponent. Re-baseline (run "
                f"SFT-vs-{opp_model} with --informational first), then pass an "
                f"explicit recalibrated --threshold — or use --informational "
                f"for the baseline measurement itself.")

    # Make `--rl-served` resolve to a specific checkpoint by pushing its
    # adapter first (so the run script can eval step0 vs final from the same
    # vLLM without bash push gymnastics). Omitted ⇒ whatever is already loaded.
    adapter_sync = None
    if args.rl_adapter_path:
        import megagem.training.adapter_sync as ADP
        adapter_sync = ADP.push_adapter_to_vllm(
            args.vllm_url, args.rl_served, args.rl_adapter_path)
        if not adapter_sync["ok"]:
            raise SystemExit(
                f"could not push RL adapter {args.rl_adapter_path!r}: "
                f"{adapter_sync['load_body']}")

    seat = args.trainable_seat
    seeds = _eval_grid(args)
    max_parallel = max(1, int(args.max_parallel))
    K = max(1, int(args.eval_samples_per_seed))
    temperature = args.temperature

    # K-sample averaging (Lever A): at T>0 the dominant noise is per-game
    # policy-sampling variance — replay each (seed, arm) K times and AVERAGE
    # the K scores BEFORE pairing. var(delta_avg) = var(delta)/K, so the CI
    # half-width drops by ~√K at K× the eval-game cost. Equivalent to N=K·N
    # single-sample but cheaper (the heuristic doesn't sweat the extra calls).
    # K=1 is the default. Greedy (T=0) auto-collapses to K=1 ONLY when the
    # opponent is deterministic (the heuristic) — then both arms are fully
    # deterministic and replays would be byte-identical. A hosted API opponent
    # is NOT bitwise-deterministic even at T=0, so K-sample averaging is kept.
    if (opp_is_det and temperature is not None
            and float(temperature) == 0.0 and K > 1):
        K_effective = 1
        K_note = (f"K={K} requested but T=0.0 (greedy) vs the deterministic "
                  f"heuristic is byte-identical — K collapsed to 1.")
    else:
        K_effective = K
        K_note = None

    async def _play_all() -> dict:
        # One loop, one semaphore — all 2·N·K games scheduled concurrently up
        # to `max_parallel`. vLLM batches concurrent requests; the heuristic
        # shim is uvicorn-async; both arms safely overlap. The pairing is
        # preserved by indexing on (seed, arm, k_idx), NOT by call order.
        sem = asyncio.Semaphore(max_parallel)
        tags: list[tuple[int, str, int]] = []
        coros = []
        total = len(seeds) * K_effective * 2  # RL + SFT arms
        counter = [0]
        label = getattr(args, "label", None) or "eval"
        print(f"  [{label}] launching {total} eval games "
              f"(seeds={len(seeds)} × K={K_effective} × 2 arms, "
              f"max_parallel={max_parallel})", flush=True)
        for s in seeds:
            for k_idx in range(K_effective):
                opp = [opp_model] * args.num_players
                rl_models = list(opp);  rl_models[seat] = args.rl_served
                sft_models = list(opp); sft_models[seat] = args.sft_served
                tags.append((s, "rl", k_idx))
                coros.append(_play_async(
                    rl_models, seed=s, value_chart=args.value_chart,
                    num_players=args.num_players, tag=f"rl{s}_k{k_idx}",
                    tmp=tmp, semaphore=sem, temperature=temperature,
                    counter=counter, total=total, eval_label=label))
                tags.append((s, "sft", k_idx))
                coros.append(_play_async(
                    sft_models, seed=s, value_chart=args.value_chart,
                    num_players=args.num_players, tag=f"sft{s}_k{k_idx}",
                    tmp=tmp, semaphore=sem, temperature=temperature,
                    counter=counter, total=total, eval_label=label))
        games = await asyncio.gather(*coros)
        by_seed: dict[int, dict[str, list[dict]]] = {}
        for (s, arm, _k), g in zip(tags, games):
            by_seed.setdefault(s, {}).setdefault(arm, []).append(g)
        return by_seed

    by_seed = asyncio.run(_play_all())

    deltas, per_game = [], []
    for s in seeds:
        # SAME seed all legs ⇒ deterministic heuristic plays identically ⇒
        # the pairing isolates the trainable policy. Pairing is by-dict-key,
        # not by-list-order, so concurrency cannot perturb deltas.
        rl_scores = [_terminal_score(g, seat) for g in by_seed[s]["rl"]]
        sft_scores = [_terminal_score(g, seat) for g in by_seed[s]["sft"]]
        rl_avg = sum(rl_scores) / len(rl_scores)
        sft_avg = sum(sft_scores) / len(sft_scores)
        deltas.append(float(rl_avg - sft_avg))
        rl_details = [_game_details(g, seat) for g in by_seed[s]["rl"]]
        sft_details = [_game_details(g, seat) for g in by_seed[s]["sft"]]
        per_game.append({"seed": s,
                         "score_rl_avg": rl_avg, "score_sft_avg": sft_avg,
                         "score_rl_samples": rl_scores,
                         "score_sft_samples": sft_scores,
                         "delta": rl_avg - sft_avg,
                         "k_samples": K_effective,
                         # §C.3 — per-K-sample decomposition (margin /
                         # illegal counts / num_rounds). Same data the engine
                         # already saw; we just stop discarding it so a future
                         # post-mortem can tell margin-driven wins from
                         # illegal-avoidance.
                         "rl_game_details": rl_details,
                         "sft_game_details": sft_details})

    ci = H.paired_bootstrap_ci(deltas, n_resamples=args.n_resamples)
    gate_pass = ci["ci_low"] > args.threshold
    # Informational baseline (step 0): a true pre-train policy ≈ SFT, so
    # delta ≈ 0 and the +2 gate is EXPECTED to "fail". That is not a run
    # failure — it must NOT poison rc / the spend decision. status reflects
    # the gate; `informational` + rc 0 (in main) keep it out of aggregation.
    passed = gate_pass or args.informational
    # A passing gate is necessary but NOT sufficient for a final spend claim.
    # Surface the standing caveats so a green ci_low is never over-read.
    n_eval = len(_eval_grid(args))
    final_gate_caveats = []
    if n_eval < 60:
        final_gate_caveats.append(
            f"eval N={n_eval} < the RL plan §3.6 N=60 — fine for a seam/smoke "
            f"read, NOT a final spend/no-spend claim.")
    # The eval-determinism caveat is GONE if we ran greedy (T=0) OR if the
    # CI tightening from K-sample averaging is large enough to dominate the
    # Δ≈0.116 per-game noise floor; otherwise it still applies.
    # Greedy alone clears the determinism caveat ONLY vs the deterministic
    # heuristic; a hosted API opponent is not deterministic at T=0, so only
    # K-sample averaging (K≥4) clears it there.
    eval_det_addressed = (
        (opp_is_det and temperature is not None
         and float(temperature) == 0.0)
        or K_effective >= 4
    )
    if not eval_det_addressed:
        final_gate_caveats.append(
            "eval-determinism is unaddressed here (measured: Δ≈0.116 ≈ the "
            "det=False effect size) — a final gate needs deterministic "
            "eval (--temperature 0 vs the heuristic) or K-sample averaging "
            "(--eval-samples-per-seed ≥4) to dominate that noise.")
    if K_note:
        final_gate_caveats.append(K_note)
    # A non-heuristic eval opponent: pairing is only approximate, and the +2
    # threshold was calibrated against the heuristic — surface both.
    if not opp_is_det:
        final_gate_caveats.append(
            f"eval opponent {opp_model!r} is a hosted API model — pairing is "
            f"approximate (provider-side nondeterminism even at T=0); rely on "
            f"K-sample averaging (--eval-samples-per-seed >=4).")
    opp_threshold_stale = (
        not opp_is_det and abs(args.threshold - GATE_THRESHOLD) < 1e-9)
    if opp_threshold_stale:
        final_gate_caveats.append(
            f"--threshold is still the heuristic-calibrated +{GATE_THRESHOLD}"
            f", NOT valid vs {opp_model!r} — re-baseline (measure "
            f"SFT-vs-opponent first) and pass an explicit --threshold.")
    valid_final = (gate_pass and not args.informational and n_eval >= 60
                   and not opp_threshold_stale)

    # Wandb (Plan §A.6) — two complementary writes:
    #  (1) per-label scalars to `wandb.summary` keep the existing one-shot
    #      summary panel intact (`eval/step50/ci_low`, etc) for run-table
    #      comparisons across replicates.
    #  (2) the same scalars as a TIME SERIES on a custom `eval/step` x-axis
    #      so the wandb chart panel renders a learning curve. wandb's
    #      built-in global step is the training-step counter, which has
    #      already advanced past these checkpoint numbers by the time the
    #      post-hoc intermediate evals run — hence a separate x-axis.
    # Per-seed deltas live in eval_step{N}.json on disk (`per_game` field);
    # the wandb.Table was duplicative and cluttered the eval panels, so it's
    # dropped here. Inspect the per-seed JSON directly for that detail.
    if _wandb_run is not None:
        try:
            import re as _re
            import wandb
            lbl = str(args.label or "final")
            wandb.summary[f"eval/{lbl}/mean_delta"] = float(ci["mean_delta"])
            wandb.summary[f"eval/{lbl}/ci_low"] = float(ci["ci_low"])
            wandb.summary[f"eval/{lbl}/ci_high"] = float(ci["ci_high"])
            wandb.summary[f"eval/{lbl}/n"] = int(ci["n"])
            wandb.summary[f"eval/{lbl}/passed_gate"] = bool(gate_pass)
            wandb.summary[f"eval/{lbl}/valid_as_final_evidence"] = bool(
                valid_final)
            wandb.summary[f"eval/{lbl}/threshold"] = float(args.threshold)

            # Time-series: parse step from label. `stepN` → N; `final` →
            # `args.total_steps` (when provided); `*_trainseeds` variants
            # go to a separate `eval_trainseeds/` namespace so the main
            # chart isn't contaminated by the secondary panel.
            trainseeds = lbl.endswith("_trainseeds")
            base_lbl = lbl[:-len("_trainseeds")] if trainseeds else lbl
            eval_step: int | None = None
            if base_lbl == "final" and args.total_steps is not None:
                eval_step = int(args.total_steps)
            elif base_lbl.startswith("step"):
                m = _re.match(r"^step(\d+)$", base_lbl)
                if m:
                    eval_step = int(m.group(1))
            if eval_step is not None:
                ns = "eval_trainseeds" if trainseeds else "eval"
                wandb.define_metric(f"{ns}/step", hidden=True)
                for k in ("mean_delta", "ci_low", "ci_high",
                          "passed_gate", "n"):
                    wandb.define_metric(
                        f"{ns}/{k}", step_metric=f"{ns}/step")
                wandb.log({
                    f"{ns}/step":        eval_step,
                    f"{ns}/mean_delta":  float(ci["mean_delta"]),
                    f"{ns}/ci_low":      float(ci["ci_low"]),
                    f"{ns}/ci_high":     float(ci["ci_high"]),
                    f"{ns}/passed_gate": int(gate_pass),
                    f"{ns}/n":           int(ci["n"]),
                })
        except Exception:  # noqa: BLE001 — telemetry never aborts an eval
            pass

    return {
        "label": args.label, "mode": "GPU",
        "status": "PASS" if passed else "FAIL",
        "informational": args.informational,
        "gate_pass": gate_pass,
        "valid_as_final_evidence": valid_final,
        "final_gate_caveats": final_gate_caveats,
        "gate": f"ci_low > +{args.threshold}",
        "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
        "mean_delta": ci["mean_delta"], "n": ci["n"],
        "threshold": args.threshold, "passed_gate": gate_pass,
        "n_resamples": args.n_resamples,
        "per_game": per_game,
        "eval_seeds": _eval_grid(args), "trainable_seat": seat,
        "value_chart": args.value_chart,
        "rl_served": args.rl_served, "sft_served": args.sft_served,
        "rl_adapter_path": args.rl_adapter_path,
        "adapter_sync": adapter_sync,
        "max_parallel": max_parallel,
        "eval_samples_per_seed": K_effective,
        "temperature": temperature,
        "eval_opponent": args.eval_opponent,
        "opponent_model": opp_model,
    }


def _dry_run(args) -> dict:
    """CPU, no GPU/net: (1) ENGINE terminal-score extraction
    (`final_scores[pid].final_score`, hands revealed) works on real
    transcripts AND differs from the hand-free mid-game proxy (regression
    guard for the #1 fix); (2) the paired-bootstrap +2 gate is not a rubber
    stamp; (3) pairing isolates the policy; (4) `--informational` keeps a
    failing baseline gate out of rc."""
    corpus = sorted(glob.glob(
        "results/phase1_corpus/chart_A/**/*.json", recursive=True))[:3]
    if not corpus:
        corpus = sorted(glob.glob("tests/fixtures/*.json"))[:3]
    scores, matches_engine, differs_from_proxy = [], True, False
    from megagem.rl.scorer import score_at  # proxy, for the regression guard only
    for f in corpus:
        g = json.load(open(f))
        seat = args.trainable_seat
        eng = _terminal_score(g, seat)
        scores.append(eng)
        raw = {e["player_id"]: e["final_score"]
               for e in g["final_results"]["final_scores"]}[seat]
        if eng != raw:
            matches_engine = False  # must read final_scores verbatim
        p = parse_transcript(g)
        if eng != score_at(p, p.num_rounds - 1, seat):
            differs_from_proxy = True  # confirms we are NOT on the proxy
    # ≥1 of 3 must differ from the mid-game proxy (corpus-wide it is 131/144);
    # if they never differed we'd be silently back on the broken path.
    score_ok = (len(scores) == len(corpus)
                and all(isinstance(x, int) for x in scores)
                and matches_engine and differs_from_proxy)

    pos = H.paired_bootstrap_ci([3.0, 4.0, 2.5, 5.0, 3.5, 2.2, 6.0, 3.0],
                                n_resamples=2000)
    zero = H.paired_bootstrap_ci([0.1, -0.2, 0.0, 0.3, -0.1, 0.05],
                                 n_resamples=2000)
    gate_pos = pos["ci_low"] > GATE_THRESHOLD          # must PASS
    gate_zero = not (zero["ci_low"] > GATE_THRESHOLD)  # must FAIL the +2 gate

    # pairing: same seed → same opponent game → delta isolates policy
    opp_model = _opponent_model(args.eval_opponent)
    pairing_ok = True
    try:
        for s in _eval_grid(args):
            opp = [opp_model] * args.num_players
            a = list(opp); a[args.trainable_seat] = "RL"
            b = list(opp); b[args.trainable_seat] = "SFT"
            if a == b or a[args.trainable_seat] == b[args.trainable_seat]:
                pairing_ok = False
    except Exception:
        pairing_ok = False

    # informational mode must turn a failing gate into a non-failing run
    info_ok = (GATE_THRESHOLD > 0)  # sanity: +2 gate (not >0)
    passed = score_ok and gate_pos and gate_zero and pairing_ok and info_ok
    return {
        "label": args.label,
        "mode": "DRY-RUN (CPU — engine terminal score + gate + pairing)",
        "status": "PASS" if passed else "FAIL",
        "terminal_score_ok": score_ok,
        "uses_engine_final_scores_not_proxy": (
            matches_engine and differs_from_proxy),
        "sample_engine_scores": scores,
        "gate_positive_set_passes": gate_pos,
        "gate_zero_set_fails": gate_zero,
        "pairing_isolates_policy": pairing_ok,
        "threshold": GATE_THRESHOLD,
        "eval_opponent": args.eval_opponent,
        "pos_ci_low": pos["ci_low"], "zero_ci_low": zero["ci_low"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__)
    p.add_argument("--label", default="final",
                   help="checkpoint label (step0 / final / step25).")
    p.add_argument("--rl-served", default="phase3-trainable",
                   help="vLLM served-model-name for the RL checkpoint "
                        "(the pushed LoRA adapter).")
    p.add_argument("--sft-served", default="qwen/qwen3-4b-instruct",
                   help="served-model-name for frozen SFT1200-v2.")
    p.add_argument("--rl-adapter-path", default=None,
                   help="if set, push this LoRA adapter dir to the rollout "
                        "vLLM under --rl-served before evaluating (lets one "
                        "vLLM serve step0 vs final).")
    p.add_argument("--vllm-url", default=None)
    p.add_argument("--heuristic-url", default=None)
    p.add_argument("--eval-opponent", default="heuristic",
                   help="opponent for ALL non-trainable eval seats. "
                        "'heuristic' (default — deterministic, exact pairing, "
                        "comparable to the N=5 history) or a model id "
                        "resolvable via megagem.endpoints (e.g. "
                        "google/gemini-3-flash-preview via PRIME_API_KEY). A "
                        "non-heuristic opponent requires re-baselining "
                        "--threshold (measure SFT-vs-opponent first).")
    p.add_argument("--eval-seeds", type=int, default=24,
                   help="fixed paired eval conditions (chart A).")
    p.add_argument("--seed-start", type=int, default=20000,
                   help="distinct from training seeds (no leakage).")
    p.add_argument("--num-players", type=int, default=3)
    p.add_argument("--trainable-seat", type=int, default=0)
    p.add_argument("--value-chart", default="A")
    p.add_argument("--n-resamples", type=int, default=10000)
    p.add_argument("--threshold", type=float, default=GATE_THRESHOLD)
    p.add_argument("--max-parallel", type=int, default=32,
                   help="max concurrent games (both arms share the same "
                        "semaphore). vLLM batches concurrent requests; ~5–8× "
                        "wall-time speedup at no GPU cost. Set to 1 to "
                        "restore the old sequential behaviour (fallback if "
                        "vLLM preempts under concurrency — check vllm_server.log).")
    p.add_argument("--eval-samples-per-seed", type=int, default=1,
                   help="Lever A — K-sample averaging. For each (seed, arm) "
                        "play K games, AVERAGE the K terminal scores before "
                        "pairing. var(delta_avg)=var(delta)/K → CI half-width "
                        "drops by √K at K× the eval-game cost. K=4 ⇒ ~2× CI "
                        "tightening, equivalent (in CI) to N=240 single-sample. "
                        "Auto-collapses to K=1 under --temperature 0 (greedy "
                        "replays are byte-identical).")
    p.add_argument("--temperature", type=float, default=None,
                   help="Lever C — symmetric T override for BOTH arms. T=0.0 "
                        "= greedy: both arms deterministic given state, "
                        "per-game policy-sampling variance collapses, CI "
                        "tightens dramatically. Measures the policy MODE not "
                        "the full trained distribution (a corroborating "
                        "low-noise signal, not a deployment-matched gate). "
                        "Unset ⇒ vLLM default (training-matched stochastic).")
    p.add_argument("--informational", action="store_true",
                   help="baseline (step 0) mode: report the gate but NEVER "
                        "fail the process on it (delta≈0 is EXPECTED for the "
                        "pre-train policy; it must not poison rc / spend).")
    p.add_argument("--total-steps", type=int, default=None,
                   help="Total training-step count for this run. Used ONLY to "
                        "place the `final` (and `final_trainseeds`) eval on "
                        "the `eval/step` wandb x-axis; without it, `final` "
                        "still logs to wandb.summary but does not appear in "
                        "the time-series chart. `stepN` labels self-locate "
                        "from the integer in the label.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.dry_run:
            result = _dry_run(args)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                result = _run_eval(args, tmp)
    except SystemExit as e:
        result = {"label": args.label, "status": "FAIL",
                  "reason": f"ABORT: {e}"}
    except Exception as e:  # noqa: BLE001
        result = {"label": args.label, "status": "FAIL",
                  "reason": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    g = result
    tag = " [INFORMATIONAL — not gated]" if g.get("informational") else ""
    print(f"\n[{g.get('mode', '?')}] Phase 3.6 eval ({args.label}){tag} — "
          f"{g['status']}")
    if g.get("mode") == "GPU" and "mean_delta" in g:
        if g.get("informational"):
            verdict = "baseline (≈0 expected; informational only)"
        else:
            verdict = "SPEND" if g.get("gate_pass") else "DO NOT SPEND"
        print(f"  mean_delta={g['mean_delta']:.3f}  "
              f"ci_low={g['ci_low']:.3f}  gate: {g['gate']}  → {verdict}")
    if g["status"] != "PASS" and "traceback" in g:
        print(g["traceback"])
    print(f"→ {args.output}")
    # Informational runs never fail the process on the gate (the baseline
    # ≈0 delta must not poison rc/spend); a hard error (no mean_delta) still
    # fails. Non-informational: rc follows the gate.
    if g.get("informational"):
        return 0 if ("mean_delta" in g or g.get("mode", "").startswith("DRY")) else 1
    return 0 if g["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
