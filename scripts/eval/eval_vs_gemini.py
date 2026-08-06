#!/usr/bin/env python3
"""Sanity eval: policy (seat 0, vLLM-served) vs N-1 copies of an API opponent
(default Gemini 3 Flash). Reports win rate and checks it against an expected
baseline (~30% vs 2× Gemini 3 Flash for the SFT model).

Win convention matches scripts/eval/_aggregate_qwen_eval.py: a "win" is the policy
score *strictly greater* than the max opponent score (ties are not wins).

  python scripts/eval/eval_vs_gemini.py \\
      --model qwen/qwen3-4b-instruct \\
      --vllm-url http://localhost:8000/v1 \\
      --num-games 10 --output results/eval_vs_gemini.json

Prereqs:
  * vLLM already serving the policy (e.g. `vllm serve <repo> --served-model-name qwen/qwen3-4b-instruct`).
  * PRIME_API_KEY exported — Gemini routes through Prime Inference per
    megagem.endpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

from megagem import endpoints
from megagem.assets import asset_path
from megagem.evals.game_runner import register_vllm_endpoint
from megagem.rollout import run_game

DEFAULT_OPPONENT = "google/gemini-3-flash-preview"
EXPECTED_WIN_RATE = 0.30  # SFT model vs 2× Gemini 3 Flash, prior observation


def _mean(xs: list[float]) -> float | None:
    return round(statistics.mean(xs), 5) if xs else None


def _median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 5) if xs else None


def _rate(xs: list[bool]) -> float | None:
    return round(sum(1 for x in xs if x) / len(xs), 5) if xs else None


def _count_by(xs: list) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        key = str(x)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _summarize_ev_decisions(decisions: list[dict]) -> dict:
    """Aggregate the E1 ev_dist selector's per-decision telemetry: how often the
    gate fired, how far deviations moved the bid, and the predicted EV margins
    (the offline-vs-live calibration check runs on these post-hoc)."""
    if not decisions:
        return {"n_decisions": 0}
    dev = [d for d in decisions if (d.get("gate") or {}).get("passed")]
    margins = [float((d.get("gate") or {}).get("margin", 0.0)) for d in decisions]
    return {
        "n_decisions": len(decisions),
        "n_deviations": len(dev),
        "gate_pass_rate": round(len(dev) / len(decisions), 4),
        "mean_gate_margin": _mean(margins),
        "mean_abs_bid_change": _mean([abs(d["b_star"] - d["b_bp"]) for d in dev]) if dev else None,
        "mean_bid_change": _mean([d["b_star"] - d["b_bp"] for d in dev]) if dev else None,
        "mean_vhat": _mean([float(d.get("vhat", 0.0)) for d in decisions]),
        "mean_p_win_star": _mean([float(d.get("p_win_star", 0.0)) for d in dev]) if dev else None,
        "mean_p_win_bp": _mean([float(d.get("p_win_bp", 0.0)) for d in decisions]),
    }


def _summarize_pikl_decisions(decisions: list[dict]) -> dict:
    """Aggregate piKL intrinsics from run_game's per-decision payloads.

    The goal is to tell whether piKL was active and value-directed, independent
    of final win-rate noise: candidate diversity, prior diversity, Q headroom,
    policy movement away from tau, and confidence-gate behavior.
    """
    if not decisions:
        return {"n_decisions": 0}
    metrics = [d.get("metrics") or {} for d in decisions]

    def vals(name: str) -> list[float]:
        return [
            float(m[name]) for m in metrics
            if name in m and isinstance(m[name], (int, float))
        ]

    def bools(name: str) -> list[bool]:
        return [
            bool(m[name]) for m in metrics
            if name in m and isinstance(m[name], bool)
        ]

    gates = [d.get("gate") for d in decisions if d.get("gate")]
    out = {
        "n_decisions": len(decisions),
        "by_phase": _count_by([d.get("phase") for d in decisions]),
        "candidate_modes": _count_by([d.get("candidate_mode") for d in decisions if d.get("candidate_mode")]),
        "mean_n_candidates": _mean(vals("n_candidates")),
        "mean_tau_eff_n": _mean(vals("tau_eff_n")),
        "mean_tau_entropy_norm": _mean(vals("tau_entropy_norm")),
        "mean_pi_eff_n": _mean(vals("pi_eff_n")),
        "mean_kl_pi_tau": _mean(vals("kl_pi_tau")),
        "mean_tv_pi_tau": _mean(vals("tv_pi_tau")),
        "mean_q_spread": _mean(vals("q_spread")),
        "mean_q_top2_margin": _mean(vals("q_top2_margin")),
        "mean_q_best_lift_vs_tau": _mean(vals("q_best_lift_vs_tau")),
        "mean_chosen_lift_vs_tau": _mean(vals("chosen_lift_vs_tau")),
        "q_best_is_tau_mode_rate": _rate(bools("q_best_is_tau_mode")),
        "chosen_is_q_best_rate": _rate(bools("chosen_is_q_best")),
        # Tie-adjusted: chosen counts as best when its Q is within ε of the argmax
        # (value-equivalent tie members are not penalized by an arbitrary tie-break).
        "chosen_is_q_best_tieadj_rate": _rate(bools("chosen_is_q_best_tieadj")),
        "chosen_is_tau_mode_rate": _rate(bools("chosen_is_tau_mode")),
        "pi_mode_is_tau_mode_rate": _rate(bools("pi_mode_is_tau_mode")),
        "pi_mode_is_q_best_rate": _rate(bools("pi_mode_is_q_best")),
        "gate_n": len(gates),
    }
    if gates:
        def _lift(gs: list[dict]) -> list[float]:
            return [float(g["lift"]) for g in gs if isinstance(g.get("lift"), (int, float))]

        passed = [g for g in gates if g.get("passed")]
        failed = [g for g in gates if not g.get("passed")]
        out.update({
            "gate_pass_rate": _rate([bool(g.get("passed")) for g in gates]),
            "gate_passed_n": len(passed),
            "gate_failed_n": len(failed),
            "mean_gate_lift": _mean(_lift(gates)),
            # Passed-gate lift is the value-direction signal for the DEVIATING nodes
            # (failed-gate lift mostly reflects near-zero-headroom fallbacks).
            "gate_pass_lift_mean": _mean(_lift(passed)),
            "gate_pass_lift_median": _median(_lift(passed)),
            "gate_fail_lift_mean": _mean(_lift(failed)),
            "gate_fail_lift_median": _median(_lift(failed)),
            "mean_gate_threshold": _mean([
                float(g["threshold"]) for g in gates
                if isinstance(g.get("threshold"), (int, float))
            ]),
        })
    return out


async def play_one_game(
    models: list[str],
    seed: int,
    value_chart: str,
    num_players: int,
    results_dir: Path | None,
    pikl_config: dict | None = None,
    value_head_config: dict | None = None,
) -> dict:
    """Run one game; return seat-0 (policy) outcome. Never raises — a failed
    game is recorded with error=... and excluded from the win-rate denominator
    so one bad game can't tank the whole run."""
    json_filename = f"vs_gemini_seed{seed}.json" if results_dir is not None else None
    t0 = time.perf_counter()
    try:
        state = await run_game(
            models=models,
            value_chart=value_chart,
            seed=seed,
            num_players=num_players,
            output_file="trajectory" if results_dir is not None else None,
            json_filename=json_filename,
            quiet=True,
            game_label=f"vs_gemini seed={seed}",
            results_dir=results_dir,
            pikl_config=pikl_config,
            value_head_config=value_head_config,
        )
    except Exception as e:  # noqa: BLE001 — one game failing must not abort the sweep
        return {"seed": seed, "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.perf_counter() - t0, 2)}

    scores = {
        int(s["player_id"]): float(s["final_score"])
        for s in state["final_scores"]
        if "player_id" in s and "final_score" in s
    }
    if len(scores) != num_players or 0 not in scores:
        return {"seed": seed, "error": f"bad final_scores: {scores}",
                "elapsed_s": round(time.perf_counter() - t0, 2)}

    policy_score = scores[0]
    opp_max = max(v for k, v in scores.items() if k != 0)
    pikl_decisions = state.get("pikl_decision_metrics") or []
    out = {
        "seed": seed,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "num_rounds": state.get("num_rounds"),
        "winner_id": state.get("winner_id"),
        "policy_score": policy_score,
        "opponent_max": opp_max,
        "policy_delta": policy_score - opp_max,
        "win": policy_score > opp_max,  # strict — matches _aggregate_qwen_eval
        "pikl_metrics": _summarize_pikl_decisions(pikl_decisions),
        "pikl_decisions": pikl_decisions,
    }
    ev = state.get("ev_dist_decisions") or []
    if ev:
        out["ev_dist_decisions"] = ev
        out["ev_dist_metrics"] = _summarize_ev_decisions(ev)
    return out


async def run_eval(
    policy_model: str,
    opponent: str,
    num_games: int,
    max_parallel: int,
    seed_start: int,
    value_chart: str,
    num_players: int,
    results_dir: Path | None,
    pikl_config: dict | None = None,
    value_head_config: dict | None = None,
) -> dict:
    # Seat 0 = policy, remaining seats = opponent (2× Gemini for num_players=3).
    models = [policy_model] + [opponent] * (num_players - 1)
    sem = asyncio.Semaphore(max_parallel)

    async def with_sem(seed: int) -> dict:
        async with sem:
            return await play_one_game(
                models, seed, value_chart, num_players, results_dir, pikl_config,
                value_head_config,
            )

    seeds = list(range(seed_start, seed_start + num_games))
    t0 = time.perf_counter()
    per_game = await asyncio.gather(*(with_sem(s) for s in seeds))
    wall = time.perf_counter() - t0

    scored = [g for g in per_game if "error" not in g]
    errored = [g for g in per_game if "error" in g]
    n = len(scored)
    wins = sum(1 for g in scored if g["win"])
    win_rate = wins / n if n else None

    # Normal-approx 95% CI on the win rate (n is small — band is wide; this is
    # to judge whether the ~0.30 baseline is plausibly consistent, not a gate).
    ci = None
    consistent = None
    if n:
        se = math.sqrt(max(win_rate * (1 - win_rate), 1e-9) / n)
        lo, hi = max(0.0, win_rate - 1.96 * se), min(1.0, win_rate + 1.96 * se)
        ci = [round(lo, 3), round(hi, 3)]
        consistent = lo <= EXPECTED_WIN_RATE <= hi

    ep = endpoints.ENDPOINTS.get(policy_model, {})
    all_pikl_decisions = [
        d for game in scored for d in game.get("pikl_decisions", [])
    ]
    all_ev_decisions = [
        d for game in scored for d in game.get("ev_dist_decisions", [])
    ]
    return {
        "config": {
            "policy_model": policy_model,
            "policy_endpoint_url": ep.get("url"),
            "policy_served_name": ep.get("model"),
            "opponent": opponent,
            "seats": models,
            "num_games": num_games,
            "num_players": num_players,
            "max_parallel": max_parallel,
            "value_chart": value_chart,
            "seed_range": [seed_start, seed_start + num_games - 1],
            "pikl": pikl_config,
            "value_head": bool(value_head_config and value_head_config.get("enabled")),
        },
        "wall_clock_total_s": round(wall, 2),
        "n_scored": n,
        "n_errored": len(errored),
        "wins": wins,
        "win_rate": round(win_rate, 3) if win_rate is not None else None,
        "win_rate_95ci": ci,
        "mean_policy_score": round(statistics.mean(g["policy_score"] for g in scored), 2) if n else None,
        "mean_policy_delta": round(statistics.mean(g["policy_delta"] for g in scored), 2) if n else None,
        "expected_win_rate": EXPECTED_WIN_RATE,
        "consistent_with_expected": consistent,
        "pikl_metrics": _summarize_pikl_decisions(all_pikl_decisions),
        "ev_dist_metrics": _summarize_ev_decisions(all_ev_decisions),
        "per_game": per_game,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--model", required=True,
                   help="Policy model id (seat 0). With --vllm-url it's "
                        "registered as a local vLLM endpoint.")
    p.add_argument("--vllm-url", default=None)
    p.add_argument("--served-model-name", default=None)
    p.add_argument("--api-key-env", default="EMPTY")
    p.add_argument("--opponent", default=DEFAULT_OPPONENT,
                   help="Opponent model id for the non-policy seats.")
    p.add_argument("--num-games", type=int, default=10)
    p.add_argument("--max-parallel", type=int, default=4,
                   help="Concurrent games. Lower if the opponent API "
                        "rate-limits.")
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--value-chart", default="A")
    p.add_argument("--num-players", type=int, default=3)
    p.add_argument("--results-dir", type=Path, default=None,
                   help="If set, per-game trajectory JSON is written here.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--value-head", default=None,
                   help="Path to a value_head.pkl; injects calibrated value estimates "
                        "into seat-0's bid prompts (Phase 2). Seat 0 must be local vLLM.")
    # piKL reveal search (seat 0 must be a local vLLM model). Omit --pikl-lambda
    # to disable — that disabled run IS the exact blueprint control. --pikl-lambda inf
    # is a sampled-τ̂ (self-consistency) control, not the exact blueprint.
    p.add_argument("--pikl-lambda", type=float, default=None,
                   help="Enable depth-1 piKL reveal search at this λ (tanh-Q units).")
    p.add_argument("--pikl-n", type=int, default=16, help="τ̂ samples per reveal node.")
    p.add_argument("--pikl-temp", type=float, default=1.0, help="Blueprint sampling temperature for τ̂.")
    p.add_argument("--pikl-alpha", type=float, default=0.5, help="τ̂ Laplace smoothing.")
    p.add_argument("--pikl-m", type=int, default=8, help="Belief determinizations per reveal node.")
    p.add_argument("--pikl-seed", type=int, default=0,
                   help="Search RNG seed — independent of the game seed (which encodes hidden state).")
    p.add_argument("--pikl-max-parallel", type=int, default=4, help="Concurrent search continuations.")
    p.add_argument("--pikl-target", choices=("reveal", "bid", "ev_dist"), default="reveal",
                   help="piKL decision surface: reveal (Gate A), bid (Gate B), or the E1 "
                        "analytic expected-surplus selector (ev_dist; enables itself — no "
                        "--pikl-lambda needed).")
    # E1 ev_dist selector (--pikl-target ev_dist). Zero extra LLM calls; the env
    # pre-samples the blueprint bid and the selector deviates only past the gate.
    p.add_argument("--ev-model-path", default=str(asset_path("ev_dist_v1.pkl")),
                   help="EvDistModel artifact (default: packaged asset ev_dist_v1.pkl).")
    p.add_argument("--ev-value-head-path", default=str(asset_path("value_head.pkl")),
                   help="Value head pkl for V-hat (default: packaged asset value_head.pkl).")
    p.add_argument("--ev-gate-min", type=float, default=1.0,
                   help="Deviate from the blueprint bid only when EV(b*) - EV(b_bp) >= this (coins).")
    p.add_argument("--ev-no-mission-bonus", action="store_true",
                   help="Exclude the exact mission bonus from V-hat (D3-faithful ablation).")
    p.add_argument("--ev-pacing-lam", type=float, default=0.0,
                   help="Budget-pacing shadow price per coin: EV = (V' - (1+lam)·b)·p. "
                        "Prices the financing cascade (E1 stage-2 fix; sim-adjudicated).")
    p.add_argument("--ev-vhat-debias", type=float, default=0.0,
                   help="Subtract this from V-hat (measured off-corpus over-statement).")
    p.add_argument("--ev-value-refit-path", default="",
                   help="V2(b) gem-refit pickle: g(x) on private-hand/unseen-pool "
                        "features REPLACES the flat debias when set.")
    p.add_argument("--ev-pacing-schedule", default="",
                   help="V3 lambda(state): 'linear:a,b' | 'pace:lam0,kappa' "
                        "(overrides --ev-pacing-lam when set).")
    p.add_argument("--history-repair", action="store_true",
                   help="Seat-0 prompts gain a full-history per-opponent bidding profile "
                        "(E1 arm 3). Opponents' prompts are untouched.")
    p.add_argument("--persona-text", default="",
                   help="E2 persona gate: strategy-card suffix appended to EVERY seat's "
                        "system prompt (homogeneous persona tables). Empty = legacy.")
    p.add_argument("--pikl-value-aware", action="store_true",
                   help="Q continues with fair-value opponents (CPU heuristic) instead of the "
                        "blueprint self-model — the Gate-B operator (strongest var-screen signal).")
    p.add_argument("--pikl-fv-shade", type=float, default=0.8,
                   help="Fair-value bid shade for --pikl-value-aware.")
    p.add_argument("--pikl-bid-all-auctions", action="store_true",
                   help="Bid target: search every auction, not just treasures (default treasure-only).")
    p.add_argument("--pikl-opp-model", choices=("fair_value", "market"), default="fair_value",
                   help="Opponent bid model in the Q rollout + fv_opponent_seats: fair_value "
                        "(default, the locked arm) or market (recent-winning-bid + budget + "
                        "tiebreak; matches how the blueprint actually bids).")
    p.add_argument("--pikl-candidate-mode", choices=("sampled", "threshold", "all"), default="sampled",
                   help="Bid target: sampled support (legacy), threshold anchors, or all legal bids.")
    p.add_argument("--pikl-max-bid-candidates", type=int, default=0,
                   help="Cap bid candidates after widening; 0 keeps all candidates.")
    p.add_argument("--pikl-gate-min-lift", type=float, default=0.0,
                   help="Conservative piKL gate: require best-Q lift over tau baseline by this "
                        "many tanh-Q units before deviating.")
    p.add_argument("--pikl-gate-z", type=float, default=0.0,
                   help="Conservative piKL gate: additionally require z * SE of the Q lift.")
    p.add_argument("--pikl-lambda-mix", default="",
                   help="Comma-separated DiL-piKL lambda mixture, e.g. 'inf,0.3,0.1,0.03,0'. "
                        "If set, enables Q search even when --pikl-lambda is omitted.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_players < 2:
        print("FATAL: --num-players must be >= 2.", file=sys.stderr)
        return 2

    if args.vllm_url:
        register_vllm_endpoint(
            args.model, args.served_model_name, args.vllm_url, args.api_key_env
        )
    elif args.model not in endpoints.ENDPOINTS:
        print(f"FATAL: policy model {args.model!r} not in megagem.endpoints "
              "and no --vllm-url given.", file=sys.stderr)
        return 2

    # Hard preflight on the opponent: better to fail now than after spinning
    # games and burning the GPU/API budget on 401s.
    opp_ep = endpoints.ENDPOINTS.get(args.opponent)
    if opp_ep is None:
        print(f"FATAL: opponent {args.opponent!r} not in megagem.endpoints. "
              f"Known: {sorted(endpoints.ENDPOINTS)}", file=sys.stderr)
        return 2
    opp_key_env = opp_ep.get("key")
    if opp_key_env and opp_key_env != "EMPTY" and not os.getenv(opp_key_env):
        print(f"FATAL: opponent {args.opponent!r} needs ${opp_key_env} but it "
              "is not set in the environment.", file=sys.stderr)
        return 2

    lambda_mix = [
        float(x.strip()) for x in args.pikl_lambda_mix.split(",") if x.strip()
    ]
    pikl_config = None
    if args.pikl_target == "ev_dist":
        if endpoints.ENDPOINTS.get(args.model, {}).get("key") != "EMPTY":
            print("FATAL: ev_dist needs seat-0 to be a local vLLM model (pass "
                  "--vllm-url).", file=sys.stderr)
            return 2
        pikl_config = {
            "enabled": True, "target": "ev_dist",
            "temperature": args.pikl_temp, "max_parallel": args.pikl_max_parallel,
            "ev_model_path": args.ev_model_path,
            "ev_value_head_path": args.ev_value_head_path,
            "ev_gate_min": args.ev_gate_min,
            "ev_mission_bonus": not args.ev_no_mission_bonus,
            "ev_pacing_lam": args.ev_pacing_lam,
            "ev_vhat_debias": args.ev_vhat_debias,
            "ev_value_refit_path": args.ev_value_refit_path,
            "ev_pacing_schedule": args.ev_pacing_schedule,
        }
    elif args.pikl_lambda is not None or lambda_mix:
        if endpoints.ENDPOINTS.get(args.model, {}).get("key") != "EMPTY":
            print("FATAL: piKL needs seat-0 to be a local vLLM model (pass "
                  "--vllm-url) so search continuations don't hit the API.", file=sys.stderr)
            return 2
        pikl_config = {
            "enabled": True, "lambda": args.pikl_lambda if args.pikl_lambda is not None else math.inf,
            "n": args.pikl_n,
            "temperature": args.pikl_temp, "alpha": args.pikl_alpha,
            "m": args.pikl_m, "max_parallel": args.pikl_max_parallel, "seed": args.pikl_seed,
            "target": args.pikl_target, "value_aware": args.pikl_value_aware,
            "fv_shade": args.pikl_fv_shade, "bid_treasure_only": not args.pikl_bid_all_auctions,
            "opp_model": args.pikl_opp_model,
            "candidate_mode": args.pikl_candidate_mode,
            "max_bid_candidates": args.pikl_max_bid_candidates,
            "gate_min_lift": args.pikl_gate_min_lift,
            "gate_z": args.pikl_gate_z,
            "lambda_mix": lambda_mix,
        }

    if args.history_repair:
        from megagem.environment import prompts as _prompts
        _prompts.HISTORY_REPAIR_SEATS = {0}
        print("History repair ON (seat 0): full-history opponent bidding profile in prompts")

    if args.persona_text:
        from megagem.environment import prompts as _prompts
        _prompts.PERSONA_SUFFIX = args.persona_text
        print(f"Persona ON (all seats): {args.persona_text[:80]!r}")

    value_head_config = None
    if args.value_head:
        if endpoints.ENDPOINTS.get(args.model, {}).get("key") != "EMPTY":
            print("FATAL: --value-head needs seat-0 to be a local vLLM model (pass "
                  "--vllm-url).", file=sys.stderr)
            return 2
        from megagem.value_head.value_estimator import ValueEstimator
        value_head_config = {"enabled": True,
                             "estimator": ValueEstimator.load(args.value_head),
                             "seats": [0]}
        print(f"Value head ON (seat 0): {args.value_head}")

    summary = asyncio.run(run_eval(
        policy_model=args.model,
        opponent=args.opponent,
        num_games=args.num_games,
        max_parallel=args.max_parallel,
        seed_start=args.seed_start,
        value_chart=args.value_chart,
        num_players=args.num_players,
        results_dir=args.results_dir,
        pikl_config=pikl_config,
        value_head_config=value_head_config,
    ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    print(f"\nPolicy: {args.model}")
    print(f"Opponent: {args.opponent} × {args.num_players - 1}  "
          f"(value_chart={args.value_chart})")
    if summary["n_errored"]:
        print(f"WARNING: {summary['n_errored']} game(s) errored and were "
              "excluded — see per_game[].error in the output JSON.")
    if not summary["n_scored"]:
        print("FAIL: no games scored.")
        return 1
    print(f"Win rate: {summary['win_rate']:.1%}  "
          f"({summary['wins']}/{summary['n_scored']}), "
          f"95% CI {summary['win_rate_95ci']}")
    print(f"Mean policy score: {summary['mean_policy_score']}  "
          f"(mean delta vs best opponent: {summary['mean_policy_delta']:+})")
    pm = summary.get("pikl_metrics") or {}
    if pm.get("n_decisions"):
        print("piKL intrinsics: "
              f"decisions={pm.get('n_decisions')}  "
              f"cands={pm.get('mean_n_candidates')}  "
              f"tau_eff={pm.get('mean_tau_eff_n')}  "
              f"Q_lift={pm.get('mean_q_best_lift_vs_tau')}  "
              f"chosen_lift={pm.get('mean_chosen_lift_vs_tau')}  "
              f"KL={pm.get('mean_kl_pi_tau')}  "
              f"gate_pass={pm.get('gate_pass_rate')}")
        if pm.get("gate_n"):
            print("piKL gate: "
                  f"pass={pm.get('gate_passed_n')}/{pm.get('gate_n')}  "
                  f"pass_lift(mean/med)={pm.get('gate_pass_lift_mean')}/{pm.get('gate_pass_lift_median')}  "
                  f"fail_lift(mean/med)={pm.get('gate_fail_lift_mean')}/{pm.get('gate_fail_lift_median')}  "
                  f"chosen_is_best={pm.get('chosen_is_q_best_rate')}  "
                  f"chosen_is_best_tieadj={pm.get('chosen_is_q_best_tieadj_rate')}")
    print(f"Expected baseline: {EXPECTED_WIN_RATE:.0%} — "
          f"{'CONSISTENT (within 95% CI)' if summary['consistent_with_expected'] else 'OUTSIDE 95% CI — investigate'}")
    print(f"Wall: {summary['wall_clock_total_s']}s  →  {args.output}")
    # Exit 0 even if outside the band: this is a diagnostic, not a gate. A
    # nonzero rc is reserved for "couldn't run the eval at all".
    return 0


if __name__ == "__main__":
    sys.exit(main())
