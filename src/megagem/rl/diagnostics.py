"""Reward-balance diagnostics + λ re-derivation (the RL plan §A.4 / Caveat 1).

Re-derives λ from a real corpus rather than the back-of-envelope: pre-λ
telescoped shaping vs §A.0 terminal mass (``trajectory_breakdowns``,
``shaping_terminal_mass``), and a sweep gated on both mass-ratio target and
within-group rank preservation (``lambda_sweep``). Mirrors the production
reward path (round_shaping / terminal_breakdowns / compute_advantages); pure
stdlib, CPU-only. Needs a K≥2 corpus to gate rank preservation.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Sequence

from .advantage import AdvantageConfig, compute_advantages
from .reward import (
    RewardConfig,
    calibrate_scale,
    per_turn_rewards,
    round_shaping,
    terminal_breakdowns,
)
from .scorer import parse_transcript

_DEFAULT_LAMBDAS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)


def _pct(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(p * len(sorted_vals)))]


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman ρ via average-rank + Pearson (no numpy); None when undefined."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None

    def ranks(vals: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie block
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    if dx <= 0 or dy <= 0:
        return None
    return num / (dx ** 0.5 * dy ** 0.5)


@dataclass(frozen=True)
class TrajectoryBreakdown:
    game_id: int
    player_id: int
    group_key: tuple  # ((seed, value_chart), pid) — matches TurnAdvantage
    s_raw: float  # pre-λ telescoped mid-game margin = Σ_round raw shaping
    true_margin: float  # engine-true §A.0 terminal margin (post-reveal)
    r_terminal: float
    is_winner: bool


def trajectory_breakdowns(
    transcripts: Sequence[dict[str, Any]], reward_cfg: RewardConfig | None = None
) -> list[TrajectoryBreakdown]:
    """Per (game, player): pre-λ telescoped shaping vs §A.0 terminal.

    ``S_raw`` is computed through the *same* ``round_shaping`` the reward
    pipeline uses, at λ=1 (so it is exactly Σ shaping / λ), not a re-derived
    formula — the diagnostic must not diverge from the production path.
    """
    reward_cfg = reward_cfg or RewardConfig()
    # replace (not a 4-field rebuild) so Phase-3 knobs (terminal_shape,
    # reveal_shaping_weight, terminal_correction, correction_scale) survive.
    raw_cfg = replace(reward_cfg, shaping_lambda=1.0)  # raw (pre-λ) shaping
    out: list[TrajectoryBreakdown] = []
    for gid, data in enumerate(transcripts):
        parsed = parse_transcript(data)
        rs = round_shaping(parsed, raw_cfg)
        term = terminal_breakdowns(parsed, reward_cfg)
        s_raw: dict[int, float] = defaultdict(float)
        for (_r_idx, pid), v in rs.items():
            s_raw[pid] += v
        for pid, bd in term.items():
            out.append(
                TrajectoryBreakdown(
                    game_id=gid,
                    player_id=pid,
                    group_key=((parsed.seed, parsed.value_chart), pid),
                    s_raw=s_raw.get(pid, 0.0),
                    true_margin=bd.margin,
                    r_terminal=bd.r_terminal,
                    is_winner=bd.is_winner,
                )
            )
    return out


def _shaping_channel_sums(
    transcripts: Sequence[dict[str, Any]], reward_cfg: RewardConfig
) -> dict[tuple[int, int], float]:
    """Per (game_id, player_id): the *actual* shaping-channel mass that flows
    into the trajectory — ``Σ_turns (shaping + terminal_correction)`` — read
    straight from the production ``per_turn_rewards`` path (§7.4: no
    re-derivation).

    Without rec 1a ``terminal_correction ≡ 0`` so this is exactly ``λ·S_raw``
    (the mass gate is byte-unchanged for the locked default). WITH rec 1a it
    is the *corrected, telescoped* mass — the quantity the λ mass-gate must
    actually weigh; ``λ·|S_raw|`` alone measures the uncorrected biased
    mid-game proxy, which the 1a/all ablations no longer feed to the policy.
    ``game_id`` is the ``enumerate`` index, matching ``trajectory_breakdowns``.
    """
    out: dict[tuple[int, int], float] = defaultdict(float)
    for gid, data in enumerate(transcripts):
        parsed = parse_transcript(data)
        for t in per_turn_rewards(parsed, reward_cfg):
            out[(gid, t.player_id)] += t.shaping + t.terminal_correction
    return dict(out)


def shaping_terminal_mass(
    transcripts: Sequence[dict[str, Any]], reward_cfg: RewardConfig | None = None
) -> dict[str, Any]:
    """Caveat 1 recommendation #1: the panel the Phase 1 report was missing.

    Per-trajectory shaping-channel mass ``|Σ(shaping+terminal_correction)|``
    vs terminal mass ``|R_terminal|``, and corr(mid-game proxy margin,
    engine-true margin) — i.e. *is the biased mid-game scorer even faithful to
    the true outcome it stands in for?* The mass numerator is the **actual**
    channel the policy sees: with rec 1a on it is the corrected/telescoped
    mass, not the uncorrected ``λ·|S_raw|`` proxy (which 1a replaces).
    """
    reward_cfg = reward_cfg or RewardConfig()
    bks = trajectory_breakdowns(transcripts, reward_cfg)
    if not bks:
        return {"trajectories": 0}
    lam = reward_cfg.shaping_lambda
    chan = _shaping_channel_sums(transcripts, reward_cfg)
    shaping = [abs(chan.get((b.game_id, b.player_id), 0.0)) for b in bks]
    terminal = [abs(b.r_terminal) for b in bks]
    ratios = sorted(s / t for s, t in zip(shaping, terminal) if t > 1e-12)
    s_raw = [b.s_raw for b in bks]
    true_m = [b.true_margin for b in bks]
    corr = _spearman(s_raw, true_m)
    return {
        "trajectories": len(bks),
        "lambda": lam,
        "scale": reward_cfg.scale,
        "shaping_mass_median": round(statistics.median(shaping), 4),
        "terminal_mass_median": round(statistics.median(terminal), 4),
        "shaping_over_terminal_median": (
            round(statistics.median(ratios), 3) if ratios else None
        ),
        "shaping_over_terminal_p90": (
            round(_pct(ratios, 0.90), 3) if ratios else None
        ),
        "frac_shaping_exceeds_terminal": (
            round(sum(1 for r in ratios if r > 1.0) / len(ratios), 3)
            if ratios
            else None
        ),
        "corr_midgame_vs_true_margin": (
            round(corr, 4) if corr is not None else None
        ),
    }


def _group_rank_corr(
    rows: Sequence[Any], margin_by: dict[tuple, float]
) -> dict[str, Any]:
    """Within each K-group: does mean trainable advantage rank the rollouts the
    same way their *true* terminal margin does? (Spearman, averaged over groups.)

    This is the operational form of "terminal dominates": shaping may reduce
    variance, but it must not flip which rollout the policy is pushed toward.
    """
    by_group: dict[tuple, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        if r.trainable:
            by_group[r.group_key][r.game_id].append(r.advantage)

    corrs: list[float] = []
    groups_total = len(by_group)
    for gkey, by_game in by_group.items():
        if len(by_game) < 2:
            continue
        pid = gkey[1]  # TurnAdvantage.group_key == ((seed, chart), pid)
        gids = sorted(by_game)
        adv = [statistics.fmean(by_game[g]) for g in gids]
        tm: list[float] = []
        ok = True
        for g in gids:
            if (g, pid) not in margin_by:
                ok = False
                break
            tm.append(margin_by[(g, pid)])
        if not ok:
            continue
        c = _spearman(adv, tm)
        if c is not None:
            corrs.append(c)
    return {
        "groups_total": groups_total,
        "groups_usable": len(corrs),
        "rank_corr_mean": round(statistics.fmean(corrs), 4) if corrs else None,
        "rank_corr_min": round(min(corrs), 4) if corrs else None,
    }


def _recommend(
    sweep: list[dict[str, Any]], ratio_target: float, rank_corr_floor: float
) -> dict[str, Any]:
    res = sorted(sweep, key=lambda r: r["lambda"])
    have_rank = any(r["rank_corr_mean"] is not None for r in res)
    chosen: float | None = None
    for r in res:  # ascending λ → keep the largest that passes both gates
        mass = r["shaping_over_terminal_median"]
        if mass is None or mass > ratio_target:
            continue
        if have_rank and (
            r["rank_corr_mean"] is None or r["rank_corr_mean"] < rank_corr_floor
        ):
            continue
        chosen = r["lambda"]
    if have_rank:
        basis = (
            f"largest λ with shaping/terminal median ≤ {ratio_target} AND "
            f"within-group rank-corr ≥ {rank_corr_floor}"
        )
    else:
        basis = (
            f"mass-ratio only (shaping/terminal median ≤ {ratio_target}); "
            "rank-corr UNAVAILABLE — no K≥2 group in this corpus. Retrieve the "
            "off-box true-K=8 Phase 1 corpus to gate on rank preservation."
        )
    note = None
    if chosen is None:
        note = (
            f"no swept λ meets the gate(s); the smallest tried "
            f"({res[0]['lambda']}) is still too large — sweep lower."
        )
    return {
        "lambda": chosen,
        "basis": basis,
        "rank_corr_available": have_rank,
        "note": note,
    }


def lambda_sweep(
    transcripts: Sequence[dict[str, Any]],
    lambdas: Sequence[float] = _DEFAULT_LAMBDAS,
    base_reward_cfg: RewardConfig | None = None,
    adv_cfg: AdvantageConfig | None = None,
    ratio_target: float = 0.33,
    rank_corr_floor: float = 0.8,
) -> dict[str, Any]:
    """Sweep λ; recommend the largest that keeps shaping a variance-reducer.

    ``ratio_target`` 0.33 ⇒ shaping ≤ 1/3 of terminal at the median (terminal
    ≥ 3x shaping). ``rank_corr_floor`` 0.8 ⇒ the standardized advantage still
    ranks rollouts ~by true outcome within the K-group.
    """
    transcripts = list(transcripts)
    base = base_reward_cfg or RewardConfig()
    adv_cfg = adv_cfg or AdvantageConfig()
    # True terminal margin is λ-independent → compute once, align by game_id
    # (compute_advantages enumerates the same list in the same order).
    margin_by = {
        (b.game_id, b.player_id): b.true_margin
        for b in trajectory_breakdowns(transcripts, base)
    }
    sweep: list[dict[str, Any]] = []
    for lam in lambdas:
        # replace (not a 4-field rebuild) so the ablation cfg's Phase-3 knobs
        # (terminal_shape / reveal_shaping_weight / terminal_correction /
        # correction_scale) thread through every swept λ.
        rc = replace(base, shaping_lambda=lam)
        mass = shaping_terminal_mass(transcripts, rc)
        rk = _group_rank_corr(compute_advantages(transcripts, rc, adv_cfg), margin_by)
        sweep.append(
            {
                "lambda": lam,
                "shaping_over_terminal_median": mass.get(
                    "shaping_over_terminal_median"
                ),
                "frac_shaping_exceeds_terminal": mass.get(
                    "frac_shaping_exceeds_terminal"
                ),
                "rank_corr_mean": rk["rank_corr_mean"],
                "rank_corr_min": rk["rank_corr_min"],
                "groups_usable": rk["groups_usable"],
            }
        )
    return {
        "sweep": sweep,
        "ratio_target": ratio_target,
        "rank_corr_floor": rank_corr_floor,
        "recommendation": _recommend(sweep, ratio_target, rank_corr_floor),
    }


def reveal_policy_diagnostic(
    transcripts: Sequence[dict[str, Any]],
    reward_cfg: RewardConfig | None = None,
    *,
    selfish_mean_floor: float = 0.0,
    payoff_corr_band: float = 0.15,
    min_reveal_turns: int = 30,
) -> dict[str, Any]:
    """Caveat-2 addendum — reveal-shaping **telemetry** (NOT a causal gate).

    CPU-only; hand-free; mirrors the production reward path per §7.4. For
    every **policy** reveal turn (treasure-auction winner whose reveal parsed
    *and* was legal — defaulted/illegal reveals are not a choice and are
    counted but excluded) we record the realised ``reveal_shaping`` and pair
    it with that revealer's engine-true §A.0 terminal margin.

    ⚠️ **CONFOUND — read before interpreting.** ``reveal_shaping`` is a
    function of *game state*, not only the policy's choice: it is large/
    positive mainly when the revealer already holds big collections in the
    revealed colour. A strong position therefore produces both larger
    ``reveal_shaping`` *and* a higher terminal margin **independently of
    which gem was chosen**. Consequently
    ``corr(reveal_shaping, terminal_margin)`` is *confounded by player
    strength* and is **not** a causal test of "is the policy chasing the
    proxy": it can read positive for a fully selfish policy (strong states →
    positive both) and near-zero from corpus composition alone. This function
    is **observational trend telemetry**, not a decision rule.

    Reported (all descriptive, none causal):

    * ``mean_signed_reveal_shaping`` / ``frac_self_favoring_of_nonzero`` —
      does realised reveal shaping skew self-favouring? The **actionable**
      use is the *trend* across training iterations measured on a **fixed
      held-out eval corpus** (same seeds/opponents every iteration) — that
      holds the state distribution roughly constant so a rising mean is more
      attributable to policy drift (still weakly confounded by the policy
      reaching stronger states as it improves; not eliminated).
    * ``corr_reveal_shaping_vs_true_margin`` — a *descriptive* association
      only; do not read it as "reveal carries (or lacks) real value".

    The confounder-free causal test is a *within-state counterfactual*
    (compare the chosen reveal against the winner's other legal hand gems);
    that necessarily reads the hand and is out of scope for this hand-free
    probe (self-play caveat 2 addendum).
    """
    reward_cfg = reward_cfg or RewardConfig()
    shp: list[float] = []  # policy reveal-turn reveal_shaping
    margin: list[float] = []  # that revealer's engine-true terminal margin
    n_reveal = n_default = 0
    for data in transcripts:
        parsed = parse_transcript(data)
        term = terminal_breakdowns(parsed, reward_cfg)
        for t in per_turn_rewards(parsed, reward_cfg):
            if t.phase != "reveal":
                continue
            n_reveal += 1
            if not (t.parse_valid and t.legal_valid):
                n_default += 1  # not a policy choice — excluded from signal
                continue
            shp.append(t.shaping)
            margin.append(term[t.player_id].margin if t.player_id in term else 0.0)

    nonzero = [s for s in shp if abs(s) > 1e-12]
    frac_pos = (
        round(sum(1 for s in nonzero if s > 0) / len(nonzero), 4)
        if nonzero
        else None
    )
    mean_signed = round(statistics.fmean(shp), 6) if shp else None
    corr = _spearman(shp, margin) if len(shp) >= 2 else None
    corr = round(corr, 4) if corr is not None else None

    if len(shp) < min_reveal_turns:
        observation = (
            "insufficient data (need ≥%d policy reveal turns) — telemetry only"
            % min_reveal_turns
        )
    elif corr is None:
        observation = "undefined: no variance in reveal_shaping or margin"
    elif mean_signed is not None and mean_signed > selfish_mean_floor and abs(
        corr
    ) < payoff_corr_band:
        observation = (
            f"reveal_shaping skews self-favoring (mean {mean_signed} > "
            f"{selfish_mean_floor}) and is weakly associated with terminal "
            f"margin (|ρ| {abs(corr)} < {payoff_corr_band}). CONSISTENT WITH "
            "proxy-tracking but NOT proof — confounded by player strength "
            "(common cause of both). Track this on a fixed held-out eval "
            "corpus across iterations; confirm with the within-state "
            "counterfactual probe before acting (drop reveal sub-delta / PBRS)."
        )
    else:
        observation = (
            "no self-favoring-and-decoupled pattern at this corpus "
            "(observational, confounded — not exculpatory)"
        )

    return {
        "policy_reveal_turns": len(shp),
        "reveal_turns_total": n_reveal,
        "reveal_turns_defaulted_excluded": n_default,
        "frac_zero_effect": (
            round((len(shp) - len(nonzero)) / len(shp), 4) if shp else None
        ),
        "frac_self_favoring_of_nonzero": frac_pos,
        "mean_signed_reveal_shaping": mean_signed,
        "corr_reveal_shaping_vs_true_margin": corr,
        "selfish_mean_floor": selfish_mean_floor,
        "payoff_corr_band": payoff_corr_band,
        "method": (
            "observational association — CONFOUNDED by game state (player "
            "strength is a common cause of reveal_shaping and terminal "
            "margin); trend telemetry on a fixed eval corpus, NOT a causal "
            "gate. Causal test = within-state counterfactual (reads hand)."
        ),
        "observation": observation,
    }


# ===========================================================================
# Phase-3 reward-design pre-spend gate.
# All CPU-only, hand-free, reuse the production path (§7.4). Report-only:
# nothing here mutates a locked default; the runner never flips config.
# ===========================================================================

_REVEAL_SETTINGS = (1.0, 0.0)  # rec 3: reveal on (locked) vs off


def telescope_identity(
    transcripts: Sequence[dict[str, Any]],
    base_reward_cfg: RewardConfig | None = None,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Rec 1a verifier (diagnostic *a*).

    With ``terminal_correction`` ON, per (game, player) check the exact
    telescoping identity

        Σ_turns (shaping + terminal_correction) == (λ / Sc) · true_margin

    (``const ≡ 0``; ``Sc = correction_scale or scale``). Run under
    ``reveal_shaping_weight ∈ {1.0, 0.0}`` — the ``0.0`` pass is precisely the
    Rev-7 check that the correction was *recomputed against the reveal-zeroed
    sum*, not bolted on. Σ is read from ``per_turn_rewards`` (the exact path
    training sums), so a pass here is a pass for the trained objective.
    """
    base = base_reward_cfg or RewardConfig()
    per_setting: dict[str, Any] = {}
    all_pass = True
    for w in _REVEAL_SETTINGS:
        cfg = replace(base, terminal_correction=True, reveal_shaping_weight=w)
        sc = cfg.correction_scale if cfg.correction_scale is not None else cfg.scale
        lam = cfg.shaping_lambda
        devs: list[float] = []
        worst = {"game_id": None, "player_id": None, "dev": 0.0}
        for gid, data in enumerate(transcripts):
            parsed = parse_transcript(data)
            term = terminal_breakdowns(parsed, cfg)
            acc: dict[int, float] = defaultdict(float)
            for t in per_turn_rewards(parsed, cfg):
                acc[t.player_id] += t.shaping + t.terminal_correction
            for pid, bd in term.items():
                expected = lam * (bd.margin / sc)
                dev = abs(acc.get(pid, 0.0) - expected)
                devs.append(dev)
                if dev > worst["dev"]:
                    worst = {"game_id": gid, "player_id": pid, "dev": dev}
        n_fail = sum(1 for d in devs if d > tol)
        max_dev = max(devs) if devs else 0.0
        passed = bool(devs) and max_dev <= tol
        all_pass = all_pass and passed
        per_setting["reveal_on" if w == 1.0 else "reveal_off"] = {
            "reveal_shaping_weight": w,
            "n_trajectories": len(devs),
            "max_abs_dev": max_dev,
            "mean_abs_dev": statistics.fmean(devs) if devs else 0.0,
            "n_fail": n_fail,
            "tol": tol,
            "pass": passed,
            "worst": worst,
        }
    return {"pass": all_pass, "tol": tol, **per_setting}


_PHASE3_ABLATIONS = ("baseline", "+1a", "+tanh", "+reveal0", "+all")


def _ablation_cfg(name: str, base: RewardConfig, scale_tanh: float) -> RewardConfig:
    if name == "baseline":
        return base
    if name == "+1a":
        return replace(base, terminal_correction=True)
    if name == "+tanh":
        return replace(base, terminal_shape="tanh", scale=scale_tanh)
    if name == "+reveal0":
        return replace(base, reveal_shaping_weight=0.0)
    if name == "+all":
        return replace(
            base,
            terminal_correction=True,
            terminal_shape="tanh",
            scale=scale_tanh,
            reveal_shaping_weight=0.0,
        )
    raise ValueError(f"unknown ablation {name!r}")


def ablation_lambda_relock(
    transcripts: Sequence[dict[str, Any]],
    lambdas: Sequence[float] = _DEFAULT_LAMBDAS,
    base_reward_cfg: RewardConfig | None = None,
    adv_cfg: AdvantageConfig | None = None,
    ratio_target: float = 0.33,
    rank_corr_floor: float = 0.8,
    scale_tanh: float | None = None,
) -> dict[str, Any]:
    """λ re-lock under each single-change ablation + the combined cut
    (diagnostic *b*).

    ``+all`` (1a + tanh@SCALE_tanh + reveal0) is the *final* shaping
    definition λ must be re-locked against (Rev 7 pt 3); the single-change
    rows make each change's marginal effect on the mass-ratio and the
    within-group rank-corr gate visible. tanh ablations re-lock at the
    *tanh-specific* SCALE (≈1.82·median, NOT 21.5) — reusing 21.5 would
    under-center and bias the re-lock.
    """
    transcripts = list(transcripts)
    base = base_reward_cfg or RewardConfig()
    if scale_tanh is None:
        scale_tanh = float(
            calibrate_scale(transcripts, replace(base, terminal_shape="tanh"))[
                "recommended_scale"
            ]
        )
    by_ablation: dict[str, Any] = {}
    table: list[dict[str, Any]] = []
    baseline_rec: float | None = None
    for name in _PHASE3_ABLATIONS:
        cfg = _ablation_cfg(name, base, scale_tanh)
        sweep = lambda_sweep(
            transcripts,
            lambdas,
            base_reward_cfg=cfg,
            adv_cfg=adv_cfg,
            ratio_target=ratio_target,
            rank_corr_floor=rank_corr_floor,
        )
        by_ablation[name] = sweep
        rec = sweep["recommendation"]["lambda"]
        if name == "baseline":
            baseline_rec = rec
        # mass ratio at the locked λ for an at-a-glance marginal read
        mass_at_lock = next(
            (
                r["shaping_over_terminal_median"]
                for r in sweep["sweep"]
                if abs(r["lambda"] - base.shaping_lambda) < 1e-12
            ),
            None,
        )
        table.append(
            {
                "ablation": name,
                "recommended_lambda": rec,
                "delta_vs_baseline": (
                    None
                    if rec is None or baseline_rec is None
                    else round(rec - baseline_rec, 6)
                ),
                "mass_ratio_at_locked_lambda": mass_at_lock,
                "rank_corr_floor_basis": sweep["recommendation"]["basis"],
            }
        )
    return {
        "scale_tanh": scale_tanh,
        "scale_clip": base.scale,
        "lambdas": list(lambdas),
        "table": table,
        "by_ablation": by_ablation,
        "final_cut": "+all",
    }


def _group_game_mean_adv(rows: Sequence[Any]) -> dict[tuple, dict[int, float]]:
    """Per (group_key, game_id) → mean trainable advantage (the same reduction
    ``_group_rank_corr`` uses, factored out for the clip-vs-tanh panel)."""
    by: dict[tuple, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        if r.trainable:
            by[r.group_key][r.game_id].append(r.advantage)
    return {
        gk: {g: statistics.fmean(v) for g, v in bg.items()}
        for gk, bg in by.items()
    }


def clip_vs_tanh_panel(
    transcripts: Sequence[dict[str, Any]],
    base_reward_cfg: RewardConfig | None = None,
    adv_cfg: AdvantageConfig | None = None,
    scale_tanh: float | None = None,
) -> dict[str, Any]:
    """Is tanh load-bearing or cosmetic? (diagnostic *c*, rev-2 sharpened —
    tie-count alone is insufficient.)

    Per true K-group, all three:

    1. **same-seat same-side clip ties** — rollouts whose trainable-seat
       ``|margin|/scale ≥ 1`` (pinned at ±1 under the hard clip) on the same
       sign: those are *tied* in RTG and unrankable by standardization.
    2. **clip-vs-tanh within-group rank-corr** — Spearman of per-rollout mean
       standardized advantage under clip vs tanh; ≈1 ⇒ tanh changed nothing.
    3. **does the recovered tail signal survive standardization?** — spread of
       the tanh per-rollout advantage among the formerly-clipped rollouts vs
       the group's standardized σ. ``spread ≤ σ_group`` ⇒ washed out.

    Verdict ``LOAD-BEARING`` iff there are clip ties AND median cross-shape
    rank-corr < 0.95 AND median recovered-spread/σ_group > 1.0; else
    ``COSMETIC`` with the raw numbers stated (no optimistic rounding).
    """
    transcripts = list(transcripts)
    base = base_reward_cfg or RewardConfig()
    if scale_tanh is None:
        scale_tanh = float(
            calibrate_scale(transcripts, replace(base, terminal_shape="tanh"))[
                "recommended_scale"
            ]
        )
    clip_cfg = replace(base, terminal_shape="clip")
    tanh_cfg = replace(base, terminal_shape="tanh", scale=scale_tanh)

    margin_by = {
        (b.game_id, b.player_id): b.true_margin
        for b in trajectory_breakdowns(transcripts, base)
    }
    adv_clip = _group_game_mean_adv(compute_advantages(transcripts, clip_cfg, adv_cfg))
    adv_tanh = _group_game_mean_adv(compute_advantages(transcripts, tanh_cfg, adv_cfg))

    n_groups = 0
    groups_with_tie = 0
    tied_rollout_counts: list[int] = []
    cross_corrs: list[float] = []
    spread_ratios: list[float] = []
    per_group: list[dict[str, Any]] = []
    for gk, clip_by_game in adv_clip.items():
        pid = gk[1]
        gids = sorted(clip_by_game)
        if len(gids) < 2:
            continue
        n_groups += 1
        tanh_by_game = adv_tanh.get(gk, {})
        # 1. clip ties (same seat, same side)
        pos = [g for g in gids if margin_by.get((g, pid), 0.0) / base.scale >= 1.0]
        neg = [g for g in gids if margin_by.get((g, pid), 0.0) / base.scale <= -1.0]
        tie_n = max(len(pos), len(neg))
        clipped = pos if len(pos) >= len(neg) else neg
        if tie_n >= 2:
            groups_with_tie += 1
            tied_rollout_counts.append(tie_n)
        # 2. cross-shape rank-corr
        c = _spearman(
            [clip_by_game[g] for g in gids],
            [tanh_by_game.get(g, 0.0) for g in gids],
        )
        if c is not None:
            cross_corrs.append(c)
        # 3. recovered spread among formerly-clipped vs group σ
        ratio = None
        if len(clipped) >= 2:
            tanh_clipped = [tanh_by_game.get(g, 0.0) for g in clipped]
            spread = max(tanh_clipped) - min(tanh_clipped)
            sigma = statistics.pstdev([tanh_by_game.get(g, 0.0) for g in gids])
            if sigma > 1e-12:
                ratio = spread / sigma
                spread_ratios.append(ratio)
        per_group.append(
            {
                "group_key": repr(gk),
                "rollouts": len(gids),
                "clipped_pos": len(pos),
                "clipped_neg": len(neg),
                "tie_n": tie_n,
                "cross_shape_rank_corr": (None if c is None else round(c, 4)),
                "recovered_spread_over_sigma": (
                    None if ratio is None else round(ratio, 4)
                ),
            }
        )

    def _med(xs: list[float]) -> float | None:
        return round(statistics.median(xs), 4) if xs else None

    med_corr = _med(cross_corrs)
    med_ratio = _med(spread_ratios)
    has_ties = groups_with_tie > 0
    load_bearing = bool(
        has_ties
        and med_corr is not None
        and med_corr < 0.95
        and med_ratio is not None
        and med_ratio > 1.0
    )
    if load_bearing:
        verdict = "LOAD-BEARING"
        reason = (
            f"{groups_with_tie}/{n_groups} groups have ≥2 same-side clip ties; "
            f"median clip-vs-tanh rank-corr {med_corr} < 0.95 (orderings "
            f"differ); median recovered-spread/σ {med_ratio} > 1.0 (survives "
            "standardization)"
        )
    else:
        verdict = "COSMETIC"
        reason = (
            f"groups_with_tie={groups_with_tie}/{n_groups}; median "
            f"clip-vs-tanh rank-corr={med_corr}; median recovered-spread/σ="
            f"{med_ratio}. tanh does not measurably re-rank within groups on "
            "this corpus (it may still help on a higher-variance Phase-3 "
            "rollout corpus — this is not an exculpation)."
        )
    return {
        "n_groups": n_groups,
        "groups_with_clip_tie": groups_with_tie,
        "mean_tied_rollouts": (
            round(statistics.fmean(tied_rollout_counts), 4)
            if tied_rollout_counts
            else 0.0
        ),
        "median_cross_shape_rank_corr": med_corr,
        "median_recovered_spread_over_sigma": med_ratio,
        "scale_clip": base.scale,
        "scale_tanh": scale_tanh,
        "verdict": verdict,
        "reason": reason,
        "per_group": per_group,
    }


def degenerate_group_sigma_panel(
    transcripts: Sequence[dict[str, Any]],
    reward_cfg: RewardConfig | None = None,
    adv_cfg: AdvantageConfig | None = None,
) -> dict[str, Any]:
    """Rec 5 — quantify the silent ``advantage.py`` degenerate-σ fallback
    (``denom = sigma if sigma > std_eps else 1.0`` ⇒ mean-centered-but-
    unscaled). Mirrors that exact grouping/pstdev so the count is faithful.

    Honest when none: ``n_degenerate=0`` ⇒ the fallback is not exercised on
    this corpus (NOT a guarantee it never will be in Phase-3 rollouts).
    """
    adv_cfg = adv_cfg or AdvantageConfig()
    rows = compute_advantages(transcripts, reward_cfg, adv_cfg)
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r.trainable:
            groups[r.group_key].append(r.advantage_raw)
    sigmas: list[float] = []
    degenerate: list[str] = []
    for gk, vals in groups.items():
        sigma = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        sigmas.append(sigma)
        if sigma <= adv_cfg.std_eps:
            degenerate.append(repr(gk))
    healthy = sorted(s for s in sigmas if s > adv_cfg.std_eps)
    return {
        "n_groups": len(groups),
        "n_degenerate": len(degenerate),
        "std_eps": adv_cfg.std_eps,
        "degenerate_group_keys": degenerate,
        "sigma_raw_min": round(min(sigmas), 6) if sigmas else None,
        "sigma_raw_median": round(statistics.median(sigmas), 6) if sigmas else None,
        "sigma_raw_max": round(max(sigmas), 6) if sigmas else None,
        "healthy_min_over_median": (
            round(healthy[0] / statistics.median(healthy), 4)
            if len(healthy) >= 1
            else None
        ),
        "note": (
            "no degenerate K-groups on this corpus — the silent "
            "mean-centered-but-unscaled fallback is NOT exercised here (not a "
            "guarantee for Phase-3 rollouts; choose the policy explicitly)"
            if not degenerate
            else (
                f"{len(degenerate)} group(s) hit the σ≤std_eps fallback and "
                "are silently mean-centered-but-unscaled → down-weighted vs "
                "unit-variance groups in a mixed batch"
            )
        ),
    }


def illegal_leverage_panel(
    transcripts: Sequence[dict[str, Any]],
    reward_cfg: RewardConfig | None = None,
) -> dict[str, Any]:
    """Rec 4 (best-effort, honest) — the illegal-penalty RTG-leverage concern
    (Problem A): one −0.5 telescopes into the RTG of every preceding turn on
    that trajectory and one such rollout can dominate a K-group σ.

    Post-SFT illegal rate is ≈5.4e-4, so this corpus is expected to have
    ~zero illegal turns. When it does, the panel reports
    ``measurable: False`` and says so plainly — an empty set is *not* a
    clearance and no leverage number is fabricated from it.
    """
    cfg = reward_cfg or RewardConfig()
    n_turns = 0
    illegal: list[dict[str, Any]] = []
    for gid, data in enumerate(transcripts):
        parsed = parse_transcript(data)
        by_player: dict[int, list[Any]] = defaultdict(list)
        for t in per_turn_rewards(parsed, cfg):
            by_player[t.player_id].append(t)
        for pid, seq in by_player.items():
            seq.sort(key=lambda t: (t.round_index, 0 if t.phase == "bid" else 1))
            for i, t in enumerate(seq):
                n_turns += 1
                if not (t.parse_valid and t.legal_valid):
                    # RTG(t)=Σ_{s≥t} r_s ⇒ this −0.5 enters the RTG of every
                    # turn at index ≤ i (itself + all preceding).
                    illegal.append(
                        {
                            "game_id": gid,
                            "player_id": pid,
                            "turn_index": i,
                            "telescopes_into_n_turns": i + 1,
                        }
                    )
    if not illegal:
        return {
            "n_illegal": 0,
            "n_turns": n_turns,
            "measurable": False,
            "note": (
                "no illegal turns on this corpus (expected; post-SFT illegal "
                "rate ≈5.4e-4) — the RTG-leverage interaction is NOT "
                "measurable here. An empty set is not a clearance."
            ),
        }
    leverage = [d["telescopes_into_n_turns"] for d in illegal]
    return {
        "n_illegal": len(illegal),
        "n_turns": n_turns,
        "measurable": True,
        "max_telescoped_turns": max(leverage),
        "mean_telescoped_turns": round(statistics.fmean(leverage), 4),
        "summed_penalty_per_illegal_traj": cfg.illegal_penalty,
        "illegal_turns": illegal[:50],
        "note": (
            "each illegal turn injects illegal_penalty into the RTG of all "
            "preceding turns on that trajectory; a single such rollout in a "
            "K-group can dominate σ after standardization (Problem A). "
            "Mitigation (apply −0.5 to the offending turn's r_t only, do not "
            "telescope) is itself a behavioural change to A/B, not a free fix."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 1 reward-balance diagnostics + λ re-derivation"
    )
    ap.add_argument("--corpus", required=True, help="dir of schema-v3 transcripts")
    ap.add_argument("--pattern", default="*.json")
    ap.add_argument(
        "--lambdas",
        default=",".join(str(x) for x in _DEFAULT_LAMBDAS),
        help="comma-separated λ values to sweep",
    )
    ap.add_argument("--ratio-target", type=float, default=0.33)
    ap.add_argument("--rank-corr-floor", type=float, default=0.8)
    ap.add_argument("--json", action="store_true", help="emit JSON, not a table")
    args = ap.parse_args(argv)

    from .export import load_corpus  # local: avoids an import cycle

    corpus = load_corpus(args.corpus, args.pattern)
    if not corpus:
        print(f"no schema-v3 transcripts under {args.corpus!r}")
        return 1

    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    current = shaping_terminal_mass(corpus)  # at the locked default λ
    out = lambda_sweep(
        corpus,
        lambdas,
        ratio_target=args.ratio_target,
        rank_corr_floor=args.rank_corr_floor,
    )

    reveal = reveal_policy_diagnostic(corpus)

    if args.json:
        print(json.dumps(
            {"current": current, **out, "reveal_policy": reveal},
            indent=2, default=str,
        ))
        return 0

    print(f"corpus: {len(corpus)} games  |  current λ={current.get('lambda')} "
          f"SCALE={current.get('scale')}")
    print(f"  shaping/terminal median = {current.get('shaping_over_terminal_median')}"
          f"  (>1 means shaping dominates; target ≤ {args.ratio_target})")
    print(f"  corr(mid-game proxy margin, engine-true margin) = "
          f"{current.get('corr_midgame_vs_true_margin')}")
    print()
    print(f"{'lambda':>8} {'shp/term med':>13} {'frac shp>term':>14} "
          f"{'rankcorr mean':>14} {'rankcorr min':>13} {'groups':>7}")
    for r in out["sweep"]:
        print(f"{r['lambda']:>8} {str(r['shaping_over_terminal_median']):>13} "
              f"{str(r['frac_shaping_exceeds_terminal']):>14} "
              f"{str(r['rank_corr_mean']):>14} {str(r['rank_corr_min']):>13} "
              f"{r['groups_usable']:>7}")
    rec = out["recommendation"]
    print()
    print(f"recommended λ = {rec['lambda']}  ({rec['basis']})")
    if not rec["rank_corr_available"]:
        print("  WARNING: rank-corr gate could not run on this corpus.")
    if rec["note"]:
        print(f"  note: {rec['note']}")
    print()
    print("reveal-shaping telemetry (Caveat 2 addendum — observational, "
          "CONFOUNDED; not a causal gate):")
    print(f"  policy reveal turns = {reveal['policy_reveal_turns']} "
          f"(of {reveal['reveal_turns_total']}; "
          f"{reveal['reveal_turns_defaulted_excluded']} defaulted/illegal excluded)")
    print(f"  mean signed reveal_shaping = {reveal['mean_signed_reveal_shaping']}  "
          f"frac self-favoring (of nonzero) = {reveal['frac_self_favoring_of_nonzero']}")
    print(f"  corr(reveal_shaping, true terminal margin) = "
          f"{reveal['corr_reveal_shaping_vs_true_margin']}  (descriptive only)")
    print(f"  observation: {reveal['observation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
