#!/usr/bin/env python3
"""Phase-3 lagged-self opponent pool (the RL plan §3.3) + PFSP weighting (repl_08).

History
-------
* **Original (repl_03–repl_07)**: stationary scripted heuristic + lagged-self
  ring buffer, draw probability annealed 0 → p_max. The trainable seat learned
  to crush the heuristic's `bid=floor(0.5·max_bid)` / lex-first-reveal exploit
  — †phase3-heuristic-exploit-confirmed: §3.6 +16.5 vs heuristic / +4.4pp vs
  Flash (n=180, noise) at repl_07 step 400.
* **repl_08 (this file)**: scripted heuristic removed from training. The pool
  now holds (a) a pre-seeded PINNED step_0 snapshot (the SFT-base anchor; never
  evicted), (b) a ring-buffered set of unpinned lagged-self snapshots, and
  (c) optional API opponents (Gemini-Flash @ low p). Snapshot picks are
  PFSP-weighted by **observed win-rate against the trainee**:
  `w_i = max(0.01, (1 - WR_i)**2)` (AlphaStar `f_hard`, p=2). When ALL
  observed WRs are below 0.25 AND every snapshot has real observations
  (≥`_PFSP_MIN_N_GAMES`), the pick falls back to `f_var(x)=x(1-x)`
  (AlphaStar's main-exploiter recipe for "stuck below uniform").
  Cold-start handling (Codex round-1 fix): under-observed snapshots get a
  per-snapshot WR=0.5 PRIOR (max-uncertainty); the bucket regime stays
  `f_hard` as long as ≥1 snapshot has real observations. True `uniform`
  regime fires only when ZERO snapshots have ≥`_PFSP_MIN_N_GAMES` of
  observations (pre-first-roll). WR storage (Codex round-3 fix):
  `update_snapshot_winrate` accumulates `n_games` monotonically and uses
  an EWMA on `win_rate` so small-sample rolls don't downgrade trust.
  Determinism preserved via the existing `_unit(...)` SHA-256 hash +
  cumulative-weight binary search; the pinned anchor gets a minimum draw
  PROBABILITY (`p_anchor_floor`, default 0.15) so PFSP can't drown it out
  once the trainee crushes it.

§A.7 — one opponent per K-group: `draw(step, seed)` is deterministic in
`(step, seed)`, so all K rollouts of one seed within one refresh face the
SAME opponent (only the trainable seat varies; clean GRPO within-group
standardisation). The opponent is re-drawn across refreshes (different
`step`) and across seeds.

Pure stdlib — no torch, no network at import (mirrors
`megagem.training.adapter_sync`) — so the CPU tests and
`phase3_grpo.py --dry-run` import it freely.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OpponentSpec:
    """One opponent assignment for a K-group.

    `served_name` is what `megagem.rollout.run_game` resolves through
    `megagem/endpoints.py` (a snapshot LoRA on the rollout vLLM, or an API
    model). `actor_id` is the
    §A.6 loss-mask tag — any non-"trainable" string masks the opponent's
    tokens identically; the distinct value is for telemetry only.
    """

    kind: str           # "current_self" | "snapshot" | "api" | "heuristic" (legacy)
    served_name: str
    actor_id: str
    step: int | None = None


@dataclass(frozen=True)
class Snapshot:
    """A frozen lagged-self adapter registered on the rollout vLLM."""

    step: int
    served_name: str
    adapter_path: str


def anneal_probability(
    step: int, *, anneal_start: int, anneal_end: int, p_max: float
) -> float:
    """P(draw a non-heuristic opponent) at optimizer `step`.

    Linear ramp 0 -> `p_max` over [`anneal_start`, `anneal_end`], clamped at
    both ends. With repl_08 (no scripted heuristic in training) the caller
    sets anneal_start=0/anneal_end=0/p_max=1.0 so this is constantly 1.0 —
    the pool is active from step 0 because the step_0 anchor is pre-seeded.
    Kept for back-compat with the repl_07 "anneal in" schedule.
    """
    p_max = float(p_max)
    if step <= anneal_start:
        return 0.0
    if step >= anneal_end or anneal_end <= anneal_start:
        return p_max
    return p_max * (step - anneal_start) / (anneal_end - anneal_start)


def _unit(rng_seed: int, step: int, seed: int, salt: str) -> float:
    """Deterministic float in [0, 1) from (rng_seed, step, seed, salt).

    Uses hashlib, NOT the builtin hash(): CPython salts hash() of str/bytes per
    process (PYTHONHASHSEED), which would make draws non-reproducible across
    processes and break the §A.7 'opponent sampling state' the run JSON logs.
    sha256 is process-independent.
    """
    key = f"{int(rng_seed)}|{int(step)}|{int(seed)}|{salt}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2.0 ** 64


def _pick(specs: list, rng_seed: int, step: int, seed: int, salt: str):
    """Deterministically pick one element of a non-empty `specs` (uniform)."""
    idx = int(_unit(rng_seed, step, seed, salt) * len(specs))
    return specs[min(idx, len(specs) - 1)]


def _weighted_pick(
    specs: list, weights: list[float],
    rng_seed: int, step: int, seed: int, salt: str,
):
    """Deterministically pick one element of `specs` with positive weights.

    Uses cumulative-weight binary search so the pick is process-independent
    (no random.choices PRNG state) and bit-exactly the same across reproduction
    runs given the same (rng_seed, step, seed, salt). Falls back to uniform
    if all weights are non-positive.
    """
    total = sum(w for w in weights if w > 0.0)
    if total <= 0.0 or len(specs) == 0:
        return _pick(specs, rng_seed, step, seed, salt + ":fallback")
    u = _unit(rng_seed, step, seed, salt) * total
    acc = 0.0
    for s, w in zip(specs, weights):
        if w <= 0.0:
            continue
        acc += w
        if u < acc:
            return s
    # Floating-point slop ⇒ return the last positively-weighted element.
    for s, w in zip(reversed(specs), reversed(weights)):
        if w > 0.0:
            return s
    return specs[-1]


# PFSP minimum-floor weight: keep ALL snapshots reachable so the policy can't
# stop learning from an opponent by virtue of (1-WR)^2 ⇒ 0 at WR≈1. AlphaStar
# uses a small lower bound for the same reason (Vinyals 2019, supplementary).
_PFSP_W_MIN = 1e-2

# PFSP "stuck" threshold: when the MAX observed snapshot WR is below this, the
# policy is being crushed by everyone and `f_hard` keeps weighting the same
# already-too-hard cohort. Switch to `f_var(x)=x(1-x)` (peaks at WR=0.5) which
# lifts matched-strength opponents — AlphaStar's main-exploiter recipe.
_PFSP_F_VAR_TRIGGER = 0.25

# Minimum per-snapshot game count before that snapshot's WR is trusted for
# PFSP weighting. Below this, the snapshot gets a max-uncertainty PRIOR weight
# (WR=0.5 ⇒ f_hard=0.25) — does NOT disable PFSP for the rest of the bucket
# (Codex round-1 fix; the pre-fix behaviour caused PFSP to flip back to
# uniform every time a fresh snapshot was added, ~3 disabled refreshes per
# snapshot rotation).
_PFSP_MIN_N_GAMES = 10
_PFSP_UNOBSERVED_WR_PRIOR = 0.5

# EWMA decay for `update_snapshot_winrate` (Codex round-3 fix): with K=8 and
# 24 seeds, a snapshot drawn for a single seed in one roll has n_games=8 < 10
# — the pre-fix unconditional-overwrite would drop a previously-trusted
# (wr, n≥10) entry back to under-min, flickering PFSP weights. The new
# storage policy:
#   * `n_games` accumulates monotonically (cumulative; trust grows over time).
#   * `win_rate` is an EWMA: `new_wr = (1-α)·old_wr + α·obs_wr`.
# α=0.3 is recent-biased enough to track a non-stationary trainee (the policy
# improves between rolls) but smooth enough that a single noisy roll doesn't
# whipsaw the weight. Once trusted, a snapshot stays trusted.
_PFSP_WR_EMA_ALPHA = 0.3

# Heuristic decay floor (repl_08 v3). Effective heuristic-draw probability:
#   p_heuristic_effective = p_heuristic · max(_P_HEURISTIC_DECAY_FLOOR, (1-WR)^2)
# Floor of 0.10 (plan-faithful — Codex round-1 review: original 0.20 silently
# doubled late-run heuristic exposure beyond what the plan defended).
# At WR=1.0 with p_heuristic=0.25, late-run heuristic share = 0.25·0.10 =
# 2.5% of all draws ≈ 0.2 games per K-group of 8. That is INTENDED: once
# the trainee crushes the heuristic, we want it to fade — the heuristic's
# job is to provide a +R signal during cold-start, not to be a perennial
# baseline. PFSP's own floor (`_PFSP_W_MIN = 0.01`) is even tighter; 0.10
# is the right order of magnitude. Squared decay is the same
# `f_hard(x) = (1-x)^2` form as PFSP — opponents the trainee crushes shrink
# quadratically. Repl_07's heuristic exploit happened with a CONSTANT share
# (no decay); this within-run mechanism is the design fix for that.
_P_HEURISTIC_DECAY_FLOOR = 0.10


class OpponentPool:
    """Lagged-self opponent pool — pinned anchors + unpinned ring buffer +
    optional API opponents. repl_08 introduces PFSP win-rate weighting for
    snapshot picks (see module docstring)."""

    def __init__(
        self,
        *,
        pinned_snapshots: list[Snapshot] | None = None,
        max_snapshots: int = 5,
        anneal_start: int = 0,
        anneal_end: int = 0,
        p_max: float = 1.0,
        api_specs: list[OpponentSpec] | None = None,
        api_weights: list[int] | None = None,
        p_api: float = 0.0,
        current_self_spec: OpponentSpec | None = None,
        p_current_self: float = 0.0,
        rng_seed: int = 0,
        heuristic_spec: OpponentSpec | None = None,
        p_anchor_floor: float = 0.0,
        p_heuristic: float = 0.0,
        heuristic_anneal_end: int = 0,
    ):
        """`pinned_snapshots` are anchors that never evict from the ring
        buffer (use for the SFT-base step_0 anchor in repl_08+). They count as
        snapshots in `draw()` and in PFSP weighting just like any unpinned
        snapshot, but `add_snapshot()` never removes them.

        `p_anchor_floor` (default 0.0): minimum draw PROBABILITY for the pinned
        anchor(s) among the snapshot bucket. Without this, once the trainee
        crushes the anchor (WR ≈ 0.95 ⇒ f_hard ≈ 0.0025 → floored to 0.01),
        the anchor receives ≈ 1% of snapshot draws — barely anchoring at all.
        With `p_anchor_floor=0.15`, 15% of all snapshot draws are reserved for
        a uniform pick across the pinned set; the remaining 85% goes through
        PFSP weighting across pinned+unpinned. Pre-Codex (and Codex-flagged)
        the floor was implicitly 0.0.

        `heuristic_spec` is the legacy single-stationary-heuristic fallback
        spec used ONLY when no snapshots exist and no API is configured —
        kept for `--no-opponent-pool` back-compat and the repl_07-era reproduction
        path. With pre-seeded `pinned_snapshots` (repl_08 default) this slot is
        never reached UNLESS `p_heuristic > 0` (see below).

        `p_heuristic` (default 0.0): the repl_08 v3 mechanism — the heuristic
        gets `p_heuristic_effective` of all draws, where
        `p_heuristic_effective = p_heuristic · max(_P_HEURISTIC_DECAY_FLOOR,
        (1 - WR_heur)²)`. WR is the trainee's observed win rate vs the
        heuristic, EWMA-smoothed across rolls (same storage policy as
        `update_snapshot_winrate`). Cold-start (n_games < `_PFSP_MIN_N_GAMES`)
        uses the FULL `p_heuristic` so the first ~10 games provide a clean
        +R baseline. Diagnosis behind this knob: seam_smoke_02 (heuristic
        REMOVED) gave ≈0 reward / ≈33% WR vs all opponents — null learning
        signal. Repl_07 (heuristic at constant share) gave +0.6 reward / 75%
        WR vs heuristic but overfit. This decay sits between the two:
        meaningful early signal, naturally shrinking share as the trainee
        crushes the heuristic. Requires `heuristic_spec is not None` when
        `p_heuristic > 0`.

        `heuristic_anneal_end` (default 0 ⇒ disabled): step-based anneal-to-zero
        for the heuristic, DECOUPLED from the `(1-WR)²` decay floor. When > 0,
        `p_heuristic_effective` is additionally multiplied by a linear factor
        `max(0, 1 - step/heuristic_anneal_end)`, so the heuristic provides a
        cold-start +R signal early and then exits COMPLETELY at
        `heuristic_anneal_end` — leaving the pinned SFT anchor + lagged
        snapshots as the sole signal source (the AlphaStar/OpenAI-Five pattern:
        bootstrap on a fixed opponent, then graduate to self-play). This is the
        principled replacement for the perennial `_P_HEURISTIC_DECAY_FLOOR`
        floor (which never reaches 0 — at p_heuristic=0.30 it bottoms out at
        ≈3% of draws forever). Default 0 keeps the floor-only behaviour
        byte-identical for back-compat.
        """
        self.heuristic_spec = heuristic_spec
        self.max_snapshots = int(max_snapshots)
        self.anneal_start = int(anneal_start)
        self.anneal_end = int(anneal_end)
        self.p_max = float(p_max)
        self.api_specs = list(api_specs or [])
        # `api_weights` is a per-spec integer multiplicity. Default None ⇒ all
        # ones ⇒ uniform pick (back-compat). Non-uniform weights are realised
        # by expanding the spec list (each spec appears `weight` times in
        # `_api_specs_expanded`) and letting `_pick` keep its uniform-by-index
        # semantics — keeps the deterministic-by-(step,seed) draw contract.
        if api_weights is None:
            self.api_weights = [1] * len(self.api_specs)
        else:
            if len(api_weights) != len(self.api_specs):
                raise ValueError(
                    f"api_weights length ({len(api_weights)}) must equal "
                    f"api_specs length ({len(self.api_specs)})")
            if any(w < 0 for w in api_weights):
                raise ValueError(
                    f"api_weights must be non-negative ints, got {api_weights}")
            if sum(api_weights) == 0 and self.api_specs:
                raise ValueError(
                    "api_weights sum to 0 with non-empty api_specs — at least "
                    "one weight must be positive (or pass api_specs=[]).")
            self.api_weights = [int(w) for w in api_weights]
        self._api_specs_expanded: list[OpponentSpec] = [
            s for s, w in zip(self.api_specs, self.api_weights)
            for _ in range(w)
        ]
        self.p_api = float(p_api)
        self.current_self_spec = current_self_spec
        if not 0.0 <= float(p_current_self) <= 1.0:
            raise ValueError(
                f"p_current_self must be in [0,1], got {p_current_self}")
        if float(p_current_self) > 0.0 and current_self_spec is None:
            raise ValueError(
                "p_current_self > 0 requires current_self_spec to be set; "
                "otherwise draw() would have nothing to return on the "
                "current-self branch.")
        self.p_current_self = float(p_current_self)
        self.rng_seed = int(rng_seed)
        if not 0.0 <= float(p_anchor_floor) <= 1.0:
            raise ValueError(
                f"p_anchor_floor must be in [0,1], got {p_anchor_floor}")
        self.p_anchor_floor = float(p_anchor_floor)
        # Pinned snapshots: anchors (e.g. step_0 SFT-base). Never evicted.
        self._pinned_snapshots: list[Snapshot] = list(pinned_snapshots or [])
        # Unpinned snapshots: bounded ring buffer (FIFO eviction).
        self._snapshots: list[Snapshot] = []
        # PFSP state: latest observed (win_rate, n_games) per snapshot step.
        self._snapshot_winrate: dict[int, tuple[float, int]] = {}
        # repl_08 v3: heuristic-with-decay state.
        if not 0.0 <= float(p_heuristic) <= 1.0:
            raise ValueError(
                f"p_heuristic must be in [0,1], got {p_heuristic}")
        if float(p_heuristic) > 0.0 and heuristic_spec is None:
            raise ValueError(
                "p_heuristic > 0 requires heuristic_spec to be set; otherwise "
                "draw() would have nothing to return on the heuristic branch.")
        # Cold-start budget check: at cold start (no observations) the heuristic
        # gets `p_heuristic` of all draws, the API gets `p_api`, current self
        # gets `p_current_self`, and the remainder goes to checkpoints. If
        # those sum > 1, the one-tier mix-gate in draw() is ill-defined. Reject
        # loud at construction so the user doesn't discover this mid-run.
        p_budget = float(p_heuristic) + float(p_api) + float(p_current_self)
        if p_budget > 1.0 + 1e-9:
            raise ValueError(
                f"p_heuristic ({p_heuristic}) + p_api ({p_api}) + "
                f"p_current_self ({p_current_self}) = {p_budget:.3f} > 1 — "
                "opponent draws cannot sum above 1. Reduce one or more.")
        self.p_heuristic = float(p_heuristic)
        self._heuristic_winrate: tuple[float, int] | None = None
        # repl_08 v3.2: step-based anneal-to-zero (decoupled from the WR floor).
        if int(heuristic_anneal_end) < 0:
            raise ValueError(
                f"heuristic_anneal_end must be >= 0, got {heuristic_anneal_end}")
        self.heuristic_anneal_end = int(heuristic_anneal_end)

    # -- snapshot ring buffer ------------------------------------------------
    def add_pinned_snapshot(self, snap: Snapshot) -> None:
        """Add an anchor that will NEVER be evicted by `add_snapshot`. The
        pool's effective size becomes `len(pinned) + max_snapshots`. Idempotent
        on `(step, served_name)`.
        """
        for existing in self._pinned_snapshots:
            if existing.step == snap.step and existing.served_name == snap.served_name:
                return
        self._pinned_snapshots.append(snap)

    def add_snapshot(self, snap: Snapshot) -> Snapshot | None:
        """Append `snap` to the unpinned ring buffer. Past `max_snapshots`,
        drop & RETURN the oldest UNPINNED so the caller can unload it from
        vLLM. Returns None when nothing was evicted. Pinned snapshots are
        NEVER returned from this method.
        """
        self._snapshots.append(snap)
        if len(self._snapshots) > self.max_snapshots:
            return self._snapshots.pop(0)
        return None

    def snapshots(self) -> list[Snapshot]:
        """All snapshots, pinned first then unpinned in insertion order."""
        return [*self._pinned_snapshots, *self._snapshots]

    def pinned_snapshots(self) -> list[Snapshot]:
        return list(self._pinned_snapshots)

    def unpinned_snapshots(self) -> list[Snapshot]:
        return list(self._snapshots)

    # -- PFSP feedback -------------------------------------------------------
    def update_snapshot_winrate(
        self, step: int, win_rate: float, n_games: int
    ) -> None:
        """Record per-snapshot win rate (fraction of games in which the
        trainee BEAT the snapshot's seat). Caller pipes this from
        `_per_snapshot_age_stats` once per training roll.

        Storage policy (Codex round-3):
          * `n_games` accumulates monotonically across calls (cumulative
            evidence — trust grows over time, never downgrades).
          * `win_rate` is an EWMA: `(1-α)·old + α·obs` with
            α=`_PFSP_WR_EMA_ALPHA` (recent-biased to track a non-stationary
            trainee, but smooth enough that a single noisy small-sample roll
            doesn't whipsaw the weight).

        Pre-fix this method unconditionally overwrote the stored tuple, so a
        roll where a snapshot was drawn for only one seed (n=8 < min=10)
        would replace a trusted (wr, n≥10) with an under-min entry, dropping
        the snapshot back into the WR=0.5-prior regime — flickering PFSP
        weights at the cadence of opponent rotation.
        """
        if not math.isfinite(win_rate) or n_games <= 0:
            return
        wr = min(1.0, max(0.0, float(win_rate)))
        n = max(0, int(n_games))
        existing = self._snapshot_winrate.get(int(step))
        if existing is None:
            self._snapshot_winrate[int(step)] = (wr, n)
            return
        old_wr, old_n = existing
        alpha = _PFSP_WR_EMA_ALPHA
        new_wr = (1.0 - alpha) * old_wr + alpha * wr
        new_n = old_n + n          # cumulative — trust grows, never falls
        self._snapshot_winrate[int(step)] = (new_wr, new_n)

    def snapshot_winrates(self) -> dict[int, tuple[float, int]]:
        return dict(self._snapshot_winrate)

    # -- heuristic decay (repl_08 v3) ----------------------------------------
    def update_heuristic_winrate(
        self, win_rate: float, n_games: int
    ) -> None:
        """Record per-roll heuristic WR (fraction of games the trainee BEAT
        the heuristic). Mirror of `update_snapshot_winrate` but for the single
        heuristic slot. Caller pipes from `_per_snapshot_age_stats` /
        `per_opponent_stats['heuristic']` once per training roll. Same EWMA
        on WR + cumulative n_games storage policy as snapshots."""
        if not math.isfinite(win_rate) or n_games <= 0:
            return
        wr = min(1.0, max(0.0, float(win_rate)))
        n = max(0, int(n_games))
        if self._heuristic_winrate is None:
            self._heuristic_winrate = (wr, n)
            return
        old_wr, old_n = self._heuristic_winrate
        alpha = _PFSP_WR_EMA_ALPHA
        self._heuristic_winrate = (
            (1.0 - alpha) * old_wr + alpha * wr, old_n + n)

    def heuristic_winrate(self) -> tuple[float, int] | None:
        """Return the stored (EWMA_win_rate, cumulative_n_games) or None."""
        return self._heuristic_winrate

    def _heuristic_step_factor(self, step: int | None) -> float:
        """Linear step-anneal multiplier for the heuristic draw probability.

        Ramps 1.0 → 0.0 over `[0, heuristic_anneal_end)` and stays 0.0 after.
        Returns 1.0 (no anneal) when the knob is disabled (`<= 0`) or `step`
        is unknown — so a `heuristic_anneal_end=0` pool (the default) and any
        no-`step` direct call are byte-identical to the pre-anneal behaviour.
        """
        e = self.heuristic_anneal_end
        if e <= 0 or step is None:
            return 1.0
        s = int(step)
        if s >= e:
            return 0.0
        if s <= 0:
            return 1.0
        return 1.0 - (float(s) / float(e))

    def _p_heuristic_effective(self, step: int | None = None) -> float:
        """Current heuristic-draw probability at optimizer `step`.

        Two multiplicative factors:
          1. WR decay (always on): `max(_P_HEURISTIC_DECAY_FLOOR,
             (1 - WR_heur)²)` once `_PFSP_MIN_N_GAMES` heuristic games have
             been observed; before that (cold start) the factor is 1.0 so the
             very first rolls give the trainee a clean +R signal.
          2. Step anneal (opt-in via `heuristic_anneal_end > 0`): a linear
             `max(0, 1 - step/heuristic_anneal_end)` that drives the share to
             EXACTLY 0 by `heuristic_anneal_end`, decoupled from the decay
             floor. `step=None` / knob disabled ⇒ factor 1.0 (back-compat).

        Returns 0.0 when `p_heuristic == 0` or no `heuristic_spec` is
        configured (both identical: the heuristic branch in `draw()` is
        disabled), or once the step anneal has fully retired the heuristic.
        """
        if self.p_heuristic <= 0.0 or self.heuristic_spec is None:
            return 0.0
        step_factor = self._heuristic_step_factor(step)
        if step_factor <= 0.0:
            return 0.0
        if (self._heuristic_winrate is None
                or self._heuristic_winrate[1] < _PFSP_MIN_N_GAMES):
            wr_factor = 1.0
        else:
            wr = self._heuristic_winrate[0]
            wr_factor = max(_P_HEURISTIC_DECAY_FLOOR, (1.0 - wr) ** 2)
        return self.p_heuristic * wr_factor * step_factor

    # -- annealed draw -------------------------------------------------------
    def anneal_probability(self, step: int) -> float:
        return anneal_probability(
            step, anneal_start=self.anneal_start,
            anneal_end=self.anneal_end, p_max=self.p_max)

    def _pfsp_weights(
        self, snapshots: list[Snapshot]
    ) -> tuple[list[float], str]:
        """Compute PFSP weights for `snapshots`. Returns (weights, regime)
        where `regime` is one of:
          * "empty"       — no snapshots in input.
          * "uniform"     — NONE of the snapshots have ≥ _PFSP_MIN_N_GAMES of
                            observations yet (true cold-start before any
                            snapshot enters the draw).
          * "f_hard"      — AlphaStar `f_hard(x) = (1-x)^2` (default).
          * "f_var"       — AlphaStar `f_var(x) = x(1-x)` fallback when ALL
                            *observed* WRs are below `_PFSP_F_VAR_TRIGGER`
                            (the policy is being crushed by every opponent;
                            switch to weighting matched-strength opponents).

        Pre-Codex (and Codex-flagged) the per-snapshot WR was treated as
        "uniform-bucket-wide if ANY entry has <10 games" — which made every
        snapshot rotation (every `snapshot_every` training steps) reset the
        regime to uniform until the freshly-added snapshot accumulated 10
        games. Effective PFSP uptime was poor. Fix: under-observed snapshots
        get a per-snapshot WR=0.5 PRIOR (max-uncertainty), and we still
        compute f_hard/f_var across the bucket — only the genuine cold-start
        (NO snapshot observed) hits the all-uniform branch.
        """
        n = len(snapshots)
        if n == 0:
            return [], "empty"
        observed_wrs: list[float] = []        # filled with WR=0.5 for unobserved
        has_real_obs = False
        for s in snapshots:
            entry = self._snapshot_winrate.get(s.step)
            if entry is None or entry[1] < _PFSP_MIN_N_GAMES:
                observed_wrs.append(_PFSP_UNOBSERVED_WR_PRIOR)
            else:
                observed_wrs.append(entry[0])
                has_real_obs = True
        if not has_real_obs:
            return [1.0] * n, "uniform"
        # f_var fires only when (a) all snapshots are REALLY observed (no
        # prior-only entries), AND (b) every observed WR is below the
        # `_PFSP_F_VAR_TRIGGER`. The first condition was added per Codex's
        # residual-risk note: in f_var mode `x(1-x)` peaks at x=0.5, so an
        # unobserved snapshot using the WR=0.5 prior would get the MAX
        # weight — fresh snapshots would be over-sampled before evidence.
        # If any snapshot is prior-only, stay in f_hard, where the WR=0.5
        # prior maps to the mid-range weight 0.25 (sensible default).
        all_observed = all(
            self._snapshot_winrate.get(s.step) is not None
            and self._snapshot_winrate[s.step][1] >= _PFSP_MIN_N_GAMES
            for s in snapshots
        )
        if all_observed and max(observed_wrs) < _PFSP_F_VAR_TRIGGER:
            weights = [
                max(_PFSP_W_MIN, x * (1.0 - x)) for x in observed_wrs
            ]
            return weights, "f_var"
        weights = [max(_PFSP_W_MIN, (1.0 - x) ** 2) for x in observed_wrs]
        return weights, "f_hard"

    def draw(self, step: int, seed: int) -> OpponentSpec:
        """The ONE opponent for the K-group at (`step`, `seed`).

        Deterministic — same (step, seed) ⇒ same opponent for all K rollouts of
        that seed (§A.7).

        Two modes, mutually exclusive at the top of the function:

        **League mix mode** (`p_heuristic > 0` or `p_current_self > 0`):
          Single-tier mix-gate over `u ∈ [0, 1)`:
            * `u < p_heuristic_effective` ⇒ heuristic
            * `u < p_heuristic_effective + p_api` ⇒ API
            * `u < p_heuristic_effective + p_api + p_current_self`
              ⇒ current self
            * else ⇒ checkpoint snapshot bucket (with `p_anchor_floor` carve-out)
          The legacy anneal-to-heuristic branch is SKIPPED in this mode (it
          would otherwise re-fire the heuristic at any non-zero anneal_start
          / p_max, swamping the configured rate — Codex round-1 bug).

        **Legacy mode** (`p_heuristic == 0`):
          1. Legacy anneal: pre-anneal step (per `anneal_probability(step)`)
             ⇒ heuristic_spec if configured.
          2. API bucket: with sub-probability `p_api`, an API model (note:
             this is `p_api · (1 - p_anneal)` of TOTAL draws — only meaningful
             when anneal is fully ramped).
          3. Anchor floor: carve `p_anchor_floor` of snapshot draws for the
             pinned anchor.
          4. Snapshot bucket: PFSP-weighted across pinned+unpinned.

        Raises RuntimeError if the pool is empty and no fallback exists —
        better a loud crash than silently training against nothing.
        """
        # League mix mode: single-tier mix-gate. `p_api` and
        # `p_current_self` are absolute fractions of all draws here (not
        # conditional on the snapshot/checkpoint bucket).
        if self.p_heuristic > 0.0 or self.p_current_self > 0.0:
            p_heur_eff = self._p_heuristic_effective(step)
            api_avail = bool(self._api_specs_expanded)
            p_api_eff = self.p_api if api_avail else 0.0
            p_self_eff = (
                self.p_current_self
                if self.current_self_spec is not None else 0.0
            )
            u = _unit(self.rng_seed, step, seed, "mix-gate-league")
            # Branch 1: heuristic (rate = p_heur_eff).
            if u < p_heur_eff:
                return self.heuristic_spec  # type: ignore[return-value]
            # Branch 2: API (rate = p_api_eff).
            if api_avail and u < p_heur_eff + p_api_eff:
                return _pick(
                    self._api_specs_expanded,
                    self.rng_seed, step, seed, "api-pick")
            # Branch 3: current live adapter as greedy opponent.
            if self.current_self_spec is not None and (
                u < p_heur_eff + p_api_eff + p_self_eff
            ):
                return self.current_self_spec
            # Branch 4: checkpoint bucket.
            return self._draw_snapshot(step, seed)
        # Legacy mode (p_heuristic == 0): preserves the pre-v3 contract.
        # Legacy anneal: pre-anneal step ⇒ always heuristic. repl_08 callers
        # set anneal_start=0/p_max=1.0 so this branch never fires; repl_07
        # callers may still rely on it.
        if _unit(self.rng_seed, step, seed, "pool") >= self.anneal_probability(step):
            if self.heuristic_spec is not None:
                return self.heuristic_spec
            # No heuristic configured: fall through to snapshot/API.
        # API bucket: with sub-probability p_api (of post-anneal draws), an
        # API model.
        if (self._api_specs_expanded
                and _unit(self.rng_seed, step, seed, "api") < self.p_api):
            return _pick(
                self._api_specs_expanded, self.rng_seed, step, seed, "api-pick")
        return self._draw_snapshot(step, seed)

    def _draw_snapshot(self, step: int, seed: int) -> OpponentSpec:
        """Pick a snapshot (pinned or unpinned) with anchor-floor + PFSP.

        Extracted from `draw()` so both v3 and legacy modes share the
        identical snapshot-selection logic — only the upstream heuristic/API
        gates differ. The anchor-floor + PFSP weighting are unchanged from
        the pre-v3 code; this method just gives them a stable call-site.
        """
        # Anchor floor (Codex fix): when a pinned anchor exists and
        # `p_anchor_floor > 0`, reserve that fraction of snapshot draws for a
        # uniform pick across pinned snapshots. Without this, once the trainee
        # crushes the anchor (WR ≈ 0.95 ⇒ f_hard floored to 0.01), the anchor
        # gets ≈ 1% of draws — barely anchoring. With floor=0.15, the anchor
        # always sees ≥ 15% of snapshot draws regardless of PFSP weights.
        if (self.p_anchor_floor > 0.0 and self._pinned_snapshots
                and _unit(self.rng_seed, step, seed, "anchor-floor")
                < self.p_anchor_floor):
            pinned_specs = [self._snapshot_spec(s)
                            for s in self._pinned_snapshots]
            return _pick(pinned_specs, self.rng_seed, step, seed, "anchor-pick")
        # Snapshot bucket (pinned + unpinned), PFSP-weighted by per-snapshot
        # observed win-rate.
        all_snaps = self.snapshots()
        if all_snaps:
            specs = [self._snapshot_spec(s) for s in all_snaps]
            weights, _regime = self._pfsp_weights(all_snaps)
            return _weighted_pick(
                specs, weights, self.rng_seed, step, seed, "snap-pick-pfsp")
        # Empty pool + no API/current-self. Last-ditch heuristic fallback if configured.
        if self.heuristic_spec is not None:
            return self.heuristic_spec
        raise RuntimeError(
            "OpponentPool.draw: pool is empty; checkpoint bucket is empty (no pinned "
            "snapshots and no unpinned snapshots). Caller must pre-seed a "
            "pinned anchor before trainer.train(); otherwise the residual "
            "checkpoint probability mass has nothing to draw."
        )

    @staticmethod
    def _snapshot_spec(snap: Snapshot) -> OpponentSpec:
        return OpponentSpec(
            kind="snapshot", served_name=snap.served_name,
            actor_id=f"snapshot_{snap.step}", step=snap.step)

    def telemetry(self, step: int) -> dict:
        """Pool state at `step` — merged into the run JSON per refresh."""
        all_snaps = self.snapshots()
        weights, regime = self._pfsp_weights(all_snaps) if all_snaps else ([], "empty")
        # Snapshot of the (snapshot_step → win_rate, n_games) feedback.
        wr_view = {
            s.step: {
                "win_rate": self._snapshot_winrate[s.step][0],
                "n_games": self._snapshot_winrate[s.step][1],
                "pfsp_weight": (weights[i] if i < len(weights) else None),
                "pinned": s in self._pinned_snapshots,
            }
            for i, s in enumerate(all_snaps)
            if s.step in self._snapshot_winrate
        }
        out = {
            "p_anneal": round(self.anneal_probability(step), 4),
            "n_pinned": len(self._pinned_snapshots),
            "n_snapshots": len(self._snapshots),
            "n_snapshots_total": len(all_snaps),
            "snapshot_steps": [s.step for s in all_snaps],
            "pinned_steps": [s.step for s in self._pinned_snapshots],
            "n_api": len(self.api_specs),
            "api_weights": list(self.api_weights),
            "p_current_self": round(self.p_current_self, 4),
            "current_self_configured": self.current_self_spec is not None,
            "pfsp_regime": regime,
            "snapshot_winrates": wr_view,
        }
        # repl_08 v3: heuristic decay state. Only emitted when configured
        # so legacy/no-heuristic runs keep clean telemetry.
        if self.p_heuristic > 0.0:
            out["p_heuristic"] = round(self.p_heuristic, 4)
            out["p_heuristic_effective"] = round(
                self._p_heuristic_effective(step), 4)
            out["heuristic_anneal_end"] = self.heuristic_anneal_end
            out["heuristic_step_factor"] = round(
                self._heuristic_step_factor(step), 4)
            if self._heuristic_winrate is not None:
                out["heuristic_winrate"] = {
                    "win_rate": self._heuristic_winrate[0],
                    "n_games": self._heuristic_winrate[1],
                }
        # Codex-flagged: expose the REALIZED marginal probabilities of each
        # opponent kind so consumers don't have to re-derive them from the
        # raw knobs. In v3 mode these are the exact gate boundaries; in
        # legacy mode they account for the anneal × p_api interaction.
        if self.p_heuristic > 0.0 or self.p_current_self > 0.0:
            p_heur = self._p_heuristic_effective(step)
            p_api = self.p_api if self._api_specs_expanded else 0.0
            p_self = (
                self.p_current_self
                if self.current_self_spec is not None else 0.0
            )
            p_snap = max(0.0, 1.0 - p_heur - p_api - p_self)
            mode = (
                "league_mix"
                if self.p_current_self > 0.0
                else "v3_heuristic_with_decay"
            )
        else:
            p_anneal = self.anneal_probability(step)
            p_legacy_heur = (1.0 - p_anneal) if self.heuristic_spec is not None else 0.0
            # Within the (1 - legacy_heur) post-anneal share, p_api carves
            # the API rate; the rest goes to snapshots.
            p_post = 1.0 - p_legacy_heur
            p_api_marginal = (
                self.p_api * p_post if self._api_specs_expanded else 0.0)
            p_heur = p_legacy_heur
            p_api = p_api_marginal
            p_self = 0.0
            p_snap = max(0.0, 1.0 - p_heur - p_api)
            mode = "legacy_anneal" if self.heuristic_spec is not None else "pure_pool"
        out["draw_mode"] = mode
        out["realized_marginal_p"] = {
            "heuristic": round(p_heur, 4),
            "api": round(p_api, 4),
            "current_self": round(p_self, 4),
            "snapshot": round(p_snap, 4),
            "checkpoint": round(p_snap, 4),
        }
        return out
