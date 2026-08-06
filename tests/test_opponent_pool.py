"""Tests for the repl_08 opponent-pool refactor: pinned anchors + PFSP.

The repl_07-era pool tests live in tests/test_phase3_grpo_run.py (kept green
by the back-compat keyword args). This file covers what's NEW in repl_08:
  * Pinned snapshots are anchors — never evicted by add_snapshot.
  * Pinned anchors have a minimum draw PROBABILITY (`p_anchor_floor`),
    so PFSP can't drown them out once the trainee crushes them.
  * PFSP `f_hard(x)=(1-x)^2` weighting bites once any snapshot has ≥10
    games (under-observed snapshots get a per-snapshot WR=0.5 prior;
    the bucket regime is NOT bucket-wide reset to uniform — Codex round-1).
  * `f_var(x)=x(1-x)` fallback fires only when ALL snapshots are
    real-observed AND every observed WR <0.25 (Codex round-2: a WR=0.5
    prior would peak f_var and over-sample fresh snapshots).
  * Cold-start `uniform` regime fires ONLY when ZERO snapshots have
    real observations (true pre-first-roll case).
  * Empty pool with no heuristic fallback raises (loud, not silent).
  * Determinism of the weighted pick across processes.
"""

from __future__ import annotations

import pytest

from megagem.training.opponent_pool import (
    OpponentPool, OpponentSpec, Snapshot, _PFSP_MIN_N_GAMES,
    _PFSP_W_MIN, _PFSP_F_VAR_TRIGGER, _PFSP_UNOBSERVED_WR_PRIOR,
    _PFSP_WR_EMA_ALPHA, _P_HEURISTIC_DECAY_FLOOR,
)


def _heuristic() -> OpponentSpec:
    return OpponentSpec("heuristic", "megagem/heuristic-v1", "heuristic")


def _current_self() -> OpponentSpec:
    return OpponentSpec("current_self", "phase3-trainable", "current_self")


def _snap(step: int) -> Snapshot:
    return Snapshot(step=step, served_name=f"phase3-snap-{step}",
                    adapter_path=f"/tmp/p/step_{step}")


# --------------------------------------------------------------------------- #
# Pinned anchors                                                               #
# --------------------------------------------------------------------------- #
def test_pinned_anchor_survives_unpinned_eviction():
    """The 6th UNPINNED snapshot in a max=5 pool evicts the oldest unpinned,
    NEVER the pinned anchor."""
    anchor = _snap(0)
    pool = OpponentPool(pinned_snapshots=[anchor], max_snapshots=5)
    for st in (25, 50, 75, 100, 125):
        assert pool.add_snapshot(_snap(st)) is None
    # 6th unpinned ⇒ evict oldest unpinned (step 25); anchor stays.
    evicted = pool.add_snapshot(_snap(150))
    assert evicted is not None and evicted.step == 25
    assert [s.step for s in pool.pinned_snapshots()] == [0]
    assert [s.step for s in pool.unpinned_snapshots()] == [50, 75, 100, 125, 150]
    assert [s.step for s in pool.snapshots()] == [0, 50, 75, 100, 125, 150]


def test_add_pinned_snapshot_idempotent():
    """add_pinned_snapshot is a no-op when the same (step, served_name) is
    already pinned — protects against double-pre-seeding on retry."""
    anchor = _snap(0)
    pool = OpponentPool(pinned_snapshots=[])
    pool.add_pinned_snapshot(anchor)
    pool.add_pinned_snapshot(anchor)
    pool.add_pinned_snapshot(Snapshot(0, "phase3-snap-0", "/different/path"))
    assert len(pool.pinned_snapshots()) == 1
    assert pool.pinned_snapshots()[0].adapter_path == "/tmp/p/step_0"


def test_pinned_snapshot_appears_in_draw_distribution():
    """draw() must include the pinned anchor in the snapshot bucket — at step
    0 with only the anchor in the pool, every draw returns the anchor."""
    anchor = _snap(0)
    pool = OpponentPool(
        pinned_snapshots=[anchor], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0, rng_seed=0)
    kinds = {pool.draw(0, s).kind for s in range(9000, 9100)}
    assert kinds == {"snapshot"}
    names = {pool.draw(0, s).served_name for s in range(9000, 9100)}
    assert names == {"phase3-snap-0"}


# --------------------------------------------------------------------------- #
# Current-self / checkpoint league mix                                        #
# --------------------------------------------------------------------------- #
def test_current_self_checkpoint_mix_is_80_20_no_heuristic():
    pool = OpponentPool(
        pinned_snapshots=[_snap(0)],
        max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0,
        current_self_spec=_current_self(),
        p_current_self=0.80,
        p_heuristic=0.0,
        rng_seed=123)
    draws = [pool.draw(50, s).kind for s in range(10_000, 12_000)]
    self_frac = draws.count("current_self") / len(draws)
    snap_frac = draws.count("snapshot") / len(draws)
    assert 0.76 <= self_frac <= 0.84
    assert 0.16 <= snap_frac <= 0.24
    assert "heuristic" not in draws
    telemetry = pool.telemetry(50)
    assert telemetry["draw_mode"] == "league_mix"
    assert telemetry["realized_marginal_p"]["current_self"] == pytest.approx(0.8)
    assert telemetry["realized_marginal_p"]["checkpoint"] == pytest.approx(0.2)


def test_current_self_mix_rejects_invalid_probability_sum():
    with pytest.raises(ValueError, match="opponent draws cannot sum"):
        OpponentPool(
            pinned_snapshots=[_snap(0)],
            current_self_spec=_current_self(),
            p_current_self=0.80,
            p_api=0.25,
            p_heuristic=0.0)


def test_current_self_probability_requires_spec():
    with pytest.raises(ValueError, match="current_self_spec"):
        OpponentPool(pinned_snapshots=[_snap(0)], p_current_self=0.80)


def test_anchor_floor_still_applies_inside_checkpoint_bucket():
    pool = OpponentPool(
        pinned_snapshots=[_snap(0)],
        max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0,
        current_self_spec=_current_self(),
        p_current_self=0.80,
        p_anchor_floor=0.50,
        rng_seed=456)
    pool.add_snapshot(_snap(25))
    pool.update_snapshot_winrate(0, win_rate=0.95, n_games=100)
    pool.update_snapshot_winrate(25, win_rate=0.05, n_games=100)
    snapshots = [
        pool.draw(100, s)
        for s in range(20_000, 24_000)
        if pool.draw(100, s).kind == "snapshot"
    ]
    assert snapshots, "expected some checkpoint-bucket draws"
    anchor_frac = sum(1 for spec in snapshots if spec.step == 0) / len(snapshots)
    assert 0.40 <= anchor_frac <= 0.60


# --------------------------------------------------------------------------- #
# PFSP weighting                                                               #
# --------------------------------------------------------------------------- #
def test_pfsp_cold_start_is_uniform_only_when_zero_observed():
    """Codex fix: 'uniform' regime fires ONLY when NO snapshot has reached
    `_PFSP_MIN_N_GAMES` observations. The pre-Codex behaviour ("any
    under-observed snapshot ⇒ uniform for all") meant every snapshot rotation
    reset PFSP to uniform — defeating the entire mechanism."""
    pool = OpponentPool(
        pinned_snapshots=[_snap(0)], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0, rng_seed=0)
    for st in (25, 50, 75):
        pool.add_snapshot(_snap(st))
    # ZERO observations ≥10 games ⇒ true cold start ⇒ uniform.
    pool.update_snapshot_winrate(25, win_rate=0.10, n_games=5)
    pool.update_snapshot_winrate(50, win_rate=0.95, n_games=5)
    assert pool.telemetry(step=200)["pfsp_regime"] == "uniform"


def test_pfsp_under_observed_snapshot_gets_neutral_prior_not_bucket_reset():
    """Codex fix: when SOME snapshots have ≥10 games and others don't, the
    under-observed ones get a WR=0.5 prior (f_hard weight=0.25) but the
    regime stays "f_hard" — adding a fresh snapshot doesn't reset PFSP for
    the rest of the bucket."""
    snaps = [_snap(25), _snap(50), _snap(75)]
    pool = OpponentPool(
        pinned_snapshots=[], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0, rng_seed=0)
    for s in snaps:
        pool.add_snapshot(s)
    # Two snapshots observed (≥10 games), one fresh (no obs).
    pool.update_snapshot_winrate(25, win_rate=0.10, n_games=100)
    pool.update_snapshot_winrate(50, win_rate=0.90, n_games=100)
    # snap 75 ⇒ no observation
    weights, regime = pool._pfsp_weights(snaps)
    assert regime == "f_hard"
    # snap 25 (hard): (1-0.10)^2 = 0.81
    assert weights[0] == pytest.approx(0.81)
    # snap 50 (easy): max(0.01, (1-0.90)^2 = 0.01) = 0.01
    assert weights[1] == pytest.approx(_PFSP_W_MIN)
    # snap 75 (unobserved): uses prior WR=0.5 ⇒ (1-0.5)^2 = 0.25
    expected_prior = (1.0 - _PFSP_UNOBSERVED_WR_PRIOR) ** 2
    assert weights[2] == pytest.approx(expected_prior)


def test_pfsp_f_var_requires_all_snapshots_observed():
    """Codex residual-risk fix: in f_var mode `x(1-x)` peaks at x=0.5, so a
    prior-only snapshot with WR=0.5 would receive the MAXIMUM f_var weight —
    fresh snapshots would be over-sampled before any real evidence. The fix:
    f_var only fires when EVERY snapshot has ≥_PFSP_MIN_N_GAMES of REAL
    observations. With one prior-only entry, we stay in f_hard (where the
    WR=0.5 prior maps to the mid-range weight 0.25)."""
    snaps = [_snap(25), _snap(50), _snap(75)]
    pool = OpponentPool(
        pinned_snapshots=[], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0, rng_seed=0)
    for s in snaps:
        pool.add_snapshot(s)
    # All REAL observations would justify f_var; but one snap is prior-only.
    pool.update_snapshot_winrate(25, win_rate=0.10, n_games=100)
    pool.update_snapshot_winrate(50, win_rate=0.20, n_games=100)
    # snap 75 unobserved ⇒ stay in f_hard
    _w, regime = pool._pfsp_weights(snaps)
    assert regime == "f_hard", (
        "prior-only snap must keep us in f_hard so it isn't over-sampled")
    # Now observe snap 75 too; f_var should fire.
    pool.update_snapshot_winrate(75, win_rate=0.18, n_games=100)
    _w, regime = pool._pfsp_weights(snaps)
    assert regime == "f_var"


def test_pfsp_weight_ratio_low_vs_mid_winrate_is_modest():
    """Sanity-check on PFSP magnitude — Codex flagged that the original plan
    overstated weight ratios. WR=0.06 vs WR=0.42 (the actual repl_07 per-
    snapshot range) ⇒ weight ratio of ~2.6×, NOT 88× as the plan first
    claimed."""
    snaps = [_snap(25), _snap(50)]
    pool = OpponentPool(
        pinned_snapshots=[], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0, rng_seed=0)
    for s in snaps:
        pool.add_snapshot(s)
    pool.update_snapshot_winrate(25, win_rate=0.06, n_games=100)
    pool.update_snapshot_winrate(50, win_rate=0.42, n_games=100)
    w, regime = pool._pfsp_weights(snaps)
    assert regime == "f_hard"
    ratio = w[0] / w[1]
    assert 2.0 < ratio < 3.5, f"PFSP ratio {ratio:.2f} not in expected band"


def test_pfsp_f_hard_weights_harder_opponents_higher():
    """`f_hard(x)=(1-x)^2`: WR=0.05 (hard, trainee usually loses) gets the
    largest weight; WR=0.95 (easy) is pinned at the _PFSP_W_MIN floor."""
    snaps = [_snap(25), _snap(50), _snap(75)]
    pool = OpponentPool(
        pinned_snapshots=[], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0, rng_seed=0)
    for s in snaps:
        pool.add_snapshot(s)
    pool.update_snapshot_winrate(25, win_rate=0.05, n_games=100)
    pool.update_snapshot_winrate(50, win_rate=0.50, n_games=100)
    pool.update_snapshot_winrate(75, win_rate=0.95, n_games=100)
    weights, regime = pool._pfsp_weights(snaps)
    assert regime == "f_hard"
    assert weights[0] == pytest.approx(0.9025)         # (1 - 0.05)^2
    assert weights[1] == pytest.approx(0.25)           # (1 - 0.50)^2
    # WR=0.95 ⇒ raw (1-0.95)^2 = 0.0025; floored to _PFSP_W_MIN=1e-2.
    assert weights[2] == pytest.approx(max(_PFSP_W_MIN, (1 - 0.95) ** 2))
    # Monotonicity: hard > medium > easy.
    assert weights[0] > weights[1] > weights[2]


def test_pfsp_f_var_fallback_when_stuck_below_uniform():
    """When ALL snapshots have WR < _PFSP_F_VAR_TRIGGER (0.25) — i.e. the
    trainee is being crushed by every opponent — f_hard would weight all the
    same too-hard opponents. Fall back to f_var(x)=x(1-x) (peaks at 0.5) so a
    matched-strength snapshot, if any later appears, will rise."""
    snaps = [_snap(25), _snap(50), _snap(75)]
    pool = OpponentPool(
        pinned_snapshots=[], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0, rng_seed=0)
    for s in snaps:
        pool.add_snapshot(s)
    pool.update_snapshot_winrate(25, win_rate=0.05, n_games=100)
    pool.update_snapshot_winrate(50, win_rate=0.15, n_games=100)
    pool.update_snapshot_winrate(75, win_rate=0.20, n_games=100)  # max < 0.25
    weights, regime = pool._pfsp_weights(snaps)
    assert regime == "f_var"
    # f_var(x)=x(1-x): WR=0.20 is closest to 0.5 ⇒ highest f_var weight.
    assert weights[2] > weights[1] > weights[0]


def test_pfsp_f_hard_active_when_at_least_one_above_trigger():
    """f_var fires ONLY when MAX(WR) < trigger; one above-trigger snapshot
    flips us back to f_hard. The trigger is on the worst-case, not the mean."""
    snaps = [_snap(25), _snap(50)]
    pool = OpponentPool(
        pinned_snapshots=[], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0, rng_seed=0)
    for s in snaps:
        pool.add_snapshot(s)
    pool.update_snapshot_winrate(25, win_rate=0.10, n_games=100)
    pool.update_snapshot_winrate(50, win_rate=_PFSP_F_VAR_TRIGGER + 0.01,
                                 n_games=100)
    _w, regime = pool._pfsp_weights(snaps)
    assert regime == "f_hard"


def test_pfsp_draw_oversamples_low_winrate_snapshot():
    """End-to-end: with f_hard active, low-WR snapshots are drawn substantially
    more often than high-WR ones. We don't pin the exact ratio (depends on the
    sha256-driven shuffle across seeds) — just that the harder one wins the
    majority of the draws by a wide margin."""
    snaps = [_snap(25), _snap(50)]
    pool = OpponentPool(
        pinned_snapshots=[], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0, rng_seed=0)
    for s in snaps:
        pool.add_snapshot(s)
    # WR 0.05 ⇒ weight ~0.9025; WR 0.95 ⇒ weight ~0.01 (floor) ⇒ ~90:1.
    pool.update_snapshot_winrate(25, win_rate=0.05, n_games=200)
    pool.update_snapshot_winrate(50, win_rate=0.95, n_games=200)
    picked = [pool.draw(1000, s).step for s in range(9000, 9500)]
    assert picked.count(25) > 400          # ~99% expected; >80% is loose
    assert picked.count(50) < 100


# --------------------------------------------------------------------------- #
# Determinism                                                                  #
# --------------------------------------------------------------------------- #
def test_weighted_pick_deterministic_across_pool_instances():
    """Two pools with identical (rng_seed, snapshots, WR observations) must
    produce identical draws across seeds — the cumulative-weight binary search
    has no PRNG state, only the sha256-derived `_unit`."""
    def _mk():
        snaps = [_snap(25), _snap(50), _snap(75)]
        p = OpponentPool(
            pinned_snapshots=[], max_snapshots=5,
            anneal_start=0, anneal_end=0, p_max=1.0, rng_seed=7)
        for s in snaps:
            p.add_snapshot(s)
        p.update_snapshot_winrate(25, 0.10, 100)
        p.update_snapshot_winrate(50, 0.50, 100)
        p.update_snapshot_winrate(75, 0.90, 100)
        return p
    a, b = _mk(), _mk()
    for seed in range(9000, 9050):
        assert a.draw(123, seed).served_name == b.draw(123, seed).served_name


# --------------------------------------------------------------------------- #
# Empty-pool failure                                                            #
# --------------------------------------------------------------------------- #
def test_empty_pool_no_heuristic_raises_runtime_error():
    """repl_08 design: with the heuristic removed and no pre-seed, draw() must
    raise loud — never silently fall back to an opponent the user removed."""
    pool = OpponentPool(
        pinned_snapshots=[], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0,
        rng_seed=0, heuristic_spec=None)
    with pytest.raises(RuntimeError, match="pool is empty"):
        pool.draw(100, 9000)


def test_empty_pool_with_legacy_heuristic_falls_back():
    """Back-compat: when the legacy heuristic_spec is provided and the pool is
    empty, draw() returns the heuristic. This is the `--no-opponent-pool` path
    used by the dry-run and repl_07-era reproduction."""
    pool = OpponentPool(
        pinned_snapshots=[], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0,
        rng_seed=0, heuristic_spec=_heuristic())
    # No snapshots, no API ⇒ heuristic.
    assert pool.draw(100, 9000).kind == "heuristic"


# --------------------------------------------------------------------------- #
# API mixing with snapshots                                                    #
# --------------------------------------------------------------------------- #
def test_api_p10_with_pinned_anchor_mixes():
    """p_api=0.10 with a pinned anchor + Flash: ~10% of draws are API; the
    rest are snapshots (the pinned anchor)."""
    anchor = _snap(0)
    api = [OpponentSpec("api", "google/gemini-3-flash-preview", "api_g")]
    pool = OpponentPool(
        pinned_snapshots=[anchor], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0,
        api_specs=api, p_api=0.10, rng_seed=0)
    kinds = [pool.draw(500, s).kind for s in range(9000, 9500)]
    n_api = kinds.count("api")
    n_snap = kinds.count("snapshot")
    assert 30 < n_api < 80                       # ~50 expected at p=0.10
    assert n_snap > 400


# --------------------------------------------------------------------------- #
# update_snapshot_winrate storage policy (Codex round-3)                       #
# --------------------------------------------------------------------------- #
def test_update_snapshot_winrate_first_call_stores_directly():
    """First observation of a snapshot stores (wr, n) verbatim."""
    pool = OpponentPool(pinned_snapshots=[_snap(0)])
    pool.update_snapshot_winrate(25, win_rate=0.4, n_games=50)
    assert pool.snapshot_winrates()[25] == (0.4, 50)


def test_update_snapshot_winrate_accumulates_n_games_monotonically():
    """Codex round-3 fix: n_games grows cumulatively across calls — small-
    sample rolls never downgrade trust. Pre-fix, a roll with n=8 would
    overwrite a trusted n=100 entry and flicker PFSP back to prior."""
    pool = OpponentPool(pinned_snapshots=[_snap(0)])
    pool.update_snapshot_winrate(25, win_rate=0.30, n_games=100)
    pool.update_snapshot_winrate(25, win_rate=0.50, n_games=8)
    wr, n = pool.snapshot_winrates()[25]
    # n_games strictly grows; never falls below previous.
    assert n == 108


def test_update_snapshot_winrate_uses_ewma_on_wr():
    """WR EWMA: new_wr = (1 - α) · old_wr + α · obs_wr. With α=0.3,
    obs=0.50 on top of stored=0.30 ⇒ new_wr = 0.7·0.30 + 0.3·0.50 = 0.36."""
    pool = OpponentPool(pinned_snapshots=[_snap(0)])
    pool.update_snapshot_winrate(25, win_rate=0.30, n_games=100)
    pool.update_snapshot_winrate(25, win_rate=0.50, n_games=8)
    wr, _n = pool.snapshot_winrates()[25]
    expected = (1.0 - _PFSP_WR_EMA_ALPHA) * 0.30 + _PFSP_WR_EMA_ALPHA * 0.50
    assert wr == pytest.approx(expected)


def test_small_sample_roll_no_longer_demotes_snapshot_to_prior():
    """End-to-end regression for the bug Codex round-3 flagged: a trusted
    snapshot (n=100) followed by a one-seed roll (n=8) must REMAIN trusted
    on the next draw — not drop into the WR=0.5-prior regime."""
    snaps = [_snap(25), _snap(50)]
    pool = OpponentPool(pinned_snapshots=[], max_snapshots=5)
    for s in snaps:
        pool.add_snapshot(s)
    # Snap 25: trusted, low WR (the kind PFSP would over-sample).
    pool.update_snapshot_winrate(25, win_rate=0.10, n_games=100)
    pool.update_snapshot_winrate(50, win_rate=0.40, n_games=100)
    # Next roll: snap 25 drawn for one seed only → n=8 < _PFSP_MIN_N_GAMES.
    pool.update_snapshot_winrate(25, win_rate=0.20, n_games=8)
    weights, regime = pool._pfsp_weights(snaps)
    assert regime == "f_hard"
    # snap 25 must still be trusted (n=108), not fall into WR=0.5 prior
    # (which would give weight 0.25; trusted-WR≈0.13 EWMA gives ~0.75).
    assert weights[0] > 0.6, (
        f"trusted snap demoted by small-sample overwrite — weight {weights[0]} "
        f"too close to prior weight 0.25")


def test_update_snapshot_winrate_rejects_invalid_inputs():
    """Non-finite WR or non-positive n_games are dropped silently — telemetry
    must never aborts a roll, even with a degenerate input."""
    pool = OpponentPool(pinned_snapshots=[_snap(0)])
    pool.update_snapshot_winrate(25, win_rate=float("nan"), n_games=100)
    pool.update_snapshot_winrate(25, win_rate=0.5, n_games=0)
    pool.update_snapshot_winrate(25, win_rate=0.5, n_games=-5)
    assert 25 not in pool.snapshot_winrates()


# --------------------------------------------------------------------------- #
# Pinned anchor probability floor (Codex fix)                                  #
# --------------------------------------------------------------------------- #
def test_anchor_floor_reserves_minimum_draw_probability():
    """With p_anchor_floor=0.2 and an anchor that PFSP would weight to ~0.01
    (trainee crushing it), the anchor still receives ~20% of snapshot draws."""
    anchor = _snap(0)
    snaps_unpinned = [_snap(25), _snap(50)]
    pool = OpponentPool(
        pinned_snapshots=[anchor], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0,
        p_anchor_floor=0.2, rng_seed=0)
    for s in snaps_unpinned:
        pool.add_snapshot(s)
    # Trainee crushes anchor (WR=0.95) but the unpinned snapshots are tough.
    pool.update_snapshot_winrate(0, win_rate=0.95, n_games=200)
    pool.update_snapshot_winrate(25, win_rate=0.10, n_games=200)
    pool.update_snapshot_winrate(50, win_rate=0.15, n_games=200)
    picked = [pool.draw(1000, s).step for s in range(9000, 9500)]
    anchor_frac = picked.count(0) / len(picked)
    # 20% expected from the anchor-floor branch + ~0% from PFSP (floored).
    assert 0.15 < anchor_frac < 0.30, f"anchor_frac={anchor_frac:.3f} not near 0.20"


def test_anchor_floor_zero_disables_branch():
    """p_anchor_floor=0.0 ⇒ identical behaviour to pre-Codex pool: PFSP alone
    governs the snapshot pick. With anchor WR=0.95 it gets ~floored weight."""
    anchor = _snap(0)
    snaps_unpinned = [_snap(25), _snap(50)]
    pool = OpponentPool(
        pinned_snapshots=[anchor], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0,
        p_anchor_floor=0.0, rng_seed=0)
    for s in snaps_unpinned:
        pool.add_snapshot(s)
    pool.update_snapshot_winrate(0, win_rate=0.95, n_games=200)
    pool.update_snapshot_winrate(25, win_rate=0.10, n_games=200)
    pool.update_snapshot_winrate(50, win_rate=0.15, n_games=200)
    picked = [pool.draw(1000, s).step for s in range(9000, 9500)]
    anchor_frac = picked.count(0) / len(picked)
    # No floor ⇒ anchor weight ~0.01 vs ~0.81+0.7225, so ~0.6% draw share.
    assert anchor_frac < 0.05, f"anchor_frac={anchor_frac:.3f} expected near-zero"


def test_p_anchor_floor_validation():
    """Out-of-range values must raise at construction — better loud than
    silently mis-configured."""
    with pytest.raises(ValueError, match="p_anchor_floor"):
        OpponentPool(pinned_snapshots=[_snap(0)], p_anchor_floor=-0.1)
    with pytest.raises(ValueError, match="p_anchor_floor"):
        OpponentPool(pinned_snapshots=[_snap(0)], p_anchor_floor=1.5)


# --------------------------------------------------------------------------- #
# Telemetry surface                                                            #
# --------------------------------------------------------------------------- #
def test_telemetry_reports_pinned_and_unpinned_counts():
    pool = OpponentPool(
        pinned_snapshots=[_snap(0)], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0, rng_seed=0)
    for st in (25, 50, 75):
        pool.add_snapshot(_snap(st))
    t = pool.telemetry(step=200)
    assert t["n_pinned"] == 1
    assert t["n_snapshots"] == 3
    assert t["n_snapshots_total"] == 4
    assert t["snapshot_steps"] == [0, 25, 50, 75]
    assert t["pinned_steps"] == [0]
    assert t["pfsp_regime"] == "uniform"          # no WR observations


# --------------------------------------------------------------------------- #
# Heuristic-with-decay (repl_08 v3)                                            #
# --------------------------------------------------------------------------- #
def test_p_heuristic_requires_heuristic_spec():
    """`p_heuristic > 0` without a `heuristic_spec` must raise — the draw()
    branch would have nothing to return otherwise."""
    with pytest.raises(ValueError, match="p_heuristic"):
        OpponentPool(
            pinned_snapshots=[_snap(0)], heuristic_spec=None,
            p_heuristic=0.25)


def test_p_heuristic_range_validation():
    """Out-of-range p_heuristic must raise at construction."""
    with pytest.raises(ValueError, match="p_heuristic"):
        OpponentPool(
            pinned_snapshots=[_snap(0)], heuristic_spec=_heuristic(),
            p_heuristic=-0.1)
    with pytest.raises(ValueError, match="p_heuristic"):
        OpponentPool(
            pinned_snapshots=[_snap(0)], heuristic_spec=_heuristic(),
            p_heuristic=1.5)


def test_p_heuristic_zero_disables_branch():
    """Default `p_heuristic=0.0` ⇒ `_p_heuristic_effective() == 0.0` and
    draw() never enters the heuristic-with-decay branch — back-compat with
    pre-v3 callers. (Step > 0 so the LEGACY anneal branch doesn't fire
    either; anneal_start=0 returns p_anneal=0 at step=0, which would route
    every draw to the legacy heuristic-fallback — a separate code path.)"""
    pool = OpponentPool(
        pinned_snapshots=[_snap(0)], heuristic_spec=_heuristic(),
        p_heuristic=0.0,
        anneal_start=0, anneal_end=0, p_max=1.0, rng_seed=0)
    assert pool._p_heuristic_effective() == 0.0
    # All draws hit the snapshot bucket (anchor) — no heuristic shows up.
    kinds = {pool.draw(100, s).kind for s in range(9000, 9100)}
    assert kinds == {"snapshot"}


def test_p_heuristic_cold_start_uses_full_p():
    """Before _PFSP_MIN_N_GAMES heuristic games observed, the effective P
    equals the configured P — gives the trainee a clean +R signal from the
    very first roll."""
    pool = OpponentPool(
        pinned_snapshots=[_snap(0)], heuristic_spec=_heuristic(),
        p_heuristic=0.25,
        anneal_start=0, anneal_end=0, p_max=1.0, rng_seed=0)
    # No observations yet ⇒ full p.
    assert pool._p_heuristic_effective() == pytest.approx(0.25)
    # Even a small-sample first observation (below MIN_N_GAMES) keeps cold.
    pool.update_heuristic_winrate(win_rate=0.80, n_games=_PFSP_MIN_N_GAMES - 1)
    assert pool._p_heuristic_effective() == pytest.approx(0.25)


def test_p_heuristic_decay_monotonic_in_winrate():
    """At fixed p_heuristic, effective P is non-increasing in WR (over the
    [0, 1] range): the more the trainee wins, the less heuristic share it
    gets. Past the floor crossing (WR > 0.553) the value is flat at the
    floor — non-strict monotonicity (≤)."""
    # Sweep WR with enough observations to leave cold-start.
    wrs = [0.0, 0.1, 0.25, 0.5, 0.7, 0.85, 1.0]
    effs = []
    for wr in wrs:
        # Reset by constructing a fresh pool (avoids EWMA accumulation).
        p = OpponentPool(
            pinned_snapshots=[_snap(0)], heuristic_spec=_heuristic(),
            p_heuristic=0.30,
            anneal_start=0, anneal_end=0, p_max=1.0, rng_seed=0)
        p.update_heuristic_winrate(wr, _PFSP_MIN_N_GAMES * 5)
        effs.append(p._p_heuristic_effective())
    # Non-strict monotone non-increasing.
    for a, b in zip(effs, effs[1:]):
        assert b <= a + 1e-12, f"effective P went up: {effs}"
    # Endpoints land where the formula predicts.
    assert effs[0] == pytest.approx(0.30)  # WR=0 ⇒ no decay
    assert effs[-1] == pytest.approx(0.30 * _P_HEURISTIC_DECAY_FLOOR)  # floor


def test_p_heuristic_floor_active_above_break_winrate():
    """The decay multiplier `(1-WR)²` falls below `_P_HEURISTIC_DECAY_FLOOR`
    once WR > 1 - sqrt(floor). With floor=0.10, that crossing is at WR≈0.684.
    For WR=0.75 and above, the floor binds: effective P = p · floor.
    For WR=0.50, decay applies."""
    p_heur = 0.25
    # WR=0.75 ⇒ (1-0.75)² = 0.0625 < 0.10 floor.
    pool = OpponentPool(
        pinned_snapshots=[_snap(0)], heuristic_spec=_heuristic(),
        p_heuristic=p_heur, rng_seed=0)
    pool.update_heuristic_winrate(0.75, _PFSP_MIN_N_GAMES * 5)
    assert pool._p_heuristic_effective() == pytest.approx(
        p_heur * _P_HEURISTIC_DECAY_FLOOR)
    # WR=0.5 ⇒ (1-0.5)² = 0.25 > 0.10 floor; decay applies.
    pool2 = OpponentPool(
        pinned_snapshots=[_snap(0)], heuristic_spec=_heuristic(),
        p_heuristic=p_heur, rng_seed=0)
    pool2.update_heuristic_winrate(0.50, _PFSP_MIN_N_GAMES * 5)
    assert pool2._p_heuristic_effective() == pytest.approx(p_heur * 0.25)


def test_p_heuristic_draw_proportional_to_effective_p():
    """End-to-end: with `p_heuristic=0.30` and WR=0.0 (cold start treats as
    full p), ~30% of draws return the heuristic. Loose bounds because the
    sha256-driven draw schedule is not perfectly uniform on small samples."""
    anchor = _snap(0)
    pool = OpponentPool(
        pinned_snapshots=[anchor], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0,
        heuristic_spec=_heuristic(), p_heuristic=0.30, rng_seed=0)
    kinds = [pool.draw(500, s).kind for s in range(9000, 9500)]
    n_heur = kinds.count("heuristic")
    n_snap = kinds.count("snapshot")
    # 30% of 500 = 150; allow ±20% slack on small-sample variance.
    assert 110 < n_heur < 200, f"n_heur={n_heur} not near expected 150"
    assert 300 < n_snap < 400, f"n_snap={n_snap} not near expected 350"


def test_p_heuristic_draw_decays_after_wr_observed():
    """Once heuristic WR climbs to the floor regime (WR>0.684 with floor=
    0.10), draws drop to ~floor × p_heuristic. With p=0.30 and floor=0.10,
    that's 0.03 ⇒ ~15 of 500 draws."""
    anchor = _snap(0)
    pool = OpponentPool(
        pinned_snapshots=[anchor], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0,
        heuristic_spec=_heuristic(), p_heuristic=0.30, rng_seed=0)
    # Make heuristic WR=0.85 (trainee dominates) with enough games to leave
    # cold-start.
    pool.update_heuristic_winrate(0.85, _PFSP_MIN_N_GAMES * 5)
    kinds = [pool.draw(500, s).kind for s in range(9000, 9500)]
    n_heur = kinds.count("heuristic")
    # Expected: 0.30 * 0.10 = 0.03 ⇒ 15 of 500. Loose bound for SHA-bucketing.
    assert 5 < n_heur < 35, f"n_heur={n_heur} not near expected ~15"


def test_update_heuristic_winrate_ewma_and_cumulative():
    """Storage policy mirrors update_snapshot_winrate: cumulative n_games,
    EWMA on win_rate."""
    pool = OpponentPool(
        pinned_snapshots=[_snap(0)], heuristic_spec=_heuristic(),
        p_heuristic=0.20)
    pool.update_heuristic_winrate(win_rate=0.60, n_games=100)
    pool.update_heuristic_winrate(win_rate=0.80, n_games=50)
    wr, n = pool.heuristic_winrate()
    # n is cumulative.
    assert n == 150
    # wr is EWMA: 0.7·0.60 + 0.3·0.80 = 0.66
    expected = (1 - _PFSP_WR_EMA_ALPHA) * 0.60 + _PFSP_WR_EMA_ALPHA * 0.80
    assert wr == pytest.approx(expected)


def test_update_heuristic_winrate_rejects_invalid_inputs():
    """Non-finite WR or non-positive n_games must be dropped silently —
    telemetry never aborts a roll."""
    pool = OpponentPool(
        pinned_snapshots=[_snap(0)], heuristic_spec=_heuristic(),
        p_heuristic=0.25)
    pool.update_heuristic_winrate(win_rate=float("nan"), n_games=100)
    pool.update_heuristic_winrate(win_rate=0.50, n_games=0)
    pool.update_heuristic_winrate(win_rate=0.50, n_games=-3)
    assert pool.heuristic_winrate() is None


def test_p_heuristic_skips_legacy_anneal_branch():
    """Regression test for Codex round-1 bug: when `p_heuristic > 0`, the
    legacy anneal-to-heuristic branch must NOT fire — otherwise non-trivial
    `anneal_start`/`p_max` (e.g., the seam-profile default
    anneal_start=10/p_max=0.7) would route every early-step draw to the
    heuristic, swamping the configured `p_heuristic` rate. Pre-fix: 100%
    heuristic at step ≤ anneal_start; post-fix: exactly `p_heuristic`.
    """
    anchor = _snap(0)
    # Repl_07-style anneal: heuristic-only for first 50 steps, then ramp.
    # If the legacy branch is incorrectly consulted in v3 mode, EVERY draw
    # at step=5 will return heuristic (p_anneal=0 ⇒ u >= 0 ⇒ always true).
    pool = OpponentPool(
        pinned_snapshots=[anchor], max_snapshots=5,
        anneal_start=50, anneal_end=400, p_max=0.7,
        heuristic_spec=_heuristic(), p_heuristic=0.30, rng_seed=0)
    # Step well within the anneal-zero region — legacy would fire 100%.
    kinds = [pool.draw(5, s).kind for s in range(9000, 9500)]
    n_heur = kinds.count("heuristic")
    # Expected ~150/500 (30%), NOT ~500/500 (100% legacy bug).
    assert 110 < n_heur < 200, (
        f"n_heur={n_heur}/500 at step=5 with p_heuristic=0.30 — should be "
        "~150, not 500 (legacy anneal bug)")
    # Step BEYOND anneal_end — same expected rate (decay path).
    kinds_late = [pool.draw(500, s).kind for s in range(9000, 9500)]
    n_heur_late = kinds_late.count("heuristic")
    assert 110 < n_heur_late < 200, (
        f"n_heur_late={n_heur_late}/500 at step=500 — should still be ~150")


def test_legacy_mode_anneal_still_works_when_p_heuristic_zero():
    """Back-compat: with `p_heuristic == 0`, the legacy anneal-to-heuristic
    branch must STILL fire (repl_07 behaviour). At step < anneal_start the
    rate is 100% heuristic; at step > anneal_end it's p_max."""
    anchor = _snap(0)
    pool = OpponentPool(
        pinned_snapshots=[anchor], max_snapshots=5,
        anneal_start=50, anneal_end=400, p_max=0.7,
        heuristic_spec=_heuristic(), p_heuristic=0.0, rng_seed=0)
    # Step in anneal-zero region: all draws should be heuristic.
    kinds_pre = [pool.draw(5, s).kind for s in range(9000, 9100)]
    assert all(k == "heuristic" for k in kinds_pre), (
        "legacy anneal-zero region should be 100% heuristic")
    # Step beyond anneal_end: 1 - p_max = 30% heuristic, 70% snapshot.
    kinds_post = [pool.draw(500, s).kind for s in range(9000, 9500)]
    n_heur_post = kinds_post.count("heuristic")
    # Expected ~150/500 ((1 - 0.7) * 500); loose ±50.
    assert 100 < n_heur_post < 200, (
        f"legacy anneal-end heuristic rate {n_heur_post}/500 not near ~150")


def test_p_api_is_absolute_share_in_v3_mode():
    """v3 mix-gate makes p_api an ABSOLUTE fraction of total draws (not
    `p_api · (1 - p_heuristic)` as the pre-fix two-tier code did). With
    p_heuristic=0.30 + p_api=0.20, exact expected rates:
      heuristic=30%, api=20%, snapshot=50%
    (pre-fix 2-tier would have given api=(1-0.30)·0.20=14%)."""
    anchor = _snap(0)
    api = [OpponentSpec("api", "google/gemini-3-flash-preview", "api_g")]
    pool = OpponentPool(
        pinned_snapshots=[anchor], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0,
        heuristic_spec=_heuristic(), p_heuristic=0.30,
        api_specs=api, p_api=0.20, rng_seed=0)
    kinds = [pool.draw(100, s).kind for s in range(9000, 10000)]
    n_heur = kinds.count("heuristic")
    n_api = kinds.count("api")
    n_snap = kinds.count("snapshot")
    # Expected: heur≈300, api≈200, snap≈500. Loose ±15% bounds.
    assert 240 < n_heur < 360, f"n_heur={n_heur} not near 300"
    assert 150 < n_api < 250, (
        f"n_api={n_api} — pre-fix 2-tier would have given ~140 (= (1-0.30)·"
        "0.20·1000); v3 must give ~200 (absolute share)")
    assert 400 < n_snap < 600, f"n_snap={n_snap} not near 500"


def test_p_heuristic_plus_p_api_validation():
    """p_heuristic + p_api > 1 is rejected at construction — the one-tier
    mix-gate has no defined behaviour otherwise."""
    api = [OpponentSpec("api", "google/gemini-3-flash-preview", "api_g")]
    with pytest.raises(ValueError, match="p_heuristic.*p_api"):
        OpponentPool(
            pinned_snapshots=[_snap(0)],
            heuristic_spec=_heuristic(), p_heuristic=0.70,
            api_specs=api, p_api=0.40)


def test_telemetry_realized_marginals_v3_mode():
    """`telemetry()` exposes realized marginal probabilities so consumers
    don't have to re-derive them. In v3 mode at cold start: heur=p_heur,
    api=p_api, snap=1-p_heur-p_api."""
    anchor = _snap(0)
    api = [OpponentSpec("api", "google/gemini-3-flash-preview", "api_g")]
    pool = OpponentPool(
        pinned_snapshots=[anchor], heuristic_spec=_heuristic(),
        p_heuristic=0.30, api_specs=api, p_api=0.20)
    t = pool.telemetry(step=100)
    assert t["draw_mode"] == "v3_heuristic_with_decay"
    m = t["realized_marginal_p"]
    assert m["heuristic"] == pytest.approx(0.30)
    assert m["api"] == pytest.approx(0.20)
    assert m["snapshot"] == pytest.approx(0.50)
    # After WR=1.0 observation with enough games, heur drops to floor.
    pool.update_heuristic_winrate(1.0, _PFSP_MIN_N_GAMES * 5)
    t2 = pool.telemetry(step=100)
    m2 = t2["realized_marginal_p"]
    assert m2["heuristic"] == pytest.approx(0.30 * _P_HEURISTIC_DECAY_FLOOR)
    assert m2["api"] == pytest.approx(0.20)
    assert m2["snapshot"] == pytest.approx(1.0 - m2["heuristic"] - 0.20)


def test_telemetry_realized_marginals_legacy_mode():
    """In legacy mode (p_heuristic=0 + nontrivial anneal) the realized
    marginals account for the anneal × p_api interaction."""
    api = [OpponentSpec("api", "google/gemini-3-flash-preview", "api_g")]
    pool = OpponentPool(
        pinned_snapshots=[_snap(0)],
        anneal_start=50, anneal_end=400, p_max=0.7,
        heuristic_spec=_heuristic(), p_heuristic=0.0,
        api_specs=api, p_api=0.10)
    # At step=500 (past anneal_end), p_anneal = p_max = 0.7 ⇒
    # legacy_heur = 1 - 0.7 = 0.30, api_marginal = 0.7 * 0.10 = 0.07,
    # snap = 0.63.
    t = pool.telemetry(step=500)
    assert t["draw_mode"] == "legacy_anneal"
    m = t["realized_marginal_p"]
    assert m["heuristic"] == pytest.approx(0.30, abs=1e-3)
    assert m["api"] == pytest.approx(0.07, abs=1e-3)
    assert m["snapshot"] == pytest.approx(0.63, abs=1e-3)


def test_telemetry_includes_heuristic_decay_when_active():
    """`telemetry()` emits `p_heuristic`, `p_heuristic_effective`, and (once
    observed) `heuristic_winrate` only when `p_heuristic > 0` — legacy runs
    keep clean output."""
    # Legacy / pure-self-play: p_heuristic=0 ⇒ NO heuristic keys in telemetry.
    pool_legacy = OpponentPool(
        pinned_snapshots=[_snap(0)], heuristic_spec=None, p_heuristic=0.0)
    t = pool_legacy.telemetry(step=10)
    assert "p_heuristic" not in t
    assert "p_heuristic_effective" not in t
    assert "heuristic_winrate" not in t
    # v3 mode: keys present.
    pool_v3 = OpponentPool(
        pinned_snapshots=[_snap(0)], heuristic_spec=_heuristic(),
        p_heuristic=0.30)
    t2 = pool_v3.telemetry(step=10)
    assert t2["p_heuristic"] == pytest.approx(0.30)
    assert t2["p_heuristic_effective"] == pytest.approx(0.30)  # cold start
    assert "heuristic_winrate" not in t2  # no observations yet
    pool_v3.update_heuristic_winrate(win_rate=0.75, n_games=100)
    t3 = pool_v3.telemetry(step=10)
    assert "heuristic_winrate" in t3
    assert t3["heuristic_winrate"]["n_games"] == 100


# --------------------------------------------------------------------------- #
# repl_08 v3.2 — heuristic step-anneal-to-zero (decoupled from the WR floor).  #
# --------------------------------------------------------------------------- #
def test_heuristic_anneal_disabled_is_backcompat():
    """`heuristic_anneal_end=0` (default) ⇒ the step has NO effect: effective P
    is identical at every step (floor-only behaviour, byte-identical to pre-v3.2
    and to the no-`step` direct call)."""
    pool = OpponentPool(
        pinned_snapshots=[_snap(0)], heuristic_spec=_heuristic(),
        p_heuristic=0.30, rng_seed=0)  # heuristic_anneal_end defaults to 0
    assert pool.heuristic_anneal_end == 0
    base = pool._p_heuristic_effective()           # no step ⇒ factor 1.0
    for step in (0, 10, 100, 10_000):
        assert pool._p_heuristic_effective(step) == pytest.approx(base)
        assert pool._heuristic_step_factor(step) == 1.0


def test_heuristic_step_factor_linear_to_zero():
    """The step factor ramps 1.0 → 0.0 linearly over [0, end) and stays 0 after."""
    pool = OpponentPool(
        pinned_snapshots=[_snap(0)], heuristic_spec=_heuristic(),
        p_heuristic=0.30, heuristic_anneal_end=40, rng_seed=0)
    assert pool._heuristic_step_factor(0) == pytest.approx(1.0)
    assert pool._heuristic_step_factor(10) == pytest.approx(0.75)
    assert pool._heuristic_step_factor(20) == pytest.approx(0.5)
    assert pool._heuristic_step_factor(39) == pytest.approx(1.0 - 39 / 40)
    assert pool._heuristic_step_factor(40) == 0.0   # fully retired at `end`
    assert pool._heuristic_step_factor(1000) == 0.0


def test_heuristic_anneal_drives_effective_p_to_zero_cold_start():
    """At cold start (WR factor 1.0), effective P tracks the linear anneal and
    hits EXACTLY 0 at `heuristic_anneal_end`."""
    pool = OpponentPool(
        pinned_snapshots=[_snap(0)], heuristic_spec=_heuristic(),
        p_heuristic=0.30, heuristic_anneal_end=40, rng_seed=0)
    assert pool._p_heuristic_effective(0) == pytest.approx(0.30)
    assert pool._p_heuristic_effective(20) == pytest.approx(0.15)
    assert pool._p_heuristic_effective(40) == 0.0
    assert pool._p_heuristic_effective(60) == 0.0


def test_heuristic_anneal_decoupled_from_wr_floor():
    """The step anneal is DECOUPLED from the `(1-WR)²` decay floor: even when
    the trainee crushes the heuristic (WR in the floor regime, where the WR
    factor pins at `_P_HEURISTIC_DECAY_FLOOR` forever), the step anneal still
    drives the share to EXACTLY 0 by `heuristic_anneal_end`. This is the whole
    point — the floor alone never reaches 0."""
    end = 40
    annealed = OpponentPool(
        pinned_snapshots=[_snap(0)], heuristic_spec=_heuristic(),
        p_heuristic=0.30, heuristic_anneal_end=end, rng_seed=0)
    floor_only = OpponentPool(
        pinned_snapshots=[_snap(0)], heuristic_spec=_heuristic(),
        p_heuristic=0.30, rng_seed=0)  # no step anneal
    for p in (annealed, floor_only):
        p.update_heuristic_winrate(0.95, _PFSP_MIN_N_GAMES * 5)  # floor regime
    # Floor-only: persists at p·floor forever (the problem we're fixing).
    assert floor_only._p_heuristic_effective(end) == pytest.approx(
        0.30 * _P_HEURISTIC_DECAY_FLOOR)
    assert floor_only._p_heuristic_effective(10_000) == pytest.approx(
        0.30 * _P_HEURISTIC_DECAY_FLOOR)
    # Annealed: WR-floor share early, but EXACTLY 0 at/after `end`.
    assert annealed._p_heuristic_effective(0) == pytest.approx(
        0.30 * _P_HEURISTIC_DECAY_FLOOR)
    assert annealed._p_heuristic_effective(end) == 0.0
    assert annealed._p_heuristic_effective(end + 100) == 0.0


def test_heuristic_anneal_no_draws_after_end():
    """End-to-end: after `heuristic_anneal_end`, draw() returns ZERO heuristic
    opponents — the share is exactly 0, not a small floor."""
    end = 20
    pool = OpponentPool(
        pinned_snapshots=[_snap(0)], max_snapshots=5,
        anneal_start=0, anneal_end=0, p_max=1.0,
        heuristic_spec=_heuristic(), p_heuristic=0.30,
        heuristic_anneal_end=end, rng_seed=0)
    # Before the anneal end, the heuristic still shows up.
    early = [pool.draw(0, s).kind for s in range(9000, 9300)]
    assert early.count("heuristic") > 0
    # At/after the anneal end, NOT a single heuristic draw.
    late = [pool.draw(end + 5, s).kind for s in range(9000, 9300)]
    assert late.count("heuristic") == 0, (
        f"heuristic still drawn after anneal end: {late.count('heuristic')}")


def test_heuristic_anneal_end_validation():
    """Negative `heuristic_anneal_end` is rejected at construction."""
    with pytest.raises(ValueError, match="heuristic_anneal_end"):
        OpponentPool(
            pinned_snapshots=[_snap(0)], heuristic_spec=_heuristic(),
            p_heuristic=0.30, heuristic_anneal_end=-5)


def test_heuristic_anneal_telemetry_exposed():
    """Telemetry surfaces the anneal knob + the current step factor so the
    operator can see the heuristic graduating out across rolls."""
    pool = OpponentPool(
        pinned_snapshots=[_snap(0)], heuristic_spec=_heuristic(),
        p_heuristic=0.30, heuristic_anneal_end=40, rng_seed=0)
    t = pool.telemetry(step=20)
    assert t["heuristic_anneal_end"] == 40
    assert t["heuristic_step_factor"] == pytest.approx(0.5)
    assert t["p_heuristic_effective"] == pytest.approx(0.15)  # 0.30·1.0·0.5
