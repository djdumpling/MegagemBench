"""Caveat 1 tooling — shaping/terminal mass panel + λ re-derivation sweep.

CPU-only, runs on a clean checkout: the telescoping/mass identities use the
tracked sanitized schema-v3 fixture; the rank-preservation metric is exercised
on synthetic zero-shaping K-groups (true outcome controlled).
"""

from __future__ import annotations

import json
import statistics

import pytest

from megagem.rl.diagnostics import (
    lambda_sweep,
    reveal_policy_diagnostic,
    shaping_terminal_mass,
    trajectory_breakdowns,
)
from megagem.rl.export import main as export_main
from megagem.rl.reward import RewardConfig
from megagem.rl.scorer import parse_transcript, score_at

from _rl_fixtures import TRACKED_DIR, fixture_paths, load as _load
from test_rl_advantage import _make_game

FIXTURES = fixture_paths()


# ---- S_raw is exactly the telescoped mid-game margin (mirrors reward path) ---


@pytest.mark.skipif(not FIXTURES, reason="no schema-v3 fixtures")
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_s_raw_equals_telescoped_midgame_margin(path) -> None:
    data = _load(path)
    parsed = parse_transcript(data)
    last = len(parsed.rounds) - 1
    by_pid = {b.player_id: b for b in trajectory_breakdowns([data])}
    for pid in range(parsed.num_players):
        others = [
            score_at(parsed, last, q)
            for q in range(parsed.num_players)
            if q != pid
        ]
        expected = score_at(parsed, last, pid) - statistics.fmean(others)
        assert by_pid[pid].s_raw == pytest.approx(expected, abs=1e-6)


# ---- mass scales linearly with λ; lower λ reduces shaping dominance ---------


@pytest.mark.skipif(not FIXTURES, reason="no schema-v3 fixtures")
def test_shaping_mass_scales_linearly_with_lambda() -> None:
    corpus = [_load(p) for p in FIXTURES]
    m2 = shaping_terminal_mass(corpus, RewardConfig(shaping_lambda=0.02))
    m1 = shaping_terminal_mass(corpus, RewardConfig(shaping_lambda=0.01))
    if not m1["shaping_mass_median"]:
        pytest.skip("tracked fixture has no shaping (empty value display)")
    assert m2["shaping_mass_median"] == pytest.approx(
        2.0 * m1["shaping_mass_median"], abs=1e-3
    )
    assert m2["shaping_over_terminal_median"] > m1["shaping_over_terminal_median"]


@pytest.mark.skipif(not FIXTURES, reason="no schema-v3 fixtures")
def test_lower_lambda_monotonically_reduces_dominance() -> None:
    corpus = [_load(p) for p in FIXTURES]
    sweep = lambda_sweep(corpus, [0.005, 0.01, 0.05, 0.2])["sweep"]
    ratios = [s["shaping_over_terminal_median"] for s in sweep]
    if any(r is None for r in ratios) or ratios[-1] == 0:
        pytest.skip("tracked fixture has no shaping")
    assert ratios == sorted(ratios), ratios  # non-decreasing in λ
    assert ratios[-1] > ratios[0]  # 0.2 strictly dominates 0.005


# ---- rank-preservation metric: terminal-only ⇒ ρ = 1 at any λ --------------


def test_rank_corr_is_one_when_terminal_only() -> None:
    """Zero shaping (empty display) ⇒ advantage is pure terminal, so the
    standardized advantage must rank rollouts exactly by true margin — at
    every λ. Validates the rank-preservation gate machinery itself."""
    seed = 77
    # P0 never wins (no +0.1), margins strictly increasing & unclipped at 21.5.
    finals = [(30, 40, 50), (30, 38, 46), (30, 36, 42), (30, 34, 38)]
    corpus = [_make_game(seed=seed, finals=f) for f in finals]
    out = lambda_sweep(corpus, [0.01, 0.2])
    for entry in out["sweep"]:
        assert entry["groups_usable"] == 2  # P0 and P2 vary; P1 is constant
        assert entry["rank_corr_mean"] == pytest.approx(1.0)
        assert entry["rank_corr_min"] == pytest.approx(1.0)
    assert out["recommendation"]["rank_corr_available"] is True


# ---- honest degradation when the corpus has no K≥2 groups ------------------


@pytest.mark.skipif(not FIXTURES, reason="no schema-v3 fixtures")
def test_sweep_degrades_honestly_without_k_groups() -> None:
    single = [_load(FIXTURES[0])]
    out = lambda_sweep(single, [0.01, 0.2])
    rec = out["recommendation"]
    assert rec["rank_corr_available"] is False
    assert "rank-corr UNAVAILABLE" in rec["basis"]
    assert all(s["groups_usable"] == 0 for s in out["sweep"])  # no crash


# ---- the Phase 1 report now carries the panel it was blind to --------------


def test_export_report_has_shaping_terminal_block(capsys) -> None:
    rc = export_main(["--corpus", str(TRACKED_DIR), "--report"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{") :])
    # additive: the pre-existing report shape is intact …
    assert {"scale_calibration", "advantage_summary", "training_rows_B"} <= payload.keys()
    # … plus the Caveat-1 panel.
    diag = payload["shaping_terminal_diagnostic"]
    assert diag["trajectories"] >= 1
    assert "shaping_over_terminal_median" in diag
    assert "corr_midgame_vs_true_margin" in diag
    assert diag["lambda"] == RewardConfig().shaping_lambda  # the locked default
    # … plus the Caveat-2-addendum reveal-shaping telemetry block.
    rp = payload["reveal_policy_diagnostic"]
    assert {"observation", "method", "mean_signed_reveal_shaping",
            "corr_reveal_shaping_vs_true_margin"} <= rp.keys()
    assert isinstance(rp["observation"], str)
    assert "CONFOUNDED" in rp["method"]  # the overclaim guard


# ---- Caveat 2 addendum: reveal-shaping telemetry --------------------------
#
# SCOPE OF THESE TESTS: they validate the *descriptive binning + arithmetic*
# (counts, exclusions, mean/corr, which observation string fires for a given
# numeric input) and that the output never claims causality. They do NOT —
# and cannot, from observational data — validate a causal "the policy is
# chasing the reveal proxy" reading: corr(reveal_shaping, margin) is
# confounded by player strength (a common cause of both). The causal test is
# the within-state counterfactual probe (reads the hand), out of scope here.


def _reveal_game(seed: int, p0_red: int, finals: tuple[int, int, int]) -> dict:
    """1-round treasure game: P0 wins and reveals Red; P0 holds ``p0_red``
    Red in collection, P1/P2 hold none, equal coins.

    Pre-round-1 display is empty (Red value 0); post-reveal Red count 1
    (chart A ⇒ value 4). So P0's reveal-turn shaping is exactly
    ``λ·(4·p0_red − 0) = 0.04·p0_red`` — strictly positive, scaled by
    ``p0_red``; P0's terminal margin is set independently via ``finals``.
    One policy reveal turn per game.
    """
    winner = max(range(3), key=lambda p: finals[p])
    players = []
    for pid, red in ((0, p0_red), (1, 0), (2, 0)):
        players.append({
            "player_id": pid, "actor_id": "trainable", "bid": 1,
            "coins_before": 35, "coins_after": 35,
            "collection": ["Red"] * red,
            "collection_counts": {"Red": red} if red else {},
            "is_winner": pid == 0, "parse_valid": True, "legal_valid": True,
            "reasoning": "", "prompt": f"P{pid}", "raw_response": "r",
        })
    players[0]["winning_bid"] = 1
    players[0]["gem_revealed"] = "Red"
    players[0]["reveal"] = {
        "actor_id": "trainable", "parse_valid": True, "legal_valid": True,
        "prompt": "P0 reveal", "raw_response": "Red",
    }
    return {
        "metadata": {"schema_version": 3, "seed": seed, "num_players": 3,
                     "value_chart": "A"},
        "rounds": [{
            "round_number": 1,
            "auction": {"type": "treasure", "description": "t"},
            "players": players,
            "value_display": {"Red": {"count": 1, "value_per_gem": 4}},
            "tiebreak_order": [0, 1, 2],
            "available_missions": [], "missions_completed": {},
        }],
        "final_results": {
            "winner_id": winner, "num_rounds": 1, "available_missions": [],
            "final_scores": [
                {"player_id": p, "final_score": finals[p],
                 "loan_payments": 0, "investment_returns": 0}
                for p in range(3)
            ],
        },
    }


def test_reveal_binning_self_favoring_and_decoupled_string() -> None:
    """Binning logic only: positive mean + low |corr| ⇒ the descriptive
    self-favoring/decoupled observation fires AND it explicitly disclaims
    causality (no "SUSPECTED"/"escalate"; says confounded + NOT proof)."""
    # p0_red cycles 1..5; P0 margin varies on a coprime period (3) ⇒ the
    # (reveal_shaping, margin) pairs are rank-decorrelated by construction.
    corpus = []
    for i in range(15):
        red = (i % 5) + 1
        p0_final = 60 + (i % 3) * 12  # independent of red
        corpus.append(_reveal_game(seed=i, p0_red=red,
                                    finals=(p0_final, 20, 20)))
    rep = reveal_policy_diagnostic(corpus, min_reveal_turns=10)
    assert rep["policy_reveal_turns"] == 15
    assert rep["mean_signed_reveal_shaping"] > 0  # all reveals self-favoring
    assert rep["frac_self_favoring_of_nonzero"] == 1.0
    assert abs(rep["corr_reveal_shaping_vs_true_margin"]) < 0.15
    obs = rep["observation"]
    assert "self-favoring" in obs
    # the overclaim guard: no causal verdict, confound stated explicitly
    assert "SUSPECTED" not in obs and "escalate" not in obs
    assert "NOT proof" in obs and "confounded" in obs.lower()
    assert "CONFOUNDED" in rep["method"]


def test_reveal_binning_no_pattern_string() -> None:
    """Positive corr ⇒ the 'no decoupled pattern' string, still labelled
    observational and explicitly NOT exculpatory (the confound cuts both
    ways — this is not an 'ok' clearance)."""
    corpus = []
    for i in range(15):
        red = (i % 5) + 1
        corpus.append(_reveal_game(seed=i, p0_red=red,
                                   finals=(40 + red * 12, 20, 20)))
    rep = reveal_policy_diagnostic(corpus, min_reveal_turns=10)
    assert rep["corr_reveal_shaping_vs_true_margin"] > 0.15
    assert "no self-favoring-and-decoupled pattern" in rep["observation"]
    assert "not exculpatory" in rep["observation"]


def test_reveal_telemetry_insufficient_data_is_honest() -> None:
    corpus = [_reveal_game(seed=i, p0_red=2, finals=(80, 10, 10))
              for i in range(4)]
    rep = reveal_policy_diagnostic(corpus)  # default min_reveal_turns=30
    assert rep["policy_reveal_turns"] == 4
    assert "insufficient data" in rep["observation"]


def test_reveal_telemetry_excludes_defaulted_reveals() -> None:
    """A defaulted/illegal reveal is not a policy choice — counted but not
    scored in the signal."""
    g = _reveal_game(seed=1, p0_red=3, finals=(80, 10, 10))
    g["rounds"][0]["players"][0]["reveal"]["parse_valid"] = False
    g["rounds"][0]["players"][0]["reveal"]["legal_valid"] = False
    rep = reveal_policy_diagnostic([g], min_reveal_turns=1)
    assert rep["reveal_turns_total"] == 1
    assert rep["reveal_turns_defaulted_excluded"] == 1
    assert rep["policy_reveal_turns"] == 0


@pytest.mark.skipif(not FIXTURES, reason="no schema-v3 fixtures")
def test_reveal_telemetry_structural_invariants_on_fixtures() -> None:
    rep = reveal_policy_diagnostic([_load(p) for p in FIXTURES])
    assert (rep["policy_reveal_turns"]
            + rep["reveal_turns_defaulted_excluded"]
            == rep["reveal_turns_total"])
    assert isinstance(rep["observation"], str) and rep["observation"]
    # output must never present itself as a causal gate
    assert "CONFOUNDED" in rep["method"] and "NOT a causal gate" in rep["method"]


# ===========================================================================
# Phase-3 reward-design pre-spend gate panels.
# ===========================================================================

from dataclasses import replace  # noqa: E402

from megagem.rl.diagnostics import (  # noqa: E402
    _ablation_cfg,
    _shaping_channel_sums,
    ablation_lambda_relock,
    clip_vs_tanh_panel,
    degenerate_group_sigma_panel,
    illegal_leverage_panel,
    telescope_identity,
)


def _p3_kgroups() -> list[dict]:
    # duplicate seeds ⇒ true K-groups; mixed reveal/no-reveal, big + small
    return [
        _reveal_game(seed=1, p0_red=3, finals=(200, 10, 5)),
        _reveal_game(seed=1, p0_red=4, finals=(40, 60, 30)),
        _make_game(seed=2, finals=(120, 20, 10)),
        _make_game(seed=2, finals=(8, 70, 70)),
    ]


def test_telescope_identity_passes_and_flags_honestly_at_tol_zero() -> None:
    """rec 1a (a): the identity holds at a sane tol under BOTH reveal
    settings; at tol=0 the residual float dev makes it FAIL (the gate can
    fail — it does not silently pass)."""
    corpus = _p3_kgroups()
    ok = telescope_identity(corpus, tol=1e-6)
    assert ok["pass"] is True
    assert ok["reveal_on"]["pass"] and ok["reveal_off"]["pass"]
    assert ok["reveal_on"]["n_fail"] == 0
    strict = telescope_identity(corpus, tol=0.0)
    assert strict["pass"] is False  # does NOT silently pass at tol=0
    # the residual float dev surfaces in at least one setting (here reveal_on;
    # reveal_off telescopes exactly to 0.0 on this integer-margin corpus)
    assert (strict["reveal_on"]["n_fail"] > 0
            or strict["reveal_off"]["n_fail"] > 0)


def test_mass_gate_weighs_corrected_channel_not_uncorrected_proxy() -> None:
    """Codex 2026-05-19 material fix. The λ mass-gate must weigh the ACTUAL
    shaping channel the policy sees — ``Σ(shaping+terminal_correction)`` — not
    the uncorrected ``λ·|S_raw|`` proxy that rec 1a replaces.

    * default (1a off): channel sum == legacy ``λ·S_raw`` exactly (the locked
      panel is byte-unchanged).
    * 1a on: channel sum == the telescoped ``(λ/Sc)·true_margin`` and is
      materially different from ``λ·S_raw`` — so the +1a/+all mass ratios
      reflect the corrected quantity, not the biased mid-game proxy.
    """
    corpus = _p3_kgroups()
    lam, sc = 0.01, RewardConfig().scale

    off = RewardConfig(shaping_lambda=lam)
    chan_off = _shaping_channel_sums(corpus, off)
    bks = trajectory_breakdowns(corpus, off)
    for b in bks:  # 1a off ⇒ channel == λ·S_raw (legacy proxy), exactly
        assert chan_off[(b.game_id, b.player_id)] == pytest.approx(
            lam * b.s_raw, abs=1e-9
        )

    on = replace(off, terminal_correction=True)
    chan_on = _shaping_channel_sums(corpus, on)
    for b in bks:  # 1a on ⇒ channel telescopes to (λ/Sc)·true_margin
        assert chan_on[(b.game_id, b.player_id)] == pytest.approx(
            lam * b.true_margin / sc, abs=1e-9
        )
    # ...and that is genuinely a different number from the uncorrected proxy
    # (else the bug would be invisible) for at least one trajectory.
    assert any(
        abs(chan_on[(b.game_id, b.player_id)] - lam * b.s_raw) > 1e-6
        for b in bks
    )
    # the panel surfaces the corrected number, so the +1a ratio != baseline.
    base_ratio = shaping_terminal_mass(corpus, off)[
        "shaping_over_terminal_median"
    ]
    on_ratio = shaping_terminal_mass(corpus, on)[
        "shaping_over_terminal_median"
    ]
    assert base_ratio != on_ratio


def test_ablation_cfg_threads_every_phase3_field() -> None:
    """Guards the dataclasses.replace fix: each ablation cfg carries exactly
    the intended Phase-3 knobs (a 4-field RewardConfig rebuild would drop
    them silently)."""
    base = RewardConfig()
    assert _ablation_cfg("baseline", base, 19.6) == base
    assert _ablation_cfg("+1a", base, 19.6).terminal_correction is True
    t = _ablation_cfg("+tanh", base, 19.6)
    assert t.terminal_shape == "tanh" and t.scale == 19.6
    assert _ablation_cfg("+reveal0", base, 19.6).reveal_shaping_weight == 0.0
    a = _ablation_cfg("+all", base, 19.6)
    assert (a.terminal_correction is True and a.terminal_shape == "tanh"
            and a.scale == 19.6 and a.reveal_shaping_weight == 0.0)


def test_ablation_lambda_relock_structure_and_threading() -> None:
    rep = ablation_lambda_relock(
        _p3_kgroups(), lambdas=(0.005, 0.01, 0.02), scale_tanh=19.6
    )
    assert set(rep["by_ablation"]) == {
        "baseline", "+1a", "+tanh", "+reveal0", "+all"
    }
    assert len(rep["table"]) == 5 and rep["final_cut"] == "+all"
    for name, sweep in rep["by_ablation"].items():
        assert "sweep" in sweep and "recommendation" in sweep
        assert len(sweep["sweep"]) == 3


def test_clip_vs_tanh_panel_returns_verdict_and_raw_numbers() -> None:
    rep = clip_vs_tanh_panel(_p3_kgroups(), scale_tanh=19.6)
    assert rep["verdict"] in ("LOAD-BEARING", "COSMETIC")
    for k in ("n_groups", "groups_with_clip_tie", "scale_clip", "scale_tanh"):
        assert k in rep
    assert rep["scale_tanh"] == 19.6 and rep["scale_clip"] == 21.5
    assert isinstance(rep["reason"], str) and rep["reason"]


def test_degenerate_sigma_panel_honest_both_ways() -> None:
    # ≥2 distinct rollouts per group ⇒ no degenerate group + honest note
    none = degenerate_group_sigma_panel(_p3_kgroups())
    assert none["n_degenerate"] == 0
    assert "NOT exercised" in none["note"]
    # a 1-round reveal game ⇒ the two non-winners have a single trainable bid
    # turn each ⇒ that group's pooled σ is 0 ⇒ the fallback IS exercised
    deg = degenerate_group_sigma_panel(
        [_reveal_game(seed=99, p0_red=2, finals=(80, 10, 10))]
    )
    assert deg["n_degenerate"] >= 2
    assert deg["degenerate_group_keys"]
    assert "down-weighted" in deg["note"]


def test_illegal_leverage_panel_honest_zero_and_measured() -> None:
    clean = illegal_leverage_panel([_make_game(seed=1, finals=(80, 10, 10))])
    assert clean["n_illegal"] == 0 and clean["measurable"] is False
    assert "not a clearance" in clean["note"]
    g = _make_game(seed=2, finals=(80, 10, 10), p0_illegal=True)
    bad = illegal_leverage_panel([g])
    assert bad["n_illegal"] >= 1 and bad["measurable"] is True
    assert bad["max_telescoped_turns"] >= 1
