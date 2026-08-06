"""Tests for the λ=0 metric clarification: passed/failed gate-lift split and the
tie-adjusted chosen-is-best rate in the eval summary, plus the paired-outcomes
analysis (paired Δ gain + bootstrap CI, win-flip McNemar exact)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load(mod_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, REPO / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


paired = _load("pikl_paired_outcomes", "scripts/analysis/pikl_paired_outcomes.py")


# --- summary: passed/failed gate-lift split + tie-adjusted rate --------------

def _decision(*, passed, lift, chosen_is_q_best, tieadj):
    return {
        "phase": "bid",
        "gate": {"passed": passed, "lift": lift},
        "metrics": {
            "chosen_is_q_best": chosen_is_q_best,
            "chosen_is_q_best_tieadj": tieadj,
        },
    }


def test_summary_splits_gate_lift_and_reports_tieadj():
    from eval_vs_gemini import _summarize_pikl_decisions
    decisions = [
        _decision(passed=True, lift=0.20, chosen_is_q_best=True, tieadj=True),
        _decision(passed=True, lift=0.10, chosen_is_q_best=False, tieadj=True),
        _decision(passed=False, lift=0.001, chosen_is_q_best=False, tieadj=False),
        _decision(passed=False, lift=0.005, chosen_is_q_best=False, tieadj=False),
    ]
    out = _summarize_pikl_decisions(decisions)
    assert out["gate_passed_n"] == 2 and out["gate_failed_n"] == 2
    assert math.isclose(out["gate_pass_lift_mean"], 0.15, abs_tol=1e-9)
    assert math.isclose(out["gate_pass_lift_median"], 0.15, abs_tol=1e-9)
    assert math.isclose(out["gate_fail_lift_mean"], 0.003, abs_tol=1e-9)
    # exact chosen-is-best = 1/4; tie-adjusted credits both passed nodes = 2/4
    assert math.isclose(out["chosen_is_q_best_rate"], 0.25, abs_tol=1e-9)
    assert math.isclose(out["chosen_is_q_best_tieadj_rate"], 0.5, abs_tol=1e-9)


# --- paired analysis primitives ----------------------------------------------

def test_mcnemar_exact_symmetric_and_bounds():
    assert paired.mcnemar_exact(0, 0) == 1.0
    assert math.isclose(paired.mcnemar_exact(5, 5), 1.0, abs_tol=1e-9)
    # 10 vs 0 flips is decisive
    assert paired.mcnemar_exact(10, 0) < 0.01
    # symmetry in the two off-diagonal counts
    assert math.isclose(paired.mcnemar_exact(3, 9), paired.mcnemar_exact(9, 3), abs_tol=1e-12)
    # reproduce the prior 10-vs-5 readout (≈0.30)
    assert 0.2 < paired.mcnemar_exact(10, 5) < 0.4


def test_bootstrap_ci_contains_mean_and_is_seeded():
    diffs = [4.0, -2.0, 6.0, 1.0, -3.0, 5.0, 2.0, 0.0]
    lo, hi = paired.bootstrap_ci(diffs, n_boot=2000, seed=0)
    lo2, hi2 = paired.bootstrap_ci(diffs, n_boot=2000, seed=0)
    assert (lo, hi) == (lo2, hi2)  # seeded ⇒ reproducible
    mean = sum(diffs) / len(diffs)
    assert lo <= mean <= hi


def test_paired_report_pairs_by_seed_and_counts_flips():
    # baseline: seeds 1,2,3 ; treatment improves seed 1 (loss→win) and seed 3
    # delta, regresses nothing.
    base = {
        "config": {"pikl": None},
        "per_game": [
            {"seed": 1, "win": False, "policy_delta": -5.0, "policy_score": 10},
            {"seed": 2, "win": True, "policy_delta": 3.0, "policy_score": 20},
            {"seed": 3, "win": False, "policy_delta": -2.0, "policy_score": 12},
        ],
    }
    treat = {
        "config": {"pikl": {"lambda": 0.0}},
        "per_game": [
            {"seed": 1, "win": True, "policy_delta": 4.0, "policy_score": 18,
             "pikl_decisions": [_decision(passed=True, lift=0.2, chosen_is_q_best=True, tieadj=True)]},
            {"seed": 2, "win": True, "policy_delta": 3.0, "policy_score": 20, "pikl_decisions": []},
            {"seed": 3, "win": False, "policy_delta": 1.0, "policy_score": 15, "pikl_decisions": []},
        ],
    }
    rep = paired.paired_report(base, treat, n_boot=1000, seed=0)
    assert rep["n_paired"] == 3
    assert rep["flips"]["base_loss_treat_win"] == 1  # seed 1
    assert rep["flips"]["base_win_treat_loss"] == 0
    assert math.isclose(rep["paired_delta_gain_mean"], (9.0 + 0.0 + 3.0) / 3, abs_tol=1e-9)
    assert rep["delta_sign"]["treat_better"] == 2 and rep["delta_sign"]["treat_worse"] == 0
    assert "mcnemar_exact_p" in rep
    # treatment-arm gate telemetry surfaces passed-gate lift
    assert math.isclose(rep["treatment_gate"]["gate_pass_lift_mean"], 0.2, abs_tol=1e-9)


# --- reproduction on the real downloaded artifacts (skipped if absent) -------

def test_reproduces_prior_numbers_on_downloaded_files():
    off = REPO / "gate_b_gemini-3-flash_market_off.json"
    l0 = REPO / "gate_b_gemini-3-flash_market_0.json"
    if not (off.exists() and l0.exists()):
        pytest.skip("downloaded Gate-B artifacts not present")
    rep = paired.paired_report(paired.load_arm(off), paired.load_arm(l0), n_boot=2000, seed=0)
    assert rep["n_paired"] == 40
    # prior manual analysis: paired Δ gain ≈ +4.0, flips 10 (loss→win) vs 5 (win→loss)
    assert 2.0 < rep["paired_delta_gain_mean"] < 6.0
    assert rep["flips"]["base_loss_treat_win"] == 10
    assert rep["flips"]["base_win_treat_loss"] == 5
    assert 0.2 < rep["mcnemar_exact_p"] < 0.4
