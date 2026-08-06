"""The evidence profile trains against self-play, never a silent heuristic.

`scripts/training/run_phase3.sh` only starts the heuristic shim when the run
actually asks for it (legacy `--no-opponent-pool`, `P_HEURISTIC>0`, or a
heuristic *eval* opponent); the driver must agree via
`_needs_heuristic_endpoint`, so an evidence run cannot quietly fall back to
the scripted opponent.
"""

from __future__ import annotations

import sys

from _rl_fixtures import REPO_ROOT

P3 = REPO_ROOT / "scripts" / "training"


def test_current_self_spec_uses_live_adapter_name():
    import megagem.training.adapter_sync as ADP

    import phase3_grpo

    spec = phase3_grpo._current_self_spec()
    assert spec.kind == "current_self"
    assert spec.served_name == ADP.ADAPTER_NAME
    assert spec.actor_id == "current_self"


def test_heuristic_url_not_required_for_default_pool(monkeypatch):
    import phase3_grpo

    monkeypatch.setattr(sys, "argv", [
        "phase3_grpo.py",
        "--output", "/tmp/phase3.json",
        "--opponent-pool",
        "--p-heuristic", "0",
        "--p-current-self", "0.8",
    ])
    args = phase3_grpo.parse_args()
    assert args.heuristic_url is None
    assert phase3_grpo._needs_heuristic_endpoint(args) is False


def test_heuristic_url_required_only_for_legacy_or_explicit_heuristic(monkeypatch):
    import phase3_grpo

    monkeypatch.setattr(sys, "argv", [
        "phase3_grpo.py",
        "--output", "/tmp/phase3.json",
        "--no-opponent-pool",
    ])
    assert phase3_grpo._needs_heuristic_endpoint(phase3_grpo.parse_args()) is True

    monkeypatch.setattr(sys, "argv", [
        "phase3_grpo.py",
        "--output", "/tmp/phase3.json",
        "--opponent-pool",
        "--p-heuristic", "0.2",
    ])
    assert phase3_grpo._needs_heuristic_endpoint(phase3_grpo.parse_args()) is True


def test_run_script_defaults_no_heuristic_80_20_evidence():
    src = (P3 / "run_phase3.sh").read_text()
    assert "_DEF_NUM_SEEDS=32" in src
    assert "_DEF_ROWS_PER_GEN=4096" in src
    assert "_DEF_K=16" in src
    assert "_DEF_MAX_PARALLEL=64" in src
    assert "_DEF_SNAPSHOT_EVERY=10" in src
    assert "_DEF_MAX_SNAPSHOTS=8" in src
    assert ': "${P_CURRENT_SELF:=0.80}"' in src
    assert '--p-current-self "${P_CURRENT_SELF}"' in src
    assert 'heuristic shim not started' in src


def test_run_script_only_starts_the_heuristic_when_asked():
    """The shim is a package entrypoint (`-m megagem.training.heuristic_endpoint`,
    run with src/ on PYTHONPATH) and `start_heuristic` is reached only through
    `heuristic_needed` — training-side (legacy pool or P_HEURISTIC>0) or a
    heuristic eval opponent. P_HEURISTIC defaults to 0 ⇒ pure self-play."""
    src = (P3 / "run_phase3.sh").read_text()
    # the shim moved into the package; it is launched as a module, not a script
    assert "-m megagem.training.heuristic_endpoint" in src
    assert "scripts/phase3" not in src
    assert 'export PYTHONPATH="${REPO_ROOT}/src' in src
    # default: heuristic OFF in training
    assert ': "${P_HEURISTIC:=0.0}"' in src
    # the gate: start_heuristic is guarded, and the guard ORs the two needs
    assert "heuristic_needed() {" in src
    assert "heuristic_training_needed || heuristic_eval_needed" in src
    assert "if heuristic_needed; then\n        start_heuristic" in src
    # heuristic_training_needed is exactly "legacy pool OR explicit p_heuristic"
    assert '[[ "${OPPONENT_POOL}" != "1" ]] || heuristic_prob_enabled' in src
    # and the driver only receives --heuristic-url in that same case
    assert ("if heuristic_training_needed; then\n"
            '            cmd+=( --heuristic-url "${HEUR_URL}" )') in src
