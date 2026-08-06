"""Phase 1 scope add-on — per-row training-row contract vs the TRL seam."""

from __future__ import annotations

import json

import pytest

from megagem.rl.export import build_training_rows, contract_check, load_corpus, main
from megagem.training.megagem_grpo import (
    PRECOMPUTED_ADVANTAGE_KEY,
    PRECOMPUTED_REWARD_KEY,
)
from test_rl_advantage import _make_game

from _rl_fixtures import TRACKED_DIR, fixture_paths

FIXTURES = fixture_paths()


def test_row_keys_and_scalar_contract() -> None:
    corpus = [_make_game(seed=1, finals=(80, 20, 20)), _make_game(seed=1, finals=(20, 80, 20))]
    rows = build_training_rows(corpus)
    assert rows, "expected trainable rows"
    b = contract_check(rows)
    assert b == len(rows)  # _replacement_advantages builds a (B,) tensor
    for row in rows:
        assert isinstance(row[PRECOMPUTED_ADVANTAGE_KEY], float)
        assert isinstance(row[PRECOMPUTED_REWARD_KEY], float)
        assert "prompt" in row and "completion" in row
        assert {"actor_id", "bucket", "group_key", "game_id"} <= row.keys()
        comps = row["reward_components"]
        assert set(comps) == {"legal", "shaping", "terminal_correction", "terminal"}
        assert row[PRECOMPUTED_REWARD_KEY] == pytest.approx(sum(comps.values()))


def test_only_trainable_rows_exported() -> None:
    g = _make_game(
        seed=2, finals=(90, 10, 10), actor_ids=("trainable", "snapshot_1", "heuristic")
    )
    rows = build_training_rows([g])
    assert rows and all(r["player_id"] == 0 for r in rows)
    assert all(r["actor_id"] == "trainable" for r in rows)


def test_contract_check_rejects_bad_rows() -> None:
    with pytest.raises(KeyError):
        contract_check([{"prompt": "x"}])
    with pytest.raises(ValueError):
        contract_check([{PRECOMPUTED_ADVANTAGE_KEY: float("nan"), PRECOMPUTED_REWARD_KEY: 0.0}])
    with pytest.raises(ValueError):
        contract_check([{PRECOMPUTED_ADVANTAGE_KEY: 1, PRECOMPUTED_REWARD_KEY: 0.0}])


def test_constants_are_the_trainer_seam_constants() -> None:
    # Reused, not re-declared — this binding is the contract bond.
    assert PRECOMPUTED_ADVANTAGE_KEY == "precomputed_advantage"
    assert PRECOMPUTED_REWARD_KEY == "precomputed_reward"


def test_cli_report_runs_on_tracked_corpus(capsys) -> None:
    rc = main(["--corpus", str(TRACKED_DIR), "--report"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{") :])
    assert payload["training_rows_B"] == payload["advantage_summary"]["turns_trainable"]
    assert payload["scale_calibration"]["recommended_scale"] > 0


def test_load_corpus_deterministic_order() -> None:
    a = load_corpus(str(TRACKED_DIR))
    b = load_corpus(str(TRACKED_DIR))
    assert a, "tracked fixture corpus must be non-empty on a clean checkout"
    assert [g["metadata"]["seed"] for g in a] == [g["metadata"]["seed"] for g in b]


def test_load_corpus_rejects_missing_actor_id(tmp_path) -> None:
    """Caveat 4 — the Phase-3 entry point itself is fail-closed."""
    good = _make_game(seed=1, finals=(80, 10, 10))
    bad = _make_game(seed=2, finals=(10, 80, 10))
    del bad["rounds"][0]["players"][0]["actor_id"]
    (tmp_path / "good.json").write_text(json.dumps(good))
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="missing/empty actor_id"):
        load_corpus(str(tmp_path))
    # The escape hatch still loads it raw (e.g. for forensic inspection).
    assert len(load_corpus(str(tmp_path), assert_schema=False)) == 2
