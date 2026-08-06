"""Tests for Phase 0.3 + 0.4 smoke harnesses (data-prep paths only).

The actual H100-bound update step in update_cost_smoke.run_update_step is
covered by running it on a real H100; these tests pin the pure-Python
data-prep so a regression there can't silently break the smoke output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from update_cost_smoke import (
    TurnRecord,
    build_batch_records,
    iter_turn_records,
    load_trajectories,
    summarize_batch,
)


def _v3_game(num_rounds: int = 2, players_per_round: int = 3) -> dict:
    rounds = []
    for r in range(1, num_rounds + 1):
        players = []
        for p in range(players_per_round):
            players.append({
                "player_id": p,
                "model": "fake",
                "bid": p + r,
                "is_winner": p == 0,
                "actor_id": "trainable" if p == 0 else "snapshot_0",
                "prompt": f"user prompt round={r} player={p}",
                "raw_response": f"<think>r{r}p{p}</think>{{\"bid\": {p + r}}}",
                "parsed_action": p + r,
                "parse_method": "strict_json",
                "parse_valid": True,
                "legal_valid": p + r < 10,
                "default_used": False,
                "length_split": {"total_chars": 30, "think_block_chars": 5, "json_chars": 9},
                "parse_error": "",
                "legal_error": "",
            })
        rounds.append({
            "round_number": r,
            "auction": {"type": "treasure", "description": "t"},
            "players": players,
        })
    return {
        "metadata": {
            "schema_version": 3,
            "models": ["fake"] * players_per_round,
            "seed": 42,
            "system_prompt": "You are a MegaGem player. Bid wisely.",
        },
        "rounds": rounds,
        "final_results": {"winner_id": 0},
        "statistics": [],
        "telemetry": {"schema_version": 2},
    }


# ---------- load_trajectories ----------

def test_load_trajectories_accepts_v3(tmp_path: Path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps(_v3_game()))
    games = load_trajectories([p])
    assert len(games) == 1
    assert games[0]["_path"] == str(p)


def test_load_trajectories_rejects_v1(tmp_path: Path):
    g = _v3_game()
    g["metadata"]["schema_version"] = 1
    p = tmp_path / "v1.json"
    p.write_text(json.dumps(g))
    with pytest.raises(ValueError, match="schema_version=1"):
        load_trajectories([p])


def test_load_trajectories_rejects_missing_version(tmp_path: Path):
    g = _v3_game()
    del g["metadata"]["schema_version"]
    p = tmp_path / "no_version.json"
    p.write_text(json.dumps(g))
    with pytest.raises(ValueError):
        load_trajectories([p])


def test_load_trajectories_rejects_v2_without_system_prompt(tmp_path: Path):
    g = _v3_game()
    g["metadata"]["schema_version"] = 2
    del g["metadata"]["system_prompt"]
    p = tmp_path / "v2.json"
    p.write_text(json.dumps(g))
    with pytest.raises(ValueError, match="schema_version=2"):
        load_trajectories([p])


def test_load_trajectories_rejects_v3_missing_system_prompt(tmp_path: Path):
    g = _v3_game()
    del g["metadata"]["system_prompt"]
    p = tmp_path / "malformed.json"
    p.write_text(json.dumps(g))
    with pytest.raises(ValueError, match="system_prompt"):
        load_trajectories([p])


# ---------- iter_turn_records ----------

def test_iter_turn_records_flattens_all_players(tmp_path: Path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps(_v3_game(num_rounds=3, players_per_round=3)))
    games = load_trajectories([p])
    records = iter_turn_records(games)
    assert len(records) == 9
    assert all(isinstance(r, TurnRecord) for r in records)
    assert {r.player_id for r in records} == {0, 1, 2}
    assert {r.actor_id for r in records} == {"trainable", "snapshot_0"}
    assert all(r.system_prompt == "You are a MegaGem player. Bid wisely." for r in records)
    assert records[0].user_prompt.startswith("user prompt round=")


def test_iter_turn_records_skips_records_without_raw_response(tmp_path: Path):
    g = _v3_game(num_rounds=1, players_per_round=2)
    del g["rounds"][0]["players"][1]["raw_response"]
    p = tmp_path / "g.json"
    p.write_text(json.dumps(g))
    records = iter_turn_records(load_trajectories([p]))
    assert len(records) == 1
    assert records[0].player_id == 0


def test_iter_turn_records_skips_records_without_prompt(tmp_path: Path):
    g = _v3_game(num_rounds=1, players_per_round=2)
    del g["rounds"][0]["players"][1]["prompt"]
    p = tmp_path / "g.json"
    p.write_text(json.dumps(g))
    records = iter_turn_records(load_trajectories([p]))
    assert len(records) == 1
    assert records[0].player_id == 0


# ---------- build_batch_records ----------

def test_build_batch_records_returns_first_n(tmp_path: Path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps(_v3_game(num_rounds=4, players_per_round=3)))
    records = iter_turn_records(load_trajectories([p]))
    batch = build_batch_records(records, num_turns=5)
    assert len(batch) == 5
    assert batch == records[:5]


def test_build_batch_records_raises_when_undersized(tmp_path: Path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps(_v3_game(num_rounds=1, players_per_round=2)))
    records = iter_turn_records(load_trajectories([p]))
    with pytest.raises(ValueError, match="Need at least"):
        build_batch_records(records, num_turns=10)


# ---------- summarize_batch ----------

def test_summarize_batch_counts_actor_split(tmp_path: Path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps(_v3_game(num_rounds=2, players_per_round=3)))
    records = iter_turn_records(load_trajectories([p]))
    summary = summarize_batch(records)
    assert summary["num_turns"] == 6
    assert summary["num_unique_games"] == 1
    assert summary["actor_counts"]["trainable"] == 2
    assert summary["actor_counts"]["snapshot_0"] == 4
    assert summary["parseable_count"] == 6
    # legal_valid is False when bid+round >= 10 in the fixture; with rounds 1-2 and players 0-2 that never happens
    assert summary["legal_count"] == 6
    assert summary["mean_prompt_chars"] > 0
    assert summary["mean_response_chars"] > 0
