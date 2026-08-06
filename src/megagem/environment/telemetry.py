"""Per-game telemetry aggregator over schema-v2 per-turn records."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..game.actions import PARSE_METHOD_STRICT_JSON


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, ``p`` in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    if p <= 0:
        return float(s[0])
    if p >= 100:
        return float(s[-1])
    rank = max(1, int(round(p / 100.0 * len(s))))
    return float(s[min(rank - 1, len(s) - 1)])


def _empty_actor_bucket() -> dict[str, int]:
    return {"turns": 0, "strict_json": 0, "parseable": 0, "legal": 0, "default_used": 0}


def _empty_phase_summary() -> dict[str, Any]:
    return {
        "total_turns": 0,
        "by_actor": defaultdict(_empty_actor_bucket),
        "lengths": {"total": [], "think": [], "json": []},
    }


def _summarize_phase(phase_state: dict[str, Any]) -> dict[str, Any]:
    lengths = phase_state["lengths"]
    return {
        "total_turns": phase_state["total_turns"],
        "by_actor": {
            actor_id: dict(bucket)
            for actor_id, bucket in phase_state["by_actor"].items()
        },
        "length_split_mean": {
            "total": (sum(lengths["total"]) / len(lengths["total"])) if lengths["total"] else 0.0,
            "think": (sum(lengths["think"]) / len(lengths["think"])) if lengths["think"] else 0.0,
            "json": (sum(lengths["json"]) / len(lengths["json"])) if lengths["json"] else 0.0,
        },
        "length_split_p95": {
            "total": _percentile(lengths["total"], 95),
            "think": _percentile(lengths["think"], 95),
            "json": _percentile(lengths["json"], 95),
        },
    }


def _accumulate(phase_state: dict[str, Any], record: dict[str, Any]) -> None:
    # v1 records (no parse_method) only contribute to total_turns.
    actor_id = record.get("actor_id", "unknown")
    bucket = phase_state["by_actor"][actor_id]
    bucket["turns"] += 1
    phase_state["total_turns"] += 1

    parse_method = record.get("parse_method")
    if parse_method is not None:
        if parse_method == PARSE_METHOD_STRICT_JSON:
            bucket["strict_json"] += 1
        if record.get("parse_valid"):
            bucket["parseable"] += 1
        if record.get("parse_valid") and record.get("legal_valid"):
            bucket["legal"] += 1
        if record.get("default_used"):
            bucket["default_used"] += 1

    length = record.get("length_split") or {}
    if length:
        phase_state["lengths"]["total"].append(int(length.get("total_chars", 0)))
        phase_state["lengths"]["think"].append(int(length.get("think_block_chars", 0)))
        phase_state["lengths"]["json"].append(int(length.get("json_chars", 0)))


def compute_telemetry(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate strict/parseable/legal counts and length splits across ``rounds``.

    Reads schema-v2 fields off ``rounds[i]["players"][j]`` (bids) and
    ``rounds[i]["players"][j]["reveal"]`` (reveals). v1 records contribute
    only to ``total_turns``.
    """
    bids_state = _empty_phase_summary()
    reveals_state = _empty_phase_summary()

    for round_data in rounds or []:
        for player_record in round_data.get("players", []) or []:
            _accumulate(bids_state, player_record)
            reveal = player_record.get("reveal")
            if isinstance(reveal, dict):
                _accumulate(reveals_state, reveal)

    return {
        "schema_version": 2,
        "bids": _summarize_phase(bids_state),
        "reveals": _summarize_phase(reveals_state),
    }
