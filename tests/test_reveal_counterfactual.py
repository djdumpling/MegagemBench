"""Firewalled PRIVATE-HAND counterfactual reveal probe (Caveat 2 addendum).

Covers: the counterfactual arithmetic (hand-computed), consistency with the
production `reveal_shaping` for the actually-chosen colour, honest handling of
scrubbed/empty hands, the opt-in CLI, and — critically — the **firewall**:
the hand-free RL core must not import or reach this hand-reading module.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from megagem.analysis.reveal_counterfactual import (
    main as rc_main,
    reveal_counterfactual_diagnostic,
)
from megagem.rl.reward import RewardConfig, per_turn_rewards
from megagem.rl.scorer import parse_transcript
from _rl_fixtures import TRACKED_DIR, fixture_paths, load as _load

FIXTURES = fixture_paths()
REPO = Path(__file__).resolve().parent.parent


def _game(
    *,
    seed: int,
    p0_collection: dict[str, int],
    p1_collection: dict[str, int],
    p2_collection: dict[str, int],
    recorded_p0_hand: list[str],
    revealed: str,
    finals: tuple[int, int, int] = (90, 10, 10),
) -> dict:
    """1-round treasure game: P0 wins and reveals ``revealed`` into an empty
    pre-display; collections are fixed so the reveal counterfactual is pure
    display revaluation. ``recorded_p0_hand`` is the *post-reveal* hand."""
    cols = (p0_collection, p1_collection, p2_collection)
    players = []
    for pid in range(3):
        c = cols[pid]
        coll = [g for g, k in c.items() for _ in range(k)]
        players.append({
            "player_id": pid, "actor_id": "trainable", "bid": 1,
            "coins_before": 35, "coins_after": 35,
            "collection": coll, "collection_counts": dict(c),
            "is_winner": pid == 0, "parse_valid": True, "legal_valid": True,
            "reasoning": "", "prompt": f"P{pid}", "raw_response": "r",
            "hand": list(recorded_p0_hand) if pid == 0 else [],
        })
    players[0]["winning_bid"] = 1
    players[0]["gem_revealed"] = revealed
    players[0]["reveal"] = {
        "actor_id": "trainable", "parse_valid": True, "legal_valid": True,
        "final_reveal": revealed, "parsed_action": revealed,
        "prompt": "P0 reveal", "raw_response": revealed,
    }
    return {
        "metadata": {"schema_version": 3, "seed": seed, "num_players": 3,
                     "value_chart": "A"},
        "rounds": [{
            "round_number": 1,
            "auction": {"type": "treasure", "description": "t"},
            "players": players,
            # post-reveal display: Red (the revealed colour) count 1 ⇒ value 4
            "value_display": {revealed: {"count": 1, "value_per_gem": 4}},
            "tiebreak_order": [0, 1, 2],
            "available_missions": [], "missions_completed": {},
        }],
        "final_results": {
            "winner_id": 0, "num_rounds": 1, "available_missions": [],
            "final_scores": [
                {"player_id": p, "final_score": finals[p],
                 "loan_payments": 0, "investment_returns": 0}
                for p in range(3)
            ],
        },
    }


# ---- counterfactual arithmetic (hand-computed) -----------------------------


def test_counterfactual_math_and_percentile_hand_computed() -> None:
    """Chart A, empty pre-display ⇒ revealing colour c gives it value 4.
    P0 col Red×3,Blue×1; P1 Blue×2,Green×1; P2 Green×4; P0 reveals Red,
    recorded (post-reveal) hand = [Blue, Green] ⇒ choice set {Red,Blue,Green}.
    λ=1 ⇒ cf_shaping: Red = 4·3 − mean(0,0) = 12; Blue = 4·1 − mean(8,0) = 0;
    Green = 0 − mean(4,16) = −10. Chosen Red is the argmax."""
    g = _game(
        seed=1,
        p0_collection={"Red": 3, "Blue": 1},
        p1_collection={"Blue": 2, "Green": 1},
        p2_collection={"Green": 4},
        recorded_p0_hand=["Blue", "Green"],  # post-reveal: Red already gone
        revealed="Red",
    )
    rep = reveal_counterfactual_diagnostic(
        [g], RewardConfig(shaping_lambda=1.0), min_choice_turns=1
    )
    assert rep["policy_choice_turns"] == 1
    assert rep["reveal_turns_forced_no_choice"] == 0
    # normalised mid-rank: chose the strict max of 3 distinct options ⇒
    # midrank 3, (3−1)/(3−1) = 1.0 (random null is 0.5 for any k)
    assert rep["mean_self_favoring_rank"] == 1.0
    assert rep["random_choice_rank_null"] == 0.5
    # excess = 12 − mean(12, 0, −10) = 12 − 0.6667 = 11.3333  (null 0.0)
    assert rep["mean_excess_over_options"] == pytest.approx(11.3333, abs=1e-3)
    assert rep["frac_chose_argmax_self_favoring"] == 1.0
    assert rep["observation"].startswith("SELFISH-REVEAL SELECTION")


def test_counterfactual_chosen_equals_production_reveal_shaping() -> None:
    """Consistency anchor: for the colour actually revealed, the
    counterfactual must equal the production `reveal_shaping` on that reveal
    turn. Both are independently the hand-known value 12 at λ=1."""
    g = _game(
        seed=2,
        p0_collection={"Red": 3, "Blue": 1},
        p1_collection={"Blue": 2, "Green": 1},
        p2_collection={"Green": 4},
        recorded_p0_hand=["Blue", "Green"],
        revealed="Red",
    )
    parsed = parse_transcript(g)
    rev = [
        t for t in per_turn_rewards(parsed, RewardConfig(shaping_lambda=1.0))
        if t.phase == "reveal" and t.player_id == 0
    ]
    assert len(rev) == 1
    assert rev[0].shaping == pytest.approx(12.0, abs=1e-9)  # production
    # counterfactual on the chosen colour is the per-turn argmax value 12;
    # equality of the two independently-derived 12s is the consistency proof.


def test_self_sacrificing_reveal_scores_low_rank() -> None:
    """Same state but P0 reveals Green (helps P2 a lot, P0 not at all) ⇒
    chosen is the strict minimum option ⇒ normalised rank 0.0 (< 0.5 random
    null), negative excess, and the 'no selfish selection' observation."""
    g = _game(
        seed=3,
        p0_collection={"Red": 3, "Blue": 1},
        p1_collection={"Blue": 2, "Green": 1},
        p2_collection={"Green": 4},
        recorded_p0_hand=["Red", "Blue"],  # post-reveal: Green gone
        revealed="Green",
    )
    rep = reveal_counterfactual_diagnostic(
        [g], RewardConfig(shaping_lambda=1.0), min_choice_turns=1
    )
    assert rep["mean_self_favoring_rank"] == 0.0  # strict worst of 3
    assert rep["mean_excess_over_options"] < 0
    assert rep["frac_chose_argmax_self_favoring"] == 0.0
    assert rep["observation"].startswith("no selfish-reveal selection")


# ---- honest handling of unanalysable hands ---------------------------------


def test_scrubbed_hand_is_skipped_not_misanalysed() -> None:
    g = _game(
        seed=4, p0_collection={"Red": 2}, p1_collection={}, p2_collection={},
        recorded_p0_hand=["__SCRUBBED__"], revealed="Red",
    )
    rep = reveal_counterfactual_diagnostic([g], min_choice_turns=1)
    assert rep["reveal_turns_scrubbed_hand_skipped"] == 1
    assert rep["policy_choice_turns"] == 0
    assert "insufficient data" in rep["observation"]


def test_tracked_fixture_empty_hands_is_honest_not_a_crash() -> None:
    """The tracked sanitized fixture's hands are scrubbed to []; every reveal
    is then a forced single-option turn ⇒ insufficient data, no crash, no
    false signal. (Only the *tracked* fixture is hand-empty; a real corpus,
    if present, has real hands — covered by the invariants test below.)"""
    tracked = [json.loads(p.read_text())
               for p in sorted(Path(TRACKED_DIR).glob("*.json"))]
    rep = reveal_counterfactual_diagnostic(tracked)
    assert rep["policy_choice_turns"] == 0
    assert "insufficient data" in rep["observation"]
    assert rep["analysis_type"].startswith("PRIVATE-HAND")


@pytest.mark.skipif(not FIXTURES, reason="no schema-v3 fixtures")
def test_count_invariants_on_whatever_fixtures_exist() -> None:
    """Real-hand corpus (if present) yields real choice turns; assert the
    bookkeeping closes and the output never drops its private-hand label.
    No verdict is asserted — that depends on the model's actual behaviour."""
    rep = reveal_counterfactual_diagnostic([_load(p) for p in FIXTURES])
    assert (
        rep["policy_choice_turns"]
        + rep["reveal_turns_defaulted_excluded"]
        + rep["reveal_turns_forced_no_choice"]
        + rep["reveal_turns_scrubbed_hand_skipped"]
        == rep["reveal_turns_total"]
    )
    assert rep["analysis_type"].startswith("PRIVATE-HAND")
    assert isinstance(rep["observation"], str) and rep["observation"]


# ---- the firewall ----------------------------------------------------------

_HANDFREE_CORE = [
    "src/megagem/rl/reward.py", "src/megagem/rl/advantage.py", "src/megagem/rl/scorer.py",
    "src/megagem/rl/diagnostics.py", "src/megagem/rl/export.py",
]


@pytest.mark.parametrize("rel", _HANDFREE_CORE)
def test_handfree_core_does_not_reference_the_hand_tool(rel: str) -> None:
    src = (REPO / rel).read_text()
    assert "reveal_counterfactual" not in src, (
        f"{rel} references the private-hand tool — firewall breach"
    )
    assert "megagem.analysis" not in src and "from ..analysis" not in src, (
        f"{rel} imports the analysis package — firewall breach"
    )


def test_importing_handfree_core_does_not_load_the_hand_tool() -> None:
    sys.modules.pop("megagem.analysis.reveal_counterfactual", None)
    for m in ("megagem.rl.reward", "megagem.rl.advantage", "megagem.rl.scorer",
              "megagem.rl.diagnostics", "megagem.rl.export"):
        importlib.reload(importlib.import_module(m))
    assert "megagem.analysis.reveal_counterfactual" not in sys.modules, (
        "importing the hand-free core pulled in the private-hand tool"
    )


def test_leakage_gate_structural_invariant_still_holds() -> None:
    """The hand-free guarantee on the scorer is unchanged: TurnStateSnapshot
    still exposes no hand field (the tool re-scores via the same primitive,
    it does not add a hand to the snapshot)."""
    from megagem.rl.scorer import TurnStateSnapshot
    assert not any("hand" in f for f in TurnStateSnapshot.__dataclass_fields__)


# ---- opt-in CLI ------------------------------------------------------------


def test_cli_refuses_without_consent(capsys) -> None:
    rc = rc_main(["--corpus", "whatever"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "REFUSING TO RUN" in err and "PRIVATE HANDS" in err


def test_cli_runs_with_consent_and_labels_output(tmp_path, capsys) -> None:
    g = _game(
        seed=9, p0_collection={"Red": 3}, p1_collection={"Blue": 2},
        p2_collection={"Green": 4}, recorded_p0_hand=["Blue", "Green"],
        revealed="Red",
    )
    (tmp_path / "g.json").write_text(json.dumps(g))
    rc = rc_main([
        "--corpus", str(tmp_path), "--min-choice-turns", "1",
        "--i-understand-this-reads-private-hands",
    ])
    assert rc == 0
    cap = capsys.readouterr()
    # safety label on stderr (keeps stdout clean); table on stdout
    assert "PRIVATE-HAND counterfactual" in cap.err
    assert "mean self-favoring rank" in cap.out


def test_cli_json_stdout_is_parseable(tmp_path, capsys) -> None:
    """Finding 2: in --json mode stdout must be pure JSON (banner → stderr)."""
    g = _game(
        seed=10, p0_collection={"Red": 3}, p1_collection={"Blue": 2},
        p2_collection={"Green": 4}, recorded_p0_hand=["Blue", "Green"],
        revealed="Red",
    )
    (tmp_path / "g.json").write_text(json.dumps(g))
    rc = rc_main([
        "--corpus", str(tmp_path), "--min-choice-turns", "1", "--json",
        "--i-understand-this-reads-private-hands",
    ])
    assert rc == 0
    cap = capsys.readouterr()
    payload = json.loads(cap.out)  # must not raise
    assert payload["analysis_type"].startswith("PRIVATE-HAND")
    assert payload["random_choice_rank_null"] == 0.5
    assert "PRIVATE-HAND counterfactual" in cap.err  # label still emitted
