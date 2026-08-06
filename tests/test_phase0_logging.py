"""Tests for raw-action logging and per-turn telemetry."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pytest

from megagem.environment.multi_agent_env import (
    BidTurnRecord,
    DEFAULT_ACTOR_ID,
    MegaGemEnv,
    RevealTurnRecord,
)
from megagem.environment.telemetry import compute_telemetry
from megagem.game.actions import (
    PARSE_METHOD_BRACE_MATCH,
    PARSE_METHOD_CODE_BLOCK,
    PARSE_METHOD_NONE,
    PARSE_METHOD_PROSE_FALLBACK,
    PARSE_METHOD_STRICT_JSON,
    extract_json_from_text,
    extract_json_with_method,
    length_split,
    parse_bid,
    parse_reveal,
)


# ---------- parse_method classification ----------

class TestParseMethod:
    def test_strict_json_bid(self):
        r = parse_bid('{"bid": 5}')
        assert r.valid is True
        assert r.bid == 5
        assert r.parse_method == PARSE_METHOD_STRICT_JSON

    def test_code_block_bid(self):
        r = parse_bid('Reasoning here.\n```json\n{"bid": 7}\n```')
        assert r.valid is True
        assert r.bid == 7
        assert r.parse_method == PARSE_METHOD_CODE_BLOCK

    def test_brace_match_bid(self):
        r = parse_bid('I think 3 is right. {"bid": 3, "extra": "ignored"}')
        assert r.valid is True
        assert r.bid == 3
        assert r.parse_method == PARSE_METHOD_BRACE_MATCH

    def test_prose_fallback_bid(self):
        r = parse_bid("After thinking it over, I will bid 8.")
        assert r.valid is True
        assert r.bid == 8
        assert r.parse_method == PARSE_METHOD_PROSE_FALLBACK

    def test_no_parse_bid(self):
        r = parse_bid("nothing parseable here")
        assert r.valid is False
        assert r.parse_method == PARSE_METHOD_NONE

    def test_invalid_bid_value_keeps_method(self):
        # parse_method must distinguish "clean JSON, bad field" from "no JSON".
        r = parse_bid('{"bid": "not a number"}')
        assert r.valid is False
        assert r.parse_method == PARSE_METHOD_STRICT_JSON

    def test_strict_json_reveal(self):
        r = parse_reveal('{"reveal": "Red"}')
        assert r.valid is True
        assert r.gem_color == "Red"
        assert r.parse_method == PARSE_METHOD_STRICT_JSON

    def test_brace_match_reveal(self):
        r = parse_reveal('Going with red. {"reveal": "Red"}')
        assert r.valid is True
        assert r.parse_method == PARSE_METHOD_BRACE_MATCH

    def test_no_parse_reveal(self):
        r = parse_reveal("blah blah")
        assert r.valid is False
        assert r.parse_method == PARSE_METHOD_NONE

    def test_unknown_color_is_unparseable(self):
        # Hallucinated color names are shape errors, not rule violations.
        r = parse_reveal('{"reveal": "PurpleZebra"}')
        assert r.valid is False
        assert r.parse_method == PARSE_METHOD_STRICT_JSON
        assert "Invalid gem color" in r.error

    def test_parse_reveal_does_not_check_hand(self):
        # Hand-membership is a legal check (validate_reveal_for_hand), not parse.
        r = parse_reveal('{"reveal": "Blue"}')
        assert r.valid is True
        assert r.gem_color == "Blue"

    def test_negative_bid_is_parseable(self):
        # {"bid": -1} is well-formed JSON; legality is validate_bid_for_auction's job.
        r = parse_bid('{"bid": -1}')
        assert r.valid is True
        assert r.parse_method == PARSE_METHOD_STRICT_JSON
        assert r.bid == -1

    def test_extract_json_with_method_returns_span(self):
        data, method, span = extract_json_with_method('Some prose. {"bid": 2}')
        assert data == {"bid": 2}
        assert method == PARSE_METHOD_BRACE_MATCH
        assert span is not None and span[1] > span[0]

    def test_backwards_compat_extract_json_from_text(self):
        # Non-method-aware wrapper still returns the dict.
        assert extract_json_from_text('{"bid": 5}') == {"bid": 5}
        assert extract_json_from_text("nothing") is None


# ---------- length_split ----------

class TestLengthSplit:
    def test_think_block_and_json(self):
        text = "<think>some reasoning</think>\n{\"bid\": 5}"
        ls = length_split(text)
        assert ls["total_chars"] == len(text)
        assert ls["think_block_chars"] == len("some reasoning")
        assert ls["json_chars"] == len('{"bid": 5}')

    def test_no_think_no_json(self):
        ls = length_split("just prose, no tags")
        assert ls["total_chars"] == len("just prose, no tags")
        assert ls["think_block_chars"] == 0
        assert ls["json_chars"] == 0

    def test_empty(self):
        assert length_split("") == {"total_chars": 0, "think_block_chars": 0, "json_chars": 0}

    def test_multiple_think_blocks(self):
        text = "<think>aaa</think> mid <think>bbbb</think> {\"bid\": 1}"
        ls = length_split(text)
        assert ls["think_block_chars"] == 3 + 4
        assert ls["json_chars"] == len('{"bid": 1}')


# ---------- BidTurnRecord / RevealTurnRecord shape ----------

class TestTurnRecordShape:
    # Pin the raw-action-logging field names so a rename surfaces immediately.
    REQUIRED_FIELDS = {
        "prompt",
        "raw_response",
        "parsed_action",
        "parse_valid",
        "legal_valid",
        "default_used",
        "actor_id",
    }

    def test_bid_record_has_all_required_fields(self):
        assert self.REQUIRED_FIELDS.issubset(set(BidTurnRecord.__dataclass_fields__.keys()))

    def test_reveal_record_has_all_required_fields(self):
        assert self.REQUIRED_FIELDS.issubset(set(RevealTurnRecord.__dataclass_fields__.keys()))


# ---------- run_bidding_phase / run_reveal_phase return shape ----------

class _FakeChoice:
    def __init__(self, content: str):
        self.message = type("M", (), {"content": content, "reasoning_content": None})()


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeChat:
    def __init__(self, contents: dict[int, str]):
        self._contents = list(contents.values())
        self._idx = 0

    async def create(self, **kwargs):
        content = self._contents[self._idx % len(self._contents)]
        self._idx += 1
        return _FakeResponse(content)


class _FakeChatCompletions:
    def __init__(self, chat: _FakeChat):
        self._chat = chat

    @property
    def completions(self):
        return self._chat


class _FakeAsyncOpenAI:
    """Smallest-possible stand-in for AsyncOpenAI used in env tests."""

    def __init__(self, content: str):
        self._chat_obj = _FakeChat({0: content})

    @property
    def chat(self):
        return _FakeChatCompletions(self._chat_obj)


def _make_env(seed: int = 42) -> MegaGemEnv:
    return MegaGemEnv(num_players=3, value_chart_id="A", seed=seed)


@pytest.mark.asyncio
async def test_run_bidding_phase_returns_records():
    env = _make_env()
    game_state = env.create_game_state(seed=42)

    # Starting coins are 35 for 3 players, so 9999 is illegal under any auction type.
    clients = [
        _FakeAsyncOpenAI('{"bid": 1}'),                  # strict_json, legal
        _FakeAsyncOpenAI("I'll bid 5 next."),            # prose_fallback, legal
        _FakeAsyncOpenAI('{"bid": 9999}'),               # strict_json, illegal
    ]
    auction = game_state.draw_auction_card()
    assert auction is not None

    bids, records = await env.run_bidding_phase(
        game_state, clients, ["fake"] * 3, sampling_args=None
    )
    assert len(bids) == 3 and all(isinstance(r, BidTurnRecord) for r in records)

    r0, r1, r2 = records

    assert r0.parse_method == PARSE_METHOD_STRICT_JSON
    assert r0.parse_valid is True
    assert r0.legal_valid is True
    assert r0.default_used is False
    assert r0.parsed_action == 1
    assert r0.final_bid == 1
    assert r0.actor_id == DEFAULT_ACTOR_ID

    assert r1.parse_method == PARSE_METHOD_PROSE_FALLBACK
    assert r1.parse_valid is True
    assert r1.legal_valid is True
    assert r1.default_used is False
    assert r1.parsed_action == 5
    assert r1.final_bid == 5

    assert r2.parse_method == PARSE_METHOD_STRICT_JSON
    assert r2.parse_valid is True
    assert r2.legal_valid is False
    assert r2.default_used is True
    assert r2.parsed_action == 9999
    assert r2.final_bid == 0
    assert r2.legal_error


@pytest.mark.asyncio
async def test_actor_ids_threaded_through_records():
    env = _make_env()
    game_state = env.create_game_state(seed=42)
    auction = game_state.draw_auction_card()
    assert auction is not None

    clients = [_FakeAsyncOpenAI('{"bid": 0}') for _ in range(3)]
    models = ["fake"] * 3
    actor_ids = ["trainable", "snapshot_0", "heuristic"]

    _, records = await env.run_bidding_phase(
        game_state, clients, models, sampling_args=None, actor_ids=actor_ids
    )
    assert [r.actor_id for r in records] == actor_ids


@pytest.mark.asyncio
async def test_run_bidding_phase_separates_parse_from_legal_for_negative_bids():
    """Negative bids: parse_valid=True, legal_valid=False, default substituted."""
    env = _make_env()
    game_state = env.create_game_state(seed=42)
    auction = game_state.draw_auction_card()
    assert auction is not None

    clients = [_FakeAsyncOpenAI('{"bid": -3}') for _ in range(3)]
    _, records = await env.run_bidding_phase(
        game_state, clients, ["fake"] * 3, sampling_args=None
    )
    for r in records:
        assert r.parse_valid is True
        assert r.parse_method == PARSE_METHOD_STRICT_JSON
        assert r.parsed_action == -3
        assert r.legal_valid is False
        assert r.default_used is True
        assert r.final_bid == 0
        assert r.legal_error


@pytest.mark.asyncio
async def test_run_reveal_phase_separates_parse_from_legal_for_out_of_hand():
    """Reveal of a color not in hand: parse_valid=True, legal_valid=False."""
    env = _make_env()
    game_state = env.create_game_state(seed=42)
    winner_id = 0
    # Force a hand that does NOT contain Blue.
    game_state.players[winner_id].hand = ["Red", "Red", "Green"]

    client = _FakeAsyncOpenAI('{"reveal": "Blue"}')
    final_reveal, record = await env.run_reveal_phase(
        game_state, winner_id, client, "fake", sampling_args=None
    )
    assert record is not None
    assert record.parse_valid is True
    assert record.parse_method == PARSE_METHOD_STRICT_JSON
    assert record.parsed_action == "Blue"
    assert record.legal_valid is False
    assert record.default_used is True
    # The env substituted a default in-hand color.
    assert final_reveal in {"Red", "Green"}
    assert record.legal_error


@pytest.mark.asyncio
async def test_run_reveal_phase_legal_when_color_in_hand():
    env = _make_env()
    game_state = env.create_game_state(seed=42)
    winner_id = 0
    game_state.players[winner_id].hand = ["Red", "Green", "Blue"]

    client = _FakeAsyncOpenAI('{"reveal": "Blue"}')
    final_reveal, record = await env.run_reveal_phase(
        game_state, winner_id, client, "fake", sampling_args=None
    )
    assert record is not None
    assert record.parse_valid is True
    assert record.legal_valid is True
    assert record.default_used is False
    assert record.parsed_action == "Blue"
    assert final_reveal == "Blue"


@pytest.mark.asyncio
async def test_run_reveal_phase_unknown_color_is_unparseable():
    env = _make_env()
    game_state = env.create_game_state(seed=42)
    winner_id = 0
    game_state.players[winner_id].hand = ["Red", "Green"]

    client = _FakeAsyncOpenAI('{"reveal": "PurpleZebra"}')
    _, record = await env.run_reveal_phase(
        game_state, winner_id, client, "fake", sampling_args=None
    )
    assert record is not None
    assert record.parse_valid is False
    assert record.legal_valid is False
    assert record.default_used is True
    # parsed_action is None because the color name didn't refer to any action.
    assert record.parsed_action is None


@pytest.mark.asyncio
async def test_run_bidding_phase_actor_ids_default_to_trainable():
    env = _make_env()
    game_state = env.create_game_state(seed=42)
    auction = game_state.draw_auction_card()
    assert auction is not None

    clients = [_FakeAsyncOpenAI('{"bid": 0}') for _ in range(3)]
    _, records = await env.run_bidding_phase(
        game_state, clients, ["fake"] * 3, sampling_args=None
    )
    assert all(r.actor_id == DEFAULT_ACTOR_ID for r in records)


# ---------- compute_telemetry ----------

class TestComputeTelemetry:
    def _round(self, players: list[dict[str, Any]]) -> dict[str, Any]:
        return {"round_number": 1, "players": players}

    def _bid_record(
        self,
        actor_id: str = "trainable",
        parse_method: str = PARSE_METHOD_STRICT_JSON,
        parse_valid: bool = True,
        legal_valid: bool = True,
        default_used: bool = False,
        total: int = 50,
        think: int = 30,
        json_: int = 10,
        reveal: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "player_id": 0,
            "actor_id": actor_id,
            "parse_method": parse_method,
            "parse_valid": parse_valid,
            "legal_valid": legal_valid,
            "default_used": default_used,
            "length_split": {
                "total_chars": total,
                "think_block_chars": think,
                "json_chars": json_,
            },
        }
        if reveal is not None:
            rec["reveal"] = reveal
        return rec

    def test_empty_rounds(self):
        t = compute_telemetry([])
        assert t["bids"]["total_turns"] == 0
        assert t["reveals"]["total_turns"] == 0
        assert t["schema_version"] == 2

    def test_counts_by_actor(self):
        rounds = [
            self._round([
                self._bid_record(actor_id="trainable"),
                self._bid_record(actor_id="trainable", parse_method=PARSE_METHOD_BRACE_MATCH),
                self._bid_record(actor_id="snapshot_0", parse_method=PARSE_METHOD_NONE,
                                  parse_valid=False, legal_valid=False, default_used=True,
                                  total=20, think=0, json_=0),
            ]),
        ]
        t = compute_telemetry(rounds)
        assert t["bids"]["total_turns"] == 3
        assert t["bids"]["by_actor"]["trainable"]["turns"] == 2
        assert t["bids"]["by_actor"]["trainable"]["strict_json"] == 1
        assert t["bids"]["by_actor"]["trainable"]["parseable"] == 2
        assert t["bids"]["by_actor"]["trainable"]["legal"] == 2
        assert t["bids"]["by_actor"]["trainable"]["default_used"] == 0
        assert t["bids"]["by_actor"]["snapshot_0"]["parseable"] == 0
        assert t["bids"]["by_actor"]["snapshot_0"]["default_used"] == 1

    def test_reveal_counted_separately(self):
        reveal = {
            "actor_id": "trainable",
            "parse_method": PARSE_METHOD_STRICT_JSON,
            "parse_valid": True,
            "legal_valid": True,
            "default_used": False,
            "length_split": {"total_chars": 20, "think_block_chars": 10, "json_chars": 8},
        }
        rounds = [self._round([self._bid_record(reveal=reveal)])]
        t = compute_telemetry(rounds)
        assert t["bids"]["total_turns"] == 1
        assert t["reveals"]["total_turns"] == 1
        assert t["reveals"]["by_actor"]["trainable"]["legal"] == 1

    def test_lengths_mean_and_p95(self):
        rounds = [self._round([
            self._bid_record(total=10, think=5, json_=3),
            self._bid_record(total=20, think=10, json_=6),
            self._bid_record(total=30, think=15, json_=9),
            self._bid_record(total=40, think=20, json_=12),
        ])]
        t = compute_telemetry(rounds)
        assert t["bids"]["length_split_mean"]["total"] == 25.0
        assert t["bids"]["length_split_p95"]["total"] == 40.0

    def test_v1_records_only_count_total_turns(self):
        rounds = [self._round([
            {"player_id": 0, "bid": 5, "is_winner": False},  # v1 shape
            self._bid_record(),
        ])]
        t = compute_telemetry(rounds)
        assert t["bids"]["total_turns"] == 2
        assert t["bids"]["by_actor"]["trainable"]["parseable"] == 1
        assert t["bids"]["by_actor"]["unknown"]["turns"] == 1
        assert t["bids"]["by_actor"]["unknown"]["parseable"] == 0


# ---------- v1 schema preservation (run_game.py round_data shape) ----------

def test_bid_turn_record_asdict_is_json_serializable():
    """asdict(BidTurnRecord) must round-trip through json.dump for on-disk JSONs."""
    import json

    record = BidTurnRecord(
        player_id=0,
        actor_id="trainable",
        prompt="bid prompt text",
        raw_response="<think>x</think>{\"bid\":1}",
        parsed_action=1,
        parse_method=PARSE_METHOD_STRICT_JSON,
        parse_valid=True,
        legal_valid=True,
        default_used=False,
        final_bid=1,
        reasoning="x",
        length_split={"total_chars": 28, "think_block_chars": 1, "json_chars": 9},
        parse_error="",
        legal_error="",
    )
    blob = json.dumps(asdict(record))
    assert "trainable" in blob
    assert '"parse_method": "strict_json"' in blob
    assert '"prompt": "bid prompt text"' in blob


# ---------- end-to-end rollout smoke test ----------

@pytest.mark.asyncio
async def test_full_rollout_emits_v2_records():
    """End-to-end rollout: every per-turn record in player_messages carries v2 fields."""
    env = _make_env(seed=7)
    clients = [_FakeAsyncOpenAI('{"bid": 0}\n{"reveal": "Red"}') for _ in range(3)]
    actor_ids = ["trainable", "snapshot_0", "heuristic"]

    _, final_state = await env.rollout(
        client=clients[0],
        model="fake",
        prompt=[],
        clients=clients,
        models=["fake"] * 3,
        actor_ids=actor_ids,
    )

    assert final_state["actor_ids"] == actor_ids

    required_bid_keys = {
        "phase",
        "round",
        "player_id",
        "actor_id",
        "prompt",
        "raw_response",
        "parsed_action",
        "parse_method",
        "parse_valid",
        "legal_valid",
        "default_used",
        "final_bid",
        "length_split",
    }
    saw_at_least_one_bid = False
    for player_id, messages in enumerate(final_state["player_messages"]):
        expected_actor = actor_ids[player_id]
        for msg in messages:
            if msg.get("phase") == "bid":
                assert required_bid_keys.issubset(msg.keys()), (
                    f"missing keys: {required_bid_keys - msg.keys()}"
                )
                assert msg["actor_id"] == expected_actor
                assert isinstance(msg["length_split"], dict)
                saw_at_least_one_bid = True
    assert saw_at_least_one_bid, "expected at least one bid record"


def test_v1_player_round_data_keys_still_present():
    """Schema v2/v3 keys must be additive — no collisions with the v1 surface."""
    v1_keys = {
        "player_id",
        "model",
        "bid",
        "coins_before",
        "coins_after",
        "collection",
        "collection_counts",
        "hand",
        "reasoning",
        "is_winner",
    }
    additive_keys = {
        "actor_id",
        "prompt",
        "raw_response",
        "parsed_action",
        "parse_method",
        "parse_valid",
        "legal_valid",
        "default_used",
        "length_split",
        "parse_error",
        "legal_error",
    }
    assert v1_keys.isdisjoint(additive_keys), (
        "schema-bumped keys must not collide with v1 keys (additive contract)"
    )
