"""Tests for the depth-1 piKL reveal search and the _run_game_loop refactor."""

from __future__ import annotations

import asyncio
import json
import math
import random
from collections import Counter

import pytest

from megagem.environment.multi_agent_env import MegaGemEnv
from megagem.environment.pikl_search import (
    PiklRevealSearcher, fair_value_bid, redeal_consistent, _weighted_var,
)
from megagem.game.cards import AuctionCard, AuctionType
from megagem.game.rules import resolve_auction

_COLORS = ["Red", "Blue", "Green", "Purple", "Yellow"]


def _first_hand_color(prompt: str) -> str:
    i = prompt.find("your_private_hand")
    seg = prompt[i:i + 200] if i >= 0 else prompt
    return next((c for c in _COLORS if c in seg), "Red")


class _Msg:
    def __init__(self, content):
        self.content = content
        self.reasoning_content = None


class _Resp:
    def __init__(self, content, n):
        self.choices = [type("C", (), {"message": _Msg(content)})() for _ in range(n)]


class _Completions:
    def __init__(self, owner):
        self._owner = owner

    async def create(self, *, model, messages, n=1, **kw):
        return _Resp(self._owner._content(messages), n)


class MockClient:
    """Deterministic, hand-aware: legal reveals and a fixed bid (no defaults)."""

    def __init__(self, bid: int = 0):
        self.bid = bid
        self.calls: list[str] = []
        self.chat = type("Chat", (), {"completions": _Completions(self)})()

    def _content(self, messages) -> str:
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        if "color string" in prompt:  # reveal schema marker
            return json.dumps({"reveal": _first_hand_color(prompt)})
        return json.dumps({"bid": self.bid})


def _env():
    return MegaGemEnv(num_players=3, value_chart_id="A", seed=123)


def _canon(state: dict) -> dict:
    return {k: v for k, v in state.items() if k not in ("timing", "model_chat_times_seconds")}


def _all_gems(gs) -> list[str]:
    out = list(gs.value_display) + list(gs.revealed_gems) + [g.color.value for g in gs.gem_deck]
    for p in gs.players:
        out += list(p.hand) + list(p.collection)
    return out


def _treasure_node(env, seed=5, bids=(9, 0, 0)):
    """A mid-round state where seat 0 has just won a 1-gem treasure auction."""
    gs = env.create_game_state(seed=seed)
    gs.current_auction = AuctionCard(id="t", type=AuctionType.TREASURE, gems=1)
    outcome = resolve_auction(gs, list(bids))
    assert outcome.winner_id == 0
    return gs, outcome, list(bids)


# --- 1. _run_game_loop refactor: a full game is deterministic -----------------

def test_run_game_loop_byte_identical():
    env, client = _env(), MockClient()

    def run():
        return asyncio.run(env.rollout(client=client, model="m", prompt=[], info={"seed": 123}))

    _, s1 = run()
    _, s2 = run()
    assert json.dumps(_canon(s1), sort_keys=True) == json.dumps(_canon(s2), sort_keys=True)
    assert s1["num_rounds"] > 0 and len(s1["final_scores"]) == 3


# --- 2. redeal_consistent: leak-safe, info-set preserving ---------------------

def test_redeal_preserves_info_set_and_conserves_gems():
    gs = _env().create_game_state(seed=7)
    own = list(gs.players[0].hand)
    public = (list(gs.value_display), list(gs.revealed_gems),
              [list(p.collection) for p in gs.players], [p.coins for p in gs.players],
              [len(p.hand) for p in gs.players])
    total = Counter(_all_gems(gs))

    w = redeal_consistent(gs, 0, random.Random(1))

    assert w.players[0].hand == own  # own hand fixed
    assert (list(w.value_display), list(w.revealed_gems),
            [list(p.collection) for p in w.players], [p.coins for p in w.players],
            [len(p.hand) for p in w.players]) == public  # public fixed
    assert Counter(_all_gems(w)) == total  # gem multiset conserved
    assert gs.players[1].hand == [p for p in gs.players if p.player_id == 1][0].hand  # original intact


def test_redeal_is_seed_deterministic_and_varies_opponents():
    gs = _env().create_game_state(seed=7)
    a = redeal_consistent(gs, 0, random.Random(1))
    b = redeal_consistent(gs, 0, random.Random(1))
    assert [p.hand for p in a.players] == [p.hand for p in b.players]  # same seed → same world
    variants = {tuple(redeal_consistent(gs, 0, random.Random(s)).players[1].hand) for s in range(12)}
    assert len(variants) > 1  # opponent hands actually move across seeds


def test_redeal_is_info_set_invariant_to_hidden_ordering():
    gs = _env().create_game_state(seed=7)
    twin = gs.copy()  # same info set for seat 0, reversed hidden ordering/partition
    for p in twin.players:
        if p.player_id != 0:
            p.hand = list(reversed(p.hand))
    twin.gem_deck = list(reversed(twin.gem_deck))
    twin.auction_deck = list(reversed(twin.auction_deck))
    a = redeal_consistent(gs, 0, random.Random(5))
    b = redeal_consistent(twin, 0, random.Random(5))
    assert [p.hand for p in a.players] == [p.hand for p in b.players]
    assert [g.color.value for g in a.gem_deck] == [g.color.value for g in b.gem_deck]
    assert [c.id for c in a.auction_deck] == [c.id for c in b.auction_deck]


# --- 3. τ̂ estimation: normalized, count-respecting ----------------------------

def test_tau_hat_normalized():
    s = _searcher(lam=1.0, n_tau=8)
    gs, _o, _b = _treasure_node(s._cont_env)
    legal = sorted(set(gs.players[0].hand))
    from megagem.environment.prompts import generate_reveal_prompt
    tau = asyncio.run(s._tau_hat(generate_reveal_prompt(gs, 0), 0, legal))
    assert set(tau) == set(legal)
    assert math.isclose(sum(tau.values()), 1.0, abs_tol=1e-9)
    top = _first_hand_color(generate_reveal_prompt(gs, 0))
    assert tau[top] == max(tau.values())  # the sampled color dominates


# --- 4. piKL formula limits ---------------------------------------------------

def test_pikl_formula_limits():
    s = _searcher(lam=1.0)
    tau = {"Red": 0.5, "Blue": 0.5}
    q = {"Red": 1.0, "Blue": 0.0}
    s.lam = 1e6
    near_tau = s._pikl(tau, q)
    assert abs(near_tau["Red"] - 0.5) < 1e-3  # λ→∞ ⇒ blueprint
    s.lam = 1e-3
    near_argmax = s._pikl(tau, q)
    assert near_argmax["Red"] > 0.99  # λ→0 ⇒ argmax Q


def test_weighted_var():
    assert _weighted_var({"a": 0.5, "b": 0.5}, {"a": 1.0, "b": -1.0}) == pytest.approx(1.0)
    assert _weighted_var({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": -1.0}) == pytest.approx(0.0)


# --- 5. recursion guard -------------------------------------------------------

def test_recursion_guard():
    s = _searcher(lam=0.1)
    s._in_search = True
    gs, o, b = _treasure_node(s._cont_env)
    with pytest.raises(AssertionError):
        asyncio.run(s.search(gs, 0, o, b))


# --- 6. leak-safety is structural --------------------------------------------

def test_continuation_uses_only_blueprint():
    s = _searcher(lam=0.1)
    assert {id(c) for c in s._cont_clients} == {id(s.bp_client)}
    assert set(s._cont_models) == {s.bp_model}
    assert "trainable" not in s._cont_actor_ids


# --- 7. search end-to-end (λ=∞ control and finite λ) --------------------------

def test_search_lambda_inf_returns_blueprint_reveal():
    s = _searcher(lam=math.inf)
    gs, o, b = _treasure_node(s._cont_env)
    color, rec = asyncio.run(s.search(gs, 0, o, b))
    assert color in gs.players[0].hand
    assert rec.parse_method == "pikl" and rec.legal_valid and not rec.default_used


def test_search_finite_lambda_picks_legal_and_audits():
    s = _searcher(lam=0.1, m_worlds=3, n_tau=4)
    gs, o, b = _treasure_node(s._cont_env)
    color, rec = asyncio.run(s.search(gs, 0, o, b))
    assert color in gs.players[0].hand
    payload = json.loads(rec.raw_response)["pikl"]
    assert set(payload["Q"]) == set(payload["tau"]) == set(sorted(set(gs.players[0].hand)))
    assert math.isclose(sum(payload["pi"].values()), 1.0, abs_tol=1e-3)


def test_search_lambda_zero_is_argmax_q():
    s = _searcher(lam=0, m_worlds=3, n_tau=4)
    gs, o, b = _treasure_node(s._cont_env)
    color, rec = asyncio.run(s.search(gs, 0, o, b))
    payload = json.loads(rec.raw_response)["pikl"]
    assert payload["lambda"] == 0 and "Q" in payload and "pi" not in payload and "tau" not in payload
    assert payload["Q"][color] >= max(payload["Q"].values()) - 1e-4  # argmax Q
    assert color in gs.players[0].hand


def test_pikl_lambda_zero_does_not_divide():
    # the formula path is never hit for λ=0 (search branches first); confirm no crash
    s = _searcher(lam=0, m_worlds=2, n_tau=2)
    gs, o, b = _treasure_node(s._cont_env)
    asyncio.run(s.search(gs, 0, o, b))  # ZeroDivisionError would fail here


def test_diagnostics_hook_records_and_keeps_blueprint_play():
    s = _searcher(lam=0.1, m_worlds=2, n_tau=4)
    s.diagnostics = []
    gs, o, b = _treasure_node(s._cont_env)
    color, rec = asyncio.run(s.search(gs, 0, o, b))
    assert color in gs.players[0].hand
    assert len(s.diagnostics) == 1
    d = s.diagnostics[0]
    assert set(d["Q_belief"]) == set(d["Q_oracle"]) == set(d["tau"])
    assert d["var_belief"] >= 0 and d["var_oracle"] >= 0
    assert "Q" not in json.loads(rec.raw_response)["pikl"]  # play stayed blueprint


# --- 8. value-aware reveal screen (fair-value opponents) ----------------------

def test_fair_value_bid_private_edge_clamp_and_nontreasure():
    gs = _env().create_game_state(seed=7)
    color = gs.revealed_gems[0]
    gs.players[0].hand = [color, color]  # seat 0 privately holds the auctioned colour
    gs.current_auction = AuctionCard(id="t", type=AuctionType.TREASURE, gems=1)
    assert fair_value_bid(gs, 0, 0) > fair_value_bid(gs, 1, 0) == 0  # private edge; opp sees empty display
    gs.players[0].coins = 1
    assert fair_value_bid(gs, 0, 0) <= 1  # clamped to coins
    gs.current_auction = AuctionCard(id="L", type=AuctionType.LOAN, amount=5)
    assert fair_value_bid(gs, 0, 0) == 0  # non-treasure → 0


def test_value_aware_rollout_is_deterministic():
    s = _searcher(lam=0.1, m_worlds=2)
    s.value_aware = True
    gs, o, b = _treasure_node(s._cont_env)
    w = s._make_worlds(gs, 0)[0]
    color = sorted(set(gs.players[0].hand))[0]
    m1 = asyncio.run(s._rollout_one(w, color, 0, o, b))
    m2 = asyncio.run(s._rollout_one(w, color, 0, o, b))
    assert m1 == m2 and isinstance(m1, float)  # heuristic continuation has no LLM/RNG noise


def test_value_aware_search_records_diagnostics():
    s = _searcher(lam=0.1, m_worlds=3, n_tau=4)
    s.value_aware = True
    s.diagnostics = []
    gs, o, b = _treasure_node(s._cont_env)
    color, _rec = asyncio.run(s.search(gs, 0, o, b))
    assert color in gs.players[0].hand and len(s.diagnostics) == 1
    assert s.diagnostics[0]["var_belief"] >= 0


def _searcher(*, lam, n_tau=8, m_worlds=4, temperature=1.0):
    return PiklRevealSearcher(
        bp_client=MockClient(), bp_model="bp", trainable_seat=0, num_players=3,
        value_chart_id="A", bp_extra={}, lam=lam, n_tau=n_tau, temperature=temperature,
        m_worlds=m_worlds, crn_seed=42, max_parallel=4,
    )
