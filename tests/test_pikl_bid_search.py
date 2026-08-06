"""Tests for the depth-1 piKL bid search (Gate B) and its env hook."""

from __future__ import annotations

import asyncio
import json
import math

import pytest

from megagem.environment.multi_agent_env import MegaGemEnv
from megagem.environment.pikl_search import PiklBidSearcher
from megagem.environment.prompts import generate_bid_prompt
from megagem.game.cards import AuctionCard, AuctionType

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
    def __init__(self, contents):
        self.choices = [type("C", (), {"message": _Msg(c)})() for c in contents]


class _Completions:
    def __init__(self, owner):
        self._owner = owner

    async def create(self, *, model, messages, n=1, **kw):
        return _Resp(self._owner._contents(messages, n))


class BidMockClient:
    """Reveals legally; bids cycle through ``bids`` across the N samples so
    candidate enumeration yields multiple legal candidates."""

    def __init__(self, bids=(2, 5, 8)):
        self.bids = list(bids)
        self.calls: list[str] = []
        self.chat = type("Chat", (), {"completions": _Completions(self)})()

    def _contents(self, messages, n) -> list[str]:
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        if "color string" in prompt:  # reveal schema marker
            return [json.dumps({"reveal": _first_hand_color(prompt)})] * n
        return [json.dumps({"bid": self.bids[i % len(self.bids)]}) for i in range(n)]


def _searcher(*, lam, n_tau=6, m_worlds=4, bids=(2, 5, 8)):
    return PiklBidSearcher(
        bp_client=BidMockClient(bids), bp_model="bp", trainable_seat=0, num_players=3,
        value_chart_id="A", bp_extra={}, lam=lam, n_tau=n_tau, m_worlds=m_worlds,
        crn_seed=42, max_parallel=4,
    )


def _bid_node(env, seed=5):
    """A pre-bid state with a live 1-gem treasure auction."""
    gs = env.create_game_state(seed=seed)
    gs.current_auction = AuctionCard(id="t", type=AuctionType.TREASURE, gems=1)
    return gs


# --- 1. candidate enumeration + τ̂ -------------------------------------------

def test_candidates_span_samples_and_tau_normalized():
    s = _searcher(lam=1.0, n_tau=9, bids=(2, 5, 8))
    gs = _bid_node(s._cont_env)
    cands, tau = asyncio.run(s._candidates(generate_bid_prompt(gs, 0), gs, 0))
    assert set(cands) == {2, 5, 8}  # the support of the sampled bids
    assert cands == sorted(cands)
    assert math.isclose(sum(tau.values()), 1.0, abs_tol=1e-9)


def test_illegal_bids_dropped_then_default():
    # a bid above starting coins (35) is illegal → filtered; only legal 3 remains
    s = _searcher(lam=1.0, n_tau=8, bids=(3, 999))
    gs = _bid_node(s._cont_env)
    cands, _ = asyncio.run(s._candidates(generate_bid_prompt(gs, 0), gs, 0))
    assert cands == [3]


# --- 2. piKL search limits ---------------------------------------------------

def test_search_lambda_inf_returns_blueprint_bid():
    s = _searcher(lam=math.inf)
    gs = _bid_node(s._cont_env)
    bid, rec = asyncio.run(s.search(gs, 0))
    assert bid in (2, 5, 8)
    assert rec.parse_method == "pikl" and rec.legal_valid and not rec.default_used
    assert rec.final_bid == bid


def test_search_finite_lambda_audits_candidates():
    s = _searcher(lam=0.1, m_worlds=3, n_tau=6)
    gs = _bid_node(s._cont_env)
    bid, rec = asyncio.run(s.search(gs, 0))
    payload = json.loads(rec.raw_response)["pikl"]
    keys = {int(k) for k in payload["Q"]}
    assert keys == {2, 5, 8} == {int(k) for k in payload["tau"]}
    assert math.isclose(sum(payload["pi"].values()), 1.0, abs_tol=1e-3)
    assert payload["metrics"]["n_candidates"] == 3
    assert "tau_eff_n" in payload["metrics"]
    assert "q_best_lift_vs_tau" in payload["metrics"]
    assert "kl_pi_tau" in payload["metrics"]
    assert bid in keys


def test_search_lambda_zero_is_argmax_q():
    s = _searcher(lam=0, m_worlds=3, n_tau=6)
    gs = _bid_node(s._cont_env)
    bid, rec = asyncio.run(s.search(gs, 0))
    payload = json.loads(rec.raw_response)["pikl"]
    # λ=0 still selects argmax-Q and never anchors (no π), but τ̂ IS now recorded
    # so the value-lift diagnostics are computed for the acted-on node (it was
    # already sampled in _candidates — recording it is free, changes no action).
    assert payload["lambda"] == 0 and "pi" not in payload
    assert "tau" in payload
    assert "q_best_lift_vs_tau" in payload["metrics"]
    assert payload["Q"][str(bid)] >= max(payload["Q"].values()) - 1e-4  # argmax Q


# --- 3. recursion guard ------------------------------------------------------

def test_recursion_guard():
    s = _searcher(lam=0.1)
    s._in_search = True
    gs = _bid_node(s._cont_env)
    with pytest.raises(AssertionError):
        asyncio.run(s.search(gs, 0))


# --- 4. leak-safety is structural -------------------------------------------

def test_continuation_uses_only_blueprint():
    s = _searcher(lam=0.1)
    assert {id(c) for c in s._cont_clients} == {id(s.bp_client)}
    assert set(s._cont_models) == {s.bp_model}
    assert "trainable" not in s._cont_actor_ids


# --- 5. diagnostics (var-screen) --------------------------------------------

def test_diagnose_records_belief_and_oracle():
    s = _searcher(lam=0.1, m_worlds=4, n_tau=6)
    gs = _bid_node(s._cont_env)
    d = asyncio.run(s.diagnose_node(gs, 0))
    assert set(d["Q_belief"]) == set(d["Q_oracle"]) == set(d["tau"]) == {2, 5, 8}
    assert d["var_belief"] >= 0 and d["var_oracle"] >= 0
    assert set(d["var_belief_M"]) == {k for k in (8, 16, 24) if k <= 4}  # only M-prefixes ≤ m_worlds


def test_diagnostics_hook_keeps_blueprint_play():
    s = _searcher(lam=0.1, m_worlds=2, n_tau=6)
    s.diagnostics = []
    gs = _bid_node(s._cont_env)
    bid, rec = asyncio.run(s.search(gs, 0))
    assert bid in (2, 5, 8)
    assert len(s.diagnostics) == 1
    assert "Q" not in json.loads(rec.raw_response)["pikl"]  # play stayed blueprint (sampled τ̂)


# --- 6. env hook replaces the trainable seat's bid ---------------------------

def test_bidding_phase_hook_replaces_trainable_bid():
    s = _searcher(lam=math.inf, n_tau=6)
    env = MegaGemEnv(num_players=3, value_chart_id="A", seed=1)
    gs = _bid_node(env)
    bids, recs = asyncio.run(env.run_bidding_phase(
        gs, [s.bp_client] * 3, ["bp"] * 3, {"temperature": 0.0}, pikl_bid_searcher=s))
    assert recs[0].parse_method == "pikl" and bids[0] in (2, 5, 8)  # seat 0 searched
    assert recs[1].parse_method != "pikl" and recs[2].parse_method != "pikl"  # opponents untouched


def test_value_aware_bid_q_is_deterministic_and_records():
    s = _searcher(lam=0.1, m_worlds=4, n_tau=6)
    s.value_aware = True
    gs = _bid_node(env=MegaGemEnv(num_players=3, value_chart_id="A", seed=1))
    d1 = asyncio.run(s.diagnose_node(gs, 0))
    d2 = asyncio.run(s.diagnose_node(gs, 0))
    assert d1["Q_belief"] == d2["Q_belief"]  # heuristic continuation has no LLM/RNG noise
    assert set(d1["Q_belief"]) == set(d1["Q_oracle"]) == set(d1["tau"])
    assert d1["var_belief"] >= 0


def test_treasure_only_skips_non_treasure_auction():
    s = _searcher(lam=math.inf, n_tau=6)
    s.treasure_only = True
    env = MegaGemEnv(num_players=3, value_chart_id="A", seed=1)
    gs = env.create_game_state(seed=5)
    gs.current_auction = AuctionCard(id="L", type=AuctionType.LOAN, amount=10)
    bids, recs = asyncio.run(env.run_bidding_phase(
        gs, [s.bp_client] * 3, ["bp"] * 3, {"temperature": 0.0}, pikl_bid_searcher=s))
    assert recs[0].parse_method != "pikl"  # non-treasure auction skipped under treasure_only


# --- 7. fail-fast sampling + degenerate-node hardening (post-review fixes) ----

class _RaisingClient:
    """A client whose completion call always raises a chosen exception — to test
    get_player_samples' transient-vs-non-transient policy."""

    def __init__(self, exc):
        async def _create(**kw):
            raise exc
        self.chat = type("Chat", (), {
            "completions": type("Cmp", (), {"create": staticmethod(_create)})()})()


def test_samples_nontransient_error_raises_not_silent():
    # A non-transient error (e.g. a backend rejecting the `n` param) must RAISE so
    # the sweep aborts rather than silently degrade τ̂ to uniform on every node.
    env = MegaGemEnv(num_players=3, value_chart_id="A", seed=1)
    client = _RaisingClient(ValueError("backend rejected n"))
    with pytest.raises(ValueError):
        asyncio.run(env.get_player_samples(
            client, "bp", "p", n=4, temperature=1.0, player_id=0))


def test_samples_transient_error_returns_empty():
    import httpx
    from openai import APIConnectionError
    env = MegaGemEnv(num_players=3, value_chart_id="A", seed=1)
    client = _RaisingClient(APIConnectionError(request=httpx.Request("POST", "http://x")))
    out = asyncio.run(env.get_player_samples(
        client, "bp", "p", n=4, temperature=1.0, player_id=0))
    assert out == []  # transient ⇒ caller falls back (τ̂→uniform), no raise


def test_diagnose_m1_skips_crossfit_no_crash():
    # m_worlds=1 ⇒ the odd cross-fit split is empty; must not crash, must omit
    # regret_crossfit and flag the degenerate-M screen.
    s = _searcher(lam=0.1, m_worlds=1, n_tau=6, bids=(2, 5, 8))
    s.value_aware = True
    gs = _bid_node(env=MegaGemEnv(num_players=3, value_chart_id="A", seed=1))
    d = asyncio.run(s.diagnose_node(gs, 0))
    assert "regret_crossfit" not in d
    assert d.get("crossfit_skipped_low_m") is True
    assert d["var_belief"] >= 0


def test_single_candidate_node_recorded_zero_variance():
    # All samples agree on one legal bid ⇒ single-candidate node. With diagnostics
    # on it must be RECORDED (zero variance/regret), not dropped — dropping these
    # zero-headroom nodes upward-biases the var-screen mean.
    s = _searcher(lam=0.1, n_tau=8, bids=(7,))
    s.value_aware = True
    s.diagnostics = []
    gs = _bid_node(s._cont_env)
    bid, rec = asyncio.run(s.search(gs, 0))
    assert bid == 7
    assert len(s.diagnostics) == 1
    d = s.diagnostics[0]
    assert d["single_candidate"] is True
    assert d["var_belief"] == 0.0 and d["regret_insample"] == 0.0 and d["regret_crossfit"] == 0.0


# --- 8. widened bid operator -------------------------------------------------

def test_threshold_candidate_mode_adds_unsampled_neighbors():
    s = _searcher(lam=0.1, n_tau=4, bids=(7,))
    s.candidate_mode = "threshold"
    gs = _bid_node(s._cont_env)
    cands, tau = asyncio.run(s._candidates(generate_bid_prompt(gs, 0), gs, 0))
    assert {5, 6, 7, 8, 9}.issubset(cands)
    assert any(b not in {7} for b in cands)
    assert tau[7] > tau[6]  # sampled support still anchors the prior
    assert math.isclose(sum(tau.values()), 1.0, abs_tol=1e-9)


def test_all_candidate_mode_enumerates_legal_bids():
    s = _searcher(lam=0.1, n_tau=0)
    s.candidate_mode = "all"
    gs = _bid_node(s._cont_env)
    cands, tau = asyncio.run(s._candidates(generate_bid_prompt(gs, 0), gs, 0))
    assert cands == list(range(gs.players[0].coins + 1))
    assert math.isclose(sum(tau.values()), 1.0, abs_tol=1e-9)


def test_confidence_gate_falls_back_to_tau(monkeypatch):
    s = _searcher(lam=0, n_tau=4, bids=(2, 8))
    s.gate_min_lift = 1.0
    gs = _bid_node(s._cont_env)

    async def fake_q_matrix(_game_state, _bidder_id, cands):
        return {b: ([0.0] * 4 if b == 2 else [0.1] * 4) for b in cands}

    monkeypatch.setattr(s, "_q_matrix", fake_q_matrix)
    _bid, rec = asyncio.run(s.search(gs, 0))
    payload = json.loads(rec.raw_response)["pikl"]
    assert payload["gate"]["passed"] is False
    assert payload["metrics"]["n_candidates"] == 2
    assert payload["metrics"]["q_best_lift_vs_tau"] < payload["gate"]["threshold"]
    assert "Q" in payload and "pi" not in payload


def test_confidence_gate_passes_and_best_of_n_selects_q_best(monkeypatch):
    s = _searcher(lam=0, n_tau=4, bids=(2, 8))
    s.gate_min_lift = 0.001
    gs = _bid_node(s._cont_env)

    async def fake_q_matrix(_game_state, _bidder_id, cands):
        return {b: ([0.0] * 4 if b == 2 else [1.0] * 4) for b in cands}

    monkeypatch.setattr(s, "_q_matrix", fake_q_matrix)
    bid, rec = asyncio.run(s.search(gs, 0))
    payload = json.loads(rec.raw_response)["pikl"]
    assert bid == 8
    assert payload["gate"]["passed"] is True
    assert payload["metrics"]["chosen_is_q_best"] is True
    # passed λ=0 gate now records τ̂ ⇒ the acted-on node carries value-lift telemetry
    assert "tau" in payload
    assert payload["metrics"]["chosen_lift_vs_tau"] > 0
    assert payload["metrics"]["q_best_lift_vs_tau"] > 0


# --- 9. intrinsic tie-break consistency (λ=0 metric clarification) ------------

def test_intrinsics_qbest_tiebreak_matches_selector():
    # Exact Q-tie at the top between bids 0 and 9 (all-losing bids share an
    # identical continuation ⇒ identical Q). The live selector is max() over the
    # ascending candidate order, which picks the SMALLEST tied action (0). The
    # generic metric MUST agree so chosen_is_q_best is not a tie-break artifact.
    from megagem.environment.pikl_search import _pikl_intrinsics
    q = {0: 0.5, 5: 0.2, 9: 0.5}
    tau = {0: 0.2, 5: 0.6, 9: 0.2}
    m = _pikl_intrinsics(tau, q, chosen=0)
    assert m["q_best"] == "0"           # smallest action among Q-ties (matches max())
    assert m["q_top2_margin"] == 0.0
    assert m["chosen_is_q_best"] is True
    assert m["chosen_is_q_best_tieadj"] is True


def test_intrinsics_tieadj_credits_the_other_tied_action():
    # If the selector had instead landed on the OTHER member of the Q-tie (9),
    # exact chosen_is_q_best is False but the tie-adjusted flag credits it: both
    # tied actions are value-equivalent argmaxes.
    from megagem.environment.pikl_search import _pikl_intrinsics
    q = {0: 0.5, 5: 0.2, 9: 0.5}
    m = _pikl_intrinsics({0: 0.2, 5: 0.6, 9: 0.2}, q, chosen=9)
    assert m["chosen_is_q_best"] is False
    assert m["chosen_is_q_best_tieadj"] is True
    # a genuinely inferior action is credited by neither flag
    m2 = _pikl_intrinsics({0: 0.2, 5: 0.6, 9: 0.2}, q, chosen=5)
    assert m2["chosen_is_q_best"] is False
    assert m2["chosen_is_q_best_tieadj"] is False


def test_lambda_mix_averages_anchor_strengths():
    s = _searcher(lam=math.inf, n_tau=0)
    s.lambda_mix = (math.inf, 0.0)
    pi = s._pikl_policy({2: 0.75, 8: 0.25}, {2: 0.0, 8: 1.0})
    assert math.isclose(pi[2], 0.375, abs_tol=1e-9)
    assert math.isclose(pi[8], 0.625, abs_tol=1e-9)
