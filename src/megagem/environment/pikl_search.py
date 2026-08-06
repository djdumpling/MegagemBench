"""Depth-1 piKL search (test-time policy) for reveals and bids.

Both searchers pick an action from ``π_piKL(a) ∝ τ̂(a)·exp(Q(a)/λ)`` where:
  * τ̂ — the blueprint's action marginal, estimated by SAMPLING N chain-of-thought
    completions and counting parsed actions (a CoT marginal, not a single-token
    logprob), Laplace-smoothed over the legal/candidate actions. τ̂ is
    VALID-CONDITIONED (unparsed/illegal samples dropped); the exact deployed-blueprint
    control is the sweep's λ=∞ point run with piKL OFF.
  * Q  — the information-state value, the belief-averaged terminal utility
    ``E_world[tanh(margin/19.6)]`` over M redeals consistent with the acting seat's
    information set (own hand + public state fixed; opponents' hands + future deck
    resampled). Every world is continued to terminal with all seats playing the local
    blueprint greedily — the search never queries the true held-out opponent, and the
    realized hidden world never leaks into the choice.
  * λ  — anchor strength. λ→∞ ⇒ blueprint; λ→0 ⇒ argmax-Q (best-of-N).

REVEAL (``PiklRevealSearcher``): the action is the gem colour revealed from hand
(≤5 enumerable). The reveal has no direct terminal-scoring channel (every hand gem
reaches the display at game end regardless), so its belief-Var is ~0 — see the
2026-06-01 var-screen finding in the design doc.

BID (``PiklBidSearcher``): the action is the integer bid. By default candidates are
the support of N blueprint bid samples (progressive widening). Opt-in wider modes
add deterministic threshold bids around sampled/market/fair-value anchors or enumerate
all legal bids. Q injects the candidate bid, lets the opponents re-bid via the
blueprint in each belief world, resolves the auction, and continues to terminal — a
DIRECT scoring channel (you win the gem and pay your bid), so bids are the decisive
operator surface.
"""

import asyncio
import json
import math
import random
from collections import Counter
from statistics import fmean, median, pstdev

from ..game.actions import (
    get_default_bid, get_default_reveal, parse_bid, parse_reveal, validate_bid_for_auction,
)
from ..game.cards import AuctionType, GemCard, GemColor
from ..game.rules import (
    apply_auction_outcome, determine_winner, resolve_auction, reveal_gem_from_hand,
)
from ..game.state import GameState
from .multi_agent_env import BidTurnRecord, DEFAULT_ACTOR_ID, MegaGemEnv, RevealTurnRecord
from .prompts import generate_bid_prompt, generate_reveal_prompt

TANH_SCALE = 19.6  # locked terminal-reward scale; Q is utility = tanh(margin/19.6)
Q_TIE_EPS = 1e-9   # Q within this of the best counts as a tie (argmax-set membership)


def redeal_consistent(state: GameState, acting_seat: int, rng: random.Random) -> GameState:
    """A world drawn from the uniform constraint-consistent belief b̂(·|I): keep
    the acting seat's hand and all public state fixed; randomly re-partition the
    unseen gems (opponents' hands + remaining deck) by public hand-counts and
    reshuffle the hidden future order of the gem and auction decks.

    The hidden multisets are CANONICALIZED (sorted) before shuffling, so the
    sampled world is a function of only (information set, rng) — two true states
    that share the acting seat's info set but differ in hidden ordering/partition
    produce the same particle under the same rng (finite-M info-set invariance)."""
    gs = state.copy()
    opponents = [p for p in gs.players if p.player_id != acting_seat]
    pool = sorted([c for p in opponents for c in p.hand] + [g.color.value for g in gs.gem_deck])
    rng.shuffle(pool)
    i = 0
    for p in opponents:
        k = len(p.hand)
        p.hand = pool[i:i + k]
        i += k
    gs.gem_deck = [GemCard(id=f"redeal-{j}", color=GemColor(c)) for j, c in enumerate(pool[i:])]
    gs.auction_deck = sorted(gs.auction_deck, key=lambda a: a.id)
    rng.shuffle(gs.auction_deck)
    return gs


def _starting_coins(num_players: int) -> int:
    """Per-player opening stake (mirrors GameState.create_new_game)."""
    return 35 if num_players == 3 else 25 if num_players == 4 else 20


def _apply_coin_effect(coins: dict, r) -> None:
    """Apply auction ``r``'s coin change to its winner (forward coin reconstruction).
    Treasure/Investment: winner pays its winning bid; Loan: receives amount − bid."""
    w = r.winner_id
    if w not in coins:
        return
    card = r.auction_card or {}
    if card.get("type") == "loan":
        coins[w] += int(card.get("amount", 0)) - r.winning_bid
    else:  # treasure / investment — pay your own winning bid
        coins[w] -= r.winning_bid


def _behavior_logweight(world: GameState, acting_seat: int, *, use_bids: bool,
                        use_reveals: bool, sigma: float, shade: float, gamma: float,
                        window: int = 0, coin_cap: bool = False) -> float:
    """log P(observed opponent behaviour | this world's hypothesised opponent hands).

    The uniform belief ``redeal_consistent`` already excludes the acting seat's hand,
    the value-display, and the in-play gems (they are fixed public state, never in the
    re-partitioned pool) — point 1 of the design. This adds the BEHAVIOURAL signal:

      * BIDS (point 2): under the fair-value model an opponent's bid on a treasure is
        ``min(coins, shade·Σ_{c∈gems} value(disp_c + hand_c))`` — increasing in how much
        of the auctioned colours they hold. We score each world by a Gaussian likelihood
        on the residual between the observed bid and the bid the hypothesised hand would
        have justified, with the historical display, the hand-at-that-time, and (when
        ``coin_cap``) the coins-at-that-time all reconstructed from ``auction_history``.
      * REVEALS (point 3): if opponents reveal greedily, a world where the revealer held
        more of the revealed colour AT REVEAL TIME (``hand_at``) is more likely — a weak
        +γ·count bonus. The rollout self-model reveals ALPHABETICALLY not greedily, so
        this is the most speculative channel (kept separable for ablation).

    ``window`` (>0) restricts the behavioural conditioning to the last ``window`` auctions
    — matching the model's observation window (prompts show only the last 5) — while still
    reconstructing display/hand/coins over the FULL history (cumulative public state). 0 =
    full history (the original behaviour). ``coin_cap`` adds the fair-value coin cap to the
    predicted bid. Both off by default ⇒ byte-identical to the pre-correction sampler.

    Returns 0.0 (⇒ uniform weight) when there is no history or no enabled channel fires.

    CAVEAT that remains: the likelihood assumes opponents bid ~fair-value (sign flips on
    chart D; misspecified if the actual policy isn't value-coherent → can hurt)."""
    history = world.auction_history
    if not history or not (use_bids or use_reveals):
        return 0.0
    chart = world.value_chart
    opp = {p.player_id for p in world.players if p.player_id != acting_seat}
    # hand_at[s] = s's hypothesised hand WHILE BIDDING/REVEALING earlier = its current hand
    # plus every gem it reveals across history (each reveal left the hand only after a win).
    # Built up front, decremented as we pass each reveal, so at auction t it holds exactly
    # the gems s still had at t. Used by BOTH the bid and the reveal channels.
    hand_at = {s: Counter(world.players[s].hand) for s in opp}
    for r in history:
        if r.gem_revealed and r.winner_id in hand_at:
            hand_at[r.winner_id][r.gem_revealed] += 1
    coins_at = {s: float(_starting_coins(world.num_players)) for s in opp} if coin_cap else None
    n = len(history)
    lo = 0 if window <= 0 else max(0, n - window)
    disp: Counter = Counter()
    lw = 0.0
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for idx, r in enumerate(history):
        in_window = idx >= lo
        gems = r.gems_won  # auctioned colours (winner took all) → nonempty iff TREASURE
        if use_bids and gems and in_window:
            for s in opp:
                if s < len(r.bids):
                    pred = shade * sum(chart.get_gem_value(disp[c] + hand_at[s][c]) for c in gems)
                    if coin_cap:
                        pred = min(max(coins_at[s], 0.0), pred)
                    lw -= inv2s2 * (r.bids[s] - pred) ** 2
        if r.gem_revealed:
            if use_reveals and r.winner_id in opp and in_window:
                lw += gamma * hand_at[r.winner_id][r.gem_revealed]  # count AT reveal time
            disp[r.gem_revealed] += 1
            if r.winner_id in hand_at and hand_at[r.winner_id][r.gem_revealed] > 0:
                hand_at[r.winner_id][r.gem_revealed] -= 1  # this gem has now left the hand
        if coin_cap:
            _apply_coin_effect(coins_at, r)
    return lw


def _resample_wor(pool: list, logw: list[float], m: int, rng: random.Random) -> list:
    """Efraimidis–Spirakis weighted sampling WITHOUT replacement: ``key_i = u_i^(1/w_i)``
    (u_i seeded uniform), keep the m largest keys. Returns m DISTINCT worlds, so the Q
    average never collapses below m even under a peaked belief (a guard against effective-
    sample-size collapse that with-replacement SIR would suffer). Seeded ⇒ CRN-reproducible."""
    if m >= len(pool):
        return list(pool)
    hi = max(logw)
    keyed = []
    for lw, item in zip(logw, pool):
        w = math.exp(lw - hi)              # normalised to (0, 1], best world → 1
        keyed.append((rng.random() ** (1.0 / max(w, 1e-12)), item))
    keyed.sort(key=lambda t: t[0], reverse=True)  # sort on the float key only (worlds uncomparable)
    return [item for _k, item in keyed[:m]]


class _PiklSearcher:
    """Shared infrastructure: belief worlds (CRN), blueprint sampling, the piKL
    softmax, and the leak-safe greedy self-model continuation env."""

    fv_shade = 0.8        # fair-value bid shade for the value-aware heuristic continuation
    value_aware = False   # continue with fair-value opponents (CPU heuristic) vs blueprint
    # Belief over the M determinization worlds. "uniform" = the constraint-consistent
    # uniform redeal (byte-identical default). "bid"/"reveal"/"both" import the behavioural
    # signal via likelihood-weighted resampling (SIR) — see _behavior_logweight.
    belief = "uniform"
    belief_sigma = 6.0       # bid-residual noise scale (coins) for the Gaussian likelihood
    belief_gamma = 0.5       # greedy-reveal affinity bonus per held gem of a revealed colour
    belief_oversample = 6    # candidate pool = belief_oversample × M before resampling to M
    belief_window = 0        # restrict behavioural conditioning to last N auctions (0 = full)
    belief_coin_cap = False  # apply the fair-value coin cap to the predicted bid
    # Rollout/live opponent bid model: "fair_value" (default, byte-identical) | "market".
    # "market" anchors to recent winning bids + budget + tiebreak (matches how the blueprint
    # actually bids; offline-fit on SFT teacher bids — see market_anchored_bid).
    opp_model = "fair_value"
    _market_params: dict | None = None  # overrides MARKET_DEFAULTS when set (megagem.rollout wires it)

    def __init__(
        self, *, bp_client, bp_model: str, trainable_seat: int, num_players: int,
        value_chart_id: str, bp_extra: dict, lam: float, n_tau: int = 16,
        temperature: float = 1.0, alpha: float = 0.5, m_worlds: int = 8,
        crn_seed: int = 0, max_parallel: int = 4,
    ):
        self.bp_client = bp_client
        self.bp_model = bp_model
        self.trainable_seat = trainable_seat
        self.num_players = num_players
        self.lam = lam
        self.n_tau = n_tau
        self.temperature = temperature
        self.alpha = alpha
        self.m_worlds = m_worlds
        self.crn_seed = crn_seed
        self.max_parallel = max_parallel
        self._sem: asyncio.Semaphore | None = None
        self._in_search = False
        self.diagnostics: list | None = None  # C2 var-screen sink; play stays blueprint
        # Dedicated greedy self-model env for every counterfactual continuation:
        # all seats use the local blueprint with its extra params, so the search
        # is leak-safe (never the true opponent) and config-isolated from the
        # live game's mixed per-seat sampling.
        self._cont_env = MegaGemEnv(
            num_players=num_players, value_chart_id=value_chart_id,
            per_seat_api_params=[bp_extra] * num_players,
        )
        self._cont_clients = [bp_client] * num_players
        self._cont_models = [bp_model] * num_players
        self._cont_actor_ids = ["blueprint_selfmodel"] * num_players

    def is_pikl_seat(self, seat: int) -> bool:
        return seat == self.trainable_seat

    def _opp_bid(self, gs, seat) -> int:
        """Opponent bid in a Q rollout. opp_model="fair_value" (default) is byte-identical
        to the original `fair_value_bid(gs, seat, self.trainable_seat, self.fv_shade)`."""
        return opponent_bid(gs, seat, self.trainable_seat, model=self.opp_model,
                            shade=self.fv_shade, market_params=self._market_params)

    def _make_worlds(self, game_state, acting_seat) -> list[GameState]:
        base = (self.crn_seed * 7919 + game_state.round_number) & 0x7FFFFFFF
        if self.belief == "uniform":
            return [redeal_consistent(game_state, acting_seat, random.Random(base * 131 + m))
                    for m in range(self.m_worlds)]
        # Behaviour-informed belief: oversample uniform constraint-consistent worlds, weight
        # each by P(observed opponent bids/reveals | world), and resample M without replacement.
        # Same CRN base ⇒ the pool (and the resample) are reproducible across q-matrix/q-value
        # calls at a node. Downstream (Q, cross-fit, variance) sees M worlds and is untouched.
        k = self.m_worlds * self.belief_oversample
        pool = [redeal_consistent(game_state, acting_seat, random.Random(base * 131 + m))
                for m in range(k)]
        use_bids = self.belief in ("bid", "both")
        use_reveals = self.belief in ("reveal", "both")
        logw = [_behavior_logweight(w, acting_seat, use_bids=use_bids, use_reveals=use_reveals,
                                    sigma=self.belief_sigma, shade=self.fv_shade,
                                    gamma=self.belief_gamma, window=self.belief_window,
                                    coin_cap=self.belief_coin_cap) for w in pool]
        return _resample_wor(pool, logw, self.m_worlds,
                             random.Random((base * 2_654_435_761) & 0x7FFFFFFF))

    async def _samples(self, prompt, player_id, *, n) -> list[str]:
        return await self._cont_env.get_player_samples(
            self.bp_client, self.bp_model, prompt,
            n=n, temperature=self.temperature, player_id=player_id)

    def _pikl(self, tau, q, lam: float | None = None) -> dict:
        lam = self.lam if lam is None else lam
        if math.isinf(lam):
            return dict(tau)
        if lam <= 0:
            best = max(tau, key=lambda c: (q[c], c))
            return {c: 1.0 if c == best else 0.0 for c in tau}
        keys = list(tau)
        logits = [math.log(max(tau[c], 1e-12)) + q[c] / lam for c in keys]
        hi = max(logits)
        exps = [math.exp(x - hi) for x in logits]
        z = sum(exps)
        return {c: e / z for c, e in zip(keys, exps)}

    def _sample(self, pi, round_number):
        r = random.Random(self.crn_seed * 1_000_003 + round_number).random()
        acc = 0.0
        for c, p in pi.items():
            acc += p
            if r <= acc:
                return c
        return next(iter(pi))

    async def _continue_to_margin(self, gs) -> float:
        """Resume a prepared post-auction state to terminal with the greedy
        self-model and return the trainable seat's margin vs the mean of others."""
        env = self._cont_env
        await env._run_game_loop(
            gs, self._cont_clients, self._cont_models, {"temperature": 0.0},
            self._cont_actor_ids,
            player_messages=[[] for _ in range(env.num_players)], game_events=[],
            model_chat_times=[0.0] * env.num_players, round_num=gs.round_number,
            round_callback=None, pikl_searcher=None,
        )
        _w, scores = determine_winner(gs)
        fs = {s["player_id"]: s["final_score"] for s in scores}
        others = [v for k, v in fs.items() if k != self.trainable_seat]
        return fs[self.trainable_seat] - fmean(others)

    def _heuristic_rollout(self, gs) -> float:
        """CPU-only continuation: every seat bids at fair value to terminal. The
        opponents price by the PUBLIC display, so a reveal that moves the display
        moves their bids — the behavioral channel a value-blind self-model misses.
        Deterministic ⇒ no rollout noise (M worlds capture only belief uncertainty)."""
        env = self._cont_env
        while not gs.is_game_over():
            auction = gs.draw_auction_card()
            if auction is None:
                break
            bids = [self._opp_bid(gs, s) for s in range(gs.num_players)]
            outcome = resolve_auction(gs, bids)
            gem_revealed = None
            if auction.type == AuctionType.TREASURE:
                gem_revealed = get_default_reveal(gs.players[outcome.winner_id].hand)
                if gem_revealed:
                    reveal_gem_from_hand(gs, outcome.winner_id, gem_revealed)
            apply_auction_outcome(gs, outcome, bids, gem_revealed)
            if auction.type == AuctionType.TREASURE and gs.available_missions:
                env.run_mission_phase(gs, outcome.winner_id)
            env._maybe_end_game(gs)
        _w, scores = determine_winner(gs)
        fs = {s["player_id"]: s["final_score"] for s in scores}
        others = [v for k, v in fs.items() if k != self.trainable_seat]
        return fs[self.trainable_seat] - fmean(others)


class PiklRevealSearcher(_PiklSearcher):
    async def search(self, game_state, winner_id, outcome, bids):
        assert not self._in_search, "piKL search re-entered (recursion guard)"
        self._in_search = True
        try:
            if self._sem is None:
                self._sem = asyncio.Semaphore(self.max_parallel)
            legal = sorted(set(game_state.players[winner_id].hand))
            prompt = generate_reveal_prompt(game_state, winner_id)
            rnd = game_state.round_number
            if len(legal) == 1:
                if self.diagnostics is not None:  # record zero-variance node, don't drop it
                    self.diagnostics.append(_degenerate_diag(legal[0], rnd))
                return self._record(winner_id, legal[0], prompt)
            if self.diagnostics is not None:
                diag = await self.diagnose_node(game_state, winner_id, outcome, bids)
                self.diagnostics.append(diag)
                return self._record(winner_id, self._sample(diag["tau"], rnd), prompt)
            if self.lam == 0:  # best-of-N: argmax Q, ties → canonical (sorted) order
                q = await self._q_values(game_state, winner_id, outcome, bids, legal)
                return self._record(winner_id, max(legal, key=lambda c: q[c]), prompt, q=q)
            tau = await self._tau_hat(prompt, winner_id, legal)
            if math.isinf(self.lam):  # π = τ (the searcher's blueprint; exact control = piKL off)
                return self._record(winner_id, self._sample(tau, rnd), prompt, tau=tau)
            q = await self._q_values(game_state, winner_id, outcome, bids, legal)
            pi = self._pikl(tau, q)
            return self._record(winner_id, self._sample(pi, rnd), prompt, tau=tau, q=q, pi=pi)
        finally:
            self._in_search = False

    async def diagnose_node(self, game_state, winner_id, outcome, bids) -> dict:
        """C2 pre-screen: τ̂, belief Q and oracle (single true-world) Q per colour,
        with the variance/regret reductions in ``_reduce_diag``."""
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_parallel)
        legal = sorted(set(game_state.players[winner_id].hand))
        prompt = generate_reveal_prompt(game_state, winner_id)
        tau = await self._tau_hat(prompt, winner_id, legal)
        mat = await self._q_matrix(game_state, winner_id, outcome, bids, legal)
        oracle = await asyncio.gather(*[
            self._rollout_one(game_state, c, winner_id, outcome, bids) for c in legal])
        q_oracle = {c: math.tanh(m / TANH_SCALE) for c, m in zip(legal, oracle)}
        return _reduce_diag(tau, mat, q_oracle, game_state.round_number)

    async def _q_matrix(self, game_state, winner_id, outcome, bids, legal) -> dict:
        """Per-colour list of world utilities tanh(margin/19.6) over the shared
        (CRN) belief worlds — the raw matrix the diagnostics reduce."""
        worlds = self._make_worlds(game_state, winner_id)

        async def col(color):
            margins = await asyncio.gather(*[
                self._rollout_one(w, color, winner_id, outcome, bids) for w in worlds])
            return [math.tanh(m / TANH_SCALE) for m in margins]

        return dict(zip(legal, await asyncio.gather(*[col(c) for c in legal])))

    async def _tau_hat(self, prompt, winner_id, legal) -> dict:
        counts: Counter = Counter()
        for s in await self._samples(prompt, winner_id, n=self.n_tau):
            p = parse_reveal(s)
            if p.valid and p.gem_color in legal:
                counts[p.gem_color] += 1
        denom = sum(counts.values()) + self.alpha * len(legal)
        return {c: (counts[c] + self.alpha) / denom for c in legal}

    async def _q_values(self, game_state, winner_id, outcome, bids, legal) -> dict:
        worlds = self._make_worlds(game_state, winner_id)

        async def q(color):
            margins = await asyncio.gather(*[
                self._rollout_one(w, color, winner_id, outcome, bids) for w in worlds])
            return fmean(math.tanh(m / TANH_SCALE) for m in margins)

        return dict(zip(legal, await asyncio.gather(*[q(c) for c in legal])))

    async def _rollout_one(self, world_state, color, winner_id, outcome, bids) -> float:
        gs = world_state.copy()
        reveal_gem_from_hand(gs, winner_id, color)
        apply_auction_outcome(gs, outcome, bids, color)
        if gs.available_missions:
            self._cont_env.run_mission_phase(gs, winner_id)
        self._cont_env._maybe_end_game(gs)
        if self.value_aware:
            return self._heuristic_rollout(gs)  # CPU-only, no LLM ⇒ no semaphore
        async with self._sem:
            return await self._continue_to_margin(gs)

    def _record(self, winner_id, color, prompt, *, tau=None, q=None, pi=None):
        payload = {"lambda": "inf" if math.isinf(self.lam) else self.lam, "chosen": color}
        for name, d in (("tau", tau), ("Q", q), ("pi", pi)):
            if d is not None:
                payload[name] = {k: round(v, 4) for k, v in d.items()}
        metrics = _pikl_intrinsics(tau, q, pi, chosen=color)
        if metrics:
            payload["metrics"] = _round_metrics(metrics)
        rec = RevealTurnRecord(
            player_id=winner_id, actor_id=DEFAULT_ACTOR_ID, prompt=prompt,
            raw_response=json.dumps({"pikl": payload}), parsed_action=color,
            parse_method="pikl", parse_valid=True, legal_valid=True, default_used=False,
            final_reveal=color, reasoning="", length_split={}, parse_error="", legal_error="",
        )
        return color, rec


class PiklBidSearcher(_PiklSearcher):
    # The fair-value edge (win an undervalued gem cheaply) lives in TREASURE
    # auctions; restrict the screen to them (set False to search all auctions).
    treasure_only = False
    # Candidate set for scalar bid search. "sampled" is the legacy support-only
    # mode. "threshold" keeps sampled bids and deterministic bid thresholds around
    # market/fair-value anchors. "all" enumerates every legal integer bid; use a
    # cap for cost if needed.
    candidate_mode = "sampled"
    max_bid_candidates = 0
    # Conservative state gate. If enabled, fall back to tau unless the best-Q bid
    # beats the tau-weighted baseline by at least gate_min_lift + gate_z * SE.
    gate_min_lift = 0.0
    gate_z = 0.0
    # DiL-piKL-style anchor mixture. None keeps the single configured lambda.
    lambda_mix: tuple[float, ...] | None = None

    def wants_auction(self, game_state) -> bool:
        return not self.treasure_only or game_state.current_auction.type == AuctionType.TREASURE

    async def search(self, game_state, bidder_id):
        assert not self._in_search, "piKL bid search re-entered (recursion guard)"
        self._in_search = True
        try:
            if self._sem is None:
                self._sem = asyncio.Semaphore(self.max_parallel)
            prompt = generate_bid_prompt(game_state, bidder_id)
            cands, tau = await self._candidates(prompt, game_state, bidder_id)
            rnd = game_state.round_number
            if len(cands) == 1:
                if self.diagnostics is not None:  # record zero-variance node, don't drop it
                    self.diagnostics.append(_degenerate_diag(cands[0], rnd))
                return self._record(bidder_id, cands[0], prompt, tau=tau)
            if self.diagnostics is not None:
                diag = await self.diagnose_node(game_state, bidder_id, cands=cands, tau=tau)
                self.diagnostics.append(diag)
                return self._record(bidder_id, self._sample(tau, rnd), prompt, tau=tau)
            if self.lam == 0 and not self.lambda_mix:
                mat = await self._q_matrix(game_state, bidder_id, cands)
                q = self._q_means(mat)
                gate = self._confidence_gate(tau, mat)
                if not gate["passed"]:
                    return self._record(
                        bidder_id, self._sample(tau, rnd), prompt, tau=tau, q=q, gate=gate)
                # Record τ̂ on the passed (acted-on) node too: it was already sampled
                # in _candidates, so logging it is free and does not change the action
                # (still argmax-Q), but it makes q_best_lift_vs_tau / chosen_lift_vs_tau
                # reflect the DEVIATING decisions, not just the fallback ones.
                return self._record(
                    bidder_id, max(cands, key=lambda b: q[b]), prompt, tau=tau, q=q, gate=gate)
            if math.isinf(self.lam) and not self.lambda_mix:
                return self._record(bidder_id, self._sample(tau, rnd), prompt, tau=tau)
            mat = await self._q_matrix(game_state, bidder_id, cands)
            q = self._q_means(mat)
            gate = self._confidence_gate(tau, mat)
            if not gate["passed"]:
                return self._record(
                    bidder_id, self._sample(tau, rnd), prompt, tau=tau, q=q, gate=gate)
            pi = self._pikl_policy(tau, q)
            return self._record(
                bidder_id, self._sample(pi, rnd), prompt, tau=tau, q=q, pi=pi, gate=gate)
        finally:
            self._in_search = False

    async def diagnose_node(self, game_state, bidder_id, *, cands=None, tau=None) -> dict:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_parallel)
        if cands is None:
            prompt = generate_bid_prompt(game_state, bidder_id)
            cands, tau = await self._candidates(prompt, game_state, bidder_id)
        mat = await self._q_matrix(game_state, bidder_id, cands)
        q_oracle = await self._oracle_q(game_state, bidder_id, cands)
        return _reduce_diag(tau, mat, q_oracle, game_state.round_number)

    async def _candidates(self, prompt, game_state, bidder_id):
        """Candidate bids and τ̂ for bid piKL.

        Legacy mode uses the legality-filtered support of N blueprint samples.
        Wider modes add deterministic bid thresholds so the scalar auction action
        space is not bottlenecked by what the blueprint happened to sample.
        """
        counts: Counter = Counter()
        if self.n_tau > 0:
            for s in await self._samples(prompt, bidder_id, n=self.n_tau):
                p = parse_bid(s)
                if p.valid and validate_bid_for_auction(game_state, bidder_id, p.bid)[0]:
                    counts[int(p.bid)] += 1
        if not counts and self.candidate_mode == "sampled":
            counts[get_default_bid()] = 1
        cands = self._candidate_set(game_state, bidder_id, counts)
        denom = sum(counts.values()) + self.alpha * len(cands)
        if denom <= 0:
            return cands, {b: 1.0 / len(cands) for b in cands}
        tau = {b: (counts[b] + self.alpha) / denom for b in cands}
        z = sum(tau.values())
        return cands, {b: v / z for b, v in tau.items()}

    def _candidate_set(self, game_state, bidder_id, counts: Counter) -> list[int]:
        mode = self.candidate_mode
        if mode not in ("sampled", "threshold", "all"):
            raise ValueError(f"unknown bid candidate_mode {mode!r}")
        if mode == "sampled":
            candidates = set(counts) or {get_default_bid()}
            anchors = set(candidates)
        elif mode == "all":
            max_bid = self._max_legal_bid(game_state, bidder_id)
            candidates = {
                b for b in range(max_bid + 1)
                if validate_bid_for_auction(game_state, bidder_id, b)[0]
            }
            anchors = self._threshold_anchors(game_state, bidder_id, counts)
        else:
            anchors = self._threshold_anchors(game_state, bidder_id, counts)
            candidates = {
                b for b in anchors | set(counts)
                if validate_bid_for_auction(game_state, bidder_id, int(b))[0]
            }
        if not candidates:
            candidates = {get_default_bid()}
            anchors = set(candidates)
        return self._trim_candidates(candidates, counts, anchors)

    def _threshold_anchors(self, game_state, bidder_id, counts: Counter) -> set[int]:
        max_bid = self._max_legal_bid(game_state, bidder_id)
        anchors: set[int] = {0, max_bid, get_default_bid(), *counts}

        def add_neighborhood(center: int, deltas=(-2, -1, 0, 1, 2)) -> None:
            for d in deltas:
                anchors.add(max(0, min(max_bid, int(center) + d)))

        for b in list(counts) or [get_default_bid()]:
            add_neighborhood(int(b))
        # Include thresholds around the rollout model's own and opponent bids.
        # These are often the bids that flip winner/tiebreak outcomes.
        for seat in range(self.num_players):
            pred = self._opp_bid(game_state, seat)
            add_neighborhood(pred, deltas=(-1, 0, 1, 2))
        return anchors

    def _trim_candidates(self, candidates: set[int], counts: Counter,
                         anchors: set[int]) -> list[int]:
        cap = int(self.max_bid_candidates or 0)
        if cap <= 0 or len(candidates) <= cap:
            return sorted(candidates)

        selected: list[int] = []

        def take(seq) -> None:
            for b in seq:
                if b in candidates and b not in selected:
                    selected.append(b)
                    if len(selected) >= cap:
                        break

        sampled = sorted((b for b in candidates if counts[b] > 0),
                         key=lambda b: (-counts[b], b))
        take(sampled)
        take([0, max(candidates), get_default_bid()])
        take(sorted(b for b in anchors if b in candidates))
        if len(selected) < cap:
            pivots = selected or sampled or sorted(anchors) or [0]
            remaining = [b for b in candidates if b not in selected]
            remaining.sort(key=lambda b: (min(abs(b - p) for p in pivots), b))
            take(remaining)
        return sorted(selected)

    @staticmethod
    def _max_legal_bid(game_state, bidder_id) -> int:
        player = game_state.players[bidder_id]
        auction = game_state.current_auction
        if auction is not None and auction.type == AuctionType.LOAN:
            return int(player.coins + auction.amount)
        return int(player.coins)

    async def _q_matrix(self, game_state, bidder_id, cands) -> dict:
        worlds = self._make_worlds(game_state, bidder_id)
        if self.value_aware:  # CPU-only: fair-value opponents, no LLM continuation
            return {b: [math.tanh(self._bid_heuristic_rollout(w, b, bidder_id) / TANH_SCALE)
                        for w in worlds] for b in cands}
        opp = await asyncio.gather(*(self._opponent_bids(w, bidder_id) for w in worlds))

        async def col(b):  # CRN: shared worlds + opponent bids across candidates
            margins = await asyncio.gather(*(
                self._bid_rollout_one(worlds[i], b, opp[i], bidder_id) for i in range(len(worlds))))
            return [math.tanh(m / TANH_SCALE) for m in margins]

        return dict(zip(cands, await asyncio.gather(*(col(b) for b in cands))))

    @staticmethod
    def _q_means(mat) -> dict:
        return {b: fmean(mat[b]) for b in mat}

    def _pikl_policy(self, tau, q) -> dict:
        if not self.lambda_mix:
            return self._pikl(tau, q)
        acc = {b: 0.0 for b in tau}
        for lam in self.lambda_mix:
            pi = self._pikl(tau, q, lam)
            for b, p in pi.items():
                acc[b] += p
        z = sum(acc.values())
        return {b: v / z for b, v in acc.items()}

    def _confidence_gate(self, tau, mat) -> dict:
        enabled = self.gate_min_lift > 0 or self.gate_z > 0
        q = self._q_means(mat)
        best = max(q, key=q.get)
        bp = sum(tau[b] * q[b] for b in q)
        lift = q[best] - bp
        n = len(next(iter(mat.values()))) if mat else 0
        se = 0.0
        if n > 1:
            diffs = []
            for i in range(n):
                bp_i = sum(tau[b] * mat[b][i] for b in q)
                diffs.append(mat[best][i] - bp_i)
            se = pstdev(diffs) / math.sqrt(n)
        threshold = self.gate_min_lift + self.gate_z * se
        return {
            "enabled": enabled,
            "passed": (not enabled) or lift >= threshold,
            "best": best,
            "lift": lift,
            "se": se,
            "threshold": threshold,
            "min_lift": self.gate_min_lift,
            "z": self.gate_z,
        }

    async def _oracle_q(self, game_state, bidder_id, cands) -> dict:
        if self.value_aware:
            return {b: math.tanh(self._bid_heuristic_rollout(game_state, b, bidder_id) / TANH_SCALE)
                    for b in cands}
        opp = await self._opponent_bids(game_state, bidder_id)  # true-world opponent bids
        margins = await asyncio.gather(*(
            self._bid_rollout_one(game_state, b, opp, bidder_id) for b in cands))
        return {b: math.tanh(m / TANH_SCALE) for b, m in zip(cands, margins)}

    def _bid_heuristic_rollout(self, world_state, b, bidder_id) -> float:
        """Value-aware bid Q (CPU-only): inject candidate bid b into the current
        auction against fair-value opponents, then continue to terminal all-fair-value.
        Tests the friend's edge mechanism — win an undervalued gem cheaply — against
        opponents who actually price by the public fair value."""
        gs = world_state.copy()
        auction = gs.current_auction
        bids = [b if s == bidder_id else self._opp_bid(gs, s)
                for s in range(self.num_players)]
        outcome = resolve_auction(gs, bids)
        gem_revealed = None
        if auction.type == AuctionType.TREASURE:
            gem_revealed = get_default_reveal(gs.players[outcome.winner_id].hand)
            if gem_revealed:
                reveal_gem_from_hand(gs, outcome.winner_id, gem_revealed)
        apply_auction_outcome(gs, outcome, bids, gem_revealed)
        if auction.type == AuctionType.TREASURE and gs.available_missions:
            self._cont_env.run_mission_phase(gs, outcome.winner_id)
        self._cont_env._maybe_end_game(gs)
        return self._heuristic_rollout(gs)

    async def _opponent_bids(self, world, bidder_id) -> dict:
        """Greedy blueprint bids for the non-trainable seats in this world."""
        async def one(seat):
            prompt = generate_bid_prompt(world, seat)
            async with self._sem:
                resp = await self._cont_env.get_player_response(
                    self.bp_client, self.bp_model, prompt, {"temperature": 0.0}, player_id=seat)
            p = parse_bid(resp)
            ok = p.valid and validate_bid_for_auction(world, seat, p.bid)[0]
            return seat, (p.bid if ok else get_default_bid())

        seats = [s for s in range(self.num_players) if s != bidder_id]
        return dict(await asyncio.gather(*(one(s) for s in seats)))

    async def _bid_rollout_one(self, world_state, b, opp_bids, bidder_id) -> float:
        async with self._sem:
            gs = world_state.copy()
            auction = gs.current_auction
            bids = [b if s == bidder_id else opp_bids[s] for s in range(self.num_players)]
            outcome = resolve_auction(gs, bids)
            gem_revealed = None
            if auction.type == AuctionType.TREASURE:  # reveal is ~neutral → default
                gem_revealed = get_default_reveal(gs.players[outcome.winner_id].hand)
                if gem_revealed:
                    reveal_gem_from_hand(gs, outcome.winner_id, gem_revealed)
            apply_auction_outcome(gs, outcome, bids, gem_revealed)
            if auction.type == AuctionType.TREASURE and gs.available_missions:
                self._cont_env.run_mission_phase(gs, outcome.winner_id)
            self._cont_env._maybe_end_game(gs)
            return await self._continue_to_margin(gs)

    def _record(self, bidder_id, bid, prompt, *, tau=None, q=None, pi=None, gate=None):
        payload = {"lambda": "inf" if math.isinf(self.lam) else self.lam, "chosen": bid}
        if self.lambda_mix:
            payload["lambda_mix"] = [
                "inf" if math.isinf(lam) else lam for lam in self.lambda_mix
            ]
        payload["candidate_mode"] = self.candidate_mode
        for name, d in (("tau", tau), ("Q", q), ("pi", pi)):
            if d is not None:
                payload[name] = {str(k): round(v, 4) for k, v in d.items()}
        metrics = _pikl_intrinsics(tau, q, pi, chosen=bid)
        if metrics:
            payload["metrics"] = _round_metrics(metrics)
        if gate is not None and gate.get("enabled"):
            payload["gate"] = {
                "passed": bool(gate["passed"]),
                "best": int(gate["best"]),
                "lift": round(gate["lift"], 5),
                "se": round(gate["se"], 5),
                "threshold": round(gate["threshold"], 5),
                "min_lift": round(gate["min_lift"], 5),
                "z": round(gate["z"], 3),
            }
        rec = BidTurnRecord(
            player_id=bidder_id, actor_id=DEFAULT_ACTOR_ID, prompt=prompt,
            raw_response=json.dumps({"pikl": payload}), parsed_action=bid,
            parse_method="pikl", parse_valid=True, legal_valid=True, default_used=False,
            final_bid=bid, reasoning="", length_split={}, parse_error="", legal_error="",
        )
        return bid, rec


def fair_value_bid(game_state, seat, trainable_seat, shade=0.8) -> int:
    """A fair-value trader (your-friend's model): bid a shade of the auctioned
    gems' value. Opponents price by the PUBLIC display count; the trainable seat
    additionally counts its own hand — its private edge, which a reveal leaks (the
    revealed gem moves hand→display, so opponents see the higher count and the edge
    on that colour closes). Non-treasure auctions → 0. Deterministic."""
    auction = game_state.current_auction
    if auction is None or auction.type != AuctionType.TREASURE:
        return 0
    gems = game_state.revealed_gems[:min(auction.gems, len(game_state.revealed_gems))]
    disp = game_state.get_value_display_counts()
    own = Counter(game_state.players[seat].hand) if seat == trainable_seat else {}
    chart = game_state.value_chart
    value = sum(chart.get_gem_value(disp.get(c, 0) + own.get(c, 0)) for c in gems)
    return max(0, min(game_state.players[seat].coins, round(shade * value)))


# Offline-fit defaults (fit by the retired opp_model_fit probe over 4990 SFT teacher bids): the
# market-anchored model halves the bid-prediction error of fair_value (treasure MAE 2.4 vs
# 4.4) and is nearly sign-balanced (|imbalance| 0.03 vs 0.46) — fair_value is systematically
# too LOW because the blueprint anchors bids to recent winning bids + budget + tiebreak, not
# to fair value (measured in the SFT teacher audit, 2026-06-03).
MARKET_DEFAULTS = {"f": 1.0, "window": 5, "agg": "median", "match_gems": True,
                   "f_floor": 0.25, "tiebreak_delta": 1}
_AGG_FN = {"mean": fmean, "median": median, "max": max, "last": lambda xs: xs[-1]}


def market_anchored_bid(game_state, seat, trainable_seat=None, *, shade=0.8,
                        f=1.0, window=5, agg="median", match_gems=True,
                        f_floor=0.25, tiebreak_delta=1) -> int:
    """Market-anchored opponent: bid the way the SFT blueprint actually bids — anchored to
    recent winning bids of comparable treasure auctions + own budget + tiebreak, NOT to fair
    value (the teacher policy has no fair-value pricing fn; see the SFT audit). Public-info
    only (no hand peek), pure and DETERMINISTIC. Non-treasure → 0 (matches fair_value_bid;
    keeps loan/investment rollout dynamics byte-identical). ``trainable_seat``/``shade`` are
    accepted only for call-site symmetry with fair_value_bid and are ignored. Cold-start (no
    comparable history) → f_floor·coins. ``tiebreak_delta`` bumps the bid when ``seat`` is
    last in tiebreak priority (the dominant teacher signal, present in 90.5% of traces)."""
    auction = game_state.current_auction
    if auction is None or auction.type != AuctionType.TREASURE:
        return 0
    coins = game_state.players[seat].coins
    # treasure ⟺ gems_won nonempty (winner takes all auctioned gems); len(gems_won) = gem count
    A = [r.winning_bid for r in game_state.auction_history[-window:]
         if r.gems_won and (not match_gems or len(r.gems_won) == auction.gems)]
    pred = (f_floor * coins) if not A else (f * _AGG_FN[agg](A))
    if tiebreak_delta and game_state.tiebreak_order and game_state.tiebreak_order[-1] == seat:
        pred += tiebreak_delta  # last in tiebreak must outbid to win → nudge up
    return max(0, min(coins, round(pred)))


def opponent_bid(game_state, seat, trainable_seat, *, model="fair_value", shade=0.8,
                 market_params=None) -> int:
    """Single source of truth for the opponent bid model — used in BOTH the piKL Q rollout
    and the live-env fv_opponent_seats injection so they never diverge. model="fair_value"
    (default) is byte-identical to calling fair_value_bid directly."""
    if model == "market":
        return market_anchored_bid(game_state, seat, trainable_seat, shade=shade,
                                   **(market_params or MARKET_DEFAULTS))
    return fair_value_bid(game_state, seat, trainable_seat, shade)


def _fmt_action(a) -> str:
    return str(a)


def _action_key(a):
    try:
        return (0, float(a))
    except (TypeError, ValueError):
        return (1, str(a))


def _entropy(dist: dict) -> float:
    return -sum(float(p) * math.log(max(float(p), 1e-12))
                for p in dist.values() if float(p) > 0)


def _top_action(d: dict):
    return max(d, key=lambda a: (d[a], _action_key(a)))


def _dist_metrics(dist: dict, prefix: str) -> dict:
    if not dist:
        return {}
    ent = _entropy(dist)
    n = len(dist)
    mode = _top_action(dist)
    return {
        f"{prefix}_entropy": ent,
        f"{prefix}_entropy_norm": ent / math.log(n) if n > 1 else 0.0,
        f"{prefix}_eff_n": math.exp(ent),
        f"{prefix}_mode": _fmt_action(mode),
        f"{prefix}_top_prob": float(dist[mode]),
    }


def _pikl_intrinsics(tau=None, q=None, pi=None, chosen=None, q_world_sd: float | None = None) -> dict:
    """Compact per-node piKL telemetry.

    The metrics separate exploration/diversity (candidate count, entropy), value
    headroom (Q spread/lift), and actual policy movement (KL/TV/action changes).
    They are intentionally scalar so Gate-B summaries can be read without opening
    every raw trajectory.
    """
    tau = dict(tau or {})
    q = dict(q or {})
    pi = dict(pi or {})
    out: dict = {"n_candidates": len(tau or q or pi)}
    if tau:
        out.update(_dist_metrics(tau, "tau"))
    if pi:
        out.update(_dist_metrics(pi, "pi"))
    if tau and pi:
        keys = set(tau) | set(pi)
        out["kl_pi_tau"] = sum(
            float(pi.get(a, 0.0)) * math.log(
                max(float(pi.get(a, 0.0)), 1e-12) / max(float(tau.get(a, 0.0)), 1e-12))
            for a in keys if float(pi.get(a, 0.0)) > 0)
        out["tv_pi_tau"] = 0.5 * sum(
            abs(float(pi.get(a, 0.0)) - float(tau.get(a, 0.0))) for a in keys)
        out["pi_mode_is_tau_mode"] = out.get("pi_mode") == out.get("tau_mode")
    if q:
        # Tie-break IDENTICALLY to the live selector: max() over the ascending
        # candidate order picks the SMALLEST tied action, so order by descending
        # Q then ascending action_key. Using reverse=True here instead would pick
        # the largest tied action and make chosen_is_q_best a tie-break artifact
        # (exact Q-ties are common — all-losing bids share one continuation).
        ordered = sorted(q, key=lambda a: (-float(q[a]), _action_key(a)))
        best = ordered[0]
        worst = min(q, key=lambda a: (q[a], _action_key(a)))
        bp = sum(float(tau.get(a, 0.0)) * float(q[a]) for a in q) if tau else None
        top2 = float(q[best]) - float(q[ordered[1]]) if len(ordered) > 1 else 0.0
        spread = float(q[best]) - float(q[worst])
        out.update({
            "q_best": _fmt_action(best),
            "q_worst": _fmt_action(worst),
            "q_spread": spread,
            "q_top2_margin": top2,
        })
        if q_world_sd is not None:
            out["q_world_sd"] = float(q_world_sd)
            out["q_noise_to_spread"] = (
                float(q_world_sd) / spread if spread > 1e-12 else None)
        if bp is not None:
            out.update({
                "q_tau_value": bp,
                "q_best_lift_vs_tau": float(q[best]) - bp,
                "q_best_tau_prob": float(tau.get(best, 0.0)),
                "q_best_is_tau_mode": _fmt_action(best) == out.get("tau_mode"),
            })
        if chosen is not None and chosen in q:
            out["chosen"] = _fmt_action(chosen)
            out["chosen_is_q_best"] = chosen == best
            # Tie-adjusted: credit the chosen action whenever its Q is within ε of
            # the best Q (i.e. it is a member of the argmax SET), so value-equivalent
            # tie members are not penalized by an arbitrary tie-break.
            out["chosen_is_q_best_tieadj"] = float(q[chosen]) >= float(q[best]) - Q_TIE_EPS
            if tau:
                out["chosen_is_tau_mode"] = _fmt_action(chosen) == out.get("tau_mode")
                out["chosen_tau_prob"] = float(tau.get(chosen, 0.0))
            if bp is not None:
                out["chosen_lift_vs_tau"] = float(q[chosen]) - bp
        if pi:
            out["pi_mode_is_q_best"] = out.get("pi_mode") == _fmt_action(best)
    return out


def _round_metrics(obj):
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float)):
        return round(float(obj), 5)
    if isinstance(obj, dict):
        return {k: _round_metrics(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_metrics(v) for v in obj]
    return obj


def _weighted_var(tau, q) -> float:
    mean = sum(tau[c] * q[c] for c in q)
    return sum(tau[c] * (q[c] - mean) ** 2 for c in q)


def _uvar(q) -> float:
    """Unweighted variance of Q across actions — the τ̂→uniform ceiling on
    Var_belief, isolating Q-structure from blueprint concentration."""
    vs = list(q.values())
    m = fmean(vs)
    return fmean((x - m) ** 2 for x in vs)


def _degenerate_diag(action, round_number) -> dict:
    """Diagnostic record for a node with a SINGLE legal action (one hand colour /
    one candidate bid). Variance and both regrets are identically 0 — there is no
    alternative to switch to — so no rollout is needed. Recording it (instead of
    silently returning early) keeps these zero-headroom nodes IN the var-screen
    mean; dropping them would upward-bias the average headroom estimate."""
    return {
        "round": round_number, "legal": [action], "tau": {action: 1.0},
        "Q_belief": {action: 0.0}, "Q_oracle": {action: 0.0},
        "var_belief": 0.0, "var_oracle": 0.0, "var_belief_M": {}, "uvar_belief_M": {},
        "q_world_sd": 0.0, "regret_insample": 0.0, "regret_crossfit": 0.0,
        "regret_vs_oracle": 0.0, "oracle_agree": 1.0,
        "single_candidate": True,
        "intrinsics": _pikl_intrinsics({action: 1.0}, {action: 0.0}, chosen=action),
    }


def _reduce_diag(tau, mat, q_oracle, round_number) -> dict:
    """Shared C2 reductions over the per-action world-utility matrix ``mat``:
    τ̂-weighted belief/oracle variance (and the clairvoyance gap between them);
    ``var/uvar_belief_M`` at nested M-prefixes (does belief-Var shrink ~1/M toward
    0 ⇒ estimation noise?); ``q_world_sd`` (world-to-world noise floor); and the
    regret of the belief-best action over the blueprint, in-sample (winner's-curse
    biased) and CROSS-FIT (select on even worlds, score on the disjoint odd ones)."""
    legal = list(tau)
    m_max = len(next(iter(mat.values())))
    q_belief = {c: fmean(mat[c]) for c in legal}
    prefixes = [k for k in (8, 12, 16, 24, 32, 48) if k <= m_max]
    q_at = lambda k: {c: fmean(mat[c][:k]) for c in legal}  # noqa: E731

    def bp_value(qd):  # blueprint = sample τ̂ ⇒ value is the τ̂-weighted mean Q
        return sum(tau[c] * qd[c] for c in legal)

    out = {
        "round": round_number, "legal": legal, "tau": tau,
        "Q_belief": q_belief, "Q_oracle": q_oracle,
        "var_belief": _weighted_var(tau, q_belief),
        "var_oracle": _weighted_var(tau, q_oracle),
        "var_belief_M": {k: _weighted_var(tau, q_at(k)) for k in prefixes},
        "uvar_belief_M": {k: _uvar(q_at(k)) for k in prefixes},
        "q_world_sd": fmean(pstdev(mat[c]) for c in legal) if m_max > 1 else 0.0,
        "regret_insample": max(q_belief.values()) - bp_value(q_belief),
    }
    # Belief FIDELITY: pick the action the belief says is best, then score it on the TRUE
    # world's oracle Q. ``regret_vs_oracle`` ≥ 0 is how much that choice underperforms the
    # true-best action (lower = the belief better tracks reality); ``oracle_agree`` is the
    # 0/1 hit. This is the metric that says whether a SHARPER belief (informed vs uniform)
    # selects better real-world actions — the cross-fit regret is belief-internal and cannot.
    a_belief = max(q_belief, key=q_belief.get)
    out["regret_vs_oracle"] = max(q_oracle.values()) - q_oracle[a_belief]
    out["oracle_agree"] = 1.0 if a_belief == max(q_oracle, key=q_oracle.get) else 0.0
    out["intrinsics"] = _pikl_intrinsics(
        tau, q_belief, chosen=a_belief, q_world_sd=out["q_world_sd"])
    # Cross-fit needs ≥2 worlds (disjoint even/odd split); with m_worlds<2 the odd
    # set is empty (fmean([]) raises). Omit the key so summarize() skips this node
    # rather than crash — but flag it so a degenerate-M screen is visible.
    if m_max >= 2:
        sel = {c: fmean(mat[c][0::2]) for c in legal}  # select set (even worlds)
        ev = {c: fmean(mat[c][1::2]) for c in legal}   # disjoint eval set (odd worlds)
        out["regret_crossfit"] = ev[max(sel, key=sel.get)] - bp_value(ev)
    else:
        out["crossfit_skipped_low_m"] = True
    return out
