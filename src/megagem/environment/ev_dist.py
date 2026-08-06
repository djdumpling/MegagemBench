"""Analytic expected-surplus bid selector — E1 `ev_dist`.

At each treasure auction the trainable seat bids
    b* = argmax_b  (V-hat − b) · P-hat(win | b)
with
  - P-hat from the frozen opponent bid-distribution artifact
    (packaged asset ``ev_dist_v1.pkl``; see ``megagem.assets``): per-opponent
    gbt point prediction
    + pooled out-of-fold residual PMF, exact tie handling via the round's live
    tiebreak order (src/megagem/environment/bid_model.py — the construction adjudicated
    offline at D = +1.30 coins/decision, t = +16.8, on 1,821 logged decisions);
  - V-hat from the supervised value head (gem value + exact mission bonus),
    consumed ANALYTICALLY — never injected into the prompt, which was the
    measured Phase-2 failure mode (in-context anchoring -> winner's curse);
  - a confidence gate: deviate from the blueprint's sampled bid b_bp only when
    EV(b*) − EV(b_bp) >= gate_min_ev (default 1.0 coin).

Pass-through fidelity (load-bearing for the paired eval): the env pre-samples
the seat's bid via the NORMAL path and hands it in (`wants_presampled`), so
gated-off decisions return the untouched normal record — the treatment arm is
byte-identical to the OFF arm wherever the gate does not fire, and the selector
adds ZERO LLM calls. Per-decision telemetry accumulates in `decision_log`
(drained by ``megagem.rollout.run_game`` into game_data["ev_dist_decisions"]).
"""

from __future__ import annotations

import asyncio
import json

from ..game.actions import get_default_bid, parse_bid, validate_bid_for_auction
from .bid_model import EvDistModel
from .ev_selector import (
    DEFAULT_MODEL_PATH,
    DEFAULT_VALUE_HEAD_PATH,
    EvDistSelector,
)
from .multi_agent_env import BidTurnRecord, DEFAULT_ACTOR_ID
from .pikl_search import PiklBidSearcher
from .prompts import generate_bid_prompt

class EvDistBidSearcher(PiklBidSearcher):
    treasure_only = True       # the artifact + V-hat are treasure objects
    wants_presampled = True    # env hands in the normal-path sample (see module doc)

    def __init__(self, *, model_path: str = DEFAULT_MODEL_PATH,
                 value_head_path: str = DEFAULT_VALUE_HEAD_PATH,
                 gate_min_ev: float = 1.0, use_mission_bonus: bool = True,
                 pacing_lam: float = 0.0, vhat_debias: float = 0.0,
                 value_refit_path: str = "", pacing_schedule: str = "",
                 ev_model: EvDistModel | None = None, value_est=None, **kw):
        kw.setdefault("lam", float("inf"))  # unused; satisfies the base ctor
        super().__init__(**kw)
        self.selector = EvDistSelector(
            model_path=model_path,
            value_head_path=value_head_path,
            gate_min_ev=gate_min_ev,
            use_mission_bonus=use_mission_bonus,
            pacing_lam=pacing_lam,
            vhat_debias=vhat_debias,
            value_refit_path=value_refit_path,
            pacing_schedule=pacing_schedule,
            ev_model=ev_model,
            value_est=value_est,
        )
        # Compatibility aliases for diagnostics and existing callers.
        self.ev_model = self.selector.ev_model
        self.value_est = self.selector.value_est
        self.gate_min_ev = self.selector.gate_min_ev
        self.use_mission_bonus = self.selector.use_mission_bonus
        self.pacing_lam = self.selector.pacing_lam
        self.vhat_debias = self.selector.vhat_debias
        self.pacing_schedule = self.selector.pacing_schedule
        self.value_refit = self.selector.value_refit
        self.decision_log = self.selector.decision_log

    async def _sample_blueprint(self, game_state, bidder_id):
        """Standalone fallback (tests / direct use): one normal-path-equivalent
        blueprint sample. In live games the env pre-samples instead."""
        prompt = generate_bid_prompt(game_state, bidder_id)
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_parallel)
        async with self._sem:
            resp = await self._cont_env.get_player_response(
                self.bp_client, self.bp_model, prompt,
                {"temperature": self.temperature}, player_id=bidder_id)
        p = parse_bid(resp)
        ok = bool(p.valid and validate_bid_for_auction(game_state, bidder_id, p.bid)[0])
        b_bp = int(p.bid) if ok else get_default_bid()
        rec = BidTurnRecord(
            player_id=bidder_id, actor_id=DEFAULT_ACTOR_ID, prompt=prompt,
            raw_response=resp or "", parsed_action=(int(p.bid) if p.valid else None),
            parse_method=p.parse_method, parse_valid=bool(p.valid), legal_valid=ok,
            default_used=not ok, final_bid=b_bp, reasoning=p.reasoning or "",
            length_split={}, parse_error=p.error or "", legal_error="")
        return b_bp, rec

    # ------------------------------------------------------------------ #
    async def search(self, game_state, bidder_id, presampled=None):
        b_bp, rec_bp = presampled if presampled is not None \
            else await self._sample_blueprint(game_state, bidder_id)
        b_bp = int(b_bp)

        chosen, payload = self.selector.select(game_state, bidder_id, b_bp)
        if chosen == b_bp:
            return b_bp, rec_bp
        rec = BidTurnRecord(
            player_id=bidder_id, actor_id=DEFAULT_ACTOR_ID, prompt=rec_bp.prompt,
            raw_response=json.dumps({"pikl": payload}), parsed_action=chosen,
            parse_method="pikl", parse_valid=True, legal_valid=True, default_used=False,
            final_bid=chosen, reasoning="", length_split={}, parse_error="",
            legal_error="")
        return chosen, rec
