"""Lightweight runtime core for the dynamics-aware EV bid selector.

Unlike ``ev_dist.py``, this module has no dependency on the training/evaluation
environment (datasets/verifiers). It is therefore safe to use in the small
interactive Modal image while sharing the exact decision rule used by evals.
"""

from __future__ import annotations

import pickle
from collections import Counter

import numpy as np

from ..assets import asset_path
from ..value_head.value_estimator import ValueEstimator
from .bid_model import EvDistModel, ev_best, node_from_live


DEFAULT_MODEL_PATH = str(asset_path("ev_dist_v1.pkl"))
DEFAULT_VALUE_HEAD_PATH = str(asset_path("value_head.pkl"))


class EvDistSelector:
    """Choose a treasure bid from a pre-sampled weights-only bid."""

    def __init__(
        self,
        *,
        model_path: str = DEFAULT_MODEL_PATH,
        value_head_path: str = DEFAULT_VALUE_HEAD_PATH,
        gate_min_ev: float = 1.0,
        use_mission_bonus: bool = True,
        pacing_lam: float = 0.0,
        vhat_debias: float = 0.0,
        value_refit_path: str = "",
        pacing_schedule: str = "",
        ev_model: EvDistModel | None = None,
        value_est=None,
    ):
        self.ev_model = (
            ev_model if ev_model is not None else EvDistModel.load(model_path)
        )
        self.value_est = (
            value_est
            if value_est is not None
            else ValueEstimator.load(value_head_path)
        )
        self.gate_min_ev = float(gate_min_ev)
        self.use_mission_bonus = bool(use_mission_bonus)
        self.pacing_lam = float(pacing_lam)
        self.vhat_debias = float(vhat_debias)

        from megagem.environment.pacing import parse_schedule

        self.pacing_schedule = parse_schedule(pacing_schedule)
        self.value_refit = None
        if value_refit_path:
            with open(value_refit_path, "rb") as file:
                self.value_refit = pickle.load(file)["model"]
        self.decision_log: list[dict] = []

    def _refit_ghat(self, game_state, seat: int, vhat_gem: float) -> float:
        auction = game_state.current_auction
        lot = list(game_state.revealed_gems[: auction.gems])
        display = game_state.get_value_display_counts()
        hand = Counter(game_state.players[seat].hand)
        collection = list(game_state.players[seat].collection)
        treasures_seen = sum(
            1
            for result in game_state.auction_history
            if (result.auction_card or {}).get("type") == "treasure"
        ) + 1
        lot_counts = Counter(lot)
        unseen = {
            color: max(
                0,
                6
                - int(display.get(color, 0))
                - hand.get(color, 0)
                - lot_counts.get(color, 0),
            )
            for color in set(lot)
        }
        features = [
            float(vhat_gem),
            float(len(lot)),
            float(game_state.round_number),
            float(sum(hand.get(color, 0) for color in set(lot))),
            float(sum(unseen.values())),
            float(min(unseen.values(), default=0)),
            float(max(0, 17 - treasures_seen)),
            float(sum(int(display.get(color, 0)) for color in set(lot))),
            float(sum(collection.count(color) for color in set(lot))),
        ]
        return float(self.value_refit.predict(np.array([features]))[0])

    def _lot_value(self, game_state, seat: int) -> dict:
        auction = game_state.current_auction
        display_counts = game_state.get_value_display_counts()
        own_hand = Counter(game_state.players[seat].hand)
        all_collections: Counter = Counter()
        for player in game_state.players:
            for color, count in player.get_collection_counts().items():
                all_collections[color] += count
        gems = list(game_state.revealed_gems[: auction.gems])
        return self.value_est.marginal_value(
            gems=gems,
            seat_collection_counts=game_state.players[seat].get_collection_counts(),
            available_mission_ids=[
                mission.id for mission in game_state.available_missions
            ],
            display_counts=display_counts,
            own_hand_counts=own_hand,
            collection_counts_all=all_collections,
            round_number=game_state.round_number,
            chart_id=game_state.value_chart.id,
        )

    def select(
        self, game_state, bidder_id: int, blueprint_bid: int
    ) -> tuple[int, dict | None]:
        """Return ``(chosen_bid, telemetry)``; non-treasure turns pass through."""
        blueprint_bid = int(blueprint_bid)
        node = node_from_live(game_state)
        if node is None:
            return blueprint_bid, None

        try:
            chart = game_state.value_chart
            coins = int(node["coins"].get(bidder_id, 0))
            marginal_value = self._lot_value(game_state, bidder_id)
            vhat = float(marginal_value["gem_value"])
            if self.use_mission_bonus:
                vhat += float(marginal_value["mission_bonus"])
            if self.value_refit is not None:
                vhat += self._refit_ghat(
                    game_state,
                    bidder_id,
                    float(marginal_value["gem_value"]),
                )
            else:
                vhat -= self.vhat_debias

            opponent_seats = [seat for seat in node["coins"] if seat != bidder_id]
            mus = {
                seat: self.ev_model.mu(node, seat, chart)
                for seat in opponent_seats
            }
            pwin = self.ev_model.win_curve_for(
                node,
                bidder_id,
                chart,
                opp_seats=opponent_seats,
            )
        except Exception as exc:  # noqa: BLE001
            payload = {
                "selector": "ev_dist",
                "round": node["round"],
                "player_id": bidder_id,
                "error": f"{type(exc).__name__}: {exc}",
                "chosen": blueprint_bid,
                "b_bp": blueprint_bid,
                "gate": {
                    "passed": False,
                    "min_ev": self.gate_min_ev,
                    "margin": 0.0,
                },
            }
            self.decision_log.append(payload)
            return blueprint_bid, payload

        lam_eff = self.pacing_lam
        if self.pacing_schedule is not None:
            from megagem.environment.pacing import pacing_lambda

            lam_eff = pacing_lambda(
                self.pacing_schedule,
                auctions_resolved=len(game_state.auction_history),
                coins=int(game_state.players[bidder_id].coins),
                flat=self.pacing_lam,
            )
        if lam_eff > 0.0:
            bid_grid = np.arange(len(pwin), dtype=float)
            ev = (vhat - (1.0 + lam_eff) * bid_grid) * pwin
            best_bid = int(np.argmax(ev))
            best_ev = float(ev[best_bid])
        else:
            best_bid, best_ev, ev = ev_best(vhat, pwin)

        blueprint_ev = float(ev[min(blueprint_bid, coins)])
        gate_margin = best_ev - blueprint_ev
        deviate = (
            gate_margin >= self.gate_min_ev and best_bid != blueprint_bid
        )
        chosen = int(best_bid if deviate else blueprint_bid)
        payload = {
            "selector": "ev_dist",
            "round": node["round"],
            "player_id": bidder_id,
            "chosen": chosen,
            "b_bp": blueprint_bid,
            "b_star": int(best_bid),
            "vhat": round(vhat, 3),
            "vhat_gem": round(float(marginal_value["gem_value"]), 3),
            "vhat_mission": round(float(marginal_value["mission_bonus"]), 3),
            "ev_star": round(best_ev, 4),
            "ev_bp": round(blueprint_ev, 4),
            "p_win_star": round(float(pwin[best_bid]), 4),
            "p_win_bp": round(float(pwin[min(blueprint_bid, coins)]), 4),
            "mu": {str(seat): int(mu) for seat, mu in mus.items()},
            "coins": coins,
            "lam_eff": round(float(lam_eff), 3),
            "gate": {
                "passed": bool(deviate),
                "min_ev": self.gate_min_ev,
                "margin": round(gate_margin, 4),
            },
        }
        self.decision_log.append(payload)
        return chosen, payload
