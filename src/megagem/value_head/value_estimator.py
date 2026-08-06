"""Probabilistic gem-valuation forecaster (P0 core).

Semantics (verified on 80 games, 2026-06-09): every gem dealt to a HAND ends in the
value display (revealed on a win, or dumped at game end), and gems won at auction go to
COLLECTIONS (never displayed). So the final display count of color c,

    n_c = (# of c dealt into the 15 hand slots),

is fixed by the deal and merely REVEALED over the game (display is monotone
non-decreasing). At a decision the acting seat knows a FLOOR (current display_c + its
own remaining hand_c, both certain to display) and a CEILING (6 - collections_c); the
hidden remainder is opponents' unrevealed hand_c. The head learns that remainder from
the public reveal trajectory. n_c is **chart-independent** -> one head across all charts.

Inference is **numpy-only** (a small MLP whose weights are extracted to plain arrays):
training imports sklearn (local-only), but a saved head loads + predicts with numpy
alone, so it runs inside the pinned vLLM Modal image (numpy<2.3, no scikit-learn).

Leak-safe: features use the acting seat's OWN hand + public board only; opponents' hands
are never read. Every feature is knowable at LIVE decision time (no final-round-count
look-ahead) so the offline-trained head transfers byte-for-byte to live play.
"""

from __future__ import annotations

import os
import pickle

import numpy as np

from megagem.game.cards import Mission, ValueChart

COLORS = ["Red", "Blue", "Green", "Purple", "Yellow"]
TOTAL_PER_COLOR = 6
HAND_TOTAL = 15  # 3 players x 5 hand gems -> total final display count

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")

# Ordered feature names — the single source of truth for vectorization order.
# Every entry is knowable at live decision time (no look-ahead at the final round count).
FEATURES = [
    "disp_c", "own_hand_c", "coll_c",
    "known_floor", "known_ceiling", "slack",
    "accounted", "remaining_unaccounted",
    "total_displayed", "displayed_frac",
    "round_number",
]


def load_charts() -> dict[str, ValueChart]:
    import json
    with open(os.path.join(_DATA, "value_charts.json")) as fh:
        raw = json.load(fh)
    return {cid: ValueChart.from_dict(cid, data) for cid, data in raw.items()}


def load_missions() -> dict[str, Mission]:
    import json
    with open(os.path.join(_DATA, "missions.json")) as fh:
        raw = json.load(fh)
    return {m["id"]: Mission.from_dict(m) for m in raw}


def value_features(*, color, display_counts, own_hand_counts, collection_counts_all,
                   round_number) -> dict:
    """Leak-safe, live-knowable features for predicting color's final display count n_c.

    display_counts / collection_counts_all are PUBLIC (per color). own_hand_counts is the
    ACTING SEAT's own hand only — never an opponent's. round_number is the 1-based current
    round (knowable live; we deliberately do NOT use the final round count).
    """
    disp_c = float(display_counts.get(color, 0))
    own_c = float(own_hand_counts.get(color, 0))
    coll_c = float(collection_counts_all.get(color, 0))
    floor = disp_c + own_c                        # certain to display
    ceiling = TOTAL_PER_COLOR - coll_c            # collected c never display
    accounted = disp_c + own_c + coll_c
    total_disp = float(sum(display_counts.values()))
    return {
        "disp_c": disp_c,
        "own_hand_c": own_c,
        "coll_c": coll_c,
        "known_floor": floor,
        "known_ceiling": ceiling,
        "slack": ceiling - floor,                 # width of the uncertainty range
        "accounted": accounted,
        "remaining_unaccounted": TOTAL_PER_COLOR - accounted,  # opp hand c + unwon deck c
        "total_displayed": total_disp,
        "displayed_frac": total_disp / HAND_TOTAL,  # reveal progress (knowable live)
        "round_number": float(round_number),
    }


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _value_dist(count_dist: dict[int, float], chart: ValueChart) -> dict[float, float]:
    out: dict[float, float] = {}
    for k, p in count_dist.items():
        v = float(chart.get_gem_value(int(k)))
        out[v] = out.get(v, 0.0) + p
    return out


def _quantile(value_dist: dict[float, float], q: float) -> float:
    cum, last = 0.0, 0.0
    for v in sorted(value_dist):
        last = v
        cum += value_dist[v]
        if cum >= q:
            return v
    return last


class ValueEstimator:
    """Numpy-inference MLP count head + chart mapping + exact mission bonus."""

    def __init__(self, charts=None, missions=None):
        self.charts = charts or load_charts()
        self.missions = missions or load_missions()
        self.feature_names = list(FEATURES)
        # numpy inference params (set by fit/load) — NO sklearn objects retained.
        self.scaler_mean = self.scaler_scale = None
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self.classes_: list[int] = []
        self.calib_x = self.calib_y = None        # isotonic breakpoints on E[value]

    # ---- training (sklearn imported locally; not needed to LOAD a saved head) -- #
    def fit(self, X, y, *, seed=0, hidden=(64, 32)):
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        X = np.asarray(X, float)
        sc = StandardScaler().fit(X)
        clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", alpha=1e-3,
                            max_iter=300, random_state=seed).fit(sc.transform(X),
                                                                 np.asarray(y, int))
        self.scaler_mean = sc.mean_.astype(float)
        self.scaler_scale = np.where(sc.scale_ == 0, 1.0, sc.scale_).astype(float)
        self.W = [w.astype(float) for w in clf.coefs_]
        self.b = [bi.astype(float) for bi in clf.intercepts_]
        self.classes_ = [int(c) for c in clf.classes_]
        return self

    def fit_calibration(self, raw_values, realized_values):
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            np.asarray(raw_values, float), np.asarray(realized_values, float))
        self.calib_x = np.asarray(iso.X_thresholds_, float)
        self.calib_y = np.asarray(iso.y_thresholds_, float)
        return self

    # ---- pure-numpy inference -------------------------------------------- #
    def proba_matrix(self, X) -> np.ndarray:
        h = (np.asarray(X, float) - self.scaler_mean) / self.scaler_scale
        for w, bi in zip(self.W[:-1], self.b[:-1]):
            h = np.maximum(h @ w + bi, 0.0)       # relu
        return _softmax(h @ self.W[-1] + self.b[-1])

    def _calibrate(self, v):
        if self.calib_x is None:
            return v
        return np.interp(v, self.calib_x, self.calib_y)

    def _vec(self, feat: dict) -> np.ndarray:
        return np.array([[feat[k] for k in self.feature_names]], float)

    def count_dist(self, feat: dict) -> dict[int, float]:
        p = self.proba_matrix(self._vec(feat))[0]
        return {int(c): float(pi) for c, pi in zip(self.classes_, p)}

    def evalue_raw(self, feat: dict, chart_id: str) -> float:
        chart = self.charts[chart_id]
        return sum(p * chart.get_gem_value(k) for k, p in self.count_dist(feat).items())

    def evalue(self, feat: dict, chart_id: str) -> float:
        return float(self._calibrate(self.evalue_raw(feat, chart_id)))

    def evalues_batch(self, X, chart_ids, *, calibrate=True) -> np.ndarray:
        """E[value] for every row at its game's chart (grouped by chart, vectorized)."""
        P = self.proba_matrix(X)
        chart_ids = np.asarray(chart_ids)
        out = np.empty(P.shape[0])
        for cid in set(chart_ids.tolist()):
            vvec = np.array([self.charts[cid].get_gem_value(k) for k in self.classes_], float)
            m = chart_ids == cid
            out[m] = P[m] @ vvec
        return self._calibrate(out) if calibrate else out

    def argmax_counts_batch(self, X) -> np.ndarray:
        return np.asarray(self.classes_)[self.proba_matrix(X).argmax(1)]

    def value_band(self, feat: dict, chart_id: str, lo=0.1, hi=0.9):
        vd = _value_dist(self.count_dist(feat), self.charts[chart_id])
        return _quantile(vd, lo), _quantile(vd, hi)

    # ---- exact mission bonus (deterministic) ----------------------------- #
    def mission_bonus(self, collection_counts: dict[str, int], candidate_gems,
                      available_mission_ids) -> int:
        base = [c for c, k in collection_counts.items() for _ in range(int(k))]
        after = base + list(candidate_gems)
        bonus = 0
        for mid in available_mission_ids:
            m = self.missions.get(mid)
            if m is None:
                continue
            if m.check_completion(after) and not m.check_completion(base):
                bonus += m.reward
        return bonus

    def marginal_value(self, *, gems, seat_collection_counts, available_mission_ids,
                       display_counts, own_hand_counts, collection_counts_all,
                       round_number, chart_id) -> dict:
        """Predicted marginal value of WINNING a treasure lot to the acting seat:
        sum of per-gem predicted end-game value + EXACT mission bonus the lot enables."""
        gem_value, lo, hi = 0.0, 0.0, 0.0
        for g in gems:
            feat = value_features(
                color=g, display_counts=display_counts, own_hand_counts=own_hand_counts,
                collection_counts_all=collection_counts_all, round_number=round_number)
            gem_value += self.evalue(feat, chart_id)
            b = self.value_band(feat, chart_id)
            lo += b[0]
            hi += b[1]
        mb = self.mission_bonus(seat_collection_counts, gems, available_mission_ids)
        return {"gem_value": gem_value, "mission_bonus": mb,
                "total": gem_value + mb, "band": (lo + mb, hi + mb)}

    # ---- persistence (numpy-only payload; loads without sklearn) ---------- #
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scaler_mean": self.scaler_mean, "scaler_scale": self.scaler_scale,
                         "W": self.W, "b": self.b, "classes_": self.classes_,
                         "calib_x": self.calib_x, "calib_y": self.calib_y,
                         "feature_names": self.feature_names}, f)

    @classmethod
    def load(cls, path):
        est = cls()
        with open(path, "rb") as f:
            d = pickle.load(f)
        est.scaler_mean, est.scaler_scale = d["scaler_mean"], d["scaler_scale"]
        est.W, est.b, est.classes_ = d["W"], d["b"], d["classes_"]
        est.calib_x, est.calib_y = d["calib_x"], d["calib_y"]
        est.feature_names = d["feature_names"]
        return est
