"""Tests for the supervised value head (megagem.value_head).

Covers: the leak firewall (a seat's features never depend on opponents' hands), the
verified semantic bounds (known_floor <= n_c <= known_ceiling on real games), exact
mission bonus via the real check_completion, chart weighting, marginal_value structure,
and save/load.
"""

import glob
import os
from collections import Counter

import numpy as np
import pytest

import megagem.value_head.train as tv
from megagem.value_head.value_estimator import (
    COLORS, FEATURES, ValueEstimator, value_features,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# --------------------------------------------------------------------------- #
# leak firewall                                                                #
# --------------------------------------------------------------------------- #
def _game(models, rounds, vdf):
    g = {"metadata": {"models": models, "value_chart": "A"},
         "rounds": rounds, "final_results": {"value_display_final": vdf}}
    g["_path"] = "mem://test"
    return g


def _round(player_hands, collections=None, display=None):
    players = []
    for k, hand in enumerate(player_hands):
        players.append({"player_id": k, "bid": 0, "is_winner": False,
                        "hand": hand,
                        "collection_counts": (collections[k] if collections else {})})
    return {"auction": {"type": "treasure", "gems_available": ["Red"]},
            "players": players, "value_display": display or {},
            "available_missions": []}


def test_build_rows_no_cross_seat_hand_leak():
    # seat 0's features must be identical regardless of opponents' hands.
    vdf = {c: {"count": 3 if c == "Red" else 3} for c in COLORS}
    g_a = _game(["m0", "m1", "m2"],
                [_round([["Red", "Red"], ["Blue"], ["Green"]])], vdf)
    g_b = _game(["m0", "m1", "m2"],
                [_round([["Red", "Red"], ["Yellow", "Yellow"], ["Purple"]])], vdf)
    rows_a = {r["_color"]: r for r in tv.build_rows([g_a]) if r["_model"] == "m0"}
    rows_b = {r["_color"]: r for r in tv.build_rows([g_b]) if r["_model"] == "m0"}
    for c in COLORS:
        for k in FEATURES:
            assert rows_a[c][k] == rows_b[c][k], f"seat-0 feature {k}/{c} leaked from opponents"
    # and seat 0 genuinely sees its OWN hand
    assert rows_a["Red"]["own_hand_c"] == 2.0


def test_value_features_signature_has_no_opponent_input():
    # structural guarantee: the only hand input is the acting seat's own hand
    import inspect
    params = set(inspect.signature(value_features).parameters)
    assert "own_hand_counts" in params
    assert not any("opp" in p or "opponent" in p for p in params)


# --------------------------------------------------------------------------- #
# verified semantics: floor <= n_c <= ceiling on REAL games                    #
# --------------------------------------------------------------------------- #
def test_semantic_bounds_hold_on_real_games():
    # Benchmark transcripts are regenerable scratch under the gitignored
    # results/ tree, so a clean checkout has none — skip gracefully.
    files = sorted(glob.glob(
        os.path.join(ROOT, "results/**/benchmark_*.json"), recursive=True))[:40]
    if not files:
        pytest.skip("no benchmark games present (results/ is gitignored scratch)")
    games = tv.load_games(files)
    rows = tv.build_rows(games)
    assert len(rows) > 1000
    bad = 0
    for r in rows:
        if not (r["known_floor"] <= r["_label"] <= r["known_ceiling"] + 1e-9):
            bad += 1
    # floor = display + own-hand (certain to display); ceiling = 6 - collections.
    assert bad == 0, f"{bad} rows violated known_floor <= n_c <= known_ceiling"


# --------------------------------------------------------------------------- #
# exact mission bonus                                                          #
# --------------------------------------------------------------------------- #
def test_mission_bonus_newly_completed_only():
    est = ValueEstimator()
    # find a 'specific' 2-color mission in the registry
    spec = next(m for m in est.missions.values()
                if m.requirement.type == "specific" and len(m.requirement.colors) == 2)
    a, b = spec.requirement.colors
    # collection has one of the two; the candidate supplies the other -> newly completes
    assert est.mission_bonus({a: 1}, [b], [spec.id]) == spec.reward
    # already completable -> not NEWLY completed -> 0
    assert est.mission_bonus({a: 1, b: 1}, ["Red"], [spec.id]) == 0
    # mission not in the available list -> 0
    assert est.mission_bonus({a: 1}, [b], []) == 0


# --------------------------------------------------------------------------- #
# chart weighting + marginal value structure                                  #
# --------------------------------------------------------------------------- #
def test_evalue_is_chart_weighted_count_dist(monkeypatch):
    est = ValueEstimator()
    # chart A: count 1->4, 3->12. A 50/50 over counts {1,3} -> E[value] = 8.
    monkeypatch.setattr(est, "count_dist", lambda feat: {1: 0.5, 3: 0.5})
    feat = value_features(color="Red", display_counts={}, own_hand_counts=Counter(),
                          collection_counts_all={}, round_number=1)
    assert est.evalue_raw(feat, "A") == pytest.approx(0.5 * 4 + 0.5 * 12)


def test_marginal_value_adds_gem_value_and_mission_bonus(monkeypatch):
    est = ValueEstimator()
    monkeypatch.setattr(est, "count_dist", lambda feat: {3: 1.0})   # value 12/gem on chart A
    spec = next(m for m in est.missions.values()
                if m.requirement.type == "specific" and len(m.requirement.colors) == 2)
    a, b = spec.requirement.colors
    mv = est.marginal_value(
        gems=[a, b], seat_collection_counts={}, available_mission_ids=[spec.id],
        display_counts={}, own_hand_counts=Counter(), collection_counts_all={},
        round_number=1, chart_id="A")
    assert mv["gem_value"] == pytest.approx(24.0)          # two gems x 12
    assert mv["mission_bonus"] == spec.reward              # a+b completes the mission
    assert mv["total"] == pytest.approx(24.0 + spec.reward)


# --------------------------------------------------------------------------- #
# the head learns + persists                                                   #
# --------------------------------------------------------------------------- #
def test_head_recovers_deterministic_signal_and_roundtrips(tmp_path):
    rng = np.random.RandomState(0)
    feats, labels = [], []
    for _ in range(3000):
        disp = {"Red": int(rng.randint(0, 3))}
        own = Counter({"Red": int(rng.randint(0, 3))})
        coll = {"Red": int(rng.randint(0, 2))}
        f = value_features(color="Red", display_counts=disp, own_hand_counts=own,
                           collection_counts_all=coll, round_number=int(rng.randint(1, 18)))
        feats.append(f)
        labels.append(int(f["known_floor"]))               # label is exactly the floor
    X = np.array([[f[k] for k in FEATURES] for f in feats])
    est = ValueEstimator().fit(X, np.array(labels))
    acc = np.mean([max(est.count_dist(f).items(), key=lambda kv: kv[1])[0] == lab
                   for f, lab in zip(feats[:300], labels[:300])])
    assert acc > 0.95                                      # floor is a feature -> easily learned

    # vectorized E[value] must match the per-row path exactly
    Xchk = np.array([[f[k] for k in FEATURES] for f in feats[:200]])
    batch = est.evalues_batch(Xchk, ["A"] * 200, calibrate=False)
    perrow = np.array([est.evalue_raw(f, "A") for f in feats[:200]])
    assert np.allclose(batch, perrow)

    p = tmp_path / "vh.pkl"
    est.save(str(p))
    est2 = ValueEstimator.load(str(p))
    f0 = feats[0]
    assert est2.count_dist(f0) == est.count_dist(f0)

    # the saved payload must be numpy/python only (loads on the sklearn-free Modal image)
    import pickle as _pk
    payload = _pk.load(open(p, "rb"))
    assert all(isinstance(w, np.ndarray) for w in payload["W"])
    flat = list(payload.values()) + [x for v in payload.values()
                                     if isinstance(v, list) for x in v]
    assert all("sklearn" not in type(v).__module__ for v in flat)
