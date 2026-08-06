#!/usr/bin/env python3
"""P0 — build, calibrate, and validate the supervised value head ($0/CPU).

Builds one row per (game, round, color, seat) from saved game logs: leak-safe features
(public board + the seat's own hand) -> label n_c (color's final display count). Fits the
probabilistic count head, calibrates E[value] on a held-out fold, and reports held-out
value R²/MAE by game progress vs naive baselines + calibration. Saves the model.

GO (P0): held-out mid-bucket value R² >= 0.5 AND beats the best naive baseline AND
calibration slope in [0.9, 1.1].

  python3 -m megagem.value_head.train --globs 'results/**/*.json'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from megagem.value_head.value_estimator import (
    COLORS, FEATURES, ValueEstimator, value_features,
)


def load_games(globs):
    seen, games = set(), []
    for g in globs:
        for f in glob.glob(g, recursive=True):
            if f in seen:
                continue
            seen.add(f)
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            md, fr = d.get("metadata") or {}, d.get("final_results") or {}
            if isinstance(md.get("models"), list) and isinstance(fr.get("value_display_final"), dict):
                d["_path"] = f
                games.append(d)
    return games


def _disp_counts(value_display):
    out = {}
    if isinstance(value_display, dict):
        for c, info in value_display.items():
            out[c] = int(info.get("count", 0) or 0) if isinstance(info, dict) else int(info or 0)
    return out


def bidtime_state(rounds, ri):
    """Reconstruct the PRE-reveal (bid-time) public state for round ``ri`` from the logged
    POST-round snapshots, so offline training features match the LIVE adapter exactly.

    A round's bid sees the state AFTER round ri-1's reveal/win; round 0's bid sees the
    initial deal (empty display + collections, and each seat's full pre-reveal hand —
    recovered by adding back the gem it revealed in round 0, if any).
    Returns (display_counts, coll_all, own_by_seat, coll_by_seat).
    """
    if ri == 0:
        players = rounds[0].get("players") or []
        own, seat_coll = [], []
        for p in players:
            h = Counter(p.get("hand") or [])
            rev = p.get("gem_revealed")
            if rev:
                h[rev] += 1                       # add back the gem revealed in round 0
            own.append(h)
            seat_coll.append({})                  # no collections before any auction
        return {}, defaultdict(int), own, seat_coll

    src = rounds[ri - 1]
    disp = _disp_counts(src.get("value_display"))
    sp = src.get("players") or []
    coll_all: defaultdict = defaultdict(int)
    seat_coll, own = [], []
    for p in sp:
        cc = {c: int(k or 0) for c, k in (p.get("collection_counts") or {}).items()}
        seat_coll.append(cc)
        for c, k in cc.items():
            coll_all[c] += k
        own.append(Counter(p.get("hand") or []))
    return disp, coll_all, own, seat_coll


def build_rows(games):
    """One row per (game, round, color, seat): leak-safe BID-TIME features + label n_c."""
    rows = []
    for g in games:
        models = g["metadata"]["models"]
        chart_id = g["metadata"].get("value_chart", "A")
        vdf = g["final_results"]["value_display_final"]
        final_n = {c: int((vdf.get(c) or {}).get("count", 0)) for c in COLORS}
        rounds = g.get("rounds") or []
        n = len(rounds)
        for ri, rnd in enumerate(rounds):
            rn = int(rnd.get("round_number", ri + 1))      # the round being DECIDED
            disp, coll_all, own_by_seat, _ = bidtime_state(rounds, ri)
            for seat, own in enumerate(own_by_seat):
                for c in COLORS:
                    feat = value_features(
                        color=c, display_counts=disp, own_hand_counts=own,
                        collection_counts_all=coll_all, round_number=rn)
                    rows.append({
                        **feat,
                        "_label": final_n[c],
                        "_game": g["_path"], "_chart": chart_id,
                        "_round_frac": (ri + 1) / max(1, n),
                        "_model": models[seat] if seat < len(models) else "?",
                        "_color": c, "_disp_c": feat["disp_c"],
                        "_floor": feat["known_floor"],
                    })
    return rows


def _split(games, seed, fracs=(0.7, 0.1, 0.2)):
    paths = sorted(g["_path"] for g in games)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(paths))
    n_fit = int(fracs[0] * len(paths))
    n_cal = int(fracs[1] * len(paths))
    fit = {paths[i] for i in perm[:n_fit]}
    cal = {paths[i] for i in perm[n_fit:n_fit + n_cal]}
    test = {paths[i] for i in perm[n_fit + n_cal:]}
    return fit, cal, test


def _bucket(rf):
    return "early" if rf <= 1 / 3 else ("mid" if rf <= 2 / 3 else "late")


def _Xy(rows):
    X = np.array([[r[k] for k in FEATURES] for r in rows], float)
    y = np.array([r["_label"] for r in rows], int)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--globs", nargs="+", default=["results/**/*.json"],
        help="glob(s) of schema-v3 game JSONs to fit on. The default points at "
             "results/ (gitignored scratch), so pass an explicit corpus unless "
             "you have generated games locally.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-out", default="results/value_probe/value_head.pkl")
    ap.add_argument("--report-out", default="results/value_probe/value_head_report.json")
    ap.add_argument("--leave-out", default="gemini-3-flash",
                    help="substring: also report value R² training WITHOUT this model, testing ON it")
    args = ap.parse_args()

    games = load_games(args.globs)
    if not games:
        raise SystemExit(
            f"no schema-v3 games matched {args.globs}. Generate a corpus "
            f"(e.g. megagem-run, or scripts/eval/eval_qwen_baseline.sh) and "
            f"point --globs at it.")
    print(f"loaded {len(games)} games")
    rows = build_rows(games)
    print(f"built {len(rows)} (state,color,seat) rows; "
          f"label n_c distribution: {dict(sorted(Counter(r['_label'] for r in rows).items()))}")

    fit_p, cal_p, test_p = _split(games, args.seed)
    fit_rows = [r for r in rows if r["_game"] in fit_p]
    cal_rows = [r for r in rows if r["_game"] in cal_p]
    test_rows = [r for r in rows if r["_game"] in test_p]
    print(f"games fit/cal/test = {len(fit_p)}/{len(cal_p)}/{len(test_p)}  "
          f"rows = {len(fit_rows)}/{len(cal_rows)}/{len(test_rows)}")

    est = ValueEstimator()
    Xf, yf = _Xy(fit_rows)
    est.fit(Xf, yf, seed=args.seed)

    # calibrate E[value] on the held-out calibration fold
    cal_raw = est.evalues_batch(_Xy(cal_rows)[0], [r["_chart"] for r in cal_rows], calibrate=False)
    cal_real = np.array([est.charts[r["_chart"]].get_gem_value(r["_label"]) for r in cal_rows])
    est.fit_calibration(cal_raw, cal_real)

    # ---- held-out evaluation ----
    def chart_v(r, count):
        return est.charts[r["_chart"]].get_gem_value(int(count))

    realized = np.array([chart_v(r, r["_label"]) for r in test_rows])
    pred_ev = est.evalues_batch(_Xy(test_rows)[0], [r["_chart"] for r in test_rows])
    pred_arg = est.argmax_counts_batch(_Xy(test_rows)[0])
    naive_now = np.array([chart_v(r, r["_disp_c"]) for r in test_rows])
    naive_floor = np.array([chart_v(r, min(6, r["_floor"])) for r in test_rows])
    buckets = np.array([_bucket(r["_round_frac"]) for r in test_rows])
    true_n = np.array([r["_label"] for r in test_rows])

    def block(mask):
        if mask.sum() < 20:
            return None
        return {
            "n": int(mask.sum()),
            "value_r2_head": float(r2_score(realized[mask], pred_ev[mask])),
            "value_mae_head": float(mean_absolute_error(realized[mask], pred_ev[mask])),
            "value_r2_naive_now": float(r2_score(realized[mask], naive_now[mask])),
            "value_r2_naive_floor": float(r2_score(realized[mask], naive_floor[mask])),
            "count_acc_head": float((pred_arg[mask] == true_n[mask]).mean()),
        }

    report = {"by_bucket": {b: block(buckets == b) for b in ("early", "mid", "late")},
              "ALL": block(np.ones(len(test_rows), bool))}

    # calibration slope on test (realized ~ a + b*pred_ev)
    b_slope = float(np.polyfit(pred_ev, realized, 1)[0])
    report["calibration_slope"] = b_slope

    print("\n" + "=" * 74)
    print("P0 — held-out value-head validation (E[value] vs realized; by game progress)")
    print("=" * 74)
    print(f"  {'bucket':<6}{'HEAD r2':>9}{'HEAD mae':>10}{'count_acc':>11}"
          f"{'naive_now r2':>14}{'naive_floor r2':>16}")
    for b in ("early", "mid", "late", "ALL"):
        d = report["by_bucket"].get(b) if b != "ALL" else report["ALL"]
        if not d:
            continue
        print(f"  {b:<6}{d['value_r2_head']:>9.3f}{d['value_mae_head']:>10.2f}"
              f"{d['count_acc_head']:>11.3f}{d['value_r2_naive_now']:>14.3f}"
              f"{d['value_r2_naive_floor']:>16.3f}")
    print(f"  calibration slope (realized ~ E[value]): {b_slope:.3f}")

    # ---- opponent-transfer robustness: train WITHOUT leave-out, test ON it ----
    lo = args.leave_out
    lo_games_test = [g for g in games if any(lo in m for m in g["metadata"]["models"])]
    lo_train = [r for r in rows if all(lo not in m for m in [r["_model"]])
                and r["_game"] not in {g["_path"] for g in lo_games_test}]
    lo_test = [r for r in rows if r["_game"] in {g["_path"] for g in lo_games_test}]
    if len(lo_train) > 1000 and len(lo_test) > 200:
        est2 = ValueEstimator()
        est2.fit(*_Xy(lo_train), seed=args.seed)
        rz = np.array([chart_v(r, r["_label"]) for r in lo_test])
        pv = est2.evalues_batch(_Xy(lo_test)[0], [r["_chart"] for r in lo_test], calibrate=False)
        mid = np.array([_bucket(r["_round_frac"]) == "mid" for r in lo_test])
        report["opponent_transfer"] = {
            "leave_out": lo, "n_test": len(lo_test),
            "value_r2_mid": float(r2_score(rz[mid], pv[mid])) if mid.sum() > 20 else None,
            "value_r2_all": float(r2_score(rz, pv)),
        }
        print(f"  opponent-transfer (train w/o {lo}, test ON {lo}): "
              f"mid R²={report['opponent_transfer']['value_r2_mid']}, "
              f"all R²={report['opponent_transfer']['value_r2_all']:.3f}")

    # ---- GO / NO-GO ----
    mid = report["by_bucket"].get("mid") or {}
    go = (mid.get("value_r2_head", 0) >= 0.5
          and mid.get("value_r2_head", 0) > max(mid.get("value_r2_naive_now", -9),
                                                mid.get("value_r2_naive_floor", -9)) + 0.05
          and 0.9 <= b_slope <= 1.1)
    report["P0_GO"] = bool(go)
    print("\n" + "-" * 74)
    print(f"P0 {'GO' if go else 'NO-GO'}: mid value R²={mid.get('value_r2_head'):.3f} "
          f"(naive best {max(mid.get('value_r2_naive_now', -9), mid.get('value_r2_naive_floor', -9)):.3f}), "
          f"calibration slope {b_slope:.3f}.")
    print("-" * 74)

    est.save(args.model_out)
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nsaved model -> {args.model_out}\nsaved report -> {args.report_out}")


if __name__ == "__main__":
    main()
