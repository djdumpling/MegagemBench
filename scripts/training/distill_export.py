#!/usr/bin/env python3
"""Level-1a corpus builder: selector-corrected SFT pairs from ev_gate game logs.

(Formerly ``scripts/phase3/distill1a_export.py``.)

Reads games recorded by the ev_dist arm (each turn stores the EXACT live prompt
and raw_response; ``ev_dist_decisions`` stores the selector's final choice), and
writes distill_train-compatible JSONL:

  {"metadata": {...}, "prompt": [{role:system},{role:user}], "completion": [{role:assistant}]}

Row kinds:
  treasure/deviation   (seat 0 only — the selector's seat) gate passed ->
                       completion = template-reasoned derivation of the corrected
                       bid, built from the selector's own logged quantities
                       (vhat, p_win at b_bp vs b*, budget). Coherent reasoning
                       matching the label; no LLM calls.
  treasure/passthrough (ALL seats) -> completion = the model's own raw_response
                       (label bid == its own bid; the anti-forgetting anchor on
                       the treasure surface; seats 1/2 prevent the adapter from
                       keying on your_player_id=0). Downsampled to --pass-dev-ratio.
  loan / investment    (ALL seats) completion = own raw_response (financing-channel
                       anchor — the liquidity-cascade lesson). Downsampled to
                       --anchor-ratio.

Split is by SEED (game-level): hash(seed) % 100 < held_pct -> held.jsonl, which
additionally embeds the F-hat win_curve per treasure decision so the parity gate
can score EV(b) for arbitrary bids with zero model deps at eval time.

Filters: parse_valid, not default_used, parsed bid consistency, prompt-length cap.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from megagem.assets import asset_path
from megagem.environment.bid_model import (
    EvDistModel, beats_tie, iter_nodes_bidtime, win_curve,
)
from megagem.data import load_value_charts
from megagem.game.cards import ValueChart

MAX_PROMPT_CHARS = 16000


def _dev_completion(dec: dict, rng: random.Random) -> str:
    """Reasoned completion deriving the corrected bid from logged selector fields."""
    chosen, b_bp = int(dec["chosen"]), int(dec["b_bp"])
    vhat = float(dec["vhat"])
    p_bp = int(round(100 * float(dec["p_win_bp"])))
    p_st = int(round(100 * float(dec["p_win_star"])))
    coins = int(dec["coins"])
    up = chosen > b_bp
    if up:
        variants = [
            (f"This treasure is worth about {vhat:.0f} to me net of bias. A bid of "
             f"{b_bp} only wins ~{p_bp}% of the time; pushing to {chosen} lifts the "
             f"win probability to ~{p_st}% while the surplus ({vhat:.0f} minus the "
             f"price, with each coin's option value priced in at my budget of "
             f"{coins}) stays clearly positive. Expected surplus is higher at {chosen}."),
            (f"Net value here is ~{vhat:.0f}. At {b_bp} I mostly lose this lot "
             f"(~{p_bp}% win); {chosen} wins ~{p_st}% and still clears value after "
             f"charging myself the option value of coins ({coins} left). The "
             f"expected-surplus-maximizing bid is {chosen}."),
        ]
    else:
        variants = [
            (f"This treasure is worth about {vhat:.0f} to me net of bias. {b_bp} "
             f"would overpay once each coin's option value is priced in (budget "
             f"{coins}); shading to {chosen} keeps the win probability at ~{p_st}% "
             f"and earns more surplus per coin. Expected surplus favors {chosen}."),
            (f"Net value is ~{vhat:.0f}, so {b_bp} is too aggressive relative to "
             f"surplus once I charge the option value of my {coins} remaining "
             f"coins. {chosen} still wins ~{p_st}% of the time with better "
             f"expected surplus. I bid {chosen}."),
        ]
    body = variants[rng.randrange(len(variants))]
    return f"{body}\n\n```json\n{{\"bid\": {chosen}}}\n```"


def _sft_row(kind: str, sysprompt: str, user: str, completion: str,
             meta: dict) -> dict:
    return {
        "metadata": {"source": "distill1a", "kind": kind, **meta},
        "prompt": [{"role": "system", "content": sysprompt},
                   {"role": "user", "content": user}],
        "completion": [{"role": "assistant", "content": completion}],
    }


def _treasure_curves(game: dict, model: EvDistModel, chart: ValueChart) -> dict[int, list[float]]:
    """round_number -> P(win|b) curve for seat 0, exactly the live selector's math."""
    out: dict[int, list[float]] = {}
    for node in iter_nodes_bidtime(game):
        if 0 not in node["bids"]:
            continue
        opp = [s for s in node["bids"] if s != 0]
        if len(opp) != 2:
            continue
        spec = [(model.mu(node, s, chart), int(node["coins"].get(s, 0)),
                 beats_tie(node["tiebreak"], 0, s)) for s in opp]
        out[int(node["round"])] = [
            float(x) for x in win_curve(spec, model.residuals,
                                        int(node["coins"].get(0, 0)))]
    return out


def export(games_dirs: list[str], out_dir: str, artifact: str, held_pct: int,
           pass_dev_ratio: float, anchor_ratio: float, seed: int,
           dev_repeat: int = 1) -> dict:
    rng = random.Random(seed)
    model = EvDistModel.load(artifact)
    ch = {cid: ValueChart.from_dict(cid, d)
          for cid, d in load_value_charts().items()}

    files = sorted(p for d in games_dirs for p in Path(d).glob("*.json"))
    train_rows: list[dict] = []
    held_rows: list[dict] = []
    pools = {"deviation": [], "passthrough": [], "anchor": []}
    counts = {"games": 0, "games_held": 0, "skipped_filter": 0,
              "skipped_join": 0, "skipped_long": 0}

    for fp in files:
        try:
            g = json.loads(fp.read_text())
        except Exception:  # noqa: BLE001
            continue
        md = g.get("metadata") or {}
        gseed = int(md.get("seed") or 0)
        sysprompt = md.get("system_prompt") or ""
        decs = g.get("ev_dist_decisions") or []
        if not sysprompt or not decs:
            continue
        counts["games"] += 1
        is_held = (gseed % 100) < held_pct
        counts["games_held"] += int(is_held)
        chart = ch.get(md.get("value_chart") or "A", ch["A"])
        rounds = {int(r.get("round_number") or 0): r for r in g.get("rounds") or []}
        curves = _treasure_curves(g, model, chart) if is_held else {}

        def _player(r: dict, sid: int) -> dict | None:
            return next((p for p in r.get("players", [])
                         if int(p.get("player_id", -1)) == sid), None)

        def _emit(kind: str, seat: int, rn: int, prompt: str, completion: str,
                  label: int, b_bp: int, gate_passed: bool, atype: str,
                  dec: dict | None) -> None:
            meta = {"seed": gseed, "round": rn, "seat": seat,
                    "label_bid": label, "b_bp": b_bp,
                    "gate_passed": gate_passed, "auction_type": atype}
            row = _sft_row(kind, sysprompt, prompt, completion, meta)
            if is_held:
                hm = dict(row["metadata"])
                hm.update({
                    "vhat": float((dec or {}).get("vhat", 0.0)),
                    "p_win_bp": float((dec or {}).get("p_win_bp", 0.0)),
                    "p_win_star": float((dec or {}).get("p_win_star", 0.0)),
                    "b_star": int((dec or {}).get("b_star", -1)),
                    "coins": int((dec or {}).get("coins", 0)),
                    # curves are seat-0's P(win|b); only seat-0 rows carry one
                    "win_curve": curves.get(rn, []) if (seat == 0 and dec) else [],
                })
                row = dict(row); row["metadata"] = hm
                held_rows.append(row)
            else:
                pools[kind].append(row)

        # --- seat-0 treasure decisions (the selector surface) ---
        for dec in decs:
            rn = int(dec.get("round") or 0)
            if int(dec.get("player_id", 0)) != 0:
                continue
            r = rounds.get(rn)
            p0 = _player(r, 0) if r else None
            if p0 is None or not p0.get("prompt"):
                counts["skipped_join"] += 1
                continue
            if len(p0["prompt"]) + len(sysprompt) > MAX_PROMPT_CHARS:
                counts["skipped_long"] += 1
                continue
            chosen = int(dec.get("chosen"))
            gate_passed = bool((dec.get("gate") or {}).get("passed"))
            if gate_passed:
                _emit("deviation", 0, rn, p0["prompt"],
                      _dev_completion(dec, rng), chosen,
                      int(dec.get("b_bp", -1)), True, "treasure", dec)
            else:
                ok = (p0.get("parse_valid") and not p0.get("default_used")
                      and int(p0.get("parsed_action", -10**9)) == chosen)
                if not ok:
                    counts["skipped_filter"] += 1
                    continue
                _emit("passthrough", 0, rn, p0["prompt"],
                      str(p0.get("raw_response") or ""), chosen,
                      int(dec.get("b_bp", -1)), False, "treasure", dec)

        # --- all-seat own-behavior anchors: treasure pass-throughs (seats 1/2,
        # no selector there => pure blueprint bids) + loan/investment (all seats,
        # the financing channel). Seats 1/2 stop the adapter keying on player 0. ---
        for rn, r in rounds.items():
            atype = ((r.get("auction") or {}).get("type") or "").lower()
            for sid in (0, 1, 2):
                if atype == "treasure" and sid == 0:
                    continue  # handled via the decision log above
                p = _player(r, sid)
                if (p is None or not p.get("prompt")
                        or not p.get("parse_valid") or p.get("default_used")):
                    counts["skipped_filter"] += 1
                    continue
                if len(p["prompt"]) + len(sysprompt) > MAX_PROMPT_CHARS:
                    counts["skipped_long"] += 1
                    continue
                label = int(p.get("parsed_action", 0))
                kind = "passthrough" if atype == "treasure" else "anchor"
                _emit(kind, sid, rn, p["prompt"],
                      str(p.get("raw_response") or ""), label, label,
                      False, atype, None)

    # --- assemble train set with pre-registered ratios ---
    n_dev = len(pools["deviation"])
    rng.shuffle(pools["passthrough"])
    rng.shuffle(pools["anchor"])
    n_pass = min(len(pools["passthrough"]), int(round(pass_dev_ratio * n_dev)))
    n_anc = min(len(pools["anchor"]), int(round(anchor_ratio * n_dev)))
    train_rows = (pools["deviation"] * max(1, dev_repeat)
                  + pools["passthrough"][:n_pass] + pools["anchor"][:n_anc])
    rng.shuffle(train_rows)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "train.jsonl").open("w") as fh:
        for row in train_rows:
            fh.write(json.dumps(row) + "\n")
    with (out / "held.jsonl").open("w") as fh:
        for row in held_rows:
            fh.write(json.dumps(row) + "\n")

    summary = {
        "games": counts["games"], "games_held": counts["games_held"],
        "train_rows": len(train_rows),
        "train_deviation": n_dev, "dev_repeat": dev_repeat,
        "train_passthrough": n_pass, "train_anchor": n_anc,
        "pool_passthrough": len(pools["passthrough"]), "pool_anchor": len(pools["anchor"]),
        "held_rows": len(held_rows),
        "held_treasure": sum(1 for r in held_rows
                             if r["metadata"]["kind"] in ("deviation", "passthrough")),
        "held_with_curve": sum(1 for r in held_rows
                               if r["metadata"].get("win_curve")),
        "skipped": {k: v for k, v in counts.items() if k.startswith("skipped")},
        "artifact": artifact,
    }
    (out / "export_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[distill1a-export] {json.dumps(summary)}", flush=True)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games-dir", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--artifact",
                    default=str(asset_path("ev_dist_bp_v1.pkl")))
    ap.add_argument("--held-pct", type=int, default=15,
                    help="seed %% 100 < held_pct -> held split")
    ap.add_argument("--pass-dev-ratio", type=float, default=2.0)
    ap.add_argument("--anchor-ratio", type=float, default=1.0)
    ap.add_argument("--dev-repeat", type=int, default=1,
                    help="replicate deviation rows N x (upweight the correction)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    export(args.games_dir, args.out_dir, args.artifact, args.held_pct,
           args.pass_dev_ratio, args.anchor_ratio, args.seed,
           dev_repeat=args.dev_repeat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
