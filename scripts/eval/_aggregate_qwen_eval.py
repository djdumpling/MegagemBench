#!/usr/bin/env python3
"""Per-panel Qwen win-rate / mean score / mean delta from eval_qwen_baseline.sh
outputs.

  python scripts/eval/_aggregate_qwen_eval.py results/eval_sft_step1200 results/eval_sft_step1000
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


QWEN_MODEL = "qwen/qwen3-4b-instruct"


def find_game_files(root: Path):
    if not root.is_dir():
        return
    for panel_dir in sorted(root.iterdir()):
        if not panel_dir.is_dir():
            continue
        panel = panel_dir.name
        if panel == "self_play":
            for f in sorted(panel_dir.glob("cost_probe_game_seed_*.json")):
                yield panel, None, f
        else:
            for seat_dir in sorted(panel_dir.iterdir()):
                if not seat_dir.is_dir():
                    continue
                for f in sorted(seat_dir.glob("cost_probe_game_seed_*.json")):
                    yield panel, seat_dir.name, f


def aggregate(root: Path) -> list[dict]:
    """Per-panel win rates over the game JSONs under ``root``.

    Every game that cannot be scored is COUNTED, not silently dropped: this
    output is the published win rate, so a shrinking denominator has to be
    visible. Skips land in each row's ``n_skipped``/``skips`` fields.
    """
    by_panel: dict[str, list[tuple[float, float | None]]] = defaultdict(list)
    skips: dict[str, Counter] = defaultdict(Counter)

    for panel, _seat, fpath in find_game_files(root):
        try:
            data = json.loads(fpath.read_text())
        except (json.JSONDecodeError, OSError):
            skips[panel]["unreadable"] += 1
            continue
        meta = data.get("metadata") or {}
        models = meta.get("models") or []
        final = (data.get("final_results") or {}).get("final_scores") or []
        if not final or len(final) != len(models):
            skips[panel]["incomplete_final_scores"] += 1
            continue

        qwen_seats = [i for i, m in enumerate(models) if m == QWEN_MODEL]
        if not qwen_seats:
            skips[panel]["no_qwen_seat"] += 1
            continue

        scores = {
            entry["player_id"]: float(entry["final_score"])
            for entry in final
            if "player_id" in entry and "final_score" in entry
        }
        if len(scores) != len(models):
            skips[panel]["malformed_scores"] += 1
            continue

        if panel == "self_play":
            qwen_scores = [scores[i] for i in qwen_seats]
            by_panel[panel].append((mean(qwen_scores), None))
        else:
            qwen_seat = qwen_seats[0]
            qwen_score = scores[qwen_seat]
            opp_scores = [v for k, v in scores.items() if k != qwen_seat]
            max_opp = max(opp_scores)
            by_panel[panel].append((qwen_score, max_opp))

    rows = []
    for panel in sorted(set(by_panel) | set(skips)):
        results = by_panel.get(panel, [])
        panel_skips = dict(skips.get(panel, Counter()))
        row = {
            "panel": panel,
            "n_games": len(results),
            "n_skipped": sum(panel_skips.values()),
            "skips": panel_skips,
        }
        if not results:
            row.update({"win_rate": None, "mean_qwen_score": None,
                        "mean_qwen_delta": None})
        elif panel == "self_play":
            row.update({
                "win_rate": None,
                "mean_qwen_score": round(mean(r[0] for r in results), 2),
                "mean_qwen_delta": None,
            })
        else:
            wins = sum(1 for q, o in results if q > o)
            row.update({
                "win_rate": round(wins / len(results), 3),
                "mean_qwen_score": round(mean(r[0] for r in results), 2),
                "mean_qwen_delta": round(mean((r[0] - r[1]) for r in results), 2),
            })
        rows.append(row)
    return rows


def print_table(label: str, rows: list[dict]) -> None:
    print(f"\n=== {label} ===")
    if not rows:
        print("  (no completed games found)")
        return
    header = f"  {'panel':<18} {'n':>4}  {'win_rate':>9}  {'qwen_score':>10}  {'delta':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        wr = f"{r['win_rate']:.3f}" if r["win_rate"] is not None else "    --"
        dt = f"{r['mean_qwen_delta']:+8.2f}" if r["mean_qwen_delta"] is not None else "      --"
        score = (f"{r['mean_qwen_score']:>10.2f}"
                 if r["mean_qwen_score"] is not None else f"{'--':>10}")
        print(f"  {r['panel']:<18} {r['n_games']:>4}  {wr:>9}  {score}  {dt}")
    total_skipped = sum(r.get("n_skipped", 0) for r in rows)
    if total_skipped:
        print(f"\n  WARNING: {total_skipped} game file(s) could not be scored and "
              f"are NOT in the rates above:")
        for r in rows:
            if r.get("n_skipped"):
                detail = ", ".join(f"{k}={v}" for k, v in sorted(r["skips"].items()))
                print(f"    {r['panel']:<18} skipped={r['n_skipped']:<4} ({detail})")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('dirs', nargs='+', type=Path)
    args = p.parse_args()

    any_results = False
    for d in args.dirs:
        rows = aggregate(d)
        print_table(d.name if d.is_dir() else str(d), rows)
        if rows:
            any_results = True
    return 0 if any_results else 1


if __name__ == "__main__":
    sys.exit(main())
