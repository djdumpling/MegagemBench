#!/usr/bin/env python3
"""Qwen3-4B baseline / post-SFT paired eval — 6 opponent panels + self-play on
the locked val seeds, each panel × seed × Qwen-seat played once (190 games
default). Shells out to ``scripts/eval/cost_probe_match.py`` per game; resumable
(any seed whose summary already exists is skipped).

This script does NOT start vLLM. Use ``scripts/eval/eval_qwen_baseline.sh`` for the
full pipeline, or start vLLM yourself first.

  .venv/bin/python scripts/eval/eval_qwen_baseline.py --dry-run
  .venv/bin/python scripts/eval/eval_qwen_baseline.py --yes
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from megagem.training.seed_splits import resolve_seeds

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

# Ctrl+C in the main thread reaches into this map to terminate live subprocs.
_running_procs: dict[int, subprocess.Popen] = {}
_procs_lock = threading.Lock()
_stop_event = threading.Event()


def _track_proc(proc: subprocess.Popen) -> None:
    with _procs_lock:
        _running_procs[proc.pid] = proc


def _untrack_proc(proc: subprocess.Popen) -> None:
    with _procs_lock:
        _running_procs.pop(proc.pid, None)


def _kill_all_running() -> None:
    with _procs_lock:
        for proc in list(_running_procs.values()):
            try:
                proc.terminate()
            except Exception:
                pass

QWEN_MODEL = "qwen/qwen3-4b-instruct"

PANELS: list[tuple[str, str]] = [
    ("vs_flash",       "google/gemini-3-flash-preview"),
    ("vs_opus46",      "anthropic/claude-opus-4.6"),
    ("vs_sonnet45",    "anthropic/claude-sonnet-4.5"),
    ("vs_maverick",    "meta-llama/llama-4-maverick"),
    ("vs_intellect3",  "prime-intellect/intellect-3"),
    ("vs_gpt_oss120b", "openai/gpt-oss-120b"),
    ("vs_gemini3pro",  "google/gemini-3.1-pro-preview"),
]

SELF_PLAY_LABEL = "self_play"


@dataclass(frozen=True)
class GameSpec:
    seed: int
    panel_label: str
    opponent: str | None         # None for self-play
    qwen_seat: int | None        # None for self-play
    models: tuple[str, str, str]

    @property
    def is_self_play(self) -> bool:
        return self.qwen_seat is None




def build_schedule(
    seeds: list[int],
    panels: list[tuple[str, str]],
    include_self_play: bool,
) -> list[GameSpec]:
    schedule: list[GameSpec] = []
    for label, opponent in panels:
        for seed in seeds:
            for seat in (0, 1, 2):
                models = [opponent, opponent, opponent]
                models[seat] = QWEN_MODEL
                schedule.append(GameSpec(
                    seed=seed,
                    panel_label=label,
                    opponent=opponent,
                    qwen_seat=seat,
                    models=(models[0], models[1], models[2]),
                ))
    if include_self_play:
        for seed in seeds:
            schedule.append(GameSpec(
                seed=seed,
                panel_label=SELF_PLAY_LABEL,
                opponent=None,
                qwen_seat=None,
                models=(QWEN_MODEL, QWEN_MODEL, QWEN_MODEL),
            ))
    return schedule


def count_by_panel(schedule: list[GameSpec]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for g in schedule:
        counts[g.panel_label] += 1
    return dict(counts)


def game_output_dir(out_root: Path, spec: GameSpec) -> Path:
    """Panel games nest by seat so seed-named files don't collide."""
    if spec.is_self_play:
        return out_root / spec.panel_label
    return out_root / spec.panel_label / f"seat{spec.qwen_seat}"


def summary_path_for(out_root: Path, spec: GameSpec) -> Path:
    return game_output_dir(out_root, spec) / f"cost_probe_summary_seed_{spec.seed}.json"


def write_manifest(
    out_dir: Path,
    schedule: list[GameSpec],
    seeds_spec: str,
    panels: list[tuple[str, str]],
) -> Path:
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seeds": str(seeds_spec),
        "qwen_model": QWEN_MODEL,
        "panels": [{"label": l, "opponent": o} for l, o in panels],
        "self_play_label": SELF_PLAY_LABEL,
        "games": [
            {
                "seed": g.seed,
                "panel": g.panel_label,
                "opponent": g.opponent,
                "qwen_seat": g.qwen_seat,
                "models": list(g.models),
            }
            for g in schedule
        ],
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def run_one_game(
    spec: GameSpec,
    out_root: Path,
    verbose: bool,
) -> dict:
    """Shell out to cost_probe_match.py for one game; return its summary JSON."""
    if _stop_event.is_set():
        raise RuntimeError("stop requested before subprocess start")

    out_dir = game_output_dir(out_root, spec)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "cost_probe_match.py"),
        "--seed", str(spec.seed),
        "--models", *spec.models,
        "--results-dir", str(out_dir),
        "--value-chart", "A",
        "--num-players", "3",
    ]
    if verbose:
        cmd.append("--verbose")

    stdout_dest = None if verbose else subprocess.DEVNULL
    stderr_dest = None if verbose else subprocess.PIPE
    proc = subprocess.Popen(cmd, stdout=stdout_dest, stderr=stderr_dest)
    _track_proc(proc)
    try:
        _, stderr_bytes = proc.communicate()
    finally:
        _untrack_proc(proc)

    if proc.returncode != 0:
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else "<no stderr>"
        raise RuntimeError(
            f"cost_probe_match.py failed for seed={spec.seed}, panel={spec.panel_label}, "
            f"seat={spec.qwen_seat}\n"
            f"return code: {proc.returncode}\n"
            f"stderr: {stderr[-2000:]}"
        )

    summary_path = summary_path_for(out_root, spec)
    if not summary_path.exists():
        raise RuntimeError(
            f"cost_probe_match.py returned 0 but no summary at {summary_path}"
        )
    return json.loads(summary_path.read_text())


def confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--seeds", default="val",
                   help="seed selection: a split name (train/val/test), an "
                        "inclusive range like 1151-1160, a comma-separated "
                        "list, or a path to a seed file. Default: val "
                        "(the locked SFT validation split).")
    p.add_argument("--panels", default=None,
                   help="comma-separated panel labels to run (e.g. 'vs_flash' "
                        "or 'vs_flash,vs_opus46'); 'self_play' is a valid "
                        "label. Default: all 6 opponent panels + self-play.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-parallel", type=int, default=4)
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def fmt_game(g: GameSpec) -> str:
    seat = "self" if g.is_self_play else f"seat{g.qwen_seat}"
    return f"{g.panel_label:<16} {seat:<5} seed={g.seed}"


def main() -> int:
    args = parse_args()

    if args.max_parallel < 1:
        print(f"ERROR: --max-parallel must be >= 1, got {args.max_parallel}", file=sys.stderr)
        return 2

    try:
        seeds = resolve_seeds(args.seeds)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not seeds:
        print(f"ERROR: no seeds resolved from {args.seeds!r}", file=sys.stderr)
        return 2

    selected_panels = PANELS
    include_self_play = True
    if args.panels:
        wanted = {x.strip() for x in args.panels.split(",") if x.strip()}
        known = {l for l, _ in PANELS} | {SELF_PLAY_LABEL}
        unknown = wanted - known
        if unknown:
            print(f"ERROR: unknown panel(s) {sorted(unknown)}; "
                  f"known: {sorted(known)}", file=sys.stderr)
            return 2
        selected_panels = [(l, o) for l, o in PANELS if l in wanted]
        include_self_play = SELF_PLAY_LABEL in wanted

    schedule = build_schedule(seeds, selected_panels, include_self_play)
    panel_counts = count_by_panel(schedule)

    out_dir = args.output_dir or (
        REPO_ROOT / "results" / f"eval_qwen_baseline_{datetime.now():%Y%m%d_%H%M%S}"
    )

    print("Qwen3-4B baseline eval")
    print(f"  Seeds:        {len(seeds)} ({args.seeds})")
    print(f"  Panels:       {len(selected_panels)} opponent panel(s)"
          + (" + self-play" if include_self_play else ""))
    print(f"  Games:        {len(schedule)} total")
    for label, _ in selected_panels:
        print(f"    {label:<18} {panel_counts.get(label, 0):>3}")
    if include_self_play:
        print(f"    {SELF_PLAY_LABEL:<18} {panel_counts.get(SELF_PLAY_LABEL, 0):>3}")
    print(f"  Max parallel: {args.max_parallel}")
    print(f"  Qwen:         local vLLM (megagem.endpoints -> {QWEN_MODEL})")
    print(f"  Output dir:   {out_dir}")
    print()

    if args.dry_run:
        print("Dry run — exiting without making API calls.")
        return 0

    if not args.yes:
        prompt = f"Proceed with {len(schedule)} games? [y/N] "
        if not confirm(prompt):
            print("Aborted.")
            return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir, schedule, args.seeds, selected_panels)

    skip_existing = not args.no_skip_existing
    skipped = 0
    todo: list[GameSpec] = []
    for game in schedule:
        summary_path = summary_path_for(out_dir, game)
        if skip_existing and summary_path.exists():
            skipped += 1
        else:
            todo.append(game)

    if skipped:
        print(f"Skipping {skipped} games with existing summaries.")
    if not todo:
        print("Nothing to do.")
        return 0

    print(f"Running {len(todo)} games at --max-parallel={args.max_parallel}")
    print()

    finished = 0
    failed = 0
    n = len(todo)

    # Explicit pool (not `with`) so Ctrl+C can cancel without waiting.
    ex = ThreadPoolExecutor(max_workers=args.max_parallel)
    future_to_game = {
        ex.submit(run_one_game, g, out_dir, args.verbose): g for g in todo
    }

    try:
        for fut in as_completed(future_to_game):
            game = future_to_game[fut]
            try:
                fut.result()
            except Exception as exc:
                failed += 1
                print(f"FAIL  {fmt_game(game)}: {exc}", flush=True)
                continue
            finished += 1
            print(f"DONE  [{finished:>3}/{n}] {fmt_game(game)}", flush=True)
        ex.shutdown(wait=True)
    except KeyboardInterrupt:
        print("\nInterrupted — cancelling pending and terminating in-flight subprocesses.",
              flush=True)
        _stop_event.set()
        ex.shutdown(wait=False, cancel_futures=True)
        _kill_all_running()
        print(f"Completed={finished}, failed={failed}, skipped={skipped}")
        print("Re-run the same command to resume (existing summaries will be skipped).")
        return 130

    print()
    print(f"Done. {finished} games run, {skipped} skipped, {failed} failed.")
    print(f"Output dir: {out_dir}")
    print()
    print("Next: aggregate Qwen win-rate / score-delta per panel from the per-game JSONs.")
    print("Each game's schema-v3 trajectory is at <panel>/<seat>/cost_probe_game_seed_<seed>.json;")
    print("the matching seat for Qwen is recorded in manifest.json.")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
