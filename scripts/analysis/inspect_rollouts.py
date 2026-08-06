#!/usr/bin/env python3
"""Inspect Phase-3 GRPO rollout dumps (schema-v3 game JSONs).

Pretty-prints the games written by ``phase3_grpo.py`` when ``--dump-rollouts``
is set (default-ON for ``profile=evidence`` per plan §C.1). For each game in a
``roll_NNN/`` dir, shows:

  * Header — seed, chart, opponent models, round count.
  * Per round, per seat — actor id (``trainable`` / opponent), bid, parse_valid,
    legal_valid, completion length (chars), and the first 200 chars of
    ``reasoning`` (use ``--full`` to keep the entire prose, ``--components``
    to skip prose).
  * Per-game reward decomposition for the trainable seat — legal + shaping
    summed over its turns, plus ``terminal_breakdowns`` (margin, r_terminal,
    is_winner) from ``megagem.rl.reward``. The reward-component arithmetic is
    re-derived from the same modules ``phase3_grpo.py`` trains on, so a
    discrepancy between this view and the on-policy run would be a real bug.

CPU-only — imports ``megagem.rl.{reward,advantage,scorer}`` (torch-free); no GPU
or network. Reads only, mutates nothing.

  # Latest roll under a synced run dir (after `modal volume get ... --recursive`):
  python scripts/analysis/inspect_rollouts.py results/phase3_evidence_<tag>_modal/rollout_dumps

  # A specific roll, full prose, filter to one seed:
  python scripts/analysis/inspect_rollouts.py <dump_dir> --roll 0 --seed 9000 --full

  # Reward-component table only (quick triage of group_std degeneracy):
  python scripts/analysis/inspect_rollouts.py <dump_dir> --components --limit 8
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from pathlib import Path

from megagem.rl.advantage import AdvantageConfig, compute_advantages
from megagem.rl.reward import (
    RewardConfig, per_turn_rewards, terminal_breakdowns,
)
from megagem.rl.scorer import parse_transcript


def _find_phase3_grpo_json(dump_root: Path) -> Path | None:
    """Locate the sibling ``phase3_grpo.json`` so we can re-derive the EXACT
    RewardConfig the run trained on. Looks up the directory chain until we hit
    the file or the filesystem root — Modal-sync dirs put the JSON adjacent to
    ``rollout_dumps/``; bare ``roll_NNN/`` dirs need one more level."""
    for parent in (dump_root, *dump_root.parents):
        cand = parent / "phase3_grpo.json"
        if cand.is_file():
            return cand
        if parent == parent.parent:  # filesystem root
            break
    return None


def _load_reward_config(
    dump_root: Path,
    *,
    overrides: dict,
) -> tuple[RewardConfig, dict]:
    """Build the RewardConfig the inspector should use.

    Resolution order (later wins):
      1. RewardConfig() defaults (locked Phase-1 cut: scale=21.5, clip, λ=0.01).
      2. ``reward_config`` block of the sibling phase3_grpo.json, if found.
      3. CLI overrides (``--scale`` etc.).

    Returns ``(cfg, provenance)`` where provenance documents which path each
    field came from — printed at the top of the inspector output so a viewer
    is never silently looking at the wrong scale/shape.
    """
    fields = {f.name for f in dataclasses.fields(RewardConfig)}
    provenance = {f: "default" for f in fields}
    cfg_kwargs: dict = {}

    sidecar = _find_phase3_grpo_json(dump_root)
    sidecar_block: dict | None = None
    if sidecar is not None:
        try:
            sidecar_block = (json.loads(sidecar.read_text())
                             or {}).get("reward_config")
        except (json.JSONDecodeError, OSError):
            sidecar_block = None
    if isinstance(sidecar_block, dict):
        for k, v in sidecar_block.items():
            if k in fields:
                cfg_kwargs[k] = v
                provenance[k] = f"phase3_grpo.json:{sidecar.parent.name}"

    for k, v in overrides.items():
        if v is None:
            continue
        if k in fields:
            cfg_kwargs[k] = v
            provenance[k] = "--cli"

    try:
        cfg = RewardConfig(**cfg_kwargs)
    except TypeError as e:
        raise SystemExit(
            f"could not build RewardConfig from "
            f"{cfg_kwargs!r}: {e}") from e
    return cfg, provenance


def _to_bool(x) -> bool:
    """Schema-v3 dumps sometimes coerce bools to strings via ``json.dumps(...,
    default=str)`` (phase3_grpo.py:207). Tolerate both."""
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in ("true", "1", "yes")
    return bool(x)


def _to_int(x):
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        try:
            return int(x)
        except ValueError:
            return None
    return None


def find_roll_dirs(root: Path) -> list[Path]:
    """Accept any of: a ``roll_NNN/`` dir directly, a ``rollout_dumps/`` parent,
    or a run dir containing ``rollout_dumps/``."""
    if not root.exists():
        raise SystemExit(f"path does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    if root.name.startswith("roll_") and any(root.glob("*.json")):
        return [root]
    rolls = sorted(p for p in root.iterdir()
                   if p.is_dir() and p.name.startswith("roll_"))
    if rolls:
        return rolls
    nested = root / "rollout_dumps"
    if nested.is_dir():
        return sorted(p for p in nested.iterdir()
                      if p.is_dir() and p.name.startswith("roll_"))
    return []


def load_game(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def fmt_header(roll_dir: Path, game_path: Path, game: dict) -> str:
    md = game.get("metadata") or {}
    return ("\n" + "═" * 78 + "\n"
            f"{roll_dir.name}/{game_path.name}\n"
            f"  seed={md.get('seed')}  chart={md.get('value_chart')}  "
            f"players={md.get('num_players')}  "
            f"rounds={len(game.get('rounds') or [])}\n"
            f"  models={md.get('models')}\n"
            + "─" * 78)


def fmt_round(r: dict, full: bool) -> str:
    out: list[str] = []
    rn = r.get("round_number", "?")
    for p in r.get("players") or []:
        pid = p.get("player_id")
        actor = p.get("actor_id", "?")
        bid = p.get("bid")
        pv = "OK" if _to_bool(p.get("parse_valid")) else "FAIL"
        lv = "OK" if _to_bool(p.get("legal_valid")) else "FAIL"
        ls = p.get("length_split") if isinstance(p.get("length_split"), dict) else {}
        chars = _to_int(ls.get("total_chars")) if ls else None
        reason = p.get("reasoning") or ""
        if not full and len(reason) > 200:
            reason = reason[:200] + " …"
        marker = "▶" if str(actor) == "trainable" else " "
        out.append(
            f"  R{rn:>2}  {marker} P{pid} {str(actor)[:18]:<18} "
            f"bid={str(bid):>3}  parse={pv}  legal={lv}  chars={chars}")
        if reason.strip():
            # Indent so the prose lines visually nest under the seat row.
            for line in reason.splitlines():
                out.append(f"        {line.rstrip()}")
        # If the player emitted a nested ``reveal`` (a second action that round),
        # surface it too — schema-v3 nests reveal under players[].reveal.
        rv = p.get("reveal")
        if isinstance(rv, dict):
            gc = rv.get("gem_color") or rv.get("parsed_action")
            ract = rv.get("actor_id", "?")
            rpv = "OK" if _to_bool(rv.get("parse_valid")) else "FAIL"
            rlv = "OK" if _to_bool(rv.get("legal_valid")) else "FAIL"
            out.append(
                f"        reveal: {ract} gem={gc}  parse={rpv}  legal={rlv}")
    return "\n".join(out)


def fmt_breakdown(game: dict, cfg: RewardConfig,
                  adv_cfg: AdvantageConfig) -> str:
    """Reward decomposition for trainable seat(s).

    ``per_turn_rewards`` gives us the per-turn ``TurnReward`` records (with
    ``legal``, ``shaping``, ``parse_valid``, ``legal_valid``,
    ``terminal_correction``); ``terminal_breakdowns`` adds the per-pid
    ``r_terminal`` / margin / is_winner. ``compute_advantages`` is also called
    so we can surface the row-level ``reward`` sum that ``phase3_grpo.py``
    actually trains on — a discrepancy between (legal+shaping+correction+
    r_terminal) and ``Σ r.reward`` would be a real wiring bug.
    """
    parsed = parse_transcript(game)
    bds = terminal_breakdowns(parsed, cfg)
    turn_rewards = per_turn_rewards(parsed, cfg)
    adv_rows = compute_advantages([game], cfg, adv_cfg)
    # legal/shaping per-pid from TurnReward (only sum the trainable seats);
    # actor_id was post-tagged before the dump, so reading it here is safe.
    agg: dict[int, dict] = defaultdict(
        lambda: {"legal": 0.0, "shaping": 0.0, "term_corr": 0.0,
                 "n_turns": 0, "n_illegal": 0, "n_unparsed": 0})
    for tr in turn_rewards:
        if tr.actor_id != "trainable":
            continue
        a = agg[tr.player_id]
        a["legal"] += float(tr.legal)
        a["shaping"] += float(tr.shaping)
        a["term_corr"] += float(tr.terminal_correction)
        a["n_turns"] += 1
        if not tr.legal_valid:
            a["n_illegal"] += 1
        if not tr.parse_valid:
            a["n_unparsed"] += 1
    # Σ r.reward across the trainable seat's TurnAdvantage rows — the value
    # actually summed into Σ precomputed_reward for that group_id.
    rsum: dict[int, float] = defaultdict(float)
    for r in adv_rows:
        if r.actor_id == "trainable" and r.trainable:
            rsum[r.player_id] += float(r.reward)
    if not agg:
        return "  (no trainable seats in this game)"
    out = ["  reward breakdown — trainable seat(s):"]
    for pid, a in sorted(agg.items()):
        bd = bds.get(pid)
        # legal+shaping+term_corr+r_terminal = the policy's total per-game
        # reward. Σ r.reward (above) must equal that.
        expected_total = (a["legal"] + a["shaping"] + a["term_corr"]
                          + (bd.r_terminal if bd else 0.0))
        actual_total = rsum.get(pid, 0.0)
        out.append(
            f"    P{pid}  legal={a['legal']:+.4f}  shaping={a['shaping']:+.4f}  "
            f"term_corr={a['term_corr']:+.4f}  "
            f"turns={a['n_turns']} unparsed={a['n_unparsed']} "
            f"illegal={a['n_illegal']}")
        if bd is not None:
            out.append(
                f"        terminal: r_terminal={bd.r_terminal:+.4f}  "
                f"margin={bd.margin:+.2f}  final_score={bd.final_score}  "
                f"is_winner={bd.is_winner}")
        out.append(
            f"        total = legal+shaping+term_corr+r_terminal "
            f"= {expected_total:+.4f}   |   Σ row.reward = {actual_total:+.4f}"
            + ("  (✓ match)" if abs(expected_total - actual_total) < 1e-6
               else f"  ⚠ MISMATCH Δ={expected_total - actual_total:+.4f}"))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dir",
                    help="rollout_dumps/ dir, a single roll_NNN/ dir, or a run "
                         "dir containing rollout_dumps/.")
    ap.add_argument("--roll", type=int, default=None,
                    help="specific roll index (default: latest).")
    ap.add_argument("--all-rolls", action="store_true",
                    help="walk every roll instead of just one.")
    ap.add_argument("--limit", type=int, default=4,
                    help="cap games shown per roll.")
    ap.add_argument("--seed", type=int, default=None,
                    help="filter to a specific seed (matches `seed{S}_` in filename).")
    ap.add_argument("--full", action="store_true",
                    help="print full reasoning text, no 200-char truncation.")
    ap.add_argument("--components", action="store_true",
                    help="reward-component table only (skip per-round prose).")
    # Reward-config overrides. All default to None ⇒ inherit from the sibling
    # phase3_grpo.json (auto-detected) or the locked RewardConfig() defaults.
    # Pass any of these to override that resolution — useful when inspecting a
    # dump dir without its sidecar JSON, or to A/B a different reward shape.
    ap.add_argument("--scale", type=float, default=None,
                    help="override RewardConfig.scale (locked default 21.5).")
    ap.add_argument("--terminal-shape", choices=("clip", "tanh"), default=None,
                    help="override RewardConfig.terminal_shape (default 'clip').")
    ap.add_argument("--shaping-lambda", type=float, default=None,
                    help="override RewardConfig.shaping_lambda (default 0.01).")
    ap.add_argument("--reveal-shaping-weight", type=float, default=None,
                    help="override RewardConfig.reveal_shaping_weight (default 1.0).")
    ap.add_argument("--terminal-correction", default=None,
                    help="1/0/true/false; override RewardConfig.terminal_correction. "
                         "Unset ⇒ inherit from sidecar / RewardConfig default.")
    ap.add_argument("--win-bonus", type=float, default=None,
                    help="override RewardConfig.win_bonus.")
    ap.add_argument("--illegal-penalty", type=float, default=None,
                    help="override RewardConfig.illegal_penalty.")
    args = ap.parse_args(argv)

    root = Path(args.dump_dir).expanduser().resolve()
    rolls = find_roll_dirs(root)
    if not rolls:
        raise SystemExit(f"no roll_NNN/ subdirs found at {root}")
    if args.roll is not None:
        want = f"roll_{int(args.roll):03d}"
        rolls = [r for r in rolls if r.name == want]
        if not rolls:
            raise SystemExit(f"--roll {args.roll}: no matching subdir under {root}")
    elif not args.all_rolls:
        rolls = [rolls[-1]]  # latest only

    tc_override: bool | None = None
    if args.terminal_correction is not None:
        tc_override = args.terminal_correction.lower() in (
            "1", "true", "yes", "on")
    overrides = {
        "scale": args.scale,
        "terminal_shape": args.terminal_shape,
        "shaping_lambda": args.shaping_lambda,
        "reveal_shaping_weight": args.reveal_shaping_weight,
        "terminal_correction": tc_override,
        "win_bonus": args.win_bonus,
        "illegal_penalty": args.illegal_penalty,
    }
    cfg, provenance = _load_reward_config(root, overrides=overrides)
    adv_cfg = AdvantageConfig()
    # Surface where each reward-config field came from — silent default vs.
    # auto-detected sidecar vs. CLI override. This is the first thing the
    # viewer reads so the reward-decomposition table can never be silently
    # off (e.g. inspecting a tanh-shaped run with the clip default).
    src_counts: dict[str, int] = {}
    for v in provenance.values():
        src_counts[v] = src_counts.get(v, 0) + 1
    src_summary = " ".join(
        f"{k}={v}" for k, v in sorted(src_counts.items()))
    print(f"[reward cfg] {dataclasses.asdict(cfg)}")
    print(f"[reward cfg sources] {src_summary}")
    # Surface fields whose VALUE diverges from the locked RewardConfig()
    # defaults — those are the things actually different about this run.
    # (A field merely re-confirmed by the sidecar gets no special call-out.)
    locked = dataclasses.asdict(RewardConfig())
    diverged = [(f, getattr(cfg, f)) for f in provenance
                if locked.get(f) != getattr(cfg, f)]
    if diverged:
        print("[reward cfg diverged-from-locked] " + ", ".join(
            f"{f}={v!r}({provenance[f]})" for f, v in diverged))

    n_printed = 0
    for roll_dir in rolls:
        paths = sorted(roll_dir.glob("*.json"))
        if args.seed is not None:
            paths = [p for p in paths if f"seed{args.seed}_" in p.name]
        if not paths:
            print(f"\n[{roll_dir.name}] no games match filters")
            continue
        shown = paths[: max(0, int(args.limit))]
        print(f"\n══ {roll_dir.name}  ({len(paths)} games total, "
              f"showing {len(shown)})  reward_cfg: scale={cfg.scale} "
              f"λ={cfg.shaping_lambda} shape={cfg.terminal_shape} "
              f"correction={cfg.terminal_correction} ══")
        for gp in shown:
            game = load_game(gp)
            if game is None:
                print(f"  {gp.name}: unreadable JSON")
                continue
            if not args.components:
                print(fmt_header(roll_dir, gp, game))
                for r in game.get("rounds") or []:
                    print(fmt_round(r, args.full))
            else:
                # Components-only still needs a one-line locator per game.
                md = game.get("metadata") or {}
                print(f"\n  {roll_dir.name}/{gp.name}  seed={md.get('seed')} "
                      f"chart={md.get('value_chart')}")
            try:
                print(fmt_breakdown(game, cfg, adv_cfg))
            except Exception as e:  # noqa: BLE001
                print(f"  [reward breakdown failed: {type(e).__name__}: {e}]")
            n_printed += 1

    print(f"\n[summary] {n_printed} game(s) printed across "
          f"{len(rolls)} roll(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
