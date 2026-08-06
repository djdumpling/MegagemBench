"""Per-turn return-to-go advantages (Phase 1.3 / 1.5).

The locked scheme (user decision; MARS + SPIRAL synthesis):

* ``r_t = legal_t + λ·shaping_t``; the player's terminal turn additionally
  carries the §A.0 ``R_terminal``.
* ``RTG(t) = Σ_{s≥t} r_s`` along that player's ordered turn sequence — so
  the terminal reward and all later shaping flow back to earlier turns
  (MARS return-to-go, ≈ GAE(γ=1, λ=1)).
* Role-conditioned EMA baseline (SPIRAL RAE), keyed on
  ``(tiebreak_rank, broad_phase)`` — 9 buckets, auto-collapsing to 3
  (tiebreak_rank only) if any bucket is sparse (§A.5 fallback). ``A = RTG − b``.
* Within-group standardization (GRPO comparison set): ``A ← (A − μ) / σ`` over
  the K rollouts sharing ``(seed, value_chart, trainable_seat)``.
* Actor masking (§A.6): only ``actor_id == "trainable"`` turns get a nonzero
  advantage or feed any statistic.

Processing order is fixed (``group_key, game_id, player_id, round_index,
phase``) so the offline pass — including the order-dependent EMA — is
byte-reproducible.

Scope note: a *true* GRPO group is K rollouts that share
``(seed, value_chart, opponent assignment, seat)`` and differ only by policy
sampling. Whether this offline pass exercises that depends on the corpus: a
1-game-per-seed corpus yields 1-rollout "groups" — the standardization math is
**code-path validated** but not K-group validated. ``summarize`` reports the
realized rollouts-per-group so the distinction is explicit; true K-group
behaviour is a Phase-3 rollout concern (and ``phase1_corpus.sh`` can emit a
K-rollout corpus via ``ROLLOUTS_PER_SEED``).
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Sequence

from .reward import RewardConfig, TurnReward, per_turn_rewards, terminal_breakdowns
from .scorer import ParsedTranscript, decision_tiebreak_order, parse_transcript

_PHASE_ORDER = {"bid": 0, "reveal": 1}


@dataclass(frozen=True)
class AdvantageConfig:
    ema_alpha: float = 0.9  # RAE baseline EMA decay
    min_bucket_count: int = 20  # below this in any 9-bucket → collapse to 3
    num_phases: int = 3  # early / mid / late
    collapse_when_sparse: bool = True
    standardize: bool = True  # within-group (A − μ)/σ
    std_eps: float = 1e-8
    trainable_actor_id: str = "trainable"


@dataclass
class TurnAdvantage:
    game_id: int
    group_key: Hashable
    round_index: int
    player_id: int
    phase: str
    actor_id: str
    trainable: bool
    legal: float
    shaping: float
    terminal_correction: float
    terminal_reward: float
    reward: float  # r_t (incl. R_terminal on the terminal turn)
    rtg: float
    baseline: float
    bucket: tuple  # (tiebreak_rank, broad_phase) — always the full key
    ema_bucket: tuple  # key actually used for the baseline (3- or 9-scheme)
    advantage_raw: float  # RTG − baseline
    advantage: float  # post group-standardization (0.0 if not trainable)
    is_terminal_turn: bool
    parse_valid: bool
    legal_valid: bool
    prompt: str
    completion: str


def broad_phase(round_index: int, num_rounds: int, num_phases: int = 3) -> int:
    """0..num_phases-1 by round thirds (early/mid/late)."""
    if num_rounds <= 0:
        return 0
    idx = int(round_index * num_phases / num_rounds)
    return max(0, min(num_phases - 1, idx))


def tiebreak_rank(tiebreak_order: Sequence[int], player_id: int) -> int:
    """Player's position in the tiebreak order (0 == breaks ties first).

    The rotating *role* the baseline conditions on (§A.5: seat is arbitrary,
    tiebreak rank is not). Callers pass the **decision-time** order
    (``scorer.decision_tiebreak_order``), not the post-winner-update one, so
    the bucket is not outcome-conditioned.
    """
    try:
        return list(tiebreak_order).index(player_id)
    except ValueError:
        return len(tiebreak_order)


def default_group_key(parsed: ParsedTranscript) -> Hashable:
    """§A.7 group identity for the Phase-1 self-play corpus.

    A GRPO group is the K rollouts sharing (seed, value_chart, opponent
    assignment, opponent sampling state, trainable seat). Phase-1 self-play
    has no opponent pool yet (all seats trainable), so (seed, value_chart) +
    the per-row player_id (== trainable seat) is the *complete* identity. The
    value_chart term is mandatory: distinct charts are different games, and
    the corpus script must not reuse a seed range across charts.
    """
    return (parsed.seed, parsed.value_chart)


def _trajectory_turns(
    parsed: ParsedTranscript,
    game_id: int,
    reward_cfg: RewardConfig,
    adv_cfg: AdvantageConfig,
    group_key_fn: Callable[[ParsedTranscript], Hashable],
) -> list[TurnAdvantage]:
    """One game → all turns with r_t, RTG, and (rank, phase) bucket (pre-EMA)."""
    turns = per_turn_rewards(parsed, reward_cfg)
    term = terminal_breakdowns(parsed, reward_cfg)
    num_rounds = max((t.round_index for t in turns), default=-1) + 1
    base_group = group_key_fn(parsed)

    by_player: dict[int, list[TurnReward]] = {}
    for t in turns:
        by_player.setdefault(t.player_id, []).append(t)

    rows: list[TurnAdvantage] = []
    for pid, seq in by_player.items():
        seq.sort(key=lambda t: (t.round_index, _PHASE_ORDER[t.phase]))
        trainable = all(s.actor_id == adv_cfg.trainable_actor_id for s in seq)

        # r_t (+ R_terminal on the terminal turn). ``terminal_correction`` is
        # 0.0 on every turn unless rec-1a is enabled, where it is nonzero only
        # on the terminal turn — so default runs are byte-identical and, when
        # on, the correction telescopes back through RTG to all earlier turns.
        rewards = []
        terminal_rewards = []
        for s in seq:
            terminal_reward = 0.0
            if s.is_terminal_turn and pid in term:
                terminal_reward = term[pid].r_terminal
            value = s.legal + s.shaping + s.terminal_correction + terminal_reward
            rewards.append(value)
            terminal_rewards.append(terminal_reward)

        # Return-to-go: suffix sums.
        rtg = [0.0] * len(rewards)
        acc = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            acc += rewards[i]
            rtg[i] = acc

        for s, r_t, ret, terminal_reward in zip(
            seq, rewards, rtg, terminal_rewards
        ):
            rank = tiebreak_rank(
                decision_tiebreak_order(parsed, s.round_index), pid
            )
            phase = broad_phase(s.round_index, num_rounds, adv_cfg.num_phases)
            rows.append(
                TurnAdvantage(
                    game_id=game_id,
                    group_key=(base_group, pid),
                    round_index=s.round_index,
                    player_id=pid,
                    phase=s.phase,
                    actor_id=s.actor_id,
                    trainable=trainable,
                    legal=s.legal,
                    shaping=s.shaping,
                    terminal_correction=s.terminal_correction,
                    terminal_reward=terminal_reward,
                    reward=r_t,
                    rtg=ret,
                    baseline=0.0,
                    bucket=(rank, phase),
                    ema_bucket=(rank, phase),
                    advantage_raw=0.0,
                    advantage=0.0,
                    is_terminal_turn=s.is_terminal_turn,
                    parse_valid=s.parse_valid,
                    legal_valid=s.legal_valid,
                    prompt=s.prompt,
                    completion=s.completion,
                )
            )
    return rows


def compute_advantages(
    transcripts: Sequence[dict[str, Any]],
    reward_cfg: RewardConfig | None = None,
    adv_cfg: AdvantageConfig | None = None,
    group_key_fn: Callable[[ParsedTranscript], Hashable] = default_group_key,
) -> list[TurnAdvantage]:
    """Full offline pass over a corpus → per-turn advantages (deterministic).

    1. Build per-player return-to-go and (rank, phase) buckets per game.
    2. Pick 9 vs 3 buckets (collapse if any 9-bucket is sparse, §A.5).
    3. Single EMA-baseline sweep in fixed order; ``A_raw = RTG − b``.
    4. Within-group standardize ``A_raw`` (trainable turns only).
    """
    reward_cfg = reward_cfg or RewardConfig()
    adv_cfg = adv_cfg or AdvantageConfig()

    rows: list[TurnAdvantage] = []
    num_players = 0
    for game_id, data in enumerate(transcripts):
        parsed = parse_transcript(data)
        num_players = max(num_players, parsed.num_players)
        rows.extend(
            _trajectory_turns(parsed, game_id, reward_cfg, adv_cfg, group_key_fn)
        )

    # Deterministic order for the order-dependent EMA + reproducible dumps.
    rows.sort(
        key=lambda r: (
            repr(r.group_key),
            r.game_id,
            r.player_id,
            r.round_index,
            _PHASE_ORDER[r.phase],
        )
    )

    trainable = [r for r in rows if r.trainable]

    # --- bucket scheme: collapse 9→3 if ANY expected (rank, phase) bucket is
    # sparse — including buckets with ZERO observations. Enumerating the full
    # structural rank × phase grid (not just observed buckets) is what makes
    # an unobserved role/phase trigger the §A.5 fallback. ---
    full_counts = Counter(r.bucket for r in trainable)
    expected_full = {
        (rank, phase)
        for rank in range(num_players)
        for phase in range(adv_cfg.num_phases)
    }
    use_full = not (
        adv_cfg.collapse_when_sparse
        and trainable
        and any(
            full_counts.get(b, 0) < adv_cfg.min_bucket_count
            for b in expected_full
        )
    )
    for r in trainable:
        r.ema_bucket = r.bucket if use_full else (r.bucket[0],)

    # --- single EMA-baseline sweep (baseline read before its own update) ---
    baseline: dict[tuple, float] = {}
    for r in trainable:
        bk = r.ema_bucket
        if bk not in baseline:
            r.baseline = 0.0  # cold start: no estimate yet
            r.advantage_raw = r.rtg
            baseline[bk] = r.rtg
        else:
            r.baseline = baseline[bk]
            r.advantage_raw = r.rtg - baseline[bk]
            a = adv_cfg.ema_alpha
            baseline[bk] = a * baseline[bk] + (1.0 - a) * r.rtg

    # --- within-group standardization (GRPO comparison set) ---
    if adv_cfg.standardize:
        groups: dict[Hashable, list[TurnAdvantage]] = {}
        for r in trainable:
            groups.setdefault(r.group_key, []).append(r)
        for grp in groups.values():
            vals = [r.advantage_raw for r in grp]
            mu = statistics.fmean(vals)
            sigma = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            denom = sigma if sigma > adv_cfg.std_eps else 1.0
            for r in grp:
                r.advantage = (r.advantage_raw - mu) / denom
    else:
        for r in trainable:
            r.advantage = r.advantage_raw

    # Non-trainable turns stay masked.
    for r in rows:
        if not r.trainable:
            r.advantage = 0.0
    return rows


def summarize(rows: Sequence[TurnAdvantage]) -> dict[str, Any]:
    """Acceptance-criteria report for Phase 1.3 / 1.5."""
    trainable = [r for r in rows if r.trainable]
    bucket_counts = Counter(r.bucket for r in trainable)
    scheme = (
        f"{len(set(r.ema_bucket for r in trainable))} EMA buckets "
        f"(arity {len(trainable[0].ema_bucket) if trainable else 0})"
    )
    groups: dict[Hashable, list[float]] = {}
    group_games: dict[Hashable, set[int]] = {}
    for r in trainable:
        groups.setdefault(r.group_key, []).append(r.advantage)
        group_games.setdefault(r.group_key, set()).add(r.game_id)
    rollouts_per_group = sorted(len(g) for g in group_games.values())
    min_k = rollouts_per_group[0] if rollouts_per_group else 0
    max_k = rollouts_per_group[-1] if rollouts_per_group else 0
    if min_k >= 2:
        mode = f"true K-group validation (every group K≥2, max K={max_k})"
    elif max_k >= 2:
        mode = (
            f"partial K-group validation (max K={max_k} but {sum(1 for n in rollouts_per_group if n < 2)} "
            f"group(s) have <2 rollouts — likely failed/missing rollouts; "
            f"those groups are code-path-only)"
        )
    else:
        mode = (
            "code-path validation only (1 rollout/group; true K-group "
            "grouping is a Phase-3 rollout concern — use "
            "phase1_corpus.sh ROLLOUTS_PER_SEED>1 for a grouped corpus)"
        )
    group_norm = {
        "groups": len(group_games),
        "rollouts_per_group_min": min_k,
        "rollouts_per_group_max": max_k,
        "mode": mode,
    }
    group_stats = {
        repr(k): {
            "n": len(v),
            "mean": round(statistics.fmean(v), 6) if v else 0.0,
            "std": round(statistics.pstdev(v), 6) if len(v) > 1 else 0.0,
        }
        for k, v in sorted(groups.items(), key=lambda kv: repr(kv[0]))
    }
    adv = [r.advantage for r in trainable]
    return {
        "turns_total": len(rows),
        "turns_trainable": len(trainable),
        "ema_scheme": scheme,
        "group_norm": group_norm,
        "buckets": {str(k): c for k, c in sorted(bucket_counts.items())},
        "advantage": {
            "mean": round(statistics.fmean(adv), 6) if adv else 0.0,
            "std": round(statistics.pstdev(adv), 6) if len(adv) > 1 else 0.0,
            "min": round(min(adv), 4) if adv else 0.0,
            "max": round(max(adv), 4) if adv else 0.0,
        },
        "groups": group_stats,
    }
