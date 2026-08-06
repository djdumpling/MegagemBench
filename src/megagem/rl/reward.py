"""Terminal + legal-action + per-turn shaping rewards (Phase 1.2).

Implements the RL plan:

* §A.0 terminal reward: ``clip(margin / SCALE, -1, 1) + WIN_BONUS·1[rank==1]``
  where ``margin_p = final_score_p - mean_{q!=p} final_score_q``. The final
  scores are the engine-true, post-reveal numbers from
  ``final_results.final_scores`` — *not* the mid-game scorer.
* §A.2 legal-action reward: ``0`` if ``parse_valid and legal_valid`` else
  ``-0.5``, for **bids and reveals**, read off the recorded telemetry (never
  re-parsed). Bid/reveal traversal mirrors ``environment.telemetry``. Every
  turn record must carry an explicit ``actor_id`` — a missing one is a hard
  error, never a silent "trainable" (§A.6 fails closed; Caveat 4).
* §A.4 per-turn shaping: round ``r``'s competitive score change
  ``Δscore_p(r) - mean_{q!=p} Δscore_q(r)`` is **split** between its bid and
  reveal turns (Caveat 2). The bid sub-delta values round-``r`` holdings at
  the *pre-reveal* display (the auction's effect); the reveal sub-delta is
  the residual the winner's reveal produced by mutating the value display
  (revaluing everyone's gems). Each turn is graded on its own competitive
  sub-delta, so the strategically pivotal reveal gets a dense signal instead
  of only its legality term. The split is exact: bid + reveal equals the old
  single round-shaping value per (round, player), so the telescoped total —
  and Caveat 1's locked λ — are unchanged. A round with no reveal turn for a
  player keeps that round's full shaping on the bid turn (nothing is lost).

This module produces the per-turn building blocks; return-to-go,
role-conditioned baseline, and normalization live in ``rl.advantage``.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace
from typing import Any, Iterable

from .scorer import (
    ParsedTranscript,
    parse_transcript,
    reconstruct_snapshots,
    score_snapshot,
    score_snapshot_with_display,
    starting_score,
)

_BID = "bid"
_REVEAL = "reveal"
_PHASE_ORDER = {_BID: 0, _REVEAL: 1}


@dataclass(frozen=True)
class RewardConfig:
    scale: float = 21.5  # §A.0 SCALE — locked from Phase 1 corpus (2026-05-16, 1.6)
    win_bonus: float = 0.1  # §A.0 WIN_BONUS
    # §A.4 λ — LOCKED (2026-05-16, Caveat 1). Was 0.2, which makes Σshaping
    # ~4-9x the terminal mass at SCALE=21.5 (median |margin|=10.75): the
    # opposite of "terminal must dominate". 0.01 is the largest λ that keeps
    # shaping/terminal median ≤ 0.33 (measured 0.215) AND within-group
    # rank-corr ≥ 0.8 (measured mean 0.874) on the off-box true-K=8 Phase 1
    # corpus (48 games / 144 trajectories / 18 K-groups) — λ=0.02 fails the
    # mass gate (median 0.43 > 0.33). Confirmed via `python -m
    # megagem.rl.diagnostics --corpus <K=8 corpus>`; the interim 0.01 held.
    # (Self-play caveat 1, recommendation #2.)
    shaping_lambda: float = 0.01
    illegal_penalty: float = -0.5  # §A.2 illegal-action reward (legal == 0.0)
    # --- Phase-3 reward-design experiment knobs.
    # EVERY default below reproduces the locked behavior EXACTLY: a default
    # RewardConfig() is byte-identical to pre-Phase-3. Opt-in only; the
    # overnight diagnostic gate is report-only and never flips these.
    terminal_shape: str = "clip"  # "clip" (locked) | "tanh" (rec 2)
    reveal_shaping_weight: float = 1.0  # 1.0 locked; 0.0 = rec 3 (drop reveal)
    terminal_correction: bool = False  # rec 1a: Σ-shaping→true-margin correction
    correction_scale: float | None = None  # None ⇒ reuse `scale` for the 1a term


@dataclass(frozen=True)
class TerminalBreakdown:
    """§A.0 telemetry; only ``r_terminal`` feeds training."""

    player_id: int
    final_score: int
    margin: float
    rank: int  # 1 == game winner
    is_winner: bool
    r_terminal: float


@dataclass(frozen=True)
class TurnReward:
    """One (player, action) emission — the unit of credit assignment."""

    round_index: int
    player_id: int
    phase: str  # "bid" | "reveal"
    actor_id: str
    legal: float  # 0.0 or illegal_penalty
    shaping: float  # λ-scaled; bid/reveal sub-delta split (§A.4, Caveat 2)
    is_terminal_turn: bool  # player's last emission → R_terminal added here
    parse_valid: bool
    legal_valid: bool
    prompt: str
    completion: str  # raw_response
    terminal_correction: float = 0.0  # rec 1a; nonzero only on the terminal turn


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _terminal_value(margin: float, cfg: RewardConfig) -> float:
    """§A.0 terminal squash — the single source of truth (§7.4).

    ``clip`` (default) reproduces the locked expression byte-identically.
    ``tanh`` (rec 2) is strictly monotone everywhere, so a win-by-60 still
    outranks a win-by-22 within its GRPO group (Problem C). ``win_bonus``
    stays additive *outside* the squash in both cases, exactly as before.
    """
    if cfg.terminal_shape == "clip":
        return _clip(margin / cfg.scale, -1.0, 1.0)
    if cfg.terminal_shape == "tanh":
        return math.tanh(margin / cfg.scale)
    raise ValueError(
        f"unknown terminal_shape {cfg.terminal_shape!r} (expected 'clip'|'tanh')"
    )


def terminal_breakdowns(
    parsed: ParsedTranscript, cfg: RewardConfig
) -> dict[int, TerminalBreakdown]:
    """§A.0, every player. ``rank`` uses ``winner_id`` for the rank-1 tiebreak."""
    scores = {s["player_id"]: int(s["final_score"]) for s in parsed.final_scores}
    out: dict[int, TerminalBreakdown] = {}
    for pid, fs in scores.items():
        others = [v for q, v in scores.items() if q != pid]
        margin = fs - (statistics.fmean(others) if others else 0.0)
        is_winner = pid == parsed.winner_id
        # Rank by score desc; the game winner is forced to rank 1 (engine
        # already resolved score ties via tiebreak_order).
        higher = sum(1 for q, v in scores.items() if v > fs)
        rank = 1 if is_winner else max(2, higher + 1)
        r_terminal = _terminal_value(margin, cfg) + (
            cfg.win_bonus if is_winner else 0.0
        )
        out[pid] = TerminalBreakdown(
            player_id=pid,
            final_score=fs,
            margin=margin,
            rank=rank,
            is_winner=is_winner,
            r_terminal=r_terminal,
        )
    return out


def _legal_reward(record: dict[str, Any], cfg: RewardConfig) -> float:
    ok = bool(record.get("parse_valid")) and bool(record.get("legal_valid"))
    return 0.0 if ok else cfg.illegal_penalty


def _competitive(values: dict[int, float], n: int, lam: float) -> dict[int, float]:
    """λ·(value_p − mean_{q≠p} value_q) for every player (§A.4 competitive form)."""
    out: dict[int, float] = {}
    for pid in range(n):
        others = [values[q] for q in range(n) if q != pid]
        out[pid] = lam * (values[pid] - (statistics.fmean(others) if others else 0.0))
    return out


def round_shaping_components(
    parsed: ParsedTranscript, cfg: RewardConfig
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """λ-scaled §A.4 shaping split into (bid, reveal) per (round, player).

    Round ``r``'s competitive score change is decomposed into the part the
    **auction** produced and the part the winner's **reveal** produced
    (Caveat 2). Let ``S0_p`` = p's score at the end of round r-1 (start
    ``starting_score``), ``S1_p`` = p's score at the end of round r (the
    existing post-reveal scorer), and ``Smid_p`` = round-r holdings scored at
    the *pre-reveal* display (= round r-1's display, empty before round 1).
    Then

        bid sub-delta    = Smid_p − S0_p     # auction outcome, display fixed
        reveal sub-delta = S1_p   − Smid_p   # display mutation, holdings fixed

    and ``bid + reveal = S1_p − S0_p = Δscore_p(r)`` exactly. Each sub-delta
    is turned into the §A.4 competitive form (own − mean opponents'). Summing
    the two components reproduces :func:`round_shaping` per (r, pid) up to
    float round-off, so the telescoped total and Caveat 1's locked λ are
    unchanged — the split only re-attributes a round's shaping between its
    bid and reveal turns.

    The pre-reveal display is round r-1's recorded ``value_display`` (only a
    reveal mutates the display between round boundaries, so r-1's post-reveal
    display *is* round r's pre-reveal display). No value chart, no hands.
    """
    snaps = reconstruct_snapshots(parsed)
    n = parsed.num_players
    base = float(starting_score(n))

    # post = score at the round's end (post-reveal display, == score_snapshot);
    # pre  = the same round-r holdings scored at the *pre-reveal* display.
    post: dict[tuple[int, int], float] = {}
    pre: dict[tuple[int, int], float] = {}
    for r_idx in range(len(parsed.rounds)):
        for pid in range(n):
            snap = snaps.get((r_idx, pid))
            if snap is None:  # defensive: reconstruct fills every (r, pid)
                post[(r_idx, pid)] = base
                pre[(r_idx, pid)] = base
                continue
            post[(r_idx, pid)] = float(score_snapshot(snap)["final_score"])
            if r_idx == 0:
                pre_display: dict[str, int] = {}  # nothing revealed pre-round-1
            else:
                prev_snap = snaps.get((r_idx - 1, pid))
                pre_display = (
                    prev_snap.display_value_per_gem if prev_snap else {}
                )
            pre[(r_idx, pid)] = float(
                score_snapshot_with_display(snap, pre_display)["final_score"]
            )

    bid_out: dict[tuple[int, int], float] = {}
    rev_out: dict[tuple[int, int], float] = {}
    lam = cfg.shaping_lambda
    for r_idx in range(len(parsed.rounds)):
        prev = (
            {pid: post[(r_idx - 1, pid)] for pid in range(n)}
            if r_idx > 0
            else {pid: base for pid in range(n)}
        )
        bid_sub = {pid: pre[(r_idx, pid)] - prev[pid] for pid in range(n)}
        rev_sub = {pid: post[(r_idx, pid)] - pre[(r_idx, pid)] for pid in range(n)}
        for pid, v in _competitive(bid_sub, n, lam).items():
            bid_out[(r_idx, pid)] = v
        # rec 3: weight the reveal sub-delta. w==1.0 (locked) is byte-identical
        # (x*1.0 is exact in IEEE-754); w==0.0 drops the structurally
        # anti-correlated reveal proxy while keeping the (round,player) key
        # present so round_shaping / per_turn_rewards fold logic is untouched.
        w_rev = cfg.reveal_shaping_weight
        for pid, v in _competitive(rev_sub, n, lam).items():
            rev_out[(r_idx, pid)] = w_rev * v
    return bid_out, rev_out


def round_shaping(
    parsed: ParsedTranscript, cfg: RewardConfig
) -> dict[tuple[int, int], float]:
    """λ-scaled §A.4 shaping per (round_index, player_id) — the round total.

    ``Δscore_p(r) = score_p(end r) - score_p(end r-1)`` with
    ``score_p(end -1) = starting_score``; the result is
    ``λ·(Δscore_p(r) − mean_{q≠p} Δscore_q(r))``. Defined as the sum of the
    bid and reveal components (:func:`round_shaping_components`) so the
    diagnostics / λ-lock path measures exactly what training attributes per
    turn — single source of truth, no harness/production drift (§7.4).
    """
    bid, rev = round_shaping_components(parsed, cfg)
    return {k: bid[k] + rev.get(k, 0.0) for k in bid}


def _iter_records(
    parsed: ParsedTranscript,
) -> Iterable[tuple[int, int, str, dict[str, Any]]]:
    """(round_index, player_id, phase, record) for every bid and reveal."""
    for r_idx, rnd in enumerate(parsed.rounds):
        for rec in rnd.get("players", []):
            pid = rec.get("player_id")
            yield r_idx, pid, _BID, rec
            reveal = rec.get("reveal")
            if isinstance(reveal, dict):
                yield r_idx, pid, _REVEAL, reveal


def _require_actor_id(
    rec: dict[str, Any],
    parsed: ParsedTranscript,
    round_index: int,
    player_id: Any,
    phase: str,
) -> str:
    """Caveat 4 — fail closed. Every turn record MUST carry a non-empty actor_id.

    The §A.6 actor mask keeps frozen-opponent (snapshot/heuristic) tokens out
    of the policy gradient. Silently defaulting a missing field to
    ``"trainable"`` is correct only because Phase-1 self-play is all-trainable;
    in Phase 3 a writer that forgets the field would un-mask opponent tokens
    into the gradient with no error and no test catching it. Refuse to guess.
    """
    actor_id = rec.get("actor_id")
    if not isinstance(actor_id, str) or not actor_id:
        raise ValueError(
            f"missing/empty actor_id on the {phase} turn of player "
            f"{player_id} in round {round_index} (seed={parsed.seed}, "
            f"chart={parsed.value_chart}); got {actor_id!r}. Every schema-v3 "
            "turn record must carry an explicit actor_id — the §A.6 actor "
            "mask fails closed rather than guessing the actor. "
            "Re-run rollouts with a writer that stamps actor_id."
        )
    return actor_id


def per_turn_rewards(
    parsed: ParsedTranscript, cfg: RewardConfig
) -> list[TurnReward]:
    """Ordered per-turn rewards (legal + λ·shaping); terminal added in advantage.

    Order: (round_index, player_id, phase) with bid before reveal — the same
    key ``rl.advantage`` sorts on for deterministic processing. The round's
    shaping is split across its bid and reveal turns (§A.4, Caveat 2); a
    player with no reveal turn that round keeps the round's full shaping on
    its bid turn so the per-(round, player) total is preserved exactly.
    """
    bid_shaping, reveal_shaping = round_shaping_components(parsed, cfg)
    records = list(_iter_records(parsed))
    has_reveal_turn = {
        (r, pid) for r, pid, ph, _ in records if ph == _REVEAL
    }

    # rec 1a — terminal-turn correction so the trajectory shaping sum
    # telescopes to the engine-true margin instead of the biased mid-game
    # proxy: with it, Σ(shaping + terminal_correction) == (λ/Sc)·true_margin
    # (const ≡ 0; the starting-score offset already cancels in the
    # competitive own−mean(others) form). ``s_raw`` is recomputed from a λ=1
    # round_shaping on THIS cfg (§7.4 single source — never t.shaping/lam,
    # which divides by zero at λ=0), so the identity stays exact under
    # reveal_shaping_weight ∈ {1.0, 0.0} (Rev 7: recomputed against the
    # reveal-zeroed sum, not bolted on). This is objective-changing, not PBRS.
    correction_by_pid: dict[int, float] = {}
    if cfg.terminal_correction:
        term_bd = terminal_breakdowns(parsed, cfg)
        raw_cfg = replace(cfg, shaping_lambda=1.0)
        s_raw_by_pid: dict[int, float] = {}
        for (_r, p_i), v in round_shaping(parsed, raw_cfg).items():
            s_raw_by_pid[p_i] = s_raw_by_pid.get(p_i, 0.0) + v
        lam = cfg.shaping_lambda
        sc = (
            cfg.correction_scale
            if cfg.correction_scale is not None
            else cfg.scale
        )
        for p_i, bd in term_bd.items():
            correction_by_pid[p_i] = lam * (bd.margin / sc) - lam * (
                s_raw_by_pid.get(p_i, 0.0)
            )

    # Each player's last emission across the game gets the terminal reward.
    last_turn: dict[int, tuple[int, int]] = {}
    for r_idx, pid, phase, _rec in records:
        key = (r_idx, _PHASE_ORDER[phase])
        if pid not in last_turn or key > last_turn[pid]:
            last_turn[pid] = key

    turns: list[TurnReward] = []
    for r_idx, pid, phase, rec in records:
        is_terminal = last_turn.get(pid) == (r_idx, _PHASE_ORDER[phase])
        if phase == _BID:
            shaping = bid_shaping.get((r_idx, pid), 0.0)
            if (r_idx, pid) not in has_reveal_turn:
                # No reveal turn for this player this round (non-winner, or a
                # loan/investment/no-reveal round): the round's reveal-side
                # shaping has nowhere else to go, so keep it on the bid turn.
                # Preserves the exact per-(round, player) total.
                shaping += reveal_shaping.get((r_idx, pid), 0.0)
        else:  # _REVEAL — graded on the display mutation it caused
            shaping = reveal_shaping.get((r_idx, pid), 0.0)
        turns.append(
            TurnReward(
                round_index=r_idx,
                player_id=pid,
                phase=phase,
                actor_id=_require_actor_id(rec, parsed, r_idx, pid, phase),
                legal=_legal_reward(rec, cfg),
                shaping=shaping,
                is_terminal_turn=is_terminal,
                parse_valid=bool(rec.get("parse_valid")),
                legal_valid=bool(rec.get("legal_valid")),
                prompt=rec.get("prompt", ""),
                completion=rec.get("raw_response", ""),
                terminal_correction=(
                    correction_by_pid.get(pid, 0.0) if is_terminal else 0.0
                ),
            )
        )
    turns.sort(key=lambda t: (t.round_index, t.player_id, _PHASE_ORDER[t.phase]))
    return turns


def assert_corpus_actor_ids(transcripts: Iterable[dict[str, Any]]) -> int:
    """Caveat 4 — corpus schema gate. Raise unless EVERY bid/reveal turn in
    every transcript carries an explicit, non-empty ``actor_id``.

    Called before any training-row export so a corpus that cannot be safely
    actor-masked is rejected up front (before Phase-3 GPU spend) with the
    exact offending locus, rather than silently un-masking opponent tokens.
    Returns the number of turn records validated (so callers can log it).
    """
    checked = 0
    for gid, data in enumerate(transcripts):
        parsed = parse_transcript(data)
        for r_idx, pid, phase, rec in _iter_records(parsed):
            try:
                _require_actor_id(rec, parsed, r_idx, pid, phase)
            except ValueError as exc:
                raise ValueError(f"corpus transcript #{gid}: {exc}") from None
            checked += 1
    return checked


def calibrate_scale(
    transcripts: Iterable[dict[str, Any]], cfg: RewardConfig | None = None
) -> dict[str, Any]:
    """§5 Q4: recommend SCALE so typical ``|R_terminal| ≈ 0.5``, rare ``≈ 1.0``.

    Aggregates ``|margin|`` over every player in every game. Recommends
    ``SCALE = 2 · median(|margin|)`` (median maps to |R_terminal| ≈ 0.5) and
    reports the ±1 clip rate that choice implies.
    """
    cfg = cfg or RewardConfig()
    margins: list[float] = []
    games = 0
    for data in transcripts:
        games += 1
        parsed = parse_transcript(data)
        for bd in terminal_breakdowns(parsed, cfg).values():
            margins.append(abs(bd.margin))
    if not margins:
        return {"games": games, "note": "no transcripts"}

    margins.sort()
    median = statistics.median(margins)
    rms = (statistics.fmean(m * m for m in margins)) ** 0.5

    def pct(p: float) -> float:
        return margins[min(len(margins) - 1, int(p * len(margins)))]

    abs_margin = {
        "median": round(median, 2),
        "rms": round(rms, 2),
        "p90": round(pct(0.90), 2),
        "p99": round(pct(0.99), 2),
        "max": round(margins[-1], 2),
    }
    if cfg.terminal_shape == "tanh":
        # rec 2 calibration: tanh-specific. Target tanh(median/SCALE)=0.5 ⇒
        # SCALE_tanh = median/atanh(0.5) ≈ 1.82·median (NOT the linear
        # 2·median). Reusing 21.5 under tanh under-centers:
        # tanh(10.75/21.5)=tanh(0.5)=0.462, not 0.5.
        recommended = round(median / math.atanh(0.5), 1) or 1.0
        soft_thresh = math.atanh(0.96)  # |tanh| ≥ 0.96 ⇒ soft-saturated tail
        saturated = sum(
            1 for m in margins if m / recommended >= soft_thresh
        ) / len(margins)
        return {
            "games": games,
            "player_margins": len(margins),
            "abs_margin": abs_margin,
            "terminal_shape": "tanh",
            "current_scale": cfg.scale,
            "recommended_scale": recommended,
            "implied_soft_saturation_rate_at_recommended": round(saturated, 3),
            "rationale": (
                "SCALE_tanh = median/atanh(0.5) ≈ 1.82·median → "
                "tanh(median/SCALE) ≈ 0.5; |tanh| ≥ 0.96 = soft-saturated"
            ),
        }
    recommended = round(2.0 * median, 1) or 1.0
    clipped = sum(1 for m in margins if m / recommended >= 1.0) / len(margins)
    return {
        "games": games,
        "player_margins": len(margins),
        "abs_margin": abs_margin,
        "current_scale": cfg.scale,
        "recommended_scale": recommended,
        "implied_clip_rate_at_recommended": round(clipped, 3),
        "rationale": "SCALE = 2·median(|margin|) → typical |R_terminal| ≈ 0.5",
    }
