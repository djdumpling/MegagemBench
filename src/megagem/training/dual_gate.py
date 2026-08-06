"""Dual-gate SPEND criterion for phase-3 GRPO runs (pure logic, no Modal).

Single-gate §3.6 (paired eval vs the training heuristic, ci_low > +2) is
permeable to opponent-overfitting: a policy can pass §3.6 with an
opponent-specific exploit yet transfer to only baseline-level win-rate against
a held-out opponent (Gemini 3 Flash). The dual gate therefore requires a
vs-Flash heldout: with ``flash_primary=True`` (default) the Flash gate is the
BINDING spend criterion and §3.6 is informational — it can no longer veto a
policy that transfers, nor rubber-stamp one that doesn't. With
``flash_primary=False`` (legacy AND) both gates must pass.

Statistical gate: when ``sft_baseline_wr > 0`` the absolute win-rate threshold
is combined with a one-sided z-test of H0: RL_WR = SFT_baseline_WR against
H1: RL_WR > SFT_baseline_WR. panel_eval plays N seeds x 3 seat rotations; the
3 rotations within a seed share game-level randomness and are positively
correlated, so an iid SE over n_games would over-count independent trials.
The conservative bound treats each SEED as one trial (full intracluster
correlation): ``n_eff = num_seeds``. At n_seeds=60, SFT=0.30, alpha=0.05 the
gate requires observed WR ~= 0.40 (vs ~0.36 under iid).

Operator interpretation (locked 2026-05-27): observed WR ~35% at n_seeds=60 is
PROMISING but not a spend-pass; WR >= ~40% is a plausible strict pass under
the conservative bound. To make +5pp passable, raise ``flash_seeds`` to ~100+
or implement a seed-cluster bootstrap once panel_eval surfaces per-seed win
counts — do not lower the gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def z_critical(alpha: float) -> float:
    """One-sided z critical value from the inverse normal CDF.

    alpha=0.05 -> ~1.645; 0.025 -> 1.960; 0.01 -> 2.326. Uses scipy when
    available, else the Abramowitz & Stegun 26.2.23 approximation (~5e-4).
    """
    try:
        from scipy.stats import norm

        return float(norm.ppf(1.0 - alpha))
    except Exception:
        p = 1.0 - alpha
        t = math.sqrt(math.log(1.0 / ((1.0 - p) ** 2)))
        c = [2.515517, 0.802853, 0.010328]
        d = [1.432788, 0.189269, 0.001308]
        num = c[0] + c[1] * t + c[2] * t * t
        den = 1 + d[0] * t + d[1] * t * t + d[2] * t * t * t
        return t - num / den


@dataclass
class FlashGateAssessment:
    """Outcome of the vs-Flash heldout gate plus the overall spend verdict."""

    win_rate: float | None
    delta: float | None
    n_games: int | None
    expected_games: int
    games_ok: bool
    z_obs: float | None
    z_crit: float | None
    n_eff: int | None
    stat_pass: bool | None          # None => stat test disabled
    flash_pass: bool
    spend: bool
    lines: list[str] = field(default_factory=list)


def assess_flash_gate(
    panel_result: dict,
    *,
    gate_36_pass: bool,
    flash_primary: bool,
    flash_seeds: int,
    flash_threshold: float,
    sft_baseline_wr: float,
    significance_alpha: float,
) -> FlashGateAssessment:
    """Assess a completed panel_eval result dict and render the report lines.

    The caller is responsible for transport-level failures (aborted runs,
    nonzero rc) — this function assumes ``panel_result`` is a trusted,
    completed panel_eval payload with a ``win_rates`` list.
    """
    expected_games = int(flash_seeds) * 3
    flash_row = next(
        (r for r in (panel_result.get("win_rates") or [])
         if r.get("panel") == "vs_flash"),
        None,
    )
    flash_wr = flash_row.get("win_rate") if flash_row else None
    flash_delta = flash_row.get("mean_qwen_delta") if flash_row else None
    flash_n_games = flash_row.get("n_games") if flash_row else None

    # 3-seat rotation => exactly num_seeds * 3 games. A short panel_eval
    # (failed games, resumability drops) is partial -> refuse.
    games_ok = (isinstance(flash_n_games, int)
                and flash_n_games == expected_games)

    stat_pass: bool | None = None
    z_obs = None
    z_crit = None
    n_eff_used = None
    if (sft_baseline_wr > 0
            and isinstance(flash_wr, (int, float))
            and isinstance(flash_n_games, int)
            and flash_n_games > 0):
        p0 = float(sft_baseline_wr)
        # Conservative: n_eff = num_seeds, NOT n_games (see module docstring).
        n_eff_used = int(flash_seeds)
        se0 = math.sqrt(p0 * (1.0 - p0) / n_eff_used)
        z_obs = ((flash_wr - p0) / se0) if se0 > 0 else None
        z_crit = z_critical(significance_alpha)
        stat_pass = (z_obs is not None and z_obs > z_crit)

    flash_pass = (
        isinstance(flash_wr, (int, float))
        and flash_wr > flash_threshold
        and games_ok
        # When the stat test is enabled, BOTH the absolute threshold AND the
        # significance test must clear. None => disabled.
        and (stat_pass is None or stat_pass))

    lines: list[str] = []
    wr_s = (f"{flash_wr*100:.1f}%"
            if isinstance(flash_wr, (int, float)) else "n/a")
    ng_s = str(flash_n_games) if flash_n_games is not None else "n/a"
    lines.append(f"[dual-gate] vs Flash heldout: win_rate={wr_s}  "
                 f"delta={flash_delta}  n_games={ng_s}/{expected_games}")

    spend = False
    if not games_ok:
        lines.append(f"[dual-gate] Flash gate FAIL — partial heldout "
                     f"(got {ng_s} games, expected {expected_games}). "
                     f"OVERALL: NO-SPEND ✗")
        return FlashGateAssessment(
            win_rate=flash_wr, delta=flash_delta, n_games=flash_n_games,
            expected_games=expected_games, games_ok=False, z_obs=z_obs,
            z_crit=z_crit, n_eff=n_eff_used, stat_pass=stat_pass,
            flash_pass=False, spend=False, lines=lines)

    abs_s = (f"abs(>{flash_threshold*100:.0f}%)="
             f"{'✓' if flash_wr > flash_threshold else '✗'}")
    if stat_pass is None:
        stat_s = "ztest=disabled"
    else:
        stat_s = (
            f"ztest(SFT={sft_baseline_wr*100:.0f}%,"
            f"α={significance_alpha},"
            f"n_eff={n_eff_used}=seeds): "
            f"z={z_obs:.2f} vs z_crit={z_crit:.2f} "
            f"{'✓' if stat_pass else '✗'}  "
            f"[CAVEAT: n_eff=num_seeds is cluster-conservative; "
            f"true effective n is between {flash_seeds} "
            f"and {expected_games} depending on the 3-rotation "
            f"intracluster correlation]"
        )
    lines.append(f"[dual-gate] Flash gate {abs_s}  {stat_s}  → "
                 f"{'PASS' if flash_pass else 'FAIL'}")

    # Disagreement note — the two gates pointing opposite ways is itself the
    # signal, so surface it explicitly.
    if gate_36_pass and not flash_pass:
        lines.append("[dual-gate] NOTE: §3.6(heuristic)=PASS but Flash=FAIL — "
                     "classic opponent-overfitting signature (the heuristic "
                     "gate was fooled).")
    elif flash_pass and not gate_36_pass:
        lines.append("[dual-gate] NOTE: Flash=PASS but §3.6(heuristic)=FAIL — "
                     "the heuristic gate would have wrongly vetoed a policy "
                     "that transfers.")

    # OVERALL verdict. flash-primary => Flash binds, §3.6 is informational;
    # legacy => both must pass (AND).
    spend = flash_pass if flash_primary else (gate_36_pass and flash_pass)
    if spend:
        basis = ("Flash primary; §3.6 informational" if flash_primary
                 else "both gates pass")
        lines.append(f"\n[dual-gate] OVERALL: SPEND ✓ ({basis})")
    else:
        parts = [
            f"§3.6={'✓' if gate_36_pass else '✗'}"
            f"{'(info)' if flash_primary else ''}",
            f"Flash={'✓' if flash_pass else '✗'}"
            f"{'(binding)' if flash_primary else ''}",
        ]
        lines.append(f"\n[dual-gate] OVERALL: NO-SPEND ✗ ({'  '.join(parts)})")

    return FlashGateAssessment(
        win_rate=flash_wr, delta=flash_delta, n_games=flash_n_games,
        expected_games=expected_games, games_ok=games_ok, z_obs=z_obs,
        z_crit=z_crit, n_eff=n_eff_used, stat_pass=stat_pass,
        flash_pass=flash_pass, spend=spend, lines=lines)
