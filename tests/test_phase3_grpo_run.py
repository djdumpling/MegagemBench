"""Phase-3 GRPO run — CPU contract tests (no GPU/network).

Two layers: (1) pure unit tests of the stationary heuristic opponent
(legality on REAL stored prompts through the REAL parser, determinism,
OpenAI envelope) and the paired-bootstrap +2 gate; (2) subprocess
`--dry-run` integration of the real driver/eval entrypoints (their true
unattended form) asserting the rich glue contract: actor-mask on bid AND
reveal, §A.7 single K-group per seed, ENGINE terminal-score extraction
(`final_scores`, NOT the mid-game proxy), and that the §3.6 gate is not a
rubber stamp.
"""

from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

import pytest

from _rl_fixtures import REPO_ROOT
from megagem.training.heuristic_endpoint import (
    _chat_completion_envelope,
    _extract_section,
    decide,
)

# scripts/training holds the drivers under test (phase3_grpo.py, phase3_eval.py,
# megagem_steps.py); tests/conftest.py puts it on sys.path.
P3 = REPO_ROOT / "scripts" / "training"

CORPUS = sorted(
    glob.glob(str(REPO_ROOT / "results/phase1_corpus/chart_A/**/*.json"),
              recursive=True)
)
# Only the tests that read real stored transcripts need the corpus (results/ is
# gitignored scratch; a clean checkout has none). The source-contract tests, the
# pure-function tests and the subprocess --dry-runs all run without it.
requires_corpus = pytest.mark.skipif(
    not CORPUS, reason="phase1 corpus not present (gitignored)")


def _prompts():
    """(bid_prompts, reveal_prompts) harvested from real schema-v3
    transcripts — the exact strings run_game would send the opponent."""
    bids, reveals = [], []
    for f in CORPUS[:8]:
        g = json.load(open(f))
        for rnd in g.get("rounds", []):
            for rec in rnd.get("players", []):
                if rec.get("prompt"):
                    bids.append(rec["prompt"])
                rv = rec.get("reveal")
                if isinstance(rv, dict) and rv.get("prompt"):
                    reveals.append(rv["prompt"])
    return bids, reveals


# --------------------------------------------------------------------------- #
# Heuristic opponent — the legality contract that makes it a valid player.     #
# --------------------------------------------------------------------------- #
@requires_corpus
def test_heuristic_bids_are_parseable_and_legal():
    from megagem.game.actions import parse_bid

    bids, _ = _prompts()
    assert bids, "no bid prompts in corpus"
    for p in bids:
        out = decide([{"role": "user", "content": p}])
        pb = parse_bid(out)
        assert pb.valid, f"unparseable heuristic bid: {out!r}"
        auction = _extract_section(p, "current_auction") or {}
        max_bid = int(auction.get("max_bid_for_you") or 0)
        # legal range is [0, max_bid_for_you]; non-Treasure ⇒ decline (0)
        assert 0 <= pb.bid <= max_bid, (pb.bid, max_bid)
        reason = str(auction.get("bid_limit_reason") or "")
        if "Loan" in reason or "Investment" in reason:
            assert pb.bid == 0  # deterministic decline


@requires_corpus
def test_heuristic_reveals_are_parseable_and_in_hand():
    from megagem.game.actions import parse_reveal

    _, reveals = _prompts()
    assert reveals, "no reveal prompts in corpus"
    for p in reveals:
        out = decide([{"role": "user", "content": p}])
        pr = parse_reveal(out)
        assert pr.valid, f"unparseable heuristic reveal: {out!r}"
        hand = (_extract_section(p, "your_private_hand") or {}).get("cards") or []
        if hand:
            assert pr.gem_color in hand, (pr.gem_color, hand)
            assert pr.gem_color == sorted(hand)[0]  # deterministic rule


@requires_corpus
def test_heuristic_is_stationary_and_deterministic():
    """The whole reason for heuristic over self-mirror: no opponent variance
    in the K=8 group. Same prompt ⇒ byte-identical action, every time."""
    bids, reveals = _prompts()
    for p in (bids[:5] + reveals[:5]):
        outs = {decide([{"role": "user", "content": p}]) for _ in range(8)}
        assert len(outs) == 1, f"non-stationary opponent: {outs}"


def test_heuristic_openai_envelope_shape():
    env = _chat_completion_envelope('{"bid": 3}', "megagem/heuristic-v1")
    assert env["object"] == "chat.completion"
    msg = env["choices"][0]["message"]
    assert msg["role"] == "assistant" and msg["content"] == '{"bid": 3}'
    assert env["choices"][0]["finish_reason"] == "stop"


def test_heuristic_unknown_prompt_defaults_to_legal_bid_zero():
    """No action_request section (malformed) ⇒ a parseable, trivially-legal
    bid 0, never an exception (a rollout must never hang on the opponent)."""
    from megagem.game.actions import parse_bid

    out = decide([{"role": "user", "content": "garbage with no sections"}])
    assert parse_bid(out).valid and parse_bid(out).bid == 0


# --------------------------------------------------------------------------- #
# §3.6 paired-bootstrap gate logic (reused dependency-free CI).               #
# --------------------------------------------------------------------------- #
def test_paired_bootstrap_plus2_gate_is_not_a_rubber_stamp():
    from megagem.training.grpo_harness import paired_bootstrap_ci

    strong = paired_bootstrap_ci([5.0, 6.0, 4.5, 7.0, 5.5, 4.0, 6.5, 5.0],
                                 n_resamples=3000)
    weak = paired_bootstrap_ci([0.2, -0.1, 0.0, 0.3, -0.2, 0.1],
                               n_resamples=3000)
    positive_small = paired_bootstrap_ci([1.0, 1.2, 0.8, 1.1, 0.9, 1.0],
                                          n_resamples=3000)
    # gate is ci_low > +2 (NOT >0): a real win passes; noise and a small but
    # real +1 improvement both correctly FAIL the +2 bar.
    assert strong["ci_low"] > 2.0
    assert not (weak["ci_low"] > 2.0)
    assert not (positive_small["ci_low"] > 2.0)


# --------------------------------------------------------------------------- #
# Driver `_health` gate honesty (a failed gate must be reported, not hidden).  #
# --------------------------------------------------------------------------- #
def test_health_gates_flag_pathologies_honestly():
    import importlib

    mod = importlib.import_module("phase3_grpo")

    class A:  # minimal args carrier
        kl_max = 0.5

    good = {"kl": [0.1, 0.2, 0.15], "loss": [1.0, 0.9], "grad_norm": [0.5]}
    rows = [{"precomputed_advantage": v, "completion": "x" * 10}
            for v in (-1.0, 0.5, 1.5, -0.7)]
    assert mod._health(good, rows, A)["all_pass"] is True

    blew = {"kl": [0.1, 9.9], "loss": [1.0], "grad_norm": [0.5]}
    assert mod._health(blew, rows, A)["gates"]["kl_bounded"] is False

    nan = {"kl": [0.1], "loss": [float("nan")], "grad_norm": [0.5]}
    assert mod._health(nan, rows, A)["gates"]["nan_free"] is False

    flat = [{"precomputed_advantage": 0.0, "completion": "x"} for _ in range(4)]
    h = mod._health(good, flat, A)
    assert h["gates"]["advantage_variance_nondegenerate"] is False
    assert h["all_pass"] is False  # degenerate σ is surfaced, not swallowed


# --------------------------------------------------------------------------- #
# Subprocess `--dry-run` integration — the real entrypoints' true form.       #
# --------------------------------------------------------------------------- #
def _dry(script: str, out: Path) -> dict:
    r = subprocess.run(
        [sys.executable, f"scripts/training/{script}", "--dry-run",
         "--output", str(out)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(out.read_text())


def test_driver_dry_run_glue_contract(tmp_path):
    rep = _dry("phase3_grpo.py", tmp_path / "d.json")
    assert rep["status"] == "PASS"
    w = rep["wiring_checks"]
    assert w["post_tag_top_level"] and w["post_tag_nested_reveal"]
    assert w["all_rows_trainable_seat"] and w["export_contract_ok"]
    # §A.7: K rollouts at one seed ⇒ exactly one (seed,chart,seat) group
    assert w["a7_single_kgroup_for_one_seed"]
    assert len(rep["group_keys"]) == 1
    lg = rep["heuristic_legality"]
    assert rep["heuristic_legal_ok"] is True
    if lg.get("skipped"):
        assert lg["bid_checked"] == 0 and lg["reveal_checked"] == 0
    else:
        assert lg["bid_checked"] > 0 and lg["reveal_checked"] > 0
        assert lg["illegal"] == [] and lg["unparsed"] == []


def test_eval_dry_run_gate_and_pairing(tmp_path):
    rep = _dry("phase3_eval.py", tmp_path / "e.json")
    assert rep["status"] == "PASS"
    assert rep["terminal_score_ok"] is True
    # the #1 fix: must read engine final_scores, not the mid-game proxy
    assert rep["uses_engine_final_scores_not_proxy"] is True
    assert rep["gate_positive_set_passes"] is True
    assert rep["gate_zero_set_fails"] is True
    assert rep["pairing_isolates_policy"] is True
    assert rep["threshold"] == 2.0


# --------------------------------------------------------------------------- #
# Regression guards for the four review fixes (#1–#4).                         #
# --------------------------------------------------------------------------- #
@requires_corpus
def test_fix1_eval_uses_engine_terminal_score_not_proxy():
    """`_terminal_score` == engine `final_scores[pid].final_score` (hands
    revealed) and is NOT the hand-free mid-game proxy `score_at` (which
    diverges on most trajectories — using it silently breaks §3.6)."""
    from phase3_eval import _terminal_score
    from megagem.rl.scorer import parse_transcript, score_at

    differ = total = 0
    for f in CORPUS:
        g = json.load(open(f))
        raw = {e["player_id"]: e["final_score"]
               for e in g["final_results"]["final_scores"]}
        p = parse_transcript(g)
        for pid in range(p.num_players):
            total += 1
            assert _terminal_score(g, pid) == raw[pid]      # engine, verbatim
            if _terminal_score(g, pid) != score_at(p, p.num_rounds - 1, pid):
                differ += 1
    # confirmed 131/144 on this corpus — assert the proxy is genuinely
    # different so a silent regression back to score_at fails loudly.
    assert differ > total // 2, (differ, total)


def test_fix3_driver_is_single_persistent_trainer():
    """#3: no per-segment trainer rebuild. The driver exposes a dynamic
    on-policy rollout + a `--rows-per-gen` fixed-N dataset, and NO
    `--refresh-every` segment knob (which implied optimizer reset)."""
    import importlib

    m = importlib.import_module("phase3_grpo")
    assert hasattr(m, "make_onpolicy_rollout_func")
    src = (P3 / "phase3_grpo.py").read_text()
    assert "for seg in range(n_segments)" not in src   # the old reset loop
    assert "SINGLE persistent trainer" in src
    assert "--rows-per-gen" in src and "--refresh-every" not in src


def test_fix4_a7_sampling_is_wired_through_run_game():
    """#4: external rollouts are actually §A.7-sampled. `run_game` accepts a
    `caller_api_params` passthrough and the driver sends T=1.0/top-p=0.95."""
    import importlib
    import inspect

    from megagem import rollout as run_game
    sig = inspect.signature(run_game.run_game)
    assert "caller_api_params" in sig.parameters

    m = importlib.import_module("phase3_grpo")
    assert m.A7_SAMPLING == {"temperature": 1.0, "top_p": 0.95}
    src = (P3 / "phase3_grpo.py").read_text()
    # the trainable seat is still §A.7-sampled; opponents now play greedy via
    # the per-seat caller_api_params list (see the §3.3 opponent-pool tests).
    assert "sampling or A7_SAMPLING" in src
    assert "caller_api_params=caller_api_params" in src


def test_fix_vllm_lora_endpoint_is_under_v1():
    """Blocking: vLLM 0.10.2 serves (un)load_lora_adapter under `/v1/`. The
    helper must NOT strip `/v1` (a prior version did → 404 on-box)."""
    import megagem.training.adapter_sync as ADP

    assert ADP._lora_api_base("http://h:8000/v1") == "http://h:8000/v1"
    assert ADP._lora_api_base("http://h:8000/v1/") == "http://h:8000/v1"
    assert not hasattr(ADP, "_admin_base")  # the /v1-stripping bug is gone
    src = (REPO_ROOT / "src/megagem/training/adapter_sync.py").read_text()
    assert "/load_lora_adapter" in src and "/unload_lora_adapter" in src
    # the POST targets must be the /v1-based api, never a stripped root
    assert "f\"{api}/load_lora_adapter\"" in src
    assert "f\"{api}/unload_lora_adapter\"" in src


def test_fix_driver_hard_aborts_on_failed_adapter_sync():
    """Blocking: a failed adapter push must HARD-ABORT (else training rolls
    against stale/base weights while GRPO stats look healthy)."""
    src = (P3 / "phase3_grpo.py").read_text()
    assert 'if not sync.get("ok"):' in src
    # the abort must be inside roll_fn, before the roll, and raise
    head = src.split("def roll_fn", 1)[1].split("_roll_onpolicy", 1)[0]
    assert 'if not sync.get("ok"):' in head
    assert "raise RuntimeError" in head


def test_balanced_select_spreads_across_kgroups_not_prefix():
    """High-2 fix: `build_training_rows` sorts by group_key, so `rows[:n]`
    would train almost entirely on the first seed's group. `_balanced_select`
    must round-robin so the kept batch covers every group, return EXACTLY n,
    and preserve each row's (already-finalised) precomputed_advantage."""
    import importlib

    m = importlib.import_module("phase3_grpo")
    rows = ([{"group_key": "((9000, 'A'), 0)", "precomputed_advantage": i}
             for i in range(100)]
            + [{"group_key": "((9001, 'A'), 0)", "precomputed_advantage": i}
               for i in range(100, 200)]
            + [{"group_key": "((9002, 'A'), 0)", "precomputed_advantage": i}
               for i in range(200, 300)])
    sel = m._balanced_select(rows, 12)
    assert len(sel) == 12
    from collections import Counter
    by_g = Counter(r["group_key"] for r in sel)
    assert len(by_g) == 3 and set(by_g.values()) == {4}  # 4 per group, even
    # naive rows[:12] would be 12× the FIRST group — guard that we're not that
    assert by_g["((9001, 'A'), 0)"] == 4 and by_g["((9002, 'A'), 0)"] == 4
    # advantages carried verbatim (selection is a subset, never a recompute)
    assert all("precomputed_advantage" in r for r in sel)


def test_balanced_select_spreads_within_group_not_prefix():
    """Row-coverage fix: within a K-group rows are (game, round)-sorted, so
    popping the prefix keeps only the earliest rollout/rounds — K=8 then helps
    advantage normalization but not gradient coverage. `_balanced_select`
    shuffles each group (seeded) so the kept batch samples rollouts × rounds
    uniformly. `precomputed_advantage` here just encodes the in-group index."""
    import importlib

    m = importlib.import_module("phase3_grpo")
    rows = [{"group_key": "g", "precomputed_advantage": i} for i in range(80)]
    sel = m._balanced_select(rows, 16, seed=0)
    picks = {r["precomputed_advantage"] for r in sel}
    assert len(sel) == 16 and len(picks) == 16          # 16 distinct rows
    # the regression guarded: a prefix selector returns exactly {0..15}.
    assert picks != set(range(16)), "within-group selection is a prefix"
    # union across rolls spans the whole group — coverage, not one window.
    union = set()
    for s in range(12):
        union |= {r["precomputed_advantage"]
                  for r in m._balanced_select(rows, 16, seed=s)}
    assert max(union) >= 64 and min(union) <= 15, sorted(union)
    # seeded ⇒ reproducible for a fixed (rows, n, seed)
    assert [r["precomputed_advantage"]
            for r in m._balanced_select(rows, 16, seed=0)] == \
           [r["precomputed_advantage"] for r in sel]


def test_train_seed_schedule_is_fresh_by_default():
    import importlib

    m = importlib.import_module("phase3_grpo")
    assert m._train_seeds_for_roll(9000, 3, 0) == [9000, 9001, 9002]
    assert m._train_seeds_for_roll(9000, 3, 1) == [9003, 9004, 9005]
    assert m._train_seeds_for_roll(9000, 3, 7) == [9021, 9022, 9023]
    assert m._train_seeds_for_roll(
        9000, 3, 7, fixed=True) == [9000, 9001, 9002]


def test_rollout_diagnostics_cover_full_and_selected_rows():
    import importlib

    m = importlib.import_module("phase3_grpo")
    gk = "((9000, 'A'), 0)"
    rows = [
        {"group_key": gk, "game_id": 0, "round_index": 0, "player_id": 0,
         "phase": "bid", "is_terminal_turn": False,
         "precomputed_reward": 0.0, "precomputed_advantage": -1.0,
         "reward_components": {
             "legal": 0.0, "shaping": 0.0,
             "terminal_correction": 0.0, "terminal": 0.0}},
        {"group_key": gk, "game_id": 0, "round_index": 1, "player_id": 0,
         "phase": "bid", "is_terminal_turn": True,
         "precomputed_reward": 0.0, "precomputed_advantage": -1.0,
         "reward_components": {
             "legal": 0.0, "shaping": 0.0,
             "terminal_correction": 0.0, "terminal": 0.0}},
        {"group_key": gk, "game_id": 1, "round_index": 0, "player_id": 0,
         "phase": "bid", "is_terminal_turn": False,
         "precomputed_reward": 1.0, "precomputed_advantage": 1.0,
         "reward_components": {
             "legal": 0.0, "shaping": 1.0,
             "terminal_correction": 0.0, "terminal": 0.0}},
        {"group_key": gk, "game_id": 1, "round_index": 1, "player_id": 0,
         "phase": "reveal", "is_terminal_turn": True,
         "precomputed_reward": 1.0, "precomputed_advantage": 1.0,
         "reward_components": {
             "legal": 0.0, "shaping": 0.0,
             "terminal_correction": 0.0, "terminal": 1.0}},
    ]
    games = [
        {"final_results": {"final_scores": [
            {"player_id": 0, "final_score": 10},
            {"player_id": 1, "final_score": 10},
            {"player_id": 2, "final_score": 10}]}},
        {"final_results": {"final_scores": [
            {"player_id": 0, "final_score": 20},
            {"player_id": 1, "final_score": 10},
            {"player_id": 2, "final_score": 10}]}},
    ]
    diag = m._rollout_diagnostics(
        rows, rows[:2], games, trainable_seat=0,
        opponents_by_seed={9000: "megagem/heuristic-v1"})
    assert diag["full_rows"]["total_rows"] == 4
    assert diag["selected_rows"]["total_rows"] == 2
    assert diag["selection"]["selected_total_fraction"] == pytest.approx(0.5)
    assert diag["full_rows"]["by_seed"]["9000"] == 4
    assert diag["full_rows"]["by_opponent"]["megagem/heuristic-v1"] == 4
    assert diag["full_rows"]["reward_components"]["sum"] == pytest.approx(2.0)
    kg = diag["kgroup_alignment"]
    assert kg["n_groups"] == 1
    assert kg["groups"][0]["spearman_reward_vs_margin"] == pytest.approx(1.0)
    assert kg["groups"][0]["degenerate"]["reward_sum_std_le_1e-6"] is False


def test_eval_surfaces_final_gate_caveats_and_validity():
    """Medium-5: a green ci_low must never be over-read. The eval must emit
    `valid_as_final_evidence` + `final_gate_caveats` (N<60, eval-determinism)
    so a smoke read can't masquerade as a spend claim."""
    src = (P3 / "phase3_eval.py").read_text()
    assert "valid_as_final_evidence" in src
    assert "final_gate_caveats" in src
    assert "N=60" in src and "det" in src.lower()  # the two standing caveats


def test_run_script_cost_and_honesty_controls():
    """High-1/Medium-3/4: the run script must (a) PROFILE seam|evidence with
    N=60 for evidence, (b) skip the costly eval on GRPO failure unless
    EVAL_ON_GRPO_FAIL=1, (c) banner SEAM-TEST shape, (d) require explicit
    SEAM_VERIFIED + check vllm/transformers pins in the prep gate."""
    s = (P3 / "run_phase3.sh").read_text()
    assert "PROFILE" in s and "evidence)" in s and "_DEF_EVAL_SEEDS=60" in s
    assert "EVAL_ON_GRPO_FAIL" in s and "SKIPPING §3.6 eval" in s
    assert "SEAM-TEST SHAPE" in s
    assert "SEAM_VERIFIED" in s
    assert "vllm==0.10.2 transformers==4.55.4" in s


def test_eval_runs_games_in_parallel_via_semaphore():
    """Parallel-eval: phase3_eval.py must schedule BOTH arms × all seeds in
    one event loop under an asyncio.Semaphore + gather (≈5-8× speedup at no
    GPU $). Per-game asyncio.run() inside the per-seed loop is the old
    sequential path — guard against regression."""
    src = (P3 / "phase3_eval.py").read_text()
    assert "asyncio.gather" in src, "eval lost the gather() concurrency"
    assert "Semaphore" in src, "eval lost the concurrency cap"
    assert "_play_async" in src, "eval must use the async per-game helper"
    assert "--max-parallel" in src, "eval must expose --max-parallel"
    # Pairing must be by-key, not by-list-order: gather result order is
    # well-defined here but the by_seed dict is the explicit contract.
    assert "by_seed[s][\"rl\"]" in src and "by_seed[s][\"sft\"]" in src, (
        "pairing must be looked up by (seed, arm) so concurrency cannot "
        "perturb delta_i")
    # Old sequential path must be gone: there must be exactly one asyncio.run
    # call in the eval (the top-level _play_all dispatch), not one per game.
    assert src.count("asyncio.run(") == 1, (
        f"phase3_eval must have EXACTLY ONE asyncio.run() call (the top-level "
        f"_play_all dispatch). Found {src.count('asyncio.run(')} — the old "
        f"per-game asyncio.run(run_game(...)) pattern is back.")
    assert "asyncio.run(_play_all" in src, (
        "the one asyncio.run() must be the _play_all dispatch")


def test_grpo_rollout_runs_games_in_parallel_via_semaphore():
    """Parallel-rollout: phase3_grpo._roll_onpolicy must concurrently roll
    seeds×k games per refresh (~5-6× rollout wall-time speedup at no GPU $).
    §A.7 standardization runs AFTER gather, so group boundaries are invariant
    to schedule order."""
    src = (P3 / "phase3_grpo.py").read_text()
    assert "asyncio.gather" in src, "rollout lost the gather() concurrency"
    assert "Semaphore" in src, "rollout lost the concurrency cap"
    assert "max_parallel" in src, "rollout must accept max_parallel"
    assert "--max-parallel" in src, "driver must expose --max-parallel"
    # The per-game asyncio.run was the bottleneck — make sure it's gone.
    assert "for s in seeds:\n        for ki in range(k):\n            fname" not in src, (
        "the old sequential per-game loop is back — should be one "
        "asyncio.run(_play_all()) over all seeds×k coros")


def test_hetero_opponents_per_seat_draw_and_a7_safe():
    """PSRO heterogeneous tables: each non-trainable seat draws its OWN opponent
    (seat-perturbed seed → independent), assigned per seat in _models_for, and
    held constant within a K-group (§A.7). Opt-in; legacy homogeneous path kept."""
    src = (P3 / "phase3_grpo.py").read_text()
    assert "opponents_for_table=None" in src, "_roll_onpolicy must accept the opt-in map"
    assert "opp_by_seed_seat" in src, "per-(seed,seat) opponent map missing"
    # per-seat assignment (NOT one opponent broadcast to all non-trainable seats)
    assert "for seat in _nontrainable:" in src
    assert "m[seat] = opp_by_seed_seat[s][seat].served_name" in src
    # roll_fn draws independently per seat with distinct-prime seed perturbation
    assert "pool.draw(step, s * 7919 + seat * 104729)" in src
    # opt-in switch: opponents_for_table only in hetero mode, else legacy callback
    assert "opponents_for_table=((lambda s: opp_table[s]) if hetero else None)" in src
    assert 'getattr(args, "hetero_opponents", False)' in src
    # legacy homogeneous draw still present (byte-identical when the flag is off)
    assert "opp_specs = {s: pool.draw(step, s) for s in roll_seeds}" in src


def test_hetero_actor_mask_is_by_seat_loss_safe():
    """Loss-safety: _post_tag_actor_mask masks by SEAT, so two different opponents
    at a table are both masked out; the opponent label is a cosmetic "+"-join."""
    src = (P3 / "phase3_grpo.py").read_text()
    assert "def _opp_actor_id(s: int)" in src
    assert "opp_by_seed_seat[s][seat].actor_id for seat in _nontrainable" in src
    assert "opponent_actor_id=_opp_actor_id(s)" in src, (
        "the actor-mask call must use the per-table combined id, not a single "
        "opponent's — otherwise heterogeneous-table labelling is wrong")


def test_hetero_opponents_flag_threaded_end_to_end():
    """--hetero-opponents is wired through CLI → run_phase3.sh → modal_train
    (phase3 + phase3_main)."""
    assert "--hetero-opponents" in (P3 / "phase3_grpo.py").read_text()
    s = (P3 / "run_phase3.sh").read_text()
    assert ': "${HETERO_OPPONENTS:=1}"' in s      # default ON (mixed tables)
    assert '[[ "${HETERO_OPPONENTS}" == "1" ]] && cmd+=( --hetero-opponents )' in s
    m = (REPO_ROOT / "modal_train.py").read_text()
    assert '"HETERO_OPPONENTS": "1" if hetero_opponents else "0"' in m
    assert m.count("hetero_opponents: bool = False") == 2   # phase3 + phase3_main
    assert "hetero_opponents=hetero_opponents" in m         # phase3_main → phase3


def test_run_script_threads_max_parallel_to_both_scripts():
    """run_phase3.sh must surface MAX_PARALLEL (PROFILE-driven default: 32 for
    seam, 64 for evidence) and thread it to BOTH phase3_grpo.py (rollout) and
    phase3_eval.py (eval); a partial wire would leave one phase serial."""
    s = (P3 / "run_phase3.sh").read_text()
    assert "MAX_PARALLEL" in s
    assert ': "${MAX_PARALLEL:=${_DEF_MAX_PARALLEL}}"' in s, (
        "MAX_PARALLEL must resolve from the PROFILE default")
    assert "_DEF_MAX_PARALLEL=32" in s and "_DEF_MAX_PARALLEL=64" in s
    # Must appear in BOTH command builders (run_grpo + run_eval).
    assert s.count('--max-parallel "${MAX_PARALLEL}"') >= 2, (
        "MAX_PARALLEL must be threaded to BOTH grpo and eval invocations")


def test_eval_supports_k_sample_averaging():
    """Lever A: phase3_eval must accept --eval-samples-per-seed and average
    K plays per (seed, arm) before pairing — var(delta_avg)=var(delta)/K, so
    CI half-width drops by √K at K× the eval-game cost. Pairing must remain
    per-seed."""
    src = (P3 / "phase3_eval.py").read_text()
    assert "--eval-samples-per-seed" in src
    assert "K_effective" in src or "k_samples" in src
    # averaging happens before pairing
    assert "score_rl_samples" in src and "score_sft_samples" in src
    assert "rl_avg" in src and "sft_avg" in src


def test_eval_supports_temperature_override():
    """Lever C: phase3_eval must accept --temperature and pass it symmetrically
    to BOTH arms (same caller_api_params on RL and SFT calls). T=0.0 = greedy,
    which auto-collapses K to 1 (replays would be byte-identical)."""
    src = (P3 / "phase3_eval.py").read_text()
    assert "--temperature" in src
    assert "caller_api_params" in src
    # T=0 collapse to K=1 to avoid wasted compute
    assert "K_effective = 1" in src or "K=1" in src.replace(" ", "")
    # both arms get the SAME temperature (no per-arm asymmetry)
    assert src.count("temperature=temperature") >= 2, (
        "temperature must be threaded into BOTH _play_async calls "
        "(rl arm AND sft arm) for symmetric eval")


def test_eval_drops_caveat_when_noise_addressed():
    """When the eval is greedy (T=0) OR uses K≥4 averaging, the standing
    "eval-determinism unaddressed" caveat is no longer applicable and should
    not be emitted (or the gate-validity check should reflect this)."""
    src = (P3 / "phase3_eval.py").read_text()
    assert "eval_det_addressed" in src
    # the caveat string should be GATED on eval_det_addressed
    assert "if not eval_det_addressed" in src


def test_run_script_threads_eval_levers_and_skip_grpo():
    """run_phase3.sh must (a) thread EVAL_SAMPLES_PER_SEED + EVAL_TEMPERATURE
    to phase3_eval.py, and (b) support SKIP_GRPO=1 + EXT_FINAL_DIR for the
    eval-only re-eval flow (Lever C corroboration on existing checkpoint)."""
    s = (P3 / "run_phase3.sh").read_text()
    assert "EVAL_SAMPLES_PER_SEED" in s
    assert "--eval-samples-per-seed" in s
    assert "EVAL_TEMPERATURE" in s
    assert "--temperature" in s
    assert "SKIP_GRPO" in s
    assert "EXT_FINAL_DIR" in s
    assert "EVAL_TRAIN_SEEDS" in s and "final_trainseeds" in s
    assert "EVAL_INTERMEDIATE" in s


def test_fix2_informational_baseline_does_not_poison_rc(tmp_path):
    """#2: `--informational` reports the gate but the process must not fail
    on it (the pre-train baseline's ≈0 delta is expected)."""
    r = subprocess.run(
        [sys.executable, "scripts/training/phase3_eval.py", "--dry-run",
         "--informational", "--label", "step0",
         "--output", str(tmp_path / "b.json")],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads((tmp_path / "b.json").read_text())
    assert rep["status"] == "PASS"


def test_grpo_dump_rollouts_is_wired_and_side_effect_only():
    """phase3_grpo.py must expose --dump-rollouts and persist each on-policy
    roll's games so the reward diagnostic can run on the TRUE phase-3
    distribution (rollouts are otherwise lost with the TemporaryDirectory).
    The dump must (a) happen AFTER _post_tag_actor_mask so games carry the
    actor_id the diagnostic's compute_advantages requires, (b) write one
    subdir per roll (a clean K-group set), and (c) be a pure side effect —
    the rows returned to the trainer are unchanged."""
    src = (P3 / "phase3_grpo.py").read_text()
    assert "--dump-rollouts" in src, "driver must expose --dump-rollouts"
    assert "dump_dir" in src and "roll_index" in src, (
        "_roll_onpolicy must accept dump_dir + roll_index")
    # dump is gated and writes per-roll subdirs.
    assert "if dump_dir is not None:" in src
    assert 'f"roll_{int(roll_index):03d}"' in src
    # dump happens AFTER the post-tag (games carry actor_id), BEFORE the
    # flatten return — locate both anchors and assert ordering.
    i_tag = src.index("_post_tag_actor_mask(")
    i_dump = src.index("if dump_dir is not None:")
    i_ret = src.index("return H.flatten_training_rows")
    assert i_tag < i_dump < i_ret, (
        "dump must sit between post-tag and the flatten return so dumped "
        "games are actor-tagged and the returned rows are untouched")


def test_run_script_threads_dump_rollouts():
    """run_phase3.sh must surface DUMP_ROLLOUTS (off by default), resolve the
    "1" shorthand to ${RESULTS_DIR}/rollout_dumps, and thread --dump-rollouts
    to phase3_grpo.py only when non-empty."""
    s = (P3 / "run_phase3.sh").read_text()
    assert ': "${DUMP_ROLLOUTS:=}"' in s, "DUMP_ROLLOUTS must default to off"
    assert '"1" ]] && DUMP_ROLLOUTS="${RESULTS_DIR}/rollout_dumps"' in s, (
        "DUMP_ROLLOUTS=1 must resolve to the standard per-run location")
    assert '[[ -n "${DUMP_ROLLOUTS}" ]] && cmd+=( --dump-rollouts' in s, (
        "--dump-rollouts must be threaded only when DUMP_ROLLOUTS is set")


def test_run_script_supports_vllm_tokenizer_override():
    """run_phase3.sh must expose VLLM_TOKENIZER (off by default) and append
    --tokenizer to the vLLM serve command when set — the escape hatch for a
    served model whose own tokenizer lacks a usable chat_template (transformers
    4.44+ then 400s every chat request)."""
    s = (P3 / "run_phase3.sh").read_text()
    assert ': "${VLLM_TOKENIZER:=}"' in s, "VLLM_TOKENIZER must default to off"
    assert 'cmd+=( --tokenizer "${VLLM_TOKENIZER}" )' in s, (
        "VLLM_TOKENIZER must be threaded into the vllm serve command")
    # gated on non-empty so the default run is unchanged.
    assert 'if [[ -n "${VLLM_TOKENIZER}" ]]; then' in s


def test_reward_diagnostic_script_exists_and_has_cli():
    """The reward-signal CPU diagnostic must exist with its --corpus CLI so
    a dumped roll can be checked: does precomputed_reward correlate with the
    engine final_score, and is the within-K-group spread trainable?"""
    diag = P3 / "reward_score_correlation.py"
    assert diag.exists(), "scripts/training/reward_score_correlation.py missing"
    src = diag.read_text()
    assert "--corpus" in src
    assert "compute_advantages" in src and "terminal_breakdowns" in src


# --------------------------------------------------------------------------- #
# §3.3 lagged-self opponent pool + telemetry + eval-opponent (this change).    #
# --------------------------------------------------------------------------- #
def test_run_game_supports_per_seat_caller_api_params():
    """run_game.caller_api_params accepts a PER-SEAT list (Phase-3 pool: the
    trainable seat sampled, opponents greedy) as well as a dict (backward-
    compatible). MegaGemEnv threads it per player_id."""
    import inspect

    from megagem import rollout as run_game
    ann = str(inspect.signature(run_game.run_game)
              .parameters["caller_api_params"].annotation)
    assert "list" in ann, f"caller_api_params must accept a list, got {ann}"
    rg = (REPO_ROOT / "src/megagem/rollout.py").read_text()
    assert "per_seat_api_params" in rg
    # the vLLM-only extra_body is computed PER SEAT (not via a global has_vllm)
    # so a mixed game never sends chat_template_kwargs to a hosted API opponent.
    assert "_vllm_base_extra" in rg and "has_vllm" not in rg
    env_src = (REPO_ROOT / "src/megagem/environment/multi_agent_env.py").read_text()
    assert "per_seat_api_params" in env_src and "player_id" in env_src


def test_spg_shape_recipe_repl_08_strict_spg2():
    """repl_08 strict spg=2 geometry: with rows_per_gen=64, num_generations=2,
    PHASE2_MICRO_CAP=64 ⇒ gen_batch=128, micro=64, spg=2. The repl_07 default
    (rows=96, cap=64) instead gives spg=3 because 192 is not divisible by 128.

    Verifies _spg_shape — the SINGLE source of truth shared by _gpu_run and
    --dry-run."""
    import os
    from phase3_grpo import _spg_shape
    old_cap = os.environ.get("PHASE2_MICRO_CAP")
    try:
        os.environ["PHASE2_MICRO_CAP"] = "64"
        # repl_08 strict spg=2. _spg_shape also reports ga / divisibility /
        # opt-step bookkeeping, so assert the geometry keys as a subset.
        r08 = _spg_shape(rows_per_gen=64, num_generations=2)
        assert {"g": 2, "gen_batch": 128, "micro_cap": 64,
                "micro": 64, "spg": 2}.items() <= r08.items(), r08
        # repl_07 baseline ⇒ spg=3
        r07 = _spg_shape(rows_per_gen=96, num_generations=2)
        assert {"g": 2, "gen_batch": 192, "micro_cap": 64,
                "micro": 64, "spg": 3}.items() <= r07.items(), r07
    finally:
        if old_cap is None:
            os.environ.pop("PHASE2_MICRO_CAP", None)
        else:
            os.environ["PHASE2_MICRO_CAP"] = old_cap


def test_dry_run_validates_opp_anchor_floor():
    """Codex round-2: the GPU path uses p_anchor_floor=args.opp_anchor_floor,
    but the dry-run originally constructed the pool with the default — so a
    direct `--dry-run --opp-anchor-floor 1.5` would silently pass while the
    GPU path would later raise. Fix: dry-run now threads the same param."""
    src = (P3 / "phase3_grpo.py").read_text()
    # The dry-run pool construction includes p_anchor_floor.
    dry_run_block = src.split("def _dry_run(args)")[1].split("def _gpu_run")[0]
    assert "p_anchor_floor=args.opp_anchor_floor" in dry_run_block, (
        "dry-run pool construction must pass p_anchor_floor so out-of-range "
        "values fail before GPU spend")


def test_opp_anchor_floor_threaded_through_modal_and_shell():
    """Codex round-2: --opp-anchor-floor existed as an argparse leaf default
    but was not threaded through run_phase3.sh or modal_train.py — so the
    intended entrypoints always used the hardcoded leaf default and the
    claim '0.0 reproduces pre-Codex behavior' was untrue through Modal."""
    s = (P3 / "run_phase3.sh").read_text()
    assert ': "${OPP_ANCHOR_FLOOR:=0.15}"' in s
    assert '--opp-anchor-floor "${OPP_ANCHOR_FLOOR}"' in s
    m = (REPO_ROOT / "modal_train.py").read_text()
    assert "opp_anchor_floor: str =" in m
    assert '"OPP_ANCHOR_FLOOR": opp_anchor_floor' in m
    # phase3_main also forwards it to phase3().
    assert "opp_anchor_floor=opp_anchor_floor" in m


def test_dual_gate_has_statistical_test_against_sft_baseline():
    """Codex round-2: dual-gate must do more than `observed_WR > threshold`;
    when sft_baseline_wr > 0 it should additionally run a one-sided z-test
    so the gate is statistically defensible, not a noisy point estimate.

    The gate math now lives in the package (megagem.training.dual_gate); the
    Modal entrypoint only exposes the knobs and threads them through."""
    m = (REPO_ROOT / "modal_train.py").read_text()
    assert "dual_gate_sft_baseline_wr: float = 0.30" in m
    assert "dual_gate_significance_alpha: float = 0.05" in m
    assert "from megagem.training.dual_gate import assess_flash_gate" in m
    assert "sft_baseline_wr=dual_gate_sft_baseline_wr" in m
    assert "significance_alpha=dual_gate_significance_alpha" in m

    d = (REPO_ROOT / "src/megagem/training/dual_gate.py").read_text()
    assert "one-sided z-test" in d
    assert "def z_critical" in d
    assert "stat_pass" in d
    assert "z_crit" in d
    # both the absolute threshold AND the z-test must clear when enabled
    assert "and (stat_pass is None or stat_pass)" in d


def test_dapo_api_drop_alert_threshold_is_at_least_2():
    """Codex round-2: with p_api=0.10 and num_seeds=24, expected API K-groups
    per roll is ~2.4. The original `api_total >= 4` floor missed the exact
    catastrophic 2/2 or 3/3 cases. Lowered to >= 2."""
    src = (P3 / "phase3_grpo.py").read_text()
    assert "api_total >= 2 and api_dropped / api_total > 0.5" in src


def _make_dapo_rows(group_seed: int, rewards: list[float]) -> list[dict]:
    """Build minimal rows for `_dapo_filter_degenerate_kgroups`. Each K-sample
    becomes one row with a fake group_key that survives `_row_seed`'s
    `ast.literal_eval` round-trip and one game per sample (per_game sum =
    that sample's reward, which is what the filter aggregates over)."""
    group_key = repr(((group_seed,), "g"))
    return [
        {
            "group_key": group_key,
            "game_id": f"game_{group_seed}_{i}",
            "precomputed_reward": r,
        }
        for i, r in enumerate(rewards)
    ]


def test_dapo_heuristic_exempt_skips_relative_threshold():
    """Heuristic K-groups with low std (= policy has learned the exploit)
    must NOT be dropped under the relative threshold when heuristic is in
    `exempt_kinds`. Snapshot K-groups with similar std SHOULD still be
    dropped (control). The absolute threshold still applies to both."""
    from megagem.training.opponent_pool import OpponentSpec
    from phase3_grpo import _dapo_filter_degenerate_kgroups

    # Two K-groups: seed=100 = heuristic, seed=200 = snapshot. Both have
    # within-group std = 0.10 (each group's 8 rewards spread by ±0.10).
    # Under rel-threshold 0.4 with EMA std seeded at 0.6 (snap-like), the
    # threshold becomes 0.4 * 0.6 = 0.24 — both groups would normally
    # be dropped under the rel path. The abs floor is 0.01.
    heur_rewards = [0.60, 0.62, 0.58, 0.59, 0.61, 0.63, 0.57, 0.60]
    snap_rewards = [-0.05, +0.05, -0.04, +0.04, -0.06, +0.06, +0.03, -0.03]

    rows = (_make_dapo_rows(100, heur_rewards)
            + _make_dapo_rows(200, snap_rewards))
    opp_specs = {
        100: OpponentSpec("heuristic", "megagem/heuristic-v1", "h"),
        200: OpponentSpec("snapshot", "phase3-snap-50", "s50", step=50),
    }
    # Seed EMA so the relative threshold has a real value to compare against
    # — first-observation seeding would set EMA = observed-std and never trip.
    opp_ema_std = {"heuristic": 0.60, "snapshot": 0.60}

    # CONTROL: no exemption — both groups should drop under rel.
    _, stats_no_exempt = _dapo_filter_degenerate_kgroups(
        list(rows), opp_specs=opp_specs,
        opp_ema_std=dict(opp_ema_std),
        abs_threshold=0.01, opp_rel_threshold=0.4, ema_alpha=0.9,
        exempt_kinds=frozenset(),
    )
    drops_no = stats_no_exempt["by_kind_dropped"]
    assert drops_no.get("heuristic", 0) == 1, (
        f"heuristic should drop under rel without exemption; got {drops_no}")
    assert drops_no.get("snapshot", 0) == 1, (
        f"snapshot should drop under rel; got {drops_no}")

    # TREATMENT: heuristic exempt — only snapshot drops.
    _, stats_exempt = _dapo_filter_degenerate_kgroups(
        list(rows), opp_specs=opp_specs,
        opp_ema_std=dict(opp_ema_std),
        abs_threshold=0.01, opp_rel_threshold=0.4, ema_alpha=0.9,
        exempt_kinds=frozenset({"heuristic"}),
    )
    drops_yes = stats_exempt["by_kind_dropped"]
    assert drops_yes.get("heuristic", 0) == 0, (
        f"heuristic should NOT drop when exempt; got {drops_yes}")
    assert drops_yes.get("snapshot", 0) == 1, (
        f"snapshot should still drop; got {drops_yes}")
    assert stats_exempt["exempt_kinds"] == ["heuristic"], (
        f"telemetry should record the exemption; got {stats_exempt}")


def test_dapo_abs_threshold_still_drops_heuristic_when_exempt():
    """Heuristic exemption skips the RELATIVE threshold only. Truly-zero-std
    heuristic groups (all 8 rewards identical) must still drop under the
    absolute floor — otherwise we'd let format-failure K-groups through."""
    from megagem.training.opponent_pool import OpponentSpec
    from phase3_grpo import _dapo_filter_degenerate_kgroups

    # All-identical rewards → std = 0 — well below abs_threshold=0.01.
    rows = _make_dapo_rows(100, [0.50] * 8)
    opp_specs = {
        100: OpponentSpec("heuristic", "megagem/heuristic-v1", "h"),
    }
    _, stats = _dapo_filter_degenerate_kgroups(
        rows, opp_specs=opp_specs, opp_ema_std={"heuristic": 0.60},
        abs_threshold=0.01, opp_rel_threshold=0.4, ema_alpha=0.9,
        exempt_kinds=frozenset({"heuristic"}),
    )
    assert stats["by_kind_dropped"].get("heuristic", 0) == 1, (
        "abs-floor must still drop truly-degenerate heuristic K-groups")


def test_dapo_heuristic_exempt_default_on_in_phase3_grpo():
    """The runtime defaults to PHASE3_DAPO_HEURISTIC_EXEMPT=1 unless
    operator explicitly opts out — encoded as the literal default in the
    env-read line."""
    src = (P3 / "phase3_grpo.py").read_text()
    assert 'PHASE3_DAPO_HEURISTIC_EXEMPT", "1"' in src, (
        "default should be ON; if you intentionally changed it, also update "
        "this test and the seam-PASS criteria.")


def test_grpo_preseeds_step0_anchor_after_trainer_built():
    """repl_08 wiring: phase3_grpo.py constructs the pool with an empty pinned
    list, then — AFTER trainer.model exists and the init adapter is saved —
    saves the snapshot_0 adapter, pushes to vLLM, registers in ENDPOINTS, and
    calls pool.add_pinned_snapshot. A failed push is FATAL (no silent fallback
    to the removed heuristic)."""
    src = (P3 / "phase3_grpo.py").read_text()
    assert "pinned_snapshots=[]" in src
    assert "heuristic_spec=None" in src
    # The seed block lives between init_ckpt and trainer.train().
    assert "repl_08: scripted heuristic removed from training" in src
    assert "pool.add_pinned_snapshot" in src
    assert "step_0 anchor adapter→vLLM sync FAILED" in src
    # WR pipe-back from per-snapshot stats into the pool.
    assert "pool.update_snapshot_winrate" in src


def test_opponent_anneal_schedule():
    """anneal_probability: 0 at/before start, p_max at/after end, monotone
    non-decreasing and linear between. Degenerate end<=start ⇒ step at start."""
    from megagem.training.opponent_pool import anneal_probability as ap
    kw = dict(anneal_start=50, anneal_end=200, p_max=0.7)
    assert ap(0, **kw) == 0.0
    assert ap(50, **kw) == 0.0
    assert ap(125, **kw) == pytest.approx(0.35)
    assert ap(200, **kw) == pytest.approx(0.7)
    assert ap(99_999, **kw) == pytest.approx(0.7)
    prev = -1.0
    for s in range(0, 260, 5):
        v = ap(s, **kw)
        assert v >= prev - 1e-12 and 0.0 <= v <= 0.7
        prev = v
    assert ap(60, anneal_start=50, anneal_end=50,
              p_max=0.7) == pytest.approx(0.7)


def test_opponent_pool_ring_buffer_evicts_oldest():
    """keep-last-N: the 6th snapshot into a max=5 pool evicts the oldest."""
    from megagem.training.opponent_pool import OpponentPool, OpponentSpec, Snapshot
    pool = OpponentPool(
        heuristic_spec=OpponentSpec("heuristic", "h", "heuristic"),
        max_snapshots=5)
    for i in range(1, 6):
        assert pool.add_snapshot(Snapshot(25 * i, f"s{i}", "/p")) is None
    assert [s.step for s in pool.snapshots()] == [25, 50, 75, 100, 125]
    evicted = pool.add_snapshot(Snapshot(150, "s6", "/p"))
    assert evicted is not None and evicted.step == 25
    assert [s.step for s in pool.snapshots()] == [50, 75, 100, 125, 150]


def test_opponent_draw_one_per_kgroup_and_anneal():
    """§A.7: draw(step,seed) is constant for a fixed (step,seed) — every K
    rollout of a seed faces the SAME opponent. Before anneal_start it is always
    the heuristic; well past anneal_end ~p_max are pooled snapshots."""
    from megagem.training.opponent_pool import OpponentPool, OpponentSpec, Snapshot
    pool = OpponentPool(
        heuristic_spec=OpponentSpec(
            "heuristic", "megagem/heuristic-v1", "heuristic"),
        max_snapshots=5, anneal_start=50, anneal_end=200, p_max=0.7,
        rng_seed=0)
    for st in (50, 75, 100, 125, 150):
        pool.add_snapshot(Snapshot(st, f"phase3-snap-{st}", "/p"))
    for seed in range(9000, 9008):
        assert len({pool.draw(300, seed).served_name
                    for _ in range(16)}) == 1
    assert all(pool.draw(10, s).kind == "heuristic"
               for s in range(9000, 9020))
    kinds = [pool.draw(5000, s).kind for s in range(9000, 9100)]
    assert kinds.count("snapshot") > 45      # ≈70 expected; loose bound
    assert kinds.count("heuristic") > 5      # the 1-p_max floor still present


def test_opponent_draw_is_reproducible_not_hash_salted():
    """draw() is process-independent — _unit uses hashlib, NOT the per-process
    salted builtin hash(). The golden value pins it against a hash() regression
    under a different PYTHONHASHSEED."""
    from megagem.training.opponent_pool import OpponentPool, OpponentSpec, Snapshot, _unit
    assert _unit(0, 1200, 9000, "pool") == pytest.approx(
        0.01716432132568204, abs=1e-12)

    def _mk():
        p = OpponentPool(
            heuristic_spec=OpponentSpec("heuristic", "h", "heuristic"),
            anneal_start=0, anneal_end=1, p_max=1.0, rng_seed=42)
        for st in (25, 50, 75):
            p.add_snapshot(Snapshot(st, f"phase3-snap-{st}", "/p"))
        return p
    a, b = _mk(), _mk()
    assert all(a.draw(99, s).served_name == b.draw(99, s).served_name
               for s in range(9000, 9050))


def test_actor_mask_masks_any_pool_opponent():
    """§A.6: the loss mask is opponent-AGNOSTIC. Post-tagging with a heuristic,
    snapshot, or API actor_id all flatten to rows that are exclusively the
    trainable seat's — the opponent's tokens are dropped identically."""
    import copy

    import megagem.training.grpo_harness as Hh
    from megagem_steps import _post_tag_actor_mask

    games = Hh.rollout_group(
        9000, trainable_policy_fn=Hh.make_stub_policy_fn("bestresp"), k=4)
    for opp_id in ("heuristic", "snapshot_75", "api_google_gemini-3-flash"):
        stripped = []
        for g in games:
            gg = copy.deepcopy(g)
            for rnd in gg["rounds"]:
                for rec in rnd["players"]:
                    rec.pop("actor_id", None)
            stripped.append(gg)
        rows = Hh.flatten_training_rows(
            [_post_tag_actor_mask(g, trainable_seat=0,
                                  opponent_actor_id=opp_id)
             for g in stripped],
            trainable_seat=0)
        assert rows, f"no rows for opponent {opp_id!r}"
        assert all(r["actor_id"] == Hh.TRAINABLE_ACTOR_ID for r in rows), (
            f"opponent {opp_id!r} leaked non-trainable rows into the loss")
        assert all(r["player_id"] == 0 for r in rows)


def test_grpo_opponent_pool_is_wired():
    """phase3_grpo.py exposes the §3.3 pool — the --opponent-pool toggle, the
    8 pool knobs, the _Snapshot callback, the per-seed draw, greedy opponents,
    and the §A.7 one-opponent-per-seed defensive assertion."""
    src = (P3 / "phase3_grpo.py").read_text()
    for flag in ("--opponent-pool", "--snapshot-every", "--max-snapshots",
                 "--opp-anneal-start", "--opp-anneal-end", "--opp-anneal-pmax",
                 "--opp-api-models", "--opp-api-prob", "--opp-pool-seed"):
        assert flag in src, f"missing pool CLI knob {flag}"
    assert "class _Snapshot" in src and "save_snapshot_adapter" in src
    assert "opponent_for_seat" in src and "OpponentPool(" in src
    assert "OPPONENT_SAMPLING" in src and '"temperature": 0.0' in src
    assert "§A.7 violation" in src       # the defensive one-opponent assertion
    # API opponents preflight PRIME_API_KEY before training — fail fast, not a
    # mid-rollout crash after GPU spend.
    assert "os.getenv(endpoints.PRIME_KEY)" in src
    # --opp-anneal-end defaults to None ⇒ resolved to --steps after parse_args,
    # so a direct python invocation anneals over the whole run.
    assert "args.opp_anneal_end is None" in src
    assert "args.opp_anneal_end = args.steps" in src


def test_grpo_exposes_epsilon_high_knob():
    """--epsilon-high (DAPO clip-higher) is an OPT-IN knob: present, defaults
    to None (⇒ TRL's symmetric 0.2 — behaviour-neutral), and is inserted into
    GRPOConfig only when set (passing epsilon=None would corrupt TRL)."""
    src = (P3 / "phase3_grpo.py").read_text()
    assert '"--epsilon-high"' in src and '"--epsilon"' in src
    assert "if args.epsilon_high is not None:" in src
    assert "if args.epsilon is not None:" in src


def test_grpo_persists_entropy_and_clip_telemetry():
    """_health harvests + emits entropy and PPO clip-fraction trajectories —
    using TRL's standard-path key clip_ratio/region_mean (NOT a bare
    clip_ratio, which is Liger-only) — and tolerates an old series dict that
    lacks them."""
    import importlib

    src = (P3 / "phase3_grpo.py").read_text()
    assert "clip_ratio/region_mean" in src
    assert "entropy_trajectory" in src and "clip_fraction_trajectory" in src

    m = importlib.import_module("phase3_grpo")

    class A:
        kl_max = 0.5

    rows = [{"precomputed_advantage": v, "completion": "xx"}
            for v in (-1.0, 0.5, 1.5, -0.7)]
    h = m._health({"kl": [0.1], "loss": [1.0], "grad_norm": [0.5]}, rows, A)
    assert "entropy_trajectory" in h and "clip_fraction_trajectory" in h
    assert h["all_pass"] is True          # missing telemetry keys != failure


def test_driver_dry_run_pool_checks_and_config(tmp_path):
    """The driver --dry-run exercises the §3.3 pool on CPU (anneal bounds,
    ring-buffer eviction, deterministic draw, opponent-agnostic mask) and
    emits a config block carrying the clip-higher knobs."""
    rep = _dry("phase3_grpo.py", tmp_path / "d.json")
    assert rep["status"] == "PASS"
    assert all(rep["pool_checks"].values()), rep["pool_checks"]
    for key in ("learning_rate", "kl_beta", "epsilon", "epsilon_high"):
        assert key in rep["config"], f"config block missing {key}"


def test_run_script_threads_opponent_pool_knobs():
    """run_phase3.sh threads the §3.3 pool knobs + the vLLM multi-LoRA flags
    and guards MAX_CPU_LORAS / MAX_LORAS with enough slack for the trainable +
    pinned step_0 anchor (repl_08) + every snapshot; modal_train.py exposes the
    matching params + env wiring."""
    s = (P3 / "run_phase3.sh").read_text()
    assert ': "${OPPONENT_POOL:=1}"' in s
    assert ': "${SNAPSHOT_EVERY:=${_DEF_SNAPSHOT_EVERY}}"' in s
    assert ': "${MAX_SNAPSHOTS:=${_DEF_MAX_SNAPSHOTS}}"' in s
    # LoRA slot counts are PROFILE-driven (seam 8/8, evidence 10/11).
    assert ': "${MAX_LORAS:=${_DEF_MAX_LORAS}}"' in s
    assert ': "${MAX_CPU_LORAS:=${_DEF_MAX_CPU_LORAS}}"' in s
    assert "_DEF_MAX_LORAS=8; _DEF_MAX_CPU_LORAS=8" in s
    assert "_DEF_MAX_LORAS=10; _DEF_MAX_CPU_LORAS=11" in s
    assert "--max-loras" in s and "--max-cpu-loras" in s
    assert "--no-opponent-pool" in s
    # repl_08 bumped the LoRA-slot guard: trainable + step_0 anchor + every
    # snapshot must fit, so the CPU cache needs +3 (was +2) and the GPU slots
    # need +2 (was +1).
    assert "MAX_CPU_LORAS < MAX_SNAPSHOTS + 3" in s   # CPU-cache guard
    assert "MAX_LORAS < MAX_SNAPSHOTS + 2" in s       # GPU per-batch guard
    m = (REPO_ROOT / "modal_train.py").read_text()
    assert "opponent_pool: bool = True" in m
    assert '"OPPONENT_POOL":' in m and '"MAX_LORAS":' in m


def test_run_script_announces_low_rows_bypass_banner():
    """When ALLOW_LOW_ROWS_PER_GEN=1 is set under PROFILE=evidence with a
    below-floor ROWS_PER_GEN, run_phase3.sh prints a loud banner naming the
    env-var and the floor it bypasses — so we never miss a quiet bypass after
    the fact."""
    s = (P3 / "run_phase3.sh").read_text()
    assert 'ALLOW_LOW_ROWS_PER_GEN' in s
    # the evidence rows/gen floor is 4096 (k=16 × 32 seeds)
    assert "BANNER" in s and "bypassing the 4096-floor guard" in s
    assert "default no-heuristic evidence floor: 4096" in s


def test_run_script_threads_clip_higher_and_eval_opponent():
    """run_phase3.sh exposes EPSILON_HIGH (opt-in clip-higher, empty default)
    and EVAL_OPPONENT, threaded into the grpo / eval commands. The eval
    opponent now defaults to the SFT-served model (no-heuristic path), not the
    scripted heuristic."""
    s = (P3 / "run_phase3.sh").read_text()
    assert ': "${EPSILON_HIGH:=}"' in s
    assert '[[ -n "${EPSILON_HIGH}" ]] && cmd+=( --epsilon-high' in s
    assert ': "${EVAL_OPPONENT:=${SERVED_NAME}}"' in s
    assert '--eval-opponent "${EVAL_OPPONENT}"' in s


def test_eval_supports_eval_opponent():
    """phase3_eval.py exposes --eval-opponent (default heuristic ⇒ unchanged,
    history-comparable) and makes the T=0 K-collapse opponent-conditional — a
    hosted API opponent is not deterministic, so K-averaging must be kept."""
    src = (P3 / "phase3_eval.py").read_text()
    assert "--eval-opponent" in src and 'default="heuristic"' in src
    assert "_opponent_is_deterministic" in src
    assert "opp_is_det and temperature is not None" in src
    # a non-heuristic opponent at the stale heuristic-+2 threshold (and not
    # --informational) must hard-abort, not emit a green SPEND — the preflight
    # and the caveat both key on the default gate.
    assert src.count("abs(args.threshold - GATE_THRESHOLD)") >= 2

    import importlib
    ev = importlib.import_module("phase3_eval")
    assert ev._opponent_model("heuristic") == "megagem/heuristic-v1"
    assert ev._opponent_model("google/gemini-3-flash-preview") == (
        "google/gemini-3-flash-preview")
    assert ev._opponent_is_deterministic("heuristic") is True
    assert ev._opponent_is_deterministic(
        "google/gemini-3-flash-preview") is False


def test_run_script_profile_specific_defaults():
    """PROFILE drives steps / eval-K / pool cadence — seam stresses the §3.3
    pool in a tiny run, evidence is the real §3.5/§3.6 sizing (K=8 eval, NOT
    the K=1 noise trap). modal_train.py uses 0-sentinels so `--profile` alone
    selects the right values."""
    s = (P3 / "run_phase3.sh").read_text()
    # seam profile — pool-stressing wiring smoke
    assert "_DEF_STEPS=70" in s
    assert "_DEF_SNAPSHOT_EVERY=10" in s and "_DEF_MAX_SNAPSHOTS=2" in s
    # evidence profile — the real run: K=8 decision-grade eval, 200 steps
    assert "_DEF_STEPS=200" in s and "_DEF_EVAL_K=8" in s
    assert "_DEF_NUM_SEEDS=32" in s and "_DEF_MICRO_CAP=64" in s
    # knobs resolve from the PROFILE defaults
    assert ': "${STEPS:=${_DEF_STEPS}}"' in s
    assert ': "${EVAL_SAMPLES_PER_SEED:=${_DEF_EVAL_K}}"' in s
    assert ': "${PHASE2_MICRO_CAP:=${_DEF_MICRO_CAP}}"' in s
    assert ': "${OPP_ANNEAL_START:=${_DEF_OPP_ANNEAL_START}}"' in s
    # OPP_ANNEAL_END=0 sentinel resolves to the run length
    assert 'OPP_ANNEAL_END}" == "0" ]] && OPP_ANNEAL_END="${STEPS}"' in s
    # modal_train.py 0-sentinels: phase3 + phase3_main only
    m = (REPO_ROOT / "modal_train.py").read_text()
    # the bare `steps` param (not gradient_accumulation_steps) on phase3 +
    # phase3_main only
    assert m.count("\n    steps: int = 0,") == 2
    assert "def phase2(" not in m
    assert 'env["STEPS"] = str(steps)' in m
    assert 'env["EVAL_SAMPLES_PER_SEED"] = str(eval_samples_per_seed)' in m
    assert 'env["SNAPSHOT_EVERY"] = str(snapshot_every)' in m


def test_run_script_guards_low_rows_and_threads_seed_mode():
    s = (P3 / "run_phase3.sh").read_text()
    assert "ALLOW_LOW_ROWS_PER_GEN" in s
    assert "ROWS_PER_GEN < 4096" in s
    assert "FIXED_TRAIN_SEEDS" in s
    assert "--fixed-train-seeds" in s


def test_opponent_pool_p_api_zero_never_draws_api():
    """p_api is the SOLE API-draw control: p_api=0.0 ⇒ an API opponent is
    NEVER drawn — even with an empty snapshot pool, the draw falls back to the
    heuristic, not API. p_api=1.0 ⇒ API is drawn for every pooled slot."""
    from megagem.training.opponent_pool import OpponentPool, OpponentSpec
    api = [OpponentSpec("api", "google/gemini-3-flash-preview", "api_g")]
    heur = OpponentSpec("heuristic", "h", "heuristic")
    # empty snapshot pool, p_api=0, anneal forced to p_max from step 0
    p0 = OpponentPool(heuristic_spec=heur, anneal_start=0, anneal_end=1,
                      p_max=1.0, api_specs=api, p_api=0.0, rng_seed=0)
    kinds = {p0.draw(500, s).kind for s in range(9000, 9100)}
    assert "api" not in kinds, kinds          # p_api=0 ⇒ no API, ever
    assert kinds == {"heuristic"}             # empty pool ⇒ heuristic fallback
    # p_api=1.0 with an empty pool ⇒ API IS drawn (the slot is API-eligible)
    p1 = OpponentPool(heuristic_spec=heur, anneal_start=0, anneal_end=1,
                      p_max=1.0, api_specs=api, p_api=1.0, rng_seed=0)
    assert all(p1.draw(500, s).kind == "api" for s in range(9000, 9020))


# --------------------------------------------------------------------------- #
# repl_08 v3.2 — seat rotation (P0-1) + heuristic anneal wiring.               #
# --------------------------------------------------------------------------- #
def test_rotated_seat_round_robin_and_disabled():
    """`_rotated_seat` is the pure SoT for the trainable seat per roll. When
    rotate=False it pins `base_seat` (legacy single-seat); when rotate=True it
    round-robins across all seats so the policy trains from every position
    (the TrueSkill/panel eval rates seat0/1/2). The seat is a function of
    roll_index ONLY — constant within a roll, hence constant within each
    K-group, so §A.7 within-group standardization is unaffected."""
    from phase3_grpo import _rotated_seat
    # Disabled ⇒ always the base seat regardless of roll_index.
    for ri in range(6):
        assert _rotated_seat(ri, base_seat=0, num_players=3, rotate=False) == 0
        assert _rotated_seat(ri, base_seat=2, num_players=3, rotate=False) == 2
    # Enabled ⇒ round-robin (base_seat + roll_index) % num_players.
    seats = [_rotated_seat(ri, base_seat=0, num_players=3, rotate=True)
             for ri in range(7)]
    assert seats == [0, 1, 2, 0, 1, 2, 0]
    # Non-zero base seat shifts the cycle but still covers all seats.
    seats2 = [_rotated_seat(ri, base_seat=1, num_players=3, rotate=True)
              for ri in range(6)]
    assert seats2 == [1, 2, 0, 1, 2, 0]
    assert set(seats2) == {0, 1, 2}


def test_run_script_threads_seat_rotation_and_heuristic_anneal():
    """The repl_08 v3.2 fixes are wired end-to-end: run_phase3.sh exposes
    ROTATE_SEATS / HEURISTIC_ANNEAL_END knobs and threads the matching flags;
    phase3_grpo.py defines the args; modal_train.py exposes the params + env."""
    s = (P3 / "run_phase3.sh").read_text()
    assert ': "${ROTATE_SEATS:=1}"' in s            # default ON (train all seats)
    assert ': "${HEURISTIC_ANNEAL_END:=0}"' in s    # default off (floor-only)
    assert '[[ "${ROTATE_SEATS}" == "1" ]] && cmd+=( --rotate-seats )' in s
    assert '--heuristic-anneal-end "${HEURISTIC_ANNEAL_END}"' in s
    g = (P3 / "phase3_grpo.py").read_text()
    assert '"--rotate-seats"' in g
    assert '"--heuristic-anneal-end"' in g
    assert "heuristic_anneal_end=args.heuristic_anneal_end" in g  # both pool sites
    assert g.count("heuristic_anneal_end=args.heuristic_anneal_end") == 2
    m = (REPO_ROOT / "modal_train.py").read_text()
    assert "rotate_seats: bool = True" in m
    assert "heuristic_anneal_end: int = 0" in m
    assert '"ROTATE_SEATS": "1" if rotate_seats else "0"' in m
    assert '"HEURISTIC_ANNEAL_END": str(heuristic_anneal_end)' in m


def test_dual_gate_flash_primary_is_wired():
    """P0-2: the held-out vs-Flash gate is the BINDING spend criterion by
    default (dual_gate_flash_primary=True); the §3.6 heuristic gate becomes
    informational and can no longer veto a transferring policy. The legacy
    AND-mode short-circuit is gated behind `not dual_gate_flash_primary`."""
    m = (REPO_ROOT / "modal_train.py").read_text()
    assert "dual_gate_flash_primary: bool = True" in m
    # dual_gate_* params are phase3_main-local (used in the post-train dual-gate
    # block), NOT threaded to phase3.remote() — so they're absent from kwargs.
    assert "flash_primary=dual_gate_flash_primary" in m
    # The §3.6-fail short-circuit must be guarded by legacy (non-primary) mode.
    assert "(not dual_gate_flash_primary) and (not gate_36_pass)" in m
    # Flash binds the verdict in primary mode; §3.6 only ANDs in legacy mode —
    # the composition itself now lives in the package.
    d = (REPO_ROOT / "src/megagem/training/dual_gate.py").read_text()
    assert ("spend = flash_pass if flash_primary "
            "else (gate_36_pass and flash_pass)") in d


# --------------------------------------------------------------------------- #
# M1/M2/M3 — signal-diagnostic instruments (pure functions, no GPU).          #
# --------------------------------------------------------------------------- #
def _p3():
    import importlib
    return importlib.import_module("phase3_grpo")


def _roll_with_anchor(step, win_rate, n_total=30, anchor_key=0):
    return {"step": step,
            "per_snapshot_age": {"by_step": {
                anchor_key: {"win_rate": win_rate, "n_total": n_total}}}}


def test_anchor_winrate_trend_positive_slope():
    m = _p3()
    rolls = [_roll_with_anchor(s, wr)
             for s, wr in [(0, 0.30), (4, 0.36), (8, 0.42), (12, 0.48)]]
    t = m._anchor_winrate_trend(rolls)
    assert t["slope"] is not None and t["slope"] > 0
    assert t["delta_half"] > 0
    assert t["n_rolls_with_anchor"] == 4
    assert t["total_anchor_games"] == 120
    # spearman of a monotone-rising series is +1
    assert t["spearman_wr_vs_step"] == pytest.approx(1.0)
    # A clean near-linear rise is highly significant ⇒ gate reads 'improving'.
    assert t["slope_t"] is not None and t["slope_t"] >= 2.0
    assert m._anchor_winrate_improving(t) is True


def test_anchor_winrate_trend_flat_is_chance():
    """A perfectly flat anchor win-rate is the EXPECTED null — slope must be
    exactly 0.0 (not None), slope_t 0.0, and the gate reads 'not improving'."""
    m = _p3()
    rolls = [_roll_with_anchor(s, 0.333) for s in (0, 4, 8, 12)]
    t = m._anchor_winrate_trend(rolls)
    assert t["slope"] == 0.0
    assert t["delta_half"] == 0.0
    assert t["slope_t"] == 0.0
    assert m._anchor_winrate_improving(t) is False


def test_anchor_winrate_gate_requires_significance():
    """The gate tests slope SIGNIFICANCE, not sign. A noisy positive slope (the
    real repl_09 anchor-WR series: slope>0 but t≈1.8) must read 'not improving'
    — guarding the false-positive the old `slope>0` rule produced."""
    m = _p3()
    repl09 = [0.271, 0.286, 0.312, 0.312, 0.304, 0.250, 0.385, 0.325, 0.288,
              0.250, 0.384, 0.341, 0.362, 0.266, 0.359, 0.284, 0.330, 0.411]
    rolls = [_roll_with_anchor(i * 4, wr, n_total=96)
             for i, wr in enumerate(repl09)]
    t = m._anchor_winrate_trend(rolls)
    assert t["slope"] > 0                        # positive drift …
    assert 1.0 < t["slope_t"] < 2.0              # … but NOT significant (t≈1.8)
    assert m._anchor_winrate_improving(t) is False           # default t_min=2.0
    assert m._anchor_winrate_improving(t, t_min=1.5) is True  # threshold tunable
    # slope=None (too few anchor draws) ⇒ non-fatal pass; <3 pts falls back to sign.
    assert m._anchor_winrate_improving({"slope": None}) is True
    assert m._anchor_winrate_improving({"slope": 0.01, "slope_t": None}) is True


def test_anchor_winrate_trend_ignores_rolls_without_anchor():
    """Rolls where the step-0 anchor wasn't drawn (n_total=0 or absent) are
    excluded; str-keyed by_step (JSON-loaded) is handled identically."""
    m = _p3()
    rolls = [
        {"step": 0, "per_snapshot_age": {"by_step": {"0": {"win_rate": 0.30, "n_total": 10}}}},
        {"step": 4, "per_snapshot_age": {"by_step": {"100": {"win_rate": 0.9, "n_total": 10}}}},  # no anchor
        {"step": 8, "per_snapshot_age": {"by_step": {"0": {"win_rate": 0.50, "n_total": 0}}}},     # n=0 excluded
        {"step": 12, "per_snapshot_age": {"by_step": {"0": {"win_rate": 0.46, "n_total": 14}}}},
    ]
    t = m._anchor_winrate_trend(rolls)
    assert t["n_rolls_with_anchor"] == 2          # only steps 0 and 12 count
    assert t["total_anchor_games"] == 24          # 10 + 14
    assert t["slope"] is not None and t["slope"] > 0


def test_component_var_decomposition_shares_sum_sensibly():
    """When only the terminal margin varies across the K rollouts (shaping
    held constant), virtually all within-group reward variance is terminal and
    the λ-shaping share is ~0 — the cosmetic-shaping signature."""
    m = _p3()
    rows = [
        {"group_key": "g1", "game_id": gid,
         "reward_components": {"terminal": term, "shaping": 0.02,
                               "legal": 0.0, "terminal_correction": 0.0}}
        for gid, term in enumerate([0.1, 0.5, -0.3, 0.9, 0.2, -0.1, 0.7, 0.0])
    ]
    d = m._reward_component_var_decomposition(rows)
    sh = d["within_group_var_share"]
    assert d["n_groups_usable"] == 1
    assert sh["shaping"] == pytest.approx(0.0, abs=1e-9)
    assert sh["terminal"] == pytest.approx(1.0, abs=1e-9)
    # components are not independent, but with one varying component the
    # cross-term is ~0 so Σshares ≈ 1.
    assert d["sum_of_component_vars_over_total"] == pytest.approx(1.0, abs=1e-9)


def test_component_var_decomposition_handles_degenerate_group():
    """A K-group whose per-game totals are all identical (var_total≈0) is
    excluded from the share averages — no divide-by-zero, shares are None."""
    m = _p3()
    rows = [
        {"group_key": "gd", "game_id": i,
         "reward_components": {"terminal": 0.5, "shaping": 0.0,
                               "legal": 0.0, "terminal_correction": 0.0}}
        for i in range(4)
    ]
    d = m._reward_component_var_decomposition(rows)
    assert d["n_groups_usable"] == 0
    assert d["within_group_var_share"]["terminal"] is None
    assert d["sum_of_component_vars_over_total"] is None


def _multi_bucket_rows(half_tilt, *, groups, k=8):
    """Rows spanning 6 board-agnostic buckets (2 phases × 3 round-thirds).
    `half_tilt(half, phase, third) -> advantage` lets a test set each half's
    systematic per-bucket tilt. group_keys g0..g{groups-1}; even-index sorted
    keys land in half 0, odd in half 1 (matches the probe's split)."""
    rows = []
    gid = 0
    for g in range(groups):
        gk = f"g{g}"
        half = g % 2  # sorted g0,g1,… → even idx half0, odd half1
        for _ in range(k):
            for rnd in range(6):          # rmax=5 → thirds 0,0,1,1,2,2
                third = (3 * rnd) // 6
                for phase in ("bid", "reveal"):
                    rows.append({
                        "group_key": gk, "game_id": gid, "round_index": rnd,
                        "phase": phase,
                        "precomputed_advantage": half_tilt(half, phase, third),
                    })
            gid += 1
    return rows


def test_transferable_probe_high_corr_when_consistent():
    """Both disjoint halves share the SAME per-bucket advantage tilt ⇒ the
    split-half correlation is ≈+1 (a durable, transferable signal)."""
    m = _p3()
    tilt = lambda half, phase, third: (1.0 if phase == "bid" else -1.0) * (third + 1)
    rows = _multi_bucket_rows(tilt, groups=6)
    p = m._transferable_signal_probe(rows)
    assert p["n_shared_buckets"] == 6
    assert p["split_half_corr"] == pytest.approx(1.0, abs=1e-9)
    assert p["transferable_magnitude"] > 0
    # roll-to-roll cosine vs the same vector is 1.
    p2 = m._transferable_signal_probe(rows, prev_vector=p["vector"])
    assert p2["self_consistency_cosine"] == pytest.approx(1.0, abs=1e-9)


def test_transferable_probe_low_corr_when_random():
    """When the two halves carry OPPOSITE per-bucket tilts (no shared
    transferable direction), the split-half correlation is strongly negative —
    the probe does not mistake board-idiosyncratic gradients for signal."""
    m = _p3()
    # half 0: bid +, reveal − ; half 1: the negation → anti-correlated.
    tilt = lambda half, phase, third: (
        (1.0 if phase == "bid" else -1.0) * (third + 1) * (1 if half == 0 else -1))
    rows = _multi_bucket_rows(tilt, groups=6)
    p = m._transferable_signal_probe(rows)
    assert p["n_shared_buckets"] == 6
    assert p["split_half_corr"] is not None and p["split_half_corr"] < -0.5


def test_transferable_probe_disjoint_halves():
    """A group's rows never split across halves: if group g0 (→half0) only ever
    plays 'bid' and g1 (→half1) only 'reveal', the halves share NO bucket, so
    split_half_corr is None — proving the split is by group_key (disjoint
    boards), exactly what makes the cross-half correlation meaningful."""
    m = _p3()
    rows = []
    for gid in range(8):
        rows.append({"group_key": "g0", "game_id": gid, "round_index": 0,
                     "phase": "bid", "precomputed_advantage": 0.5})
    for gid in range(8, 16):
        rows.append({"group_key": "g1", "game_id": gid, "round_index": 0,
                     "phase": "reveal", "precomputed_advantage": -0.5})
    p = m._transferable_signal_probe(rows)
    assert p["n_groups"] == 2
    assert p["n_shared_buckets"] == 0
    assert p["split_half_corr"] is None


def _seat_rows(tilt, *, seat, groups=6, gid0=0, rounds=3):
    """K-group rows for ONE seat (player_id=seat); disjoint group_keys so the
    split-half is by board. ``tilt(phase, third)`` sets the per-bucket advantage."""
    rows = []
    gid = gid0
    for g in range(groups):
        gk = f"s{seat}_g{g}"
        for rnd in range(rounds):
            third = min(2, (3 * rnd) // rounds)
            for phase in ("bid", "reveal"):
                rows.append({"group_key": gk, "game_id": gid,
                             "round_index": rnd, "phase": phase,
                             "player_id": seat,
                             "precomputed_advantage": tilt(phase, third)})
            gid += 1
    return rows


def test_transferable_probe_tags_seat():
    """The trainable seat rotates per roll, so a single roll is one seat; the
    probe tags it from player_id."""
    m = _p3()
    rows = _seat_rows(lambda phase, third: 1.0 if phase == "bid" else -1.0, seat=2)
    p = m._transferable_signal_probe(rows)
    assert p["seat"] == 2
    assert p["n_seats_in_roll"] == 1


def test_transferable_probe_cross_seat_invariant_high_cosine():
    """A tilt that is the SAME across seats ⇒ cross_seat_cosine ≈ +1: a genuine
    seat-INVARIANT transferable direction, NOT a positional artifact. (The test
    the fixed-seat-0 v3_1/v3_2 runs could not run.)"""
    m = _p3()
    tilt = lambda phase, third: (1.0 if phase == "bid" else -1.0) * (third + 1)
    state: dict = {}
    a = m._transferable_signal_probe(
        _seat_rows(tilt, seat=0), seat_vectors=state.get("seat_vectors"))
    m._accumulate_seat_vector(state, a)
    b = m._transferable_signal_probe(
        _seat_rows(tilt, seat=1, gid0=1000),
        seat_vectors=state.get("seat_vectors"))
    assert b["seat"] == 1
    assert b["cross_seat_cosine_min"] == pytest.approx(1.0, abs=1e-9)


def test_transferable_probe_cross_seat_artifact_negative_cosine():
    """When seat 1's tilt is the NEGATION of seat 0's, the cross-seat cosine is
    strongly negative — the smoking gun for a POSITIONAL artifact (seat-specific
    tilt masquerading as transferable signal)."""
    m = _p3()
    tilt0 = lambda phase, third: (1.0 if phase == "bid" else -1.0) * (third + 1)
    state: dict = {}
    a = m._transferable_signal_probe(_seat_rows(tilt0, seat=0))
    m._accumulate_seat_vector(state, a)
    b = m._transferable_signal_probe(
        _seat_rows(lambda phase, third: -tilt0(phase, third), seat=1, gid0=1000),
        seat_vectors=state.get("seat_vectors"))
    assert b["cross_seat_cosine_min"] is not None
    assert b["cross_seat_cosine_min"] < -0.5


def test_transferable_probe_length_tercile_populates():
    """Games of varying length ⇒ by_length_tercile reports per-tercile magnitude
    so a length-localised tilt is visible (and is None when lengths are uniform)."""
    m = _p3()
    rows = []
    gid = 0
    for length in (1, 3, 6):          # 3 games each at 3 distinct lengths
        for _ in range(3):
            for rnd in range(length):
                for phase in ("bid", "reveal"):
                    rows.append({"group_key": f"g{gid}", "game_id": gid,
                                 "round_index": rnd, "phase": phase,
                                 "player_id": 0,
                                 "precomputed_advantage": 0.5 if phase == "bid"
                                 else -0.5})
            gid += 1
    p = m._transferable_signal_probe(rows)
    assert p["by_length_tercile"] is not None
    assert set(p["by_length_tercile"]) <= {"short", "mid", "long"}
    assert len(p["by_length_tercile"]) >= 2
