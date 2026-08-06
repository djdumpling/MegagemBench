"""Thin TRL GRPO glue for MegaGem RL.

The trainer should not own MegaGem reward or RAE logic. Rollout/reward code
computes scalar per-turn rewards and advantages, then passes them through TRL's
``rollout_func`` extra fields. This subclass keeps TRL's generation, logprob,
masking, reward logging, and KL plumbing intact, replacing only the advantage
tensor consumed by ``_compute_loss``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:  # Optional RL dependency; base MegaGem workflows do not require TRL.
    import torch
    from trl import GRPOTrainer as _TRLGRPOTrainer
except (ImportError, RuntimeError) as exc:  # pragma: no cover - exercised in non-RL envs
    torch = None  # type: ignore[assignment]
    _TRLGRPOTrainer = object  # type: ignore[assignment,misc]
    _TRL_IMPORT_ERROR: BaseException | None = exc
else:
    _TRL_IMPORT_ERROR = None

    # §5 #6-class compatibility (transformers axis). The pinned TRL commit
    # assumes a newer transformers than our pinned 4.55.4 (held back for the
    # Qwen tokenizer; see docs/training.md for the pinned stack). TRL's
    # GRPOConfig.__post_init__ reads TrainingArguments fields that 4.55.4 does
    # not define (e.g. ``parallelism_config``, an accelerate ParallelismConfig
    # integration added later), raising AttributeError at config construction.
    # We run single-rank, so backfilling benign class-level defaults is a
    # no-op for behaviour and unblocks GRPOConfig. Add siblings here if more
    # missing attrs surface from the same version gap.
    try:
        from trl import GRPOConfig as _GRPOConfig

        _TF_COMPAT_DEFAULTS: dict[str, Any] = {"parallelism_config": None}
        for _attr, _default in _TF_COMPAT_DEFAULTS.items():
            if not hasattr(_GRPOConfig, _attr):
                setattr(_GRPOConfig, _attr, _default)
    except Exception:  # noqa: BLE001 - a compat shim must never block import
        pass


PRECOMPUTED_REWARD_KEY = "precomputed_reward"
PRECOMPUTED_ADVANTAGE_KEY = "precomputed_advantage"


def _ensure_rl_stack() -> None:
    if _TRL_IMPORT_ERROR is not None:
        raise ImportError(
            "MegaGemGRPOTrainer requires the optional RL stack. Install it with "
            "`uv sync --extra rl` before importing or constructing the trainer."
        ) from _TRL_IMPORT_ERROR


def precomputed_reward_func(
    *,
    prompts: Sequence[Any],
    completions: Sequence[Any],
    precomputed_reward: Sequence[float | int | None],
    **_: Any,
) -> list[float | None]:
    """TRL reward-function shim that returns rollout-provided rewards.

    TRL requires at least one reward function even when the environment has
    already scored the turns. The output remains useful for TRL's reward logs;
    policy optimization uses ``precomputed_advantage`` via
    ``MegaGemGRPOTrainer`` below.
    """
    del completions
    if len(precomputed_reward) != len(prompts):
        raise ValueError(
            f"{PRECOMPUTED_REWARD_KEY!r} length ({len(precomputed_reward)}) "
            f"does not match batch size ({len(prompts)})."
        )
    return [None if reward is None else float(reward) for reward in precomputed_reward]


def _replacement_advantages(
    inputs: Sequence[dict[str, Any]],
    output: dict[str, Any],
    *,
    key: str = PRECOMPUTED_ADVANTAGE_KEY,
):
    """Build the replacement advantage tensor from per-row input metadata."""
    _ensure_rl_stack()
    if "advantages" not in output:
        raise KeyError("TRL output is missing 'advantages'; cannot replace them.")

    missing = [i for i, row in enumerate(inputs) if key not in row]
    if missing:
        preview = ", ".join(str(i) for i in missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise KeyError(
            f"Missing {key!r} on {len(missing)} input rows: {preview}{suffix}. "
            "The rollout_func must return it as an extra field."
        )

    reference = output["advantages"]
    advantages = torch.as_tensor(  # type: ignore[union-attr]
        [row[key] for row in inputs],
        dtype=reference.dtype,
        device=reference.device,
    )

    valid_shapes = {tuple(reference.shape)}
    if "completion_mask" in output:
        valid_shapes.add(tuple(output["completion_mask"].shape))
    if tuple(advantages.shape) not in valid_shapes:
        valid = ", ".join(str(shape) for shape in sorted(valid_shapes))
        raise ValueError(
            f"{key!r} produced advantage shape {tuple(advantages.shape)}, "
            f"expected one of: {valid}."
        )
    return advantages


# TRL metrics whose grouping / EOS assumptions do not match our wiring:
#   * ``frac_reward_zero_std`` and ``rewards/precomputed_reward_func/std``
#     reshape rewards by ``num_generations`` over our flattened
#     (seed, k, round, pid) rows — that std lands on heterogeneous rows and is
#     uninterpretable (sticks at 1.0). The honest K-group std is computed in
#     ``phase3_grpo._gpu_run`` on the FULL roll output (before the
#     ``_balanced_select`` subset), keyed on ``(group_key, game_id)``; see
#     ``rolls[*].kgroup_reward_stats`` in ``phase3_grpo.json``.
#   * ``completions/clipped_ratio`` and ``completions/mean_terminated_length``
#     check ``last_completion_token == tokenizer.eos_token_id``. vLLM's chat
#     endpoint strips the stop token from the returned text, so completion_ids
#     never end with EOS regardless of what eos_token_id is set to —
#     clipped_ratio sticks at 1.0 even when every completion ended naturally.
#     The honest replacement, computed below from ``completion_mask`` vs
#     ``args.max_completion_length``, is ``megagem/completions/at_budget_rate``.
_MISLEADING_TRL_METRIC_KEYS = (
    "frac_reward_zero_std",
    "rewards/precomputed_reward_func/std",
    "completions/clipped_ratio",
    "completions/mean_terminated_length",
    "completions/min_terminated_length",
    "completions/max_terminated_length",
)


class MegaGemGRPOTrainer(_TRLGRPOTrainer):  # type: ignore[misc]
    """GRPOTrainer variant that consumes environment-computed advantages."""

    advantage_key = PRECOMPUTED_ADVANTAGE_KEY

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _ensure_rl_stack()
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        # Snapshot pre-super metric lengths so we can pop ONLY the entries TRL
        # appended this step (not values left from earlier steps that log()
        # hasn't drained yet). See _suppress_misleading_trl_metrics below.
        pre_lens = self._snapshot_metric_lens(_MISLEADING_TRL_METRIC_KEYS)
        output = super()._generate_and_score_completions(inputs)
        advantages = _replacement_advantages(
            inputs, output, key=self.advantage_key
        )
        output["advantages"] = advantages
        self._log_megagem_advantages(advantages)
        self._log_megagem_completion_health(output)
        self._suppress_misleading_trl_metrics(pre_lens)
        return output

    def _snapshot_metric_lens(
        self, keys: Sequence[str]
    ) -> dict[str, dict[str, int]]:
        metrics = getattr(self, "_metrics", None)
        if metrics is None:
            return {}
        out: dict[str, dict[str, int]] = {}
        for mode in ("train", "eval"):
            section = metrics.get(mode)
            if section is None:
                continue
            out[mode] = {k: len(section.get(k, ())) for k in keys}
        return out

    def _log_megagem_advantages(self, advantages: Any) -> None:
        """Log replacement-advantage stats without editing TRL's stock logs."""
        metrics = getattr(self, "_metrics", None)
        model = getattr(self, "model", None)
        if metrics is None or model is None:
            return

        mode = "train" if model.training else "eval"
        values = advantages.detach().float()
        finite = values[torch.isfinite(values)]  # type: ignore[index,operator]
        if finite.numel() == 0:
            metrics[mode]["megagem/advantages/count"].append(0)
            return

        metrics[mode]["megagem/advantages/count"].append(int(finite.numel()))
        metrics[mode]["megagem/advantages/mean"].append(finite.mean().item())
        metrics[mode]["megagem/advantages/std"].append(
            finite.std(unbiased=False).item()
        )
        metrics[mode]["megagem/advantages/min"].append(finite.min().item())
        metrics[mode]["megagem/advantages/max"].append(finite.max().item())

    def _log_megagem_completion_health(self, output: dict[str, Any]) -> None:
        """Honest replacement for TRL's ``completions/clipped_ratio`` and
        ``completions/mean_terminated_length``.

        TRL's checks compare ``last_completion_token == tokenizer.eos_token_id``,
        but vLLM's chat endpoint strips the stop token from the returned text
        — so completion_ids never end with EOS regardless of how
        ``eos_token_id`` is configured. The TRL values stick at
        ``clipped_ratio=1.0`` / ``mean_terminated_length=0`` even when every
        completion finished naturally at ``<|im_end|>``. They are popped from
        the log by ``_suppress_misleading_trl_metrics`` and replaced here.

        Length-budget definition (well-formed regardless of how the chat
        template chooses to terminate): a completion is "at budget" iff its
        non-pad token count reaches ``args.max_completion_length``. Anything
        shorter ended via vLLM's stop machinery → ``terminated``. This is
        exactly the operational question (is generation getting cut off by
        the budget?) without depending on which special token the chat
        template emits last.
        """
        metrics = getattr(self, "_metrics", None)
        model = getattr(self, "model", None)
        if metrics is None or model is None or "completion_mask" not in output:
            return
        mode = "train" if model.training else "eval"
        mask = output["completion_mask"]
        try:
            lengths = mask.detach().int().sum(dim=1).cpu().tolist()  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 — telemetry never aborts a step
            return
        if not lengths:
            return
        budget = int(getattr(self.args, "max_completion_length", 0) or 0)
        at_budget = (sum(1 for L in lengths if budget and L >= budget)
                     if budget else 0)
        n = len(lengths)
        metrics[mode]["megagem/completions/at_budget_rate"].append(
            at_budget / n)
        metrics[mode]["megagem/completions/terminated_rate"].append(
            (n - at_budget) / n)
        metrics[mode]["megagem/completions/n"].append(n)
        metrics[mode]["megagem/completions/max_budget"].append(budget)

    def _suppress_misleading_trl_metrics(
        self, pre_lens: dict[str, dict[str, int]]
    ) -> None:
        """Drop the entries TRL appended this step for metrics whose grouping
        or EOS-presence assumption doesn't match our wiring.

        Two failure modes:

        * **Reward-std keys** (``frac_reward_zero_std``,
          ``rewards/precomputed_reward_func/std``): TRL reshapes
          ``rewards.view(-1, num_generations).std(dim=1)``; our rollout
          returns ``rows_per_gen`` distinct ``(seed,k,round,pid)`` rows whose
          ``num_generations`` reshape lands on rows that don't belong
          together. Replacement (honest K-group std on the FULL roll output,
          keyed on ``(group_key, game_id)``) lives in
          ``phase3_grpo._gpu_run`` and is surfaced as
          ``phase3_grpo.json::rolls[*].kgroup_reward_stats``.

        * **Completion-termination keys** (``completions/clipped_ratio``,
          ``completions/{mean,min,max}_terminated_length``): TRL compares
          ``completion_ids[-1]`` to ``[eos_token_id, pad_token_id]``, but
          vLLM's chat endpoint strips the stop token from the returned text,
          so ``completion_ids`` never end with EOS regardless of
          ``eos_token_id``. Every completion looks "truncated" even when it
          finished naturally. Replacement (length-budget based) is
          ``megagem/completions/at_budget_rate`` from
          :meth:`_log_megagem_completion_health` — same step, same dict.

        Implementation: snapshot list lengths before super() and pop only the
        new appends — we never touch values from earlier steps that ``log()``
        hasn't drained yet.
        """
        metrics = getattr(self, "_metrics", None)
        if metrics is None or not pre_lens:
            return
        for mode, expected in pre_lens.items():
            section = metrics.get(mode)
            if section is None:
                continue
            for key, before in expected.items():
                series = section.get(key)
                if series is None:
                    continue
                while len(series) > before:
                    series.pop()
                # Popping leaves an empty list under the key; TRL's log()
                # computes ``mean(values) if values else None`` and emits
                # ``"<key>": None`` — noisier than the metric we were trying
                # to suppress. If the series is empty AND was empty before
                # super(), drop the key entirely so it never reaches the log.
                if before == 0 and not series:
                    try:
                        del section[key]
                    except KeyError:
                        pass
