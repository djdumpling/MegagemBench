"""TRL introspection helpers shared by the GRPO harness.

Small, dependency-light utilities for adapting to the pinned TRL commit's
actual signatures: ``_filter_kwargs`` drops kwargs a target does not accept,
``print_trl_env`` reports which fields the installed ``GRPOConfig`` exposes.
The live rollout-contract seam is ``megagem.training.grpo_harness`` — see
docs/training.md for the pinned stack and the fork recipe.
"""

from __future__ import annotations

import inspect
from dataclasses import fields, is_dataclass
from typing import Any


def _filter_kwargs(target: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs the target (callable or dataclass) actually accepts."""
    if is_dataclass(target):
        allowed = {f.name for f in fields(target)}
    else:
        sig = inspect.signature(target)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return dict(kwargs)
        allowed = set(sig.parameters)
    dropped = sorted(set(kwargs) - allowed)
    if dropped:
        print(f"[harness] note: dropping unsupported kwargs for "
              f"{getattr(target, '__name__', target)}: {dropped}")
    return {k: v for k, v in kwargs.items() if k in allowed}


def print_trl_env() -> None:
    """Diagnostic preamble — version + the seam signatures, so a first-run
    failure is self-explanatory without source spelunking."""
    import trl
    from trl import GRPOConfig, GRPOTrainer

    print(f"[harness] trl {getattr(trl, '__version__', '?')}")
    init_params = list(inspect.signature(GRPOTrainer.__init__).parameters)
    print(f"[harness] GRPOTrainer.__init__ params: {init_params}")
    for seam in ("rollout_func", "peft_config", "reward_funcs", "processing_class"):
        print(f"[harness]   has '{seam}': {seam in init_params}")
    cfg_fields = {f.name for f in fields(GRPOConfig)} if is_dataclass(GRPOConfig) else set()
    for f in ("num_generations", "use_liger_loss", "use_liger_kernel",
              "use_vllm", "max_completion_length", "steps_per_generation"):
        print(f"[harness]   GRPOConfig.{f}: {'yes' if f in cfg_fields else 'MISSING'}")
