"""Test bootstrap.

The ``megagem`` package itself is imported from the editable install
(``uv sync``); nothing here touches its import path. This file only exposes
the script *drivers* (scripts/ is deliberately not installed) and shared skip
markers for the optional heavy training deps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Script drivers under test (phase3_grpo.py, phase3_eval.py, bibd_eval.py,
# flash_bid_model_fit.py, ...) live in scripts/, outside the package.
for _p in (
    REPO_ROOT / "scripts" / "training",
    REPO_ROOT / "scripts" / "eval",
    REPO_ROOT / "scripts" / "analysis",
):
    sys.path.insert(0, str(_p))


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


# TRL/torch are deliberately not project deps: training runs inside the Modal
# image, which rebuilds the TRL fork from public upstream (docs/training.md).
requires_trl = pytest.mark.skipif(
    not _importable("trl"),
    reason="TRL fork not installed (training runs in the Modal image; "
    "see docs/training.md)",
)
requires_torch = pytest.mark.skipif(
    not _importable("torch"), reason="torch not installed"
)
