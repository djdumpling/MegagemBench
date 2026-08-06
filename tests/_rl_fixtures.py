"""Shared schema-v3 fixture discovery for the Phase 1 RL tests.

``results/`` is gitignored scratch: regenerable output that contains private
hands, so it is never committed and a clean checkout has no corpus at all.
``tests/fixtures/*.json`` therefore ships one tiny *sanitized* schema-v3 game
(hands scrubbed, free text stubbed, ``tiebreak_order_before`` injected) so the
high-value oracle tests run everywhere; a local ``results/sft_v2_val_only``
corpus is picked up opportunistically when it happens to be present.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKED_DIR = REPO_ROOT / "tests" / "fixtures"
CORPUS_DIR = REPO_ROOT / "results" / "sft_v2_val_only"


def _is_schema_v3(path: Path) -> bool:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(data, dict) and data.get("metadata", {}).get("schema_version") == 3


def fixture_paths() -> list[Path]:
    """Tracked sanitized fixtures (always) + the local corpus (if present)."""
    paths = sorted(TRACKED_DIR.glob("*.json"))
    if CORPUS_DIR.is_dir():
        paths += sorted(CORPUS_DIR.glob("*.json"))
    return [p for p in paths if _is_schema_v3(p)]


def load(path: Path) -> dict:
    return json.loads(path.read_text())
