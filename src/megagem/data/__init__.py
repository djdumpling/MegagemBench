"""MegaGem data loading utilities."""

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent


def load_json(filename: str) -> Any:
    """Load a JSON file from the data directory."""
    with open(DATA_DIR / filename) as f:
        return json.load(f)


def load_gems() -> list[dict]:
    """Load gem card definitions."""
    return load_json("gems.json")


def load_auctions() -> list[dict]:
    """Load auction card definitions."""
    return load_json("auctions.json")


def load_missions() -> list[dict]:
    """Load mission card definitions."""
    return load_json("missions.json")


def load_value_charts() -> dict[str, list[dict]]:
    """Load value chart definitions."""
    return load_json("value_charts.json")
