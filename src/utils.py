"""Shared utility functions."""

import json
import os
from pathlib import Path


def get_project_root() -> str:
    """Return the project root directory."""
    return str(Path(__file__).parent.parent)


def load_config(config_path: str = None) -> dict:
    """Load project configuration."""
    if config_path is None:
        config_path = os.path.join(get_project_root(), "configs", "default.json")
    with open(config_path) as f:
        return json.load(f)


def load_split(bench_dir: str, logic: str) -> dict:
    """Load the three-tier benchmark split."""
    split_path = os.path.join(bench_dir, logic, "split_three_tier.json")
    if not os.path.exists(split_path):
        # Fall back to original split
        split_path = os.path.join(bench_dir, logic, "split.json")
    with open(split_path) as f:
        return json.load(f)
