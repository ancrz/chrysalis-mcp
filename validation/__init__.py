"""Validation battery for I4-M4: hypotheses H-a, H-a', H-b, H-c, H-mcp.

Topos: this package consumes ``chrysalis_mcp`` (public stdio tool, algorithms, substrate);
``chrysalis_mcp`` must never import ``validation`` (enforced by tests).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR_ENV = "CHRYSALIS_RESULTS_DIR"


def _results_dir() -> Path:
    """Env override (absolute, or relative to ROOT) so runs of different models do not collide."""
    raw = os.environ.get(RESULTS_DIR_ENV, "").strip()
    return ROOT / raw if raw else ROOT / "validation" / "results"  # absolute raw wins in ROOT / raw


RESULTS_DIR = _results_dir()
REPORT_PATH = RESULTS_DIR / "RESULTS.md"
BYTES_PER_TOKEN = 4  # same estimate the server envelope uses for est_tokens


def est_tokens(n_bytes: int) -> int:
    return max(1, round(n_bytes / BYTES_PER_TOKEN))
