"""Validation battery for I4-M4: hypotheses H-a, H-a', H-b, H-c, H-mcp.

Topos: this package consumes ``chrysalis_mcp`` (public stdio tool, algorithms, substrate);
``chrysalis_mcp`` must never import ``validation`` (enforced by tests).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "validation" / "results"
REPORT_PATH = RESULTS_DIR / "RESULTS.md"
BYTES_PER_TOKEN = 4  # same estimate the server envelope uses for est_tokens


def est_tokens(n_bytes: int) -> int:
    return max(1, round(n_bytes / BYTES_PER_TOKEN))
