"""Server-wide switch for the opt-in bio features (``CHRYSALIS_BIO_DEFAULTS``).

The bio features of ``memory`` (usage tracking, adaptive inhibition, FlyVec encoder, replay,
usage weighting, tagging/capture) are per-call params that default to off, so a client that never
passes them never exercises them. ``CHRYSALIS_BIO_DEFAULTS=all`` flips what an OMITTED param
means; an explicit param always wins. The table below is the single place those defaults live.

The environment is read at call time (never cached), so a test or a launcher can change it.
``mode`` is deliberately absent: ``sdm`` is an alternative read mode, not an add-on.
"""

import os
from typing import Any

ENV_VAR = "CHRYSALIS_BIO_DEFAULTS"
PROFILES = ("off", "all")

# action -> param -> default applied when the caller omits it under the "all" profile.
ALL_DEFAULTS: dict[str, dict[str, Any]] = {
    "recall": {"track_usage": True, "inhibition": "adaptive", "encoder": "flyvec"},
    "consolidate": {"replay": True, "usage_weighting": True, "capture": True},
}
# Auto-train FlyVec after ingest only up to this many nodes (training cost grows with the KB).
AUTO_FLYVEC_MAX_NODES = 5000


def profile() -> str:
    """Active profile; anything but ``all`` (including garbage) means ``off``."""
    value = os.environ.get(ENV_VAR, "off").strip().lower()
    return value if value in PROFILES else "off"


def apply(action: str, params: dict[str, Any]) -> dict[str, Any]:
    """Params with the profile defaults filled in for omitted keys (``off``: same object)."""
    if profile() != "all":
        return params
    extra = {k: v for k, v in ALL_DEFAULTS.get(action, {}).items() if k not in params}
    return {**params, **extra} if extra else params
