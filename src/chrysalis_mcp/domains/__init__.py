"""Domain registration. New domains (memory, reason) are added here with one line each."""

from ..core.registry import Registry
from . import memory, meta, reason


def build_registry() -> Registry:
    registry = Registry()
    meta.register(registry)
    reason.register(registry)
    memory.register(registry)
    return registry
