"""``.env.example`` documents exactly the CHRYSALIS_* variables the code reads."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = re.compile(r"CHRYSALIS_[A-Z0-9_]*[A-Z0-9]")
# Names that appear in source only as a family prefix (``CHRYSALIS_LLM_*``), never as a variable.
PREFIXES = frozenset({"CHRYSALIS_LLM"})


def _names(paths: list[Path]) -> set[str]:
    found: set[str] = set()
    for path in paths:
        found.update(NAME.findall(path.read_text(encoding="utf-8")))
    return found - PREFIXES


def _source_names() -> set[str]:
    files = [p for d in ("src", "validation") for p in (ROOT / d).rglob("*.py")]
    return _names(files)


def _documented_names() -> set[str]:
    return _names([ROOT / ".env.example"])


def test_every_variable_in_code_is_documented() -> None:
    missing = _source_names() - _documented_names()
    assert not missing, f"missing from .env.example: {sorted(missing)}"


def test_no_stale_entries_in_env_example() -> None:
    stale = _documented_names() - _source_names()
    assert not stale, f"in .env.example but unused in code: {sorted(stale)}"
