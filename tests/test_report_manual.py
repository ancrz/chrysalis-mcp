"""Report regeneration keeps the hand-written section; evidence sources resolve to real headings."""

import json
import re
from pathlib import Path

from validation import RESULTS_DIR, report

EVIDENCE = Path(__file__).resolve().parent.parent / "src/chrysalis_mcp/data/evidence.json"


def _slug(heading: str) -> str:
    """GitHub rule: lowercase, drop punctuation except '-' and spaces, spaces become '-'."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def test_manual_section_survives_regeneration(tmp_path):
    out = tmp_path / "RESULTS.md"
    manual = "## Hand-written\n\nkeep  this\n  verbatim, <b>incl</b> spacing\n\n"
    report.write(RESULTS_DIR, out)  # first write: no marker yet, nothing to preserve
    assert report.split_manual(out.read_text()) == ""
    out.write_text(out.read_text() + manual)
    report.write(RESULTS_DIR, out)
    text = out.read_text()
    assert text.count(report.MANUAL_MARKER) == 1 and report.split_manual(text) == manual
    assert text.startswith(report.render(RESULTS_DIR).rstrip("\n"))
    report.write(RESULTS_DIR, out)  # idempotent
    assert out.read_text() == text


def test_content_before_the_marker_is_regenerated(tmp_path):
    out = tmp_path / "RESULTS.md"
    out.write_text("stale generated text\n" + report.MANUAL_MARKER + "\nmanual\n")
    report.write(RESULTS_DIR, out)
    text = out.read_text()
    assert "stale generated text" not in text and report.split_manual(text) == "manual\n"


def test_every_evidence_source_anchor_matches_a_report_heading():
    rendered = report.render(RESULTS_DIR)
    slugs = {_slug(m.group(1)) for m in re.finditer(r"^#{1,6} (.+)$", rendered, re.M)}
    sources = {e["source"] for e in json.loads(EVIDENCE.read_text())["entries"]}
    assert sources
    for source in sources:
        path, _, anchor = source.partition("#")
        assert path == "validation/results/RESULTS.md" and anchor in slugs, source
