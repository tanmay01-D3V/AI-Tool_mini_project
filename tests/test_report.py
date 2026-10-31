# tests/test_report.py
"""Smoke test for the HTML report rendering."""

from __future__ import annotations

import pathlib as _pl

import pandas as pd
import pytest

from cleaner.report import build_report_context, render_report


@pytest.fixture
def tiny_context():
    # Minimal data frames – 2 rows, 2 columns
    raw = pd.DataFrame(
        {"col_a": ["A", pd.NA], "col_b": ["", "placeholder"]},
        dtype="string",
    )
    clean = raw.copy()
    clean["col_a"] = clean["col_a"].str.upper()
    flags = pd.DataFrame(
        [
            {"row": 2, "column": "col_b", "value": "", "issue_type": "missing_value", "reason": "blank"},
            {"row": 2, "column": "col_b", "value": "placeholder", "issue_type": "missing_value", "reason": "placeholder: placeholder"},
        ]
    )
    change_log = pd.DataFrame(
        [
            {"row": 1, "column": "col_a", "old_value": "A", "new_value": "A", "rule": "title_case"},
        ]
    )
    failures = pd.DataFrame(columns=["row", "column", "value", "check", "reason"])
    return {
        "raw_df": raw,
        "clean_df": clean,
        "flags_before": flags,
        "flags_after": flags,
        "change_log": change_log,
        "failures_before": failures,
        "failures_after": failures,
        "ai_summary": "<script>alert('x')</script>",
        "meta": {"source_file": "tiny.csv"},
    }


def test_report_renders_all_sections(tiny_context, tmp_path):
    ctx = build_report_context(
        tiny_context["raw_df"],
        tiny_context["clean_df"],
        tiny_context["flags_before"],
        tiny_context["flags_after"],
        tiny_context["change_log"],
        tiny_context["failures_before"],
        tiny_context["failures_after"],
        ai_summary=tiny_context["ai_summary"],
        meta=tiny_context["meta"],
    )
    out_file = tmp_path / "report.html"
    render_report(ctx, out_file)

    html = out_file.read_text(encoding="utf-8")
    # Section headings (13 total)
    headings = [
        "AI-generated (Gemini)",
        "Dataset overview",
        "Quality score",
        "Issues by type",
        "Issues by column",
        "Missing values",
        "Duplicates",
        "Format inconsistencies",
        "Invalid values",
        "Changes applied",
        "Items needing human review",
        "Pandera validation",
        "Gemini usage",
        "Method and limitations",
    ]
    for h in headings:
        assert f"<h2>{h}</h2>" in html

    # Ensure the injected script tag is escaped (no raw <script>)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html

    # No external URLs
    assert "http://" not in html
    assert "https://" not in html