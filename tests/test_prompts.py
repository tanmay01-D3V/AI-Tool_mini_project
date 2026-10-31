"""Tests for cleaner.prompts schemas, version constants, and rendering."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from cleaner.prompts import (
    DEFAULT_TEMPERATURE,
    P1ColumnProfile,
    P2FormatRules,
    P2Rule,
    P3CategoryMapping,
    P3_v3,
    P4MissingAdvice,
    P5Summary,
    render_p1_v1,
    render_p2_v1,
    render_p3_v1,
    render_p3_v2,
    render_p3_v3,
    render_p4_v1,
    render_p5_v1,
)


def test_default_temperature_is_zero() -> None:
    """Verify default temperature is 0."""
    assert DEFAULT_TEMPERATURE == 0.0


def test_p1_schema_and_render() -> None:
    """Verify P1 schema validation and prompt rendering."""
    data = {
        "semantic_type": "email",
        "expected_format": "standard email",
        "nullable": False,
        "confidence": 0.95,
        "reason": "All non-null values contain @ and valid domain.",
    }
    obj = P1ColumnProfile.model_validate(data)
    assert obj.semantic_type == "email"

    rendered = render_p1_v1("email", ["a@b.com", "c@d.com"], 1.5)
    assert 'column "email"' in rendered
    assert "1.50%" in rendered
    assert "ROLE: You are a careful data quality analyst." in rendered


def test_p2_schema_and_render() -> None:
    """Verify P2 schema validation and prompt rendering."""
    data = {
        "rules": [
            {
                "action": "parse_date",
                "params": {"day_first": True},
                "reason": "Converts mixed dates to ISO format.",
            }
        ]
    }
    obj = P2FormatRules.model_validate(data)
    assert len(obj.rules) == 1
    assert obj.rules[0].action == "parse_date"

    rendered = render_p2_v1("order_date", "date", {"DD/MM/YYYY": 120})
    assert 'column "order_date"' in rendered
    assert "DD/MM/YYYY" in rendered


def test_p3_v3_exact_text_and_schema() -> None:
    """Verify P3 v3 contains expected structural text and renders correctly."""
    assert "ROLE: You are a careful data quality analyst." in P3_v3
    assert 'TASK: Group spelling, casing and abbreviation variants of the same category' in P3_v3
    assert '1. Every input value must appear exactly once as "raw" in "mapping".' in P3_v3
    assert "2. Pick the most frequent clean spelling as the canonical value. Do not invent new categories." in P3_v3
    assert '3. Merge only values that clearly mean the same thing. If unsure, put the value in' in P3_v3
    assert '   "needs_review" and map it to itself.' in P3_v3
    assert "4. Never translate, drop or reinterpret a value." in P3_v3
    rendered = render_p3_v3("payment_method", {"UPI": 100, "upi": 10})
    assert 'column "payment_method"' in rendered
    assert '"UPI": 100' in rendered

    # Schema validation (list-based CategoryMapping used by prompt_lab)
    from cleaner.prompts import CategoryMapping, MappingItem
    sample = CategoryMapping(
        canonical=["UPI", "Cash on Delivery"],
        mapping=[
            MappingItem(raw="UPI", canonical="UPI"),
            MappingItem(raw="upi", canonical="UPI"),
            MappingItem(raw="COD", canonical="Cash on Delivery"),
        ],
        needs_review=["UnknownPayment"],
    )
    assert sample.canonical == ["UPI", "Cash on Delivery"]
    assert sample.mapping[1].raw == "upi"
    assert sample.mapping[1].canonical == "UPI"
    assert sample.needs_review == ["UnknownPayment"]

    # Legacy P3CategoryMapping (dict-based) still works
    legacy = P3CategoryMapping.model_validate({
        "canonical": ["UPI", "Cash on Delivery"],
        "mapping": {"UPI": "UPI", "upi": "UPI", "COD": "Cash on Delivery"},
        "needs_review": ["UnknownPayment"],
    })
    assert legacy.mapping["upi"] == "UPI"
    assert legacy.needs_review == ["UnknownPayment"]


def test_p4_schema_and_render() -> None:
    """Verify P4 schema validation and prompt rendering."""
    data = {
        "strategy": "leave_null",
        "rationale": "Field is optional for undelivered orders.",
        "risk": "Low risk of distribution distortion.",
    }
    obj = P4MissingAdvice.model_validate(data)
    assert obj.strategy == "leave_null"

    rendered = render_p4_v1("rating", "integer", 33.2, {"5": 126, "4": 111})
    assert 'column "rating"' in rendered
    assert "33.20%" in rendered


def test_p5_schema_and_render() -> None:
    """Verify P5 schema validation and prompt rendering."""
    data = {"summary": "The dataset has 545 rows with 299 missing values."}
    obj = P5Summary.model_validate(data)
    assert "545 rows" in obj.summary

    rendered = render_p5_v1({"total_rows": 545, "issues": 205})
    assert '"total_rows": 545' in rendered
