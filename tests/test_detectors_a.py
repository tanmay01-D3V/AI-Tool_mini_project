"""Tests for missing value and duplicate detectors."""

import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure project root is in sys.path when running pytest directly
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from cleaner.loader import load_csv
from cleaner.detectors.missing import detect_missing
from cleaner.detectors.duplicates import (
    detect_exact_duplicates,
    detect_key_duplicates,
)
from cleaner.detectors import run_all


def test_blank_and_placeholder_reported_with_right_reason() -> None:
    """1. blank and placeholder cells are reported with the right reason."""
    df = pd.DataFrame({
        "name": ["Alice", pd.NA, pd.NA],
        "email": ["alice@example.com", "bob@example.com", pd.NA],
    })
    null_log = pd.DataFrame([
        {"row": 2, "column": "name", "original_value": "", "kind": "blank"},
        {"row": 3, "column": "name", "original_value": "null", "kind": "placeholder"},
        {"row": 3, "column": "email", "original_value": "N/A", "kind": "placeholder"},
    ])

    issues = detect_missing(df, null_log)
    assert len(issues) == 3

    r2_name = issues[(issues["row"] == 2) & (issues["column"] == "name")].iloc[0]
    assert r2_name["issue_type"] == "missing_value"
    assert r2_name["reason"] == "blank"

    r3_name = issues[(issues["row"] == 3) & (issues["column"] == "name")].iloc[0]
    assert r3_name["issue_type"] == "missing_value"
    assert r3_name["reason"] == "placeholder: null"

    r3_email = issues[(issues["row"] == 3) & (issues["column"] == "email")].iloc[0]
    assert r3_email["issue_type"] == "missing_value"
    assert r3_email["reason"] == "placeholder: N/A"


def test_blank_rating_pending_vs_delivered() -> None:
    """2. a blank rating on a Pending order is NOT reported; on a Delivered order it is."""
    df = pd.DataFrame({
        "order_id": ["ORD-1", "ORD-2"],
        "status": ["Pending", "Delivered"],
        "rating": [pd.NA, pd.NA],
    })
    null_log = pd.DataFrame([
        {"row": 1, "column": "rating", "original_value": "", "kind": "blank"},
        {"row": 2, "column": "rating", "original_value": "", "kind": "blank"},
    ])

    issues = detect_missing(df, null_log)
    assert len(issues) == 1

    flagged = issues.iloc[0]
    assert flagged["row"] == 2
    assert flagged["column"] == "rating"
    assert flagged["issue_type"] == "missing_value"
    assert flagged["reason"] == "blank"


def test_exact_duplicates_only_later_copy_flagged() -> None:
    """3. exact duplicates: only the later copy is flagged."""
    df = pd.DataFrame({
        "order_id": ["ORD-1", "ORD-2", "ORD-1"],
        "item": ["Book", "Pen", "Book"],
    })

    issues = detect_exact_duplicates(df)
    assert len(issues) == 1

    flagged = issues.iloc[0]
    assert flagged["row"] == 3
    assert flagged["column"] == "*row*"
    assert flagged["issue_type"] == "exact_duplicate"
    assert flagged["reason"] == "identical to row 1"


def test_near_duplicate_both_flagged() -> None:
    """4. near-duplicate: same order_id, name typo and email case -> both flagged."""
    df = pd.DataFrame({
        "order_id": ["ORD-100", "ORD-100"],
        "name": ["Jonathan Doe", "Jonathon Doe"],
        "email": ["jdoe@example.com", "JDOE@EXAMPLE.COM"],
    })

    issues = detect_key_duplicates(df, key="order_id")
    assert len(issues) == 2
    assert (issues["issue_type"] == "near_duplicate").all()
    assert (issues["column"] == "*row*").all()
    assert issues.loc[0, "reason"] == "near-duplicate of row 2, similarity 94.4" or "near-duplicate of row 2" in issues.loc[0, "reason"]
    assert "near-duplicate of row 1" in issues.loc[1, "reason"]


def test_key_conflict_both_flagged() -> None:
    """5. key conflict: same order_id, completely different rows -> both flagged."""
    df = pd.DataFrame({
        "order_id": ["ORD-200", "ORD-200"],
        "name": ["Alice Smith", "Zack Miller"],
        "email": ["asmith@example.com", "zmiller@example.com"],
    })

    issues = detect_key_duplicates(df, key="order_id")
    assert len(issues) == 2
    assert (issues["issue_type"] == "duplicate_key_conflict").all()
    assert (issues["column"] == "order_id").all()
    assert (issues["reason"] == "key reused with different content").all()


def test_dirty_orders_exact_duplicates_count() -> None:
    """6. on data/orders_dirty.csv, run_all flags exactly 25 exact_duplicate rows."""
    csv_path = Path("data/orders_dirty.csv")
    df, null_log = load_csv(csv_path)

    all_issues = run_all(df, null_log)
    exact_dups = all_issues[all_issues["issue_type"] == "exact_duplicate"]
    assert len(exact_dups) == 25
