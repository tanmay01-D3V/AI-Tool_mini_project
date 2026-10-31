"""Tests for cleaner.rules transformation actions and change logging."""

import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from cleaner.rules import apply_rules


def test_rule_strip_trim() -> None:
    """Verify strip_trim removes stray spaces and logs changes only when modified."""
    df = pd.DataFrame({"city": ["  Mumbai  ", "Delhi", pd.NA]})
    rules = [{"column": "city", "action": "strip_trim", "params": {}}]

    clean_df, change_log = apply_rules(df, rules)

    # Changed Mumbai, Delhi remained unchanged
    assert clean_df.at[0, "city"] == "Mumbai"
    assert clean_df.at[1, "city"] == "Delhi"
    assert pd.isna(clean_df.at[2, "city"])

    assert len(change_log) == 1
    record = change_log.iloc[0]
    assert record["row"] == 1
    assert record["column"] == "city"
    assert record["old_value"] == "  Mumbai  "
    assert record["new_value"] == "Mumbai"
    assert record["rule"] == "strip_trim"


def test_rule_title_case() -> None:
    """Verify title_case standardizes text casing and logs changes."""
    df = pd.DataFrame({"name": ["john doe", "Alice Smith"]})
    rules = [{"column": "name", "action": "title_case", "params": {}}]

    clean_df, change_log = apply_rules(df, rules)

    assert clean_df.at[0, "name"] == "John Doe"
    assert clean_df.at[1, "name"] == "Alice Smith"

    # Alice Smith was already title-cased, so only row 1 is logged
    assert len(change_log) == 1
    assert change_log.iloc[0]["old_value"] == "john doe"
    assert change_log.iloc[0]["new_value"] == "John Doe"


def test_rule_map_values() -> None:
    """Verify map_values replaces variants and preserves already clean values."""
    df = pd.DataFrame({"payment": ["COD", "Cash on Delivery", "UPI"]})
    mapping = {"COD": "Cash on Delivery", "UPI": "UPI"}
    rules = [{"column": "payment", "action": "map_values", "params": {"mapping": mapping}}]

    clean_df, change_log = apply_rules(df, rules)

    assert clean_df.at[0, "payment"] == "Cash on Delivery"
    assert clean_df.at[1, "payment"] == "Cash on Delivery"
    assert clean_df.at[2, "payment"] == "UPI"

    # Only COD was modified
    assert len(change_log) == 1
    assert change_log.iloc[0]["old_value"] == "COD"
    assert change_log.iloc[0]["new_value"] == "Cash on Delivery"


def test_rule_parse_date() -> None:
    """Verify parse_date normalizes multiple date formats to ISO YYYY-MM-DD."""
    df = pd.DataFrame({"order_date": ["15/05/2024", "24 Mar 2023", "2024-05-15", "invalid_date"]})
    rules = [{"column": "order_date", "action": "parse_date", "params": {"day_first": True}}]

    clean_df, change_log = apply_rules(df, rules)

    assert clean_df.at[0, "order_date"] == "2024-05-15"
    assert clean_df.at[1, "order_date"] == "2023-03-24"
    assert clean_df.at[2, "order_date"] == "2024-05-15"  # unchanged
    assert clean_df.at[3, "order_date"] == "invalid_date"  # unchanged invalid

    # Exactly 2 changes logged
    assert len(change_log) == 2
    assert change_log.iloc[0]["old_value"] == "15/05/2024"
    assert change_log.iloc[0]["new_value"] == "2024-05-15"
    assert change_log.iloc[1]["old_value"] == "24 Mar 2023"
    assert change_log.iloc[1]["new_value"] == "2023-03-24"


def test_rule_normalize_phone() -> None:
    """Verify normalize_phone standardizes numbers to E.164 format."""
    df = pd.DataFrame({"phone": ["+91 92813 70661", "+919281370661", "invalid"]})
    rules = [{"column": "phone", "action": "normalize_phone", "params": {"default_country": "IN"}}]

    clean_df, change_log = apply_rules(df, rules)

    assert clean_df.at[0, "phone"] == "+919281370661"
    assert clean_df.at[1, "phone"] == "+919281370661"  # unchanged
    assert clean_df.at[2, "phone"] == "invalid"  # unchanged

    assert len(change_log) == 1
    assert change_log.iloc[0]["old_value"] == "+91 92813 70661"
    assert change_log.iloc[0]["new_value"] == "+919281370661"


def test_rule_parse_money() -> None:
    """Verify parse_money parses currency symbols and words to decimal strings."""
    df = pd.DataFrame({"price": ["₹1,500.00", "Rs. 450", "1500.00"]})
    rules = [{"column": "price", "action": "parse_money", "params": {}}]

    clean_df, change_log = apply_rules(df, rules)

    assert clean_df.at[0, "price"] == "1500.00"
    assert clean_df.at[1, "price"] == "450.00"
    assert clean_df.at[2, "price"] == "1500.00"  # unchanged

    assert len(change_log) == 2
    assert change_log.iloc[0]["old_value"] == "₹1,500.00"
    assert change_log.iloc[0]["new_value"] == "1500.00"
    assert change_log.iloc[1]["old_value"] == "Rs. 450"
    assert change_log.iloc[1]["new_value"] == "450.00"


def test_unknown_action_raises_error() -> None:
    """Verify applying an unsupported action raises a ValueError."""
    df = pd.DataFrame({"val": ["A"]})
    rules = [{"column": "val", "action": "unsupported_action", "params": {}}]

    with pytest.raises(ValueError, match="Unknown action: 'unsupported_action'"):
        apply_rules(df, rules)


def test_exact_duplicates_dropped_and_near_duplicates_kept() -> None:
    """Verify exact duplicates are dropped and logged, while near duplicates remain."""
    df = pd.DataFrame({
        "order_id": ["ORD-1", "ORD-1", "ORD-1"],
        "customer": ["Alice", "Alice", "Alicia"],
    })

    clean_df, change_log = apply_rules(df, [])

    # Row 1 and Row 2 are exact duplicates -> Row 2 is dropped
    # Row 3 is a near duplicate (Alicia) -> NOT auto-dropped
    assert len(clean_df) == 2
    assert list(clean_df["customer"]) == ["Alice", "Alicia"]

    drop_records = change_log[change_log["rule"] == "drop_exact_duplicate"]
    assert len(drop_records) == 1
    assert drop_records.iloc[0]["row"] == 2
    assert drop_records.iloc[0]["column"] == "*row*"


def test_missing_values_flagged_in_flags_column() -> None:
    """Verify missing values are recorded in the _flags column."""
    df = pd.DataFrame({
        "id": ["1", "2"],
        "rating": ["5", pd.NA],
    })

    clean_df, _ = apply_rules(df, [])

    assert "_flags" in clean_df.columns
    assert clean_df.at[0, "_flags"] == ""
    assert clean_df.at[1, "_flags"] == "missing: rating"


def test_input_immutability() -> None:
    """Verify apply_rules does not mutate the input DataFrame."""
    df = pd.DataFrame({"city": ["  Delhi  "]})
    df_orig = df.copy(deep=True)

    rules = [{"column": "city", "action": "strip_trim", "params": {}}]
    clean_df, _ = apply_rules(df, rules)

    assert clean_df.at[0, "city"] == "Delhi"
    # Original remains untouched
    assert df.at[0, "city"] == "  Delhi  "
    pd.testing.assert_frame_equal(df, df_orig)

