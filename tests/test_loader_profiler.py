"""Tests for CSV loader and profiler."""

import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure project root is in sys.path when running pytest directly
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from cleaner.loader import load_csv, PLACEHOLDERS
from cleaner.profiler import profile


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    """Create a 6-row sample CSV file containing various edge cases."""
    csv_file = tmp_path / "sample.csv"
    content = (
        "name,city\n"
        "Alice,Mumbai\n"
        ",Delhi\n"
        "N/A,Bangalore\n"
        "null,Pune\n"
        "  Bob,Chennai\n"
        "cHaRlIe,Kolkata\n"
    )
    csv_file.write_text(content, encoding="utf-8")
    return csv_file


def test_load_csv_detects_placeholders(sample_csv_path: Path) -> None:
    """Verify placeholder null detection, row logging, and value preservation."""
    df, null_log = load_csv(sample_csv_path)

    # DataFrame checks
    assert len(df) == 6
    assert pd.isna(df.loc[1, "name"])
    assert pd.isna(df.loc[2, "name"])
    assert pd.isna(df.loc[3, "name"])

    # Ensure spaced value and mixed-case value remain unchanged
    assert df.loc[4, "name"] == "  Bob"
    assert df.loc[5, "name"] == "cHaRlIe"

    # Null log checks: right row numbers (1-based data rows) and kinds
    assert len(null_log) == 3
    expected_rows = [
        {"row": 2, "column": "name", "original_value": "", "kind": "blank"},
        {"row": 3, "column": "name", "original_value": "N/A", "kind": "placeholder"},
        {"row": 4, "column": "name", "original_value": "null", "kind": "placeholder"},
    ]
    assert null_log.to_dict(orient="records") == expected_rows


def test_profile_returns_correct_null_metrics(sample_csv_path: Path) -> None:
    """Verify profile computes accurate null_count and null_pct."""
    df, _ = load_csv(sample_csv_path)
    prof = profile(df)

    expected_columns = [
        "column",
        "null_count",
        "null_pct",
        "unique_count",
        "top_values",
        "samples",
    ]
    assert list(prof.columns) == expected_columns

    name_row = prof[prof["column"] == "name"].iloc[0]
    assert name_row["null_count"] == 3
    assert name_row["null_pct"] == 50.0
    assert name_row["unique_count"] == 3

    city_row = prof[prof["column"] == "city"].iloc[0]
    assert city_row["null_count"] == 0
    assert city_row["null_pct"] == 0.0
    assert city_row["unique_count"] == 6


def test_orders_dirty_csv_shape_and_nulls() -> None:
    """Verify loading data/orders_dirty.csv matches expected shape and null count."""
    csv_path = Path("data/orders_dirty.csv")
    df, null_log = load_csv(csv_path)

    assert df.shape == (545, 16)
    assert df.isna().sum().sum() == 299
    assert len(null_log) == 299


def test_all_placeholders_recognized(tmp_path: Path) -> None:
    """Verify every placeholder in the PLACEHOLDERS list is recognized."""
    rows = "\n".join(f"{p},dummy" for p in PLACEHOLDERS)
    csv_file = tmp_path / "placeholders.csv"
    csv_file.write_text(f"val,extra\n{rows}\n", encoding="utf-8")

    df, null_log = load_csv(csv_file)
    assert df["val"].isna().all()
    assert len(null_log) == len(PLACEHOLDERS)


def test_profile_formatting_and_empty(tmp_path: Path) -> None:
    """Verify top_values, samples format, and behavior on empty dataframe."""
    csv_file = tmp_path / "repeated.csv"
    content = "col\nA\nA\nB\nC\nD\nE\nF\n"
    csv_file.write_text(content, encoding="utf-8")

    df, _ = load_csv(csv_file)
    prof = profile(df)
    row = prof.iloc[0]

    # top_values: at most 5 items formatted as 'val (count)'
    assert "A (2)" in row["top_values"]
    assert len(row["top_values"].split(" | ")) == 5

    # samples: 5 values joined with ' | '
    samples_split = row["samples"].split(" | ")
    assert len(samples_split) == 5

    # Empty df
    empty_df = pd.DataFrame(columns=["a", "b"])
    empty_prof = profile(empty_df)
    assert len(empty_prof) == 2
    assert empty_prof.loc[0, "null_count"] == 0
    assert empty_prof.loc[0, "null_pct"] == 0.0
    assert empty_prof.loc[0, "top_values"] == ""
    assert empty_prof.loc[0, "samples"] == ""
