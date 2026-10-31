"""Detector pipeline entry point."""

import pandas as pd

from cleaner.detectors.missing import detect_missing
from cleaner.detectors.duplicates import (
    detect_exact_duplicates,
    detect_key_duplicates,
)


def run_all(df: pd.DataFrame, null_log: pd.DataFrame) -> pd.DataFrame:
    """Run all detectors and return aggregated issues sorted by row then column.

    Args:
        df: Input DataFrame.
        null_log: Null log DataFrame from load_csv.

    Returns:
        DataFrame with columns: row, column, value, issue_type, reason.
    """
    exact_dups = detect_exact_duplicates(df)
    exact_dup_rows = set(exact_dups["row"])
    key_dups = detect_key_duplicates(df, exact_dup_rows=exact_dup_rows)
    missing = detect_missing(df, null_log)

    combined = pd.concat([missing, exact_dups, key_dups], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(
            columns=["row", "column", "value", "issue_type", "reason"]
        )

    return combined.sort_values(by=["row", "column"]).reset_index(drop=True)
