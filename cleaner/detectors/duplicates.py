"""Duplicate detectors for exact matches and key conflicts / near matches."""

import pandas as pd
from rapidfuzz import fuzz


def _normalize_cell(val: object) -> str:
    """Normalize cell for comparison: NA to empty string, strip, lowercase."""
    if pd.isna(val):
        return ""
    return str(val).strip().lower()


def detect_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows identical in all columns to an earlier row.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns: row, column, value, issue_type, reason.
    """
    issues: list[dict[str, object]] = []
    seen: dict[tuple, int] = {}

    for idx in range(len(df)):
        row_num = idx + 1
        row_tuple = tuple(df.iloc[idx].tolist())
        if row_tuple in seen:
            earlier_row = seen[row_tuple]
            issues.append({
                "row": row_num,
                "column": "*row*",
                "value": "",
                "issue_type": "exact_duplicate",
                "reason": f"identical to row {earlier_row}",
            })
        else:
            seen[row_tuple] = row_num

    return pd.DataFrame(
        issues,
        columns=["row", "column", "value", "issue_type", "reason"],
    )


def _compute_row_similarity(df: pd.DataFrame, idx1: int, idx2: int) -> float:
    """Compute mean column similarity between two rows using rapidfuzz."""
    if len(df.columns) == 0:
        return 0.0
    ratios = [
        fuzz.ratio(
            _normalize_cell(df.iat[idx1, c]),
            _normalize_cell(df.iat[idx2, c]),
        )
        for c in range(len(df.columns))
    ]
    return sum(ratios) / len(ratios)


def detect_key_duplicates(
    df: pd.DataFrame,
    key: str = "order_id",
    threshold: float = 90,
    exact_dup_rows: set[int] | None = None,
) -> pd.DataFrame:
    """Detect near-duplicates and key conflicts among non-exact duplicate rows.

    Args:
        df: Input DataFrame.
        key: Column name used as identifier key.
        threshold: Minimum mean similarity percentage to classify as near-duplicate.
        exact_dup_rows: Optional set of 1-based row numbers already flagged as exact dups.

    Returns:
        DataFrame with columns: row, column, value, issue_type, reason.
    """
    if key not in df.columns:
        return pd.DataFrame(
            columns=["row", "column", "value", "issue_type", "reason"]
        )

    if exact_dup_rows is None:
        exact_dups = detect_exact_duplicates(df)
        exact_dup_rows = set(exact_dups["row"])

    groups: dict[str, list[int]] = {}
    key_col_idx = df.columns.get_loc(key)

    for idx in range(len(df)):
        row_num = idx + 1
        if row_num in exact_dup_rows:
            continue
        key_val = df.iat[idx, key_col_idx]
        if pd.isna(key_val):
            continue
        groups.setdefault(str(key_val), []).append(row_num)

    issues: list[dict[str, object]] = []

    for _, row_nums in groups.items():
        if len(row_nums) <= 1:
            continue

        first_row_num = row_nums[0]
        first_idx = first_row_num - 1
        later_sims: dict[int, float] = {}

        for later_row_num in row_nums[1:]:
            later_idx = later_row_num - 1
            sim = _compute_row_similarity(df, first_idx, later_idx)
            later_sims[later_row_num] = sim

        is_near_dup = all(sim >= threshold for sim in later_sims.values())

        if is_near_dup:
            for r in row_nums:
                other = row_nums[1] if r == first_row_num else first_row_num
                sim_val = later_sims[row_nums[1]] if r == first_row_num else later_sims[r]
                issues.append({
                    "row": r,
                    "column": "*row*",
                    "value": "",
                    "issue_type": "near_duplicate",
                    "reason": f"near-duplicate of row {other}, similarity {round(sim_val, 1)}",
                })
        else:
            for r in row_nums:
                issues.append({
                    "row": r,
                    "column": key,
                    "value": str(df.iat[r - 1, key_col_idx]),
                    "issue_type": "duplicate_key_conflict",
                    "reason": "key reused with different content",
                })

    issues.sort(key=lambda x: x["row"])
    return pd.DataFrame(
        issues,
        columns=["row", "column", "value", "issue_type", "reason"],
    )
