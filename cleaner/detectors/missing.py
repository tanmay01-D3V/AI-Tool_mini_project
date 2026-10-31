"""Missing value detector considering legitimately blank cells."""

from collections.abc import Callable
import pandas as pd


def _default_rating_blank_allowed(df: pd.DataFrame) -> pd.Series:
    """Check if blank rating is allowed based on order status."""
    if "status" not in df.columns:
        return pd.Series(True, index=df.index)
    status_clean = df["status"].fillna("").astype(str).str.strip().str.lower()
    return ~status_clean.isin({"delivered", "deliverd", "returned"})


DEFAULT_LEGIT_BLANK_WHEN: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "rating": _default_rating_blank_allowed,
}


def detect_missing(
    df: pd.DataFrame,
    null_log: pd.DataFrame,
    legit_blank_when: dict[str, Callable[[pd.DataFrame], pd.Series]] | None = None,
) -> pd.DataFrame:
    """Detect missing values from null_log, filtering out allowed blanks.

    Args:
        df: Input DataFrame.
        null_log: Null log DataFrame from load_csv.
        legit_blank_when: Mapping of column names to predicate functions.

    Returns:
        DataFrame with columns: row, column, value, issue_type, reason.
    """
    if null_log.empty:
        return pd.DataFrame(
            columns=["row", "column", "value", "issue_type", "reason"]
        )

    rules = (
        DEFAULT_LEGIT_BLANK_WHEN
        if legit_blank_when is None
        else legit_blank_when
    )

    allowed_masks: dict[str, pd.Series] = {}
    for col, pred in rules.items():
        if col in df.columns:
            allowed_masks[col] = pred(df)

    issues: list[dict[str, object]] = []
    for _, null_row in null_log.iterrows():
        row_num = int(null_row["row"])
        col_name = str(null_row["column"])
        orig_val = null_row["original_value"]
        kind = str(null_row["kind"])

        # Check allowed blank
        if kind == "blank" and col_name in allowed_masks:
            mask = allowed_masks[col_name]
            idx = row_num - 1
            if 0 <= idx < len(mask) and bool(mask.iloc[idx]):
                continue

        reason = "blank" if kind == "blank" else f"placeholder: {orig_val}"
        issues.append({
            "row": row_num,
            "column": col_name,
            "value": orig_val,
            "issue_type": "missing_value",
            "reason": reason,
        })

    return pd.DataFrame(
        issues,
        columns=["row", "column", "value", "issue_type", "reason"],
    )
