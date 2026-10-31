"""CSV loader with placeholder null detection and logging."""

from pathlib import Path
import pandas as pd

PLACEHOLDERS: list[str] = [
    "",
    "N/A",
    "NA",
    "null",
    "NULL",
    "nan",
    "none",
    "-",
    "?",
    "Unknown",
]


def _classify_placeholder(val: object, placeholders: set[str]) -> str | None:
    """Classify a cell value as 'blank', 'placeholder', or None."""
    if not isinstance(val, str):
        return None
    stripped = val.strip()
    if stripped.lower() in placeholders:
        return "blank" if stripped == "" else "placeholder"
    return None


def load_csv(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a CSV file as strings and replace placeholder values with pd.NA.

    Args:
        path: Path to the target CSV file.

    Returns:
        A tuple of (cleaned DataFrame, null log DataFrame).
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    placeholder_lookup = {p.strip().lower() for p in PLACEHOLDERS}
    null_records: list[dict[str, object]] = []

    for row_idx in range(len(df)):
        row_num = row_idx + 1
        for col_idx, col_name in enumerate(df.columns):
            val = df.iat[row_idx, col_idx]
            kind = _classify_placeholder(val, placeholder_lookup)
            if kind is not None:
                null_records.append({
                    "row": row_num,
                    "column": col_name,
                    "original_value": val,
                    "kind": kind,
                })
                df.iat[row_idx, col_idx] = pd.NA

    null_log = pd.DataFrame(
        null_records,
        columns=["row", "column", "original_value", "kind"],
    )
    return df, null_log
