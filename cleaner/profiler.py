"""Dataset profiling utilities for tabular data."""

import pandas as pd


def _compute_top_values(non_null: pd.Series, limit: int = 5) -> str:
    """Format the most frequent non-null values with their counts."""
    if non_null.empty:
        return ""
    top_counts = non_null.value_counts().head(limit)
    return " | ".join(f"{val} ({cnt})" for val, cnt in top_counts.items())


def _compute_samples(
    non_null: pd.Series, limit: int = 5, random_state: int = 42
) -> str:
    """Sample random non-null values joined by pipe."""
    if non_null.empty:
        return ""
    sample_size = min(limit, len(non_null))
    sampled = non_null.sample(n=sample_size, random_state=random_state)
    return " | ".join(str(val) for val in sampled)


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary profile statistics for each column in df.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns: column, null_count, null_pct, unique_count,
        top_values, samples.
    """
    total_rows = len(df)
    rows: list[dict[str, object]] = []

    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = round((null_count / total_rows) * 100.0, 2) if total_rows > 0 else 0.0
        non_null = series.dropna()
        unique_count = int(non_null.nunique())

        rows.append({
            "column": str(col),
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": unique_count,
            "top_values": _compute_top_values(non_null),
            "samples": _compute_samples(non_null),
        })

    return pd.DataFrame(
        rows,
        columns=[
            "column",
            "null_count",
            "null_pct",
            "unique_count",
            "top_values",
            "samples",
        ],
    )
