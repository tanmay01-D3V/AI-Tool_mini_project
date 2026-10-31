# cleaner/schema.py
"""Pandera schema builder and validator for the orders dataset.

The public API consists of three functions:
* ``build_schema`` – construct a ``pa.DataFrameSchema`` from a column‑type
  specification.
* ``validate`` – lazily validate a DataFrame against the schema and return a
  DataFrame of failures with columns ``[row, column, value, check, reason]``.
* ``summarize`` – produce a concise summary dictionary from the failure DataFrame.

All functions are pure; there is no module‑level mutable state.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import pandera as pa
import pandera.errors as pa_errors

# Helper type alias
ColumnSpec = Dict[str, Any]


def _regex_check(pattern: str, name: str) -> pa.Check:
    """Return a Pandera ``Check`` that validates a string against *pattern*.

    ``ignore_na`` is left to the caller – most checks set it based on column
    nullability.
    """
    regex = re.compile(pattern)
    return pa.Check(lambda s: s.astype(str).str.match(regex), name=name)


def _numeric_range_check(min_val: Optional[float] = None, max_val: Optional[float] = None, name: str = "") -> pa.Check:
    """Return a ``Check`` that ensures numeric values lie within ``[min_val, max_val]``.
    ``None`` means the bound is not applied.
    """
    def _check(series: pd.Series) -> pd.Series:
        num = pd.to_numeric(series, errors="coerce")
        ok = pd.Series(True, index=series.index)
        if min_val is not None:
            ok &= num >= min_val
        if max_val is not None:
            ok &= num <= max_val
        return ok
    return pa.Check(_check, name=name or "numeric_range")


def _date_check(today: date) -> pa.Check:
    """Check that a string is ``YYYY‑MM‑DD`` and not later than *today*.
    """
    def _check(series: pd.Series) -> pd.Series:
        parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce").dt.date
        ok = parsed <= today
        return ok
    return pa.Check(_check, name="date_valid")


def build_schema(
    column_types: Dict[str, ColumnSpec],
    *,
    cross_rules: bool = True,
    allowed_values: Optional[Dict[str, Iterable[str]]] = None,
    today: Optional[date] = None,
) -> pa.DataFrameSchema:
    """Create a Pandera ``DataFrameSchema`` from a column‑type mapping.

    Parameters
    ----------
    column_types:
        Mapping ``column_name → spec`` where *spec* contains keys ``semantic_type``
        (e.g. "id", "email", …), ``nullable`` (bool) and optional ``unique``,
        ``min`` and ``max``.
    cross_rules:
        If ``True`` adds the two dataframe‑level checks described in the spec.
    allowed_values:
        Optional mapping ``column → iterable_of_allowed_strings`` for the
        categorical types.
    today:
        Date used for future‑date validation; defaults to ``date.today()``.
    """
    today = today or date.today()
    columns: Dict[str, pa.Column] = {}

    for col, spec in column_types.items():
        stype = spec["semantic_type"]
        nullable: bool = spec.get("nullable", False)
        unique: bool = spec.get("unique", False)
        col_checks: List[pa.Check] = []
        dtype = str

        if stype == "id":
            if unique:
                col_checks.append(pa.Check(lambda s: s.is_unique, name="unique_id"))
        elif stype == "person_name":
            col_checks.append(_regex_check(r"^\S.*\S$", "no_edge_spaces"))
            col_checks.append(
                pa.Check(lambda s: ~s.str.fullmatch(r"[A-Z]+") & ~s.str.fullmatch(r"[a-z]+"), name="mixed_case")
            )
        elif stype == "email":
            col_checks.append(_regex_check(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", "email_format"))
            col_checks.append(pa.Check(lambda s: s == s.str.lower(), name="email_lowercase"))
        elif stype == "phone":
            col_checks.append(_regex_check(r"^\+\d{1,3} \d{5} \d{5}$", "phone_format"))
        elif stype == "date":
            col_checks.append(_regex_check(r"^\d{4}-\d{2}-\d{2}$", "date_pattern"))
            col_checks.append(_date_check(today))
        elif stype == "money":
            col_checks.append(_regex_check(r"^\d+(?:\.\d{1,2})?$", "money_pattern"))
        elif stype == "integer":
            col_checks.append(_regex_check(r"^-?\d+$", "integer_pattern"))
            if "min" in spec or "max" in spec:
                col_checks.append(_numeric_range_check(spec.get("min"), spec.get("max"), name="int_range"))
        elif stype in {"category", "country", "city"}:
            col_checks.append(_regex_check(r"^\S.*\S$", "no_edge_spaces"))
            if allowed_values and col in allowed_values:
                allowed = set(allowed_values[col])
                col_checks.append(pa.Check(lambda s: s.isin(allowed), name="allowed_values"))
        elif stype == "free_text":
            pass
        else:
            raise ValueError(f"Unsupported semantic_type: {stype}")

        column = pa.Column(dtype, nullable=nullable, unique=unique, checks=col_checks, name=col)
        columns[col] = column

    df_checks: List[pa.Check] = []
    if cross_rules:
        def _amount_check(df: pd.DataFrame) -> pd.Series:
            qty = pd.to_numeric(df["quantity"], errors="coerce")
            unit = pd.to_numeric(df["unit_price"], errors="coerce")
            total = pd.to_numeric(df["total_amount"], errors="coerce")
            expected = qty * unit
            diff = (total - expected).abs()
            ok = diff <= 0.01
            mask = qty.isna() | unit.isna() | total.isna()
            return ok | mask
        df_checks.append(pa.Check(_amount_check, name="amount_consistency"))

        def _date_order_check(df: pd.DataFrame) -> pd.Series:
            signup = pd.to_datetime(df["signup_date"], errors="coerce")
            order = pd.to_datetime(df["order_date"], errors="coerce")
            ok = signup <= order
            mask = signup.isna() | order.isna()
            return ok | mask
        df_checks.append(pa.Check(_date_order_check, name="date_order_consistency"))

    schema = pa.DataFrameSchema(columns=columns, checks=df_checks, strict=False)
    return schema


def validate(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Validate *df* against *schema* lazily.

    Returns a DataFrame with columns ``[row, column, value, check, reason]``. ``row``
    is 1‑based to match the project's conventions.
    """
    try:
        schema.validate(df, lazy=True)
        return pd.DataFrame(columns=["row", "column", "value", "check", "reason"])
    except pa_errors.SchemaErrors as err:
        fc = err.failure_cases.copy()
        fc["row"] = fc["index"] + 1
        fc["column"] = fc["column"].fillna("*")
        fc["value"] = fc["failure_case"].astype(str)
        fc["check"] = fc["check"].astype(str)
        # In Pandera 0.35, failure_case contains the error description
        fc["reason"] = fc["failure_case"].astype(str)
        result = fc[["row", "column", "value", "check", "reason"]].reset_index(drop=True)
        return result


def summarize(failures: pd.DataFrame, n_rows: int) -> Dict[str, Any]:
    """Return a summary dict for *failures*.

    ``n_rows`` is the total number of rows in the validated DataFrame.
    """
    total = len(failures)
    rows = failures["row"].nunique() if total else 0
    by_col = failures.groupby("column").size().to_dict() if total else {}
    by_chk = failures.groupby("check").size().to_dict() if total else {}
    return {
        "total_failures": total,
        "rows_with_failures": rows,
        "by_column": by_col,
        "by_check": by_chk,
        "passed": total == 0,
        "n_rows": n_rows,
    }

# End of file

