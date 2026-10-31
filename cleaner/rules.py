"""Rule-based data cleaning engine and transformation logger."""

import re
from typing import Any
from dateutil import parser as date_parser
import pandas as pd
import phonenumbers

ACTION_ORDER: list[str] = [
    "strip_trim",
    "title_case",
    "map_values",
    "parse_date",
    "normalize_phone",
    "parse_money",
]

SUPPORTED_ACTIONS: set[str] = set(ACTION_ORDER)


def _apply_strip_trim(val: object) -> object:
    """Trim leading and trailing whitespace from strings."""
    if pd.isna(val) or not isinstance(val, str):
        return val
    return val.strip()


def _apply_title_case(val: object) -> object:
    """Convert string to title case while preserving nulls."""
    if pd.isna(val) or not isinstance(val, str):
        return val
    return val.strip().title()


def _apply_map_values(val: object, mapping: dict[str, str]) -> object:
    """Map raw categorical values to canonical forms."""
    if pd.isna(val):
        return val
    s = str(val)
    if s in mapping:
        return mapping[s]
    stripped = s.strip()
    if stripped in mapping:
        return mapping[stripped]
    return val


def _apply_parse_date(val: object, day_first: bool = True) -> object:
    """Parse dates in various standard formats into ISO YYYY-MM-DD."""
    if pd.isna(val) or not isinstance(val, str) or not val.strip():
        return val
    s = val.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    try:
        if re.match(r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}$", s):
            dt = date_parser.parse(s, yearfirst=True, dayfirst=False)
        else:
            dt = date_parser.parse(s, dayfirst=day_first)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return val


def _apply_normalize_phone(val: object, default_country: str = "IN") -> object:
    """Normalize phone numbers to international E.164 format."""
    if pd.isna(val) or not isinstance(val, str) or not val.strip():
        return val
    s = val.strip()
    try:
        parsed = phonenumbers.parse(s, default_country)
        if phonenumbers.is_possible_number(parsed) or phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except Exception:
        pass
    return val


def _apply_parse_money(val: object) -> object:
    """Normalize currency strings to standard two-decimal number strings."""
    if pd.isna(val) or not isinstance(val, str) or not val.strip():
        return val
    s = val.strip()
    # Strip currency keywords and symbols
    cleaned = re.sub(r"(?i)rs\.?|inr|usd|eur|gbp|[₹$€£]", "", s)
    cleaned = cleaned.replace(",", "").strip()
    try:
        amount = float(cleaned)
        return f"{amount:.2f}"
    except ValueError:
        return val


def _execute_single_action(
    val: object, action: str, params: dict[str, Any]
) -> object:
    """Dispatch cell value transformation based on action name."""
    if action == "strip_trim":
        return _apply_strip_trim(val)
    elif action == "title_case":
        return _apply_title_case(val)
    elif action == "map_values":
        mapping = params.get("mapping", params)
        return _apply_map_values(val, mapping)
    elif action == "parse_date":
        day_first = params.get("day_first", True)
        return _apply_parse_date(val, day_first=day_first)
    elif action == "normalize_phone":
        country = params.get("default_country", "IN")
        return _apply_normalize_phone(val, default_country=country)
    elif action == "parse_money":
        return _apply_parse_money(val)
    raise ValueError(f"Unsupported action: '{action}'")


def _update_flags_column(df: pd.DataFrame) -> None:
    """Populate _flags column marking missing values in the DataFrame."""
    feature_cols = [c for c in df.columns if c != "_flags"]
    flags: list[str] = []
    for idx in range(len(df)):
        missing_in_row = [
            c for c in feature_cols if pd.isna(df.iat[idx, df.columns.get_loc(c)])
        ]
        flags.append(f"missing: {', '.join(missing_in_row)}" if missing_in_row else "")
    df["_flags"] = flags


def apply_rules(
    df: pd.DataFrame, rules: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply approved data cleaning rules to df in fixed order and return change log.

    Args:
        df: Input DataFrame (not modified).
        rules: List of rule dicts with format {column, action, params}.

    Returns:
        Tuple of (cleaned DataFrame, change log DataFrame).
    """
    # 1. Validate rules
    for r in rules:
        action = r.get("action")
        if action not in SUPPORTED_ACTIONS:
            raise ValueError(f"Unknown action: '{action}'")

    # 2. Never mutate input
    clean_df = df.copy()
    change_records: list[dict[str, object]] = []

    # 3. Exact duplicates: keep first, drop rest, log what was dropped
    dup_mask = clean_df.duplicated(keep="first")
    if dup_mask.any():
        for row_idx, is_dup in enumerate(dup_mask):
            if is_dup:
                change_records.append({
                    "row": row_idx + 1,
                    "column": "*row*",
                    "old_value": "duplicate",
                    "new_value": "<dropped>",
                    "rule": "drop_exact_duplicate",
                })
        clean_df = clean_df[~dup_mask].reset_index(drop=True)

    # 4. Sort rules by documented fixed execution order
    sorted_rules = sorted(
        rules, key=lambda rule: ACTION_ORDER.index(rule["action"])
    )

    # 5. Apply column transformation rules
    for rule in sorted_rules:
        col = rule.get("column", "")
        action = rule.get("action", "")
        params = rule.get("params", {})

        if col not in clean_df.columns:
            continue

        col_idx = clean_df.columns.get_loc(col)
        for row_idx in range(len(clean_df)):
            row_num = row_idx + 1
            old_val = clean_df.iat[row_idx, col_idx]
            new_val = _execute_single_action(old_val, action, params)

            # Record changes if cell value changed (handling NA equality)
            val_changed = False
            if pd.isna(old_val) and pd.isna(new_val):
                val_changed = False
            elif pd.isna(old_val) != pd.isna(new_val):
                val_changed = True
            elif old_val != new_val:
                val_changed = True

            if val_changed:
                clean_df.iat[row_idx, col_idx] = new_val
                change_records.append({
                    "row": row_num,
                    "column": col,
                    "old_value": old_val,
                    "new_value": new_val,
                    "rule": action,
                })

    # 6. Missing values: flag in _flags column
    _update_flags_column(clean_df)

    change_log_df = pd.DataFrame(
        change_records,
        columns=["row", "column", "old_value", "new_value", "rule"],
    )
    return clean_df, change_log_df

