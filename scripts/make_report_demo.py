# scripts/make_report_demo.py
"""Demo that builds a full validation report for ``orders_dirty.csv``."""

from __future__ import annotations

import pathlib as _pl

import pandas as pd

from cleaner.loader import load_csv
from cleaner.detectors import run_all as run_detectors
from cleaner.rules import apply_rules
from cleaner.schema import build_schema, validate
from cleaner.report import build_report_context, render_report
from cleaner.orders_config import ORDERS_COLUMN_TYPES

# ---------------------------------------------------------------------------


def main() -> None:
    dirty_path = _pl.Path("data/orders_dirty.csv")
    raw_df, _null_log = load_csv(dirty_path)

    # 1️⃣ Run detectors on the raw data
    flags_before = run_detectors(raw_df, _null_log)

    # 2️⃣ Pandera validation before cleaning
    schema = build_schema(ORDERS_COLUMN_TYPES, cross_rules=True)
    failures_before = validate(raw_df, schema)

    # 3️⃣ Apply a hand‑written rule list
    rule_list = [
        {"column": "order_date", "action": "parse_date", "params": {}},
        {"column": "signup_date", "action": "parse_date", "params": {}},
        {"column": "customer_name", "action": "title_case", "params": {}},
        {"column": "phone", "action": "normalize_phone", "params": {}},
        {"column": "unit_price", "action": "parse_money", "params": {}},
        {"column": "total_amount", "action": "parse_money", "params": {}},
        {
            "column": "country",
            "action": "map_values",
            "params": {"mapping": {"USA": "United States", "UK": "United Kingdom"}},
        },
        {
            "column": "payment_method",
            "action": "map_values",
            "params": {"mapping": {"cc": "Credit Card", "pp": "PayPal"}},
        },
    ]
    clean_df, change_log = apply_rules(raw_df, rule_list)

    # 4️⃣ Run detectors on the cleaned data
    flags_after = run_detectors(clean_df, _null_log)

    # 5️⃣ Validate after cleaning
    failures_after = validate(clean_df, schema)

    # 6️⃣ Build report context
    context = build_report_context(
        raw_df,
        clean_df,
        flags_before,
        flags_after,
        change_log,
        failures_before,
        failures_after,
        meta={"source_file": str(dirty_path), "output_file": "outputs/orders_clean.csv"},
    )

    # 7️⃣ Render HTML
    out_html = render_report(context, "outputs/report.html")
    print(f"Report written to {out_html}")


if __name__ == "__main__":
    main()