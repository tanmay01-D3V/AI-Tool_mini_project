"""Quick evaluation script comparing detector outputs against planted issues."""

import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from cleaner.loader import load_csv
from cleaner.detectors import run_all


def main() -> None:
    """Evaluate detector performance against planted issues dataset."""
    df, null_log = load_csv("data/orders_dirty.csv")
    detected = run_all(df, null_log)
    planted = pd.read_csv("data/planted_issues.csv", dtype=str)

    # Map row numbers to order_id
    row_to_order_id = {
        idx + 1: df.iat[idx, df.columns.get_loc("order_id")]
        for idx in range(len(df))
    }
    detected_with_oid = detected.copy()
    detected_with_oid["order_id"] = detected_with_oid["row"].map(row_to_order_id)

    # Sets for fast matching
    missing_flags = set(
        zip(
            detected[detected["issue_type"] == "missing_value"]["row"],
            detected[detected["issue_type"] == "missing_value"]["column"],
        )
    )

    dup_types = {"exact_duplicate", "near_duplicate", "duplicate_key_conflict"}
    flagged_dup_order_ids = set(
        detected_with_oid[detected_with_oid["issue_type"].isin(dup_types)]["order_id"]
    )

    # Flag counts per issue_type
    flag_counts = detected["issue_type"].value_counts().to_dict()

    print(f"{'Issue Type':<25} {'Planted':<10} {'Detected':<10} {'Recall':<10} {'Total Flags':<12}")
    print("-" * 70)

    for issue_type, group in planted.groupby("issue_type", sort=True):
        planted_count = len(group)
        detected_count = 0

        for _, row in group.iterrows():
            if issue_type == "missing_value":
                data_row = int(row["data_row"])
                col = row["column"]
                if (data_row, col) in missing_flags:
                    detected_count += 1
            elif issue_type in dup_types:
                oid = row["order_id"]
                if oid in flagged_dup_order_ids:
                    detected_count += 1

        recall = (detected_count / planted_count * 100.0) if planted_count > 0 else 0.0
        flags = flag_counts.get(issue_type, 0)
        print(f"{issue_type:<25} {planted_count:<10} {detected_count:<10} {recall:>6.1f}%    {flags:<12}")

    print("-" * 70)
    print(f"Total planted issues: {len(planted)}")
    print(f"Total detected flags: {len(detected)}")


if __name__ == "__main__":
    main()
