# scripts/prompt_lab.py
"""Prompt‑lab driver for P3 canonicalisation experiments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

from cleaner.loader import load_csv
from cleaner.llm import call_gemini

from cleaner.prompts import (
    render_p1_v1, render_p2_v1, render_p3_v3, render_p4_v1, render_p5_v1, CategoryMapping, MappingItem
)

# ---------------------------------------------------------------------------




def render_p3(version: str, column: str, value_counts: dict) -> str:
    """Render a P3 prompt for *column* using the template of *version*.

    Parameters
    ----------
    version:
        One of ``"v3"``.
    column:
        Column name to be canonicalized.
    value_counts:
        Mapping of raw value → occurrence count.

    Returns
    -------
    str
        Fully-formatted prompt ready for ``call_gemini``.
    """
    if version not in P3_VERSIONS:
        raise ValueError(f"Unsupported P3 version '{version}'. Choose from {list(P3_VERSIONS)}")
    tmpl = P3_VERSIONS[version]
    # Use str.replace() instead of str.format() because the P3 v3 template
    # contains literal curly braces in the example ({{"UPI": ...}}); .format()
    # would collapse {{ to {, while .replace() only matches exact {column}/{values_json}.
    prompt = tmpl.replace("{column}", column)
    prompt = tmpl.replace("{values_json}", json.dumps(value_counts))
    return prompt
def call_gemini_no_cache(
    prompt: str,
    schema: type,
    *,
    model: str = "gemini-3.5-flash-lite",
    temperature: float = 0.0,
    client: object | None = None,
) -> any:
    """Force a live Gemini request (no cache) – thin wrapper around ``call_gemini``.

    Uses a longer base_delay (3 s) so that exponential back-off handles both
    429 (rate limit) and transient 503 (service unavailable) errors.
    """
    return call_gemini(
        prompt,
        schema,
        model=model,
        temperature=temperature,
        client=client,
        cache_dir=Path("outputs/.tmp_no_cache_dir_never_exists"),
        log_dir=Path("prompt_log/runs"),
        max_tries=5,
        base_delay=3.0,   # gives delays: 3 s, 6 s, 12 s, 24 s + jitter
        sleep_fn=time.sleep,
    )


def build_truth_maps(
    dirty_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    """Return ``{column → {raw → clean}}``.

    * From ``planted_issues.csv`` we get rows where ``issue_type ==
      "format_inconsistency"`` – those map the dirty value to the correct one.
    * Every distinct value that already appears in the ground‑truth file maps to
      itself.
    """
    # Load planted issues (read‑only, no assumptions about its contents)
    planted_path = Path("data/planted_issues.csv")
    planted_df = pd.read_csv(planted_path, dtype=str)

    truth: dict[str, dict[str, str]] = defaultdict(dict)

    # 1️⃣ format_inconsistency rows
    fmt_rows = planted_df[
        (planted_df["issue_type"] == "format_inconsistency")
    ][["column", "dirty_value", "clean_value"]]
    for _, r in fmt_rows.iterrows():
        truth[r["column"]][r["dirty_value"]] = r["clean_value"]

    # 2️⃣ self‑mapping from ground‑truth values
    for col in dirty_df.columns:
        for val in ground_truth_df[col].dropna().unique():
            truth[col][val] = val

    return truth


def score_run(
    column: str,
    mapping: CategoryMapping,
    truth: dict[str, str],
) -> tuple[dict[str, int], list[dict[str, str]]]:
    """Score a single run for *column*.

    Returns a tuple of:
    - dict with counters: valid, coverage, correct, safe, wrong, invented.
    - list of wrong mapping entries, each as {raw, mapped_to, expected}.
    """
    scores = Counter(valid=1, coverage=0, correct=0, safe=0, wrong=0, invented=0)
    wrong_mappings: list[dict[str, str]] = []

    # 1️⃣ valid JSON – already guaranteed by the Pydantic model
    scores["valid"] = 1

    # 2️⃣ coverage – every raw value in truth must appear exactly once in mapping
    raw_vals = set(truth.keys())
    mapped_raw = {item.raw for item in mapping.mapping}
    scores["coverage"] = int(raw_vals <= mapped_raw)

    # 3️⃣ correctness & safety
    for item in mapping.mapping:
        raw = item.raw
        can = item.canonical
        true = truth.get(raw)
        if true is None:
            # Invented raw value (should not happen)
            scores["invented"] += 1
        elif can == true:
            scores["correct"] += 1
        else:
            # Wrong mapping; if we put it in needs_review and map to itself, count as safe
            if raw in mapping.needs_review and can == raw:
                scores["safe"] += 1
            else:
                scores["wrong"] += 1
                wrong_mappings.append({"raw": raw, "mapped_to": can, "expected": true})

    # 4️⃣ any raw values not present at all → invented
    missing_raw = raw_vals - mapped_raw
    scores["invented"] += len(missing_raw)

    return dict(scores), wrong_mappings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompt‑lab for P3 category canonicaliser."
    )
    parser.add_argument(
        "prompt",
        choices=["p3"],
        help="Which prompt family to run (currently only p3).",
    )
    parser.add_argument(
        "version",
        help="Prompt version identifier (e.g. v1, v2, v3).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of independent Gemini calls per column.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        required=True,
        help="Columns to test (e.g. country city product_category).",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=7,
        help="Seconds to sleep between successive Gemini calls (rate‑limit safety).",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load data
    dirty_df, _ = load_csv(Path("data/orders_dirty.csv"))
    gt_df, _ = load_csv(Path("data/orders_ground_truth.csv"))

    # Build truth maps (raw → clean) for every column we will test
    truth_maps = build_truth_maps(dirty_df, gt_df)

    # Run the lab
    overall = Counter()
    per_column: dict[str, list[dict[str, int]]] = defaultdict(list)
    # Store actual CategoryMapping responses for consistency check (separate from scores)
    per_column_responses: dict[str, list[CategoryMapping]] = defaultdict(list)
    per_column_wrong_mappings: dict[str, list[list[dict]]] = defaultdict(list)

    for col in args.columns:
        # Distinct non‑null raw values + counts from the dirty file
        value_counts = (
            dirty_df[col]
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        )

        # -------------------------------------------------------------------
        # Repeat the Gemini call ``runs`` times
        for run_idx in range(args.runs):
            prompt = render_p3(args.version, col, value_counts)
            # Force a live request (no cache)
            response = call_gemini_no_cache(
                prompt,
                schema=CategoryMapping,
                temperature=0.0,
                model="gemini-3.5-flash-lite",
            )
            # Store response for consistency check
            per_column_responses[col].append(response)
            # Score this run
            run_score, run_wrong_mappings = score_run(col, response, truth_maps[col])
            per_column[col].append(run_score)
            overall.update(run_score)

            # Respect rate limit
            if run_idx < args.runs - 1:
                time.sleep(args.sleep)

    # Consistency check (identical mapping across runs)
    consistency: dict[str, bool] = {}
    for col, responses in per_column_responses.items():
        # Serialize each response's mapping list for equality comparison
        serialized = [
            json.dumps(
                {
                    "canonical": sorted(r.canonical),
                    "mapping": sorted((m.raw, m.canonical) for m in r.mapping),
                },
                sort_keys=True,
            )
            for r in responses
        ]
        consistency[col] = len(set(serialized)) == 1

    # Print table per column
    print()
    print("Column | Valid | Coverage | Correct % | Safe % | Wrong % | Invented | Consistent")
    header = (
        "Column | Valid | Coverage | Correct % | Safe % | Wrong % | Invented | Consistent"
    )
    print(header)
    print("-" * len(header))
    for col, runs in per_column.items():
        # Aggregate scores across runs
        agg = Counter()
        for r in runs:
            agg.update(r)
        n = len(runs)
        correct_pct = agg["correct"] / (agg["correct"] + agg["wrong"] + agg["safe"]) * 100 if (
            agg["correct"] + agg["wrong"] + agg["safe"]
        ) else 0
        safe_pct = agg["safe"] / (agg["correct"] + agg["wrong"] + agg["safe"]) * 100 if (
            agg["correct"] + agg["wrong"] + agg["safe"]
        ) else 0
        wrong_pct = agg["wrong"] / (agg["correct"] + agg["wrong"] + agg["safe"]) * 100 if (
            agg["correct"] + agg["wrong"] + agg["safe"]
        ) else 0
        print(
            f"{col} | {agg['valid']}/{n} | {agg['coverage']}/{n} | "
            f"{correct_pct:.1f}% | {safe_pct:.1f}% | {wrong_pct:.1f}% | "
            f"{agg['invented']} | {'yes' if consistency[col] else 'no'}"
        )

    # Overall summary
    total_runs = sum(len(v) for v in per_column.values())
    overall_correct = overall["correct"]
    overall_safe = overall["safe"]
    overall_wrong = overall["wrong"]
    overall_invented = overall["invented"]
    total_mapped = overall_correct + overall_wrong + overall_safe

    print("\n=== Overall summary ===")
    print(f"Runs executed                : {total_runs}")
    print(f"Valid JSON (should be 100%)   : {overall['valid']}/{total_runs}")
    print(f"Coverage (truth ⊆ mapping)   : {overall['coverage']}/{total_runs}")
    print(f"Correct mappings %           : {overall_correct/total_mapped*100:.1f}%")
    print(f"Safe (needs_review) %        : {overall_safe/total_mapped*100:.1f}%")
    print(f"Wrong mappings %             : {overall_wrong/total_mapped*100:.1f}%")
    print(f"Invented raw values          : {overall_invented}")

    # Print every wrong mapping
    if overall_wrong > 0 or any(per_column_wrong_mappings[col] for col in args.columns):
        print("\n=== Wrong mappings ===")
        for col in args.columns:
            for run_idx, wrong_map in enumerate(per_column_wrong_mappings[col]):
                if wrong_map:
                    print(f"\nColumn: {col} (run {run_idx + 1}):")
                    for entry in wrong_map:
                        print(f"  raw: '{entry['raw']}' | mapped to: '{entry['mapped_to']}' | expected: '{entry['expected']}'")



    # Log summary line to prompt_log/tool_prompts.md
    md_line = (
        f"| Prompt | Version | Date | What changed | Test input | "
        f"Valid JSON | Correct % | Invented values | Consistent | Problem seen | Next fix |\n"
        f"|---|---|---|---|---|---|---|---|---|---|---|\n"
        f"| p3 | {args.version} | {time.strftime('%Y-%m-%d')} |  | "
        f"{', '.join(args.columns)} | {overall['valid']}/{total_runs} | "
        f"{overall_correct/total_mapped*100:.1f}% | {overall_invented} | "
        f"{'yes' if all(consistency.values()) else 'no'} |  |  |\n"
    )
    tool_md = Path("prompt_log/tool_prompts.md")
    tool_md.parent.mkdir(parents=True, exist_ok=True)
    # Write header if file is new, then append the summary line
    if not tool_md.exists():
        tool_md.write_text(
            "| Prompt | Version | Date | What changed | Test input | "
            f"Valid JSON | Correct % | Invented values | Consistent | Problem seen | Next fix |\n"
            f"|---|---|---|---|---|---|---|---|---|---|---|\n", encoding="utf-8"
        )
    with open(tool_md, encoding="utf-8", mode="a") as f:
        f.write(md_line)


if __name__ == "__main__":
    main()