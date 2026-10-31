# cleaner/report.py
"""HTML validation report generation.

The module aggregates the results of the loader, detectors, rule engine and
Pandera validation and renders a self-contained HTML file with Jinja2.
"""

from __future__ import annotations

import datetime as _dt
import pathlib as _pl
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Local imports – use the real function names from the project
from .loader import load_csv
from .detectors import run_all as run_detectors
from .schema import build_schema, validate, summarize
from .rules import apply_rules


def _null_log_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Create a null‑log DataFrame for a cleaned DataFrame.

    The null log has columns [row, column, original_value, kind] where ``kind``
    is ``blank`` for empty/whitespace cells and ``placeholder`` for any other
    placeholder value.  It mirrors the behaviour of ``loader.load_csv`` but works
    on an existing DataFrame.
    """
    rows, cols = df.shape
    records = []
    for r in range(rows):
        for c in df.columns:
            val = df.iat[r, df.columns.get_loc(c)]
            if pd.isna(val):
                continue
            if isinstance(val, str) and (val.strip() == "" or val.lower() == "nan"):
                kind = "blank"
            else:
                continue
            records.append(
                {
                    "row": r + 1,  # 1‑based
                    "column": c,
                    "original_value": val,
                    "kind": kind,
                }
            )
    return pd.DataFrame(records, columns=["row", "column", "original_value", "kind"])


def _quality_score(flags: pd.DataFrame, total_cells: int) -> float:
    """Return a quality score between 0 and 1.

    ``flags`` contains detector output rows.  Cells whose column is ``"*row*``
    are row‑level issues and are ignored for the score.
    """
    if total_cells == 0:
        return 1.0
    flagged = flags[flags["column"] != "*row*"].shape[0]
    return 1.0 - flagged / total_cells


def build_report_context(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    flags_before: pd.DataFrame,
    flags_after: pd.DataFrame,
    change_log: pd.DataFrame,
    failures_before: pd.DataFrame,
    failures_after: pd.DataFrame,
    *,
    gemini_usage: Optional[Dict[str, Any]] = None,
    ai_summary: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect everything the HTML template needs.

    The function is deliberately pure – it does not read/write files.
    """
    # ------------------------------------------------------------------ meta
    meta = meta or {}
    meta.setdefault("processed_at", _dt.datetime.utcnow().isoformat() + "Z")
    meta.setdefault("source_file", "unknown.csv")
    meta.setdefault("output_file", "outputs/orders_clean.csv")

    # ---------------------------------------------------------------- overview
    rows, cols = raw_df.shape
    total_cells = rows * cols

    # ---------------------------------------------------------------- quality
    score_before = _quality_score(flags_before, total_cells)
    score_after = _quality_score(flags_after, total_cells)

    # ---------------------------------------------------------------- flags for null‑log (needed by detectors)
    # The detectors expect a null_log for missing‑value detection.  We build a
    # minimal version from the cleaned data (so that flags_after can be recomputed
    # after rule application if desired).
    null_log_before = _null_log_from_df(raw_df)
    null_log_after = _null_log_from_df(clean_df)

    # ---------------------------------------------------------------- change‑log helpers
    # Show up‑to‑3 examples per column before/after editing.
    examples_before: Dict[str, List[str]] = {}
    examples_after: Dict[str, List[str]] = {}
    for col in change_log["column"].unique():
        vals_before = (
            change_log.loc[change_log["column"] == col, "old_value"]
            .dropna()
            .astype(str)
            .head(3)
            .tolist()
        )
        vals_after = (
            change_log.loc[change_log["column"] == col, "new_value"]
            .dropna()
            .astype(str)
            .head(3)
            .tolist()
        )
        examples_before[col] = vals_before
        examples_after[col] = vals_after

    # ---------------------------------------------------------------- null‑log summary
    # Pre‑compute blank/placeholder counts per column to avoid Jinja2
    # parsing issues with & in boolean indexing.
    null_log_after_blank: Dict[str, int] = {}
    null_log_after_placeholder: Dict[str, int] = {}
    for col in null_log_after["column"].unique():
        n_blank = int(
            ((null_log_after["column"] == col) & (null_log_after["kind"] == "blank")).sum()
        )
        n_ph = int(
            ((null_log_after["column"] == col) & (null_log_after["kind"] == "placeholder")).sum()
        )
        null_log_after_blank[col] = n_blank
        null_log_after_placeholder[col] = n_ph

    # ---------------------------------------------------------------- summary dict
    context: Dict[str, Any] = {
        "meta": meta,
        "ai_summary": ai_summary,
        "rows": rows,
        "cols": cols,
        "source_file": meta["source_file"],
        "processed_at": meta["processed_at"],
        "quality_before": round(score_before, 4),
        "quality_after": round(score_after, 4),
        "flags_before": flags_before,
        "flags_after": flags_after,
        "null_log_before": null_log_before,
        "null_log_after": null_log_after,
        "detector_counts_by_type_before": Counter(flags_before["issue_type"]),
        "detector_counts_by_type_after": Counter(flags_after["issue_type"]),
        "detector_counts_by_column_before": Counter(flags_before["column"]),
        "detector_counts_by_column_after": Counter(flags_after["column"]),
        "change_log": change_log,
        "examples_before": examples_before,
        "examples_after": examples_after,
        "failures_before": failures_before,
        "failures_after": failures_after,
        "validation_summary_before": summarize(failures_before, rows),
        "validation_summary_after": summarize(failures_after, rows),
        "gemini_usage": gemini_usage,
        "rule_counts": pd.Series(change_log["rule"]).value_counts().to_dict(),
        "null_log_after_blank": null_log_after_blank,
        "null_log_after_placeholder": null_log_after_placeholder,
    }
    return context


def render_report(context: Dict[str, Any], out_path: str | _pl.Path) -> str:
    """Render the Jinja2 template and write the HTML file.

    Parameters
    ----------
    context:
        The dictionary returned by :func:`build_report_context`.
    out_path:
        Destination path for the HTML file.  Parent directories are created if
        necessary.

    Returns
    -------
    str
        The absolute path of the written file.
    """
    tmpl_dir = _pl.Path(__file__).parent.parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(tmpl_dir)),
        autoescape=select_autoescape(["html", "xml", "j2"]),
        enable_async=False,
    )
    tmpl = env.get_template("report.html.j2")
    rendered = tmpl.render(context)

    out_path = _pl.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    return str(out_path.resolve())
