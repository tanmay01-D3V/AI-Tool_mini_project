"""In-tool prompts P1 to P5 with versioning and Pydantic schemas."""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Schemas for Structured JSON Outputs
# ---------------------------------------------------------------------------

class P1ColumnProfile(BaseModel):
    """P1 schema: column semantic type, format pattern, and nullability."""

    semantic_type: Literal[
        "id",
        "person_name",
        "email",
        "phone",
        "date",
        "money",
        "integer",
        "category",
        "country",
        "city",
        "free_text",
    ]
    expected_format: str = Field(
        ...,
        description="Expected format pattern or regex (e.g., YYYY-MM-DD or standard email).",
    )
    nullable: bool = Field(
        ..., description="True if column is permitted to have blank/null values."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0."
    )
    reason: str = Field(
        ..., description="One-sentence explanation supporting the semantic type."
    )


class P2Rule(BaseModel):
    """Single proposed cleaning rule."""

    action: str = Field(
        ..., description="Cleaning action, e.g., parse_date, normalize_phone, strip_trim."
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Parameters required to apply the action."
    )
    reason: str = Field(
        ..., description="One-sentence reason describing why this rule is proposed."
    )


class P2FormatRules(BaseModel):
    """P2 schema: list of proposed formatting rules for a column."""

    rules: list[P2Rule] = Field(
        default_factory=list, description="Ordered list of transformation rules."
    )


class P3CategoryMapping(BaseModel):
    """P3 schema (legacy): canonical category values and raw-to-clean mapping.

    Uses a plain ``dict`` for mapping – kept for backward compatibility with
    existing tests.  New code should use ``CategoryMapping`` (list form) which
    is more reliable with Gemini structured output.
    """

    canonical: list[str] = Field(
        default_factory=list,
        description="List of distinct standard canonical categories.",
    )
    mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Complete mapping where every input raw value is a key.",
    )
    needs_review: list[str] = Field(
        default_factory=list,
        description="Raw values that could not be reliably mapped and need human review.",
    )


# ---------------------------------------------------------------------------
# NEW P3 models – list-based mapping (more reliable with Gemini JSON output)
# ---------------------------------------------------------------------------
class MappingItem(BaseModel):
    """A single raw → canonical mapping entry."""

    raw: str = Field(..., description="Raw value observed in the dirty data.")
    canonical: str = Field(..., description="Chosen clean canonical value.")


class CategoryMapping(BaseModel):
    """P3 schema (v2+): canonical category values and raw-to-clean mapping.

    Uses a list of ``MappingItem`` objects rather than a plain dict because
    dict field ordering is not reliable in Gemini structured-output mode.
    """

    canonical: list[str] = Field(
        default_factory=list,
        description="List of distinct standard canonical categories.",
    )
    mapping: list[MappingItem] = Field(
        default_factory=list,
        description="Complete mapping from every raw value to a canonical value.",
    )
    needs_review: list[str] = Field(
        default_factory=list,
        description="Raw values that could not be reliably mapped and need human review.",
    )


class P4MissingAdvice(BaseModel):
    """P4 schema: imputation or handling strategy for null values."""

    strategy: Literal[
        "leave_null",
        "fill_constant",
        "fill_mode",
        "fill_median",
        "drop_row",
        "manual",
    ]
    rationale: str = Field(
        ..., description="One-sentence rationale supporting the selected strategy."
    )
    risk: str = Field(
        ..., description="Potential data quality or bias risk introduced by the strategy."
    )


class P5Summary(BaseModel):
    """P5 schema: executive data quality summary."""

    summary: str = Field(
        ...,
        description="Concise summary (about 150 words) referencing only provided input numbers.",
    )


# ---------------------------------------------------------------------------
# Prompt Templates and Functions
# ---------------------------------------------------------------------------

DEFAULT_TEMPERATURE: float = 0.0

# --------------------------------------------------
# P1: Column Profiler
# --------------------------------------------------

P1_v1 = """ROLE: You are a careful data quality analyst.
TASK: Infer the semantic type and constraints for the dataset column "{column}".
CONTEXT:
- Column name: "{column}"
- Null percentage: {null_pct}%
- 15 sample values: {samples_json}
ALLOWED SEMANTIC TYPES:
- id, person_name, email, phone, date, money, integer, category, country, city, free_text
RULES:
1. Select exactly one semantic_type from the allowed list.
2. If unsure between category and free_text, choose category only if values repeat.
3. State expected_format concisely (e.g. "YYYY-MM-DD", "E.164", "alphanumeric").
4. Provide a single-sentence reason justifying the classification.
5. Do not invent types outside the allowed list.
Return JSON matching schema."""


def render_p1_v1(column: str, samples: list, null_pct: float) -> str:
    """Render P1 column profiler prompt (v1)."""
    prompt = P1_v1.replace("{column}", column)
    prompt = prompt.replace("{null_pct}", f"{null_pct:.2f}")
    prompt = prompt.replace("{samples_json}", json.dumps(samples[:15]))
    return prompt


# --------------------------------------------------
# P2: Format Rule Proposer
# --------------------------------------------------

P2_v1 = """ROLE: You are a careful data quality analyst.
TASK: Propose non-destructive format cleaning rules for column "{column}".
CONTEXT:
- Column: "{column}"
- Inferred type: "{semantic_type}"
- Detected format patterns (pattern: count, sample): {patterns_json}
RULES:
1. Propose transformation rules from: strip_trim, title_case, uppercase, lowercase, parse_date, normalize_phone, parse_money.
2. Every rule must include action, params dict, and a one-sentence reason.
3. Do not invent custom transformations. If the column is already uniform, return an empty rules list.
4. Never modify raw numeric values or reinterpret ambiguous dates without flagging.
Return JSON matching schema."""


def render_p2_v1(column: str, semantic_type: str, patterns: dict) -> str:
    """Render P2 format rule proposer prompt (v1)."""
    # Generate patterns_json: format signatures with counts and 2 examples each
    patterns_parts = []
    for ptype, pdata in patterns.items():
        # Handle both dict format {signature, count, samples} and integer count format
        if isinstance(pdata, int):
            sig = ptype  # use the key as the signature when value is a plain count
            count = pdata
            samples = []
        else:
            sig = pdata.get("signature", "unknown")
            count = pdata.get("count", 0)
            samples = pdata.get("samples", [])[:2]
        samples_str = ", ".join(str(s) for s in samples)
        patterns_parts.append(f"{sig}: {count}, {samples_str}")
    patterns_json = " | ".join(patterns_parts)

    prompt = P2_v1.replace("{column}", column)
    prompt = prompt.replace("{semantic_type}", semantic_type)
    prompt = prompt.replace("{patterns_json}", patterns_json)
    return prompt


# --------------------------------------------------
# P3: Category Canonicalizer (v1, v2, v3)
# --------------------------------------------------

P3_v1 = """ROLE: You are a careful data quality analyst.
TASK: Group spelling, casing and abbreviation variants of the same category in the
column "{column}".
RULES:
1. Every input value must appear exactly once as "raw" in "mapping".
2. Pick the most frequent clean spelling as the canonical value. Do not invent new categories.
3. Merge only values that clearly mean the same thing. If unsure, put the value in
   "needs_review" and map it to itself.
4. Never translate, drop or reinterpret a value.
INPUT (value: count): {values_json}
Return JSON only."""

P3_v2 = """ROLE: You are a careful data quality analyst.
TASK: Group spelling, casing and abbreviation variants of the same category in the
column "{column}".
RULES:
1. Every input value must appear exactly once as "raw" in "mapping".
2. Pick the most frequent clean spelling as the canonical value. Do not invent new categories.
3. Merge only values that clearly mean the same thing. If unsure, put the value in
   "needs_review" and map it to itself.
4. Never translate, drop or reinterpret a value.
INPUT (value: count): {values_json}
Return JSON only (no example)."""


def render_p3_v1(column: str, values: dict) -> str:
    """Render P3 category canonicalizer prompt (v1: simple)."""
    prompt = P3_v1.replace("{column}", column)
    prompt = prompt.replace("{values_json}", json.dumps(values))
    return prompt


def render_p3_v2(column: str, values: dict) -> str:
    """Render P3 category canonicalizer prompt (v2: simple without example)."""
    prompt = P3_v2.replace("{column}", column)
    prompt = prompt.replace("{values_json}", json.dumps(values))
    return prompt


P3_v3 = """ROLE: You are a careful data quality analyst.
TASK: Group spelling, casing and abbreviation variants of the same category in the
column "{column}".
RULES:
1. Every input value must appear exactly once as "raw" in "mapping".
2. Pick the most frequent clean spelling as the canonical value. Do not invent new categories.
3. Merge only values that clearly mean the same thing. If unsure, put the value in
   "needs_review" and map it to itself.
4. Never translate, drop or reinterpret a value.
EXAMPLE
Input: {"UPI": 120, "upi": 9, "U.P.I": 2, "Cash on Delivery": 80, "COD": 14, "Cheque": 1}
Output: canonical ["UPI","Cash on Delivery","Cheque"]; mapping UPI->UPI, upi->UPI,
U.P.I->UPI, Cash on Delivery->Cash on Delivery, COD->Cash on Delivery, Cheque->Cheque;
needs_review [].
INPUT (value: count): {values_json}
Return JSON only."""


def render_p3_v3(column: str, values: dict) -> str:
    """Render P3 category canonicalizer prompt (v3: few-shot with escape hatch)."""
    prompt = P3_v3.replace("{column}", column)
    prompt = prompt.replace("{values_json}", json.dumps(values))
    return prompt


# Mapping version → template string (used by render_p3 and prompt_lab.py)

P3_VERSIONS: dict[str, str] = {"v1": P3_v1, "v2": P3_v2, "v3": P3_v3}


def render_p3(version: str, column: str, value_counts: dict) -> str:
    """Render a P3 prompt for *column* using the template of *version*.

    Parameters
    ----------
    version:
        One of ``"v1"``, ``"v2"``, or ``"v3"``.
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
    prompt = tmpl.replace("{column}", column)
    prompt = tmpl.replace("{values_json}", json.dumps(value_counts))
    return prompt
    """Return a flat dict of numbers for P5 summary and lab scoring.

    Parameters
    ----------
    context : dict
        The dict returned by build_report_context.

    Returns
    -------
    dict
        Flat dict with: rows, cols, null_before, placeholder_before,
        flags_before_per_type, flags_after_per_type, failures_before_total,
        failures_after_total, quality_before, quality_after, changes_applied.
    """
    stats = {
        "rows": context["rows"],
        "cols": context["cols"],
        "null_before": sum(context["null_log_after_blank"].values()) + sum(context["null_log_after_placeholder"].values()),
        "placeholder_before": sum(context["null_log_after_placeholder"].values()),
        "flags_before_per_type": dict(context["detector_counts_by_type_before"]),
        "flags_after_per_type": dict(context["detector_counts_by_type_after"]),
        "failures_before_total": context["validation_summary_before"]["total_failures"],
        "failures_after_total": context["validation_summary_after"]["total_failures"],
        "quality_before": context["quality_before"],
        "quality_after": context["quality_after"],
        "changes_applied": sum(context["rule_counts"].values()) if context["rule_counts"] else 0,
    }
    return stats

P3_VERSIONS: dict[str, str] = {"v3": P3_v3}


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


# --------------------------------------------------
# P4: Missing-Value Advisor
# --------------------------------------------------

P4_v1 = """ROLE: You are a careful data quality analyst.
TASK: Recommend a missing-value handling strategy for column "{column}".
CONTEXT:
- Column name: {column}
- Semantic type: {semantic_type}
- Null percentage: {null_pct}
- Distinct value distribution (top values): {distribution_json}
ALLOWED STRATEGIES:
- leave_null: Keep as null (e.g. conditional/optional field).
- fill_constant: Impute fixed known default.
- fill_mode: Impute most frequent categorical value.
- fill_median: Impute median for numerical values.
- drop_row: Drop row if key identifier is missing.
- manual: Flag for human decision.
RULES:
1. Choose exactly one strategy from the allowed list.
2. Provide a one-sentence rationale explaining why this strategy fits.
3. Explicitly state the risk (e.g. distortion of distribution, introduction of bias).
4. Negative instruction: Do not recommend drop_row if null percentage > 5% or if column is non-critical.
5. If unsure or if column is a primary key, recommend manual.
Return JSON matching schema."""


def render_p4_v1(
    column: str,
    semantic_type: str,
    null_pct: float,
    distribution: dict,
) -> str:
    """Render P4 missing-value advisor prompt (v1)."""
    # Generate distribution_json: top values with counts
    top_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:5]
    dist_parts = []
    for val, count in top_items:
        val_str = str(val).replace("'", "\'")
        dist_parts.append(f"{{{val_str}}}: {count}")
    distribution_json = "; ".join(dist_parts)

    prompt = P4_v1.replace("{column}", column)
    prompt = prompt.replace("{semantic_type}", semantic_type)
    prompt = prompt.replace("{null_pct}", f"{null_pct:.2f}%")
    prompt = prompt.replace("{distribution_json}", distribution_json)
    return prompt


# --------------------------------------------------
# P5: Report Summarizer
# --------------------------------------------------

P5_v1 = """ROLE: You are a careful data quality analyst.
TASK: Write an executive summary of dataset cleaning and validation findings.
INPUT STATISTICS (JSON):
{stats_json}
CONSTRAINTS:
1. Target length: approximately 150 words.
2. Factuality constraint: You may use ONLY numbers present in the input statistics JSON.
3. Do not invent, extrapolate, or guess any counts, percentages, or metrics.
4. Highlight major issue types (duplicates, missing values, format inconsistencies) and overall dataset health.
Return JSON matching schema."""


def render_p5_v1(stats: dict) -> str:
    """Render P5 report summarizer prompt (v1)."""
    stats_json = json.dumps(stats, indent=2)
    prompt = P5_v1.replace("{stats_json}", stats_json)
    return prompt


def build_summary_stats(context: dict) -> dict:
    """Return a flat dict of numbers for P5 summary and lab scoring.

    Parameters
    ----------
    context : dict
        The dict returned by build_report_context.

    Returns
    -------
    dict
        Flat dict with: rows, cols, null_before, placeholder_before,
        flags_before_per_type, flags_after_per_type, failures_before_total,
        failures_after_total, quality_before, quality_after, changes_applied.
    """
    stats = {
        "rows": context["rows"],
        "cols": context["cols"],
        "null_before": sum(context["null_log_after_blank"].values()) + sum(context["null_log_after_placeholder"].values()),
        "placeholder_before": sum(context["null_log_after_placeholder"].values()),
        "flags_before_per_type": dict(context["detector_counts_by_type_before"]),
        "flags_after_per_type": dict(context["detector_counts_by_type_after"]),
        "failures_before_total": context["validation_summary_before"]["total_failures"],
        "failures_after_total": context["validation_summary_after"]["total_failures"],
        "quality_before": context["quality_before"],
        "quality_after": context["quality_after"],
        "changes_applied": sum(context["rule_counts"].values()) if context["rule_counts"] else 0,
    }
    return stats
