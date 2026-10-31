import streamlit as st
import pandas as pd
import json
import os
import zipfile
from pathlib import Path
from datetime import datetime

from cleaner.loader import load_csv
from cleaner.detectors import detect_missing, detect_exact_duplicates, detect_key_duplicates
from cleaner.rules import apply_rules, SUPPORTED_ACTIONS
from cleaner.schema import build_schema, validate, summarize
from cleaner.prompts import (
    render_p1_v1, render_p2_v1, render_p3_v3, render_p4_v1, render_p5_v1,
    build_summary_stats,
)
from cleaner.llm import call_gemini, DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Data Quality Lab", layout="wide")

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None
if "change_log" not in st.session_state:
    st.session_state.change_log = None
if "failures_before" not in st.session_state:
    st.session_state.failures_before = None
if "failures_after" not in st.session_state:
    st.session_state.failures_after = None
if "gemini_enabled" not in st.session_state:
    st.session_state.gemini_enabled = True
if "gemini_calls" not in st.session_state:
    st.session_state.gemini_calls = 0
if "gemini_cache" not in st.session_state:
    st.session_state.gemini_cache = {}
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = "gemini-3.5-flash-lite"


# ---------------------------------------------------------------------------
# Helper: Gemini call with caching and spinner
# ---------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def cached_gemini(prompt, model):
    """Cache Gemini calls by prompt hash."""
    from cleaner.llm import call_gemini
    return call_gemini(prompt, schema=None, model=model, temperature=0.0)


def gemini_call(prompt, model_key, render_func, *render_args):
    """Execute a Gemini call with spinner and caching."""
    with st.spinner(f"Gemini call ({model_key})..."):
        try:
            result = call_gemini(prompt, schema=None, model=model, temperature=0.0)
            st.session_state.gemini_calls += 1
            return result
        except Exception as e:
            st.error(f"Gemini call failed: {e}")
            st.session_state.gemini_calls += 1
            return None


# ---------------------------------------------------------------------------
# Page 1: Upload
# ---------------------------------------------------------------------------
def page_upload():
    st.header("1. Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            st.session_state.df, st.session_state.null_log = load_csv(uploaded_file)
            st.success(f"Loaded {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")


# ---------------------------------------------------------------------------
# Page 2: Profile
# ---------------------------------------------------------------------------
def page_profile():
    st.header("2. Profile")
    if st.session_state.df is None:
        st.info("Upload a CSV first.")
        return

    st.write("### Column profile")
    from cleaner.profiler import profile as profiler
    profile_df = profiler(st.session_state.df)
    st.dataframe(profile_df, use_container_width=True)

    st.write("### Detected types (with Gemini)")
    if st.session_state.gemini_enabled:
        with st.spinner("Running P1 per column (cached)..."):
            for col in st.session_state.df.columns:
                col_data = st.session_state.df[col].dropna().astype(str).tolist()
                null_pct = st.session_state.df[col].isna().mean() * 100
                prompt = render_p1_v1(col, col_data, null_pct)
                # Cache by prompt text
                prompt_hash = hash(prompt)
                if prompt_hash not in st.session_state.gemini_cache:
                    result = call_gemini(prompt, schema=None, model=st.session_state.gemini_model, temperature=0.0)
                    st.session_state.gemini_calls += 1
                    st.session_state.gemini_cache[prompt_hash] = result
                else:
                    st.session_state.gemini_calls += 0  # cached, don't increment
                # Display result next to column
                st.caption(f"{col}: {st.session_state.gemini_cache[prompt_hash]}")
    else:
        st.info("Gemini disabled — no AI type detection.")


# ---------------------------------------------------------------------------
# Page 3: Issues found
# ---------------------------------------------------------------------------
def page_issues():
    st.header("3. Issues found")
    if st.session_state.df is None:
        st.info("Upload a CSV first.")
        return

    # Run detectors
    null_log = st.session_state.get("null_log") if st.session_state.df is not None else None
    if null_log is None:
        _, null_log = load_csv(Path("data/orders_dirty.csv"))  # fallback

    missing_df = detect_missing(st.session_state.df, null_log)
    exact_dups = detect_exact_duplicates(st.session_state.df)
    key_dups = detect_key_duplicates(st.session_state.df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Missing")
        st.dataframe(missing_df, use_container_width=True)
    with col2:
        st.subheader("Duplicates")
        st.dataframe(exact_dups, use_container_width=True)
    with col3:
        st.subheader("Formats/Invalid")
        # Schema validation
        from cleaner.orders_config import ORDERS_COLUMN_TYPES
        schema = build_schema(ORDERS_COLUMN_TYPES, cross_rules=True)
        failures_before = validate(st.session_state.df, schema)
        st.dataframe(failures_before, use_container_width=True)

    # P4 advice (display only)
    st.subheader("P4 advice (null handling)")
    if st.session_state.gemini_enabled:
        with st.spinner("Generating P4 advice..."):
            for col in st.session_state.df.columns:
                col_data = st.session_state.df[col].dropna()
                null_count = st.session_state.df[col].isna().sum()
                if null_count > 0:
                    prompt = render_p4_v1(col, "category", null_count/len(st.session_state.df), {})
                    result = call_gemini(prompt, schema=None, model=st.session_state.gemini_model, temperature=0.0)
                    st.caption(f"{col}: {result}")


# ---------------------------------------------------------------------------
# Page 4: Proposed rules
# ---------------------------------------------------------------------------
def page_rules():
    st.header("4. Proposed rules")
    if st.session_state.df is None:
        st.info("Upload a CSV first.")
        return

    # Gemini proposals (P3 category mappings, P2 format rules)
    if st.session_state.gemini_enabled:
        st.write("### Gemini proposals")
        # P3 category mappings for columns with categorical data
        categorical_cols = ["category", "country", "city"]
        for col in categorical_cols:
            if col in st.session_state.df.columns:
                value_counts = st.session_state.df[col].dropna().astype(str).value_counts().to_dict()
                if value_counts:
                    prompt = render_p3_v3(col, value_counts)
                    result = call_gemini(prompt, schema=None, model=st.session_state.gemini_model, temperature=0.0)
                    st.caption(f"{col} (P3): {result}")

        # P2 format rules
        format_cols = ["signup_date", "order_date", "phone", "unit_price", "total_amount", "customer_name", "email"]
        for col in format_cols:
            if col in st.session_state.df.columns:
                value_counts = st.session_state.df[col].dropna().astype(str).value_counts().to_dict()
                if value_counts:
                    # Generate format patterns
                    patterns = {}
                    for val in value_counts:
                        # Simple pattern generation
                        if val.isdigit():
                            patterns[val] = {"signature": "digit", "count": value_counts[val]}
                        else:
                            patterns[val] = {"signature": "alpha", "count": value_counts[val]}
                    prompt = render_p2_v1(col, "category", patterns)
                    result = call_gemini(prompt, schema=None, model=st.session_state.gemini_model, temperature=0.0)
                    st.caption(f"{col} (P2): {result}")

    # Rule-based fixes
    st.write("### Rule-based fixes")
    st.write(f"Available actions: {', '.join(SUPPORTED_ACTIONS)}")

    # Rule inputs
    with st.form(key="rules_form"):
        st.write("Add cleaning rules (column, action, params):")
        col1, col2, col3 = st.columns(3)
        with col1:
            rule_col = st.selectbox("Column", st.session_state.df.columns.tolist() if st.session_state.df is not None and len(st.session_state.df) > 0 else [])
        with col2:
            rule_action = st.selectbox("Action", SUPPORTED_ACTIONS)
        with col3:
            rule_param = st.text_input("Param (JSON dict, e.g. {\"mapping\": {\"US\": \"USA\"}})", "{}")

        submit_rule = st.form_submit_button("Add rule")
        if submit_rule and rule_col and rule_action:
            try:
                params = json.loads(rule_param) if rule_param else {}
                new_rule = {"column": rule_col, "action": rule_action, "params": params}
                st.session_state.rules = st.session_state.get("rules", []) + [new_rule]
                st.success("Rule added!")
            except json.JSONDecodeError:
                st.error("Invalid JSON in params")

    # Display existing rules with accept/reject checkboxes
    st.write("### Proposed rules")
    rules = st.session_state.get("rules", [])
    accepted_rules = []

    for i, rule in enumerate(rules):
        # Use unique keys for each checkbox to avoid duplicate key errors
        checkbox_key = f"rule_accept_{i}"
        st.checkbox(f"{rule['action']} on {rule['column']}", key=checkbox_key, value=False)
        accepted_rules.append(rule)  # In real app, would check which are checked

    # Apply accepted rules
    if st.button("Apply accepted rules"):
        if 'rules' in st.session_state and st.session_state.rules:
            # Filter to accepted rules (all start unchecked, so none accepted unless changed)
            accepted = [r for i, r in enumerate(rules) if st.session_state.get(f"rule_accept_{i}", False)]
            with st.spinner("Applying rules..."):
                st.session_state.clean_df, st.session_state.change_log = apply_rules(
                    st.session_state.df, accepted
                )
                # Run schema validation after
                from cleaner.orders_config import ORDERS_COLUMN_TYPES
                schema = build_schema(ORDERS_COLUMN_TYPES, cross_rules=True)
                st.session_state.failures_before = validate(st.session_state.df, schema)
                st.session_state.failures_after = validate(st.session_state.clean_df, schema)
                st.success("Rules applied!")
        else:
            st.warning("No rules to apply.")


# ---------------------------------------------------------------------------
# Page 5: Before / after
# ---------------------------------------------------------------------------
def page_before_after():
    st.header("5. Before / after")
    if st.session_state.clean_df is None:
        st.info("Apply rules first on Page 4.")
        return

    st.write("### Before sample")
    st.dataframe(st.session_state.df.head(10), use_container_width=True)
    st.write("### After sample")
    st.dataframe(st.session_state.clean_df.head(10), use_container_width=True)

    st.write("### Change log")
    if st.session_state.change_log is not None and not st.session_state.change_log.empty:
        st.dataframe(st.session_state.change_log, use_container_width=True)
    else:
        st.info("No changes logged.")

    st.write("### Quality scores")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Before", f"{st.session_state.get('quality_before', 0):.4f}")
    with col2:
        st.metric("After", f"{st.session_state.get('quality_after', 0):.4f}")

    st.write("### Pandera failures")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Before:")
        st.dataframe(st.session_state.get('failures_before', pd.DataFrame()), use_container_width=True)
    with col2:
        st.write("After:")
        st.dataframe(st.session_state.get('failures_after', pd.DataFrame()), use_container_width=True)


# ---------------------------------------------------------------------------
# Page 6: Download
# ---------------------------------------------------------------------------
def page_download():
    st.header("6. Download")
    if st.session_state.clean_df is None:
        st.info("Apply rules first.")
        return

    # Clean CSV
    csv = st.session_state.clean_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download cleaned CSV", csv, file_name="cleaned.csv", mime="text/csv")

    # Change log CSV
    if st.session_state.change_log is not None and not st.session_state.change_log.empty:
        cl_csv = st.session_state.change_log.to_csv(index=False).encode("utf-8")
        st.download_button("Download change log CSV", cl_csv, file_name="change_log.csv", mime="text/csv")
    else:
        st.info("No change log available.")

    # HTML validation report
    if st.session_state.gemini_enabled:
        if st.button("Generate HTML report"):
            with st.spinner("Generating HTML report..."):
                from cleaner.orders_config import ORDERS_COLUMN_TYPES
                schema = build_schema(ORDERS_COLUMN_TYPES, cross_rules=True)
                failures_after = validate(st.session_state.clean_df, schema)
                stats = build_summary_stats({
                    "null_log_after_blank": {},
                    "null_log_after_placeholder": {},
                    "rule_counts": {},
                    "validation_summary_before": {"total_failures": 0},
                    "validation_summary_after": {"total_failures": failures_after["total_failures"]},
                    "quality_before": 1.0,
                    "quality_after": 1.0,
                    "rows": len(st.session_state.clean_df),
                    "cols": len(st.session_state.clean_df.columns),
                })
                prompt = render_p5_v1(stats)
                result = call_gemini(prompt, schema=None, model=st.session_state.gemini_model, temperature=0.0)
                st.session_state.gemini_calls += 1
                report_html = result if result else "# Gemini report generation failed"
                st.download_button("Download HTML report", report_html, file_name="report.html", mime="text/html")
    else:
        st.info("Gemini disabled — HTML report not generated.")

    # Zip of prompt_log/runs
    runs_dir = Path("prompt_log/runs")
    zip_path = Path("prompt_log/output.zip")
    if runs_dir.exists():
        with zipfile.ZipFile(zip_path, "w") as zf:
            for root, dirs, files in os.walk(runs_dir):
                for file in files:
                    zf.write(os.path.join(root, file), arcname=file)
        with open(zip_path, "rb") as f:
            st.download_button("Download prompt_log zip", f.read(), file_name="prompt_log.zip", mime="application/zip")
    else:
        st.info("No prompt logs yet.")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Upload", "Profile", "Issues found", "Proposed rules", "Before / after", "Download"],
    )

    # Sidebar Gemini toggle - only once, with unique key
    st.session_state.gemini_enabled = st.sidebar.checkbox("Use Gemini", value=True, key="toggle_gemini")

    # ORDERS_COLUMN_TYPES for schema building
    from cleaner.orders_config import ORDERS_COLUMN_TYPES

    # Profiler instance
    from cleaner.profiler import profile as profiler

    # Route to pages
    if page == "Upload":
        page_upload()
    elif page == "Profile":
        page_profile()
    elif page == "Issues found":
        page_issues()
    elif page == "Proposed rules":
        page_rules()
    elif page == "Before / after":
        page_before_after()
    elif page == "Download":
        page_download()


if __name__ == "__main__":
    main()