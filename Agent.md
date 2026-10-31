# Project instructions
Streamlit tool in Python 3.11+ that cleans and validates CSV files.
Stack: pandas, Pandera, RapidFuzz, google-genai, Streamlit, Jinja2, pytest.
Code rules: type hints, short docstrings, small functions, no global state.
Data rules: all data is read as strings; nulls are pd.NA; row numbers are 1-based
data rows (header not counted).
Detectors return a DataFrame with columns [row, column, value, issue_type, reason].
Do not call an LLM in detectors, rules or schema code.
Never open or read data/planted_issues.csv, data/answer_key_summary.json,
data/orders_ground_truth.csv or make_dirty_data.py while writing code.
Never read, print or modify .env.
Run pytest after every change and commit only when tests pass.