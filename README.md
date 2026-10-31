# AI Tool Mini Project - Data Quality Lab

A Streamlit-based data quality analysis application that provides 6 pages of CSV analysis, integrating rule-based cleaning with Gemini AI calls for type detection, category canonicalization, and report generation.

## Project Structure

The project implements **B1-B10** features across multiple commits:

| Commit | Feature |
|--------|---------|
| B1 | CSV loader and profiler |
| B2a | Missing and duplicate detectors |
| B3 | Gemini client with retry and cache |
| B5 | Rules engine and change log |
| B6 | Pandera schema and validation |
| B7 | HTML validation report |
| B9 and B10 | Prompt lab with P1 to P5 prompts |
| **B8** | **Streamlit app** (latest) |

## Features

### 6-Page Streamlit App

1. **Upload** - CSV file uploader with row/column display and null detection
2. **Profile** - Column profiles + P1 Gemini type detection (cached, under 40 calls/file)
3. **Issues found** - Missing values, duplicates, format validation with schema checks
4. **Proposed rules** - Gemini category mappings (P3) + format rules (P2) + rule-based fixes
5. **Before / after** - Sample comparison with quality scores and Pandera failures
6. **Download** - Cleaned CSV, change log, HTML validation report, prompt_log zip

### Gemini Integration

- **Sidebar toggle** "Use Gemini" (default ON)
- **Spinner during calls** with caching by prompt hash
- **Under 40 Gemini calls per file**: P1 per column, P3 for categories, P2 for formats, P4 per column with nulls, P5 once
- **Graceful fallback**: If API key missing or quota exceeded, continues in rule-only mode
- All detection/cleaning features still work without Gemini

### Rule-Based Cleaning

- `apply_rules()` from `cleaner/rules.py`
- Supported actions: strip_trim, title_case, map_values, parse_date, normalize_phone, parse_money
- Accept/reject checkboxes for proposed rules
- Before/after quality score comparison

### P1-P5 Prompt Templates

- **P1**: Column profiler - semantic type and format detection
- **P2**: Format rule proposer - transformation rules
- **P3**: Category canonicalizer - grouping spelling variants (v1, v2, v3)
- **P4**: Missing-value advisor - imputation strategies
- **P5**: Report summarizer - 150-word executive summary
- All rendered using `str.replace()` for placeholder filling

### Download Options

- Cleaned CSV
- Change log CSV
- HTML validation report (Gemini-generated P5 summary)
- Prompt log zip (all runs/)

## Installation

```bash
# Clone the repository
git clone https://github.com/tanmay01-D3V/AI-Tool_mini_project.git
cd AI-Tool_mini_project

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up Gemini API key
# Copy .env.example to .env and add your GEMINI_API_KEY
cp .env.example .env

# Run the app
streamlit run app.py
```

## Requirements

- Python 3.14.0+
- Streamlit
- Google Generative AI (google-generativeai)
- Pandas
- Pandera (schema validation)
- RapidFuzz (fuzzy matching)
- Python-dateutil (date parsing)
- Phonenumbers (phone validation)

```bash
pip install -r requirements.txt
```

## Test Results

All tests pass:

```
tests/test_app.py: 6/6 passed
tests/test_prompts.py: 6/6 passed
Total: 12/12 passed
```

Run tests:
```bash
cd AI-Tool_mini_project
source .venv/bin/activate
python -m pytest tests/test_app.py tests/test_prompts.py -v
```

## Usage

1. **Start the app**: `streamlit run app.py`
2. **Navigate pages** using sidebar radio
3. **Toggle "Use Gemini"** on/off (default ON)
4. **Upload a CSV** file
5. **Navigate through pages** to analyze data quality
6. **Use Download** buttons to export results

## Project Highlights

- **9 incremental commits** (B1-B10 + initial setup)
- **Rule-only mode** works without Gemini API key
- **Prompt caching** keeps Gemini calls under 40 per file
- **12 pytest tests** verify app structure and prompt rendering
- **HTML reports** generated via Gemini P5 summary
- **Downloadable outputs**: CSV, change log, HTML report, prompt log zip

## Repository

```
https://github.com/tanmay01-D3V/AI-Tool_mini_project.git
```

---
*Built with Streamlit, Gemini AI, Pandera, and RapidFuzz*