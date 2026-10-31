# tests/test_prompt_lab.py
"""Unit tests for the prompt-lab scoring helpers (offline – no Gemini calls)."""

from __future__ import annotations

from scripts.prompt_lab import score_run
from cleaner.prompts import CategoryMapping, MappingItem


def _make_mapping(
    canonical: list[str],
    pairs: list[tuple[str, str]],
    needs_review: list[str] | None = None,
) -> CategoryMapping:
    """Convenience factory for CategoryMapping."""
    return CategoryMapping(
        canonical=canonical,
        mapping=[MappingItem(raw=r, canonical=c) for r, c in pairs],
        needs_review=needs_review or [],
    )


# ---------------------------------------------------------------------------
# Test 1 – basic correct + safe + wrong + invented counting
# ---------------------------------------------------------------------------

def test_score_run_correct():
    """Both values mapped correctly → correct=2, wrong=0, safe=0, invented=0."""
    truth = {"foo": "FOO", "bar": "BAR"}
    m = _make_mapping(["FOO", "BAR"], [("foo", "FOO"), ("bar", "BAR")])
    s = score_run("col", m, truth)
    assert s[0]["valid"] == 1
    assert s[0]["correct"] == 2
    assert s[0]["wrong"] == 0
    assert s[0]["safe"] == 0
    assert s[0]["invented"] == 0


def test_score_run_safe():
    """Value put in needs_review and mapped to itself → safe=1, wrong=0.

    For a 'safe' result the truth canonical must differ from the raw value so that
    mapping raw→raw is NOT correct, but because the model put it in needs_review
    the result counts as 'safe' rather than 'wrong'.
    """
    # truth: foo→FOO (correct), bar→Bar (model is uncertain, maps bar→bar → safe)
    truth = {"foo": "FOO", "bar": "Bar"}
    m = _make_mapping(
        ["FOO"],
        [("foo", "FOO"), ("bar", "bar")],
        needs_review=["bar"],
    )
    s = score_run("col", m, truth)
    assert s[0]["correct"] == 1   # foo is correct
    assert s[0]["safe"] == 1      # bar is in needs_review and maps to itself → safe
    assert s[0]["wrong"] == 0


def test_score_run_wrong():
    """Value mapped to wrong canonical → wrong=1."""
    truth = {"foo": "FOO", "bar": "BAR"}
    # model maps bar → FOO (wrong)
    m = _make_mapping(["FOO"], [("foo", "FOO"), ("bar", "FOO")])
    s = score_run("col", m, truth)
    assert s[0]["wrong"] == 1
    assert s[0]["correct"] == 1


def test_score_run_invented_raw():
    """Model includes a raw value not in truth → invented += 1."""
    truth = {"foo": "FOO"}
    # model adds "baz" which is not in truth
    m = _make_mapping(["FOO"], [("foo", "FOO"), ("baz", "BAZ")])
    s = score_run("col", m, truth)
    assert s[0]["invented"] == 1
    assert s[0]["correct"] == 1


def test_score_run_missing_raw():
    """Truth has a raw value absent from mapping → counts as invented (uncovered)."""
    truth = {"foo": "FOO", "bar": "BAR"}
    # model only maps foo
    m = _make_mapping(["FOO"], [("foo", "FOO")])
    s = score_run("col", m, truth)
    # bar is not in mapping → treated as missing (invented += 1)
    assert s[0]["invented"] == 1
    assert s[0]["correct"] == 1
    assert s[0]["coverage"] == 0   # truth not fully covered


def test_score_run_coverage_flag():
    """coverage=1 only when every truth key appears in mapping."""
    truth = {"a": "A", "b": "B"}
    full = _make_mapping(["A", "B"], [("a", "A"), ("b", "B")])
    partial = _make_mapping(["A"], [("a", "A")])
    assert score_run("col", full, truth)[0]["coverage"] == 1
    assert score_run("col", partial, truth)[0]["coverage"] == 0
