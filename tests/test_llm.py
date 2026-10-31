"""Tests for cleaner.llm module using injected fake Gemini client."""

import json
import sys
from pathlib import Path
from typing import Any
import pytest
from pydantic import BaseModel

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from cleaner.llm import call_gemini


class DummySchema(BaseModel):
    summary: str
    confidence: float


class FakeUsageMetadata:
    def __init__(self, prompt_tokens: int = 12, candidates_tokens: int = 8, total_tokens: int = 20) -> None:
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = candidates_tokens
        self.total_token_count = total_tokens


class FakeResponse:
    def __init__(self, text: str, usage: FakeUsageMetadata | None = None) -> None:
        self.text = text
        self.usage_metadata = usage or FakeUsageMetadata()


class FakeRateLimitError(Exception):
    def __init__(self, message: str = "Rate limit exceeded", code: int = 429) -> None:
        super().__init__(message)
        self.code = code


class FakeModels:
    def __init__(self, items: list[Any]) -> None:
        self.items = list(items)
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, model: str, contents: str, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        if not self.items:
            raise RuntimeError("FakeModels called with no remaining responses")
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, items: list[Any]) -> None:
        self.models = FakeModels(items)


def test_call_gemini_valid_response(tmp_path: Path) -> None:
    """Verify call_gemini parses valid response, logs run, and populates cache."""
    cache_dir = tmp_path / "cache"
    log_dir = tmp_path / "runs"

    response_json = json.dumps({"summary": "Data clean", "confidence": 0.98})
    fake_client = FakeClient([FakeResponse(text=response_json)])

    result = call_gemini(
        prompt="Analyze data",
        schema=DummySchema,
        model="gemini-3.5-flash-lite",
        client=fake_client,
        cache_dir=cache_dir,
        log_dir=log_dir,
    )

    assert isinstance(result, DummySchema)
    assert result.summary == "Data clean"
    assert result.confidence == 0.98
    assert len(fake_client.models.calls) == 1

    # Check run log
    log_files = list(log_dir.glob("*.json"))
    assert len(log_files) == 1
    log_data = json.loads(log_files[0].read_text(encoding="utf-8"))
    assert log_data["prompt"] == "Analyze data"
    assert log_data["model"] == "gemini-3.5-flash-lite"
    assert log_data["response"] == response_json
    assert log_data["token_counts"]["prompt_tokens"] == 12
    assert "timestamp" in log_data

    # Check cache file created
    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 1


def test_call_gemini_429_then_success(tmp_path: Path) -> None:
    """Verify exponential backoff retries when encountering HTTP 429 errors."""
    cache_dir = tmp_path / "cache"
    log_dir = tmp_path / "runs"

    response_json = json.dumps({"summary": "Recovered", "confidence": 0.92})
    fake_client = FakeClient([
        FakeRateLimitError("Quota exhausted", code=429),
        FakeResponse(text=response_json),
    ])

    sleeps: list[float] = []

    result = call_gemini(
        prompt="Retry test",
        schema=DummySchema,
        client=fake_client,
        cache_dir=cache_dir,
        log_dir=log_dir,
        sleep_fn=sleeps.append,
    )

    assert isinstance(result, DummySchema)
    assert result.summary == "Recovered"
    assert len(fake_client.models.calls) == 2
    assert len(sleeps) == 1
    assert sleeps[0] >= 1.0  # base delay + jitter


def test_call_gemini_cache_hit(tmp_path: Path) -> None:
    """Verify subsequent call with identical prompt and model hits disk cache."""
    cache_dir = tmp_path / "cache"
    log_dir = tmp_path / "runs"

    response_json = json.dumps({"summary": "Cached entry", "confidence": 0.99})
    first_client = FakeClient([FakeResponse(text=response_json)])

    # First call stores in cache
    res1 = call_gemini(
        prompt="Common prompt",
        schema=DummySchema,
        client=first_client,
        cache_dir=cache_dir,
        log_dir=log_dir,
    )
    assert res1.summary == "Cached entry"
    assert len(first_client.models.calls) == 1

    # Second client has zero responses and raises if called
    second_client = FakeClient([])
    res2 = call_gemini(
        prompt="Common prompt",
        schema=DummySchema,
        client=second_client,
        cache_dir=cache_dir,
        log_dir=log_dir,
    )

    assert res2.summary == "Cached entry"
    assert res2.confidence == 0.99
    # Verify second client was not touched
    assert len(second_client.models.calls) == 0


def test_call_gemini_429_max_tries_exhausted(tmp_path: Path) -> None:
    """Verify rate limit exception is raised when all retries are exhausted."""
    cache_dir = tmp_path / "cache"
    log_dir = tmp_path / "runs"

    fake_client = FakeClient([
        FakeRateLimitError("Rate limit 1", code=429),
        FakeRateLimitError("Rate limit 2", code=429),
        FakeRateLimitError("Rate limit 3", code=429),
        FakeRateLimitError("Rate limit 4", code=429),
    ])

    sleeps: list[float] = []

    with pytest.raises(FakeRateLimitError):
        call_gemini(
            prompt="Exhaust test",
            schema=DummySchema,
            client=fake_client,
            cache_dir=cache_dir,
            log_dir=log_dir,
            max_tries=4,
            sleep_fn=sleeps.append,
        )

    assert len(fake_client.models.calls) == 4
    assert len(sleeps) == 3

