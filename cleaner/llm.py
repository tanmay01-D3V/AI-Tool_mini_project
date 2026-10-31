"""LLM caller module using google-genai with JSON schema, caching, and retries."""

import hashlib
import json
import os
import random
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_CACHE_DIR = Path("outputs/.llm_cache")
DEFAULT_RUNS_DIR = Path("prompt_log/runs")


def _is_retryable_error(exc: Exception) -> bool:
    """Return True for HTTP 429 (rate-limit) or 503 (service unavailable) errors."""
    for code in (429, 503):
        if hasattr(exc, "code") and getattr(exc, "code") == code:
            return True
        if hasattr(exc, "status_code") and getattr(exc, "status_code") == code:
            return True
    msg = str(exc)
    return "429" in msg or "503" in msg or "UNAVAILABLE" in msg


# Backward-compat alias
_is_rate_limit_error = _is_retryable_error


def _compute_cache_key(prompt: str, model: str) -> str:
    """Compute SHA-256 cache key from model name and prompt string."""
    payload = f"{model}:{prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _extract_token_counts(response: object) -> dict[str, int | None]:
    """Extract token counts from response usage metadata if present."""
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return {
            "prompt_tokens": None,
            "candidates_tokens": None,
            "total_tokens": None,
        }
    return {
        "prompt_tokens": getattr(metadata, "prompt_token_count", None),
        "candidates_tokens": getattr(metadata, "candidates_token_count", None),
        "total_tokens": getattr(metadata, "total_token_count", None),
    }


def _clean_json_text(text: str) -> str:
    """Strip markdown code block wrappers from JSON string if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _save_run_log(
    log_dir: Path,
    prompt: str,
    response_text: str,
    model: str,
    token_counts: dict[str, int | None],
) -> Path:
    """Persist raw request and response details as a JSON file in runs dir."""
    log_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    timestamp_iso = now.isoformat()
    filename = f"run_{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
    log_path = log_dir / filename

    payload = {
        "timestamp": timestamp_iso,
        "model": model,
        "prompt": prompt,
        "response": response_text,
        "token_counts": token_counts,
    }
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return log_path


def _save_cache(
    cache_dir: Path,
    cache_key: str,
    prompt: str,
    model: str,
    response_text: str,
) -> Path:
    """Save raw response text in disk cache under outputs/.llm_cache/."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.json"
    payload = {
        "prompt": prompt,
        "model": model,
        "response_text": response_text,
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cache_path


def _execute_with_retry(
    client: object,
    model: str,
    prompt: str,
    config: types.GenerateContentConfig,
    max_tries: int = 4,
    base_delay: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> object:
    """Execute generate_content with exponential backoff on HTTP 429."""
    for attempt in range(max_tries):
        try:
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            if _is_rate_limit_error(exc) and attempt < max_tries - 1:
                delay = (base_delay * (2 ** attempt)) + random.uniform(0.0, 0.25)
                sleep_fn(delay)
                continue
            raise


def call_gemini(
    prompt: str,
    schema: type[T] | None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    client: object | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    log_dir: Path | str = DEFAULT_RUNS_DIR,
    max_tries: int = 4,
    base_delay: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T | str:
    """Call Gemini API with structured JSON output, caching, and retry on 429.

    Args:
        prompt: User prompt sent to Gemini.
        schema: Target Pydantic model for structured JSON decoding. Use None for raw text response.
        model: Model identifier.
        temperature: Sampling temperature.
        client: Optional injected Gemini client (for testing).
        cache_dir: Directory where response cache files are stored.
        log_dir: Directory where run log files are stored.
        max_tries: Maximum attempts for HTTP 429 retries.
        base_delay: Base delay for exponential backoff in seconds.
        sleep_fn: Sleep function (overridable in tests).

    Returns:
        Validated Pydantic model instance, or raw response text if schema is None.
    """
    cache_dir_path = Path(cache_dir)
    log_dir_path = Path(log_dir)

    # 1. Check disk cache
    cache_key = _compute_cache_key(prompt, model)
    cache_file = cache_dir_path / f"{cache_key}.json"
    if cache_file.exists():
        cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
        cached_text = cached_data.get("response_text", "")
        if schema is not None:
            return schema.model_validate_json(_clean_json_text(cached_text))
        return _clean_json_text(cached_text)

    # 2. Setup client if not injected
    if client is None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

    # 3. Configure JSON schema output
    if schema is not None:
        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )
    else:
        config = types.GenerateContentConfig(
            temperature=temperature,
        )

    # 4. Generate content with retries
    response = _execute_with_retry(
        client=client,
        model=model,
        prompt=prompt,
        config=config,
        max_tries=max_tries,
        base_delay=base_delay,
        sleep_fn=sleep_fn,
    )

    response_text = getattr(response, "text", "") or ""
    token_counts = _extract_token_counts(response)

    # 5. Log raw request and response
    _save_run_log(
        log_dir=log_dir_path,
        prompt=prompt,
        response_text=response_text,
        model=model,
        token_counts=token_counts,
    )

    # 6. Save to disk cache
    _save_cache(
        cache_dir=cache_dir_path,
        cache_key=cache_key,
        prompt=prompt,
        model=model,
        response_text=response_text,
    )

    # 7. Parse and return
    if schema is not None:
        return schema.model_validate_json(_clean_json_text(response_text))
    return _clean_json_text(response_text)

