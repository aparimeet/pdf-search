from __future__ import annotations

"""API key auth and simple in-memory rate limiting."""

import os
import time
from typing import Dict

from fastapi import Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_429_TOO_MANY_REQUESTS

API_KEY_HEADER = APIKeyHeader(name="x-api-key", auto_error=False)

# Simple burst + sliding window in-memory rate limiter (per key)
_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "60"))
_WINDOW_SECS = int(os.getenv("RATE_LIMIT_WINDOW_SECS", "60"))
_request_log: Dict[str, list] = {}


def _rate_limit_check(api_key: str) -> None:
    """Track requests per API key and raise when exceeding window limits."""
    now = time.time()
    window_start = now - _WINDOW_SECS
    entries = _request_log.setdefault(api_key, [])
    # Drop old
    while entries and entries[0] < window_start:
        entries.pop(0)
    if len(entries) >= _MAX_REQUESTS:
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )
    entries.append(now)


def get_api_key(api_key_header: str | None = Security(API_KEY_HEADER)) -> str:
    """Validate x-api-key header and enforce per-key rate limiting."""
    if not api_key_header:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Missing API key")
    expected = os.getenv("SEARCH_API_KEY", "dev-secret-change-me")
    if api_key_header != expected:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    _rate_limit_check(api_key_header)
    return api_key_header
