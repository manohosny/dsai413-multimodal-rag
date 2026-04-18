"""Shared retry helper for Gemini API calls with exponential backoff."""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_RETRIES = 5
BASE_DELAY = 2.0  # seconds


def with_retry(fn: Callable[[], T]) -> T:
    """Call fn() with exponential backoff on transient errors.

    Catches: ReadTimeout, ConnectTimeout, ConnectError, RemoteProtocolError,
    429 rate-limit, 503 service-unavailable, RESOURCE_EXHAUSTED.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as exc:
            exc_name = type(exc).__name__
            is_transient = (
                "ReadTimeout" in exc_name
                or "ConnectTimeout" in exc_name
                or "ConnectError" in exc_name
                or "RemoteProtocolError" in exc_name
                or "429" in str(exc)
                or "503" in str(exc)
                or "RESOURCE_EXHAUSTED" in str(exc)
            )
            if not is_transient or attempt == MAX_RETRIES - 1:
                raise
            delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                "Transient error on attempt %d/%d (%s), retrying in %.1fs",
                attempt + 1, MAX_RETRIES, exc_name, delay,
            )
            time.sleep(delay)
    raise RuntimeError("Unreachable")  # pragma: no cover
