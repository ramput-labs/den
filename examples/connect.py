"""One place for where the examples' server is: `den serve` on this machine unless TYPESAFE_BASE_URL says otherwise.

The key is DEN_API_KEY when the server was started with one; any non-empty key works for a server without one (the
SDK refuses an empty key).
"""

from __future__ import annotations

import os

from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient

BASE_URL = os.environ.get("TYPESAFE_BASE_URL") or "http://127.0.0.1:8009"
API_KEY = os.environ.get("DEN_API_KEY") or "local"
MODEL = "den-latest"  # `den serve` answers with any model name and echoes it back
TIMEOUT = 60.0  # the SDK's 10 s default is short for a first request or a long state on a laptop


def client() -> TypeSafeClient:
    return TypeSafeClient(api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=TIMEOUT)


def async_client() -> AsyncTypeSafeClient:
    return AsyncTypeSafeClient(api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=TIMEOUT)
