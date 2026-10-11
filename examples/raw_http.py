"""The wire format with no SDK: one JSON POST, standard library only. Port this to any language.

uv run python examples/raw_http.py [http://127.0.0.1:8009]
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Any

REQUEST = {
    "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
    "questions": {
        "department": {
            "type": "choice",  # criteria: option name -> description (or null)
            "instructions": "Which team should handle this?",
            "criteria": {"returns": "Exchanges, refunds", "shipping": "Delivery, delays", "billing": "Charges"},
        },
        "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},  # criteria optional
        "frustration": {
            "type": "score",  # criteria: levels, lowest first
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}


def post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/systemone", data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    if key := os.environ.get("DEN_API_KEY"):
        request.add_header("authorization", f"Bearer {key}")
    with urllib.request.urlopen(request) as response:  # a 4xx raises HTTPError; its body is {"error": "..."}
        reply: dict[str, Any] = json.loads(response.read())
        return reply


def main(url: str) -> None:
    print(json.dumps(post(url, REQUEST), indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SYSTEMONE_URL", "http://127.0.0.1:8009"))
