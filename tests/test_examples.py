"""TypeSafe's SDK (typesafe-sdk) and every script in examples/ against the real `den serve` handler, with
test_serve's fake model: the published client must work against our server unchanged."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer

import async_client
import pytest
import quickstart
import raw_http
import tool_gate
import triage
import typed_response
from pydantic import TypeAdapter
from test_serve import Fake
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    Noul,
    Questions,
    RetryPolicy,
    Score,
    TypeSafeAPIResponseValidationError,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeClient,
)

from den.runtime import respond
from den.serve import handler

NO_RETRY = RetryPolicy(max_retries=0)


@pytest.fixture
def url() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler(Fake(), "den-test", "secret", "2026-10-10"))  # type: ignore[arg-type]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture
def client(url: str) -> Iterator[TypeSafeClient]:
    with TypeSafeClient(api_key="secret", base_url=url, model="kev-latest", retry=NO_RETRY) as c:
        yield c


def test_the_sdk_reads_den_serve_answers(client: TypeSafeClient) -> None:
    questions: Questions = {
        "billing": Noul(instructions="Is this ticket about billing?"),
        "tone": Choice(instructions="Tone?", criteria={"calm": None, "frustrated": None, "angry": None}),
        "urgency": Score(instructions="How urgent?", criteria=["can wait", "this week", "today"]),
    }
    r = client.system_one(state="I was charged twice.", questions=questions)
    wire = json.loads(TypeAdapter(Questions).dump_json(questions))
    expected = respond(Fake(), {"state": "I was charged twice.", "questions": wire})  # type: ignore[arg-type]
    assert r.model == "kev-latest" and r.usage.input_tokens == 42 and r.usage.output_tokens == 0
    assert r.nouls["billing"].noul == expected["answers"]["billing"]["noul"]
    assert r.choices["tone"].choice == "frustrated"
    assert r.choices["tone"].probabilities == expected["answers"]["tone"]["probabilities"]
    assert r.scores["urgency"].score == expected["answers"]["urgency"]["score"]
    assert r.scores["urgency"].legend == {0: "can wait", 1: "this week", 2: "today"}


def test_models_list_has_typesafes_shape(client: TypeSafeClient) -> None:
    (model,) = client.models.list().models
    assert model.name == "den-test" and model.release_date == "2026-10-10"
    assert model.description.startswith("qwen3.5-4b + LoRA r16 + set head")


def test_errors_arrive_as_the_sdks_types(url: str, client: TypeSafeClient) -> None:
    question = {"q": Noul(instructions="?")}
    with TypeSafeClient(api_key="wrong", base_url=url, retry=NO_RETRY) as c, pytest.raises(TypeSafeAuthenticationError):
        c.system_one(state="s", questions=question)
    with pytest.raises(TypeSafeBadRequestError, match="empty state"):
        client.system_one(state="", questions=question)
    with pytest.raises(TypeSafeAPIResponseValidationError):  # the fake picks option 2, outside typed_response.Tone
        client.system_one(
            state="s",
            questions={
                **typed_response.QUESTIONS,
                "tone": Choice(instructions="Tone?", criteria={"calm": None, "bored": None}),
            },
            response_model=typed_response.Ticket,
        )


def test_every_example_runs(
    url: str, client: TypeSafeClient, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    quickstart.main(client)
    typed_response.main(client)
    triage.main(client)
    tool_gate.main(client)
    monkeypatch.setenv("DEN_API_KEY", "secret")
    raw_http.main(url)

    async def run_async() -> None:
        async with AsyncTypeSafeClient(api_key="secret", base_url=url, retry=NO_RETRY) as c:
            await async_client.main(c)

    asyncio.run(run_async())
    out = capsys.readouterr().out
    assert "tone:    frustrated" in out and "serving den-test (2026-10-10)" in out
    assert "refused as expected: 400 {'error': 'empty state'}" in out
    assert sum(line.startswith(("finance", "logistics", "engineering", "human")) for line in out.splitlines()) == 4
