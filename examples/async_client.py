"""`AsyncTypeSafeClient`, for code that already runs in an event loop (a web app, a queue worker).

`den serve` reads one request at a time, so these requests queue on the server: async keeps your loop free while
they wait; it does not make the model faster. Errors are typed: a refused request raises `TypeSafeBadRequestError`
with the server's reason.

    uv run python examples/async_client.py
"""

from __future__ import annotations

import asyncio

from connect import async_client
from typesafe_sdk import AsyncTypeSafeClient, Noul, Questions, RetryPolicy, Score, TypeSafeBadRequestError

QUESTIONS: Questions = {
    "urgent": Noul(instructions="Does this message need a reply today?"),
    "sentiment": Score(instructions="What is the overall sentiment?", criteria=["Negative", "Neutral", "Positive"]),
}
INBOX = [
    "Server is down for all EU customers since 09:00, please advise ASAP.",
    "Thanks for the quick fix last week, everything works great now!",
    "Could you send the Q3 invoice when you get a chance?",
]


async def main(client: AsyncTypeSafeClient) -> None:
    for model in (await client.models.list()).models:
        print(f"serving {model.name} ({model.release_date}): {model.description}")

    responses = await asyncio.gather(*(client.system_one(state=text, questions=QUESTIONS) for text in INBOX))
    for text, r in zip(INBOX, responses, strict=True):
        print(f"urgent={r.nouls['urgent'].noul:.2f} sentiment={r.scores['sentiment'].score:.2f}  {text}")

    try:
        await client.system_one(state="", questions=QUESTIONS, retry=RetryPolicy(max_retries=0))
    except TypeSafeBadRequestError as e:
        print(f"refused as expected: {e.status} {e.body}")


async def run() -> None:
    async with async_client() as client:
        await main(client)


if __name__ == "__main__":
    asyncio.run(run())
