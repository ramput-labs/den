"""Answers as typed fields: pass `response_model=` and the SDK validates each answer into its own field.

A field named after a question gets that question's answer, so `r.billing.noul` is a float, `r.tone.choice` is
one of the declared names, and a typo is a mypy error rather than a KeyError at run time. Narrowing `choice` to a
`Literal` also makes pydantic reject an answer outside it (`TypeSafeAPIResponseValidationError`).

    uv run python examples/typed_response.py
"""

from __future__ import annotations

from typing import Literal

from connect import client
from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Questions,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    TypeSafeClient,
)

type Tone = Literal["calm", "frustrated", "angry"]


class ToneAnswer(ChoiceAnswer):
    choice: Tone


class Ticket(SystemOneResponse):
    billing: NoulAnswer
    tone: ToneAnswer
    urgency: ScoreAnswer


QUESTIONS: Questions = {
    "billing": Noul(instructions="Is this ticket about billing?"),
    "tone": Choice(
        instructions="What is the customer's tone?",
        criteria={"calm": None, "frustrated": None, "angry": None},
    ),
    "urgency": Score(instructions="How urgent is this ticket?", criteria=["can wait", "this week", "today"]),
}


def read(client: TypeSafeClient, text: str) -> Ticket:
    return client.system_one(state=text, questions=QUESTIONS, response_model=Ticket)


def main(client: TypeSafeClient) -> None:
    ticket = read(client, "I was charged twice. Please fix this ASAP.")
    print(f"billing: {ticket.billing.noul:.2f}")
    print(f"tone:    {ticket.tone.choice} ({ticket.tone.probabilities[ticket.tone.choice]:.2f})")
    print(f"urgency: {ticket.urgency.score:.2f} of {max(ticket.urgency.legend)}")


if __name__ == "__main__":
    with client() as c:
        main(c)
