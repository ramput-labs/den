"""Route tickets on the model's calibrated probabilities, and hand the unsure ones to a person.

The team names are a `Literal`, so the `match` is checked as exhaustive by mypy, and an answer outside them fails
validation instead of falling through.

    uv run python examples/triage.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never

from connect import client
from typesafe_sdk import Choice, ChoiceAnswer, Noul, NoulAnswer, Questions, SystemOneResponse, TypeSafeClient

type Team = Literal["billing", "shipping", "returns", "technical"]


class TeamAnswer(ChoiceAnswer):
    choice: Team


class Triage(SystemOneResponse):
    team: TeamAnswer
    churn: NoulAnswer


QUESTIONS: Questions = {
    "team": Choice(
        instructions="Which team should handle this ticket?",
        criteria={
            "billing": "Charges, refunds of charges, invoices, payment methods",
            "shipping": "Delivery times, tracking, lost parcels",
            "returns": "Exchanges, wrong or damaged items, sending items back",
            "technical": "The website or app not working, login problems",
        },
    ),
    "churn": Noul(
        instructions="Is the customer threatening to leave or cancel?",
        criteria={"true": "they say they will leave or cancel"},
    ),
}
TICKETS = [
    "I was charged twice for order #1182. Please refund one of them.",
    "The tracking page has said 'label created' for nine days now.",
    "The app logs me out every time I open the cart. Fix this or I'm cancelling my membership.",
    "It came, but it's not what I expected.",
]
SURE = 0.80  # route automatically at or above this probability; calibration makes it mean ~80% right


@dataclass(frozen=True, slots=True)
class Route:
    queue: str
    p: float
    churn: float


def route(client: TypeSafeClient, ticket: str) -> Route:
    r = client.system_one(state=ticket, questions=QUESTIONS, response_model=Triage)
    p, risk = r.team.probabilities[r.team.choice], r.churn.noul
    if p < SURE:
        return Route("human-review", p, risk)
    match r.team.choice:
        case "billing":
            queue = "finance"
        case "shipping" | "returns":
            queue = "logistics"
        case "technical":
            queue = "engineering"
        case _:
            assert_never(r.team.choice)
    return Route(f"{queue}{'+retention' if risk >= 0.5 else ''}", p, risk)


def main(client: TypeSafeClient) -> None:
    for ticket in TICKETS:
        r = route(client, ticket)
        print(f"{r.queue:<22} p={r.p:.2f} churn={r.churn:.2f}  {ticket}")


if __name__ == "__main__":
    with client() as c:
        main(c)
