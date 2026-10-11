"""A cheap gate in front of an agent: should it call a tool now, and which one?

The state need not be a string: a JSON object goes in as data and the server renders it to indented text, the same
way the model saw it in training. Questions can be plain dictionaries too (the SDK's `QuestionModel` TypedDicts).

    uv run python examples/tool_gate.py
"""

from __future__ import annotations

from connect import client
from typesafe_sdk import JSONContent, Questions, TypeSafeClient

TOOLS: dict[str, JSONContent] = {
    "get_weather": "Current weather for a city. Arguments: city",
    "convert_currency": "Convert an amount between currencies. Arguments: amount, from, to",
    "book_table": "Reserve a restaurant table. Arguments: restaurant, time, people",
}
QUESTIONS: Questions = {
    "call": {
        "type": "noul",
        "instructions": "Should the assistant call one of the available tools to answer the user's last message?",
        "criteria": {
            "true": "a tool is needed and fits the request",
            "false": "answer directly, or the request is outside every tool",
        },
    },
    "ready": {"type": "noul", "instructions": "Does the user's last message give every argument the tool needs?"},
    "tool": {"type": "choice", "instructions": "Which tool fits the user's last message best?", "criteria": TOOLS},
}


def gate(client: TypeSafeClient, conversation: list[JSONContent]) -> str:
    r = client.system_one(state={"tools": TOOLS, "conversation": conversation}, questions=QUESTIONS)
    call, ready, tool = r.nouls["call"].noul, r.nouls["ready"].noul, r.choices["tool"]
    if call < 0.5:
        return f"answer directly (P(call) = {call:.2f})"
    if ready < 0.5:
        return f"ask for missing arguments of {tool.choice} (P(ready) = {ready:.2f})"
    return f"call {tool.choice} (p = {tool.probabilities[tool.choice]:.2f})"


def main(client: TypeSafeClient) -> None:
    conversations: list[list[JSONContent]] = [
        [{"user": "How much is 250 euros in yen?"}],
        [{"user": "Can you book a table for us?"}],
        [{"user": "Write me a haiku about autumn."}],
    ]
    for conversation in conversations:
        print(f"{gate(client, conversation):<55} <- {conversation[-1]}")


if __name__ == "__main__":
    with client() as c:
        main(c)
