"""The three question types through TypeSafe's SDK, against your own model.

uv run den serve --run hf:ramput-labs/den-qwen3.5-4b@v1      # in another terminal
uv run python examples/quickstart.py
"""

from __future__ import annotations

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient


def main(client: TypeSafeClient) -> None:
    response = client.system_one(
        state="I was charged twice. Please fix this ASAP.",
        questions={
            "billing": Noul(instructions="Is this ticket about billing?"),
            "tone": Choice(
                instructions="What is the customer's tone?",
                criteria={"calm": None, "frustrated": None, "angry": None},
            ),
            "urgency": Score(
                instructions="How urgent is this ticket?",
                criteria=["can wait", "this week", "today"],
            ),
        },
    )
    print(response.nouls["billing"].noul)  # P(yes)
    print(response.choices["tone"].choice, response.choices["tone"].probabilities)
    print(response.scores["urgency"].score, response.scores["urgency"].legend)  # expected level, 0..2
    print(f"{response.model}: {response.usage.input_tokens} tokens")


if __name__ == "__main__":
    with TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009", model="den-latest") as c:
        main(c)
