# Examples: your den model through TypeSafe's SDK

`den serve` speaks TypeSafe's `POST /v1/systemone` and `GET /v1/models`, so
[`typesafe-sdk`](https://pypi.org/project/typesafe-sdk/) works against a den model as it is. Point `base_url` at
your server.

## 1. Serve the model

```bash
uv run den serve --run hf:ramput-labs/den-qwen3.5-4b@v1 --release-date 2026-10-10   # merged/ + head, ~9.4 GB once
DEN_API_KEY=secret uv run den serve --run ... --host 0.0.0.0                         # require a bearer key
```

| Route | Body | Returns |
|---|---|---|
| `POST /v1/systemone` | `{"state", "model", "questions": {name: {type, instructions, criteria}}}` | `{"model", "answers", "usage", "latency_ms"}` |
| `GET /v1/models` | | `{"models": [{"name", "description", "release_date"}], "data": [...]}` |
| `GET /health` | | `{"status": "ok"}` (never needs the key) |

Errors are `{"error": "..."}` with status 400, 401, 404 or 500, which the SDK raises as `TypeSafeBadRequestError`,
`TypeSafeAuthenticationError`, `TypeSafeNotFoundError` or `TypeSafeInternalServerError`. The model name in a request
is echoed back, and any name is accepted. The server reads one request at a time.

## 2. Call it

```bash
uv pip install typesafe-sdk          # already in den's dev group
```

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009", model="den-latest")
response = client.system_one(
    state="I was charged twice. Please fix this ASAP.",
    questions={
        "billing": Noul(instructions="Is this ticket about billing?"),
        "tone": Choice(
            instructions="What is the customer's tone?",
            criteria={"calm": None, "frustrated": None, "angry": None},
        ),
        "urgency": Score(instructions="How urgent is this ticket?", criteria=["can wait", "this week", "today"]),
    },
)
response.nouls["billing"].noul  # 0.93: P(yes)
response.choices["tone"].choice  # "frustrated" (0.52; "angry" 0.46)
response.scores["urgency"].score  # 1.43: the expected level, 0..2
```

These are the v1 model's real answers. Use any non-empty `api_key` for a server started without `DEN_API_KEY`
(the SDK refuses an empty key), and the server's key otherwise. The SDK's default timeout is 10 s. `connect.py`
raises it to 60 s for laptops.

**Typed fields:** pass `response_model=` a subclass of `SystemOneResponse` with one field per question
(`billing: NoulAnswer`). Each answer is then validated into its own typed attribute. Narrowing a choice to a
`Literal` (`class ToneAnswer(ChoiceAnswer): choice: Literal[...]`) makes mypy check your `match` statements and
makes pydantic reject an answer outside that set. `typed_response.py` and `triage.py` show both.

**Instructions are required by den.** TypeSafe's API lets a choice or score question leave out `instructions`. den
refuses that with a 400 (`<name>: empty instructions`), because the model was only trained on questions that have
them.

## 3. Examples

```bash
export PYTHONPATH=examples            # plus DEN_API_KEY=... if the server has one; TYPESAFE_BASE_URL for another host
uv run python examples/quickstart.py
```

| Script | Shows |
|---|---|
| `quickstart.py` | the three question types, `nouls` / `choices` / `scores` |
| `typed_response.py` | `response_model=`: answers as typed, validated fields |
| `triage.py` | `Literal` teams, an exhaustive `match`, routing on a calibrated probability threshold |
| `tool_gate.py` | a JSON-object state and dictionary questions: should an agent call a tool, and which |
| `async_client.py` | `AsyncTypeSafeClient`, `models.list()`, a typed 400 |
| `raw_http.py` | the wire format with the standard library only, to port to any language |

`tests/test_examples.py` runs the SDK and every script against the real `den serve` handler (with a fake model),
so a server change that breaks the SDK fails `make check`.
