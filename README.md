# den

**Small System One decision models you can run and fine-tune yourself.** A model reads a document (the *state*) and
typed questions about it, and returns a calibrated probability for every option of every question, in one forward
pass, with no text generation. den follows [Kev](https://github.com/jaredpalmer/kev), the open reconstruction of
TypeSafe's Jev: a Qwen base model with a LoRA adapter and a pointer head, served behind TypeSafe's
`POST /v1/systemone`, so TypeSafe's Python SDK works against it unchanged. The goal: match each Kev model at its
size, then beat it.

## Highlights

- Yes/no (`noul`), multiple-choice (`choice`, 1–255 options) and rating (`score`) questions in one request.
- Calibrated probabilities by default: every release ships with fitted temperatures.
- Drop-in for Jev and Kev: the [TypeSafe SDK](examples/README.md) works against `den serve` as it is.
- Runs on Apple Silicon (MLX), NVIDIA (CUDA) and CPU. Fine-tunes with Unsloth + LoRA on one NVIDIA GPU.
- Honest numbers: dev for every decision, the locked test read once per release, every evaluation kept in git.

## Model

| Release | Base | Hub | Held-out dev accuracy vs Kev-4B |
|---|---|---|---|
| **den v1** | Qwen3.5-4B-Base, LoRA r16 + set head | [`ramput-labs/den-4b@v1`](https://huggingface.co/ramput-labs/den-4b/tree/v1) | tied: +0.2 points [−0.4, +0.8] over 10,459 questions |

v1 trained on exactly Kev-4B's data. It is better calibrated than Kev-4B (ECE 1.9% vs 2.9%) and ahead on reasoning
skills (+2.5) and entity binding (+2.1), behind out of domain (−2.0) and on dates/unknowable items (−1.6). The full,
paired, reproducible comparison is in [`benchmarks/`](benchmarks/README.md).

## Quick start

You need [uv](https://docs.astral.sh/uv/) and about 10 GB of disk for the 4B model.

```bash
git clone https://github.com/ramput-labs/den.git && cd den
make setup                    # Python 3.12 + this machine's runtime (MLX on Apple Silicon, CUDA on Linux)
make model-download           # den v1 into models/den-4b, once, checked against its sha256
make serve                    # http://127.0.0.1:8009/v1/systemone, from models/den-4b (no network)
```

In another terminal:

```bash
make predict                  # answers examples/request.jsonl without a server
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d @examples/request.jsonl
make example                  # the same through TypeSafe's SDK (examples/quickstart.py)
```

### Use it from Python

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009", model="den-latest")
response = client.system_one(
    state="I was charged twice. Please fix this ASAP.",
    questions={
        "billing": Noul(instructions="Is this ticket about billing?"),
        "tone": Choice(instructions="What is the customer's tone?",
                       criteria={"calm": None, "frustrated": None, "angry": None}),
        "urgency": Score(instructions="How urgent is this ticket?", criteria=["can wait", "this week", "today"]),
    },
)
print(response.nouls["billing"].noul, response.choices["tone"].choice, response.scores["urgency"].score)
```

Or in-process, no server: `den.load("hf:ramput-labs/den-4b@v1").predict({...})` returns the same response.
[`examples/`](examples/README.md) has typed responses, routing, a tool-calling gate, async and raw HTTP.

## API

`POST /v1/systemone` takes a state (text or any JSON) and named questions; `GET /v1/models` lists the model;
`GET /health` needs no key. Set `DEN_API_KEY` to require `Authorization: Bearer <key>`.

```json
{"state": "Shoes arrived late and I see two charges.",
 "questions": {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"shipping": "Delivery", "billing": "Charges"}},
               "urgent": {"type": "noul", "instructions": "Does this need a reply today?"}}}
```

Each answer holds `choice` (likeliest option), `score` (expected level) or `noul` (P(yes)), every option's
probability, and `confidence` = (p_max − 1/K)/(1 − 1/K); the response adds `model`, `usage` and `latency_ms`.

## Fine-tune

Training needs Linux and an NVIDIA GPU (driver ≥ 580). Start small, then go big:

```bash
make train-smoke              # any machine, Mac included (~35 min): every code path at toy scale
make gpu-setup                # on the GPU box: Unsloth, the model, data/, the readiness gate
make rehearse                 # any 8 GB+ card (~2.5 h): the whole pipeline on qwen3.5-0.8b
make gpu-gates gpu-timing gpu-round1 gpu-round2 gpu-eval   # the real run (H100: 5-6 h)
```

[`docs/fine-tuning.md`](docs/fine-tuning.md) walks through every step and what each gate should print;
[`docs/gpu-runbook.md`](docs/gpu-runbook.md) is the same plan for a cloud agent. To train on your own labelled
records, put them in `data/clean/train/` and continue from the last release:
`make train-next FROM=v1 OUT=runs/v2 DATA="train/<yours>.jsonl"` ([details](docs/fine-tuning.md#training-on-your-own-data)).

## How it works

The backbone prefills `state + question + options`; the head scores each option's span against the question's final
token; a softmax and a fitted temperature give the probabilities. Each question is read with the state alone. Round 1
of training is Kev-4B's four-stage recipe; round 2 adds 18 public sources. [`docs/architecture.md`](docs/architecture.md)
covers the head, the prompt, the backends, experiment modes and versioning; [`docs/data.md`](docs/data.md) covers the
data: sources, layout, leakage rules, the audit and licences.

## Development

```bash
make check                    # ruff, mypy --strict, pytest, claims vs evidence: run before every commit
make help                     # every target, grouped
make test-model               # slow tests on the real 4B weights
```

| Path | What |
|---|---|
| `den/` | the package: one flat module per concern (`api`, `prompt`, `head`, `lora`, `train`, `evaluate`, `serve`, ...); `den <command>` is `den/cli.py` |
| `scripts/` | operational tools beside the library: data mirror, breadth builder, GPU checks, benchmarks |
| `tests/` | one test file per area |
| `examples/` | TypeSafe SDK examples and the sample request |
| `benchmarks/` | den vs Kev-4B: results, per-question predictions, plots |
| `docs/` | fine-tuning guide, agent runbook, architecture, data |
| `releases/`, `reports/` | release records and evaluation evidence (committed, no weights) |
| `locks/`, `licences/` | sha256 of every placed file; canonical licence texts for the data |

[`CLAUDE.md`](CLAUDE.md) holds the rules every change follows (never train on test, determinism, pinned files) and is
the best map of the code. Use `uv run` for everything; Python 3.12, line length 120, full type annotations.

## License

den's code is [MIT](LICENSE). Model weights keep their base model's licence (Qwen3.5: Apache-2.0); each dataset keeps
its own, recorded in [`docs/data.md`](docs/data.md#licences) and shipped with the data.
