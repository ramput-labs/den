"""Served System One models head to head: every record of each dev panel goes to each model's own /v1/systemone
server, and `den.metrics` scores the answers, so neither model is read through the other's code.

Each model is judged the way it ships (its server, its prompt template, its temperature) on the same pinned rows
under data/clean. Test partitions are refused: they are read once per release candidate (hard rule 1), so test
numbers come from each model's own release record instead. Predictions are written as they arrive to
`benchmarks/predictions/<model>/<panel>.jsonl`, so an interrupted run resumes where it stopped.

    uv run den serve --run hf:ramput-labs/den-qwen3.5-4b@v1               # :8009
    (cd kev && uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b@139fdd94...)   # :8008
    uv run python -m scripts.run_benchmarks --model den-v1=http://127.0.0.1:8009 --model kev-4b=http://127.0.0.1:8008
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx

from den.api import Json, Question, Record, parse
from den.catalog import record_path, role
from den.doctor import commit
from den.evaluate import metas
from den.fetch import digest
from den.metrics import BINS, Answer, Row, argmax, paired_bootstrap, robustness, summarize
from den.paths import CLEAN

OUT = Path("benchmarks")
PANELS = (
    "dev/transfer.jsonl",  # out of domain: QNLI, SciQ, TweetEval, PAWS, MMLU, Emotion, held-out policy rules
    "dev/breadth.jsonl",  # 14 held-out public datasets
    "dev/skills.jsonl",  # hard-v1: policies, trade-offs, probability, multi-hop, dates, judging, abstention
    "dev/devtools.jsonl",  # code review, commit type, flaky tests, safety, when2call, prompt injection
    "dev/documents.jsonl",  # CFPB complaint narratives, up to ~7k tokens
    "dev/core.jsonl",  # decision-v7: public classification, policy minimal pairs, rule structures
    "dev/probes.jsonl",  # dates and unknowable items (confidence on the unknowable is scored separately)
    "dev/semif.jsonl",  # authored if-then decisions
    "dev/binding.jsonl",  # which entity a fact binds to
)
CONFIDENT = 0.9  # an unknowable question answered at or above this probability is a confident guess
QUESTION_FIELDS = ("type", "instructions", "criteria")


def records(path: Path) -> Iterator[tuple[int, dict[str, Json], Record]]:
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            raw = json.loads(line)
            meta = raw.get("_meta") if isinstance(raw.get("_meta"), dict) else {}
            yield n, raw, parse(raw, str(meta.get("id") or f"{path.stem}:{n}"))


def request(raw: dict[str, Json]) -> dict[str, Json]:
    """The /v1/systemone request a client would send: the state and each question, without labels or metadata."""
    questions = raw["questions"]
    assert isinstance(questions, dict)
    asked: dict[str, Json] = {}
    for qid, q in questions.items():
        assert isinstance(q, dict)
        asked[qid] = {k: q[k] for k in QUESTION_FIELDS if k in q}
    return {"state": raw["state"], "questions": asked}


def probabilities(question: Question, answer: dict[str, Any]) -> list[float]:
    """One answer's probabilities in the record's option order (den.api's `keys`), renormalized after rounding."""
    match question.type:
        case "noul":
            p = [1.0 - float(answer["noul"]), float(answer["noul"])]
        case _:
            given = answer["probabilities"]
            p = [float(given[k]) for k in question.keys]
    total = sum(p)
    return [x / total for x in p] if total > 0 else [1.0 / len(p)] * len(p)


def predict(client: httpx.Client, url: str, panel: str, out: Path) -> None:
    """Every record of `panel` through one server, appended to `out`; records already there are skipped."""
    done = {json.loads(line)["line"] for line in out.read_text().splitlines()} if out.exists() else set()
    out.parent.mkdir(parents=True, exist_ok=True)
    todo = [(n, raw, rec) for n, raw, rec in records(CLEAN / panel) if n not in done]
    started = time.perf_counter()
    with out.open("a", encoding="utf-8") as f:
        for i, (n, raw, record) in enumerate(todo, 1):
            t = time.perf_counter()
            response = client.post(f"{url}/v1/systemone", json=request(raw))
            ms = 1000 * (time.perf_counter() - t)
            if response.status_code != 200:
                raise SystemExit(f"{url} on {panel}:{n}: {response.status_code} {response.text[:300]}")
            body = response.json()
            row = {
                "line": n,
                "record": record.id,
                "ms": round(ms, 1),
                "tokens": body.get("usage", {}).get("input_tokens"),
                "probs": {
                    q.id: [round(x, 6) for x in probabilities(q, body["answers"][q.id])] for q in record.questions
                },
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 100 == 0 or i == len(todo):
                rate = i / (time.perf_counter() - started)
                print(f"  {panel}: {i}/{len(todo)} records, {rate:.1f}/s, eta {(len(todo) - i) / rate / 60:.1f} min")


def score(panel: str, predictions: Path) -> tuple[dict[str, Any], dict[tuple[str, str], tuple[str, bool]]]:
    """`metrics.summarize` over one model's answers, Kev's robustness checks from each record's `_meta` (its `clean`
    numbers are the ones Kev's cards print), confident guesses on unknowable items, and latency; and each hard
    question's (record, correct), for pairing."""
    rows = {r["line"]: r for r in map(json.loads, predictions.read_text().splitlines())}
    meta = metas(CLEAN / panel)
    answers: list[Answer] = []
    checks: list[Row] = []
    correct: dict[tuple[str, str], tuple[str, bool]] = {}
    unknowable = confident = 0
    bins = [[0.0, 0.0, 0.0] for _ in range(BINS)]  # per confidence bin: sum of confidence, correct, questions
    for n, _, record in records(CLEAN / panel):
        for q in record.questions:
            a = Answer(rows[n]["probs"][q.id], q.label, q.type, q.source, q.target, q.teacher)
            answers.append(a)
            m = meta.get(record.id, {})
            checks.append(Row(a, record.id, q.id, q.keys, variant=_text(m, "variant") or "clean",
                              parent=_text(m, "parent_id"), pair=_text(m, "pair_id"), sibling=_text(m, "sibling"),
                              control=_text(m, "control_id"), origin=_text(m, "source") or ""))  # fmt: skip
            if a.unknowable:
                unknowable += 1
                confident += max(a.probs) >= CONFIDENT
            else:
                top = argmax(a.probs)
                correct[(record.id, q.id)] = (record.id, top == q.label)
                cell = bins[min(int(a.probs[top] * BINS), BINS - 1)]
                cell[0] += a.probs[top]
                cell[1] += top == q.label
                cell[2] += 1
    result = summarize(answers)
    if extra := robustness(checks):
        result["robustness"] = extra
    if unknowable:
        result["unknowable_questions"] = unknowable
        result[f"unknowable_answered_at_{CONFIDENT}"] = confident / unknowable
    result["reliability"] = bins
    ms = sorted(r["ms"] for r in rows.values())
    per_question = sorted(r["ms"] / len(r["probs"]) for r in rows.values())
    result["latency_ms"] = {"request_p50": ms[len(ms) // 2], "request_p95": ms[int(0.95 * (len(ms) - 1))],
                            "question_p50": per_question[len(per_question) // 2]}  # fmt: skip
    return result, correct


def _text(meta: dict[str, Json], key: str) -> str | None:
    return str(value) if (value := meta.get(key)) else None


def paired(a: dict[tuple[str, str], tuple[str, bool]], b: dict[tuple[str, str], tuple[str, bool]]) -> dict[str, float]:
    """b's accuracy minus a's over the questions both answered, resampling whole records."""
    keys = sorted(a.keys() & b.keys())
    return paired_bootstrap([a[k][0] for k in keys], [float(a[k][1]) for k in keys], [float(b[k][1]) for k in keys])


def machine() -> str:
    """The chip (`Apple M5 Pro`) where the OS says, else the platform."""
    if platform.system() == "Darwin":
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if brand.returncode == 0 and brand.stdout.strip():
            return f"{brand.stdout.strip()}, macOS {platform.mac_ver()[0]}"
    return f"{platform.system()} {platform.machine()}"


def models(client: httpx.Client, url: str) -> Json:
    try:
        reply: Json = client.get(f"{url}/v1/models").json()
        return reply
    except (httpx.HTTPError, ValueError) as e:
        return {"error": str(e)}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="run_benchmarks", description=__doc__.split("\n\n")[0])
    p.add_argument("--model", action="append", required=True, metavar="NAME=URL", help="a served model; repeat")
    p.add_argument("--baseline", help="the model each other one is paired against (default: the last --model)")
    p.add_argument("--key", help="bearer key, if the servers require one")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("panels", nargs="*", default=list(PANELS), help="files under data/clean (dev only)")
    args = p.parse_args(argv)
    served = dict(m.split("=", 1) for m in args.model)
    baseline = args.baseline or list(served)[-1]
    panels = [record_path(x) for x in args.panels]
    if refused := [x for x in panels if role(x) != "dev"]:
        raise SystemExit(f"benchmarks read dev partitions only (test is read once per release candidate): {refused}")

    headers = {"authorization": f"Bearer {args.key}"} if args.key else {}
    with httpx.Client(timeout=600, headers=headers) as client:
        for name, url in served.items():
            for panel in panels:
                print(f"{name}: {panel}", flush=True)
                predict(client, url, panel, args.out / "predictions" / name / panel)
        servers = {name: models(client, url) for name, url in served.items()}

    results: dict[str, Any] = {name: {} for name in served}
    correct: dict[str, dict[str, dict[tuple[str, str], tuple[str, bool]]]] = {name: {} for name in served}
    for name in served:
        for panel in panels:
            results[name][panel], correct[name][panel] = score(panel, args.out / "predictions" / name / panel)
    comparisons: dict[str, Any] = {}
    for name in served:
        if name != baseline:
            each = {x: paired(correct[baseline][x], correct[name][x]) for x in panels}
            pooled_a = {(x, *k): v for x in panels for k, v in correct[baseline][x].items()}
            pooled_b = {(x, *k): v for x in panels for k, v in correct[name][x].items()}
            keys = sorted(pooled_a.keys() & pooled_b.keys())
            each["pooled"] = paired_bootstrap(
                [f"{k[0]}:{pooled_a[k][0]}" for k in keys],
                [float(pooled_a[k][1]) for k in keys],
                [float(pooled_b[k][1]) for k in keys],
            )
            comparisons[f"{name} vs {baseline}"] = each
    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": commit(),
        "machine": machine(),
        "baseline": baseline,
        "servers": {name: {"url": url, "models": servers[name]} for name, url in served.items()},
        "files": {x: digest(CLEAN / x) for x in panels},
        "results": results,
        "paired_accuracy": comparisons,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
