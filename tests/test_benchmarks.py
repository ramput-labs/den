"""scripts/run_benchmarks.py and scripts/plot_benchmarks.py: requests carry no labels, answers map back to each
record's option order, test partitions are refused, and a run against two servers yields results, plots and a README."""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from test_serve import Fake

from den.api import parse
from den.serve import handler
from scripts import plot_benchmarks, run_benchmarks

LABELLED: dict[str, Any] = {
    "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"returns": "Exchanges", "shipping": "Delivery", "billing": "Charges"},
            "label": "billing",
            "src": "core",
        },
        "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?", "label": True},
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
            "label": 2,
            "target": {"1": 1, "2": 1},
        },
    },
    "_meta": {"id": "r1", "variant": "clean"},
}


def test_requests_carry_no_labels_or_metadata() -> None:
    sent = run_benchmarks.request(LABELLED)
    assert set(sent) == {"state", "questions"}
    questions = sent["questions"]
    assert isinstance(questions, dict)
    assert all(isinstance(q, dict) and set(q) <= {"type", "instructions", "criteria"} for q in questions.values())


def test_answers_map_to_the_records_option_order() -> None:
    department, escalate, frustration = parse(LABELLED, "r1").questions
    choice = {"probabilities": {"billing": 0.2, "returns": 0.5, "shipping": 0.3}}
    assert run_benchmarks.probabilities(department, choice) == [0.5, 0.3, 0.2]  # returns, shipping, billing
    assert run_benchmarks.probabilities(escalate, {"noul": 0.8}) == pytest.approx([0.2, 0.8])
    score = {"probabilities": {"2": 0.6, "0": 0.1, "1": 0.2}}  # rounded: renormalized
    assert run_benchmarks.probabilities(frustration, score) == pytest.approx([1 / 9, 2 / 9, 6 / 9])


def test_test_partitions_are_refused() -> None:
    with pytest.raises(SystemExit, match="dev partitions only"):
        run_benchmarks.main(["--model", "a=http://127.0.0.1:9", "test/core.jsonl"])


@pytest.fixture
def servers() -> Iterator[list[str]]:
    started = [ThreadingHTTPServer(("127.0.0.1", 0), handler(Fake(), name, None)) for name in ("den", "kev")]  # type: ignore[arg-type]
    for s in started:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    yield [f"http://127.0.0.1:{s.server_address[1]}" for s in started]
    for s in started:
        s.shutdown()


def test_a_run_writes_results_plots_and_readme(
    servers: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean = tmp_path / "clean"
    (clean / "dev").mkdir(parents=True)
    rows = [{**LABELLED, "_meta": {"id": f"r{i}", "variant": "clean"}} for i in range(30)]
    (clean / "dev/semif.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(run_benchmarks, "CLEAN", clean)
    out = tmp_path / "bench"
    den, kev = servers
    args = ["--model", f"kev-4b={kev}", "--model", f"den-v1={den}", "--baseline", "kev-4b", "--out", str(out)]
    assert run_benchmarks.main([*args, "dev/semif.jsonl"]) == 0
    report = json.loads((out / "results.json").read_text())
    semif = report["results"]["den-v1"]["dev/semif.jsonl"]
    assert semif["questions"] == 60 and semif["unknowable_questions"] == 30  # frustration's target is unknowable
    assert sum(cell[2] for cell in semif["reliability"]) == 60
    assert report["paired_accuracy"]["den-v1 vs kev-4b"]["pooled"]["diff"] == 0  # the same fake behind both
    assert len((out / "predictions/den-v1/dev/semif.jsonl").read_text().splitlines()) == 30
    assert run_benchmarks.main([*args, "dev/semif.jsonl"]) == 0  # resumes: nothing left to ask
    assert len((out / "predictions/den-v1/dev/semif.jsonl").read_text().splitlines()) == 30

    shutil.copy(plot_benchmarks.OUT / "kev-4b-card.json", out)
    assert plot_benchmarks.main(["--out", str(out)]) == 0
    assert {p.name for p in out.glob("*.png")} == {
        f"{name}{suffix}.png" for name in ("accuracy", "delta", "calibration", "test") for suffix in ("", "-dark")
    }
    readme = (out / "README.md").read_text()
    assert "+0.0 [+0.0, +0.0]" in readme and "Authored if-then rules" in readme and "Long documents" in readme
