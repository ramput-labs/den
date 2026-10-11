"""`runtime.rebuild`: a run published without merged/ gets it back from its adapter and the pinned base, byte for byte
the merged/ that was evaluated (integrity.json's sha256), or not at all."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from den import runtime
from den.fetch import digest
from den.integrity import write
from den.lora import merge

Q = "model.language_model.layers.{}.self_attn.q_proj.weight"


def published(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny base, and a run as `den publish` uploads it: adapter, head files, integrity.json with merged/'s sha256,
    and no merged/."""
    torch.manual_seed(0)
    base, run = tmp_path / "base", tmp_path / "run"
    base.mkdir()
    run.mkdir()
    save_file({Q.format(i): torch.randn(8, 8).to(torch.bfloat16) for i in range(2)}, base / "model.safetensors")
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": {"hidden_size": 8}}))
    adapter = {f"base_model.model.model.language_model.layers.{i}.self_attn.q_proj.lora_{ab}.weight": torch.randn(*s)
               for i in range(2) for ab, s in (("A", (2, 8)), ("B", (8, 2)))}  # fmt: skip
    save_file(adapter, run / "adapter_model.safetensors")
    (run / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
    merge(base, run, run / "merged")
    (run / "integrity.json").write_text(json.dumps({"files": {f"merged/{f.name}": digest(f)
                                                             for f in (run / "merged").iterdir()}}))  # fmt: skip
    shutil.rmtree(run / "merged")
    return run, base


@pytest.fixture
def models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(runtime, "MODELS", tmp_path / "models")
    return tmp_path / "models"


def test_rebuild_matches_the_evaluated_merge_and_is_built_once(
    tmp_path: Path, models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, base = published(tmp_path)
    monkeypatch.setattr(runtime, "base_weights", lambda key: base)
    out = runtime.rebuild(run, "tiny")
    recorded = json.loads((run / "integrity.json").read_text())["files"]
    assert out.parent == models / "merged" and (out / runtime.VERIFIED).is_file()
    assert {f"merged/{f.name}": digest(f) for f in out.iterdir() if not f.name.startswith(".")} == recorded

    def merge_again(*_: object) -> int:
        raise AssertionError("merged twice")

    monkeypatch.setattr(runtime, "merge", merge_again)
    assert runtime.rebuild(run, "tiny") == out  # the verified copy is reused


def test_a_rebuild_that_differs_from_the_evaluated_merge_is_refused(
    tmp_path: Path, models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, base = published(tmp_path)
    monkeypatch.setattr(runtime, "base_weights", lambda key: base)
    report = json.loads((run / "integrity.json").read_text())
    report["files"]["merged/model.safetensors"] = "0" * 64  # e.g. v1's TF32 merge, or another base
    (run / "integrity.json").write_text(json.dumps(report))
    with pytest.raises(SystemExit, match="not the evaluated one"):
        runtime.rebuild(run, "tiny")
    assert not any((models / "merged").iterdir())  # nothing half-built or unverified is left to load


def test_a_run_with_neither_merged_nor_its_hashes_is_refused(tmp_path: Path, models: Path) -> None:
    run, _ = published(tmp_path)
    (run / "integrity.json").write_text(json.dumps({"files": {}}))
    with pytest.raises(SystemExit, match="no merged/"):
        runtime.rebuild(run, "tiny")


def test_a_trained_run_passes_integrity_and_rebuilds_to_the_same_bytes(
    tmp_path: Path, models: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole path: merged/ as training writes it, integrity.json as check-run writes it, then the server's
    rebuild from the adapter alone gives every merged/ file its recorded sha256."""
    from den.head import HeadConfig, make_head, save_head

    run, base = published(tmp_path)
    merge(base, run, run / "merged")
    save_head(run, make_head(8, HeadConfig(kind="set", dim=8, heads=2)), base="tiny", lora=2, temperature=1.0)
    (run / "run.json").write_text(json.dumps({"base": "tiny", "lora": {"rank": 2, "modules": 2}}))
    assert write(run, base)["ok"]
    shutil.rmtree(run / "merged")
    monkeypatch.setattr(runtime, "base_weights", lambda key: base)
    assert (runtime.rebuild(run, "tiny") / "model.safetensors").is_file()
