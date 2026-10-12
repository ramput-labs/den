from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import huggingface_hub
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from den.fetch import digest
from scripts import download_model


def sharded(root: Path) -> dict[str, torch.Tensor]:
    """A run with `merged/` in two indexed shards (bf16 and fp32 tensors) and a head; returns every tensor."""
    merged = root / "merged"
    merged.mkdir(parents=True)
    first = {"a": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3), "b": torch.ones(2)}
    second = {"c": torch.full((4,), 0.5, dtype=torch.bfloat16)}
    save_file(first, merged / "model-00001-of-00002.safetensors", metadata={"format": "pt"})
    save_file(second, merged / "model-00002-of-00002.safetensors", metadata={"format": "pt"})
    index = {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00001-of-00002.safetensors",
             "c": "model-00002-of-00002.safetensors"}}  # fmt: skip
    (merged / download_model.INDEX).write_text(json.dumps(index))
    (merged / "config.json").write_text("{}")
    (root / "head.json").write_text("{}")
    (root / "head.safetensors").write_bytes(b"head")
    return first | second


def hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}


def test_join_keeps_every_tensor_bit_for_bit(tmp_path: Path) -> None:
    tensors = sharded(tmp_path)
    download_model.join(tmp_path / "merged")
    assert sorted(p.name for p in (tmp_path / "merged").iterdir()) == ["config.json", download_model.WEIGHTS]
    with safe_open(tmp_path / "merged" / download_model.WEIGHTS, "pt") as f:
        assert f.metadata() == {"format": "pt"}
        assert set(f.keys()) == set(tensors)
        for key, tensor in tensors.items():
            joined = f.get_tensor(key)
            assert joined.dtype == tensor.dtype and torch.equal(joined, tensor)


def test_verify_refuses_a_changed_or_unlisted_file(tmp_path: Path) -> None:
    sharded(tmp_path)
    expected = hashes(tmp_path)
    download_model.verify(tmp_path, expected)
    (tmp_path / "head.safetensors").write_bytes(b"other")
    with pytest.raises(SystemExit, match="sha256 differs"):
        download_model.verify(tmp_path, expected)
    (tmp_path / "extra.bin").write_bytes(b"")
    with pytest.raises(SystemExit, match="doesn't list"):
        download_model.verify(tmp_path, expected)


def test_download_verifies_joins_and_leaves_only_what_serving_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = tmp_path / "hub"
    sharded(hub)
    (hub / "integrity.json").write_text(json.dumps({"files": hashes(hub)}))
    calls: list[dict[str, Any]] = []

    def snapshot(repo: str, revision: str | None, allow_patterns: list[str], local_dir: Path) -> str:
        calls.append({"repo": repo, "revision": revision, "allow": allow_patterns})
        for rel in hashes(hub):
            if rel != "integrity.json":
                (local_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                (local_dir / rel).write_bytes((hub / rel).read_bytes())
        (local_dir / ".cache").mkdir()
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(hub / "integrity.json"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    out = tmp_path / "models" / "den-4b"
    download_model.main(["--repo", "org/den-4b@v1", "--out", str(out)])
    assert calls == [{"repo": "org/den-4b", "revision": "v1", "allow": download_model.SERVED}]
    assert sorted(hashes(out)) == ["head.json", "head.safetensors", "merged/config.json", "merged/model.safetensors"]
    download_model.download("org/den-4b@v1", out)  # a finished copy is not downloaded again
    assert len(calls) == 1
