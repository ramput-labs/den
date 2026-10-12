"""A released den model as a local folder: `models/den-4b`, only what serving reads, the weights in one file.

`make serve` reads `RUN` (default `models/den-4b`), so a local copy never asks the Hub. The download takes the head
and `merged/` from the release tag, checks every file against the sha256 the run's `integrity.json` recorded for the
evaluated model, then joins the shards into one `merged/model.safetensors` (the same tensors, bit for bit; both MLX and
transformers read a single unindexed file) and drops what serving never reads: the shards' index, `integrity.json`
and the download cache. A release without `merged/` (v2 on) can't be joined here; serve it as `hf:<repo>@<version>`,
which rebuilds `merged/` on first load.

    uv run python -m scripts.download_model --repo ramput-labs/den-4b@v1 --out models/den-4b   # make model-download
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from den.fetch import digest
from den.paths import MODELS

REPO = "ramput-labs/den-4b@v1"
OUT = MODELS / "den-4b"
WEIGHTS = "model.safetensors"
INDEX = "model.safetensors.index.json"
SERVED = ["head.json", "head.safetensors", "merged/*"]


def verify(root: Path, expected: dict[str, str]) -> None:
    """Every downloaded file under `root` has the sha256 the evaluated run recorded."""
    present = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts)
    if unknown := [rel for rel in present if rel not in expected]:
        raise SystemExit(f"{root}: files integrity.json doesn't list: {', '.join(unknown)}")
    if wrong := [rel for rel in present if digest(root / rel) != expected[rel]]:
        raise SystemExit(f"{root}: sha256 differs from integrity.json: {', '.join(wrong)}")


def join(merged: Path) -> None:
    """The indexed shards in `merged/` -> one `model.safetensors` with the same tensors and metadata."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    shards = sorted(set(json.loads((merged / INDEX).read_text())["weight_map"].values()))
    tensors: dict[str, Any] = {}
    metadata: dict[str, str] | None = None
    for shard in shards:
        with safe_open(merged / shard, "pt") as f:
            metadata = metadata or f.metadata()
            tensors |= {key: f.get_tensor(key) for key in f.keys()}  # noqa: SIM118 (a safetensors handle is not a dict)
    save_file(tensors, merged / f"{WEIGHTS}.part", metadata=metadata or {"format": "pt"})
    (merged / f"{WEIGHTS}.part").rename(merged / WEIGHTS)
    for name in [*shards, INDEX]:
        (merged / name).unlink()


def download(spec: str = REPO, out: Path = OUT) -> Path:
    """`<org>/<name>@<revision>` -> `out`, verified and joined; a finished copy is left as it is."""
    from huggingface_hub import hf_hub_download, snapshot_download

    if (out / "merged" / WEIGHTS).is_file() and (out / "head.safetensors").is_file():
        print(f"{out} is already downloaded")
        return out
    repo, _, revision = spec.partition("@")
    expected = json.loads(Path(hf_hub_download(repo, "integrity.json", revision=revision or None)).read_text())
    snapshot_download(repo, revision=revision or None, allow_patterns=SERVED, local_dir=out)
    if not (out / "merged" / "config.json").is_file():
        raise SystemExit(f"{spec} has no merged/: serve it as RUN=hf:{spec}, which rebuilds merged/ on first load")
    verify(out, expected["files"])
    if (out / "merged" / INDEX).is_file():
        join(out / "merged")
    shutil.rmtree(out / ".cache", ignore_errors=True)
    print(f"{spec} -> {out}: verified against integrity.json, weights in merged/{WEIGHTS}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="download_model", description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=REPO, help="<org>/<name>@<revision> on the Hub (default %(default)s)")
    parser.add_argument("--out", type=Path, default=OUT, help="the local run folder (default %(default)s)")
    args = parser.parse_args(argv)
    download(args.repo, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
