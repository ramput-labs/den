# Fine-tuning den, start to finish

A step-by-step path for a person at the keyboard: rent (or use) an NVIDIA GPU, train Qwen3.5-4B in two rounds,
choose a round on dev, read the test set once, and release the model. Each step is one `make` target and says what
you should see. [`gpu-runbook.md`](gpu-runbook.md) is the same plan written for a cloud agent, with more detail on
every gate.

When something fails, keep the step name (e.g. "Step 4"), the command, and the last 40 lines of its log
(`tail -40 logs/<log>`). That is almost always enough to tell what went wrong.

## Which machine

Training needs **Linux (or WSL2) and an NVIDIA GPU with driver ≥ 580** (the locked PyTorch is the CUDA 13 build).
A Mac can do everything except train: `make train-smoke` runs every code path at toy scale on its CPU.

| Machine | What to run | Model | Time |
|---|---|---|---|
| Mac, or Linux without NVIDIA | `make train-smoke` | qwen3.5-0.8b, 4 steps through PEFT | ~35 min |
| any NVIDIA card ≥ 8 GB (RTX 3070, 4090, ...) | `make rehearse` | qwen3.5-0.8b, ≤ 1,000 records a file | ~2.5 h |
| H100 80 GB, RTX PRO 6000 96 GB | `make gpu-*`, steps 1–9 below | qwen3.5-4b, all data | 5–6 h (3.5–4 h without round 2) |

**Rehearse first.** A small run on any card goes through the same pipeline as the real one: the Unsloth install, the
four-stage round 1 with replay and `--init-from` handoffs, round 2, merge, integrity, evaluation, the ship decision
and a real request. A bug in setup, memory, handoffs or merging then costs nothing to find, and the big GPU's hours
go to training. Qwen3.5-4B's bf16 weights alone are 9.3 GB, so an 8 GB card rehearses on the 0.8B: the code path is
identical for every Qwen3.5 size.

## The plan at a glance

| Step | What | Command | Time (H100) |
|---|---|---|---|
| 0 | Prepare: check, push, tokens | `make check` | 10 min |
| 1 | Machine: driver, disk, tmux | `nvidia-smi` | 5 min |
| 2 | Code, environment, model, data, doctor | `make gpu-setup` | 40–50 min |
| 3 | Gates: CUDA tests, overfit tests | `make gpu-gates` | 15 min |
| 4 | Timing run and the go/no-go decision | `make gpu-timing` | 15 min |
| 5 | Round 1: Kev-4B's four stages | `make gpu-round1` | 1.6–2.5 h |
| 6 | Safety copy to the Hub | `den publish` | 5–10 min |
| 7 | Round 2: public sources + replay (optional) | `make gpu-round2` | 50–70 min |
| 8 | Evaluate both rounds, choose, test once | `make gpu-eval`, `make eval-test` | 30–40 min |
| 9 | Release, prove it answers, keep the evidence | `make release` | 15 min |

Every `gpu-*` target writes its log under `logs/` (gitignored).

---

## Step 0: prepare, before renting anything

The GPU box gets the code from GitHub, so push first:

```bash
make check                     # must end with "... passed" and "claims: 0 of 0 trace to their evidence"
git checkout -b train-v2 && git push -u origin train-v2
```

If `data/` was regenerated since the last upload (a change to `sources.py`, `normalize.py` or `clean.py`), upload it
first, or the box trains on the old copy: `make data-upload DATA_REPO=<org>/<name> TAG=data-v2`.

Write down:
- a **Hugging Face token** with *write* access (<https://huggingface.co/settings/tokens>)
- a **GitHub token** if the repo is private (read to clone; write to push the release record back)
- the **model repo**, e.g. `ramput-labs/den-4b` (created private by the first upload)
- the **data copy**: `a1i6ek/duck-datasets` (the default) or a pinned `<repo>@<tag>`
- your **budget** in GPU hours

## Step 1: the machine

Rent or use Ubuntu 22.04/24.04 (or WSL2 on Windows, with the repo under `~/`, not `/mnt/c`), **driver ≥ 580**, and
disk: ≥ 120 GB on a big GPU (base 9.3 GB, data ~6 GB, venv ~15 GB, merged runs, checkpoints), ≥ 30 GB for a rehearsal.

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h ~ | tail -1
sudo apt update && sudo apt install -y git build-essential tmux curl    # Triton compiles kernels with gcc
tmux new -s den           # an SSH drop can't kill a run; reconnect with: tmux attach -t den
```

**Driver below 580?** On a rented box, pick another image; don't install drivers. On your own Ubuntu machine:
`sudo ubuntu-drivers install` and reboot. On WSL2, update the *Windows* driver; never install one inside WSL.

## Step 2: code, environment, model and data (40–50 min)

```bash
export HF_TOKEN=hf_... HF_REPO=ramput-labs/den-4b DATA_REPO=a1i6ek/duck-datasets BUDGET_HOURS=6
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone --branch train-v2 https://github.com/ramput-labs/den.git && cd den    # private: https://$GITHUB_TOKEN@github.com/...
uvx hf auth login --token "$HF_TOKEN"
mkdir -p logs && make gpu-setup 2>&1 | tee logs/setup.log
export UV_NO_SYNC=1 && echo 'export UV_NO_SYNC=1' >> ~/.bashrc    # your own `uv run`s never undo Unsloth
```

`make gpu-setup` stops at the first failure; rerun it after a fix (finished downloads are verified, not redone):
1. `setup-gpu`: `uv sync` (torch for CUDA 13, transformers 5.x, flash-linear-attention), a check that torch sees the
   GPU, then Unsloth on top **without** replacing torch or transformers, and a second check (`bf16 True`, no `BAD`).
2. `model`: `qwen3.5-4b` (9.3 GB), every shard checked against `locks/`.
3. `data-download`: `data/` from `$DATA_REPO` (~3.6 GB), then a rebuild of the files the licences keep out of the
   copy (~20 min). It first prints `data commit: <repo>@<commit>` (step 9 records it) and ends with
   `ok, every file matches`.
4. `data-check` twice: round 1's and round 2's data shapes.
5. `gpu-doctor`: the readiness gate, saved to `logs/doctor.log`.

**The gate: these numbers must match exactly.** The first `data-check` (round 1) prints:
```
train      40352 records    40352 questions  tokens median 176 p99 4455 max 7390 total 17.6M  skipped 0
dev         1468 records     1468 questions  tokens median 111 p99 788 max 858 total 0.3M  skipped 0
calib       1148 records     1148 questions  tokens median 120 p99 774 max 815 total 0.2M  skipped 0
```
and the second (round 2) starts with
`train      42528 records    42528 questions  tokens median 126 p99 2191 max 6058 total 10.2M  skipped 0`.
`gpu-doctor` prints `ok` on every line, including `GPU memory fits qwen3.5-4b in bf16`. Different numbers mean the
code or data differs from what was validated: stop there.

- **Unsloth fails to install:** `uv pip install --no-deps unsloth unsloth_zoo`, then each missing import with
  `uv pip install -c .unsloth-pins.txt <package>`. **Never install `vllm` or `bitsandbytes`.**
- **`transformers < 5.17` after that:** `uv pip install "transformers==$(grep -i '^transformers==' .unsloth-pins.txt | cut -d= -f3)"`.

## Step 3: gates, can it train at all? (~15 min)

```bash
make gpu-gates
```
- `logs/test-cuda.log`: **3 passed** (the third loads Qwen3.5-4B through Unsloth with LoRA on exactly 248 text
  projections in fp32 and takes one real step).
- `den overfit` ends with **PASS**; `content_agreement` in `logs/overfit-lora.json` near 1.0 means the head reads the
  options' content, not their position.
- `logs/overfit-train.log`: the last `epoch dev` line shows **acc ≥ 0.95**.

A model that can't fit 100 rows has a bug (positions, labels, masking, loss). More training won't fix it: stop.

## Step 4: timing run and the decision (~15 min)

Count the minutes spent since the machine started (≈ 75 if all went well):
```bash
make gpu-timing BUDGET_HOURS="$BUDGET_HOURS" SPENT=75
```
It trains both rounds for 20 steps a stage, checks both merges, answers `examples/request.jsonl` with the merged
model, prints the time estimate with a decision, and deletes `runs/timing`. Watch memory from a second tmux pane
(`Ctrl-b %`): `watch -n 5 nvidia-smi`.

In `logs/timing.log`: every stage has `engine unsloth  dtype bf16 ... lora_modules 248 ... trainable_dtype float32`
(`bfloat16` there is a bug: stop), every stage after the first has `init   from runs/timing/...`, no `nan`, no
`CUDA out of memory`, and both merges print `(248 merged weights)` then `integrity ok`.

The decision: **run both rounds** (steps 5–7), **round 1 only** (skip step 7), or **stop and report**.
Out of memory in a stage? Halve its `--batch` and double its `--accum` on its line in the `Makefile` (same
effective batch), and rerun `make gpu-timing`.

## Step 5: round 1, Kev-4B's four stages (1.6–2.5 h)

```bash
make gpu-round1          # -> runs/kev-recipe/{1-base,2-dates,3-documents,4-skills}; log: logs/round1.log
```

| Stage | Data | lr | Output |
|---|---|---|---|
| 1-base | `train/core`, 2 epochs, 25% none-of-the-above pairs | 5e-5 (head 2e-4) | `runs/kev-recipe/1-base` |
| 2-dates | `train/dates-unknowable` + 2,000 `core` replayed | 2e-5 | `.../2-dates` |
| 3-documents | `train/documents` (states up to 7.4k tokens) + 2,000 replayed | 2e-5 | `.../3-documents` |
| 4-skills | `train/skills` + `train/devtools` + 4,000 replayed, merged | 2e-5 | `.../4-skills` |

Checkpoints every 500 steps; each stage is capped at 100 minutes (`STAGE_MINUTES`). Check every ~15 minutes:
```bash
grep -E "^setup|^init|dev|calibration|saved|stopping|nan|Error" logs/round1.log | tail -20
```
Good: each stage ends with `epoch dev ...`, `calibration  T ...` (T between 0.5 and 3), `dev at T ...` and
`saved runs/kev-recipe/<stage>`. 1-base reaches dev nll < 1.0 and acc ≥ 0.63 (chance: 0.31). Later stages keep
`dev/core` within ~3 points of 1-base. The last prints `(248 merged weights)` and `integrity ok`.

**If the machine dies mid-stage:** finished stages are kept. `make -n gpu-round1` prints the commands; rerun only the
unfinished stage's line (adding `--resume latest`) and the ones after it.

## Step 6: safety copy to the Hub (5–10 min)

```bash
uv run den publish --run runs/kev-recipe/4-skills --repo "$HF_REPO" --logs logs/timing.log logs/round1.log
```
If the machine is lost after this, the model isn't. Interrupted? Run the same command again: it resumes. It uploads
the adapter and head (~140 MB) with the run's data and logs, never `merged/`: servers rebuild that from the adapter,
checked against `integrity.json`. This isn't a release yet (step 9).

## Step 7: round 2 (skip if step 4 said so)

```bash
make gpu-round2          # -> runs/round2; log: logs/round2.log
```
1,500 records from each of the 18 public sources plus 1,500 replayed from each Kev suite, from round 1's `4-skills`.
Good: `calibration  T ...` (0.5–3), `saved runs/round2`, `(248 merged weights)`.

## Step 8: evaluate, choose, test once

```bash
make gpu-eval            # log: logs/eval.log
```
For each round: integrity, every dev file with Kev's robustness checks, the augmentations, and 500 records of every
source; then `den compare` ends with `ship: <run>` (dev only: mean accuracy up, no file down more than a point), and
`den compare --paired` says whether the difference is beyond noise.

```bash
export SHIP=runs/round2          # or runs/kev-recipe/4-skills, as `compare` printed
make eval-test RUN=$SHIP 2>&1 | tee logs/test.log
SRC_TEST=$(cd data/clean && ls test/sources/*/*.jsonl | tr '\n' ' ')
uv run den evaluate --final --run "$SHIP" --limit 1000 $SRC_TEST 2>&1 | tee -a logs/test.log
```
**The test set is read once,** for the shipped run only. The command refuses a second read. Never test the other round.

## Step 9: release, prove it, keep the evidence

```bash
DATA_PIN=$(grep -oP '^data commit: \K\S+' logs/setup.log)      # the <repo>@<commit> data-download fetched
make release VERSION=v2 RUN=$SHIP REPO="$HF_REPO" PARENT=v1 DATA_REPO="$DATA_PIN" \
  ARGS="--logs logs/timing.log logs/round1.log logs/round2.log"
```
This publishes the run, tags that Hub commit, and writes `releases/<version>.json` (lineage, data hashes, dev and test
results, file hashes, the paired comparison with its parent). Add `--dry-run` to `ARGS` to see every check pass
first. A release refuses a version that falls more than a point below its parent on any dev file.

Prove the published model answers, from nothing but the Hub:
```bash
make predict RUN=release:v2
make serve RUN=release:v2 > logs/serve.log 2>&1 &
until grep -q "den serving" logs/serve.log; do sleep 2; done
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d @examples/request.jsonl; echo
kill %1
```

Keep the evidence: `releases/` and `reports/runs/` (every evaluation's numbers and per-question answers) are what the
next version is compared against.
```bash
git add releases/ reports/runs/ && git commit -m "Release v2: record and evaluation evidence" && git push
```
Then **destroy the instance** (some providers bill stopped disks).

---

## Training on your own data

New labelled records go in the `/v1/systemone` shape plus a label on each question (see `den/api.py`):
```json
{"state": "...", "questions": {"team": {"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": "..."}, "label": "a"}}}
```
Labels are the option name (`choice`), a bool (`noul`) or a level index (`score`). Put the file under
`data/clean/train/`, then continue from the last release so nothing earlier is forgotten:
```bash
make train-next FROM=v1 OUT=runs/v2 DATA="train/<your-file>.jsonl"   # replays Kev's suites and 500 of every source
make eval-dev RUN=runs/v2
```
and finish with steps 8–9. `make train` runs a single stage with any flags (`make train ARGS="--help"` lists them).

## Knobs

| Variable | Default | Effect |
|---|---|---|
| `MODEL` | `qwen3.5-4b` | the backbone (`make models` lists every size) |
| `STAGE_MINUTES`, `ROUND2_MINUTES` | 100, 90 | per-stage time caps (a capped stage calibrates, saves and hands over) |
| `SAVE_STEPS` | 500 | checkpoint interval, for `--resume latest` |
| `LOGS` | `logs` | where `gpu-*` logs go |
| `N` | 1000 | records per file in `make rehearse` |
| `ARGS` | | extra `den train` flags for every stage, e.g. `ARGS="--max-state 4096"` |

`make -n <target>` prints every command without running it.

## Quick fixes

| Symptom | Fix |
|---|---|
| `uv run` hangs at 0% CPU | another uv process holds the lock: use `.venv/bin/den ...` |
| a package vanished after a `uv run` | `UV_NO_SYNC=1` wasn't set: `make setup-gpu`, then export it (step 2) |
| `... falling back to its reference PyTorch implementation` | flash-linear-attention is missing (many times slower): fix before training |
| `causal_conv1d_fn is falling back ...` | expected: the fallback is cuDNN's conv1d, at most ~10% of a step |
| `CUDA out of memory` | halve that stage's `--batch`, double its `--accum` in the `Makefile`; in a rehearsal, `ARGS="--max-state 4096"` |
| SSH dropped | `tmux attach -t den`; training kept running |
| instance died mid-stage | `make -n gpu-round1`, rerun the unfinished stage's line with `--resume latest`, then the rest |
| `merge: replaced N of 248` | the adapter is fine: keep the log, don't retrain |
| an upload stopped | run the same command again: uploads resume |
| `has already read [...]: the locked test set is read once` | working as intended: don't re-read test |
