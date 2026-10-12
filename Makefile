# den: `make help` lists every target. New here? Start with README.md's Quick start.
#
#   any machine (Mac or Linux)   make setup, make serve, make predict      use the released model
#   any machine                  make check, make train-smoke              develop; the whole pipeline at toy scale
#   Linux + NVIDIA GPU           make gpu-setup, make rehearse, gpu-*      fine-tune (docs/fine-tuning.md)
#
# Model targets take MODEL=<key> (`make models`); the default is qwen3.5-4b. ARGS reach the underlying `den` command.
.DEFAULT_GOAL := help
SHELL := /bin/bash
.SHELLFLAGS := -eo pipefail -c
MAKEFLAGS += --no-print-directory

# After `make setup-gpu` has put Unsloth on top of the lockfile (.unsloth-pins.txt marks it), every `uv run` under
# make skips the sync that would take it away again.
UV := uv run
ifneq ($(wildcard .unsloth-pins.txt),)
export UV_NO_SYNC := 1
endif

# What to serve: the released model, pinned (its merged weights live in the v1 tag).
RUN ?= models/den-4b
PORT ?= 8009
REQUEST ?= examples/request.jsonl

MODEL ?= $(or $(DEN_MODEL),qwen3.5-4b)
RUNS ?= runs/kev-recipe
ROUND2 ?= runs/round2
CAP ?= 1500
EVAL ?= --eval-steps 400 --eval-max 600
SUITES_SCORED := core documents skills devtools transfer probes breadth
DEV := $(SUITES_SCORED:%=dev/%.jsonl) dev/binding.jsonl dev/semif.jsonl
TEST := $(SUITES_SCORED:%=test/%.jsonl)
SHIP_DEV := dev/core.jsonl dev/documents.jsonl dev/skills.jsonl dev/devtools.jsonl
KEV_SUITES := train/core.jsonl train/dates-unknowable.jsonl train/documents.jsonl train/skills.jsonl train/devtools.jsonl
# What shipped runs fit T on: Kev's calibration set plus every source's calibration split (`den normalize`), one T per
# question type. Read when a target runs, so it sees the data that data-download placed.
CALIB = --calibration calibration/core.jsonl $(patsubst data/clean/%,%,$(sort $(wildcard data/clean/calibration/sources/*/*.jsonl))) \
	--calibrate-by-type
NEXT_SOURCES ?= 500
TRAIN = $(UV) den train --model $(MODEL)
DATA_REPO ?= $(or $(DEN_DATA_REPO),a1i6ek/duck-datasets)
GIVEN_DATA_REPO := $(filter command line environment,$(origin DATA_REPO))$(DEN_DATA_REPO)
log = 2>&1 | tee $(1)

# The gated GPU path (docs/fine-tuning.md), full scale by default. Logs go to LOGS, where `den publish --logs` and
# `make release ARGS="--logs ..."` pick them up.
LOGS ?= logs
MIN_DISK ?= 80
STAGE_MINUTES ?= 100
ROUND2_MINUTES ?= 90
SAVE_STEPS ?= 500
OVERFIT_LORA ?=
OVERFIT_TRAIN ?= --batch 4 --accum 1
BUDGET_HOURS ?= 5
SPENT ?= 75

# `make rehearse`: the same gates and both rounds, small enough for an 8 GB card (Qwen3.5-4B's bf16 weights alone are
# 9.3 GB): qwen3.5-0.8b, at most N records from any one file, batch 1 x 8 (the recipe's effective batch of 8).
N ?= 1000
REHEARSE_RUNS ?= runs/rehearse
REHEARSE := MODEL=qwen3.5-0.8b RUNS=$(REHEARSE_RUNS) ROUND2=$(REHEARSE_RUNS)/round2 CAP=100 LOGS=$(REHEARSE_RUNS)/logs \
	MIN_DISK=30 STAGE_MINUTES=30 ROUND2_MINUTES=30 SAVE_STEPS=0 EVAL="--eval-steps 50 --eval-max 200" \
	OVERFIT_LORA="--batch 8" OVERFIT_TRAIN="--batch 2 --accum 2" ARGS="--limit $(N) --batch 1 --accum 8 $(ARGS)"

.PHONY: help setup setup-gpu check doctor env models model smoke serve predict example train-smoke test-model test-cuda \
	list data-download data-upload data-verify data-check data-licences data-breadth data-normalize data-clean data-audit \
	train train-round1 train-round2 train-next train-rebuild eval-dev eval-test release release-list bench-run bench-plot \
	gpu-setup gpu-doctor gpu-gates gpu-timing gpu-round1 gpu-round2 gpu-eval rehearse rehearse-clean linux-gpu

# Targets that train or test on CUDA stop at once on any other machine, instead of failing gate by gate.
setup-gpu test-cuda gpu-setup gpu-doctor gpu-gates gpu-timing gpu-round1 gpu-round2 rehearse: linux-gpu

linux-gpu:
	@if [ "$$(uname -s)" != Linux ] || ! command -v nvidia-smi >/dev/null; then \
		echo "make $(MAKECMDGOALS) runs on a Linux (or WSL2) machine with an NVIDIA GPU: docs/fine-tuning.md"; \
		echo "On this machine: make check, make train-smoke, make serve."; \
		exit 1; \
	fi

help: ## show targets
	@awk 'BEGIN {FS = ":.*## "} /^##@ / {printf "\n%s\n", substr($$0, 5)} /^[a-z0-9-]+:.*## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ Get started (any machine: Mac with MLX, or Linux)

setup: ## install Python 3.12 and the locked dependencies (uv sync)
	uv sync

serve: ## serve RUN (default: the released v1) at http://127.0.0.1:PORT/v1/systemone; DEN_API_KEY=... requires a key
	$(UV) den serve --run $(RUN) --port $(PORT)

predict: ## answer REQUEST (default examples/request.jsonl, one /v1/systemone request a line) with RUN, no server
	$(UV) den predict --run $(RUN) $(REQUEST)

example: ## the TypeSafe SDK quickstart against a running `make serve` (examples/README.md has the rest)
	TYPESAFE_BASE_URL=http://127.0.0.1:$(PORT) PYTHONPATH=examples $(UV) python examples/quickstart.py

check: ## lint, strict type-check, unit tests, claims vs evidence: run before every commit
	$(UV) ruff check den scripts tests examples
	$(UV) ruff format --check den scripts tests examples
	$(UV) mypy
	$(UV) pytest -q
	$(UV) python -m scripts.verify_claims

doctor: ## versions, GPU, BF16, driver, Unsloth, model revision; ARGS="--require cuda" to gate on them
	$(UV) den doctor --model $(MODEL) $(ARGS)

env: ## the backend this machine uses (mlx | cuda | cpu)
	$(UV) den env

models: ## every backbone (any size) and the Kev reference checkpoints
	$(UV) den models

model: ## download MODEL (a key from `make models`, or hf:<org>/<name>@<commit>)
	$(UV) den model $(MODEL)

smoke: ## run MODEL's backbone once on this machine's backend
	$(UV) den smoke --model $(MODEL)

# Every code path a GPU run takes except Unsloth and the CUDA kernels, on any machine: 4 LoRA steps on qwen3.5-0.8b
# through PEFT on the CPU, best/, calibration, merge, integrity, then evaluation and serving. About 35 min on an M5 Pro
# (DeltaNet's PyTorch fallback is the slow part); `make rehearse` and the GPU gates cover the rest.
train-smoke: ## the whole pipeline at toy scale on this machine (~35 min on a Mac): train, merge, check, evaluate, predict
	$(MAKE) model MODEL=qwen3.5-0.8b
	rm -rf runs/train-smoke
	$(UV) den train --model qwen3.5-0.8b --engine peft --limit 50 --overfit 8 --max-steps 4 --batch 2 --accum 1 \
		--eval-steps 2 --eval-max 8 --calibration calibration/sources/topic/trec.jsonl --calibrate-by-type --merge \
		--out runs/train-smoke
	$(UV) den check-run --run runs/train-smoke --load
	$(UV) den evaluate --run runs/train-smoke --limit 30 --no-evidence dev/core.jsonl dev/sources/decisions/typed-decisions.jsonl
	$(UV) den evaluate --run runs/train-smoke --limit 20 --no-evidence --augment pairs dev/core.jsonl
	$(UV) den predict --run runs/train-smoke $(REQUEST)

test-model: ## tests on the real qwen3.5-4b weights: MLX vs PyTorch parity, LoRA wiring, merge (slow)
	$(UV) pytest -q -m model

##@ Data (data/ is gitignored; data-download is the one way to get it)

data-download: ## the Hub copy into data/, rebuilding what the licences left out, every file checked: [DATA_REPO=<org>/<name>[@rev]] [NO_REBUILD=1]
	$(UV) python -m scripts.mirror download --repo $(DATA_REPO) $(if $(NO_REBUILD),--no-rebuild)

data-check: ## tokenize the clean training data and print its shape (no GPU; the data gate)
	$(UV) den train --dry-run --model $(MODEL) $(ARGS)

data-verify: ## re-hash every downloaded file against locks/
	$(UV) den verify

list: ## every pinned dataset
	$(UV) den list

data-upload: ## data/ to the Hub, licence notices included, restricted sources left out: DATA_REPO=<org>/<name> [TAG=data-v1] [PUBLIC=1]
	@test -n "$(GIVEN_DATA_REPO)" || (echo "name the repo to upload to: DATA_REPO=<org>/<name> (or DEN_DATA_REPO)" && exit 1)
	$(UV) python -m scripts.mirror upload --repo $(DATA_REPO) $(if $(PUBLIC),--public) $(if $(TAG),--tag $(TAG))

data-licences: ## write each source's licence, evidence and texts into data/ (LICENSES.md, LICENSES/, README.md card)
	$(UV) den licences $(ARGS)

data-breadth: ## rebuild Kev's breadth-v1 panel -> data/{dev,test}/breadth.jsonl, sha256-checked; before data-normalize
	$(UV) python -m scripts.build_breadth

data-normalize: ## raw sources -> leakage-free train / dev / test under data/*/sources/ (after changing normalize.py)
	$(UV) den normalize

data-clean: ## rewrite the cleaned, deduplicated copies under data/clean/ (after changing clean.py)
	$(UV) den clean

data-audit: ## check every record and source is fit to train on; writes reports/data-audit.json
	$(UV) den audit --model $(MODEL)

##@ Fine-tune and evaluate (training: Linux + NVIDIA, after setup-gpu; see docs/fine-tuning.md)

setup-gpu: ## Linux + NVIDIA (driver >= 580): setup, then Unsloth on top without touching torch/transformers, both checked
	uv sync
	uv run --no-sync python -m scripts.check_env
	uv pip freeze | grep -iE '^(torch|transformers|tokenizers|huggingface-hub|numpy|triton|flash-linear-attention|accelerate|peft)==' > .unsloth-pins.txt
	uv pip install "unsloth>=2026.9" -c .unsloth-pins.txt
	uv run --no-sync python -m scripts.check_env --unsloth
	@echo "make targets now skip uv's sync; for your own uv runs: export UV_NO_SYNC=1 && echo 'export UV_NO_SYNC=1' >> ~/.bashrc"

test-cuda: ## CUDA checks: devices, dtypes, padding, and one real Unsloth LoRA + head step on MODEL
	DEN_MODEL=$(MODEL) $(UV) pytest -q -m cuda

train: ## one LoRA + pointer-head fine-tune of MODEL; ARGS="--data train/<file>.jsonl --epochs 2 ..."
	$(TRAIN) $(ARGS)

# Stage 1's head is fresh, so it trains at --head-lr 2e-4, not at the adapters' 5e-5: our change to Kev's recipe, which
# records no head rate. Later stages continue it at --lr. 1e-3 (the overfit gate's rate) diverged on the H100 at step
# ~1450 of 4450 (2026-10-10): the head's weights grew, dev nll 0.41 -> 1.87, grad norm median 6 -> 30; the LoRA didn't.
train-round1: ## round 1, Kev-4B's four stages, each from the last -> RUNS/{1-base,...,4-skills}; ARGS go to every stage
	$(TRAIN) --data train/core.jsonl --epochs 2 --lr 5e-5 --head-lr 2e-4 --batch 4 --accum 2 --p-none-pair 0.25 $(EVAL) \
		--dev dev/core.jsonl --out $(RUNS)/1-base $(ARGS)
	$(TRAIN) --data train/dates-unknowable.jsonl --replay 2000 --init-from $(RUNS)/1-base --lr 2e-5 --batch 4 --accum 2 \
		$(EVAL) --dev dev/core.jsonl --out $(RUNS)/2-dates $(ARGS)
	$(TRAIN) --data train/documents.jsonl --replay 2000 --init-from $(RUNS)/2-dates --lr 2e-5 --batch 2 --accum 4 \
		$(EVAL) --dev dev/documents.jsonl dev/core.jsonl --out $(RUNS)/3-documents $(ARGS)
	$(TRAIN) --data train/skills.jsonl train/devtools.jsonl --replay 4000 --init-from $(RUNS)/3-documents --lr 2e-5 \
		--batch 2 --accum 4 $(EVAL) --dev dev/skills.jsonl dev/devtools.jsonl dev/core.jsonl $(CALIB) --merge \
		--out $(RUNS)/4-skills $(ARGS)

train-round2: ## round 2 from round 1 -> ROUND2: the 18 public sources (CAP each) + CAP replayed from each Kev suite
	$(TRAIN) --data --sources-cap $(CAP) --replay $(CAP) --replay-from $(KEV_SUITES) --init-from $(RUNS)/4-skills \
		--lr 2e-5 --batch 4 --accum 2 $(EVAL) --dev $(SHIP_DEV) $(CALIB) --merge --out $(ROUND2) $(ARGS)

# A new version continues from the last release on new data, replaying Kev's suites (REPLAY each) and the public
# sources (SOURCES each, default NEXT_SOURCES) so neither round's skills are forgotten.
train-next: ## the next version from a released one: FROM=v1 OUT=runs/v2 DATA="train/<new>.jsonl ..." [SOURCES=500] [REPLAY=1500]
	$(TRAIN) --data $(DATA) --sources-cap $(or $(SOURCES),$(NEXT_SOURCES)) --replay $(or $(REPLAY),$(CAP)) \
		--replay-from $(KEV_SUITES) --init-from release:$(FROM) --lr 2e-5 --batch 4 --accum 2 $(EVAL) \
		--dev $(SHIP_DEV) $(CALIB) --merge --out $(OUT) $(ARGS)

# The periodic rebuild: one run from the base on everything (Kev's suites in full, CAP of every source, any new DATA),
# so a chain of --init-from versions can be checked against a fresh model. Ship it only if it wins on dev
# (`den compare --paired`). Untested at full scale: the one-run mix is ours, not Kev's.
train-rebuild: ## one run from the base on all data -> OUT: OUT=runs/v3-rebuild [DATA="train/<new>.jsonl ..."] [EPOCHS=1]
	$(TRAIN) --data $(KEV_SUITES) $(DATA) --sources-cap $(CAP) --epochs $(or $(EPOCHS),1) --lr 5e-5 --head-lr 1e-3 \
		--batch 4 --accum 2 --p-none-pair 0.25 $(EVAL) --dev $(SHIP_DEV) $(CALIB) --merge --out $(OUT) $(ARGS)

eval-dev: ## a trained RUN on dev: integrity, metrics + Kev's robustness checks, augmentations
	$(UV) den check-run --run $(RUN)
	$(UV) den evaluate --run $(RUN) $(DEV)
	for AUG in pairs permute none-replace none-add distract; do \
		$(UV) den evaluate --run $(RUN) --augment $$AUG dev/core.jsonl dev/documents.jsonl dev/skills.jsonl || exit 1; \
	done

eval-test: ## the locked test set, read once, for the shipped RUN only (refuses a second read)
	$(UV) den evaluate --final --run $(RUN) $(TEST)

release: ## publish RUN as VERSION, tag it on REPO, record releases/VERSION.json: VERSION=v2 RUN=runs/v2 REPO=<org>/<name> [PARENT=v1] [DATA_REPO=<org>/<name>@<tag>]
	$(UV) den release create --version $(VERSION) --run $(RUN) --repo $(REPO) $(if $(PARENT),--parent $(PARENT)) \
		$(if $(GIVEN_DATA_REPO),--data-repo $(DATA_REPO)) $(ARGS)

release-list: ## every released version, its parent and headline numbers
	$(UV) den release list

bench-run: ## den vs Kev-4B on every dev panel through both servers (benchmarks/): [DEN_URL=http://127.0.0.1:8009] [KEV_URL=http://127.0.0.1:8008]
	$(UV) python -m scripts.run_benchmarks --model kev-4b=$(or $(KEV_URL),http://127.0.0.1:8008) \
		--model den-v1=$(or $(DEN_URL),http://127.0.0.1:8009) --baseline kev-4b

bench-plot: ## benchmarks/results.json -> plots (light + dark) and benchmarks/README.md
	$(UV) python -m scripts.plot_benchmarks

##@ GPU box: the gated path, one target per step (any NVIDIA GPU; docs/fine-tuning.md, docs/gpu-runbook.md)

gpu-setup: setup-gpu ## setup-gpu, MODEL, data/, both rounds' data shapes, then gpu-doctor
	$(MAKE) model data-download
	$(MAKE) data-check
	$(MAKE) data-check ARGS="--data --sources-cap $(CAP) --replay $(CAP) --replay-from $(KEV_SUITES) $(ARGS)"
	$(MAKE) gpu-doctor

gpu-doctor: ## the readiness gate: CUDA, BF16, driver, Unsloth, kernels, VRAM for MODEL, MIN_DISK GB free
	@mkdir -p $(LOGS)
	$(MAKE) doctor ARGS="--require cuda --min-disk $(MIN_DISK)" $(call log,$(LOGS)/doctor.log)

gpu-gates: ## can it train at all: the CUDA tests on MODEL, then both overfit gates (LoRA + head must fit 100 rows)
	@mkdir -p $(LOGS)
	$(MAKE) test-cuda $(call log,$(LOGS)/test-cuda.log)
	$(UV) den overfit --model $(MODEL) --n 100 --steps 300 $(OVERFIT_LORA) --out $(LOGS)/overfit-lora.json \
		$(call log,$(LOGS)/overfit-lora.log)
	$(UV) den train --model $(MODEL) --data train/core.jsonl --overfit 100 --epochs 25 --lr 2e-4 --head-lr 1e-3 \
		$(OVERFIT_TRAIN) --out runs/overfit $(call log,$(LOGS)/overfit-train.log)
	rm -rf runs/overfit

gpu-timing: ## 20 steps of every stage, both merges checked, the time estimate and decision: BUDGET_HOURS=5 SPENT=<min>
	@mkdir -p $(LOGS)
	$(MAKE) train-round1 RUNS=runs/timing ARGS="--max-steps 20" $(call log,$(LOGS)/timing.log)
	$(MAKE) train-round2 RUNS=runs/timing ROUND2=runs/timing/round2 ARGS="--max-steps 20" 2>&1 | tee -a $(LOGS)/timing.log
	$(UV) den check-run --run runs/timing/4-skills && $(UV) den check-run --run runs/timing/round2
	$(UV) den predict --run runs/timing/round2 $(REQUEST)
	$(UV) python -m scripts.estimate_time --budget-hours $(BUDGET_HOURS) --spent-minutes $(SPENT)
	rm -rf runs/timing

gpu-round1: ## round 1 -> RUNS, each stage capped at STAGE_MINUTES, checkpoints every SAVE_STEPS
	@mkdir -p $(LOGS)
	$(MAKE) train-round1 ARGS="--max-minutes $(STAGE_MINUTES) --save-steps $(SAVE_STEPS) $(ARGS)" $(call log,$(LOGS)/round1.log)

gpu-round2: ## round 2 -> ROUND2, capped at ROUND2_MINUTES (skip it if gpu-timing said so)
	@mkdir -p $(LOGS)
	$(MAKE) train-round2 ARGS="--max-minutes $(ROUND2_MINUTES) --save-steps $(SAVE_STEPS) $(ARGS)" $(call log,$(LOGS)/round2.log)

gpu-eval: ## eval-dev on each round, 500 records of every source, then the ship decision on dev (paired too)
	@mkdir -p $(LOGS)
	( SRC=$$(cd data/clean && ls dev/sources/*/*.jsonl | tr '\n' ' '); \
	for RUN in $(RUNS)/4-skills $(patsubst %/run.json,%,$(wildcard $(ROUND2)/run.json)); do \
		$(MAKE) eval-dev RUN=$$RUN && $(UV) den evaluate --run $$RUN --limit 500 $$SRC || exit 1; \
	done; \
	if [ -f $(ROUND2)/run.json ]; then \
		$(UV) den compare $(RUNS)/4-skills $(ROUND2) --files $(SHIP_DEV) dev/transfer.jsonl $$SRC && \
		$(UV) den compare --paired $(RUNS)/4-skills $(ROUND2); \
	fi ) $(call log,$(LOGS)/eval.log)

# The same gates and both rounds at small scale: the cheap way to find a setup, memory, handoff or merge bug before
# paying for a big GPU. Dev samples only, nothing written to reports/, the test set never read.
rehearse: ## doctor, gates, both rounds and a quick evaluation at N records a file on qwen3.5-0.8b (any 8 GB card, ~2.5 h)
	$(MAKE) model MODEL=qwen3.5-0.8b
	$(MAKE) gpu-doctor gpu-gates gpu-round1 gpu-round2 $(REHEARSE)
	for RUN in $(REHEARSE_RUNS)/4-skills $(REHEARSE_RUNS)/round2; do \
		$(UV) den check-run --run $$RUN && $(UV) den evaluate --run $$RUN --limit 200 --no-evidence $(SHIP_DEV) || exit 1; \
	done $(call log,$(REHEARSE_RUNS)/logs/eval.log)
	$(UV) den compare $(REHEARSE_RUNS)/4-skills $(REHEARSE_RUNS)/round2 --files $(SHIP_DEV) | tee -a $(REHEARSE_RUNS)/logs/eval.log
	$(UV) den predict --run $(REHEARSE_RUNS)/round2 $(REQUEST) | tee $(REHEARSE_RUNS)/logs/predict.json
	@echo "rehearsal passed: qwen3.5-0.8b, $(N) records a file; logs in $(REHEARSE_RUNS)/logs"

rehearse-clean: ## delete the rehearsal's runs and logs
	rm -rf $(REHEARSE_RUNS)
