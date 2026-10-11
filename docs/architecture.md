# How den works

## System One rules

1. **One pass, no generation.** The backbone prefills `state + question + options`. The head scores each option
   against the question, and a softmax turns the scores into probabilities. No logits are sampled, no text generated.
2. **Typed questions.** `choice` (1–255 named options), `score` (ordered levels), `noul` (yes/no). The request shape
   is the API, and training records use the same bytes the server sends to the model.
3. **Calibrated.** A temperature `T` (one per question type for shipped runs) is fitted on held-out calibration
   splits, never on dev or test.
4. **Honest evaluation.** Eval-only sources are never trained on. Locked test partitions are read once per release
   candidate. The catalog enforces both (`den/catalog.py`, `tests/`).
5. **Pinned.** Every download names a commit and a sha256. `locks/*.json` records every byte that was placed.

## The model

A Qwen base model (Qwen3.5-4B-Base by default) with a LoRA adapter, rank 16 (alpha 32, dropout 0) on all 248
text-decoder projections (attention, Gated DeltaNet and MLP; none in the vision tower), fp32 adapters over a bf16
base, and a head:

- **The prompt** (`den/prompt.py`) is plain text, tokenized once. Pointer positions come from the tokenizer's
  character offsets: each option's span runs from its `-` to its closing newline, and the question's final token is
  the `:` of `Answer:`. Each question is read with the state alone.
- **The head** (`den/head.py`) starts as Kev's pointer, q(question's final token) · k(option's closing token), and adds
  span pooling, one cross-option attention block and a per-option prior, all starting at zero, so training begins
  from Kev's scoring. It is permutation-equivariant over options. `--head-kind pointer` is Kev's head exactly;
  `--head-layers 0` drops the block.
- **Score questions:** `--ordinal-weight` adds Kev's ranked probability score, so near-miss levels cost less.

A run exports Hugging Face-style files only, no pickle: the PEFT adapter, the head as `head.safetensors` +
`head.json` (its `hidden_size` must match the backbone's), and `merged/`, the base with the LoRA folded in
(`lora.fold`: W + scale·B@A in float64 on the CPU, rounded once to bf16, so every machine builds the same bytes).
Releases upload the adapter only; servers rebuild `merged/` on first load and check it against `integrity.json`.

## Backends

| Machine | Backend | Precision | Runtime |
|---|---|---|---|
| Apple Silicon | `mlx` | bf16 | mlx-lm, Metal |
| Linux + NVIDIA GPU | `cuda` | bf16 | PyTorch + transformers, flash-linear-attention kernels |
| anything else | `cpu` | fp32 | PyTorch + transformers |

`DEN_BACKEND=mlx|cuda|cpu` overrides the choice. Both runtimes load only the model's text tower, so dense, MoE and
vision-language checkpoints all work, and both return the same final hidden states: on Qwen3.5-4B, MLX in bf16 and
PyTorch in fp32 agree to a per-token cosine of at least 0.999 (`make test-model`). The head always runs in PyTorch on
the CPU. The Linux wheels are built for CUDA 13 (driver ≥ 580); `causal-conv1d` is optional.

Any Qwen backbone works (`make models`): Qwen3 0.6B–30B-A3B, Qwen3.5 0.8B–35B-A3B, Qwen3.8-27B, and the Kev 1.0
references (`kev-0.8b`, `kev-4b`, `kev-9b`, `kev-27b`). Any other Hub model: `make model MODEL=hf:<org>/<name>@<commit>`
(its digests are frozen in the lock on first download).

## Training

Two rounds, each a chain of stages where every stage continues from the last (`--init-from`); the commands are in the
`Makefile` and the full walk-through in [`fine-tuning.md`](fine-tuning.md).

- **Round 1** (`make train-round1`): Kev-4B's own recipe, from its model card and code. `core` ×2 at 5e-5 with 25%
  none-of-the-above minimal pairs, then dates, then documents, then skills + devtools, at 2e-5 with 2k/2k/4k `core`
  records replayed. One change of ours: stage 1's fresh head trains at `--head-lr 2e-4` (Kev records no head rate;
  1e-3 diverged on the H100).
- **Round 2** (`make train-round2`): continues on 18 public sources Kev-4B never trained on (`CAP` records each) with
  `CAP` replayed from each Kev suite. `den compare` ships it only if it beats round 1 on dev without losing more than a
  point on any file.

Every stage: one row per question, Kev's option augmentation (shuffle; none-of-the-above 10%/12%; distractor 15%),
validation on a fixed dev sample every `--eval-steps`, and the lowest-dev-NLL checkpoint kept in `best/`. The
`transformers.Trainer` (patched by Unsloth) owns bf16, accumulation, clipping and the cosine schedule; `den/trainer.py`
overrides only length-grouped sampling and the head's learning rate.

### Experiment modes and gates

| Mode | Command |
|---|---|
| A zero-shot Qwen (its own answer-letter probabilities) | `den train --head-kind letters --lora 0 --epochs 0` |
| B Qwen + LoRA, answering with the letter | `den train --head-kind letters` |
| C frozen Qwen + pointer head | `den train --lora 0 --head-lr 1e-3` |
| D Qwen + LoRA + head (the release recipe) | `make train-round1`, `make train-round2` |

Ablations: `--lora-targets all|attention-mlp|attention`, `--option-rep end|marker|mean|attn`, `--head-proj linear|mlp`,
`--head-kind set|pointer`, `--ordinal-weight`, `--lora-alpha`, `--rslora`. `--report-to trackio|wandb|tensorboard`
logs to a tracker. `den probe` compares modes A and C quickly on cached frozen-backbone features (on a Mac too).

Gates before any long run: `den overfit` (LoRA + head must fit 100 real rows; then determinism, option-permutation and
option-replacement checks) and `den train --overfit 100` (the same through the real training loop).

## Evaluation and evidence

`den evaluate` reports accuracy, NLL, Brier, ECE, coverage at 5% error, accuracy and ECE per question type, accuracy
per source, and score-level MAE/RPS, plus Kev's robustness checks where a file carries them (option-order flips,
contrastive pairs, confidence on unknowable items). `--augment pairs|permute|none-replace|none-add|distract` measures
Kev's augmentations. All metric math lives in `den/metrics.py`.

Every evaluation is kept in git: `den evaluate` writes `reports/runs/<run>/` (each file's metrics, every question's
probabilities, provenance), and `den compare --paired A B` says whether B beats A beyond noise, question by question
with 95% intervals. Numbers printed in the docs are listed in `reports/claims.json` and checked by `make check`.

## Versions

`make release VERSION=v1 RUN=<run> REPO=<org>/<name>` publishes a run, tags the Hub commit and records
`releases/v1.json` (lineage, data hashes, dev and test results, file hashes). A new version is trained when enough new
data has gathered, not on a schedule: `make train-next FROM=v1 OUT=runs/v2 DATA="train/<new>.jsonl"` continues from v1,
replaying Kev's suites and 500 records of every public source. It is released with `PARENT=v1`, which refuses it if
any dev file fell more than a point below v1. When the base model, LoRA rank or head shape changes, or after three
versions in a chain, `make train-rebuild` trains one fresh model on everything; it ships only if it beats the chain
on dev. Every decision comes from dev; test numbers are reported, never used to choose. `release:v1` works wherever a
run does (`--init-from`, `--run`).
