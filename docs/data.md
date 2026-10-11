# Data

Everything a run trains and is measured on: where it comes from, how it's laid out, how it's cleaned, and the rules
that keep evaluation honest. `data/` is gitignored; `make data-download` is the one way to get it.

## Getting it

```bash
make data-download                       # the Hub copy (~3.6 GB), plus a rebuild of what the licences keep out of it
make data-download DATA_REPO=<org>/<name>@<commit>   # a pinned copy (default: a1i6ek/duck-datasets)
make data-verify                         # re-hash every placed file against locks/
make data-check                          # tokenize data/clean and print its shape (no GPU)
```

`make data-download` downloads the copy `make data-upload DATA_REPO=<org>/<name>` made, checks every file against its
sha256 manifest and every pinned file against `locks/`, then rebuilds only the left-out files whose step has work to
do (fetch, breadth, normalize, clean, ~20 min on a fresh machine), checking them against the uploader's hashes. A
rerun skips what is already there and verified; `NO_REBUILD=1` skips the rebuild. The underlying fetches stay
available as `uv run den data <set>` (`den list` shows the sets) for work on the data pipeline itself.

| Set | Size | Lands in |
|---|---|---|
| `suites` | ~100 MB | `train/ calibration/ dev/ test/ manifests/` |
| `raw-train` | ~0.8 GB | `sources/train/{intent,topic,reading,sentiment,safety}` |
| `raw-new` | ~0.3 GB | `sources/train/{knowledge,intent,tools,decisions}` |
| `raw-eval` | ~0.6 GB | `sources/eval/` |
| `raw-bulk` | ~5 GB | `sources/train/{documents,devtools}` (not in the Hub copy: `uv run den data raw-bulk`) |

## Layout

Files are organized by **role**, so the path says how a file may be used. The catalog (`den/catalog.py`) refuses any
entry whose path doesn't match its role. Training reads only `data/clean/`, built from `train/` and `sources/train/`.

```
data/
├── train/                  frozen training records, one file per Kev 1.0 stage
│   ├── core.jsonl              stage 1  12,576  public classification + policy pairs + rule structures
│   ├── dates-unknowable.jsonl  stage 2   1,425  date policies + deciding sentence removed
│   ├── documents.jsonl         stage 3   5,219  CFPB complaint narratives (up to ~7k tokens)
│   ├── skills.jsonl            stage 4   6,000  long policies, trade-offs, probability, multi-hop, dates, judging, abstention
│   ├── devtools.jsonl          stage 4   5,320  code review, commit type, flaky tests, content safety
│   └── sources/<skill>/<ds>.jsonl  normalized from sources/train (make data-normalize)
├── calibration/            fit the temperature only
│   ├── core.jsonl              in-distribution
│   ├── heldout.jsonl           held-out sources
│   └── sources/<skill>/<ds>.jsonl  at most 500 records carved from each trainable source's train
├── dev/                    model selection: core, documents, skills, devtools, transfer, probes, breadth, binding, semif
│   └── sources/…               normalized dev splits, trainable and eval-only sources
├── test/                   locked, read once per release candidate: same suites as dev/
│   └── sources/…               normalized test splits
├── clean/                  cleaned, deduplicated copies of all of the above (make data-clean): what runs read
├── manifests/              Kev's manifest for each suite (sources, revisions, counts)
└── sources/
    ├── train/              raw, may become training records
    │   ├── intent/             banking77 · massive-en
    │   ├── topic/              ag-news · trec · dbpedia14
    │   ├── reading/            boolq · multinli
    │   ├── sentiment/          sst5 · yelp-full · amazon-reviews · imdb
    │   ├── safety/             aegis · aart-filter
    │   ├── knowledge/          arc · openbookqa · commonsenseqa · qasc
    │   ├── tools/              glaive-function-calling
    │   ├── documents/          cfpb-complaints                          (raw-bulk)
    │   └── devtools/           commitpackft · codereviewer · flakeflagger (raw-bulk)
    └── eval/               raw, NEVER trained on
        ├── transfer/           qnli · paws · sciq · mmlu · mmlu-pro · emotion · tweeteval-offensive
        ├── devtools/           when2call · prompt-injection-deepset · prompt-injection-gandalf
        └── breadth/            Decision Index areas (knowledge, language, retrieval, tools, arts)
```

Some repositories can never be trained on, and the catalog refuses to mark any of them trainable: `tasksource/*`
(backs Kev's private tasksource-heldout-v1), CUAD (longdoc-v1), SNLI and Rotten Tomatoes (Kev excluded them), and MMLU
and MMLU-Pro.

### Our names, Kev's names

| Ours | Kev |
|---|---|
| `core` | `decision-v7` |
| `dates-unknowable` | `night2/dates_unknowable` |
| `documents` | `documents-v1` |
| `skills` | `hard-v1` |
| `devtools` | `devtools-v1` |
| `transfer` | `transfer-v4` (its test is Kev's headline number) |
| `probes` | `transfer-v9` (date deadlines, MMLU, MMLU-Pro) |
| `heldout` | `round3/transfer-r3` |

Eval-only extras beside Kev's suites: `make data-breadth` rebuilds Kev's breadth-v1 panel (14 public datasets) from the
pinned raws, byte for byte against Kev's published sha256, into `data/{dev,test}/breadth.jsonl`; `binding` and `semif`
(Kev's role-binding diagnostic and SemIf's authored decisions) come with Kev's suites.

## Normalized sources

`make data-normalize` turns every trainable and eval-only raw source (17 trainable, 9 eval-only) into the same
canonical record format as Kev's suites, in about 2 minutes. The output is deterministic: the same inputs always give
byte-identical files. Each output's counts, label distribution and sha256 go to `reports/normalize.json`, which
`make data-audit` checks.

- **Text:** NFC Unicode and tidy whitespace. Literal `\n` and `\"` escapes are undone (Yelp, AG News), HTML entities
  are decoded, including AG News' broken `#39;` and `quot;`, and `<br />` tags become newlines (IMDB, Aegis). Datasets
  that are pre-tokenized, like SST-5, are left as written.
- **Questions:** each source keeps its native label set. For the sources Kev used, the instructions are Kev's
  wording. Multiple-choice options get neutral keys (`opt_1`…). SciQ's options and Glaive's functions are put in a
  seeded order, so the answer's position carries no signal.
- **Cleaning:** unlabeled or invalid rows are rejected; Aegis prompts that are REDACTED or match AART are rejected, as
  Kev does; records whose identical question carries different labels are dropped (mostly Glaive and Aegis); a
  repeated wrong option collapses to one copy, and an item whose correct answer is repeated is dropped; every
  duplicate question is kept once, in the most held-out split it appears in.
- **Splits:** native splits where they exist (MultiNLI: matched for dev, mismatched for test). BoolQ, CommonsenseQA and
  QASC have unlabeled tests, so their validation split becomes the test. A missing dev split is carved from train by a
  hash of the source text, so every question about one document lands in the same split. Dev and test are capped at
  5,000 records each, and a calibration split (at most 500 records, 5% of train) is carved from every trainable source.
- **No leakage:** matches use both the rendered state and the raw-text provenance (`_meta.text_sha256`, the same hash
  Kev records). Test drops anything in Kev's train, calibration or dev; dev drops anything in Kev's train or
  calibration, or in any test; train drops anything in any calibration, dev or test.

## Audit

`make data-audit` parses every record against the `/v1/systemone` schema, then checks it for training: counts against
Kev's manifests, duplicates, conflicting labels, leakage across train, calibration, dev and test, label balance, token
lengths, and that every raw source opens. The report is `reports/data-audit.json`. Current result: **0 errors** over
the Kev suites and all 69 normalized files.

- All 42,866 suite records are valid, and every count matches Kev's manifests. No training question or state appears
  in any calibration, dev or test file. The longest training state is 7,329 tokens, under Kev's 7,552 limit.
- **Exact duplicates:** `train/core` has 420, mostly from Kev's rule-structure generator. `make data-clean` drops them
  in `data/clean` only, so the pinned files stay byte-identical to Kev 1.0.
- **Unknowable probes:** `probes` and `calibration/heldout` repeat some questions with different labels on purpose:
  Kev's *unknowable* items, where the deciding fact is missing, scored on confidence, not accuracy.
- **Not fully held out:** 18 of the 22 unknowable families in `probes` and `calibration/heldout` come from generators
  that stage 2 (`train/dates-unknowable`) trains on, so T is never fitted on `calibration/heldout` by default. The
  `transfer` test is fully out of domain.

## Licences

Every source's licence is recorded with its evidence in `den/licences.py`. `make data-licences` writes them into
`data/`: `README.md` as the dataset card, `LICENSES.md` with each file's sources, `LICENSES/` with per-source terms and
licence texts (`licences/` holds the canonical texts), and a `SOURCE-LICENSE.md` beside every dataset saying what its
files hold, under which licence, and whether the uploaded copy has them.

`make data-upload` never uploads sources whose terms forbid redistribution (Yelp, Amazon reviews, the raw HellaSwag
and RouterBench files), and keeps unlicensed ones out of public copies. Kev's `core` suite holds Yelp and Amazon
reviews, so `make data-download` rebuilds it, and every other left-out file, from its pinned original.
`den train --max-licence share-alike` trains only on records whose sources are at most that class; the model card
lists the classes a run trained on.
