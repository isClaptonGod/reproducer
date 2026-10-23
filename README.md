# Can agents learn to need less history?

This repository implements a controlled experiment on **Context Language Models (CLMs)**:
can an agent learn engineering actions that reduce the cost of continuing its work after
conversation history and dedicated memory are removed?

It supports **Qwen3.6-27B** through an OpenAI-compatible endpoint and **Claude 4.6 Sonnet**
through Anthropic Messages. Each backbone gets its own learned action policy; model weights
remain fixed. Installation, tests, and the offline demo make no real model calls.

| Condition | Intervention |
| --- | --- |
| `standard_clm` | Editable conversation file, standard instructions |
| `handoff_clm` | Same CLM, instructed to produce good handoffs and persistent state |
| `learned_clm` | Same CLM, engineering-action policy learned from fresh-continuation rewards |

All three have the same tools, context budget, primary call budget, engineering-window
budget, container resources, and continuation limits. Engineering calls count against the
primary budget. This is an independent minimal implementation of CLM's editable-context
mechanism, **not an exact reproduction of the published CLM experiments**.

## Quick start

**Windows / Git Bash users:** follow [the step-by-step Windows and WSL guide](docs/windows.md).
`/workspace/reproducer` is a cloud-only path. Native Windows virtual environments use
`.venv/Scripts/python.exe`; the `.venv/bin/python` commands below are for Linux and WSL.

Python 3.11+ is sufficient for the core harness and offline validation; there are no runtime
Python dependencies. Real agents execute shell commands only inside Docker.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m clm_experiment demo --out runs/demo
```

The demo trains four action-policy episodes, then evaluates three conditions on two held-out
**synthetic fixture tasks**. `runs/demo/report.json` labels these synthetic and inconclusive.
It validates the pipeline; it provides no evidence for the hypothesis or either real model.

Docker integration checks run separately:

```bash
docker pull python:3.12-slim
CLM_TEST_DOCKER=1 .venv/bin/python -m unittest discover -s tests -p test_docker.py -v
```

## Benchmarks

### Mathematical optimization

Four evaluation-only instances have independent, host-side numeric checkers:

| Problem | Definition used here | Objective |
| --- | --- | --- |
| Circle packing | 26 possibly unequal circles in the unit square | Maximize sum of radii |
| Erdős' minimum overlap | 200 equal-width bins on `[0,2]`, heights in `[0,1]`, integral 1 | Minimize maximum correlation with the complementary, zero-extended function |
| Min–max distance ratio | 16 distinct points in the plane | Minimize maximum / minimum pairwise distance |
| Heilbronn triangle | 11 points in the unit-area triangle `(0,0), (2,0), (0,1)` | Maximize the minimum area over all triples |

These choices follow the relevant AlphaEvolve problem definitions. The Erdős representation
is a finite search family, not a proof of a global optimum. Checkers use floating-point
feasibility with a `1e-9` tolerance, not formal proofs. Milestones in the generated manifest
are explicit experimental choices; freeze them before testing, using separate pilot data
if calibration is needed.

```bash
docker build -t clm-work:0.1 .
.venv/bin/python -m clm_experiment prepare-math --out data/math --image clm-work:0.1
```

The work image supplies NumPy and SciPy. Omitting `--image` uses the plain Python image for
stdlib algorithms. The real runner pins installed image IDs before execution.

In a cloud environment with a custom trusted TLS root, supply its public CA bundle without
disabling verification: `docker build --secret id=ca_bundle,src=/etc/ssl/certs/ca-certificates.crt
-t clm-work:0.1 .` (one shell command). If Docker's home directory is read-only, set
`BUILDX_CONFIG=/tmp/clm-buildx` for that command.

There is **one primary run per problem × method × backbone**, totaling 24 math primary runs.
Checkpoints and diagnostic continuation forks are repeated measurements of those runs,
never extra independent optimization trials or candidates for best-run selection. Math
instances are never used for policy training; their findings remain descriptive.

### BrowseComp-Plus

Use the [official dataset/decryption instructions](https://github.com/texttron/BrowseComp-Plus)
to obtain decrypted queries and evidence qrels. Export the fixed corpus locally as JSONL
with `docid` (or `id`), `text`, and optional `title`.

```bash
.venv/bin/python -m clm_experiment index-corpus \
  --corpus data/corpus.jsonl --out data/corpus.db
.venv/bin/python -m clm_experiment prepare-browse \
  --queries data/browsecomp_plus_decrypted.jsonl \
  --qrels data/qrel_evidence.txt --corpus-db data/corpus.db --out data/browse
```

Queries require `query_id`, `query`, and `answer` (string) or `answers` (list). The importer
creates a deterministic 30% training / 70% held-out split by query. Edit task `group` to keep
known related questions together; exact instance/group overlap is checked, but paraphrased
duplicates need curation. Answers, relevance labels, and the corpus index remain host-side;
agents access documents through the same read-only search tool.

The built-in retriever uses SQLite FTS5 BM25, top 5, at most 2,400 characters per document.
The local progress score is `1` for an exact answer match, otherwise `0.5 × evidence-recall`.
Correct answers need not cite every gold document. Partial evidence is checkpointed after
each producer action; agents are instructed to save citations incrementally.
**This is an experimental proxy, not official BrowseComp-Plus grading or its 512-token
retriever setting.** Freeze it across methods. For official final-answer grading, export
each model/method's primary outputs and run the upstream evaluator:

```bash
.venv/bin/python -m clm_experiment export-browse \
  --runs runs/main --out runs/browse-official
# In the upstream checkout, once for each exported model/method directory:
python scripts_evaluation/evaluate_run.py --input_dir /absolute/path/to/exported/directory
```

Upstream final grading is separate from the proxy-based continuation measurements; it does
not retroactively make those measurements official benchmark results.

### EdgeBench repository tasks

Use [ByteDance-Seed/EdgeBench and SForge](https://github.com/ByteDance-Seed/EdgeBench) to fetch
task definitions, pull work/judge images, and start the hidden-test judge:

```bash
# Install SForge separately, using the version matching your task definitions.
sforge fetch-tasks edgebench
sforge pull --task ad_placement_optimization --registry seededge
sforge pull --task integer_compression_codec --registry seededge
sforge serve
```

Copy `configs/edge-selection.example.json` to a local selection file; replace task JSON paths
and image tags with actual fetched/pulled values. Choose disjoint train/test task groups and
add the desired tasks explicitly. The example illustrates the schema; **it does not identify
the paper's EdgeBench-10 subset**. Calibrate task-specific milestones before evaluation.
Use `score_0_100` only when the task judge supplies it.
For noisy performance judges, set each selection entry's `quality_tolerance` using independent
pilot measurements; it defaults to a strict `1e-8`. Do not tune tolerances on held-out results.

```bash
.venv/bin/python -m clm_experiment prepare-edge \
  --selection configs/edge-selection.local.json --out data/edge
```

The importer records immutable work-image IDs and task-definition hashes. The runner uses
SForge's register, submit, and result APIs, with hidden judging outside agent containers.
Configure `SFORGE_JUDGE_URL` and `SFORGE_ADMIN_SECRET` in the **host process only**, using
the admin secret from your judge server. Judge credentials and tests never enter workspaces.

Supported tasks must be filesystem-restartable repository tasks, include `python3` and `tar`
in their work images, and work without agent-container networking. Stateful games, GPU tasks,
symlink-containing workspaces, and online-service tasks need another backend or explicit
preparation. Increase configured resources and budgets for demanding tasks. The default
256-call setting is for development; it does not reproduce the paper's 12-hour setting.

## Run the two backbones

`configs/experiment.json` declares both model IDs and a 32K context budget. For Qwen, use a
compatible vLLM server:

```bash
vllm serve Qwen/Qwen3.6-27B --max-model-len 65536 --reasoning-parser qwen3
```

Set `QWEN_BASE_URL` if needed. Local serving may omit authentication; hosted serving should
use `require_key: true` and `QWEN_API_KEY`. Set `ANTHROPIC_API_KEY` securely for Sonnet.
The harness never substitutes another model on failure. Presence checks print no secrets:

```bash
.venv/bin/python -m clm_experiment doctor --config configs/experiment.json
```

Default token counting uses vLLM's `/tokenize` or Anthropic's count-tokens API. For servers
without a compatible counter, `token_counter: "bytes"` supplies a conservative byte bound;
label that deviation, since it is not an actual 32K tokenizer budget. Anthropic does not
support seeded Messages sampling; logs distinguish this from seeded OpenAI-compatible calls.

Train each policy on the training portions of BrowseComp-Plus and/or EdgeBench:

```bash
.venv/bin/python -m clm_experiment train --config configs/experiment.json \
  --tasks data/browse/tasks.jsonl --tasks data/edge/tasks.jsonl \
  --model qwen27b --episodes 32 --out policies/qwen27b
.venv/bin/python -m clm_experiment train --config configs/experiment.json \
  --tasks data/browse/tasks.jsonl --tasks data/edge/tasks.jsonl \
  --model sonnet46 --episodes 32 --out policies/sonnet46

.venv/bin/python -m clm_experiment run --config configs/experiment.json \
  --tasks data/browse/tasks.jsonl --tasks data/math/tasks.jsonl \
  --tasks data/edge/tasks.jsonl --policy-dir policies --out runs/main
```

The grid is randomized and saved before execution. Existing output directories cannot be
overwritten. Failed cells remain in the planned grid and block a complete-result claim.
Use `--models` and `--conditions` for an explicit pilot; do not reuse held-out pilot instances
for confirmation or retrospectively select favorable runs.

## Measurement and learning

At each first observed crossing of a **fixed absolute score milestone**, the runner:

1. Exports the task filesystem and dedicated memory separately.
2. Removes dedicated notes, conversation files, caches, and disallowed file types. Detects
   common transcript relocation patterns and checks that filtering preserved the task score.
3. Runs fresh and retained-context continuations from the same functional filesystem in
   separate new containers, measuring work to the next fixed milestone.
4. Gives the producer its assigned, fixed-budget engineering window.
5. Repeats the paired probes after that action. A meaningful solution-score change during
   engineering invalidates the measurement instead of counting ordinary task progress.

Fresh probes receive the original task, neutral continuation instructions, and functional
artifacts. They receive no producer context, dedicated notes, condition label, learned arm,
live process state, or earlier probe outcomes. The producer never sees probe results.
Retained probes receive the producer's editable CLM context and dedicated memory, but also
restart their container, isolating the memory contrast from live processes.

Let `C_f` and `C_r` be fresh and retained continuation cost. The policy learns from:

```text
gap_gain = (C_f_before - C_r_before) - (C_f_after - C_r_after)
fresh_gain = C_f_before - C_f_after
reward = min(gap_gain, fresh_gain) - engineering_cost
```

The learner is a UCB bandit over reproducible entrypoints, executable verification,
structured functional state, and restartable checkpoints. It learns rewards per task
family and freezes before testing. Unseen math families use pooled training rewards.
This learns intervention selection for a fixed backbone; it does not fine-tune weights
or learn an unrestricted vocabulary of actions. Untrained policies cannot be labeled learned.
The fresh-gain cap prevents rewards for merely making retained-history continuation slower.

Logs include model input/output/cache tokens, calls, tools, inspection bytes/calls, tool and
wall time, judging time, and engineering costs. Shell calls also count as inspection.
Cost weights are fixed in the config. USD is computed only if current prices are supplied
as `usd_per_million_tokens: {input, output, cache_read, cache_write}`; otherwise zero means
**unpriced**, not free. No FLOP or prefix-reuse PFLOP estimates are invented.

Failed continuations are censored and charged at least `probe_cost_cap + failure_penalty`,
so early failure cannot appear cheap. Overshot milestones do not count as zero-cost successful
probes. Missing, invalid, and unreached stages are reported rather than silently dropped.

Policy-training cost includes training producers, actions, and all diagnostic probes.
Reports charge that cost divided by `training_amortization_tasks` per held-out task (default
**1**, a conservative single-task deployment). Freeze the horizon before testing. Evaluation
diagnostic probe costs are recorded separately from hypothetical deployment burden.

## A result that can reject the hypothesis

```bash
.venv/bin/python -m clm_experiment analyze --runs runs/main --out runs/analysis.json \
  --minimum-effect 1.0 --quality-margin 0.05 --min-groups 8 --max-censoring 0.1
```

Read [the protocol](docs/protocol.md) before confirmatory runs. Reports compare the learned
method with **both** controls, separately for each backbone and suite. Task groups are
bootstrapped, never checkpoints, with multiplicity-adjusted intervals for cost savings,
retained-adjusted history-dependence reduction, and task quality.

An interval ruling out the minimum required gain, or establishing excess quality loss,
can reject the preregistered **operational hypothesis**. Missing significance alone is
inconclusive. The four one-run math instances remain descriptive.

Transcript detection is heuristic: paraphrases, encoding, comments, and unused fields can
hide memory. Confirmatory decisions require an artifact-review attestation matching run-file
checksums. Without review, enough independent groups, and usable measurements, the report
stays inconclusive.

## Sources

- [Context Language Models](https://github.com/facebookresearch/context-language-models),
  arXiv:2609.37725. No upstream implementation is vendored.
- [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus), arXiv:2508.06600.
- [AlphaEvolve definitions](https://github.com/google-deepmind/alphaevolve_results), notebook
  sections B.5, B.8, B.9, B.12. Checkers are independently implemented here.
- [EdgeBench / SForge](https://github.com/ByteDance-Seed/EdgeBench), arXiv:2607.05155.

Keep downloaded benchmark data, credentials, and generated results in ignored `data/`,
`runs/`, and `policies/` paths.
