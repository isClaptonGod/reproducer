# reproducer

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
