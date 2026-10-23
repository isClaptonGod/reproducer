# Analysis and validity protocol

Freeze task/version hashes, groups, milestones, targets, budgets, cost weights, sampling,
minimum effect, quality margin, censoring limit, and training amortization before testing.
The default configuration is a starting protocol, not a power analysis. Eight independent
groups is a reporting gate, not a guarantee of power; estimate variance on separate pilots.

## Estimands

Average continuation measurements across each task's preregistered stages, then form:

- **Net fresh-start burden:** producer and engineering cost up to each checkpoint, plus that
  checkpoint's fresh continuation cost, averaged across stages, plus the preregistered share
  of policy-training cost. Primary work after a checkpoint is not counted a second time.
- **Restricted history-dependence gap:** post-action fresh minus retained continuation cost
  from identical functional snapshots. Fixed budgets and failure penalties restrict this
  estimand; it does not estimate unlimited completion time.
- **Quality:** whether the original task's independent final target was achieved.

Positive control-minus-learned burden and gap differences favor learning. Positive
learned-minus-control quality differences favor learning. Average task variants within
their declared original-task group before bootstrapping.

Gap contrasts use matched stages with valid measurements in both conditions. Burden also
penalizes invalid/unreached stages. Any missing/invalid stage blocks confirmation, so a
conditional cost comparison cannot silently discard unsuccessful tasks. Excessive censoring
also makes results inconclusive. Publish the reachability and quality outcomes alongside costs.

## Decision rule

For each backbone × non-math suite × control, compute task-group bootstrap intervals.
Familywise alpha is 0.05, Bonferroni-adjusted over contrasts and their three endpoints.
Both burden savings and gap reduction use the preregistered minimum effect in cost units.
Changing the estimator after inspecting signs is a new exploratory analysis.

Subject to complete runs, sufficient groups, artifact review, valid checkpoints, and an
acceptable censoring rate:

- **Support:** lower bounds exceed minimum savings and gap reduction, while the quality
  lower bound stays above the negative noninferiority margin.
- **Reject the operational hypothesis:** an upper bound rules out the minimum savings or
  gap reduction, or the quality upper bound establishes loss beyond the margin.
- **Inconclusive:** everything else, including insufficient precision.

Rejection concerns this action library, backbones, datasets, budgets, and effect threshold;
it cannot rule out every possible engineering action or learning algorithm. Math results
remain descriptive. Checkpoints and forks from one math trial are not independent replications.

## Functional-artifact review

Inspect `stage-*/pre/functional`, `stage-*/post/functional`, and their audit manifests. Confirm
that new code is exercised, structured state is consumed, tests check task behavior, and
other retained files are functional outputs. Reject unused JSON fields, encoded reasoning,
conversation extracts, or comments serving as hidden handoffs. Inspect both controls too.
Automated filename, message-envelope, and n-gram checks are triage rather than a guarantee.

Write a local review file naming every `run.json` relative to the analysis input directory,
with the SHA256 of its exact bytes:

```json
{
  "reviewer": "researcher identifier",
  "functional_artifacts_only": true,
  "reviewed_runs": {
    "model-directory/task-directory/standard_clm/run.json": "sha256-of-run-json"
  }
}
```

Pass `--artifact-review runs/review.json` to analysis. The runner never fabricates this
attestation. If violations are found, retain their failed results and register a follow-up;
do not replace an unfavorable result with another seed.

## Remaining threats

Milestones are first observed crossings and may be overshot between judge calls. Reachability
can vary by method; reports expose that selection effect. Overshot zero-cost probes are
invalid. Unchanged score is weaker than semantic equivalence, making artifact review essential.

The handoff condition is an instruction-based control, not an equally trained handoff policy.
Useful additional preregistered controls include a learned handoff policy, shuffled rewards,
removing the retained-gap term, and varying training amortization horizons.

Seed support does not ensure identical provider randomness. Anthropic sampling, performance
judges, and floating-point optimizers remain nondeterministic. Sample more non-math task
groups rather than secretly adding math restarts.

BrowseComp-Plus uses an exact-match/evidence proxy, so missing aliases can mark correct
answers wrong; report official final grading separately. Pin SForge/task versions. Stateful,
network-dependent, and GPU-specific tasks require another backend.
