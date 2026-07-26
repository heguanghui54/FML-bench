# Self-Evolving ML Scientist

This branch adds a provenance-first control plane around FML-bench. It learns
the checked-out agent, task, metric, and baseline contracts; plans one unified
research-and-paper dependency graph at Stage 3; gives every node a type-specific
inner execution/review loop; and emits paper-ready data tables and vector
figures without fabricating missing experiments.

## Outer stages

0. Freeze retrievable memory and the four skill registries.
1. Freeze research, evaluation, budget, and publication claim contracts.
2. Select ML methods, baseline-agent search operators, reviewers, and skills.
3. Plan the complete experiment-to-paper DAG and every task-local OR tree.
4. Execute dependency-ready candidates.
5. Apply node-specific metrics, scientific reviewers, and integrity gates.
6. Advance, refine, replicate, debug, backtrack, or create a versioned amendment.
7. Freeze evidence and govern research, research-review, writing, and paper-review skills.

The final paper-writing node exists from Stage 3 but is locked behind the
frozen final evidence bundle. Provisional outline and method-description nodes
may run earlier, but they carry a no-results-claims gate.

After final drafting, publication is also dependency-gated: blocking
pre-review integrity and AI-failure-mode audit, full multi-role peer review,
issue-by-issue amendment triage, revision, focused re-review, optional
re-revision, an independent final integrity pass, finalization, and a process
record. Review-discovered empirical errors point to a new graph version rooted
at the affected code/experiment node; they are never patched only in prose.

## Protected-test boundary

The visible search loop uses validation results. The best candidate is frozen
before one protected test execution, and that result never returns to the
search controller. After test exposure, a code change requires a new hidden
evaluation or must be reported as post-hoc.

## Bootstrap the auditable knowledge and paper-data layer

```bash
python3 -m ml_scientist.cli bootstrap \
  --out artifacts/ml_scientist/bootstrap \
  --results benchmark_results
```

This writes:

- source-hashed cards for all seven agents;
- contracts and baseline values for all eighteen tasks;
- eighteen task metrics and twelve process metrics;
- one Stage-3 research-and-paper graph per task;
- CSV tables and colorblind-safe SVG inventory figures;
- an experiment-data status file that refuses to invent performance charts
  when no real `summary.json` records exist.
- a baseline/evaluation handbook and source-hashed provenance catalog;
- a strictly separated `published_prior/` evidence layer transcribing the
  current paper's 108 agent-task aggregate cells, seven-agent summary, twelve
  process correlations, eighteen task cards, and four statistical SVG charts;
- a provisional paper outline, methods section, and locked claim registry;
- a preregistered pilot plus balanced 7-agent x 8-task x 3-trial FML-Lite
  confirmatory matrix, with every independent trial stored in a separate result
  root so the upstream scorer cannot silently select the wrong replicate.
- a frozen retrieval snapshot, append-only skill-evolution event ledger, and
  four separate project-local registries for research execution, research
  review, paper writing, and paper review.

Skill maturity is evidence-gated: source audits remain observations; one real
complete downstream no-regression pass permits provisional success; two
materially distinct complete passes permit repeated success. Contradictions
remain in history and roll the active rule back. A captured global-skill base
checksum must still match before promotion, and an experiment arm cannot see
memory or skill writes made after its snapshot was frozen.

The default protocol contains `SET_MODEL` and is therefore a non-runnable
template. Freeze it with one model/provider before execution:

```bash
python3 -m ml_scientist.cli protocol \
  --out artifacts/ml_scientist/protocol \
  --model YOUR_FIXED_MODEL --provider OpenAI \
  --results benchmark_results/controlled

python3 -m ml_scientist.cli preflight \
  --out artifacts/ml_scientist/preflight \
  --model YOUR_FIXED_MODEL --provider OpenAI
```

`run_commands.sh` uses matched step budgets, explicit controller seeds, unique
workspaces, one-way protected tests, and a separate result root per trial. The
FML scorer now supports `--suite lite` and `--tasks ...`, allowing the published
Lite subset and engineering pilots to be scored without pretending that missing
full-suite tasks were run.

Prefer the resumable campaign runner to executing the shell file directly. It
refuses an unfrozen `SET_MODEL`, skips result-complete rows, records command
hashes/exit codes/logs in an append-only event ledger, and continues to make
failed executions visible:

```bash
python3 -m ml_scientist.cli campaign \
  --matrix artifacts/ml_scientist/protocol/run_matrix.csv \
  --logs artifacts/ml_scientist/campaign-logs \
  --phase pilot --dry-run
```

## Run data ingestion after experiments

```bash
python3 -m ml_scientist.cli report \
  --results benchmark_results \
  --metric-reports metric_reports \
  --out artifacts/ml_scientist/experiment-report
```

Every ingested record retains its `summary.json` path and SHA-256 digest. Group
statistics report attempted and successful counts, FML baseline-fallback credit,
normalized improvement, sample standard deviation, and a two-sided 95% Student-t
interval only when replication exists. Performance figures are emitted only for
real replicated groups. The twelve upstream process metrics are ingested from
their scorer CSVs into a long provenance table, replicated statistics, and one
separate uncertainty plot per metric so incompatible units are never combined
on a single axis. Overall agent estimates first average all tasks within each
complete trial block and then compute uncertainty across trial means; incomplete
blocks are surfaced and excluded. Pairwise agent tables use matched task-trial
differences and are explicitly marked descriptive and unadjusted for multiplicity.

## Published prior versus new campaign evidence

`python3 -m ml_scientist.cli published-prior --out DIR` writes aggregate data
from arXiv:2605.17373v2 with table-level source locators and file hashes. This
layer can guide Stage-3 hypotheses and Stage-5 review expectations, but it is
never loaded by the new-experiment reporter, never fills missing trials, and
never unlocks a paper result claim. The opportunity-density split is labeled
post-hoc, and the pooled process correlations retain warnings about unadjusted
p-values, dependence among cells, and AUC/final-score definitional overlap.

For each task, the Stage-3 method-frontier node receives a non-claim prior:
dense-opportunity tasks start with a greedy/adaptive hypothesis, while
sparse-opportunity tasks start with multi-branch exploration. A matched-budget
counterfactual search family is always required, local frozen evidence may
override the prior, and peer review verifies that published and local evidence
classes never merge.
