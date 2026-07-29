# Self-Evolving ML Scientist

This branch adds a provenance-first control plane around FML-bench. It learns
the checked-out agent, task, metric, and baseline contracts into a typed
knowledge graph; freezes that graph for each arm; plans one task-conditioned
conditional research-and-paper graph at Stage 3; composes node-local atomic
policies from visible evidence; gives every node a type-specific inner execution/review
loop; and emits paper-ready data tables and vector figures without fabricating
missing experiments.

## Outer stages

0. Freeze retrievable memory and the four skill registries.
1. Freeze research, evaluation, budget, and publication claim contracts.
2. Retrieve stage-eligible state, proposal, selection, debugging, memory, and diversity policies, plus reviewers and skills.
3. Plan the complete experiment-to-paper DAG and every task-local OR tree.
4. Execute dependency-ready candidates.
5. Apply node-specific metrics, scientific reviewers, and integrity gates.
6. Advance, refine, replicate, debug, backtrack, or create a versioned amendment.
7. Gate completed transitions, distill skills, assess contextual utility, and govern research, research-review, writing, and paper-review skills.

Stage 3 produces one conditional pipeline per research job. A complete baseline
agent is retained as a provenance bundle and known-compatible reference
composition, but it is not the selectable unit for research nodes. Instead,
state, proposal, selection, debugging, memory, and diversity mechanisms are
typed `Policy` nodes. Each executable research node carries role-specific
candidate pools, capability requirements, and a frozen metric-visibility
contract. The controller selects one policy per required role and rejects
compositions with missing state or conflicting capabilities. Full-pipeline
variants remain available only for explicit, preregistered pipeline ablations.

Every selected policy must produce a trace row containing its role, whether it
was applied, a `PASSED`, `FAILED`, `NOT_APPLIED`, or `INCONCLUSIVE` status,
role-specific gate checks, and evidence artifact references. A node-level pass
does not silently promote every component. Only explicit applied-policy traces
are aggregated into future policy success or failure counts, and those counts
remain separated by node type so hypothesis evidence cannot silently become
code or experiment evidence.

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
- source-grounded JSON and Markdown dossiers for every agent and task, including
  audited control-method line numbers, effective parameters, failure hypotheses,
  prompts, commands, target-file boundaries, auxiliary metrics, and baseline hashes;
- a strictly separated `published_prior/` evidence layer transcribing the
  current paper's 108 agent-task aggregate cells, seven-agent summary, twelve
  process correlations, eighteen task cards, and six statistical SVG charts;
- a provisional paper outline, methods section, and locked claim registry;
- a separate manuscript-quality protocol with six blocking integrity gates,
  seven weighted review criteria, five blinded reviewer roles, and matched
  one-shot/gated/full-pipeline writing ablations;
- a preregistered pilot plus balanced 7-agent x 8-task x 3-trial FML-Lite
  confirmatory matrix, with every independent trial stored in a separate result
  root so the upstream scorer cannot silently select the wrong replicate.
- a frozen retrieval snapshot, append-only skill-evolution event ledger, and
  four separate project-local registries for research execution, research
  review, paper writing, and paper review.
- `benchmark_contracts/`, where validation-visible, post-stage, protected-final,
  paper-review, and operator-only metric views are kept distinct;
- `knowledge_graph/knowledge_graph.json`, the canonical typed memory migrated
  from the prior registry, source-grounded agent/task dossiers, decomposed
  baseline-policy ontology, metric audit,
  benchmark contracts, paper-quality protocol, and skill registry;
- task-conditioned Stage-3 plans bound to the frozen graph content hash;
- `reports/adaptive_pipeline_checkpoint.html`, a self-contained architecture
  and memory-migration checkpoint report.

Skill maturity is evidence-gated: source audits and newly distilled candidates remain observations; one later held-out
complete downstream no-regression pass permits provisional success; two
materially distinct held-out complete passes permit repeated success. An authoring episode cannot validate its own candidate. Contradictions
remain in history and roll the active rule back. A captured global-skill base
checksum must still match before promotion, and an experiment arm cannot see
memory or skill writes made after its snapshot was frozen.

## Learn additional benchmarks and metrics

The checked-out FML source is learned and verified during bootstrap. A new
user-supplied benchmark manifest enters as an observation and cannot become
active merely because its JSON shape is valid:

```bash
python3 -m ml_scientist.cli benchmark-learn \
  --source path/to/new_benchmark_contract.json \
  --out artifacts/ml_scientist/bootstrap/benchmark_contracts
```

Activation additionally requires implementation-audit and boundary-test
evidence. Protected-final metrics are structurally forbidden from search
routing. Metric semantics are versioned; a corrected metric supersedes rather
than silently overwrites an already frozen campaign definition.

## Build, validate, and query graph memory

```bash
python3 -m ml_scientist.cli memory-build-graph \
  --artifact-root artifacts/ml_scientist/bootstrap \
  --out artifacts/ml_scientist/bootstrap/knowledge_graph

python3 -m ml_scientist.cli memory-validate-graph \
  --graph artifacts/ml_scientist/bootstrap/knowledge_graph/knowledge_graph.json

python3 -m ml_scientist.cli memory-query-graph \
  --graph artifacts/ml_scientist/bootstrap/knowledge_graph/knowledge_graph.json \
  --query "privacy low diversity limited budget" \
  --type Policy \
  --stage code_modification
```

The old `governance/memory_registry.json` remains only as a compatibility view
and points to the graph as its canonical store. The execution DAG is deliberately
separate from this persistent semantic graph.

Completed decision episodes are appended after an arm and become visible only
to a future snapshot:

```bash
python3 -m ml_scientist.cli memory-append-episode \
  --episodes artifacts/ml_scientist/bootstrap/knowledge_graph/episodes.jsonl \
  --input completed_episode.json
```

Enriched episodes include the user research request and task family, state and
visible evidence before the decision, selected policies, route and state after
the decision, executor-reported or derived token/GPU/wall-time/diversity
telemetry, evidence artifacts, role-level policy traces, and optional structured
learning signals. The post-arm lifecycle is invoked explicitly:

```bash
python3 -m ml_scientist.cli skill-evolve \
  --episodes artifacts/ml_scientist/bootstrap/knowledge_graph/episodes.jsonl \
  --governance-root artifacts/ml_scientist/bootstrap/governance \
  --out artifacts/ml_scientist/bootstrap/skill_evolution
```

The writer uses an informative-trajectory gate and a two-pass analysis/mutation
contract. Each applied policy emits `CREATE`, `PATCH`, or `NONE`; a candidate
must contain a measurable trigger, ordered procedure, applicability branch and
stage, expected effect, acceptance gates, and rollback condition. Exact
duplicates are ignored and close matches become non-active patch versions.
Writes enter only the next frozen snapshot.
When the planned `skill-evolution` node becomes dependency-ready,
`adaptive-runtime-evolve` invokes the same engine, records the four lifecycle
reports as node evidence, and completes or fails the Stage-7 gate without
starting an experiment.

Later held-out validation episodes update utility by research context and stage
using reward relative to a context baseline. A complete no-regression pass can
promote a candidate; a second materially distinct context can establish repeated
success; comparable harmful evidence rolls the affected version back while
retaining its negative history. Protected-final metrics cannot drive this loop.

## Initialize the adaptive Stage 4-6 runtime

Initializing or inspecting runtime state never launches an experiment. The
executor remains a separate explicit action, which preserves the project pause
boundary:

```bash
python3 -m ml_scientist.cli adaptive-runtime-init \
  --plan artifacts/ml_scientist/bootstrap/plans/Privacy_privacymeter.json \
  --graph artifacts/ml_scientist/bootstrap/knowledge_graph/knowledge_graph.json \
  --run-id privacy-adaptive-001 \
  --state artifacts/ml_scientist/adaptive_runs/privacy-adaptive-001/state.json
```

`adaptive-runtime-decide` records a compatible atomic-policy composition from a
JSON list of visible metric evidence. Baseline bundles remain provenance only;
the operator contract contains only the policies explicitly selected for the
current node. The contract is suitable for the text-only Codex CLI during
planning or for the existing `CodeEditor` on declared target files, but still
requires an explicit executor call. `adaptive-runtime-record` accepts executor evidence and moves
the node forward, retries, selects another policy composition, backtracks, or stops for
budget. `adaptive-runtime-amend` routes paper-review errors to writing,
statistics, experiment, replication, metric semantics, or code and invalidates
dependent descendants when required.

For an atomic-policy node, `adaptive-runtime-record` also requires
`--policy-evaluations policy_trace.json`. The file must account for every
selected role. Applied policies cannot receive success or failure credit without
at least one evidence artifact reference. Non-applied policies receive no
credit. Completed episodes expose those explicit role-level outcomes only to a
future frozen graph snapshot.

`adaptive-runtime-record` can additionally accept `--budget-after`,
`--telemetry`, `--state-observations`, and `--skill-validation-evaluations` JSON
files. `adaptive-runtime-export --node NODE --episodes episodes.jsonl` exports
and appends the complete transition required by the skill-evolution engine.

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

## Live campaign, trajectory bridge, and adaptive executor

During a long frozen campaign, refresh the self-contained HTML and its machine-readable tables with:

```bash
python3 -m ml_scientist.cli fml-import-episodes \
  --results benchmark_results/controlled \
  --episodes artifacts/ml_scientist/trajectory_memory/baseline_observations.jsonl \
  --graph artifacts/ml_scientist/bootstrap/knowledge_graph/knowledge_graph.json

python3 -m ml_scientist.cli live-pipeline-report \
  --results benchmark_results/controlled \
  --campaign-dir artifacts/ml_scientist/codex_ssh_protocol/pilot_campaign_v1 \
  --matrix artifacts/ml_scientist/codex_ssh_protocol/run_matrix.csv \
  --graph artifacts/ml_scientist/bootstrap/knowledge_graph/knowledge_graph.json \
  --issues artifacts/ml_scientist/pipeline_diagnostics/issue_registry.json \
  --episodes artifacts/ml_scientist/trajectory_memory/baseline_observations.jsonl \
  --new-pipeline-results benchmark_results/adaptive_pipeline \
  --out artifacts/ml_scientist/reports/live-pipeline
```

Legacy FML summaries become bundle-level observation episodes only. The bridge
does not invent atomic Policy credit or a learning signal. A separately labeled
`adaptive_pipeline` summary can carry the executor's explicit atomic selection,
post-execution assessment, reachability audit, and resource telemetry into an
enriched episode. Planning and assessment are deliberately separate: the
pre-execution operator declares `planned_policy_use`, while PASSED/FAILED credit
and CREATE/PATCH/NONE signals are permitted only after real artifacts exist.

The adaptive executor also checks token, wall-clock, and candidate-step budgets
before calls, and rejects newly added Python functions with no call site in the
allowed target-file set before validation. That static pilot gate catches the
known unreachable-helper failure mode but does not replace the runtime marker
required for confirmatory attribution.

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
blocks are surfaced and excluded. Pairwise agent tables use the same matched
complete seed blocks as the primary uncertainty unit and report the mean and
median paired difference, Student-t interval, Cohen dz, seed-block win/tie/loss
counts, an exact sign test, and Holm family-wise correction across all agent
pairs. A separate task-effect table exposes heterogeneity and task/task-seed
win rates, but labels those repeated-measure views descriptive rather than
pretending every cell is an independent replication. The generated statistical
analysis contract warns that three seeds have weak inferential resolution and
that a non-significant comparison is not evidence of equivalence.

The report also writes `adaptive_opportunity_interactions.csv`. Within each
matched seed and baseline, it subtracts the AdaptiveSearch advantage on the
published sparse-opportunity tasks from its advantage on the published
dense-opportunity tasks. The six baseline interactions receive a separate Holm
correction. The published post-hoc partition supplies frozen hypothesis labels;
only new campaign outcomes enter the estimator.

## Evidence-locked paper package

Bootstrap writes a substantive Methods manuscript, research-question brief,
argument blueprint, claim registry, and table/figure plan. To refresh the paper
handoff after ingesting results independently:

```bash
python3 -m ml_scientist.cli paper-package \
  --artifact-root artifacts/ml_scientist/bootstrap \
  --out artifacts/ml_scientist/bootstrap/paper
```

`paper_readiness.json` checks the human paper configuration, repository
knowledge, frozen protocol, execution preflight, all 182 planned records, all 21
complete agent comparisons, process metrics, scorer sensitivities, empirical
claim gates, and the paper-writing ablation. Until those gates pass, the
manuscript contains real system and protocol prose but keeps the Abstract,
Results, empirical Discussion, and Conclusion explicitly locked.

## Published prior versus new campaign evidence

`python3 -m ml_scientist.cli published-prior --out DIR` writes aggregate data
from arXiv:2605.17373v2 with table-level source locators and file hashes. This
layer can guide Stage-3 hypotheses and Stage-5 review expectations, but it is
never loaded by the new-experiment reporter, never fills missing trials, and
never unlocks a paper result claim. The opportunity-density split is labeled
post-hoc, and the pooled process correlations retain warnings about unadjusted
p-values, dependence among cells, and AUC/final-score definitional overlap.
It also produces a paper-ready prior-evidence brief with admissible wording,
locked claims, diagnostic task rankings, preregistered questions, result-package
requirements, and captions for every published/derived chart.

The source audit additionally freezes a metric-implementation report. It makes
the Unlearning raw/display direction explicit and records two current
paper-versus-scorer differences: exploration uses every persisted step snapshot,
and best-improvement step uses the last exact match. New experiments retain the
official checked-in scorer. The report writes a separate scorer-sensitivity
table containing the first-achieved-best step and valid-only snapshot membership
diagnostics beside the official last-match/all-snapshot definitions. GraphCodeBERT
exploration values are recomputed only under an explicitly named sensitivity
analysis when real snapshots exist; official metrics are never replaced and
metric semantics never change after the campaign begins.

For each task, the Stage-3 method-frontier node receives a non-claim prior:
dense-opportunity tasks start with a greedy/adaptive hypothesis, while
sparse-opportunity tasks start with multi-branch exploration. A matched-budget
counterfactual search family is always required, local frozen evidence may
override the prior, and peer review verifies that published and local evidence
classes never merge.

## Paper-quality evaluation is not an FML metric

The controlled FML comparison removed native paper-writing and automated-review
modules, so manuscript quality cannot honestly inherit an FML score. Bootstrap
therefore writes `paper_evaluation/` as a separate local extension. Its six
hard gates cover traceability, published/local evidence separation, citations,
method provenance, protected-test leakage, and amendment closure. Only papers
passing every hard gate proceed to a blinded five-role, seven-criterion rubric;
scores are never averaged with normalized improvement or process metrics.

The protocol also preregisters three writing arms—one-shot, gated without peer
revision, and the full review/amendment/re-review pipeline—with three independent
drafts per arm and matched evidence, model, and token budget. The generated
templates contain no scores until real manuscripts and reviews exist.

After reviewers fill `review_scores.csv`, `hard_gate_results.csv`, and optional
`issue_dispositions.csv`, reduce them without modifying any manuscript:

```bash
python3 -m ml_scientist.cli paper-evaluation-report \
  --data paper_review_data \
  --out artifacts/ml_scientist/bootstrap/paper_evaluation
```

The reducer requires all 315 criterion-review rows and all 54 manuscript hard
gates, verifies consistent blind IDs, blocks unresolved major issues, aggregates
reviewer medians before weighting, verifies 45 distinct review run IDs, seeds,
randomized orders, report paths, and report hashes, reports ICC(2,k) agreement,
and computes the three matched writing-arm contrasts with intervals, effect
sizes, sign tests, and Holm correction. A complete evaluation can still contain failed manuscripts;
completion means the comparison was fully observed, not that every arm passed.
Finalization additionally requires `paper_release_decision.json`: a named human
editor must select a passing W2 manuscript, record a rationale, and point to a
real final-integrity report.
