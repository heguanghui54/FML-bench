# Pipeline experiment pause checkpoint — 2026-07-29

Recorded at: 2026-07-29T23:34:35+08:00

Status: `PAUSED_BY_USER`

Resume policy: do not restart any ML experiment, Ubuntu GPU workload, i4h workload, or cloud-compute action until the user explicitly asks to continue this objective.

## Current confirmatory experiment

- Phase: `confirmatory_heldout_transfer`
- Interrupted row: `adaptivesearch × Generalization_domainbed_officehome × trial01`
- Seed: `1103`
- Budget profile: `matched-v1`
- Outcome: `ABORTED_OPERATOR_INTERRUPT`
- Paper-result eligibility: `false`
- Protected test exposed: `false`
- Performance score available: `false`
- GPU compute activated: `false`
- Last log root: `artifacts/ml_scientist/post_fix/heldout_transfer_confirmatory/campaign_logs_firstarm_real_retry06`
- Canonical detailed checkpoint: `artifacts/ml_scientist/post_fix/heldout_transfer_confirmatory/pause_checkpoint_20260729_224044.json`

The candidate at interruption proposed an EMA-network modification for ERM. Runtime evidence observed `ERM.__init__`, but mandatory activation markers for `ERM._update_ema`, `ERM.update`, and `ERM.predict` were still missing. The interrupted attempt must not be treated as performance evidence and must restart from the frozen matrix row with a new result root.

## Verified stop state

- No local FML campaign or adaptive-runtime process was active at this checkpoint.
- The detailed stop checkpoint recorded zero remote FML processes and zero remote GPU-compute processes.
- i4h compute was inactive. Its local static dashboard server may remain open; it is not an i4h experiment.
- No cloud compute was purchased or authorized.

## Repairs already present

- Candidate activation contracts reject Python and YAML semantic no-ops.
- Runtime probes exclude forked DataLoader workers and attribute subprocess-launched trainers.
- CUDA seeding is deferred until the candidate-call boundary.
- Operator interruption terminates owned local and remote process groups and records an aborted outcome.
- Future GPU executions use a 900-second startup-liveness gate.
- The controlled test checkpoint reported 145 passed and 0 failed tests; real GPU verification of the 900-second startup gate remains outstanding.

## Research objective to resume

The next objective is not merely to maximize the official FML protected-test metric. The paper-grade experiment should evaluate the pipeline on multiple frozen planes:

1. ML task performance and held-out generalization.
2. Search efficiency, reliability, diversity, and resource cost.
3. Runtime activation, causal attribution, replication, and ablation validity.
4. Literature, hypothesis, experimental-design, statistical-analysis, and review quality.
5. Memory/skill adaptation and transfer to held-out tasks.
6. End-to-end paper integrity and blinded paper-quality assessment, kept separate from FML performance.

The intended claim is a robust-generalist result: superiority on predefined primary dimensions, formal non-inferiority on dimensions not won, strong average rank, and low worst-case regret. Metrics, normalization, task/seed allocation, non-inferiority margins, and multiplicity correction must be frozen before protected results are inspected. Search-only agents must be compared like-for-like on search; complete-pipeline comparisons must use complete systems or identical downstream modules.

## Exact resume point

After explicit authorization, rerun the same first held-out matrix row under the repaired harness and 900-second GPU startup gate using a new log root. Do not reuse any interrupted attempt, restart i4h automatically, expose protected metrics to search or skill evolution, or purchase cloud compute without separate approval.
