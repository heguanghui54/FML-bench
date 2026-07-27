# Codex CLI controller with Ubuntu SSH evaluation

This condition is designed to learn transferable ML-research pipeline skills. It is
not labeled as an exact reproduction of the API-model condition in the published
FML-bench results.

## Boundary

- The Mac runs the seven research-agent controllers and an authenticated local
  `CodexCLI` text-only model adapter.
- The Ubuntu host runs task validation/test commands through `SSHExecutor`.
- Codex calls are ephemeral, execute in an empty read-only directory, and are
  rejected if their JSONL stream contains tool activity.
- Ubuntu GPU tasks refuse to start while another compute process (including i4h)
  is visible in `nvidia-smi`.
- Numerical paper claims remain locked until protected-test results and integrity
  checks are present.

## External-disk layout

All heavyweight state is rooted at `/media/heshi/game/fml-scientist`:

```text
repo/  conda-envs/  conda-pkgs/  cache/  datasets/
checkpoints/  results/  tmp/  bootstrap/  packages/  asset-tools/
```

Docker already uses `/media/heshi/game/docker-data` on `ubuntu-heshi`.

CIFAR-10 and CIFAR-100 are downloaded once into the external-disk shared cache,
checked against torchvision's canonical MD5 values, and linked into task
templates. This avoids repeated downloads and duplicate experiment data.

## Bootstrap

From the Mac controller:

```bash
FML_DOWNLOAD_PROXY=http://127.0.0.1:17891 \
  bash scripts/bootstrap_codex_controller.sh
scripts/sync_to_ubuntu.sh
ssh ubuntu-heshi \
  'cd /media/heshi/game/fml-scientist/repo && \
   FML_DOWNLOAD_PROXY=http://127.0.0.1:17891 \
   bash scripts/bootstrap_ubuntu_external.sh'
```

Then source `scripts/ubuntu_external_env.sh` before `setup.py`.
Because the benchmark harness runs on the Mac in this architecture, pass
`--skip-harness-env` when adding Ubuntu task environments. This avoids installing
API SDKs and the large general-purpose harness stack on the runner.

For example:

```bash
python setup.py --skip-harness-env --task Causality_gcastle
```

Prepare all eight confirmatory Lite tasks, with per-task resumable logs under
the external-disk result root:

```bash
FML_DOWNLOAD_PROXY=http://127.0.0.1:17891 \
  bash scripts/setup_ubuntu_lite.sh
```

Prepare AdaptiveSearch's GraphCodeBERT runtime after the Ubuntu external-disk
bootstrap. Ubuntu stages the fixed macOS arm64 wheels and model snapshot on the
2 TB disk; the script then copies them to this Mac external-volume repository,
installs offline, and verifies a deterministic 768-dimensional embedding:

```bash
FML_DOWNLOAD_PROXY=http://127.0.0.1:17891 \
  bash scripts/prepare_adaptivesearch_assets.sh
```

The frozen condition uses `torch==2.10.0`, `transformers==5.3.0`,
`microsoft/graphcodebert-base` revision
`2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d`, CPU inference on the Mac
controller, and `local_files_only=true`. Task training and evaluation continue
to use the Ubuntu NVIDIA GPU.

The controller also needs source-only templates for code editing, but never the
Ubuntu datasets. Create them on the Mac external volume with:

```bash
.controller-venv/bin/python setup.py --skip-envs --skip-data \
  --task Generalization_domainbed \
  --task Continual_Learning_pycil \
  --task Data_Efficiency_usb \
  --task Generalization_domainbed_officehome \
  --task Privacy_opacus \
  --task Privacy_privacymeter \
  --task Robustness_and_Reliability_art \
  --task Robustness_openood
```

## Controlled run

```bash
.controller-venv/bin/python run_agent_benchmark.py \
  --agent-config configs/agents/autoresearch.yaml \
  --task-config configs/tasks/causality_causalml.yaml \
  --model gpt-5.6-sol --provider CodexCLI \
  --eval-backend ssh --ssh-host ubuntu-heshi \
  --remote-project-root /media/heshi/game/fml-scientist/repo \
  --seed 1103 agent.autoresearch.max_steps=1
```

Every Codex call and SSH execution writes a versioned audit record. Formal GPU
runs additionally require an idle-GPU preflight.

Run the full readiness check before a campaign:

```bash
.controller-venv/bin/python -m ml_scientist.cli preflight \
  --out artifacts/ml_scientist/controller_ready \
  --model gpt-5.6-sol --provider CodexCLI \
  --eval-backend ssh --ssh-host ubuntu-heshi \
  --remote-project-root /media/heshi/game/fml-scientist/repo
```

The checked-in CodexCLI/SSH condition is intentionally resource limited. Its
diagnostic pilot uses one agent step and its balanced confirmatory matrix uses
three steps for every agent-task-seed run (7 agents x 8 Lite tasks x 3 seeds).
This preserves task code, metrics, seeds, and the protected-test boundary while
making the campaign executable on one RTX 3060 Ti. It estimates early-search
performance, not the published 100-step condition. AdaptiveSearch's 50-step
phase transition cannot fire in this matrix, so a branching advantage must not
be claimed from these results.

A quarantined engineering calibration measured one native PrivacyMeter
validation at 1918.81 seconds on this RTX 3060 Ti. Because the unchanged
protected-test path regenerates its checkpoint with one additional validation,
the PrivacyMeter portion alone is forecast at 7.46 validation-GPU-hours for the
pilot and 44.77 for the confirmatory matrix, before controller and test
overhead. Calibration scores remain ineligible for model selection or paper
claims.

If i4h is using the GPU through the paper-extension transient service, pause
the owning unit (stopping only its child container lets the service continue):

```bash
ssh ubuntu-heshi 'systemctl --user stop i4h-p1-visual-robustness-runner.service'
```

Provider-specific packages such as Anthropic are imported only when that
provider is selected; CodexCLI does not require an API key or those SDKs.
