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
checkpoints/  results/  tmp/  bootstrap/
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

The controller environment is deliberately small. Provider-specific packages
such as Anthropic are imported only when that provider is selected; CodexCLI
does not require an API key or those SDKs.
