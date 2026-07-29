#!/bin/zsh

set -eu

cd "/Volumes/mac外接盘/self-evolve ml"

while true; do
  .controller-venv/bin/python -m ml_scientist.cli fml-import-episodes \
    --results benchmark_results/controlled \
    --episodes artifacts/ml_scientist/trajectory_memory/baseline_observations.jsonl \
    --graph artifacts/ml_scientist/post_arm/knowledge_graph/final_graph_snapshot/knowledge_graph.json \
    >/dev/null
  .controller-venv/bin/python -m ml_scientist.cli live-pipeline-report \
    --results benchmark_results/controlled \
    --campaign-dir artifacts/ml_scientist/codex_ssh_protocol/pilot_campaign_v1 \
    --matrix artifacts/ml_scientist/codex_ssh_protocol/run_matrix.csv \
    --graph artifacts/ml_scientist/post_arm/knowledge_graph/final_graph_snapshot/knowledge_graph.json \
    --issues artifacts/ml_scientist/pipeline_diagnostics/issue_registry.json \
    --episodes artifacts/ml_scientist/trajectory_memory/baseline_observations.jsonl \
    --new-pipeline-results benchmark_results/adaptive_pipeline \
    --out artifacts/ml_scientist/reports/live-pipeline \
    >/dev/null
  sleep 45
done
