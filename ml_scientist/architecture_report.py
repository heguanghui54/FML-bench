"""Self-contained HTML checkpoint report for the adaptive graph-backed pipeline."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _esc(value: Any) -> str:
    return html.escape(str(value))


def write_architecture_checkpoint_report(
    *,
    graph_path: Path,
    benchmark_contract_path: Path,
    plans_dir: Path,
    out_path: Path,
    experiment_status: str = "PAUSED_BY_USER",
) -> dict[str, Any]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    contract = json.loads(benchmark_contract_path.read_text(encoding="utf-8"))
    plans = []
    for path in sorted(plans_dir.glob("*.json")):
        try:
            plans.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    node_counts: dict[str, int] = {}
    for node in graph.get("nodes", []):
        node_counts[node["node_type"]] = node_counts.get(node["node_type"], 0) + 1
    observability: dict[str, int] = {}
    for metric in contract.get("metrics", []):
        key = metric.get("observability", "UNKNOWN")
        observability[key] = observability.get(key, 0) + 1
    graph_bound = sum(plan.get("knowledge_graph_snapshot", {}).get("status") == "FROZEN_GRAPH_BOUND" for plan in plans)
    cards = "".join(
        f'<div class="card"><span>{_esc(name)}</span><strong>{count}</strong></div>'
        for name, count in sorted(node_counts.items()) if count
    )
    metric_rows = "".join(
        f"<tr><td>{_esc(name)}</td><td>{count}</td><td>{'No' if name == 'PROTECTED_FINAL_ONLY' else 'Stage-dependent'}</td></tr>"
        for name, count in sorted(observability.items())
    )
    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Self-Evolving ML Scientist — Architecture Checkpoint</title>
<style>
:root{{--bg:#08111f;--panel:#111d31;--ink:#edf4ff;--muted:#9db0ce;--line:#294263;--cyan:#5ee7f0;--green:#76e6a5;--amber:#ffc76b;--red:#ff8b8b}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(145deg,#07101d,#0b1730);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui;padding:36px}}
main{{max-width:1180px;margin:auto}} h1{{font-size:34px;margin:.2rem 0}} h2{{margin-top:34px}} .muted{{color:var(--muted)}}
.status{{display:inline-block;padding:6px 12px;border:1px solid var(--amber);color:var(--amber);border-radius:999px;font-weight:700}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px;margin:18px 0}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;display:flex;flex-direction:column;gap:7px}} .card span{{color:var(--muted)}} .card strong{{font-size:26px;color:var(--cyan)}}
.flow{{display:grid;grid-template-columns:repeat(8,minmax(100px,1fr));gap:9px;align-items:stretch;margin:20px 0}} .step{{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 10px;text-align:center}} .step:not(:last-child)::after{{content:'→';position:absolute;right:-10px;top:38%;color:var(--cyan);z-index:2}} .step b{{display:block;color:var(--cyan)}}
.loop{{border-left:4px solid var(--green);background:#0e2230;padding:14px 18px;border-radius:8px}} table{{width:100%;border-collapse:collapse;background:var(--panel);border-radius:12px;overflow:hidden}} th,td{{border-bottom:1px solid var(--line);padding:10px 12px;text-align:left}} th{{color:var(--cyan)}} code{{color:var(--green)}}
.warning{{border:1px solid var(--amber);background:#2a2115;padding:14px;border-radius:10px}} @media(max-width:900px){{.flow{{grid-template-columns:repeat(2,1fr)}}.step::after{{display:none}}body{{padding:18px}}}}
</style></head><body><main>
<span class="status">Experiment { _esc(experiment_status) }</span>
<h1>Graph-Backed Adaptive ML Scientist</h1>
    <p class="muted">Implementation checkpoint: benchmark learning, typed memory migration, adaptive Stage 3 planning, node-local atomic-policy composition, enriched transitions, post-run skill distillation, contextual utility, evidence-gated governance, paper evidence locks, and backward amendments.</p>

<h2>Pipeline</h2><div class="flow">
    <div class="step"><b>Stage 0</b>Freeze graph</div><div class="step"><b>Stage 1</b>Metric contract</div><div class="step"><b>Stage 2</b>Value-aware read</div><div class="step"><b>Stage 3</b>Compose DAG</div><div class="step"><b>Stage 4</b>Execute node</div><div class="step"><b>Stage 5</b>Assess policies</div><div class="step"><b>Stage 6</b>Route / repair</div><div class="step"><b>Stage 7</b>Write / govern</div>
</div>
    <div class="loop"><strong>Inner loop:</strong> retrieve role-specific policies → validate capabilities → compose → execute → record research context, state transition, resource telemetry, and one evidence trace per applied policy → advance, retry, change composition, backtrack, or create a versioned amendment. After the arm, informative transitions enter CREATE/PATCH/NONE distillation; later held-out outcomes update contextual utility and promotion or rollback. Full baseline bundles remain provenance references. Protected final metrics never return to search or skill evolution.</div>

<h2>Canonical knowledge graph</h2><div class="grid">{cards}</div>
<p><code>{_esc(graph.get('content_sha256'))}</code></p>

<h2>Benchmark and metric learning</h2><table><thead><tr><th>Observability</th><th>Metrics</th><th>May route current search?</th></tr></thead><tbody>{metric_rows}</tbody></table>

<h2>Stage 3 migration status</h2><div class="grid">
<div class="card"><span>Task plans</span><strong>{len(plans)}</strong></div><div class="card"><span>Graph-bound plans</span><strong>{graph_bound}</strong></div><div class="card"><span>Cartesian expansion</span><strong>OFF</strong></div><div class="card"><span>Paper node</span><strong>LOCKED</strong></div>
</div>

    <h2>Self-evolving skill memory</h2><p><strong>Read:</strong> active skills are ranked by semantic relevance, maturity, contextual utility, diversity value, and cost at stage boundaries. <strong>Write:</strong> a gated post-run buffer removes invalid executions and distills explicit policy traces into general, task-specific, or action skills. <strong>Assess:</strong> a candidate cannot validate itself on its authoring episode; later held-out downstream evidence supplies contextual advantage. <strong>Govern:</strong> useful skills are promoted, near-duplicates are patched, and comparable harmful evidence rolls back the affected version.</p>

    <h2>Paper correction routes</h2><p>Review findings are typed. Wording returns to writing; statistical errors invalidate analysis and claims; missing experiments return to experiment or replication; implementation errors return to code and invalidate descendants. After protected-test exposure, empirical changes require a new hidden evaluation or explicit post-hoc labeling.</p>

<div class="warning"><strong>Evidence boundary:</strong> this report proves local architecture construction, migration, and unit validation. It does not claim that the adaptive pipeline outperforms any FML baseline. The existing GPU campaign remains paused.</div>
</main></body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    return {
        "schema_version": "ml-scientist-architecture-report-v3",
        "path": str(out_path.resolve()),
        "graph_node_count": len(graph.get("nodes", [])),
        "graph_edge_count": len(graph.get("edges", [])),
        "plan_count": len(plans),
        "graph_bound_plan_count": graph_bound,
        "experiment_status": experiment_status,
    }
