"""Published FML-bench v2 aggregate results, kept separate from new evidence.

These constants transcribe aggregate tables from arXiv:2605.17373v2. They are
useful for protocol design, sanity checks, and reviewer context, but they are
not observations produced by this repository checkout. The writer therefore
uses a dedicated directory and emits an explicit provenance manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PAPER = {
    "title": "FML-bench: A Controlled Study of AI Research Agent Strategies from the Perspective of Search Dynamics",
    "arxiv_id": "2605.17373",
    "version": "v2",
    "date": "2026-05-29",
    "abstract_url": "https://arxiv.org/abs/2605.17373",
    "html_url": "https://arxiv.org/html/2605.17373",
    "pdf_url": "https://arxiv.org/pdf/2605.17373",
    "evidence_class": "published_prior_aggregate",
}

AGENTS = [
    ("theaiscientist", "TAS v1"),
    ("ai_scientist_v2", "TAS v2"),
    ("aide", "AIDE"),
    ("aira_mcts", "AIRA"),
    ("autoresearch", "AutoR"),
    ("openevolve", "OEvolve"),
]

TASK_IDS = {
    "DomainBed-CM": "Generalization_domainbed",
    "DomainBed-OH": "Generalization_domainbed_officehome",
    "EasyFSL": "Data_Efficiency_easyfsl",
    "USB": "Data_Efficiency_usb",
    "Lightly": "Representation_Learning_lightly",
    "Solo-learn": "Representation_Learning_solo_learn",
    "Cont.-Learn.": "Continual_Learning_continual_learning",
    "PyCIL": "Continual_Learning_pycil",
    "CausalML": "Causality_causalml",
    "gCastle": "Causality_gcastle",
    "ART": "Robustness_and_Reliability_art",
    "OpenOOD": "Robustness_openood",
    "PrivacyMeter": "Privacy_privacymeter",
    "Opacus": "Privacy_opacus",
    "AIF360": "Fairness_and_Bias_aif360",
    "Fairlearn": "Fairness_fairlearn",
    "Unlearning": "Unlearning_open_unlearning",
    "PFLlib": "Federated_Learning_PFLlib",
}

# Table 2: each pair is (mean, standard deviation) across three rounds.
TABLE2 = {
    "DomainBed-CM": [(0.071, .041), (.245, .182), (.231, .198), (.165, .211), (.067, .053), (.242, .200)],
    "DomainBed-OH": [(.006, .004), (.014, .003), (.011, .005), (.012, .002), (.005, .004), (.013, .005)],
    "EasyFSL": [(.013, .020), (.034, .017), (.030, .046), (.063, .025), (.032, .028), (.046, .028)],
    "USB": [(.040, .011), (.039, .008), (.029, .008), (.019, .005), (.048, .029), (.067, .012)],
    "Lightly": [(.020, .025), (.019, .016), (.044, .023), (.000, .000), (.069, .029), (.043, .012)],
    "Solo-learn": [(.076, .018), (.113, .004), (.069, .025), (.066, .034), (.003, .003), (.093, .005)],
    "Cont.-Learn.": [(.048, .011), (.349, .339), (.113, .163), (.072, .124), (.375, .230), (.164, .089)],
    "PyCIL": [(.030, .025), (.064, .004), (.056, .005), (.020, .007), (.026, .007), (.037, .021)],
    "CausalML": [(.020, .006), (.040, .068), (.023, .024), (.033, .057), (.007, .012), (.014, .024)],
    "gCastle": [(.014, .014), (.066, .114), (.127, .207), (.145, .045), (.127, .143), (.169, .146)],
    "ART": [(.429, .026), (.395, .026), (.447, .035), (.303, .127), (.448, .032), (.459, .011)],
    "OpenOOD": [(.023, .033), (.039, .017), (.012, .020), (.021, .002), (.057, .013), (.006, .011)],
    "PrivacyMeter": [(.008, .014), (.482, .068), (.530, .018), (.011, .020), (.575, .025), (.032, .056)],
    "Opacus": [(.000, .000), (.000, .000), (.006, .010), (.000, .000), (.054, .017), (.003, .003)],
    "AIF360": [(.208, .028), (.218, .031), (.231, .005), (.136, .120), (.218, .016), (.156, .135)],
    "Fairlearn": [(.170, .005), (.173, .000), (.162, .003), (.152, .002), (.170, .001), (.173, .000)],
    "Unlearning": [(.968, .004), (.944, .016), (.831, .142), (.921, .045), (.896, .094), (.794, .039)],
    "PFLlib": [(.236, .018), (.236, .013), (.250, .007), (.234, .042), (.277, .053), (.201, .005)],
}

TABLE3 = [
    ("adaptivesearch", "Adaptive", .208, 58.6),
    ("ai_scientist_v2", "TAS v2", .193, 56.2),
    ("autoresearch", "AutoR", .192, 56.2),
    ("aide", "AIDE", .178, 49.4),
    ("openevolve", "OEvolve", .151, 52.8),
    ("theaiscientist", "TAS v1", .132, 40.4),
    ("aira_mcts", "AIRA", .132, 28.4),
]

# Table 4. The exact p-values stated in Section 5.2 are used for AUC and first
# improvement; all other p-values are transcribed from the table.
TABLE4 = [
    ("Exploration", "Exploration Spread", [23.65, 28.44, 16.23, 14.60, 23.35, 10.89], .094, .091),
    ("Exploration", "Exploration Uniqueness", [.0628, .0474, .0357, .0406, .0341, .0796], .012, .823),
    ("Exploration", "Exploration Reach", [122.6, 139.6, 72.97, 71.19, 110.6, 42.39], .115, .039),
    ("Exploration", "Effective dim", [1.784, 1.450, 2.974, 2.745, 1.709, 3.718], -.140, .011),
    ("Generalization", "Val-test |gap|", [.0362, .0247, .0274, .0401, .0334, .0448], .097, .080),
    ("Reliability", "Valid step ratio", [.7744, .7339, .8887, .7893, .8241, .8281], .086, .122),
    ("Efficiency", "AUC-over-steps", [.1356, .1579, .1612, .1303, .1755, .1412], .784, 1.6e-68),
    ("Efficiency", "First-improvement step", [12.78, 16.46, 10.62, 15.56, 9.185, 15.11], -.291, 2.3e-7),
    ("Efficiency", "Late-gain fraction", [.1067, .3016, .1185, .3205, .1882, .2246], -.042, .462),
    ("Efficiency", "Best-improvement step", [48.70, 74.17, 79.09, 71.78, 91.24, 70.94], .104, .060),
    ("Cost", "Token cost (M)", [2.288, 1.426, 1.737, 1.075, 1.736, .9343], -.005, .935),
    ("Cost", "Wall-clock time (h)", [30.27, 32.49, 27.35, 29.08, 29.41, 28.70], -.094, .090),
]

TASK_CARDS = [
    ("Generalization", "DomainBed-CM", "ColoredMNIST", "ERM", "accuracy", "higher", 1.0, .118, "SPARSE-OPP", .287),
    ("Generalization", "DomainBed-OH", "OfficeHome", "ERM with pretrained ResNet-50 features", "average OOD accuracy", "higher", 1.0, .030, "SPARSE-OPP", .863),
    ("Data Efficiency", "EasyFSL", "Mini-ImageNet", "Prototypical Networks", "episodic accuracy", "higher", 1.0, .025, "SPARSE-OPP", .653),
    ("Data Efficiency", "USB", "CIFAR-100 with 200 labels", "FixMatch", "test accuracy", "higher", 1.0, .047, "SPARSE-OPP", .079),
    ("Representation Learning", "Lightly", "CIFAR-10", "MoCo", "linear-probe accuracy", "higher", 1.0, .055, "SPARSE-OPP", .756),
    ("Representation Learning", "Solo-learn", "CIFAR-100", "Barlow Twins", "linear-eval accuracy", "higher", 1.0, .102, "SPARSE-OPP", .490),
    ("Continual Learning", "Cont.-Learn.", "splitMNIST", "Synaptic Intelligence without replay", "final average accuracy", "higher", 1.0, .254, "DENSE-OPP", .271),
    ("Continual Learning", "PyCIL", "CIFAR-100 incremental", "iCaRL", "average incremental accuracy", "higher", 1.0, .049, "SPARSE-OPP", .595),
    ("Causality", "CausalML", "IHDP", "Dragonnet", "individual-treatment-effect MAE", "lower", 0.0, .077, "SPARSE-OPP", 1.328),
    ("Causality", "gCastle", "50-node nonlinear synthetic DAG", "NOTEARS", "structural Hamming distance", "lower", 0.0, .347, "DENSE-OPP", 71.0),
    ("Robustness", "ART", "poisoned MNIST", "dp-instahide", "defense score", "higher", 1.0, .573, "DENSE-OPP", .477),
    ("Robustness", "OpenOOD", "CIFAR-10 versus SVHN/OOD", "maximum softmax probability", "near-OOD AUROC", "higher", 1.0, .045, "SPARSE-OPP", .930),
    ("Privacy", "PrivacyMeter", "CIFAR-10 WRN-28-2", "membership-inference audit", "MIA/RMIA AUC gap", "lower", 0.0, .454, "DENSE-OPP", .321),
    ("Privacy", "Opacus", "CIFAR-10", "CNN+GroupNorm DP-SGD", "test accuracy at epsilon=8", "higher", 1.0, .144, "DENSE-OPP", .588),
    ("Fairness", "AIF360", "COMPAS", "Adversarial Debiasing", "absolute average-odds difference", "lower", 0.0, .356, "DENSE-OPP", .240),
    ("Fairness", "Fairlearn", "Adult Census", "unconstrained logistic regression", "absolute demographic-parity difference", "lower", 0.0, .447, "DENSE-OPP", .173),
    ("Unlearning", "Unlearning", "TOFU with Llama-3.2-1B", "naive gradient ascent", "-log10(KS-test p-value)", "lower", 0.0, 1.207, "DENSE-OPP", 166.80),
    ("Federated Learning", "PFLlib", "CIFAR-10, 20 non-IID clients", "FedAvg", "global test accuracy", "higher", 1.0, .352, "DENSE-OPP", .453),
]

SEARCH_STRATEGIES = [
    ("theaiscientist", "TAS v1", "parallel linear multi-idea", "independent idea chains; no cross-idea feedback"),
    ("ai_scientist_v2", "TAS v2", "four-stage best-first tree search", "best metric first; LLM journal; stage shares 0.10/0.20/0.50/0.20"),
    ("aide", "AIDE", "solution-space tree", "improve best good node or stochastically debug a buggy leaf"),
    ("aira_mcts", "AIRA", "UCT tree search", "multi-child expansion; running mean fitness; no rollout"),
    ("autoresearch", "AutoR", "greedy hill climbing", "strict keep/discard; bounded crash debugging; one incumbent"),
    ("openevolve", "OEvolve", "island MAP-Elites", "5x5 feature grids; mixed parent selection; ring migration"),
    ("adaptivesearch", "Adaptive", "greedy then multi-branch", "irreversible switch after 50-step stagnation within epsilon 0.0005"),
]


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _save_svg(path: Path, body: str, width: int, height: int, title: str, desc: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">\n'
        f'<title>{_escape(title)}</title><desc>{_escape(desc)}</desc>\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.axis{stroke:#222}.grid{stroke:#ddd}.label{font-size:12px}.small{font-size:10px}.title{font-size:17px;font-weight:bold}</style>\n'
        f'{body}\n</svg>\n',
        encoding="utf-8",
    )


def agent_task_rows() -> list[dict[str, Any]]:
    rows = []
    for task_name, values in TABLE2.items():
        for (agent_id, agent_label), (mean, sd) in zip(AGENTS, values, strict=True):
            rows.append(
                {
                    "evidence_class": PAPER["evidence_class"],
                    "paper": PAPER["arxiv_id"] + PAPER["version"],
                    "source_locator": "Table 2",
                    "task_id": TASK_IDS[task_name],
                    "paper_task": task_name,
                    "agent_id": agent_id,
                    "paper_agent": agent_label,
                    "round_n": 3,
                    "mean_normalized_test_improvement": mean,
                    "sd_across_rounds": sd,
                }
            )
    return rows


def agent_summary_rows() -> list[dict[str, Any]]:
    return [
        {
            "evidence_class": PAPER["evidence_class"],
            "paper": PAPER["arxiv_id"] + PAPER["version"],
            "source_locator": "Table 3",
            "agent_id": agent_id,
            "paper_agent": label,
            "task_n": 18,
            "mean_normalized_test_improvement": improvement,
            "pairwise_win_rate_percent": win_rate,
        }
        for agent_id, label, improvement, win_rate in TABLE3
    ]


def process_rows() -> list[dict[str, Any]]:
    rows = []
    for dimension, metric, values, rho, p_value in TABLE4:
        row: dict[str, Any] = {
            "evidence_class": PAPER["evidence_class"],
            "paper": PAPER["arxiv_id"] + PAPER["version"],
            "source_locator": "Table 4 and Section 5.2",
            "dimension": dimension,
            "metric": metric,
            "pooled_cell_n": 324,
            "pooled_spearman_rho": rho,
            "p_value_unadjusted": p_value,
            "significant_unadjusted_p_lt_0_05": p_value < .05,
        }
        for (_, label), value in zip(AGENTS, values, strict=True):
            row[label] = value
        rows.append(row)
    return rows


def task_card_rows() -> list[dict[str, Any]]:
    fields = [
        "domain", "paper_task", "dataset", "baseline_method", "native_metric",
        "direction", "theoretical_best", "opportunity_density", "partition", "baseline_raw",
    ]
    rows = []
    for values in TASK_CARDS:
        row = dict(zip(fields, values, strict=True))
        row.update(
            {
                "evidence_class": PAPER["evidence_class"],
                "paper": PAPER["arxiv_id"] + PAPER["version"],
                "source_locator": "Table 5 and Appendix A",
                "task_id": TASK_IDS[row["paper_task"]],
                "partition_is_post_hoc": True,
            }
        )
        rows.append(row)
    return rows


def published_guidance_for_task(task_id: str) -> dict[str, Any]:
    """Return hypothesis guidance without granting empirical claim credit."""
    try:
        card = next(row for row in task_card_rows() if row["task_id"] == task_id)
    except StopIteration as exc:
        raise KeyError(f"No published FML task card for {task_id}") from exc
    dense = card["partition"] == "DENSE-OPP"
    if dense:
        strategy = {
            "initial_mode": "greedy_exploitation",
            "primary_operators": ["autoresearch", "adaptivesearch"],
            "required_counterfactual_operators": ["ai_scientist_v2", "openevolve"],
            "reason": "The published post-hoc partition found frequent gains per unit code-space distance.",
        }
    else:
        strategy = {
            "initial_mode": "multi_branch_exploration",
            "primary_operators": ["adaptivesearch", "ai_scientist_v2", "openevolve"],
            "required_counterfactual_operators": ["autoresearch"],
            "reason": "The published post-hoc partition found sparse gains; preserve multiple frontiers before exploitation.",
        }
    return {
        "evidence_class": PAPER["evidence_class"],
        "source": PAPER["arxiv_id"] + PAPER["version"],
        "source_locator": "Table 5, Section 4.3, and Appendix G",
        "task_id": task_id,
        "published_baseline_method": card["baseline_method"],
        "published_native_metric": card["native_metric"],
        "published_opportunity_density": card["opportunity_density"],
        "published_partition": card["partition"],
        "strategy_hypothesis": strategy,
        "review_requirements": [
            "treat the opportunity partition as post-hoc",
            "include the counterfactual search family under a matched budget",
            "do not use this prior to unlock a result claim",
            "replace this guidance when local frozen validation evidence contradicts it",
        ],
        "claim_unlock_eligible": False,
    }


def _agent_summary_figure(rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 1120, 450
    left, top, panel_w, row_h = 145, 62, 365, 45
    body = ['<text x="24" y="28" class="title">Published FML-bench agent summary (prior evidence)</text>']
    body.append('<text x="145" y="50" class="label">Mean normalized test improvement</text>')
    body.append('<text x="665" y="50" class="label">Pairwise win rate (%)</text>')
    for index, row in enumerate(rows):
        y = top + index * row_h
        label = _escape(row["paper_agent"])
        improvement = float(row["mean_normalized_test_improvement"])
        win = float(row["pairwise_win_rate_percent"])
        body.append(f'<text x="{left-10}" y="{y+20}" text-anchor="end" class="label">{label}</text>')
        body.append(f'<rect x="{left}" y="{y+4}" width="{improvement/.25*panel_w:.1f}" height="21" fill="#0072B2"/>')
        body.append(f'<text x="{left+improvement/.25*panel_w+7:.1f}" y="{y+20}" class="small">{improvement:.3f}</text>')
        body.append(f'<rect x="665" y="{y+4}" width="{win/65*panel_w:.1f}" height="21" fill="#E69F00"/>')
        body.append(f'<text x="{665+win/65*panel_w+7:.1f}" y="{y+20}" class="small">{win:.1f}</text>')
    body.append('<text x="24" y="425" class="small">Source: arXiv:2605.17373v2 Table 3. Aggregate published values; not produced by this campaign.</text>')
    _save_svg(path, "\n".join(body), width, height, "Published FML agent summary", "Published mean normalized improvement and pairwise win rate for seven agents.")


def _heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    by_cell = {(row["paper_task"], row["paper_agent"]): float(row["mean_normalized_test_improvement"]) for row in rows}
    task_names = list(TABLE2)
    agent_names = [label for _, label in AGENTS]
    width, height = 930, 690
    left, top, cell_w, cell_h = 190, 72, 112, 30
    body = ['<text x="24" y="28" class="title">Published mean normalized test improvement by agent and task</text>']
    for j, agent in enumerate(agent_names):
        body.append(f'<text x="{left+j*cell_w+cell_w/2}" y="55" text-anchor="middle" class="label">{_escape(agent)}</text>')
    for i, task in enumerate(task_names):
        y = top + i * cell_h
        body.append(f'<text x="{left-8}" y="{y+20}" text-anchor="end" class="label">{_escape(task)}</text>')
        for j, agent in enumerate(agent_names):
            value = by_cell[(task, agent)]
            intensity = min(value / .6, 1.0)
            blue = int(245 - 155 * intensity)
            green = int(249 - 90 * intensity)
            fill = f'rgb({blue},{green},220)'
            text_fill = "white" if intensity > .60 else "#222"
            body.append(f'<rect x="{left+j*cell_w}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" fill="{fill}"/>')
            body.append(f'<text x="{left+j*cell_w+(cell_w-2)/2}" y="{y+20}" text-anchor="middle" style="font-size:10px;fill:{text_fill}">{value:.3f}</text>')
    body.append('<text x="24" y="655" class="small">Each cell is a 3-round mean from Table 2; uncertainty remains in published_agent_task_mean_sd.csv.</text>')
    body.append('<text x="24" y="672" class="small">Color is capped at 0.6 so the Unlearning row does not erase contrasts among other tasks.</text>')
    _save_svg(path, "\n".join(body), width, height, "Published FML task-agent heatmap", "Published three-round mean normalized improvement for six agents on eighteen tasks.")


def _process_correlation_figure(rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 970, 535
    center, scale, top, row_h = 610, 380, 62, 35
    body = ['<text x="24" y="28" class="title">Published process metrics versus final improvement</text>']
    body.append(f'<line x1="{center}" y1="46" x2="{center}" y2="480" stroke="#555"/>')
    for tick in (-.8, -.4, 0, .4, .8):
        x = center + tick * scale
        body.append(f'<line x1="{x}" y1="46" x2="{x}" y2="480" class="grid"/>')
        body.append(f'<text x="{x}" y="500" text-anchor="middle" class="small">{tick:+.1f}</text>')
    for index, row in enumerate(rows):
        y = top + index * row_h
        rho = float(row["pooled_spearman_rho"])
        significant = bool(row["significant_unadjusted_p_lt_0_05"])
        color = "#D55E00" if significant else "#999999"
        anchor = "start" if rho >= 0 else "end"
        offset = 10 if rho >= 0 else -10
        body.append(f'<text x="400" y="{y+4}" text-anchor="end" class="label">{_escape(row["metric"])}</text>')
        body.append(f'<line x1="{center}" y1="{y}" x2="{center+rho*scale:.1f}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<circle cx="{center+rho*scale:.1f}" cy="{y}" r="6" fill="{color}"/>')
        body.append(f'<text x="{center+rho*scale+offset:.1f}" y="{y+4}" text-anchor="{anchor}" class="small">{rho:+.3f}{" *" if significant else ""}</text>')
    body.append('<text x="24" y="524" class="small">Pooled Spearman over 324 cells. * unadjusted p&lt;0.05; no multiplicity correction or cluster adjustment is claimed.</text>')
    _save_svg(path, "\n".join(body), width, height, "Published process correlations", "Pooled Spearman correlations between twelve process metrics and normalized test improvement.")


def _opportunity_figure(rows: list[dict[str, Any]], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: float(row["opportunity_density"]))
    width, height = 1050, 680
    left, top, plot_w, row_h = 190, 55, 770, 31
    maximum = max(float(row["opportunity_density"]) for row in ordered)
    body = ['<text x="24" y="28" class="title">Published post-hoc opportunity-density partition</text>']
    median_x = left + .1315 / maximum * plot_w
    body.append(f'<line x1="{median_x:.1f}" y1="45" x2="{median_x:.1f}" y2="615" stroke="#333" stroke-dasharray="5,4"/>')
    body.append(f'<text x="{median_x+5:.1f}" y="49" class="small">median 0.1315</text>')
    for index, row in enumerate(ordered):
        y = top + index * row_h
        value = float(row["opportunity_density"])
        width_value = value / maximum * plot_w
        color = "#009E73" if row["partition"] == "DENSE-OPP" else "#56B4E9"
        body.append(f'<text x="{left-8}" y="{y+18}" text-anchor="end" class="label">{_escape(row["paper_task"])}</text>')
        body.append(f'<rect x="{left}" y="{y+3}" width="{width_value:.1f}" height="20" fill="{color}"/>')
        body.append(f'<text x="{left+width_value+7:.1f}" y="{y+18}" class="small">{value:.3f}</text>')
    body.append('<text x="24" y="649" class="small">The partition was computed after observing agent trajectories; treat it as hypothesis-generating, not preregistered evidence.</text>')
    _save_svg(path, "\n".join(body), width, height, "Published opportunity density", "Post-hoc opportunity density for eighteen FML tasks and the published median split.")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_published_prior(out_dir: Path) -> dict[str, Any]:
    """Write source-labeled published tables, charts, notes, and manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    agent_task = agent_task_rows()
    agent_summary = agent_summary_rows()
    processes = process_rows()
    task_cards = task_card_rows()
    search_rows = [
        {
            "evidence_class": PAPER["evidence_class"],
            "paper": PAPER["arxiv_id"] + PAPER["version"],
            "source_locator": "Table 6 and Appendix D",
            "agent_id": agent_id,
            "paper_agent": label,
            "search_topology": topology,
            "selection_and_memory": details,
        }
        for agent_id, label, topology, details in SEARCH_STRATEGIES
    ]

    _write_csv(out_dir / "published_agent_task_mean_sd.csv", agent_task, list(agent_task[0]))
    _write_csv(out_dir / "published_agent_summary.csv", agent_summary, list(agent_summary[0]))
    _write_csv(out_dir / "published_process_metrics.csv", processes, list(processes[0]))
    _write_csv(out_dir / "published_task_cards.csv", task_cards, list(task_cards[0]))
    _write_csv(out_dir / "published_search_strategies.csv", search_rows, list(search_rows[0]))
    _write_csv(
        out_dir / "paper_version_lineage.csv",
        [
            {
                "arxiv_id": "2510.10472",
                "role": "legacy",
                "task_scope": "8 tasks",
                "status": "retained on upstream legacy branch; do not mix protocols",
                "url": "https://arxiv.org/abs/2510.10472",
            },
            {
                "arxiv_id": "2605.17373v2",
                "role": "current",
                "task_scope": "18 tasks, 10 domains, 12 process metrics",
                "status": "source for every published-prior table in this directory",
                "url": PAPER["abstract_url"],
            },
        ],
        ["arxiv_id", "role", "task_scope", "status", "url"],
    )
    evaluation_contract = {
        "evidence_class": PAPER["evidence_class"],
        "paper": PAPER["arxiv_id"] + PAPER["version"],
        "source_locators": ["Sections 3.2-3.4", "Appendix E"],
        "search_observes": "validation metric only",
        "protected_test": "best-validated frozen codebase is evaluated once after the step budget; no test feedback returns to search",
        "final_metrics": {
            "normalized_test_improvement": {
                "definition": "non-negative native-metric improvement over the untouched baseline divided by the distance from the baseline to the theoretical best",
                "range": "[0, 1] under bounded contracts",
                "lower_is_better": "reverse the improvement direction",
                "unbounded_worst": "CausalML, gCastle, and Unlearning set the worst reference to the untouched baseline",
                "negative_improvement": "clamped to zero",
            },
            "pairwise_win_rate": {
                "definition": "strict native-metric wins against every other agent, averaged across opponents and tasks",
                "ties": "not wins",
                "property": "invariant under monotone rescaling",
            },
        },
        "process_metric_dimensions": {
            "Exploration": ["Exploration Spread", "Exploration Uniqueness", "Exploration Reach", "Effective dim"],
            "Generalization": ["Val-test |gap|"],
            "Reliability": ["Valid step ratio"],
            "Efficiency": ["AUC-over-steps", "First-improvement step", "Best-improvement step", "Late-gain fraction"],
            "Cost": ["Token cost (M)", "Wall-clock time (h)"],
        },
    }
    (out_dir / "published_evaluation_contract.json").write_text(
        json.dumps(evaluation_contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    figures = out_dir / "figures"
    _agent_summary_figure(agent_summary, figures / "published_agent_summary.svg")
    _heatmap(agent_task, figures / "published_agent_task_heatmap.svg")
    _process_correlation_figure(processes, figures / "published_process_correlations.svg")
    _opportunity_figure(task_cards, figures / "published_opportunity_density.svg")

    notes = out_dir / "README.md"
    notes.write_text(
        "\n".join(
            [
                "# Published FML-bench Prior Evidence",
                "",
                "Status: PUBLISHED PRIOR AGGREGATE — not a result of this campaign.",
                "",
                f"Source: [{PAPER['title']}]({PAPER['abstract_url']}), {PAPER['arxiv_id']}{PAPER['version']} ({PAPER['date']}).",
                "",
                "These tables teach the planning and review layers what the published baseline agents, tasks, native metrics, normalized metric, and process diagnostics look like. They may support hypotheses, resource planning, sanity checks, and related-work comparisons. They must never unlock a new-paper empirical claim or be concatenated with locally produced trial rows.",
                "",
                "## Published protocol represented here",
                "",
                "- Main comparison: six agents x eighteen tasks x three independent rounds x 100 validation steps = 324 runs, using one GPT-5.4 backbone and NVIDIA A100 80GB GPUs.",
                "- Final metrics: non-negative range-normalized held-out test improvement and native-metric pairwise win rate.",
                "- Process analysis: twelve metrics over exploration, generalization, reliability, efficiency, and cost.",
                "- AdaptiveSearch is a seventh, separately added strategy; its aggregate result is included only where published in Table 3.",
                "",
                "## Review warnings carried with the data",
                "",
                "- Opportunity density and its median split are explicitly post-hoc.",
                "- Table 4 correlations pool agent-task-round cells and report unadjusted p-values; task/agent dependence and multiple testing should be addressed in new work.",
                "- AUC-over-steps has definitional overlap with final improvement.",
                "- The benchmark port preserves core search strategies while uniformly replacing native execution infrastructure and removing paper writing, reviewing, literature retrieval, VLM plot analysis, and within-step multi-seed averaging.",
                "- The published values are aggregate tables, not raw trajectories; they cannot support our planned paired or hierarchical reanalysis without raw records.",
                "",
                "## Version lineage",
                "",
                "The current 18-task, 12-process-metric study supersedes the older arXiv:2510.10472 eight-task benchmark paper. Do not silently mix their protocols or results.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    artifact_paths = sorted(path for path in out_dir.rglob("*") if path.is_file() and path.name != "provenance_manifest.json")
    manifest = {
        "status": "published_prior_only",
        "source": PAPER,
        "separation_contract": {
            "may_inform": ["Stage-2 skill selection", "Stage-3 planning", "review expectations", "sanity checks", "related work"],
            "may_not_inform": ["protected-test search feedback", "new empirical claim unlock", "new campaign sample size", "locally measured uncertainty"],
            "merge_with_new_experiment_rows": False,
        },
        "protocol": {
            "main_agent_n": 6,
            "task_n": 18,
            "round_n": 3,
            "step_budget": 100,
            "main_run_n": 324,
            "backbone": "GPT-5.4",
            "accelerator": "NVIDIA A100 80GB",
        },
        "row_counts": {
            "agent_task": len(agent_task),
            "agent_summary": len(agent_summary),
            "process_metrics": len(processes),
            "task_cards": len(task_cards),
            "search_strategies": len(search_rows),
            "paper_versions": 2,
        },
        "artifacts": [
            {"path": str(path.relative_to(out_dir)), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in artifact_paths
        ],
    }
    manifest_path = out_dir / "provenance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
