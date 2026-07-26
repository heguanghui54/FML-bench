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
import statistics
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


def derived_agent_dispersion_rows() -> list[dict[str, Any]]:
    rows = []
    for agent_index, (agent_id, label) in enumerate(AGENTS):
        values = [cells[agent_index][0] for cells in TABLE2.values()]
        rows.append(
            {
                "evidence_class": PAPER["evidence_class"],
                "paper": PAPER["arxiv_id"] + PAPER["version"],
                "derived_from": "Table 2 rounded per-task means",
                "agent_id": agent_id,
                "paper_agent": label,
                "task_n": len(values),
                "mean_across_task_means": statistics.fmean(values),
                "sample_sd_across_task_means": statistics.stdev(values),
                "minimum_task_mean": min(values),
                "maximum_task_mean": max(values),
                "interpretation": "task heterogeneity, not uncertainty across rounds",
            }
        )
    return rows


def derived_task_discrimination_rows() -> list[dict[str, Any]]:
    cards = {row["paper_task"]: row for row in task_card_rows()}
    rows = []
    for task, cells in TABLE2.items():
        values = [cell[0] for cell in cells]
        best = max(values)
        winners = [AGENTS[index][1] for index, value in enumerate(values) if value == best]
        rows.append(
            {
                "evidence_class": PAPER["evidence_class"],
                "paper": PAPER["arxiv_id"] + PAPER["version"],
                "derived_from": "Table 2 rounded six-agent means and Table 5 opportunity density",
                "task_id": TASK_IDS[task],
                "paper_task": task,
                "agent_n": len(values),
                "mean_across_agents": statistics.fmean(values),
                "sample_sd_across_agents": statistics.stdev(values),
                "range_across_agents": max(values) - min(values),
                "highest_mean_agent": ";".join(winners),
                "highest_mean": best,
                "opportunity_density_published_post_hoc": cards[task]["opportunity_density"],
                "partition_published_post_hoc": cards[task]["partition"],
                "interpretation": "descriptive strategy discrimination based on rounded published means",
            }
        )
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


def _agent_dispersion_figure(rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 820, 560
    left, top, plot_w, plot_h = 105, 58, 620, 410
    x_max, y_max = .22, .26
    body = ['<text x="24" y="28" class="title">Published mean performance versus cross-task variability</text>']
    body.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>')
    for tick in (0, .05, .10, .15, .20):
        x = left + tick / x_max * plot_w
        body.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top+plot_h}" class="grid"/>')
        body.append(f'<text x="{x}" y="{top+plot_h+18}" text-anchor="middle" class="small">{tick:.2f}</text>')
    for tick in (0, .05, .10, .15, .20, .25):
        y = top + plot_h - tick / y_max * plot_h
        body.append(f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" class="grid"/>')
        body.append(f'<text x="{left-9}" y="{y+4}" text-anchor="end" class="small">{tick:.2f}</text>')
    for index, row in enumerate(rows):
        mean = float(row["mean_across_task_means"])
        sd = float(row["sample_sd_across_task_means"])
        x = left + mean / x_max * plot_w
        y = top + plot_h - sd / y_max * plot_h
        color = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9")[index]
        label_dx, label_dy = {
            "TAS v2": (10, 18),
            "AutoR": (10, -10),
        }.get(row["paper_agent"], (10, -7))
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}"/>')
        body.append(f'<text x="{x+label_dx:.1f}" y="{y+label_dy:.1f}" class="label">{_escape(row["paper_agent"])}</text>')
    body.append(f'<text x="{left+plot_w/2}" y="{height-52}" text-anchor="middle" class="label">Mean normalized improvement across 18 task means</text>')
    body.append(f'<text x="22" y="{top+plot_h/2}" transform="rotate(-90 22 {top+plot_h/2})" text-anchor="middle" class="label">Sample SD across task means</text>')
    body.append('<text x="24" y="538" class="small">Derived from rounded Table 2 means. Vertical variation is task heterogeneity, not a confidence interval.</text>')
    _save_svg(path, "\n".join(body), width, height, "Agent performance and task heterogeneity", "Mean published agent improvement plotted against sample standard deviation across eighteen task means.")


def _task_discrimination_figure(rows: list[dict[str, Any]], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: float(row["range_across_agents"]), reverse=True)
    width, height = 1000, 690
    left, top, plot_w, row_h = 190, 52, 650, 32
    maximum = max(float(row["range_across_agents"]) for row in ordered)
    body = ['<text x="24" y="28" class="title">Published task discrimination among six search strategies</text>']
    for index, row in enumerate(ordered):
        y = top + index * row_h
        value = float(row["range_across_agents"])
        bar = value / maximum * plot_w
        color = "#009E73" if row["partition_published_post_hoc"] == "DENSE-OPP" else "#56B4E9"
        body.append(f'<text x="{left-8}" y="{y+19}" text-anchor="end" class="label">{_escape(row["paper_task"])}</text>')
        body.append(f'<rect x="{left}" y="{y+3}" width="{bar:.1f}" height="21" fill="{color}"/>')
        body.append(f'<text x="{left+bar+7:.1f}" y="{y+19}" class="small">range {value:.3f}; best {_escape(row["highest_mean_agent"])}</text>')
    body.append('<text x="24" y="666" class="small">Range of rounded three-round means; descriptive only. Green=dense and blue=sparse under the published post-hoc partition.</text>')
    _save_svg(path, "\n".join(body), width, height, "Task discrimination among strategies", "Across-agent range of published mean normalized improvement for each FML task.")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_paper_synthesis(
    path: Path,
    agent_summary: list[dict[str, Any]],
    processes: list[dict[str, Any]],
    task_discrimination: list[dict[str, Any]],
) -> None:
    significant = [row for row in processes if row["significant_unadjusted_p_lt_0_05"]]
    discriminating = sorted(task_discrimination, key=lambda row: float(row["range_across_agents"]), reverse=True)[:6]
    lines = [
        "# Paper-Ready FML Prior Evidence Brief",
        "",
        "Status: descriptive synthesis of published aggregate evidence, not a manuscript result from the new campaign.",
        "",
        f"Primary source: [{PAPER['title']}]({PAPER['abstract_url']}), `{PAPER['arxiv_id']}{PAPER['version']}`.",
        "",
        "## What may be claimed from the published study",
        "",
        "The controlled comparison isolates search strategy by sharing code editing, execution, metric presentation, and validation/test separation. Six main agents were evaluated on eighteen tasks for three rounds and 100 validation steps per run; AdaptiveSearch was subsequently reported as a seventh strategy. These facts provide prior context and hypotheses, not evidence that the present pipeline is superior.",
        "",
        "| Agent | Published mean normalized improvement | Published pairwise win rate |",
        "| --- | ---: | ---: |",
    ]
    for row in agent_summary:
        lines.append(f"| {row['paper_agent']} | {float(row['mean_normalized_test_improvement']):.3f} | {float(row['pairwise_win_rate_percent']):.1f}% |")
    lines.extend(
        [
            "",
            "Safe interpretation: AdaptiveSearch has the highest reported aggregate values in Table 3; TAS v2 and AutoResearch are nearly tied in mean improvement. Do not call these differences statistically significant from the aggregate table alone.",
            "",
            "## Published process associations",
            "",
            "The following four pooled Spearman correlations have unadjusted `p < 0.05` in the paper:",
            "",
            "| Process metric | rho | Unadjusted p | Required qualification |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in significant:
        p = float(row["p_value_unadjusted"])
        p_label = f"{p:.2g}" if p >= .001 else f"{p:.1e}"
        qualification = "partly overlaps with final improvement" if row["metric"] == "AUC-over-steps" else "pooled cells; dependence and multiplicity remain"
        lines.append(f"| {row['metric']} | {float(row['pooled_spearman_rho']):+.3f} | {p_label} | {qualification} |")
    lines.extend(
        [
            "",
            "Use association language only. The pooled correlations do not establish that changing a process metric will causally improve final performance.",
            "",
            "## Checked-in scorer fidelity notes",
            "",
            "The source-level audit must accompany any reproduced result. The current scorer embeds every persisted step snapshot for exploration metrics, whereas the paper describes valid-step embeddings. It also reports the last successful exact match to `best_val_metric`, whereas the paper describes the step where the peak was first achieved. The campaign freezes the checked-in implementation for official comparability and may report valid-only/first-achieved variants only as separately named sensitivity analyses.",
            "",
            "## Tasks most useful for diagnosing strategy differences",
            "",
            "This ranking is derived from the across-agent range of rounded Table 2 means. It is descriptive and should guide diagnostic pilots, not determine the confirmatory task set after outcomes are known.",
            "",
            "| Task | Across-agent range | Highest published mean | Post-hoc regime |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in discriminating:
        lines.append(f"| {row['paper_task']} | {float(row['range_across_agents']):.3f} | {row['highest_mean_agent']} ({float(row['highest_mean']):.3f}) | {row['partition_published_post_hoc']} |")
    lines.extend(
        [
            "",
            "## Questions to preregister for the new campaign",
            "",
            "1. Under the same model, step budget, task version, and independent seeds, does AdaptiveSearch improve mean FML-Lite normalized improvement over fixed greedy, tree, MCTS, and evolutionary search?",
            "2. Do published dense/sparse opportunity labels predict the direction of paired strategy differences on new independent runs? This is a confirmatory test of a published post-hoc hypothesis, not reuse of the original evidence.",
            "3. Do evidence-gated memory and node-local review improve early AUC, valid-step ratio, and final normalized improvement without increasing protected-test leakage or invalid executions?",
            "4. Given the same frozen empirical evidence, does the full writing/review/amendment pipeline improve blinded paper-quality scores and hard-gate pass rates over one-shot writing?",
            "",
            "## Minimum result package for the new paper",
            "",
            "- Native task metrics, normalized improvement, failures, and constraints for every planned run.",
            "- Per-task/per-trial rows before aggregate tables; no removal after protected-test exposure.",
            "- Complete task blocks within each trial, paired agent comparisons, uncertainty, and multiplicity labels.",
            "- All twelve process metrics with separate units and figures.",
            "- Search-regime, memory, review-loop, and paper-writing ablations under matched budgets.",
            "- Claim-to-evidence hashes, code/task/model versions, seeds, hardware, costs, and negative results.",
            "",
            "## Figure captions ready for adaptation",
            "",
            "- `published_agent_summary.svg`: Published aggregate normalized improvement and pairwise win rate for seven FML search strategies. Values are prior evidence from Table 3 and are not new campaign measurements.",
            "- `published_agent_task_heatmap.svg`: Three-round mean normalized test improvement for six main agents on eighteen tasks. Cell-level standard deviations remain in the companion CSV.",
            "- `published_process_correlations.svg`: Pooled Spearman associations between twelve process metrics and final improvement. Highlighting denotes unadjusted significance and does not imply causality.",
            "- `published_opportunity_density.svg`: Published post-hoc task opportunity density and median split; the classification is hypothesis-generating.",
            "- `derived_agent_mean_vs_cross_task_sd.svg`: Aggregate performance versus sample standard deviation across task means; the vertical axis measures task heterogeneity, not round-level uncertainty.",
            "- `derived_task_strategy_discrimination.svg`: Across-agent range of rounded task means, used only to choose diagnostic pilot coverage before new outcomes are observed.",
            "",
            "## Claims that remain locked",
            "",
            "No statement about the new pipeline's experimental superiority, generalization, cost effectiveness, paper quality, or skill evolution is permitted until the corresponding frozen local campaign and blinded paper-evaluation artifacts exist.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_published_prior(out_dir: Path) -> dict[str, Any]:
    """Write source-labeled published tables, charts, notes, and manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    agent_task = agent_task_rows()
    agent_summary = agent_summary_rows()
    processes = process_rows()
    task_cards = task_card_rows()
    agent_dispersion = derived_agent_dispersion_rows()
    task_discrimination = derived_task_discrimination_rows()
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
    _write_csv(out_dir / "derived_agent_cross_task_dispersion.csv", agent_dispersion, list(agent_dispersion[0]))
    _write_csv(out_dir / "derived_task_strategy_discrimination.csv", task_discrimination, list(task_discrimination[0]))
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
    _agent_dispersion_figure(agent_dispersion, figures / "derived_agent_mean_vs_cross_task_sd.svg")
    _task_discrimination_figure(task_discrimination, figures / "derived_task_strategy_discrimination.svg")
    _write_paper_synthesis(out_dir / "paper_ready_prior_evidence_brief.md", agent_summary, processes, task_discrimination)

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
            "derived_agent_dispersion": len(agent_dispersion),
            "derived_task_discrimination": len(task_discrimination),
        },
        "artifacts": [
            {"path": str(path.relative_to(out_dir)), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in artifact_paths
        ],
    }
    manifest_path = out_dir / "provenance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
