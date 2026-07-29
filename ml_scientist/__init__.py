"""Evidence-gated, self-evolving ML research-to-paper pipeline."""

from .catalog import build_catalog
from .benchmark_learning import build_fml_benchmark_contract
from .governance import EvidenceGatedSkillRegistry
from .knowledge_graph import build_project_knowledge_graph, query_knowledge_graph, validate_knowledge_graph
from .policy_model import build_agent_policy_specs, validate_policy_composition
from .planner import build_research_paper_plan, validate_plan
from .adaptive_runtime import AdaptiveResearchRuntime
from .skill_evolution import SkillEvolutionEngine

__all__ = [
    "AdaptiveResearchRuntime",
    "EvidenceGatedSkillRegistry",
    "SkillEvolutionEngine",
    "build_catalog",
    "build_fml_benchmark_contract",
    "build_project_knowledge_graph",
    "build_agent_policy_specs",
    "build_research_paper_plan",
    "query_knowledge_graph",
    "validate_knowledge_graph",
    "validate_policy_composition",
    "validate_plan",
]
