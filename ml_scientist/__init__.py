"""Evidence-gated, self-evolving ML research-to-paper pipeline."""

from .catalog import build_catalog
from .governance import EvidenceGatedSkillRegistry
from .planner import build_research_paper_plan, validate_plan

__all__ = ["build_catalog", "build_research_paper_plan", "validate_plan", "EvidenceGatedSkillRegistry"]
