"""Reusable AutoML experiment skeleton extracted from the March Madness project."""

from .architecture import build_architecture_report
from .exploration import ExplorationPolicy
from .generator import ExperimentGenerator
from .models import ConfigChange, ExperimentProposal, ExperimentResult, Hypothesis, ProjectSpec
from .plugins import PluginRegistry, PluginSpec
from .registry import ExperimentRegistry
from .runner import AutoMLRunner, ProjectAdapter
from .tracking import BacklogStore, DailyReportBuilder

__all__ = [
    "AutoMLRunner",
    "BacklogStore",
    "build_architecture_report",
    "ConfigChange",
    "DailyReportBuilder",
    "ExperimentGenerator",
    "ExperimentProposal",
    "ExperimentRegistry",
    "ExperimentResult",
    "ExplorationPolicy",
    "Hypothesis",
    "PluginRegistry",
    "PluginSpec",
    "ProjectAdapter",
    "ProjectSpec",
]
