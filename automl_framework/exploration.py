from __future__ import annotations

from dataclasses import dataclass

from .models import ExperimentProposal, ExperimentResult, Hypothesis
from .registry import ExperimentRegistry


@dataclass(slots=True)
class ExplorationPolicy:
    max_attempts_per_path: int = 2
    require_single_change: bool = True
    improvement_threshold: float = 0.0

    def rank_hypotheses(self, hypotheses: list[Hypothesis], registry: ExperimentRegistry) -> list[Hypothesis]:
        candidates = []
        abandoned = set(registry.data.get("summary", {}).get("abandoned_paths", []))
        for hypothesis in hypotheses:
            attempts = registry.path_attempts(hypothesis.path_key)
            if hypothesis.status != "pending":
                continue
            if hypothesis.path_key in abandoned:
                continue
            if attempts >= self.max_attempts_per_path and not registry.path_has_improvement(hypothesis.path_key):
                continue
            candidates.append((attempts, -hypothesis.priority, hypothesis.title.lower(), hypothesis))
        return [item[-1] for item in sorted(candidates)]

    def validate_proposal(self, proposal: ExperimentProposal) -> None:
        if self.require_single_change and len(proposal.changes) != 1:
            raise ValueError("Exploration policy requires exactly one controlled change per experiment")

    def mark_improvement(self, result: ExperimentResult, registry: ExperimentRegistry) -> bool:
        metric_name = registry.metric
        if not metric_name or metric_name not in result.metrics:
            return False
        best = registry.best_metric()
        value = result.metrics[metric_name]
        if best is None:
            return True
        if registry.metric_mode == "max":
            return value > (best + self.improvement_threshold)
        return value < (best - self.improvement_threshold)
