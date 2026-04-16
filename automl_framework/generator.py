from __future__ import annotations

import copy
from typing import Any

from .exploration import ExplorationPolicy
from .models import ConfigChange, ExperimentProposal, Hypothesis
from .registry import ExperimentRegistry


def apply_change(config: dict[str, Any], change: ConfigChange) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    cursor: Any = updated
    parts = change.target.split('.')
    for key in parts[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    leaf = parts[-1]
    if change.change_type == 'set':
        cursor[leaf] = change.value
    elif change.change_type == 'increment':
        cursor[leaf] = cursor.get(leaf, 0) + change.value
    elif change.change_type == 'merge':
        existing = cursor.get(leaf, {})
        if not isinstance(existing, dict) or not isinstance(change.value, dict):
            raise ValueError(
                "merge changes require dict values; "
                f"got existing={type(existing).__name__}, value={type(change.value).__name__}"
            )
        merged = dict(existing)
        merged.update(change.value)
        cursor[leaf] = merged
    else:
        raise ValueError(f'Unsupported change_type: {change.change_type}')
    return updated


class ExperimentGenerator:
    def __init__(self, policy: ExplorationPolicy | None = None):
        self.policy = policy or ExplorationPolicy()

    def propose(
        self,
        baseline_config: dict[str, Any],
        base_config_id: str | None,
        hypotheses: list[Hypothesis],
        mutation_space: dict[str, list[ConfigChange]],
        registry: ExperimentRegistry,
        prefix: str = 'exp',
    ) -> ExperimentProposal:
        for hypothesis in self.policy.rank_hypotheses(hypotheses, registry):
            for change in mutation_space.get(hypothesis.id, []):
                if self._is_noop(baseline_config, change):
                    continue
                experiment_id = f"{prefix}_{registry.next_experiment_number():03d}_{self._slug(hypothesis.title)}"
                config = apply_change(baseline_config, change)
                config["id"] = experiment_id
                if base_config_id is not None:
                    config["parent"] = base_config_id
                proposal = ExperimentProposal(
                    id=experiment_id,
                    parent_id=registry.best_experiment_id() or base_config_id,
                    hypothesis_id=hypothesis.id,
                    path_key=hypothesis.path_key,
                    rationale=hypothesis.description,
                    base_config_id=base_config_id,
                    config=config,
                    changes=[change],
                    tags=sorted(set(hypothesis.tags + change.tags)),
                )
                self.policy.validate_proposal(proposal)
                return proposal
        ranked = self.policy.rank_hypotheses(hypotheses, registry)
        raise ValueError(
            "No experiment proposal available with the current policy and mutation space "
            f"(ranked_hypotheses={len(ranked)}, mutation_sets={len(mutation_space)})"
        )

    def _is_noop(self, config: dict[str, Any], change: ConfigChange) -> bool:
        cursor: Any = config
        parts = change.target.split('.')
        for key in parts[:-1]:
            if not isinstance(cursor, dict) or key not in cursor:
                return False
            cursor = cursor[key]
        if not isinstance(cursor, dict):
            return False
        return change.change_type == 'set' and cursor.get(parts[-1]) == change.value

    def _slug(self, text: str) -> str:
        slug = ''.join(ch.lower() if ch.isalnum() else '_' for ch in text).strip('_')
        slug = '_'.join(part for part in slug.split('_') if part)
        return slug[:48] or 'proposal'
