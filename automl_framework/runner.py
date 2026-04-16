from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from .generator import ExperimentGenerator
from .models import ExperimentProposal, ExperimentResult, Hypothesis, ProjectSpec
from .plugins import PluginRegistry
from .registry import ExperimentRegistry
from .tracking import BacklogStore, DailyReportBuilder


class ProjectAdapter(ABC):
    @abstractmethod
    def project_spec(self) -> ProjectSpec:
        raise NotImplementedError

    @abstractmethod
    def baseline_config(self) -> tuple[str | None, dict]:
        raise NotImplementedError

    @abstractmethod
    def feature_registry(self) -> PluginRegistry:
        raise NotImplementedError

    @abstractmethod
    def model_registry(self) -> PluginRegistry:
        raise NotImplementedError

    @abstractmethod
    def transform_registry(self) -> PluginRegistry:
        raise NotImplementedError

    @abstractmethod
    def hypotheses(self) -> list[Hypothesis]:
        raise NotImplementedError

    @abstractmethod
    def mutation_space(self) -> dict[str, list]:
        raise NotImplementedError

    def state_dir(self) -> Path:
        project = self.project_spec()
        state_path = project.paths.get('framework_state')
        if state_path:
            return Path(state_path)
        raise ValueError('ProjectSpec.paths.framework_state is required')


class AutoMLRunner:
    def __init__(self, adapter: ProjectAdapter, generator: ExperimentGenerator | None = None):
        self.adapter = adapter
        self.project = adapter.project_spec()
        self.generator = generator or ExperimentGenerator()
        state_dir = adapter.state_dir()
        self.registry = ExperimentRegistry(state_dir / 'registry.json', project=self.project)
        self.backlog = BacklogStore(state_dir / 'backlog.json')

    def bootstrap(self) -> None:
        state_dir = self.adapter.state_dir()
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / 'reports').mkdir(exist_ok=True)
        (state_dir / 'proposals').mkdir(exist_ok=True)
        with open(state_dir / 'project.json', 'w', encoding='utf-8') as handle:
            json.dump(self.project.to_dict(), handle, indent=2)
        if not self.backlog.path.exists():
            self.backlog.save(self.adapter.hypotheses())
        self.registry.save()

    def propose_next(self) -> ExperimentProposal:
        hypotheses = self.backlog.load() or self.adapter.hypotheses()
        base_config_id, baseline_config = self.adapter.baseline_config()
        proposal = self.generator.propose(
            baseline_config=baseline_config,
            base_config_id=base_config_id,
            hypotheses=hypotheses,
            mutation_space=self.adapter.mutation_space(),
            registry=self.registry,
        )
        proposal_path = self.adapter.state_dir() / 'proposals' / f'{proposal.id}.json'
        proposal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(proposal_path, 'w', encoding='utf-8') as handle:
            json.dump(proposal.to_dict(), handle, indent=2)
        return proposal

    def record_result(self, result: ExperimentResult) -> dict:
        result.improved = result.improved or self.generator.policy.mark_improvement(result, self.registry)
        saved = self.registry.record(result)
        if result.hypothesis_id:
            self.backlog.mark_status(result.hypothesis_id, 'tested')
        return saved

    def build_report(self) -> str:
        hypotheses = self.backlog.load() or self.adapter.hypotheses()
        report = DailyReportBuilder(self.project, self.registry, hypotheses).build()
        report_path = self.adapter.state_dir() / 'reports' / 'latest.md'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding='utf-8')
        return report
