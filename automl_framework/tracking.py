        from __future__ import annotations

        import json
        from dataclasses import replace
        from datetime import date
        from pathlib import Path

        from .models import Hypothesis, ProjectSpec
        from .registry import ExperimentRegistry


        class BacklogStore:
            def __init__(self, path: str | Path):
                self.path = Path(path)

            def load(self) -> list[Hypothesis]:
                if not self.path.exists():
                    return []
                with open(self.path, 'r', encoding='utf-8') as handle:
                    payload = json.load(handle)
                items = payload.get('hypotheses', payload)
                return [Hypothesis(**item) for item in items]

            def save(self, hypotheses: list[Hypothesis]) -> None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                payload = {'hypotheses': [hypothesis.to_dict() for hypothesis in hypotheses]}
                with open(self.path, 'w', encoding='utf-8') as handle:
                    json.dump(payload, handle, indent=2)

            def mark_status(self, hypothesis_id: str, status: str) -> None:
                hypotheses = []
                for hypothesis in self.load():
                    if hypothesis.id == hypothesis_id:
                        hypotheses.append(replace(hypothesis, status=status))
                    else:
                        hypotheses.append(hypothesis)
                self.save(hypotheses)


        class DailyReportBuilder:
            def __init__(self, project: ProjectSpec, registry: ExperimentRegistry, hypotheses: list[Hypothesis]):
                self.project = project
                self.registry = registry
                self.hypotheses = hypotheses

            def build(self, report_date: date | None = None) -> str:
                report_date = report_date or date.today()
                pending = [hyp for hyp in self.hypotheses if hyp.status == 'pending']
                recent = self.registry.experiments()[-5:]
                summary = self.registry.data.get('summary', {})
                lines = [
                    f"# Daily AutoML Report — {report_date.isoformat()}",
                    '',
                    '## Project state',
                    f"- Project: {self.project.name}",
                    f"- Objective: {self.project.objective}",
                    f"- Metric: {self.project.metric} ({self.project.metric_mode})",
                    f"- Best metric: {summary.get('best_metric')}",
                    f"- Best experiment: {summary.get('best_experiment_id')}",
                    f"- Total experiments: {summary.get('total_experiments', 0)}",
                    '',
                    '## Exploration status',
                    f"- Promoted paths: {', '.join(summary.get('promoted_paths', [])) or 'None'}",
                    f"- Abandoned paths: {', '.join(summary.get('abandoned_paths', [])) or 'None'}",
                    '',
                    '## Recent experiments',
                ]
                if recent:
                    for experiment in recent:
                        metric = experiment.get('metrics', {}).get(self.project.metric)
                        lines.append(f"- {experiment.get('experiment_id')}: {metric} | improved={experiment.get('improved')} | path={experiment.get('path_key')}")
                else:
                    lines.append('- No experiments recorded yet')
                lines.extend(['', '## Pending hypotheses'])
                if pending:
                    for hypothesis in sorted(pending, key=lambda item: (-item.priority, item.title.lower())):
                        lines.append(f"- [{hypothesis.change_kind}] {hypothesis.title} (priority={hypothesis.priority}, path={hypothesis.path_key})")
                else:
                    lines.append('- No pending hypotheses')
                return '
'.join(lines) + '
'
