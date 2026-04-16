from __future__ import annotations

import argparse
import json
from pathlib import Path

from automl_framework.models import ProjectSpec
from automl_framework.runner import AutoMLRunner
from examples.march_madness.adapter import MarchMadnessAdapter


def default_registry_payload(project: ProjectSpec) -> dict:
    return {
        "project": project.to_dict(),
        "summary": {
            "metric": "cv_score",
            "metric_mode": "min",
            "best_metric": None,
            "best_experiment_id": None,
            "total_experiments": 0,
            "path_attempts": {},
            "promoted_paths": [],
            "abandoned_paths": [],
        },
        "experiments": [],
    }


def bootstrap_generic(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / 'reports').mkdir(exist_ok=True)
    (target / 'proposals').mkdir(exist_ok=True)
    project = ProjectSpec(
        name='automl-template',
        objective='replace with your supervised learning objective',
        metric='cv_score',
        metric_mode='min',
        dataset={'train_path': 'data/train.csv', 'target': 'target'},
        splitter={'type': 'temporal', 'min_train_periods': 5},
        paths={'framework_state': str(target)},
    )
    (target / 'project.json').write_text(json.dumps(project.to_dict(), indent=2), encoding='utf-8')
    (target / 'registry.json').write_text(
        json.dumps(default_registry_payload(project), indent=2),
        encoding='utf-8',
    )
    (target / 'backlog.json').write_text(json.dumps({'hypotheses': []}, indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Generic AutoML experiment skeleton CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    bootstrap = subparsers.add_parser('bootstrap-template', help='Create a generic framework_state skeleton')
    bootstrap.add_argument('target', type=Path)

    march = subparsers.add_parser('march-madness-proposal', help='Generate the next proposal for this repository')
    march.add_argument('--repo-root', type=Path, required=True)
    march.add_argument('--output', type=Path)

    report = subparsers.add_parser('march-madness-report', help='Build the latest framework report for this repository')
    report.add_argument('--repo-root', type=Path, required=True)

    args = parser.parse_args()
    if args.command == 'bootstrap-template':
        bootstrap_generic(args.target)
        return

    adapter = MarchMadnessAdapter(args.repo_root)
    runner = AutoMLRunner(adapter)
    runner.bootstrap()

    if args.command == 'march-madness-proposal':
        proposal = runner.propose_next()
        payload = json.dumps(proposal.to_dict(), indent=2)
        if args.output:
            args.output.write_text(payload, encoding='utf-8')
        else:
            print(payload)
    elif args.command == 'march-madness-report':
        print(runner.build_report())


if __name__ == '__main__':
    main()
