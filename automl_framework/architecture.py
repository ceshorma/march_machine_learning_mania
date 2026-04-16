from __future__ import annotations

from pathlib import Path

from .runner import ProjectAdapter


CORE_MODULES: tuple[tuple[str, str], ...] = (
    ("automl_framework.models", "Canonical contracts for project specs, hypotheses, config deltas, proposals, and results."),
    ("automl_framework.plugins", "Registries for reusable feature, model, and transform switches."),
    ("automl_framework.exploration", "Policy that enforces single-change proposals and path-level exploration limits."),
    ("automl_framework.generator", "Proposal generator that applies one mutation to a baseline config."),
    ("automl_framework.registry", "Structured experiment tracking, best-metric summary, lineage, and path attempts."),
    ("automl_framework.tracking", "Backlog persistence and daily-report generation."),
    ("automl_framework.runner", "Adapter contract plus the orchestration entrypoint that glues the core together."),
    ("automl_framework.cli", "Bootstrap and reporting commands that expose the framework from the repository root."),
)


def _state_name(adapter: ProjectAdapter) -> str:
    return adapter.state_dir().name


def build_architecture_report(adapter: ProjectAdapter) -> str:
    project = adapter.project_spec()
    repo_root = Path(project.paths.get("repo_root", ".")).resolve()
    state_name = _state_name(adapter)
    adapter_name = project.metadata.get("adapter", "project_adapter")
    legacy_runner = project.paths.get("legacy_runner", "")
    legacy_registry = project.paths.get("legacy_registry", "")

    lines = [
        "# AutoML Framework Architecture",
        "",
        "## Operational tree",
        "```text",
        ".",
        "├── automl_framework/",
        "│   ├── models.py",
        "│   ├── plugins.py",
        "│   ├── exploration.py",
        "│   ├── generator.py",
        "│   ├── registry.py",
        "│   ├── tracking.py",
        "│   ├── runner.py",
        "│   ├── architecture.py",
        "│   └── cli.py",
        f"├── examples/{adapter_name}/",
        "│   └── adapter.py",
        f"├── automl_state/{state_name}/",
        "│   ├── project.json",
        "│   ├── backlog.json",
        "│   ├── registry.json",
        "│   ├── proposals/",
        "│   └── reports/",
        "├── templates/project_template/",
        "├── templates/dataset_template/",
        "└── notebooks/run_experiment.py",
        "```",
        "",
        "## Module responsibilities",
    ]
    for module_name, description in CORE_MODULES:
        lines.append(f"- `{module_name}`: {description}")

    lines.extend(
        [
            "",
            "## Core vs adapter boundary",
            "### Core owns",
            "- experiment contracts and serialization",
            "- plugin registries and config mutations",
            "- proposal ranking and single-change generation",
            "- experiment registry, path promotion, and abandonment rules",
            "- backlog storage and framework-level reports",
            "",
            "### Adapter owns",
            "- project objective, metric, dataset metadata, and filesystem bindings",
            "- baseline experiment/config selection",
            "- hypothesis backlog for the domain",
            "- feature/model/transform registrations tied to the domain",
            "- mapping between generic config changes and the legacy runner/config schema",
            "",
            "### March Madness stays outside the core",
            f"- adapter module: `{repo_root / 'examples' / adapter_name / 'adapter.py'}`",
            f"- legacy runner: `{legacy_runner}`",
            f"- legacy experiment registry: `{legacy_registry}`",
            f"- framework incubator state: `{adapter.state_dir()}`",
            "",
            "## Migration sequence",
            "1. Stabilize the reusable core interfaces (`runner`, `registry`, `hypothesis`, `plugins`, `generator`).",
            "2. Keep March Madness as the incubator/example project through the adapter.",
            "3. Move new NCAA/Kaggle feature logic behind the adapter instead of the core.",
            "4. Open a separate repository only after the core contract and at least one reusable template are stable.",
        ]
    )
    return "\n".join(lines) + "\n"
