from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ExperimentResult, ProjectSpec, to_json_ready


class ExperimentRegistry:
    def __init__(self, path: str | Path, project: ProjectSpec | None = None):
        self.path = Path(path)
        self.project = project
        self.data = self._load_or_create()

    def _default_data(self) -> dict[str, Any]:
        return {
            "project": self.project.to_dict() if self.project else {},
            "summary": {
                "metric": self.project.metric if self.project else None,
                "metric_mode": self.project.metric_mode if self.project else "min",
                "best_metric": None,
                "best_experiment_id": None,
                "total_experiments": 0,
                "path_attempts": {},
                "promoted_paths": [],
                "abandoned_paths": [],
            },
            "experiments": [],
        }

    def _load_or_create(self) -> dict[str, Any]:
        if self.path.exists():
            with open(self.path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            if self.project and not data.get("project"):
                data["project"] = self.project.to_dict()
            data.setdefault("summary", self._default_data()["summary"])
            data.setdefault("experiments", [])
            return data
        data = self._default_data()
        self.save(data)
        return data

    @property
    def metric(self) -> str | None:
        return self.data.get("summary", {}).get("metric") or (self.project.metric if self.project else None)

    @property
    def metric_mode(self) -> str:
        return self.data.get("summary", {}).get("metric_mode") or (self.project.metric_mode if self.project else "min")

    def save(self, data: dict[str, Any] | None = None) -> None:
        if data is not None:
            self.data = data
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as handle:
            json.dump(self.data, handle, indent=2)

    def experiments(self) -> list[dict[str, Any]]:
        return list(self.data.get("experiments", []))

    def best_metric(self) -> float | None:
        return self.data.get("summary", {}).get("best_metric")

    def best_experiment_id(self) -> str | None:
        return self.data.get("summary", {}).get("best_experiment_id")

    def next_experiment_number(self) -> int:
        return len(self.data.get("experiments", [])) + 1

    def path_attempts(self, path_key: str) -> int:
        return int(self.data.get("summary", {}).get("path_attempts", {}).get(path_key, 0))

    def path_has_improvement(self, path_key: str) -> bool:
        return any(exp.get("path_key") == path_key and exp.get("improved") for exp in self.data.get("experiments", []))

    def last_experiment(self) -> dict[str, Any] | None:
        experiments = self.data.get("experiments", [])
        return experiments[-1] if experiments else None

    def record(self, result: ExperimentResult) -> dict[str, Any]:
        metric_name = self.metric
        result_dict = result.to_dict()
        existing = any(exp.get("experiment_id") == result.experiment_id for exp in self.data.get("experiments", []))
        experiments = [exp for exp in self.data.get("experiments", []) if exp.get("experiment_id") != result.experiment_id]
        experiments.append(result_dict)
        experiments.sort(key=lambda item: item.get("created_at", ""))
        self.data["experiments"] = experiments
        self.data["summary"]["total_experiments"] = len(experiments)

        attempts = self.data["summary"].setdefault("path_attempts", {})
        if not existing:
            attempts[result.path_key] = attempts.get(result.path_key, 0) + 1
        else:
            attempts.setdefault(result.path_key, 0)

        if result.improved and result.path_key not in self.data["summary"].setdefault("promoted_paths", []):
            self.data["summary"]["promoted_paths"].append(result.path_key)
        if not result.improved and attempts[result.path_key] >= 2 and not self.path_has_improvement(result.path_key):
            abandoned = self.data["summary"].setdefault("abandoned_paths", [])
            if result.path_key not in abandoned:
                abandoned.append(result.path_key)

        if metric_name and metric_name in result.metrics:
            metric_value = result.metrics[metric_name]
            current_best = self.best_metric()
            if current_best is None or self._is_better(metric_value, current_best):
                self.data["summary"]["best_metric"] = metric_value
                self.data["summary"]["best_experiment_id"] = result.experiment_id

        self.save()
        return result_dict

    def _is_better(self, candidate: float, incumbent: float) -> bool:
        if self.metric_mode == "max":
            return candidate > incumbent
        return candidate < incumbent

    def to_dict(self) -> dict[str, Any]:
        return to_json_ready(self.data)
