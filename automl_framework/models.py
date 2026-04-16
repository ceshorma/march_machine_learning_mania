from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def to_json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_json_ready(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: to_json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_ready(v) for v in value]
    return value


@dataclass(slots=True)
class ProjectSpec:
    name: str
    objective: str
    metric: str
    metric_mode: str = "min"
    dataset: dict[str, Any] = field(default_factory=dict)
    splitter: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_json_ready(self)


@dataclass(slots=True)
class Hypothesis:
    id: str
    title: str
    description: str
    change_kind: str
    priority: int = 50
    path_key: str = "default"
    dependencies: list[str] = field(default_factory=list)
    success_criteria: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_json_ready(self)


@dataclass(slots=True)
class ConfigChange:
    target: str
    value: Any
    change_type: str = "set"
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_json_ready(self)


@dataclass(slots=True)
class ExperimentProposal:
    id: str
    parent_id: str | None
    hypothesis_id: str
    path_key: str
    rationale: str
    base_config_id: str | None
    config: dict[str, Any]
    changes: list[ConfigChange]
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return to_json_ready(self)


@dataclass(slots=True)
class ExperimentResult:
    experiment_id: str
    status: str
    metrics: dict[str, float]
    hypothesis_id: str | None = None
    path_key: str = "default"
    parent_id: str | None = None
    improved: bool = False
    key_findings: str = ""
    changes: list[ConfigChange] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    finished_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_json_ready(self)
