from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .models import ConfigChange, to_json_ready


@dataclass(slots=True)
class PluginSpec:
    name: str
    kind: str
    description: str
    config_path: str
    default_value: object | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def build_change(self, value: object, description: str | None = None) -> ConfigChange:
        return ConfigChange(
            target=self.config_path,
            value=value,
            description=description or f"Set {self.config_path}={value!r}",
            tags=list(self.tags),
        )

    def to_dict(self) -> dict[str, object]:
        return to_json_ready(self)


class PluginRegistry:
    def __init__(self, kind: str):
        self.kind = kind
        self._plugins: dict[str, PluginSpec] = {}

    def register(self, plugin: PluginSpec) -> PluginSpec:
        if plugin.kind != self.kind:
            raise ValueError(f"Plugin {plugin.name!r} has kind {plugin.kind!r}, expected {self.kind!r}")
        self._plugins[plugin.name] = plugin
        return plugin

    def get(self, name: str) -> PluginSpec:
        return self._plugins[name]

    def values(self) -> list[PluginSpec]:
        return list(self._plugins.values())

    def items(self) -> Iterable[tuple[str, PluginSpec]]:
        return self._plugins.items()

    def names(self) -> list[str]:
        return sorted(self._plugins)

    def to_dict(self) -> dict[str, object]:
        return {name: plugin.to_dict() for name, plugin in self._plugins.items()}
