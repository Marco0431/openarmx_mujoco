"""运行数据独立于控制器；目录从入口显式传入。"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class RunPaths:
    project_root: Path
    cache_root: Path
    output_root: Path
    entry_file: Path


@dataclass(slots=True)
class RunResult:
    log: list[dict[str, Any]] = field(default_factory=list)
    horizon: list[dict[str, Any]] = field(default_factory=list)
    startup: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    failure: dict[str, Any] = field(default_factory=dict)
    joint_limit_trace: dict[str, Any] = field(default_factory=dict)
    physics_trace: dict[str, Any] = field(default_factory=dict)
    motion_force_trace: dict[str, Any] = field(default_factory=dict)
    motion_force_path: str | None = None
    dynamics_trace: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    disturbance_log: str | None = None
