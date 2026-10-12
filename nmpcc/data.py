"""控制周期的数据契约；数组顺序始终为左臂、右臂，采用 SI 单位。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def check_scalar(value: Any, name: str) -> None:
    scalar = np.asarray(value, dtype=float)
    if scalar.shape != () or not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite scalar")


def checked(value: Any, shape: tuple[int, ...], name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {result.shape}")
    return result.copy()


@dataclass(slots=True)
class JointState:
    q: Array
    dq: Array
    time: float = 0.0

    def __post_init__(self) -> None:
        self.q = checked(self.q, (2, 7), "q")
        self.dq = checked(self.dq, (2, 7), "dq")
        check_scalar(self.time, "time")

    def vector(self) -> Array:
        return np.concatenate((self.q[0], self.dq[0], self.q[1], self.dq[1]))


@dataclass(slots=True)
class ContactMeasurement:
    normal_magnitude: Array
    force_world: Array
    contact_torque_world: Array  # 原始接触力矩之和；不含参考点平移项。
    wrench_world: Array  # 各 pad 接触面中心处的真实 wrench，仅用于观测。
    reference_points_world: Array
    joint_load: Array
    tangent_joint_load: Array
    time: float = 0.0

    def __post_init__(self) -> None:
        for name, shape in (("normal_magnitude", (2,)), ("force_world", (2, 3)),
                            ("contact_torque_world", (2, 3)), ("wrench_world", (2, 6)),
                            ("reference_points_world", (2, 3)), ("joint_load", (2, 7)),
                            ("tangent_joint_load", (2, 7))):
            setattr(self, name, checked(getattr(self, name), shape, name))
        check_scalar(self.time, "time")


@dataclass(slots=True)
class DynamicsSnapshot:
    mass: Array
    gravity: Array
    damping: Array
    contact: ContactMeasurement
    time: float = 0.0

    def __post_init__(self) -> None:
        self.mass = checked(self.mass, (2, 7, 7), "mass")
        self.gravity = checked(self.gravity, (2, 7), "gravity")
        self.damping = checked(self.damping, (2, 7), "damping")
        check_scalar(self.time, "time")


@dataclass(slots=True)
class ReferenceHorizon:
    node_times: Array  # 相对当前周期的节点时刻，秒；必须覆盖终端 N。
    poses: Array
    velocities: Array
    forces: Array
    force_rates: Array

    def __post_init__(self) -> None:
        self.node_times = np.asarray(self.node_times, dtype=float).copy()
        if self.node_times.ndim != 1 or not np.isfinite(self.node_times).all():
            raise ValueError("node_times must be a finite vector")
        for name in ("poses", "velocities", "forces", "force_rates"):
            setattr(self, name, checked(getattr(self, name), (len(self.node_times), 2, 6), name))


@dataclass(slots=True)
class Prediction:
    states: Array
    controls: Array
    references: ReferenceHorizon
    poses: Array
    velocities: Array
    force_nominal: Array
    forces: Array
    force_error: Array
    position_error_x: Array
    residuals: Array
    residual_rates: Array


@dataclass(slots=True)
class SolveResult:
    torque: Array
    prediction: Prediction
    status: int
    initial_status: int
    statistics: dict[str, float | int]
    failure: dict[str, Any] | None
    anchoring: dict[str, Array]
    contact_load: dict[str, Array]
