"""结构配置与单周期覆盖项。默认值保持当前实验的控制配置。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral
import numpy as np
from .data import Array, checked, check_scalar


class ForceFeedbackSource(StrEnum):
    NORMAL_MAGNITUDE = "normal_magnitude"
    WORLD_FORCE_X = "world_force_x"


class BiasPolicy(StrEnum):
    X_ONLY = "x_only"
    LEGACY_SIX_AXIS = "legacy_six_axis"


class ContactLoadMode(StrEnum):
    FULL_RESIDUAL = "full_residual"
    TANGENT = "tangent"


class NormalForceLaw(StrEnum):
    SMOOTH_POSITIVE = "smooth_positive"
    IDEAL_UNILATERAL = "ideal_unilateral"
    MEASURED_ANCHORED_IDEAL = "measured_anchored_ideal"


@dataclass(slots=True)
class ControllerConfig:
    N: int = 10
    Tp: float = 0.1
    eta_cop: Array = field(default_factory=lambda: np.tile([0., .8, 0., 0., 0., 0.], (2, 1)))
    eta_f: Array = field(default_factory=lambda: np.tile([.8, 0., 0., 0., 0., 0.], (2, 1)))
    Q: float = 1.0
    P: float = 2.0
    eta_dev: float = 0.05
    eta_u: float = 5e-8
    kx: float = 3750.0
    rest_gap: float = 0.248
    kf_scale: float = 0.003
    smooth_eps: float = 0.002
    vmax: float = 0.05
    vz_max: float = 0.05
    reg_method: str = "MIRROR"
    lm_reg: float = 1e-2
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    sim_method_num_steps: int = 2
    sim_method_num_stages: int = 4
    integrator_type: str = "ERK"
    qp_solver_cond_N: int | None = None
    qp_solver_iter_max: int = 15
    nlp_solver_max_iter: int = 20
    fail_on_nonzero_status: bool = True
    rebuild_on_failure: bool = True
    force_feedback_source: ForceFeedbackSource = ForceFeedbackSource.NORMAL_MAGNITUDE
    bias_policy: BiasPolicy = BiasPolicy.X_ONLY
    contact_load_mode: ContactLoadMode = ContactLoadMode.FULL_RESIDUAL
    # 缺少此字段的历史配置仍重建原平滑模型；场景显式选择新接触律。
    normal_force_law: NormalForceLaw = NormalForceLaw.SMOOTH_POSITIVE

    def __post_init__(self) -> None:
        for name in ("Tp", "Q", "P", "eta_dev", "eta_u", "kx", "rest_gap", "kf_scale",
                     "smooth_eps", "vmax", "vz_max", "lm_reg"):
            check_scalar(getattr(self, name), name)
        for name in ("eta_cop", "eta_f"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape == (6,):
                value = np.tile(value, (2, 1))
            setattr(self, name, checked(value, (2, 6), name))
        if isinstance(self.N, bool) or not isinstance(self.N, Integral) or self.N < 1 or self.Tp <= 0:
            raise ValueError("N must be a positive integer and Tp must be positive")
        self.force_feedback_source = ForceFeedbackSource(self.force_feedback_source)
        self.bias_policy = BiasPolicy(self.bias_policy)
        self.contact_load_mode = ContactLoadMode(self.contact_load_mode)
        self.normal_force_law = NormalForceLaw(self.normal_force_law)


@dataclass(slots=True)
class RuntimeWeights:
    Q: float | None = None
    P: float | None = None
    eta_dev: float | None = None
    eta_u: float | None = None
    ed_x: float | None = None

    def __post_init__(self) -> None:
        for name in ("Q", "P", "eta_dev", "eta_u", "ed_x"):
            value = getattr(self, name)
            if value is not None:
                check_scalar(value, name)


@dataclass(slots=True)
class SolveOptions:
    u_seed: Array | None = None
    u_prev: Array | None = None
    du_max: Array | float | None = None
    u_lim: Array | float | None = None
    kf_scale: float | None = None
    eta_f: Array | None = None
    eta_cop: Array | None = None
    weights: RuntimeWeights | None = None
