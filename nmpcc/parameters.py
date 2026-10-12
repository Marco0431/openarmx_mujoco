"""扁平参数布局：旧律物理参数 201 + eta 24 = 225，新测量锚定律追加 3 = 228。

模型和控制流程访问命名字段，数字槽位仅出现在此模块。质量矩阵按 CasADi
列主序编码。0:225 索引含义不变；225 为当前总间隙，226:228 为两侧正法向力。
旧 191/201 维输出只作历史日志兼容视图，无法独立重放测量锚定律。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import casadi as ca
import numpy as np
from .config import ControllerConfig, NormalForceLaw, SolveOptions
from .data import DynamicsSnapshot, JointState, ReferenceHorizon, Array

PHYSICAL_SIZE = 201
FULL_SIZE = 225
LAYOUT_VERSION = "physical201_eta24_v1"
ANCHORED_SIZE = FULL_SIZE + 3
ANCHORED_LAYOUT_VERSION = "physical201_eta24_anchor_gap1_normal2_v2"


def parameter_size(cfg: ControllerConfig) -> int:
    return ANCHORED_SIZE if cfg.normal_force_law == NormalForceLaw.MEASURED_ANCHORED_IDEAL else FULL_SIZE


def layout_version(cfg: ControllerConfig) -> str:
    return ANCHORED_LAYOUT_VERSION if parameter_size(cfg) == ANCHORED_SIZE else LAYOUT_VERSION


def checked_anchor(anchor_gap: Any, anchor_normal: Any) -> tuple[float, Array]:
    """法向锚点是正压力；仅消除不超过 1e-9 N 的负浮点噪声。"""
    gap, normal = np.asarray(anchor_gap, float), np.asarray(anchor_normal, float)
    if gap.size != 1 or normal.shape != (2,) or not np.isfinite(gap).all() or not np.isfinite(normal).all():
        raise ValueError("measured_anchored_ideal requires finite anchor_gap and anchor_normal shape (2,)")
    if np.any(normal < -1e-9):
        raise ValueError(f"negative normal-force anchor: {normal.tolist()}; check left/right world-x force signs")
    return float(gap.ravel()[0]), np.maximum(normal, 0.)


@dataclass(slots=True)
class ParameterView:
    mass: tuple[Any, Any]
    gravity: tuple[Any, Any]
    load: tuple[Any, Any]
    poses: tuple[Any, Any]
    velocities: tuple[Any, Any]
    forces: tuple[Any, Any]
    damping: tuple[Any, Any]
    bias: tuple[Any, Any]
    kf: Any
    force_rates: tuple[Any, Any]
    eta_cop: tuple[Any, Any]
    eta_f: tuple[Any, Any]
    anchor_gap: Any = None
    anchor_normal: Any = None


def view(p: Any) -> ParameterView:
    symbolic = isinstance(p, (ca.SX, ca.MX, ca.DM))
    size = int(p.numel()) if symbolic else np.asarray(p).size
    if size not in (FULL_SIZE, ANCHORED_SIZE):
        raise ValueError("expected shared parameter width 225 or 228")
    reshape = (lambda value: ca.reshape(value, 7, 7)) if symbolic else (lambda value: value.reshape(7, 7, order="F"))
    return ParameterView((reshape(p[0:49]), reshape(p[56:105])),
                         (p[49:56], p[105:112]), (p[112:119], p[119:126]),
                         (p[126:132], p[132:138]), (p[138:144], p[144:150]),
                         (p[150:156], p[156:162]), (p[162:169], p[169:176]),
                         (p[176:182], p[182:188]), p[188],
                         (p[189:195], p[195:201]),
                         (p[201:207], p[207:213]), (p[213:219], p[219:225]),
                         p[225] if size == ANCHORED_SIZE else None,
                         p[226:228] if size == ANCHORED_SIZE else None)


def encode(dynamics: DynamicsSnapshot, references: ReferenceHorizon,
           bias: Array, load: Array, cfg: ControllerConfig,
           options: SolveOptions | None = None, *,
           anchor_gap: Any = None, anchor_normal: Any = None) -> Array:
    options = options or SolveOptions()
    count = cfg.N + 1
    if len(references.node_times) != count:
        raise ValueError(f"references must include all {count} nodes")
    p = np.zeros((count, parameter_size(cfg)))
    if parameter_size(cfg) == ANCHORED_SIZE:
        gap, normal = checked_anchor(anchor_gap, anchor_normal)
        p[:, 225], p[:, 226:228] = gap, normal
    p[:, :49] = dynamics.mass[0].ravel(order="F")
    p[:, 49:56] = dynamics.gravity[0]
    p[:, 56:105] = dynamics.mass[1].ravel(order="F")
    p[:, 105:112] = dynamics.gravity[1]
    p[:, 112:126] = np.asarray(load).reshape(14)
    p[:, 126:138] = references.poses.reshape(count, 12)
    p[:, 138:150] = references.velocities.reshape(count, 12)
    p[:, 150:162] = references.forces.reshape(count, 12)
    p[:, 162:176] = dynamics.damping.reshape(14)
    p[:, 176:188] = np.asarray(bias).reshape(12)
    p[:, 188] = cfg.kf_scale if options.kf_scale is None else options.kf_scale
    p[:, 189:201] = references.force_rates.reshape(count, 12)
    p[:, 201:213] = np.broadcast_to(cfg.eta_cop if options.eta_cop is None else options.eta_cop, (2, 6)).reshape(12)
    p[:, 213:225] = np.broadcast_to(cfg.eta_f if options.eta_f is None else options.eta_f, (2, 6)).reshape(12)
    if not np.isfinite(p).all():
        raise ValueError("parameters contain non-finite values")
    return p


def to_legacy(p: Array, variant: str) -> Array:
    """旧日志/oracle 的有损兼容视图；228 参数的锚点须另存，不可据此重放新律。"""
    values = np.asarray(p, dtype=float)
    if values.ndim == 0 or values.shape[-1] not in (PHYSICAL_SIZE, FULL_SIZE, ANCHORED_SIZE):
        raise ValueError("expected shared parameter width 201, 225 or 228")
    if variant == "handheld":
        return values[..., :PHYSICAL_SIZE].copy()
    if variant != "flat":
        raise ValueError("variant must be flat or handheld")
    return np.concatenate((values[..., :176], values[..., 176:177],
                           values[..., 182:183], values[..., 188:201]), axis=-1)


def to_legacy_full(p: Array, variant: str) -> Array:
    """失败参数的旧兼容视图：保留全部 eta，但不包含新增锚点。"""
    values = np.asarray(p, dtype=float)
    if values.ndim == 0 or values.shape[-1] not in (FULL_SIZE, ANCHORED_SIZE):
        raise ValueError("expected shared full parameter width 225 or 228")
    return np.concatenate((to_legacy(values, variant), values[..., PHYSICAL_SIZE:FULL_SIZE]), axis=-1)


def from_legacy(p: Array, variant: str, cfg: ControllerConfig, *,
                anchor_gap: Any = None, anchor_normal: Any = None) -> Array:
    values = np.asarray(p, dtype=float)
    expected = 191 if variant == "flat" else 201 if variant == "handheld" else -1
    if values.shape[-1] != expected:
        raise ValueError(f"expected {expected} legacy parameters")
    result = np.zeros((*values.shape[:-1], parameter_size(cfg)))
    if parameter_size(cfg) == ANCHORED_SIZE:
        gap, normal = checked_anchor(anchor_gap, anchor_normal)
        result[..., 225], result[..., 226:228] = gap, normal
    result[..., :176] = values[..., :176]
    if variant == "flat":
        result[..., 176] = values[..., 176]
        result[..., 182] = values[..., 177]
        result[..., 188:201] = values[..., 178:191]
    else:
        result[..., :201] = values
    result[..., 201:213] = cfg.eta_cop.ravel()
    result[..., 213:225] = cfg.eta_f.ravel()
    return result


def encode_legacy_horizon(state: JointState, dynamics: DynamicsSnapshot,
                          references: ReferenceHorizon, anchoring: dict[str, Array],
                          contact_load: dict[str, Array], cfg: ControllerConfig,
                          variant: str = "flat", options: SolveOptions | None = None) -> Array:
    """有损旧格式视图；新律的精确重放请保存 encode_shared_horizon 的完整结果。"""
    return to_legacy(encode_shared_horizon(state, dynamics, references, anchoring,
                                          contact_load, cfg, options), variant)


def encode_shared_horizon(state: JointState, dynamics: DynamicsSnapshot,
                          references: ReferenceHorizon, anchoring: dict[str, Array],
                          contact_load: dict[str, Array], cfg: ControllerConfig,
                          options: SolveOptions | None = None) -> Array:
    """重建完整参数；state 不重新锚定，始终使用该次求解保存的锚点。"""
    return encode(dynamics, references, anchoring["bias"], contact_load["frozen_residual"],
                  cfg, options, anchor_gap=anchoring.get("anchor_gap"),
                  anchor_normal=anchoring.get("anchor_normal"))
