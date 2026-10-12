"""唯一的扁平参数布局边界：物理参数 201 + eta 24 = 225。

模型和控制流程访问命名字段，数字槽位仅出现在此模块。质量矩阵按 CasADi
列主序编码。旧 191 维输出只在保存历史格式日志时使用。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import casadi as ca
import numpy as np
from .config import ControllerConfig, SolveOptions
from .data import DynamicsSnapshot, JointState, ReferenceHorizon, Array

PHYSICAL_SIZE = 201
FULL_SIZE = 225
LAYOUT_VERSION = "physical201_eta24_v1"


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


def view(p: Any) -> ParameterView:
    symbolic = isinstance(p, (ca.SX, ca.MX, ca.DM))
    reshape = (lambda value: ca.reshape(value, 7, 7)) if symbolic else (lambda value: value.reshape(7, 7, order="F"))
    return ParameterView((reshape(p[0:49]), reshape(p[56:105])),
                         (p[49:56], p[105:112]), (p[112:119], p[119:126]),
                         (p[126:132], p[132:138]), (p[138:144], p[144:150]),
                         (p[150:156], p[156:162]), (p[162:169], p[169:176]),
                         (p[176:182], p[182:188]), p[188],
                         (p[189:195], p[195:201]),
                         (p[201:207], p[207:213]), (p[213:219], p[219:225]))


def encode(dynamics: DynamicsSnapshot, references: ReferenceHorizon,
           bias: Array, load: Array, cfg: ControllerConfig,
           options: SolveOptions | None = None) -> Array:
    options = options or SolveOptions()
    count = cfg.N + 1
    if len(references.node_times) != count:
        raise ValueError(f"references must include all {count} nodes")
    p = np.zeros((count, FULL_SIZE))
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
    """新布局到旧日志/oracle 布局；不作为 Controller 输入接口。"""
    values = np.asarray(p, dtype=float)
    if values.shape[-1] not in (PHYSICAL_SIZE, FULL_SIZE):
        raise ValueError("expected shared parameter width 201 or 225")
    if variant == "handheld":
        return values[..., :PHYSICAL_SIZE].copy()
    if variant != "flat":
        raise ValueError("variant must be flat or handheld")
    return np.concatenate((values[..., :176], values[..., 176:177],
                           values[..., 182:183], values[..., 188:201]), axis=-1)


def to_legacy_full(p: Array, variant: str) -> Array:
    """完整失败参数的日志边界：物理段转旧格式，末尾保留全部 eta。"""
    values = np.asarray(p, dtype=float)
    if values.ndim == 0 or values.shape[-1] != FULL_SIZE:
        raise ValueError("expected shared full parameter width 225")
    return np.concatenate((to_legacy(values, variant), values[..., PHYSICAL_SIZE:]), axis=-1)


def from_legacy(p: Array, variant: str, cfg: ControllerConfig) -> Array:
    values = np.asarray(p, dtype=float)
    expected = 191 if variant == "flat" else 201 if variant == "handheld" else -1
    if values.shape[-1] != expected:
        raise ValueError(f"expected {expected} legacy parameters")
    result = np.zeros((*values.shape[:-1], FULL_SIZE))
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
    return to_legacy(encode(dynamics, references, anchoring["bias"],
                            contact_load["frozen_residual"], cfg, options), variant)
