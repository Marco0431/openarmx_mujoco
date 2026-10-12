"""OCP 约束、数值选项和缓存身份；不执行仿真。"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from .config import ControllerConfig, RuntimeWeights
from .data import Array
from .model import SymbolicModel
from .parameters import FULL_SIZE, LAYOUT_VERSION

MODEL_VERSION = "shared_v2"


@dataclass(slots=True)
class Bounds:
    lower_torque: Array
    upper_torque: Array
    lower_joint: Array
    upper_joint: Array

    @classmethod
    def create(cls, umin: Any = None, umax: Any = None,
               qmin: Any = None, qmax: Any = None) -> Bounds:
        if (qmin is None) != (qmax is None):
            raise ValueError("qmin and qmax must be supplied together")
        result = cls(np.full(14, -8.) if umin is None else np.asarray(umin, float).copy(),
                     np.full(14, 8.) if umax is None else np.asarray(umax, float).copy(),
                     np.full(14, -np.inf) if qmin is None else np.asarray(qmin, float).copy(),
                     np.full(14, np.inf) if qmax is None else np.asarray(qmax, float).copy())
        if any(value.shape != (14,) for value in (result.lower_torque, result.upper_torque,
                                                  result.lower_joint, result.upper_joint)):
            raise ValueError("bounds must have shape (14,)")
        if not np.isfinite(result.lower_torque).all() or not np.isfinite(result.upper_torque).all() or np.any(result.lower_torque >= result.upper_torque):
            raise ValueError("torque bounds must be finite with lower < upper")
        lo, hi = result.lower_joint, result.upper_joint
        if not np.all((np.isfinite(lo) & np.isfinite(hi)) | (np.isneginf(lo) & np.isposinf(hi))) or np.any(lo >= hi):
            raise ValueError("joint bounds need lower < upper, or (-inf,+inf)")
        return result


def cost_weights(cfg: ControllerConfig, override: RuntimeWeights | None = None) -> tuple[Array, Array]:
    values = {key: getattr(cfg, key) for key in ("Q", "P", "eta_dev", "eta_u")}
    if override is not None:
        values.update({key: getattr(override, key) for key in values if getattr(override, key) is not None})
    derivative = np.full(12, values["eta_dev"])
    if override is not None and override.ed_x is not None:
        derivative[[0, 6]] = override.ed_x
    return (np.diag(values["Q"] * np.r_[np.ones(12), derivative, np.full(14, values["eta_u"])]),
            np.diag(values["P"] * np.r_[np.ones(12), derivative]))


def solver_identity(cfg: ControllerConfig, fk: dict[str, Any], bounds: Bounds) -> str:
    settings = asdict(cfg)
    # 运行时 kf、eta 和失败策略不影响生成代码；其余结构均进入指纹。
    for key in ("kf_scale", "eta_cop", "eta_f", "fail_on_nonzero_status", "rebuild_on_failure"):
        settings.pop(key)
    payload = {"version": MODEL_VERSION, "layout": LAYOUT_VERSION, "settings": settings,
               "bounds": {key: value.tolist() for key, value in asdict(bounds).items()},
               "tips": fk["tips_fn"].serialize(),
               "contacts": fk["contact_tips_fn"].serialize() if "contact_tips_fn" in fk else None}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return f"nmpcc_{MODEL_VERSION}_{digest}"


def build_ocp(cfg: ControllerConfig, symbolic: SymbolicModel, bounds: Bounds,
              name: str, cache_root: str | Path, verbose: bool = False) -> tuple[AcadosOcp, Path]:
    model = AcadosModel()
    model.name = name
    model.x, model.u, model.p = symbolic.x, symbolic.u, symbolic.p
    model.f_expl_expr = symbolic.dynamics
    model.cost_y_expr = symbolic.residual
    model.cost_y_expr_e = symbolic.terminal_residual
    model.con_h_expr = symbolic.constraints
    ocp = AcadosOcp()
    ocp.model = model
    options = ocp.solver_options
    options.N_horizon, options.tf = cfg.N, cfg.Tp
    options.time_steps = np.full(cfg.N, cfg.Tp / cfg.N)
    ocp.cost.cost_type = ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W, ocp.cost.W_e = cost_weights(cfg)
    ocp.cost.yref, ocp.cost.yref_e = np.zeros(38), np.zeros(24)
    ocp.constraints.idxbu = np.arange(14)
    ocp.constraints.lbu, ocp.constraints.ubu = bounds.lower_torque.copy(), bounds.upper_torque.copy()
    finite = np.isfinite(bounds.lower_joint)
    if finite.any():
        indices = np.r_[0:7, 14:21][finite]
        for suffix in ("", "_e"):
            setattr(ocp.constraints, "idxbx" + suffix, indices.copy())
            setattr(ocp.constraints, "lbx" + suffix, bounds.lower_joint[finite].copy())
            setattr(ocp.constraints, "ubx" + suffix, bounds.upper_joint[finite].copy())
    ocp.constraints.lh = np.array([0., 0., -cfg.vz_max, -cfg.vz_max])
    ocp.constraints.uh = np.array([cfg.vmax, cfg.vmax, cfg.vz_max, cfg.vz_max])
    ocp.constraints.x0 = np.zeros(28)
    ocp.parameter_values = np.zeros(FULL_SIZE)
    options.nlp_solver_type = "SQP_RTI"
    options.qp_solver = cfg.qp_solver
    options.qp_solver_iter_max = cfg.qp_solver_iter_max
    options.hessian_approx = "GAUSS_NEWTON"
    options.regularize_method = cfg.reg_method
    if cfg.lm_reg > 0:
        options.levenberg_marquardt = cfg.lm_reg
    options.integrator_type = cfg.integrator_type
    options.sim_method_num_steps = cfg.sim_method_num_steps
    options.sim_method_num_stages = cfg.sim_method_num_stages
    if cfg.qp_solver_cond_N is not None:
        options.qp_solver_cond_N = cfg.qp_solver_cond_N
    options.nlp_solver_max_iter = cfg.nlp_solver_max_iter
    options.print_level = int(verbose)
    directory = Path(cache_root).resolve() / name.rsplit("_", 1)[-1]
    json_file = directory / f"{name}.json"
    ocp.code_gen_options.code_export_directory = str(directory)
    ocp.code_gen_options.json_file = str(json_file)
    return ocp, json_file


_dll_directories: list[Any] = []


def create_solver(ocp: AcadosOcp, json_file: Path, *, verbose: bool = False,
                  rebuild: bool = False) -> Any:
    """新建求解器实例，优先复用当前静态 OCP 的代码缓存。

    rebuild 表示失败后的实例重建，不强制重编译。Windows 不能覆盖仍被旧
    实例加载的 DLL；只有缓存缺失或加载失败时才生成和编译。
    """
    # Python 3.8+ 的 Windows DLL 搜索不保证使用 PATH，保留句柄至进程结束。
    source = os.environ.get("ACADOS_SOURCE_DIR")
    if os.name == "nt" and source:
        directory = Path(source) / "bin"
        if directory.is_dir():
            _dll_directories.append(os.add_dll_directory(str(directory)))
    json_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        # 本模块的结构指纹已隔离缓存；禁止模板层把复用尝试自动升级为编译。
        return AcadosOcpSolver(ocp, json_file=str(json_file), build=False, generate=False,
                               check_reuse_possible=False, verbose=verbose)
    except (FileNotFoundError, OSError):
        return AcadosOcpSolver(ocp, json_file=str(json_file), build=True, generate=True, verbose=verbose)
