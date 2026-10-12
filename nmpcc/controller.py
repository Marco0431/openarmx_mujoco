"""每周期采样参数、锚定接触力、冻结负载残差和执行 SQP_RTI。"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
import time
import numpy as np
from .config import BiasPolicy, ContactLoadMode, ControllerConfig, ForceFeedbackSource, SolveOptions
from .data import Array, DynamicsSnapshot, JointState, Prediction, ReferenceHorizon, SolveResult, checked
from .model import build_model
from .ocp import Bounds, build_ocp, cost_weights, create_solver, solver_identity
from .parameters import encode, view


class SolverError(RuntimeError):
    """公开的求解器异常；failure 保存独立的当前与首次失败现场。"""

    def __init__(self, message: str, failure: dict[str, Any]) -> None:
        super().__init__(message)
        self.failure = {key: value.copy() if isinstance(value, np.ndarray) else value
                        for key, value in failure.items()}


@dataclass(slots=True)
class PreparedCycle:
    state: Array
    parameters: Array
    anchoring: dict[str, Array]
    contact_load: dict[str, Array]


class Controller:
    """共享控制器；构造时创建求解器，几何 FK 和缓存根目录由调用方指定。"""

    def __init__(self, cfg: ControllerConfig, fk: dict[str, Any], *, cache_root: str | Path,
                 umin: Array | None = None, umax: Array | None = None,
                 qmin: Array | None = None, qmax: Array | None = None,
                 verbose: bool = False, solver_factory: Callable[..., Any] | None = None) -> None:
        self.config = cfg
        self.bounds = Bounds.create(umin, umax, qmin, qmax)
        self.symbolic = build_model(cfg, fk)
        self.name = solver_identity(cfg, fk, self.bounds)
        self._ocp, self._json_file = build_ocp(cfg, self.symbolic, self.bounds, self.name, cache_root, verbose)
        self.node_times = np.r_[0., np.cumsum(self._ocp.solver_options.time_steps)]
        self._solver_factory = solver_factory or create_solver
        self._solver = self._solver_factory(self._ocp, self._json_file, verbose=verbose, rebuild=False)
        self._weights_overridden = False
        self.reset()

    @property
    def metadata(self) -> dict[str, Any]:
        weights, terminal_weights = cost_weights(self.config)
        return {"name": self.name, "W": weights, "W_e": terminal_weights,
                "config": asdict(self.config)}

    def reset(self) -> None:
        """丢弃热启动与首次失败现场；下一次求解写满所有 N+1 状态节点。"""
        self._warm: tuple[Array, Array] | None = None
        self._first_failure: dict[str, Any] | None = None
        if hasattr(self._solver, "reset"):
            self._solver.reset()

    def _prepare(self, state: JointState, dynamics: DynamicsSnapshot,
                 references: ReferenceHorizon, options: SolveOptions) -> PreparedCycle:
        if references.node_times.shape != self.node_times.shape or not np.allclose(references.node_times, self.node_times, rtol=0., atol=1e-12):
            raise ValueError("reference node times must match controller node_times")
        x = state.vector()
        nominal = np.asarray(self.symbolic.nominal_force(x), float).reshape(2, 6)
        measured = np.zeros((2, 6))
        contact = dynamics.contact
        if self.config.force_feedback_source == ForceFeedbackSource.NORMAL_MAGNITUDE:
            measured[:, 0] = [-contact.normal_magnitude[0], contact.normal_magnitude[1]]
        else:
            measured[:, 0] = contact.force_world[:, 0]
        if self.config.bias_policy == BiasPolicy.LEGACY_SIX_AXIS:
            measured[:, 1:3] = contact.force_world[:, 1:3]
            # 保留手持旧反馈口径；真实接触面 wrench 不进入旧控制反馈。
            measured[:, 3:] = contact.contact_torque_world
        bias = measured - nominal
        tangent = contact.tangent_joint_load.copy()
        p = encode(dynamics, references, bias, tangent, self.config, options)
        # 按动力学回算法保留负载抵消的浮点顺序；矩阵使用参数中的列主序视图。
        current = p[0].copy()
        params = view(current)
        for load in params.load:
            load[:] = 0.
        derivative = np.asarray(self.symbolic.dynamics_fn(x, np.zeros(14), current), float).ravel()
        normal = (np.r_[params.mass[0] @ derivative[7:14], params.mass[1] @ derivative[21:28]]
                  + np.r_[params.gravity[0], params.gravity[1]]
                  + np.r_[*params.damping] * state.dq.ravel()).reshape(2, 7)
        residual = contact.joint_load - normal if self.config.contact_load_mode == ContactLoadMode.FULL_RESIDUAL else tangent
        # 同一周期只求一次当前法向负载，并把差值冻结到包括终端的所有节点。
        p = encode(dynamics, references, bias, residual, self.config, options)
        return PreparedCycle(x, p,
                             {"nominal": nominal, "measured": measured, "bias": bias, "anchored": nominal+bias},
                             {"full_contact": contact.joint_load.copy(), "model_normal": normal,
                              "tangent_contact": tangent, "frozen_residual": residual,
                              "correction": residual-tangent})

    def equilibrium_seed(self, state: JointState, dynamics: DynamicsSnapshot,
                         references: ReferenceHorizon, options: SolveOptions | None = None) -> Array:
        prepared = self._prepare(state, dynamics, references, options or SolveOptions())
        derivative = np.asarray(self.symbolic.dynamics_fn(prepared.state, np.zeros(14), prepared.parameters[0]), float).ravel()
        params = view(prepared.parameters[0])
        seed = -np.r_[params.mass[0] @ derivative[7:14], params.mass[1] @ derivative[21:28]]
        return checked(seed, (14,), "equilibrium seed")

    def acceleration(self, state: JointState, dynamics: DynamicsSnapshot,
                     references: ReferenceHorizon, torque: Array,
                     options: SolveOptions | None = None, *,
                     frozen_solution: SolveResult | None = None) -> Array:
        """诊断加速度；指定求解结果时沿用其冻结载荷，避免质量矩阵对照重算残差。"""
        if frozen_solution is None:
            prepared = self._prepare(state, dynamics, references, options or SolveOptions())
            x, parameters = prepared.state, prepared.parameters
        else:
            x = state.vector()
            parameters = encode(dynamics, references, frozen_solution.anchoring["bias"],
                                frozen_solution.contact_load["frozen_residual"], self.config, options)
        derivative = np.asarray(self.symbolic.dynamics_fn(x, checked(torque, (14,), "torque"), parameters[0]), float).ravel()
        return np.stack((derivative[7:14], derivative[21:28]))

    def _initialize(self, x: Array, seed: Array | None) -> None:
        if self._warm is None:
            states = np.broadcast_to(x, (self.config.N+1, 28))
            controls = np.broadcast_to(np.zeros(14) if seed is None else checked(seed, (14,), "u_seed"), (self.config.N, 14))
        else:
            states, controls = self._warm
        for i, value in enumerate(states):
            self._solver.set(i, "x", value)
        for i, value in enumerate(controls):
            self._solver.set(i, "u", value)

    def _apply_cycle(self, prepared: PreparedCycle, options: SolveOptions) -> None:
        self._initialize(prepared.state, options.u_seed)
        for i, row in enumerate(prepared.parameters):
            self._solver.set(i, "p", row)
            self._solver.set(i, "yref", np.zeros(24 if i == self.config.N else 38))
        lo, hi = self.bounds.lower_torque.copy(), self.bounds.upper_torque.copy()
        if options.u_lim is not None:
            limit = np.broadcast_to(np.asarray(options.u_lim, float), (14,))
            if not np.isfinite(limit).all() or np.any(limit < 0):
                raise ValueError("u_lim must be finite and nonnegative")
            lo, hi = np.maximum(lo, -limit), np.minimum(hi, limit)
        if np.any(lo > hi):
            raise ValueError("runtime torque bounds have empty intersection")
        for i in range(self.config.N):
            self._solver.constraints_set(i, "lbu", lo)
            self._solver.constraints_set(i, "ubu", hi)
        if (options.u_prev is None) != (options.du_max is None):
            raise ValueError("u_prev and du_max must be supplied together")
        if options.u_prev is not None:
            center = checked(options.u_prev, (14,), "u_prev")
            delta = np.broadcast_to(np.asarray(options.du_max, float), (14,))
            first_lo, first_hi = np.maximum(lo, center-delta), np.minimum(hi, center+delta)
            if not np.isfinite(delta).all() or np.any(delta < 0) or np.any(first_lo > first_hi):
                raise ValueError("invalid first-stage torque increment bounds")
            self._solver.constraints_set(0, "lbu", first_lo)
            self._solver.constraints_set(0, "ubu", first_hi)
        if options.weights is not None or self._weights_overridden:
            weights, terminal = cost_weights(self.config, options.weights)
            for i in range(self.config.N):
                self._solver.cost_set(i, "W", weights)
            self._solver.cost_set(self.config.N, "W", terminal)
            self._weights_overridden = options.weights is not None

    def _run_solver(self, state: Array) -> int:
        try:
            self._solver.solve_for_x0(x0_bar=state,
                                      fail_on_nonzero_status=self.config.fail_on_nonzero_status,
                                      print_stats_on_failure=True)
            return int(self._solver.status)
        except RuntimeError:
            return 4  # 保留原程序 fail_on_nonzero_status 抛出异常时的状态约定。

    def solve(self, state: JointState, dynamics: DynamicsSnapshot,
              references: ReferenceHorizon, options: SolveOptions | None = None) -> SolveResult:
        options = options or SolveOptions()
        prepared = self._prepare(state, dynamics, references, options)
        self._apply_cycle(prepared, options)
        started = time.perf_counter()
        initial_status = status = self._run_solver(prepared.state)
        if status:
            # 重建前保存当前迭代；此前已记录的首次失败不能替代本次现场。
            current_failure = {"status": status, "x0": prepared.state.copy(),
                               "p_full": prepared.parameters.copy(),
                               "x": np.array([self._solver.get(i, "x") for i in range(self.config.N+1)]),
                               "u": np.array([self._solver.get(i, "u") for i in range(self.config.N)])}
            if self._first_failure is None:
                self._first_failure = current_failure
        if status and self.config.rebuild_on_failure:
            try:
                self._solver = self._solver_factory(self._ocp, self._json_file, verbose=False, rebuild=True)
            except Exception as exc:
                failure = {"status": status, "x0": current_failure["x0"],
                           "p_full": current_failure["p_full"],
                           "pred_x": current_failure["x"], "pred_u": current_failure["u"]}
                failure.update({"first_"+key: value for key, value in self._first_failure.items()})
                raise SolverError(f"solver recreation failed after status {status}: {exc}", failure) from exc
            self._apply_cycle(prepared, options)
            status = self._run_solver(prepared.state)
        statistics: dict[str, float | int] = {"solve_wall_ms": (time.perf_counter()-started)*1000.}
        for source, target, multiplier in (("time_tot", "solver_ms", 1000.), ("qp_iter", "qp_iter", 1.),
                                            ("time_qp", "time_qp", 1.), ("time_sim", "time_sim", 1.), ("time_lin", "time_lin", 1.)):
            try:
                value = float(np.asarray(self._solver.get_stats(source)).ravel()[0])*multiplier
                statistics[target] = int(value) if source == "qp_iter" else value
            except (KeyError, ValueError, IndexError, AttributeError, RuntimeError):
                statistics[target] = -1 if source == "qp_iter" else float("nan")
        states = np.array([self._solver.get(i, "x") for i in range(self.config.N+1)])
        controls = np.array([self._solver.get(i, "u") for i in range(self.config.N)])
        self._warm = states.copy(), controls.copy()
        diagnostics = [tuple(np.asarray(value, float).ravel() for value in self.symbolic.diagnostic_fn(x, p))
                       for x, p in zip(states, prepared.parameters)]
        fields = [np.array([row[i] for row in diagnostics]) for i in range(8)]
        prediction = Prediction(states, controls, references, fields[0].reshape(-1, 2, 6),
                                fields[1].reshape(-1, 2, 6), fields[2], fields[3], fields[4], fields[5],
                                fields[6].reshape(-1, 2, 6), fields[7].reshape(-1, 2, 6))
        return SolveResult(controls[0].copy(), prediction, status, initial_status, statistics,
                           self._first_failure, prepared.anchoring, prepared.contact_load)
