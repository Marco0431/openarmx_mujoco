"""唯一的 28 状态/14 力矩符号模型，场景差异由 FK 和测量策略提供。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import casadi as ca
from .config import ControllerConfig, NormalForceLaw
from .parameters import FULL_SIZE, view


@dataclass(slots=True)
class SymbolicModel:
    x: Any
    u: Any
    p: Any
    dynamics: Any
    residual: Any
    terminal_residual: Any
    constraints: Any
    nominal_force: ca.Function
    normal_load: ca.Function
    dynamics_fn: ca.Function
    residual_fn: ca.Function
    diagnostic_fn: ca.Function


def build_model(cfg: ControllerConfig, fk: dict[str, Any]) -> SymbolicModel:
    if fk["n_arm"] != 7:
        raise ValueError("the shared model requires seven joints per arm")
    x, u, p = ca.SX.sym("x", 28), ca.SX.sym("u", 14), ca.SX.sym("p", FULL_SIZE)
    params = view(p)
    q, dq = (x[:7], x[14:21]), (x[7:14], x[21:28])
    source = fk["tips_qt"]
    tips = ca.substitute([fk["tips"]], [source[:7], source[7:]], list(q))[0]
    contact_expr = fk.get("contact_tips", ca.vertcat(fk["tips"][:3], fk["tips"][6:9]))
    contacts = ca.substitute([contact_expr], [source[:7], source[7:]], list(q))[0]
    compression = cfg.rest_gap - (contacts[3] - contacts[0])
    if cfg.normal_force_law == NormalForceLaw.IDEAL_UNILATERAL:
        # 严格 c>0 分支：c=0 选择零导数，避免 fmax 在折点采用 1/2。
        compressed = ca.if_else(compression > 0., compression, 0.)
    else:
        compressed = .5 * (compression + ca.sqrt(compression * compression + cfg.smooth_eps ** 2))
    left = ca.vertcat(-cfg.kx * compressed, ca.SX.zeros(5))
    nominal = (left, -left)
    predicted = tuple(nominal[i] + params.bias[i] for i in range(2))
    # 只有世界 x 方向法向力进入预测动力学；剩余负载由冻结残差包含。
    normal_load = tuple(ca.jacobian(contacts[3*i:3*i+3], q[i]).T @
                        ca.vertcat(predicted[i][0], 0., 0.) for i in range(2))
    accelerations = tuple(ca.solve(params.mass[i], u[7*i:7*i+7] + params.load[i]
                          - params.gravity[i] - params.damping[i] * dq[i])
                          + ca.solve(params.mass[i], normal_load[i]) for i in range(2))
    dynamics = ca.vertcat(dq[0], accelerations[0], dq[1], accelerations[1])
    velocities = ca.jacobian(tips, x) @ dynamics
    compression_gradient = ca.jacobian(compressed, x)
    rate = compression_gradient[:7] @ dq[0] + compression_gradient[14:21] @ dq[1]
    left_rate = ca.vertcat(-cfg.kx * rate, ca.SX.zeros(5))
    force_rates = (left_rate, -left_rate)
    errors, error_rates = [], []
    for i in range(2):
        j = 1 - i
        cop, force = params.eta_cop[i], params.eta_f[i]
        errors.append((1-force) * (tips[6*i:6*i+6] - (1-cop)*params.poses[i]
                       - cop*tips[6*j:6*j+6]) - force*params.kf*(predicted[i]-params.forces[i]))
        error_rates.append((1-force) * (velocities[6*i:6*i+6] - (1-cop)*params.velocities[i]
                            - cop*velocities[6*j:6*j+6]) - force*params.kf*(force_rates[i]-params.force_rates[i]))
    terminal = ca.vertcat(*errors, *error_rates)
    residual = ca.vertcat(terminal, u)
    speed = [ca.sqrt(ca.dot(velocities[6*i:6*i+3], velocities[6*i:6*i+3])+1e-10) for i in range(2)]
    constraints = ca.vertcat(*speed, velocities[2], velocities[8])
    diagnostics = [tips, velocities, ca.vertcat(nominal[0][0], nominal[1][0]),
                   ca.vertcat(predicted[0][0], predicted[1][0]),
                   ca.vertcat(predicted[0][0]-params.forces[0][0], predicted[1][0]-params.forces[1][0]),
                   ca.vertcat(tips[0]-params.poses[0][0], tips[6]-params.poses[1][0]),
                   ca.vertcat(*errors), ca.vertcat(*error_rates)]
    return SymbolicModel(x, u, p, dynamics, residual, terminal, constraints,
                         ca.Function("shared_nominal", [x], [ca.vertcat(*nominal)]),
                         ca.Function("shared_normal_load", [x, p], [ca.vertcat(*normal_load)]),
                         ca.Function("shared_dynamics", [x, u, p], [dynamics]),
                         ca.Function("shared_residual", [x, u, p], [residual]),
                         ca.Function("shared_diagnostic", [x, p], diagnostics))
