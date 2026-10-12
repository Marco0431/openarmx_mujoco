"""接触测量、同拍动力学采样和物理步关节限位记录。

测量不推进仿真。六维接触面力偶仅用于对账，旧接触自身力矩口径单独保留。
Euler 的关节限位反力与步前状态配对，不额外刷新主仿真。
"""
from __future__ import annotations
import json
import time
import mujoco
import numpy as np

def measure_contact(model, data, arms, box_geom_id, *, joint_loads=True):
    """一次遍历得到法向幅值、世界力、真实力偶和关节载荷。

    共同参考点取每侧垫片朝内接触面中心。关节载荷仍在实际接触点映射，
    因此不能再把共同参考点的力臂项加入接触点 Jacobian。
    """
    from nmpcc.data import ContactMeasurement
    normal = np.zeros(2)
    force = np.zeros((2, 3))
    torque = np.zeros((2, 3))
    wrench = np.zeros((2, 6))
    points = np.empty((2, 3))
    full = np.zeros((2, 7))
    tangent = np.zeros((2, 7))
    for side, arm in enumerate(arms):
        pad = arm['pad_id']
        axis = data.geom_xmat[pad].reshape(3, 3)[:, 2]
        inward = data.geom_xpos[box_geom_id] - data.geom_xpos[pad]
        direction = 1.0 if np.dot(axis, inward) >= 0.0 else -1.0
        points[side] = data.geom_xpos[pad] + direction * model.geom_size[pad, 2] * axis
    force6 = np.zeros(6)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    pad_to_side = {arm['pad_id']: side for side, arm in enumerate(arms)}
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.geom1 == box_geom_id and contact.geom2 in pad_to_side:
            side, sign = pad_to_side[contact.geom2], 1.0
        elif contact.geom2 == box_geom_id and contact.geom1 in pad_to_side:
            side, sign = pad_to_side[contact.geom1], -1.0
        else:
            continue
        arm = arms[side]
        mujoco.mj_contactForce(model, data, i, force6)
        frame = contact.frame.reshape(3, 3)
        f = sign * (frame.T @ force6[:3])
        tau = sign * (frame.T @ force6[3:])
        normal[side] += abs(float(force6[0]))
        force[side] += f
        torque[side] += tau
        wrench[side, :3] += f
        wrench[side, 3:] += tau + np.cross(contact.pos - points[side], f)
        if joint_loads:
            mujoco.mj_jac(model, data, jacp, jacr, contact.pos, arm['body_id'])
            jp, jr = jacp[:, arm['dofs']], jacr[:, arm['dofs']]
            full[side] += jp.T @ f + jr.T @ tau
            ft = f.copy()
            ft[0] = 0.0
            tangent[side] += jp.T @ ft + jr.T @ tau
    return ContactMeasurement(normal, force, torque, wrench, points, full, tangent, float(data.time))


def sample_dynamics(model, data, arms, box_geom_id, *, diagnostic=None):
    """每拍一次采样；保持两次 forward 和 finally 恢复真实速度的顺序。"""
    from nmpcc.data import DynamicsSnapshot
    started = time.perf_counter()
    if diagnostic is not None:
        old_mass = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, data, old_mass)
        diagnostic['mass_old'] = np.array([old_mass[np.ix_(a['dofs'], a['dofs'])] for a in arms])
    diagnostic_ms = (time.perf_counter() - started) * 1000.0
    velocity = data.qvel.copy()
    try:
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        gravity = np.array([data.qfrc_bias[a['dofs']] for a in arms])
    finally:
        data.qvel[:] = velocity
        mujoco.mj_forward(model, data)
    full_mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full_mass)
    mass = np.array([full_mass[np.ix_(a['dofs'], a['dofs'])] for a in arms])
    damping = np.array([model.dof_damping[a['dofs']] for a in arms])
    contact = measure_contact(model, data, arms, box_geom_id)
    if diagnostic is not None:
        diagnostic.update(mass_current=mass.copy(), sampling_diagnostic_ms=diagnostic_ms)
    return DynamicsSnapshot(mass, gravity, damping, contact, float(data.time))


def contact_forces(model, data, arms, box_geom_id):
    """每侧法向力绝对值之和（与标定路径同口径，用于兼容比较）。"""
    result = {a["side"]: 0.0 for a in arms}
    force = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        for a in arms:
            if ((c.geom1 == a["pad_id"] and c.geom2 == box_geom_id) or
                    (c.geom2 == a["pad_id"] and c.geom1 == box_geom_id)):
                mujoco.mj_contactForce(model, data, i, force)
                result[a["side"]] += abs(float(force[0]))
    return result


def _rotvec_of_R(R):
    """世界旋转矩阵 (3,3) 转为主值旋转向量，单位 rad。"""
    R = np.asarray(R, float)
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-6:
        return 0.5 * vee
    return 0.5 * theta / np.sin(theta) * vee


def box_attitude(data, box_qpos):
    """freejoint 四元数 -> (精确 rotvec(3), quat_wxyz)。"""
    q = data.qpos[box_qpos + 3:box_qpos + 7].copy()
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return _rotvec_of_R(R.reshape(3, 3)), q


def dynamics_phase(reference, t):
    """将参考时序细分为夹紧、保持、加速、匀速、减速和停止，仅用于统计。"""
    if t < reference.t_ramp:
        return "clamp"
    if t < reference.t_hold_end or reference.move_duration <= 0:
        return "hold"
    tau = t - reference.t_hold_end
    if tau < reference.accel_time:
        return "acceleration"
    if tau < reference.base_time:
        return "cruise"
    if t < reference.move_end:
        return "deceleration"
    return "stopped"


def arm_joint_bounds(model, arms):
    """按控制器左右关节顺序读取物理角度上下界，无限位关节使用无穷边界。"""
    ids = np.concatenate([a["joint_ids"] for a in arms]).astype(int)
    limited = model.jnt_limited[ids].astype(bool)
    return (np.where(limited, model.jnt_range[ids, 0], -np.inf),
            np.where(limited, model.jnt_range[ids, 1], np.inf))


class JointLimitRecorder:
    """按物理步记录关节状态、边界余量和实际限位反力。
    Euler 在更新位置之前计算约束力，因此用步前 q/v 配对该步留下的反力。
    数据先存内存，结束时统一导出，不增加主仿真的前向刷新或逐步磁盘写入。"""
    def __init__(self, model, data, arms, metadata):
        if model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
            raise ValueError("joint-limit trace currently requires Euler step timing")
        self.ids = np.concatenate([a["joint_ids"] for a in arms]).astype(int)
        self.qadr = np.concatenate([a["qpos"] for a in arms]).astype(int)
        self.dofs = np.concatenate([a["dofs"] for a in arms]).astype(int)
        self.lo, self.hi = arm_joint_bounds(model, arms)
        self.metadata = dict(metadata, joint_names=[model.joint(int(j)).name for j in self.ids],
                             state_timing="pre-step q/v paired with Euler step constraint forces")
        self.initial_qpos = data.qpos.copy()
        self.initial_qvel = data.qvel.copy()
        self.rows = []
        self.failure = {}

    def before_step(self, data):
        started = time.perf_counter()
        state = (float(data.time), data.qpos[self.qadr].copy(), data.qvel[self.dofs].copy())
        return state, (time.perf_counter() - started) * 1000.0

    def after_step(self, model, data, captured):
        started = time.perf_counter()
        (t, q, v), prepare_ms = captured
        rows = np.flatnonzero(data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
        force = np.zeros(data.nefc)
        force[rows] = data.efc_force[rows]
        generalized = np.zeros(model.nv)
        if rows.size:
            mujoco.mj_mulJacTVec(model, data, generalized, force)
        active = np.isin(self.ids, data.efc_id[rows])
        lower, upper = q - self.lo, self.hi - q
        self.rows.append((t, q, v, lower, upper, active, generalized[self.dofs].copy(),
                          prepare_ms + (time.perf_counter() - started) * 1000.0))

    def export(self, horizon):
        fields = ("t", "q", "v", "lower_margin", "upper_margin", "active", "limit_torque", "diagnostic_ms")
        out = {key: np.asarray([r[i] for r in self.rows]) for i, key in enumerate(fields)}
        out.update(qmin=self.lo, qmax=self.hi, initial_qpos=self.initial_qpos,
                   initial_qvel=self.initial_qvel,
                   metadata_json=np.array(json.dumps(self.metadata, default=lambda v: np.asarray(v).tolist())))
        for key in ("t", "node_times", "joint_q", "joint_v", "joint_violation", "joint_diagnostic_ms",
                    "prepare_ms", "reference_ms", "solver_ms", "control_ms", "solver_status"):
            out["pred_" + key] = np.asarray([h[key] for h in horizon])
        out.update({"failure_" + k: np.asarray(v) for k, v in self.failure.items()})
        return out
