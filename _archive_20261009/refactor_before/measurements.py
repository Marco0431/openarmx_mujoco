"""控制循环使用的状态采样、接触测量和诊断记录。

测量函数不推进仿真；关节边界记录与 Euler 物理步的反力保持同一时刻。
仅开启 dynamics_diag 时才额外在独立 MjData 上计算加速度对账。
"""
from __future__ import annotations
import json
import time
import mujoco
import numpy as np
from ik import get_ee_tip, name_to_id

def sample_plant_parameters(model, data, arms, box_geom_id, *, diagnostic=None,
                            contact_snapshot=None):
    """采样当前惯性、重力、切向载荷和阻尼，返回两臂共 140 维参数。
    先临时清零速度读取重力，再恢复真实速度并刷新；惯性和接触载荷随后读取。
    只使用原有两次 mj_forward，finally 保证速度恢复，不改变位置或仿真时间。
    diagnostic 可保留刷新前后惯性用于对账；contact_snapshot 提供同次完整接触测量。"""
    diagnostic_ms = 0.0
    if diagnostic is not None:
        started = time.perf_counter()
        old_mass = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, data, old_mass)
        diagnostic["mass_old"] = np.array([old_mass[np.ix_(a["dofs"], a["dofs"])] for a in arms])
        diagnostic_ms = (time.perf_counter() - started) * 1000.0
    saved_velocity = data.qvel.copy()
    try:
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        gravity = data.qfrc_bias.copy()
    finally:
        data.qvel[:] = saved_velocity
        mujoco.mj_forward(model, data)
    current_mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, current_mass)
    mass_gravity, loads, blocks, contacts = [], [], [], []
    for arm in arms:
        block = current_mass[np.ix_(arm["dofs"], arm["dofs"])]
        mass_gravity.extend([block.reshape(-1), gravity[arm["dofs"]]])
        if contact_snapshot is None:
            loads.append(contact_wrench_on_arm(model, data, arm, box_geom_id)[0])
        else:
            contact = {}
            loads.append(contact_wrench_on_arm(model, data, arm, box_geom_id,
                                              snapshot=contact)[0])
            contacts.append(contact)
        if diagnostic is not None:
            blocks.append(block)
    damping = np.concatenate([model.dof_damping[a["dofs"]] for a in arms])
    if contact_snapshot is not None:
        contact_snapshot.update(full_contact=np.concatenate([c["full_contact"] for c in contacts]),
                                tangent_contact=np.concatenate(loads),
                                world_wrench=np.concatenate([c["world_wrench"] for c in contacts]),
                                normal_magnitude=np.array([c["normal_magnitude"] for c in contacts]))
    if diagnostic is not None:
        started = time.perf_counter()
        diagnostic["mass_current"] = np.array(blocks)
        diagnostic["sampling_diagnostic_ms"] = diagnostic_ms + (time.perf_counter() - started) * 1000.0
    return np.concatenate(mass_gravity + loads + [damping])


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


class DynamicsRecorder:
    """在相同状态、输入和其他参数下，只替换惯性矩阵比较预测加速度。
    MuJoCo 对照值由独立 MjData 的即时前向计算得到，不额外刷新主仿真对象。
    两种预测都沿用冻结接触载荷；剩余差异不等同于惯性采样问题。"""
    def __init__(self, model, arms, metadata):
        self.model, self.arms = model, arms
        self.shadow = mujoco.MjData(model)
        self.dofs = np.concatenate([a["dofs"] for a in arms])
        self.act_ids = np.concatenate([a["act_ids"] for a in arms])
        self.metadata = dict(metadata, diagnostic="same-state instantaneous acceleration; only M changes")
        self.rows = []

    def record(self, data, controller, mass_snapshot, reference, t):
        started = time.perf_counter()
        x = np.concatenate([part for a in self.arms
                            for part in (data.qpos[a["qpos"]], data.qvel[a["dofs"]])])
        u = data.ctrl[self.act_ids].copy()
        current_p = controller.last_parameters[0].copy()
        old_p = current_p.copy()
        old_p[:49] = mass_snapshot["mass_old"][0].reshape(-1)
        old_p[56:105] = mass_snapshot["mass_old"][1].reshape(-1)
        acceleration_idx = np.r_[7:14, 21:28]
        predicted_old = np.asarray(controller._dynamics_fn(x, u, old_p)).ravel()[acceleration_idx]
        predicted_current = np.asarray(controller._dynamics_fn(x, u, current_p)).ravel()[acceleration_idx]
        mujoco.mj_copyData(self.shadow, self.model, data)
        mujoco.mj_forward(self.model, self.shadow)
        actual = self.shadow.qacc[self.dofs].copy()
        self.rows.append(dict(
            t=float(t), phase=dynamics_phase(reference, t), x=x, u=u,
            p_old=old_p, p_current=current_p,
            mass_old=mass_snapshot["mass_old"].copy(), mass_current=mass_snapshot["mass_current"].copy(),
            acceleration_old=predicted_old, acceleration_current=predicted_current,
            acceleration_mujoco=actual, error_old=predicted_old-actual,
            error_current=predicted_current-actual,
            sampling_diagnostic_ms=mass_snapshot["sampling_diagnostic_ms"],
            diagnostic_ms=(time.perf_counter() - started) * 1000.0))

    def export(self):
        out = {k: np.asarray([row[k] for row in self.rows]) for k in self.rows[0]} if self.rows else {}
        out["metadata_json"] = np.array(json.dumps(self.metadata, default=lambda v: np.asarray(v).tolist()))
        return out


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


def _rotvec_of_R(R):
    """世界旋转矩阵 (3,3) → 旋转向量（主值；与 nmpcc_cart.rotvec_of_R 同口径）。"""
    R = np.asarray(R, float)
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-6:
        return 0.5 * vee
    return 0.5 * theta / np.sin(theta) * vee


def contact_wrench_on_arm(model, data, arm, box_geom_id, *, snapshot=None):
    """将垫片-箱接触的世界系切向力回射到关节：fJ=J^T f。

    ``mj_contactForce`` 给出作用在 geom2 上的力；按垫片所在侧调整符号。
    接触点可能与 EE tip
    相差垫片半厚，因此必须在实际接触点计算 Jacobian。x 向法向力仍由
    NMPCC 的虚拟弹簧模型承担，这里只回射实测 y/z 切向载荷。
    snapshot 可同时读取全量接触关节载荷，复用相同接触力和 Jacobian。
    """
    f_world = np.zeros(3)
    fJ = np.zeros(len(arm["dofs"]))
    full_contact = np.zeros_like(fJ)
    world_wrench = np.zeros(6)
    normal_magnitude = 0.0
    force6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if not ((c.geom1 == arm["pad_id"] and c.geom2 == box_geom_id) or
                (c.geom2 == arm["pad_id"] and c.geom1 == box_geom_id)):
            continue
        mujoco.mj_contactForce(model, data, i, force6)
        frame = c.frame.reshape(3, 3)
        f = frame.T @ force6[:3]
        tau = frame.T @ force6[3:]
        sign = 1.0 if c.geom2 == arm["pad_id"] else -1.0
        f *= sign
        tau *= sign
        full_force = f.copy() if snapshot is not None else None
        f[0] = 0.0
        f_world += f
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, c.pos, arm["body_id"])
        fJ += jacp[:, arm["dofs"]].T @ f + jacr[:, arm["dofs"]].T @ tau
        if snapshot is not None:
            full_contact += jacp[:, arm["dofs"]].T @ full_force + jacr[:, arm["dofs"]].T @ tau
            # 与垫片测量接口一致：汇总世界系力和接触本身的力矩。
            # 这里不把多个接触点的力矩平移到同一参考点；广义载荷由上方雅可比计算。
            world_wrench += np.r_[full_force, tau]
            normal_magnitude += abs(float(force6[0]))
    if snapshot is not None:
        snapshot.update(full_contact=full_contact, normal_magnitude=normal_magnitude,
                        world_wrench=world_wrench)
    return fJ, f_world


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


def pad_box_wrench_3d(model, data, arms, box_geom_id):
    """汇总作用在每侧垫片上的世界坐标力和接触自身力矩。

    返回 {side: (f_world(3), tau_world(3))}。力矩项不包含平移到共同原点的力臂贡献；
    关节广义载荷需使用 contact_wrench_on_arm 中接触点处的雅可比映射。
    """
    out = {a["side"]: (np.zeros(3), np.zeros(3)) for a in arms}
    force6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        for a in arms:
            if ((c.geom1 == a["pad_id"] and c.geom2 == box_geom_id) or
                    (c.geom2 == a["pad_id"] and c.geom1 == box_geom_id)):
                mujoco.mj_contactForce(model, data, i, force6)
                frame = c.frame.reshape(3, 3)
                f, tau = out[a["side"]]
                sign = 1.0 if c.geom2 == a["pad_id"] else -1.0
                f += sign * (frame.T @ force6[:3])
                tau += sign * (frame.T @ force6[3:])
                out[a["side"]] = (f, tau)
    return out


def box_net_wrench(model, data, arms, box_geom_id, box_com):
    """箱体质心处的净接触力偶（来自两垫片接触；箱体自由体受力 = 接触 + 重力）。"""
    f_net = np.zeros(3)
    tau_net = np.zeros(3)
    force6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        is_pair = False
        for a in arms:
            if ((c.geom1 == a["pad_id"] and c.geom2 == box_geom_id) or
                    (c.geom2 == a["pad_id"] and c.geom1 == box_geom_id)):
                is_pair = True
                break
        if not is_pair:
            continue
        mujoco.mj_contactForce(model, data, i, force6)
        frame = c.frame.reshape(3, 3)
        sign = 1.0 if c.geom2 == box_geom_id else -1.0
        f_on_box = sign * (frame.T @ force6[:3])
        t_on_box = sign * (frame.T @ force6[3:])
        r = c.pos - box_com
        tau_net += t_on_box + np.cross(r, f_on_box)
        f_net += f_on_box
    return f_net, tau_net


def box_attitude(data, box_qpos):
    """freejoint 四元数 -> (精确 rotvec(3), quat_wxyz)。"""
    q = data.qpos[box_qpos + 3:box_qpos + 7].copy()
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return _rotvec_of_R(R.reshape(3, 3)), q



def __getattr__(name):
    """旧代码仍可按原名称访问历史离线对账函数。"""
    if name in ['SCENE_CHECK_CONST', '_load_grasp_pose', '_load_slender', 'main', 'press_delta_mm', 'reconciliation']:
        from . import legacy_reconciliation
        return getattr(legacy_reconciliation, name)
    raise AttributeError(name)
