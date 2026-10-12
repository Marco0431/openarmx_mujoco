#!/usr/bin/env python3
"""手持双臂搬运实验入口。

负责建立场景和初始抓取、生成逐节点参考、执行 NMPCC 控制以及保存结果。
控制器在对应 nmpcc_cart 文件中独立定义；测量和观测在 diagnostics/，
统计、绘图调用和扰动管理在 exp2_experiments/。时间单位为秒，位置为米，力为牛顿。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

from pathlib import Path

import nmpcc_cart_grasp_pose as nc
from exp2_experiments import handheld_tools as experiment_tools
from diagnostics.handheld_observation import Observation
from exp2_experiments.sweeps import has_sweep, run_sweep
from exp2_experiments import disturbance as disturbance_tools
from diagnostics.measurements import sample_plant_parameters, arm_joint_bounds, contact_forces
from ik import ARM_CONFIGS, BASE_DIR, clamp_qpos, get_ee_tip, name_to_id, solve_pose_ik

# 直接使用 grasp_pose_scene.xml，不修改原始场景文件。
BOX_HALF = np.array([0.12, 0.20, 0.10], dtype=np.float64)
BOX_CENTER = np.array([0.5, 0.0, 0.72], dtype=np.float64)
GRIP_CLEARANCE = 0.004
IK_ITERS = 1500
SCENE_XML = BASE_DIR / "grasp_pose_scene.xml"
K_STIFF = 2419.3
# 双侧串联等效刚度（N/m）；标定工具位于 diagnostics/calibration/。
REST_GAP = 2 * (BOX_HALF[0] + GRIP_CLEARANCE)
DT = 0.002  # MuJoCo 物理步长；每步均检查关节边界。
CTRL_EVERY = 3  # 每三步更新控制，对应 6 ms；其余步保持力矩。
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "exp2_latest_grasp_pose")
HOLD_BOX_Y_TOL = 0.002


def build_scene() -> Path:
    """从固定抓取场景建立运行模型，补齐缺失的关节力矩电机。
    电机限幅继承模型的物理范围，原始场景文件不作覆盖。"""
    if not SCENE_XML.is_file():
        raise FileNotFoundError(SCENE_XML)
    original = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    root = ET.Element('mujoco', model='grasp_pose_nmpcc_runtime')
    ET.SubElement(root, 'include', file=str(SCENE_XML.resolve()))
    actuator = ET.SubElement(root, 'actuator')
    missing = 0
    for physical_side in ('left', 'right'):
        for i in range(1, 8):
            name = f'openarmx_{physical_side}_joint{i}'
            jid = name_to_id(original, mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_ACTUATOR, name + '_motor')
            if aid >= 0:
                continue
            if not original.jnt_actfrclimited[jid]:
                raise ValueError(f'{name} lacks an explicit physical torque range')
            bounds = ' '.join(format(float(v), '.17g') for v in original.jnt_actfrcrange[jid])
            ET.SubElement(actuator, 'motor', name=name + '_motor', joint=name,
                          gear='1', ctrllimited='true', forcelimited='true',
                          ctrlrange=bounds, forcerange=bounds)
            missing += 1
    if not missing:
        return SCENE_XML
    runtime = BASE_DIR / 'grasp_pose_handheld_nmpcc_runtime.xml'
    content = ET.tostring(root, encoding='unicode')
    if not runtime.exists() or runtime.read_text(encoding='utf-8') != content:
        runtime.write_text(content, encoding='utf-8')
    return runtime


def make_arms(model):
    """收集双臂关节、执行器和垫片索引，并转换到控制器的左右排序。
    手持场景的物理左右臂与控制器左右约定相反，后续数据统一使用此映射。"""
    arms = []
    # 物理左臂在 +x，物理右臂在 -x；转换成控制器 left/right 约定。
    side_of = {"left": "right", "right": "left"}
    for cfg in ARM_CONFIGS:
        side = side_of[cfg["side"]]
        joint_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg["joint_names"]]
        arms.append({
            "side": side, "sign": -1.0 if side == "left" else 1.0,
            "q_rest": cfg["q_rest"],
            "body_id": name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["jacobi_body"]),
            "joint_ids": joint_ids,
            "dofs": [int(model.jnt_dofadr[j]) for j in joint_ids],
            "qpos": [int(model.jnt_qposadr[j]) for j in joint_ids],
            "pad_id": name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                 f"openarmx_{cfg['side']}_ee_pad"),
        })
    return sorted(arms, key=lambda a: 0 if a["side"] == "left" else 1)


def grasp_targets(arms, center=None):
    """根据箱体中心和手臂所在侧生成对夹目标点。"""
    center = BOX_CENTER if center is None else np.asarray(center)
    return {a["side"]: np.array(
        [center[0] + a["sign"] * (BOX_HALF[0] + GRIP_CLEARANCE), center[1], center[2]])
        for a in arms}


def solve_end_grip(model, data, arms, targets, iters=IK_ITERS):
    """用位置与朝向 IK 建立初始夹持姿态；仅用于初始化，不运行动态控制。"""
    for a in arms:
        for _ in range(iters):
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            mujoco.mj_forward(model, data)
            dq, _ = solve_pose_ik(
                model, data, a["body_id"], targets[a["side"]],
                np.array([-a["sign"], 0.0, 0.0]), get_ee_tip(data, a["body_id"]),
                a["dofs"], a["qpos"], a["q_rest"])
            data.qpos[a["qpos"]] += dq
            clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)


def calibrate_initial_preload(model, data, arms, box_geom_id, target_force,
                              target_fn, tolerance=0.25, max_iter=12):
    """按实测接触力迭代修正初始压入量，返回初始夹持目标和对应实测力。
    只校准本次启动的几何预压，不改接触刚度或控制器增益。"""
    lo = 0.0
    hi = max(3.0 * float(target_force), 5.0)
    best = None
    for _ in range(max_iter):
        command_force = 0.5 * (lo + hi)
        solve_end_grip(model, data, arms, target_fn(command_force),
                       max(250, IK_ITERS // 3))
        measured = contact_forces(model, data, arms, box_geom_id)
        mean_force = 0.5 * (measured["left"] + measured["right"])
        error = mean_force - target_force
        if best is None or abs(error) < abs(best[0]):
            best = (error, command_force, measured)
        if abs(error) <= tolerance:
            break
        if error < 0.0:
            lo = command_force
        else:
            hi = command_force
    if best is None or abs(best[0]) > max(tolerance, 0.05 * target_force):
        raise RuntimeError(
            f"t=0 preload calibration failed: target={target_force:.3f} N, best={best}")
    command_force = best[1]
    solve_end_grip(model, data, arms, target_fn(command_force), IK_ITERS)
    measured = contact_forces(model, data, arms, box_geom_id)
    print(f"[t0 preload] target={target_force:.3f}N geometry_command={command_force:.3f}N "
          f"static=({measured['left']:.3f},{measured['right']:.3f})N")
    return target_fn(command_force), measured


def disable_robot_collisions(model, arms, box_geom_id):
    """设置机器人与箱体的碰撞筛选，保留本实验需要的垫片接触。"""
    pads = {a["pad_id"] for a in arms}
    for gid in range(model.ngeom):
        if gid == box_geom_id or gid in pads:
            continue
        if (model.geom(gid).name or "").startswith("openarmx_"):
            model.geom_contype[gid] = 0
            model.geom_conaffinity[gid] = 0


def soften_contacts(model, box_geom_id, pad_ids):
    """统一箱体和垫片的接触求解参数，保持当前标定所用的软接触设置。"""
    for gid in [box_geom_id, *pad_ids]:
        model.geom_solref[gid] = [0.02, 1.0]
        model.geom_solimp[gid][:3] = [0.999, 0.999, 0.001]



def arm_damping(model, dofs, value=0.5):
    """设置指定关节的粘性阻尼；预测参数随后从模型读取同一组系数。"""
    for d in dofs:
        model.dof_damping[d] = value


def smoothstep(s: float) -> float:
    """将归一化进度限制在 0～1，并生成两端斜率为零的平滑过渡。"""
    s = min(1.0, max(0.0, float(s)))
    return 3.0 * s * s - 2.0 * s * s * s


def inject_motors(runtime_xml: Path) -> None:
    """为指定运行模型补齐力矩电机；已有电机保留不重复创建。"""
    tree = ET.parse(runtime_xml)
    root = tree.getroot()
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    existing = {motor.get("name") for motor in actuator.findall("motor")}
    added = 0
    for side in ("left", "right"):
        for i in range(1, 8):
            joint_name = f"openarmx_{side}_joint{i}"
            motor_name = f"{joint_name}_motor"
            if motor_name in existing:
                continue
            joint = root.find(f".//joint[@name='{joint_name}']")
            if joint is None:
                raise ValueError(f"runtime XML 中找不到关节 {joint_name}")
            force_range = joint.get("actuatorfrcrange", "-120 120")
            ET.SubElement(actuator, "motor", {
                "name": motor_name,
                "joint": joint_name,
                "forcerange": force_range,
                "ctrlrange": force_range,
            })
            added += 1
    if added:
        tree.write(runtime_xml, encoding="utf-8", xml_declaration=True)
        print(f"[scene] injected {added} torque motors into {runtime_xml.name}")


# 阶段参考：夹紧、保持、平滑搬运与停止
def motion_timing(distance, vy):
    """计算基准时长、单段加减速时长和实际搬运时长。
    速度按幅值使用，轨迹始终沿世界负 y；零距离或零速度返回静止时序。"""
    distance, speed = abs(float(distance)), abs(float(vy))
    if distance <= 1e-9 or speed <= 1e-9:
        return 0.0, 0.0, 0.0
    base = distance / speed
    ramp = min(0.20, 0.5 * base)
    return base, ramp, base + ramp

@dataclass
class ReferenceSample:
    """一个时刻的双臂参考：六维位姿/速度、法向力幅值及其变化率。
    phase 用于日志分段，ramp_s 是夹紧过程的归一化平滑进度。"""
    refs: dict
    vrefs: dict
    fmag: float
    fmag_dot: float
    phase: str
    ramp_s: float

class PhaseReference:
    """按仿真时间生成夹紧、保持、搬运和停止参考。
    夹紧位置与夹持力使用同一平滑进度；搬运含加速、匀速和对称减速。
    预测节点用各自未来时间采样，因此位置、速度和力导数在时间上保持一致。"""

    def __init__(self, force, vy, distance, t_ramp, t_hold, press0, press,
                 rv_ref, f_start):
        self.force, self.f_start = float(force), float(f_start)
        self.speed, self.distance = abs(float(vy)), abs(float(distance))
        self.t_ramp = max(0.0, float(t_ramp))
        self.t_hold_end = self.t_ramp + max(0.0, float(t_hold))
        self.press0, self.press, self.rv_ref = press0, press, rv_ref
        self.base_time, self.accel_time, self.move_duration = motion_timing(distance, vy)
        self.move_end = self.t_hold_end + self.move_duration

    def sample(self, t):
        """返回指定时刻的参考值；移动结束后保持终点位置、速度归零。"""
        z = float(t) / self.t_ramp if self.t_ramp > 1e-9 else 1.0
        s = smoothstep(z)
        sd = 6.0 * z * (1.0 - z) / self.t_ramp if 0.0 < z < 1.0 else 0.0
        fmag = self.f_start + (self.force - self.f_start) * s
        fmag_dot = (self.force - self.f_start) * sd
        phase = "ramp" if t < self.t_ramp else "hold"
        offset_y = velocity_y = 0.0
        if self.move_duration > 0.0 and t >= self.t_hold_end:
            # 停止尾段仍标记为 move，保持日志阶段定义一致。
            phase = "move"
            tau, ramp, speed = t - self.t_hold_end, self.accel_time, self.speed
            if tau < ramp:
                z = tau / ramp
                offset_y = -speed * ramp * (z**3 - 0.5*z**4)
                velocity_y = -speed * smoothstep(z)
            elif tau < self.base_time:
                offset_y = -speed * (tau - 0.5*ramp)
                velocity_y = -speed
            elif t < self.move_end:
                z = (tau - self.base_time) / ramp
                offset_y = -speed * (self.base_time - 0.5*ramp
                                     + ramp * (z - z**3 + 0.5*z**4))
                velocity_y = -speed * (1.0 - smoothstep(z))
            else:
                offset_y = -self.distance
        refs, vrefs = {}, {}
        for side in ("left", "right"):
            delta = self.press[side] - self.press0[side]
            position = self.press0[side] + delta * s + np.array([0.0, offset_y, 0.0])
            velocity = delta * sd + np.array([0.0, velocity_y, 0.0])
            refs[side] = np.r_[position, self.rv_ref[side]]
            vrefs[side] = np.r_[velocity, np.zeros(3)]
        return ReferenceSample(refs, vrefs, fmag, fmag_dot, phase, s)

    def __call__(self, t):
        """兼容返回五个值的参考采样接口。"""
        r = self.sample(t)
        return r.refs, r.vrefs, r.fmag, r.phase, r.ramp_s

def phase_profile(force, vy, distance, t_ramp, t_hold, press0, press, rv_ref,
                  f_start=2.0):
    """建立可反复采样的参考生成器，同时返回保持阶段结束时间。"""
    profile = PhaseReference(force, vy, distance, t_ramp, t_hold, press0, press,
                             rv_ref, f_start)
    return profile, profile.t_hold_end


# 场景复位
def reset_world(model, data, arms, box_qpos) -> None:
    """恢复初始关节和箱体姿态，清零速度并刷新模型。
    这是每次实验开始时的复位，不是控制循环内的状态修正。"""
    global BOX_CENTER
    data.time = 0.0
    data.ctrl[:] = 0.0
    data.qacc_warmstart[:] = 0.0
    for a in arms:
        data.qpos[a["qpos"]] = a["q_rest"]
        clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    # 根据休息姿态设置箱体中心，保证初始抓取目标沿用已验证的位置。
    tips_mid = np.mean([get_ee_tip(data, a["body_id"]) for a in arms], axis=0)
    BOX_CENTER = tips_mid + np.array([0.0, 0.0, 0.10])
    data.qpos[box_qpos:box_qpos + 3] = BOX_CENTER
    data.qpos[box_qpos + 3:box_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


# 单次仿真：采样、求解、施加命令与物理积分
def run_case(model, data, arms, ids, args, nmpc, ref_fn, t_hold_end,
             viewer=None):
    """执行一次实验，并管理可选的外力脉冲。
    外力只进入 MuJoCo；无论正常结束还是异常退出，都撤销本次施力并保存日志。"""
    end_time = (t_hold_end + 0.3 if args.hold_only else
                max(args.steps * DT, t_hold_end + ref_fn.move_duration + 0.4))
    pulse = disturbance_tools.ForcePulse(model, data, args, end_time, OUT_DIR)
    nmpc.last_disturbance_log = str(pulse.path.resolve()) if pulse.enabled else None
    try:
        return _run_case(model, data, arms, ids, args, nmpc, ref_fn, t_hold_end,
                         viewer=viewer, disturbance=pulse)
    finally:
        pulse.release()
        pulse.save()


def _run_case(model, data, arms, ids, args, nmpc, ref_fn, t_hold_end,
             disturbance,
             viewer=None) -> (list, dict):
    """执行预热与闭环控制，返回观测序列和评价指标。
    每个物理步推进 2 ms，每三个物理步更新一次控制，其间保持上一拍力矩。
    测量、预测轨迹及异常现场由 diagnostics 中的观测器集中记录。"""
    box_body, box_geom, box_qpos = ids["box_body"], ids["box_geom"], ids["box_qpos"]
    mass_snapshot = {} if getattr(args, "dynamics_diag", False) else None
    # 默认冻结实测完整接触载荷与当前模型法向载荷的差值。
    contact_load_mode = getattr(args, "contact_load_mode", "residual")
    contact_snapshot = {}  # 同次采样提供左右各六维力与力矩

    def joint_state():
        """按控制器顺序拼接左臂角度/速度、右臂角度/速度，共 28 维。"""
        return np.concatenate([
            data.qpos[arms[0]["qpos"]], data.qvel[arms[0]["dofs"]],
            data.qpos[arms[1]["qpos"]], data.qvel[arms[1]["dofs"]],
        ])

    def plant_params():
        """在同一当前状态采样惯性、重力、阻尼和接触载荷，不推进仿真时间。"""
        options = dict(diagnostic=mass_snapshot)
        if contact_snapshot is not None:
            options["contact_snapshot"] = contact_snapshot
        return sample_plant_parameters(model, data, arms, box_geom, **options)

    def measured_normal_force():
        """返回本次采样的左右六维世界坐标力与力矩。"""
        if contact_snapshot is not None:
            return contact_snapshot["world_wrench"].copy()
        forces = contact_forces(model, data, arms, box_geom)
        return np.array([-forces["left"], forces["right"]], dtype=float)

    reference_ms = 0.0
    def build_p(t):
        """构造 N+1 行节点参数：动力学量每拍测一次，参考按节点时间分别计算。
        载荷残差在整段预测中冻结，避免当前测量法向力与模型预测法向力重复计入。"""
        nonlocal reference_ms
        pp = plant_params()
        started = time.perf_counter()
        p = np.zeros((nmpc.cfg.N + 1, nc.NP))
        p[:, :126] = pp[:126]
        p[:, 162:176] = pp[126:140]
        p[:, nc.KF_PARAM_IDX] = float(nmpc.cfg.kf_scale)
        # 第 0 行对应当前时刻，最后一行对应预测终端，不能漏掉终端参考。
        for i, tau in enumerate(nmpc.node_times):
            r = ref_fn.sample(t + tau)
            p[i, 126:132], p[i, 132:138] = r.refs["left"], r.refs["right"]
            p[i, 138:144], p[i, 144:150] = r.vrefs["left"], r.vrefs["right"]
            p[i, 150], p[i, 156] = -r.fmag, r.fmag
            p[i, nc.FREF_DOT_IDX] = -r.fmag_dot
            p[i, nc.FREF_DOT_IDX + nc.NT] = r.fmag_dot
            if i == 0:
                fmag, phase = r.fmag, r.phase
        reference_ms = (time.perf_counter() - started) * 1000.0
        if contact_load_mode == "residual":
            p = nmpc.contact_residual_parameters(
                joint_state(), p, contact_snapshot["full_contact"], measured_normal_force())
        return p, fmag, phase

    trace = Observation(model, data, arms, ids, args, nmpc, ref_fn, DT, CTRL_EVERY,
                        smoothstep, mass_snapshot, contact_snapshot, contact_load_mode)

    # IK 建立几何预压，但不代表箱体已静态平衡。种子仅平衡当前预测模型，
    # 预热不推进仿真，实际力矩仍由随后首拍求解返回。
    nmpc._warm = None
    x0, measured = joint_state(), np.full(2, np.nan)
    p0 = np.full((nmpc.cfg.N + 1, nc.NP), np.nan)
    u_seed = np.full(nc.NU, np.nan)
    try:
        p0, _, _ = build_p(0.0)
        x0, measured = joint_state(), measured_normal_force()
        u_seed = nmpc.equilibrium_seed(x0, p0, f_meas=measured)
        lower, upper = nmpc.umin.copy(), nmpc.umax.copy()
        if args.u_lim is not None:
            limit = np.broadcast_to(np.asarray(args.u_lim, dtype=float), (nc.NU,))
            lower, upper = np.maximum(lower, -limit), np.minimum(upper, limit)
        if (not np.isfinite(u_seed).all() or np.isnan(lower).any() or np.isnan(upper).any()
                or np.any(lower > upper) or np.any(u_seed < lower) or np.any(u_seed > upper)):
            raise ValueError("model-equilibrium seed is outside effective torque bounds; startup stopped")
        trace.recorder.metadata["startup_seed"] = u_seed.copy()
        trace.recorder.metadata["startup_seed_kind"] = "current prediction model equilibrium, not plant equilibrium"
    except Exception as exc:
        trace.capture_failure(exc, 0.0, x0, p0, measured, u_seed)
        raise
    # 两次预热只迭代求解器，不调用 mj_step，也不把种子直接施加给机器人。
    for _ in range(2):
        x0, measured = joint_state(), measured_normal_force()
        try:
            nmpc.solve(x0, p0, u_ref=u_seed, f_meas=measured, u_seed=u_seed,
                       u_lim=args.u_lim)
            if nmpc.last_status != 0:
                raise RuntimeError(f"NMPCC warmup status {nmpc.last_status}")
            trace.capture_recovered_failure(0.0)
            trace.record_warmup()
        except Exception as exc:
            trace.capture_failure(exc, 0.0, x0, p0, measured, u_seed)
            raise

    initial_box_y = float(data.xpos[box_body][1])
    print(f"[timed] ramp={args.t_ramp:.3f}s, holding={args.t_hold:.3f}s, "
          f"moving starts at t={t_hold_end:.3f}s"
          if abs(args.y_dist) > 1e-9 and abs(args.v_y) > 1e-9 and not args.hold_only
          else f"[timed] ramp={args.t_ramp:.3f}s, then holding; translation disabled")

    u_last = u_seed.copy()
    torque_lo = np.concatenate([model.actuator_ctrlrange[a["act_ids"], 0]
                                for a in arms])
    torque_hi = np.concatenate([model.actuator_ctrlrange[a["act_ids"], 1]
                                for a in arms])
    phase = "ramp"
    u_cmd = np.asarray(u_seed, dtype=float).copy()
    u_clip_mask = np.zeros(nc.NU, dtype=bool)
    u_sat_mask = np.isclose(u_cmd, torque_lo, atol=1e-6) | np.isclose(
        u_cmd, torque_hi, atol=1e-6)
    move_duration = ref_fn.move_duration
    required_end = t_hold_end + (0.3 if args.hold_only else move_duration + 0.4)
    steps = max(args.steps, int(np.ceil(required_end / DT)))
    stop_file = getattr(args, 'disturb_stop_file', None)
    for step in range(steps):
        t = step * DT
        # 先施加本步外力，再采样当前动力学；预测器不接收未来扰动时间表。
        disturbance.begin_step(t)
        # 非控制步保留上一拍的阶段；耗时仅在控制更新时记录。
        solve_ms = np.nan
        settling = step < int(round(args.settle_time / DT))
        if settling:
            # 可选稳定阶段实验；默认关闭。
            u_cmd = experiment_tools.settling_command(data, arms, torque_lo, torque_hi)
        elif step % CTRL_EVERY == 0:
            t0 = time.perf_counter()
            x_now = joint_state()
            p, f_meas = np.full((nmpc.cfg.N + 1, nc.NP), np.nan), np.full(2, np.nan)
            try:
                p, fmag, phase = build_p(t)
                prepare_ms = (time.perf_counter() - t0) * 1000.0
                f_meas = measured_normal_force()
                u_ref_now = np.r_[p[0, 49:56], p[0, 105:112]]
                u_last = nmpc.solve(x_now, p, u_ref=u_ref_now, f_meas=f_meas,
                                    u_seed=u_seed, u_lim=args.u_lim)
                if nmpc.last_status != 0:
                    raise RuntimeError(f"NMPCC returned status {nmpc.last_status}")
                if not (np.isfinite(u_last).all() and np.isfinite(nmpc.last_pred_x).all()):
                    raise RuntimeError("non-finite NMPCC solution")
                trace.capture_recovered_failure(t)
            except Exception as exc:
                trace.capture_failure(exc, t, x_now, p, f_meas, u_seed)
                raise
            u_seed = u_last.copy()          # 下一拍使用本次命令作为控制初值
            trace.record_prediction(t, phase, p, fmag, u_ref_now, prepare_ms, reference_ms)
            u_cmd = np.clip(u_last, torque_lo, torque_hi)
            u_clip_mask = np.abs(u_cmd - np.asarray(u_last, dtype=float)) > 1e-9
            # 同时记录命令裁剪和恰好落在执行器边界上的解。
            u_sat_mask = u_clip_mask | np.isclose(u_cmd, torque_lo, atol=1e-6) | np.isclose(
                u_cmd, torque_hi, atol=1e-6)
            solve_ms = (time.perf_counter() - t0) * 1000.0
            trace.horizon_log[-1]["control_ms"] = solve_ms
        for a in arms:
            arm_i = arms.index(a)
            data.ctrl[a["act_ids"]] = u_cmd[7 * arm_i:7 * arm_i + 7]
        if not settling:
            trace.record_dynamics(step, t)
        # 先保存步前状态，再与本次积分计算的约束反力配对，避免时间错位。
        trace.before_step()
        mujoco.mj_step(model, data)
        disturbance.end_step()
        trace.after_step(u_cmd)
        if viewer is not None:
            viewer.sync()
            time.sleep(DT)
        trace.record_sample(step, t, phase, solve_ms,
                            prepare_ms if step % CTRL_EVERY == 0 and not settling else np.nan,
                            reference_ms, u_cmd, u_clip_mask, u_sat_mask, settling)
        if stop_file and Path(stop_file).exists():
            print('[disturbance] 已收到停止请求，撤力并保存当前日志')
            break
        if args.hold_only and t >= t_hold_end + 0.3:
            break
    if viewer is not None:
        viewer.close()
    return trace.finish(initial_box_y, move_duration, summarize)

def summarize(log, args, initial_box_y, move_duration=None):
    """使用本场景的参考和评价口径统计结果。"""
    return experiment_tools.summarize(log, args, initial_box_y, move_duration,
        box_center=BOX_CENTER, hold_box_y_tol=HOLD_BOX_Y_TOL, motion_timing=motion_timing, physics_dt=DT)

# 主流程
def save_run(output, args, model, data, arms, ids, controller, reference, log, horizon, metrics, error):
    """归档参数、源码、预测轨迹和失败信息，便于复查一次实验。"""
    return experiment_tools.save_run(output, args, model, data, arms, ids,
        controller, reference, log, horizon, metrics, error, entry_file=__file__,
        base_dir=BASE_DIR, box_center=BOX_CENTER, kx=K_STIFF,
        rest_gap=REST_GAP, scene_path=build_scene())
def main():
    """解析参数、建立抓取姿态和控制器，再运行实验并保存结果。
    普通调用只运行一次；参数扫描由独立实验辅助模块处理。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--force", type=float, default=10, help="稳定夹持力参考 (N)")
    parser.add_argument("--kf_scale", type=float, default=0.003,
                        help="力误差直接乘此增益，与平放版一致；不再除以接触刚度")
    parser.add_argument("--Q", type=float, default=1.0, help="阶段代价的独立整体权重 Q")
    parser.add_argument("--P", type=float, default=2.0, help="独立终端权重 P")
    parser.add_argument("--eta_dev", type=float, default=0.05, help="误差导数相对权重 eta_dev")
    parser.add_argument("--eta_u", type=float, default=5e-8, help="输入相对权重 eta_u")
    parser.add_argument("--u_lim", type=float, default=None,
                        help="求解时 u 全 horizon 硬限幅 (N·m)；为力矩优化提供有限可行域、"
                             "防止力矩偏移过大（标准 MPC 输入约束 u∈U）")
    parser.add_argument("--kx_model", type=float, default=K_STIFF,
                        help="预测弹簧刚度 (N/m)，默认保留手持标定2419.3；力残差不再除以kx。"
                             "本轮不调整该参数或物理接触标定。")
    parser.add_argument("--eta_cop_y", type=float, choices=[0.8], default=0.8,
                        help="协作映射 eta_cop 的 y 分量；0=绝对 y 跟踪，0.8=同步优先")
    parser.add_argument("--If1_x", type=float, choices=[0.8], default=0.8,
                        help="eta_f 的 x 分量权重 If1（p 参数，运行时可变，不触发重建）；"
                             "0.8=力-位置混合，1.0=纯力误差")

    parser.add_argument("--t_ramp", type=float, default=0.5, help="10N→目标力的平滑时间 (s)")
    parser.add_argument("--t_hold", type=float, default=1.0, help="固定 Holding 时间，结束后直接 Moving (s)")
    parser.add_argument("--f_start", type=float, default=10.0,
                        help="启动夹持力参考 (N)，同时用于初始 IK 压紧和控制器首拍")
    parser.add_argument("--v_y", type=float, default=-0.01,
                        help="世界坐标 -y 前进速度 (m/s)，正值表示沿 -y 前进")
    parser.add_argument("--y_dist", type=float, default=0.05,
                        help="沿世界 -y 的前进距离 (m)")
    parser.add_argument("--vz_max", type=float, default=0.05,
                        help="OCP 末端 z 速度约束上限 (m/s)")
    parser.add_argument("--settle_time", type=float, default=0.0,
                        help="t0 沉降时长 (s)：用全接触平衡扭矩静置，让软接触竖直摩擦发育、"
                             "接住箱子后再交给控制器；0=关闭")
    parser.add_argument("--hold_only", action="store_true")
    parser.add_argument("--sliding_friction", type=float, default=0.8,
                        help="箱体及两侧夹持垫的滑动摩擦系数；两侧同时设置，避免取较大值覆盖")
    parser.add_argument("--noslip_iterations", type=int, default=3,
                        help="MuJoCo 每个物理步的 NoSlip 最大迭代次数；0 为关闭")
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--Tp", type=float, default=0.10)
    parser.add_argument("--sim_steps", type=int, default=2,
                        help="acados ERK 每个预测区间的积分步数")
    parser.add_argument("--no_viewer", action="store_true", help="关闭 MuJoCo 可视化窗口")
    parser.add_argument('--contact_load_mode', choices=['residual', 'tangent'], default='residual',
                        help='默认residual，与平放版一致；tangent仅供载荷近似对照')
    parser.add_argument('--output_dir', default=None, help='new directory for complete NPZ diagnostics')
    experiment_tools.add_arguments(parser)
    disturbance_tools.add_arguments(parser)
    args = parser.parse_args()
    if not np.isfinite(args.sliding_friction) or args.sliding_friction < 0:
        parser.error("sliding_friction 必须是非负有限数")
    if args.noslip_iterations < 0:
        parser.error("noslip_iterations 必须是非负整数")
    print(f"[command] y_dist={args.y_dist:.6f} m, v_y={args.v_y:.6f} m/s, "
          f"Q={args.Q:g}, P={args.P:g}, eta_dev={args.eta_dev:g}, "
          f"eta_u={args.eta_u:g}, hold_only={args.hold_only}")
    print(f"[output] CSV and Fig. 12 files: {os.path.abspath(OUT_DIR)}")

    any_sweep = has_sweep(args)

    scene_xml = build_scene()
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    arms = make_arms(model)
    for a in arms:
        joint_ids = a["joint_ids"]
        a["act_ids"] = [name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                    f"{model.joint(j).name}_motor") for j in joint_ids]

    box_body = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_geom = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    box_joint = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")
    box_qpos = int(model.jnt_qposadr[box_joint])
    ids = {"box_body": box_body, "box_geom": box_geom, "box_qpos": box_qpos}

    disable_robot_collisions(model, arms, box_geom)
    soften_contacts(model, box_geom, [a["pad_id"] for a in arms])

    # 仅覆盖本入口的运行模型；同步设置箱体与垫片，保留扭转、滚动摩擦和手持接触刚度。
    model.geom_friction[[box_geom] + [a["pad_id"] for a in arms], 0] = args.sliding_friction
    model.opt.noslip_iterations = args.noslip_iterations
    print(f"[contact] box/pads sliding_friction={args.sliding_friction:g}, "
          f"noslip_iterations={model.opt.noslip_iterations}")

    contact_offsets = nc.pad_contact_offsets(model, arms, .1001)
    # 标定使用垫片中心；将零力间距换算到朝内接触面，保持物理标定不变。
    contact_rest_gap = REST_GAP - sum(model.geom_size[a['pad_id'], 2] for a in arms)
    fk = nc.mujoco_fk_symbols(
        model,
        [{"body_id": a["body_id"], "joint_ids": a["joint_ids"]} for a in arms],
        0.1001,
        contact_offsets=contact_offsets,
    )
    # 以当前抓取姿态固定双臂的姿态参考
    def make_rv_ref():
        return {a["side"]: nc.rotvec_of_R(data.xmat[a["body_id"]].reshape(3, 3))
                for a in arms}
    arm_damping(model, [d for a in arms for d in a["dofs"]], 0.5)

    umax_parts = []
    for a in arms:
        umax_parts.append(model.actuator_ctrlrange[a["act_ids"], 1])
    umax = np.concatenate(umax_parts)

    NMPC_CACHE = {}

    def build_nmpc(kf):
        # grasp_v6 使用直接力误差增益；编译配置和关节界进入缓存指纹。
        model_name = (f"nmpcc_cart_grasp_pose_v6"
                      f"_Q{args.Q:g}_P{args.P:g}"
                      f"_ed{args.eta_dev:g}_eu{args.eta_u:g}"
                      f"_vz{args.vz_max:g}_ss{args.sim_steps}_kx{args.kx_model:g}"
                      f"_N{args.N}_Tp{args.Tp:g}_If{args.If1_x:g}")
        model_name = model_name.replace(".", "p").replace("-", "m")
        cfg = nc.NMPCCCartConfig(
            N=args.N, Tp=args.Tp,
            eta_cop=np.array([[0.0, args.eta_cop_y, 0.0, 0.0, 0.0, 0.0]] * 2),
            eta_f=np.array([[args.If1_x, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2),
            Q=args.Q,
            eta_dev=args.eta_dev,
            eta_u=args.eta_u,
            P=args.P,
            model_name=model_name,
            kx=args.kx_model,
            rest_gap=contact_rest_gap,
            kf_scale=kf,
            vmax=0.05,
            vz_max=args.vz_max,
            sim_method_num_steps=args.sim_steps,
            sim_method_num_stages=4,
        )
        qmin, qmax = arm_joint_bounds(model, arms)
        return nc.NMPCCCart(cfg, fk, umin=-umax, umax=umax, qmin=qmin, qmax=qmax)

    def run_one(kf, force, tag):
        args.force = force
        args.kf_scale = kf
        # 每次运行复位场景，并按启动力重建抓取姿态
        reset_world(model, data, arms, box_qpos)
        # reset_world 根据休息位重新计算了 BOX_CENTER；每个 run 必须同步重算目标。
        base_targets_now = grasp_targets(arms, BOX_CENTER)
        def press_for_now(force_n):
            delta = force_n / (2.0 * K_STIFF)
            return {
                a["side"]: base_targets_now[a["side"]] +
                np.array([-a["sign"] * delta, 0.0, 0.0])
                for a in arms
            }
        pl, _ = calibrate_initial_preload(
            model, data, arms, box_geom, args.f_start, press_for_now)
        rv_ref_now = make_rv_ref()
        pf = press_for_now(force)
        rfn, th_end = phase_profile(
            force, 0.0 if args.hold_only else args.v_y, args.y_dist,
            args.t_ramp, args.t_hold, pl, pf, rv_ref_now,
            f_start=args.f_start)
        viewer = None
        if not args.no_viewer and not any_sweep:
            viewer = mujoco.viewer.launch_passive(model, data)
            viewer.cam.lookat[:] = [0.0, -0.13, 0.11]
            viewer.cam.distance = 1.8
            viewer.cam.elevation = -12
            viewer.cam.azimuth = 180
            viewer.sync()
            print("viewer opened; close the window to stop the experiment")
        # 复用同配置求解器；每次运行清空轨迹和内部状态，保证冷启动一致。
        okey = (round(kf, 6), args.Q, args.P, args.eta_dev, args.eta_u,
                args.eta_cop_y, args.sim_steps, args.N, args.Tp, args.vz_max,
                args.kx_model)
        nmpc = NMPC_CACHE.get(okey)
        if nmpc is None:
            nmpc = build_nmpc(kf)
            NMPC_CACHE[okey] = nmpc
            print(f"NMPCC built: K={K_STIFF:.3f}, kx_model={args.kx_model:.3f}, "
                  f"pad_center_gap={REST_GAP:.6f}, contact_face_gap={nmpc.cfg.rest_gap:.6f}, "
                  f"N={args.N}, Tp={args.Tp:.3f}, force={force:.1f}N, kf_scale={kf:.3f}, "
                  f"Q={args.Q:g}, P={args.P:g}, eta_dev={args.eta_dev:g}, eta_u={args.eta_u:g}, "
                  f"eta_cop_y={args.eta_cop_y:.2f}")
        else:
            print(f"NMPCC cached (same OCP params); force={force:.1f}N, "
                  f"t_ramp={args.t_ramp:.2f}, t_hold={args.t_hold:.2f}, f_start={args.f_start:.1f}")
        experiment_tools.configure_runtime_experiment(nmpc, args)
        nmpc._warm = None
        # 清空求解器内部记忆，避免不同运行之间相互影响。
        try:
            nmpc.solver.reset()
        except Exception as e:
            print(f"[warn] nmpc.solver.reset() failed: {e}")
        # 接口支持时关闭 QP 热启动。
        try:
            nmpc.solver.options_set("qp_warm_start", 0)
            nmpc.solver.options_set("warm_start_first_qp", 0)
        except Exception as e:
            print(f"[warn] qp_warm_start disable failed: {e}")
        trace_dir = (Path(args.output_dir) if args.output_dir else
                     Path(OUT_DIR) / ('trace_' + tag + '_' + time.strftime('%Y%m%d_%H%M%S')))
        if any_sweep and args.output_dir:
            trace_dir = trace_dir / tag
        trace_dir.mkdir(parents=True, exist_ok=False)
        log, m, failure = [], {}, None
        t0 = time.perf_counter()
        try:
            log, m = run_case(model, data, arms, ids, args, nmpc, rfn, th_end, viewer=viewer)
        except Exception as exc:
            failure = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            if viewer is not None:
                viewer.close()
            save_run(trace_dir, args, model, data, arms, ids, nmpc, rfn, log,
                     getattr(nmpc, 'joint_limit_horizon', []), m, failure)
            print(f'[saved] {trace_dir}')
        dt = time.perf_counter() - t0
        return experiment_tools.finish_run(log, getattr(nmpc, "joint_limit_horizon", []), m, args, nmpc, tag, dt, OUT_DIR, __file__)

    if any_sweep:
        run_sweep(args, run_one, OUT_DIR)
        return

    m = run_one(args.kf_scale, args.force, "single")
    print("--- latest Experiment II (summary) ---")
    for k, v in m.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
