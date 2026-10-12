#!/usr/bin/env python3
"""平放双臂搬运实验入口。

负责建立场景和初始抓取、生成逐节点参考、执行 NMPCC 控制以及保存结果。
控制器在对应 nmpcc_cart 文件中独立定义；测量和观测在 diagnostics/，
统计、绘图调用和扰动管理在 exp2_experiments/。时间单位为秒，位置为米，力为牛顿。"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

import nmpcc_cart as nc
from exp2_experiments import flat_tools as experiment_tools
from diagnostics.flat_observation import Observation
from exp2_experiments.sweeps import has_sweep, run_sweep
from exp2_experiments import disturbance as disturbance_tools
import exp2_slender_scene as scene_mod
from diagnostics.measurements import contact_forces, arm_joint_bounds, sample_plant_parameters
from exp2_slender_scene import (
    BOX_CENTER, CAN_HALF_H, CAN_OFFSETS_X, IK_ITERS, SLENDER_HALF,
    build_grasped_scene, grasp_targets, solve_end_grip,
)
from ik import clamp_qpos, name_to_id

# 串联等效刚度（N/m），由 diagnostics/calibration/ 中的接触标定工具得到。
K_STIFF = 2793.166
# 零力时两侧模型接触点的间距；接触点偏置使其不同于箱体长度。
REST_GAP = 0.507537
PAD_HALF_THICKNESS = 0.004
DT = 0.002  # MuJoCo 物理步长；每步均检查关节边界。
CTRL_EVERY = 3  # 每三步更新控制，对应 6 ms；其余步保持力矩。
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "exp2_latest")

# 仅用于结果评价
HOLD_BOX_Y_TOL = 0.002


def arm_damping(model, dofs, value=0.5):
    """设置指定关节的粘性阻尼；预测参数随后从模型读取同一组系数。"""
    for d in dofs:
        model.dof_damping[d] = value


def smoothstep(s: float) -> float:
    """将归一化进度限制在 0～1，并生成两端斜率为零的平滑过渡。"""
    s = min(1.0, max(0.0, float(s)))
    return 3.0 * s * s - 2.0 * s * s * s


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
    for a in arms:
        data.qpos[a["qpos"]] = a["q_rest"]
        clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    data.qpos[box_qpos:box_qpos + 3] = BOX_CENTER
    data.qpos[box_qpos + 3:box_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    for k, ox in enumerate(CAN_OFFSETS_X):
        cq = int(model.jnt_qposadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                               f"can{k}_freejoint")])
        data.qpos[cq:cq + 3] = [BOX_CENTER[0] + ox, BOX_CENTER[1],
                                BOX_CENTER[2] + SLENDER_HALF[2] + CAN_HALF_H + 0.002]
        data.qpos[cq + 3:cq + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
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
                         disturbance=pulse, viewer=viewer)
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
    # 未传接触模式的旧调用保持切向载荷预测。
    contact_load_mode = getattr(args, "contact_load_mode", "tangent")
    contact_snapshot = {} if contact_load_mode == "residual" else None

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
        """返回与 OCP 左负 x、右正 x 约定一致的当前法向力。"""
        if contact_snapshot is not None:
            return contact_snapshot["normal_magnitude"] * np.array([-1.0, 1.0])
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
        if contact_snapshot is not None:
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
    u_cmd = np.zeros(nc.NU)
    u_clip_mask = np.zeros(nc.NU, dtype=bool)
    u_sat_mask = np.isclose(u_cmd, torque_lo, atol=1e-6) | np.isclose(
        u_cmd, torque_hi, atol=1e-6)
    move_duration = ref_fn.move_duration
    required_end = t_hold_end + (0.3 if args.hold_only else move_duration + 0.4)
    steps = max(args.steps, int(np.ceil(required_end / DT)))
    stop_file = getattr(args, 'disturb_stop_file', None)
    for step in range(steps):
        t = step * DT
        disturbance.begin_step(t)
        # 非控制步保留上一拍的阶段；耗时仅在控制更新时记录。
        solve_ms = np.nan
        if step % CTRL_EVERY == 0:
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
                            prepare_ms if step % CTRL_EVERY == 0 else np.nan,
                            reference_ms, u_cmd, u_clip_mask, u_sat_mask)
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
def main():
    """解析参数、建立抓取姿态和控制器，再运行实验并保存结果。
    普通调用只运行一次；参数扫描由独立实验辅助模块处理。"""
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--force", type=float, default=10, help="稳定夹持力参考 (N)")
    parser.add_argument("--kf_scale", type=float, default=0.003,
                        help="力误差增益，直接乘力误差；当前验证值为 0.003")
    parser.add_argument("--kx", type=float, default=None,
                        help="OCP 夹持弹簧刚度 (N/m)；None=用标定 K_STIFF；"
                             "模型力=kx*max(0,rest_gap-接触间距)，与 MuJoCo 软接触刚度标定对齐")
    parser.add_argument("--qp_solver", type=str, default="PARTIAL_CONDENSING_HPIPM",
                        help="QP 求解器（PARTIAL_CONDENSING_HPIPM/FULL_CONDENSING_HPIPM/"
                             "FULL_CONDENSING_QPOASES 等）")
    parser.add_argument("--qp_solver_iter_max", type=int, default=15)
    parser.add_argument("--qp_solver_cond_N", type=int, default=None,
                        help="部分凝聚块数（仅 PARTIAL_CONDENSING_HPIPM 生效；None=不设置）")
    parser.add_argument("--nlp_solver_max_iter", type=int, default=20)
    parser.add_argument("--Q", type=float, default=1.0,
                        help="阶段代价独立权重")
    parser.add_argument("--P", type=float, default=2.0,
                        help="终端代价独立权重")
    parser.add_argument("--eta_dev", type=float, default=0.05,
                        help="误差导数相对权重 eta_dev")
    parser.add_argument("--eta_u", type=float, default=5e-8,
                        help="输入相对权重 eta_u")
    parser.add_argument("--u_lim", type=float, default=None,
                        help="运行时输入盒约束 ±u_lim (Nm)：标准 MPC 输入限幅 u∈U，"
                             "只改 lbu/ubu 数值，免重新编译；None=不额外钳制，"
                             "沿用物理 actuator_ctrlrange 结构界")
    parser.add_argument("--eta_cop_y", type=float, default=0.8,
                        help="协作映射 eta_cop 的 y 分量；0=绝对 y 跟踪，0.8=同步优先")
    parser.add_argument("--t_ramp", type=float, default=0.5, help="10N→目标力的平滑时间 (s)")
    parser.add_argument("--t_hold", type=float, default=1, help="固定 Holding 时间，结束后直接 Moving (s)")
    parser.add_argument("--f_start", type=float, default=10.0,
                        help="启动夹持力参考 (N)，同时用于初始 IK 压紧和控制器首拍")
    parser.add_argument("--grip_z", type=float, default=0.02,
                        help="垫片接触高度相对箱体重心的竖直偏移 (m)；正=重心上方（摆式稳定），"
                             "负=重心下方")
    parser.add_argument("--pad_half_w", type=float, default=0.060,
                        help="垫片横向半尺寸 (m)；0.060 = 覆盖箱面全宽（12cm）")
    parser.add_argument("--pad_half_h", type=float, default=0.030,
                        help="垫片竖直半尺寸 (m)；加大可增强抗滚转力臂")
    parser.add_argument("--v_y", type=float, default=-0.01)
    parser.add_argument("--y_dist", type=float, default=0.5)
    parser.add_argument("--vz_max", type=float, default=0.05,
                        help="OCP 末端 z 速度约束上限 (m/s)")
    parser.add_argument("--hold_only", action="store_true")
    parser.add_argument("--sliding_friction", type=float, default=0.8,
                        help="箱体及两侧夹持垫的滑动摩擦系数；两侧同时设置，避免取较大值覆盖")
    parser.add_argument("--noslip_iterations", type=int, default=3,
                        help="MuJoCo 每个物理步的 NoSlip 最大迭代次数；0 为关闭")
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--Tp", type=float, default=0.10)
    parser.add_argument("--sim_steps", type=int, default=2,
                        help="acados ERK 每个预测区间的积分步数")
    parser.add_argument("--sim_stages", type=int, default=4,
                        help="acados ERK 积分级数；当前验证配置为 4")
    parser.add_argument("--contact_load_mode", choices=("tangent", "residual"), default="residual",
                        help="接触预测载荷：默认 residual 冻结完整实测载荷减当前模型法向载荷；tangent 回退原切向载荷")
    parser.add_argument("--no_viewer", action="store_true", help="关闭 MuJoCo 可视化窗口")
    experiment_tools.add_arguments(parser)
    disturbance_tools.add_arguments(parser)
    parser.add_argument('--output_dir', default=None, help='本次实验输出目录（须为新目录）')
    args = parser.parse_args()
    if not np.isfinite(args.sliding_friction) or args.sliding_friction < 0:
        parser.error("sliding_friction 必须是非负有限数")
    if args.noslip_iterations < 0:
        parser.error("noslip_iterations 必须是非负整数")
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=False)
        OUT_DIR = str(Path(args.output_dir).resolve())
    print(f"[command] y_dist={args.y_dist:.6f} m, v_y={args.v_y:.6f} m/s, "
          f"hold_only={args.hold_only}")
    print(f"[output] CSV and Fig. 12 files: {os.path.abspath(OUT_DIR)}")

    any_sweep = has_sweep(args)

    scene_mod.GRIP_Z_OFFSET = args.grip_z          # 摆式稳定偏移（场景模块全局常量，供两阶段构建读取）
    scene_mod.PAD_HALF_W = args.pad_half_w         # 垫片横向半宽（加宽垫片）
    scene_mod.PAD_HALF_H = args.pad_half_h         # 垫片竖直半高
    scene_xml, model, data, arms = build_grasped_scene(IK_ITERS)
    for a in arms:
        joint_ids = a["joint_ids"]
        a["act_ids"] = [name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                    f"{model.joint(j).name}_motor") for j in joint_ids]

    box_body = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_geom = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    box_joint = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")
    box_qpos = int(model.jnt_qposadr[box_joint])
    ids = {"box_body": box_body, "box_geom": box_geom, "box_qpos": box_qpos}

    # 仅覆盖本入口的运行模型；箱体与垫片同步修改，保留扭转、滚动摩擦和接触刚度。
    model.geom_friction[[box_geom] + [a["pad_id"] for a in arms], 0] = args.sliding_friction
    model.opt.noslip_iterations = args.noslip_iterations
    print(f"[contact] box/pads sliding_friction={args.sliding_friction:g}, "
          f"noslip_iterations={model.opt.noslip_iterations}")

    reset_world(model, data, arms, box_qpos)

    # --- 目标姿态：轻接触起点 + 压入终点（不改几何）---
    base_targets = grasp_targets(arms)

    def press_for(force_n):
        delta = force_n / (2.0 * K_STIFF)
        return {
            a["side"]: base_targets[a["side"]] + np.array([-a["sign"] * delta, 0.0, 0.0])
            for a in arms
        }

    press_light = press_for(args.f_start)      # 与首拍参考一致的启动预压（默认 10 N）
    press_full_for = press_for
    solve_end_grip(model, data, arms, press_light, IK_ITERS)

    contact_offsets = nc.pad_contact_offsets(model, arms, 0.1001)
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
    rv_ref = make_rv_ref()
    arm_damping(model, [d for a in arms for d in a["dofs"]], 0.5)

    umax_parts = []
    for a in arms:
        umax_parts.append(model.actuator_ctrlrange[a["act_ids"], 1])
    umax = np.concatenate(umax_parts)
    qmin, qmax = arm_joint_bounds(model, arms)

    NMPC_CACHE = {}

    def build_nmpc(kf):
        # 模型名及控制器指纹区分编译配置；kf 为运行期参数。
        kx_eff = float(args.kx) if args.kx is not None else float(K_STIFF)
        parts = [f"Q{args.Q:g}", f"P{args.P:g}",
                 f"ed{args.eta_dev:g}", f"eu{args.eta_u:g}",
                 f"vz{args.vz_max:g}", f"ss{args.sim_steps}"]
        extra = []
        if abs(kx_eff - float(K_STIFF)) > 1e-9:
            extra.append(f"kx{float(kx_eff):.9g}")
        if abs(REST_GAP - 2.0 * SLENDER_HALF[0]) > 1e-9:
            extra.append(f"rg{float(REST_GAP):.9g}")
        # 注意：N 标签的基准保留历史值 20（默认已是 10），使默认配置生成的
        # 模型名继续带 "N10"，与既有 codegen 缓存目录一致、可直接复用。
        if args.N != 20:
            extra.append(f"N{args.N}")
        if abs(args.Tp - 0.10) > 1e-12:
            extra.append(f"Tp{args.Tp:g}")
        if args.qp_solver != "PARTIAL_CONDENSING_HPIPM":
            # 短标签：全名会让 codegen 路径超 Windows MAX_PATH（260）导致 CasADi 写文件失败
            qp_tag = {"FULL_CONDENSING_HPIPM": "FCH",
                      "FULL_CONDENSING_QPOASES": "FCQ"}.get(args.qp_solver,
                                                            args.qp_solver)
            extra.append(f"qp{qp_tag}")
        if args.sim_stages != 4:
            extra.append(f"st{args.sim_stages}")
        if args.qp_solver_iter_max != 15:
            extra.append(f"qi{args.qp_solver_iter_max}")
        if args.qp_solver_cond_N is not None:
            extra.append(f"qc{args.qp_solver_cond_N}")
        if args.nlp_solver_max_iter != 20:
            extra.append(f"ni{args.nlp_solver_max_iter}")
        model_name = ("nmpcc_cart_dual_force_bias_v7_contact_" + "_".join(parts)
                      + (("_" + "_".join(extra)) if extra else ""))
        model_name = model_name.replace(".", "p").replace("-", "m")
        cfg = nc.NMPCCCartConfig(
            N=args.N, Tp=args.Tp,
            eta_cop=np.array([[0.0, args.eta_cop_y, 0.0, 0.0, 0.0, 0.0]] * 2),
            eta_f=np.array([[0.8, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2),
            Q=args.Q,
            eta_dev=args.eta_dev,
            eta_u=args.eta_u,
            P=args.P,
            model_name=model_name,
            kx=kx_eff,
            rest_gap=REST_GAP,
            kf_scale=kf,
            vmax=0.05,
            vz_max=args.vz_max,
            sim_method_num_steps=args.sim_steps,
            sim_method_num_stages=args.sim_stages,
            qp_solver=args.qp_solver,
            qp_solver_iter_max=args.qp_solver_iter_max,
            qp_solver_cond_N=args.qp_solver_cond_N,
            nlp_solver_max_iter=args.nlp_solver_max_iter,
        )
        return nc.NMPCCCart(cfg, fk, umin=-umax, umax=umax, qmin=qmin, qmax=qmax)

    def run_one(kf, force, tag):
        args.force = force
        args.kf_scale = kf
        # 每次运行复位场景，并按启动力重建抓取姿态
        reset_world(model, data, arms, box_qpos)
        pl = press_for(args.f_start)
        solve_end_grip(model, data, arms, pl, IK_ITERS)
        rv_ref_now = make_rv_ref()
        pf = press_full_for(force)
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
                args.eta_cop_y, args.sim_steps, args.sim_stages, args.N, args.Tp, args.vz_max,
                args.kx, args.qp_solver, args.qp_solver_iter_max,
                args.qp_solver_cond_N, args.nlp_solver_max_iter)
        nmpc = NMPC_CACHE.get(okey)
        if nmpc is None:
            nmpc = build_nmpc(kf)
            NMPC_CACHE[okey] = nmpc
            print(f"NMPCC built: K={K_STIFF:.3f}, rest_gap={REST_GAP:.6f}, "
                  f"N={args.N}, Tp={args.Tp:.3f}, force={force:.1f}N, kf_scale={kf:.3f}, "
                  f"kx={nmpc.cfg.kx:.3f}, "
                  f"Q={args.Q:.3g}, P={args.P:.3g}, eta_dev={args.eta_dev:.3g}, "
                  f"eta_u={args.eta_u:.3g}, "
                  f"qp_solver={args.qp_solver}, qi={args.qp_solver_iter_max}, "
                  f"qc={args.qp_solver_cond_N}, ni={args.nlp_solver_max_iter}, "
                  f"eta_cop_y={args.eta_cop_y:.2f}")
        else:
            print(f"NMPCC cached (same OCP params); force={force:.1f}N, "
                  f"t_ramp={args.t_ramp:.2f}, t_hold={args.t_hold:.2f}, f_start={args.f_start:.1f}")
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
        t0 = time.perf_counter()
        try:
            log, horizon_log, m = run_case(model, data, arms, ids, args, nmpc, rfn, th_end,
                                         viewer=viewer)
        except Exception:
            experiment_tools.save_failure(nmpc, tag, OUT_DIR)
            raise
        dt = time.perf_counter() - t0
        return experiment_tools.finish_run(log, horizon_log, m, args, nmpc, tag, dt, OUT_DIR, __file__)

    if any_sweep:
        run_sweep(args, run_one, OUT_DIR)
        return

    m = run_one(args.kf_scale, args.force, "single")
    print("--- latest Experiment II (summary) ---")
    for k, v in m.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
