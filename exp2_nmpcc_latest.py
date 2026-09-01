#!/usr/bin/env python3
"""exp2_nmpcc_latest.py —— 最新地面场景的实验二 NMPCC（Holding 偏转修复版）

使用 exp2_slender_scene.py 的地面双臂场景（BASE_X=0.65、Z_MOUNT=0.06、箱中心 z=0.11；
箱 0.50 x 0.12 x 0.08 m，4 罐沿 x；K_stiff=2793.166 N/m、rest_gap=0.507537 m；
x 轴 eta_f=0.8、y 轴 eta_cop=0.8、z/姿态跟踪；acados 每 6ms 求解一次）：
  - 摆式稳定夹持：垫片接触相对箱体重心上偏 GRIP_Z_OFFSET=+0.02 m，箱体重心悬挂在
    夹持线下方 → 重力对绕夹持轴(x)的滚转提供恢复力矩（中性稳定 → 摆式稳定），
    同时约束箱体 y 漂移；x 向 K_STIFF/REST_GAP 标定不变。

针对 Holding 偏转的修复（控制律不动：nmpcc_cart.py 仅恢复过一次并发工作流误写损坏并合并其求解器调参改动，见交接说明）：
  1. 启动几何/FK 对账与接触测量等检测类函数已移到 exp2_diagnostics.py
     （可独立运行，见下方"运行"）；主程序只保留仿真与作图；
  2. 日志扩展：三维世界系接触力 + 接触力矩（每块垫片，含对箱体质心的净力偶）、
     箱体完整姿态（quaternion + rotvec），逐拍落 CSV；
  3. 高力启动：初始 IK 与控制器首拍统一使用 f_start=10 N，随后在 0.5 s 内平滑到
     默认 12 N，避免低力接手瞬间箱体先下落；
  4. 固定时序：ramp 后 Holding 3 s，随后无条件进入 Moving，不使用稳定 gate；
  5. 控制层精简：夹持力完全由 NMPCC 力残差承担（力参考进 OCP 参数 p），已移除
     clamp_balance_comp 开环预紧前馈与 y_guard 附加保护层；
  6. CLI：--force --kf_scale --Q --P --eta_dev --eta_u --t_ramp --t_hold
     --sweep_forces --sweep_kf；保留 --no_viewer --no_csv。

运行：
  启动几何/FK 对账（检测工具，独立运行，不占用主程序）：
    D:\\acados_env\\python.exe -u exp2_diagnostics.py
  仅保持：
    D:\\acados_env\\python.exe -u exp2_nmpcc_latest.py --hold_only --no_viewer
  力/增益扫描（同进程复用求解器；force 是参数不触发重建，kf_scale 每值重建一次）：
    D:\\acados_env\\python.exe -u exp2_nmpcc_latest.py --hold_only --no_viewer --sweep_forces 10,12,15 --sweep_kf 0.05,0.10,0.20
  定时搬运（0.5 s ramp + 3 s Holding 后进入 Moving）：
    D:\\acados_env\\python.exe -u exp2_nmpcc_latest.py --y_dist 0.08
"""
from __future__ import annotations

import argparse
import csv
import os
import json
import time

_DEBUG_LOG = r"D:\RobotSim\.cursor\debug-068a93.log"

def _debug_log(hypothesis_id, location, message, data):
    record = {"sessionId": "068a93", "runId": "pre-fix", "hypothesisId": hypothesis_id,
              "location": location, "message": message, "data": data,
              "timestamp": int(time.time() * 1000)}
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception:
        pass

import mujoco
import mujoco.viewer
import numpy as np

import nmpcc_cart as nc
import exp2_slender_scene as scene_mod
from exp2_diagnostics import (
    box_attitude, box_net_wrench, contact_forces,
    contact_wrench_on_arm, pad_box_wrench_3d,
)
from exp2_slender_scene import (
    BOX_CENTER, CAN_COUNT, CAN_HALF_H, CAN_OFFSETS_X,
    IK_ITERS, SLENDER_HALF, build_grasped_scene, grasp_targets, make_arms, solve_end_grip,
)
from ik import clamp_qpos, get_ee_tip, name_to_id

# 由 calib_exp2_contact.py 得到；场景改变后必须重新标定
K_STIFF = 2793.166
# 接触面口径的零压入间距：左右 pad 内侧面分别对齐箱体两端面，
# 因此等于箱体沿夹持轴的实际长度，而不是 link7 tip 中心间距。
REST_GAP = 2.0 * SLENDER_HALF[0]
PAD_HALF_THICKNESS = 0.004
DT = 0.002
CTRL_EVERY = 3
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "exp2_latest")

# 仅用于结果评价
HOLD_BOX_Y_TOL = 0.002


def arm_damping(model, dofs, value=0.5):
    for d in dofs:
        model.dof_damping[d] = value


def smoothstep(s: float) -> float:
    s = min(1.0, max(0.0, float(s)))
    return 3.0 * s * s - 2.0 * s * s * s


# ======================================================================
# 阶段参考：平滑预紧 + holding -> moving（参数进 OCP p，控制器恒定）
# ======================================================================

#三个阶段，前期夹紧夹紧的时候平缓输入预期的力，到移动阶段同理平缓输入预期速度
def phase_profile(force, vy, distance, t_ramp, t_hold, press0, press, rv_ref,
                  f_start=2.0):
    """返回 sample(t) -> refs, vrefs, fmag, phase, ramp_s。

    参考按固定时间切换：先平滑预紧 ``t_ramp``，再保持 ``t_hold``，随后直接
    进入 Moving；不再使用稳定门控。``f_start`` 与启动 IK 压紧位一致，因此
    提高它可以避免控制器接手瞬间先释放夹持力。
    """
    distance = abs(distance)
    speed = abs(vy)
    motion_enabled = distance > 1e-9 and speed > 1e-9
    t_hold_end = t_ramp + t_hold
    segment = distance / speed if motion_enabled else 0.0

    def sample(t):
        s = smoothstep(t / t_ramp) if t_ramp > 1e-9 else 1.0
        fmag = f_start + (force - f_start) * s

        press_now = {
            side: press0[side] + (press[side] - press0[side]) * s
            for side in press0
        }
        if t < t_ramp:
            phase = "ramp"
        elif not motion_enabled or t < t_hold_end:
            phase = "hold"
        else:
            phase = "move"
        offset_y = 0.0
        velocity_y = 0.0

        if phase == "move":
            tau = t - t_hold_end
            ramp = min(0.20, 0.5 * segment)
            if tau < ramp:
                ss = tau / ramp
                smth = 3.0 * ss * ss - 2.0 * ss * ss * ss
                offset_y = -speed * ramp * (ss ** 3 - 0.5 * ss ** 4)
                velocity_y = -speed * smth
            elif tau < segment:
                offset_y = -(0.5 * speed * ramp + speed * (tau - ramp))
                velocity_y = -speed
            else:
                offset_y = -distance
                velocity_y = 0.0
        refs, vrefs = {}, {}
        print("t=", t, "phase=", phase)
        for side in ("left", "right"):
            base = press_now[side]
            refs[side] = np.concatenate([
                [base[0], base[1] + offset_y, base[2]], rv_ref[side],
            ])
            vrefs[side] = np.array([0.0, velocity_y, 0.0, 0.0, 0.0, 0.0])
        return refs, vrefs, fmag, phase, s

    return sample, t_hold_end


# ======================================================================
# 场景复位（sweep 复用同一 model/data）
# ======================================================================
def reset_world(model, data, arms, box_qpos) -> None:
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


# ======================================================================
# 单次仿真（供 main 与 sweep 共用）
# ======================================================================
def run_case(model, data, arms, ids, args, nmpc, ref_fn, t_hold_end,
             viewer=None) -> (list, dict):
    box_body, box_geom, box_qpos = ids["box_body"], ids["box_geom"], ids["box_qpos"]

    def joint_state():
        return np.concatenate([
            data.qpos[arms[0]["qpos"]], data.qvel[arms[0]["dofs"]],
            data.qpos[arms[1]["qpos"]], data.qvel[arms[1]["dofs"]],
        ])

    def plant_params():
        full_m = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, full_m)
        mgs = []
        fjs = []
        for a in arms:
            mm = full_m[np.ix_(a["dofs"], a["dofs"])]
            saved = data.qvel.copy()
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            bias = data.qfrc_bias[a["dofs"]].copy()
            data.qvel[:] = saved
            mujoco.mj_forward(model, data)
            fJ, _ = contact_wrench_on_arm(model, data, a, box_geom)
            mgs.extend([mm.reshape(-1), bias])
            fjs.append(fJ)
        damping = np.concatenate([model.dof_damping[a["dofs"]] for a in arms])
        return np.concatenate(mgs + fjs + [damping])

    def measured_normal_force():
        """返回与 OCP 左负 x、右正 x 约定一致的当前法向力。"""
        forces = contact_forces(model, data, arms, box_geom)
        return np.array([-forces["left"], forces["right"]], dtype=float)

    def gravity_comp():
        out = []
        for a in arms:
            saved = data.qvel.copy()
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            out.append(data.qfrc_bias[a["dofs"]].copy())
            data.qvel[:] = saved
            mujoco.mj_forward(model, data)
        return np.concatenate(out)

    def reference_for(t):
        refs, vrefs, fmag, phase, ramp_s = ref_fn(t)
        return refs, vrefs, fmag, phase, ramp_s

    def build_p(t):
        refs, vrefs, fmag, phase, ramp_s = reference_for(t)
        p = np.zeros(nc.NP)
        pp = plant_params()
        p[0:112] = pp[:112]
        p[112:126] = pp[112:126]
        p[162:176] = pp[126:140]
        p[126:132] = refs["left"]
        p[132:138] = refs["right"]
        p[138:144] = vrefs["left"]
        p[144:150] = vrefs["right"]
        p[150:156] = [-fmag, 0, 0, 0, 0, 0]
        p[156:162] = [fmag, 0, 0, 0, 0, 0]
        return p, fmag, phase

    # t=0 前 IK 已建立 f_start 预压；预热读取该姿态的真实接触力且不推进仿真。
    p0, _, _ = build_p(0.0)
    u_seed = gravity_comp()
    nmpc._warm = None
    for _ in range(2):
        nmpc.solve(joint_state(), p0, u_ref=u_seed,
                   f_meas=measured_normal_force(), u_seed=u_seed)

    initial_box_y = float(data.xpos[box_body][1])
    print(f"[timed] ramp={args.t_ramp:.3f}s, holding={args.t_hold:.3f}s, "
          f"moving starts at t={t_hold_end:.3f}s"
          if abs(args.y_dist) > 1e-9 and abs(args.v_y) > 1e-9 and not args.hold_only
          else f"[timed] ramp={args.t_ramp:.3f}s, then holding; translation disabled")

    log = []
    horizon_log = []
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
    force_nominal = np.full(2, np.nan)
    force_measured_control = np.full(2, np.nan)
    force_bias = np.full(2, np.nan)
    force_anchored = np.full(2, np.nan)
    force_ref = np.array([-args.f_start, args.f_start], dtype=float)
    pred_force = np.full((3, 2), np.nan)
    pred_force_error = np.full((3, 2), np.nan)
    pred_position_error_x = np.full((3, 2), np.nan)
    pred_eps_x = np.full((3, 2), np.nan)
    pred_epsd_x = np.full((3, 2), np.nan)
    pred_u = np.full((3, nc.NU), np.nan)
    move_duration = (0.0 if args.hold_only or abs(args.v_y) <= 1e-9
                     else abs(args.y_dist) / abs(args.v_y))
    required_end = t_hold_end + (0.3 if args.hold_only else move_duration + 0.4)
    steps = max(args.steps, int(np.ceil(required_end / DT)))
    for step in range(steps):
        t = step * DT
        # solve_ms is populated only on control updates; phase is held between
        # updates so the CSV reflects the actual reference phase.
        solve_ms = np.nan
        if step % CTRL_EVERY == 0:
            t0 = time.perf_counter()
            p, fmag, phase = build_p(t)
            f_meas = measured_normal_force()
            u_ref_now = gravity_comp()
            u_last = nmpc.solve(joint_state(), p,
                                u_ref=u_ref_now,
                                f_meas=f_meas,
                                u_seed=u_seed)
            force_nominal = nmpc.last_force_nominal.copy()
            force_measured_control = nmpc.last_force_measured.copy()
            force_bias = nmpc.last_force_bias.copy()
            force_anchored = nmpc.last_force_anchored.copy()
            force_ref = np.array([-fmag, fmag], dtype=float)
            state_nodes = [0, args.N // 2, args.N]
            input_nodes = [0, max(0, args.N // 2), args.N - 1]
            pred_force = nmpc.last_pred_force[state_nodes].copy()
            pred_force_error = nmpc.last_pred_force_error[state_nodes].copy()
            pred_position_error_x = nmpc.last_pred_position_error_x[state_nodes].copy()
            pred_eps_x = nmpc.last_pred_eps_x[state_nodes].copy()
            pred_epsd_x = nmpc.last_pred_epsd_x[state_nodes].copy()
            pred_u = nmpc.last_pred_u[input_nodes].copy()
            horizon_log.append({
                "t": t,
                "phase": phase,
                "force_ref": force_ref.copy(),
                "force": nmpc.last_pred_force.copy(),
                "force_error": nmpc.last_pred_force_error.copy(),
                "position_error_x": nmpc.last_pred_position_error_x.copy(),
                "eps_x": nmpc.last_pred_eps_x.copy(),
                "epsd_x": nmpc.last_pred_epsd_x.copy(),
                "u": nmpc.last_pred_u.copy(),
                "u_ref": u_ref_now.copy(),
            })
            u_cmd = np.clip(u_last, torque_lo, torque_hi)
            u_clip_mask = np.abs(u_cmd - np.asarray(u_last, dtype=float)) > 1e-9
            # Count both explicit clipping and an optimizer solution lying on an
            # actuator bound. The latter is a real saturation even if np.clip
            # does not numerically change the command.
            u_sat_mask = u_clip_mask | np.isclose(u_cmd, torque_lo, atol=1e-6) | np.isclose(
                u_cmd, torque_hi, atol=1e-6)
            solve_ms = (time.perf_counter() - t0) * 1000.0
            if step == 0:
                print(f"[force-anchor t0] meas={np.array2string(f_meas, precision=4)} "
                      f"nominal={np.array2string(force_nominal, precision=4)} "
                      f"bias={np.array2string(force_bias, precision=4)} "
                      f"anchored={np.array2string(force_anchored, precision=4)} "
                      f"ref={np.array2string(force_ref, precision=4)}")
                print(f"[u0] {np.array2string(u_cmd, precision=12, floatmode='unique')}")
                print(f"[nmpc-pred t0] F(k0,mid,N)={np.array2string(pred_force, precision=3)}")
                print(f"[nmpc-pred t0] Ferr={np.array2string(pred_force_error, precision=3)} "
                      f"xerr={np.array2string(pred_position_error_x, precision=5)} "
                      f"eps_x={np.array2string(pred_eps_x, precision=5)}")
                print(f"[nmpc-pred t0] u(k0,mid,last)={np.array2string(pred_u, precision=3)}")
                eps0 = nmpc.last_pred_eps_x[0]
                epsd0 = nmpc.last_pred_epsd_x[0]
                u0 = nmpc.last_pred_u[0]
                udev0 = u0  # 论文式(31)：输入项零参考 ūᵀηᵤū，不做 u-u_ref 差值
                cost_eps = args.Q * float(eps0 @ eps0)
                cost_epsd = args.Q * args.eta_dev * float(epsd0 @ epsd0)
                cost_u = args.Q * args.eta_u * float(udev0 @ udev0)
                print(f"[nmpc-cost k0] eps2={cost_eps / max(args.Q, 1e-12):.9g} "
                      f"epsd2={cost_epsd / max(args.Q * args.eta_dev, 1e-12):.9g} "
                      f"udev2={cost_u / max(args.Q * args.eta_u, 1e-12):.9g}")
                print(f"[nmpc-cost k0 weighted] Qeps={cost_eps:.9g} "
                      f"Qeta_devepsd={cost_epsd:.9g} Qeta_uudev={cost_u:.9g} "
                      f"total={cost_eps + cost_epsd + cost_u:.9g}")
                print(f"[nmpc-state k0] r={np.array2string(nmpc.last_pred_r[0], precision=9)} "
                      f"rdot={np.array2string(nmpc.last_pred_rdot[0], precision=9)}")
                print(f"[nmpc-force k0] nominal={np.array2string(nmpc.last_pred_force_nominal[0], precision=9)} "
                      f"anchored={np.array2string(pred_force[0], precision=9)} "
                      f"ref={np.array2string(force_ref, precision=9)}")
        for a in arms:
            arm_i = arms.index(a)
            data.ctrl[a["act_ids"]] = u_cmd[7 * arm_i:7 * arm_i + 7]
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
            time.sleep(DT)
        if step % 5 == 0:
            forces = contact_forces(model, data, arms, box_geom)
            w = pad_box_wrench_3d(model, data, arms, box_geom)
            rv, quat = box_attitude(data, box_qpos)

            print(f"t={t + DT:.3f}s  左臂对箱 F={-w['left'][0]} N | 力矩={-w['left'][1]} N·m"
                  f" | boxz={data.xpos[box_body][2]:.4f} tiltx={np.degrees(rv[0]):+.2f}°")

            fb, tb = box_net_wrench(model, data, arms, box_geom,
                                    data.xpos[box_body].copy())
            if step == 0:
                print(f"[t0] FL={forces['left']:.4f} FR={forces['right']:.4f} "
                      f"boxz={data.xpos[box_body][2]:.6f} "
                      f"tiltx={np.degrees(rv[0]):.4f} "
                      f"tipL={get_ee_tip(data, arms[0]['body_id'])}")
            log.append({
                "t": t + DT,
                "phase": phase,
                "ramp_s": smoothstep(t / args.t_ramp) if args.t_ramp > 1e-9 else 1.0,
                "moving_active": 1.0 if phase == "move" else 0.0,
                "box": data.xpos[box_body].copy(),
                "quat": quat.copy(),
                "rotvec": rv.copy(),
                "tipL": get_ee_tip(data, arms[0]["body_id"]).copy(),
                "tipR": get_ee_tip(data, arms[1]["body_id"]).copy(),
                "wL": w["left"], "wR": w["right"],
                "box_f": fb, "box_t": tb,
                "u_cmd": u_cmd.copy(),
                "u_clip_mask": u_clip_mask.copy(),
                "u_sat_mask": u_sat_mask.copy(),
                "FL": forces["left"], "FR": forces["right"],
                "force_nominal": force_nominal.copy(),
                "force_measured_control": force_measured_control.copy(),
                "force_bias": force_bias.copy(),
                "force_anchored": force_anchored.copy(),
                "force_ref": force_ref.copy(),
                "pred_force": pred_force.copy(),
                "pred_force_error": pred_force_error.copy(),
                "pred_position_error_x": pred_position_error_x.copy(),
                "pred_eps_x": pred_eps_x.copy(),
                "pred_epsd_x": pred_epsd_x.copy(),
                "pred_u": pred_u.copy(),
                "solve_ms": solve_ms if step % CTRL_EVERY == 0 else np.nan,
                "solver_status": (float(nmpc.last_status)
                                   if step % CTRL_EVERY == 0 else np.nan),
                "qp_iter": (float(nmpc.last_qp_iter)
                             if step % CTRL_EVERY == 0 else np.nan),
                "solver_qp_ms": (nmpc.last_time_qp * 1000.0
                                  if step % CTRL_EVERY == 0 else np.nan),
                "solver_sim_ms": (nmpc.last_time_sim * 1000.0
                                   if step % CTRL_EVERY == 0 else np.nan),
                "solver_lin_ms": (nmpc.last_time_lin * 1000.0
                                   if step % CTRL_EVERY == 0 else np.nan),
            })
        if args.hold_only and t >= t_hold_end + 0.3:
            break
    if viewer is not None:
        viewer.close()
    metrics = summarize(log, args, initial_box_y)
    return log, horizon_log, metrics


def summarize(log, args, initial_box_y) -> dict:
    tt = np.array([r["t"] for r in log])
    box = np.array([r["box"] for r in log])
    rv = np.array([r["rotvec"] for r in log])
    tip_l = np.array([r["tipL"] for r in log])
    tip_r = np.array([r["tipR"] for r in log])
    fl = np.array([r["FL"] for r in log]); fr = np.array([r["FR"] for r in log])
    w_l_t = np.array([r["wL"][1] for r in log])
    w_r_t = np.array([r["wR"][1] for r in log])
    box_t = np.array([r["box_t"] for r in log])
    u_clip = np.array([r["u_clip_mask"] for r in log], dtype=bool)
    u_sat = np.array([r["u_sat_mask"] for r in log], dtype=bool)
    motion_commanded = (not args.hold_only and abs(float(args.y_dist)) > 1e-9
                        and abs(float(args.v_y)) > 1e-9)
    # hold 全窗（worst-case 评估用）：hold_only = [t_ramp, 结束]；moving = [t_ramp, hold_end]
    if not motion_commanded:
        sel = tt >= max(0.0, tt[-1] - 0.3)
        sel_hold = tt >= args.t_ramp
        sel_move = np.zeros_like(tt, dtype=bool)
    elif motion_commanded:
        sel = (tt >= args.t_ramp + args.t_hold - 0.3) & (tt <= args.t_ramp + args.t_hold)
        sel_hold = (tt >= args.t_ramp) & (tt <= args.t_ramp + args.t_hold)
        sel_move = np.array([r["phase"] == "move" for r in log], dtype=bool)
    if not sel.any():
        sel = tt >= tt[-1] - 0.3
    if not sel_hold.any():
        sel_hold = sel
    if not sel_move.any():
        sel_move = np.zeros_like(tt, dtype=bool)
    ys_mm = (tip_l[:, 1] - tip_r[:, 1]) * 1000.0
    rv_deg = np.abs(rv) * 180.0 / np.pi
    initial_box_y = float(initial_box_y)

    def max_or_nan(values):
        values = np.asarray(values)
        return float(np.max(values)) if values.size else float("nan")

    def mean_or_nan(values):
        values = np.asarray(values)
        return float(np.mean(values)) if values.size else float("nan")

    def drift_mm(values):
        values = np.asarray(values)
        return (float(np.max(np.abs(values - np.mean(values))) * 1000.0)
                if values.size else float("nan"))

    box_y_hold_max_mm = max_or_nan(np.abs(box[sel_hold, 1] - initial_box_y) * 1000.0)
    box_y_move_delta_mm = ((box[sel_move, 1] - box[sel_move, 1][0]) * 1000.0
                           if sel_move.any() else np.array([]))
    box_y_move_travel_mm = (-float(box_y_move_delta_mm[-1])
                            if box_y_move_delta_mm.size else float("nan"))
    box_y_move_travel_error_mm = (abs(box_y_move_travel_mm - abs(float(args.y_dist)) * 1000.0)
                                  if np.isfinite(box_y_move_travel_mm) else float("nan"))
    control_rows = [r for r in log if np.isfinite(r["solver_status"])]
    statuses = np.array([r["solver_status"] for r in control_rows], dtype=float)
    qp_iters = np.array([r["qp_iter"] for r in control_rows], dtype=float)
    status_bad = statuses[(statuses >= 0) & (statuses != 0)]
    move_observed = bool(sel_move.any())
    commanded_distance = 0.0 if args.hold_only else abs(float(args.y_dist))
    speed = abs(float(args.v_y))
    move_duration = commanded_distance / speed if speed > 1e-9 else float("inf")
    expected_move_end = args.t_ramp + args.t_hold + move_duration
    box_y_final_error_mm = abs((box[-1, 1] - initial_box_y) + commanded_distance) * 1000.0
    # The plant is still stepped for a short settling tail after the commanded
    # translation.  Evaluate the endpoint at the end of the commanded motion,
    # rather than at the final tail sample.
    move_end_t = args.t_ramp + args.t_hold + move_duration
    endpoint_idx = int(np.argmin(np.abs(tt - move_end_t)))
    box_y_endpoint_error_mm = abs(
        (box[endpoint_idx, 1] - initial_box_y) + commanded_distance
    ) * 1000.0
    box_y_initial_error_mm = box_y_endpoint_error_mm
    # 评价固定时间切换后的实际搬运行程。
    move_complete_after_settle = int(
        move_observed and tt[-1] + DT >= expected_move_end
        and box_y_move_travel_error_mm <= 2.0
    )
    move_complete = int(
        move_complete_after_settle
        and box_y_hold_max_mm <= HOLD_BOX_Y_TOL * 1000.0
        and box_y_initial_error_mm <= 2.0
    )
    m = {
        "force": args.force, "kf_scale": args.kf_scale,
        "Q": args.Q, "P": args.P, "eta_dev": args.eta_dev,
        "eta_u": args.eta_u, "eta_cop_y": args.eta_cop_y,
        "t_ramp": args.t_ramp, "t_hold": args.t_hold,
        "f_start": args.f_start,
        "transition_mode": "timed",
        "motion_commanded": int(motion_commanded),
        "moving_start_s": args.t_ramp + args.t_hold,
        "tilt_x_max_deg": max_or_nan(rv_deg[sel, 0]),
        "tilt_y_max_deg": max_or_nan(rv_deg[sel, 1]),
        "tilt_z_max_deg": max_or_nan(rv_deg[sel, 2]),
        "tilt_x_hold_max_deg": max_or_nan(rv_deg[sel_hold, 0]),
        "tilt_y_hold_max_deg": max_or_nan(rv_deg[sel_hold, 1]),
        "tilt_z_hold_max_deg": max_or_nan(rv_deg[sel_hold, 2]),
        "z_sag_mm": mean_or_nan(box[sel, 2] - BOX_CENTER[2]) * 1000.0,
        "z_drift_mm": drift_mm(box[sel, 2]),
        "z_drift_hold_mm": drift_mm(box[sel_hold, 2]),
        "FL_mean": mean_or_nan(fl[sel]), "FR_mean": mean_or_nan(fr[sel]),
        "F_diff_max_N": max_or_nan(np.abs(fl[sel] - fr[sel])),
        "F_diff_hold_max_N": max_or_nan(np.abs(fl[sel_hold] - fr[sel_hold])),
        "y_sync_mean_mm": mean_or_nan(np.abs(ys_mm)),
        "y_sync_max_mm": max_or_nan(np.abs(ys_mm)),
        "y_sync_hold_max_mm": max_or_nan(np.abs(ys_mm[sel_hold])),
        "tilt_x_move_max_deg": max_or_nan(rv_deg[sel_move, 0]),
        "tilt_y_move_max_deg": max_or_nan(rv_deg[sel_move, 1]),
        "tilt_z_move_max_deg": max_or_nan(rv_deg[sel_move, 2]),
        "z_drift_move_mm": drift_mm(box[sel_move, 2]),
        "F_diff_move_max_N": max_or_nan(np.abs(fl[sel_move] - fr[sel_move])),
        "contact_moment_left_move_max_Nm": max_or_nan(
            np.linalg.norm(w_l_t[sel_move], axis=1)),
        "contact_moment_right_move_max_Nm": max_or_nan(
            np.linalg.norm(w_r_t[sel_move], axis=1)),
        "contact_moment_box_move_max_Nm": max_or_nan(
            np.linalg.norm(box_t[sel_move], axis=1)),
        "y_sync_move_mean_mm": mean_or_nan(np.abs(ys_mm[sel_move])),
        "y_sync_move_max_mm": max_or_nan(np.abs(ys_mm[sel_move])),
        "actuator_clip_fraction": mean_or_nan(u_clip),
        "actuator_saturation_fraction": mean_or_nan(u_sat),
        "actuator_clip_move_fraction": mean_or_nan(u_clip[sel_move]),
        "actuator_saturation_move_fraction": mean_or_nan(u_sat[sel_move]),
        "move_observed": int(move_observed),
        "move_complete": move_complete,
        "move_complete_after_settle": move_complete_after_settle,
        "move_expected_end_s": expected_move_end,
        "move_end_t_s": float(tt[-1]),
        "initial_box_y": initial_box_y,
        "box_y_final_drift_mm": (float(box[-1, 1] - initial_box_y) * 1000.0),
        "box_y_hold_max_mm": box_y_hold_max_mm,
        "box_y_initial_error_mm": box_y_initial_error_mm,
        "box_y_move_travel_mm": box_y_move_travel_mm,
        "box_y_move_travel_error_mm": box_y_move_travel_error_mm,
        "box_y_endpoint_error_mm": box_y_endpoint_error_mm,
        "box_y_final_error_mm": box_y_final_error_mm,
        "solver_status_bad_count": int(status_bad.size),
        "solver_status_zero_fraction": (float(np.mean(statuses == 0.0))
                                         if statuses.size else float("nan")),
        "qp_iter_mean": mean_or_nan(qp_iters),
        "qp_iter_max": max_or_nan(qp_iters),
        "solver_qp_ms_mean": mean_or_nan(
            [r["solver_qp_ms"] for r in control_rows]),
        "solver_sim_ms_mean": mean_or_nan(
            [r["solver_sim_ms"] for r in control_rows]),
        "solver_lin_ms_mean": mean_or_nan(
            [r["solver_lin_ms"] for r in control_rows]),
        "solve_ms_mean": mean_or_nan([r["solve_ms"] for r in log
                                       if np.isfinite(r["solve_ms"])]),
        "solve_ms_max": max_or_nan([r["solve_ms"] for r in log
                                     if np.isfinite(r["solve_ms"])]),
    }
    return m


# ======================================================================
# 主流程
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--force", type=float, default=10, help="稳定夹持力参考 (N)")
    parser.add_argument("--kf_scale", type=float, default=0.008, help="力误差增益 (nmpcc_cart 配置)")
    parser.add_argument("--Q", type=float, default=1.0,
                        help="论文式(31)阶段损失外部标量 Q")
    parser.add_argument("--P", type=float, default=2.0,
                        help="论文式(32)终端损失外部标量 P")
    parser.add_argument("--eta_dev", type=float, default=3000.0,
                        help="论文式(31)误差导数相对权重 eta_dev")
    parser.add_argument("--eta_u", type=float, default=0.8,
                        help="论文式(31)输入相对权重 eta_u")
    parser.add_argument("--eta_cop_y", type=float, default=0.8,
                        help="论文合作映射 eta_cop 的 y 分量；0=绝对 y 跟踪，0.8=同步优先")
    parser.add_argument("--t_ramp", type=float, default=0.5, help="10N→目标力的平滑时间 (s)")
    parser.add_argument("--t_hold", type=float, default=1, help="固定 Holding 时间，结束后直接 Moving (s)")
    parser.add_argument("--f_start", type=float, default=10.0,
                        help="启动夹持力参考 (N)，同时用于初始 IK 压紧和控制器首拍")
    parser.add_argument("--grip_z", type=float, default=0.02,
                        help="垫片接触高度相对箱体重心的竖直偏移 (m)；正=重心上方（摆式稳定），"
                             "负=重心下方（顶重倒摆，_probe_grip_z 实测放大滚转）")
    parser.add_argument("--v_y", type=float, default=-0.01)
    parser.add_argument("--y_dist", type=float, default=0.5)
    parser.add_argument("--hold_only", action="store_true")
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--Tp", type=float, default=0.10)
    parser.add_argument("--sim_steps", type=int, default=4,
                        help="acados ERK 每个 shooting interval 的积分步数（推荐2；旧值4）")
    parser.add_argument("--no_csv", action="store_true")
    parser.add_argument("--no_viewer", action="store_true", help="关闭 MuJoCo 可视化窗口")
    parser.add_argument("--sweep_forces", type=str, default=None,
                        help="逗号分隔力扫描（同求解器，不重建）")
    parser.add_argument("--sweep_kf", type=str, default=None,
                        help="逗号分隔 kf_scale 扫描（每值重建一次求解器）")
    parser.add_argument("--sweep_tramp", type=str, default=None,
                        help="逗号分隔 t_ramp 扫描（仅参考 profile，不重建求解器）")
    parser.add_argument("--sweep_thold", type=str, default=None,
                        help="逗号分隔 t_hold 扫描（仅参考 profile，不重建求解器）")
    parser.add_argument("--sweep_fstart", type=str, default=None,
                        help="逗号分隔 f_start 扫描（每值重新收敛轻压位）")
    parser.add_argument("--repeats", type=int, default=1,
                        help="每候选重复次数（同进程 reset_world + 重收敛 + 求解器种子复位）")
    args = parser.parse_args()
    print(f"[command] y_dist={args.y_dist:.6f} m, v_y={args.v_y:.6f} m/s, "
          f"hold_only={args.hold_only}")
    print(f"[output] CSV and Fig. 12 files: {os.path.abspath(OUT_DIR)}")

    sweep_forces = ([float(x) for x in args.sweep_forces.split(",")]
                    if args.sweep_forces else None)
    sweep_kfs = ([float(x) for x in args.sweep_kf.split(",")]
                 if args.sweep_kf else None)
    sweep_tramps = ([float(x) for x in args.sweep_tramp.split(",")]
                    if args.sweep_tramp else None)
    sweep_tholds = ([float(x) for x in args.sweep_thold.split(",")]
                    if args.sweep_thold else None)
    sweep_fstarts = ([float(x) for x in args.sweep_fstart.split(",")]
                     if args.sweep_fstart else None)
    any_sweep = any(x is not None for x in (sweep_forces, sweep_kfs,
                                            sweep_tramps, sweep_tholds, sweep_fstarts))

    scene_mod.GRIP_Z_OFFSET = args.grip_z          # 摆式稳定偏移（场景模块全局常量，供两阶段构建读取）
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

    reset_world(model, data, arms, box_qpos)

    # --- 目标姿态：轻接触起点 + 压入终点（不改几何）---
    base_targets = grasp_targets(arms)

    def press_for(force_n):
        delta = force_n / (2.0 * K_STIFF)
        return {
            a["side"]: base_targets[a["side"]] + np.array([-a["sign"] * delta, 0.0, 0.0])
            for a in arms
        }

    press_light = press_for(args.f_start)      # 安全轻接触（默认 2N，摩擦持重容量 20N > 箱重 8.4N）
    press_full_for = press_for
    solve_end_grip(model, data, arms, press_light, IK_ITERS)

    contact_offsets = nc.pad_contact_offsets(model, arms, 0.1001)
    fk = nc.mujoco_fk_symbols(
        model,
        [{"body_id": a["body_id"], "joint_ids": a["joint_ids"]} for a in arms],
        0.1001,
        contact_offsets=contact_offsets,
    )
    # 按给定 force 重建 rv_ref 用的抓取姿态（solve_end_grip 已放在轻压位）
    def make_rv_ref():
        return {a["side"]: nc.rotvec_of_R(data.xmat[a["body_id"]].reshape(3, 3))
                for a in arms}
    rv_ref = make_rv_ref()
    arm_damping(model, [d for a in arms for d in a["dofs"]], 0.5)

    umax_parts = []
    for a in arms:
        umax_parts.append(model.actuator_ctrlrange[a["act_ids"], 1])
    umax = np.concatenate(umax_parts)

    NMPC_CACHE = {}

    def build_nmpc(kf):
        # CasADi names allow letters, digits, and underscores only.
        model_name = (f"nmpcc_cart_dual_force_bias_v4_contact_kf{float(kf):.9g}"
                      f"_Q{args.Q:g}_P{args.P:g}"
                      f"_ed{args.eta_dev:g}_eu{args.eta_u:g}")
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
            kx=K_STIFF,
            rest_gap=REST_GAP,
            kf_scale=kf,
            vmax=0.05,
            vz_max=0.001,
            sim_method_num_steps=args.sim_steps,
            sim_method_num_stages=4,
        )
        return nc.NMPCCCart(cfg, fk, umin=-umax, umax=umax)

    def run_one(kf, force, tag):
        args.force = force
        args.kf_scale = kf
        # 每 run 复位世界并按 **当前 f_start** 重新收敛轻压位（确定性）
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
        # 求解器按 OCP 参数缓存（kf/Q/sim_steps 相同则复用已编译实例）；
        # 每 run 用 u_seed 冷启动（nmpc._warm=None + 预求解），保证重复运行等价于独立进程冷启动。
        okey = (round(kf, 6), args.Q, args.P, args.eta_dev, args.eta_u,
                args.eta_cop_y,
                args.sim_steps, args.N, args.Tp)
        nmpc = NMPC_CACHE.get(okey)
        if nmpc is None:
            nmpc = build_nmpc(kf)
            NMPC_CACHE[okey] = nmpc
            print(f"NMPCC built: K={K_STIFF:.3f}, rest_gap={REST_GAP:.6f}, "
                  f"N={args.N}, Tp={args.Tp:.3f}, force={force:.1f}N, kf_scale={kf:.3f}, "
                  f"Q={args.Q:.3g}, P={args.P:.3g}, eta_dev={args.eta_dev:.3g}, "
                  f"eta_u={args.eta_u:.3g}, "
                  f"eta_cop_y={args.eta_cop_y:.2f}")
        else:
            print(f"NMPCC cached (same OCP params); force={force:.1f}N, "
                  f"t_ramp={args.t_ramp:.2f}, t_hold={args.t_hold:.2f}, f_start={args.f_start:.1f}")
        nmpc._warm = None
        # 关键：复位 acados 内部求解器内存（含 HPIPM 因子分解热身状态）。
        # 若不复位，上一 run 的 QP 分解状态会以 warm start 形式进入本 run 的第一次 QP，
        # 造成同配置重复运行出现 0.5°~4.3° 的 tilt 分支分歧（初值敏感性的主要来源）。
        try:
            nmpc.solver.reset()
        except Exception as e:
            print(f"[warn] nmpc.solver.reset() failed: {e}")
        # 进一步：关闭 SQP-RTI 的 QP 热身（若支持；hpipm 记忆热身是 run 间串扰的通道）
        try:
            nmpc.solver.options_set("qp_warm_start", 0)
            nmpc.solver.options_set("warm_start_first_qp", 0)
        except Exception as e:
            print(f"[warn] qp_warm_start disable failed: {e}")
        t0 = time.perf_counter()
        log, horizon_log, m = run_case(model, data, arms, ids, args, nmpc, rfn, th_end,
                                       viewer=viewer)
        dt = time.perf_counter() - t0
        print(f"[run {tag}] F={force:.1f} kf={kf:.3f} "
              f"timed_hold={args.t_hold:.2f}s move_start={m['moving_start_s']:.2f}s | "
              f"{'move=DISABLED' if not m['motion_commanded'] else ('move=COMPLETE' if m['move_complete'] else 'move=INCOMPLETE')} "
              f"(after_settle={m['move_complete_after_settle']}, observed={m['move_observed']}, "
              f"hold_dy={m['box_y_hold_max_mm']:.2f}mm, "
              f"travel={m['box_y_move_travel_mm']:.2f}mm, "
              f"initial_end_err={m['box_y_initial_error_mm']:.2f}mm) | "
              f"tilt=({m['tilt_x_max_deg']:.2f},"
              f"{m['tilt_y_max_deg']:.2f},{m['tilt_z_max_deg']:.2f})° | "
              f"z_sag={m['z_sag_mm']:+.2f}mm z_drift={m['z_drift_mm']:.2f}mm | "
              f"F=({m['FL_mean']:.2f},{m['FR_mean']:.2f})N diff={m['F_diff_max_N']:.2f}N | "
              f"ysync={m['y_sync_mean_mm']:.2f}/{m['y_sync_max_mm']:.2f}mm | "
              f"status_bad={m['solver_status_bad_count']} qp={m['qp_iter_mean']:.1f} "
              f"solve={m['solve_ms_mean']:.2f}ms ({dt:.1f}s)")
        if not args.no_csv:
            os.makedirs(OUT_DIR, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(OUT_DIR, f"exp2_latest_{tag}_{stamp}.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["t", "phase", "ramp_s", "moving_active",
                            "boxx", "boxy", "boxz", "q_w", "q_x", "q_y", "q_z",
                            "rvx", "rvy", "rvz",
                            "tipLx", "tipLy", "tipLz", "tipRx", "tipRy", "tipRz",
                            "FLx", "FLy", "FLz", "FLtx", "FLty", "FLtz",
                            "FRx", "FRy", "FRz", "FRtx", "FRty", "FRtz",
                            "BCFx", "BCFy", "BCFz", "BCtx", "BCty", "BCtz",
                            "FL", "FR",
                            "FmeasCtrlLx", "FmeasCtrlRx",
                            "FmodelLx", "FmodelRx", "bFLx", "bFRx",
                            "FanchorLx", "FanchorRx", "FrefLx", "FrefRx",
                            *[f"Fpred_{k}_{side}" for k in ("k0", "mid", "N") for side in ("L", "R")],
                            *[f"Ferr_{k}_{side}" for k in ("k0", "mid", "N") for side in ("L", "R")],
                            *[f"Xerr_{k}_{side}" for k in ("k0", "mid", "N") for side in ("L", "R")],
                            *[f"EpsX_{k}_{side}_{axis}" for k in ("k0", "mid", "N")
                              for side in ("L", "R") for axis in ("x", "y", "z", "rx", "ry", "rz")],
                            *[f"EpsdX_{k}_{side}_{axis}" for k in ("k0", "mid", "N")
                              for side in ("L", "R") for axis in ("x", "y", "z", "rx", "ry", "rz")],
                            *[f"Upred_{k}_{i + 1}" for k in ("k0", "mid", "last") for i in range(nc.NU)],
                            *[f"u{i + 1}" for i in range(nc.NU)],
                            *[f"uclip{i + 1}" for i in range(nc.NU)],
                            *[f"usat{i + 1}" for i in range(nc.NU)],
                            "solve_ms", "solver_status", "qp_iter",
                            "solver_qp_ms", "solver_sim_ms", "solver_lin_ms"])
                for r in log:
                    w.writerow([r["t"], r["phase"], r["ramp_s"], r["moving_active"],
                                *r["box"], *r["quat"], *r["rotvec"],
                                *r["tipL"], *r["tipR"], *r["wL"][0], *r["wL"][1],
                                *r["wR"][0], *r["wR"][1], *r["box_f"], *r["box_t"],
                                r["FL"], r["FR"], *r["force_measured_control"],
                                *r["force_nominal"], *r["force_bias"],
                                *r["force_anchored"], *r["force_ref"],
                                *r["pred_force"].reshape(-1),
                                *r["pred_force_error"].reshape(-1),
                                *r["pred_position_error_x"].reshape(-1),
                                *r["pred_eps_x"].reshape(-1),
                                *r["pred_epsd_x"].reshape(-1),
                                *r["pred_u"].reshape(-1), *r["u_cmd"],
                                *r["u_clip_mask"].astype(int), *r["u_sat_mask"].astype(int),
                                r["solve_ms"], r["solver_status"],
                                r["qp_iter"], r["solver_qp_ms"], r["solver_sim_ms"],
                                r["solver_lin_ms"]])
            print(f"saved {path}")
            horizon_path = os.path.join(OUT_DIR, f"exp2_latest_{tag}_{stamp}_horizon.csv")
            with open(horizon_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "t", "phase", "node", "tau_node",
                    "FrefLx", "FrefRx", "FpredLx", "FpredRx",
                    "FerrLx", "FerrRx", "XerrLx", "XerrRx",
                    "EpsXL", "EpsXR", "EpsdXL", "EpsdXR",
                    *[f"Upred{i + 1}" for i in range(nc.NU)],
                    *[f"Uref{i + 1}" for i in range(nc.NU)],
                    *[f"Udev{i + 1}" for i in range(nc.NU)],
                ])
                node_dt = args.Tp / args.N
                for h in horizon_log:
                    for node in range(args.N + 1):
                        u_pred = (h["u"][node] if node < args.N
                                  else np.full(nc.NU, np.nan))
                        u_dev = u_pred - h["u_ref"]
                        w.writerow([
                            h["t"], h["phase"], node, node * node_dt,
                            *h["force_ref"], *h["force"][node],
                            *h["force_error"][node], *h["position_error_x"][node],
                            *h["eps_x"][node], *h["epsd_x"][node],
                            *u_pred, *h["u_ref"], *u_dev,
                        ])
            print(f"saved {horizon_path}")
            # 每次保存 CSV 后自动生成 Fig. 12 的 PNG 和 SVG；绘图失败不影响仿真结果。
            plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "plot_exp2_fig12.py")
            plot_prefix = os.path.join(OUT_DIR, f"fig12_{tag}_{stamp}")
            try:
                subprocess.run(
                    [sys.executable, plot_script, path, "--out", plot_prefix],
                    check=True,
                )
            except Exception as exc:
                print(f"[warn] automatic Fig. 12 generation failed: {exc}")
        return m

    if any_sweep:
        kfs = sweep_kfs or [args.kf_scale]
        forces = sweep_forces or [args.force]
        tramps = sweep_tramps or [args.t_ramp]
        tholds = sweep_tholds or [args.t_hold]
        fstarts = sweep_fstarts or [args.f_start]
        rows = []
        for kf in kfs:
            for force in forces:
                for tramp in tramps:
                    for thold in tholds:
                        for fstart in fstarts:
                            args.t_ramp = tramp
                            args.t_hold, args.f_start = thold, fstart
                            for rep in range(args.repeats):
                                tag = (f"f{force:g}_kf{kf:g}"
                                       f"_tr{tramp:g}_th{thold:g}_fs{fstart:g}_r{rep}")
                                rows.append(run_one(kf, force, tag))
        os.makedirs(OUT_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(OUT_DIR, f"sweep_hold_{stamp}.csv")
        cols = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\n=== sweep summary: {path} ===")
        print("rep | force | kf | tramp | thold | fstart | "
              "tiltx_hold | tilty_hold | tiltz_hold | z_sag | z_drift_hold | "
              "FL | FR | Fdiff_hold | ysync_hold | solve_ms")
        for i, m in enumerate(rows):
            print(f"{i % args.repeats:3d} | {m['force']:5.1f} | {m['kf_scale']:.2f} | "
                  f"{m['t_ramp']:.2f} | {m['t_hold']:.2f} | "
                  f"{m['f_start']:.1f} | "
                  f"{m['tilt_x_hold_max_deg']:5.2f} | {m['tilt_y_hold_max_deg']:5.2f} | "
                  f"{m['tilt_z_hold_max_deg']:5.2f} | {m['z_sag_mm']:6.2f} | "
                  f"{m['z_drift_hold_mm']:5.2f} | {m['FL_mean']:5.2f} | {m['FR_mean']:5.2f} | "
                  f"{m['F_diff_hold_max_N']:5.2f} | {m['y_sync_hold_max_mm']:6.2f} | "
                  f"{m['solve_ms_mean']:5.2f}")
        return

    m = run_one(args.kf_scale, args.force, "single")
    print("--- latest Experiment II (summary) ---")
    for k, v in m.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()