#!/usr/bin/env python3
"""exp2_nmpcc_latest_grasp_pose.py —— 空中抓取姿态场景的实验二 NMPCC（Holding 偏转修复版）

直接使用 grasp_pose_scene.xml（不改几何/不重标定接触）：
  - 双臂、箱 0.24 x 0.40 x 0.20 m，中心 (0.5, 0, 0.72)；K_stiff=3575 N/m、rest_gap=0.248 m；
  - x 轴 eta_f=0.8、y 轴 eta_cop=0.8、z/姿态跟踪；acados 每 6ms 求解一次。

针对 Holding 偏转的修复（控制律不动：nmpcc_cart.py 仅恢复过一次并发工作流误写损坏并合并其求解器调参改动，见交接说明）：
  1. 启动几何/FK 对账与接触测量等检测类函数已移到 exp2_diagnostics.py
     （可独立运行，见下方"运行"）；主程序只保留仿真与作图；
  2. 日志扩展：三维世界系接触力 + 接触力矩（每块垫片，含对箱体质心的净力偶）、
     箱体完整姿态（quaternion + rotvec），逐拍落 CSV；
  3. t=0 预压力：初始 IK 使用 f_start=10 N，随后在 0.5 s 内平滑到目标力；
  4. 固定时序：ramp 后 Holding 3 s，随后无条件进入 Moving，不使用稳定 gate；
  5. 纯 NMPCC：t=0 之后每个执行器命令都直接来自 NMPCC；无夹持力外环、
     笛卡尔保持外环、启动保持或 handoff 混合；f_meas 仅作为 OCP 模型参数；
     首个预测步包含以上一拍 NMPCC 解为中心的输入变化率硬约束；
  6. CLI：--force --kf_scale --Q --P --eta_dev --eta_u --t_ramp --t_hold
     --sweep_forces --sweep_kf；保留 --no_viewer --no_csv。

运行：
  启动几何/FK 对账（检测工具，独立运行，不占用主程序）：
    D:\\acados_env\\python.exe -u exp2_diagnostics.py --scene grasp_pose
  仅保持：
    D:\\acados_env\\python.exe -u exp2_nmpcc_latest_grasp_pose.py --hold_only --no_viewer
  力/增益扫描（同进程复用求解器；force 是参数不触发重建，kf_scale 每值重建一次）：
    D:\\acados_env\\python.exe -u exp2_nmpcc_latest_grasp_pose.py --hold_only --no_viewer --sweep_forces 10,12,15 --sweep_kf 0.05,0.10,0.20
  定时搬运（0.5 s ramp + 3 s Holding 后进入 Moving）：
    D:\\acados_env\\python.exe -u exp2_nmpcc_latest_grasp_pose.py --y_dist 0.08
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

from pathlib import Path

import nmpcc_cart_grasp_pose as nc
from exp2_diagnostics import (
    box_attitude, box_net_wrench, contact_forces,
    contact_wrench_on_arm, pad_box_wrench_3d,
)
from ik import ARM_CONFIGS, BASE_DIR, clamp_qpos, get_ee_tip, name_to_id, solve_pose_ik

# 直接使用 grasp_pose_scene.xml，不修改原始场景文件。
BOX_HALF = np.array([0.12, 0.20, 0.10], dtype=np.float64)
BOX_CENTER = np.array([0.5, 0.0, 0.72], dtype=np.float64)
GRIP_CLEARANCE = 0.004
IK_ITERS = 1500
SCENE_XML = BASE_DIR / "grasp_pose_scene.xml"
K_STIFF = 3575.0
REST_GAP = 2 * (BOX_HALF[0] + GRIP_CLEARANCE)
DT = 0.002
CTRL_EVERY = 3
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "exp2_latest_grasp_pose")
HOLD_BOX_Y_TOL = 0.002


def build_scene() -> Path:
    if not SCENE_XML.is_file():
        raise FileNotFoundError(f"场景文件不存在: {SCENE_XML}")
    return SCENE_XML


def make_arms(model):
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
    center = BOX_CENTER if center is None else np.asarray(center)
    return {a["side"]: np.array(
        [center[0] + a["sign"] * (BOX_HALF[0] + GRIP_CLEARANCE), center[1], center[2]])
        for a in arms}


def solve_end_grip(model, data, arms, targets, iters=IK_ITERS):
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
    """At t=0 only, calibrate IK penetration against static contact force."""
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
    pads = {a["pad_id"] for a in arms}
    for gid in range(model.ngeom):
        if gid == box_geom_id or gid in pads:
            continue
        if (model.geom(gid).name or "").startswith("openarmx_"):
            model.geom_contype[gid] = 0
            model.geom_conaffinity[gid] = 0


def soften_contacts(model, box_geom_id, pad_ids):
    for gid in [box_geom_id, *pad_ids]:
        model.geom_solref[gid] = [0.02, 1.0]
        model.geom_solimp[gid][:3] = [0.999, 0.999, 0.001]



def arm_damping(model, dofs, value=0.5):
    for d in dofs:
        model.dof_damping[d] = value


def smoothstep(s: float) -> float:
    s = min(1.0, max(0.0, float(s)))
    return 3.0 * s * s - 2.0 * s * s * s


def inject_motors(runtime_xml: Path) -> None:
    """Ensure the IK runtime XML contains one torque motor per arm joint."""
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


# ======================================================================
# 阶段参考：平滑预紧 + holding -> moving（参数进 OCP p，控制器恒定）
# ======================================================================
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
    global BOX_CENTER
    for a in arms:
        data.qpos[a["qpos"]] = a["q_rest"]
        clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    tips_mid = np.mean([get_ee_tip(data, a["body_id"]) for a in arms], axis=0)
    BOX_CENTER = tips_mid + np.array([0.0, 0.0, 0.10])
    data.qpos[box_qpos:box_qpos + 3] = BOX_CENTER
    data.qpos[box_qpos + 3:box_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
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

    def measured_force():
        """读取当前两侧垫片接触的世界系 6D 力/力矩，供 NMPCC 每拍更新。"""
        measured = pad_box_wrench_3d(model, data, arms, box_geom)
        return np.concatenate([
            measured["left"][0], measured["left"][1],
            measured["right"][0], measured["right"][1],
        ])

    def reference_for(t):
        refs, vrefs, fmag, phase, ramp_s = ref_fn(t)
        return refs, vrefs, fmag, phase, ramp_s

    def build_p(t):
        refs, vrefs, fmag, phase, _ = reference_for(t)
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

    torque_lo = np.concatenate([model.actuator_ctrlrange[a["act_ids"], 0]
                                for a in arms])
    torque_hi = np.concatenate([model.actuator_ctrlrange[a["act_ids"], 1]
                                for a in arms])

    def nominal_input():
        """Input center for NMPCC, matching its split contact dynamics."""
        out = []
        for a in arms:
            fJ, _ = contact_wrench_on_arm(model, data, a, box_geom)
            out.append(data.qfrc_bias[a["dofs"]] - fJ)
        return np.concatenate(out)

    # t=0 may contain an IK preload, but every executed torque from the first
    # simulation step onward is the direct NMPCC result.  u_seed and u_ref are
    # numerical OCP quantities; neither is sent directly to the actuators.
    p0, _, _ = build_p(0.0)
    u_seed = np.clip(nominal_input(), torque_lo, torque_hi)
    nmpc._warm = None
    u_last = nmpc.solve(joint_state(), p0, u_ref=u_seed,
                        f_meas=measured_force(), u_seed=u_seed)

    initial_box_y = float(data.xpos[box_body][1])
    motion_commanded = (not args.hold_only and abs(args.y_dist) > 1e-9
                        and abs(args.v_y) > 1e-9)
    print(f"[pure NMPCC] active from t=0; ramp={args.t_ramp:.3f}s, "
          f"holding={args.t_hold:.3f}s, moving starts at t={t_hold_end:.3f}s"
          if motion_commanded
          else "[pure NMPCC] active from t=0; translation disabled")

    log = []
    phase = "ramp"
    u_cmd = np.clip(np.asarray(u_last, dtype=float), torque_lo, torque_hi)
    u_clip_mask = np.abs(u_cmd - np.asarray(u_last, dtype=float)) > 1e-9
    u_sat_mask = u_clip_mask | np.isclose(u_cmd, torque_lo, atol=1e-6) | np.isclose(
        u_cmd, torque_hi, atol=1e-6)
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
            p, current_f_ref, phase = build_p(t)
            u_ref = np.clip(nominal_input(), torque_lo, torque_hi)
            u_last = nmpc.solve(
                joint_state(), p,
                u_ref=u_ref,
                f_meas=measured_force(),
                u_seed=u_seed,
                u_prev=u_cmd,
                du_max=du_max)
            u_raw = np.asarray(u_last, dtype=float)
            u_cmd = np.clip(u_raw, torque_lo, torque_hi)
            u_clip_mask = np.abs(u_cmd - u_raw) > 1e-9
            u_sat_mask = u_clip_mask | np.isclose(u_cmd, torque_lo, atol=1e-6) | np.isclose(
                u_cmd, torque_hi, atol=1e-6)
            solve_ms = (time.perf_counter() - t0) * 1000.0
            if step == 0:
                print(f"[u0] {np.array2string(u_cmd, precision=12, floatmode='unique')}")
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
            print(f"t={t + DT:.3f}s  左臂对箱 F={-w['left'][0]} N | 力矩={-w['left'][1]} N·m")
            fb, tb = box_net_wrench(model, data, arms, box_geom,
                                    data.xpos[box_body].copy())
            rv, quat = box_attitude(data, box_qpos)
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
                "f_ref": current_f_ref,
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
    return log, metrics


def summarize(log, args, initial_box_y) -> dict:
    tt = np.array([r["t"] for r in log])
    box = np.array([r["box"] for r in log])
    rv = np.array([r["rotvec"] for r in log])
    tip_l = np.array([r["tipL"] for r in log])
    tip_r = np.array([r["tipR"] for r in log])
    fl = np.array([r["FL"] for r in log]); fr = np.array([r["FR"] for r in log])
    f_ref = np.array([r["f_ref"] for r in log])
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
    box_y_move_travel_mm = (float(box_y_move_delta_mm[-1])
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
    box_y_final_error_mm = abs((box[-1, 1] - initial_box_y) - commanded_distance) * 1000.0
    # The plant is still stepped for a short settling tail after the commanded
    # translation.  Evaluate the endpoint at the end of the commanded motion,
    # rather than at the final tail sample.
    move_end_t = args.t_ramp + args.t_hold + move_duration
    endpoint_idx = int(np.argmin(np.abs(tt - move_end_t)))
    box_y_endpoint_error_mm = abs(
        (box[endpoint_idx, 1] - initial_box_y) - commanded_distance
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
        "control_mode": "pure_nmpcc_from_t0",
        "initial_FL_N": float(fl[0]), "initial_FR_N": float(fr[0]),
        "initial_f_ref_N": float(f_ref[0]),
        "FL_max_step_N": max_or_nan(np.abs(np.diff(fl))),
        "FR_max_step_N": max_or_nan(np.abs(np.diff(fr))),
        "contact_loss_samples": int(np.sum((fl < 0.1) | (fr < 0.1))),
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
    parser.add_argument("--force", type=float, default=40, help="稳定夹持力参考 (N)")
    parser.add_argument("--kf_scale", type=float, default=0.8, help="力误差增益 (nmpcc_cart 配置)")
    parser.add_argument("--Q", type=float, default=1.0, help="论文式(31)阶段损失外部标量 Q")
    parser.add_argument("--P", type=float, default=3.0, help="论文式(32)终端损失外部标量 P")
    parser.add_argument("--eta_dev", type=float, default=6000.0, help="误差导数相对权重 eta_dev")
    parser.add_argument("--eta_u", type=float, default=0.20, help="输入相对权重 eta_u")
    parser.add_argument("--eta_cop_y", type=float, default=0.8,
                        help="论文合作映射 eta_cop 的 y 分量；0=绝对 y 跟踪，0.8=同步优先")
    parser.add_argument("--t_ramp", type=float, default=0.5, help="10N→目标力的平滑时间 (s)")
    parser.add_argument("--t_hold", type=float, default=0.1, help="固定 Holding 时间，结束后直接 Moving (s)")
    parser.add_argument("--f_start", type=float, default=10.0,
                        help="启动夹持力参考 (N)，同时用于初始 IK 压紧和控制器首拍")
    parser.add_argument("--v_y", type=float, default=-0.01,
                        help="世界坐标 -y 前进速度 (m/s)，正值表示沿 -y 前进")
    parser.add_argument("--y_dist", type=float, default=0.05,
                        help="沿世界 -y 的前进距离 (m)")
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
          f"Q={args.Q:g}, P={args.P:g}, eta_dev={args.eta_dev:g}, "
          f"eta_u={args.eta_u:g}, hold_only={args.hold_only}")
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

    scene_xml = build_scene()
    inject_motors(SCENE_XML.parent / "openarmx_ik_runtime.xml")
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

    fk = nc.mujoco_fk_symbols(
        model,
        [{"body_id": a["body_id"], "joint_ids": a["joint_ids"]} for a in arms],
        0.1001,
    )
    # 按给定 force 重建 rv_ref 用的抓取姿态（solve_end_grip 已放在轻压位）
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
        # CasADi names allow letters, digits, and underscores only.
        model_name = (f"nmpcc_cart_grasp_pose_v3_kf{float(kf):.9g}"
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
        # 求解器按 OCP 参数缓存（kf/Q/sim_steps 相同则复用已编译实例）；
        # 每 run 用 u_seed 冷启动（nmpc._warm=None + 预求解），保证重复运行等价于独立进程冷启动。
        okey = (round(kf, 6), args.Q, args.P, args.eta_dev, args.eta_u,
                args.eta_cop_y, args.sim_steps, args.N, args.Tp)
        nmpc = NMPC_CACHE.get(okey)
        if nmpc is None:
            nmpc = build_nmpc(kf)
            NMPC_CACHE[okey] = nmpc
            print(f"NMPCC built: K={K_STIFF:.3f}, rest_gap={REST_GAP:.6f}, "
                  f"N={args.N}, Tp={args.Tp:.3f}, force={force:.1f}N, kf_scale={kf:.3f}, "
                  f"Q={args.Q:g}, P={args.P:g}, eta_dev={args.eta_dev:g}, eta_u={args.eta_u:g}, "
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
        log, m = run_case(model, data, arms, ids, args, nmpc, rfn, th_end,
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
              f"F=({m['FL_mean']:.2f},{m['FR_mean']:.2f})N diff={m['F_diff_max_N']:.2f}N "
              f"step=({m['FL_max_step_N']:.2f},{m['FR_max_step_N']:.2f})N "
              f"loss={m['contact_loss_samples']} | "
              f"ysync={m['y_sync_mean_mm']:.2f}/{m['y_sync_max_mm']:.2f}mm | "
              f"status_bad={m['solver_status_bad_count']} qp={m['qp_iter_mean']:.1f} "
              f"solve={m['solve_ms_mean']:.2f}ms ({dt:.1f}s)")
        if not args.no_csv:
            os.makedirs(OUT_DIR, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(OUT_DIR, f"exp2_latest_{tag}_{stamp}.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["t", "phase", "ramp_s", "moving_active", "f_ref",
                            "boxx", "boxy", "boxz", "q_w", "q_x", "q_y", "q_z",
                            "rvx", "rvy", "rvz",
                            "tipLx", "tipLy", "tipLz", "tipRx", "tipRy", "tipRz",
                            "FLx", "FLy", "FLz", "FLtx", "FLty", "FLtz",
                            "FRx", "FRy", "FRz", "FRtx", "FRty", "FRtz",
                            "BCFx", "BCFy", "BCFz", "BCtx", "BCty", "BCtz",
                            "FL", "FR", *[f"u{i + 1}" for i in range(nc.NU)],
                            *[f"uclip{i + 1}" for i in range(nc.NU)],
                            *[f"usat{i + 1}" for i in range(nc.NU)],
                            "solve_ms", "solver_status", "qp_iter",
                            "solver_qp_ms", "solver_sim_ms", "solver_lin_ms"])
                for r in log:
                    w.writerow([r["t"], r["phase"], r["ramp_s"], r["moving_active"],
                                r["f_ref"],
                                *r["box"], *r["quat"], *r["rotvec"],
                                *r["tipL"], *r["tipR"], *r["wL"][0], *r["wL"][1],
                                *r["wR"][0], *r["wR"][1], *r["box_f"], *r["box_t"],
                                r["FL"], r["FR"], *r["u_cmd"],
                                *r["u_clip_mask"].astype(int), *r["u_sat_mask"].astype(int),
                                r["solve_ms"], r["solver_status"],
                                r["qp_iter"], r["solver_qp_ms"], r["solver_sim_ms"],
                                r["solver_lin_ms"]])
            print(f"saved {path}")
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