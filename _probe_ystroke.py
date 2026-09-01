#!/usr/bin/env python3
"""_probe_ystroke.py —— y 行程与 z 耦合实测（当前布局 + 基座 yaw 变体）

任务：箱沿 y 以 -10 mm/s 匀速搬运，z 应保持恒定（论文 z 不动）。
实测：从抓取姿态出发，目标点沿 y 双向扫描（warm-start IK 模拟连续运动），
报告每个布局的：
  - 可用 y 行程（端面误差 <2mm 且关节余量 >0.05 rad 的最大 |dy|）
  - 行程内的 z 耦合（tip z 最大偏差）和 x 漂移（夹持面距变化 = 影响夹持力）
布局：base_x=0.55, z_mount=0.15, 镜像侧装 ±90°；变体 = 基座 yaw ∈ {-15°, 0, +15°}。
"""
from __future__ import annotations

import numpy as np
import mujoco

from ik import ARM_CONFIGS, clamp_qpos, get_ee_tip, name_to_id, solve_pose_ik
from grasp_pose_static import GRIP_CLEARANCE
from floor_mount import generate_floor_runtime_xml

BOX_L = 0.50
BOX_Y = -0.13
ERR_MAX = 0.002          # 端面误差上限 (m)
MARGIN_MIN = 0.05        # 关节余量下限 (rad)
DY_STEP = 0.02           # y 扫描步长 (m)
DY_MAX = 0.25            # 扫描上限 (m)——找出真实行程极限


def build(yaw_deg: float, z_mount: float = 0.15, base_x: float = 0.55):
    xml = generate_floor_runtime_xml(
        base_x=base_x, z_mount=z_mount,
        mount_roll={"left": np.pi / 2, "right": -np.pi / 2},
        mount_yaw=np.radians(yaw_deg))
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    arms = []
    for cfg in ARM_CONFIGS:
        body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["jacobi_body"])
        joint_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg["joint_names"]]
        arms.append({
            "side": cfg["side"], "sign": -1.0 if cfg["side"] == "left" else 1.0,
            "q_rest": cfg["q_rest"], "body_id": body_id, "joint_ids": joint_ids,
            "dofs": [int(model.jnt_dofadr[j]) for j in joint_ids],
            "qpos": [int(model.jnt_qposadr[j]) for j in joint_ids],
        })
    return model, data, arms


def ik_to(model, data, a, target, iters):
    inward = np.array([-a["sign"], 0.0, 0.0])
    for _ in range(iters):
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        dq, _ = solve_pose_ik(model, data, a["body_id"], target, inward,
                              get_ee_tip(data, a["body_id"]),
                              a["dofs"], a["qpos"], a["q_rest"])
        data.qpos[a["qpos"]] += dq
        clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    mujoco.mj_forward(model, data)


def margin_of(model, data, a):
    q = data.qpos[a["qpos"]]
    lo = model.jnt_range[a["joint_ids"], 0]
    hi = model.jnt_range[a["joint_ids"], 1]
    return float(np.min(np.minimum(q - lo, hi - q)))


def sweep(yaw_deg: float, z_mount: float = 0.15, box_z: float = 0.20, base_x: float = 0.55):
    model, data, arms = build(yaw_deg, z_mount, base_x)
    # 初始抓取（冷启动 1500 次）
    tgt0 = {a["side"]: np.array([a["sign"] * (BOX_L / 2 + GRIP_CLEARANCE), BOX_Y, box_z])
            for a in arms}
    q_sol = {}
    for a in arms:
        ik_to(model, data, a, tgt0[a["side"]], 1500)
        q_sol[a["side"]] = data.qpos[a["qpos"]].copy()
    err0 = {a["side"]: float(np.linalg.norm(get_ee_tip(data, a["body_id"]) - tgt0[a["side"]]))
            for a in arms}

    out = {}
    for direction, dsign in (("dy+", +1.0), ("dy-", -1.0)):
        recs = []
        dy = DY_STEP
        while dy <= DY_MAX:
            ok = True
            for a in arms:
                data.qpos[a["qpos"]] = q_sol[a["side"]].copy()   # warm start（上一可行点）
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                tgt = tgt0[a["side"]].copy()
                tgt[1] += dsign * dy
                ik_to(model, data, a, tgt, 400)
                tip = get_ee_tip(data, a["body_id"])
                err = float(np.linalg.norm(tip - tgt))
                mrg = margin_of(model, data, a)
                if err > ERR_MAX or mrg < MARGIN_MIN:
                    ok = False
                q_sol[a["side"]] = data.qpos[a["qpos"]].copy()
                recs.append({"arm": a["side"], "dy": dsign * dy, "err": err, "margin": mrg,
                             "dz": abs(tip[2] - box_z), "dx": abs(tip[0] - tgt0[a["side"]][0])})
            if not ok:
                break
            dy += DY_STEP
        good = [r for r in recs if r["err"] <= ERR_MAX and r["margin"] >= MARGIN_MIN]
        if good:
            out[direction] = {
                "max_abs_dy": max(abs(r["dy"]) for r in good),
                "max_dz": max(r["dz"] for r in good),
                "max_dx": max(r["dx"] for r in good),
                "min_margin": min(r["margin"] for r in good),
            }
        else:
            out[direction] = None
    print(f"[bx={base_x:.2f} zm={z_mount:.2f} bz={box_z:.2f}] 抓取误差 L={err0['left']*1e3:.2f} R={err0['right']*1e3:.2f} mm")
    for direction in ("dy+", "dy-"):
        r = out[direction]
        if r:
            print(f"    {direction}: 行程 {r['max_abs_dy']*1e3:.0f} mm | "
                  f"z 耦合 {r['max_dz']*1e3:.2f} mm | x 漂移 {r['max_dx']*1e3:.2f} mm | "
                  f"最小余量 {r['min_margin']:.3f} rad")
        else:
            print(f"    {direction}: 无可行步长")


def main() -> None:
    print("=== y 行程 / z 耦合扫描（warm-start IK，任务连续运动模拟）===")
    print("--- 低基座 × 宽间距 × 箱高 联合扫描（yaw=+15°）---")
    for z_mount in (0.06, 0.08, 0.10):
        for base_x in (0.55, 0.60, 0.65, 0.70):
            for dz_off in (0.0, 0.05):
                sweep(15.0, z_mount, z_mount + dz_off, base_x)


if __name__ == "__main__":
    main()