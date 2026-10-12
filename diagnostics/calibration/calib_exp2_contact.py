#!/usr/bin/env python3
"""calib_exp2_contact.py —— 最新地面双臂细长箱场景的接触标定

标定目标：
  1. 在最新 exp2_slender_scene 场景中，把箱体固定在初始位置；
  2. 用对称 IK 将左右垫片同时向箱内压入 delta；
  3. 读取 MuJoCo 的左右法向接触力；
  4. 拟合 F_side = k_side * delta + b，得到单侧有效刚度 k_side；
  5. NMPCC 的串联等效刚度 K_stiff = k_side / 2，rest_gap 由零力几何间距确定。

只做几何/接触标定，不运行 NMPCC，不修改控制器。
运行：D:\\acados_env\\python.exe -u calib_exp2_contact.py
"""
from __future__ import annotations

# 支持从工程外直接调用标定脚本。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv
from pathlib import Path

import mujoco
import numpy as np

from nmpcc import kinematics as nc

from exp2_slender_scene import (
    BOX_CENTER,
    GRIP_CLEARANCE,
    IK_ITERS,
    SCENE_XML,
    SLENDER_HALF,
    build_scene,
    make_arms,
    grasp_targets,
    solve_end_grip,
)
from ik import clamp_qpos, get_ee_tip, name_to_id, solve_pose_ik

OUT = Path("results/exp2_contact_calibration.csv")

# 压入量范围：先小范围线性区，避免把垫片或箱体压入过深
DELTAS = np.arange(0.0, 0.0101, 0.0005)  # 0..10 mm


def contact_forces(model, data, arms, box_geom_id: int) -> dict[str, float]:
    """提取每个垫片与箱体接触的法向力绝对值。"""
    out = {a["side"]: 0.0 for a in arms}
    force = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        for a in arms:
            pair = ((c.geom1 == a["pad_id"] and c.geom2 == box_geom_id) or
                    (c.geom2 == a["pad_id"] and c.geom1 == box_geom_id))
            if pair:
                mujoco.mj_contactForce(model, data, i, force)
                out[a["side"]] += abs(float(force[0]))
    return out


def set_box_fixed(data, box_qpos: int, box_pos: np.ndarray) -> None:
    data.qpos[box_qpos:box_qpos + 3] = box_pos
    data.qpos[box_qpos + 3:box_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0


def solve_press_pose(model, data, arms, delta: float, base_targets: dict[str, np.ndarray]) -> None:
    """对称压入：left 向 +x，right 向 -x。"""
    for a in arms:
        target = base_targets[a["side"]].copy()
        target[0] += -a["sign"] * delta
        for _ in range(IK_ITERS):
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            mujoco.mj_forward(model, data)
            inward = np.array([-a["sign"], 0.0, 0.0])
            dq, _ = solve_pose_ik(
                model, data, a["body_id"], target,
                inward, get_ee_tip(data, a["body_id"]),
                a["dofs"], a["qpos"], a["q_rest"],
            )
            data.qpos[a["qpos"]] += dq
            clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)


def main() -> None:
    scene = build_scene()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    arms = make_arms(model)
    box_body = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_geom = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    box_joint = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")
    box_qpos = int(model.jnt_qposadr[box_joint])

    # 关节回到最新场景休息位，箱体固定在 BOX_CENTER
    for a in arms:
        data.qpos[a["qpos"]] = a["q_rest"].copy()
        clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    set_box_fixed(data, box_qpos, BOX_CENTER)
    mujoco.mj_forward(model, data)

    base_targets = grasp_targets(arms)
    solve_end_grip(model, data, arms, base_targets)
    # 零压入距离按实际 pad 接触面中心记录；tip 距离只作为兼容对照。
    contact_offsets = nc.pad_contact_offsets(model, arms, 0.1001)
    tip_distance = abs(
        get_ee_tip(data, arms[0]["body_id"])[0] -
        get_ee_tip(data, arms[1]["body_id"])[0]
    )
    q_grasp = {a["side"]: data.qpos[a["qpos"]].copy() for a in arms}
    qL, qR = (data.qpos[a["qpos"]].copy() for a in arms)
    contact0 = np.asarray(
        nc.mujoco_fk_symbols(
            model,
            [{"body_id": a["body_id"], "joint_ids": a["joint_ids"]} for a in arms],
            0.1001,
            contact_offsets=contact_offsets,
        )["contact_tips_fn"](qL, qR)
    ).ravel()
    contact_distance = abs(float(contact0[0] - contact0[3]))

    rows = []
    for delta in DELTAS:
        for a in arms:
            data.qpos[a["qpos"]] = q_grasp[a["side"]].copy()
        set_box_fixed(data, box_qpos, BOX_CENTER)
        solve_press_pose(model, data, arms, float(delta), base_targets)
        force = contact_forces(model, data, arms, box_geom)
        rows.append({
            "delta_m": float(delta),
            "delta_mm": float(delta * 1000.0),
            "force_left_N": force["left"],
            "force_right_N": force["right"],
            "force_mean_N": 0.5 * (force["left"] + force["right"]),
            "force_diff_N": force["left"] - force["right"],
        })
        print(
            f"delta={delta*1000:5.1f} mm | "
            f"FL={force['left']:8.3f} N FR={force['right']:8.3f} N "
            f"diff={force['left']-force['right']:+7.3f} N"
        )

    # 只用接触力已经明显高于噪声、且未到大压入端的线性段拟合
    fit_rows = [r for r in rows if 0.001 <= r["delta_m"] <= 0.006]
    x = np.array([r["delta_m"] for r in fit_rows])
    y = np.array([r["force_mean_N"] for r in fit_rows])
    k_side, intercept = np.polyfit(x, y, 1)
    K_stiff = 0.5 * k_side

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== 最新场景接触标定 ===")
    print(f"场景: {SCENE_XML}")
    print(f"箱中心: {BOX_CENTER} | 箱端面几何间距: {2*SLENDER_HALF[0]:.4f} m")
    print(f"tip 中心间距（兼容对照）: {tip_distance:.6f} m")
    print(f"pad 接触面零压入中心间距: {contact_distance:.6f} m")
    print(f"单侧线性拟合: F_side = {k_side:.3f} * delta + {intercept:.3f} N")
    print(f"串联等效刚度 K_stiff = k_side/2 = {K_stiff:.3f} N/m")
    print(f"左右力差 RMS = {np.sqrt(np.mean([r['force_diff_N']**2 for r in rows])):.4f} N")
    print(f"结果已保存: {OUT}")


if __name__ == "__main__":
    main()
