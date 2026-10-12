#!/usr/bin/env python3
"""calib_grasp_pose_contact.py —— grasp_pose 变体的接触刚度标定

与 calib_exp2_contact.py 同法：箱固定在设计抓取位 (0.5, 0, 0.72)，
对称 IK 把垫片压入 delta（0..10mm），读左右法向接触力，
拟合 F_side = k_side * delta + b；NMPCC 串联等效刚度 K = k_side / 2。
只做几何/接触标定，不运行控制器。
"""
from __future__ import annotations

# 支持从工程外直接调用标定脚本。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from experiments.scenes import handheld as gp
from ik import clamp_qpos, get_ee_tip

DELTAS = np.arange(0.0, 0.0101, 0.0005)  # 0..10 mm


def contact_forces(model, data, arms, box_geom_id):
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


def main():
    scene_xml = gp.build_scene()
    gp.inject_motors(gp.SCENE_XML.parent / "openarmx_ik_runtime.xml")
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    arms = gp.make_arms(model)
    for a in arms:
        a["act_ids"] = [gp.name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                      f"{model.joint(j).name}_motor")
                        for j in a["joint_ids"]]
    box_body = gp.name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_geom = gp.name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    box_joint = gp.name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")
    box_qpos = int(model.jnt_qposadr[box_joint])
    gp.disable_robot_collisions(model, arms, box_geom)
    gp.soften_contacts(model, box_geom, [a["pad_id"] for a in arms])
    gp.arm_damping(model, [d for a in arms for d in a["dofs"]], 0.5)

    # 实际运行位（reset_world 同款）：箱 = 休息位 tips_mid + 0.10
    for a in arms:
        data.qpos[a["qpos"]] = a["q_rest"]
        clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    tips_mid = np.mean([get_ee_tip(data, a["body_id"]) for a in arms], axis=0)
    center = tips_mid + np.array([0.0, 0.0, 0.10])
    data.qpos[box_qpos:box_qpos + 3] = center
    data.qpos[box_qpos + 3:box_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    print(f"实际运行位 box={np.round(center, 3)}  tipL={np.round(get_ee_tip(data, arms[0]['body_id']), 3)} "
          f"tipR={np.round(get_ee_tip(data, arms[1]['body_id']), 3)}")

    base_targets = gp.grasp_targets(arms, center)

    rows = []
    for delta in DELTAS:
        targets = {
            a["side"]: base_targets[a["side"]] + np.array([-a["sign"] * delta, 0.0, 0.0])
            for a in arms
        }
        gp.solve_end_grip(model, data, arms, targets, gp.IK_ITERS)
        fl = contact_forces(model, data, arms, box_geom)
        tipL = get_ee_tip(data, arms[0]["body_id"])
        tipR = get_ee_tip(data, arms[1]["body_id"])
        # IK 跟踪误差（x 方向，向箱压入的目标 vs 实际）
        err = ((tipL[0] - targets["left"][0]), (tipR[0] - targets["right"][0]))
        rows.append((delta, fl["left"], fl["right"], err))
        if abs(delta * 1000 - 5.0) < 1e-9:
            print(f"  delta={delta*1000:.1f}mm  FL={fl['left']:.2f} FR={fl['right']:.2f} "
                  f"xerr=({err[0]*1000:+.1f},{err[1]*1000:+.1f})mm")

    d = np.array([r[0] for r in rows])
    fL = np.array([r[1] for r in rows])
    fR = np.array([r[2] for r in rows])
    # 线性区拟合（取力 > 0 的部分）
    sel = (fL > 0.5) & (fR > 0.5)
    kL, bL = np.polyfit(d[sel], fL[sel], 1)
    kR, bR = np.polyfit(d[sel], fR[sel], 1)
    print(f"线性区拟合: F_left = {kL:.1f}*delta + {bL:.2f}   "
          f"F_right = {kR:.1f}*delta + {bR:.2f}")
    k_side = 0.5 * (kL + kR)
    k_stiff = 0.5 * k_side
    print(f"k_side = {k_side:.1f} N/m   K_stiff(串联) = {k_stiff:.1f} N/m  (当前代码 K_STIFF={gp.K_STIFF})")
    # 零力间隙：从拟合截距反推
    delta0 = -0.5 * (bL / kL + bR / kR)
    print(f"零力间隙估计 delta0 = {delta0*1000:.2f} mm  -> 对应 rest_gap = "
          f"{2 * (gp.BOX_HALF[0] + gp.GRIP_CLEARANCE) - 2 * delta0:.5f} m "
          f"(当前 rest_gap={gp.REST_GAP})")


if __name__ == "__main__":
    main()
