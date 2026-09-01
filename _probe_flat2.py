#!/usr/bin/env python3
"""_probe_flat2.py —— 全平布局细化：围绕最优点 (zm=0.20, pitch=-60°, bz=0.25, by=-0.13) 细化，
并打印各 link 的 z 坐标剖面（看"平"到什么程度、哪节翘起来）。"""
from __future__ import annotations

import numpy as np
import mujoco

from ik import ARM_CONFIGS, clamp_qpos, get_ee_tip, name_to_id, solve_pose_ik
from grasp_pose_static import GRIP_CLEARANCE
from floor_mount import generate_floor_runtime_xml

BOX_L = 0.50
TILT = 0.30
JOINT_SHORT = ("j1", "j2", "j3", "j4", "j5", "j6", "j7")


def run(base_x, z_mount, pitch_deg, box_z, box_y, show_profile=False):
    xml = generate_floor_runtime_xml(base_x=base_x, z_mount=z_mount,
                                     mount_pitch=np.radians(pitch_deg))
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
    msgs = []
    for a in arms:
        data.qpos[a["qpos"]] = a["q_rest"].copy()
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        tgt = np.array([a["sign"] * (BOX_L / 2 + GRIP_CLEARANCE), box_y, box_z])
        inward = np.array([-a["sign"] * np.cos(TILT), 0.0, np.sin(TILT)])
        for _ in range(1500):
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            dq, _ = solve_pose_ik(
                model, data, a["body_id"], tgt, inward,
                get_ee_tip(data, a["body_id"]),
                a["dofs"], a["qpos"], a["q_rest"])
            data.qpos[a["qpos"]] += dq
            clamp_qpos(model, data, a["qpos"], a["joint_ids"])
        mujoco.mj_forward(model, data)
        err = float(np.linalg.norm(get_ee_tip(data, a["body_id"]) - tgt))
        q = data.qpos[a["qpos"]]
        lo = model.jnt_range[a["joint_ids"], 0]
        hi = model.jnt_range[a["joint_ids"], 1]
        mrg = np.minimum(q - lo, hi - q)
        j_min = int(np.argmin(mrg))
        zs = np.array([data.xpos[name_to_id(model, mujoco.mjtObj.mjOBJ_BODY,
                                            f"openarmx_{a['side']}_link{i}")][2] for i in range(1, 8)])
        plan = float(np.max(np.abs(zs - box_z)))
        msg = (f"{a['side']}: err={err*1e3:.2f}mm 余量={mrg[j_min]:.3f}({JOINT_SHORT[j_min]}) "
               f"平面偏差={plan:.3f}m")
        if show_profile:
            msg += f"\n      z剖面(l1..l7)={np.round(zs, 3)} q={np.round(q, 2)}"
        msgs.append(msg)
    print(f"[zm={z_mount:.2f} pitch={pitch_deg:+.0f}° bz={box_z:.2f} by={box_y:+.2f}] "
          + " | ".join(msgs))


def main() -> None:
    print("=== 细化扫描 ===")
    for pitch in (-50, -55, -60, -65):
        for zm, bz in ((0.15, 0.20), (0.20, 0.25), (0.25, 0.30)):
            run(0.55, zm, pitch, bz, -0.13)
    print("\n=== 最优候选的 z 剖面 ===")
    run(0.55, 0.20, -60, 0.25, -0.13, show_profile=True)
    run(0.55, 0.20, -55, 0.25, -0.13, show_profile=True)
    run(0.55, 0.25, -60, 0.30, -0.13, show_profile=True)


if __name__ == "__main__":
    main()