#!/usr/bin/env python3
"""_probe_flat.py —— "全平"布局扫描：基座落地 + 臂近水平面内伸展（mount_pitch 接近 -90°）

用户目标：像论文一样立在地上，且整条臂尽量在一个 z 水平面内。
做法：臂安装俯仰 pitch → -90°（臂链从地面基座直接水平伸出），箱高跟着降到臂平面。
扫描：z_mount × pitch × box_z × box_y，报告 端面误差 / 限位关节余量 / 平面偏差。
"""
from __future__ import annotations

import numpy as np
import mujoco

from ik import ARM_CONFIGS, clamp_qpos, get_ee_tip, name_to_id, solve_pose_ik
from grasp_pose_static import GRIP_CLEARANCE
from floor_mount import generate_floor_runtime_xml

BOX_L = 0.50
TILT = 0.30            # 杆身上仰角（j7 限位妥协，垫片由 pad_fix 修正回竖直）
JOINT_SHORT = ("j1", "j2", "j3", "j4", "j5", "j6", "j7")


def run(base_x, z_mount, pitch_deg, box_z, box_y):
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
        zs = [data.xpos[name_to_id(model, mujoco.mjtObj.mjOBJ_BODY,
                                   f"openarmx_{a['side']}_link{i}")][2] for i in range(1, 8)]
        plan = float(np.max(np.abs(np.array(zs) - box_z)))
        msgs.append(f"{a['side']}: err={err*1e3:.2f}mm 余量={mrg[j_min]:.3f}({JOINT_SHORT[j_min]}) "
                    f"平面偏差={plan:.3f}m")
    print(f"[zm={z_mount:.2f} pitch={pitch_deg:+.0f}° bz={box_z:.2f} by={box_y:+.2f}] "
          + " | ".join(msgs))


def main() -> None:
    print("=== 全平布局扫描：基座落地，臂近水平面伸展 ===")
    for pitch in (-60, -75, -90):
        for z_mount, box_z in ((0.10, 0.15), (0.15, 0.20), (0.20, 0.25)):
            for box_y in (0.0, -0.13):
                run(0.55, z_mount, pitch, box_z, box_y)


if __name__ == "__main__":
    main()