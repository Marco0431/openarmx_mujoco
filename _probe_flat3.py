#!/usr/bin/env python3
"""_probe_flat3.py —— 侧装（mount_roll→±90°）扫描：让臂链"自然下垂方向"变成水平伸向箱内。

臂的链节偏移沿 link 局部 +z；把臂侧滚 90°，局部 +z 转为水平 → 整条链沿水平面行进，
有望真正实现"基座落地 + 全臂一个水平面"。
"""
from __future__ import annotations

import numpy as np
import mujoco

from ik import ARM_CONFIGS, clamp_qpos, get_ee_tip, name_to_id, solve_pose_ik
from grasp_pose_static import GRIP_CLEARANCE
from floor_mount import generate_floor_runtime_xml

BOX_L = 0.50
TILT = 0.30
JOINT_SHORT = ("j1", "j2", "j3", "j4", "j5", "j6", "j7")


def run(base_x, z_mount, roll_deg, box_z, box_y, show_profile=False):
    xml = generate_floor_runtime_xml(base_x=base_x, z_mount=z_mount,
                                     mount_roll=np.radians(roll_deg))
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
        zmin = float(zs.min())
        msg = (f"{a['side']}: err={err*1e3:.2f}mm 余量={mrg[j_min]:.3f}({JOINT_SHORT[j_min]}) "
               f"平面偏差={plan:.3f}m 最低link={zmin:.3f}m")
        if show_profile:
            msg += f"\n      PROFILE(l1..l7)={np.round(zs, 3)} q={np.round(q, 2)}"
        msgs.append(msg)
    print(f"[zm={z_mount:.2f} roll={roll_deg:+.0f}° bz={box_z:.2f} by={box_y:+.2f}] "
          + " | ".join(msgs))


def main() -> None:
    print("=== 镜像侧装（左+90°/右-90°）：tilt 是否还需要 ===")
    for tilt in (0.0, 0.15, 0.30):
        global TILT
        TILT = tilt
        xml = generate_floor_runtime_xml(base_x=0.55, z_mount=0.15,
                                         mount_roll={"left": np.pi / 2, "right": -np.pi / 2})
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
            tgt = np.array([a["sign"] * (BOX_L / 2 + GRIP_CLEARANCE), -0.13, 0.20])
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
            plan = float(np.max(np.abs(zs[1:] - 0.20)))   # 平面性看 l2..l7（l1 是基座原点）
            msgs.append(f"{a['side']}: err={err*1e3:.2f}mm 余量={mrg[j_min]:.3f}({JOINT_SHORT[j_min]}) "
                        f"平面偏差(l2-7)={plan:.3f}m PROFILE={np.round(zs, 3)}")
        print(f"[tilt={np.degrees(tilt):.0f}°] " + " | ".join(msgs))


if __name__ == "__main__":
    main()