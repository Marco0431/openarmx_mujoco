#!/usr/bin/env python3
"""双臂平端面（无夹具）夹持大箱子演示。

- 开重力：箱子（2 kg）自由悬在（/落在）两臂之间
- 左/右臂 link7 的端面垫片通过位置 IK 分别钉在箱子对应侧面的中点，
并让 link7 底面（法向 +Z）垂直于箱面对立贴合
- 箱子摩擦设得极大（sliding=50），顶住之后不会滑落 → 验证"顶夹 + 摩擦力"链路
- 不做力矩/力控：纯位置驱动（约定阶段，先验证能不能夹住）

运行：
    python grasp_box_demo.py --check      # 只构建加载
    python grasp_box_demo.py              # 打开 MuJoCo 被动 viewer
    python grasp_box_demo.py --headless   # 无 GUI，跑 N 步打印稳定性
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from ik import (
    ARM_CONFIGS,
    BASE_DIR,
    ROBOT_YAW_LEFT_90,
    TARGET_MAX,
    TARGET_MIN,
    clamp_qpos,
    clamp_target,
    generate_runtime_xml,
    get_ee_tip,
    name_to_id,
    solve_pose_ik,
)

# ---------- 箱子与夹持参数 ----------
BOX_HALF = np.array([0.12, 0.12, 0.10], dtype=np.float64)  # 箱子半尺寸（x y z）：实体前臂粗，箱须小到连杆不入箱
BOX_MASS = 1.0                                              # 箱子质量 (kg)
BOX_FRICTION = "5 0.05 0.005"   # 大摩擦（滑动 5，远超所需 0.35；数值可收敛的范围内取大）
GRIP_CLEARANCE = 0.001  # 底面与箱面的贴合余量（m）：贴着但不插入（实体物理下不能压进刚体）
IK_DT = 0.001          # 主循环步长（与场景 timestep 一致）


def build_scene() -> Path:
    """生成 runtime XML（含端面垫片）并写夹持场景：重力开 + 大摩擦箱子。"""
    runtime_xml = generate_runtime_xml()
    scene_xml = BASE_DIR / "grasp_box_scene.xml"
    scene_xml.write_text(
        f"""<mujoco model="grasp_box_scene">
  <compiler angle="radian"/>
  <option timestep="{IK_DT}" gravity="0 0 -9.81" iterations="500" solver="Newton" cone="elliptic"/>

  <asset>
    <texture name="studio_sky" type="skybox" builtin="gradient"
             rgb1="0.08 0.10 0.14" rgb2="0.01 0.012 0.018"
             width="512" height="512"/>
    <texture name="floor_grid" type="2d" builtin="checker"
             rgb1="0.16 0.18 0.21" rgb2="0.24 0.27 0.31"
             width="1024" height="1024" mark="edge" markrgb="0.55 0.60 0.68"/>
    <material name="floor_mat" texture="floor_grid" texrepeat="7 7"
              reflectance="0.22" shininess="0.35" specular="0.25"/>
    <material name="wall_mat" rgba="0.10 0.12 0.16 1" reflectance="0.08"/>
    <material name="box_mat" rgba="0.85 0.55 0.20 1" shininess="0.6" specular="0.4"/>
  </asset>

  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.24 0.24 0.26" specular="0.25 0.25 0.25"/>
    <map znear="0.01" zfar="8"/>
    <scale forcewidth="0.06" contactwidth="0.05"/>
    <rgba haze="0.03 0.035 0.045 1"/>
  </visual>

  <worldbody>
    <light name="key_light" pos="-1.4 -2.2 3.2" dir="0.35 0.55 -1" directional="true"
           diffuse="0.95 0.92 0.86" specular="0.35 0.35 0.35"/>
    <light name="fill_light" pos="1.8 1.4 2.2" dir="-0.45 -0.35 -1" directional="true"
           diffuse="0.35 0.45 0.65" specular="0.12 0.12 0.16"/>
    <light name="rim_light" pos="0 2.2 1.8" dir="0 -1 -0.45" directional="true"
           diffuse="0.55 0.65 0.85" specular="0.20 0.22 0.28"/>

    <geom name="floor" type="plane" pos="0 0 -0.012" size="3.0 3.0 0.02" material="floor_mat"/>
    <geom name="back_wall" type="box" pos="0 0.95 0.78" size="2.6 0.025 0.80" material="wall_mat" contype="0" conaffinity="0"/>

    <body name="box" pos="0.5 0 0.72">
      <freejoint name="box_freejoint"/>
      <geom name="box" type="box" size="{BOX_HALF[0]} {BOX_HALF[1]} {BOX_HALF[2]}"
            material="box_mat" mass="{BOX_MASS}"
            friction="{BOX_FRICTION}" condim="3" contype="1" conaffinity="1"
            solref="0.002 1" solimp="0.999 0.999 0.001"/>
    </body>
  </worldbody>

  <include file="{runtime_xml.name}"/>
</mujoco>
""",
        encoding="utf-8",
    )
    return scene_xml


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenArmX 双臂平端面顶夹大箱子演示。")
    parser.add_argument("--check", action="store_true", help="Only build and load the model")
    parser.add_argument("--headless", action="store_true", help="No viewer; run fixed steps and print stability")
    parser.add_argument("--steps", type=int, default=600, help="Headless steps")
    args = parser.parse_args()

    scene_xml = build_scene()
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)

    # ---------- 解析对象 ----------
    box_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_geom_id = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    box_qpos = int(model.jnt_qposadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")])
    box_qvel = int(model.jnt_dofadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")])

    arms = []
    for cfg in ARM_CONFIGS:
        jac_body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["jacobi_body"])
        joint_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg["joint_names"]]
        dofs = [int(model.jnt_dofadr[j]) for j in joint_ids]
        qpos = [int(model.jnt_qposadr[j]) for j in joint_ids]
        arms.append(
            {
                "side": cfg["side"],
                # runtime yaw 后：左臂沿 +x 伸出、右臂沿 -x 伸出，夹持轴是 x
                "sign": 1.0 if cfg["side"] == "left" else -1.0,
                "q_rest": cfg["q_rest"],
                "jac_body_id": jac_body_id,
                "joint_ids": joint_ids,
                "dofs": dofs,
                "qpos": qpos,
            }
        )

    # ---------- 复位：机器人回休息位，再算两臂端面，把箱子摆到端面中点（z 同高、y 居中） ----------
    for arm in arms:
        data.qpos[arm["qpos"]] = arm["q_rest"].copy()
        clamp_qpos(model, data, arm["qpos"], arm["joint_ids"])
    mujoco.mj_forward(model, data)

    tips = [get_ee_tip(data, arm["jac_body_id"]).copy() for arm in arms]
    mid = np.mean(tips, axis=0)
    data.qpos[box_qpos : box_qpos + 3] = mid   # 箱子放在两臂端面正中间，初始不穿任何杆
    data.qpos[box_qpos + 3 : box_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[box_qvel : box_qvel + 6] = 0.0
    mujoco.mj_forward(model, data)

    box_start = data.xpos[box_id].copy()
    grasp_y = float(mid[1])   # 固定抓取位：跟随箱子 x（夹紧），y/z 锁死避免顶飞/侧滑
    grasp_z = float(mid[2])
    print("✅ 双臂顶夹演示已加载")
    print(f"nbody={model.nbody} ngeom={model.ngeom}")
    try:
        pad_id = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "openarmx_left_ee_pad")
        print(f"左垫片 geom id={pad_id} 中心={data.geom_xpos[pad_id].round(4)}")
    except ValueError as exc:
        print("警告：", exc)
    print(f"箱子中心 {box_start.round(4)} | 端面初始 {[t.round(4) for t in tips]} | 抓取高度 {grasp_z:.4f}")

    if args.check:
        return

    # ---------- 每帧：双臂各自钉住箱子对应侧面中点，再推进动力学一步 ----------
    def step_control() -> None:
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_forward(model, data)

        box_pos = data.xpos[box_id]
        solutions = []
        for arm in arms:
            target_pos = clamp_target(
                [box_pos[0] + arm["sign"] * (BOX_HALF[0] + GRIP_CLEARANCE), grasp_y, grasp_z]
            )
            # 用 link7 底端面对顶：端面点到位，且底面法向（+Z）垂直于箱子侧面
            desired_dir = np.array([arm["sign"], 0.0, 0.0], dtype=np.float64)
            current_pos = get_ee_tip(data, arm["jac_body_id"])
            dq, _ = solve_pose_ik(
                model, data, arm["jac_body_id"], target_pos, desired_dir, current_pos,
                arm["dofs"], arm["qpos"], arm["q_rest"],
            )
            solutions.append((arm, dq, target_pos))

        for arm, dq, _ in solutions:
            data.qpos[arm["qpos"]] += dq
            clamp_qpos(model, data, arm["qpos"], arm["joint_ids"])

        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)

    def contact_force_sum() -> float:
        """与箱子接触的几何对的总法向力（N）。"""
        total = 0.0
        force = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 == box_geom_id or c.geom2 == box_geom_id:
                mujoco.mj_contactForce(model, data, i, force)
                total += force[0]
                print(
                    f"   接触 geom{c.geom1}({mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)})"
                    f"<->geom{c.geom2}({mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)}) "
                    f"dist={c.dist:.5f} Fn={force[0]:.1f}N "
                    f"norm={c.frame[0:3].round(3)}"
                )
        print(f"    ncon(总)={data.ncon}")
        return total

    if args.headless:
        last_report = 0
        for t in range(args.steps):
            step_control()
            if t - last_report >= 100:
                drift = np.linalg.norm(data.xpos[box_id] - box_start)
                tips = [get_ee_tip(data, arm["jac_body_id"]) for arm in arms]
                box_pos = data.xpos[box_id]
                print(
                    f"step={t:5d} 箱位漂移={drift*1000:.1f} mm "
                    f"法向合力={contact_force_sum():7.1f} N "
                    f"箱高={data.xpos[box_id][2]:.4f}"
                )
                for arm, tip in zip(arms, tips):
                    target = np.array(
                        [box_pos[0] + arm["sign"] * (BOX_HALF[0] + GRIP_CLEARANCE), grasp_y, grasp_z]
                    )
                    print(
                        f"   {arm['side']}: tip={tip.round(3)} target={target.round(3)} "
                        f"err={np.linalg.norm(tip - target)*1000:.1f}mm"
                    )
                last_report = t
        print("完成。")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.4, 0.0, 0.65]
        viewer.cam.distance = 2.6
        viewer.cam.elevation = -14
        viewer.cam.azimuth = 180

        while viewer.is_running():
            step_control()
            viewer.sync()
            time.sleep(IK_DT)


if __name__ == "__main__":
    main()