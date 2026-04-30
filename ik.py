#!/usr/bin/env python3
"""Left-arm position IK demo using the ROS2 visual OpenArmX MuJoCo model."""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from view_ros2_visual_meshes import OUTPUT_URDF, SOURCE_URDF, generate_visual_urdf


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_XML = BASE_DIR / "openarmx_ik_runtime.xml"
SCENE_XML = BASE_DIR / "openarmx_ik_scene.xml"

FINGER_BODIES = ["openarmx_left_right_finger", "openarmx_left_left_finger"]
JACOBIAN_BODY = "openarmx_left_link7"
ARM_JOINTS = [f"openarmx_left_joint{i}" for i in range(1, 8)]
GRIPPER_JOINTS = ["openarmx_left_finger_joint1", "openarmx_left_finger_joint2"]

GRIPPER_OPEN = 0.040
GRIPPER_CLOSE = 0.002
GRIPPER_SMOOTH = 0.15

DAMPING = 0.20
K_POS = 1.2
K_NULL = 0.001
MAX_DQ = 0.02
Q_REST = np.array([0.0, -0.35, 0.15, 0.85, 0.0, 0.45, 0.0], dtype=np.float64)

TARGET_MIN = np.array([-0.45, -0.55, 0.02], dtype=np.float64)
TARGET_MAX = np.array([0.80, 0.55, 1.05], dtype=np.float64)
ROBOT_YAW_LEFT_90 = "-1.57079632679"


def name_to_id(model, obj_type, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"找不到对象: {name}")
    return obj_id


def clamp_qpos(model, data, qpos_ids: list[int], joint_ids: list[int]) -> None:
    for qpos_id, joint_id in zip(qpos_ids, joint_ids):
        if model.jnt_limited[joint_id]:
            low, high = model.jnt_range[joint_id]
            data.qpos[qpos_id] = np.clip(data.qpos[qpos_id], low, high)


def get_body_pos(model, data, body_name: str) -> np.ndarray:
    body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy()


def get_body_quat(model, data, body_name: str) -> np.ndarray:
    body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.xmat[body_id])
    return quat


def get_gripper_center(model, data) -> np.ndarray:
    positions = []
    for body_name in FINGER_BODIES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            positions.append(data.xpos[body_id].copy())
    if not positions:
        raise ValueError(f"找不到手指 body: {FINGER_BODIES}")
    return np.mean(positions, axis=0)


def clamp_target(pos: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(pos, TARGET_MIN), TARGET_MAX)


def solve_position_ik(
    model,
    data,
    body_id: int,
    target_pos: np.ndarray,
    current_pos: np.ndarray,
    arm_dofs: list[int],
    arm_qpos: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    error = target_pos - current_pos
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, current_pos, body_id)

    jac = jacp[:, arm_dofs]
    damping_matrix = (DAMPING**2) * np.eye(3)
    jac_damped_pinv = jac.T @ np.linalg.solve(jac @ jac.T + damping_matrix, np.eye(3))
    dq_primary = jac_damped_pinv @ (K_POS * error)

    null_projector = np.eye(len(arm_dofs)) - jac_damped_pinv @ jac
    dq_secondary = K_NULL * (Q_REST - data.qpos[arm_qpos])
    dq = dq_primary + null_projector @ dq_secondary

    norm = np.linalg.norm(dq)
    if norm > MAX_DQ and norm > 1e-12:
        dq = dq / norm * MAX_DQ

    return dq, error


def generate_runtime_xml() -> Path:
    visual_urdf = generate_visual_urdf(SOURCE_URDF, OUTPUT_URDF)
    model = mujoco.MjModel.from_xml_path(str(visual_urdf))
    mujoco.mj_saveLastXML(str(RUNTIME_XML), model)

    tree = ET.parse(RUNTIME_XML)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is not None:
        robot_root = ET.Element("body", {"name": "openarmx_robot_yaw_root", "euler": f"0 0 {ROBOT_YAW_LEFT_90}"})
        for child in list(worldbody):
            worldbody.remove(child)
            robot_root.append(child)
        worldbody.append(robot_root)
    tree.write(RUNTIME_XML, encoding="utf-8", xml_declaration=True)

    return RUNTIME_XML


def write_scene(runtime_xml: Path) -> Path:
    SCENE_XML.write_text(
        f"""<mujoco model="openarmx_ik_scene">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>

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
    <material name="platform_mat" rgba="0.20 0.22 0.25 1" reflectance="0.18" shininess="0.45"/>
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

    <geom name="floor" type="plane" pos="0 0 -0.012" size="2.4 2.4 0.02" material="floor_mat"/>
    <geom name="back_wall" type="box" pos="0 0.95 0.78" size="2.4 0.025 0.80" material="wall_mat" contype="0" conaffinity="0"/>
    <geom name="left_wall" type="box" pos="-1.25 0 0.55" size="0.025 1.15 0.56" material="wall_mat" contype="0" conaffinity="0"/>
    <geom name="right_wall" type="box" pos="1.25 0 0.55" size="0.025 1.15 0.56" material="wall_mat" contype="0" conaffinity="0"/>
    <geom name="display_platform" type="cylinder" pos="0 0 -0.006" size="0.72 0.012" material="platform_mat" contype="0" conaffinity="0"/>

    <body name="ik_target" pos="0.25 0.18 0.45" mocap="true">
      <geom name="ik_target" type="sphere" size="0.035" rgba="1 0 0 0.45" contype="0" conaffinity="0"/>
    </body>
    <body name="gripper_center_viz" pos="0 0 0" mocap="true">
      <geom name="gripper_center_viz" type="sphere" size="0.015" rgba="0 1 1 0.8" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <include file="{runtime_xml.name}"/>
</mujoco>
""",
        encoding="utf-8",
    )
    return SCENE_XML


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenArmX ROS2 visual MuJoCo left-arm IK.")
    parser.add_argument("--check", action="store_true", help="Only build and load the model")
    args = parser.parse_args()

    runtime_xml = generate_runtime_xml()
    scene_xml = write_scene(runtime_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)

    jac_body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, JACOBIAN_BODY)
    target_body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "ik_target")
    target_mocap_id = model.body_mocapid[target_body_id]
    viz_body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_center_viz")
    viz_mocap_id = model.body_mocapid[viz_body_id]

    arm_joint_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]
    arm_dofs = [int(model.jnt_dofadr[joint_id]) for joint_id in arm_joint_ids]
    arm_qpos = [int(model.jnt_qposadr[joint_id]) for joint_id in arm_joint_ids]

    gripper_joint_ids = []
    for joint_name in GRIPPER_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            gripper_joint_ids.append(joint_id)
    gripper_qpos = [int(model.jnt_qposadr[joint_id]) for joint_id in gripper_joint_ids]

    state = {
        "running": True,
        "pause": False,
        "reset": False,
        "debug": True,
        "gripper_target": GRIPPER_OPEN,
    }

    def reset_robot() -> None:
        data.qpos[arm_qpos] = Q_REST.copy()
        clamp_qpos(model, data, arm_qpos, arm_joint_ids)
        for qpos_id in gripper_qpos:
            data.qpos[qpos_id] = GRIPPER_OPEN
        mujoco.mj_forward(model, data)

        target_pos = clamp_target(get_gripper_center(model, data).copy())
        data.mocap_pos[target_mocap_id] = target_pos
        data.mocap_quat[target_mocap_id] = get_body_quat(model, data, JACOBIAN_BODY)
        data.mocap_pos[viz_mocap_id] = target_pos
        mujoco.mj_forward(model, data)

    def key_callback(keycode: int) -> None:
        try:
            key = chr(keycode).upper()
        except ValueError:
            return
        if key == "O":
            state["gripper_target"] = GRIPPER_OPEN
            print("👉 左夹爪张开")
        elif key == "C":
            state["gripper_target"] = GRIPPER_CLOSE
            print("👉 左夹爪闭合")
        elif key == "R":
            state["reset"] = True
            print("👉 复位")
        elif key == "P":
            state["pause"] = not state["pause"]
            print(f"👉 IK {'暂停' if state['pause'] else '继续'}")
        elif key == "D":
            state["debug"] = not state["debug"]
            print(f"👉 调试输出 {'开启' if state['debug'] else '关闭'}")
        elif key == "Q":
            state["running"] = False
            print("👉 退出")

    reset_robot()

    print("✅ OpenArmX IK 脚本已加载")
    print("操作: 鼠标拖动红球, O/C 夹爪, R 复位, P 暂停, D 调试, Q 退出")
    print(f"nq={model.nq} nv={model.nv} nbody={model.nbody} ngeom={model.ngeom}")

    if args.check:
        return

    last_debug = 0.0
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = [0.0, -0.02, 0.58]
        viewer.cam.distance = 2.25
        viewer.cam.elevation = -16
        viewer.cam.azimuth = 180

        while viewer.is_running() and state["running"]:
            if state["reset"]:
                reset_robot()
                state["reset"] = False
                viewer.sync()
                continue

            target_pos = clamp_target(data.mocap_pos[target_mocap_id].copy())
            data.mocap_pos[target_mocap_id] = target_pos

            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            mujoco.mj_forward(model, data)

            current_pos = get_gripper_center(model, data)
            if not state["pause"]:
                dq, error = solve_position_ik(model, data, jac_body_id, target_pos, current_pos, arm_dofs, arm_qpos)
                data.qpos[arm_qpos] += dq
                clamp_qpos(model, data, arm_qpos, arm_joint_ids)

            for qpos_id in gripper_qpos:
                data.qpos[qpos_id] += GRIPPER_SMOOTH * (state["gripper_target"] - data.qpos[qpos_id])

            mujoco.mj_forward(model, data)
            gripper_center = get_gripper_center(model, data)
            data.mocap_pos[viz_mocap_id] = gripper_center

            now = time.time()
            if state["debug"] and now - last_debug > 0.5:
                print(f"[DEBUG] target={target_pos.round(4)} cur={gripper_center.round(4)} err={np.linalg.norm(target_pos - gripper_center):.5f}")
                last_debug = now

            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
