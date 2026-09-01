#!/usr/bin/env python3
"""Dual-arm position IK demo using the ROS2 visual OpenArmX MuJoCo model.

========== 文件总览 ==========
这是一个 OpenArmX 双臂机器人【双臂】的 MuJoCo 位置级 IK（逆运动学）交互演示脚本：
左右两条 7-DOF 手臂各自独立求解 IK，共同追逐同一个可拖动的目标点（红球）。

整体流程:
    1. 从 ROS2 视觉 URDF 生成 MuJoCo 模型 XML（generate_runtime_xml）
    2. 手写一个展示场景 XML：无重力、灯光、地板、红色目标球（mocap，可拖动）（write_scene）
    3. 启动 MuJoCo passive viewer，鼠标拖动红球设定目标点
    4. 每帧分别求解左右臂 7 关节的 IK 增量并直接写入各自 qpos，让两套圆柱末端端面都追踪红球
    5. 键盘控制复位 / 暂停 / 调试 / 退出

关键技术: 阻尼最小二乘（DLS）+ 零空间投影，解 7-DOF 冗余臂的位置 IK。
注意: 这是纯运动学驱动（手动清零 qvel/qacc 并直接改 qpos），不做动力学仿真。
========== 文件总览 ==========
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from view_ros2_visual_meshes import EE_HAND_OFFSET, OUTPUT_URDF, SOURCE_URDF, generate_visual_urdf


# ---------- 文件与路径 ----------
BASE_DIR = Path(__file__).resolve().parent
RUNTIME_XML = BASE_DIR / "openarmx_ik_runtime.xml"  # 从 URDF 转出的机器人模型 XML（运行时生成）
SCENE_XML = BASE_DIR / "openarmx_ik_scene.xml"      # 展示场景 XML（运行时生成，include 上面的 runtime XML）

# ---------- 双臂配置（与 openarmx_control_runtime.xml 中的名字一一对应） ----------
# 每项包含一条手臂的全部关键对象命名；右臂是左臂的镜像结构，休息位取镜像值
# （joint2/3/6 取反；joint4 因右臂 range 为 [0, 1.8] 无法取反，保持同值）
# 无末端夹具：link7 末端本身就是平的（圆柱杆的顶面），顶夹面 = link7 端面中心，
# 即 link7 局部 +Z 的 EE_TIP_Z 处（原 hand 法兰连接面）。
EE_TIP_Z = EE_HAND_OFFSET  # link7 原点 -> 端面中心 (0.1001)
ARM_CONFIGS = [
    {
        "side": "left",   # 侧别，用于调试输出与可视化球命名
        "joint_names": [f"openarmx_left_joint{i}" for i in range(1, 8)],  # 7 个旋转关节（7-DOF 冗余臂）
        "jacobi_body": "openarmx_left_link7",                             # 末端连杆（其端面即顶夹面）
        "viz_body": "left_gripper_viz",      # 场景中对应的端面可视化球（青色）
        "q_rest": np.array([0.0, -0.35, 0.15, 0.85, 0.0, 0.45, 0.0], dtype=np.float64),  # 休息位姿态
    },
    {
        "side": "right",
        "joint_names": [f"openarmx_right_joint{i}" for i in range(1, 8)],
        "jacobi_body": "openarmx_right_link7",
        "viz_body": "right_gripper_viz",     # 端面可视化球（橙色）
        "q_rest": np.array([0.0, 0.35, -0.15, 0.85, 0.0, -0.45, 0.0], dtype=np.float64),
    },
]

# ---------- IK 求解参数 ----------
DAMPING = 0.20    # 阻尼最小二乘 (DLS) 的阻尼系数 λ：奇异位形附近防止伪逆发散
K_POS = 1.2       # 位置增益：误差放大系数，越大追踪越快（但可能振荡）
K_ROT = 1.2       # 姿态增益：让 link7 底面法向对齐目标方向（对顶用）
K_NULL = 0.001    # 零空间（次要任务）增益：让关节缓慢朝休息位靠拢
MAX_DQ = 0.02     # 单步最大关节增量（弧度），限幅防止 IK 一步跳变

# ---------- 目标点可拖动范围（包围盒，防止拖出可达工作空间） ----------
TARGET_MIN = np.array([-0.45, -0.55, 0.02], dtype=np.float64)
TARGET_MAX = np.array([0.80, 0.55, 1.05], dtype=np.float64)
ROBOT_YAW_LEFT_90 = "-1.57079632679"  # 把整棵机器人树绕 z 轴旋转 -90°（π/2）的 euler 角


def name_to_id(model, obj_type, name: str) -> int:
    """按名字查 MuJoCo 对象 id，找不到直接抛错（避免静默出错）。"""
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"找不到对象: {name}")
    return obj_id


def clamp_qpos(model, data, qpos_ids: list[int], joint_ids: list[int]) -> None:
    """把 qpos 中指定的关节值限制到 MuJoCo 模型定义的 joint range 内。"""
    for qpos_id, joint_id in zip(qpos_ids, joint_ids):
        if model.jnt_limited[joint_id]:  # 只有设置了限位的关节才约束
            low, high = model.jnt_range[joint_id]
            data.qpos[qpos_id] = np.clip(data.qpos[qpos_id], low, high)


def get_body_quat(model, data, body_name: str) -> np.ndarray:
    """取指定 body 的世界姿态（旋转矩阵 xmat -> 四元数）。"""
    body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.xmat[body_id])
    return quat


def get_ee_tip(data, link7_body_id: int) -> np.ndarray:
    """link7 端面中心（世界系）：link7 原点沿其 +Z（手臂末端指向）外推 EE_TIP_Z。

    无夹具，顶夹面就是 link7 平整的端面。
    """
    offset = np.array([0.0, 0.0, EE_TIP_Z])
    return data.xpos[link7_body_id] + data.xmat[link7_body_id].reshape(3, 3) @ offset


def clamp_target(pos: np.ndarray) -> np.ndarray:
    """把目标点限制在 TARGET_MIN~TARGET_MAX 包围盒内。"""
    return np.minimum(np.maximum(pos, TARGET_MIN), TARGET_MAX)


def solve_position_ik(
    model,
    data,
    body_id: int,
    target_pos: np.ndarray,
    current_pos: np.ndarray,
    arm_dofs: list[int],
    arm_qpos: list[int],
    q_rest: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """位置 IK 核心：阻尼最小二乘 (DLS) + 零空间投影，返回关节增量 dq 和位置误差。

    数学原理:
      - 主任务:  Δq主 = J† · (K_pos · e)，其中 J† = Jᵀ(JJᵀ + λ²I)⁻¹ 是阻尼伪逆
        λ = DAMPING，奇异位形附近 JJᵀ 接近奇异时 λ²I 保证可逆，不会发散。
      - 零空间:  Δq次 = (I - J†J) · K_null · (q_rest - q)
        该分量不改变末端位置（在 J 的零空间内），只让 7-DOF 冗余臂的姿态
        朝休息位靠拢，从而在无穷多 IK 解中挑出"自然"的那个。
      - 限幅:    总增量范数超过 MAX_DQ 时等比缩放，配合主循环逐帧积分逐步逼近。
    """
    error = target_pos - current_pos                      # 位置误差 (3,)
    jacp = np.zeros((3, model.nv))                        # 平移雅可比
    jacr = np.zeros((3, model.nv))                        # 旋转雅可比（本脚本只用平移，但 mj_jac 要求两者都传）
    mujoco.mj_jac(model, data, jacp, jacr, current_pos, body_id)

    jac = jacp[:, arm_dofs]                               # 只取这条手臂 7 个自由度对应的列 -> (3,7)
    damping_matrix = (DAMPING**2) * np.eye(3)             # λ²I，用于 DLS
    jac_damped_pinv = jac.T @ np.linalg.solve(jac @ jac.T + damping_matrix, np.eye(3))  # J† = Jᵀ(JJᵀ+λ²I)⁻¹
    dq_primary = jac_damped_pinv @ (K_POS * error)        # 主任务：沿误差梯度方向缩小位置误差

    null_projector = np.eye(len(arm_dofs)) - jac_damped_pinv @ jac   # 零空间投影算子 (I - J†J)
    dq_secondary = K_NULL * (q_rest - data.qpos[arm_qpos])           # 次要任务：拉向休息位
    dq = dq_primary + null_projector @ dq_secondary        # 主 + 零空间分量

    norm = np.linalg.norm(dq)
    if norm > MAX_DQ and norm > 1e-12:                    # 单步增量限幅（防跳变）
        dq = dq / norm * MAX_DQ

    return dq, error


def solve_pose_ik(
    model,
    data,
    body_id: int,
    target_pos: np.ndarray,
    desired_dir: np.ndarray,
    current_pos: np.ndarray,
    arm_dofs: list[int],
    arm_qpos: list[int],
    q_rest: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """6 维位姿 IK：端面点到位 + link7 底面法向（局部 +Z）对齐 desired_dir。

    用于"link7 底部对立对顶"：只把端面中心挪到目标还不够，还要保证
    link7 的底端面法向垂直于目标表面（两臂底面正对、不歪斜）。
    主任务从 3 维位置扩展成 6 维位姿 —— e_p 与 e_r = cross(n_cur, n_des)，
    一并进阻尼伪逆；7-DOF 余下的 1 维仍喂给零空间。
    """
    current_dir = data.xmat[body_id].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
    current_dir /= np.linalg.norm(current_dir)
    desired_dir = desired_dir / np.linalg.norm(desired_dir)

    error = np.concatenate([K_POS * (target_pos - current_pos), K_ROT * np.cross(current_dir, desired_dir)])

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, current_pos, body_id)

    jac = np.vstack([jacp, jacr])[:, arm_dofs]            # (6,7)
    damping_matrix = (DAMPING**2) * np.eye(6)             # λ²I
    jac_damped_pinv = jac.T @ np.linalg.solve(jac @ jac.T + damping_matrix, np.eye(6))
    dq_primary = jac_damped_pinv @ error                  # 主任务：位置+对顶朝向

    null_projector = np.eye(len(arm_dofs)) - jac_damped_pinv @ jac
    dq_secondary = K_NULL * (q_rest - data.qpos[arm_qpos])
    dq = dq_primary + null_projector @ dq_secondary

    norm = np.linalg.norm(dq)
    if norm > MAX_DQ and norm > 1e-12:
        dq = dq / norm * MAX_DQ

    return dq, error[:3]


def generate_runtime_xml() -> Path:
    """从 ROS2 视觉 URDF 生成 MuJoCo 机器人模型 XML。

    1. 先由 view_ros2_visual_meshes 生成"纯视觉" URDF（contype=0，无碰撞）
    2. 加载后再 mj_saveLastXML 转成 MuJoCo XML
    3. 把整棵机器人树包进一个绕 z 轴旋转 -90° 的 openarmx_robot_yaw_root body，
       让机器人朝向观察者（便于演示）
    """
    visual_urdf = generate_visual_urdf(SOURCE_URDF, OUTPUT_URDF)
    model = mujoco.MjModel.from_xml_path(str(visual_urdf))
    mujoco.mj_saveLastXML(str(RUNTIME_XML), model)

    tree = ET.parse(RUNTIME_XML)
    root = tree.getroot()

    # link7 端面加上高摩擦碰撞垫片：机器人 URDF 视觉 mesh 全为 contype=0
    # （纯视觉无碰撞），要"直接受力顶夹"必须有真实接触面。垫片是薄圆盘，
    # 位于 end 面，摩擦设得极大以便先把箱子"牢牢钳住"再谈其它。
    for side in ("left", "right"):
        link7 = root.find(f".//body[@name='openarmx_{side}_link7']")
        if link7 is None:
            raise ValueError(f"runtime XML 中找不到 {side} link7 body")
        ET.SubElement(
            link7,
            "geom",
            {
                "name": f"openarmx_{side}_ee_pad",
                "type": "box",                    # 薄方板，端面贴合箱面（圆柱侧壁抓不住竖向力）
                "pos": f"0 0 {EE_HAND_OFFSET:.4f}",  # 端面中心
                "size": "0.030 0.030 0.004",       # 6cm 见方、8mm 厚
                "rgba": "0 1 1 0.35",              # 半透明青色，看得见接触面
                "contype": "1",
                "conaffinity": "1",
                "condim": "3",
                "friction": "5 0.05 0.005",
            },
        )

    worldbody = root.find("worldbody")
    if worldbody is not None:
        robot_root = ET.Element("body", {"name": "openarmx_robot_yaw_root", "euler": f"0 0 {ROBOT_YAW_LEFT_90}"})
        for child in list(worldbody):   # 把 worldbody 下所有子节点（整棵机器人树）挪进新 body
            worldbody.remove(child)
            robot_root.append(child)
        worldbody.append(robot_root)
    tree.write(RUNTIME_XML, encoding="utf-8", xml_declaration=True)

    return RUNTIME_XML


def write_scene(runtime_xml: Path) -> Path:
    """写展示场景 XML：无重力 + 灯光 + 地板/背景墙/展台 + 三个 mocap 球，并 include 机器人模型。

    - ik_target:         红色半透明球，mocap 可被鼠标拖动，是双臂 IK 的共同目标点
    - left_gripper_viz:  青色小球，跟随左圆柱端面
    - right_gripper_viz: 橙色小球，跟随右圆柱端面
    """
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
    <body name="left_gripper_viz" pos="0 0 0" mocap="true">
      <geom name="left_gripper_viz" type="sphere" size="0.015" rgba="0 1 1 0.8" contype="0" conaffinity="0"/>
    </body>
    <body name="right_gripper_viz" pos="0 0 0" mocap="true">
      <geom name="right_gripper_viz" type="sphere" size="0.015" rgba="1 0.6 0 0.8" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <include file="{runtime_xml.name}"/>
</mujoco>
""",
        encoding="utf-8",
    )
    return SCENE_XML


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenArmX ROS2 visual MuJoCo dual-arm IK.")
    parser.add_argument("--check", action="store_true", help="Only build and load the model")
    args = parser.parse_args()

    # ---------- 生成模型并加载 ----------
    runtime_xml = generate_runtime_xml()   # URDF -> runtime XML（含整树 -90° 旋转）
    scene_xml = write_scene(runtime_xml)   # 场景 XML（include runtime XML）
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)

    # ---------- 预解析 id：一次查好，主循环内不再按名字查找 ----------
    target_body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "ik_target")  # 红色目标球
    target_mocap_id = model.body_mocapid[target_body_id]                       # mocap 位置在 data 中的索引

    # 每条手臂解析出一组运行时句柄：末端连杆 id、tcp id、dof/qpos 索引、可视化球 mocap id
    arms = []
    for cfg in ARM_CONFIGS:
        jac_body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["jacobi_body"])  # 末端连杆（IK 的"端"）
        joint_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in cfg["joint_names"]]
        dofs = [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids]   # 7 个关节在速度空间 (nv) 的索引
        qpos = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]  # 7 个关节在 qpos 中的索引

        viz_body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["viz_body"])  # 端面可视化球

        arms.append(
            {
                "side": cfg["side"],
                "q_rest": cfg["q_rest"],
                "jacobi_body": cfg["jacobi_body"],  # body 名字（供 get_body_quat 等按名查询）
                "jac_body_id": jac_body_id,
                "joint_ids": joint_ids,
                "dofs": dofs,
                "qpos": qpos,
                "viz_mocap_id": model.body_mocapid[viz_body_id],
            }
        )

    # ---------- 运行时状态（被键盘回调修改，主循环读取） ----------
    state = {
        "running": True,          # Q 键退出
        "pause": False,           # P 键暂停/继续 IK
        "reset": False,           # R 键复位
        "debug": True,            # D 键开关调试输出
    }

    def reset_robot() -> None:
        """复位：双臂回到各自休息位，并把目标球放到两臂圆柱端面的中间位置。"""
        for arm in arms:
            data.qpos[arm["qpos"]] = arm["q_rest"].copy()
            clamp_qpos(model, data, arm["qpos"], arm["joint_ids"])
        mujoco.mj_forward(model, data)

        # 让红球吸附到两臂圆柱端面中心的中点，青色/橙色球分别吸附到各自端面，
        # 避免复位瞬间 IK 误差突变
        tips = [get_ee_tip(data, arm["jac_body_id"]).copy() for arm in arms]
        target_pos = clamp_target(np.mean(tips, axis=0))
        data.mocap_pos[target_mocap_id] = target_pos
        data.mocap_quat[target_mocap_id] = get_body_quat(model, data, arms[0]["jacobi_body"])
        for arm, tip in zip(arms, tips):
            data.mocap_pos[arm["viz_mocap_id"]] = tip
        mujoco.mj_forward(model, data)

    def key_callback(keycode: int) -> None:
        """键盘回调：R 复位，P 暂停，D 调试，Q 退出。"""
        try:
            key = chr(keycode).upper()
        except ValueError:
            return
        if key == "R":
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

    print("✅ OpenArmX 双臂 IK 脚本已加载")
    print("操作: 鼠标拖动红球, R 复位, P 暂停, D 调试, Q 退出")
    print(f"nq={model.nq} nv={model.nv} nbody={model.nbody} ngeom={model.ngeom}")

    if args.check:  # --check 只验证模型能否构建加载
        return

    last_debug = 0.0
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # 相机初始视角
        viewer.cam.lookat[:] = [0.0, -0.02, 0.58]
        viewer.cam.distance = 2.25
        viewer.cam.elevation = -16
        viewer.cam.azimuth = 180

        while viewer.is_running() and state["running"]:
            # ---------- 复位请求：恢复初始状态后继续下一帧 ----------
            if state["reset"]:
                reset_robot()
                state["reset"] = False
                viewer.sync()
                continue

            # ---------- 读取鼠标拖动的目标位置（限制在可拖动范围内） ----------
            target_pos = clamp_target(data.mocap_pos[target_mocap_id].copy())
            data.mocap_pos[target_mocap_id] = target_pos

            # ---------- 纯运动学驱动：清零速度/加速度，只做正运动学 ----------
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            mujoco.mj_forward(model, data)

            # ---------- 求解 IK：双臂各自基于当前状态独立求 dq（顺序无关） ----------
            solutions = []
            for arm in arms:
                current_pos = get_ee_tip(data, arm["jac_body_id"])
                dq, _ = solve_position_ik(
                    model, data, arm["jac_body_id"], target_pos, current_pos,
                    arm["dofs"], arm["qpos"], arm["q_rest"],
                )
                solutions.append((arm, dq))

            # 统一写回 qpos 后再 forward，避免后求解的手臂受到先写回关节的影响
            for arm, dq in solutions:
                data.qpos[arm["qpos"]] += dq
                clamp_qpos(model, data, arm["qpos"], arm["joint_ids"])  # 关节限位

            # ---------- 更新可视化：青色/橙色球跟随各自圆柱端面 ----------
            mujoco.mj_forward(model, data)
            for arm in arms:
                data.mocap_pos[arm["viz_mocap_id"]] = get_ee_tip(data, arm["jac_body_id"])

            # ---------- 调试输出（每 0.5 秒一次，显示两臂误差） ----------
            now = time.time()
            if state["debug"] and now - last_debug > 0.5:
                errs = []
                for arm in arms:
                    tip = data.mocap_pos[arm["viz_mocap_id"]]
                    errs.append(f"{arm['side']}={np.linalg.norm(target_pos - tip):.5f}")
                print(f"[DEBUG] target={target_pos.round(4)} " + " ".join(errs))
                last_debug = now

            viewer.sync()
            time.sleep(model.opt.timestep)  # 按仿真步长节流主循环


if __name__ == "__main__":
    main()