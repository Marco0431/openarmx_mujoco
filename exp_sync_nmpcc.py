#!/usr/bin/env python3
"""exp_sync_nmpcc.py —— 论文 Experiment I (Motion Synchronization) 场景与实验主脚本

论文: Zhao 等, "A Dual-Arm Robot Cooperation Framework Based on a Nonlinear Model
Predictive Cooperative Control", IEEE/ASME TMech 2024, V 节 Experiment I (Fig. 9/10)。

Experiment I 语义（用户确认）：
  - 双臂机器人摆得很开（对应论文 Fig. 9：两机器人分居桌两侧），双臂"运动同步模式"
  - ηcop,1 = ηcop,2 = 0.8，ηf,1 = ηf,2 = 0（式 11/12 只保留跟踪+协同项，无力项）
  - 每臂参考速度 -20 mm/s（y 方向），两臂一起动、相互位置保持（名义间距恒定）
  - 初始 x 向 2 mm 同步位置误差（相对名义间距的偏差），y 向无初始误差
  - 同步误差定义（相对参考）: e_sync = (r_L - r_R) - (r_ref,L - r_ref,R)
    对应 NMPCC 用"参考相对协同误差"：ε_mult,i = e_trac,i - ηcop,i·e_trac,j
    （= 论文式 12 的 ε 同构，协同项取 e_cop' = e_trac,j - e_trac,i；
      平衡点：两臂都回自己参考 + 相对偏差 = 0，即"一起动、误差归零"）
  - 对照：论文式 (7) 状态反馈（PD）由用户自己实现，出图脚本留 --overlay_pd 接口

起始姿态（用户要求：用 ik.py 的原始松弛姿态，不做收敛性 IK 条件化）：
  - 双臂设为 ARM_CONFIGS 的 q_rest 休息位（原始即可达；tips 左右对称 ±0.335 m，z=0.295）
  - 仅左臂端面 +2 mm x 偏移（右臂保持名义位）→ 初始 e_sync_x = 2 mm、e_sync_y = 0
  - 参考：r_ref,i(t) = r_i^0 + (0, -0.02·t, 0)（各臂自己的名义起点,共享 -20 mm/s）
  - 时长默认 6 s（行程 0.12 m：探针实测该区段内绝对跟踪误差 ≤~10 mm，对称漂移不影响相对同步误差）

场景要点：无箱、重力开、timestep=0.001 s（1 kHz）、14 个力矩电机、垫片无碰撞、
双目标球(红=各臂参考点) + 双端面球(青=左,橙=右) + 两排参考路径点。

运行（必须用 acados 环境）:
  D:\\acados_env\\python.exe -u exp_sync_nmpcc.py --check        # 只构建加载
  D:\\acados_env\\python.exe -u exp_sync_nmpcc.py --screenshot  # 渲染预览图到 results/exp1_sync/
  D:\\acados_env\\python.exe -u exp_sync_nmpcc.py --viewer      # 交互窗口（运动学保持姿态）
  D:\\acados_env\\python.exe -u exp_sync_nmpcc.py --run         # NMPCC 闭环 1 kHz → CSV + Fig.10 图
  D:\\acados_env\\python.exe -u exp_sync_nmpcc.py --run --controller pd --overlay_pd <你的pd.csv>
"""

from __future__ import annotations

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from ik import (
    ARM_CONFIGS,
    BASE_DIR,
    EE_TIP_Z,
    clamp_qpos,
    generate_runtime_xml,
    get_ee_tip,
    name_to_id,
    solve_position_ik,
)
from ikforce import inject_motors, strip_collisions

# Windows 控制台默认 GBK：重配 stdout/stderr 为 UTF-8（view_ros2_visual_meshes 等打印 emoji）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ======================================================================
# 实验参数区（论文 Experiment I 口径 + 本机可达性探针结论）
# ======================================================================
DT = 0.001                 # 控制/仿真步长 1 ms（论文 1 kHz 控制周期）
DURATION = 8.0             # 仿真时长 (s)：P0 处双臂公共窗口 y∈[-0.468,+0.468]，-20mm/s×8s=0.16m
SYNC_ERR_X0 = 0.002        # 初始 x 向同步位置误差 2 mm（左 +1 mm / 右 -1 mm）
V_REF_Y = -0.020           # 参考速度 -20 mm/s（y 方向，两臂参考相同——论文式(18) 平衡点要求 Δ_ref=0）
# 共同参考点起点（可达性探针选定）：x=0（双臂镜像中面）z=0.40 处双臂公共工作空间
# y ∈ [-0.468, +0.468]，参考行程 0.02*20 = 0.4 m → 边界余量 68 mm
P0 = np.array([0.0, 0.0, 0.40])
ETA_COP_AXES = (0.8, 0.8)  # 论文 ηcop,1 = ηcop,2 = 0.8（x/y 平移轴；z/姿态纯跟踪）
ETA_F = 0.0                # 论文 ηf = 0（无力项）
TORQUE_LIMIT = 8.0         # 力矩限幅 ±8 N·m（论文 10 N for safety，沿用实验二 ±8）
JOINT_DAMPING = 0.5        # 关节粘性阻尼：未建模摩擦 = 论文"不确定项"（预测模型不含）
# ======================================================================

FIG_DIR = BASE_DIR / "results" / "exp1_sync"
SCENE_XML = BASE_DIR / "sync_scene.xml"


# ======================================================================
# 场景构建
# ======================================================================
def disable_pad_collision(runtime_xml: Path) -> None:
    """把 *.ee_pad 垫片设为纯可视化（contype=0）：本实验无任何接触目标。"""
    tree = ET.parse(runtime_xml)
    root = tree.getroot()
    n = 0
    for body in root.iter("body"):
        for geom in body.findall("geom"):
            if (geom.get("name", "") or "").endswith("_ee_pad"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                n += 1
    tree.write(runtime_xml, encoding="utf-8", xml_declaration=True)
    print(f"✅ 已设 {n} 个垫片 contype=0（纯可视化，无碰撞）")


def nominal_tips(model, data, arms) -> dict:
    """原始名义姿态下的各臂端面（世界系）：q_rest 休息位。"""
    for a in arms:
        data.qpos[a["qpos"]] = a["q_rest"].copy()
        clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    out = {}
    for a in arms:
        out[a["side"]] = get_ee_tip(data, a["body_id"]).copy()
    return out


def write_scene_xml(runtime_xml: Path, nom_tips: dict) -> None:
    """写 Experiment I 场景 XML：无箱、重力开、1 ms、双目标/双端面可视球 + 共同参考路径点。"""
    y_end = P0[1] + V_REF_Y * DURATION
    dots = []
    for k, py in enumerate(np.linspace(P0[1], y_end, 9)):
        dots.append(
            f'    <geom name="pathdot_{k}" type="sphere" pos="{P0[0]:.4f} {py:.3f} {P0[2]:.4f}" '
            f'size="0.006" rgba="0.7 0.15 0.15 0.55" contype="0" conaffinity="0"/>'
        )
    dots_xml = "\n".join(dots)
    SCENE_XML.write_text(
        f"""<mujoco model="sync_scene">
  <compiler angle="radian"/>
  <option timestep="{DT}" gravity="0 0 -9.81" iterations="500" solver="Newton" cone="elliptic"/>

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
  </asset>

  <visual>
    <global offwidth="1600" offheight="1200"/>
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

    <body name="sync_ref_left" pos="0.3 0 -0.1" mocap="true">
      <geom name="sync_ref_left" type="sphere" size="0.03" rgba="1 0 0 0.4" contype="0" conaffinity="0"/>
    </body>
    <body name="sync_ref_right" pos="-0.3 0 -0.1" mocap="true">
      <geom name="sync_ref_right" type="sphere" size="0.03" rgba="1 0 0 0.4" contype="0" conaffinity="0"/>
    </body>
    <body name="sync_tip_left" pos="0 0 0" mocap="true">
      <geom name="sync_tip_left" type="sphere" size="0.022" rgba="0 1 1 0.8" contype="0" conaffinity="0"/>
    </body>
    <body name="sync_tip_right" pos="0 0 0" mocap="true">
      <geom name="sync_tip_right" type="sphere" size="0.022" rgba="1 0.6 0 0.8" contype="0" conaffinity="0"/>
    </body>
    {dots_xml}
  </worldbody>

  <include file="{runtime_xml.name}"/>
</mujoco>
""",
        encoding="utf-8",
    )


def build_scene():
    """生成 runtime XML（电机、去碰撞、垫片纯视觉）→ 写/加载场景（参考点 P0 为常量）。"""
    runtime_xml = generate_runtime_xml()
    inject_motors(runtime_xml)
    strip_collisions(runtime_xml)
    disable_pad_collision(runtime_xml)
    write_scene_xml(runtime_xml, None)
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    return model, data, None


def parse_arms(model, data):
    """左右臂运行时句柄（关节/自由度/qpos/电机索引 + 端面相关 id）。"""
    arms = []
    for cfg in ARM_CONFIGS:
        body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["jacobi_body"])
        joint_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg["joint_names"]]
        dofs = [int(model.jnt_dofadr[j]) for j in joint_ids]
        qpos = [int(model.jnt_qposadr[j]) for j in joint_ids]
        act_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_motor") for n in cfg["joint_names"]]
        viz_id = ref_id = None                       # 场景可视球（仅完整场景模型才有）
        try:
            viz_id = model.body_mocapid[name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, f"sync_tip_{cfg['side']}")]
            ref_id = model.body_mocapid[name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, f"sync_ref_{cfg['side']}")]
        except ValueError:
            pass
        arms.append({
            "side": cfg["side"],
            "q_rest": cfg["q_rest"],
            "body_id": body_id,
            "joint_ids": joint_ids,
            "dofs": dofs,
            "qpos": qpos,
            "act_ids": act_ids,
            "viz_id": viz_id,
            "ref_id": ref_id,
        })
    return arms


def initial_pose(model, data, arms, nom_tips) -> dict:
    """初始位形（论文条件）：双臂端面收敛到共同参考点 P0，左臂 +1 mm / 右臂 -1 mm（x 向）。

    参考：r_ref,i(t) = P0 + (0, −0.02t, 0)（两臂相同）+ 各臂名义 rotvec（镜像臂各自保留）。
    """
    for a in arms:
        data.qpos[a["qpos"]] = a["q_rest"].copy()
        clamp_qpos(model, data, a["qpos"], a["joint_ids"])
        data.qvel[a["dofs"]] = 0.0
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)

    # 各自 IK 到 P0 ± 1 mm（2 mm 初始同步误差）
    targets = {
        "left": P0 + np.array([SYNC_ERR_X0 / 2, 0.0, 0.0]),
        "right": P0 - np.array([SYNC_ERR_X0 / 2, 0.0, 0.0]),
    }
    import ik as _ik
    _dmp_saved = _ik.DAMPING
    _ik.DAMPING = 0.05
    for _ in range(800):
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_forward(model, data)
        sols = []
        for a in arms:
            cur = get_ee_tip(data, a["body_id"])
            dq, _ = solve_position_ik(model, data, a["body_id"], targets[a["side"]], cur,
                                      a["dofs"], a["qpos"], a["q_rest"])
            sols.append((a, dq))
        for a, dq in sols:
            data.qpos[a["qpos"]] += dq
            clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    _ik.DAMPING = _dmp_saved
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    for a in arms:
        a["q_init"] = data.qpos[a["qpos"]].copy()

    # 参考：位置 = P0（两臂相同，论文式(18) 的 Δ_ref=0），姿态 = 各臂名义 rotvec
    ref = {}
    for a in arms:
        R = data.xmat[a["body_id"]].reshape(3, 3)
        ref[a["side"]] = np.concatenate([P0, np.zeros(3)])
        ref[a["side"]][3:] = _rotvec_of_R(R)
    return ref


def _rotvec_of_R(R):
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-6:
        return 0.5 * vee
    return 0.5 * theta / np.sin(theta) * vee


def place_viz(model, data, arms, ref):
    """可视化球：红=各臂参考点（随时间移动）、青/橙=左右端面。ref 为当前时刻参考位置(3)。"""
    for a in arms:
        if a["ref_id"] is not None:
            data.mocap_pos[a["ref_id"]] = ref[a["side"]][:3]
        if a["viz_id"] is not None:
            data.mocap_pos[a["viz_id"]] = get_ee_tip(data, a["body_id"])
    mujoco.mj_forward(model, data)


def save_screenshot(model, data, arms, lookat, tag: str):
    """渲染预览图（正面 + 俯视 + 3/4 视角三张）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    renderer = mujoco.Renderer(model, height=900, width=1200)
    shots = {
        f"exp1_scene_{tag}_front.png": dict(lookat=lookat, distance=2.6, elevation=-14, azimuth=180),
        f"exp1_scene_{tag}_top.png": dict(lookat=lookat, distance=2.2, elevation=-89, azimuth=180),
        f"exp1_scene_{tag}_persp.png": dict(lookat=lookat, distance=2.4, elevation=-30, azimuth=135),
    }
    for name, c in shots.items():
        cam = mujoco.MjvCamera()
        cam.lookat[:] = c["lookat"]
        cam.distance = c["distance"]
        cam.elevation = c["elevation"]
        cam.azimuth = c["azimuth"]
        renderer.update_scene(data, camera=cam)
        plt.imsave(FIG_DIR / name, renderer.render())
        print(f"📷 {FIG_DIR / name}")


# ======================================================================
# PD 对照组（论文式(7) 状态反馈控制；Experiment I 中 ηf=0 → u_force=0）
# ======================================================================
# 增益（按 "红色状态反馈振荡更多、蓝色 NMPCC 平滑" 的对比目标整定；
# K/Λ 用操作空间阻抗公式：平动刚度 600、阻尼 22、臂间协同 150、协同阻尼 8，
# 关节阻尼 0.8、零空间姿势弹簧 2（参考 ikforce 已验证的 OSC 配方））：
K_TRAC = np.array([600.0, 600.0, 600.0, 0.0, 0.0, 0.0])   # 位置跟踪刚度 (N/m)
D_TRAC = np.array([22.0, 22.0, 22.0, 1.0, 1.0, 1.0])      # 速度跟踪阻尼 (N·s/m)
K_COP = np.array([150.0, 150.0, 150.0, 0.0, 0.0, 0.0])    # 协同刚度（施加于相对偏差）
D_COP = np.array([8.0, 8.0, 8.0, 0.0, 0.0, 0.0])          # 协同阻尼
K_OSC_DQ = 0.8                    # 关节粘性阻尼（控制器侧，同 ikforce K_DQ）
K_OSC_PQ = 2.0                    # 零空间姿势弹簧（拉向 q_init，稳定冗余模态）
MAX_TASK_FORCE_PD = 500.0
MAX_NULL_TORQUE_PD = 10.0


def arm_cart_state(model, data, arm):
    """当前臂的六维笛卡尔状态 r(6)=[位置(3), rotvec(3)] 与速度 ṙ(6)，及 6×7 雅可比。"""
    tip = get_ee_tip(data, arm["body_id"])
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, tip, arm["body_id"])
    jac = np.vstack([jacp, jacr])[:, arm["dofs"]]          # (6,7)
    R = data.xmat[arm["body_id"]].reshape(3, 3)
    r6 = np.concatenate([tip, _rotvec_of_R(R)])
    rd6 = jac @ data.qvel[arm["dofs"]]
    return r6, rd6, jac


def apply_state_feedback(model, data, arms, ref6, vref6):
    """[PD —— 论文式(7) 状态反馈控制（Experiment I 对照组）]

    ui = u_comp,i + u_trac,i + u_coop,i + u_force,i
    - u_comp,i = C + G：MuJoCo qfrc_bias 前馈（符号实测确认：+qfrc_bias 抵消重力/科氏）
    - u_trac,i = JᵀΛ[K_trac·(r_i−r_ir) + D_trac·(ṙ_i−ṙ_ir)]：平动位置阻抗。
      操作空间惯量 Λ=(J M⁻¹ Jᵀ)⁻¹ 做惯性归一化（论文式(4) 的 JᵀF 的标准实现；
      本轻量臂不做 Λ 加权时 Jᵀ 映射会把刚度放大成 >400 rad/s 高频模态，闭环必然发散）
    - u_coop,i = Jᵀ[K_cop·((r_i−r_j)−(r_ir−r_jr)) + D_cop·(ṙ_i−ṙ_j)]：相对偏差协同（论文式5）
    - u_force,i = 0（ηf=0：无量测力回路）
    - 附加: −K_DQ·q̇（关节阻尼）+ N·[K_PQ·(q_init−q) − K_DQ·q̇]（零空间姿势弹簧，不干扰末端）
    返回 u(14)（左右臂拼接）。同步误差口径与 NMPCC 完全一致（相对参考）。
    """
    tips = {a["side"]: get_ee_tip(data, a["body_id"]) for a in arms}
    lam_map = {}
    for a in arms:
        dofs = a["dofs"]
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, tips[a["side"]], a["body_id"])
        jac = jacp[:, dofs]
        m = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, m)
        mm = m[np.ix_(dofs, dofs)]
        m_inv = np.linalg.inv(mm)
        lam_inv = jac @ m_inv @ jac.T
        if np.linalg.det(lam_inv) > 1e-6:
            lam = np.linalg.inv(lam_inv)
        else:
            lam = np.linalg.pinv(lam_inv, rcond=1e-6)
        x_dot = jac @ data.qvel[dofs]
        lam_map[a["side"]] = (jac, lam, m_inv, x_dot)

    f_cop = {}
    for a in arms:
        other = next(b for b in arms if b["side"] != a["side"])
        e_cop = (tips[a["side"]] - tips[other["side"]]) \
            - (ref6[a["side"]][:3] - ref6[other["side"]][:3])
        v_cop = lam_map[a["side"]][3] - lam_map[other["side"]][3]
        f_cop[a["side"]] = -K_COP[:3] * e_cop - D_COP[:3] * v_cop

    taus = []
    for a in arms:
        dofs = a["dofs"]
        jac, lam, m_inv, x_dot = lam_map[a["side"]]
        e_pos = ref6[a["side"]][:3] - tips[a["side"]]
        v_des = vref6[a["side"]][:3]
        force = lam @ (K_TRAC[:3] * e_pos - D_TRAC[:3] * (x_dot - v_des)) + f_cop[a["side"]]
        fn = np.linalg.norm(force)
        if fn > MAX_TASK_FORCE_PD and fn > 1e-9:
            force = force / fn * MAX_TASK_FORCE_PD
        tau = data.qfrc_bias[dofs].copy() + jac.T @ force - K_OSC_DQ * data.qvel[dofs]
        null_proj = np.eye(len(dofs)) - jac.T @ lam @ jac @ m_inv
        tau_posture = K_OSC_PQ * (a["q_init"] - data.qpos[a["qpos"]]) - K_OSC_DQ * data.qvel[dofs]
        tau_null = null_proj @ tau_posture
        nn = np.linalg.norm(tau_null)
        if nn > MAX_NULL_TORQUE_PD and nn > 1e-9:
            tau_null = tau_null / nn * MAX_NULL_TORQUE_PD
        taus.append(tau + tau_null)
    return np.concatenate(taus)


# ======================================================================
# 闭环实验（NMPCC；PD 由用户填好 apply_state_feedback 后接入）
# ======================================================================
def _refs_at(ref0, t):
    """当前时刻参考：位置 + 旋转向量（y 向 -20 mm/s 平移，姿态保持）。"""
    off6 = np.array([0.0, V_REF_Y * t, 0.0, 0.0, 0.0, 0.0])
    v6 = np.array([0.0, V_REF_Y, 0.0, 0.0, 0.0, 0.0])
    return {s: ref0[s] + off6 for s in ref0}, {s: v6 for s in ref0}


def run_experiment(model, data, arms, ref0, args):
    import nmpcc_sync as ns

    arm_cfgs = [{"body_id": a["body_id"]} for a in arms]
    fk = ns.mujoco_fk_symbols(model, arm_cfgs, EE_TIP_Z)

    per_arm = np.array([30.0, 30.0, 10.0, 10.0, 10.0, 10.0, 10.0])   # 弱关节重罚（防边界解）
    nmpc = None
    if args.controller == "nmpcc":
        cfg = ns.NMPCCSyncConfig(
            N=args.N, Tp=args.Tp,
            Q=args.qscale * np.array([1500.0, 1500.0, 1000.0, 500.0, 500.0, 500.0]),
            Q_sync=np.array([args.qs, args.qs, args.qs, 0.0, 0.0, 0.0]),
            eta_dev=args.eta_dev,
            eta_u=args.eta_u,
            P_mul=2.0,
            qp_solver=args.qp,
        )
        t0 = time.perf_counter()
        nmpc = ns.NMPCCSync(cfg, fk, umin=-TORQUE_LIMIT * np.ones(14),
                            umax=TORQUE_LIMIT * np.ones(14))
        print(f"✅ NMPCCSync 构建完成 {time.perf_counter() - t0:.1f} s (N={cfg.N}, Tp={cfg.Tp})")
    else:
        print("ℹ️ PD 对照组（论文式7）：apply_state_feedback 已实现，本会话跑的是状态反馈 PD")

    def joint_state():
        return np.concatenate([
            data.qpos[arms[0]["qpos"]], data.qvel[arms[0]["dofs"]],
            data.qpos[arms[1]["qpos"]], data.qvel[arms[1]["dofs"]],
        ])

    def plant_params():
        m = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, m)
        out = []
        for a in arms:
            MM = m[np.ix_(a["dofs"], a["dofs"])]
            v = data.qvel.copy()
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            G = data.qfrc_bias[a["dofs"]].copy()
            data.qvel[:] = v
            mujoco.mj_forward(model, data)
            out.extend([MM.reshape(-1), G])
        return np.concatenate(out)

    def gravity_comp():
        """（已移除每拍使用——论文方法无 u_ref 机制，重力保持由模型 G 自洽。）
        若需显式重力补偿可直接读 data.qfrc_bias。"""
        out = []
        for a in arms:
            v = data.qvel.copy()
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            g = data.qfrc_bias[a["dofs"]].copy()
            data.qvel[:] = v
            mujoco.mj_forward(model, data)
            out.append(g)
        return np.concatenate(out)

    N = int(round(args.duration / DT))
    log = {"t": [], "esx": [], "esy": [], "tipL": [], "tipR": [], "tau": [], "solve_ms": [], "tsolve": []}
    u_last = np.zeros(14)
    sat = 0
    for k in range(N):
        t = k * DT
        mujoco.mj_forward(model, data)
        refs, vrefs = _refs_at(ref0, t)
        if k % args.ctrl_every == 0:
            if args.controller == "nmpcc":
                p_coef = np.zeros(ns.NP)
                p_coef[0:112] = plant_params()
                p_coef[112:118] = refs["left"]
                p_coef[118:124] = refs["right"]
                p_coef[124:130] = vrefs["left"]
                p_coef[130:136] = vrefs["right"]
                p_coef[136:150] = np.zeros(14)     # u_ff 通道保留（0=仅用 yref 基线，防双份补偿）
                p_coef[150:164] = np.concatenate([model.dof_damping[arms[0]["dofs"]],
                                                  model.dof_damping[arms[1]["dofs"]]])  # Dv 系数
                t0s = time.perf_counter()
                u_last = nmpc.solve(joint_state(), p_coef, u_ref=gravity_comp())
                log["solve_ms"].append((time.perf_counter() - t0s) * 1e3)
                log["tsolve"].append(t)
            else:
                t0s = time.perf_counter()
                u_last = apply_state_feedback(model, data, arms, refs, vrefs)
                log["solve_ms"].append((time.perf_counter() - t0s) * 1e3)
                log["tsolve"].append(t)
        for a in arms:
            low, high = model.actuator_ctrlrange[a["act_ids"]].T
            if np.any(u_last[a["act_ids"]] < low - 1e-9) or np.any(u_last[a["act_ids"]] > high + 1e-9):
                sat += 1
            data.ctrl[a["act_ids"]] = u_last[a["act_ids"]]
        mujoco.mj_step(model, data)
        if k % 5 == 0:
            tL = get_ee_tip(data, arms[0]["body_id"])
            tR = get_ee_tip(data, arms[1]["body_id"])
            log["t"].append(t + DT)
            log["esx"].append((tL[0] - tR[0]) - (refs["left"][0] - refs["right"][0]))
            log["esy"].append((tL[1] - tR[1]) - (refs["left"][1] - refs["right"][1]))
            log["tipL"].append(tL)
            log["tipR"].append(tR)
            log["tau"].append(u_last.copy())

    tts = np.array(log["t"])
    ex = np.array(log["esx"]); ey = np.array(log["esy"])
    sm = np.array(log["solve_ms"])
    print(f"--- Experiment I {'NMPCC' if args.controller == 'nmpcc' else 'PD'} {tts[-1]:.1f} s 报表 ---")
    print(f"初始同步误差: x = {ex[0]*1000:.2f} mm | y = {ey[0]*1000:.2f} mm（目标 2 mm / 0 mm）")
    ok = np.abs(ex) < 0.5e-3
    if np.any(ok):
        print(f"x 向同步误差到 <0.5 mm 用时: {tts[ok][0]:.2f} s | 末端 {ex[-1]*1000:.3f} mm")
    else:
        print(f"x 向同步误差未收敛到 <0.5 mm: 末端 {ex[-1]*1000:.3f} mm")
    print(f"y 向同步误差: RMS={np.sqrt((ey**2).mean())*1000:.2f} mm 峰值={np.abs(ey).max()*1000:.2f} mm")
    if len(sm):
        ctrl_name = "NMPC" if args.controller == "nmpcc" else "PD"
        print(f"{ctrl_name} 求解: 均值 {sm.mean():.2f} ms 峰值 {sm.max():.2f} ms（论文 0.28 ms/<1 ms）")
    print(f"力矩限幅命中: {sat} 拍 | 末端 t={tts[-1]:.2f}s: L={np.round(log['tipL'][-1],4)} R={np.round(log['tipR'][-1],4)}")

    FIG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = "nmpcc" if args.controller == "nmpcc" else "pd"
    if not args.no_csv:
        import csv
        path = FIG_DIR / f"exp1_sync_{tag}_{ts}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "esync_x", "esync_y", "tipLx", "tipLy", "tipLz",
                        "tipRx", "tipRy", "tipRz", *[f"tau{i+1}" for i in range(14)], "solve_ms"])
            for i in range(len(tts)):
                w.writerow([tts[i], ex[i], ey[i], *log["tipL"][i], *log["tipR"][i],
                            *log["tau"][i], sm[i] if len(sm) else ""])
        print(f"📄 {path}")
    if not args.no_plot:
        plot_fig10(tts, ex, ey, log["tsolve"], sm, FIG_DIR / f"exp1_sync_fig10_{tag}_{ts}", args.overlay_pd)


def plot_fig10(tts, ex, ey, tsolve, sm, out_prefix: Path, pd_csv=None):
    """Fig.10 同款三面板：(a) x 同步误差 (b) y 同步误差 (c) 每拍求解耗时。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(8, 9))
    axs[0].plot(tts, ex * 1000, "tab:blue", lw=1.4, label="NMPCC")
    axs[1].plot(tts, ey * 1000, "tab:blue", lw=1.4, label="NMPCC")
    if pd_csv and Path(pd_csv).exists():
        import csv
        rows = list(csv.DictReader(open(pd_csv, encoding="utf-8")))
        t = np.array([float(r["t"]) for r in rows])
        axs[0].plot(t, np.array([float(r["esync_x"]) for r in rows]) * 1000,
                    "tab:red", lw=1.1, ls="--", label="PD (Eq.7)")
        axs[1].plot(t, np.array([float(r["esync_y"]) for r in rows]) * 1000,
                    "tab:red", lw=1.1, ls="--", label="PD (Eq.7)")
        axs[0].set_title("(a) Synchronous error in x (blue=NMPCC, red=PD)")
        axs[1].set_title("(b) Synchronous error in y")
    else:
        axs[0].set_title("(a) Synchronous error in x [mm]")
        axs[1].set_title("(b) Synchronous error in y [mm]")
    axs[2].plot(tsolve, sm, "tab:blue", lw=0.8)
    axs[2].set_title("(c) Time consumption per control cycle [ms]")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.set_xlabel("t [s]")
    axs[0].legend(fontsize=8)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_prefix) + ".svg", format="svg")
    fig.savefig(str(out_prefix) + ".png", dpi=160)
    print(f"📈 {out_prefix}.svg (+png)")


# ======================================================================
# 主流程（当前只做场景；NMPC 闭环在第 2/3 步接入）
# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="论文 Experiment I（运动同步）场景与复用栈。")
    parser.add_argument("--check", action="store_true", help="只构建加载，不渲染")
    parser.add_argument("--screenshot", action="store_true", help="渲染预览图到 results/exp1_sync/")
    parser.add_argument("--viewer", action="store_true", help="打开交互窗口（运动学保持姿态）")
    parser.add_argument("--run", action="store_true", help="跑闭环实验（1 kHz，出 CSV+Fig.10 图）")
    parser.add_argument("--controller", choices=["nmpcc", "pd"], default="nmpcc",
                        help="pd 需先自己实现 apply_state_feedback（论文式7）")
    parser.add_argument("--duration", type=float, default=DURATION, help="仿真时长 (s)")
    parser.add_argument("--N", type=int, default=20, help="NMPC 视界步数")
    parser.add_argument("--Tp", type=float, default=0.1, help="NMPC 视界时长 (s，论文 100 ms)")
    parser.add_argument("--ctrl_every", type=int, default=1, help="每几步解一次 NMPC（1 = 1 kHz）")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--no_csv", action="store_true")
    parser.add_argument("--overlay_pd", default=None, help="PD 结果 CSV 路径（叠画红色对比）")
    parser.add_argument("--qscale", type=float, default=0.02, help="跟踪 Q 缩放（RTI 增益上限内：0.02 稳定/实测最优）")
    parser.add_argument("--qs", type=float, default=20.0, help="同步残差权重（x/y/z；论文 ηcop 耦合语义）")
    parser.add_argument("--eta_u", type=float, default=0.01, help="式(31) ηu")
    parser.add_argument("--eta_dev", type=float, default=40.0)
    parser.add_argument("--qp", default="PARTIAL_CONDENSING_HPIPM",
                        choices=["PARTIAL_CONDENSING_HPIPM", "FULL_CONDENSING_HPIPM",
                                 "FULL_CONDENSING_QPOASES"],
                        help="QP 求解器（论文=qpOASES；本构建 QPOASES 对约束支持不全时用 HPIPM）")
    args = parser.parse_args()

    FIG_DIR.mkdir(exist_ok=True)
    model, data, nom_tips = build_scene()
    arms = parse_arms(model, data)
    ref = initial_pose(model, data, arms, nom_tips)
    place_viz(model, data, arms, ref)

    tL = get_ee_tip(data, arms[0]["body_id"])
    tR = get_ee_tip(data, arms[1]["body_id"])
    sync_x = (tL[0] - tR[0]) - (ref["left"][0] - ref["right"][0])
    sync_y = (tL[1] - tR[1]) - (ref["left"][1] - ref["right"][1])
    print(f"✅ Experiment I 场景已加载: nq={model.nq} nv={model.nv} nbody={model.nbody} ngeom={model.ngeom}")
    print(f"   共同参考点 P0 = {np.round(P0, 4)}（两臂参考相同，论文式(18) Δ_ref=0）")
    print(f"   起始端面: L={np.round(tL, 4)} R={np.round(tR, 4)}")
    print(f"   初始同步误差 e_cop: x = {sync_x*1000:.2f} mm（目标 {SYNC_ERR_X0*1000:.0f} mm）"
          f"| y = {sync_y*1000:.2f} mm（目标 0）")
    print(f"   参考速度 = {V_REF_Y*1000:.0f} mm/s (y) | 时长默认 {DURATION:.0f} s "
          f"(行程 {abs(V_REF_Y)*DURATION*1000:.0f} mm) | 控制周期 {DT*1000:.0f} ms | 力矩限幅 ±{TORQUE_LIMIT:.0f} N·m")

    if args.check:
        return
    if args.run:
        run_experiment(model, data, arms, ref, args)
        return
    if args.screenshot:
        save_screenshot(model, data, arms, P0, "preview")
        return
    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = P0
            viewer.cam.distance = 2.6
            viewer.cam.elevation = -14
            viewer.cam.azimuth = 180
            while viewer.is_running():
                data.qvel[:] = 0.0
                data.qacc[:] = 0.0
                mujoco.mj_forward(model, data)
                place_viz(model, data, arms, ref)
                viewer.sync()
                time.sleep(DT)
        return
    parser.print_help()


if __name__ == "__main__":
    main()