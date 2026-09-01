#!/usr/bin/env python3
"""ikforce_grasp.py —— 力矩级(OSC)双臂夹箱实验（ikforce + 箱子 + 平放对夹姿态）

把两条线合起来：
  1. ikforce.py 的力矩级笛卡尔位置控制：任务力 JᵀΛ(Kp·e − Kd·ẋ)（操作空间惯量加权，
     零空间投影 N = I − JᵀΛJM⁻¹，投影后姿势力矩软限幅 MAX_NULL_TORQUE）
  2. 夹箱姿态：grasp_pose_static 收敛出的"平放对夹"姿态作为
     - 初始关节状态（机器人直接以持箱位姿起步）
     - 零空间姿势目标 q_grasp（关节弹簧拉向"我们做好的姿势"）
  3. 在任务力上叠加每侧期望夹持力 F_GRASP（向箱内）：
        F_task = Λ(Kp·e − Kd·ẋ) + F_des,   F_des = −sign·F_GRASP·x̂
     接触后由摩擦把 1.5kg 箱子托住；本步验证"能不能夹住"。

摩擦设置（有效 μ = 箱×垫 相乘）：
  箱体滑动摩擦 0.6（正常），垫片（手部顶端）2.5（高摩擦），有效 μ≈1.5

位置跟踪（默认）：夹稳 1s 后，参考轨迹让目标整体移动——抬升 12cm → 平移 12cm → x 向
正弦摆动（分段 smoothstep，起止零加速度），控制器带 v_des 速度前馈跟踪，箱子跟随参考走。
--hold_only 可回到纯静态保持（对照）。
运行:
    python ikforce_grasp.py --check
    python ikforce_grasp.py --headless --steps 5000                 # 夹持 + 移动轨迹跟踪
    python ikforce_grasp.py --headless --steps 5000 --hold_only     # 只静态夹持（对照）
    python ikforce_grasp.py --force 20                              # 换夹持力
    python ikforce_grasp.py                                         # viewer 看动态跟踪
"""

from __future__ import annotations

import argparse
import ctypes
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
    clamp_qpos,
    generate_runtime_xml,
    get_ee_tip,
    name_to_id,
)
from ikforce import (
    GRAVITY,
    IK_DT,
    K_DQ,
    K_POS,
    K_PQ,
    MAX_NULL_TORQUE,
    MAX_TASK_FORCE,
    inject_motors,
    strip_collisions,
)
from grasp_pose_static import (
    BOX_HALF,
    BOX_MASS,
    GRIP_CLEARANCE,
    solve_grasp_pose,
)

# ---------- 夹持参数 ----------
F_GRASP = 25.0                      # 每侧期望夹持力 (N)，向箱内（μ≈1.5、N≥4.9N 即静态够，取 25N 留 4~5 倍裕度）
BOX_FRICTION = "0.6 0.05 0.005"    # 箱体：正常摩擦（滑动 0.6）
PAD_FRICTION = "2.5 0.03 0.001"    # 垫片(手部顶端)：高摩擦，有效 μ = 0.6×2.5 = 1.5
LEVEL_K = 60.0                      # 箱体水平控制增益 (N/rad)：绕 y 俯仰 → 差动 z 向夹持力
LEVEL_MAX = 6.0                     # 差动力上限 (N)，防水平修正顶掉夹持
GRASP_Z = 0.55                      # 抓取/搬运高度（箱中心 z, m）：高处水平搬运"不翘起"行程最大（扫表 0.55 档）
K_D_GRASP = 55.0                    # 抓取回路笛卡尔阻尼 (N·s/m)：ζ=Kd/(2√Kp)≈0.71（ikforce 的 30 偏欠阻尼会震荡）
K_COP = 400.0                       # 臂间协同刚度 (N/m)：论文式(5) u_cop=JᵀK_cop(r_i−r_j)+D_cop(ṙ_i−ṙ_j)
D_COP = 20.0                        # 臂间协同阻尼 (N·s/m)
K_FORCE = 0.6                       # 夹持力闭环增益：指令 = f_ref + K·(f_ref − f_meas)（论文式(6) 力反馈）
F_FILT_ALPHA = 0.2                  # 接触力测量低通系数（EMA，防闭环抖振）
USE_NULLSPACE = True               # 零空间姿势用"做好的夹持姿态"
SCENE_XML = BASE_DIR / "ikforce_grasp_scene.xml"


def set_pad_friction(runtime_xml: Path, friction: str) -> None:
    """把两端 ee_pad 垫片的摩擦调高、接触调硬（硬接触才不会有摩擦"渗滑"）。"""
    tree = ET.parse(runtime_xml)
    root = tree.getroot()
    n = 0
    for body in root.iter("body"):
        for geom in body.findall("geom"):
            if (geom.get("name", "")).endswith("_ee_pad"):
                geom.set("friction", friction)
                geom.set("solref", "0.0005 1")          # 硬接触：timeconst 5e-4 s
                geom.set("solimp", "0.999 0.999 0.001")
                n += 1
    tree.write(runtime_xml, encoding="utf-8", xml_declaration=True)
    print(f"✅ 已设置 {n} 个垫片 friction={friction}, solref=0.0005 1")


def build_scene() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """生成 runtime XML（+力矩电机，去掉 link/手/指碰撞，垫片高摩擦）并写夹箱场景。"""
    runtime_xml = generate_runtime_xml()
    inject_motors(runtime_xml)            # 14 个力矩电机
    strip_collisions(runtime_xml)         # 移除 *_collision（含手/指网格与粗碰撞体）；垫片保留
    set_pad_friction(runtime_xml, PAD_FRICTION)
    SCENE_XML.write_text(
        f"""<mujoco model="ikforce_grasp_scene">
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
    <geom name="back_wall" type="box" pos="0 0.95 0.78" size="2.6 0.025 0.80" material="wall_mat"
          contype="0" conaffinity="0"/>

    <!-- 箱子：freejoint，若不夹住会直接下落；硬接触(solref 收紧)防摩擦渗滑 -->
    <body name="box" pos="0.5 0 0.72">
      <freejoint name="box_freejoint"/>
      <geom name="box" type="box" size="{BOX_HALF[0]} {BOX_HALF[1]} {BOX_HALF[2]}"
            material="box_mat" mass="{BOX_MASS}"
            friction="{BOX_FRICTION}" condim="3" contype="1" conaffinity="1"
            solref="0.0005 1" solimp="0.999 0.999 0.001"/>
    </body>
  </worldbody>

  <include file="{runtime_xml.name}"/>
</mujoco>
""",
        encoding="utf-8",
    )
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    model.opt.gravity[:] = GRAVITY
    data = mujoco.MjData(model)
    return model, data


def solve_torque_grasp(
    model,
    data,
    arm: dict,
    target_pos: np.ndarray,
    f_des: np.ndarray,
    v_des: np.ndarray | None = None,
    use_nullspace: bool = True,
    f_extra: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """力矩级夹持求解：操作空间阻抗(+速度前馈做位置跟踪) + 期望夹持力 + 重力/科氏补偿 (+零空间持姿弹簧)。

    F_task = Λ·(Kp·e − Kd·(ẋ − v_des)) + f_des   （Λ 任务空间惯量加权，同 ikforce/opspace；
    v_des 为参考轨迹速度 → 位置跟踪不掉队；None 时退化回纯保持）
    零空间: N = I − JᵀΛJM⁻¹，姿势目标 = 平放对夹关节角 q_grasp（软限幅 MAX_NULL_TORQUE）
    """
    body_id = arm["body_id"]
    dofs = arm["dofs"]

    tip = get_ee_tip(data, body_id)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, tip, body_id)
    jac = jacp[:, dofs]                     # 平移雅可比 (3,7)

    m = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, m)
    mm = m[np.ix_(dofs, dofs)]
    m_inv = np.linalg.inv(mm)               # 关节空间 M⁻¹ (7×7)

    lam_inv = jac @ m_inv @ jac.T
    if np.linalg.det(lam_inv) > 1e-6:
        lam = np.linalg.inv(lam_inv)
    else:
        lam = np.linalg.pinv(lam_inv, rcond=1e-6)

    e_pos = target_pos - tip
    x_dot = jac @ data.qvel[dofs]
    if v_des is None:
        v_des = np.zeros(3)
    if f_extra is None:
        f_extra = np.zeros(3)
    # 阻抗+速度前馈 + 期望夹持力(力闭环) + 臂间协同项 u_cop（论文式5）
    force = lam @ (K_POS * e_pos - K_D_GRASP * (x_dot - v_des)) + f_des + f_extra

    f_norm = np.linalg.norm(force)
    if f_norm > MAX_TASK_FORCE and f_norm > 1e-9:
        force = force / f_norm * MAX_TASK_FORCE

    tau = (
        data.qfrc_bias[dofs].copy()
        + jac.T @ force
        - K_DQ * data.qvel[dofs]
    )

    if use_nullspace:
        null_proj = np.eye(len(dofs)) - jac.T @ lam @ jac @ m_inv
        tau_posture = (
            K_PQ * (arm["q_grasp"] - data.qpos[arm["qpos"]])
            - K_DQ * data.qvel[dofs]
        )
        tau_null = null_proj @ tau_posture
        n_null = np.linalg.norm(tau_null)
        if n_null > MAX_NULL_TORQUE and n_null > 1e-9:
            tau_null = tau_null / n_null * MAX_NULL_TORQUE   # 轻量臂投影放大保护（同 ikforce）
        tau = tau + tau_null

    return tau, e_pos


# ---------- 参考轨迹（默认：夹稳后左右来回平稳移动；--with_lift 时先抬升再来回） ----------
TRACK_SEGS = [
    ("HOLD", 1.0, np.zeros(3), None),                                # 先夹稳 1s
    ("LIFT", 3.0, np.array([0.0, 0.0, 0.12]), None),                 # --with_lift 时启用：抬升 12cm
    ("SWAY", 1e9, np.zeros(3), dict(axis=0, amp=0.03, half=2.5)),    # 左右来回：±0.03m 默认（命令行可调），每程 2.5s，无限
]


def make_reference(lift: bool = True, amp: float = 0.03, half: float = 3.5, axis: int = 0):
    """返回 ref(t)/vdes(t) 采样器。

    - 平移段用 smoothstep（起止零加速度）；
    - SWAY 段是正弦来回：±amp 沿 axis（0=x 1=y 2=z），半周期 half 秒
      （回中、掉头全程平滑，峰值速度也仅 amp·π/half，不会出现"回原位飞快"的感觉）；
    - lift=False 时跳过 LIFT 段（不占时间）。
    """
    segs = [s for s in TRACK_SEGS if lift or s[0] != "LIFT"]
    sway_cfg = dict(axis=axis, amp=amp, half=half)

    def sample(t: float) -> tuple[np.ndarray, np.ndarray]:
        r, v = np.zeros(3), np.zeros(3)
        t0 = 0.0
        for _, dur, d, cfg in segs:
            t1 = t0 + dur
            cfg = sway_cfg if cfg is not None and "half" in cfg else cfg   # 用命令行幅度/周期
            if t >= t1:                                   # 该段已走完
                if cfg is None:
                    r = r + d
                elif "freq" in cfg:                       # 正弦段终点值
                    r[cfg["axis"]] += cfg["amp"] * np.sin(2*np.pi*cfg["freq"]*dur)
                t0 = t1
                continue
            if cfg is not None and "half" in cfg:         # —— 左右来回（正弦，周期 2*half）——
                tt = t - t0
                ph = np.pi * (tt % (2 * cfg["half"])) / cfg["half"]
                r[cfg["axis"]] += cfg["amp"] * np.sin(ph)
                v[cfg["axis"]] += cfg["amp"] * (np.pi / cfg["half"]) * np.cos(ph)
            elif cfg is not None:                         # 正弦摆动
                tt = t - t0
                r[cfg["axis"]] += cfg["amp"] * np.sin(2*np.pi*cfg["freq"]*tt)
                v[cfg["axis"]] += cfg["amp"] * 2*np.pi*cfg["freq"] * np.cos(2*np.pi*cfg["freq"]*tt)
            else:                                         # smoothstep 平移
                tau = (t - t0) / dur
                s = tau * tau * (3.0 - 2.0 * tau)
                sd = 6.0 * tau * (1.0 - tau) / dur
                r = r + d * s
                v = v + d * sd
            return r, v
        return r, v
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="力矩级(OSC)双臂夹箱实验。")
    parser.add_argument("--check", action="store_true", help="Only build and load the model")
    parser.add_argument("--headless", action="store_true", help="无 GUI：夹持 N 步后打印持稳报表")
    parser.add_argument("--steps", type=int, default=5000, help="Headless 总步数（0.002s/步）")
    parser.add_argument("--no_nullspace", action="store_true", help="关闭持姿零空间（对照）")
    parser.add_argument("--force", type=float, default=F_GRASP, help="每侧期望夹持力 (N)，A/B 用")
    parser.add_argument("--box_z", type=float, default=GRASP_Z,
                        help="抓取/搬运高度：箱中心 z (m)，默认最高档 0.55（平搬行程最大）")
    parser.add_argument("--sway_axis", type=str, default="x", choices=["x", "y", "z"],
                        help="左右来回的移动轴（默认 x=沿夹持轴；y=垂直夹持轴搬运，论文实验2 构型）")
    parser.add_argument("--amp", type=float, default=0.03,
                        help="左右来回幅度 ±m（默认 0.03：平放垫片在反复掉头下实测稳定上限≈±0.03，更大必翻）")
    parser.add_argument("--half", type=float, default=3.5,
                        help="正弦半周期 s（默认 3.5，越大越慢越稳）")
    parser.add_argument("--hold_only", action="store_true", help="只做静态保持，不加移动轨迹（对照）")
    parser.add_argument("--no_level", action="store_true", help="关闭箱子水平控制（差动 z 力），A/B 用")
    parser.add_argument("--no_coop", action="store_true",
                        help="关闭臂间协同项 u_cop（论文式5 对照：无协同的独立双臂）")
    parser.add_argument("--no_force_fb", action="store_true",
                        help="关闭夹持力闭环（回到开环 25N 指令），A/B 用")
    parser.add_argument("--dist_t", type=float, default=6.0,
                        help="扰动时刻 (s)：给指定臂端部一个外力脉冲（论文实验2 的扰动对比）")
    parser.add_argument("--dist_f", type=float, default=30.0, help="扰动力大小 (N)")
    parser.add_argument("--dist_dur", type=float, default=0.5, help="扰动持续 (s)")
    parser.add_argument("--dist_axis", type=str, default="x", choices=["x", "y", "z"], help="扰动力方向")
    parser.add_argument("--dist_side", type=str, default="left", choices=["left", "right"], help="扰动哪条臂")
    parser.add_argument("--record", type=str, default="",
                        help="保存时间序列到 .npz（t/err/follow/tau/力，供 plot_grasp_figs.py 画图）")
    parser.add_argument("--grasp_raise", type=float, default=0.0,
                        help="抓取点相对箱中心抬高 (m)：抬高后质心在接触点下方，俯仰/侧倾被动稳定")
    parser.add_argument("--preload", type=float, default=0.3,
                        help="起步预加载 (s)：该时段冻结箱子让夹持力建满再放开，掐掉接触瞬态")
    parser.add_argument("--angle", type=float, default=0.0,
                        help="垫片法向俯仰角 (deg)：>0 时内倾+上托防抬升侧倾；只做水平搬运时 0=纯平放")
    parser.add_argument("--with_lift", action="store_true",
                        help="启用抬升段（只做水平搬运时不用，垫片保持 0° 平放即可）")
    args = parser.parse_args()

    use_nullspace = USE_NULLSPACE and not args.no_nullspace
    f_grasp = args.force
    alpha = np.radians(args.angle)
    sway_axis_idx = {"x": 0, "y": 1, "z": 2}[args.sway_axis]

    model, data = build_scene()
    box_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_geom_id = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    box_qpos = int(model.jnt_qposadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")])
    box_qvel = int(model.jnt_dofadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")])

    arms = []
    for cfg in ARM_CONFIGS:
        body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["jacobi_body"])
        joint_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg["joint_names"]]
        dofs = [int(model.jnt_dofadr[j]) for j in joint_ids]
        qpos = [int(model.jnt_qposadr[j]) for j in joint_ids]
        act_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_motor") for n in cfg["joint_names"]]
        sign = 1.0 if cfg["side"] == "left" else -1.0
        arms.append(
            {
                "side": cfg["side"],
                "sign": sign,
                "q_rest": cfg["q_rest"],
                "body_id": body_id,
                "joint_ids": joint_ids,
                "dofs": dofs,
                "qpos": qpos,
                "act_ids": act_ids,
                "pad_id": name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"openarmx_{cfg['side']}_ee_pad"),
                "q_grasp": np.zeros(7),
            }
        )

    # ---------- 复位：机器人回休息位，箱子放端面中点上方 ----------
    def reset_robot() -> None:
        for arm in arms:
            data.qpos[arm["qpos"]] = arm["q_rest"].copy()
            clamp_qpos(model, data, arm["qpos"], arm["joint_ids"])
            data.qvel[arm["dofs"]] = 0.0
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_forward(model, data)
        mid = np.mean([get_ee_tip(data, arm["body_id"]) for arm in arms], axis=0)
        data.qpos[box_qpos : box_qpos + 3] = [mid[0], mid[1], args.box_z]
        data.qpos[box_qpos + 3 : box_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[box_qvel : box_qvel + 6] = 0.0
        mujoco.mj_forward(model, data)

    reset_robot()
    print("✅ 力矩级夹箱场景已加载")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")

    if args.check:
        return

    # ---------- 运动学收敛出"平放对夹"姿态，作为初始状态 + 零空间姿势目标 ----------
    q_grasp = solve_grasp_pose(model, data, arms, box_id, iters=800, tilt=alpha)
    for arm in arms:
        arm["q_grasp"] = q_grasp[arm["side"]]
        data.qpos[arm["qpos"]] = q_grasp[arm["side"]]
        data.qvel[arm["dofs"]] = 0.0
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    box_start = data.xpos[box_id].copy()

    # 夹持基准目标：初始箱面中心；位置跟踪模式下每步叠加参考偏移 off(t)，
    # 静态保持下目标就是它自己（箱子动了目标不追，靠夹持力真托住才叫夹住）
    grasp_targets = {
        arm["side"]: np.array(
            [float(box_start[0] + arm["sign"] * (BOX_HALF[0] + GRIP_CLEARANCE)),
             float(box_start[1]), float(box_start[2] + args.grasp_raise)],
            dtype=np.float64,
        )
        for arm in arms
    }
    f_des = {
        arm["side"]: np.array(
            [-arm["sign"] * f_grasp * np.cos(alpha), 0.0, f_grasp * np.sin(alpha)]
        )
        for arm in arms
    }
    ref_fn = None if args.hold_only else make_reference(
        lift=args.with_lift, amp=args.amp, half=args.half, axis=sway_axis_idx)
    print(
        f"🔧 期望夹持力 = {f_grasp} N/侧 | 垫片角 = {np.degrees(alpha):.0f}° | 轨迹 = "
        + ("静态保持" if args.hold_only else
           (f"来回平稳移动: {args.sway_axis} 向 ±{args.amp}m 正弦（半周期 {args.half}s）" if not args.with_lift
            else f"抬起0.12m → {args.sway_axis} 向来回 ±{args.amp}m"))
    )

    def pad_contact_forces() -> dict[str, float]:
        """各臂垫片与箱体接触的法向力（N），mj_contactForce 第 0 分量。"""
        out = {arm["side"]: 0.0 for arm in arms}
        force = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 == box_geom_id or c.geom2 == box_geom_id:
                for arm in arms:
                    if c.geom1 == arm["pad_id"] or c.geom2 == arm["pad_id"]:
                        mujoco.mj_contactForce(model, data, i, force)
                        out[arm["side"]] += abs(force[0])
        return out

    def box_tilt_rad() -> np.ndarray:
        """箱体倾角向量 (axis·angle, rad)：分量[0]=绕x(侧倾) [1]=绕y(俯仰/翘起) [2]=绕z。"""
        rv = np.zeros(3)
        mujoco.mju_quat2Vel(rv, data.qpos[box_qpos + 3 : box_qpos + 7], 1.0)
        return rv

    use_level = not args.no_level
    use_coop = not args.no_coop
    use_force_fb = not args.no_force_fb
    f_filt = {arm["side"]: float(f_grasp) for arm in arms}   # 接触力 EMA 状态

    # —— 实验2 扰动：对指定臂端部施加外力脉冲（xfrc_applied，世界系）——
    dist_body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, f"openarmx_{args.dist_side}_link7")
    dist_axis_idx = {"x": 0, "y": 1, "z": 2}[args.dist_axis]
    print(f"🔧 扰动: {args.dist_side}臂 +{args.dist_f}N 沿 {args.dist_axis} @ {args.dist_t}s 持续 {args.dist_dur}s")

    def apply_disturbance(t_sim: float) -> None:
        if args.dist_t <= t_sim < args.dist_t + args.dist_dur:
            fv = np.zeros(6)
            fv[dist_axis_idx] = args.dist_f
            data.xfrc_applied[dist_body_id] = fv
        else:
            data.xfrc_applied[dist_body_id] = np.zeros(6)

    # ---------- 主循环 ----------
    def step_control(targets: dict, v_des_map: dict | None = None,
                     f_map: dict | None = None) -> list[np.ndarray]:
        if v_des_map is None:
            v_des_map = {arm["side"]: None for arm in arms}
        if f_map is None:
            f_map = f_des
        mujoco.mj_forward(model, data)
        # —— 臂间协同 u_cop（论文式5）：相对误差耦合弹簧 ——
        tips = {arm["side"]: get_ee_tip(data, arm["body_id"]) for arm in arms}
        xdots = {}
        for arm in arms:
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp, jacr, tips[arm["side"]], arm["body_id"])
            xdots[arm["side"]] = jacp[:, arm["dofs"]] @ data.qvel[arm["dofs"]]
        f_cop = {}
        for arm in arms:
            other = next(a for a in arms if a["side"] != arm["side"])
            e_cop = (tips[arm["side"]] - tips[other["side"]]) \
                - (targets[arm["side"]] - targets[other["side"]])     # 相对位姿误差（含名义距离）
            v_cop = xdots[arm["side"]] - xdots[other["side"]]
            f_cop[arm["side"]] = np.zeros(3) if not use_coop else (-K_COP * e_cop - D_COP * v_cop)
        taus, errs = [], []
        for arm in arms:
            tau, e = solve_torque_grasp(
                model, data, arm, targets[arm["side"]], f_map[arm["side"]],
                v_des_map[arm["side"]], use_nullspace, f_cop[arm["side"]],
            )
            taus.append((arm, tau))
            errs.append(e)
        for arm, tau in taus:
            data.ctrl[arm["act_ids"]] = tau
        mujoco.mj_step(model, data)
        return errs

    def clamp_force_map() -> dict[str, np.ndarray]:
        """每臂指令力：x=夹持力闭环（论文式6 力反馈），z=水平差动，y=0。"""
        fc = pad_contact_forces()
        m = {arm["side"]: f_des[arm["side"]].copy() for arm in arms}
        if use_force_fb:                                   # 力闭环：F = f_ref + K(f_ref − f_meas)
            for arm in arms:
                s = arm["side"]
                f_filt[s] = F_FILT_ALPHA * fc[s] + (1.0 - F_FILT_ALPHA) * f_filt[s]
                cmd = f_grasp + K_FORCE * (f_grasp - f_filt[s])
                m[s][0] = -arm["sign"] * float(np.clip(cmd, 0.0, 2.0 * f_grasp))
        if use_level:
            tilt = box_tilt_rad()
            dz = float(np.clip(LEVEL_K * tilt[1], -LEVEL_MAX, LEVEL_MAX))
            m["left"][2] += dz
            m["right"][2] -= dz
        return m

    if args.headless:
        N = args.steps
        sat = {arm["side"]: 0 for arm in arms}
        fc_sum = {arm["side"]: 0.0 for arm in arms}
        err_sum = {arm["side"]: 0.0 for arm in arms}
        fc_track_min = {arm["side"]: 1e9 for arm in arms}   # 运动段最小夹持力
        track_sum, track_max, track_n = 0.0, 0.0, 0         # 箱子跟随参考路径的误差统计
        hold_drift_max = 0.0                                # HOLD 段(前1s)最大漂移
        tilt_max_x, tilt_max_y = 0.0, 0.0                   # 箱子倾角峰值（绕x侧倾 / 绕y翘起）
        tilt_tl = []                                    # 每500步倾角采样 (ms, 侧倾°, 翘起°)
        force_sync_sum, force_sync_max = 0.0, 0.0       # 力同步误差统计（实验2 口径）
        rel_yz_sum = 0.0                                # 臂间相对位姿误差(y,z)累计
        dist_drift_max = dist_fs_max = dist_rel_max = 0.0   # 扰动窗口内峰值
        dist_recover = None                             # 扰动结束后 1s 的箱漂移
        rec = ({"t": [], "err_l": [], "err_r": [], "follow": [],
                "tau_l": [], "tau_r": [], "f_l": [], "f_r": []} if args.record else None)
        for t in range(N):
            t_sim = t * IK_DT
            apply_disturbance(t_sim)
            off, vd = (np.zeros(3), np.zeros(3)) if ref_fn is None else ref_fn(t_sim)
            targets = {arm["side"]: grasp_targets[arm["side"]] + off for arm in arms}
            v_map = {arm["side"]: vd for arm in arms}
            frozen = t_sim < args.preload                    # 预加载：先冻结箱子把力建满
            if frozen:
                saved_box = data.qpos[box_qpos : box_qpos + 7].copy()
            errs = step_control(targets, v_map, clamp_force_map())
            if frozen:
                data.qpos[box_qpos : box_qpos + 7] = saved_box   # 箱子锁死，力照常建
                data.qvel[box_qvel : box_qvel + 6] = 0.0
            tilt = box_tilt_rad()
            tilt_max_x = max(tilt_max_x, abs(tilt[0]))
            tilt_max_y = max(tilt_max_y, abs(tilt[1]))
            if t % 500 == 0:
                tilt_tl.append((t * 2, np.degrees(tilt[0]), np.degrees(tilt[1])))

            expected = box_start + off
            terr = float(np.linalg.norm(data.xpos[box_id] - expected))
            track_sum += terr
            track_n += 1
            track_max = max(track_max, terr)
            if t_sim < 1.0:
                hold_drift_max = max(hold_drift_max, terr)   # 首段 HOLD：参考不动，误差≈漂移

            fc = pad_contact_forces()
            if rec is not None:
                rec["t"].append(t_sim)
                rec["err_l"].append(np.linalg.norm(errs[0]))
                rec["err_r"].append(np.linalg.norm(errs[1]))
                rec["follow"].append(terr)
                rec["tau_l"].append(data.ctrl[arms[0]["act_ids"]].copy())
                rec["tau_r"].append(data.ctrl[arms[1]["act_ids"]].copy())
                rec["f_l"].append(fc["left"])
                rec["f_r"].append(fc["right"])
            for arm, e in zip(arms, errs):
                fc_sum[arm["side"]] += fc[arm["side"]]
                err_sum[arm["side"]] += np.linalg.norm(e)
                if t_sim >= 1.0:
                    fc_track_min[arm["side"]] = min(fc_track_min[arm["side"]], fc[arm["side"]])
            # —— 实验2 口径指标：力同步误差 |F_L−F_R|、臂间相对位姿误差(y,z) ——
            fs = abs(fc["left"] - fc["right"])
            force_sync_sum += fs
            force_sync_max = max(force_sync_max, fs)
            rel_yz = np.linalg.norm(
                (get_ee_tip(data, arms[0]["body_id"]) - get_ee_tip(data, arms[1]["body_id"]))[1:]
            )
            rel_yz_sum += rel_yz
            if args.dist_t <= t_sim < args.dist_t + args.dist_dur:      # 扰动窗口统计
                dist_drift_max = max(dist_drift_max, terr)
                dist_fs_max = max(dist_fs_max, fs)
                dist_rel_max = max(dist_rel_max, rel_yz)
            if dist_recover is None and t_sim >= args.dist_t + args.dist_dur + 1.0:
                dist_recover = terr * 1000                               # 扰动结束后 1s 的箱漂移
            for arm in arms:
                low, high = model.actuator_ctrlrange[arm["act_ids"]].T
                tau = data.ctrl[arm["act_ids"]]
                if np.any(tau < low - 1e-9) or np.any(tau > high + 1e-9):
                    sat[arm["side"]] += 1

        if ref_fn is None:
            final_off = np.zeros(3)
        else:
            final_off, _ = ref_fn((N - 1) * IK_DT)   # 以最后一拍的参考位置为准（来回轨迹没有终点）
        final_ref = box_start + final_off
        final_err = np.linalg.norm(data.xpos[box_id] - final_ref)
        print(f"--- 夹持+跟踪 {N*IK_DT:.1f}s 报表 ---")
        for arm in arms:
            fc = fc_sum[arm["side"]] / N
            err = err_sum[arm["side"]] / N
            print(
                f"[{arm['side']}] 垫片接触法向力均值={fc:.2f} N (目标 {f_grasp} N) "
                f"| 运动段最小夹持力={fc_track_min[arm['side']]:.2f} N "
                f"| 端面误差均值={err*1000:.2f} mm | 饱和={sat[arm['side']]}/{N}"
            )
        print(f"箱子跟随参考路径: 误差均值={track_sum/track_n*1000:.2f} mm 峰值={track_max*1000:.2f} mm")
        print(f"力同步误差(实验2): 均值={force_sync_sum/track_n:.2f} N 峰值={force_sync_max:.2f} N")
        print(f"臂间相对位姿误差(y,z): 均值={rel_yz_sum/track_n*1000:.2f} mm")
        if args.dist_f > 0:
            print(f"[扰动] {args.dist_side}臂窗口内: 箱漂移峰值={dist_drift_max*1000:.2f} mm | "
                  f"力同步峰值={dist_fs_max:.2f} N | 相对位姿峰值={dist_rel_max*1000:.2f} mm")
            print(f"[扰动] 结束后 1s 箱漂移={dist_recover if dist_recover is not None else float('nan'):.2f} mm（恢复情况）")
        print(f"箱子倾角峰值: 绕x(侧倾)={np.degrees(tilt_max_x):.2f}° 绕y(翘起)={np.degrees(tilt_max_y):.2f}° "
              f"| 协同={'开' if use_coop else '关'} 力闭环={'开' if use_force_fb else '关'} 水平控制={'开' if use_level else '关'}")
        print("倾角时间线(ms→侧倾°/翘起°): " + " ".join(f"{m}ms:{a:.1f}/{b:.1f}" for m, a, b in tilt_tl))
        print(f"HOLD 段(0~1s)最大漂移 {hold_drift_max*1000:.2f} mm | 终点跟随误差 {final_err*1000:.2f} mm")
        print(f"箱终点 {data.xpos[box_id].round(4)} vs 参考终点 {final_ref.round(4)}")
        ok = final_err * 1000 < 20.0 and np.degrees(tilt_max_y) < 3.0 \
            and np.degrees(tilt_max_x) < 10.0 and data.xpos[box_id][2] > box_start[2] - 0.05
        print(">>> 判定: " + ("✅ 夹住、来回平稳且箱子没掉（终点跟错<20mm，翘起<3°，侧倾<10°）" if ok else "❌ 跟踪/夹持/水平失败"))
        if rec is not None:
            np.savez(
                args.record,
                t=np.array(rec["t"]),
                err_l=np.array(rec["err_l"]),
                err_r=np.array(rec["err_r"]),
                follow=np.array(rec["follow"]),
                tau_l=np.array(rec["tau_l"]),
                tau_r=np.array(rec["tau_r"]),
                f_l=np.array(rec["f_l"]),
                f_r=np.array(rec["f_r"]),
            )
            print(f"✅ 已保存时间序列 → {args.record}")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, -0.26, 0.5]
        viewer.cam.distance = 2.4
        viewer.cam.elevation = -14
        viewer.cam.azimuth = 180
        if sys.platform == "win32":
            ctypes.windll.winmm.timeBeginPeriod(1)   # 1ms 定时精度：Windows 默认 sleep 粒度≈15ms 会抖
        last = 0.0
        next_t = time.time()                         # 漂移补偿节拍：不追赶，只对齐下一拍
        while viewer.is_running():
            t_sim = data.time                        # 用仿真时钟取参考轨迹，物理与轨迹严格同步
            apply_disturbance(t_sim)
            off, vd = (np.zeros(3), np.zeros(3)) if ref_fn is None else ref_fn(t_sim)
            targets = {arm["side"]: grasp_targets[arm["side"]] + off for arm in arms}
            step_control(targets, {arm["side"]: vd for arm in arms}, clamp_force_map())
            now = time.time()
            if now - last > 0.5:
                fc = pad_contact_forces()
                terr = np.linalg.norm(data.xpos[box_id] - (box_start + off)) * 1000
                tilt = box_tilt_rad()
                print(
                    f"[DEBUG] t={t_sim:.1f}s 跟错={terr:.1f}mm F_L={fc['left']:.1f}N "
                    f"F_R={fc['right']:.1f}N 翘起={np.degrees(tilt[1]):.2f}° z={data.xpos[box_id][2]:.3f}"
                )
                last = now
            viewer.sync()
            next_t += IK_DT                          # 下一个节拍点
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.time()                 # 已落后真实时间：重置，避免积压追赶（赶会造成快进）


if __name__ == "__main__":
    main()