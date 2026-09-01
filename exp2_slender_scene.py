#!/usr/bin/env python3
"""exp2_slender_scene.py —— 实验二·论文式场景（Fig.11：双臂落地、细长箱平放、端面对夹）

与论文 Fig.11 的对应：
  - 两条 7-DOF 臂**从躯干拆下、矮基座直接立在地面**（floor_mount.py，z_mount=0.15），
    左臂 -x 侧、右臂 +x 侧面对面——论文两台 Panda 立在台面/地面的同款布局（无躯干"架子"）；
  - **镜像侧装 ±90°**（_probe_flat3 标定）：臂链的"自然下垂方向"转成水平伸向箱内，
    整条臂躺在同一水平面（l2..l7 平面偏差 0.119 m），箱在臂平面内低位平举，z 全程不变；
  - 垫片法向纯水平（TILT=0）：侧装后腕部不再顶 j7 限位（余量 0.450 rad @j4），
    垫片天然竖直贴箱端面，夹持力纯水平，不需要斜角转接板；
  - 垫片接触相对箱体重心上偏 GRIP_Z_OFFSET=+0.02 m（摆式稳定）：箱体重心悬挂在
    夹持线下方，重力对绕夹持轴(x)的滚转提供恢复力矩（中性稳定 → 摆式稳定），
    同时约束箱体 y 漂移；_probe_grip_z.py 实测：上偏恢复滚转、下偏（顶重倒摆）放大滚转；
  - 细长板条箱（长轴 = x = 夹持轴），力轴与箱长轴共线，4 罐沿箱长 x 摆放。

几何标定：箱长 L=0.50 m（L≥0.60 双臂够不到）；箱 0.8 kg → μ_eff≈1.5 时论文原文 10 N
夹持力摩擦容量 ≈30 N ≈ 3.8×箱重，10 N 可行。IK 端面误差 0.24 mm。

运行：
    python exp2_slender_scene.py --headless   # 无 GUI：生成场景 + IK 收敛 + 贴合报表
    python exp2_slender_scene.py              # 打开 viewer（R 复位, G 重新夹持）
"""
from __future__ import annotations

import argparse

import mujoco
import mujoco.viewer
import numpy as np

from ik import ARM_CONFIGS, BASE_DIR, EE_TIP_Z, clamp_qpos, get_ee_tip, name_to_id, solve_pose_ik
from ikforce import inject_motors
from grasp_pose_static import GRIP_CLEARANCE
from floor_mount import generate_floor_runtime_xml

# ---------- 布局参数（扫描标定值，_probe_ystroke 三维联合扫描最终标定） ----------
BASE_X = 0.65                  # 基座中心到原点距离（宽间距：臂更收拢 → 低基座可行 + 关节裕度大）
Z_MOUNT = 0.06                 # 臂 joint1 离地高度（6cm 矮桩几乎贴地，link1 最低可行位）
MOUNT_ROLL = {"left": np.pi / 2, "right": -np.pi / 2}
                               # 镜像侧装 90°：臂链自然下垂方向转为水平伸向箱内 →
                               # 整条臂躺在同一水平面（l2..l7 平面偏差 0.119 m）
MOUNT_YAW = np.radians(15.0)   # 基座偏航 +15°：y 行程双向均衡（_probe_ystroke 实测：
                               # ±145 mm 满量程，最坏关节余量 0.052 → 0.181 rad）
MOUNT_PITCH = 0.0
TILT = 0.0                     # 杆身法向 = 纯水平：镜像侧装后腕部不再顶限位
                               # （余量 0.450 rad @j4），垫片天然竖直贴箱，无需斜角修正
GRIP_Z_OFFSET = 0.02           # 垫片接触高度相对箱体重心的竖直偏移（m，正=接触在重心上方）
                               # 正偏移 → 箱体重心悬挂在夹持线下方 → 重力成为恢复力矩（摆式稳定），
                               # 同时约束箱体 y 漂移；负偏移 → 顶重倒摆（_probe_grip_z 实测放大滚转）。
                               # 垫片半高 0.015、箱面半高 0.04 → 偏移须 ∈ [-0.025, +0.025] 才满接触。

# ---------- 细长箱参数（论文 Fig.11 等比缩小版） ----------
SLENDER_HALF = np.array([0.25, 0.06, 0.04], dtype=np.float64)  # 半尺寸：x=长(夹持轴) y=宽 z=厚
BOX_MASS = 0.8                 # kg：轻箱 → 论文 10 N 力参考可行（摩擦容量 3.8×）
BOX_FRICTION = "100  0.05 0.005"
BOX_CENTER = np.array([0.0, -0.13, 0.11], dtype=np.float64)    # 低位平举（联合扫描最优：
                               # ±y 行程 220mm、z 耦合 ≤0.05mm、抓取误差 0.23mm），z 恒定
CONTACT_SOLREF = "0.02 1"      # 软接触 = 式(27) 形变世界（同 CONTACT_TC=0.02 标定路径）

# 罐（空可乐罐）
CAN_COUNT = 4
CAN_R = 0.033
CAN_HALF_H = 0.0575
CAN_MASS = 0.015
CAN_FRICTION = "1.0 0.05 0.005"
CAN_OFFSETS_X = (-0.18, -0.08, 0.08, 0.18)   

SCENE_XML = BASE_DIR / "exp2_slender_scene.xml"
IK_ITERS = 1500


def compute_pad_fix(model, data, arms) -> dict:
    """由当前抓取姿态计算垫片安装量（两阶段构建的第 1 阶段输出）。

    杆身（link7 +z）上仰 TILT 后，把垫片在 link7 局部系内反向旋转，使垫片面法向
    = 世界系纯水平 (∓x)：n_local = R_link7ᵀ · n_world；垫片中心 = 杆端 + 半厚·n_local，
    并沿世界竖直方向额外偏移 GRIP_Z_OFFSET（摆式稳定：接触相对箱体重心上下移）。
    返回 {side: (quat_wxyz, pos_xyz)}，喂给 generate_floor_runtime_xml(pad_fix=...)。
    """
    fix = {}
    for a in arms:
        R = data.xmat[a["body_id"]].reshape(3, 3)
        n_world = np.array([-a["sign"], 0.0, 0.0])            # 期望垫片法向：纯水平指向箱内
        n_local = R.T @ n_world
        n_local /= np.linalg.norm(n_local)
        v_local = R.T @ np.array([0.0, 0.0, 1.0])             # 世界 +z 在 link7 局部系（竖直偏移方向）
        v_local /= np.linalg.norm(v_local)
        # 局部旋转矩阵：z' = n_local；板面竖直轴取世界竖直方向投影 → 垫片板面保持竖直
        zp = n_local
        ref = v_local
        if abs(float(zp @ ref)) > 0.95:
            ref = np.array([1.0, 0.0, 0.0])
        xp = np.cross(ref, zp); xp /= np.linalg.norm(xp)
        yp = np.cross(zp, xp)
        R_pad = np.column_stack([xp, yp, zp])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R_pad.reshape(-1))
        pos = np.array([0.0, 0.0, EE_TIP_Z]) + GRIP_Z_OFFSET * v_local
        fix[a["side"]] = (quat, pos)
    return fix


def build_scene(pad_fix: dict | None = None) -> tuple:
    """生成场景 XML：地面安装双臂（矮基座+镜像侧装） + 细长箱(freejoint) + 4 罐(freejoint)在箱顶。"""
    runtime_xml = generate_floor_runtime_xml(base_x=BASE_X, z_mount=Z_MOUNT,
                                             mount_roll=MOUNT_ROLL, mount_pitch=MOUNT_PITCH,
                                             mount_yaw=MOUNT_YAW, pad_fix=pad_fix)
    inject_motors(runtime_xml)
    bx, by, bz = BOX_CENTER
    hx, hy, hz = SLENDER_HALF
    cans_xml = "".join(
        f"""
    <body name="can{i}" pos="{bx + ox:.4f} {by:.4f} {bz + hz + CAN_HALF_H + 0.002:.4f}">
      <freejoint name="can{i}_freejoint"/>
      <geom name="can{i}" type="cylinder" size="{CAN_R} {CAN_HALF_H}"
            material="can_mat" mass="{CAN_MASS}"
            friction="{CAN_FRICTION}" condim="3" contype="1" conaffinity="1"
            solref="0.02 1" solimp="0.999 0.999 0.001"/>
    </body>"""
        for i, ox in enumerate(CAN_OFFSETS_X)
    )
    SCENE_XML.write_text(
        f"""<mujoco model="exp2_slender_scene">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" iterations="500" solver="Newton" cone="elliptic"/>

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
    <material name="box_mat" rgba="0.76 0.60 0.42 1" shininess="0.25" specular="0.15"/>
    <material name="can_mat" rgba="0.75 0.20 0.16 1" shininess="0.5" specular="0.6"/>
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

    <!-- 细长箱：freejoint（不夹住会下落）；软接触 solref {CONTACT_SOLREF} = 式(27) 形变世界 -->
    <body name="box" pos="{bx:.4f} {by:.4f} {bz:.4f}">
      <freejoint name="box_freejoint"/>
      <geom name="box" type="box" size="{hx} {hy} {hz}"
            material="box_mat" mass="{BOX_MASS}"
            friction="{BOX_FRICTION}" condim="3" contype="1" conaffinity="1"
            solref="{CONTACT_SOLREF}" solimp="0.999 0.999 0.001"/>
    </body>
{cans_xml}
  </worldbody>

  <include file="{runtime_xml.name}"/>
</mujoco>
""",
        encoding="utf-8",
    )
    return SCENE_XML


def grasp_targets(arms) -> dict[str, np.ndarray]:
    """端面对夹目标：垫片中心 = 箱端面外 GRIP_CLEARANCE（垫片半厚），法向上仰 TILT 指向箱内。

    地面布局：left 臂在 -x 侧（sign=-1，法向 +x），right 臂在 +x 侧（sign=+1，法向 -x）。
    """
    bx, by, bz = BOX_CENTER
    return {
        a["side"]: np.array(
            [bx + a["sign"] * (SLENDER_HALF[0] + GRIP_CLEARANCE), by, bz], dtype=np.float64
        )
        for a in arms
    }


def inward_dir(side_sign: float) -> np.ndarray:
    """垫片法向：指向箱内且上仰 TILT（左臂 +x、右臂 -x）。"""
    return np.array([-side_sign * np.cos(TILT), 0.0, np.sin(TILT)])


def solve_end_grip(model, data, arms, targets: dict[str, np.ndarray], iters: int = IK_ITERS) -> None:
    """6D 位姿 IK：双臂各自独立收敛到端面对夹位。"""
    for a in arms:
        for _ in range(iters):
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            mujoco.mj_forward(model, data)
            dq, _ = solve_pose_ik(
                model, data, a["body_id"], targets[a["side"]],
                inward_dir(a["sign"]), get_ee_tip(data, a["body_id"]),
                a["dofs"], a["qpos"], a["q_rest"],
            )
            data.qpos[a["qpos"]] += dq
            clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)


def make_arms(model) -> list[dict]:
    """地面布局侧别：openarmx_left 装在 -x 基座（sign=-1），openarmx_right 在 +x（sign=+1）。"""
    arms = []
    for cfg in ARM_CONFIGS:
        body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["jacobi_body"])
        joint_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg["joint_names"]]
        arms.append({
            "side": cfg["side"],
            "sign": -1.0 if cfg["side"] == "left" else 1.0,   # 基座所在侧（箱端面所在侧）
            "q_rest": cfg["q_rest"], "body_id": body_id, "joint_ids": joint_ids,
            "dofs": [int(model.jnt_dofadr[j]) for j in joint_ids],
            "qpos": [int(model.jnt_qposadr[j]) for j in joint_ids],
            "pad_id": name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"openarmx_{cfg['side']}_ee_pad"),
        })
    return arms


def load_scene(pad_fix=None):
    """加载场景并把臂/箱/罐放到初始位（休息位 + 箱/罐初始坐标），返回 (scene_xml, model, data, arms)。"""
    sc = build_scene(pad_fix)
    m = mujoco.MjModel.from_xml_path(str(sc))
    d = mujoco.MjData(m)
    arm_list = make_arms(m)
    for a in arm_list:
        d.qpos[a["qpos"]] = a["q_rest"].copy()
        clamp_qpos(m, d, a["qpos"], a["joint_ids"])
    d.qvel[:] = 0.0
    d.qacc[:] = 0.0
    bq = int(m.jnt_qposadr[name_to_id(m, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")])
    d.qpos[bq:bq + 3] = BOX_CENTER
    d.qpos[bq + 3:bq + 7] = [1.0, 0.0, 0.0, 0.0]
    for k, ox in enumerate(CAN_OFFSETS_X):
        cq = int(m.jnt_qposadr[name_to_id(m, mujoco.mjtObj.mjOBJ_JOINT, f"can{k}_freejoint")])
        d.qpos[cq:cq + 3] = [BOX_CENTER[0] + ox, BOX_CENTER[1],
                             BOX_CENTER[2] + SLENDER_HALF[2] + CAN_HALF_H + 0.002]
        d.qpos[cq + 3:cq + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    return sc, m, d, arm_list


def build_grasped_scene(iters: int = IK_ITERS):
    """两阶段构建 + 收敛端面对夹姿态，返回 (scene_xml, model, data, arms)。

    pass1 先 IK 出抓取姿态；TILT≠0 或 GRIP_Z_OFFSET≠0 时计算垫片斜角/竖直修正
    （pad_fix），再重建场景并重新收敛。主程序与诊断工具共用，保证接触几何一致。
    """
    scene_xml, model, data, arms = load_scene(None)
    targets = grasp_targets(arms)
    solve_end_grip(model, data, arms, targets, iters)
    if abs(TILT) > 1e-6 or abs(GRIP_Z_OFFSET) > 1e-9:
        pad_fix = compute_pad_fix(model, data, arms)
        scene_xml, model, data, arms = load_scene(pad_fix)
        solve_end_grip(model, data, arms, targets, iters)
    return scene_xml, model, data, arms


def main() -> None:
    parser = argparse.ArgumentParser(description="实验二·地面安装细长箱端面对夹场景（论文 Fig.11）。")
    parser.add_argument("--headless", action="store_true", help="无 GUI：IK 收敛后打印贴合报表")
    parser.add_argument("--check", action="store_true", help="只生成并加载场景")
    parser.add_argument("--iters", type=int, default=IK_ITERS)
    args = parser.parse_args()

    # ---- 两阶段构建：pass1 先 IK 出抓取姿态；TILT/GRIP_Z_OFFSET 非零时重建带垫片修正的场景 ----
    scene_xml, model, data, arms = build_grasped_scene(args.iters)
    targets = grasp_targets(arms)
    box_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "box")

    def reset() -> None:
        for a in arms:
            data.qpos[a["qpos"]] = a["q_rest"].copy()
            clamp_qpos(model, data, a["qpos"], a["joint_ids"])
            data.qvel[a["dofs"]] = 0.0
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        bq = int(model.jnt_qposadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")])
        data.qpos[bq:bq + 3] = BOX_CENTER
        data.qpos[bq + 3:bq + 7] = [1.0, 0.0, 0.0, 0.0]
        for k, ox in enumerate(CAN_OFFSETS_X):
            cq = int(model.jnt_qposadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"can{k}_freejoint")])
            data.qpos[cq:cq + 3] = [BOX_CENTER[0] + ox, BOX_CENTER[1],
                                    BOX_CENTER[2] + SLENDER_HALF[2] + CAN_HALF_H + 0.002]
            data.qpos[cq + 3:cq + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

    print("✅ 地面安装细长箱场景已加载:", scene_xml.name)
    print(f"   基座: (±{BASE_X}, 0, 0) 安装高 {Z_MOUNT} m 镜像侧装 ±90° | "
          f"箱: 长 {2*SLENDER_HALF[0]:.2f} × 宽 {2*SLENDER_HALF[1]:.2f} × 厚 {2*SLENDER_HALF[2]:.2f} m, "
          f"{BOX_MASS} kg, 中心 {np.round(BOX_CENTER, 3)}")
    print(f"   罐: {CAN_COUNT} 个沿箱长 x (offsets {CAN_OFFSETS_X}) | "
          f"垫片法向纯水平（tilt={np.degrees(TILT):.0f}°）| "
          f"接触高度偏移 grip_z={GRIP_Z_OFFSET*1000:+.0f} mm（重心{'+上方' if GRIP_Z_OFFSET>0 else '下方/同高'}）")

    if args.check:
        return

    # ---------- 贴合报表 ----------
    print("\n--- 端面对夹贴合报表（地面安装·平面化·斜角垫片） ---")
    for a in arms:
        tip = get_ee_tip(data, a["body_id"])
        err = float(np.linalg.norm(tip - targets[a["side"]]))
        # 垫片自身姿态（geom_xmat）：斜角修正后应 = 纯水平指向箱内 (∓x)
        pad_R = data.geom_xmat[a["pad_id"]].reshape(3, 3)
        normal = pad_R @ np.array([0.0, 0.0, 1.0])
        normal /= np.linalg.norm(normal)
        align = float(normal @ np.array([-a["sign"], 0.0, 0.0]))   # 1.0 = 完全水平朝箱
        pad_center = data.geom_xpos[a["pad_id"]]
        pad_front_x = pad_center[0] + 0.004 * normal[0]
        face_x = BOX_CENTER[0] + a["sign"] * SLENDER_HALF[0]
        gap_out = (pad_front_x - face_x) * a["sign"]   # >0 间隙 / <0 压入
        q = data.qpos[a["qpos"]]
        lo = model.jnt_range[a["joint_ids"], 0]
        hi = model.jnt_range[a["joint_ids"], 1]
        mrg = np.minimum(q - lo, hi - q)
        j_min = int(np.argmin(mrg))
        # 平面性：link1..7 中心相对箱高水平面的最大偏差
        zs = [data.xpos[name_to_id(model, mujoco.mjtObj.mjOBJ_BODY,
                                   f"openarmx_{a['side']}_link{i}")][2] for i in range(1, 8)]
        plan = float(np.max(np.abs(np.array(zs) - BOX_CENTER[2])))
        print(f"  [{a['side']}] 端面误差={err*1e3:.2f} mm | 垫片法向水平对齐={align:.4f} | "
              f"垫片-箱端面: {'间隙' if gap_out >= 0 else '压入'} {abs(gap_out)*1e3:.2f} mm | "
              f"关节余量={mrg[j_min]:.3f} rad (j{j_min+1}) | 平面偏差={plan:.3f} m")
    can_z_ok = all(
        abs(data.xpos[name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, f"can{k}")][2]
            - (BOX_CENTER[2] + SLENDER_HALF[2] + CAN_HALF_H + 0.002)) < 1e-6
        for k in range(CAN_COUNT)
    )
    print(f"  罐底贴箱顶: {'✓' if can_z_ok else '✗'} | "
          f"箱端面间距={2*SLENDER_HALF[0]:.3f} m | 垫片间距={2*(SLENDER_HALF[0]+GRIP_CLEARANCE):.3f} m")

    if args.headless:
        print("完成（端面对夹姿态已收敛）。")
        return

    state = {"reset": False, "grasp": False}

    def key_callback(keycode: int) -> None:
        try:
            key = chr(keycode).upper()
        except ValueError:
            return
        if key == "R":
            state["reset"] = True
            print("👉 复位到休息位")
        elif key == "G":
            state["grasp"] = True
            print("👉 重新收敛端面对夹姿态")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = [0.0, -0.13, 0.18]
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -10
        viewer.cam.azimuth = 180
        mujoco.mj_forward(model, data)
        while viewer.is_running():
            if state["reset"]:
                reset()
                state["reset"] = False
            elif state["grasp"]:
                solve_end_grip(model, data, arms, targets, args.iters)
                state["grasp"] = False
            viewer.sync()


if __name__ == "__main__":
    main()