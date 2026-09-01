#!/usr/bin/env python3
"""exp2_nmpcc_grasppose.py —— 把人形「抱箱」场景（grasp_pose_scene.xml）接入实验二 NMPCC

从 exp2_nmpcc_latest.py 复制改造，唯一实质改动 = 场景/几何常量换成人形抱箱版：
  - 场景几何 = grasp_pose_scene.xml（双臂装在人形躯干上、箱顶在肩高前方 0.5/0/0.72，
    箱半尺寸 [0.12, 0.20, 0.10] m、质量 1.5 kg、摩擦 25），无罐（旧场景本来就没有）；
  - sign 约定与人形抱箱一致：左臂在箱 +x 侧（sign=+1，压向 −x），右臂在 −x 侧（sign=−1，
    压向 +x）——与 exp2_slender 地面镜像布局（左 −1/右 +1）相反，这是必须翻的；
  - 接触标定用旧场景标定值（calib_out.txt）：kx_model = k1/2 = 3575 N/m，
    rest_gap = 2·(0.12+0.004)+F/kx；控制律/时序/日志/扫描结构全部沿用 exp2_nmpcc_latest。

运行：
  仅保持：
    D:\\acados_env\\python.exe -u exp2_nmpcc_grasppose.py --hold_only --no_viewer
  力/增益扫描：
    D:\\acados_env\\python.exe -u exp2_nmpcc_grasppose.py --hold_only --no_viewer --sweep_forces 15,20,25
  定时搬运（0.5 s ramp + 3 s Holding 后进入 Moving，人形场景 y 行程取 ±0.03）：
    D:\\acados_env\\python.exe -u exp2_nmpcc_grasppose.py --y_dist 0.03
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

import nmpcc_cart as nc
from ik import (
    ARM_CONFIGS,
    BASE_DIR,
    clamp_qpos,
    generate_runtime_xml,
    get_ee_tip,
    name_to_id,
    solve_pose_ik,
)
from ikforce import inject_motors, strip_collisions
from grasp_pose_static import BOX_HALF, BOX_MASS, GRIP_CLEARANCE, strip_ee_geoms
from ikforce_grasp import PAD_FRICTION, set_pad_friction

# ---------- 人形抱箱场景（grasp_pose_scene.xml 同款几何） ----------
BOX_CENTER = np.array([0.5, 0.0, 0.72], dtype=np.float64)    # 箱中心 = 肩高前方
BOX_FRICTION = "25 0.05 0.005"    # 与 grasp_pose_scene.xml 一致
CAN_COUNT = 0
CAN_OFFSETS_X = ()
CAN_HALF_H = 0.0
SCENE_XML = BASE_DIR / "exp2_grasppose_scene.xml"
IK_ITERS = 1500

# ---------- 接触标定（grasp_pose/人形场景旧标定值 calib_out.txt；换场景必须重标） ----------
# calib_contact.py（human-pose 场景）：tc=0.02 → k1=7151 N/m → 模型串联 kx=k1/2=3575 N/m；
# 25N 对应每臂压深 δ*=3.50mm。
K_STIFF = 3575.0
REST_GAP = 2 * (BOX_HALF[0] + GRIP_CLEARANCE) + 15.0 / K_STIFF  # 0.248 + F/kx ≈ 0.2522
CALIB_INTERCEPT = 0.8   # 与 exp2 兼容的软接触底数（delta=0 处）
DT = 0.002
CTRL_EVERY = 3
PAD_HALF_T = 0.004      # 垫片半厚（与场景一致）
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "exp2_grasppose")

# 仅用于结果评价，不参与 Holding→Moving 切换。
HOLD_BOX_Y_TOL = 0.002


# ======================================================================
# 场景/手臂/目标 —— 人形抱箱版（grasp_pose_scene.xml 同款几何，注入电机后跑 NMPCC）
# ======================================================================
def build_scene() -> Path:
    """生成人形抱箱场景 XML（几何与 grasp_pose_scene.xml 一致，注入 14 电机 + 软接触垫摩擦）。

    与 exp2_slender_scene 的区别只有这一段：双臂装在**躯干上（人形）**而非地面矮基座，
    箱在肩高前方 (0.5, 0, 0.72)（箱半尺寸 [0.12,0.20,0.10]、1.5 kg、摩擦 25）。
    """
    runtime_xml = generate_runtime_xml()          # ik.py：人形双臂 runtime + 端面垫片
    strip_ee_geoms(runtime_xml)                   # 删 手/指 网格（人形抱箱必须！否则手会先插进箱里挡住垫片）
    strip_collisions(runtime_xml)                 # 删 *_collision
    inject_motors(runtime_xml)                    # 14 个 {joint}_motor（NMPCC 力矩控制必需）
    set_pad_friction(runtime_xml, PAD_FRICTION)   # 垫片高摩擦（抱箱不滑）
    hx, hy, hz = BOX_HALF
    SCENE_XML.write_text(
        f"""<mujoco model="exp2_grasppose_scene">
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

    <!-- 箱子：freejoint，人形抱箱（0.5,0,0.72），软接触 = 式(27) 形变世界 -->
    <body name="box" pos="{BOX_CENTER[0]:.4f} {BOX_CENTER[1]:.4f} {BOX_CENTER[2]:.4f}">
      <freejoint name="box_freejoint"/>
      <geom name="box" type="box" size="{hx:.4f} {hy:.4f} {hz:.4f}"
            material="box_mat" mass="{BOX_MASS}"
            friction="{BOX_FRICTION}" condim="3" contype="1" conaffinity="1"
            solref="0.02 1" solimp="0.999 0.999 0.001"/>
    </body>
  </worldbody>

  <include file="{runtime_xml.name}"/>
</mujoco>
""",
        encoding="utf-8",
    )
    return SCENE_XML


def make_arms(model) -> list[dict]:
    """双臂注册 —— 控制器“物理臂 ↔ 侧”对调版（人形抱箱关键修正）。

    人形静态场景（grasp_pose_static）已验证的物理布局：
      openarmx_left  ≙ 箱 +x 面（垫片中心 0.624）
      openarmx_right ≙ 箱 −x 面（垫片中心 0.376）
    而 nmpcc_cart 控制器模型内部按 left=−x / right=+x 建力（d=xR−xL、f_left=−kx·c）。
    因此若按 cfg['side'] 直送，物理在 +x 的 left 臂会被当成控制器 left（−x），
    力和参考方向全部反掉，IK 也够不到对面（实测 err 272/795mm、接触力 0）。

    正确映射（本副本）：
      控制器 left （sign=−1，目标箱 −x 面 0.376） ← openarmx_right 臂
      控制器 right（sign=+1，目标箱 +x 面 0.624） ← openarmx_left  臂
    控制器数学约定一字不改，只有“哪个物理臂担任控制器哪一侧”对调。
    """
    arms = []
    side_of = {"left": "right", "right": "left"}    # 物理臂名 → 控制器侧（对调）
    for cfg in ARM_CONFIGS:
        ctrl_side = side_of[cfg["side"]]
        sign = -1.0 if ctrl_side == "left" else 1.0
        body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["jacobi_body"])
        joint_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                     for n in cfg["joint_names"]]
        arms.append({
            "side": ctrl_side,
            "sign": sign,
            "q_rest": cfg["q_rest"],
            "body_id": body_id,
            "joint_ids": joint_ids,
            "dofs": [int(model.jnt_dofadr[j]) for j in joint_ids],
            "qpos": [int(model.jnt_qposadr[j]) for j in joint_ids],
            "pad_id": name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                 f"openarmx_{cfg['side']}_ee_pad"),
        })
    return arms


def grasp_targets(arms) -> dict[str, np.ndarray]:
    """端面对夹目标：垫片中心 = 箱侧面外 GRIP_CLEARANCE（垫片半厚），法向纯水平指向箱内。"""
    bx, by, bz = BOX_CENTER
    return {
        a["side"]: np.array(
            [bx + a["sign"] * (BOX_HALF[0] + GRIP_CLEARANCE), by, bz], dtype=np.float64
        )
        for a in arms
    }


def solve_end_grip(model, data, arms, targets: dict[str, np.ndarray],
                   iters: int = IK_ITERS) -> None:
    """6D 位姿 IK：双臂各自独立收敛到端面对夹位（人形：不斜角，TILT=0）。"""
    for a in arms:
        for _ in range(iters):
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            mujoco.mj_forward(model, data)
            dq, _ = solve_pose_ik(
                model, data, a["body_id"], targets[a["side"]],
                np.array([-a["sign"], 0.0, 0.0]), get_ee_tip(data, a["body_id"]),
                a["dofs"], a["qpos"], a["q_rest"],
            )
            data.qpos[a["qpos"]] += dq
            clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)


def soften_contacts(model, box_geom_id, pad_ids) -> None:
    """接触软化（箱/垫片同 tc=0.02）：模型式(27) 弹簧与世界软接触刚度一一对应（与 exp2 同标定路径）。"""
    for gid in [box_geom_id, *pad_ids]:
        sr = model.geom_solref[gid].copy()
        sr[:] = [0.02, 1.0]
        model.geom_solref[gid] = sr
        si = model.geom_solimp[gid].copy()
        si[:3] = [0.999, 0.999, 0.001]
        model.geom_solimp[gid] = si


def arm_damping(model, dofs, value=0.5):
    for d in dofs:
        model.dof_damping[d] = value


def smoothstep(s: float) -> float:
    s = min(1.0, max(0.0, float(s)))
    return 3.0 * s * s - 2.0 * s * s * s


# ======================================================================
# 接触/姿态测量（日志与动力学共用）
# ======================================================================
def contact_wrench_on_arm(model, data, arm, box_geom_id):
    """将垫片-箱接触的世界系切向力回射到关节：fJ=J^T f。

    ``mj_contactForce`` 给出的力作用在垫片一侧；接触点可能与 EE tip
    相差垫片半厚，因此必须在实际接触点计算 Jacobian。x 向法向力仍由
    NMPCC 的虚拟弹簧模型承担，这里只回射实测 y/z 切向载荷。
    """
    f_world = np.zeros(3)
    fJ = np.zeros(len(arm["dofs"]))
    force6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if not ((c.geom1 == arm["pad_id"] and c.geom2 == box_geom_id) or
                (c.geom2 == arm["pad_id"] and c.geom1 == box_geom_id)):
            continue
        mujoco.mj_contactForce(model, data, i, force6)
        frame = c.frame.reshape(3, 3)
        f = frame.T @ force6[:3]
        tau = frame.T @ force6[3:]
        # The contact force convention is verified against qfrc_constraint:
        # this is the wrench acting on the pad/arm, independent of geom order.
        f[0] = 0.0
        f_world += f
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, c.pos, arm["body_id"])
        fJ += jacp[:, arm["dofs"]].T @ f + jacr[:, arm["dofs"]].T @ tau
    return fJ, f_world


def contact_forces(model, data, arms, box_geom_id):
    """每侧法向力绝对值之和（与标定路径同口径，用于兼容比较）。"""
    result = {a["side"]: 0.0 for a in arms}
    force = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        for a in arms:
            if ((c.geom1 == a["pad_id"] and c.geom2 == box_geom_id) or
                    (c.geom2 == a["pad_id"] and c.geom1 == box_geom_id)):
                mujoco.mj_contactForce(model, data, i, force)
                result[a["side"]] += abs(float(force[0]))
    return result


def pad_box_wrench_3d(model, data, arms, box_geom_id):
    """每侧垫片-箱接触的世界系 3D 合力 + 3D 合力矩（力矩按接触点，由 mj_contactForce 映射）。

    返回 {side: (f_world(3), tau_world(3))}，为"作用在垫片(臂端)上的全量接触力偶”。
    """
    out = {a["side"]: (np.zeros(3), np.zeros(3)) for a in arms}
    force6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        for a in arms:
            if ((c.geom1 == a["pad_id"] and c.geom2 == box_geom_id) or
                    (c.geom2 == a["pad_id"] and c.geom1 == box_geom_id)):
                mujoco.mj_contactForce(model, data, i, force6)
                frame = c.frame.reshape(3, 3)
                f, tau = out[a["side"]]
                f += frame.T @ force6[:3]
                tau += frame.T @ force6[3:]
                out[a["side"]] = (f, tau)
    return out


def box_net_wrench(model, data, arms, box_geom_id, box_com):
    """箱体质心处的净接触力偶（来自两垫片接触；箱体自由体受力 = 接触 + 重力）。"""
    f_net = np.zeros(3)
    tau_net = np.zeros(3)
    force6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        is_pair = False
        for a in arms:
            if ((c.geom1 == a["pad_id"] and c.geom2 == box_geom_id) or
                    (c.geom2 == a["pad_id"] and c.geom1 == box_geom_id)):
                is_pair = True
                break
        if not is_pair:
            continue
        mujoco.mj_contactForce(model, data, i, force6)
        frame = c.frame.reshape(3, 3)
        f_on_pad = frame.T @ force6[:3]          # 作用在垫片方向（近箱侧）
        t_on_pad = frame.T @ force6[3:]
        f_on_box = -f_on_pad                       # 牛顿第三定律：箱受反力
        t_on_box = -t_on_pad
        r = c.pos - box_com
        tau_net += t_on_box + np.cross(r, f_on_box)
        f_net += f_on_box
    return f_net, tau_net


def box_attitude(data, box_qpos):
    """freejoint 四元数 -> (精确 rotvec(3), quat_wxyz)。"""
    q = data.qpos[box_qpos + 3:box_qpos + 7].copy()
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return nc.rotvec_of_R(R.reshape(3, 3)), q


# ======================================================================
# 阶段参考：平滑预紧 + holding -> moving（参数进 OCP p，控制器恒定）
# ======================================================================
def phase_profile(force, vy, distance, t_ramp, t_hold, press0, press, rv_ref,
                  f_start=2.0):
    """返回 sample(t) -> refs, vrefs, fmag, phase, ramp_s。

    参考按固定时间切换：先平滑预紧 ``t_ramp``，再保持 ``t_hold``，随后直接
    进入 Moving；不再使用稳定门控。``f_start`` 与启动 IK 压紧位一致，因此
    提高它可以避免控制器接手瞬间先释放夹持力。
    """
    distance = abs(distance)
    speed = abs(vy)
    motion_enabled = distance > 1e-9 and speed > 1e-9
    t_hold_end = t_ramp + t_hold
    segment = distance / speed if motion_enabled else 0.0

    def sample(t):
        s = smoothstep(t / t_ramp) if t_ramp > 1e-9 else 1.0
        fmag = f_start + (force - f_start) * s

        press_now = {
            side: press0[side] + (press[side] - press0[side]) * s
            for side in press0
        }
        if t < t_ramp:
            phase = "ramp"
        elif not motion_enabled or t < t_hold_end:
            phase = "hold"
        else:
            phase = "move"
        offset_y = 0.0
        velocity_y = 0.0

        if phase == "move":
            tau = t - t_hold_end
            ramp = min(0.20, 0.5 * segment)
            if tau < ramp:
                ss = tau / ramp
                smth = 3.0 * ss * ss - 2.0 * ss * ss * ss
                offset_y = -speed * ramp * (ss ** 3 - 0.5 * ss ** 4)
                velocity_y = -speed * smth
            elif tau < segment:
                offset_y = -(0.5 * speed * ramp + speed * (tau - ramp))
                velocity_y = -speed
            else:
                offset_y = -distance
                velocity_y = 0.0
        refs, vrefs = {}, {}
        print(
        f"[PROFILE] t={t:.3f}s | "
        f"phase={phase} | "
        f"distance={distance:.4f} | "
        f"offset_y={offset_y:.4f} | "
        f"velocity_y={velocity_y:.4f} | "
        )
        for side in ("left", "right"):
            base = press_now[side]
            refs[side] = np.concatenate([
                [base[0], base[1] + offset_y, base[2]], rv_ref[side],
            ])
            vrefs[side] = np.array([0.0, velocity_y, 0.0, 0.0, 0.0, 0.0])
        return refs, vrefs, fmag, phase, s

    return sample, t_hold_end


# ======================================================================
# 启动几何/FK 对账
# ======================================================================
def reconciliation(model, data, arms, box_qpos, fk, base_targets, press_light,
                   force) -> None:
    print("=" * 78)
    print("启动几何/FK 对账 (reconciliation)")
    print("=" * 78)
    bq = box_qpos
    print(f"[箱] freejoint pos={np.round(data.qpos[bq:bq + 3], 6)} "
          f"目标={BOX_CENTER} ok={np.allclose(data.qpos[bq:bq + 3], BOX_CENTER, atol=1e-4)}")
    print(f"[箱] quat={np.round(data.qpos[bq + 3:bq + 7], 6)}"
          f" | 质量={BOX_MASS}kg 重心高={BOX_CENTER[2]}m")
    qL = data.qpos[arms[0]["qpos"]].copy()
    qR = data.qpos[arms[1]["qpos"]].copy()
    fk_r = np.asarray(fk["tips_fn"](qL, qR)).reshape(-1)
    for i, a in enumerate(arms):
        side = a["side"]
        tip = get_ee_tip(data, a["body_id"])
        tgt = press_light[side]
        err_mm = np.linalg.norm(tip - tgt) * 1000.0
        fkp = fk_r[i * 6:i * 6 + 3]
        fk_err_mm = np.linalg.norm(fkp - tip) * 1000.0
        pad_R = data.geom_xmat[a["pad_id"]].reshape(3, 3)
        n = pad_R @ np.array([0.0, 0.0, 1.0])
        n /= (np.linalg.norm(n) + 1e-12)
        align = float(n @ np.array([-a["sign"], 0.0, 0.0]))
        pad_c = data.geom_xpos[a["pad_id"]]
        front_x = pad_c[0] + PAD_HALF_T * n[0]
        face_x = BOX_CENTER[0] + a["sign"] * BOX_HALF[0]
        gap_mm = (front_x - face_x) * a["sign"] * 1000.0
        pad_z_lo, pad_z_hi = pad_c[2] - 0.030, pad_c[2] + 0.030
        box_z_lo, box_z_hi = BOX_CENTER[2] - BOX_HALF[2], BOX_CENTER[2] + BOX_HALF[2]
        print(f"[{side}] IK err={err_mm:.3f}mm | FK-vs-MuJoCo err={fk_err_mm:.4f}mm | "
              f"垫片法向对齐(n·入箱)={align:.4f} | 垫片-箱端面间隙={gap_mm:+.3f}mm")
        print(f"[{side}] 垫片面 z∈[{pad_z_lo:.4f},{pad_z_hi:.4f}] 箱面 z∈[{box_z_lo:.4f},{box_z_hi:.4f}]"
              f" | {'箱面下缘<垫片下缘:纯摩擦持重(需法向>0.9N)' if box_z_lo < pad_z_lo else '箱面下缘受垫片托载'}")
    d = float(get_ee_tip(data, arms[1]["body_id"])[0]
               - get_ee_tip(data, arms[0]["body_id"])[0])
    print(f"[几何] 垫片中心间距 d={d:.6f} m vs REST_GAP={REST_GAP:.6f} m "
          f"(差 {(d - REST_GAP) * 1000:+.3f} mm → 零压入接触量)")
    press_in = press_light["left"][0] - base_targets["left"][0]
    expect = K_STIFF * 2.0 * abs(press_in) + CALIB_INTERCEPT
    forces = contact_forces(model, data, arms,
                            name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "box"))
    print(f"[预紧] 起始轻压 |{press_in * 1000:.3f}| mm | 标定预期力 "
          f"K_side*δ+{CALIB_INTERCEPT} = {expect:.2f} N | 实测 FL={forces['left']:.2f}N "
          f"FR={forces['right']:.2f}N | 目标力 {force:.1f} N")
    ok_light = forces["left"] > 0.3 and forces["right"] > 0.3
    print(f"[对账] {'✓ 轻接触成立（摩擦持重安全）' if ok_light else '✗ 无接触/欠压！箱可能下落'}"
          f" | 重压目标 δ={press_delta_mm(force):.3f} mm（标定 F={force:.0f}N 对应）")
    print("=" * 78)


def press_delta_mm(force):
    """重压目标压入量（mm）= F/(2K)/1e-3。"""
    return force / (2.0 * K_STIFF) * 1000.0


# ======================================================================
# 场景复位（sweep 复用同一 model/data）
# ======================================================================
def reset_world(model, data, arms, box_qpos) -> None:
    for a in arms:
        data.qpos[a["qpos"]] = a["q_rest"]
        clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    data.qpos[box_qpos:box_qpos + 3] = BOX_CENTER
    data.qpos[box_qpos + 3:box_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)


# ======================================================================
# 单次仿真（供 main 与 sweep 共用）
# ======================================================================
def run_case(model, data, arms, ids, args, nmpc, ref_fn, t_hold_end,
             viewer=None) -> (list, dict):
    box_body, box_geom, box_qpos = ids["box_body"], ids["box_geom"], ids["box_qpos"]

    def joint_state():
        return np.concatenate([
            data.qpos[arms[0]["qpos"]], data.qvel[arms[0]["dofs"]],
            data.qpos[arms[1]["qpos"]], data.qvel[arms[1]["dofs"]],
        ])

    def plant_params():
        full_m = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, full_m)
        mgs = []
        fjs = []
        for a in arms:
            mm = full_m[np.ix_(a["dofs"], a["dofs"])]
            saved = data.qvel.copy()
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            bias = data.qfrc_bias[a["dofs"]].copy()
            data.qvel[:] = saved
            mujoco.mj_forward(model, data)
            fJ, _ = contact_wrench_on_arm(model, data, a, box_geom)
            mgs.extend([mm.reshape(-1), bias])
            fjs.append(fJ)
        damping = np.concatenate([model.dof_damping[a["dofs"]] for a in arms])
        return np.concatenate(mgs + fjs + [damping])

    def gravity_comp():
        out = []
        for a in arms:
            saved = data.qvel.copy()
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            out.append(data.qfrc_bias[a["dofs"]].copy())
            data.qvel[:] = saved
            mujoco.mj_forward(model, data)
        return np.concatenate(out)

    normal_force_filt = {a["side"]: None for a in arms}

    def clamp_balance_comp(fmag):
        """Map the desired Cartesian clamp preload to joint torque.

        ``clamp_ff_scale=1`` preserves the historical open-loop preload.  A
        nonzero ``clamp_force_kp`` adds a filtered measured-force correction;
        this is an engineering feedback layer and is reported separately from
        the paper NMPCC result.
        """
        measured = pad_box_wrench_3d(model, data, arms, box_geom)
        out = []
        for a in arms:
            side = a["side"]
            f_now = abs(float(measured[side][0][0]))
            previous = normal_force_filt[side]
            f_filtered = f_now if previous is None else 0.2 * f_now + 0.8 * previous
            normal_force_filt[side] = f_filtered
            f_cmd = args.clamp_ff_scale * fmag + args.clamp_force_kp * (fmag - f_filtered)
            f_cmd = float(np.clip(f_cmd, 0.0, 1.5 * max(fmag, args.f_start)))
            fr = [a["sign"] * f_cmd, 0.0, 0.0]
            jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp, jacr,
                          get_ee_tip(data, a["body_id"]), a["body_id"])
            out.append(-(jacp[:, a["dofs"]].T @ np.array(fr)))
        return np.concatenate(out)

    def reference_for(t):
        refs, vrefs, fmag, phase, ramp_s = ref_fn(t)
        return refs, vrefs, fmag, phase, ramp_s


    def build_p(t):
        refs, vrefs, fmag, phase, ramp_s = reference_for(t)
        p = np.zeros(nc.NP)
        pp = plant_params()
        p[0:112] = pp[:112]
        p[112:126] = pp[112:126]
        p[162:176] = pp[126:140]
        p[126:132] = refs["left"]
        p[132:138] = refs["right"]
        p[138:144] = vrefs["left"]
        p[144:150] = vrefs["right"]
        p[150:156] = [-fmag, 0, 0, 0, 0, 0]
        p[156:162] = [fmag, 0, 0, 0, 0, 0]
        return p, fmag, phase

    # 预热：t=0 已处在较高启动夹持力，避免控制器接手时先释放箱体。
    p0, fmag0, _ = build_p(0.0)
    u_seed = gravity_comp() + clamp_balance_comp(fmag0)
    nmpc._warm = None
    for _ in range(2):
        nmpc.solve(joint_state(), p0, u_ref=u_seed, u_seed=u_seed)

    initial_box_y = float(data.xpos[box_body][1])
    print(f"[timed] ramp={args.t_ramp:.3f}s, holding={args.t_hold:.3f}s, "
          f"moving starts at t={t_hold_end:.3f}s"
          if abs(args.y_dist) > 1e-9 and abs(args.v_y) > 1e-9 and not args.hold_only
          else f"[timed] ramp={args.t_ramp:.3f}s, then holding; translation disabled")

    log = []
    u_last = u_seed.copy()
    torque_lo = np.concatenate([model.actuator_ctrlrange[a["act_ids"], 0]
                                for a in arms])
    torque_hi = np.concatenate([model.actuator_ctrlrange[a["act_ids"], 1]
                                for a in arms])
    phase = "ramp"
    u_cmd = np.clip(u_seed.copy(), torque_lo, torque_hi)
    u_clip_mask = np.zeros(nc.NU, dtype=bool)
    u_sat_mask = np.isclose(u_cmd, torque_lo, atol=1e-6) | np.isclose(
        u_cmd, torque_hi, atol=1e-6)
    move_duration = (0.0 if args.hold_only or abs(args.v_y) <= 1e-9
                     else abs(args.y_dist) / abs(args.v_y))
    required_end = t_hold_end + (0.3 if args.hold_only else move_duration + 0.4)
    steps = max(args.steps, int(np.ceil(required_end / DT)))
    for step in range(steps):
        t = step * DT
        # solve_ms is populated only on control updates; phase is held between
        # updates so the CSV reflects the actual reference phase.
        solve_ms = np.nan
        if step % CTRL_EVERY == 0:
            t0 = time.perf_counter()
            p, fmag, phase = build_p(t)
            u_last = nmpc.solve(joint_state(), p,
                                u_ref=gravity_comp() + clamp_balance_comp(fmag),
                                u_seed=u_seed)
            u_raw = np.asarray(u_last, dtype=float)
            u_cmd = np.clip(u_raw, torque_lo, torque_hi)
            u_clip_mask = np.abs(u_cmd - u_raw) > 1e-9
            # Count both explicit clipping and an optimizer solution lying on an
            # actuator bound. The latter is a real saturation even if np.clip
            # does not numerically change the command.
            u_sat_mask = u_clip_mask | np.isclose(u_cmd, torque_lo, atol=1e-6) | np.isclose(
                u_cmd, torque_hi, atol=1e-6)
            solve_ms = (time.perf_counter() - t0) * 1000.0
            if step == 0:
                print(f"[u0] {np.array2string(u_cmd, precision=12, floatmode='unique')}")
        for a in arms:
            arm_i = arms.index(a)
            data.ctrl[a["act_ids"]] = u_cmd[7 * arm_i:7 * arm_i + 7]
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
            time.sleep(DT)
        if step % 5 == 0:
            forces = contact_forces(model, data, arms, box_geom)
            w = pad_box_wrench_3d(model, data, arms, box_geom)
            fb, tb = box_net_wrench(model, data, arms, box_geom,
                                    data.xpos[box_body].copy())
            rv, quat = box_attitude(data, box_qpos)
            if step == 0:
                print(f"[t0] FL={forces['left']:.4f} FR={forces['right']:.4f} "
                      f"boxz={data.xpos[box_body][2]:.6f} "
                      f"tiltx={np.degrees(rv[0]):.4f} "
                      f"tipL={get_ee_tip(data, arms[0]['body_id'])}")
            log.append({
                "t": t + DT,
                "phase": phase,
                "ramp_s": smoothstep(t / args.t_ramp) if args.t_ramp > 1e-9 else 1.0,
                "moving_active": 1.0 if phase == "move" else 0.0,
                "box": data.xpos[box_body].copy(),
                "quat": quat.copy(),
                "rotvec": rv.copy(),
                "tipL": get_ee_tip(data, arms[0]["body_id"]).copy(),
                "tipR": get_ee_tip(data, arms[1]["body_id"]).copy(),
                "wL": w["left"], "wR": w["right"],
                "box_f": fb, "box_t": tb,
                "u_cmd": u_cmd.copy(),
                "u_clip_mask": u_clip_mask.copy(),
                "u_sat_mask": u_sat_mask.copy(),
                "FL": forces["left"], "FR": forces["right"],
                "solve_ms": solve_ms if step % CTRL_EVERY == 0 else np.nan,
                "solver_status": (float(nmpc.last_status)
                                   if step % CTRL_EVERY == 0 else np.nan),
                "qp_iter": (float(nmpc.last_qp_iter)
                             if step % CTRL_EVERY == 0 else np.nan),
                "solver_qp_ms": (nmpc.last_time_qp * 1000.0
                                  if step % CTRL_EVERY == 0 else np.nan),
                "solver_sim_ms": (nmpc.last_time_sim * 1000.0
                                   if step % CTRL_EVERY == 0 else np.nan),
                "solver_lin_ms": (nmpc.last_time_lin * 1000.0
                                   if step % CTRL_EVERY == 0 else np.nan),
            })
        if args.hold_only and t >= t_hold_end + 0.3:
            break
    if viewer is not None:
        viewer.close()
    metrics = summarize(log, args, initial_box_y)
    return log, metrics


def summarize(log, args, initial_box_y) -> dict:
    tt = np.array([r["t"] for r in log])
    box = np.array([r["box"] for r in log])
    rv = np.array([r["rotvec"] for r in log])
    tip_l = np.array([r["tipL"] for r in log])
    tip_r = np.array([r["tipR"] for r in log])
    fl = np.array([r["FL"] for r in log]); fr = np.array([r["FR"] for r in log])
    w_l_t = np.array([r["wL"][1] for r in log])
    w_r_t = np.array([r["wR"][1] for r in log])
    box_t = np.array([r["box_t"] for r in log])
    u_clip = np.array([r["u_clip_mask"] for r in log], dtype=bool)
    u_sat = np.array([r["u_sat_mask"] for r in log], dtype=bool)
    motion_commanded = (not args.hold_only and abs(float(args.y_dist)) > 1e-9
                        and abs(float(args.v_y)) > 1e-9)
    # hold 全窗（worst-case 评估用）：hold_only = [t_ramp, 结束]；moving = [t_ramp, hold_end]
    if not motion_commanded:
        sel = tt >= max(0.0, tt[-1] - 0.3)
        sel_hold = tt >= args.t_ramp
        sel_move = np.zeros_like(tt, dtype=bool)
    elif motion_commanded:
        sel = (tt >= args.t_ramp + args.t_hold - 0.3) & (tt <= args.t_ramp + args.t_hold)
        sel_hold = (tt >= args.t_ramp) & (tt <= args.t_ramp + args.t_hold)
        sel_move = np.array([r["phase"] == "move" for r in log], dtype=bool)
    if not sel.any():
        sel = tt >= tt[-1] - 0.3
    if not sel_hold.any():
        sel_hold = sel
    if not sel_move.any():
        sel_move = np.zeros_like(tt, dtype=bool)
    ys_mm = (tip_l[:, 1] - tip_r[:, 1]) * 1000.0
    rv_deg = np.abs(rv) * 180.0 / np.pi
    initial_box_y = float(initial_box_y)

    def max_or_nan(values):
        values = np.asarray(values)
        return float(np.max(values)) if values.size else float("nan")

    def mean_or_nan(values):
        values = np.asarray(values)
        return float(np.mean(values)) if values.size else float("nan")

    def drift_mm(values):
        values = np.asarray(values)
        return (float(np.max(np.abs(values - np.mean(values))) * 1000.0)
                if values.size else float("nan"))

    box_y_hold_max_mm = max_or_nan(np.abs(box[sel_hold, 1] - initial_box_y) * 1000.0)
    box_y_move_delta_mm = ((box[sel_move, 1] - box[sel_move, 1][0]) * 1000.0
                           if sel_move.any() else np.array([]))
    box_y_move_travel_mm = (-float(box_y_move_delta_mm[-1])
                            if box_y_move_delta_mm.size else float("nan"))
    box_y_move_travel_error_mm = (abs(box_y_move_travel_mm - abs(float(args.y_dist)) * 1000.0)
                                  if np.isfinite(box_y_move_travel_mm) else float("nan"))
    control_rows = [r for r in log if np.isfinite(r["solver_status"])]
    statuses = np.array([r["solver_status"] for r in control_rows], dtype=float)
    qp_iters = np.array([r["qp_iter"] for r in control_rows], dtype=float)
    status_bad = statuses[(statuses >= 0) & (statuses != 0)]
    move_observed = bool(sel_move.any())
    commanded_distance = 0.0 if args.hold_only else abs(float(args.y_dist))
    speed = abs(float(args.v_y))
    move_duration = commanded_distance / speed if speed > 1e-9 else float("inf")
    expected_move_end = args.t_ramp + args.t_hold + move_duration
    box_y_final_error_mm = abs((box[-1, 1] - initial_box_y) + commanded_distance) * 1000.0
    # The plant is still stepped for a short settling tail after the commanded
    # translation.  Evaluate the endpoint at the end of the commanded motion,
    # rather than at the final tail sample.
    move_end_t = args.t_ramp + args.t_hold + move_duration
    endpoint_idx = int(np.argmin(np.abs(tt - move_end_t)))
    box_y_endpoint_error_mm = abs(
        (box[endpoint_idx, 1] - initial_box_y) + commanded_distance
    ) * 1000.0
    box_y_initial_error_mm = box_y_endpoint_error_mm
    # 评价固定时间切换后的实际搬运行程。
    move_complete_after_settle = int(
        move_observed and tt[-1] + DT >= expected_move_end
        and box_y_move_travel_error_mm <= 2.0
    )
    move_complete = int(
        move_complete_after_settle
        and box_y_hold_max_mm <= HOLD_BOX_Y_TOL * 1000.0
        and box_y_initial_error_mm <= 2.0
    )
    m = {
        "force": args.force, "kf_scale": args.kf_scale, "q_y": args.q_y, "q_z": args.q_z,
        "eta_cop_y": args.eta_cop_y,
        "clamp_ff_scale": args.clamp_ff_scale,
        "clamp_force_kp": args.clamp_force_kp,
        "q_rv": args.q_rv, "q_x": args.q_x, "t_ramp": args.t_ramp, "t_hold": args.t_hold,
        "f_start": args.f_start,
        "transition_mode": "timed",
        "motion_commanded": int(motion_commanded),
        "moving_start_s": args.t_ramp + args.t_hold,
        "tilt_x_max_deg": max_or_nan(rv_deg[sel, 0]),
        "tilt_y_max_deg": max_or_nan(rv_deg[sel, 1]),
        "tilt_z_max_deg": max_or_nan(rv_deg[sel, 2]),
        "tilt_x_hold_max_deg": max_or_nan(rv_deg[sel_hold, 0]),
        "tilt_y_hold_max_deg": max_or_nan(rv_deg[sel_hold, 1]),
        "tilt_z_hold_max_deg": max_or_nan(rv_deg[sel_hold, 2]),
        "z_sag_mm": mean_or_nan(box[sel, 2] - BOX_CENTER[2]) * 1000.0,
        "z_drift_mm": drift_mm(box[sel, 2]),
        "z_drift_hold_mm": drift_mm(box[sel_hold, 2]),
        "FL_mean": mean_or_nan(fl[sel]), "FR_mean": mean_or_nan(fr[sel]),
        "F_diff_max_N": max_or_nan(np.abs(fl[sel] - fr[sel])),
        "F_diff_hold_max_N": max_or_nan(np.abs(fl[sel_hold] - fr[sel_hold])),
        "y_sync_mean_mm": mean_or_nan(np.abs(ys_mm)),
        "y_sync_max_mm": max_or_nan(np.abs(ys_mm)),
        "y_sync_hold_max_mm": max_or_nan(np.abs(ys_mm[sel_hold])),
        "tilt_x_move_max_deg": max_or_nan(rv_deg[sel_move, 0]),
        "tilt_y_move_max_deg": max_or_nan(rv_deg[sel_move, 1]),
        "tilt_z_move_max_deg": max_or_nan(rv_deg[sel_move, 2]),
        "z_drift_move_mm": drift_mm(box[sel_move, 2]),
        "F_diff_move_max_N": max_or_nan(np.abs(fl[sel_move] - fr[sel_move])),
        "contact_moment_left_move_max_Nm": max_or_nan(
            np.linalg.norm(w_l_t[sel_move], axis=1)),
        "contact_moment_right_move_max_Nm": max_or_nan(
            np.linalg.norm(w_r_t[sel_move], axis=1)),
        "contact_moment_box_move_max_Nm": max_or_nan(
            np.linalg.norm(box_t[sel_move], axis=1)),
        "y_sync_move_mean_mm": mean_or_nan(np.abs(ys_mm[sel_move])),
        "y_sync_move_max_mm": max_or_nan(np.abs(ys_mm[sel_move])),
        "actuator_clip_fraction": mean_or_nan(u_clip),
        "actuator_saturation_fraction": mean_or_nan(u_sat),
        "actuator_clip_move_fraction": mean_or_nan(u_clip[sel_move]),
        "actuator_saturation_move_fraction": mean_or_nan(u_sat[sel_move]),
        "move_observed": int(move_observed),
        "move_complete": move_complete,
        "move_complete_after_settle": move_complete_after_settle,
        "move_expected_end_s": expected_move_end,
        "move_end_t_s": float(tt[-1]),
        "initial_box_y": initial_box_y,
        "box_y_final_drift_mm": (float(box[-1, 1] - initial_box_y) * 1000.0),
        "box_y_hold_max_mm": box_y_hold_max_mm,
        "box_y_initial_error_mm": box_y_initial_error_mm,
        "box_y_move_travel_mm": box_y_move_travel_mm,
        "box_y_move_travel_error_mm": box_y_move_travel_error_mm,
        "box_y_endpoint_error_mm": box_y_endpoint_error_mm,
        "box_y_final_error_mm": box_y_final_error_mm,
        "solver_status_bad_count": int(status_bad.size),
        "solver_status_zero_fraction": (float(np.mean(statuses == 0.0))
                                         if statuses.size else float("nan")),
        "qp_iter_mean": mean_or_nan(qp_iters),
        "qp_iter_max": max_or_nan(qp_iters),
        "solver_qp_ms_mean": mean_or_nan(
            [r["solver_qp_ms"] for r in control_rows]),
        "solver_sim_ms_mean": mean_or_nan(
            [r["solver_sim_ms"] for r in control_rows]),
        "solver_lin_ms_mean": mean_or_nan(
            [r["solver_lin_ms"] for r in control_rows]),
        "solve_ms_mean": mean_or_nan([r["solve_ms"] for r in log
                                       if np.isfinite(r["solve_ms"])]),
        "solve_ms_max": max_or_nan([r["solve_ms"] for r in log
                                     if np.isfinite(r["solve_ms"])]),
    }
    return m


# ======================================================================
# 主流程
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--force", type=float, default=15, help="稳定夹持力参考 (N)")
    parser.add_argument("--kf_scale", type=float, default=0.05, help="力误差增益 (nmpcc_cart 配置)")
    parser.add_argument("--q_x", type=float, default=500.0, help="Q x 位置权重")
    parser.add_argument("--q_y", type=float, default=25000.0, help="Q y 位置权重")
    parser.add_argument("--q_z", type=float, default=10000.0, help="Q z 位置权重")
    parser.add_argument("--q_rv", type=float, default=600.0, help="Q 姿态(rotvec)权重（三轴同值）")
    parser.add_argument("--eta_cop_y", type=float, default=0.8,
                        help="论文合作映射 eta_cop 的 y 分量；0=绝对 y 跟踪，0.8=同步优先")
    parser.add_argument("--clamp_ff_scale", type=float, default=1.0,
                        help="夹持力 actuator-reference 前馈比例；1=历史基线，0=消融")
    parser.add_argument("--clamp_force_kp", type=float, default=0.0,
                        help="附加实测法向力反馈增益；0=关闭，非论文 NMPCC 层")
    parser.add_argument("--t_ramp", type=float, default=0.5, help="10N→目标力的平滑时间 (s)")
    parser.add_argument("--t_hold", type=float, default=0.1, help="固定 Holding 时间，结束后直接 Moving (s)")
    parser.add_argument("--f_start", type=float, default=10.0,
                        help="启动夹持力参考 (N)，同时用于初始 IK 压紧和控制器首拍")
    parser.add_argument("--v_y", type=float, default=-0.01)
    parser.add_argument("--y_dist", type=float, default=0.05)
    parser.add_argument("--hold_only", action="store_true")
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--Tp", type=float, default=0.10)
    parser.add_argument("--sim_steps", type=int, default=4,
                        help="acados ERK 每个 shooting interval 的积分步数（推荐2；旧值4）")
    parser.add_argument("--no_csv", action="store_true")
    parser.add_argument("--no_viewer", action="store_true", help="关闭 MuJoCo 可视化窗口")
    parser.add_argument("--sweep_forces", type=str, default=None,
                        help="逗号分隔力扫描（同求解器，不重建）")
    parser.add_argument("--sweep_kf", type=str, default=None,
                        help="逗号分隔 kf_scale 扫描（每值重建一次求解器）")
    parser.add_argument("--sweep_qrv", type=str, default=None,
                        help="逗号分隔 Q_rv 扫描（每值重建一次求解器）")
    parser.add_argument("--sweep_tramp", type=str, default=None,
                        help="逗号分隔 t_ramp 扫描（仅参考 profile，不重建求解器）")
    parser.add_argument("--sweep_thold", type=str, default=None,
                        help="逗号分隔 t_hold 扫描（仅参考 profile，不重建求解器）")
    parser.add_argument("--sweep_fstart", type=str, default=None,
                        help="逗号分隔 f_start 扫描（每值重新收敛轻压位）")
    parser.add_argument("--repeats", type=int, default=1,
                        help="每候选重复次数（同进程 reset_world + 重收敛 + 求解器种子复位）")
    args = parser.parse_args()
    print(f"[command] y_dist={args.y_dist:.6f} m, v_y={args.v_y:.6f} m/s, "
          f"hold_only={args.hold_only}")
    print(f"[output] CSV and Fig. 12 files: {os.path.abspath(OUT_DIR)}")

    sweep_forces = ([float(x) for x in args.sweep_forces.split(",")]
                    if args.sweep_forces else None)
    sweep_kfs = ([float(x) for x in args.sweep_kf.split(",")]
                 if args.sweep_kf else None)
    sweep_qrvs = ([float(x) for x in args.sweep_qrv.split(",")]
                  if args.sweep_qrv else None)
    sweep_tramps = ([float(x) for x in args.sweep_tramp.split(",")]
                    if args.sweep_tramp else None)
    sweep_tholds = ([float(x) for x in args.sweep_thold.split(",")]
                    if args.sweep_thold else None)
    sweep_fstarts = ([float(x) for x in args.sweep_fstart.split(",")]
                     if args.sweep_fstart else None)
    any_sweep = any(x is not None for x in (sweep_forces, sweep_kfs, sweep_qrvs,
                                            sweep_tramps, sweep_tholds, sweep_fstarts))

    scene_xml = build_scene()
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    arms = make_arms(model)
    for a in arms:
        joint_ids = a["joint_ids"]
        a["act_ids"] = [name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                    f"{model.joint(j).name}_motor") for j in joint_ids]

    box_body = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_geom = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    box_joint = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")
    box_qpos = int(model.jnt_qposadr[box_joint])
    ids = {"box_body": box_body, "box_geom": box_geom, "box_qpos": box_qpos}

    # 软接触 tc=0.02（= 旧人形场景 calib_out.txt kx=3575 的标定路径）——场景几何换掉后，
    # 接触标定也必须换（K_STIFF/REST_GAP/CALIB_INTERCEPT 全部来自 grasp_pose 人形旧标定值）。
    soften_contacts(model, box_geom, [a["pad_id"] for a in arms])

    reset_world(model, data, arms, box_qpos)

    # --- 目标姿态：轻接触起点 + 压入终点（不改几何）---
    base_targets = grasp_targets(arms)

    def press_for(force_n):
        delta = force_n / (2.0 * K_STIFF)
        return {
            a["side"]: base_targets[a["side"]] + np.array([-a["sign"] * delta, 0.0, 0.0])
            for a in arms
        }

    press_light = press_for(args.f_start)      # 安全轻接触（默认 2N，摩擦持重容量 20N > 箱重 8.4N）
    press_full_for = press_for
    solve_end_grip(model, data, arms, press_light, IK_ITERS)

    fk = nc.mujoco_fk_symbols(
        model,
        [{"body_id": a["body_id"], "joint_ids": a["joint_ids"]} for a in arms],
        0.1001,
    )
    # 按给定 force 重建 rv_ref 用的抓取姿态（solve_end_grip 已放在轻压位）
    def make_rv_ref():
        return {a["side"]: nc.rotvec_of_R(data.xmat[a["body_id"]].reshape(3, 3))
                for a in arms}
    rv_ref = make_rv_ref()
    arm_damping(model, [d for a in arms for d in a["dofs"]], 0.5)

    umax_parts = []
    for a in arms:
        umax_parts.append(model.actuator_ctrlrange[a["act_ids"], 1])
    umax = np.concatenate(umax_parts)

    NMPC_CACHE = {}

    def build_nmpc(kf):
        cfg = nc.NMPCCCartConfig(
            N=args.N, Tp=args.Tp,
            eta_cop=np.array([[0.0, args.eta_cop_y, 0.0, 0.0, 0.0, 0.0]] * 2),
            eta_f=np.array([[0.8, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2),
            Q=np.array([args.q_x, args.q_y, args.q_z, args.q_rv, args.q_rv, args.q_rv]),
            eta_dev=3000.0,
            eta_u=0.02,
            eta_u_vec=np.tile([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0], 2),
            P_mul=3.0,
            kx=K_STIFF,
            rest_gap=REST_GAP,
            kf_scale=kf,
            vmax=0.05,
            vz_max=0.001,
            sim_method_num_steps=args.sim_steps,
            sim_method_num_stages=4,
        )
        return nc.NMPCCCart(cfg, fk, umin=-umax, umax=umax)

    def run_one(kf, force, tag):
        args.force = force
        args.kf_scale = kf
        # 每 run 复位世界并按 **当前 f_start** 重新收敛轻压位（确定性）
        reset_world(model, data, arms, box_qpos)
        pl = press_for(args.f_start)
        solve_end_grip(model, data, arms, pl, IK_ITERS)
        rv_ref_now = make_rv_ref()
        pf = press_full_for(force)
        rfn, th_end = phase_profile(
            force, 0.0 if args.hold_only else args.v_y, args.y_dist,
            args.t_ramp, args.t_hold, pl, pf, rv_ref_now,
            f_start=args.f_start)
        if not any_sweep:
            reconciliation(model, data, arms, box_qpos, fk, base_targets,
                           press_light, force)
        viewer = None
        if not args.no_viewer and not any_sweep:
            viewer = mujoco.viewer.launch_passive(model, data)
            viewer.cam.lookat[:] = [0.5, 0.0, 0.72]   # 人形抱箱：箱中心在肩高前方
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -12
            viewer.cam.azimuth = 180
            viewer.sync()
            print("viewer opened; close the window to stop the experiment")
        # 求解器按 OCP 参数缓存（kf/Q/sim_steps 相同则复用已编译实例）；
        # 每 run 用 u_seed 冷启动（nmpc._warm=None + 预求解），保证重复运行等价于独立进程冷启动。
        okey = (round(kf, 6), args.q_x, args.q_y, args.q_z, args.q_rv,
                args.eta_cop_y, args.sim_steps, args.N, args.Tp)
        nmpc = NMPC_CACHE.get(okey)
        if nmpc is None:
            nmpc = build_nmpc(kf)
            NMPC_CACHE[okey] = nmpc
            print(f"NMPCC built: K={K_STIFF:.3f}, rest_gap={REST_GAP:.6f}, "
                  f"N={args.N}, Tp={args.Tp:.3f}, force={force:.1f}N, kf_scale={kf:.3f}, "
                  f"Q_y={args.q_y:.0f}, Q_z={args.q_z:.0f}, Q_rv={args.q_rv:.0f}, "
                  f"eta_cop_y={args.eta_cop_y:.2f}")
        else:
            print(f"NMPCC cached (same OCP params); force={force:.1f}N, "
                  f"t_ramp={args.t_ramp:.2f}, t_hold={args.t_hold:.2f}, f_start={args.f_start:.1f}")
        nmpc._warm = None
        # 关键：复位 acados 内部求解器内存（含 HPIPM 因子分解热身状态）。
        # 若不复位，上一 run 的 QP 分解状态会以 warm start 形式进入本 run 的第一次 QP，
        # 造成同配置重复运行出现 0.5°~4.3° 的 tilt 分支分歧（初值敏感性的主要来源）。
        try:
            nmpc.solver.reset()
        except Exception as e:
            print(f"[warn] nmpc.solver.reset() failed: {e}")
        # 进一步：关闭 SQP-RTI 的 QP 热身（若支持；hpipm 记忆热身是 run 间串扰的通道）
        try:
            nmpc.solver.options_set("qp_warm_start", 0)
            nmpc.solver.options_set("warm_start_first_qp", 0)
        except Exception as e:
            print(f"[warn] qp_warm_start disable failed: {e}")
        t0 = time.perf_counter()
        log, m = run_case(model, data, arms, ids, args, nmpc, rfn, th_end,
                          viewer=viewer)
        dt = time.perf_counter() - t0
        print(f"[run {tag}] F={force:.1f} kf={kf:.3f} "
              f"timed_hold={args.t_hold:.2f}s move_start={m['moving_start_s']:.2f}s | "
              f"{'move=DISABLED' if not m['motion_commanded'] else ('move=COMPLETE' if m['move_complete'] else 'move=INCOMPLETE')} "
              f"(after_settle={m['move_complete_after_settle']}, observed={m['move_observed']}, "
              f"hold_dy={m['box_y_hold_max_mm']:.2f}mm, "
              f"travel={m['box_y_move_travel_mm']:.2f}mm, "
              f"initial_end_err={m['box_y_initial_error_mm']:.2f}mm) | "
              f"tilt=({m['tilt_x_max_deg']:.2f},"
              f"{m['tilt_y_max_deg']:.2f},{m['tilt_z_max_deg']:.2f})° | "
              f"z_sag={m['z_sag_mm']:+.2f}mm z_drift={m['z_drift_mm']:.2f}mm | "
              f"F=({m['FL_mean']:.2f},{m['FR_mean']:.2f})N diff={m['F_diff_max_N']:.2f}N | "
              f"ysync={m['y_sync_mean_mm']:.2f}/{m['y_sync_max_mm']:.2f}mm | "
              f"status_bad={m['solver_status_bad_count']} qp={m['qp_iter_mean']:.1f} "
              f"solve={m['solve_ms_mean']:.2f}ms ({dt:.1f}s)")
        if not args.no_csv:
            os.makedirs(OUT_DIR, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(OUT_DIR, f"exp2_latest_{tag}_{stamp}.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["t", "phase", "ramp_s", "moving_active",
                            "boxx", "boxy", "boxz", "q_w", "q_x", "q_y", "q_z",
                            "rvx", "rvy", "rvz",
                            "tipLx", "tipLy", "tipLz", "tipRx", "tipRy", "tipRz",
                            "FLx", "FLy", "FLz", "FLtx", "FLty", "FLtz",
                            "FRx", "FRy", "FRz", "FRtx", "FRty", "FRtz",
                            "BCFx", "BCFy", "BCFz", "BCtx", "BCty", "BCtz",
                            "FL", "FR", *[f"u{i + 1}" for i in range(nc.NU)],
                            *[f"uclip{i + 1}" for i in range(nc.NU)],
                            *[f"usat{i + 1}" for i in range(nc.NU)],
                            "solve_ms", "solver_status", "qp_iter",
                            "solver_qp_ms", "solver_sim_ms", "solver_lin_ms"])
                for r in log:
                    w.writerow([r["t"], r["phase"], r["ramp_s"], r["moving_active"],
                                *r["box"], *r["quat"], *r["rotvec"],
                                *r["tipL"], *r["tipR"], *r["wL"][0], *r["wL"][1],
                                *r["wR"][0], *r["wR"][1], *r["box_f"], *r["box_t"],
                                r["FL"], r["FR"], *r["u_cmd"],
                                *r["u_clip_mask"].astype(int), *r["u_sat_mask"].astype(int),
                                r["solve_ms"], r["solver_status"],
                                r["qp_iter"], r["solver_qp_ms"], r["solver_sim_ms"],
                                r["solver_lin_ms"]])
            print(f"saved {path}")
            # 每次保存 CSV 后自动生成 Fig. 12 的 PNG 和 SVG；绘图失败不影响仿真结果。
            plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "plot_exp2_fig12.py")
            plot_prefix = os.path.join(OUT_DIR, f"fig12_{tag}_{stamp}")
            try:
                subprocess.run(
                    [sys.executable, plot_script, path, "--out", plot_prefix],
                    check=True,
                )
            except Exception as exc:
                print(f"[warn] automatic Fig. 12 generation failed: {exc}")
        return m

    if any_sweep:
        # sweep：几何对账打印一次（每次 run 几何/复位完全相同）
        reconciliation(model, data, arms, box_qpos, fk, base_targets, press_light,
                       args.force)
        kfs = sweep_kfs or [args.kf_scale]
        forces = sweep_forces or [args.force]
        qrvs = sweep_qrvs or [args.q_rv]
        tramps = sweep_tramps or [args.t_ramp]
        tholds = sweep_tholds or [args.t_hold]
        fstarts = sweep_fstarts or [args.f_start]
        rows = []
        for kf in kfs:
            for force in forces:
                for qrv in qrvs:
                    for tramp in tramps:
                        for thold in tholds:
                            for fstart in fstarts:
                                args.q_rv, args.t_ramp = qrv, tramp
                                args.t_hold, args.f_start = thold, fstart
                                for rep in range(args.repeats):
                                    tag = (f"f{force:g}_kf{kf:g}_rv{qrv:g}"
                                           f"_tr{tramp:g}_th{thold:g}_fs{fstart:g}_r{rep}")
                                    rows.append(run_one(kf, force, tag))
        os.makedirs(OUT_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(OUT_DIR, f"sweep_hold_{stamp}.csv")
        cols = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\n=== sweep summary: {path} ===")
        print("rep | force | kf | qrv | tramp | thold | fstart | "
              "tiltx_hold | tilty_hold | tiltz_hold | z_sag | z_drift_hold | "
              "FL | FR | Fdiff_hold | ysync_hold | solve_ms")
        for i, m in enumerate(rows):
            print(f"{i % args.repeats:3d} | {m['force']:5.1f} | {m['kf_scale']:.2f} | "
                  f"{m['q_rv']:5.0f} | {m['t_ramp']:.2f} | {m['t_hold']:.2f} | "
                  f"{m['f_start']:.1f} | "
                  f"{m['tilt_x_hold_max_deg']:5.2f} | {m['tilt_y_hold_max_deg']:5.2f} | "
                  f"{m['tilt_z_hold_max_deg']:5.2f} | {m['z_sag_mm']:6.2f} | "
                  f"{m['z_drift_hold_mm']:5.2f} | {m['FL_mean']:5.2f} | {m['FR_mean']:5.2f} | "
                  f"{m['F_diff_hold_max_N']:5.2f} | {m['y_sync_hold_max_mm']:6.2f} | "
                  f"{m['solve_ms_mean']:5.2f}")
        return

    m = run_one(args.kf_scale, args.force, "single")
    print("--- latest Experiment II (summary) ---")
    for k, v in m.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()