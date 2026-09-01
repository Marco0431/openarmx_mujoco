#!/usr/bin/env python3
"""ikforce_impedance.py —— 论文阻抗模式（ηf 软度旋钮）+ ikforce 力矩级 OSC（OpenArmX 双臂）

论文: Zhao 等, "A Dual-Arm Robot Cooperation Framework Based on a Nonlinear Model
Predictive Cooperative Control", IEEE/ASME TMech 2024（III-B 阻抗模式 / 第 V 节实验）。

修正记录（基于上一轮诊断）：
  原 x 轴版本在"双臂尖端靠中心面"位形下失稳：沿 x 的操作空间惯量 Λ_xx≈0.0002 kg 接近奇异，
  控制器该方向的力与阻尼 Λ·K·e、Λ·K_D·ẋ 几乎为零 → 尖端沿 x 自由漂移（发散根源；实验一
  能稳是因尖端钉在奇异点上）。本版夹持轴 = 两尖端连线 û（q_rest 下两尖端相距约 0.53 m，
  沿 ±y 对望，Λ 良态）——该位形正是 ikforce 头测验证稳定的跟踪区。

语义（论文对照）：
  式(12) 合作指标 ε_mult = (1−ηf)[(1−ηcop)e_trac − ηcop·e_cop] − ηf·e_f；阻抗模式 = ηcop=0、
  ηf∈(0,I)：双臂像弹簧一样与环境互动，"柔度"由 ηf 决定，ηf=0 刚度最大、ηf→1 刚度为零（力优先）。
  式(27) 成对力：双臂间虚拟负载弹簧 f_1 = K(r_2−r_1)、f_2 = −f_1（夹持轴上）。
  式(6) 力反馈闭环 F_fb = F_ref + K_FORCE·(F_ref − P)。

控制律（每臂、每控制周期，ikforce 求解结构 + 式12 的 ηf 混合）：
  τ = qfrc_bias + Jᵀ·F_task − K_DQ·q̇ + N·[K_PQ·(q_init−q) − K_DQ·q̇]
  F_task = Λ·[K_POS·e − K_D_GRASP·ẋ]        （除夹持轴外为纯位置阻抗）
  夹持轴(û)分量：F_axis = (1−ηf)·pos_axis − damp_axis − ηf·F_fb
    （pos/damp 为位置/阻尼沿轴分量；力反馈取负号 = 补夹向内）
  稳态平衡：ηf=0 → P ≈ K_pair·2Δ（跟踪）；ηf→1 → P → F_ref（恒力、位置让位）→ 曲线斜率随 ηf 单调下降。

实验（输出 results/ SVG+PNG，同 fig8/fig10 风格）：
  一、阻抗曲线扫描：命令夹持口收缩 Δ 0→6mm，记录 P(Δ)、压缩量 δ、让位量 e_in
    (a) P vs Δ（不同 ηf 的阻抗曲线，斜率=有效刚度）(b) Keff vs ηf (c) 稳态压缩 vs ηf
    (d) 让位量 vs ηf（Fig.7 同款"位置随 ηf 迁移"）
  二、扰动会话（论文 ExpIII 语义）：ηf=0.8 时给左臂端面沿夹持轴向 40N×0.4s 脉动
    → 弹簧力冲高、双臂协同让位、释放后自行回位

运行（acados 环境，仅用 mujoco/numpy/matplotlib）:
  D:\\acados_env\\python.exe -u ikforce_impedance.py --check
  D:\\acados_env\\python.exe -u ikforce_impedance.py          # 扫描+扰动+出图
  D:\\acados_env\\python.exe -u ikforce_impedance.py --viewer # 拖红球改夹持中心
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import mujoco
import mujoco.viewer

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

# ======================================================================
# 实验参数
# ======================================================================
K_PAIR = 2500.0              # 虚拟负载弹簧刚度 (N/m)：±6mm 收缩 → ±30N（论文 diag(1000,0) 同量级）
F_REF = 8.0                  # 期望夹持力 (N)（论文 ExpII 的"恒定夹持力"）
K_FORCE = 2.0                # 力反馈增益（论文式6 的 K_force）
K_D_GRASP = 55.0             # 夹持回路笛卡尔阻尼 (N·s/m)
DELTA_MAX = 0.006            # 收缩命令上限 Δ (m)
ETA_F_SWEEP = (0.0, 0.4, 0.8, 1.0)
RAMP_S = 1.0                 # 收缩 ramp 时长 (s)
HOLD_S = 1.2                 # 保持时长 (s)
DIST_F = 40.0                # 扰动幅值 (N)
DIST_DUR = 0.4               # 扰动持续 (s)
DIST_T = 1.2                 # 扰动开始时刻 (s)
DIST_SIDE = "left"
DIST_SESSION_S = 4.0
SCENE_XML = BASE_DIR / "ikforce_imp_scene.xml"
FIG_DIR = BASE_DIR / "results" / "impedance"


# ======================================================================
# 场景：地面/灯光/中点红球(mocap 可拖)+ 端面可视化球；物理 = 虚拟弹簧（无碰撞）
# ======================================================================
def build_scene():
    runtime_xml = generate_runtime_xml()
    inject_motors(runtime_xml)
    strip_collisions(runtime_xml)
    SCENE_XML.write_text(
        f"""<mujoco model="ikforce_imp_scene">
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

    <!-- 中点红球：mocap 可拖动（viewer 改夹持中心）；contype=0 纯可视化 -->
    <body name="imp_target" pos="0.35 0 0.55" mocap="true">
      <geom name="imp_target" type="sphere" size="0.03" rgba="1 0 0 0.45" contype="0" conaffinity="0"/>
    </body>
    <body name="imp_tip_left" pos="0 0 0" mocap="true">
      <geom name="imp_tip_left" type="sphere" size="0.018" rgba="0 1 1 0.8" contype="0" conaffinity="0"/>
    </body>
    <body name="imp_tip_right" pos="0 0 0" mocap="true">
      <geom name="imp_tip_right" type="sphere" size="0.018" rgba="1 0.6 0 0.8" contype="0" conaffinity="0"/>
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


def parse_arms(model, data):
    arms = []
    for cfg in ARM_CONFIGS:
        body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["jacobi_body"])
        joint_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg["joint_names"]]
        dofs = [int(model.jnt_dofadr[j]) for j in joint_ids]
        qpos = [int(model.jnt_qposadr[j]) for j in joint_ids]
        act_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_motor") for n in cfg["joint_names"]]
        arms.append({
            "side": cfg["side"],
            "q_rest": cfg["q_rest"],
            "body_id": body_id,
            "joint_ids": joint_ids,
            "dofs": dofs,
            "qpos": qpos,
            "act_ids": act_ids,
        })
    return arms


def nominal_geometry(model, data, arms):
    """q_rest 下的两尖端间距 D0、连线单位向量 û(R→L)、各臂轴向（L 向外=û / R 向外=−û）。
    该 q_rest 位形是 ikforce 已验证稳定的跟踪区（夹持轴处操作空间惯量良态）。"""
    for a in arms:
        data.qpos[a["qpos"]] = a["q_rest"].copy()
        clamp_qpos(model, data, a["qpos"], a["joint_ids"])
        data.qvel[a["dofs"]] = 0.0
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    tips = {a["side"]: get_ee_tip(data, a["body_id"]) for a in arms}
    d_vec = tips["left"] - tips["right"]          # 由 R 指向 L
    d0 = float(np.linalg.norm(d_vec))
    u = d_vec / d0
    for a in arms:
        a["outward"] = u if a["side"] == "left" else -u
        a["inward"] = -a["outward"]
        a["tip0"] = tips[a["side"]].copy()
    return d0, u, tips


# ======================================================================
# 虚拟负载弹簧（论文式27）：δ = D0 − ||rL−rR||，f_L = +K_pair·δ·û，f_R = −f_L
# 反力经 mj_applyFT 累加进每步 qfrc_target 缓冲（mj_step 消费）
# ======================================================================
def _apply_pair_spring(model, data, arms, d0, u, k_pair, qfrc_target):
    tips = {a["side"]: get_ee_tip(data, a["body_id"]) for a in arms}
    gap = float(np.linalg.norm(tips["left"] - tips["right"]))
    delta = d0 - gap
    p = k_pair * delta
    fL = p * u
    fR = -p * u
    mujoco.mj_applyFT(model, data, fL, np.zeros(3), tips["left"], arms[0]["body_id"], qfrc_target)
    mujoco.mj_applyFT(model, data, fR, np.zeros(3), tips["right"], arms[1]["body_id"], qfrc_target)
    return p


# ======================================================================
# 阻抗控制求解（ikforce 结构 + 论文式12 的 ηf 混合，夹持轴 û）
# ======================================================================
def solve_torque_impedance(model, data, arm, ref, p, u, eta_f, f_ref, use_nullspace=True):
    body_id = arm["body_id"]
    dofs = arm["dofs"]
    tip = get_ee_tip(data, body_id)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, tip, body_id)
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

    e = ref - tip
    x_dot = jac @ data.qvel[dofs]
    pos_sp = lam @ (K_POS * e)
    damp_sp = lam @ (K_D_GRASP * x_dot)
    force = pos_sp - damp_sp
    # 夹持轴 û 的混合：位置弹簧按 (1−ηf) 保留，剩余权重给力反馈（补夹向内）
    out = u if arm["side"] == "left" else -u
    pos_axis = float(np.dot(pos_sp, out))
    damp_axis = float(np.dot(damp_sp, out))
    f_axis = float(np.dot(force, out))
    fb = f_ref + K_FORCE * (f_ref - p)                 # >0 = 补夹（作用方向 = 向内 = −out）
    f_axis_new = (1.0 - eta_f) * pos_axis - damp_axis - eta_f * fb
    force = force + (f_axis_new - f_axis) * out
    e_in = float(np.dot(e, out))                        # 让位量：>0 = 垫片落后参考（向外让位）

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
        tau_posture = K_PQ * (arm["q_init"] - data.qpos[arm["qpos"]]) - K_DQ * data.qvel[dofs]
        tau_null = null_proj @ tau_posture
        n_null = np.linalg.norm(tau_null)
        if n_null > MAX_NULL_TORQUE and n_null > 1e-9:
            tau_null = tau_null / n_null * MAX_NULL_TORQUE
        tau = tau + tau_null
    return tau, e, e_in


def refs_at_delta(arms, tips0, u, center_off, delta):
    """各臂参考端面：ref_i = tip0_i + center_off + Δ·inward_i（收缩命令 Δ）。"""
    refs = {}
    for a in arms:
        off = delta * a["inward"]
        refs[a["side"]] = a["tip0"] + center_off + off
    return refs


# ======================================================================
# 会话：单 ηf 的收缩扫描 + 可选扰动
# ======================================================================
def run_session(model, data, arms, d0, u, eta_f, f_ref, with_disturb=False):
    for a in arms:
        data.qpos[a["qpos"]] = a["q_rest"].copy()
        clamp_qpos(model, data, a["qpos"], a["joint_ids"])
        data.qvel[a["dofs"]] = 0.0
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)

    t_ramp0 = 0.4
    t_ramp1 = t_ramp0 + RAMP_S
    dur = t_ramp1 + HOLD_S
    if with_disturb:
        dur = DIST_SESSION_S
    n = int(round(dur / IK_DT))

    log = {"t": [], "D": [], "P": [], "e_in": {}, "tip": {}, "ref": {}, "tau": [], "dist": []}
    for a in arms:
        log["e_in"][a["side"]] = []
        log["tip"][a["side"]] = []
        log["ref"][a["side"]] = []
    sat = 0
    center_off = np.zeros(3)

    for k in range(n):
        t = k * IK_DT
        if with_disturb:
            delta = 0.0
        else:
            if t <= t_ramp0:
                delta = 0.0
            elif t <= t_ramp1:
                tt = (t - t_ramp0) / RAMP_S
                delta = DELTA_MAX * tt * tt * (3.0 - 2.0 * tt)
            else:
                delta = DELTA_MAX

        refs = refs_at_delta(arms, None, u, center_off, delta)
        qf_t = np.zeros(model.nv)
        dist = 0.0
        if with_disturb and DIST_T <= t < DIST_T + DIST_DUR:
            side = arms[0]["side"] if DIST_SIDE == "left" else arms[1]["side"]
            a = next(x for x in arms if x["side"] == side)
            tip = get_ee_tip(data, a["body_id"])
            fv = DIST_F * a["inward"]                    # 向负载方向推（挤压弹簧）
            mujoco.mj_applyFT(model, data, fv, np.zeros(3), tip, a["body_id"], qf_t)
            dist = DIST_F
        p = _apply_pair_spring(model, data, arms, d0, u, K_PAIR, qf_t)
        mujoco.mj_forward(model, data)

        taus, stat = [], []
        for a in arms:
            tau, e, e_in = solve_torque_impedance(model, data, a, refs[a["side"]], p, u,
                                                  eta_f, f_ref, use_nullspace=True)
            taus.append((a, tau))
            stat.append((a, e, e_in))
        for a, tau in taus:
            data.ctrl[a["act_ids"]] = tau
        mujoco.mj_step(model, data)
        if t >= t_ramp0:
            for a, tau in taus:
                low, high = model.actuator_ctrlrange[a["act_ids"]].T
                if np.any(tau < low - 1e-9) or np.any(tau > high + 1e-9):
                    sat += 1

        log["t"].append(t)
        log["D"].append(delta)
        log["P"].append(p)
        log["dist"].append(dist)
        log["tau"].append(np.concatenate([tau for _, tau in taus])[None, :])
        for a, e, e_in in stat:
            log["e_in"][a["side"]].append(e_in)
            log["tip"][a["side"]].append(get_ee_tip(data, a["body_id"]))
            log["ref"][a["side"]].append(refs[a["side"]])

    out = dict(log)
    out["tau"] = np.concatenate(out["tau"], axis=0)
    out["sat"] = sat
    for a in arms:
        out["e_in"][a["side"]] = np.array(out["e_in"][a["side"]])
        out["tip"][a["side"]] = np.array(out["tip"][a["side"]])
        out["ref"][a["side"]] = np.array(out["ref"][a["side"]])
    return out


# ======================================================================
# 报表与出图
# ======================================================================
def t_ramp1_for(_res) -> float:
    return 0.4 + RAMP_S


def report_sweep(results, eta_list, d0):
    print("\n--- 阻抗曲线扫描报表（稳态点：保持段均值）---")
    print(f"负载弹簧 K_PAIR={K_PAIR:.0f} N/m | 目标夹持力 F_ref={F_REF:.1f} N | "
          f"名义间距 D0={d0*1000:.1f} mm | 收缩 Δ_max={DELTA_MAX*1000:.0f} mm")
    for res, eta in zip(results, eta_list):
        t = np.array(res["t"])
        hold = t >= t_ramp1_for(res)
        p = np.array(res["P"])
        P_set = float(np.mean(p[hold])) if np.any(hold) else float(np.mean(p))
        D = np.array(res["D"])
        ramp = (t >= 0.45) & (t <= max(t) - 0.05)
        slope = float(np.polyfit(D[ramp], p[ramp], 1)[0]) if np.any(ramp) else 0.0
        eL = np.array(res["e_in"]["left"])
        eR = np.array(res["e_in"]["right"])
        tipL = np.array(res["tip"]["left"])
        tipR = np.array(res["tip"]["right"])
        comp = float((d0 - np.linalg.norm(tipL[hold].mean(axis=0) - tipR[hold].mean(axis=0))) * 1000)
        print(f"  ηf={eta:<4} P_set={P_set:7.2f} N | Keff={slope:7.1f} N/m | 压缩δ={comp:6.2f} mm "
              f"| 让位 e_in_L/R={np.mean(eL[hold])*1000:6.2f}/{np.mean(eR[hold])*1000:6.2f} mm")


def report_disturb(res):
    print("\n--- 扰动会话报表（论文 ExpIII 语义：强推 → 协同让位 → 释放回位）---")
    t = np.array(res["t"])
    p = np.array(res["P"])
    p_rest = float(np.mean(p[t < DIST_T]))
    win = (t >= DIST_T) & (t < DIST_T + DIST_DUR)
    p_peak = float(np.max(np.abs(p[win]))) if np.any(win) else float("nan")
    p_back = float(np.mean(p[t >= DIST_T + DIST_DUR + 1.0])) if max(t) > DIST_T + DIST_DUR + 1.0 else float("nan")
    tips = {s: np.array(res["tip"][s]) for s in ("left", "right")}
    d_rest = {s: tips[s][t < DIST_T].mean(axis=0) for s in tips}
    d_peak = {s: float(np.max(np.linalg.norm(tips[s][win] - d_rest[s], axis=1))) for s in tips}
    recover = {s: float(np.mean(np.linalg.norm(tips[s][t >= DIST_T + DIST_DUR + 1.0] - d_rest[s], axis=1)) * 1000)
               for s in tips}
    print(f"夹持力: 静息 {p_rest:.2f} N → 扰动峰值 {p_peak:.2f} N → 释放后 1s 均值 {p_back:.2f} N")
    print(f"端面偏移峰值: left={d_peak['left']*1000:.2f} mm right={d_peak['right']*1000:.2f} mm "
          f"（双臂协同让位）| 释放后 1s 残留: left={recover['left']:.2f} mm right={recover['right']:.2f} mm")
    print(f"扰动: {DIST_SIDE}臂 沿夹持轴 {DIST_F:.0f}N ×{DIST_DUR:.1f}s @ t={DIST_T:.1f}s")


def plot_sweep(results, eta_list, ts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for res, eta, c in zip(results, eta_list, colors):
        t = np.array(res["t"])
        D = np.array(res["D"])
        P = np.array(res["P"])
        ramp = (t >= 0.45) & (t <= max(t) - 0.05)
        axs[0, 0].plot(D[ramp] * 1000, P[ramp], c, lw=1.6, label=f"eta_f={eta}")
        slope = float(np.polyfit(D[ramp], P[ramp], 1)[0])
        axs[0, 0].text(D[ramp].mean() * 1000, P[ramp].max() * 0.96, f"K={slope:.0f}", color=c, fontsize=8)
    axs[0, 0].axhline(F_REF, color="k", ls=":", lw=1.0)
    axs[0, 0].text(0.02, 0.99, f"F_ref={F_REF:.0f} N", transform=axs[0, 0].transAxes, fontsize=8, va="top")
    axs[0, 0].set_xlabel("squeeze cmd D [mm]")
    axs[0, 0].set_ylabel("grip force P [N]")
    axs[0, 0].set_title("(a) Impedance curves: P vs D (slope = effective stiffness)")
    axs[0, 0].legend(fontsize=8)

    eta = np.array(eta_list)
    comps, eins, keffs = [], [], []
    for res in results:
        t = np.array(res["t"])
        hold = t >= t_ramp1_for(res)
        tipL = np.array(res["tip"]["left"])
        tipR = np.array(res["tip"]["right"])
        comps.append(np.linalg.norm(tipL[hold].mean(axis=0) - tipR[hold].mean(axis=0)) * 1000)
        eins.append(float(np.mean(np.array(res["e_in"]["left"])[hold])) * 1000)
        ramp = (t >= 0.45) & (t <= max(t) - 0.05)
        keffs.append(float(np.polyfit(np.array(res["D"])[ramp], np.array(res["P"])[ramp], 1)[0]))
    axs[0, 1].plot(eta, keffs, "o-", color="tab:purple", lw=1.8)
    axs[0, 1].set_xlabel("eta_f")
    axs[0, 1].set_ylabel("Keff [N/m]")
    axs[0, 1].set_title("(b) Effective stiffness vs eta_f (softness knob)")
    axs[1, 0].plot(eta, comps, "s-", color="tab:brown", lw=1.8)
    axs[1, 0].set_xlabel("eta_f")
    axs[1, 0].set_ylabel("gap [mm]")
    axs[1, 0].set_title("(c) Steady gap vs eta_f (compliance grows)")
    axs[1, 1].plot(eta, eins, "^-", color="tab:cyan", lw=1.8)
    axs[1, 1].set_xlabel("eta_f")
    axs[1, 1].set_ylabel("give-way e_in [mm]")
    axs[1, 1].set_title("(d) Position give-way vs eta_f (Fig.7 style)")
    for ax in axs.flat:
        ax.grid(alpha=0.3)
    fig.suptitle("Impedance-mode curves (OpenArmX dual arm, ikforce OSC + Eq.(12) eta_f blend)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / f"impedance_curves_{ts}.svg", format="svg")
    fig.savefig(FIG_DIR / f"impedance_curves_{ts}.png", dpi=160)
    print(f"📈 {FIG_DIR / f'fig_imp_curves_{ts}.svg'} (+png)")


def plot_disturb(res, ts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array(res["t"])
    fig, axs = plt.subplots(2, 1, figsize=(9, 7))
    for ax in axs:
        ax.axvspan(DIST_T, DIST_T + DIST_DUR, color="0.9", alpha=0.8)
        ax.axvline(DIST_T, color="k", ls="--", lw=0.8)
        ax.axvline(DIST_T + DIST_DUR, color="k", ls="--", lw=0.8)
    axs[0].text(DIST_T + DIST_DUR / 2, 0.98, f"+{DIST_F:.0f} N ({DIST_SIDE})",
                transform=axs[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    axs[0].plot(t, np.array(res["P"]), "tab:red", lw=1.5, label="grip force P (spring)")
    axs[0].set_ylabel("N")
    axs[0].set_title("(a) Force response to arm push (impedance eta_f=0.8)")
    axs[0].legend(fontsize=8)
    for s, c in (("left", "tab:blue"), ("right", "tab:orange")):
        dev = np.array(res["tip"][s]) - np.array(res["tip"][s])[0]
        d = np.linalg.norm(dev, axis=1) * 1000
        axs[1].plot(t, d, c, lw=1.4, label=f"pad_{s} deviation from rest [mm]")
    axs[1].set_xlabel("t [s]")
    axs[1].set_ylabel("deviation [mm]")
    axs[1].set_title("(b) Both arms yield together and recover")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"impedance_dist_{ts}.svg", format="svg")
    fig.savefig(FIG_DIR / f"impedance_dist_{ts}.png", dpi=160)
    print(f"📈 {FIG_DIR / f'fig_imp_dist_{ts}.svg'} (+png)")


def export_sweep_csv(results, eta_list, ts):
    path = FIG_DIR / f"impedance_sweep_{ts}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["eta_f", "t", "delta_cmd", "P", "gap", "e_in_left", "e_in_right", "tipL", "tipR"])
        for res, eta in zip(results, eta_list):
            tipL = np.array(res["tip"]["left"])
            tipR = np.array(res["tip"]["right"])
            for k in range(0, len(res["t"]), 10):
                gap = np.linalg.norm(tipL[k] - tipR[k])
                w.writerow([eta, round(float(res["t"][k]), 4), res["D"][k], round(res["P"][k], 3),
                            round(gap, 5), round(float(res["e_in"]["left"][k] * 1000), 3),
                            round(float(res["e_in"]["right"][k] * 1000), 3),
                            np.round(tipL[k], 5).tolist(), np.round(tipR[k], 5).tolist()])
    print(f"📄 {path}")


def export_dist_csv(res, ts):
    path = FIG_DIR / f"impedance_dist_{ts}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "P", "dist_f", "e_in_left", "e_in_right"])
        for k in range(len(res["t"])):
            w.writerow([round(res["t"][k], 4), round(res["P"][k], 3), res["dist"][k],
                        round(float(res["e_in"]["left"][k] * 1000), 3),
                        round(float(res["e_in"]["right"][k] * 1000), 3)])
    print(f"📄 {path}")


# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="论文阻抗模式 + ikforce OSC：OpenArmX 双臂阻抗曲线。")
    parser.add_argument("--check", action="store_true", help="只构建加载")
    parser.add_argument("--viewer", action="store_true", help="交互 viewer（拖红球=改夹持中心）")
    parser.add_argument("--eta_sweep", type=str, default=",".join(map(str, ETA_F_SWEEP)),
                        help="ηf 扫描列表（逗号分隔）")
    parser.add_argument("--f_ref", type=float, default=F_REF, help="期望夹持力 (N)")
    parser.add_argument("--no_dist", action="store_true", help="跳过扰动会话")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--no_csv", action="store_true")
    args = parser.parse_args()

    FIG_DIR.mkdir(exist_ok=True)
    model, data = build_scene()
    arms = parse_arms(model, data)
    d0, u, tips0 = nominal_geometry(model, data, arms)
    for a in arms:
        a["q_init"] = np.array(data.qpos[a["qpos"]])     # q_rest 即初始/零空间目标
        a["tip0"] = tips0[a["side"]]
    target_mocap = model.body_mocapid[name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "imp_target")]
    tip_viz = {a["side"]: model.body_mocapid[name_to_id(model, mujoco.mjtObj.mjOBJ_BODY,
                                                        f"imp_tip_{a['side']}")] for a in arms}
    mid = 0.5 * (tips0["left"] + tips0["right"])
    data.mocap_pos[target_mocap][:] = mid

    print("✅ 阻抗控制场景已加载（虚拟负载弹簧，夹持轴 = 两尖端连线，q_rest 位形）")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")
    print(f"名义间距 D0={d0*1000:.1f} mm | 弹簧 K_PAIR={K_PAIR:.0f} N/m | "
          f"收缩 Δ_max={DELTA_MAX*1000:.0f} mm")
    if args.check:
        return

    if args.viewer:
        state = {"pause": False, "eta": 0.8, "f_ref": args.f_ref}

        def key_callback(keycode):
            try:
                key = chr(keycode).upper()
            except ValueError:
                return
            if key == "R":
                for a in arms:
                    data.qpos[a["qpos"]] = a["q_rest"].copy()
                    data.qvel[a["dofs"]] = 0.0
                data.qvel[:] = 0.0
                data.mocap_pos[target_mocap][:] = mid
                mujoco.mj_forward(model, data)
            elif key == "P":
                state["pause"] = not state["pause"]

        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            viewer.cam.lookat[:] = mid
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -14
            viewer.cam.azimuth = 180
            while viewer.is_running():
                off = data.mocap_pos[target_mocap].copy() - mid
                if not state["pause"]:
                    refs = refs_at_delta(arms, tips0, u, off, 0.0)
                    qf_t = np.zeros(model.nv)
                    p = _apply_pair_spring(model, data, arms, d0, u, K_PAIR, qf_t)
                    mujoco.mj_forward(model, data)
                    for a in arms:
                        tau, _, _ = solve_torque_impedance(model, data, a, refs[a["side"]], p, u,
                                                           state["eta"], state["f_ref"])
                        data.ctrl[a["act_ids"]] = tau
                    mujoco.mj_step(model, data)
                for a in arms:
                    data.mocap_pos[tip_viz[a["side"]]] = get_ee_tip(data, a["body_id"])
                viewer.sync()
        return

    eta_list = [float(x) for x in args.eta_sweep.split(",")]
    t0 = time.perf_counter()
    results = []
    for eta in eta_list:
        res = run_session(model, data, arms, d0, u, eta, args.f_ref, with_disturb=False)
        results.append(res)
        print(f"✅ ηf={eta} 会话完成 ({time.perf_counter()-t0:.1f}s, 饱和={res['sat']}拍)")
    report_sweep(results, eta_list, d0)

    res_dist = None
    if not args.no_dist:
        res_dist = run_session(model, data, arms, d0, u, 0.8, args.f_ref, with_disturb=True)
        print(f"✅ 扰动会话完成 (饱和={res_dist['sat']}拍)")
        report_disturb(res_dist)

    ts = time.strftime("%Y%m%d_%H%M%S")
    if not args.no_plot:
        plot_sweep(results, eta_list, ts)
        if res_dist is not None:
            plot_disturb(res_dist, ts)
    if not args.no_csv:
        export_sweep_csv(results, eta_list, ts)
        if res_dist is not None:
            export_dist_csv(res_dist, ts)
    print("完成。")


if __name__ == "__main__":
    main()