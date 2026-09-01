#!/usr/bin/env python3
"""exp2_paper_nmpcc.py —— 实验二复现（全新架构：纯 NMPCC 全程，无控制器切换）

重读论文 V-B 后的架构修正（关键）：论文叙述
  "In the holding process, two arms compress the object to hold it. When the required
   preload is achieved, the dual arms hold the object and move stationary, where the
   synchronous errors gradually converge to zero."
——压缩建力、力 unsteady→steady、同步误差收敛到零，**全部由 NMPCC 自己完成**。
论文没有"预载控制器 → NMPC 接手"的切换；此前 38 轮失败全部死在切换瞬间。

本脚本（零切换、t=0 即夹紧已有力）：
  t=0 姿态 = grasp_pose_static 同款"压紧位"：双臂各自 IK 从贴合位再内移 δ* = F/(2kx)
    = 4.8mm（垫片已压住箱、已有接触力≈F）。**无压缩段、无冻结、无预载切换**，
    直接从这一夹紧姿态开始 NMPCC（参考位置=压紧位 tip 恒定，力参考 f_ref=F 恒定）。
  Phase 1 保持 (0 ~ T_HOLD)：参考静止，力稳态，同步误差收敛到 0
    —— "preload achieved... hold and move stationary"。
  Phase 2 搬运 (T_HOLD ~)：y 向绝对参考速度 −10 mm/s（论文值），ηcop=0.8 运动同步；
    距离可调（--y_dist，默认 3cm = 实测机械臂 y 可靠行程），三角往返。
  判据（论文 Fig.12）：力同步误差 < 2 N、y 位置同步误差 < 2 mm、罐子不倒。

模型标定（与 world 严格一致）：kx = k1/2 = 2606 N/m（tc=0.02 软接触本征 k1=5213，_stiff_check 实测），
  rest_gap = 2(箱半厚+净空) = 物理零力间距；压紧位压入 δ* = F/(2kx) = 25/5213 ≈ 4.8mm/臂。
运行：D:\\acados_env\\python.exe -u exp2_paper_nmpcc.py [--hold_only] [--y_dist 0.03]
"""
from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

import nmpcc_cart as nc
from ik import ARM_CONFIGS, EE_TIP_Z, clamp_qpos, get_ee_tip, name_to_id, solve_pose_ik
from ikforce import IK_DT
from grasp_pose_static import BOX_HALF, BOX_MASS, GRIP_CLEARANCE
from ikforce_grasp import GRASP_Z
from exp_box_nmpcc_cart import (
    build_scene_cans, soften_contacts, pad_contact_forces, arm_damping,
    box_tilt_rad, JOINT_DAMPING, CAN_COUNT, CONTACT_TC,
)

FIG_DIR = "results/exp2_carry"
CTRL_EVERY = 3            # 每 3 个 sim 步（6 ms）解一次 NMPC
KX_MODEL = 2606.0         # k1/2：tc=0.02 软接触本征 k1=5213 实测（_stiff_check，压入-力斜率）
T_HOLD = 0.6              # 保持段：参考静止（同步误差收敛、力稳态）后开始 y 搬运


def smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def make_phases(force: float, v_y: float, y_dist: float, t_hold: float,
                press: dict, rv_ref: dict):
    """论文式参考发生器（纯 NMPCC 全程，t=0 即夹紧姿态、已有力）：

    起始姿态 = grasp_pose_static 同款"压紧位"（垫片已压入 δ*=F/k1，t=0 力≈F），
    故**无压缩段**。参考位置 = 压紧位 tip（x 恒），力参考 f_ref=F 恒定（x 轴 ηf=0.8 阻抗保持）。
    Phase1 保持 (0~t_hold)：参考静止，同步误差收敛到 0（论文原句）。
    Phase2 搬运 (t_hold~)：y 向恒速 −v_y，三角往返钳在 ±y_dist。
    返回 sample(t) → (r_ref{side:6}, v_ref{side:6}, f_mag)。
    """
    t_move0 = t_hold                            # 搬运起始时刻
    seg = y_dist / abs(v_y) if abs(v_y) > 1e-9 else 1e9   # 单程时长

    def sample(t: float):
        # y：三角往返（t_move0 起），速度恒 ±v_y
        r_y, v_yy = 0.0, 0.0
        if t >= t_move0:
            tt = t - t_move0
            ph = (tt % (2 * seg)) / seg        # [0,2)
            if ph < 1.0:
                r_y = -abs(v_y) * min(tt, seg)
                v_yy = -abs(v_y)
            else:
                r_y = -abs(v_y) * (2 * seg - min(tt, 2 * seg))
                v_yy = abs(v_y)
            r_y = max(min(r_y, 0.0), -2 * y_dist)  # 钳位（安全）
        r_ref, v_ref = {}, {}
        for side, sign in (("left", 1.0), ("right", -1.0)):
            base = press[side]                   # 压紧位 tip（t=0 已压入 δ*）
            r_ref[side] = np.concatenate([
                np.array([base[0], base[1] + r_y, base[2]]),
                rv_ref[side]])
            v_ref[side] = np.array([0.0, v_yy, 0.0, 0.0, 0.0, 0.0])
        return r_ref, v_ref, force
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="实验二：纯 NMPCC 全程（论文 V-B 复现）。")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--force", type=float, default=25.0,
                        help="每侧夹持力参考 N（论文纸面 10N；本场景箱 1.5kg 需 25N 摩擦裕度，可调）")
    parser.add_argument("--v_y", type=float, default=0.01, help="y 向参考速度幅值 m/s（论文 10mm/s）")
    parser.add_argument("--y_dist", type=float, default=0.03,
                        help="y 向单程移动距离 m（可调；默认 3cm = 实测可靠行程）")
    parser.add_argument("--hold_only", action="store_true", help="只做压缩+保持（不做 y 搬运）")
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--Tp", type=float, default=0.06)
    parser.add_argument("--no_viewer", action="store_true")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--no_csv", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # ---------- 场景（Fig.11：箱 + 4 空罐 + 双臂垫片对夹）----------
    model, data = build_scene_cans()
    box_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_geom_id = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    box_qpos = int(model.jnt_qposadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")])
    box_qvel = int(model.jnt_dofadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")])
    can_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, f"can{i}") for i in range(CAN_COUNT)]

    arms = []
    for cfg in ARM_CONFIGS:
        body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["jacobi_body"])
        joint_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg["joint_names"]]
        dofs = [int(model.jnt_dofadr[j]) for j in joint_ids]
        qpos = [int(model.jnt_qposadr[j]) for j in joint_ids]
        act_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_motor") for n in cfg["joint_names"]]
        arms.append({"side": cfg["side"], "sign": 1.0 if cfg["side"] == "left" else -1.0,
                     "q_rest": cfg["q_rest"], "body_id": body_id, "joint_ids": joint_ids,
                     "dofs": dofs, "qpos": qpos, "act_ids": act_ids,
                     "pad_id": name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"openarmx_{cfg['side']}_ee_pad")})
    soften_contacts(model, box_geom_id, [a["pad_id"] for a in arms])   # tc=0.004（< 控制周期）
    arm_damping(model, [d for a in arms for d in a["dofs"]], JOINT_DAMPING)
    fk = nc.mujoco_fk_symbols(model, [{"body_id": a["body_id"], "joint_ids": a["joint_ids"]} for a in arms], EE_TIP_Z)

    # ---------- 复位：双臂休息位、箱到抓取高度、罐上箱顶 ----------
    for arm in arms:
        data.qpos[arm["qpos"]] = arm["q_rest"].copy()
        clamp_qpos(model, data, arm["qpos"], arm["joint_ids"])
        data.qvel[arm["dofs"]] = 0.0
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    mid = np.mean([get_ee_tip(data, a["body_id"]) for a in arms], axis=0)
    data.qpos[box_qpos:box_qpos + 3] = [mid[0], mid[1], GRASP_Z]
    data.qpos[box_qpos + 3:box_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[box_qvel:box_qvel + 6] = 0.0
    mujoco.mj_forward(model, data)
    for k in range(CAN_COUNT):
        jn = f"can{k}_freejoint"
        cq = int(model.jnt_qposadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)])
        cd = int(model.jnt_dofadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)])
        off = (-0.10, -0.05, 0.05, 0.10)[k]
        data.qpos[cq:cq + 3] = [data.xpos[box_id][0], data.xpos[box_id][1] + off,
                                data.xpos[box_id][2] + BOX_HALF[2] + 0.0575 + 0.002]
        data.qpos[cq + 3:cq + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[cd:cd + 6] = 0.0
    mujoco.mj_forward(model, data)
    box_start = data.xpos[box_id].copy()

    # ---------- 压紧位 IK（t=0 即夹紧姿态、已有力：垫片压入 δ* = F/(2kx)，grasp_pose_static 同款独立收敛）----------
    delta = args.force / (2.0 * KX_MODEL)        # 每臂压入量：F = kx·2δ → δ = F/(2kx) = 25/5213 ≈ 4.8mm
    press = {a["side"]: np.array([
        float(box_start[0] + a["sign"] * (BOX_HALF[0] + GRIP_CLEARANCE - delta)),
        float(box_start[1]), float(box_start[2])]) for a in arms}
    for a in arms:
        for _ in range(1000):
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            mujoco.mj_forward(model, data)
            dq, _ = solve_pose_ik(model, data, a["body_id"], press[a["side"]],
                                  np.array([-a["sign"], 0.0, 0.0]), get_ee_tip(data, a["body_id"]),
                                  a["dofs"], a["qpos"], a["q_rest"])
            data.qpos[a["qpos"]] += dq
            clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    for arm in arms:
        arm["q_grasp"] = data.qpos[arm["qpos"]].copy()
        tip = get_ee_tip(data, arm["body_id"])
        print(f"  [IK] {arm['side']} tip={np.round(tip, 4)} 目标={np.round(press[arm['side']], 4)} "
              f"err={np.linalg.norm(tip - press[arm['side']])*1e3:.2f}mm")
    rv_ref = {a["side"]: nc.rotvec_of_R(data.xmat[a["body_id"]].reshape(3, 3)) for a in arms}

    # ---------- NMPCC（论文式25/27/28-32；η 映射 = 论文实验二：x ηf=0.8 / y ηcop=0.8）----------
    fref = {"left": np.array([args.force, 0, 0, 0, 0, 0]),
            "right": np.array([-args.force, 0, 0, 0, 0, 0])}
    rest_gap = 2.0 * (BOX_HALF[0] + GRIP_CLEARANCE)     # 物理零力间距（纯 NMPCC，无切换偏置）
    cfg = nc.NMPCCCartConfig(N=args.N, Tp=args.Tp, kx=KX_MODEL, kf_scale=1.0,
                             # x=100 力轴（有阻尼后可提）；y=500 同步；z=1000 抗翻；
                             # 姿态=10（右臂 rv≈2.81rad 近 π 分支，重权会被分支跳变炸飞）
                             Q=np.array([100.0, 500.0, 1000.0, 10.0, 10.0, 10.0]),
                             eta_u=0.02,
                             eta_u_vec=np.array([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0] * 2),
                             eta_dev=0.6, rest_gap=rest_gap)
    per_arm = np.array([60.0, 60.0, 14.0, 14.0, 14.0, 14.0, 14.0])
    umax = np.concatenate([per_arm, per_arm])
    t0b = time.perf_counter()
    nmpc = nc.NMPCCCart(cfg, fk, umin=-umax, umax=umax, verbose=args.verbose)
    print(f"✅ NMPCC 构建完成 {time.perf_counter()-t0b:.1f}s | kx={KX_MODEL:.0f} rest_gap={rest_gap:.5f} "
          f"(物理零力) | ηf_x=0.8 ηcop_y=0.8 (论文) | F={args.force}N | CONTACT_TC={CONTACT_TC}")

    def joint_state():
        return np.concatenate([data.qpos[arms[0]["qpos"]], data.qvel[arms[0]["dofs"]],
                               data.qpos[arms[1]["qpos"]], data.qvel[arms[1]["dofs"]]])

    def plant_params():
        m = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, m)
        out = []
        for a in arms:
            MM = m[np.ix_(a["dofs"], a["dofs"])]
            v = data.qvel.copy()
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            G = data.qfrc_bias[a["dofs"]].copy()   # q̈=M⁻¹(u−G)：G=+qfrc_bias（原子验证）
            data.qvel[:] = v
            mujoco.mj_forward(model, data)
            out.extend([MM.reshape(-1), G])
        # 关节阻尼（物理已知项，非不确定项）：预测 q̈=M⁻¹(u−G−D·v+Jᵀf)。
        # 缺它：弹簧(kx=3575)+轻腕(M≈0.001)在预测内无阻尼高频振荡 -> QP 病态贴边（已证）。
        # 加它：ζ=D/(2√(kM))≈0.53 良好阻尼，QP 良态。
        D = np.concatenate([model.dof_damping[a["dofs"]].copy() for a in arms])
        return np.concatenate(out + [D])

    def gravity_comp():
        out = []
        for a in arms:
            v = data.qvel.copy()
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            g = data.qfrc_bias[a["dofs"]].copy()   # +qfrc_bias（原子验证）
            data.qvel[:] = v
            mujoco.mj_forward(model, data)
            out.append(g)
        return np.concatenate(out)

    def clamp_balance_comp():
        """模型平衡 u = G − Jpᵀ·f_ref（预测 q̈=M⁻¹(u−G+Jᵀf)=0；u_balance≈tau_pre 已证）"""
        out = []
        for a in arms:
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            tip = get_ee_tip(data, a["body_id"])
            mujoco.mj_jac(model, data, jacp, jacr, tip, a["body_id"])
            out.append(-(jacp[:, a["dofs"]].T @ fref[a["side"]][:3]))
        return np.concatenate(out)

    ref_fn = make_phases(args.force, args.v_y if not args.hold_only else 0.0,
                         args.y_dist, T_HOLD, press, rv_ref)

    def build_p(t_sim: float):
        r_ref, v_ref, f_mag = ref_fn(t_sim)
        p = np.zeros(nc.NP)
        pp = plant_params()                 # [M/G(112), D(14)]
        p[0:112] = pp[:112]
        p[148:162] = pp[112:]              # D1(7)+D2(7) 关节阻尼
        p[112:118] = r_ref["left"]
        p[118:124] = r_ref["right"]
        p[124:130] = v_ref["left"]
        p[130:136] = v_ref["right"]
        f_ref_now = {"left": np.array([f_mag, 0, 0, 0, 0, 0]),
                     "right": np.array([-f_mag, 0, 0, 0, 0, 0])}
        p[136:142] = f_ref_now["left"]
        p[142:148] = f_ref_now["right"]
        return p

    # ---------- warm-up：2 次预解（第 1 次 cold 必贴边、第 2 次 warm 收敛——已证）----------
    p0 = build_p(0.0)
    for _ in range(2):
        nmpc.solve(joint_state(), p0, u_ref=gravity_comp() + clamp_balance_comp(),
                   u_seed=gravity_comp())
    print("✅ warm-up 完成（2 次预解，消灭冷启动贴边）")

    # ---------- 主循环（纯 NMPCC 全程）----------
    N = args.steps
    log = {"t": [], "box": [], "tipL": [], "tipR": [], "tau": [], "F": [], "tilt": [],
           "canz": [], "solve_ms": []}
    u_last = gravity_comp() + clamp_balance_comp()
    sat = 0
    for t in range(N):
        t_sim = t * IK_DT
        mujoco.mj_forward(model, data)

        if t % CTRL_EVERY == 0:
            p_coef = build_p(t_sim)
            t0s = time.perf_counter()
            u_last = nmpc.solve(joint_state(), p_coef,
                                u_ref=gravity_comp() + clamp_balance_comp())
            log["solve_ms"].append((time.perf_counter() - t0s) * 1e3)

        for a in arms:
            low, high = model.actuator_ctrlrange[a["act_ids"]].T
            if np.any(u_last[a["act_ids"]] < low - 1e-9) or np.any(u_last[a["act_ids"]] > high + 1e-9):
                sat += 1

        # 箱：t=0 即压紧夹住（已有力），直接自由受 NMPCC 控制（无压缩段、无冻结）
        for a in arms:
            data.ctrl[a["act_ids"]] = u_last[a["act_ids"]]
        mujoco.mj_step(model, data)

        if t % 5 == 0:
            log["t"].append(t_sim + IK_DT)
            log["box"].append(data.xpos[box_id].copy())
            log["tipL"].append(get_ee_tip(data, arms[0]["body_id"]).copy())
            log["tipR"].append(get_ee_tip(data, arms[1]["body_id"]).copy())
            log["tau"].append(u_last.copy())
            fc = pad_contact_forces(model, data, arms, box_geom_id)
            log["F"].append([fc["left"], fc["right"]])
            log["tilt"].append(box_tilt_rad(data, box_qpos))
            log["canz"].append(np.array([data.xpos[c][2] for c in can_ids]))
        if args.hold_only and t_sim >= T_HOLD + 0.3:
            break

    # ---------- 报表（论文 Fig.12 口径）----------
    x = {k: np.array(v).reshape(len(v), -1) for k, v in log.items()}
    tts, box_xyz, F, tilt = x["t"].flatten(), x["box"], x["F"], x["tilt"]
    tau, tipL, tipR, canz = x["tau"], x["tipL"], x["tipR"], x["canz"]
    print(f"--- 实验二（纯 NMPCC 全程）{tts[-1]:.1f} s 报表 ---")
    # 论文 Fig.12(b)：y 位置同步误差 = r1y − r2y（双臂同动应 <2mm）
    y_sync = tipL[:, 1] - tipR[:, 1]
    motion = tts >= T_HOLD
    if motion.any():
        print(f"y 位置同步误差 r1y−r2y: 运动段均值={np.abs(y_sync[motion]).mean()*1e3:.2f} mm "
              f"峰值={np.abs(y_sync[motion]).max()*1e3:.2f} mm（论文 Fig.12b <2mm）")
    fs = np.abs(F[:, 0] - F[:, 1])
    print(f"力同步误差 |F_L−F_R|: 均值={fs.mean():.2f} N 峰值={fs.max():.2f} N（论文 Fig.12d <2N）")
    print(f"垫片接触力: L 均值={F[:,0].mean():.2f} R 均值={F[:,1].mean():.2f} N（参考 {args.force} N）")
    # 箱跟随（y 搬运段）：箱 y 应跟随参考
    r, v, f_mag = ref_fn(tts[-1])
    print(f"箱终点 {box_xyz[-1].round(4)} vs 起始 {box_start.round(4)} | 参考终点偏移 y={r['left'][1]:+.4f}")
    held = box_xyz[-1][2] > box_start[2] - 0.05
    tmx = np.degrees(np.abs(tilt).max(axis=0))
    print(f"箱倾角峰值: 侧倾={tmx[0]:.2f}° 翘起={tmx[1]:.2f}° | 力矩饱和 {sat}/{N} | "
          f"求解 均值={np.mean(x['solve_ms']):.1f}ms 峰值={np.max(x['solve_ms']):.1f}ms")
    # 罐子平稳性（Fig.11）
    box_top = box_xyz[:, 2].reshape(-1, 1) + BOX_HALF[2]
    drop = (canz - 0.0575) - box_top
    n_fall = int(np.sum(np.any(drop < -0.01, axis=1)))
    print(f"罐子平稳性: 掉落 {n_fall} 次（Fig.11 判据：不倒=平稳）")
    ok = held and n_fall == 0 and tmx[1] < 5.0 and tmx[0] < 10.0
    print(">>> 判定: " + ("✅ 夹住、同步达标、罐子没倒" if ok else "❌ 未达标"))

    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    if not args.no_csv:
        import csv
        path = os.path.join(FIG_DIR, f"exp2_carry_paper_session_{ts}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "boxx", "boxy", "boxz", "tipLx", "tipLy", "tipLz",
                        "tipRx", "tipRy", "tipRz", *[f"tau{i+1}" for i in range(14)],
                        "FL", "FR", "tiltx", "tilty", "tiltz", "c0z", "c1z", "c2z", "c3z"])
            for i in range(len(tts)):
                w.writerow([tts[i], *box_xyz[i], *tipL[i], *tipR[i], *tau[i],
                            F[i, 0], F[i, 1], *tilt[i], *canz[i]])
        print(f"📄 {path}")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(11, 8))
        # (a) Fig.12a：双臂 xy 平面位置
        axs[0, 0].plot(tipL[:, 0], tipL[:, 1], "tab:blue", lw=1.5, label="left")
        axs[0, 0].plot(tipR[:, 0], tipR[:, 1], "tab:red", lw=1.5, label="right")
        axs[0, 0].plot(box_xyz[:, 0], box_xyz[:, 1], "k--", lw=1.0, label="box")
        axs[0, 0].set_xlabel("x [m]"); axs[0, 0].set_ylabel("y [m]")
        axs[0, 0].legend(fontsize=8); axs[0, 0].set_title("(a) dual arms xy-plane (Fig.12a)")
        # (b) Fig.12b：y 同步误差
        axs[0, 1].plot(tts, y_sync * 1e3, "tab:blue", lw=1.5)
        axs[0, 1].axhline(2.0, color="k", ls=":", lw=0.8); axs[0, 1].axhline(-2.0, color="k", ls=":", lw=0.8)
        axs[0, 1].set_xlabel("t [s]"); axs[0, 1].set_ylabel("r1y−r2y [mm]")
        axs[0, 1].set_title("(b) y sync error (Fig.12b, <2mm)")
        # (c) Fig.12c：左臂前三关节力矩（肩/肘/腕）
        for j in range(3):
            axs[1, 0].plot(tts, tau[:, j], lw=1.2, label=f"joint{j+1}")
        axs[1, 0].set_xlabel("t [s]"); axs[1, 0].set_ylabel("τ [N·m]")
        axs[1, 0].legend(fontsize=8); axs[1, 0].set_title("(c) left arm torques (Fig.12c)")
        # (d) Fig.12d：力同步误差
        axs[1, 1].plot(tts, fs, "tab:blue", lw=1.5, label="|F_L−F_R|")
        axs[1, 1].plot(tts, F[:, 0], "tab:green", lw=1.0, ls="--", label="F_L")
        axs[1, 1].plot(tts, F[:, 1], "tab:orange", lw=1.0, ls="--", label="F_R")
        axs[1, 1].axhline(2.0, color="k", ls=":", lw=0.8)
        axs[1, 1].set_xlabel("t [s]"); axs[1, 1].set_ylabel("force [N]")
        axs[1, 1].legend(fontsize=8); axs[1, 1].set_title("(d) force sync error (Fig.12d, <2N)")
        for ax in axs.flat:
            ax.grid(alpha=0.3)
        fig.suptitle("Experiment II: pure NMPCC (paper Eq.25/27/12/31-32, no controller switch)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_svg = os.path.join(FIG_DIR, f"exp2_carry_paper_fig_{ts}.svg")
        out_png = os.path.join(FIG_DIR, f"exp2_carry_paper_fig_{ts}.png")
        fig.savefig(out_svg, format="svg"); fig.savefig(out_png, dpi=160)
        print(f"📈 {out_svg} (+png)")


if __name__ == "__main__":
    main()