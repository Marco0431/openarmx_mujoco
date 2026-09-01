#!/usr/bin/env python3
"""exp_imp_pd.py —— 多误差量 PD 阻抗控制（论文式(7) 状态反馈 + 式(11) 三类误差）+ x/y 偏移曲线

用户要求（要点明确）：只用 PD 阻抗控制 —— "很多个误差量那个"（论文式(11) 的 e_trac 跟踪误差、
e_cop 协同误差、e_f 力误差，两臂共 3×2 个误差量），最后输出参考图同款的双面板：
  (a) x 偏移 [m] vs t (b) y 偏移 [m] vs t，仿真 20 s。

控制律（每臂、每控制周期，论文式(7) 完整展开）：
  τ = u_comp + Jᵀ·[ K_trac·e_trac + D_trac·ė_trac            （式4：位置跟踪误差）
                   + K_cop·e_cop + D_cop·ė_cop                （式5：协同相对误差）
                   + K_force·e_f·û ]                          （式6：力误差，夹持轴 û）
  u_comp = qfrc_bias（C+G 前馈）；e_trac = r_i − r_ref,i；
  e_cop = (r_i − r_j) − (r_ref,i − r_ref,j)（双臂相对运动协调误差，运动同步通道）；
  e_f = P − F_ref（虚拟负载弹簧实测力与目标夹持力之差 → 阻抗/恒力通道）。
  三个误差量分别加权（K_trac/K_cop/K_force = 式(12) 中 η=1 各族全保留的 PD 版合作指标）。
  稳定性：任务力经 Λ=(J M⁻¹ Jᵀ)⁻¹ 惯性归一化（轻量臂无 Λ 的纯 JᵀK 会发散，见 exp_sync 排障），
  另加关节阻尼 K_DQ 与零空间姿势弹簧（同 ikforce/ikforce_impedance 已验证配方）。

场景与任务（ikforce_impedance 已验证位形：两臂 q_rest 对望、尖端相距 D0≈0.67 m、夹持轴 û=两尖端连线）：
  - x 偏移 = 实测间距 − D0（力轴偏移；初始 +2 mm → 力反馈把它收敛到 P=F_ref 的压缩量）
  - y 偏移 = 两臂相对 y 向同步误差（初始 +2 mm 相对偏移 → e_trac/e_cop 收敛）
  - 参考/目标全程静止；20 s。

运行（acados 环境）：
  D:\\acados_env\\python.exe -u exp_imp_pd.py --check
  D:\\acados_env\\python.exe -u exp_imp_pd.py          # 20s 会话 → CSV + 图(results/impedance/impedance_offset_<ts>)
  D:\\acados_env\\python.exe -u exp_imp_pd.py --duration 20 --e0 2 --y0 2
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

from ik import ARM_CONFIGS, BASE_DIR, clamp_qpos, generate_runtime_xml, get_ee_tip, name_to_id, solve_position_ik
from ikforce import (
    GRAVITY,
    IK_DT,
    K_DQ,
    K_PQ,
    MAX_NULL_TORQUE,
    MAX_TASK_FORCE,
    inject_motors,
    strip_collisions,
)
from ikforce_impedance import build_scene, parse_arms, nominal_geometry  # 复用已验证场景/位形

SCENE_ND = BASE_DIR / "ikforce_imp_scene.xml"
FIG_DIR = BASE_DIR / "results" / "impedance"

# ======================================================================
# 三族误差的 PD 阻抗增益（与 exp_sync PD、ikforce_impedance 同一量级）
# ======================================================================
K_TRAC = np.array([200.0, 200.0, 200.0])   # e_trac 位置跟踪刚度 (N/m)
D_TRAC = np.array([12.0, 12.0, 12.0])      # e_trac 速度阻尼 (N·s/m)
K_COP = np.array([3000.0, 3000.0, 3000.0])    # e_cop 协同刚度 (N/m)（运动同步通道，强同步）
D_COP = np.array([25.0, 25.0, 25.0])       # e_cop 协同阻尼 (N·s/m)
K_FORCE = 2.0                              # e_f 力误差增益（论文式6 K_force；F_ref 处精确恒力）
F_REF = 10.0                               # 目标夹持力 (N)：x 方向绝对参考 10 N（论文 ExpII 恒定夹持力）
K_PAIR = 2500.0                            # 虚拟负载弹簧 (N/m)（论文式27）
GAP0X = 2.0e-3                             # 初始 x 向同步位置误差 2 mm（间距 − D0 = +2 mm）
GAP0Y = 0.0e-0                             # 初始 y 向同步位置误差 0 mm
V_REF_Y = 0.010                            # y 方向绝对参考速度 −10 mm/s（两臂相同，Δ_ref=0）
T_Y_STOP = 4.0                             # y 参考运行时长 (s)：走 6 s（行程 0.06 m，远离限位）后保持


# ======================================================================
# 多误差量 PD 阻抗求解
# ======================================================================
def solve_pd_impedance(model, data, arm, ref, f_cop, p, u, f_ref):
    """返回 (tau, e_trac, e_cop, e_f)。τ = qfrc_bias + JᵀΛ[K_trac e_trac + D_trac ė_trac]
    + Jᵀ[K_cop e_cop + D_cop ė_cop] + Jᵀ[K_force e_f û]。"""
    dofs = arm["dofs"]
    tip = get_ee_tip(data, arm["body_id"])
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, tip, arm["body_id"])
    jac = jacp[:, dofs]

    m = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, m)
    mm = m[np.ix_(dofs, dofs)]
    m_inv = np.linalg.inv(mm)
    lam_inv = jac @ m_inv @ jac.T
    lam = np.linalg.inv(lam_inv) if np.linalg.det(lam_inv) > 1e-6 \
        else np.linalg.pinv(lam_inv, rcond=1e-6)

    # —— 误差族 1：跟踪误差 e_trac / ė_trac（Λ 惯性归一化）——
    e_trac = ref - tip
    ed_trac = jac @ data.qvel[dofs]
    f_trac = lam @ (K_TRAC * e_trac - D_TRAC * ed_trac)

    # —— 误差族 2：协同相对误差 e_cop（外部传入，含相对阻尼）——
    f_cop_a = np.zeros(3) if f_cop is None else f_cop

    # —— 误差族 3：力误差 e_f = P − F_ref（沿夹持轴 û；欠压→补夹 / 过压→松开）——
    out = u if arm["side"] == "left" else -u
    e_f = float(p - f_ref)               # >0 = 夹过头（需松），<0 = 夹不够（需压）
    f_fce = (K_FORCE * e_f) * out        # 沿 outward 修正：欠压(e_f<0)→ 沿 −out（向内压）

    force = f_trac + f_cop_a + f_fce
    f_norm = np.linalg.norm(force)
    if f_norm > MAX_TASK_FORCE and f_norm > 1e-9:
        force = force / f_norm * MAX_TASK_FORCE

    tau = (
        data.qfrc_bias[dofs].copy()
        + jac.T @ force
        - K_DQ * data.qvel[dofs]
    )
    # 零空间姿势弹簧（稳定冗余模态，不干扰末端任务）
    null_proj = np.eye(len(dofs)) - jac.T @ lam @ jac @ m_inv
    tau_posture = K_PQ * (arm["q_init"] - data.qpos[arm["qpos"]]) - K_DQ * data.qvel[dofs]
    tau_null = null_proj @ tau_posture
    n_null = np.linalg.norm(tau_null)
    if n_null > MAX_NULL_TORQUE and n_null > 1e-9:
        tau_null = tau_null / n_null * MAX_NULL_TORQUE
    tau = tau + tau_null
    e_cop = float(np.dot(f_cop_a, out)) / (K_COP[0] + 1e-9) if np.linalg.norm(K_COP) > 0 else 0.0
    return tau, e_trac, e_cop, e_f


# ======================================================================
# 20 s 会话：初始 x/y 偏移 → 多误差 PD 阻抗收敛
# ======================================================================
def run_offsets(model, data, arms, d0, u, duration, gap0x, gap0y, f_ref, v_ref_y):
    # 初始位形：双臂 q_rest；随后在"真实初始状态"里直接制造 x 向 2 mm 同步位置误差
    # （左臂沿 û 外移 gap0x/2、右臂沿 −û 外移 gap0x/2 → 间距 − D0 = gap0x），y 向无误差。
    y_hat = np.array([0.0, 1.0, 0.0])
    for a in arms:
        data.qpos[a["qpos"]] = a["q_rest"].copy()
        clamp_qpos(model, data, a["qpos"], a["joint_ids"])
        data.qvel[a["dofs"]] = 0.0
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    left = next(a for a in arms if a["side"] == "left")
    right = next(a for a in arms if a["side"] == "right")

    import ik as _ik
    dmp0 = _ik.DAMPING
    _ik.DAMPING = 0.05
    # —— 力平衡参考位形：每臂沿夹持轴向内的目标 = F_ref/(2·K_pair)（参考间距 = D0 − F_ref/K_pair）——
    d_ref = f_ref / (2.0 * K_PAIR)
    refs0 = {
        "left": left["tip0"] - d_ref * u,
        "right": right["tip0"] + d_ref * u,
    }
    gap_ref = float(np.dot(refs0["left"] - refs0["right"], u))   # = D0 − F_ref/K_pair
    # 初始状态 = 力平衡参考位形 + x 向 2 mm 同步位置误差（真实状态偏移，经位置 IK 落位）
    dsh = gap0x / 2.0
    tgt = {"left": refs0["left"] + dsh * u, "right": refs0["right"] - dsh * u}
    for _ in range(300):
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_forward(model, data)
        sols = []
        for a in arms:
            cur = get_ee_tip(data, a["body_id"])
            dq, _ = solve_position_ik(model, data, a["body_id"], tgt[a["side"]], cur,
                                      a["dofs"], a["qpos"], a["q_rest"])
            sols.append((a, dq))
        for a, dq in sols:
            data.qpos[a["qpos"]] += dq
            clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    _ik.DAMPING = dmp0
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    for a in arms:
        a["q_init"] = np.array(data.qpos[a["qpos"]])     # 零空间姿势目标 = 带 2mm 误差的初始位形

    # 绝对参考：r_ref,i(t) = tip0,i + (0, −v_ref_y·t, 0)，两臂完全相同（Δ_ref=0 → 纯同步）；
    # x 位置参考 = 标称（力轴交给力误差 e_f 的 10 N 恒力闭环）。
    def refs_at(t):
        dy = -v_ref_y * min(t, T_Y_STOP)
        return {
            "left": refs0["left"] + dy * y_hat,
            "right": refs0["right"] + dy * y_hat,
        }

    n = int(round(duration / IK_DT))
    log = {"t": [], "x_off": [], "y_off": [], "P": [], "e_trac": {}, "e_cop": {}, "e_f": [],
           "tau": [], "sat": 0, "tip": {}}
    for a in arms:
        log["e_trac"][a["side"]] = []
        log["tip"][a["side"]] = []
    sat = 0
    for k in range(n):
        t = k * IK_DT
        refs = refs_at(t)
        qf_t = np.zeros(model.nv)
        # 虚拟负载弹簧（论文式27；反力经 mj_applyFT 累加进 qf_t）
        tips = {a["side"]: get_ee_tip(data, a["body_id"]) for a in arms}
        gap = float(np.linalg.norm(tips["left"] - tips["right"]))
        p = K_PAIR * (d0 - gap)
        mujoco.mj_applyFT(model, data, p * u, np.zeros(3), tips["left"], left["body_id"], qf_t)
        mujoco.mj_applyFT(model, data, -p * u, np.zeros(3), tips["right"], right["body_id"], qf_t)
        mujoco.mj_forward(model, data)

        # 协同误差项（误差族 2）：相对位姿偏差（y/z 为主同步通道；x 交由力误差通道）
        e_cop_vec = {}
        for a in arms:
            other = right if a["side"] == "left" else left
            e_cop_vec[a["side"]] = (
                (tips[a["side"]] - tips[other["side"]])
                - (refs[a["side"]] - refs[other["side"]])
            )
        v_cop = {}
        for a in arms:
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp, jacr, tips[a["side"]], a["body_id"])
            v_cop[a["side"]] = jacp[:, a["dofs"]] @ data.qvel[a["dofs"]]
        f_cop = {}
        for a in arms:
            other = right if a["side"] == "left" else left
            f_cop[a["side"]] = -K_COP * e_cop_vec[a["side"]] - D_COP * (v_cop[a["side"]] - v_cop[other["side"]])

        taus = []
        for a in arms:
            tau, e_trac, e_cop, e_f = solve_pd_impedance(model, data, a, refs[a["side"]],
                                                         f_cop[a["side"]], p, u, f_ref)
            taus.append((a, tau, e_trac, e_cop, e_f))
        for a, tau, _, _, _ in taus:
            data.ctrl[a["act_ids"]] = tau
        mujoco.mj_step(model, data)

        for a, tau, e_trac, e_cop, e_f in taus:
            low, high = model.actuator_ctrlrange[a["act_ids"]].T
            if np.any(tau < low - 1e-9) or np.any(tau > high + 1e-9):
                sat += 1

        tips = {a["side"]: get_ee_tip(data, a["body_id"]) for a in arms}
        x_off = (np.linalg.norm(tips["left"] - tips["right"]) - abs(gap_ref))   # 两臂运动误差 x（相减）→0
        y_off = (tips["left"] - tips["right"])[1]                               # 两臂运动误差 y（相减）→0
        log["t"].append(t)
        log["x_off"].append(x_off)
        log["y_off"].append(y_off)
        log["P"].append(p)
        for a, tau, e_trac, e_cop, e_f in taus:
            log["e_trac"][a["side"]].append(np.linalg.norm(e_trac))
            log["tip"][a["side"]].append(tips[a["side"]])
            if a["side"] == "left":
                log["e_f"].append(e_f)
        log["sat"] = sat
    for a in arms:
        log["e_trac"][a["side"]] = np.array(log["e_trac"][a["side"]])
        log["tip"][a["side"]] = np.array(log["tip"][a["side"]])
    log["x_off"] = np.array(log["x_off"])
    log["y_off"] = np.array(log["y_off"])
    log["P"] = np.array(log["P"])
    return log


# ======================================================================
# 出图：参考图同款双面板 (a) x 偏移 [m] (b) y 偏移 [m]，20 s
# ======================================================================
def plot_offsets(log, ts, nmpcc_csv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array(log["t"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    if nmpcc_csv is not None:
        import csv as _csv
        from pathlib import Path as _P
        if _P(nmpcc_csv).exists():
            rows = list(_csv.DictReader(open(nmpcc_csv, encoding="utf-8")))
            tn = np.array([float(r["t"]) for r in rows])
            exn = np.array([float(r["esync_x"]) for r in rows])
            eyn = np.array([float(r["esync_y"]) for r in rows])
            axs[0].plot(tn, exn, "tab:blue", lw=1.2, label="NMPCC")
            axs[1].plot(tn, eyn, "tab:blue", lw=1.2, label="NMPCC")
            axs[0].text(0.98, 0.02, "NMPCC (0-8s)", transform=axs[0].transAxes,
                        ha="right", fontsize=7, color="tab:blue")
    axs[0].plot(t, log["x_off"], "tab:red", lw=1.0, label="PD impedance (Eq.7)")
    axs[0].set_xlabel("t [s]")
    axs[0].set_ylabel("e_x [m]")
    axs[0].set_title("(a) x motion error = (r_L - r_R) - (ref_L - ref_R)")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.3)
    axs[1].plot(t, log["y_off"], "tab:red", lw=1.0, label="PD impedance (Eq.7)")
    axs[1].set_xlabel("t [s]")
    axs[1].set_ylabel("e_y [m]")
    axs[1].set_title("(b) y motion error = (r_L - r_R) - (ref_L - ref_R)")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"impedance_offset_{ts}.svg", format="svg")
    fig.savefig(FIG_DIR / f"impedance_offset_{ts}.png", dpi=160)
    print(f"📈 {FIG_DIR / f'fig_imp_offset_{ts}.svg'} (+png)")


def export_csv(log, ts):
    path = FIG_DIR / f"impedance_offset_{ts}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "x_offset", "y_offset", "P", "e_trac_L", "e_trac_R"])
        for k in range(len(log["t"])):
            w.writerow([round(log["t"][k], 4), round(float(log["x_off"][k]), 7),
                        round(float(log["y_off"][k]), 7), round(float(log["P"][k]), 3),
                        round(float(log["e_trac"]["left"][k]), 6),
                        round(float(log["e_trac"]["right"][k]), 6)])
    print(f"📄 {path}")


# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="多误差量 PD 阻抗控制：x/y 偏移曲线。")
    parser.add_argument("--check", action="store_true", help="只构建加载")
    parser.add_argument("--duration", type=float, default=20.0, help="会话时长 (s)")
    parser.add_argument("--gap0x", type=float, default=GAP0X, help="初始 x 偏移 (m)")
    parser.add_argument("--gap0y", type=float, default=GAP0Y, help="初始 y 相对偏移 (m)")
    parser.add_argument("--v_ref_y", type=float, default=V_REF_Y, help="y 向绝对参考速度 (m/s, 默认 0.010=−10mm/s)")
    parser.add_argument("--f_ref", type=float, default=F_REF, help="x 向绝对参考力/目标夹持力 (N)")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--no_csv", action="store_true")
    parser.add_argument("--overlay_nmpcc", default=None,
                        help="NMPCC 结果 CSV 路径（叠画蓝色对比，如 results/exp1_nmpcc_*.csv）")
    args = parser.parse_args()

    FIG_DIR.mkdir(exist_ok=True)
    model, data = build_scene()
    arms = parse_arms(model, data)
    d0, u, tips0 = nominal_geometry(model, data, arms)
    for a in arms:
        a["q_init"] = np.array(data.qpos[a["qpos"]])
        a["tip0"] = tips0[a["side"]]
    print("✅ 多误差量 PD 阻抗场景已加载（q_rest 对望位形，夹持轴 = 两尖端连线）")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")
    print(f"名义间距 D0={d0*1000:.1f} mm | 弹簧 K_PAIR={K_PAIR:.0f} N/m | "
          f"初始同步误差 x={args.gap0x*1000:.1f} mm y={args.gap0y*1000:.1f} mm | "
          f"绝对参考: y {args.v_ref_y*1000:.0f} mm/s, x {args.f_ref:.0f} N | 时长 {args.duration:.0f} s")
    if args.check:
        return

    t0 = time.perf_counter()
    log = run_offsets(model, data, arms, d0, u, args.duration, args.gap0x, args.gap0y,
                      args.f_ref, args.v_ref_y)
    print(f"✅ 会话完成 {time.perf_counter()-t0:.1f} s (饱和={log['sat']}拍)")
    t = np.array(log["t"])
    hold = t >= 5.0
    print(f"--- 多误差量 PD 阻抗 报表 ({args.duration:.0f} s) ---")
    print(f"x 偏移: 初始 {log['x_off'][0]*1000:+.2f} mm → 末端 {log['x_off'][-1]*1000:+.2f} mm "
          f"| 后段 RMS {np.sqrt((log['x_off'][hold]**2).mean())*1000:.2f} mm")
    print(f"y 偏移: 初始 {log['y_off'][0]*1000:+.2f} mm → 末端 {log['y_off'][-1]*1000:+.2f} mm "
          f"| 后段 RMS {np.sqrt((log['y_off'][hold]**2).mean())*1000:.2f} mm")
    print(f"夹持力 P: 后段均值 {np.mean(log['P'][hold]):.2f} N（目标 {args.f_ref:.1f} N）")
    ts = time.strftime("%Y%m%d_%H%M%S")
    if not args.no_plot:
        plot_offsets(log, ts, args.overlay_nmpcc)
    if not args.no_csv:
        export_csv(log, ts)
    print("完成。")


if __name__ == "__main__":
    main()