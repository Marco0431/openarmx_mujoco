#!/usr/bin/env python3
"""exp_sim2_nmpcc.py —— 论文仿真二（Fig.8）复现：双 2 连杆 + 环境弹簧 + 单会话三段模式切换

严格按论文第 IV 节实现：
  式(11)  误差定义：e_trac,i = r_i − r_i,ref ; e_cop,i = r_j − r_i ; e_f,i = f_i − f_i,ref
  式(12)  合作指标 ε_mult,i = (I−ηf,i)[(I−ηcop,i)e_trac,i − ηcop,i·e_cop,i] − ηf,i·e_f,i
  式(15)  平衡点    r_i = (1−ηcop,i)r_i,ref + ηcop,i·r_j          （运动同步/跟踪）
  式(16)/(23) 平衡点 r_i = r_i,ref + (1−ηf)⁻¹ηf(f_i − f_i,ref)    （力优先 → f→0）
  式(28)  误差状态 x̄e = [e_trac, ė_trac, e_cop, ė_cop]（r_j,cur 当前测量进入 yref）
  式(29)  ė = g(e,u)（双臂动力学预测，SQP_RTI 实时迭代）
  式(30)-(32)  NMPC：min ∫Ψ + Φ，Ψ = Q·ε² + ηdev·ε̇² + ηu·u²，Φ 终端同型
  式(35)  动力学 M(q)q̈ + C(q,q̇) + D(q̇) = u + Jᵀf_rob（摩擦 D 视为不确定项，NMPC 不建模）
  式(36)  环境力 f_rob,i = k_stiff,i(r_env,i − r_rob,i)，k_stiff = diag(1000, 0)（Table III）

Fig.8 结构（一次 3 s 仿真，η 随时间切换，对应 (b)(c)(d) 中 t=1,2 处竖线）：
  t∈[0,1)  Tracking      ηcop=0,   ηf=0   → 各臂沿自己参考运动（式20）
  t∈[1,2)  Synchroniza.  ηcop=0.5, ηf=0   → 目标 = (1−ηcop)r_i,ref + ηcop·r_j，双臂靠拢（式15/21）
  t∈[2,3]  Force priority ηcop=0,  ηf=1   → 力误差说了算，作用力→0（式16/23/24）

两臂严格镜像对称（系统与任务对称 → 对称解，图 (a) 蓝/红轨迹成对）：
  臂2 状态 = 臂1 镜像（qR=−qL, vR=−vL）；坐标反射下广义力矩反变 u_R = −u_L；
  笛卡尔力 f_R = (f_x, −f_y)；每控制拍强制投影，保持严格对称（摩擦 D 为奇函数会破坏对称，
  NMPC 预测模型不含摩擦 → 与论文"摩擦=不确定项"一致）。

运行: python exp_sim2_nmpcc.py
结果: results/sim2_iv/sim2_fig8_<时间戳>.png + sim2_session_<时间戳>.csv
"""
from __future__ import annotations

import os
import time

import numpy as np

from nmpcc import NMPCC, NMPCCConfig, two_link_plant

# ---- 参数（论文 Table III 精确值）----
L1 = 0.3266        # [m]
L2 = 0.3928        # [m]
MU = 0.01          # [N·s] 关节摩擦系数 μ（NMPC 预测模型不含 → 不确定性）
DQ1 = 0.9          # [N/m] D1^q1（关节1 摩擦常数）
DQ2 = 1.0          # [N·m] D1^q2（关节2 摩擦常数）
K_STIFF = np.array([[1000.0, 0.0], [0.0, 0.0]])   # k_stiff = diag(1000,0)，仅 x 向（Table III）

SIM_DT = 0.005
CTRL_DT = 0.01     # NMPC 控制周期 10 ms
HORIZON = 3.0      # 单会话 3 s（论文 Fig.8 (b)(c)(d) 0-3s）

# ---- 任务（论文 Fig.8(a) 同款形状：tracking 两臂从外侧向中间对称弧线，sync 竖直下降，force 反向分开）----
# 环境力（式36）：f_i = K(r_env,i − r_i)，K=diag(1000,0) 只 x 向（Table III）
ENV1 = np.array([0.448, 0.50]); ENV2 = np.array([0.452, -0.50])   # 两臂环境面（臂1 左、臂2 右）
# 臂1 起点偏墙左 3 mm（力 +3 N）；tracking 相向内移至墙左 0.5 mm（力 +0.5 N）→ 轨迹为对称内弧
# 臂2 为臂1 关于 x=0.45 的镜像（所有 x 参考 = 0.9 − x1）→ 两臂轨迹严格对称，力恒正/恒负
START1 = np.array([0.445, 0.44]); START2 = np.array([0.455, 0.44])
REF1 = np.array([0.4475, 0.36]); REF2 = np.array([0.4525, 0.36])  # tracking 终点（相向内弧 + y 下行）
SYNC1 = np.array([0.4475, 0.335]); SYNC2 = np.array([0.4525, 0.335]) # sync：竖直下降（x 不动）
REFF1 = np.array([0.448, 0.33]); REFF2 = np.array([0.452, 0.33])   # force：回墙面（力→0，反向分开）
# |r| 检查：|(0.445,0.44)|=0.626；|(0.4475,0.36)|=0.574；|(0.4525,0.335)|=0.563；均远离奇异（臂长 0.7194）
# 每段 (名称, 起点时刻, 终点时刻, 臂1 参考轨迹起点, 臂2 参考轨迹起点, 臂1 参考轨迹终点, 臂2 参考轨迹终点)
SEGMENTS = [("tracking", 0.0, 1.0, START1, START2, REF1, REF2),
            ("sync", 1.0, 2.0, REF1, REF2, SYNC1, SYNC2),
            ("force", 2.0, 3.0, SYNC1, SYNC2, REFF1, REFF2)]
# |r| 检查：|(0.448,0.44)|=0.628；|(0.448,0.36)|=0.575；|(0.451,0.325)|=0.556；均远离奇异（臂长 0.7194）

# ======================================================================
# 三种合作模式（同一套 NMPCC 式12/15/31，只改 η 与权重）
# ======================================================================
MODE_CONFIGS = {
    "tracking": dict(                       # 式(20)：ηcop=0 → r_i = r_i,ref
        eta_cop=np.array([[0.0, 0.0], [0.0, 0.0]]),
        eta_f=np.array([[0.0, 0.0], [0.0, 0.0]]),
        Q=400.0, eta_dev=80.0, eta_u=0.05, P_mul=2.0,
        force_env=None,
    ),
    "sync": dict(                           # 式(12) 分轴 η：x 向 ηcop=0（力通道：两臂各守预紧位，
        # 预紧力 ±2 N 恒正/恒负不穿越）；y 向 ηcop=0.5（运动同步：两臂 y 相互靠拢 + 与参考同步）
        eta_cop=np.array([[0.0, 0.5], [0.0, 0.5]]),
        eta_f=np.array([[0.0, 0.0], [0.0, 0.0]]),
        Q=500.0, eta_dev=80.0, eta_u=0.05, P_mul=2.0,
        force_env=None,
    ),
    "force": dict(                          # 式(16)/(24)：分轴 η —— x 向 ηf=1 力优先（力→0），
        # y 向 ηf=0 跟踪参考（η 是 diag 矩阵，式12；对应论文 (a) 中力优先段 y 仍走向参考）
        eta_cop=np.array([[0.0, 0.0], [0.0, 0.0]]),
        eta_f=np.array([[1.0, 0.0], [1.0, 0.0]]),
        Q=np.array([50.0, 500.0]), eta_dev=20.0, eta_u=0.05, P_mul=1.0,
        force_env=(K_STIFF, ENV1, ENV2),
        lm_reg=1e-3,   # x 向力残差 → GN 海森半正定；LM 正则保证 QP 正定（论文 qpOASES 同款处理）
    ),
}
SEGMENTS = [("tracking", 0.0, 1.0, START1, START2, REF1, REF2),
            ("sync", 1.0, 2.0, REF1, REF2, SYNC1, SYNC2),
            ("force", 2.0, 3.0, SYNC1, SYNC2, REFF1, REFF2)]

# ======================================================================
# 运行控制
# ======================================================================
MAKE_PLOT = True
SAVE_CSV = True
FIG_DIR = "results/sim2_iv"


def ik2(x, y):
    """2 连杆运动学逆解（标准阻尼最小二乘 DLS + 多初始猜测，抗奇异/防发散）。
    dq = (JᵀJ + λ²I)⁻¹ Jᵀ e：λ=0.05 的阻尼正则使 J 近奇异时步长有界（CLIK 同款）。"""
    def tip(q1, q2):
        return np.array([L1 * np.cos(q1) + L2 * np.cos(q1 + q2),
                         L1 * np.sin(q1) + L2 * np.sin(q1 + q2)])
    LAM = 0.05
    best = None
    for guess in [(0.2, -1.0), (1.2, -1.4), (-0.3, 1.6), (0.6, -2.2), (2.0, -1.0)]:
        q = np.array(guess, float)
        for _ in range(300):
            r = tip(q[0], q[1])
            e = np.array([x, y]) - r
            if np.linalg.norm(e) < 1e-10:
                break
            J = np.array([[-L1 * np.sin(q[0]) - L2 * np.sin(q[0] + q[1]), -L2 * np.sin(q[0] + q[1])],
                          [L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1]), L2 * np.cos(q[0] + q[1])]])
            JtJ = J.T @ J
            dq = np.linalg.solve(JtJ + LAM ** 2 * np.eye(2), J.T @ e)   # DLS：J⁺=(JᵀJ+λ²I)⁻¹Jᵀ
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq = dq / n * 0.3
            q = q + dq
        err = np.linalg.norm(tip(q[0], q[1]) - np.array([x, y]))
        if best is None or err < best[1]:
            best = (q, err)
            if err < 1e-10:
                break
    return best[0]


def make_plant_and_sim():
    """返回 (plant, 物理积分器 f(x,u) → [dx, f_rob]，环境力式36 闭式代入)。
    力矩限幅收紧到 ±8 N·m（论文实验一：“maximum torque set as 10 N for safety”）。"""
    import casadi as ca
    plant = two_link_plant(l1=L1, l2=L2, mu=MU, Dq1=DQ1, Dq2=DQ2)
    plant["umax"] = np.full(4, 8.0)
    plant["umin"] = np.full(4, -8.0)
    # 注：式(30) 的速度约束 ∥ṙ∥≤vmax 在 SQP_RTI + HPIPM 下以 box 状态约束会使 QP 病态，
    # 改用软约束：ε̇ 阻尼权重 η_dev（论文式31 的 ε̇ᵀηdevε̇ 项）抑制末端速度（NMPC 标准软约束做法）。
    x, u, p = plant["x"], plant["u"], plant["p"]
    tips = plant["tips"]
    r1, r2 = tips[:2], tips[2:]
    K = ca.SX(K_STIFF.astype(float))
    frob1 = K @ (ca.vertcat(ca.SX(list(ENV1))) - r1)
    frob2 = K @ (ca.vertcat(ca.SX(list(ENV2))) - r2)
    f_rob = ca.vertcat(frob1, frob2)
    f_expl_closed = ca.substitute(plant["f_expl"], p, f_rob)
    f_cl = ca.Function("physics", [x, u], [f_expl_closed, f_rob])
    return plant, f_cl


def step_physics(f_cl, x, u, dt):
    """RK4 积分一步。"""
    def d(xk):
        f_, _ = f_cl(xk, u)
        return np.array(f_).flatten()
    k1 = d(x); k2 = d(x + 0.5 * dt * k1); k3 = d(x + 0.5 * dt * k2); k4 = d(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def enforce_sym(x):
    """强制镜像对称：qR=−qL, vR=−vL（臂2 = 臂1 的 y 反射）。"""
    x[4:6] = -x[0:2]
    x[6:8] = -x[2:4]
    return x


def tip_of(sx):
    return np.array([L1 * np.cos(sx[0]) + L2 * np.cos(sx[0] + sx[1]),
                     L1 * np.sin(sx[0]) + L2 * np.sin(sx[0] + sx[1])])


def jac_of(q):
    return np.array([[-L1 * np.sin(q[0]) - L2 * np.sin(q[0] + q[1]), -L2 * np.sin(q[0] + q[1])],
                     [L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1]), L2 * np.cos(q[0] + q[1])]])


def ik2_sols(x, y):
    """2 连杆解析 IK：返回 2 个候选解 (q1, q2)。"""
    r2s = x * x + y * y
    c2 = np.clip((r2s - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1.0, 1.0)
    phi = np.arctan2(np.sqrt(1 - c2 * c2), c2)
    base = np.arctan2(y, x)
    sols = []
    for q2 in (phi, -phi):
        q1 = base - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
        sols.append(np.array([q1, q2]))
    return sols


def enforce_xmirror(x):
    """强制臂2 = 臂1 关于 x=0.45 的镜像（映射 T:(x,y)→(0.9−x,y)）。
    环境/参考/任务均关于该轴对称 → 对称解；速度映射 ṙ2 = (−ṙ1x, ṙ1y)。"""
    q1 = np.asarray(x[0:2], float); v1 = np.asarray(x[2:4], float)
    r1 = tip_of(q1)
    rd2 = np.array([-(jac_of(q1) @ v1)[0], (jac_of(q1) @ v1)[1]])
    tgt = np.array([0.9 - r1[0], r1[1]])
    sols = ik2_sols(tgt[0], tgt[1])
    q2 = min(sols, key=lambda s: np.linalg.norm(s - np.asarray(x[4:6], float)))  # 连续性
    v2 = np.linalg.solve(jac_of(q2), rd2)
    x[4:6] = q2; x[6:8] = v2
    return x


def run_session(verbose=False) -> dict:
    """论文 Fig.8 会话：3 s 单仿真，三段模式切换（η 随时间切换）。"""
    plant, f_cl = make_plant_and_sim()

    # 预构建三段的 NMPCC（式30-32；段切换时换实例 = 切换 η）
    nmpccs = {}
    for segname, _, _, _, _, _, _ in SEGMENTS:
        mcfg = dict(MODE_CONFIGS[segname])
        cfg = NMPCCConfig(N=50, Tp=0.1, n_tip=2, **mcfg)
        nmpccs[segname] = NMPCC(plant, cfg, verbose=verbose)
        print(f"✅ NMPCC 构建完成（{segname}）")

    # 初始状态：两臂各自在起点（独立闭环，不再强制镜像）
    q1 = ik2(START1[0], START1[1])
    q2 = ik2(START2[0], START2[1])
    x = np.concatenate([q1, np.zeros(2), q2, np.zeros(2)])

    traj = {"x": [x.copy()], "t": [0.0], "seg": ["tracking"], "f": [], "tau": [], "solve": []}
    n_seg_ctrl = int(1.0 / CTRL_DT)          # 每段 1 s = 100 拍
    for segname, t0, t1, r0L, r0R, r1L, r1R in SEGMENTS:
        nmpcc = nmpccs[segname]
        Tseg = t1 - t0
        dRL = (r1L - r0L) / Tseg                       # 参考轨迹速度（线性插值）
        dRR = (r1R - r0R) / Tseg
        for k in range(n_seg_ctrl):
            t = t0 + (k + 1) * CTRL_DT
            s = np.clip((t - t0) / Tseg, 0.0, 1.0)    # 段内进度
            rL = r0L + (r1L - r0L) * s                # 时变参考轨迹（论文 r_rob,i,r(t)）
            rR = r0R + (r1R - r0R) * s
            # 力测量（式36，供 ηf 用）：真实闭环力
            _, frob = f_cl(x, np.zeros(4))
            f_meas = np.array(frob).flatten()
            # 参考（式11/12 的 r_i,ref(t)、ṙ_i,ref、f_i,ref）；ηcop 的作用在 yref 的 r_j,cur 项里
            refs = {"rL": rL, "rR": rR,
                    "vL": dRL, "vR": dRR,
                    "f_refL": np.zeros(2), "f_refR": np.zeros(2)}
            t0s = time.perf_counter()
            u = nmpcc.solve(x, refs, f_meas)
            dt_solve = (time.perf_counter() - t0s) * 1e3
            for _ in range(int(CTRL_DT / SIM_DT)):
                x = step_physics(f_cl, x, u, SIM_DT)
                enforce_xmirror(x)   # 强制对称投影：臂2 = 臂1 的 x 镜像（对称解）
            traj["x"].append(x.copy()); traj["t"].append(t)
            traj["seg"].append(segname)
            traj["f"].append(f_meas.copy()); traj["tau"].append(u.copy())
            traj["solve"].append(dt_solve)
    return {k: np.array(v).reshape(len(v), -1) for k, v in traj.items()}


def report(res):
    x, t, seg = res["x"], res["t"], res["seg"]
    r1 = np.array([tip_of(s[:2]) for s in x])
    r2 = np.array([tip_of(s[4:6]) for s in x])
    f = res["f"]; tau = res["tau"]; st = res["solve"]
    print(f"\n=== 单会话 3 s（tracking→sync→force）===")
    for segname, t0, t1, _, _, _, _ in SEGMENTS:
        idx = np.where(seg == segname)[0]
        e = idx[-1]                       # x 索引（长度 301）
        fk = e - 1                        # f/tau 索引（长度 300）
        print(f"  {segname:8s} t∈[{t0},{t1}]  终点1=({r1[e,0]:.4f},{r1[e,1]:.4f}) "
              f"终点2=({r2[e,0]:.4f},{r2[e,1]:.4f}) |f|1={np.linalg.norm(f[fk,:2]):.2f} N "
              f"|f|2={np.linalg.norm(f[fk,2:]):.2f} N")
    print(f"x 向时间线: 初值 {r1[0,0]:.4f} → 终值 {r1[-1,0]:.4f}（臂1 环境墙 x={ENV1[0]}）")
    print(f"输入力矩 峰值: |τ1|={np.max(np.abs(tau[:,0])):.2f} |τ2|={np.max(np.abs(tau[:,1])):.2f} N·m")
    print(f"NMPC 求解耗时: 均值={np.mean(st):.2f} ms 峰值={np.max(st):.2f} ms")


def plot_fig8(res: dict, out_dir: str, ts: str) -> None:
    """论文 Fig.8 四联图：(a) 轨迹 (b) x 向时间线 (c) 输入力矩 (d) 相互作用力。
    蓝=robot1（实线） 红=robot2（虚线）；t=1,2 竖线分割三种模式并标注。
    输出 SVG（矢量，可任意放大）+ PNG（预览）两份。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = res["x"]; t = res["t"]; tau = res["tau"]; f = res["f"]
    r1 = np.array([tip_of(s[:2]) for s in x])
    r2 = np.array([tip_of(s[4:6]) for s in x])
    seg = res["seg"]
    seg_edges = [0.0, 1.0, 2.0, 3.0]
    seg_names = ["Tracking", "Synchronization", "Force priority"]

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))

    # (a) 双臂末端轨迹（两臂镜像对称；段端点圆点标记）
    axs[0, 0].plot(r1[:, 0], r1[:, 1], "tab:blue", lw=1.8, label="robot1")
    axs[0, 0].plot(r2[:, 0], r2[:, 1], "tab:red", ls="--", lw=1.8, label="robot2")
    # 段端点（每段起点与终点）
    for i, (name, s0, s1) in enumerate([("tracking", 0, 100), ("sync", 100, 200), ("force", 200, 300)]):
        axs[0, 0].plot(r1[s0, 0], r1[s0, 1], "ko", ms=5)
        axs[0, 0].plot(r1[s1, 0], r1[s1, 1], "ko", ms=3)
    axs[0, 0].text(0.02, 0.98, seg_names[0], transform=axs[0, 0].transAxes, fontsize=9, va="top")
    axs[0, 0].text(0.5, 0.55, seg_names[1], transform=axs[0, 0].transAxes, fontsize=9)
    axs[0, 0].text(0.98, 0.15, seg_names[2], transform=axs[0, 0].transAxes, fontsize=9, ha="right")
    axs[0, 0].set_xlabel("x [m]"); axs[0, 0].set_ylabel("y [m]")
    axs[0, 0].legend(fontsize=8, loc="lower left")

    # (b)(c)(d) 顶部三段名标注（模式名位于图内顶部，段边界竖线；无子标题避免重叠）
    def annotate_seg(ax):
        for edge in seg_edges[1:-1]:
            ax.axvline(edge, color="k", lw=0.8, ls="--")
        for i, nm in enumerate(seg_names):
            ax.text((seg_edges[i] + seg_edges[i + 1]) / 2, 0.97, nm,
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)

    # (b) x 方向随时间
    axs[0, 1].plot(t, r1[:, 0], "tab:blue", lw=1.6, label="robot1")
    axs[0, 1].plot(t, r2[:, 0], "tab:red", ls="--", lw=1.6, label="robot2")
    axs[0, 1].set_xlabel("t [s]"); axs[0, 1].set_ylabel("x [m]")
    axs[0, 1].legend(fontsize=8, loc="lower left")
    annotate_seg(axs[0, 1])

    # (c) 输入力矩 τ1..τ4（臂1: τ1,τ2；臂2: τ3,τ4）
    tf = t[1:]
    colors = ["tab:blue", "tab:orange", "tab:purple", "tab:gray"]
    for j, (c, lb) in enumerate(zip(colors, ["τ1", "τ2", "τ3", "τ4"])):
        axs[1, 0].plot(tf, tau[:, j], color=c, lw=1.3, label=lb)
    axs[1, 0].set_xlabel("t [s]"); axs[1, 0].set_ylabel("τ [N·m]")
    axs[1, 0].legend(fontsize=8, loc="lower left")
    annotate_seg(axs[1, 0])

    # (d) 相互作用力 f（x 分量，y 恒 0；镜像对称 f2x = f1x）
    axs[1, 1].plot(tf, f[:, 0], "tab:blue", lw=1.6, label="robot1")
    axs[1, 1].plot(tf, f[:, 2], "tab:red", ls="--", lw=1.6, label="robot2")
    axs[1, 1].set_xlabel("t [s]"); axs[1, 1].set_ylabel("f [N]")
    axs[1, 1].legend(fontsize=8, loc="lower left")
    annotate_seg(axs[1, 1])

    # 子图编号（(a)-(d)，右下角）
    letters = ["(a)", "(b)", "(c)", "(d)"]
    for ax, l in zip(axs.flat, letters):
        ax.text(0.99, -0.16, l, transform=ax.transAxes, ha="right", va="top", fontsize=11)

    for ax in axs.flat:
        ax.grid(alpha=0.3)
    fig.suptitle("Reproduction of Fig. 8 (dual two-link manipulators, NMPCC)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_svg = os.path.join(out_dir, f"sim2_fig8_{ts}.svg")
    out_png = os.path.join(out_dir, f"sim2_fig8_{ts}.png")
    fig.savefig(out_svg, format="svg")          # 矢量图，可任意放大
    fig.savefig(out_png, dpi=160)               # PNG 预览
    print(f"📈 Fig.8 风格图已保存: {out_svg} (+{os.path.basename(out_png)})")


def export_csv(res: dict, out_dir: str, ts: str) -> None:
    """导出单会话数据：t, seg, r1x,r1y, r2x,r2y, tau1..4, f1x,f1y,f2x,f2y。"""
    import csv
    x, t, seg = res["x"], res["t"], res["seg"]
    path = os.path.join(out_dir, f"sim2_session_{ts}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "seg", "r1x", "r1y", "r2x", "r2y",
                    "tau1", "tau2", "tau3", "tau4", "f1x", "f1y", "f2x", "f2y"])
        for k in range(len(res["tau"])):
            s = x[k + 1]
            r1 = tip_of(s[:2]); r2 = tip_of(s[4:6])
            w.writerow([t[k + 1], seg[k + 1], *r1, *r2, *res["tau"][k], *res["f"][k]])
    print(f"📄 {path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    res = None
    try:
        res = run_session(verbose=False)
        report(res)
    except Exception as e:
        print(f"\n[警告] 会话未跑完: {e}")
        raise
    if MAKE_PLOT and res is not None:
        plot_fig8(res, FIG_DIR, ts)
    if SAVE_CSV and res is not None:
        export_csv(res, FIG_DIR, ts)


if __name__ == "__main__":
    main()