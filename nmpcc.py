#!/usr/bin/env python3
"""nmpcc.py —— 论文 NMPCC (式30–32) 的 acados 实现（一个可复用模块）

结构（对应论文）：
  式(12)   ε_mult,i = (I−ηf)[(I−ηcop)e_trac − ηcop·e_cop] − ηf·e_f   （按轴 diag 的 ηcop/ηf）
  式(28)   误差状态 x̄e = [e_trac, ė_trac, e_cop, ė_cop]
  式(29)   ė = g(e, u)
  式(30)   min_u ∫ Ψ + Φ，受动力学/速度/力矩约束 (Tp=100ms, N=50)
  式(31)   Ψ = Q·ε_mult² + ηdev·ε̇_mult² + ηu·u²
  式(32)   Φ = 终端同型 (Q→P)

acados 实现要点：
  - 代价用 NONLINEAR_LS（acados 原生非线性残差平方和）：
      y_expr = [ ε_mult_L(3), ε_mult_R(3), ε̇_mult_L(3), ε̇_mult_R(3), u(14) ]
      W      = blkdiag(Q, Q, ηdev, ηdev, ηu)；参考轨迹走 OCP 参数 p（每拍更新），yref ≡ 0
  - ε_mult 完整表达式放残差（r_i、r_j 均状态函数 → 预测中协同误差 e_cop 演化，= 论文式28/29）；
    参考拆解：η 为按轴权重
  - 力误差 e_f = f_meas − f_ref：f_meas(接触力测量)是 ocp 参数 p，每周期更新
  - 求解器 SQP_RTI = 论文/ACADO 的 real-time iteration（每周期一次高斯牛顿+QP）
  - Plant 接口：提供 casadi 前向动力学 f_expl(x,u,p) 与端面 FK tips(q) 即插即用

内置 plant：
  - two_link_plant():         论文仿真二双连杆 (式35) + 弹簧环境力走参数(式36)
  - mujoco_dual_arm_plant():  openarmx 双臂 plant（MuJoCo 数据自动生成 casadi 动力学
                              + 复刻 ik.get_ee_tip 端面 FK），供实验一/二闭环

运行：
  from nmpcc import NMPCC, NMPCCConfig, two_link_plant
  plant = two_link_plant()
  cfg = NMPCCConfig(N=50, Tp=0.1, eta_cop=0.5*np.ones(2), ...)
  nmpcc = NMPCC(plant, cfg)
  u = nmpcc.solve(x0, refs, f_meas)
"""

from __future__ import annotations

import numpy as np
import casadi as ca

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpOptions, AcadosOcpSolver
    _HAS_ACADOS = True
except ImportError:
    _HAS_ACADOS = False


# ======================================================================
# Plant 接口（返回 dict）：
#   x        casadi SX 状态 [q_i..., v_i...]（双臂拼接）
#   u        casadi SX 输入
#   p        casadi SX 参数（接触力测量）
#   f_expl   casadi 前向动力学
#   tips     casadi 端面位置 [r_L; r_R]（每臂 nt 维）
#   umin/umax 力矩限幅
# ======================================================================
def _rot_axis(axis, q):
    """Rodrigues 旋转矩阵（绕任意单位轴）。"""
    k = axis / ca.norm_2(axis)
    K = ca.vertcat(
        ca.horzcat(0.0, -k[2], k[1]),
        ca.horzcat(k[2], 0.0, -k[0]),
        ca.horzcat(-k[1], k[0], 0.0))
    return ca.SX.eye(3) + ca.sin(q) * K + (1 - ca.cos(q)) * (K @ K)


def two_link_plant(l1=0.3266, l2=0.3928,
                   b1=1.937, b2=1.437, b3=0.873, b4=0.718, b5=0.873, c1=-0.718,
                   Dq1=0.9, Dq2=1.0, mu=0.01, umax=20.0):
    """论文仿真二：平面双 2 连杆（式35），参数直接用论文 Table III 数值。

    Table III: b1=1.937, b2=1.437, b3=0.873, b4=0.718, b5=0.873, c1=-0.718
               [kg·m²/rad]；l1=0.3266, l2=0.3928 [m]；
               D1^q1=0.9 [N/m], D1^q2=1 [N·m]（两关节不同！）；μ=0.01 [N·s]。
    摩擦项（按论文原式）：D(q̇) = [D1^q1 + μ·q̇1; D1^q2 + μ·q̇2]。
    力的去向：plant 动力学用参数 p 接收 f_rob（=k_stiff(r_env−r)由外环计算），
    NMPC 每周期把同样的测量力作为参数。
    """
    def arm_fd(q, v, u, f_rob):
        q1, q2 = q[0], q[1]
        M = ca.vertcat(ca.horzcat(b1 + b2 * ca.cos(q2), b3 + b4 * ca.cos(q2)),
                       ca.horzcat(b3 + b4 * ca.cos(q2), b5))
        C = c1 * ca.sin(q2) * ca.vertcat(
            ca.horzcat(v[1], v[0] + v[1]),
            ca.horzcat(-v[0], 0.0))                      # C(q,q̇)（需再乘 q̇ → 下）
        Cq = C @ v
        # 式(35) 摩擦：D(q̇) 列向量，线性形式 [D1^q1 + μ·q̇1; D1^q2 + μ·q̇2]（两关节常数不同）
        D = ca.vertcat(Dq1 + mu * v[0], Dq2 + mu * v[1])
        J = ca.vertcat(ca.horzcat(-l1 * ca.sin(q1) - l2 * ca.sin(q1 + q2), -l2 * ca.sin(q1 + q2)),
                       ca.horzcat(l1 * ca.cos(q1) + l2 * ca.cos(q1 + q2), l2 * ca.cos(q1 + q2)))
        qdd = ca.solve(M, u + J.T @ f_rob - Cq - D)
        return qdd

    def tip(q):
        q1, q2 = q[0], q[1]
        return ca.vertcat(l1 * ca.cos(q1) + l2 * ca.cos(q1 + q2),
                          l1 * ca.sin(q1) + l2 * ca.sin(q1 + q2))

    qL = ca.SX.sym("qL", 2); vL = ca.SX.sym("vL", 2)
    qR = ca.SX.sym("qR", 2); vR = ca.SX.sym("vR", 2)
    uL = ca.SX.sym("uL", 2); uR = ca.SX.sym("uR", 2)
    f1 = ca.SX.sym("f1", 2); f2 = ca.SX.sym("f2", 2)
    p = ca.vertcat(f1, f2)
    ddL = arm_fd(qL, vL, uL, f1)
    ddR = arm_fd(qR, vR, uR, f2)
    x = ca.vertcat(qL, vL, qR, vR)
    u = ca.vertcat(uL, uR)
    return {"x": x, "u": u, "p": p,
            "f_expl": ca.vertcat(vL, ddL, vR, ddR),
            "tips": ca.vertcat(tip(qL), tip(qR)),
            "umin": -umax * np.ones(4), "umax": umax * np.ones(4),
            "n_tip": 2, "name": "two_link_dual"}


def mujoco_dual_arm_plant(model, arm_cfgs, ee_tip_z: float):
    """由 MuJoCo 模型自动生成 openarmx 双臂 casadi 动力学 + 端面 FK。

    动能法符号构造：
      T = Σ ½[m‖ṗ_cm‖² + ωᵀ(R R_ic I R_icᵀ Rᵀ)ω]，M=∂²T/∂q̇²，C 用 Christoffel，G=∂V/∂q
      q̈ = M⁻¹(τ − C q̇ − G)，预测模型不含关节摩擦（论文同款不确定性假设）
    端面 FK 与 ik.get_ee_tip 一致：link7 体架 +Z 外推 EE_TIP_Z。
    """
    import mujoco

    def quat2mat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    def collect_chain(body_id):
        chain_ids = []
        b = body_id
        while b != 0 and len(chain_ids) < 12:
            chain_ids.append(b)
            b = model.body_parentid[b]
        chain_ids.reverse()
        out = []
        for bid in chain_ids:
            jt = model.body_jntadr[bid]
            has_joint = jt >= 0
            axis = model.jnt_axis[jt].copy() if has_joint else np.zeros(3)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            out.append({
                "pos": model.body_pos[bid].copy(),
                "quat": model.body_quat[bid].copy(),
                "axis": axis,
                "mass": float(model.body_mass[bid]),
                "ipos": model.body_ipos[bid].copy(),
                "iquat": model.body_iquat[bid].copy(),
                "inertia": model.body_inertia[bid].copy(),
                "jointed": has_joint,
            })
        return out

    def body_stack(chain, q):
        H_prev = ca.SX.eye(4)
        st = []
        qi = 0
        for d in chain:
            Rb = quat2mat(d["quat"])
            if d["jointed"]:
                Rj = _rot_axis(ca.SX(d["axis"]), q[qi])
                qi += 1
            else:
                Rj = ca.SX.eye(3)      # 静止根/支架：固定变换，无自由度
            H = H_prev @ ca.vertcat(
                ca.horzcat(Rb @ Rj, ca.reshape(ca.SX(d["pos"]), 3, 1)),
                ca.horzcat(ca.SX.zeros(1, 3), ca.SX.ones(1, 1)))
            H_prev = H
            Rc = quat2mat(d["iquat"])
            pc = (H @ ca.vertcat(ca.reshape(ca.SX(d["ipos"]), 3, 1), ca.SX.ones(1, 1)))[:3]
            st.append({"H": H, "R": H[:3, :3], "pc": pc, "m": d["mass"],
                       "I_body": ca.SX(np.diag(d["inertia"]).astype(float)),
                       "R_ic": ca.SX(Rc)})
        return st

    def arm_fd(q, v, u, chain):
        st = body_stack(chain, q)
        T_n = 0.0
        V = 0.0
        for s in st:
            pc = s["pc"]
            pd = ca.jacobian(pc, q) @ v
            R = s["R"]
            Rd = ca.reshape(ca.jacobian(R, q) @ v, 3, 3)
            S = Rd @ R.T
            omega = ca.vertcat(S[2, 1], S[0, 2], S[1, 0])
            I_base = R @ s["R_ic"] @ s["I_body"] @ s["R_ic"].T @ R.T
            T_n += 0.5 * (s["m"] * ca.dot(pd, pd) + ca.dot(omega, I_base @ omega))
            V += s["m"] * 9.81 * pc[2]
        M = ca.hessian(T_n, v)[0]
        nq = q.numel()
        Cq = ca.SX.zeros(nq)
        for i in range(nq):
            for j in range(nq):
                cij = 0.0
                for k in range(nq):
                    cij += 0.5 * (ca.jacobian(M[i, j], q)[k] + ca.jacobian(M[i, k], q)[j]
                                  - ca.jacobian(M[j, k], q)[i]) * v[k]
                Cq[i] += cij
        G = ca.jacobian(V, q).T
        return ca.solve(M, u - Cq - G)

    def tip_of(chain, q):
        H = body_stack(chain, q)[-1]["H"]
        return (H @ ca.vertcat(ca.SX(0.0), ca.SX(0.0), ca.SX(ee_tip_z), ca.SX(1.0)))[:3]

    chainL = collect_chain(arm_cfgs[0]["body_id"])
    chainR = collect_chain(arm_cfgs[1]["body_id"])
    nq = sum(1 for d in chainL if d["jointed"])
    assert nq == sum(1 for d in chainR if d["jointed"])
    qL = ca.SX.sym("qL", nq); vL = ca.SX.sym("vL", nq)
    qR = ca.SX.sym("qR", nq); vR = ca.SX.sym("vR", nq)
    uL = ca.SX.sym("uL", nq); uR = ca.SX.sym("uR", nq)
    fL = ca.SX.sym("fL", 3); fR = ca.SX.sym("fR", 3)
    p = ca.vertcat(fL, fR)
    ddL = arm_fd(qL, vL, uL, chainL)
    ddR = arm_fd(qR, vR, uR, chainR)
    x = ca.vertcat(qL, vL, qR, vR)
    u = ca.vertcat(uL, uR)
    cr = model.actuator_ctrlrange
    umin = np.array([float(cr[i][0]) for i in range(nq)] + [float(cr[nq + i][0]) for i in range(nq)])
    umax = np.array([float(cr[i][1]) for i in range(nq)] + [float(cr[nq + i][1]) for i in range(nq)])
    rL_ = tip_of(chainL, qL)
    rR_ = tip_of(chainR, qR)
    return {"x": x, "u": u, "p": p,
            "f_expl": ca.vertcat(vL, ddL, vR, ddR),
            "tips": ca.vertcat(rL_, rR_),
            "umin": umin, "umax": umax, "n_tip": 3, "name": "openarmx_dual"}


# ======================================================================
class NMPCCConfig:
    def __init__(self, N=50, Tp=0.1,
                 eta_cop=np.array([0.8, 0.8, 0.8]), eta_f=np.zeros(3),
                 Q=1.0, eta_dev=1e-3, eta_u=1e-3, P_mul=1.0, n_tip=3,
                 with_epsdot=True, force_env=None, Q_force=None,
                 reg_method="MIRROR", lm_reg=0.0, qp_solver="PARTIAL_CONDENSING_HPIPM"):
        """force_env=(k_stiff, env_L(nt), env_R(nt)) 时，力误差用论文式(36)
        f = k_stiff·(r_env − r(q)) 作为状态的函数（力优先模式）——否则用测量力 p。
        Q_force：力误差块的权重；ηf 混合时有效权重 = (1−ηf)²·Q + ηf²·Q_force
        （力余量量纲是 N，取 Q_force=1/k² 可使 GN-Hessian 与位置块同量级）。
        eta_cop/eta_f：传 (nt,) 两臂共用；传 (2, nt) 可每臂独立（论文式12 的 ηcop,1/ηcop,2）。
        reg_method：acados QP Hessian 正则化（'NO_REGULARIZE'/'MIRROR'/'CONVEXIFY'/...）。
        lm_reg：>0 时给 QP 加 Levenberg–Marquardt 正则 λI（半正定 GN 海森时的兜底）。
        qp_solver：QP 求解器（PARTIAL_CONDENSING_HPIPM 默认 / FULL_CONDENSING_HPIPM）。"""
        self.N = N
        self.Tp = Tp
        self.eta_cop = np.asarray(eta_cop, float).copy()   # (nt,) 按轴
        self.eta_f = np.asarray(eta_f, float).copy()
        self.Q = Q
        self.Q_force = Q if Q_force is None else Q_force
        self.eta_dev = eta_dev
        self.eta_u = eta_u
        self.P_mul = P_mul
        self.n_tip = n_tip
        self.with_epsdot = with_epsdot
        self.force_env = force_env
        self.reg_method = reg_method
        self.lm_reg = lm_reg
        self.qp_solver = qp_solver


class NMPCC:
    """论文 NMPCC（acados, SQP_RTI）。

    solve(x0, refs, f_meas) → u(14)
      x0     [q_L, v_L, q_R, v_R] 当前状态
      refs   dict: rL/rR（当前参考端面）、vL/vR（参考速度）、f_refL/f_refR（参考力）
      f_meas [f_L(3), f_R(3)] 接触力测量（ηf≠0 时用；无接触传 0）
    """

    def __init__(self, plant: dict, cfg: NMPCCConfig, verbose: bool = False):
        if not _HAS_ACADOS:
            raise RuntimeError("acados_template 不可用")
        self.plant = plant
        self.cfg = cfg
        nt = cfg.n_tip
        self.nt = nt
        x, u, p = plant["x"], plant["u"], plant["p"]
        nx, nu = x.shape[0], u.shape[0]
        self.nq = nx // 4
        self.nu = nu

        # ---- 合作指标 η：支持每臂独立（论文式(12) 的 ηcop,1/ηcop,2, ηf,1/ηf,2）----
        # 传 (2, nt) → 第 1 行=臂1、第 2 行=臂2；传 (nt,) → 两臂共用（兼容旧写法）
        def _norm2(eta, name):
            eta = np.asarray(eta, float)
            if eta.shape == (nt,):
                return eta.copy(), eta.copy()
            if eta.shape == (2, nt):
                return eta[0].copy(), eta[1].copy()
            raise ValueError(f"{name} 应为 (nt,) 或 (2, nt)，收到 {eta.shape}")
        self.Ic1, self.If1 = None, None
        self.Ic1, self.Ic2 = _norm2(cfg.eta_cop, "eta_cop")
        self.If1, self.If2 = _norm2(cfg.eta_f, "eta_f")

        qL = x[:self.nq]; vL = x[self.nq:2 * self.nq]
        qR = x[2 * self.nq:3 * self.nq]; vR = x[3 * self.nq:]
        tips = plant["tips"]
        rL, rR = tips[:nt], tips[nt:]
        # ---- 扩展 OCP 参数：p = [f_meas(2nt), rLref(nt), rRref(nt), ṙLref(nt), ṙRref(nt)] ----
        # 参考轨迹作为外生输入（论文式(29) 误差动态中的 r_i,ref(t) 同款）；yref ≡ 0
        p_new = ca.vertcat(p, ca.SX.sym("rlref", nt), ca.SX.sym("rrref", nt),
                           ca.SX.sym("vlref", nt), ca.SX.sym("vrref", nt))
        fLm, fRm = p_new[:nt], p_new[nt:2 * nt]
        rLref = p_new[2 * nt:3 * nt]; rRref = p_new[3 * nt:4 * nt]
        vLref = p_new[4 * nt:5 * nt]; vRref = p_new[5 * nt:6 * nt]
        # 力模型（ηf≠0 时的状态相关项，防止 GN 雅可比为零）：
        #   force_env=("pair", k_stiff)      → 论文式(27)：双臂间虚拟弹簧 f_1 = K(r_2−r_1), f_2 = −f_1
        #   force_env=(k_stiff, env_L, env_R) → 论文式(36)：各自环境力 f_i = K(r_env,i − r_i)
        # k_stiff 可为标量（各向同性）或 (nt,nt) 矩阵（论文 Table III: diag(1000, 0)，仅 x 向）
        if cfg.force_env is not None:
            fe = cfg.force_env
            if isinstance(fe[0], str) and fe[0] == "pair":   # 式(27)：双臂间（夹持物形变）力
                k_stiff = fe[1]
                if np.asarray(k_stiff).ndim == 2:
                    K = ca.SX(np.asarray(k_stiff, float)[:nt, :nt])
                else:
                    K = ca.SX(float(k_stiff)) * ca.SX.eye(nt)
                fLm = K @ (rR - rL)
                fRm = -fLm
            else:                                     # 式(36)：各自环境力
                k_stiff, env_L, env_R = fe
                envLs = ca.vertcat(*[ca.SX(float(v)) for v in np.asarray(env_L)[:nt]])
                envRs = ca.vertcat(*[ca.SX(float(v)) for v in np.asarray(env_R)[:nt]])
                if np.asarray(k_stiff).ndim == 2:
                    K = ca.SX(np.asarray(k_stiff, float)[:nt, :nt])
                else:
                    K = ca.SX(float(k_stiff)) * ca.SX.eye(nt)
                fLm = K @ (envLs - rL)
                fRm = K @ (envRs - rR)

        # 式(11)/(12)：e_trac,i = r_i − r_i,ref ; e_cop,i = r_j − r_i ; e_f,i = f_i − f_i,ref（f_ref ≡ 0）
        #   E_mult,i = (1−If,i)[(1−Ic,i)e_trac,i − Ic,i·e_cop,i] − If,i·e_f,i
        #            = (1−If,i)[r_i − (1−Ic,i)r_i,ref − Ic,i·r_j] − If,i·f_i   （展开等价形式）
        # 实现：完整 E 直接放残差表达式 —— r_i 与 r_j 都是状态函数，GN 雅可比含 r_j → 预测中协同误差
        # e_cop 随状态演化（= 论文式(28)(29) 的误差状态动态）；r_i,ref / ṙ_i,ref 走 OCP 参数 p（每拍更新），yref ≡ 0。
        # ε̇ 同理（力导 ḟ_i 视为不确定项不建模）；平衡点 = 论文式(15)/(16)
        def eps_expr(r_i, r_j, fmeas, r_ref, Ic, If):
            return (1 - If) * (r_i - (1 - Ic) * r_ref - Ic * r_j) - If * fmeas

        def epsd_expr(rd_i, rd_j, v_ref, Ic, If):
            return (1 - If) * (rd_i - (1 - Ic) * v_ref - Ic * rd_j)

        r_dot = ca.jacobian(ca.vertcat(rL, rR), ca.vertcat(qL, qR)) @ ca.vertcat(vL, vR)
        rdL, rdR = r_dot[:nt], r_dot[nt:]

        epsL = eps_expr(rL, rR, fLm, rLref, self.Ic1, self.If1)
        epsR = eps_expr(rR, rL, fRm, rRref, self.Ic2, self.If2)
        edL = epsd_expr(rdL, rdR, vLref, self.Ic1, self.If1)
        edR = epsd_expr(rdR, rdL, vRref, self.Ic2, self.If2)
        y_expr = ca.vertcat(epsL, epsR, u) if not cfg.with_epsdot \
            else ca.vertcat(epsL, epsR, edL, edR, u)
        y_expr_e = ca.vertcat(epsL, epsR) if not cfg.with_epsdot \
            else ca.vertcat(epsL, epsR, edL, edR)
        ny = (2 * nt if not cfg.with_epsdot else 4 * nt) + nu
        ne = 2 * nt if not cfg.with_epsdot else 4 * nt
        W = np.zeros((ny, ny))
        # 残差 y_expr−yref 已是 ε_mult 本体（式12）→ W 直接是论文式(31)的 Q / ηdev / ηu。
        # Q 可传标量或 (nt,) 按轴权重（论文 Q 矩阵按轴 diag）
        Qw = np.asarray(cfg.Q, float)
        if Qw.ndim == 0:
            Qw = Qw * np.ones(nt)
        W[:nt, :nt] = np.diag(Qw)
        W[nt:2 * nt, nt:2 * nt] = np.diag(Qw)
        if cfg.with_epsdot:
            W[2 * nt:3 * nt, 2 * nt:3 * nt] = cfg.eta_dev * np.eye(nt)
            W[3 * nt:4 * nt, 3 * nt:4 * nt] = cfg.eta_dev * np.eye(nt)
            W[4 * nt:, 4 * nt:] = cfg.eta_u * np.eye(nu)
        else:
            W[2 * nt:, 2 * nt:] = cfg.eta_u * np.eye(nu)
        self.W = W
        self.ny = ny

        # 式(28) 误差状态的"当前测量"：由 x0 算 [rL, rR, rdL, rdR]（solve 时更新 yref 用）
        self._fk_cur = ca.Function("fk_cur", [qL, qR, vL, vR],
                                   [rL, rR, r_dot[:nt], r_dot[nt:]])
        self._nq = self.nq

        model = AcadosModel()
        model.name = f"nmpcc_{plant.get('name', 'p')}"
        model.x = x
        model.u = u
        model.p = p_new
        model.f_expl_expr = ca.substitute(plant["f_expl"], p, p_new[:2 * nt])   # 剪切成新的 p 定义
        model.cost_y_expr = y_expr
        model.cost_y_expr_e = y_expr_e

        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = cfg.N
        ocp.solver_options.tf = cfg.Tp

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = W
        ocp.cost.W_e = W[:ne, :ne].copy()
        ocp.cost.W_e[:2 * nt if not cfg.with_epsdot else 4 * nt] = \
            cfg.P_mul * ocp.cost.W_e[:2 * nt if not cfg.with_epsdot else 4 * nt]
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ne)

        ocp.constraints.idxbu = np.arange(nu)
        ocp.constraints.lbu = plant["umin"]
        ocp.constraints.ubu = plant["umax"]
        # 官方姿势：初态用 constraints.x0 + solve_for_x0（本分支手动 set lbx/ubx 会坏 QP）
        ocp.constraints.x0 = np.zeros(nx)
        # 状态约束（式30：∥ṙ∥ ≤ vmax ↔ 关节速度限幅 vq；q 范围可选）：
        # x = [qL, vL, qR, vR]
        if "v_range" in plant:
            qr = plant.get("q_range")
            nq_ = self.nq
            vq = np.asarray(plant["v_range"], float).repeat(2)     # [vq_L(2), vq_R(2)]
            if qr is None:
                q_lo, q_hi = np.full(nq_, -5.0), np.full(nq_, 5.0)
            else:
                q_lo, q_hi = np.asarray(qr[0], float), np.asarray(qr[1], float)
            ocp.constraints.idxbx = np.arange(nx)
            ocp.constraints.lbx = np.concatenate([q_lo, -vq[:nq_], q_lo, -vq[nq_:]])
            ocp.constraints.ubx = np.concatenate([q_hi, vq[:nq_], q_hi, vq[nq_:]])
        elif "q_range" in plant:
            qr = plant["q_range"]
            ocp.constraints.idxbx = np.arange(2 * self.nq)
            ocp.constraints.lbx = np.concatenate([qr[0], np.full(nx - 2 * self.nq, -1e5)])
            ocp.constraints.ubx = np.concatenate([qr[1], np.full(nx - 2 * self.nq, 1e5)])

        ocp.parameter_values = np.zeros(6 * nt)   # p = [f(2nt), rLref, rRref, ṙLref, ṙRref]

        opts = ocp.solver_options
        opts.nlp_solver_type = "SQP_RTI"          # 论文/ACADO real-time iteration
        opts.qp_solver = cfg.qp_solver
        opts.qp_solver_iter_max = 15
        opts.hessian_approx = "GAUSS_NEWTON"
        # 力优先模式：K=diag(1000,0) 只有 x 向 → y 向力残差恒 0 → GN 海森半正定；
        # CONVEXIFY/MIRROR 正定化 + Levenberg-Marquardt 兜底（正定时无影响）
        opts.regularize_method = cfg.reg_method
        if cfg.lm_reg > 0:
            opts.levenberg_marquardt = cfg.lm_reg
        opts.integrator_type = "ERK"
        opts.sim_method_num_stages = 4
        opts.sim_method_num_steps = 4
        opts.nlp_solver_max_iter = 20
        opts.print_level = 0 if not verbose else 1

        self.solver = AcadosOcpSolver(ocp, json_file=f"nmpcc_{model.name}.json",
                                      build=True, generate=True, verbose=verbose)
        # 失败重试用：HQP 重建对象（复用已编译产物，约 1–2 s）
        self._ocp = ocp
        self._json_file = f"nmpcc_{model.name}.json"

    def yref(self, refs, r_j_cur=None):
        """当前时刻 yref（论文式12 的参考部分）：
          yref_i = (1−If,i)[(1−Ic,i)·r_i,ref + Ic,i·r_j,cur] − If,i·f_i,ref
        r_j,cur 是另一臂当前测量位置（式28 误差状态的当前值），由 solve 传入；
        不传时退化为旧行为（假设 r_j,cur = r_j,ref，仅当 Ic=0 严格成立）。"""
        nt = self.nt
        Ic1, If1 = self.Ic1, self.If1
        Ic2, If2 = self.Ic2, self.If2
        rLr = np.asarray(refs["rL"], float)[:nt]
        rRr = np.asarray(refs["rR"], float)[:nt]
        vLr = np.asarray(refs.get("vL", np.zeros(nt)), float)[:nt]
        vRr = np.asarray(refs.get("vR", np.zeros(nt)), float)[:nt]
        fLr = np.asarray(refs.get("f_refL", np.zeros(nt)), float)[:nt]
        fRr = np.asarray(refs.get("f_refR", np.zeros(nt)), float)[:nt]
        if r_j_cur is None:
            rLc, rRc, rdLc, rdRc = rLr, rRr, vLr, vRr
        else:
            rLc, rRc, rdLc, rdRc = r_j_cur
        y = np.zeros(self.ny)
        y[:nt] = (1 - If1) * ((1 - Ic1) * rLr + Ic1 * rRc) - If1 * fLr
        y[nt:2 * nt] = (1 - If2) * ((1 - Ic2) * rRr + Ic2 * rLc) - If2 * fRr
        if self.cfg.with_epsdot:
            y[2 * nt:3 * nt] = (1 - If1) * ((1 - Ic1) * vLr + Ic1 * rdRc)
            y[3 * nt:4 * nt] = (1 - If2) * ((1 - Ic2) * vRr + Ic2 * rdLc)
        return y

    def yref_e(self, refs, r_j_cur=None):
        """终端 yref（无 u 项）。"""
        return self.yref(refs, r_j_cur)[:self.ny - self.nu] if not self.cfg.with_epsdot \
            else self.yref(refs, r_j_cur)[:4 * self.nt]

    def solve(self, x0, refs, f_meas):
        """求解一拍：p = [f_meas(2nt), rL_ref(nt), rR_ref(nt), ṙL_ref(nt), ṙR_ref(nt)]。
        ε_mult 完整表达式在残差内（r_j 为状态 → 预测内协同误差演化），yref ≡ 0。"""
        x0 = np.asarray(x0, float).reshape(-1)
        nt = self.nt
        f_meas = np.asarray(f_meas, float).reshape(-1)[:2 * nt]
        rLr = np.asarray(refs["rL"], float)[:nt]
        rRr = np.asarray(refs["rR"], float)[:nt]
        vLr = np.asarray(refs.get("vL", np.zeros(nt)), float)[:nt]
        vRr = np.asarray(refs.get("vR", np.zeros(nt)), float)[:nt]
        p_cur = np.concatenate([f_meas, rLr, rRr, vLr, vRr])
        for i in range(self.cfg.N):
            self.solver.set(i, "p", p_cur)
        self.solver.set(self.cfg.N, "p", p_cur)
        try:
            status = self.solver.solve_for_x0(x0_bar=x0)
        except RuntimeError:
            # 首次 QP 偶发 ACADOS_MINSTEP（半正定海森边界情形）→ 重建对象复用编译产物再解一次
            self.solver = AcadosOcpSolver(self._ocp, json_file=self._json_file,
                                          build=True, generate=True, verbose=False)
            for i in range(self.cfg.N):
                self.solver.set(i, "p", p_cur)
            self.solver.set(self.cfg.N, "p", p_cur)
            status = self.solver.solve_for_x0(x0_bar=x0)
        return self.solver.get(0, "u")


if __name__ == "__main__":
    print("nmpcc 模块自检：")
    print("  acados 可用:", _HAS_ACADOS)
    plant = two_link_plant()
    print("  双连杆 plant: nx =", plant["x"].shape[0], " nu =", plant["u"].shape[0])