#!/usr/bin/env python3
"""nmpcc_sync.py —— 论文 NMPCC 的独立实现（Experiment I 运动同步专用，无任何力/接触模型）

对应论文：
  式(11)      e_trac,i = r_i − r_i,ref ; e_cop,i = r_j − r_i ; e_f,i = f_i − f_i,ref
  式(12)      ε_mult,i = (1−ηf,i)[(1−ηcop,i)e_trac,i − ηcop,i·e_cop,i] − ηf,i·e_f,i
  式(28)(29)  误差动态（ε_mult 完整表达式放残差 → 预测中误差演化），yref ≡ 0
  式(30)      min ∫Ψ + Φ，受动力学/力矩约束
  式(31)(32)  Ψ = Q·ε² + ηdev·ε̇² + ηu·u² ；Φ 终端同型（P_mul·Q）

Experiment I 语义（用户确认）：双臂摆得很开，每臂跟自己的参考轨迹（共享 −20 mm/s y），
"运动同步"约束的是**相对位置的偏差**：
  ε_mult,i = e_trac,i − ηcop,i·e_trac,j          （= 式12 的协同项取参考相对形式
        e_cop' = (r_j − r_i) − (r_ref,j − r_ref,i) = e_trac,j − e_trac,i）
  平衡点：两臂各自回到参考 + 相对偏差 → 0（一起动、间距保持、误差归零）。
  这是与论文 Fig.10 "同步误差(相对参考) → 0" 一致的同构实现：
  直接把 ε 写成 (r_i − r_ref,i) − ηcop·(r_j − r_ref,j)，符号上只多一个参考参数 p。

实现（同实验二验证过的数值参数化路线，但独立自写、不含力项）：
  - 状态 x = [qL(7), vL(7), qR(7), vR(7)]（28 维，关节空间→动力学严格）
  - 预测 q̈ = M⁻¹(u + u_ff − G)：M/G 每拍从 MuJoCo 读入作 OCP 参数（horizon 冻结），
    u_ff = 重力补偿前馈（= −qfrc_bias 的负值，作为 yref 的 u 基线——与实验二一致；
    EPS 代价就是"稳态前馈 + 误差回馈"，RTI 一拍收敛）
  - 参数布局 p（136）= [M1(49), G1(7), M2(49), G2(7), r1ref(6), r2ref(6), v1ref(6), v2ref(6)]
  - FK 符号由 MuJoCo 模型数据生成（与 nmpcc_cart.mujoco_fk_symbols 同族，本文件自带全套）
  - acados: SQP_RTI + GAUSS_NEWTON + MIRROR 正则 + lm_reg + 热启动

运行/接口：
  from nmpcc_sync import NMPCCSync, NMPCCSyncConfig, mujoco_fk_symbols
  fk = mujoco_fk_symbols(model, arm_cfgs, ee_tip_z)
  nmpc = NMPCCSync(NMPCCSyncConfig(...), fk, umin=-8n, umax=8n)
  u = nmpc.solve(x0, p_coef, u_ref=gravity_comp())
  p_coef（136.）
"""

from __future__ import annotations

import numpy as np
import casadi as ca

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpOptions, AcadosOcpSolver
    _HAS_ACADOS = True
except ImportError:
    _HAS_ACADOS = False

NT = 6          # r = [pos(3), rotvec(3)]
NU = 14         # 双臂力矩
NQ = 7          # 每臂关节数
NX = 4 * NQ     # [qL, vL, qR, vR] = 28
# 参数布局：M1(49) + G1(7) + M2(49) + G2(7) + r1ref(6) + r2ref(6) + v1ref(6) + v2ref(6)
#            + u_ffL(7) + u_ffR(7) + DvL(7) + DvR(7) = 164
#   u_ff = 重力保持前馈（模型内补偿；代价保持纯 ηu·u² 结构）
#   Dv   = 关节粘性阻尼力矩（模型内建模；否则相位移-失配限制闭环增益上限）
NP = 2 * (49 + 7) + 4 * 6 + 2 * 7 + 2 * 7


# ======================================================================
# 数值参数化 plant 符号（预测内 M/G 冻结）
# ======================================================================
def build_ocp_symbolics():
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    p = ca.SX.sym("p", NP)
    qL = x[:NQ]; vL = x[NQ:2 * NQ]
    qR = x[2 * NQ:3 * NQ]; vR = x[3 * NQ:]
    M1 = ca.reshape(p[0:49], 7, 7)
    G1 = p[49:56]
    M2 = ca.reshape(p[56:105], 7, 7)
    G2 = p[105:112]
    uffL = p[136:143]
    uffR = p[143:150]
    # 粘性阻尼力矩：Dv_i·v_i（参数 p[150:164] = 实际 dof_damping 系数，每拍读数）
    dd1 = ca.solve(M1, u[:7] + uffL - G1 - p[150:157] * vL)
    dd2 = ca.solve(M2, u[7:] + uffR - G2 - p[157:164] * vR)
    return {"x": x, "u": u, "p": p,
            "f_expl": ca.vertcat(vL, dd1, vR, dd2)}


# ======================================================================
# 端面 FK 符号（来自 MuJoCo 模型数据）
# ======================================================================
def _quat2mat_sx(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        ca.horzcat(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        ca.horzcat(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def _rot_axis_sx(axis, q):
    k = axis / ca.norm_2(axis)
    K = ca.vertcat(
        ca.horzcat(0.0, -k[2], k[1]),
        ca.horzcat(k[2], 0.0, -k[0]),
        ca.horzcat(-k[1], k[0], 0.0))
    return ca.SX.eye(3) + ca.sin(q) * K + (1 - ca.cos(q)) * (K @ K)


def _rotvec_of_R_sx(R):
    tr = (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0
    theta = ca.acos(ca.fmax(ca.fmin(tr, 1.0), -1.0))
    vee = ca.vertcat(R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1])
    small = theta < 1e-6
    return ca.if_else(small, 0.5 * vee, 0.5 * theta / ca.sin(theta) * vee)


def mujoco_fk_symbols(model, arm_cfgs, ee_tip_z: float):
    """由 MuJoCo 模型数据生成符号端面 FK：tips = [rL(6), rR(6)]（position + rotvec）。"""
    import numpy as _np

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
            axis = model.jnt_axis[jt].copy() if has_joint else _np.zeros(3)
            axis = axis / (_np.linalg.norm(axis) + 1e-12)
            out.append({
                "pos": model.body_pos[bid].copy(),
                "quat": model.body_quat[bid].copy(),
                "axis": axis,
                "jointed": has_joint,
            })
        return out

    def tip_of(chain, q):
        H = ca.SX.eye(4)
        qi = 0
        for d in chain:
            Rb = _quat2mat_sx(ca.SX(d["quat"]))
            Rj = _rot_axis_sx(ca.SX(d["axis"]), q[qi]) if d["jointed"] else ca.SX.eye(3)
            if d["jointed"]:
                qi += 1
            H = H @ ca.vertcat(
                ca.horzcat(Rb @ Rj, ca.reshape(ca.SX(d["pos"]), 3, 1)),
                ca.horzcat(ca.SX.zeros(1, 3), ca.SX.ones(1, 1)))
        pos = (H @ ca.vertcat(ca.SX(0.0), ca.SX(0.0), ca.SX(ee_tip_z), ca.SX(1.0)))[:3]
        rv = _rotvec_of_R_sx(H[:3, :3])
        return ca.vertcat(pos, rv)

    chainL = collect_chain(arm_cfgs[0]["body_id"])
    chainR = collect_chain(arm_cfgs[1]["body_id"])
    nq = sum(1 for d in chainL if d["jointed"])
    nq2 = sum(1 for d in chainR if d["jointed"])
    assert nq == nq2
    qL = ca.SX.sym("qL", nq); qR = ca.SX.sym("qR", nq)
    tips = ca.vertcat(tip_of(chainL, qL), tip_of(chainR, qR))
    tips_fn = ca.Function("fk_tips", [qL, qR], [tips])
    return {"tips_qt": ca.vertcat(qL, qR), "tips": tips, "tips_fn": tips_fn, "n_arm": nq}


# ======================================================================
# 配置与求解器
# ======================================================================
class NMPCCSyncConfig:
    def __init__(self, N=20, Tp=0.1,
                 Q=None, Q_sync=None, eta_dev=80.0, eta_u=1e-4, eta_u_vec=None, P_mul=2.0,
                 reg_method="MIRROR", lm_reg=1e-2,
                 qp_solver="PARTIAL_CONDENSING_HPIPM"):
        """Experiment I（无接触/无力项）配置——性能版残差结构。

        跟踪残差（满权重，保留对移动参考的刚度）+ 同步残差（相对偏差，权重 Q_sync，
        即论文式(12) ηcop 耦合语义与 Fig.10 的同步误差量）。Q：(6,) 按轴；
        Q_sync：(6,) 同步按轴（x/y 有用，其余 0）；eta_dev：ε̇ 阻尼；
        eta_u/eta_u_vec：式(31) 输入正则。
        """
        if Q is None:
            Q = np.array([300.0, 300.0, 200.0, 100.0, 100.0, 100.0])
        if Q_sync is None:
            Q_sync = np.array([150.0, 150.0, 150.0, 0.0, 0.0, 0.0])
        self.N = N
        self.Tp = Tp
        self.Q = np.asarray(Q, float).copy()
        self.Q_sync = np.asarray(Q_sync, float).copy()
        self.eta_dev = eta_dev
        self.eta_u = eta_u
        self.eta_u_vec = eta_u_vec
        self.P_mul = P_mul
        self.reg_method = reg_method
        self.lm_reg = lm_reg
        self.qp_solver = qp_solver


class NMPCCSync:
    """论文 NMPCC（acados, SQP_RTI）—— Experiment I 运动同步专用、无接触力项。

    solve(x0, p_coef) -> u(14)
      x0       [qL(7), vL(7), qR(7), vR(7)]
      p_coef   (150,) 布局 = [M1(49), G1(7), M2(49), G2(7),
                             r1ref(6), r2ref(6), v1ref(6), v2ref(6),
                             u_ffL(7), u_ffR(7)]（u_ff = 重力保持前馈，模型内补偿）
    """

    def __init__(self, cfg: NMPCCSyncConfig, fk: dict, umin=None, umax=None,
                 verbose: bool = False):
        if not _HAS_ACADOS:
            raise RuntimeError("acados_template 不可用")
        self.cfg = cfg
        nt = NT
        n_arm = fk["n_arm"]

        sym = build_ocp_symbolics()
        x, u, p, f_expl = sym["x"], sym["u"], sym["p"], sym["f_expl"]

        # FK 注入：fk 的 qL/qR 符号 → OCP 状态切片
        q_qt = fk["tips_qt"]
        qL_s, qR_s = q_qt[:n_arm], q_qt[n_arm:]
        x_qL = x[:n_arm]; x_qR = x[2 * n_arm:3 * n_arm]
        tips = ca.substitute([fk["tips"]], [qL_s, qR_s], [x_qL, x_qR])[0]
        rL, rR = tips[:nt], tips[nt:2 * nt]
        r_dot = ca.jacobian(tips, x) @ f_expl
        rdL, rdR = r_dot[:nt], r_dot[nt:2 * nt]

        # 参考（参数 p，每拍更新，horizon 冻结）：r_i,ref / ṙ_i,ref
        r1ref = p[112:118]; r2ref = p[118:124]
        v1ref = p[124:130]; v2ref = p[130:136]

        # ---- 残差结构（性能版：论文式(12) 的"跟踪+同步"分解为独立残差）----
        #   跟踪残差（满权重，不削弱 common mode）: ε_trac,i = r_i − r_i,ref
        #   同步残差（论文式(12) 的协同项，即 Fig.10 的同步误差）:
        #     ε_sync = (r_L − r_ref,L) − (r_R − r_ref,R) = e_trac,L − e_trac,R
        #   说明：论文式(12) 把两者混在 ηcop=0.8 的一个残差里 → common mode 被
        #   (1−η)²=0.04 削弱 25 倍、relative 通道刚度/阻尼失配 → 闭环只能低增益慢收敛
        #   （论文 Fig.10 本身收敛 ~15 s 的根源）。拆开后可独立加权达到 PD 级性能。
        epsL = (rL - r1ref)
        epsR = (rR - r2ref)
        epsdL = (rdL - v1ref)
        epsdR = (rdR - v2ref)
        has_sync = np.any(np.asarray(cfg.Q_sync, float) > 0)
        if has_sync:
            epsS = epsL - epsR
            epsdS = epsdL - epsdR
            y_expr = ca.vertcat(epsL, epsR, epsS, epsdL, epsdR, epsdS, u)
            y_expr_e = ca.vertcat(epsL, epsR, epsS, epsdL, epsdR, epsdS)
        else:
            y_expr = ca.vertcat(epsL, epsR, epsdL, epsdR, u)
            y_expr_e = ca.vertcat(epsL, epsR, epsdL, epsdR)

        nblocks_eps = 2 + (1 if has_sync else 0)
        nblocks_epsd = 2 + (1 if has_sync else 0)
        W = np.zeros((y_expr.shape[0], y_expr.shape[0]))
        W[:nt, :nt] = np.diag(cfg.Q)                       # 跟踪：左（满权重）
        W[nt:2 * nt, nt:2 * nt] = np.diag(cfg.Q)           # 跟踪：右
        if has_sync:
            W[2 * nt:3 * nt, 2 * nt:3 * nt] = np.diag(cfg.Q_sync)   # 同步（相对偏差）
        d_off = nt * nblocks_eps
        for k in range(nblocks_epsd):
            off = d_off + nt * k
            W[off:off + nt, off:off + nt] = cfg.eta_dev * np.eye(nt)
        u_off = d_off + nt * nblocks_epsd
        eta_u_vec = (np.full(NU, cfg.eta_u) if cfg.eta_u_vec is None
                     else np.asarray(cfg.eta_u_vec, float))
        W[u_off:, u_off:] = np.diag(eta_u_vec)
        self.W = W
        self.ny = y_expr.shape[0]
        self._u_off = u_off
        self.nu = NU

        model = AcadosModel()
        model.name = "nmpcc_sync"
        model.x = x
        model.u = u
        model.p = p
        model.f_expl_expr = f_expl
        model.cost_y_expr = y_expr
        model.cost_y_expr_e = y_expr_e

        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = cfg.N
        ocp.solver_options.tf = cfg.Tp
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = W
        ocp.cost.W_e = W[:y_expr_e.shape[0], :y_expr_e.shape[0]].copy()
        ocp.cost.W_e[:] = cfg.P_mul * ocp.cost.W_e[:]
        ocp.cost.yref = np.zeros(self.ny)
        ocp.cost.yref_e = np.zeros(y_expr_e.shape[0])

        ocp.constraints.idxbu = np.arange(NU)
        ocp.constraints.lbu = (np.full(NU, -8.0) if umin is None else np.asarray(umin, float))
        ocp.constraints.ubu = (np.full(NU, 8.0) if umax is None else np.asarray(umax, float))
        ocp.constraints.x0 = np.zeros(NX)
        ocp.parameter_values = np.zeros(NP)

        opts = ocp.solver_options
        opts.nlp_solver_type = "SQP_RTI"
        opts.qp_solver = cfg.qp_solver
        opts.qp_solver_iter_max = 15
        opts.hessian_approx = "GAUSS_NEWTON"
        opts.regularize_method = cfg.reg_method
        if cfg.lm_reg > 0:
            opts.levenberg_marquardt = cfg.lm_reg
        opts.integrator_type = "ERK"
        opts.sim_method_num_stages = 4
        opts.sim_method_num_steps = 4
        opts.nlp_solver_max_iter = 20
        opts.print_level = 0 if not verbose else 1

        self.solver = AcadosOcpSolver(ocp, json_file="nmpcc_sync.json",
                                      build=True, generate=True, verbose=verbose)
        self._ocp = ocp
        self._json_file = "nmpcc_sync.json"

    def solve(self, x0, p_coef, u_ref=None):
        """论文 III-D 实时框架的每拍调用：更新参数 → RTI 一步 → 取首段控制。

        - RTI 由 acados 内部延续上一拍迭代（= ACADO 的 warm start 机制），无需手动
          get/set 轨迹（比旧版少 ~40 次跨界调用）。
        - 仅当传入 u_ref（重力保持基线）时设置 yref 的 u 段（式(31) 的 ηu·‖u‖² 以
          ‖u − u_ref‖² 形式使能——等价于论文把 G 内置模型；经实测参数注入版有偏差，
          故此形式）。
        """
        x0 = np.asarray(x0, float).reshape(-1)[:NX]
        p_full = np.asarray(p_coef, float).reshape(-1)[:NP]
        if not getattr(self, "_seeded", False):
            # 首拍播种：内部轨迹全零 = q=0 位形 → RTI 在错误位形线性化（首拍满幅爆炸）
            wx = np.tile(x0, (self.cfg.N, 1))
            wu = np.tile(u_ref if u_ref is not None else np.zeros(NU), (self.cfg.N, 1))
            for i in range(self.cfg.N):
                self.solver.set(i, "x", wx[i])
                self.solver.set(i, "u", wu[i])
            self._seeded = True
        yref = np.zeros(self.ny)
        if u_ref is not None:
            yref[self._u_off:] = np.asarray(u_ref, float).reshape(-1)[:NU]
        for i in range(self.cfg.N):
            self.solver.set(i, "p", p_full)
            self.solver.set(i, "yref", yref)
        self.solver.set(self.cfg.N, "p", p_full)
        self.solver.set(self.cfg.N, "yref", yref[:self.ny - NU])
        try:
            status = self.solver.solve_for_x0(x0_bar=x0)
        except RuntimeError:
            # 偶发 QP 失败 → 重建复用编译产物再解一次（兜底）
            self.solver = AcadosOcpSolver(self._ocp, json_file=self._json_file,
                                          build=True, generate=True, verbose=False)
            if not getattr(self, "_seeded", False):
                self._seeded = True
            for i in range(self.cfg.N):
                self.solver.set(i, "p", p_full)
                self.solver.set(i, "yref", yref)
            self.solver.set(self.cfg.N, "p", p_full)
            self.solver.set(self.cfg.N, "yref", yref[:self.ny - NU])
            status = self.solver.solve_for_x0(x0_bar=x0)
        return self.solver.get(0, "u")


if __name__ == "__main__":
    print("nmpcc_sync 模块自检：")
    print("  acados 可用:", _HAS_ACADOS)
    print("  NP =", NP, " NX =", NX, " NU =", NU)
    sym = build_ocp_symbolics()
    print("  f_expl 维度:", sym["f_expl"].shape)