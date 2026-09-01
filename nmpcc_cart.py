#!/usr/bin/env python3
"""nmpcc_cart.py —— 论文 NMPCC 实现（关节空间状态 + 严格笛卡尔代价，实验二搬箱子）

坐标系说明（用户确认的工程折衷，数学上同构）：
  论文式(25) 状态是末端笛卡尔 [r, ṙ]；本机 OpenARM-X 轻量臂（腕关节惯量 0.0009 kg·m²，
  抓取位形 cond(Λ⁻¹)≈2500）下笛卡尔状态的"冻结操作空间系数"预报被弱关节方向放大，
  预测/真实加速度失配 2200+ rad/s²，闭环不可行。采用同构等价实现：
  关节空间状态 x = [qL, vL, qR, vR]（28 维）+ 数值参数化动力学
  M(q)q̈ + G(q) = u + Jᵀf（M/G/J 每拍从 MuJoCo 读入作参数，horizon 冻结；
  科氏/离心按论文"摩擦=不确定项"同类处理略去，低速搬运任务贡献 <2 N·m），
  代价/误差/η/参考全部按论文原文（式11/12/28/29/31/32）——即 GUIDE 5.2/5.3 论证的
  "动力学等价、状态坐标不同"的严格翻译变体。

严格对应论文：
  式(11)      e_trac,i = r_i − r_i,ref ；e_cop,i = r_j − r_i ；e_f,i = f_i − f_i,ref
  式(12)      ε_mult,i = (I−ηf,i)[(I−ηcop,i)e_trac,i − ηcop,i·e_cop,i] − ηf,i·e_f,i
  式(28)(29)  误差动态：ε_mult 完整表达式放 NONLINEAR_LS 残差（r_i、r_j 均为
              状态函数 → 预测中 e_cop 演化 = 式(29) 误差动态），yref ≡ 0
  式(31)(32)  Ψ = Q·(ε_multᵀε_mult + ηdev·ε̇_multᵀε̇_mult + ηu·ūᵀū)；Φ 使用独立标量 P。
  相互力 f_rob,i 走测量值（F/T 传感器语义；每拍更新、horizon 冻结），进动力学与 ε；
  ε̇ 的 ḟ 为模型外信息（论文 ACADO 同款，不建模）

r 表示：r_i = [pos(3); rotvec(R_link7)(3)]（论文 αβγ 的工程等价）；
速度 ṙ_i = [v(3); ω(3)]（世界系，小角度 rv̇≈ω，θ<0.05 rad 内误差 O(θ²)）。

参数 p 布局（共 176）：
  [M1(49), G1(7), fJ1(7), u_ff1(7), M2(49), G2(7), fJ2(7), u_ff2(7),
   r1ref(6), r2ref(6), v1ref(6), v2ref(6), f1ref(6), f2ref(6)]
  M_i = 该臂 7×7 惯性（列主序）；G_i = 重力广义力（=∂V/∂q）；
  fJ_i = Jᵀ·f_meas（接触力回射，测量）；u_ff_i = Jᵀ·F_des（**力前馈指令**，每拍更新）；
  预测动力学 q̈ = M⁻¹(u + u_ff − G + fJ)：力前馈进动力学 → 预测与执行一致，
  25 N 夹持力由常数前馈承担（对照组已验证机制），ε 力项退为温和测量修正。

运行：
  from nmpcc_cart import NMPCCCart, NMPCCCartConfig
  nmpc = NMPCCCart(NMPCCCartConfig(...))
  u = nmpc.solve(x0_q, p_full)
"""

from __future__ import annotations

import json
import hashlib
import os
import time
import numpy as np

_DEBUG_LOG = r"D:\RobotSim\.cursor\debug-068a93.log"

def _debug_log(hypothesis_id, location, message, data):
    record = {"sessionId": "068a93", "runId": "post-fix", "hypothesisId": hypothesis_id,
              "location": location, "message": message, "data": data,
              "timestamp": int(time.time() * 1000)}
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception:
        pass
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
# 参数布局（实际使用）：
# [M1(49), G1(7), M2(49), G2(7), fJ1(7), fJ2(7),
#  r1ref(6), r2ref(6), v1ref(6), v2ref(6), f1ref(6), f2ref(6), D1(7), D2(7),
#  bFL(1), bFR(1)]
# bFL/bFR 每拍由实测法向力减去当前状态名义力得到，并在 horizon 内冻结。
N_FORCE_BIAS = 2
NP_BASE = 2 * (49 + 7) + 14 + 6 * 6 + 14
NP = NP_BASE + N_FORCE_BIAS


# ======================================================================
# 数值参数化 plant 符号（预测内 M/G 冻结；无深度符号展开——编译快）
# ======================================================================
def _rotvec_of_R_sx(R):
    tr = (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0
    theta = ca.acos(ca.fmax(ca.fmin(tr, 1.0), -1.0))
    vee = ca.vertcat(R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1])
    small = theta < 1e-6
    return ca.if_else(small, 0.5 * vee, 0.5 * theta / ca.sin(theta) * vee)


def build_ocp_symbolics(kx: float = 3750.0, rest_gap: float = 0.248,
                        smooth_eps: float = 0.002):
    """数值参数化动力学符号 + 参数符号。ε 残差由 NMPCCCart 直接构造（见其 __init__）。

    数值参数（每拍从 MuJoCo 读入、horizon 冻结）：M(7×7)/G(7)；其余全部状态函数：
      式(27) 夹持物形变力（单边压缩弹簧，状态函数 → NMPC 可预测夹紧）：
        f_1 = −f_2 = kx·max(0, d0 − (x1 − x2))  （x 向；臂1=左臂=+x 侧；平滑死区）
      动力学 q̈ = M⁻¹(u − G + Jpᵀ·f) ；ε 力误差与 ε̇ 力导由该状态力解析给出。
    """
    x = ca.SX.sym("x", NX)          # [qL(7), vL(7), qR(7), vR(7)]
    u = ca.SX.sym("u", NU)
    p = ca.SX.sym("p", NP)
    qL = x[:NQ]; vL = x[NQ:2 * NQ]
    qR = x[2 * NQ:3 * NQ]; vR = x[3 * NQ:]

    M1 = ca.reshape(p[0:49], 7, 7)
    G1 = p[49:56]
    M2 = ca.reshape(p[56:105], 7, 7)
    G2 = p[105:112]
    fJ1 = p[112:119]
    fJ2 = p[119:126]
    D1 = p[162:169]
    D2 = p[169:176]
    dd1 = ca.solve(M1, u[:7] + fJ1 - G1 - D1 * vL)
    dd2 = ca.solve(M2, u[7:] + fJ2 - G2 - D2 * vR)
    f_expl = ca.vertcat(vL, dd1, vR, dd2)

    Ic1 = ca.SX.sym("Ic1", NT); Ic2 = ca.SX.sym("Ic2", NT)
    If1 = ca.SX.sym("If1", NT); If2 = ca.SX.sym("If2", NT)
    return {"x": x, "u": u, "p": p, "f_expl": f_expl,
            "Ic": [Ic1, Ic2], "If": [If1, If2]}


# ======================================================================
# 端点 FK 符号（来自 MuJoCo 模型数据；轻量：无 hessian，编译快）
# ======================================================================
def _quat2mat_sx(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        ca.horzcat(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        ca.horzcat(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def _rot_axis(axis, q):
    k = axis / ca.norm_2(axis)
    K = ca.vertcat(
        ca.horzcat(0.0, -k[2], k[1]),
        ca.horzcat(k[2], 0.0, -k[0]),
        ca.horzcat(-k[1], k[0], 0.0))
    return ca.SX.eye(3) + ca.sin(q) * K + (1 - ca.cos(q)) * (K @ K)


def _quat_to_mat_np(q):
    """将 MuJoCo wxyz 四元数转换为旋转矩阵。"""
    w, x, y, z = np.asarray(q, dtype=float)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pad_contact_offsets(model, arms, ee_tip_z: float):
    """返回左右 pad 朝箱体接触面中心相对 link7 tip 的局部 offset。

    ``geom_pos`` 是 pad 中心相对 link7 原点的位置，``geom_size[2]``
    是 pad 半厚度；pad 的局部 +z 面是实际朝向箱体的受力面。
    """
    offsets = []
    tip_offset = np.array([0.0, 0.0, float(ee_tip_z)])
    for arm in arms:
        pad_id = int(arm["pad_id"])
        pad_pos = np.asarray(model.geom_pos[pad_id], dtype=float)
        pad_R = _quat_to_mat_np(model.geom_quat[pad_id])
        pad_face = pad_pos + pad_R @ np.array(
            [0.0, 0.0, float(model.geom_size[pad_id, 2])])
        offsets.append(pad_face - tip_offset)
    return offsets


def mujoco_fk_symbols(model, arm_cfgs, ee_tip_z: float,
                      contact_offsets=None):
    """生成符号 FK；轨迹点为 tip，接触点用于夹持弹簧和法向力 Jacobian。

    ``contact_offsets`` 是每侧 pad 实际接触面中心相对 link7 tip 的局部
    offset。若省略，接触点退化为 tip，兼容不含实体 pad 的旧调用。
    返回字典中的 ``contact_tips`` 为 [pL_contact, pR_contact]。
    """
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
            # #region agent log
            _debug_log("H1-H4", "nmpcc_cart.py:159", "before FK joint-axis access", {
                "module": os.path.abspath(__file__),
                "numpy_global_present": "np" in globals(),
                "numpy_global_name": type(globals().get("np")).__name__ if "np" in globals() else None,
                "casadi_global_present": "ca" in globals(),
            })
            # #endregion
            axis = model.jnt_axis[jt].copy() if has_joint else np.zeros(3)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            out.append({
                "pos": model.body_pos[bid].copy(),
                "quat": model.body_quat[bid].copy(),
                "axis": axis,
                "jointed": has_joint,
            })
        return out

    def chain_H(chain, q):
        H = ca.SX.eye(4)
        qi = 0
        for d in chain:
            Rb = _quat2mat_sx(ca.SX(d["quat"]))
            Rj = _rot_axis(ca.SX(d["axis"]), q[qi]) if d["jointed"] else ca.SX.eye(3)
            if d["jointed"]:
                qi += 1
            H = H @ ca.vertcat(
                ca.horzcat(Rb @ Rj, ca.reshape(ca.SX(d["pos"]), 3, 1)),
                ca.horzcat(ca.SX.zeros(1, 3), ca.SX.ones(1, 1)))
        return H

    def tip_of(chain, q):
        H = chain_H(chain, q)
        pos = (H @ ca.vertcat(ca.SX(0.0), ca.SX(0.0), ca.SX(ee_tip_z), ca.SX(1.0)))[:3]
        R = H[:3, :3]
        rv = _rotvec_of_R_sx(R)
        return ca.vertcat(pos, rv)

    def contact_pos_of(chain, q, offset):
        H = chain_H(chain, q)
        tip = H @ ca.vertcat(ca.SX(0.0), ca.SX(0.0), ca.SX(ee_tip_z), ca.SX(1.0))
        offset = ca.SX(np.asarray(offset, dtype=float).reshape(3, 1))
        return (tip[:3] + H[:3, :3] @ offset)

    chainL = collect_chain(arm_cfgs[0]["body_id"])
    chainR = collect_chain(arm_cfgs[1]["body_id"])
    nq = sum(1 for d in chainL if d["jointed"])
    qL = ca.SX.sym("qL", nq); qR = ca.SX.sym("qR", nq)
    rL_ = tip_of(chainL, qL)
    rR_ = tip_of(chainR, qR)
    tips = ca.vertcat(rL_, rR_)
    if contact_offsets is None:
        contact_offsets = (np.zeros(3), np.zeros(3))
    if len(contact_offsets) != 2:
        raise ValueError("contact_offsets 应包含左右两侧各一个 3D offset")
    cL_ = contact_pos_of(chainL, qL, contact_offsets[0])
    cR_ = contact_pos_of(chainR, qR, contact_offsets[1])
    contact_tips = ca.vertcat(cL_, cR_)
    tips_fn = ca.Function("fk_tips", [qL, qR], [tips])
    contact_tips_fn = ca.Function("fk_contact_tips", [qL, qR], [contact_tips])
    return {"tips_qt": ca.vertcat(qL, qR), "tips": tips, "tips_fn": tips_fn,
            "contact_tips": contact_tips, "contact_tips_fn": contact_tips_fn,
            "n_arm": nq}


def rotvec_of_R(R):
    """世界旋转矩阵 (3,3) → 旋转向量（主值，与符号实现一致）。"""
    R = np.asarray(R, float)
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-6:
        return 0.5 * vee
    return 0.5 * theta / np.sin(theta) * vee


class NMPCCCartConfig:
    def __init__(self, N=50, Tp=0.1,
                 eta_cop=None, eta_f=None,
                 Q=1.0, eta_dev=80.0, eta_u=0.05,
                 P=2.0,
                 kx=3750.0, rest_gap=0.248, kf_scale=0.15, smooth_eps=0.002,
                 vmax=0.05, vz_max=0.002,
                 reg_method="MIRROR", lm_reg=1e-2,
                 qp_solver="PARTIAL_CONDENSING_HPIPM",
                 sim_method_num_steps=4, sim_method_num_stages=4,
                 integrator_type="ERK",
                 qp_solver_cond_N=None, qp_solver_iter_max=15,
                 nlp_solver_max_iter=20,
                 model_name="nmpcc_cart_dual_force_bias_uweight_fix",
                 fail_on_nonzero_status=True, rebuild_on_failure=True):
        """实验二（搬箱子）配置。η 映射（论文 Experiment II 轴向语义）：
          x 平移轴：ηf=0.8, ηcop=0（阻抗：保持夹持力）
          y 平移轴：ηcop=0.8, ηf=0（运动同步）；z 平移轴：ηcop=0（纯跟踪托重）
          姿态 αβγ：ηcop=0, ηf=0（纯跟踪）
        Q：(6,) 按轴；kx：力误差单位化刚度；kf_scale：力误差增益（0-1，按闭环稳定性整定：
        有效位移增益 = (If/(1−If))·kf_scale/kx 需 < 1/接触刚度，避免"力-位置"极限环）。

        求解器调优选项（默认值保持既有行为不变）：
          qp_solver            QP 求解器（PARTIAL_CONDENSING_HPIPM / FULL_CONDENSING_HPIPM /
                               FULL_CONDENSING_QPOASES 等）
          sim_method_num_steps/stages  每段 RK 步数/级数；integrator_type 默认 ERK
          qp_solver_cond_N    部分凝聚块数（<N 时 block_size=N/cond_N；仅在 partial 生效）
          model_name          独立模型名 → 独立 codegen 目录与 json（多求解器并行编译不冲突）
          fail_on_nonzero_status / rebuild_on_failure  求解失败处理（默认与原行为一致）
        """
        if eta_cop is None:
            eta_cop = np.array([[0.0, 0.8, 0.0, 0.0, 0.0, 0.0]] * 2)
        if eta_f is None:
            eta_f = np.array([[0.8, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2)
        self.N = int(N)
        self.Tp = float(Tp)
        self.eta_cop = np.asarray(eta_cop, float).copy()
        self.eta_f = np.asarray(eta_f, float).copy()
        # #region agent log
        _debug_log("H1-H5", "nmpcc_cart.py:246", "config eta attributes initialized", {
            "module": os.path.abspath(__file__),
            "has_eta_cop": hasattr(self, "eta_cop"),
            "has_eta_f": hasattr(self, "eta_f"),
            "eta_cop_shape": list(self.eta_cop.shape),
            "eta_f_shape": list(self.eta_f.shape),
        })
        # #endregion
        self.Q = float(Q)
        self.eta_dev = float(eta_dev)
        self.eta_u = float(eta_u)
        self.P = float(P)
        self.kx = kx
        self.rest_gap = rest_gap
        self.kf_scale = kf_scale
        self.smooth_eps = smooth_eps
        self.vmax = float(vmax)
        self.vz_max = float(vz_max)
        self.reg_method = reg_method
        self.lm_reg = lm_reg
        self.qp_solver = qp_solver
        self.sim_method_num_steps = sim_method_num_steps
        self.sim_method_num_stages = sim_method_num_stages
        self.integrator_type = integrator_type
        self.qp_solver_cond_N = qp_solver_cond_N
        self.qp_solver_iter_max = qp_solver_iter_max
        self.nlp_solver_max_iter = nlp_solver_max_iter
        self.model_name = model_name
        self.fail_on_nonzero_status = fail_on_nonzero_status
        self.rebuild_on_failure = rebuild_on_failure


class NMPCCCart:
    """论文 NMPCC（acados, SQP_RTI）—— 关节空间状态 + 笛卡尔 ε_mult 代价。

    solve(x0, p) → u(14)
      x0  [qL(7), vL(7), qR(7), vR(7)] 当前关节状态
      p   (176,) 布局见模块 docstring
    """

    def __init__(self, cfg: NMPCCCartConfig, fk: dict, umin=None, umax=None,
                 verbose: bool = False):
        if not _HAS_ACADOS:
            raise RuntimeError("acados_template 不可用")
        self.cfg = cfg
        # #region agent log
        _debug_log("H1-H5", "nmpcc_cart.py:289", "NMPCC constructor received config", {
            "module": os.path.abspath(__file__),
            "cfg_type": type(cfg).__name__,
            "cfg_attrs": sorted(name for name in ("eta_cop", "eta_f", "Q", "P")
                                 if hasattr(cfg, name)),
            "eta_cop_shape": list(cfg.eta_cop.shape) if hasattr(cfg, "eta_cop") else None,
            "eta_f_shape": list(cfg.eta_f.shape) if hasattr(cfg, "eta_f") else None,
        })
        # #endregion
        nt = NT
        n_arm = fk["n_arm"]

        def _norm2(eta, name):
            eta = np.asarray(eta, float)
            if eta.shape == (nt,):
                return eta.copy(), eta.copy()
            if eta.shape == (2, nt):
                return eta[0].copy(), eta[1].copy()
            raise ValueError(f"{name} 应为 (6,) 或 (2,6)，收到 {eta.shape}")
        self.Ic1, self.Ic2 = _norm2(cfg.eta_cop, "eta_cop")
        self.If1, self.If2 = _norm2(cfg.eta_f, "eta_f")

        # 状态/动力学（build 符号）
        sym = build_ocp_symbolics(cfg.kx, cfg.rest_gap)
        x, u, p, f_expl = sym["x"], sym["u"], sym["p"], sym["f_expl"]
        fJ1 = p[112:119]
        fJ2 = p[119:126]
        # 端面 FK 注入：fk 的 qL/qR 符号 → 替换为 OCP 状态切片（同树）
        q_qt = fk["tips_qt"]
        qL_s, qR_s = q_qt[:n_arm], q_qt[n_arm:]
        x_qL = x[:n_arm]; x_qR = x[2 * n_arm:3 * n_arm]
        tips = ca.substitute([fk["tips"]], [qL_s, qR_s], [x_qL, x_qR])[0]
        rL, rR = tips[:nt], tips[nt:2 * nt]
        pL, pR = rL[:3], rR[:3]
        contact_tips = fk.get("contact_tips", fk["tips"])
        contact_tips = ca.substitute([contact_tips], [qL_s, qR_s], [x_qL, x_qR])[0]
        cL, cR = contact_tips[:3], contact_tips[3:6]

        # ---- 式(27) 夹持物形变力（最新场景：left=-x，right=+x）----
        kx_uni = cfg.kx
        kf_scale = cfg.kf_scale
        # 最新场景的物理编号：left 在 -x、right 在 +x。
        # 因此间距 d = x_right - x_left；压缩 c = rest_gap - d。
        # 箱对 left 的反力向 -x，箱对 right 的反力向 +x。
        d = cR[0] - cL[0]
        c = cfg.rest_gap - d
        c_s = 0.5 * (c + ca.sqrt(c * c + cfg.smooth_eps ** 2))
        f1x = -kx_uni * c_s                       # left 臂：向 -x 的外推力
        f1mod = ca.vertcat(f1x, ca.SX.zeros(2))
        f2mod = -f1mod                             # right 臂：向 +x
        f1m6 = ca.vertcat(f1mod, ca.SX.zeros(3))
        f2m6 = ca.vertcat(f2mod, ca.SX.zeros(3))
        # 每拍用实测法向力锚定预测起点；左右偏差在当前 horizon 内冻结。
        bFL = p[NP_BASE]
        bFR = p[NP_BASE + 1]
        f1pred = f1m6 + ca.vertcat(bFL, ca.SX.zeros(5))
        f2pred = f2m6 + ca.vertcat(bFR, ca.SX.zeros(5))
        f1dyn = ca.vertcat(f1pred[0], ca.SX.zeros(2))
        f2dyn = ca.vertcat(f2pred[0], ca.SX.zeros(2))
        dcd_full = ca.jacobian(c_s, x)               # (1, 28)
        cd = (dcd_full[0, 0:NQ] @ x[NQ:2 * NQ]) + (dcd_full[0, 2 * NQ:3 * NQ] @ x[3 * NQ:])
        fd1x = -kx_uni * cd                  # f_left = -K*c，保持符号与 f1mod 一致
        fdot1 = ca.vertcat(ca.vertcat(fd1x, ca.SX.zeros(2)), ca.SX.zeros(3))
        fdot2 = -fdot1

        # 法向接触力作用在 pad 实际接触面，Jacobian 也必须在同一点计算。
        Jp1q = ca.jacobian(cL, x_qL)
        Jp2q = ca.jacobian(cR, x_qR)
        M1 = ca.reshape(p[0:49], 7, 7)
        M2 = ca.reshape(p[56:105], 7, 7)
        dd1 = (sym["f_expl"][NQ:2 * NQ]
               + ca.solve(M1, Jp1q.T @ f1dyn - fJ1))
        dd2 = (sym["f_expl"][3 * NQ:]
               + ca.solve(M2, Jp2q.T @ f2dyn - fJ2))
        f_expl_full = ca.vertcat(x[NQ:2 * NQ], dd1, x[3 * NQ:], dd2)
        # 末端速度必须由完整预测动力学计算，包含式(27)夹持力对 q̈ 的影响。
        r_dot = ca.jacobian(tips, x) @ f_expl_full
        rdL, rdR = r_dot[:nt], r_dot[nt:2 * nt]

        Ic1, Ic2 = sym["Ic"]; If1, If2 = sym["If"]
        r1ref = p[126:132]; r2ref = p[132:138]
        v1ref = p[138:144]; v2ref = p[144:150]
        f1ref = p[150:156]; f2ref = p[156:162]
        # ε（式12 全表达式；预测力用当前实测法向力锚定）
        eps1 = (1 - If1) * (rL - (1 - Ic1) * r1ref - Ic1 * rR) - If1 * kf_scale * (f1pred - f1ref) 
        eps2 = (1 - If2) * (rR - (1 - Ic2) * r2ref - Ic2 * rL) - If2 * kf_scale * (f2pred - f2ref) 
        # ε̇（式12 时间导数，存在 式27 → ė_f 严格解析）
        epsd1 = (1 - If1) * (rdL - (1 - Ic1) * v1ref - Ic1 * rdR) - If1 * kf_scale * fdot1
        epsd2 = (1 - If2) * (rdR - (1 - Ic2) * v2ref - Ic2 * rdL) - If2 * kf_scale * fdot2 
        y_expr = ca.vertcat(eps1, eps2, epsd1, epsd2, u)
        y_expr_e = ca.vertcat(eps1, eps2, epsd1, epsd2)

        # 参数拼接：model.p = [p(148), Ic1,Ic2,If1,If2(24)] = 172
        self._np_full = NP + 4 * NT
        model = AcadosModel()
        model.name = cfg.model_name
        model.x = x
        model.u = u
        model.p = ca.vertcat(p, Ic1, Ic2, If1, If2)
        model.f_expl_expr = f_expl_full
        model.cost_y_expr = y_expr
        model.cost_y_expr_e = y_expr_e

        ny = y_expr.shape[0]
        ne = y_expr_e.shape[0]
        W = np.zeros((ny, ny))
        W[:2 * nt, :2 * nt] = cfg.Q * np.eye(2 * nt)
        W[2 * nt:4 * nt, 2 * nt:4 * nt] = cfg.Q * cfg.eta_dev * np.eye(2 * nt)
        # 式(31)：Q 乘整个括号，eta_dev/eta_u 是括号内相对权重。
        input_offset = 4 * nt
        W[input_offset:, input_offset:] = cfg.Q * cfg.eta_u * np.eye(NU)
        self.W = W
        self.ny = ny
        self.input_offset = input_offset
        self._force_nom_fn = ca.Function(
            f"{cfg.model_name}_force_nominal", [x], [f1x, f2mod[0]])
        self._diagnostic_fn = ca.Function(
            f"{cfg.model_name}_diagnostic",
            [x, p, Ic1, Ic2, If1, If2],
            [ca.vertcat(rL, rR), ca.vertcat(rdL, rdR),
             ca.vertcat(f1m6[0], f2m6[0]),
             ca.vertcat(f1pred[0], f2pred[0]),
             ca.vertcat(f1pred[0] - f1ref[0], f2pred[0] - f2ref[0]),
             ca.vertcat(rL[0] - r1ref[0], rR[0] - r2ref[0]),
             ca.vertcat(eps1, eps2), ca.vertcat(epsd1, epsd2)])

        ocp = AcadosOcp()
        ocp.model = model
        # #region agent log
        _debug_log("H-N", "nmpcc_cart.py:424", "before solver horizon configuration", {
            "cfg_attrs": sorted(vars(cfg).keys()),
            "has_N": hasattr(cfg, "N"),
            "has_Tp": hasattr(cfg, "Tp"),
            "N": getattr(cfg, "N", None),
            "Tp": getattr(cfg, "Tp", None),
        })
        # #endregion
        ocp.solver_options.N_horizon = cfg.N
        ocp.solver_options.tf = cfg.Tp
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = W
        # 终端残差没有输入项，直接从阶段权重取前 ne=24 维，再乘独立 P。
        ocp.cost.W_e = cfg.P * W[:ne, :ne].copy()
        # #region agent log
        _debug_log("H-COST", "nmpcc_cart.py:441", "cost dimensions prepared", {
            "ny": ny, "ne": ne, "W_shape": list(W.shape),
            "W_e_shape": list(ocp.cost.W_e.shape),
            "yref_e_shape": list(np.zeros(ne).shape),
            "Q": cfg.Q, "P": cfg.P,
        })
        # #endregion
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ne)

        ocp.constraints.idxbu = np.arange(NU)
        ocp.constraints.lbu = (np.full(NU, -8.0) if umin is None else np.asarray(umin, float))
        ocp.constraints.ubu = (np.full(NU, 8.0) if umax is None else np.asarray(umax, float))
        # z速度限幅
        speed_L = ca.sqrt(ca.dot(rdL[:3], rdL[:3]) + 1e-10)
        speed_R = ca.sqrt(ca.dot(rdR[:3], rdR[:3]) + 1e-10)
        ocp.model.con_h_expr = ca.vertcat(speed_L, speed_R, rdL[2], rdR[2])
        ocp.constraints.lh = np.array([0.0, 0.0, -cfg.vz_max, -cfg.vz_max])
        ocp.constraints.uh = np.array([cfg.vmax, cfg.vmax, cfg.vz_max, cfg.vz_max])
        ocp.constraints.x0 = np.zeros(NX)
        ocp.parameter_values = np.zeros(self._np_full)

        opts = ocp.solver_options
        opts.nlp_solver_type = "SQP_RTI"
        opts.qp_solver = cfg.qp_solver
        opts.qp_solver_iter_max = cfg.qp_solver_iter_max
        opts.hessian_approx = "GAUSS_NEWTON"
        opts.regularize_method = cfg.reg_method
        if cfg.lm_reg > 0:
            opts.levenberg_marquardt = cfg.lm_reg
        opts.integrator_type = cfg.integrator_type
        opts.sim_method_num_stages = cfg.sim_method_num_stages
        opts.sim_method_num_steps = cfg.sim_method_num_steps
        if cfg.qp_solver_cond_N is not None:
            opts.qp_solver_cond_N = cfg.qp_solver_cond_N
        opts.nlp_solver_max_iter = cfg.nlp_solver_max_iter
        opts.print_level = 0 if not verbose else 1

        # acados 默认把 codegen 目录放在当前工作目录并拼接完整模型名。
        # Windows/CasADi 对深层路径较敏感，因此使用基于模型名的短哈希目录；
        # 模型名仍保留完整形式，保证不同参数配置的生成函数不会冲突。
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(module_dir)
        codegen_key = hashlib.sha1(cfg.model_name.encode("utf-8")).hexdigest()[:12]
        codegen_dir = os.path.join(project_dir, "acados_codegen", codegen_key)
        os.makedirs(codegen_dir, exist_ok=True)
        json_file = os.path.join(codegen_dir, f"{cfg.model_name}.json")
        ocp.code_gen_options.code_export_directory = codegen_dir
        ocp.code_gen_options.json_file = json_file

        # 复用上次已编译的求解器（acados 代码复用，跳过重新编译）；
        # 缓存不可用（首次运行/参数目录不存在）时才重新 build+generate。
        try:
            self.solver = AcadosOcpSolver(ocp,
                                          build=False, generate=False,
                                          verbose=verbose)
        except (FileNotFoundError, OSError) as exc:
            print(f"[acados] cached solver unavailable; rebuilding: {exc}")
            self.solver = AcadosOcpSolver(ocp,
                                          build=True, generate=True,
                                          verbose=verbose)
        self._ocp = ocp
        self._json_file = json_file
        # #region agent log
        _debug_log("H-N", "nmpcc_cart.py:480", "acados solver constructed", {
            "module": os.path.abspath(__file__),
            "N": self.cfg.N,
            "Tp": self.cfg.Tp,
            "ny": self.ny,
            "ne": ne,
            "input_offset": self.input_offset,
        })
        # #endregion

    def _full_p(self, p_coef):
        """p_full = [plant+force_bias, eta(24)]。"""
        p_coef = np.asarray(p_coef, float).reshape(-1)
        if p_coef.size == NP_BASE:
            p_coef = np.concatenate([p_coef, np.zeros(N_FORCE_BIAS)])
        if p_coef.size != NP:
            raise ValueError(f"p_coef 应为 {NP} 个参数（或旧版 {NP_BASE} 个），收到 {p_coef.size}")
        eta = np.concatenate([self.Ic1, self.Ic2, self.If1, self.If2])
        return np.concatenate([p_coef, eta])

    def _anchor_measured_force(self, x0, p_coef, f_meas):
        p_coef = np.asarray(p_coef, float).reshape(-1).copy()
        if p_coef.size == NP_BASE:
            p_coef = np.concatenate([p_coef, np.zeros(N_FORCE_BIAS)])
        if p_coef.size != NP:
            raise ValueError(f"p_coef 应为 {NP} 个参数，收到 {p_coef.size}")
        nominal = np.asarray(self._force_nom_fn(x0), dtype=float).reshape(-1)
        measured = nominal.copy() if f_meas is None else np.asarray(f_meas, float).reshape(-1)
        if measured.size < N_FORCE_BIAS:
            raise ValueError(f"f_meas 应包含左右带符号法向力，收到 {measured.size} 个数")
        bias = measured[:N_FORCE_BIAS] - nominal
        p_coef[NP_BASE:NP] = bias
        self.last_force_nominal = nominal.copy()
        self.last_force_measured = measured[:N_FORCE_BIAS].copy()
        self.last_force_bias = bias.copy()
        self.last_force_anchored = nominal + bias
        return p_coef

    def solve(self, x0, p_coef, u_ref=None, f_meas=None, u_seed=None):
        x0 = np.asarray(x0, float).reshape(-1)[:NX]
        p_coef = self._anchor_measured_force(x0, p_coef, f_meas)
        p_full = self._full_p(p_coef)
        yref = np.zeros(self.ny)
        # 论文式(31)：输入项为 ūᵀηᵤū（零参考），不把 u_ref 写入 yref。
        # u_ref 参数仅保留向后兼容，不再进入代价。
        cold = False
        if getattr(self, "_warm", None) is not None:
            wx, wu = self._warm
            for i in range(self.cfg.N):
                self.solver.set(i, "x", wx[i])
                self.solver.set(i, "u", wu[i])
        elif u_seed is not None:
            # 冷启动退化保护：首次接手用预压平衡力矩 + 当前状态作种子
            # （论文：预载完成 → 双臂静止过渡，同步误差渐收敛；不作冷启动贴边大解）
            us = np.asarray(u_seed, float).reshape(-1)[:NU]
            xs = x0.copy()
            for i in range(self.cfg.N):
                self.solver.set(i, "x", xs)
                self.solver.set(i, "u", us)
            cold = True
        for i in range(self.cfg.N):
            self.solver.set(i, "p", p_full)
            self.solver.set(i, "yref", yref)
        self.solver.set(self.cfg.N, "p", p_full)
        self.solver.set(self.cfg.N, "yref", yref[:self.ny - NU])
        try:
            self.solver.solve_for_x0(
                x0_bar=x0,
                fail_on_nonzero_status=self.cfg.fail_on_nonzero_status,
                print_stats_on_failure=True)
            status = int(self.solver.status)
        except RuntimeError as e:
            status = 4  # ACADOS_SOLVER_FAILURE（兼容旧路径：fail_on_nonzero_status=True 时抛出）
        if status != 0 and self.cfg.rebuild_on_failure:
            self.solver = AcadosOcpSolver(self._ocp, json_file=self._json_file,
                                          build=True, generate=True, verbose=False)
            if getattr(self, "_warm", None) is not None:
                wx, wu = self._warm
                for i in range(self.cfg.N):
                    self.solver.set(i, "x", wx[i])
                    self.solver.set(i, "u", wu[i])
            elif cold:
                us = np.asarray(u_seed, float).reshape(-1)[:NU]
                xs = x0.copy()
                for i in range(self.cfg.N):
                    self.solver.set(i, "x", xs)
                    self.solver.set(i, "u", us)
            for i in range(self.cfg.N):
                self.solver.set(i, "p", p_full)
                self.solver.set(i, "yref", yref)
            self.solver.set(self.cfg.N, "p", p_full)
            self.solver.set(self.cfg.N, "yref", yref[:self.ny - NU])
            status = self.solver.solve_for_x0(
                x0_bar=x0,
                fail_on_nonzero_status=self.cfg.fail_on_nonzero_status,
                print_stats_on_failure=True)
            status = int(self.solver.status)
        # 记录本拍求解统计（QP 状态/迭代/耗时），供基准测试与诊断
        self.last_status = int(status)
        try:
            self.last_qp_iter = int(self.solver.get_stats("qp_iter")[0])
        except Exception:
            self.last_qp_iter = -1
        try:
            self.last_time_qp = float(self.solver.get_stats("time_qp")[0])
        except Exception:
            self.last_time_qp = float("nan")
        try:
            self.last_time_sim = float(self.solver.get_stats("time_sim")[0])
        except Exception:
            self.last_time_sim = float("nan")
        try:
            self.last_time_lin = float(self.solver.get_stats("time_lin")[0])
        except Exception:
            self.last_time_lin = float("nan")
        # 存本拍轨迹（warm start 与内部诊断用）
        wx = np.array([self.solver.get(i, "x") for i in range(self.cfg.N + 1)])
        wu = np.array([self.solver.get(i, "u") for i in range(self.cfg.N)])
        diag = []
        for xi in wx:
            values = self._diagnostic_fn(
                xi, p_coef, self.Ic1, self.Ic2, self.If1, self.If2)
            diag.append([np.asarray(v, dtype=float).reshape(-1) for v in values])
        self.last_pred_r = np.array([d[0] for d in diag])
        self.last_pred_rdot = np.array([d[1] for d in diag])
        self.last_pred_force_nominal = np.array([d[2] for d in diag])
        self.last_pred_force = np.array([d[3] for d in diag])
        self.last_pred_force_error = np.array([d[4] for d in diag])
        self.last_pred_position_error_x = np.array([d[5] for d in diag])
        self.last_pred_eps_x = np.array([d[6] for d in diag])
        self.last_pred_epsd_x = np.array([d[7] for d in diag])
        self.last_pred_u = wu.copy()
        self._warm = (wx[:self.cfg.N], wu)
        return self.solver.get(0, "u")


if __name__ == "__main__":
    print("nmpcc_cart 模块自检：")
    print("  acados 可用:", _HAS_ACADOS)
    print("  NP =", NP, " NX =", NX, " NU =", NU)
    sym = build_ocp_symbolics(3125.0)
    print("  f_expl 维度:", sym["f_expl"].shape)