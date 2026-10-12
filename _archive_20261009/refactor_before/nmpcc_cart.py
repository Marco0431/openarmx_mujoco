#!/usr/bin/env python3
"""双臂搬箱 NMPCC 控制器。

状态为 [qL(7), vL(7), qR(7), vR(7)]，输入为 14 个关节力矩。每次控制更新
采样 M/G 和附加广义负载，并在预测区间内保持不变；默认入口传入完整实测
接触负载减当前模型法向负载的残差。模型不含科氏项，包含关节
粘性阻尼。法向接触力由平滑压缩弹簧和本次求解共享的测量偏置组成，不额外
施加力前馈。

笛卡尔位姿为 [位置, 旋转向量]，速度由解析 FK 雅可比乘关节速度得到。混合
残差包含跟踪、协作和缩放后的力误差，导数包含位置参考速度与力参考导数。
Q/P 分别作用于过程和终端代价，终端残差不含力矩。预测采用关节空间的近似动力学，不把笛卡尔动力学展开成完整坐标变换。

参数布局（索引必须保持兼容）：
  M1(49), G1(7), M2(49), G2(7), tangential_fJ1(7), tangential_fJ2(7),
  r1ref(6), r2ref(6), v1ref(6), v2ref(6), f1ref(6), f2ref(6), D1(7), D2(7),
  bFL(1), bFR(1), kf_scale(1), f1ref_dot(6), f2ref_dot(6).
solve 接受单个向量（广播）或 N+1 行。兼容宽度 176/178/179 会补零力参考导数，
缺少时补入配置中的 kf。内部另追加 24 个 eta 参数；求解器缓存按编译设置和
运动学指纹区分。
"""

from __future__ import annotations

import json
import hashlib
import os
import time
import warnings
import numpy as np

# acados_template 从 JSON 重建时会检查若干不会序列化的派生属性，并产生无害警告。
# 仅屏蔽这类提示，避免掩盖真正的求解器错误。
warnings.filterwarnings("ignore", message=r"Attribute .* not in dictionary")

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
# NP 的基础段按模块 docstring 排列，尾部追加左右六维参考力导数。
# bFL/bFR 每拍由测量法向力与当前名义力之差得到，并在整个预测区间冻结。
# kf_scale 是运行期参数，改变力误差增益无需重新生成代码。
N_FORCE_BIAS = 2
NP_BASE = 2 * (49 + 7) + 14 + 6 * 6 + 14
KF_PARAM_IDX = NP_BASE + N_FORCE_BIAS
NP_LEGACY = NP_BASE + N_FORCE_BIAS + 1
FREF_DOT_IDX = NP_LEGACY
NP = NP_LEGACY + 2 * NT
MODEL_VERSION = "v7"
Q_STATE_IDX = np.r_[0:NQ, 2*NQ:3*NQ]


def cost_weights(cfg):
    """返回独立的过程 Q 和终端 P；终端残差不含输入。"""
    base = np.diag(np.r_[np.ones(2 * NT),
                         np.full(2 * NT, cfg.eta_dev),
                         np.full(NU, cfg.eta_u)])
    return cfg.Q * base, cfg.P * base[:4 * NT, :4 * NT]


def _solver_identity(cfg, fk, umin=None, umax=None, qmin=None, qmax=None):
    """按编译设置和 FK 为所有调用生成缓存标识。"""
    runtime = {"kf_scale", "eta_cop", "eta_f", "fail_on_nonzero_status",
               "rebuild_on_failure"}
    settings = {k: v for k, v in vars(cfg).items() if k not in runtime}
    settings["input_bounds"] = [None if v is None else np.asarray(v).tolist()
                                for v in (umin, umax)]
    settings["joint_bounds"] = [None if v is None else np.asarray(v).tolist()
                                for v in (qmin, qmax)]
    payload = MODEL_VERSION + json.dumps(settings, sort_keys=True)
    payload += fk["tips_fn"].serialize()
    if "contact_tips_fn" in fk:
        payload += fk["contact_tips_fn"].serialize()
    return "nmpcc_" + MODEL_VERSION + "_" + hashlib.sha1(payload.encode()).hexdigest()[:12]


# ======================================================================
# 数值参数化动力学符号（预测内冻结 M/G；不展开深层符号）。
# ======================================================================
def _rotvec_of_R_sx(R):
    tr = (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0
    theta = ca.acos(ca.fmax(ca.fmin(tr, 1.0), -1.0))
    vee = ca.vertcat(R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1])
    small = theta < 1e-6
    return ca.if_else(small, 0.5 * vee, 0.5 * theta / ca.sin(theta) * vee)


def build_ocp_symbolics(kx: float = 3750.0, rest_gap: float = 0.248,
                        smooth_eps: float = 0.002):
    """构造预测用的近似关节动力学符号模型。

    质量矩阵 M、重力 G、附加关节负载和阻尼由参数提供，并在一个 horizon
    内冻结；接触法向力在接触雅可比处加入。112:126 槽位兼容旧接口，
    也可存放“完整测量负载减去当前模型法向负载”的残差。
    kx、rest_gap、smooth_eps 保留为兼容参数。
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
# 端点 FK 符号（来自 MuJoCo 模型数据；不构造 Hessian）。
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
    """返回左右 pad 朝箱体接触面中心相对 link7 tip 的局部偏移。

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
    """生成符号 FK；轨迹点为 tip，接触点用于夹持弹簧和法向力雅可比。

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
        """搬箱配置。η 映射保持各轴的协作/力控制语义：
          x 平移轴：ηf=0.8, ηcop=0（阻抗：保持夹持力）
          y 平移轴：ηcop=0.8, ηf=0（运动同步）；z 平移轴：ηcop=0（纯跟踪托重）
          姿态 αβγ：ηcop=0, ηf=0（纯跟踪）
        Q/P：独立的过程/终端标量权重；kx：法向弹簧刚度。
        kf_scale：力误差的额外缩放，直接乘力误差，不再除以 kx。
        kf_scale 为运行期参数（索引 KF_PARAM_IDX），改动不触发重新编译；cfg 值仅作
        缺失参数的默认补齐值。

        求解器设置：
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
    """NMPCC（acados, SQP_RTI）：关节空间状态与笛卡尔 ε_mult 代价。

    solve(x0, p) → u(14)
      x0  [qL(7), vL(7), qR(7), vR(7)] 当前关节状态
      p   (191,) 或 (N+1,191)，兼容旧版向量，布局见模块 docstring
    """

    def __init__(self, cfg: NMPCCCartConfig, fk: dict, umin=None, umax=None,
                 verbose: bool = False, *, qmin=None, qmax=None):
        """可选的 14 个关节角物理边界，顺序为左臂后右臂，单位为弧度。

        边界作用于节点 1..N；节点 0 仍是测量状态等式。两者均省略时保持无
        位置约束的兼容行为。(-inf, +inf) 表示该关节不设限；不做边界收紧、
        松弛代价或状态裁剪。
        """
        if not _HAS_ACADOS:
            raise RuntimeError("acados_template 不可用")
        self.cfg = cfg
        if (qmin is None) != (qmax is None):
            raise ValueError("qmin and qmax must be supplied together")
        self.qmin = self.qmax = None
        self.q_bound_mask = np.zeros(NU, dtype=bool)
        if qmin is not None:
            lo, hi = np.asarray(qmin, float), np.asarray(qmax, float)
            if lo.shape != (NU,) or hi.shape != (NU,):
                raise ValueError("qmin/qmax must have shape (14,), left then right")
            finite = np.isfinite(lo) & np.isfinite(hi)
            unlimited = np.isneginf(lo) & np.isposinf(hi)
            if not np.all(finite | unlimited) or np.any(lo >= hi):
                raise ValueError("joint bounds need lower < upper, or (-inf, +inf)")
            self.qmin, self.qmax = lo.copy(), hi.copy()
            self.q_bound_mask = finite
        # 物理执行器限幅（结构界，构建时写入 OCP；运行时 _apply_u_lim 可在其内收紧）
        self.umin = (np.full(NU, -8.0) if umin is None
                     else np.asarray(umin, dtype=float).copy())
        self.umax = (np.full(NU, 8.0) if umax is None
                     else np.asarray(umax, dtype=float).copy())
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

        # 夹持弹簧的法向力：左 pad 位于 -x，右 pad 位于 +x。
        kx_uni = cfg.kx
        # kf_scale 为运行期参数（索引 KF_PARAM_IDX），调力增益不触发重新 codegen/编译。
        kf_scale = p[KF_PARAM_IDX]
        # 物理编号固定为左侧 -x、右侧 +x。
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
               + ca.solve(M1, Jp1q.T @ f1dyn))
        dd2 = (sym["f_expl"][3 * NQ:]
               + ca.solve(M2, Jp2q.T @ f2dyn))
        f_expl_full = ca.vertcat(x[NQ:2 * NQ], dd1, x[3 * NQ:], dd2)
        # fJ 是冻结的附加关节载荷；残差模式下它表示测量负载减模型法向负载。
        # 预测法向力随后单独加入动力学，避免重复计入。
        # FK 仅依赖 q，所以此处 r_dot = J(q) q_dot（姿态为 rotvec 的解析导数）。
        r_dot = ca.jacobian(tips, x) @ f_expl_full
        rdL, rdR = r_dot[:nt], r_dot[nt:2 * nt]

        Ic1, Ic2 = sym["Ic"]; If1, If2 = sym["If"]
        r1ref = p[126:132]; r2ref = p[132:138]
        v1ref = p[138:144]; v2ref = p[144:150]
        f1ref = p[150:156]; f2ref = p[156:162]
        fdref1 = p[FREF_DOT_IDX:FREF_DOT_IDX + NT]
        fdref2 = p[FREF_DOT_IDX + NT:FREF_DOT_IDX + 2 * NT]
        # 混合残差：位姿、协作/力误差和输入分别进入代价。
        eps1 = (1 - If1) * (rL - (1 - Ic1) * r1ref - Ic1 * rR) - If1 * kf_scale * (f1pred - f1ref) 
        eps2 = (1 - If2) * (rR - (1 - Ic2) * r2ref - Ic2 * rL) - If2 * kf_scale * (f2pred - f2ref) 
        # 残差导数由 FK 速度和参考速度组成；弹簧力导数显式包含在内。
        epsd1 = (1 - If1) * (rdL - (1 - Ic1) * v1ref - Ic1 * rdR) - If1 * kf_scale * (fdot1 - fdref1)
        epsd2 = (1 - If2) * (rdR - (1 - Ic2) * v2ref - Ic2 * rdL) - If2 * kf_scale * (fdot2 - fdref2)
        y_expr = ca.vertcat(eps1, eps2, epsd1, epsd2, u)
        y_expr_e = ca.vertcat(eps1, eps2, epsd1, epsd2)

        # 参数拼接：p(191) + eta(24) = 215。
        self._np_full = NP + 4 * NT
        model = AcadosModel()
        model.name = _solver_identity(cfg, fk, self.umin, self.umax, self.qmin, self.qmax)
        model.x = x
        model.u = u
        model.p = ca.vertcat(p, Ic1, Ic2, If1, If2)
        model.f_expl_expr = f_expl_full
        model.cost_y_expr = y_expr
        model.cost_y_expr_e = y_expr_e

        ny = y_expr.shape[0]
        ne = y_expr_e.shape[0]
        W, W_e = cost_weights(cfg)
        # Q/P 分别作用于过程/终端；eta_dev/eta_u 是各自括号内的相对权重。
        input_offset = 4 * nt
        self.W = W
        self.W_e = W_e
        self.ny = ny
        self.input_offset = input_offset
        self._dynamics_fn = ca.Function(
            f"{model.name}_dynamics", [x, u, p], [f_expl_full])
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
        ocp.solver_options.N_horizon = cfg.N
        ocp.solver_options.tf = cfg.Tp
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = W
        ocp.cost.W_e = W_e
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ne)

        ocp.constraints.idxbu = np.arange(NU)
        ocp.constraints.lbu = self.umin
        ocp.constraints.ubu = self.umax
        if self.q_bound_mask.any():
            indices = Q_STATE_IDX[self.q_bound_mask]
            lower, upper = self.qmin[self.q_bound_mask], self.qmax[self.q_bound_mask]
            ocp.constraints.idxbx = indices.copy()
            ocp.constraints.lbx = lower.copy()
            ocp.constraints.ubx = upper.copy()
            ocp.constraints.idxbx_e = indices.copy()
            ocp.constraints.lbx_e = lower.copy()
            ocp.constraints.ubx_e = upper.copy()
        # 约束末端平移速度，并单独限制 z 方向速度。
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
        # Windows/CasADi 对深层路径较敏感；版本化指纹同时用于短模型名和目录。
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(module_dir)
        codegen_key = hashlib.sha1(model.name.encode("utf-8")).hexdigest()[:12]
        codegen_dir = os.path.join(project_dir, "acados_codegen", codegen_key)
        os.makedirs(codegen_dir, exist_ok=True)
        json_file = os.path.join(codegen_dir, f"{model.name}.json")
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
        self.node_times = np.r_[0.0, np.cumsum(ocp.solver_options.time_steps)]

    def _normalize_p(self, p_coef):
        """接受单个向量或 N+1 行，并补齐缺失的兼容字段。"""
        values = np.asarray(p_coef, dtype=float)
        if values.ndim not in (1, 2):
            raise ValueError("p_coef must be a vector or an (N+1, NP) array")
        size = values.shape[-1]
        if size not in (NP_BASE, NP_BASE + N_FORCE_BIAS, NP_LEGACY, NP):
            raise ValueError(f"parameter width must be 176, 178, 179 or {NP}, got {size}")
        if values.ndim == 2 and values.shape[0] != self.cfg.N + 1:
            raise ValueError(f"expected {self.cfg.N + 1} parameter rows")
        result = np.zeros((*values.shape[:-1], NP))
        result[..., :size] = values
        if size <= KF_PARAM_IDX:
            result[..., KF_PARAM_IDX] = self.cfg.kf_scale
        if not np.isfinite(result).all():
            raise ValueError("p_coef contains non-finite values")
        return result

    def _full_p(self, p_coef):
        """向单个归一化向量或节点行追加 eta。"""
        values = self._normalize_p(p_coef)
        eta = np.concatenate([self.Ic1, self.Ic2, self.If1, self.If2])
        eta = np.broadcast_to(eta, (*values.shape[:-1], eta.size))
        return np.concatenate([values, eta], axis=-1)

    def _anchor_measured_force(self, x0, p_coef, f_meas):
        p_coef = self._normalize_p(p_coef)
        nominal = np.asarray(self._force_nom_fn(x0), dtype=float).reshape(-1)
        measured = nominal.copy() if f_meas is None else np.asarray(f_meas, float).reshape(-1)
        if measured.size < N_FORCE_BIAS:
            raise ValueError(f"f_meas 应包含左右带符号法向力，收到 {measured.size} 个数")
        bias = measured[:N_FORCE_BIAS] - nominal
        if not np.isfinite(measured[:N_FORCE_BIAS]).all():
            raise ValueError("f_meas contains non-finite values")
        p_coef[..., NP_BASE:NP_BASE + N_FORCE_BIAS] = bias
        self.last_force_nominal = nominal.copy()
        self.last_force_measured = measured[:N_FORCE_BIAS].copy()
        self.last_force_bias = bias.copy()
        self.last_force_anchored = nominal + bias
        return p_coef

    def _apply_u_lim(self, u_lim=None):
        """运行时输入盒约束：u ∈ [max(umin, -u_lim), min(umax, u_lim)]，全预测区间硬限幅。

        u_lim=None 恢复物理界 self.umin/self.umax（与构建时 ocp.constraints.lbu/ubu
        一致）。只改 lbu/ubu 数值（idxbu 结构不变），acados 运行期 constraints_set
        不触发重新 codegen/编译——u_lim 扫描免编译的关键。
        """
        base_lo = np.asarray(self.umin, dtype=float)
        base_hi = np.asarray(self.umax, dtype=float)
        if u_lim is not None:
            lim = np.broadcast_to(np.asarray(u_lim, dtype=float), (NU,))
            base_lo = np.maximum(base_lo, -lim)
            base_hi = np.minimum(base_hi, lim)
        for i in range(self.cfg.N):
            self.solver.constraints_set(i, "lbu", base_lo)
            self.solver.constraints_set(i, "ubu", base_hi)

    def contact_residual_parameters(self, x0, p, full_contact, f_meas):
        """锚定当前总接触负载，不改变现有预测表达式。

        用完整测量接触负载减去模型当前法向关节负载，替换 112:126 槽位，并
        在每个节点冻结该差值；未来法向力仍随状态变化。这是预测参数，不是
        额外电机指令，也不修改输入和求解器迭代量。
        """
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        measured = np.asarray(full_contact, dtype=float).reshape(-1)
        if x0.shape != (NX,) or not np.isfinite(x0).all():
            raise ValueError(f"contact residual requires a finite {NX}-state vector")
        if measured.shape != (NU,) or not np.isfinite(measured).all():
            raise ValueError(f"contact residual requires {NU} finite joint loads")
        anchored = self._anchor_measured_force(x0, p, f_meas)
        current = (anchored if anchored.ndim == 1 else anchored[0]).copy()
        original_load = current[112:126].copy()
        current[112:126] = 0.0
        derivative = np.asarray(self._dynamics_fn(x0, np.zeros(NU), current)).ravel()
        ml, mr = current[:49].reshape(NQ, NQ, order="F"), current[56:105].reshape(NQ, NQ, order="F")
        normal = (np.r_[ml @ derivative[NQ:2*NQ], mr @ derivative[3*NQ:4*NQ]]
                  + np.r_[current[49:56], current[105:112]]
                  + current[162:176] * x0[np.r_[NQ:2*NQ, 3*NQ:4*NQ]])
        residual = measured - normal
        if not np.isfinite(residual).all():
            raise ValueError("contact residual contains non-finite values")
        anchored[..., 112:126] = residual
        self.last_contact_load_snapshot = dict(
            full_contact=measured.copy(), model_normal=normal,
            tangent_contact=original_load, frozen_residual=residual,
            correction=residual-original_load)
        return anchored

    def equilibrium_seed(self, x0, p, f_meas=None):
        """返回模型平衡力矩初值，不是保持真实系统的指令。

        使用与 solve 相同的参数归一化和测量力锚定；节点数组只用当前节点初值。
        该初值抵消包含冻结接触负载和阻尼在内的模型加速度，不改变测量状态或
        求解器迭代量。调用方应在使用前检查有效力矩边界。
        """
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.shape != (NX,) or not np.isfinite(x0).all():
            raise ValueError(f"equilibrium_seed requires a finite {NX}-state vector")
        parameters = self._anchor_measured_force(x0, p, f_meas)
        current = parameters if parameters.ndim == 1 else parameters[0]
        derivative = np.asarray(self._dynamics_fn(x0, np.zeros(NU), current)).reshape(-1)
        # CasADi reshape 使用列主序；MuJoCo 质量矩阵对称。
        ml = current[:49].reshape(NQ, NQ, order="F")
        mr = current[56:105].reshape(NQ, NQ, order="F")
        seed = -np.r_[ml @ derivative[NQ:2*NQ], mr @ derivative[3*NQ:4*NQ]]
        if not np.isfinite(seed).all():
            raise ValueError("model-equilibrium torque seed contains non-finite values")
        return seed

    def _initialize_iterate(self, x0, u_seed):
        """初始化所有状态节点（含 N），不移动预测区间。"""
        if getattr(self, "_warm", None) is not None:
            states, controls = self._warm
            for i in range(self.cfg.N + 1):
                self.solver.set(i, "x", states[i])
            for i in range(self.cfg.N):
                self.solver.set(i, "u", controls[i])
        else:
            # 冷启动时填充 N+1 个状态节点，避免终端节点保留无关的零状态。
            for i in range(self.cfg.N + 1):
                self.solver.set(i, "x", x0)
            if u_seed is not None:
                seed = np.asarray(u_seed, dtype=float).reshape(-1)[:NU]
                for i in range(self.cfg.N):
                    self.solver.set(i, "u", seed)

    def solve(self, x0, p_coef, u_ref=None, f_meas=None, u_seed=None,
              u_lim=None):
        """使用共享参数向量或 N+1 行节点参数求解。

        测量法向力偏置只锚定一次，并由所有节点共享。u_ref 为兼容参数；输入
        代价的参考值为零。
        """
        x0 = np.asarray(x0, float).reshape(-1)[:NX]
        p_coef = self._anchor_measured_force(x0, p_coef, f_meas)
        if p_coef.ndim == 1:
            p_coef = np.broadcast_to(p_coef, (self.cfg.N + 1, NP)).copy()
        self.last_parameters = p_coef.copy()
        p_full = self._full_p(p_coef)
        yref = np.zeros(self.ny)
        # 输入代价以零为参考；u_ref 仅保留为兼容参数，不写入 yref。
        self._initialize_iterate(x0, u_seed)
        for i in range(self.cfg.N):
            self.solver.set(i, "p", p_full[i])
            self.solver.set(i, "yref", yref)
        self.solver.set(self.cfg.N, "p", p_full[self.cfg.N])
        self.solver.set(self.cfg.N, "yref", yref[:self.ny - NU])
        self._apply_u_lim(u_lim)
        solver_started = time.perf_counter()
        try:
            self.solver.solve_for_x0(
                x0_bar=x0,
                fail_on_nonzero_status=self.cfg.fail_on_nonzero_status,
                print_stats_on_failure=True)
            status = int(self.solver.status)
        except RuntimeError as e:
            status = 4  # ACADOS_SOLVER_FAILURE（兼容旧路径：fail_on_nonzero_status=True 时抛出）
        self.last_initial_status = status
        if status != 0 and getattr(self, "first_failure", None) is None:
            self.first_failure = dict(
                status=status, x0=x0.copy(), p_full=p_full.copy(),
                x=np.array([self.solver.get(i, "x") for i in range(self.cfg.N + 1)]),
                u=np.array([self.solver.get(i, "u") for i in range(self.cfg.N)]))
        if status != 0 and self.cfg.rebuild_on_failure:
            self.solver = AcadosOcpSolver(self._ocp, json_file=self._json_file,
                                          build=True, generate=True, verbose=False)
            self._initialize_iterate(x0, u_seed)
            for i in range(self.cfg.N):
                self.solver.set(i, "p", p_full[i])
                self.solver.set(i, "yref", yref)
            self.solver.set(self.cfg.N, "p", p_full[self.cfg.N])
            self.solver.set(self.cfg.N, "yref", yref[:self.ny - NU])
            self._apply_u_lim(u_lim)   # 重建对象会回落物理界，需重贴 u_lim
            status = self.solver.solve_for_x0(
                x0_bar=x0,
                fail_on_nonzero_status=self.cfg.fail_on_nonzero_status,
                print_stats_on_failure=True)
            status = int(self.solver.status)
        self.last_solve_wall_ms = (time.perf_counter() - solver_started) * 1000.0
        self.last_solver_ms = float(np.asarray(
            self.solver.get_stats("time_tot")).reshape(-1)[0]) * 1000.0
        # 保存本拍的求解状态和耗时，便于运行时诊断。
        self.last_status = int(status)
        try:
            self.last_qp_iter = int(np.asarray(self.solver.get_stats("qp_iter")).reshape(-1)[0])
        except Exception:
            self.last_qp_iter = -1
        try:
            self.last_time_qp = float(np.asarray(self.solver.get_stats("time_qp")).reshape(-1)[0])
        except Exception:
            self.last_time_qp = float("nan")
        try:
            self.last_time_sim = float(np.asarray(self.solver.get_stats("time_sim")).reshape(-1)[0])
        except Exception:
            self.last_time_sim = float("nan")
        try:
            self.last_time_lin = float(np.asarray(self.solver.get_stats("time_lin")).reshape(-1)[0])
        except Exception:
            self.last_time_lin = float("nan")
        # 保存 N+1 个状态和 N 个输入，下一拍优先作为 warm start。
        wx = np.array([self.solver.get(i, "x") for i in range(self.cfg.N + 1)])
        wu = np.array([self.solver.get(i, "u") for i in range(self.cfg.N)])
        self.last_pred_x = wx.copy()
        diag = []
        for i, xi in enumerate(wx):
            values = self._diagnostic_fn(
                xi, p_coef[i], self.Ic1, self.Ic2, self.If1, self.If2)
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
        self._warm = (wx.copy(), wu.copy())
        return self.solver.get(0, "u")


if __name__ == "__main__":
    print("nmpcc_cart 模块自检：")
    print("  acados 可用:", _HAS_ACADOS)
    print("  NP =", NP, " NX =", NX, " NU =", NU)
    sym = build_ocp_symbolics(3125.0)
    print("  f_expl 维度:", sym["f_expl"].shape)
