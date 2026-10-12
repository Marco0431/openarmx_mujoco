#!/usr/bin/env python3
"""手持抓取位姿 NMPCC 控制器。

状态是双臂关节位置和速度，输入是 14 个关节力矩；FK 同时计算跟踪 tip
和实际 pad 接触面。每个控制周期用测量接触负载锚定预测起点，预测区间内
冻结该负载残差，并使用独立的过程权重 Q 和终端权重 P。

参数槽位保持兼容：176 个基础参数、12 个力偏置、运行期 kf_scale、左右力
参考导数，以及内部追加的 24 个 eta。力误差使用 kf_scale 直接缩放；支持
12 维接触力锚定、首阶段力矩增量边界和运行期权重更新。该控制器使用抓取
位姿的完整预测接触表达式，保留平滑接触模型的近似限制。
"""
from __future__ import annotations

import os
import hashlib
import json
import time
import warnings
from pathlib import Path

import casadi as ca
import numpy as np

# acados_template 在 JSON 回读（rebuild_on_failure 路径）时对只读/派生属性
# （tol、qp_tol、Jbu/Jbx 等）打 "Attribute X not in dictionary" 警告；
# 这些属性本就不序列化到 JSON，属上游无害噪音，这里定向屏蔽。
warnings.filterwarnings("ignore", message=r"Attribute .* not in dictionary")

# Windows 下 acados 基础 DLL 与生成求解器 DLL 需要能找到同目录依赖。
_DLL_DIR_HANDLES = []
if os.name == "nt":
    _acados_bin_candidates = [
        Path(os.environ.get("ACADOS_SOURCE_DIR", "")) / "bin",
        Path(r"D:\acados\bin"),
    ]
    for _dll_dir in _acados_bin_candidates:
        if _dll_dir.is_dir():
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(_dll_dir)))
            except (AttributeError, OSError):
                pass
            os.environ["PATH"] = str(_dll_dir) + os.pathsep + os.environ.get("PATH", "")
            break

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    if os.name == "nt":
        # acados_template 的默认 winmode 会把 D:\\acados\\bin 排除在依赖搜索外。
        AcadosOcpSolver.winmode = None
    _HAS_ACADOS = True
except ImportError:
    _HAS_ACADOS = False

NT = 6
NU = 14
NQ = 7
NX = 4 * NQ
N_FORCE_BIAS = 2 * NT
NP_BASE = 176
KF_PARAM_IDX = NP_BASE + N_FORCE_BIAS  # 保留 12 个力偏置槽位
NP_LEGACY = KF_PARAM_IDX + 1
FREF_DOT_IDX = NP_LEGACY
NP = FREF_DOT_IDX + 2 * NT  # 与另一套控制器独立的参数宽度
MODEL_VERSION = 'grasp_v6'
Q_STATE_IDX = np.r_[0:7, 14:21]



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

def _rotvec_of_R_sx(R):
    tr = (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0
    theta = ca.acos(ca.fmax(ca.fmin(tr, 1.0), -1.0))
    vee = ca.vertcat(R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1])
    return ca.if_else(theta < 1e-6, 0.5 * vee,
                      0.5 * theta / ca.sin(theta) * vee)


def build_ocp_symbolics(kx: float = 3750.0, rest_gap: float = 0.248,
                        smooth_eps: float = 0.002):
    """构造 OCP 使用的参数化关节动力学近似。

    ``M/G/fJ/D`` 是每次控制更新从 MuJoCo 读取并冻结的参数。弹簧/接触项在
    ``NMPCCCart`` 中组装，此时可使用 FK 雅可比。
    """
    del kx, rest_gap, smooth_eps  # 保留参数以兼容既有接口
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    p = ca.SX.sym("p", NP)
    qL = x[:NQ]
    vL = x[NQ:2 * NQ]
    qR = x[2 * NQ:3 * NQ]
    vR = x[3 * NQ:]

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

    Ic1 = ca.SX.sym("Ic1", NT)
    Ic2 = ca.SX.sym("Ic2", NT)
    If1 = ca.SX.sym("If1", NT)
    If2 = ca.SX.sym("If2", NT)
    return {"x": x, "u": u, "p": p, "f_expl": f_expl,
            "Ic": [Ic1, Ic2], "If": [If1, If2]}


def _quat2mat_sx(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                   2 * (x * z + w * y)),
        ca.horzcat(2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                   2 * (y * z - w * x)),
        ca.horzcat(2 * (x * z - w * y), 2 * (y * z + w * x),
                   1 - 2 * (x * x + y * y)))


def _rot_axis(axis, q):
    k = axis / ca.norm_2(axis)
    K = ca.vertcat(
        ca.horzcat(0.0, -k[2], k[1]),
        ca.horzcat(k[2], 0.0, -k[0]),
        ca.horzcat(-k[1], k[0], 0.0))
    return ca.SX.eye(3) + ca.sin(q) * K + (1 - ca.cos(q)) * (K @ K)


def pad_contact_offsets(model, arms, ee_tip_z: float):
    """返回 link7 坐标中 pad +z 接触面中心相对跟踪 tip 的偏移。"""
    offsets = []
    for arm in arms:
        gid = int(arm['pad_id'])
        rotation = np.asarray(_quat2mat_sx(ca.DM(model.geom_quat[gid])), dtype=float)
        face = model.geom_pos[gid] + rotation @ np.array([0., 0., model.geom_size[gid, 2]])
        offsets.append(face - np.array([0., 0., ee_tip_z]))
    return offsets


def mujoco_fk_symbols(model, arm_cfgs, ee_tip_z: float, contact_offsets=None):
    """生成跟踪 tip 与接触面 FK，采用平面箱体约定。"""
    def collect_chain(body_id):
        chain_ids = []
        b = int(body_id)
        while b != 0 and len(chain_ids) < 12:
            chain_ids.append(b)
            b = int(model.body_parentid[b])
        chain_ids.reverse()
        out = []
        for bid in chain_ids:
            jt = int(model.body_jntadr[bid])
            has_joint = jt >= 0
            axis = model.jnt_axis[jt].copy() if has_joint else np.zeros(3)
            axis /= np.linalg.norm(axis) + 1e-12
            out.append({"pos": model.body_pos[bid].copy(),
                        "quat": model.body_quat[bid].copy(),
                        "axis": axis, "jointed": has_joint})
        return out

    def chain_H(chain, q):
        H = ca.SX.eye(4)
        qi = 0
        for item in chain:
            Rb = _quat2mat_sx(ca.SX(item["quat"]))
            Rj = (_rot_axis(ca.SX(item["axis"]), q[qi])
                  if item["jointed"] else ca.SX.eye(3))
            if item["jointed"]:
                qi += 1
            H = H @ ca.vertcat(
                ca.horzcat(Rb @ Rj, ca.reshape(ca.SX(item["pos"]), 3, 1)),
                ca.horzcat(ca.SX.zeros(1, 3), ca.SX.ones(1, 1)))
        return H

    def tip_of(chain, q):
        H = chain_H(chain, q)
        pos = (H @ ca.vertcat(0.0, 0.0, ee_tip_z, 1.0))[:3]
        return ca.vertcat(pos, _rotvec_of_R_sx(H[:3, :3]))

    chainL = collect_chain(arm_cfgs[0]["body_id"])
    chainR = collect_chain(arm_cfgs[1]["body_id"])
    nq = sum(1 for item in chainL if item["jointed"])
    qL = ca.SX.sym("qL", nq)
    qR = ca.SX.sym("qR", nq)
    tips = ca.vertcat(tip_of(chainL, qL), tip_of(chainR, qR))
    if contact_offsets is None:
        contact_offsets = (np.zeros(3), np.zeros(3))
    if len(contact_offsets) != 2:
        raise ValueError('contact_offsets must contain two 3D offsets')
    contact_points = []
    for chain, q, offset in zip((chainL, chainR), (qL, qR), contact_offsets):
        H = chain_H(chain, q)
        local = np.array([0., 0., ee_tip_z]) + np.asarray(offset).reshape(3)
        contact_points.append((H @ ca.vertcat(ca.SX(local), 1.))[:3])
    contacts = ca.vertcat(*contact_points)
    return {"tips_qt": ca.vertcat(qL, qR),
            "tips": tips,
            "tips_fn": ca.Function("fk_tips_grasp_pose", [qL, qR], [tips]),
            "contact_tips": contacts,
            "contact_tips_fn": ca.Function("fk_contacts_grasp_pose", [qL, qR], [contacts]),
            "n_arm": nq}


def rotvec_of_R(R):
    """将世界坐标系 3x3 旋转矩阵转换为主值旋转向量。"""
    R = np.asarray(R, dtype=float)
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-6:
        return 0.5 * vee
    return 0.5 * theta / np.sin(theta) * vee


class NMPCCCartConfig:
    def __init__(self, N=50, Tp=0.1, eta_cop=None, eta_f=None,
                 Q=1.0, eta_dev=80.0, eta_u=0.05,
                 P=2.0, kx=3750.0, rest_gap=0.248, kf_scale=0.15,
                 smooth_eps=0.002, vmax=0.05, vz_max=0.002,
                 reg_method="MIRROR", lm_reg=1e-2,
                 qp_solver="PARTIAL_CONDENSING_HPIPM",
                 sim_method_num_steps=4, sim_method_num_stages=4,
                 integrator_type="ERK", qp_solver_cond_N=None,
                 qp_solver_iter_max=15, nlp_solver_max_iter=20,
                 model_name="nmpcc_cart_grasp_pose_dual",
                 fail_on_nonzero_status=True, rebuild_on_failure=True):
        if eta_cop is None:
            eta_cop = np.array([[0.0, 0.8, 0.0, 0.0, 0.0, 0.0]] * 2)
        if eta_f is None:
            eta_f = np.array([[0.8, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2)
        self.N = int(N)
        self.Tp = float(Tp)
        self.eta_cop = np.asarray(eta_cop, dtype=float).copy()
        self.eta_f = np.asarray(eta_f, dtype=float).copy()
        self.Q = float(Q)
        self.eta_dev = float(eta_dev)
        self.eta_u = float(eta_u)
        self.P = float(P)
        self.kx = float(kx)
        self.rest_gap = float(rest_gap)
        self.kf_scale = float(kf_scale)
        self.smooth_eps = float(smooth_eps)
        self.vmax = float(vmax)
        self.vz_max = float(vz_max)
        self.reg_method = reg_method
        self.lm_reg = float(lm_reg)
        self.qp_solver = qp_solver
        self.sim_method_num_steps = int(sim_method_num_steps)
        self.sim_method_num_stages = int(sim_method_num_stages)
        self.integrator_type = integrator_type
        self.qp_solver_cond_N = qp_solver_cond_N
        self.qp_solver_iter_max = int(qp_solver_iter_max)
        self.nlp_solver_max_iter = int(nlp_solver_max_iter)
        self.model_name = model_name
        self.fail_on_nonzero_status = bool(fail_on_nonzero_status)
        self.rebuild_on_failure = bool(rebuild_on_failure)


class NMPCCCart:
    """每次更新锚定测量六维接触力的 NMPCC。"""
    def __init__(self, cfg: NMPCCCartConfig, fk: dict, umin=None, umax=None,
                 verbose: bool = False, *, qmin=None, qmax=None):
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
        self.umin = (np.full(NU, -8.0) if umin is None
                     else np.asarray(umin, dtype=float).copy())
        self.umax = (np.full(NU, 8.0) if umax is None
                     else np.asarray(umax, dtype=float).copy())
        nt = NT
        n_arm = int(fk["n_arm"])

        def norm_eta(value, name):
            value = np.asarray(value, dtype=float)
            if value.shape == (nt,):
                return value.copy(), value.copy()
            if value.shape == (2, nt):
                return value[0].copy(), value[1].copy()
            raise ValueError(f"{name} 应为 (6,) 或 (2,6)，收到 {value.shape}")

        self.Ic1, self.Ic2 = norm_eta(cfg.eta_cop, "eta_cop")
        self.If1, self.If2 = norm_eta(cfg.eta_f, "eta_f")
        sym = build_ocp_symbolics(cfg.kx, cfg.rest_gap, cfg.smooth_eps)
        x, u, p, f_expl = sym["x"], sym["u"], sym["p"], sym["f_expl"]
        q_qt = fk["tips_qt"]
        qL_s, qR_s = q_qt[:n_arm], q_qt[n_arm:]
        x_qL = x[:n_arm]
        x_qR = x[2 * n_arm:3 * n_arm]
        tips = ca.substitute([fk["tips"]], [qL_s, qR_s], [x_qL, x_qR])[0]
        rL, rR = tips[:nt], tips[nt:2 * nt]
        pL, pR = rL[:3], rR[:3]

        contact_tips = fk.get('contact_tips', ca.vertcat(fk['tips'][:3], fk['tips'][nt:nt+3]))
        contact_tips = ca.substitute([contact_tips], [qL_s, qR_s], [x_qL, x_qR])[0]
        cL, cR = contact_tips[:3], contact_tips[3:6]

        kx = cfg.kx
        kf_scale = p[KF_PARAM_IDX]
        d = cR[0] - cL[0]
        compression = cfg.rest_gap - d
        compression_s = 0.5 * (
            compression + ca.sqrt(compression * compression + cfg.smooth_eps ** 2))
        f1x = -kx * compression_s
        f1mod = ca.vertcat(f1x, ca.SX.zeros(2))
        f2mod = -f1mod
        f1m6 = ca.vertcat(f1mod, ca.SX.zeros(3))
        f2m6 = ca.vertcat(f2mod, ca.SX.zeros(3))

        # 每拍用左右六维接触测量修正名义值；这些偏置在预测期间保持不变。
        force_bias1 = p[176:182]
        force_bias2 = p[182:188]
        f1pred = f1m6 + force_bias1
        f2pred = f2m6 + force_bias2
        # 附加载荷已包含接触残差；这里只把预测法向力映射到动力学，避免重复计入。
        f1dyn = ca.vertcat(f1pred[0], ca.SX.zeros(2))
        f2dyn = ca.vertcat(f2pred[0], ca.SX.zeros(2))

        dcdx = ca.jacobian(compression_s, x)
        cd = (dcdx[0, :NQ] @ x[NQ:2 * NQ]
              + dcdx[0, 2 * NQ:3 * NQ] @ x[3 * NQ:])
        fd1x = -kx * cd
        fdot1 = ca.vertcat(fd1x, 0.0, 0.0, 0.0, 0.0, 0.0)
        fdot2 = -fdot1

        Jp1q = ca.jacobian(cL, x_qL)
        Jp2q = ca.jacobian(cR, x_qR)
        M1 = ca.reshape(p[0:49], 7, 7)
        M2 = ca.reshape(p[56:105], 7, 7)
        # 保留采样得到的附加关节负载，并只加入一次预测法向负载。
        fJ1 = p[112:119]
        fJ2 = p[119:126]
        dd1 = f_expl[NQ:2 * NQ] + ca.solve(M1, Jp1q.T @ f1dyn)
        dd2 = f_expl[3 * NQ:] + ca.solve(M2, Jp2q.T @ f2dyn)
        f_expl_full = ca.vertcat(x[NQ:2 * NQ], dd1,
                                 x[3 * NQ:], dd2)
        rdot = ca.jacobian(tips, x) @ f_expl_full
        rdL, rdR = rdot[:nt], rdot[nt:2 * nt]

        Ic1, Ic2 = sym["Ic"]
        If1, If2 = sym["If"]
        r1ref, r2ref = p[126:132], p[132:138]
        v1ref, v2ref = p[138:144], p[144:150]
        f1ref, f2ref = p[150:156], p[156:162]
        fdref1 = p[FREF_DOT_IDX:FREF_DOT_IDX + NT]
        fdref2 = p[FREF_DOT_IDX + NT:FREF_DOT_IDX + 2 * NT]
        # 先在绝对位姿参考与另一臂状态之间混合，再加入经过力增益缩放的力误差。
        eps1 = ((1 - If1) *
                (rL - (1 - Ic1) * r1ref - Ic1 * rR)
                - If1 * kf_scale * (f1pred - f1ref))
        eps2 = ((1 - If2) *
                (rR - (1 - Ic2) * r2ref - Ic2 * rL)
                - If2 * kf_scale * (f2pred - f2ref))
        # 导数项同时扣除参考速度与参考力变化率，避免移动或夹紧时出现不一致目标。
        epsd1 = ((1 - If1) *
                 (rdL - (1 - Ic1) * v1ref - Ic1 * rdR)
                 - If1 * kf_scale * (fdot1 - fdref1))
        epsd2 = ((1 - If2) *
                 (rdR - (1 - Ic2) * v2ref - Ic2 * rdL)
                 - If2 * kf_scale * (fdot2 - fdref2))
        y_expr = ca.vertcat(eps1, eps2, epsd1, epsd2, u)
        y_expr_e = ca.vertcat(eps1, eps2, epsd1, epsd2)

        self._np_full = NP + 4 * NT
        model = AcadosModel()
        model.name = _solver_identity(cfg, fk, self.umin, self.umax, self.qmin, self.qmax)
        model.x = x
        model.u = u
        model.p = ca.vertcat(p, Ic1, Ic2, If1, If2)
        model.f_expl_expr = f_expl_full
        model.cost_y_expr = y_expr
        model.cost_y_expr_e = y_expr_e

        ny = int(y_expr.shape[0])
        ne = int(y_expr_e.shape[0])
        # 阶段代价由 Q 加权，终端代价独立由 P 加权；终端不包含控制输入项。
        W, W_e = cost_weights(cfg)
        self.W_e = W_e
        self.W = W
        self.ny = ny
        self._W_rt = None
        self._We_rt = None
        self.debug_eps = False
        self._force_nom_fn = ca.Function(
            f"{cfg.model_name}_force_nominal", [x], [f1m6, f2m6])
        self._y_fn = ca.Function(
            f"{cfg.model_name}_cost_y", [x, u, model.p], [y_expr])

        self._dynamics_fn = ca.Function(
            f"{model.name}_dynamics", [x, u, p], [f_expl_full])
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
        # 预测中间节点和终端使用同一物理关节范围；第 0 节点在 solve 中固定为实测状态。
        if self.q_bound_mask.any():
            indices = Q_STATE_IDX[self.q_bound_mask]
            lower, upper = self.qmin[self.q_bound_mask], self.qmax[self.q_bound_mask]
            ocp.constraints.idxbx = indices.copy()
            ocp.constraints.lbx = lower.copy()
            ocp.constraints.ubx = upper.copy()
            ocp.constraints.idxbx_e = indices.copy()
            ocp.constraints.lbx_e = lower.copy()
            ocp.constraints.ubx_e = upper.copy()
        # 末端速度边界约束预测节点的运动速度，不能替代每个物理步的限位观察。
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
        opts.print_level = 1 if verbose else 0

        # Windows/CasADi 对深层路径敏感，使用短哈希目录保存 codegen 结果。
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(module_dir)
        codegen_key = hashlib.sha1(model.name.encode("utf-8")).hexdigest()[:12]
        codegen_dir = os.path.join(project_dir, "acados_codegen", codegen_key)
        os.makedirs(codegen_dir, exist_ok=True)
        json_file = os.path.join(codegen_dir, f"{model.name}.json")
        ocp.code_gen_options.code_export_directory = codegen_dir
        ocp.code_gen_options.json_file = json_file
        try:
            self.solver = AcadosOcpSolver(
                ocp, json_file=json_file, build=False, generate=False, verbose=verbose)
        except (FileNotFoundError, OSError) as exc:
            print(f"[acados] cached grasp-pose solver unavailable; rebuilding: {exc}")
            self.solver = AcadosOcpSolver(
                ocp, json_file=json_file, build=True, generate=True, verbose=verbose)
        self._ocp = ocp
        self._json_file = json_file
        self.node_times = np.r_[0.0, np.cumsum(ocp.solver_options.time_steps)]
        self.first_failure = None
        self._warm = None
        self.last_status = -1
        self.last_qp_iter = -1
        self.last_time_qp = float("nan")
        self.last_time_sim = float("nan")
        self.last_time_lin = float("nan")

    def _normalize_p(self, p_coef):
        """接受单个向量或 N+1 行，并补齐缺失的兼容字段。"""
        values = np.asarray(p_coef, dtype=float)
        if values.ndim not in (1, 2):
            raise ValueError("p_coef must be a vector or an (N+1, NP) array")
        size = values.shape[-1]
        if size not in (NP_BASE, NP_BASE + N_FORCE_BIAS, NP_LEGACY, NP):
            raise ValueError(f"parameter width must be 176, 188, 189 or {NP}, got {size}")
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
        current = p_coef if p_coef.ndim == 1 else p_coef[0]
        measured = (nominal + current[NP_BASE:NP_BASE+N_FORCE_BIAS] if f_meas is None
                    else np.asarray(f_meas, float).reshape(-1))
        if measured.size < N_FORCE_BIAS:
            raise ValueError(f"f_meas 应包含左右世界系六维力/接触力矩，收到 {measured.size} 个数")
        bias = measured[:N_FORCE_BIAS] - nominal
        if not np.isfinite(measured[:N_FORCE_BIAS]).all():
            raise ValueError("f_meas contains non-finite values")
        p_coef[..., NP_BASE:NP_BASE + N_FORCE_BIAS] = bias
        self.last_force_nominal = nominal.copy()
        self.last_force_measured = measured[:N_FORCE_BIAS].copy()
        self.last_force_bias = bias.copy()
        self.last_force_anchored = nominal + bias
        return p_coef

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

    def _set_initial_guess(self, x0, u_seed):
        """兼容别名；包含终端状态 N。"""
        self._initialize_iterate(x0, u_seed)

    def _set_parameters(self, p_full, yref):
        for i in range(self.cfg.N):
            self.solver.set(i, "p", p_full)
            self.solver.set(i, "yref", yref)
        self.solver.set(self.cfg.N, "p", p_full)
        self.solver.set(self.cfg.N, "yref", yref[:self.ny - NU])

    def set_If(self, if_l, if_r=None):
        """运行时覆盖 eta_f 的 If1/If2（p 参数，无需重建求解器）。"""
        self.If1 = np.asarray(if_l, dtype=float).reshape(-1).copy()
        self.If2 = (self.If1.copy() if if_r is None
                    else np.asarray(if_r, dtype=float).reshape(-1).copy())

    def set_runtime_weights(self, Q=None, eta_dev=None, eta_u=None, P=None,
                            ed_x=None):
        """运行时覆盖过程/终端权重 W/W_e（cost_set，不重建求解器）。

        只重设已有块的数值：Q（残差）、eta_dev（残差导数）、eta_u（输入）、
        P（终端）、ed_x（仅 epsd 分量的 x 轴阻尼通道：前向压缩弛豫率），
        结构/对象与构建时完全一致。
        """
        Q = self.cfg.Q if Q is None else float(Q)
        eta_dev = self.cfg.eta_dev if eta_dev is None else float(eta_dev)
        eta_u = self.cfg.eta_u if eta_u is None else float(eta_u)
        P = self.cfg.P if P is None else float(P)
        ed_x = eta_dev if ed_x is None else float(ed_x)
        W = np.zeros((self.ny, self.ny))
        W[:2 * NT, :2 * NT] = Q * np.eye(2 * NT)
        W[2 * NT:4 * NT, 2 * NT:4 * NT] = Q * eta_dev * np.eye(2 * NT)
        # epsd 排列：[epsd1(6), epsd2(6)]，x 分量位于 12 与 18
        W[2 * NT, 2 * NT] = Q * ed_x
        W[3 * NT, 3 * NT] = Q * ed_x
        W[4 * NT:, 4 * NT:] = Q * eta_u * np.eye(NU)
        self._W_rt = W
        terminal_base = np.eye(self.ny - NU)
        terminal_base[2*NT:, 2*NT:] *= eta_dev
        terminal_base[2*NT, 2*NT] = ed_x
        terminal_base[3*NT, 3*NT] = ed_x
        self._We_rt = P * terminal_base
        return self

    def _apply_rt_W(self):
        if self._W_rt is None:
            return
        try:
            for i in range(self.cfg.N):
                self.solver.cost_set(i, "W", self._W_rt)
            # 终端代价在 acados C 接口中也使用 "W" 字段（stage N）。
            self.solver.cost_set(self.cfg.N, "W", self._We_rt)
        except Exception as exc:
            raise RuntimeError("runtime cost weights could not be applied") from exc

    def _apply_u_lim(self, u_lim=None, u_prev=None, du_max=None):
        """运行时输入盒约束：u ∈ [max(umin, -u_lim), min(umax, u_lim)]，全 horizon 硬限幅。

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
        if u_prev is not None and du_max is not None:
            center = np.asarray(u_prev, dtype=float).reshape(NU)
            delta = np.broadcast_to(np.asarray(du_max, dtype=float), (NU,))
            lo, hi = np.maximum(base_lo, center-delta), np.minimum(base_hi, center+delta)
            if not np.isfinite(center).all() or not np.isfinite(delta).all() or np.any(delta < 0) or np.any(lo > hi):
                raise ValueError('invalid first-stage torque increment bounds')
            self.solver.constraints_set(0, 'lbu', lo)
            self.solver.constraints_set(0, 'ubu', hi)
        self._apply_rt_W()


    def contact_residual_parameters(self, x0, p, full_contact, f_meas):
        """用手持抓取位姿的完整预测表达式锚定当前总接触负载。

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

    def solve(self, x0, p_coef, u_ref=None, f_meas=None, u_seed=None,
              u_prev=None, du_max=None, u_lim=None):
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
        self._apply_u_lim(u_lim, u_prev, du_max)
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
            self._apply_u_lim(u_lim, u_prev, du_max)   # 重建对象会回落物理界，需重贴 u_lim
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
        if self.debug_eps:
            y0 = np.asarray(self._y_fn(x0, wu[0], p_full[0])).ravel()
            print(f"[epsdbg] status={status} eps1={y0[:3]} eps2={y0[6:9]} "
                  f"epsd1x={y0[12]:+.5f} epsd2x={y0[18]:+.5f}")
        return self.solver.get(0, "u")


__all__ = [
    "NMPCCCart", "NMPCCCartConfig", "build_ocp_symbolics",
    "mujoco_fk_symbols", "rotvec_of_R", "NT", "NU", "NQ", "NX",
    "NP", "N_FORCE_BIAS", "KF_PARAM_IDX", "FREF_DOT_IDX", "MODEL_VERSION",
]


if __name__ == "__main__":
    print("nmpcc_cart_grasp_pose 模块自检：")
    print("  acados 可用:", _HAS_ACADOS)
    print("  NP =", NP, "NX =", NX, "NU =", NU)
    print("  f_expl 维度:", build_ocp_symbolics()["f_expl"].shape)
