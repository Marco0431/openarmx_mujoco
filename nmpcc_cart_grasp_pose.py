#!/usr/bin/env python3
"""独立的 grasp-pose NMPCC 实验模块。

本文件是 ``nmpcc_cart.py`` 论文基线的实验副本，允许实验专用修改。
模型仍使用关节状态作为 MuJoCo 的接口，但保留论文笛卡尔误差代价；
新增的 12 个参数把每拍的两侧 6D 接触力测量锚定到预测夹持力。

参数布局（188 个 plant 参数）：
  [M1(49), G1(7), M2(49), G2(7), fJ1(7), fJ2(7),
   r1ref(6), r2ref(6), v1ref(6), v2ref(6), f1ref(6), f2ref(6),
   D1(7), D2(7), force_bias1(6), force_bias2(6)]
随后模型参数追加 Ic1/Ic2/If1/If2 四个 6D 合作指标，共 212 个参数。
"""
from __future__ import annotations

import os
from pathlib import Path

import casadi as ca
import numpy as np

# acados 的基础 DLL 与生成求解器 DLL 在 Windows 上必须能找到同目录依赖。
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
NP = 2 * (49 + 7) + 14 + 6 * 6 + 14 + N_FORCE_BIAS


def _rotvec_of_R_sx(R):
    tr = (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0
    theta = ca.acos(ca.fmax(ca.fmin(tr, 1.0), -1.0))
    vee = ca.vertcat(R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1])
    return ca.if_else(theta < 1e-6, 0.5 * vee,
                      0.5 * theta / ca.sin(theta) * vee)


def build_ocp_symbolics(kx: float = 3750.0, rest_gap: float = 0.248,
                        smooth_eps: float = 0.002):
    """Build the numerical-parameterized joint plant used by the OCP.

    ``M/G/fJ/D`` are frozen parameters read from MuJoCo each control update.
    The spring/contact term itself is assembled in :class:`NMPCCCart`, where
    the FK Jacobian is available.
    """
    del kx, rest_gap, smooth_eps  # retained for API compatibility
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


def mujoco_fk_symbols(model, arm_cfgs, ee_tip_z: float):
    """Create symbolic FK for the two 7-DOF arm tips."""
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
    return {"tips_qt": ca.vertcat(qL, qR),
            "tips": tips,
            "tips_fn": ca.Function("fk_tips_grasp_pose", [qL, qR], [tips]),
            "n_arm": nq}


def rotvec_of_R(R):
    """Convert a world-frame 3x3 rotation matrix to a principal rotvec."""
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
    """NMPCC with per-update measured 6D contact-force anchoring."""
    def __init__(self, cfg: NMPCCCartConfig, fk: dict, umin=None, umax=None,
                 verbose: bool = False):
        if not _HAS_ACADOS:
            raise RuntimeError("acados_template 不可用")
        self.cfg = cfg
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

        kx = cfg.kx
        kf_scale = cfg.kf_scale
        d = pR[0] - pL[0]
        compression = cfg.rest_gap - d
        compression_s = 0.5 * (
            compression + ca.sqrt(compression * compression + cfg.smooth_eps ** 2))
        f1x = -kx * compression_s
        f1mod = ca.vertcat(f1x, ca.SX.zeros(2))
        f2mod = -f1mod
        f1m6 = ca.vertcat(f1mod, ca.SX.zeros(3))
        f2m6 = ca.vertcat(f2mod, ca.SX.zeros(3))

        force_bias1 = p[176:182]
        force_bias2 = p[182:188]
        f1pred = f1m6 + force_bias1
        f2pred = f2m6 + force_bias2
        # fJ contains the measured tangential generalized force.  Only the
        # measured/anchored x normal is added here, avoiding y/z double count.
        f1dyn = ca.vertcat(f1pred[0], ca.SX.zeros(2))
        f2dyn = ca.vertcat(f2pred[0], ca.SX.zeros(2))

        dcdx = ca.jacobian(compression_s, x)
        cd = (dcdx[0, :NQ] @ x[NQ:2 * NQ]
              + dcdx[0, 2 * NQ:3 * NQ] @ x[3 * NQ:])
        fd1x = -kx * cd
        fdot1 = ca.vertcat(fd1x, 0.0, 0.0, 0.0, 0.0, 0.0)
        fdot2 = -fdot1

        Jp1q = ca.jacobian(pL, x_qL)
        Jp2q = ca.jacobian(pR, x_qR)
        M1 = ca.reshape(p[0:49], 7, 7)
        M2 = ca.reshape(p[56:105], 7, 7)
        dd1 = f_expl[NQ:2 * NQ] + ca.solve(M1, Jp1q.T @ f1dyn - fJ1)
        dd2 = f_expl[3 * NQ:] + ca.solve(M2, Jp2q.T @ f2dyn - fJ2)
        f_expl_full = ca.vertcat(x[NQ:2 * NQ], dd1,
                                 x[3 * NQ:], dd2)
        rdot = ca.jacobian(tips, x) @ f_expl_full
        rdL, rdR = rdot[:nt], rdot[nt:2 * nt]

        Ic1, Ic2 = sym["Ic"]
        If1, If2 = sym["If"]
        r1ref, r2ref = p[126:132], p[132:138]
        v1ref, v2ref = p[138:144], p[144:150]
        f1ref, f2ref = p[150:156], p[156:162]
        eps1 = ((1 - If1) *
                (rL - (1 - Ic1) * r1ref - Ic1 * rR)
                - If1 * kf_scale * (f1pred - f1ref) / kx)
        eps2 = ((1 - If2) *
                (rR - (1 - Ic2) * r2ref - Ic2 * rL)
                - If2 * kf_scale * (f2pred - f2ref) / kx)
        epsd1 = ((1 - If1) *
                 (rdL - (1 - Ic1) * v1ref - Ic1 * rdR)
                 - If1 * kf_scale * fdot1 / kx)
        epsd2 = ((1 - If2) *
                 (rdR - (1 - Ic2) * v2ref - Ic2 * rdL)
                 - If2 * kf_scale * fdot2 / kx)
        y_expr = ca.vertcat(eps1, eps2, epsd1, epsd2, u)
        y_expr_e = ca.vertcat(eps1, eps2, epsd1, epsd2)

        self._np_full = NP + 4 * NT
        model = AcadosModel()
        model.name = cfg.model_name
        model.x = x
        model.u = u
        model.p = ca.vertcat(p, Ic1, Ic2, If1, If2)
        model.f_expl_expr = f_expl_full
        model.cost_y_expr = y_expr
        model.cost_y_expr_e = y_expr_e

        ny = int(y_expr.shape[0])
        ne = int(y_expr_e.shape[0])
        W = np.zeros((ny, ny))
        W[:2 * nt, :2 * nt] = cfg.Q * np.eye(2 * nt)
        W[2 * nt:4 * nt, 2 * nt:4 * nt] = cfg.Q * cfg.eta_dev * np.eye(2 * nt)
        W[4 * nt:, 4 * nt:] = cfg.Q * cfg.eta_u * np.eye(NU)
        self.W = W
        self.ny = ny
        self._force_nom_fn = ca.Function(
            f"{cfg.model_name}_force_nominal", [x], [f1m6, f2m6])

        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = cfg.N
        ocp.solver_options.tf = cfg.Tp
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = W
        ocp.cost.W_e = cfg.P * W[:ne, :ne].copy()
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ne)
        ocp.constraints.idxbu = np.arange(NU)
        ocp.constraints.lbu = self.umin
        ocp.constraints.ubu = self.umax
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

        json_file = f"{cfg.model_name}.json"
        try:
            self.solver = AcadosOcpSolver(
                ocp, json_file=json_file, build=False, generate=False, verbose=verbose)
        except (FileNotFoundError, OSError) as exc:
            print(f"[acados] cached grasp-pose solver unavailable; rebuilding: {exc}")
            self.solver = AcadosOcpSolver(
                ocp, json_file=json_file, build=True, generate=True, verbose=verbose)
        self._ocp = ocp
        self._json_file = json_file
        self._warm = None
        self.last_status = -1
        self.last_qp_iter = -1
        self.last_time_qp = float("nan")
        self.last_time_sim = float("nan")
        self.last_time_lin = float("nan")

    def _full_p(self, p_coef):
        p_coef = np.asarray(p_coef, dtype=float).reshape(-1)
        if p_coef.size == NP - N_FORCE_BIAS:
            p_coef = np.concatenate([p_coef, np.zeros(N_FORCE_BIAS)])
        if p_coef.size != NP:
            raise ValueError(
                f"p_coef 应为 {NP} 个参数（或旧版 {NP - N_FORCE_BIAS} 个），收到 {p_coef.size}")
        return np.concatenate([p_coef,
                               self.Ic1, self.Ic2, self.If1, self.If2])

    def _anchor_measured_force(self, x0, f_meas, p_coef):
        if f_meas is None:
            return np.asarray(p_coef, dtype=float).reshape(-1)
        f_meas = np.asarray(f_meas, dtype=float).reshape(-1)
        if f_meas.size < N_FORCE_BIAS:
            raise ValueError(f"f_meas 应包含 {N_FORCE_BIAS} 个数，收到 {f_meas.size}")
        p_coef = np.asarray(p_coef, dtype=float).reshape(-1).copy()
        if p_coef.size == NP - N_FORCE_BIAS:
            p_coef = np.concatenate([p_coef, np.zeros(N_FORCE_BIAS)])
        if p_coef.size != NP:
            raise ValueError(f"p_coef 应为 {NP} 个参数，收到 {p_coef.size}")
        nominal = np.concatenate([
            np.asarray(v, dtype=float).reshape(-1)
            for v in self._force_nom_fn(np.asarray(x0, dtype=float).reshape(-1)[:NX])
        ])
        p_coef[-N_FORCE_BIAS:] = f_meas[:N_FORCE_BIAS] - nominal
        return p_coef

    def _set_initial_guess(self, x0, u_seed):
        if self._warm is not None:
            wx, wu = self._warm
            for i in range(self.cfg.N):
                self.solver.set(i, "x", wx[i])
                self.solver.set(i, "u", wu[i])
            return
        if u_seed is not None:
            xs = np.asarray(x0, dtype=float).reshape(-1)[:NX]
            us = np.asarray(u_seed, dtype=float).reshape(-1)[:NU]
            for i in range(self.cfg.N):
                self.solver.set(i, "x", xs)
                self.solver.set(i, "u", us)

    def _set_parameters(self, p_full, yref):
        for i in range(self.cfg.N):
            self.solver.set(i, "p", p_full)
            self.solver.set(i, "yref", yref)
        self.solver.set(self.cfg.N, "p", p_full)
        self.solver.set(self.cfg.N, "yref", yref[:self.ny - NU])

    def solve(self, x0, p_coef, u_ref=None, f_meas=None, u_seed=None,
              u_prev=None, du_max=None):
        x0 = np.asarray(x0, dtype=float).reshape(-1)[:NX]
        p_coef = self._anchor_measured_force(x0, f_meas, p_coef)
        p_full = self._full_p(p_coef)
        yref = np.zeros(self.ny)
        if u_ref is not None:
            yref[4 * NT:] = np.asarray(u_ref, dtype=float).reshape(-1)[:NU]
        self._set_initial_guess(x0, u_seed)
        self._set_parameters(p_full, yref)
        if u_prev is not None and du_max is not None:
            center = np.asarray(u_prev, dtype=float).reshape(-1)[:NU]
            delta = np.broadcast_to(np.asarray(du_max, dtype=float), (NU,))
            self.solver.constraints_set(0, "lbu", np.maximum(self.umin, center - delta))
            self.solver.constraints_set(0, "ubu", np.minimum(self.umax, center + delta))
        else:
            self.solver.constraints_set(0, "lbu", self.umin)
            self.solver.constraints_set(0, "ubu", self.umax)
        try:
            status = self.solver.solve_for_x0(
                x0_bar=x0,
                fail_on_nonzero_status=self.cfg.fail_on_nonzero_status,
                print_stats_on_failure=True)
            status = int(self.solver.status)
        except RuntimeError:
            status = 4
        if status != 0 and self.cfg.rebuild_on_failure:
            self.solver = AcadosOcpSolver(
                self._ocp, json_file=self._json_file,
                build=True, generate=True, verbose=False)
            self._warm = None
            self._set_initial_guess(x0, u_seed)
            self._set_parameters(p_full, yref)
            try:
                status = self.solver.solve_for_x0(
                    x0_bar=x0,
                    fail_on_nonzero_status=self.cfg.fail_on_nonzero_status,
                    print_stats_on_failure=True)
                status = int(self.solver.status)
            except RuntimeError:
                status = 4
        self.last_status = int(status)
        for name, attr, default in (
            ("qp_iter", "last_qp_iter", -1),
            ("time_qp", "last_time_qp", float("nan")),
            ("time_sim", "last_time_sim", float("nan")),
            ("time_lin", "last_time_lin", float("nan")),
        ):
            try:
                val = np.asarray(self.solver.get_stats(name)).reshape(-1)[0]
                setattr(self, attr, int(val) if name == "qp_iter" else float(val))
            except Exception:
                setattr(self, attr, default)
        wx = np.array([self.solver.get(i, "x") for i in range(self.cfg.N)])
        wu = np.array([self.solver.get(i, "u") for i in range(self.cfg.N)])
        self._warm = (wx, wu)
        return np.asarray(self.solver.get(0, "u"), dtype=float).reshape(-1)[:NU]


__all__ = [
    "NMPCCCart", "NMPCCCartConfig", "build_ocp_symbolics",
    "mujoco_fk_symbols", "rotvec_of_R", "NT", "NU", "NQ", "NX",
    "NP", "N_FORCE_BIAS",
]


if __name__ == "__main__":
    print("nmpcc_cart_grasp_pose 模块自检：")
    print("  acados 可用:", _HAS_ACADOS)
    print("  NP =", NP, "NX =", NX, "NU =", NU)
    print("  f_expl 维度:", build_ocp_symbolics()["f_expl"].shape)
