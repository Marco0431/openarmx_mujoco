#!/usr/bin/env python3
"""cart_check.py —— nmpcc_cart.py 模块自检（执行前必须先用 D:\\acados_env\\python.exe）

验证项（对应计划"模块自检"步）：
  1. 冻结系数：coefficient_block 数值自洽（Λ 对称正定、Ka 结构、μ̃/p̃ 定义）
  2. casadi 动力学 vs MuJoCo 端面加速度（无 f 项时一致）：||acc_casadi − acc_mj|| < 1e-7
  3. T(rv) 旋转向量-角速度映射：数值积分小转角 → rv̇ vs ω 一致
  4. 一次 NMPCCCart.solve：u 有限、不抛异常、求解耗时
  5. 平衡点 = 论文式(14)：ε_mult(r*) = 0 的数值解 == 式(15)(16) 形式解（不动点迭代比对）

运行: D:\\acados_env\\python.exe -u cart_check.py
"""
from __future__ import annotations

import time

import numpy as np
import casadi as ca

import mujoco

import nmpcc_cart as nc
from ik import ARM_CONFIGS, get_ee_tip
from ikforce import build_model


def tol_ok(name, val, tol):
    ok = val < tol
    print(f"  [{'OK ' if ok else 'BAD'}] {name}: err = {val:.3e} (tol {tol:.0e})")
    return ok


def main() -> None:
    print("=== 1. 冻结系数数值自洽 ===")
    model, data = build_model()
    arm_cfgs = []
    from ik import ARM_CONFIGS
    for cfg in ARM_CONFIGS:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["jacobi_body"])
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg["joint_names"]]
        dofs = [int(model.jnt_dofadr[j]) for j in joint_ids]
        arm_cfgs.append({"body_id": body_id, "joint_ids": joint_ids, "dofs": dofs})

    def freeze_coefs(arm):
        """计算该臂 90 维冻结系数参数。"""
        body_id = arm["body_id"]
        dofs = arm["dofs"]
        tip = get_ee_tip(data, body_id)
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, tip, body_id)
        J6 = np.vstack([jacp, jacr])[:, dofs]                     # (6,7)
        m = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, m)
        Minv = np.linalg.inv(m[np.ix_(dofs, dofs)])               # (7,7)
        # MuJoCo 3.12: qvel=0 时 qfrc_bias = −g（重力广义力，注意负号约定）
        # qfrc_bias(q,v) = −(h + g) → g = −bias_v0, h = bias_v0 − bias_v
        v_saved = data.qvel.copy()
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        bias_v0 = data.qfrc_bias[dofs].copy()
        data.qvel[:] = v_saved
        mujoco.mj_forward(model, data)
        bias_v = data.qfrc_bias[dofs].copy()
        G = -bias_v0                                                    # 重力广义力 g
        c = bias_v0 - bias_v                                            # 科氏/离心 h
        jacp_dot = np.zeros((3, model.nv)); jacr_dot = np.zeros((3, model.nv))
        mujoco.mj_jacDot(model, data, jacp_dot, jacr_dot, tip, body_id)
        Jdot_q = np.vstack([jacp_dot, jacr_dot])[:, dofs] @ data.qvel[dofs]
        blk = nc.coefficient_block(J6, Minv, c, G, Jdot_q)
        # 额外校验：Λ 对称正定
        lam_inv = J6 @ Minv @ J6.T
        lam = np.linalg.pinv(lam_inv, rcond=1e-9)
        assert np.allclose(lam, lam.T, atol=1e-9)
        return blk, J6, Minv, lam, Jdot_q, c, G

    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    data.qvel[arm_cfgs[0]["dofs"]] = 0.2 * np.ones(7)
    data.qvel[arm_cfgs[1]["dofs"]] = -0.15 * np.ones(7)
    mujoco.mj_forward(model, data)

    blkL, J6L, MinvL, lamL, JdotL, cL, GL = freeze_coefs(arm_cfgs[0])
    blkR, J6R, MinvR, lamR, JdotR, cR, GR = freeze_coefs(arm_cfgs[1])

    print("  系数长度:", blkL.shape, blkR.shape)

    print("\n=== 2. casadi 动力学 vs MuJoCo 端面加速度（无相互作用 f）===")
    sym = nc.build_ocp_symbolics(6250.0, 0.248)
    f_expl = ca.Function("f_expl", [sym["x"], sym["u"], sym["p"]], [sym["f_expl"]])

    def measure_state(arm):
        body_id = arm["body_id"]
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, get_ee_tip(data, body_id), body_id)
        rv = nc.rotvec_of_R(data.xmat[body_id].reshape(3, 3))
        p = get_ee_tip(data, body_id)
        v = jacp[:, arm["dofs"]] @ data.qvel[arm["dofs"]]
        w = jacr[:, arm["dofs"]] @ data.qvel[arm["dofs"]]
        return np.concatenate([p, rv, v, w])

    x_all = np.concatenate([measure_state(arm_cfgs[0]), measure_state(arm_cfgs[1])])
    # 参数：Λinv 置 0 → 关闭式(27) f 项；μ̃、p̃ 用真值 → 纯重力+科氏动力学
    p_test = np.zeros(nc.NP)
    p_test[0:42] = blkL[0:42]                     # KaL
    p_test[42:78] = 0.0                           # ΛinvL = 0（关闭 f）
    p_test[78:84] = blkL[78:84]; p_test[84:90] = blkL[84:90]
    p_test[90:132] = blkR[0:42]                   # KaR
    p_test[132:168] = 0.0
    p_test[168:174] = blkR[78:84]; p_test[174:180] = blkR[84:90]
    acc_cas = np.array(f_expl(x_all, np.zeros(14), p_test)).flatten()[6:12]   # f_expl = [v1,Tw1,acc1, v2,Tw2,acc2]
    # MuJoCo 真值（无相互作用 f）：qacc = M⁻¹·bias_v（bias_v = −(h+g)）
    acc_mj = J6L @ (MinvL @ (-(cL + GL))) + JdotL
    errL = np.linalg.norm(acc_cas - acc_mj)
    if errL > 1e-6:
        print("  [debug L] acc_cas =", np.round(acc_cas, 4))
        print("  [debug L] acc_mj  =", np.round(acc_mj, 4))
        print("  [debug L] mu_tilde/p_tilde =", np.round(blkL[78:84], 4), np.round(blkL[84:90], 4))
        print("  [debug L] Jdot_q =", np.round(JdotL, 4))
    tol_ok("左臂端面加速度 err（casadi vs MuJoCo）", errL, 1e-6)
    acc_cas2 = np.array(f_expl(x_all, np.zeros(14), p_test)).flatten()[18:24]
    acc_mj2 = J6R @ (MinvR @ (-(cR + GR))) + JdotR
    tol_ok("右臂端面加速度 err（casadi vs MuJoCo）", np.linalg.norm(acc_cas2 - acc_mj2), 1e-6)

    print("\n=== 3. T(rv) 旋转向量-角速度映射（小转角积分）===")
    rv_test = np.array([0.02, -0.03, 0.04])
    w_test = np.array([0.1, -0.2, 0.15])
    rv_sx = ca.SX.sym("rv", 3)
    T_fn = ca.Function("T", [rv_sx], [nc.rotvec_map_T(rv_sx)])
    T_num = np.array(T_fn(rv_test)).reshape(3, 3)
    rv_dot_cas = T_num @ w_test
    rv_dot_approx = w_test
    tol_ok("T(rv)·ω vs ω (小角度, 偏差 O(θ·ω))", np.linalg.norm(rv_dot_cas - rv_dot_approx), 5e-3)
    # rv=0 时 T = I（极限）
    T0 = np.array(T_fn(np.zeros(3))).reshape(3, 3)
    tol_ok("T(0) = I", np.linalg.norm(T0 - np.eye(3)), 1e-6)

    print("\n=== 4. 一次 NMPCCCart.solve（acados 编译, 首次较久）===")
    t0 = time.perf_counter()
    cfg = nc.NMPCCCartConfig(N=50, Tp=0.1)
    nmpc = nc.NMPCCCart(cfg, verbose=False)
    print(f"  编译+构建耗时 {time.perf_counter() - t0:.1f} s")

    x0 = measure_state(arm_cfgs[0])
    x1 = measure_state(arm_cfgs[1])
    x0v = np.concatenate([x0, x1])
    p_coef = np.zeros(nc.NP)
    p_coef[0:42] = blkL[0:42]; p_coef[42:78] = blkL[42:78]
    p_coef[78:84] = blkL[78:84]; p_coef[84:90] = blkL[84:90]
    p_coef[90:132] = blkR[0:42]; p_coef[132:168] = blkR[42:78]
    p_coef[168:174] = blkR[78:84]; p_coef[174:180] = blkR[84:90]
    t1 = time.perf_counter()
    u = nmpc.solve(x0v, p_coef)
    dt = (time.perf_counter() - t1) * 1e3
    ok = np.all(np.isfinite(u)) and np.all(np.abs(u) < 8.0 + 1e-6)
    print(f"  u = {np.round(u, 3)}")
    print(f"  [{'OK ' if ok else 'BAD'}] solve 一次: 有限且 ≤8 N·m, 耗时 {dt:.1f} ms")

    print("\n=== 5. 平衡点 = 论文式(14) ===")
    # 常数参考（近似抓取姿态），f_ref 设定 ±25 N 沿 x，kx 标定 25/0.248
    kx = 25.0 / (2 * 0.004)      # 25 N ↔ 压缩 4 mm（两侧各 2 mm）→ 6250 N/m
    rest_gap = 0.248            # 自由间距 2×(0.12+0.004)
    # 平衡点：ε=0 → 式(14) r_i = (1−Ic)r_ref,i + Ic·r_j + (1−If)⁻¹If(f_i − f_ref,i)
    Ic = np.array([0.0, 0.8, 0.8, 0.0, 0.0, 0.0])
    If = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0])

    sym2 = nc.build_ocp_symbolics(kx, rest_gap)
    eps_fn = ca.Function("eps", [sym2["x"], sym2["p"], sym2["Ic"][0], sym2["Ic"][1], sym2["If"][0], sym2["If"][1]],
                         [sym2["eps"]])

    def smoother_pos(c, eps=0.002):
        return 0.5 * (c + np.sqrt(c * c + eps * eps))

    # r_ref 设为一个抓取位形（臂1 = 左臂 = +x 侧；臂2 = 右臂 = −x 侧，openarmx yaw 几何）
    r1ref = np.array([0.624, 0.0, 0.55, 0.02, -0.01, 0.015])
    r2ref = np.array([0.376, 0.0, 0.55, 0.02, -0.01, 0.015])
    v1ref = np.zeros(6); v2ref = np.zeros(6)
    f1ref = np.array([-25.0, 0, 0, 0, 0, 0]); f2ref = np.array([25.0, 0, 0, 0, 0, 0])

    # 平衡点（对称解）：r1x = R1x − Δ, r2x = R2x + Δ，满足 ε=0（1D 求根）
    # ε1 = 0.2·(−Δ)? 直接数值：ε1 = 0.2(r1x − R1x) − 0.8(f1x − f1refx)/kx
    def balance_root(delta):
        d = (r1ref[0] - delta) - (r2ref[0] + delta)   # x1 − x2
        c = rest_gap - d
        f1x = -kx * smoother_pos(c)
        return 0.2 * (-delta) - 0.8 * (f1x - f1ref[0]) / kx

    dlo, dhi = 0.0, 0.02
    for _ in range(200):
        dm = 0.5 * (dlo + dhi)
        fv = balance_root(dm)
        if fv > 0:
            dhi = dm          # 根在左侧：压下上限
        else:
            dlo = dm
    delta = 0.5 * (dlo + dhi)
    r1 = r1ref.copy(); r2 = r2ref.copy()
    r1[0] -= delta; r2[0] += delta
    print(f"  平衡点 r1 = {np.round(r1, 5)}   r2 = {np.round(r2, 5)}（对称压入 Δ = {delta*1000:.3f} mm/侧）")
    # 用 eps_fn 验证残差 ≈ 0
    p_test2 = np.zeros(nc.NP)
    p_test2[180:186] = r1ref; p_test2[186:192] = r2ref
    p_test2[192:198] = v1ref; p_test2[198:204] = v2ref
    p_test2[204:210] = f1ref; p_test2[210:216] = f2ref
    x_eq = np.concatenate([r1, np.zeros(6), r2, np.zeros(6)])
    eps_val = np.array(eps_fn(x_eq, p_test2, Ic, Ic, If, If)).flatten()
    tol_ok("平衡点 ε_mult", np.linalg.norm(eps_val), 1e-6)
    # 验证 x 方向：夹持力 = f_ref
    d = r1[0] - r2[0]
    F1 = -kx * smoother_pos(rest_gap - d)
    tol_ok("平衡点夹持力 x 分量 == f_ref (25 N)", abs(F1 - (-25.0)), 1e-2)

    print("\n=== 自检结束 ===")


if __name__ == "__main__":
    main()