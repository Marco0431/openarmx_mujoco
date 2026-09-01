#!/usr/bin/env python3
"""bench_nmpcc_solvers.py —— acados 求解组合基准（短 Holding headless）

对 nmpcc_cart.py 的 NMPCC（与 exp2_nmpcc_latest.py 同控制器参数），横测：
  - QP 求解器/凝聚：PARTIAL_CONDENSING_HPIPM（当前默认）、
    PARTIAL_CONDENSING_HPIPM + qp_solver_cond_N（多步凝聚）、
    FULL_CONDENSING_HPIPM、FULL_CONDENSING_QPOASES（本机 ACADOS_WITH_QPOASES=ON）；
  - 关键选项：N、Tp、sim_method_num_steps/stages（RK 精度 vs 成本）。

独立脚本（不 import exp2_nmpcc_latest，避免其被并行编辑影响）：场景/接触/常数
全部从 exp2_slender_scene.py 与 exp2 标定值内联复制。测试序列为已验证的
\"IK 预压到位 → 重力+夹持平衡种子 → 短 Holding\"（恒定参考，纯测量求解器表现）。

每个配置独立 subprocess 构建（model.name / json / codegen 目录互不相同，
避免并发编译与 DLL 名冲突）；单进程内只跑一个求解器。

运行（驱动全部配置）:
  D:\\acados_env\\python.exe -u bench_nmpcc_solvers.py --all

运行单个配置:
  D:\\acados_env\\python.exe -u bench_nmpcc_solvers.py --config pch_n20

输出:
  results/nmpcc_bench/<config>.json  （单配置明细）
  控制台摘要（驱动聚合）
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

OUT_DIR = "results/nmpcc_bench"

# ---- 与 exp2_nmpcc_latest.py 相同的标定/场景常数（内联，不 import 之）----
K_STIFF = 2793.166
REST_GAP = 0.507537
DT = 0.002
CTRL_EVERY = 3
T_HOLD = 0.6
T_RAMP = 0.5

CONFIGS = {
    # 基线：exp2 默认（PARTIAL_CONDENSING_HPIPM, N=20, Tp=0.10, ERK 4/4）
    "pch_n20": dict(N=20, Tp=0.10, qp_solver="PARTIAL_CONDENSING_HPIPM",
                    sim_method_num_steps=4, sim_method_num_stages=4),
    # 部分凝聚：cond_N=10（块长 2）
    "pch_n20_cond10": dict(N=20, Tp=0.10, qp_solver="PARTIAL_CONDENSING_HPIPM",
                           sim_method_num_steps=4, sim_method_num_stages=4,
                           qp_solver_cond_N=10),
    # 部分凝聚：cond_N=5（块长 4）
    "pch_n20_cond5": dict(N=20, Tp=0.10, qp_solver="PARTIAL_CONDENSING_HPIPM",
                          sim_method_num_steps=4, sim_method_num_stages=4,
                          qp_solver_cond_N=5),
    # 全凝聚 HPIPM
    "fch_n20_hpipm": dict(N=20, Tp=0.10, qp_solver="FULL_CONDENSING_HPIPM",
                          sim_method_num_steps=4, sim_method_num_stages=4),
    # 全凝聚 QPOASES
    "fch_n20_qpoases": dict(N=20, Tp=0.10, qp_solver="FULL_CONDENSING_QPOASES",
                            sim_method_num_steps=4, sim_method_num_stages=4),
    # 更少 RK 子步（动力学成本减半）
    "pch_n20_sim2": dict(N=20, Tp=0.10, qp_solver="PARTIAL_CONDENSING_HPIPM",
                         sim_method_num_steps=2, sim_method_num_stages=4),
    # 更短时域（步长不变）
    "pch_n10_tp05": dict(N=10, Tp=0.05, qp_solver="PARTIAL_CONDENSING_HPIPM",
                         sim_method_num_steps=4, sim_method_num_stages=4),
    # 更长时域
    "pch_n40_tp20": dict(N=40, Tp=0.20, qp_solver="PARTIAL_CONDENSING_HPIPM",
                         sim_method_num_steps=4, sim_method_num_stages=4),
    # 组合：sim2 + cond10（最快候选）
    "pch_n20_sim2_cond10": dict(N=20, Tp=0.10, qp_solver="PARTIAL_CONDENSING_HPIPM",
                                sim_method_num_steps=2, sim_method_num_stages=4,
                                qp_solver_cond_N=10),
    # 激进：RK 1 子步（动力学成本 1/4）
    "pch_n20_sim1": dict(N=20, Tp=0.10, qp_solver="PARTIAL_CONDENSING_HPIPM",
                         sim_method_num_steps=1, sim_method_num_stages=4),
    # 同级时域但阶段减半（dt=10ms，sim2 → 有效积分 5ms）
    "pch_n10_tp10_sim2": dict(N=10, Tp=0.10, qp_solver="PARTIAL_CONDENSING_HPIPM",
                              sim_method_num_steps=2, sim_method_num_stages=4),
}


def _run_one() -> None:
    """单体：加载场景 → 单配置构建（计时）→ 暖机 → 短 Holding → JSON 明细。"""
    import mujoco

    import nmpcc_cart as nc
    from exp2_slender_scene import (
        BOX_CENTER, CAN_OFFSETS_X, CAN_HALF_H, IK_ITERS, SLENDER_HALF,
        build_scene, grasp_targets, make_arms, solve_end_grip,
    )
    from ik import clamp_qpos, get_ee_tip, name_to_id

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--force", type=float, default=10.0)
    args = ap.parse_args()
    cfg_kwargs = dict(CONFIGS[args.config])
    force = args.force

    # ---- 场景/接触辅助（自包含）----
    def contact_wrench_on_arm(model, data, arm, box_geom_id):
        f_world = np.zeros(3)
        force6 = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            if not ((c.geom1 == arm["pad_id"] and c.geom2 == box_geom_id) or
                    (c.geom2 == arm["pad_id"] and c.geom1 == box_geom_id)):
                continue
            mujoco.mj_contactForce(model, data, i, force6)
            frame = c.frame.reshape(3, 3)
            f_world += frame.T @ force6[:3]
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, get_ee_tip(data, arm["body_id"]), arm["body_id"])
        f_world[0] = 0.0
        return jacp[:, arm["dofs"]].T @ f_world, f_world

    def contact_forces(model, data, arms, box_geom_id):
        result = {a["side"]: 0.0 for a in arms}
        force = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            for a in arms:
                if ((c.geom1 == a["pad_id"] and c.geom2 == box_geom_id) or
                        (c.geom2 == a["pad_id"] and c.geom1 == box_geom_id)):
                    mujoco.mj_contactForce(model, data, i, force)
                    result[a["side"]] += abs(float(force[0]))
        return result

    def box_tilt(model, data, box_qpos):
        rv = np.zeros(3)
        mujoco.mju_quat2Vel(rv, data.qpos[box_qpos + 3:box_qpos + 7], 1.0)
        return rv

    # ---- 场景加载 ----
    scene_xml = build_scene()
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    arms = make_arms(model)
    for a in arms:
        a["act_ids"] = [name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                   f"{model.joint(j).name}_motor") for j in a["joint_ids"]]

    box_body = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_geom = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    box_joint = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")
    box_qpos = int(model.jnt_qposadr[box_joint])

    # 复位世界（与 exp2 相同）
    for a in arms:
        data.qpos[a["qpos"]] = a["q_rest"]
        clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    data.qpos[box_qpos:box_qpos + 3] = BOX_CENTER
    data.qpos[box_qpos + 3:box_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    for k, ox in enumerate(CAN_OFFSETS_X):
        cq = int(model.jnt_qposadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                               f"can{k}_freejoint")])
        data.qpos[cq:cq + 3] = [BOX_CENTER[0] + ox, BOX_CENTER[1],
                                BOX_CENTER[2] + SLENDER_HALF[2] + CAN_HALF_H + 0.002]
        data.qpos[cq + 3:cq + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)

    # IK 预压到全量目标（F/(2K) 每侧）
    base_targets = grasp_targets(arms)
    delta = force / (2.0 * K_STIFF)
    press = {}
    for a in arms:
        target = base_targets[a["side"]].copy()
        target[0] += -a["sign"] * delta
        press[a["side"]] = target
    solve_end_grip(model, data, arms, press, IK_ITERS)

    rv_ref = {a["side"]: nc.rotvec_of_R(data.xmat[a["body_id"]].reshape(3, 3))
              for a in arms}
    fref = {
        "left": np.array([-force, 0, 0, 0, 0, 0]),
        "right": np.array([force, 0, 0, 0, 0, 0]),
    }
    fk = nc.mujoco_fk_symbols(
        model,
        [{"body_id": a["body_id"], "joint_ids": a["joint_ids"]} for a in arms],
        0.1001,
    )
    for d in [d for a in arms for d in a["dofs"]]:
        model.dof_damping[d] = 0.5

    umax_parts = []
    for a in arms:
        umax_parts.append(model.actuator_ctrlrange[a["act_ids"], 1])
    umax = np.concatenate(umax_parts)

    cfg = nc.NMPCCCartConfig(
        N=cfg_kwargs["N"], Tp=cfg_kwargs["Tp"],
        eta_cop=np.array([[0.0, 0.8, 0.0, 0.0, 0.0, 0.0]] * 2),
        eta_f=np.array([[0.8, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2),
        Q=np.array([50.0, 20000.0, 5000.0, 20.0, 20.0, 20.0]),
        eta_dev=5.0,
        eta_u=0.02,
        eta_u_vec=np.tile([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0], 2),
        P_mul=3.0,
        kx=K_STIFF,
        rest_gap=REST_GAP,
        kf_scale=0.10,
        vmax=0.05,
        vz_max=0.001,
        qp_solver=cfg_kwargs.get("qp_solver", "PARTIAL_CONDENSING_HPIPM"),
        sim_method_num_steps=cfg_kwargs.get("sim_method_num_steps", 4),
        sim_method_num_stages=cfg_kwargs.get("sim_method_num_stages", 4),
        qp_solver_cond_N=cfg_kwargs.get("qp_solver_cond_N"),
        model_name=f"nmpcc_bench_{args.config}",
        fail_on_nonzero_status=False,
        rebuild_on_failure=False,
    )

    # ---- 构建计时（generate + compile）----
    t0 = time.perf_counter()
    nmpc = nc.NMPCCCart(cfg, fk, umin=-umax, umax=umax, verbose=False)
    build_s = time.perf_counter() - t0

    def joint_state():
        return np.concatenate([
            data.qpos[arms[0]["qpos"]], data.qvel[arms[0]["dofs"]],
            data.qpos[arms[1]["qpos"]], data.qvel[arms[1]["dofs"]],
        ])

    def plant_params():
        full_m = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, full_m)
        mgs, fjs = [], []
        for a in arms:
            mm = full_m[np.ix_(a["dofs"], a["dofs"])]
            saved = data.qvel.copy()
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            bias = data.qfrc_bias[a["dofs"]].copy()
            data.qvel[:] = saved
            mujoco.mj_forward(model, data)
            fJ, _ = contact_wrench_on_arm(model, data, a, box_geom)
            mgs.extend([mm.reshape(-1), bias])
            fjs.append(fJ)
        damping = np.concatenate([model.dof_damping[a["dofs"]] for a in arms])
        return np.concatenate(mgs + fjs + [damping])

    def gravity_comp():
        out = []
        for a in arms:
            saved = data.qvel.copy()
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            out.append(data.qfrc_bias[a["dofs"]].copy())
            data.qvel[:] = saved
            mujoco.mj_forward(model, data)
        return np.concatenate(out)

    def clamp_balance_comp():
        out = []
        for a in arms:
            jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp, jacr,
                          get_ee_tip(data, a["body_id"]), a["body_id"])
            out.append(-(jacp[:, a["dofs"]].T @ fref[a["side"]][:3]))
        return np.concatenate(out)

    def build_p():
        p = np.zeros(nc.NP)
        pp = plant_params()
        p[0:112] = pp[:112]
        p[112:126] = pp[112:126]
        p[162:176] = pp[126:140]
        p[126:132] = np.concatenate([press["left"], rv_ref["left"]])
        p[132:138] = np.concatenate([press["right"], rv_ref["right"]])
        p[138:144] = np.zeros(6)
        p[144:150] = np.zeros(6)
        p[150:156] = fref["left"]
        p[156:162] = fref["right"]
        return p

    p0 = build_p()
    u_seed = gravity_comp() + clamp_balance_comp()
    nmpc._warm = None
    for _ in range(2):
        nmpc.solve(joint_state(), p0, u_ref=u_seed, u_seed=u_seed)

    # ---- Holding 采样循环（headless）----
    u_last = u_seed.copy()
    solve_ms = []
    time_qp_ms = []
    time_sim_ms = []
    time_lin_ms = []
    qp_iters = []
    statuses = []
    log = []
    for step in range(args.steps):
        t = step * DT
        if step % CTRL_EVERY == 0:
            p = build_p()
            t0 = time.perf_counter()
            try:
                u_last = nmpc.solve(joint_state(), p,
                                    u_ref=gravity_comp() + clamp_balance_comp(),
                                    u_seed=u_seed)
                ms = (time.perf_counter() - t0) * 1000.0
            except Exception as e:
                ms = float("nan")
                statuses.append(-2)
                solve_ms.append(ms)
                continue
            solve_ms.append(ms)
            statuses.append(nmpc.last_status)
            qp_iters.append(nmpc.last_qp_iter)
            time_qp_ms.append(nmpc.last_time_qp * 1000.0)
            time_sim_ms.append(getattr(nmpc, "last_time_sim", float("nan")) * 1000.0)
            time_lin_ms.append(getattr(nmpc, "last_time_lin", float("nan")) * 1000.0)
        for a in arms:
            data.ctrl[a["act_ids"]] = u_last[a["act_ids"]]
        mujoco.mj_step(model, data)
        if step % 5 == 0:
            forces = contact_forces(model, data, arms, box_geom)
            log.append({
                "t": t + DT,
                "box": data.xpos[box_body].copy(),
                "tipL": get_ee_tip(data, arms[0]["body_id"]).copy(),
                "tipR": get_ee_tip(data, arms[1]["body_id"]).copy(),
                "F_L": forces["left"], "F_R": forces["right"],
                "tilt": box_tilt(model, data, box_qpos),
            })

    # ---- 聚合（稳定窗 = 最后 0.3s）----
    v = [x for x in solve_ms if np.isfinite(x)]
    n_fail = sum(1 for s in statuses if s != 0 and s != -2)
    n_exc = sum(1 for s in statuses if s == -2)
    tt = np.array([r["t"] for r in log])
    sel = tt >= max(0.0, tt[-1] - 0.3) if len(tt) else np.zeros(0, bool)
    tip_l = np.array([r["tipL"] for r in log])[sel]
    tip_r = np.array([r["tipR"] for r in log])[sel]
    fl = np.array([r["F_L"] for r in log])[sel]
    fr = np.array([r["F_R"] for r in log])[sel]
    tilt = np.array([r["tilt"] for r in log])[sel]
    ys = (tip_l[:, 1] - tip_r[:, 1]) * 1000.0
    result = {
        "config": args.config,
        "params": dict(cfg_kwargs),
        "build_s": round(build_s, 1),
        "n_solve": len(v),
        "solve_mean_ms": round(float(np.mean(v)), 2) if v else None,
        "solve_median_ms": round(float(statistics.median(v)), 2) if v else None,
        "solve_p95_ms": round(float(np.percentile(v, 95)), 2) if v else None,
        "solve_max_ms": round(float(np.max(v)), 2) if v else None,
        "time_qp_mean_ms": round(float(np.nanmean(time_qp_ms)), 2) if time_qp_ms else None,
        "time_qp_max_ms": round(float(np.nanmax(time_qp_ms)), 2) if time_qp_ms else None,
        "time_sim_mean_ms": round(float(np.nanmean(time_sim_ms)), 2) if time_sim_ms else None,
        "time_lin_mean_ms": round(float(np.nanmean(time_lin_ms)), 2) if time_lin_ms else None,
        "qp_iter_mean": round(float(np.mean(qp_iters)), 2) if qp_iters else None,
        "qp_iter_max": int(np.max(qp_iters)) if qp_iters else None,
        "n_status_nonzero": n_fail,
        "n_exceptions": n_exc,
        "status_hist": {str(s): statuses.count(s) for s in sorted(set(statuses))},
        "hold_s": round(float(tt[-1]), 3),
        "y_sync_mean_mm": round(float(np.mean(np.abs(ys))), 3) if sel.any() else None,
        "y_sync_max_mm": round(float(np.max(np.abs(ys))), 3) if sel.any() else None,
        "force_sync_mean_N": round(float(np.mean(np.abs(fl - fr))), 3) if sel.any() else None,
        "force_sync_max_N": round(float(np.max(np.abs(fl - fr))), 3) if sel.any() else None,
        "contact_L_mean_N": round(float(np.mean(fl)), 3) if sel.any() else None,
        "contact_R_mean_N": round(float(np.mean(fr)), 3) if sel.any() else None,
        "force_target_N": force,
        "box_z_drift_max_mm": round(
            float(np.max(np.abs(np.array([r["box"] for r in log])[sel, 2]
                                - BOX_CENTER[2]))) * 1000, 3) if sel.any() else None,
        "tilt_max_deg": np.round(np.max(np.abs(tilt), axis=0) * 180 / np.pi, 3).tolist()
                        if sel.any() else None,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{args.config}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_all() -> None:
    """驱动：逐配置 subprocess（避免并发编译/DLL 冲突 + 崩溃隔离）→ 聚合表格。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--config", choices=list(CONFIGS), default=None)
    ap.add_argument("--steps", type=int, default=700)
    args, _ = ap.parse_known_args()
    if args.config is not None:
        _run_one()
        return
    results = {}
    for name in CONFIGS:
        print(f"\n========== [{name}] ==========")
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, "-u", __file__,
                            "--config", name, "--steps", str(args.steps)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        wall = time.perf_counter() - t0
        if r.returncode != 0:
            results[name] = {"error": r.returncode, "stderr_tail": r.stderr[-800:],
                             "stdout_tail": r.stdout[-400:]}
            print(f"[{name}] FAILED rc={r.returncode} in {wall:.1f}s")
            print(r.stdout[-400:])
            print(r.stderr[-800:])
            continue
        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, "r", encoding="utf-8") as f:
            results[name] = json.load(f)
        print(f"[{name}] done in {wall:.1f}s")

    print("\n\n================ 汇总表 ================")
    print(f"{'config':<18}{'build_s':>8}{'mean_ms':>9}{'p95_ms':>8}{'max_ms':>8}"
          f"{'qp_ms':>8}{'qpIt':>6}{'fail':>5}"
          f"{'ySync_mm':>10}{'y_max_mm':>9}{'Fdiff_N':>8}{'zDrift_mm':>10}")
    for name, rr in results.items():
        if "error" in rr:
            print(f"{name:<18}ERROR rc={rr['error']}")
            continue
        print(f"{name:<18}{rr['build_s']:>8.1f}{rr.get('solve_mean_ms', 0):>9.2f}"
              f"{rr.get('solve_p95_ms', 0):>8.2f}{rr.get('solve_max_ms', 0):>8.2f}"
              f"{rr.get('time_qp_mean_ms', 0):>8.2f}{rr.get('qp_iter_mean', 0):>6.1f}"
              f"{rr['n_status_nonzero']:>5}"
              f"{rr.get('y_sync_mean_mm', float('nan')):>10.3f}"
              f"{rr.get('y_sync_max_mm', float('nan')):>9.3f}"
              f"{rr.get('force_sync_max_N', float('nan')):>8.3f}"
              f"{rr.get('box_z_drift_max_mm', float('nan')):>10.3f}")


if __name__ == "__main__":
    run_all()