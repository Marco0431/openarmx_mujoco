"""实验统计、CSV/NPZ导出及辅助命令行；不参与控制求解。"""
import csv,json,os,subprocess,sys,time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
NU = 14

def summarize(log, args, initial_box_y, move_duration=None, *, box_center, hold_box_y_tol, motion_timing, physics_dt) -> dict:
    tt = np.array([r["t"] for r in log])
    box = np.array([r["box"] for r in log])
    rv = np.array([r["rotvec"] for r in log])
    tip_l = np.array([r["tipL"] for r in log])
    tip_r = np.array([r["tipR"] for r in log])
    fl = np.array([r["FL"] for r in log]); fr = np.array([r["FR"] for r in log])
    w_l_t = np.array([r["wL"][1] for r in log])
    w_r_t = np.array([r["wR"][1] for r in log])
    box_t = np.array([r["box_t"] for r in log])
    u_clip = np.array([r["u_clip_mask"] for r in log], dtype=bool)
    u_sat = np.array([r["u_sat_mask"] for r in log], dtype=bool)
    motion_commanded = (not args.hold_only and abs(float(args.y_dist)) > 1e-9
                        and abs(float(args.v_y)) > 1e-9)
    # hold 全窗（worst-case 评估用）：hold_only = [t_ramp, 结束]；moving = [t_ramp, hold_end]
    if not motion_commanded:
        sel = tt >= max(0.0, tt[-1] - 0.3)
        sel_hold = tt >= args.t_ramp
        sel_move = np.zeros_like(tt, dtype=bool)
    elif motion_commanded:
        sel = (tt >= args.t_ramp + args.t_hold - 0.3) & (tt <= args.t_ramp + args.t_hold)
        sel_hold = (tt >= args.t_ramp) & (tt <= args.t_ramp + args.t_hold)
        sel_move = np.array([r["phase"] == "move" for r in log], dtype=bool)
    if not sel.any():
        sel = tt >= tt[-1] - 0.3
    if not sel_hold.any():
        sel_hold = sel
    if not sel_move.any():
        sel_move = np.zeros_like(tt, dtype=bool)
    ys_mm = (tip_l[:, 1] - tip_r[:, 1]) * 1000.0
    rv_deg = np.abs(rv) * 180.0 / np.pi
    initial_box_y = float(initial_box_y)

    def max_or_nan(values):
        values = np.asarray(values)
        return float(np.max(values)) if values.size else float("nan")

    def mean_or_nan(values):
        values = np.asarray(values)
        return float(np.mean(values)) if values.size else float("nan")

    def drift_mm(values):
        values = np.asarray(values)
        return (float(np.max(np.abs(values - np.mean(values))) * 1000.0)
                if values.size else float("nan"))

    box_y_hold_max_mm = max_or_nan(np.abs(box[sel_hold, 1] - initial_box_y) * 1000.0)
    box_y_move_delta_mm = ((box[sel_move, 1] - box[sel_move, 1][0]) * 1000.0
                           if sel_move.any() else np.array([]))
    box_y_move_travel_mm = (-float(box_y_move_delta_mm[-1])
                            if box_y_move_delta_mm.size else float("nan"))
    box_y_move_travel_error_mm = (abs(box_y_move_travel_mm - abs(float(args.y_dist)) * 1000.0)
                                  if np.isfinite(box_y_move_travel_mm) else float("nan"))
    control_rows = [r for r in log if np.isfinite(r["solver_status"])]
    statuses = np.array([r["solver_status"] for r in control_rows], dtype=float)
    qp_iters = np.array([r["qp_iter"] for r in control_rows], dtype=float)
    status_bad = statuses[(statuses >= 0) & (statuses != 0)]
    move_observed = bool(sel_move.any())
    commanded_distance = abs(float(args.y_dist)) if motion_commanded else 0.0
    if move_duration is None:
        move_duration = motion_timing(commanded_distance, args.v_y)[2]
    expected_move_end = args.t_ramp + args.t_hold + move_duration
    box_y_final_error_mm = abs((box[-1, 1] - initial_box_y) + commanded_distance) * 1000.0
    # The plant is still stepped for a short settling tail after the commanded
    # translation.  Evaluate the endpoint at the end of the commanded motion,
    # rather than at the final tail sample.
    move_end_t = args.t_ramp + args.t_hold + move_duration
    endpoint_idx = int(np.argmin(np.abs(tt - move_end_t)))
    box_y_endpoint_error_mm = abs(
        (box[endpoint_idx, 1] - initial_box_y) + commanded_distance
    ) * 1000.0
    box_y_initial_error_mm = box_y_endpoint_error_mm
    # 评价固定时间切换后的实际搬运行程。
    move_complete_after_settle = int(
        move_observed and tt[-1] + physics_dt >= expected_move_end
        and box_y_move_travel_error_mm <= 2.0
    )
    move_complete = int(
        move_complete_after_settle
        and box_y_hold_max_mm <= hold_box_y_tol * 1000.0
        and box_y_initial_error_mm <= 2.0
    )
    m = {
        "force": args.force, "kf_scale": args.kf_scale,
        "Q": args.Q, "P": args.P, "eta_dev": args.eta_dev,
        "eta_u": args.eta_u, "eta_cop_y": args.eta_cop_y,
        "t_ramp": args.t_ramp, "t_hold": args.t_hold,
        "f_start": args.f_start,
        "transition_mode": "timed",
        "motion_commanded": int(motion_commanded),
        "moving_start_s": args.t_ramp + args.t_hold,
        "tilt_x_max_deg": max_or_nan(rv_deg[sel, 0]),
        "tilt_y_max_deg": max_or_nan(rv_deg[sel, 1]),
        "tilt_z_max_deg": max_or_nan(rv_deg[sel, 2]),
        "tilt_x_hold_max_deg": max_or_nan(rv_deg[sel_hold, 0]),
        "tilt_y_hold_max_deg": max_or_nan(rv_deg[sel_hold, 1]),
        "tilt_z_hold_max_deg": max_or_nan(rv_deg[sel_hold, 2]),
        "z_sag_mm": mean_or_nan(box[sel, 2] - box_center[2]) * 1000.0,
        "z_drift_mm": drift_mm(box[sel, 2]),
        "z_drift_hold_mm": drift_mm(box[sel_hold, 2]),
        "FL_mean": mean_or_nan(fl[sel]), "FR_mean": mean_or_nan(fr[sel]),
        "F_diff_max_N": max_or_nan(np.abs(fl[sel] - fr[sel])),
        "F_diff_hold_max_N": max_or_nan(np.abs(fl[sel_hold] - fr[sel_hold])),
        "y_sync_mean_mm": mean_or_nan(np.abs(ys_mm)),
        "y_sync_max_mm": max_or_nan(np.abs(ys_mm)),
        "y_sync_hold_max_mm": max_or_nan(np.abs(ys_mm[sel_hold])),
        "tilt_x_move_max_deg": max_or_nan(rv_deg[sel_move, 0]),
        "tilt_y_move_max_deg": max_or_nan(rv_deg[sel_move, 1]),
        "tilt_z_move_max_deg": max_or_nan(rv_deg[sel_move, 2]),
        "z_drift_move_mm": drift_mm(box[sel_move, 2]),
        "F_diff_move_max_N": max_or_nan(np.abs(fl[sel_move] - fr[sel_move])),
        "contact_moment_left_move_max_Nm": max_or_nan(
            np.linalg.norm(w_l_t[sel_move], axis=1)),
        "contact_moment_right_move_max_Nm": max_or_nan(
            np.linalg.norm(w_r_t[sel_move], axis=1)),
        "contact_moment_box_move_max_Nm": max_or_nan(
            np.linalg.norm(box_t[sel_move], axis=1)),
        "y_sync_move_mean_mm": mean_or_nan(np.abs(ys_mm[sel_move])),
        "y_sync_move_max_mm": max_or_nan(np.abs(ys_mm[sel_move])),
        "actuator_clip_fraction": mean_or_nan(u_clip),
        "actuator_saturation_fraction": mean_or_nan(u_sat),
        "actuator_clip_move_fraction": mean_or_nan(u_clip[sel_move]),
        "actuator_saturation_move_fraction": mean_or_nan(u_sat[sel_move]),
        "move_observed": int(move_observed),
        "move_complete": move_complete,
        "move_complete_after_settle": move_complete_after_settle,
        "move_expected_end_s": expected_move_end,
        "move_end_t_s": float(tt[-1]),
        "initial_box_y": initial_box_y,
        "box_y_final_drift_mm": (float(box[-1, 1] - initial_box_y) * 1000.0),
        "box_y_hold_max_mm": box_y_hold_max_mm,
        "box_y_initial_error_mm": box_y_initial_error_mm,
        "box_y_move_travel_mm": box_y_move_travel_mm,
        "box_y_move_travel_error_mm": box_y_move_travel_error_mm,
        "box_y_endpoint_error_mm": box_y_endpoint_error_mm,
        "box_y_final_error_mm": box_y_final_error_mm,
        "solver_status_bad_count": int(status_bad.size),
        "solver_status_zero_fraction": (float(np.mean(statuses == 0.0))
                                         if statuses.size else float("nan")),
        "qp_iter_mean": mean_or_nan(qp_iters),
        "qp_iter_max": max_or_nan(qp_iters),
        "solver_qp_ms_mean": mean_or_nan(
            [r["solver_qp_ms"] for r in control_rows]),
        "solver_sim_ms_mean": mean_or_nan(
            [r["solver_sim_ms"] for r in control_rows]),
        "solver_lin_ms_mean": mean_or_nan(
            [r["solver_lin_ms"] for r in control_rows]),
        "solve_ms_mean": mean_or_nan([r["solve_ms"] for r in log
                                       if np.isfinite(r["solve_ms"])]),
        "solve_ms_max": max_or_nan([r["solve_ms"] for r in log
                                     if np.isfinite(r["solve_ms"])]),
    }
    return m


def finish_run(log, horizon_log, m, args, result, tag, dt, output_dir, entry_file):
    """打印结果并按原格式导出；绘图失败不影响已完成的仿真。"""
    from experiments.motion_trace import save_motion_outputs
    # 完整节点参数独立于旧 CSV，测量锚定律的额外参数不能在旧布局转换中丢失。
    if horizon_log:
        os.makedirs(output_dir, exist_ok=True)
        packed = {k: np.asarray([row[k] for row in horizon_log])
                  for k in horizon_log[0] if k != 'contact_load'}
        if 'contact_load' in horizon_log[0]:
            packed.update({'contact_' + k: np.asarray([row['contact_load'][k] for row in horizon_log])
                           for k in horizon_log[0]['contact_load']})
        np.savez_compressed(Path(output_dir) / 'prediction.npz', **packed)
        np.savez_compressed(Path(output_dir) / 'physics.npz', **result.physics_trace)
        if result.startup:
            np.savez_compressed(Path(output_dir) / 'startup.npz',
                **{k: np.asarray([row[k] for row in result.startup]) for k in result.startup[0]})
    # 1. 独立诊断和首次失败现场，即使禁用 CSV 也按原选项保存。
    if args.dynamics_diag:
        os.makedirs(output_dir, exist_ok=True)
        dynamics_path = os.path.join(output_dir, f"dynamics_{tag}_{time.time_ns()}.npz")
        np.savez_compressed(dynamics_path, **result.dynamics_trace)
        print(f"[dynamics] diagnostic trace: {dynamics_path}")
    if result.failure:
        os.makedirs(output_dir, exist_ok=True)
        recovered_path = os.path.join(output_dir, f"joint_limit_retry_{tag}_{time.time_ns()}.npz")
        np.savez_compressed(recovered_path, **result.joint_limit_trace)
        print(f"[joint limits] first failed solve before retry: {recovered_path}")
    # 2. 人可读的单次结果摘要；字段含义与 summarize 保持一致。
    print(f"[run {tag}] F={args.force:.1f} kf={args.kf_scale:.3f} "
          f"timed_hold={args.t_hold:.2f}s move_start={m['moving_start_s']:.2f}s | "
          f"{'move=DISABLED' if not m['motion_commanded'] else ('move=COMPLETE' if m['move_complete'] else 'move=INCOMPLETE')} "
          f"(after_settle={m['move_complete_after_settle']}, observed={m['move_observed']}, "
          f"hold_dy={m['box_y_hold_max_mm']:.2f}mm, "
          f"travel={m['box_y_move_travel_mm']:.2f}mm, "
          f"initial_end_err={m['box_y_initial_error_mm']:.2f}mm) | "
          f"tilt=({m['tilt_x_max_deg']:.2f},"
          f"{m['tilt_y_max_deg']:.2f},{m['tilt_z_max_deg']:.2f})° | "
          f"z_sag={m['z_sag_mm']:+.2f}mm z_drift={m['z_drift_mm']:.2f}mm | "
          f"F=({m['FL_mean']:.2f},{m['FR_mean']:.2f})N diff={m['F_diff_max_N']:.2f}N | "
          f"ysync={m['y_sync_mean_mm']:.2f}/{m['y_sync_max_mm']:.2f}mm | "
          f"status_bad={m['solver_status_bad_count']} qp={m['qp_iter_mean']:.1f} "
          f"solve={m['solve_ms_mean']:.2f}ms ({dt:.1f}s)")
    if not args.no_csv:
        # 3. 每五个物理步一行：保持旧 CSV 列顺序和展开形状。
        os.makedirs(output_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"exp2_latest_{tag}_{stamp}.csv")
        np.savez_compressed(path[:-4] + "_joint_limits.npz", **result.joint_limit_trace)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "phase", "ramp_s", "moving_active",
                        "boxx", "boxy", "boxz", "q_w", "q_x", "q_y", "q_z",
                        "rvx", "rvy", "rvz",
                        "tipLx", "tipLy", "tipLz", "tipRx", "tipRy", "tipRz",
                        "FLx", "FLy", "FLz", "FLtx", "FLty", "FLtz",
                        "FRx", "FRy", "FRz", "FRtx", "FRty", "FRtz",
                        "BCFx", "BCFy", "BCFz", "BCtx", "BCty", "BCtz",
                        "FL", "FR",
                        "FmeasCtrlLx", "FmeasCtrlRx",
                        "FmodelLx", "FmodelRx", "bFLx", "bFRx",
                        "FanchorLx", "FanchorRx", "FrefLx", "FrefRx",
                        *[f"Fpred_{k}_{side}" for k in ("k0", "mid", "N") for side in ("L", "R")],
                        *[f"Ferr_{k}_{side}" for k in ("k0", "mid", "N") for side in ("L", "R")],
                        *[f"Xerr_{k}_{side}" for k in ("k0", "mid", "N") for side in ("L", "R")],
                        *[f"EpsX_{k}_{side}_{axis}" for k in ("k0", "mid", "N")
                          for side in ("L", "R") for axis in ("x", "y", "z", "rx", "ry", "rz")],
                        *[f"EpsdX_{k}_{side}_{axis}" for k in ("k0", "mid", "N")
                          for side in ("L", "R") for axis in ("x", "y", "z", "rx", "ry", "rz")],
                        *[f"Upred_{k}_{i + 1}" for k in ("k0", "mid", "last") for i in range(NU)],
                        *[f"u{i + 1}" for i in range(NU)],
                        *[f"uclip{i + 1}" for i in range(NU)],
                        *[f"usat{i + 1}" for i in range(NU)],
                        "solve_ms", "solver_status", "qp_iter",
                        "solver_qp_ms", "solver_sim_ms", "solver_lin_ms",
                        "prepare_ms", "reference_ms", "solver_ms"])
            for r in log:
                w.writerow([r["t"], r["phase"], r["ramp_s"], r["moving_active"],
                            *r["box"], *r["quat"], *r["rotvec"],
                            *r["tipL"], *r["tipR"], *r["wL"][0], *r["wL"][1],
                            *r["wR"][0], *r["wR"][1], *r["box_f"], *r["box_t"],
                            r["FL"], r["FR"], *r["force_measured_control"],
                            *r["force_nominal"], *r["force_bias"],
                            *r["force_anchored"], *r["force_ref"],
                            *r["pred_force"].reshape(-1),
                            *r["pred_force_error"].reshape(-1),
                            *r["pred_position_error_x"].reshape(-1),
                            *r["pred_eps_x"].reshape(-1),
                            *r["pred_epsd_x"].reshape(-1),
                            *r["pred_u"].reshape(-1), *r["u_cmd"],
                            *r["u_clip_mask"].astype(int), *r["u_sat_mask"].astype(int),
                            r["solve_ms"], r["solver_status"],
                            r["qp_iter"], r["solver_qp_ms"], r["solver_sim_ms"],
                            r["solver_lin_ms"], r["prepare_ms"],
                            r["reference_ms"], r["solver_ms"]])
        print(f"saved {path}")
        # 4. 每个控制周期的全部 N+1 节点；终端没有控制量，以 NaN 占位。
        horizon_path = os.path.join(output_dir, f"exp2_latest_{tag}_{stamp}_horizon.csv")
        with open(horizon_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "t", "phase", "node", "tau_node",
                "FrefLx", "FrefRx", "FpredLx", "FpredRx",
                "FerrLx", "FerrRx", "XerrLx", "XerrRx",
                *[f"Eps_{side}_{axis}" for side in ("L", "R")
                  for axis in ("x", "y", "z", "rx", "ry", "rz")],
                *[f"EpsDot_{side}_{axis}" for side in ("L", "R")
                  for axis in ("x", "y", "z", "rx", "ry", "rz")],
                *[f"Upred{i + 1}" for i in range(NU)],
                *[f"Uref{i + 1}" for i in range(NU)],
                *[f"Udev{i + 1}" for i in range(NU)],
                "t_node",
                *[f"Rref_{side}_{axis}" for side in ("L", "R")
                  for axis in ("x", "y", "z", "rx", "ry", "rz")],
                *[f"Vref_{side}_{axis}" for side in ("L", "R")
                  for axis in ("x", "y", "z", "rx", "ry", "rz")],
                *[f"Fref_{side}_{axis}" for side in ("L", "R")
                  for axis in ("x", "y", "z", "rx", "ry", "rz")],
                *[f"FrefDot_{side}_{axis}" for side in ("L", "R")
                  for axis in ("x", "y", "z", "rx", "ry", "rz")],
                "prepare_ms", "reference_ms", "solver_ms", "control_ms",
            ])
            for h in horizon_log:
                for node in range(args.N + 1):
                    u_pred = (h["u"][node] if node < args.N
                              else np.full(NU, np.nan))
                    u_dev = u_pred - h["u_ref"]
                    w.writerow([
                        h["t"], h["phase"], node, h["node_times"][node],
                        *h["force_ref"][node], *h["force"][node],
                        *h["force_error"][node], *h["position_error_x"][node],
                        *h["eps_x"][node], *h["epsd_x"][node],
                        *u_pred, *h["u_ref"], *u_dev,
                        h["t"] + h["node_times"][node],
                        *h["reference"][node], *h["force_ref_dot"][node],
                        h["prepare_ms"], h["reference_ms"], h["solver_ms"], h["control_ms"],
                    ])
        print(f"saved {horizon_path}")
        # 5. 从已落盘的 CSV 生成 Fig. 12；绘图失败不影响原始记录。
        plot_script = os.path.join(os.path.dirname(os.path.abspath(entry_file)),
                                   "plot_exp2_fig12.py")
        plot_prefix = os.path.join(output_dir, f"fig12_{tag}_{stamp}")
        plot_command = [sys.executable, plot_script, path, "--out", plot_prefix]
        if getattr(result, 'disturbance_log', None):
            plot_command += ['--disturbance_log', result.disturbance_log]
        try:
            subprocess.run(
                plot_command,
                check=True,
            )
        except Exception as exc:
            print(f"[warn] automatic Fig. 12 generation failed: {exc}")
    save_motion_outputs(result, output_dir, tag,
                        stamp if not args.no_csv else time.strftime('%Y%m%d_%H%M%S'), entry_file)
    return m

def add_arguments(parser):
    """实验选项；默认单次运行不启用扫描或额外动力学对账。"""
    parser.add_argument("--no_csv", action="store_true")
    parser.add_argument("--dynamics_diag", action="store_true",
                        help="在独立数据副本上比较新旧惯性缓存的加速度预测，结束后输出 NPZ")
    parser.add_argument("--sweep_forces", type=str, default=None,
                        help="逗号分隔力扫描（同求解器，不重建）")
    parser.add_argument("--sweep_kf", type=str, default=None,
                        help="逗号分隔 kf_scale 扫描（v5 起 kf 为运行期参数，不重建求解器）")
    parser.add_argument("--sweep_tramp", type=str, default=None,
                        help="逗号分隔 t_ramp 扫描（仅参考 profile，不重建求解器）")
    parser.add_argument("--sweep_thold", type=str, default=None,
                        help="逗号分隔 t_hold 扫描（仅参考 profile，不重建求解器）")
    parser.add_argument("--sweep_fstart", type=str, default=None,
                        help="逗号分隔 f_start 扫描（每值重新收敛轻压位）")
    parser.add_argument("--repeats", type=int, default=1,
                        help="每候选重复次数（同进程 reset_world + 重收敛 + 求解器种子复位）")



def save_failure(result, tag, output_dir):
    """保存失败现场，供关节限位和动力学对账。"""
    os.makedirs(output_dir, exist_ok=True)
    failed_path = os.path.join(output_dir, f"joint_limit_failure_{tag}_{time.time_ns()}.npz")
    np.savez_compressed(failed_path, **getattr(result, "joint_limit_trace", {}))
    if result.motion_force_trace:
        from experiments.motion_trace import save_motion_trace
        save_motion_trace(result, failed_path[:-4] + '_motion_force.npz')
    if result.dynamics_trace:
        np.savez_compressed(failed_path[:-4] + "_dynamics.npz", **result.dynamics_trace)
    print(f"[joint limits] failure snapshot: {failed_path}")
