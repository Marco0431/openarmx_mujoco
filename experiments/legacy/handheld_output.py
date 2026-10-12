"""实验统计、CSV/NPZ导出及辅助命令行；不参与控制求解。"""
import csv,json,os,subprocess,sys,time
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
NU = 14


def settling_command(data, arms, torque_lo, torque_hi):
    """旧稳定阶段实验的力矩命令；默认 settle_time=0，不执行此分支。"""
    return np.concatenate([np.clip(
        data.qfrc_bias[a['dofs']] - data.qfrc_constraint[a['dofs']],
        torque_lo[7*i:7*i+7], torque_hi[7*i:7*i+7]) for i,a in enumerate(arms)])

def summarize(log, args, initial_box_y, move_duration=None, *, box_center, hold_box_y_tol, motion_timing, physics_dt) -> dict:
    tt = np.array([r["t"] for r in log])
    box = np.array([r["box"] for r in log])
    rv = np.array([r["rotvec"] for r in log])
    tip_l = np.array([r["tipL"] for r in log])
    tip_r = np.array([r["tipR"] for r in log])
    fl = np.array([r["FL"] for r in log]); fr = np.array([r["FR"] for r in log])
    f_ref = np.array([r["f_ref"] for r in log])
    w_l_t = np.array([r["wL"][1] for r in log])
    w_r_t = np.array([r["wR"][1] for r in log])
    box_t = np.array([r["box_t"] for r in log])
    u_clip = np.array([r["u_clip_mask"] for r in log], dtype=bool)
    u_sat = np.array([r["u_sat_mask"] for r in log], dtype=bool)
    motion_commanded = (not args.hold_only and abs(float(args.y_dist)) > 1e-9
                        and abs(float(args.v_y)) > 1e-9)
    t_hold_end = args.t_ramp + args.t_hold
    commanded_distance = abs(float(args.y_dist)) if motion_commanded else 0.0
    speed = abs(float(args.v_y))
    if move_duration is None:
        move_duration = motion_timing(commanded_distance, args.v_y)[2]
    # phase_profile 的参考偏移恒为 -distance（世界 -y 向），命令位移为 -y。
    ref_offset_y_final_m = -commanded_distance
    ref_offset_y_final_mm = ref_offset_y_final_m * 1000.0
    # hold 全窗（worst-case 评估用）：hold_only = [t_ramp, 结束]；moving = [t_ramp, hold_end]
    if not motion_commanded:
        sel = tt >= max(0.0, tt[-1] - 0.3)
        sel_hold = tt >= args.t_ramp
        sel_move = np.zeros_like(tt, dtype=bool)
    elif motion_commanded:
        sel = (tt >= t_hold_end - 0.3) & (tt <= t_hold_end)
        sel_hold = (tt >= args.t_ramp) & (tt <= t_hold_end)
        # move 统计窗：仅 [t_hold_end, t_hold_end + move_duration]。
        # 旧实现用 phase 标签 == "move"，把参考冻结后（tau >= segment）和
        # 命令结束后 0.4s 尾巴都算进 travel，导致统计窗口错位。
        sel_move = (tt >= t_hold_end) & (tt < t_hold_end + move_duration)
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
    # 带符号搬运行程（+y 方向为正；命令为 -y，所以完美搬运应约为 -distance）。
    box_y_move_travel_mm = (float(box_y_move_delta_mm[-1])
                            if box_y_move_delta_mm.size else float("nan"))
    # 方向断言：箱子必须朝 -y（负向）移动，反向移动时 direction_ok = 0。
    box_y_move_direction_ok = int(
        np.isfinite(box_y_move_travel_mm) and box_y_move_travel_mm <= -0.5)
    # 带符号/带方向的行程误差：|实际位移 - 参考终值(-y)|。
    # 旧实现 abs(travel - |y_dist|) 把"完美搬运(-50mm)"评成 100mm 误差、
    # 把"反向 +70mm"评成 20mm 误差——符号颠倒。
    box_y_move_travel_error_mm = (abs(box_y_move_travel_mm - ref_offset_y_final_mm)
                                  if np.isfinite(box_y_move_travel_mm) else float("nan"))
    control_rows = [r for r in log if np.isfinite(r["solver_status"])]
    statuses = np.array([r["solver_status"] for r in control_rows], dtype=float)
    qp_iters = np.array([r["qp_iter"] for r in control_rows], dtype=float)
    status_bad = statuses[(statuses >= 0) & (statuses != 0)]
    move_observed = bool(sel_move.any())
    expected_move_end = t_hold_end + move_duration
    box_y_final_error_mm = abs(
        (box[-1, 1] - initial_box_y) * 1000.0 - ref_offset_y_final_mm)
    # The plant is still stepped for a short settling tail after the commanded
    # translation.  Evaluate the endpoint at the end of the commanded motion,
    # rather than at the final tail sample.
    move_end_t = t_hold_end + move_duration
    endpoint_idx = int(np.argmin(np.abs(tt - move_end_t)))
    box_y_endpoint_error_mm = abs(
        (box[endpoint_idx, 1] - initial_box_y) * 1000.0 - ref_offset_y_final_mm
    )
    box_y_initial_error_mm = box_y_endpoint_error_mm

    # 逐拍跟踪误差 err(t) = (box_y(t) - y0) - ref_off(t)，ref_off 与
    # phase_profile 的 move 段一致（含 ramp，tau=segment 处收尾到 -distance）。
    def ref_offset_y_at(t):
        if not motion_commanded or t <= t_hold_end:
            return 0.0
        tau = t - t_hold_end
        base, ramp, duration = motion_timing(commanded_distance, args.v_y)
        if tau < ramp:
            ss = tau / ramp
            return -speed * ramp * (ss ** 3 - 0.5 * ss ** 4)
        if tau < base:
            return -speed * (tau - 0.5 * ramp)
        if tau < duration:
            s = (tau - base) / ramp
            return -speed * (base - 0.5*ramp + ramp*(s-s**3+0.5*s**4))
        return -commanded_distance

    track_err_mm = np.full(box_y_move_delta_mm.shape, np.nan)
    if sel_move.any():
        track_err_mm = (
            (box[sel_move, 1] - initial_box_y - np.array(
                [ref_offset_y_at(t) for t in tt[sel_move]])) * 1000.0)
    box_y_move_track_err_max_mm = (
        float(np.max(np.abs(track_err_mm))) if track_err_mm.size else float("nan"))
    box_y_move_track_err_rms_mm = (
        float(np.sqrt(np.mean(track_err_mm ** 2))) if track_err_mm.size else float("nan"))

    # z 下沉门限：以 hold 窗平均箱体 z 为基准，move 窗平均 z 的下沉量（正=下沉）。
    z_hold_base_m = mean_or_nan(box[sel_hold, 2])
    z_move_mean_m = mean_or_nan(box[sel_move, 2])
    box_z_sink_move_mm = (z_hold_base_m - z_move_mean_m) * 1000.0

    # 评价固定时间切换后的实际搬运行程。
    move_complete_after_settle = int(
        move_observed and tt[-1] + physics_dt >= expected_move_end
        and np.isfinite(box_y_move_travel_error_mm)
        and box_y_move_travel_error_mm <= 2.0
        and bool(box_y_move_direction_ok)
        and np.isfinite(box_z_sink_move_mm)
        and box_z_sink_move_mm <= 5.0
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
        "control_mode": "pure_nmpcc_from_t0",
        "initial_FL_N": float(fl[0]), "initial_FR_N": float(fr[0]),
        "initial_f_ref_N": float(f_ref[0]),
        "FL_max_step_N": max_or_nan(np.abs(np.diff(fl))),
        "FR_max_step_N": max_or_nan(np.abs(np.diff(fr))),
        "contact_loss_samples": int(np.sum((fl < 0.1) | (fr < 0.1))),
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
        "box_y_move_direction_ok": box_y_move_direction_ok,
        "box_y_ref_offset_final_mm": ref_offset_y_final_mm,
        "box_y_move_track_err_max_mm": box_y_move_track_err_max_mm,
        "box_y_move_track_err_rms_mm": box_y_move_track_err_rms_mm,
        "box_z_sink_move_mm": box_z_sink_move_mm,
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
    # 1. 人可读摘要；完整物理步、预测和退出现场由 save_run 另行保存。
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
          f"F=({m['FL_mean']:.2f},{m['FR_mean']:.2f})N diff={m['F_diff_max_N']:.2f}N "
          f"step=({m['FL_max_step_N']:.2f},{m['FR_max_step_N']:.2f})N "
          f"loss={m['contact_loss_samples']} | "
          f"ysync={m['y_sync_mean_mm']:.2f}/{m['y_sync_max_mm']:.2f}mm | "
          f"status_bad={m['solver_status_bad_count']} qp={m['qp_iter_mean']:.1f} "
          f"solve={m['solve_ms_mean']:.2f}ms ({dt:.1f}s)")
    if not args.no_csv:
        # 2. 保持旧轨迹 CSV 的列顺序，供现有 Fig. 12 绘图脚本使用。
        os.makedirs(output_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"exp2_latest_{tag}_{stamp}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "phase", "ramp_s", "moving_active", "f_ref",
                        "boxx", "boxy", "boxz", "q_w", "q_x", "q_y", "q_z",
                        "rvx", "rvy", "rvz",
                        "tipLx", "tipLy", "tipLz", "tipRx", "tipRy", "tipRz",
                        "FLx", "FLy", "FLz", "FLtx", "FLty", "FLtz",
                        "FRx", "FRy", "FRz", "FRtx", "FRty", "FRtz",
                        "BCFx", "BCFy", "BCFz", "BCtx", "BCty", "BCtz",
                        "FL", "FR", *[f"u{i + 1}" for i in range(NU)],
                        *[f"uclip{i + 1}" for i in range(NU)],
                        *[f"usat{i + 1}" for i in range(NU)],
                        "solve_ms", "solver_status", "qp_iter",
                        "solver_qp_ms", "solver_sim_ms", "solver_lin_ms",
                        "refLy", "refRy", "vrefLy", "vrefRy", "box_vy"])
            for r in log:
                w.writerow([r["t"], r["phase"], r["ramp_s"], r["moving_active"],
                            r["f_ref"],
                            *r["box"], *r["quat"], *r["rotvec"],
                            *r["tipL"], *r["tipR"], *r["wL"][0], *r["wL"][1],
                            *r["wR"][0], *r["wR"][1], *r["box_f"], *r["box_t"],
                            r["FL"], r["FR"], *r["u_cmd"],
                            *r["u_clip_mask"].astype(int), *r["u_sat_mask"].astype(int),
                            r["solve_ms"], r["solver_status"],
                            r["qp_iter"], r["solver_qp_ms"], r["solver_sim_ms"],
                            r["solver_lin_ms"],
                            r["refLy"], r["refRy"], r["vrefLy"], r["vrefRy"],
                            r["box_vy"]])
        print(f"saved {path}")
        # 3. CSV 完成后调用绘图；失败时保留可重新绘图的原始记录。
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

def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    return np.asarray(value).tolist()


def archive_sources(output, base_dir, entry_file, scene_path):
    """保留包目录结构，避免同名 data.py、__init__.py 在归档时互相覆盖。"""
    base_dir = Path(base_dir).resolve()
    source_dir = Path(output) / 'source'
    source_dir.mkdir(exist_ok=True)
    sources = [Path(entry_file), base_dir / 'ik.py', base_dir / 'grasp_pose_static.py', Path(scene_path)]
    # 新图可从归档单独重绘；它复用原图的阶段和外力标注工具。
    sources.extend(path for name in ('plot_exp2_fig12.py', 'plot_exp2_motion_force.py')
                   if (path := base_dir / name).is_file())
    package_roots = []
    for package in ('nmpcc', 'experiments', 'diagnostics'):
        package_dir = base_dir / package
        if not package_dir.is_dir():
            package_dir = base_dir.parent / package
        package_roots.append(package_dir.resolve())
        sources.extend(package_dir.rglob('*.py'))
    hashes = {}
    while sources:
        path = sources.pop().resolve()
        if str(path) in hashes:
            continue
        payload = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(payload).hexdigest()
        if path.is_relative_to(base_dir):
            relative = path.relative_to(base_dir)
        else:
            # 便携包把公共实现放在场景目录上一级；只归档已知公共包。
            package_dir = next((root for root in package_roots if path.is_relative_to(root)), None)
            if package_dir is None:
                raise ValueError(f'source is outside the scene and shared packages: {path}')
            relative = Path(package_dir.name) / path.relative_to(package_dir)
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        if path.suffix == '.xml':
            sources.extend(path.parent / e.attrib['file'] for e in ET.fromstring(payload).iter('include'))
    return hashes


def save_run(output, args, model, data, arms, ids, result, reference, log, horizon, metrics, error, *, entry_file, base_dir, box_center, kx, rest_gap, scene_path):
    """退出时一次写盘，包含启动失败现场和全部节点参数。"""
    # 1. 参数与物理模型说明：读日志时无需重新加载 MuJoCo。
    metadata = dict(args=vars(args), model_version="typed_nmpcc_v1",
        source_sha256={}, error=error, metrics=metrics,
        box_center=box_center, kx=kx, rest_gap=rest_gap,
        rest_gap_convention='pad_center_calibration',
        actual_box_mass=model.body_mass[ids['box_body']],
        actual_box_friction=model.geom_friction[ids['box_geom']],
        actual_pad_friction=[model.geom_friction[a['pad_id']] for a in arms],
        contact_solref=model.geom_solref[ids['box_geom']],
        move_duration=reference.move_duration,
        arms=[dict(side=a['side'], joints=[model.joint(j).name for j in a['joint_ids']],
                   joint_ids=a['joint_ids'], qpos=a['qpos'], dofs=a['dofs'],
                   act_ids=a['act_ids']) for a in arms])
    # 先写原始现场，再归档源码；即使源码复制失败，已有诊断文件仍可检查。
    if result is not None:
        metadata.update(result.metadata)
        metadata['first_solver_failure'] = result.failure
        np.savez_compressed(output / 'joint_limits.npz', **getattr(result, 'joint_limit_trace', {}))
        np.savez_compressed(output / 'physics.npz', **getattr(result, 'physics_trace', {}))
        if result.motion_force_trace:
            from experiments.motion_trace import save_motion_trace
            save_motion_trace(result, output / 'motion_force.npz')
        warmups = getattr(result, 'startup', [])
        if warmups:
            np.savez_compressed(output / 'startup.npz',
                **{k: np.asarray([row[k] for row in warmups]) for k in warmups[0]})
        if args.dynamics_diag:
            np.savez_compressed(output / 'dynamics.npz', **result.dynamics_trace)
        if not horizon:
            horizon = getattr(result, 'horizon', [])
        trace = getattr(result, 'joint_limit_trace', {})
        physics = getattr(result, 'physics_trace', {})
        if len(trace.get('t', [])) and len(physics.get('t', [])):
            limited = trace['active'].any(axis=1)
            lost = (physics['force'] <= 0).any(axis=1)
            metadata['physical_checks'] = dict(
                limit_joint_steps=int(trace['active'].sum()),
                zero_contact_side_steps=int((physics['force'] <= 0).sum()),
                actual_violation_rad=float(max(0.0, -min(
                    trace['lower_margin'].min(), trace['upper_margin'].min()))),
                future_node_violation_rad=(float(trace['pred_joint_violation'][:, 1:].max())
                    if horizon else float('nan')),
                passed_without_limit_or_contact_loss=not bool(limited.any() or lost.any()))
            events = np.flatnonzero(limited | lost)
            if events.size:
                i = int(events[0])
                event = {k: physics[k][i] for k in ('t', 'qpos', 'qvel', 'u', 'box', 'force')}
                event.update(q=trace['q'][i], v=trace['v'][i],
                    limit_active=trace['active'][i], limit_torque=trace['limit_torque'][i])
                if horizon:
                    j = max(0, int(np.searchsorted([h['t'] for h in horizon], physics['t'][i], side='right')-1))
                    event.update(parameters=horizon[j]['parameters'], prediction=horizon[j]['state'],
                                 control_time=horizon[j]['t'])
                np.savez_compressed(output / 'first_physical_event.npz', **event)
            print(f"[physical checks] {metadata['physical_checks']}")
    # 3. 预测与观测按字段堆叠，NPZ 字段名和数组轴顺序保持兼容。
    if horizon:
        packed = {k: np.asarray([r[k] for r in horizon]) for k in horizon[0] if k != 'contact_load'}
        if 'contact_load' in horizon[0]:
            packed.update({'contact_' + k: np.asarray([r['contact_load'][k] for r in horizon])
                           for k in horizon[0]['contact_load']})
        np.savez_compressed(output / 'prediction.npz', **packed)
    if log:
        # Wrenches are two 3-vectors; every field is a numeric array or string.
        np.savez_compressed(output / 'trajectory.npz',
                            **{k: np.asarray([r[k] for r in log]) for k in log[0]})
        if not args.no_csv:
            with (output / 'trajectory.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['t', 'phase', 'box_x', 'box_y', 'box_z', 'FL', 'FR', 'solver_status'])
                writer.writerows([r['t'], r['phase'], *r['box'], r['FL'], r['FR'], r['solver_status']] for r in log)
    # 4. 退出现场和源码清单最后写入，覆盖正常结束及异常退出。
    np.savez_compressed(output / 'exit_state.npz', qpos=data.qpos, qvel=data.qvel,
                        ctrl=data.ctrl, time=data.time)
    metadata['source_sha256'] = archive_sources(output, base_dir, entry_file, scene_path)
    (output / 'summary.json').write_text(json.dumps(metadata, default=_json_value,
        ensure_ascii=False, indent=2), encoding='utf-8')


def add_arguments(parser):
    """实验选项；默认单次运行不启用扫描或额外动力学对账。"""
    parser.add_argument("--ed_rt", type=float, default=None,
                        help="运行时覆盖 eta_dev（cost_set W，不重建）；None=用构建值")
    parser.add_argument("--ed_x_rt", type=float, default=None,
                        help="运行时仅覆盖 epsd 的 x 轴分量权重（前向压缩弛豫通道，"
                             "cost_set W，不重建）；None=同 eta_dev")
    parser.add_argument("--eu_rt", type=float, default=None,
                        help="运行时覆盖 eta_u（cost_set W，不重建）；None=用构建值")
    parser.add_argument("--Q_rt", type=float, default=None,
                        help="运行时覆盖 Q（cost_set W，不重建）；None=用构建值")
    parser.add_argument("--P_rt", type=float, default=None,
                        help="运行时覆盖 P（cost_set W_e，不重建）；None=用构建值")
    parser.add_argument("--debug_eps", action="store_true",
                        help="每控制拍打印 eps/epsd/u 残差（诊断）")
    parser.add_argument("--no_csv", action="store_true")
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
    parser.add_argument('--dynamics_diag', action='store_true')


