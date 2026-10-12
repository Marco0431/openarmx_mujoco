"""参数扫描工具；控制公式仍由各入口对应的独立控制器执行。"""
import csv,os,time

def has_sweep(args):
    return any(getattr(args, name, None) for name in
        ('sweep_forces','sweep_kf','sweep_tramp','sweep_thold','sweep_fstart'))

def run_sweep(args, run_one, output_dir):
    """按原顺序执行显式请求的扫描，并保存汇总表。"""
    sweep_forces = ([float(x) for x in args.sweep_forces.split(",")]
                    if args.sweep_forces else None)
    sweep_kfs = ([float(x) for x in args.sweep_kf.split(",")]
                 if args.sweep_kf else None)
    sweep_tramps = ([float(x) for x in args.sweep_tramp.split(",")]
                    if args.sweep_tramp else None)
    sweep_tholds = ([float(x) for x in args.sweep_thold.split(",")]
                    if args.sweep_thold else None)
    sweep_fstarts = ([float(x) for x in args.sweep_fstart.split(",")]
                     if args.sweep_fstart else None)
    kfs = sweep_kfs or [args.kf_scale]
    forces = sweep_forces or [args.force]
    tramps = sweep_tramps or [args.t_ramp]
    tholds = sweep_tholds or [args.t_hold]
    fstarts = sweep_fstarts or [args.f_start]
    rows = []
    for kf in kfs:
        for force in forces:
            for tramp in tramps:
                for thold in tholds:
                    for fstart in fstarts:
                        args.t_ramp = tramp
                        args.t_hold, args.f_start = thold, fstart
                        for rep in range(args.repeats):
                            tag = (f"f{force:g}_kf{kf:g}"
                                   f"_tr{tramp:g}_th{thold:g}_fs{fstart:g}_r{rep}")
                            rows.append(run_one(kf, force, tag))
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"sweep_hold_{stamp}.csv")
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n=== sweep summary: {path} ===")
    print("rep | force | kf | tramp | thold | fstart | "
          "tiltx_hold | tilty_hold | tiltz_hold | z_sag | z_drift_hold | "
          "FL | FR | Fdiff_hold | ysync_hold | solve_ms")
    for i, m in enumerate(rows):
        print(f"{i % args.repeats:3d} | {m['force']:5.1f} | {m['kf_scale']:.2f} | "
              f"{m['t_ramp']:.2f} | {m['t_hold']:.2f} | "
              f"{m['f_start']:.1f} | "
              f"{m['tilt_x_hold_max_deg']:5.2f} | {m['tilt_y_hold_max_deg']:5.2f} | "
              f"{m['tilt_z_hold_max_deg']:5.2f} | {m['z_sag_mm']:6.2f} | "
              f"{m['z_drift_hold_mm']:5.2f} | {m['FL_mean']:5.2f} | {m['FR_mean']:5.2f} | "
              f"{m['F_diff_hold_max_N']:5.2f} | {m['y_sync_hold_max_mm']:6.2f} | "
              f"{m['solve_ms_mean']:5.2f}")
    return rows
