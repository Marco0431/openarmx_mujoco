#!/usr/bin/env python3
"""Plot four panels in the requested Experiment-II Fig. 12 layout.

(a) Both end-effectors in the world xy plane.
(b) y-position synchronization error in mm.
(c) Three representative left-arm torques: shoulder, elbow, wrist.
(d) Contact-force synchronization error F_L - F_R in N.

Example:
    D:\\acados_env\\python.exe plot_exp2_fig12.py \\
        results/exp2_latest/exp2_latest_single_20260826_225436.csv \\
        --out results/exp2_latest/fig12.png

The CSV remains the source of truth. The script does not smooth or alter data.
Use --torque_indices to select other left-arm actuators (1-based indices 1..7).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np


PHASE_COLORS = {"ramp": "#fff2cc", "hold": "#e2f0d9", "move": "#d9eaf7"}
PHASE_LABELS = {"ramp": "Ramp", "hold": "Holding", "move": "Moving"}


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")

    def col(name: str) -> np.ndarray:
        if name not in rows[0]:
            raise KeyError(f"Missing CSV column: {name}")
        return np.asarray([float(r[name]) for r in rows], dtype=float)

    return {
        "t": col("t"),
        "phase": np.asarray([r.get("phase", "") for r in rows]),
        "tipLx": col("tipLx"), "tipLy": col("tipLy"),
        "tipRx": col("tipRx"), "tipRy": col("tipRy"),
        "FL": col("FL"), "FR": col("FR"),
        **{f"u{i}": col(f"u{i}") for i in range(1, 15)},
    }


def add_phase_spans(ax, t: np.ndarray, phase: np.ndarray) -> None:
    if len(t) < 2:
        return
    start = 0
    for i in range(1, len(t) + 1):
        if i == len(t) or phase[i] != phase[start]:
            p = str(phase[start])
            if p in PHASE_COLORS:
                ax.axvspan(t[start], t[i - 1], color=PHASE_COLORS[p], alpha=0.55,
                           lw=0, zorder=0)
                if i - start >= 3:
                    ax.text((t[start] + t[i - 1]) / 2.0, 0.97, PHASE_LABELS[p],
                            transform=ax.get_xaxis_transform(), ha="center", va="top",
                            fontsize=9, color="#444444")
            start = i


def add_limit(ax, value: float, label: str, color: str) -> None:
    ax.axhline(value, color=color, ls="--", lw=0.9, alpha=0.8, label=label)
    ax.axhline(-value, color=color, ls="--", lw=0.9, alpha=0.8)


def refine_force_axis(ax, error: np.ndarray) -> tuple[float, float]:
    """按完整数据范围细化力误差纵轴；阈值线不再把小误差压成直线。"""
    finite = error[np.isfinite(error)]
    low = min(0.0, float(finite.min())) if finite.size else 0.0
    high = max(0.0, float(finite.max())) if finite.size else 0.0
    # 保留零点和全部峰值，不滤波、不截掉启动尖峰；常值曲线也留可读范围。
    padding = max((high - low) * 0.15, 0.005)
    limits = (low - padding, high + padding)
    ax.set_ylim(*limits)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(axis="y", which="minor", alpha=0.12, linestyle=":")
    return limits


def draw_disturbance(axes, path: Path) -> str:
    """按实际外力日志标记一次脉冲；右轴显示扰动力，左轴仍为夹持力差。"""
    with np.load(path, allow_pickle=False) as log:
        t = log['t'].copy()
        forces = log['force_world'].copy()
        meta = json.loads(str(log['metadata_json'].item()))
    if forces.shape != (len(t), 3) or not np.isfinite(forces).all() or not np.isfinite(t).all():
        raise ValueError('Invalid disturbance time/force data')
    axis = str(meta['axis'])
    force = forces[:, 'xyz'.index(axis)]
    active = np.flatnonzero(force != 0)
    target = str(meta['body']).replace('openarmx_', '').replace('_', ' ')
    if not active.size:
        return f'Disturbance: {target}, world {axis}; no nonzero force applied'
    start = max(float(meta['start_s']), float(t[0]))
    end = min(float(meta['start_s']) + float(meta['duration_s']), float(t[-1]) + float(meta['timestep']))
    peak = float(force[np.argmax(np.abs(force))])
    color = '#9c2780'
    for ax in axes:
        ax.axvspan(start, end, color=color, alpha=0.14, zorder=1)
        for boundary in (start, end):
            ax.axvline(boundary, color=color, ls=':', lw=0.85, alpha=0.8)
    right = axes[-1].twinx()
    line, = right.step(t, force, where='post', color=color, ls='--', lw=1.3,
                       label=f'Applied F{axis} (right axis)')
    span = max(float(np.ptp(force)), abs(peak), 0.01)
    right.set_ylim(min(0, float(force.min())) - 0.10 * span,
                   max(0, float(force.max())) + 0.35 * span)
    right.set_ylabel(f'Applied F{axis} [N]', color=color)
    right.tick_params(axis='y', colors=color)
    right.yaxis.set_major_locator(MaxNLocator(nbins=5))
    right.ticklabel_format(axis='y', style='plain', useOffset=False)
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles + [line], labels + [line.get_label()], fontsize=7, loc='upper right')
    partial = ' (interrupted)' if end < float(meta['start_s']) + float(meta['duration_s']) - 1e-10 else ''
    return f'Single pulse: {target}, world {axis}, peak {peak:+.3g} N | {start:g}-{end:g} s{partial}'


def main() -> None:
    ap = argparse.ArgumentParser(description="Create Fig. 12(a)-(d) from exp2 CSV")
    ap.add_argument("csv", type=Path, help="exp2_nmpcc_latest CSV")
    ap.add_argument("--out", type=Path, default=None,
                    help="output PNG/SVG prefix; default next to CSV as fig12_exp2")
    ap.add_argument("--torque_indices", default="2,4,7",
                    help="three left-arm actuator indices, 1-based; default 2,4,7")
    ap.add_argument("--y_limit_mm", type=float, default=2.0,
                    help="synchronization limit for panel (b), default 2 mm")
    ap.add_argument("--force_limit_n", type=float, default=2.0,
                    help="force synchronization limit for panel (d), default 2 N")
    ap.add_argument("--title", default="Experiment II: Cooperative grasping and transport")
    ap.add_argument('--disturbance_log', type=Path, default=None,
                    help='actual disturbance NPZ; mark pulse interval and plot external force on right axis')
    args = ap.parse_args()

    indices = [int(x.strip()) for x in args.torque_indices.split(",")]
    if len(indices) != 3 or any(i < 1 or i > 7 for i in indices):
        ap.error("--torque_indices must contain exactly three values in 1..7")

    d = read_csv(args.csv)
    t = d["t"]
    y_sync_mm = (d["tipLy"] - d["tipRy"]) * 1000.0
    force_error_n = d["FL"] - d["FR"]
    torque_labels = {indices[0]: "Shoulder", indices[1]: "Elbow", indices[2]: "Wrist"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # (a) world xy trajectories
    ax_a.plot(d["tipLx"], d["tipLy"], color="#1565c0", lw=1.5, label="Left arm tip")
    ax_a.plot(d["tipRx"], d["tipRy"], color="#d84315", lw=1.5, label="Right arm tip")
    ax_a.scatter(d["tipLx"][0], d["tipLy"][0], color="#1565c0", s=28, marker="o")
    ax_a.scatter(d["tipRx"][0], d["tipRy"][0], color="#d84315", s=28, marker="o")
    ax_a.scatter(d["tipLx"][-1], d["tipLy"][-1], color="#1565c0", s=34, marker="x")
    ax_a.scatter(d["tipRx"][-1], d["tipRy"][-1], color="#d84315", s=34, marker="x")
    ax_a.set_xlabel("x [m]")
    ax_a.set_ylabel("y [m]")
    ax_a.set_title("(a) Dual-arm end-effector positions in the xy plane")
    ax_a.legend(fontsize=8, loc="best")
    ax_a.grid(alpha=0.3)
    ax_a.set_aspect("equal", adjustable="datalim")

    # (b) y synchronization error
    ax_b.plot(t, y_sync_mm, color="#1746d1", lw=1.0, label=r"$y_L-y_R$")
    add_limit(ax_b, args.y_limit_mm, r"$\pm 2$ mm limit", "#b71c1c")
    ax_b.axhline(0.0, color="#555555", lw=0.7)
    add_phase_spans(ax_b, t, d["phase"])
    ax_b.set_xlabel("t [s]")
    ax_b.set_ylabel(r"$y_L-y_R$ [mm]")
    ax_b.set_title("(b) y-direction position synchronization")
    ax_b.grid(alpha=0.3)
    ax_b.legend(fontsize=8, loc="best")
    ax_b.text(0.99, 0.04, f"max |error| = {np.max(np.abs(y_sync_mm)):.2f} mm",
              transform=ax_b.transAxes, ha="right", va="bottom", fontsize=8,
              bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    # (c) representative left-arm torques
    styles = ["#1565c0", "#ef6c00", "#6a1b9a"]
    for idx, color in zip(indices, styles):
        ax_c.plot(t, d[f"u{idx}"], lw=1.0, color=color,
                  label=fr"$\tau_{{{idx}}}$ ({torque_labels[idx]})")
    add_phase_spans(ax_c, t, d["phase"])
    ax_c.set_xlabel("t [s]")
    ax_c.set_ylabel(r"$\tau$ [N m]")
    ax_c.set_title("(c) Left-arm representative joint torque inputs")
    ax_c.grid(alpha=0.3)
    ax_c.legend(fontsize=8, loc="best")

    # (d) force synchronization error; keep signed curve and show absolute limit
    ax_d.plot(t, force_error_n, color="#1746d1", lw=1.0, label=r"$F_L-F_R$")
    add_limit(ax_d, args.force_limit_n, f"±{args.force_limit_n:g} N limit", "#b71c1c")
    ax_d.axhline(0.0, color="#555555", lw=0.7)
    add_phase_spans(ax_d, t, d["phase"])
    ax_d.set_xlabel("t [s]")
    ax_d.set_ylabel(r"$F_L-F_R$ [N]")
    ax_d.set_title("(d) Contact-force synchronization error")
    ax_d.grid(alpha=0.3)
    low, high = refine_force_axis(ax_d, force_error_n)
    if not (low <= -abs(args.force_limit_n) and high >= abs(args.force_limit_n)):
        ax_d.text(0.01, 0.04, f"Limit: ±{args.force_limit_n:g} N (outside view)",
                  transform=ax_d.transAxes, ha="left", va="bottom", fontsize=8,
                  color="#b71c1c", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    ax_d.legend(fontsize=8, loc="best")
    ax_d.text(0.99, 0.04, f"max |error| = {np.max(np.abs(force_error_n)):.2f} N",
              transform=ax_d.transAxes, ha="right", va="bottom", fontsize=8,
              bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    if args.disturbance_log is not None:
        caption = draw_disturbance([ax_b, ax_c, ax_d], args.disturbance_log)
        fig.suptitle(args.title + '\n' + caption, fontsize=12)
    else:
        fig.suptitle(args.title, fontsize=14)
    out = args.out or args.csv.with_name("fig12_exp2")
    out.parent.mkdir(parents=True, exist_ok=True)
    png = out.with_suffix(".png")
    svg = out.with_suffix(".svg")
    fig.savefig(png, dpi=220)
    fig.savefig(svg)
    plt.close(fig)
    print(f"saved {png}")
    print(f"saved {svg}")
    print(f"y synchronization max = {np.max(np.abs(y_sync_mm)):.3f} mm")
    print(f"force synchronization max = {np.max(np.abs(force_error_n)):.3f} N")


if __name__ == "__main__":
    main()
