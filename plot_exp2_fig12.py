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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    add_limit(ax_d, args.force_limit_n, r"$\pm 2$ N limit", "#b71c1c")
    ax_d.axhline(0.0, color="#555555", lw=0.7)
    add_phase_spans(ax_d, t, d["phase"])
    ax_d.set_xlabel("t [s]")
    ax_d.set_ylabel(r"$F_L-F_R$ [N]")
    ax_d.set_title("(d) Contact-force synchronization error")
    ax_d.grid(alpha=0.3)
    ax_d.legend(fontsize=8, loc="best")
    ax_d.text(0.99, 0.04, f"max |error| = {np.max(np.abs(force_error_n)):.2f} N",
              transform=ax_d.transAxes, ha="right", va="bottom", fontsize=8,
              bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

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