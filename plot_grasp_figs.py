#!/usr/bin/env python3
"""plot_grasp_figs.py —— 把 --record 存下的时间序列画成三张图

(a) 位置误差：左/右臂端面误差 + 箱子跟随误差 (mm)
(b) 关节力矩：左右臂各 7 关节力矩 (N·m)（左实线 / 右虚线）
(c) 两臂作用力：左右垫片接触法向力 F_L、F_R 及力差 F_L−F_R (N)

用法：
    python ikforce_grasp_y.py --headless --steps 8000 --record grasp_rec_y.npz
    python plot_grasp_figs.py grasp_rec_y.npz --out results/control --dist_t 6.0 --dist_dur 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="绘制 ikforce_grasp 时间序列：误差/关节力矩/两臂作用力差。")
    ap.add_argument("npz", nargs="?", default="grasp_rec.npz", help="--record 保存的 .npz")
    ap.add_argument("--out", default="results/control", help="输出目录（存 svg + png）")
    ap.add_argument("--dist_t", type=float, default=6.0, help="扰动起始时刻（画阴影）")
    ap.add_argument("--dist_dur", type=float, default=0.5, help="扰动持续（画阴影）")
    ap.add_argument("--t0", type=float, default=None, help="只取 [t0, t1] 段并以其为 0 时刻（如 --t0 2 --t1 4）")
    ap.add_argument("--t1", type=float, default=None, help="同上")
    ap.add_argument("--suffix", type=str, default="", help="输出文件名后缀（如 _2to4）")
    args = ap.parse_args()

    d = np.load(args.npz)
    t = d["t"]
    err_l, err_r, follow = d["err_l"], d["err_r"], d["follow"]
    tau_l, tau_r = d["tau_l"], d["tau_r"]
    f_l, f_r = d["f_l"], d["f_r"]

    if args.t0 is not None or args.t1 is not None:
        lo = args.t0 if args.t0 is not None else float(t[0])
        hi = args.t1 if args.t1 is not None else float(t[-1])
        mask = (t >= lo) & (t <= hi)
        t, err_l, err_r, follow = t[mask], err_l[mask], err_r[mask], follow[mask]
        tau_l, tau_r, f_l, f_r = tau_l[mask], tau_r[mask], f_l[mask], f_r[mask]
        t = t - lo                                     # 窗口起点作为 0 时刻
        print(f"🔬 仅画 [{lo:.2f}s, {hi:.2f}s] 段（{lo:.2f}s 记为 0s），共 {len(t)} 点")

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 12), sharex=True)
    # ---- (a) 误差 ----
    ax = axes[0]
    ax.plot(t, err_l * 1000, lw=1.0, label="left tip err", color="tab:blue")
    ax.plot(t, err_r * 1000, lw=1.0, label="right tip err", color="tab:orange")
    ax.plot(t, follow * 1000, lw=1.4, ls="--", label="box follow err", color="tab:green")
    ax.set_ylabel("error (mm)")
    ax.set_title("(a) Position errors")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    # ---- (b) 关节力矩 ----
    ax = axes[1]
    for i in range(tau_l.shape[1]):
        ax.plot(t, tau_l[:, i], lw=0.8, color="tab:blue", alpha=0.75)
        ax.plot(t, tau_r[:, i], lw=0.8, ls="--", color="tab:orange", alpha=0.75)
    ax.set_ylabel("joint torque (N\u00b7m)")
    ax.set_title("(b) Joint torques (left solid / right dashed, 7 joints each)")
    ax.grid(alpha=0.3)
    # ---- (c) 两臂作用力 & 力差 ----
    ax = axes[2]
    ax.plot(t, f_l, lw=1.0, label="F_left", color="tab:blue")
    ax.plot(t, f_r, lw=1.0, label="F_right", color="tab:orange")
    ax.plot(t, f_l - f_r, lw=1.5, label="F_left \u2212 F_right", color="k")
    ax.axhline(0.0, color="gray", lw=0.6)
    ax.set_ylabel("force (N)")
    ax.set_xlabel("time (s)")
    ax.set_title("(c) Pad contact forces & difference (force sync error)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    # ---- 扰动阴影（裁剪模式下跟随平移；超出窗口则不画）----
    win_lo = float(args.t0) if args.t0 is not None else 0.0
    shade_lo, shade_hi = args.dist_t - win_lo, args.dist_t + args.dist_dur - win_lo
    if args.t0 is not None and (shade_hi < float(t[0]) or shade_lo > float(t[-1])):
        shade_lo, shade_hi = None, None
    for ax in axes:
        if shade_lo is not None:
            ax.axvspan(shade_lo, shade_hi, color="r", alpha=0.18, label="disturbance" if ax is axes[0] else None)
        ax.set_xlim(float(t[0]), float(t[-1]))

    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    svg = out / f"control_grasp_err_tau_fdiff{args.suffix}.svg"
    png = out / f"control_grasp_err_tau_fdiff{args.suffix}.png"
    fig.savefig(svg)
    fig.savefig(png, dpi=160)
    print(f"✅ 已保存: {svg}")
    print(f"✅ 已保存: {png}")


if __name__ == "__main__":
    main()