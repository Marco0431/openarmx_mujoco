#!/usr/bin/env python3
"""从独立 motion_force NPZ 重绘实际速度与法向夹持力，不修改原 Fig. 12。

用法：D:/acados_env/python.exe plot_exp2_motion_force.py path/to/motion_force.npz
新图使用本轮步前相对时刻；旧 CSV 若没有配套 sidecar，不能还原直接实测速度。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_exp2_fig12 import add_phase_spans, draw_disturbance


def read_motion_trace(path: Path):
    """只接受带有实际状态速度口径的新记录，旧 CSV 不作差分伪装。"""
    if path.suffix.lower() != '.npz':
        raise ValueError('需要本次运行的 motion_force NPZ；旧 CSV 无法还原直接实测速度。')
    if not path.is_file():
        raise FileNotFoundError(f'缺少独立速度/力记录：{path}；请使用新版本重新运行。')
    with np.load(path, allow_pickle=False) as archive:
        if 'metadata_json' not in archive:
            raise ValueError('该 NPZ 不是 motion_force_v1 实测记录。')
        metadata = json.loads(str(archive['metadata_json'].item()))
        if metadata.get('schema') != 'motion_force_v1':
            raise ValueError('该 NPZ 不是 motion_force_v1 实测记录。')
        result = {key: archive[key].copy() for key in archive.files}
    t = result['t']
    if not t.size:
        raise ValueError('速度/力记录为空：运行尚未完成第一个物理采样步。')
    expected = dict(tip_velocity_world=(len(t), 2, 3),
                    tip_velocity_reference_world=(len(t), 2, 3),
                    box_velocity_world=(len(t), 3), box_speed=(len(t),),
                    normal_force=(len(t), 2), normal_force_reference=(len(t), 2))
    for key in ('t', 'run_time'):
        times = result[key]
        if (times.shape != (len(t),) or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)):
            raise ValueError(f'采样时刻 {key} 必须是一维、有限且严格递增。')
    for key, shape in expected.items():
        if result[key].shape != shape or not np.isfinite(result[key]).all():
            raise ValueError(f'实际记录字段 {key} 的形状或有限性无效。')
    if not np.allclose(result['box_speed'], np.linalg.norm(result['box_velocity_world'], axis=1)):
        raise ValueError('箱体总速度必须是世界系 xyz 线速度的非负模长。')
    if np.any(result['normal_force'] < 0) or np.any(result['normal_force_reference'] < 0):
        raise ValueError('法向夹持力须使用非负幅值。')
    return result, metadata


def plot_motion_trace(path: Path, output: Path, disturbance_log: Path | None = None):
    """完整显示全部采样，不平滑、不降采样、不隐藏瞬态峰值。"""
    trace, metadata = read_motion_trace(path)
    t = trace['run_time']
    fig = plt.figure(figsize=(13.5, 11.), constrained_layout=True)
    grid = fig.add_gridspec(3, 2)
    velocity_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
                     fig.add_subplot(grid[1, 0])]
    box, force = fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[2, :])
    axes = [*velocity_axes, box, force]
    colors = ['#1565c0', '#d84315']
    # 每个世界轴单独比较两臂实际有符号速度，便于直接观察同轴同步性。
    for component, (name, ax) in enumerate(zip('xyz', velocity_axes)):
        for side, (label, color) in enumerate(zip(('Left', 'Right'), colors)):
            ax.plot(t, trace['tip_velocity_world'][:, side, component]*1000.,
                    color=color, lw=1., label=f'{label} actual')
        ax.set_title(f'({"abc"[component]}) End-effector tip: world {name} velocity')
        ax.set_ylabel('Linear velocity [mm/s]')
        ax.legend(fontsize=8, ncol=2, loc='lower right')
    box.plot(t, trace['box_speed']*1000., color='#6a1b9a', lw=1., label='Box COM speed (xyz norm)')
    box.set_title('(d) Box total linear speed')
    box.set_ylabel('Speed [mm/s]')
    box.set_ylim(bottom=0.)
    box.legend(fontsize=8, loc='upper right')
    for side, (name, color) in enumerate(zip(('Left', 'Right'), colors)):
        force.plot(t, trace['normal_force'][:, side], color=color, lw=1., label=f'{name} actual')
        force.plot(t, trace['normal_force_reference'][:, side], color=color, lw=1.1,
                   ls='--' if side == 0 else ':', label=f'{name} reference')
    force.set_title('(e) Each arm normal gripping force')
    force.set_ylabel('Normal force magnitude [N]')
    force.set_ylim(bottom=0.)
    force.legend(fontsize=8, ncol=2, loc='upper right')
    for ax in axes:
        # 提前退出仅留下一个有效采样时，也显示该点，避免零长度线段不可见。
        if len(t) == 1:
            for line in ax.lines:
                line.set_marker('o')
                line.set_markersize(3.)
        add_phase_spans(ax, t, trace['phase'])
        ax.set_xlabel('Run-relative time before step [s]')
        ax.grid(alpha=.25)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        if len(t) > 1:
            ax.set_xlim(t[0], t[-1])
    title = 'Experiment II: Measured motion and normal gripping force'
    if disturbance_log is not None:
        title += '\n' + draw_disturbance(axes, disturbance_log)
    # 保留控制器左右顺序；手持模型的物理臂名称在图注中明确列出。
    names = [name.replace('openarmx_', '') for name in metadata['arm_body_names']]
    fig.suptitle(title, fontsize=13)
    fig.supxlabel(f'Controller L/R: {names[0]} / {names[1]} | '
                  f'Actual state, {metadata["sample_period_s"]*1000:g} ms samples, unsmoothed', fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for extension in ('.png', '.svg'):
            target = output.with_suffix(extension)
            fig.savefig(target, dpi=220)
            print(f'saved {target}')
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('npz', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--disturbance_log', type=Path)
    args = parser.parse_args()
    plot_motion_trace(args.npz, args.out or args.npz.with_name('motion_force'), args.disturbance_log)


if __name__ == '__main__':
    main()
