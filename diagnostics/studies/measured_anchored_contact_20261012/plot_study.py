"""Create three-law physical and joint-limit comparison plots from saved NPZs."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

ROOT = Path(r'D:\RobotSim\results\measured_anchored_contact_20261012')
FIG = ROOT / 'figures'
LAWS = ('smooth_positive', 'ideal_unilateral', 'measured_anchored_ideal')
COLORS = {'smooth_positive': '#466b99', 'ideal_unilateral': '#c45c35',
          'measured_anchored_ideal': '#35966f'}
LABELS = {'smooth_positive': 'Smooth positive', 'ideal_unilateral': 'Ideal unilateral',
          'measured_anchored_ideal': 'Measured anchored ideal'}


def load_npz(path):
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def slug(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('_')


def case_filename(run):
    spec = run['scenario']
    dist_part = 'hold' if spec['hold_only'] else f"dist{int(round(abs(float(spec['y_dist'])) * 1000))}mm"
    return slug(f"{run['scene']}_etaCop{spec['eta_cop_y']}_etaF{spec['eta_f_x']}_{dist_part}_{spec['case']}")


def reference_offset(t, spec):
    t = np.asarray(t, dtype=float)
    offset = np.zeros_like(t)
    if spec['hold_only']:
        return offset
    distance, speed = abs(float(spec['y_dist'])), .01
    base = distance / speed if distance else 0.
    ramp = min(.20, .5*base) if base else 0.
    start = float(spec['t_ramp']) + float(spec['t_hold'])
    tau = t - start
    duration = base + ramp
    up, cruise, down = (tau >= 0)&(tau < ramp), (tau >= ramp)&(tau < base), (tau >= base)&(tau < duration)
    su = tau[up]/ramp if ramp else np.zeros(0)
    offset[up] = -speed*ramp*(su**3-.5*su**4)
    offset[cruise] = -speed*(tau[cruise]-.5*ramp)
    sd = (tau[down]-base)/ramp if ramp else np.zeros(0)
    offset[down] = -speed*(base-.5*ramp+ramp*(sd-sd**3+.5*sd**4))
    offset[t >= start+duration] = -distance
    return offset


def vertical_tilt_deg(rotvec):
    rv = np.asarray(rotvec, dtype=float)
    if rv.ndim != 2 or rv.shape[-1] != 3:
        return np.array([])
    angle = np.linalg.norm(rv, axis=1)
    scale = np.zeros_like(angle)
    nz = angle > 1e-12
    scale[nz] = np.sin(angle[nz]/2.)/angle[nz]
    qx, qy = rv[:, 0]*scale, rv[:, 1]*scale
    return np.degrees(np.arccos(np.clip(1.-2.*(qx*qx+qy*qy), -1., 1.)))


def plot_case(group):
    if not group:
        return None
    first = group[0]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    force_ax, track_ax, descent_ax, tilt_ax, margin_ax, sync_ax = axes.ravel()
    spec = first['scenario']
    initial_doc = json.loads(Path(first['evidence_prefix'] + '.reference.json').read_text(encoding='utf-8-sig'))
    initial_box = np.asarray(initial_doc['box_initial_position'], dtype=float)
    any_data = False
    for run in sorted(group, key=lambda r: LAWS.index(r['law']) if r['law'] in LAWS else 99):
        folder = Path(run['output_dir'])
        physics = load_npz(folder / 'physics.npz')
        joint = load_npz(folder / 'joint_limits.npz')
        if 't' not in physics:
            continue
        any_data = True
        law = run['law']; color = COLORS.get(law, '#555'); label = LABELS.get(law, law)
        t = np.asarray(physics['t'], dtype=float)
        force = np.asarray(physics.get('force', np.empty((0, 2))), dtype=float)
        box = np.asarray(physics.get('box', np.empty((0, 3))), dtype=float)
        if force.ndim == 2 and len(force) == len(t):
            force_ax.plot(t, force[:, 0], color=color, lw=1.15, label=label + ' L')
            force_ax.plot(t, force[:, 1], color=color, lw=1.15, ls='--', label=label + ' R')
        if box.ndim == 2 and len(box) == len(t):
            offset = reference_offset(t, spec)
            yerr = ((box[:, 1]-initial_box[1])-offset)*1000.
            track_ax.plot(t, yerr, color=color, lw=1.25, label=label)
            descent_ax.plot(t, (initial_box[2] - box[:, 2]) * 1000., color=color, lw=1.25, label=label)
            rv = np.asarray(physics.get('rotvec', []), dtype=float)
            if rv.ndim == 2 and len(rv) == len(t):
                tilt_ax.plot(t, vertical_tilt_deg(rv), color=color, lw=1.15, label=label)
        tip_l, tip_r = physics.get('tipL'), physics.get('tipR')
        if tip_l is not None and tip_r is not None:
            sync = (np.asarray(tip_l)[:, 1] - np.asarray(tip_r)[:, 1]) * 1000.
            sync_ax.plot(t, sync, color=color, lw=1.2, label=label)
        lower, upper = joint.get('lower_margin'), joint.get('upper_margin')
        jt = joint.get('t')
        if lower is not None and upper is not None and jt is not None:
            margin = np.minimum(np.asarray(lower), np.asarray(upper))
            margin_ax.plot(jt, np.rad2deg(np.min(margin, axis=1)), color=color,
                           lw=1.15, label=label)
    if not any_data:
        plt.close(figure)
        return None
    force_ax.axhline(10., color='#444', ls=':', lw=.9, label='10 N reference')
    track_ax.axhline(0., color='#444', ls=':', lw=.9, label='zero tracking error')
    for axis, title, ylabel in (
        (force_ax, 'Measured physical normal force at every 2 ms step', 'Force (N)'),
        (track_ax, 'Box y tracking error against timed reference', 'Error (mm)'),
        (descent_ax, 'Box descent from initial height', 'Descent (mm)'),
        (tilt_ax, 'Box z-axis deviation from world vertical', 'Tilt (degrees)'),
        (margin_ax, 'Minimum actual joint-limit margin', 'Margin (degrees)'),
        (sync_ax, 'Left-right pad tip y difference', 'Difference (mm)'),
    ):
        axis.set(title=title, xlabel='Time (s)', ylabel=ylabel)
        axis.grid(alpha=.25); axis.legend(fontsize=7, ncol=2)
        axis.yaxis.set_major_locator(MaxNLocator(7))
        axis.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ramp_end = float(spec['t_ramp'])
    hold_end = ramp_end + float(spec['t_hold'])
    move_end = hold_end
    if not spec['hold_only']:
        distance = abs(float(spec['y_dist']))
        base = distance/.01 if distance else 0.
        move_end += base + min(.20, .5*base)
    for axis in axes.ravel():
        axis.axvline(ramp_end, color='#777', ls=':', lw=.8,
                     label=f'ramp end {ramp_end:g}s' if axis is force_ax else None)
        axis.axvline(hold_end, color='#aaa', ls='--', lw=.8,
                     label=f'hold end {hold_end:g}s' if axis is force_ax else None)
        if not spec['hold_only']:
            axis.axvline(move_end, color='#777', ls='-.', lw=.8,
                         label=f'move end {move_end:g}s' if axis is force_ax else None)
    scene = first['scene']
    dist_mm = 0. if spec['hold_only'] else -abs(float(spec['y_dist'])) * 1000.
    figure.suptitle(f"{scene}: eta_cop={spec['eta_cop_y']}, eta_f={spec['eta_f_x']}, "
                    f"command={dist_mm:g} mm | three fixed contact laws", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .96))
    name = case_filename(first)
    for suffix in ('png', 'svg'):
        figure.savefig(FIG / f'{name}_comparison.{suffix}', dpi=180)
    plt.close(figure)
    return name


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'run_manifest.json').read_text(encoding='utf-8-sig'))
    groups = {}
    for run in manifest.get('runs', []):
        if run.get('status') in ('completed', 'failed', 'running'):
            groups.setdefault(run['case'], []).append(run)
    names = [name for group in groups.values() if (name := plot_case(group))]
    body = ['<!doctype html><html lang="en"><meta charset="utf-8">',
            '<title>Measured anchored contact comparison</title>',
            '<style>body{max-width:1250px;margin:28px auto;font:15px system-ui;line-height:1.5}img{width:100%;border:1px solid #ddd}h2{margin-top:42px}</style>',
            '<h1>Fixed three-law contact comparison</h1>',
            '<p>All three curves use the same saved plant, initial state, and scenario when the recorded MJB/reference fidelity checks pass. Physical force comes from the 2 ms closed-loop plant log.</p>']
    for name in sorted(names):
        body.append(f'<h2>{html.escape(name)}</h2><a href="{name}_comparison.svg"><img src="{name}_comparison.png"></a>')
    body.append('</html>')
    index = FIG / 'index.html'
    index.write_text('\n'.join(body), encoding='utf-8')
    print(str(index))


if __name__ == '__main__':
    main()
