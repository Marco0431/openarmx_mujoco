"""Read-only simulation audit. Recomputes every saved prediction node; never mj_step/builds a solver.

Run with D:/acados_env/python.exe -B -X utf8 audit_nodes.py [--case CASE_ID] [--limit N].
The only writes are audit artifacts under the results directory.
"""
from pathlib import Path
import argparse
import csv
from dataclasses import replace
import hashlib
import importlib
import json
import os
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import casadi as ca
import mujoco
import numpy as np

BASE = Path(r'D:/RobotSim')
ROOT = BASE / 'results/measured_anchored_contact_20261012'
WORKSPACE = BASE / 'scratch/measured_anchored_contact_20261012/workspace'
sys.path.insert(0, str(WORKSPACE))
from nmpcc.config import ControllerConfig
from nmpcc.model import build_model
from nmpcc.parameters import checked_anchor, from_legacy, parameter_size, to_legacy
from nmpcc.kinematics import mujoco_fk_symbols, pad_contact_offsets


def read_npz(path):
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def maxabs(value):
    value = np.asarray(value)
    return float(np.max(np.abs(value))) if value.size else 0.


def evaluate(fn, x, p=None, chunk=256):
    """Serial CasADi map, bounded memory; yield between chunks during the running batch."""
    outputs = []
    for begin in range(0, len(x), chunk):
        xx = x[begin:begin+chunk]
        mapped = fn.map(len(xx), 'serial')
        value = mapped(xx.T) if p is None else mapped(xx.T, p[begin:begin+chunk].T)
        outputs.append(np.asarray(value, float).T)
        time.sleep(.002)
    return np.concatenate(outputs) if outputs else np.empty((0, fn.size1_out(0)))


def geometry_for(run, summary):
    variant = run['scene']
    sys.path.insert(0, str(WORKSPACE / variant))
    path = Path(run['evidence_prefix'] + '.model.mjb')
    model = mujoco.MjModel.from_binary_path(str(path))
    # These factories only name existing model joints/bodies; no grasp or simulation.
    module = importlib.import_module('exp2_slender_scene' if variant == 'flat' else 'experiments.scenes.handheld')
    arms = module.make_arms(model)
    offsets = pad_contact_offsets(model, arms, .1001)
    kwargs = ({'contact_offsets': offsets} if variant == 'flat' else
              {'contact_local_points': [np.array([0., 0., .1001])+v for v in offsets]})
    fk = mujoco_fk_symbols(model, [dict(body_id=a['body_id'], joint_ids=a['joint_ids']) for a in arms], .1001, **kwargs)
    cfg = ControllerConfig(**summary['config'])
    symbolic = build_model(cfg, fk)
    q14 = ca.vertcat(symbolic.x[:7], symbolic.x[14:21])
    dq14 = ca.vertcat(symbolic.x[7:14], symbolic.x[21:28])
    tips = ca.substitute(fk['tips'], fk['tips_qt'], q14)
    contacts = ca.substitute(fk['contact_tips'], fk['tips_qt'], q14)
    qdot = ca.vertcat(symbolic.x[7:14], ca.SX.zeros(7), symbolic.x[21:28], ca.SX.zeros(7))
    gap = contacts[3]-contacts[0]
    velocity = ca.jacobian(tips, symbolic.x) @ qdot
    gap_dot = ca.jacobian(gap, symbolic.x) @ qdot
    jx = ca.vertcat(ca.jacobian(contacts[0], symbolic.x)[:, :7].T,
                    ca.jacobian(contacts[3], symbolic.x)[:, 14:21].T)
    geom_fn = ca.Function('offline_geometry', [symbolic.x], [ca.vertcat(tips, velocity, gap, gap_dot, jx)])
    diag = symbolic.diagnostic_fn(symbolic.x, symbolic.p)
    all_fn = ca.Function('offline_model', [symbolic.x, symbolic.p], [ca.vertcat(
        symbolic.predicted_force(symbolic.x, symbolic.p), symbolic.force_rate_fn(symbolic.x, symbolic.p),
        diag[6], diag[7], symbolic.normal_load(symbolic.x, symbolic.p))])
    return cfg, model, arms, symbolic, geom_fn, all_fn, path, fk


def numpy_force(cfg, gap, gap_dot, p):
    bias = p[:, 176:188].reshape(-1, 2, 6)
    force = bias.copy()
    rate = np.zeros_like(force)
    law = str(cfg.normal_force_law)
    if law == 'measured_anchored_ideal':
        raw = p[:, 226:228]+cfg.kx*(p[:, 225]-gap)[:, None]
        magnitude = np.maximum(raw, 0.)
        stiffness = cfg.kx*(raw > 0.)
        force[:, :, 0] = magnitude*[-1., 1.]
    else:
        compression = cfg.rest_gap-gap
        if law == 'smooth_positive':
            root = np.sqrt(compression**2+cfg.smooth_eps**2)
            positive = .5*(compression+root)
            slope = .5*(1.+compression/root)
        else:
            positive = np.maximum(compression, 0.)
            slope = (compression > 0.).astype(float)
        raw = np.repeat((cfg.kx*compression)[:, None], 2, axis=1)
        magnitude = np.repeat((cfg.kx*positive)[:, None], 2, axis=1)
        stiffness = np.repeat((cfg.kx*slope)[:, None], 2, axis=1)
        force[:, :, 0] += magnitude*[-1., 1.]
    rate[:, :, 0] = -stiffness*gap_dot[:, None]*[-1., 1.]
    return force, rate, raw, stiffness


def residual_parts(tips, velocities, force, rate, p):
    shape = (-1, 2, 6)
    pose, velocity = p[:, 126:138].reshape(shape), p[:, 138:150].reshape(shape)
    fref, fdref = p[:, 150:162].reshape(shape), p[:, 189:201].reshape(shape)
    cop, ef = p[:, 201:213].reshape(shape), p[:, 213:225].reshape(shape)
    kf = p[:, 188, None, None]
    pos = (1-ef)*(tips-(1-cop)*pose-cop*tips[:, ::-1])
    frc = -ef*kf*(force-fref)
    vel = (1-ef)*(velocities-(1-cop)*velocity-cop*velocities[:, ::-1])
    frd = -ef*kf*(rate-fdref)
    return pos, frc, vel, frd


def geometric_spotchecks(model, arms, summary, geom_fn, x, g):
    data = mujoco.MjData(model)
    data.qpos[:] = summary['initial_state']['initial_qpos']
    indices = np.unique(np.linspace(0, len(x)-1, min(12, len(x)), dtype=int))
    rows = []
    for index in indices:
        for side, arm in enumerate(arms):
            data.qpos[arm['qpos']] = x[index, :7] if side == 0 else x[index, 14:21]
        # Independent MuJoCo geometry, including actual geom local rotation/half-thickness.
        mujoco.mj_kinematics(model, data)
        faces = [data.geom_xpos[a['pad_id']]+data.geom_xmat[a['pad_id']].reshape(3, 3) @
                 np.array([0., 0., model.geom_size[a['pad_id'], 2]]) for a in arms]
        gap = faces[1][0]-faces[0][0]
        tips = np.concatenate([data.xpos[a['body_id']]+data.xmat[a['body_id']].reshape(3, 3) @
                               np.array([0., 0., .1001]) for a in arms])
        expected_tips = np.r_[g[index, :3], g[index, 6:9]]
        rows.append(dict(flat_node=int(index), gap_mujoco=float(gap), gap_symbolic=float(g[index, 24]),
                         gap_abs_error=abs(float(gap-g[index, 24])), tip_position_abs_error=maxabs(tips-expected_tips)))
    return rows


def finite_difference_checks(cfg, geom_fn, x, p, g, analytic_rate):
    # Include each side's smallest nonzero raw force to stress near-unloading nodes.
    _, _, raw, _ = numpy_force(cfg, g[:, 24], g[:, 25], p)
    indices = list(np.linspace(0, len(x)-1, min(12, len(x)), dtype=int))
    for side in range(2):
        valid = np.flatnonzero(np.abs(raw[:, side]) > 1e-7)
        if len(valid):
            indices += valid[np.argsort(np.abs(raw[valid, side]))[:8]].tolist()
    rows = []
    for index in sorted(set(indices)):
        direction = np.zeros(28)
        direction[:7], direction[14:21] = x[index, 7:14], x[index, 21:28]
        if np.linalg.norm(direction) < 1e-10:
            continue
        # Keep both ends on the same side of every hard kink; exact kink tested in core tests.
        h = 1e-5
        if str(cfg.normal_force_law) != 'smooth_positive':
            distance = float(np.min(np.abs(raw[index])))
            if distance < 1e-7:
                continue
            h = min(h, .2*distance/max(abs(cfg.kx*g[index, 25]), 1e-8))
        if h < 1e-9:
            continue
        gg = evaluate(geom_fn, np.stack((x[index]-h*direction, x[index]+h*direction)))
        ff, _, rr, _ = numpy_force(cfg, gg[:, 24], gg[:, 25], np.tile(p[index], (2, 1)))
        if str(cfg.normal_force_law) != 'smooth_positive' and np.any((rr[0] > 0.) != (rr[1] > 0.)):
            continue
        finite = (ff[1]-ff[0])/(2*h)
        rows.append(dict(flat_node=int(index), step_s=h, min_abs_raw_force=float(np.min(np.abs(raw[index]))),
                         force_rate_abs_error=maxabs(finite-analytic_rate[index])))
    return rows


def audit_trace(label, arrays, cfg, model, arms, symbolic, geom_fn, all_fn, summary, run, out, trajectory):
    xkey = 'state' if 'state' in arrays else 'x'
    required = (xkey, 'parameters_shared_full', 'force')
    missing = [k for k in required if k not in arrays]
    if missing:
        return {'status': 'unavailable', 'missing': missing}
    states, parameters = arrays[xkey], arrays['parameters_shared_full']
    if not states.size:
        return {'status': 'unavailable', 'reason': 'empty trace'}
    updates, nodes = states.shape[:2]
    x, p = states.reshape(-1, 28), parameters.reshape(-1, parameters.shape[-1])
    if p.shape[1] != parameter_size(cfg):
        raise ValueError(f'wrong parameter width {p.shape[1]} for {cfg.normal_force_law}')
    if not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError('nonfinite state or parameters; cannot claim validated formulas')
    g, symbolic_values = evaluate(geom_fn, x), evaluate(all_fn, x, p)
    tips, velocities = g[:, :12].reshape(-1, 2, 6), g[:, 12:24].reshape(-1, 2, 6)
    gap, gap_dot = g[:, 24], g[:, 25]
    force, rate, raw, stiffness = numpy_force(cfg, gap, gap_dot, p)
    pos, frc, vel, frd = residual_parts(tips, velocities, force, rate, p)
    eps, epsd = pos+frc, vel+frd
    checks, coverage = {}, {}

    def check(name, actual, expected, tol=1e-8):
        error = maxabs(np.asarray(actual)-np.asarray(expected))
        checks[name] = {'max_abs_error': error, 'tolerance': tol, 'pass': bool(np.isfinite(error) and error <= tol)}

    check('force_numpy_vs_saved', force[:, :, 0], arrays['force'].reshape(-1, 2))
    check('force_numpy_vs_symbolic', force.reshape(-1, 12), symbolic_values[:, :12])
    check('force_rate_numpy_vs_symbolic', rate.reshape(-1, 12), symbolic_values[:, 12:24])
    check('mixed_error_numpy_vs_symbolic', eps.reshape(-1, 12), symbolic_values[:, 24:36])
    check('mixed_error_rate_numpy_vs_symbolic', epsd.reshape(-1, 12), symbolic_values[:, 36:48])
    for key, value in (('force_pred_dot', rate), ('eps_x', eps), ('epsd_x', epsd)):
        coverage[key] = key in arrays
        if key in arrays:
            check(key+'_numpy_vs_saved', value.reshape(-1, 12), arrays[key].reshape(-1, 12))
    if 'reference' in arrays:
        check('references_vs_parameters', arrays['reference'].reshape(-1, 36), p[:, 126:162])
    if 'force_ref_dot' in arrays:
        check('force_reference_rate_vs_parameters', arrays['force_ref_dot'].reshape(-1, 12), p[:, 189:201])
    check('frozen_residual_all_nodes', parameters[:, :, 112:126], np.broadcast_to(parameters[:, :1, 112:126], (updates, nodes, 14)))
    check('frozen_bias_all_nodes', parameters[:, :, 176:188], np.broadcast_to(parameters[:, :1, 176:188], (updates, nodes, 12)))
    anchored = str(cfg.normal_force_law) == 'measured_anchored_ideal'
    if anchored:
        check('anchor_frozen_all_nodes', parameters[:, :, 225:], np.broadcast_to(parameters[:, :1, 225:], (updates, nodes, 3)))
        check('anchor_gap_at_current_state', gap.reshape(updates, nodes)[:, 0], parameters[:, 0, 225], 1e-10)
        check('current_force_equals_measured_anchor', force.reshape(updates, nodes, 2, 6)[:, 0, :, 0]*[-1., 1.], parameters[:, 0, 226:228])
        for key, expected in (('force_anchor_gap', parameters[:, 0, 225:226]), ('force_anchor_normal', parameters[:, 0, 226:228])):
            if key in arrays:
                check(key+'_vs_parameters', arrays[key], expected)
        for variant in ('flat', 'handheld'):
            try:
                from_legacy(to_legacy(parameters[0], variant), variant, cfg)
            except ValueError:
                checks['legacy_'+variant+'_replay_without_anchor_rejected'] = {'pass': True}
            else:
                checks['legacy_'+variant+'_replay_without_anchor_rejected'] = {'pass': False}
    if 'parameters' in arrays:
        check('legacy_view_preserves_original_slots', arrays['parameters'], to_legacy(parameters, run['scene']))
    measured_matches = []
    if label == 'prediction' and 'force_measured_control' in trajectory and 't' in trajectory:
        # Recorder.sample logs the same controller measurement independently every 10 ms.
        # Flat legacy timestamps are one physics step ahead; strict intersections occur every 30 ms.
        control_times = np.asarray(arrays['t'], float)
        sample_times = np.asarray(trajectory['t'], float)-(.002 if run['scene'] == 'flat' else 0.)
        measured = np.asarray(trajectory['force_measured_control'], float)
        pairs = []
        for sample_index, sample_time in enumerate(sample_times):
            control_index = int(np.searchsorted(control_times, sample_time-1e-8))
            if control_index < updates and abs(control_times[control_index]-sample_time) < 1e-8 and np.isfinite(measured[sample_index]).all():
                pairs.append((control_index, sample_index))
        coverage['independent_measurement_log_intersections'] = len(pairs)
        if pairs:
            ci, si = np.asarray(pairs).T
            measured_x = measured[si] if run['scene'] == 'flat' else measured[si].reshape(-1, 2, 6)[:, :, 0]
            current_force = force.reshape(updates, nodes, 2, 6)[ci, 0, :, 0]
            check('current_force_vs_independent_measured_control_log', current_force, measured_x)
            if anchored:
                check('normal_anchor_vs_independent_measured_control_log', parameters[ci, 0, 226:228], measured_x*[-1., 1.])
            if run['scene'] == 'handheld':
                check('frozen_six_axis_remainder_vs_independent_measured_control_log',
                      parameters[ci, 0, 176:188].reshape(-1, 2, 6)[:, :, 1:], measured[si].reshape(-1, 2, 6)[:, :, 1:])
            measured_matches = [dict(control_update=int(c), trajectory_index=int(s), t=float(control_times[c])) for c,s in pairs]
    # Independent Jacobian mapping; distinguish true sampled contact evidence from a model identity.
    normal = (g[:, 26:40].reshape(-1, 2, 7)*force[:, :, 0, None]).reshape(updates, nodes, 14)
    check('normal_joint_load_jacobian_vs_symbolic', normal.reshape(-1, 14), symbolic_values[:, 48:62])
    full_key = next((k for k in ('contact_full_contact', 'contact_load_full_contact') if k in arrays), None)
    normal_key = next((k for k in ('contact_model_normal', 'contact_load_model_normal') if k in arrays), None)
    residual_key = next((k for k in ('contact_frozen_residual', 'contact_load_frozen_residual') if k in arrays), None)
    coverage['sampled_full_contact'] = full_key is not None
    if full_key:
        check('current_normal_plus_frozen_residual_vs_sampled_full_contact', normal[:, 0]+parameters[:, 0, 112:126], arrays[full_key].reshape(updates, 14))
    if normal_key:
        check('current_normal_vs_saved_model_normal', normal[:, 0], arrays[normal_key].reshape(updates, 14))
    if residual_key:
        check('frozen_residual_vs_saved', parameters[:, 0, 112:126], arrays[residual_key].reshape(updates, 14))
    spot = geometric_spotchecks(model, arms, summary, geom_fn, x, g)
    check('independent_mujoco_face_gap', [r['gap_abs_error'] for r in spot], np.zeros(len(spot)), 1e-9)
    check('independent_mujoco_tip_positions', [r['tip_position_abs_error'] for r in spot], np.zeros(len(spot)), 1e-9)
    fd = finite_difference_checks(cfg, geom_fn, x, p, g, rate)
    if fd:
        check('total_force_rate_finite_difference', [r['force_rate_abs_error'] for r in fd], np.zeros(len(fd)), 2e-4)
    times = np.asarray(arrays.get('t', np.arange(updates, dtype=float)), float)
    key_indices = sorted(set([int(np.argmin(np.abs(times-t))) for t in (0., .1, .5, 1., 1.5)] + [updates-1])) if label == 'prediction' else list(range(updates))
    reshape = lambda v: np.asarray(v).reshape(updates, nodes, *np.asarray(v).shape[1:])
    np.savez_compressed(out / (label+'_nodes.npz'), **({'t': times} if label == 'prediction' else {'warmup_iteration': times}),
        state=states, parameters_shared_full=parameters,
        gap=reshape(gap), gap_dot=reshape(gap_dot), force_numpy=reshape(force), force_rate_numpy=reshape(rate),
        nominal_compression=reshape(cfg.rest_gap-gap), nominal_ideal_active=reshape(cfg.rest_gap-gap > 0.),
        unclipped_force_input=reshape(raw), active_stiffness=reshape(stiffness),
        mixed_position_term=reshape(pos), mixed_force_term=reshape(frc),
        mixed_velocity_term=reshape(vel), mixed_force_rate_term=reshape(frd),
        mixed_error=reshape(eps), mixed_error_rate=reshape(epsd), normal_joint_load=normal,
        node_times=arrays.get('node_times', np.linspace(0., cfg.Tp, nodes)),
        **({k: arrays[k] for k in arrays if k.startswith('contact_')}))
    with (out / (label+'_key_nodes.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['update', 't' if label == 'prediction' else 'warmup_iteration', 'node', 'gap', 'gap_dot', 'force_L', 'force_R', 'force_rate_L', 'force_rate_R',
                  'raw_L', 'raw_R', 'stiffness_L', 'stiffness_R', 'position_term_Lx', 'force_term_Lx',
                  'position_term_Rx', 'force_term_Rx', 'velocity_term_Lx', 'force_rate_term_Lx', 'velocity_term_Rx', 'force_rate_term_Rx']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for update in key_indices:
            for node in range(nodes):
                k = update*nodes+node
                vals = [update, times[update], node, gap[k], gap_dot[k], *force[k, :, 0], *rate[k, :, 0],
                        *raw[k], *stiffness[k], pos[k, 0, 0], frc[k, 0, 0], pos[k, 1, 0], frc[k, 1, 0],
                        vel[k, 0, 0], frd[k, 0, 0], vel[k, 1, 0], frd[k, 1, 0]]
                writer.writerow(dict(zip(fields, vals)))
    passed = all(v['pass'] for v in checks.values())
    return dict(status='passed' if passed else 'failed', updates=updates, nodes_per_update=nodes,
        total_nodes=len(x), parameter_width=p.shape[1], checks=checks, coverage=coverage,
        independent_geometry_samples=spot, finite_difference_samples=fd,
        independent_measurement_log_matches=measured_matches,
        key_samples=times[key_indices].tolist(), key_sample_coordinate='control_time_s' if label == 'prediction' else 'warmup_iteration',
        force_min_signed_normal=np.min(force[:, :, 0]*[-1., 1.], axis=0).tolist(),
        force_max_signed_normal=np.max(force[:, :, 0]*[-1., 1.], axis=0).tolist(),
        activation=dict(gap_min_max_m=[float(np.min(gap)), float(np.max(gap))],
            nominal_compression_min_max_m=[float(np.min(cfg.rest_gap-gap)), float(np.max(cfg.rest_gap-gap))],
            gap_driven_nominal_ideal_active_fraction=float(np.mean(cfg.rest_gap-gap > 0.)),
            each_side_law_input_positive_fraction=np.mean(raw > 0., axis=0).tolist(),
            each_side_zero_output_fraction=np.mean(force[:, :, 0] == 0., axis=0).tolist(),
            each_side_active_stiffness_min_N_per_m=np.min(stiffness, axis=0).tolist(),
            each_side_active_stiffness_max_N_per_m=np.max(stiffness, axis=0).tolist(),
            new_and_old_ideal_both_on_linear_positive_branch_fraction=(float(np.mean((cfg.rest_gap-gap > 0.) & np.all(raw > 0., axis=1))) if anchored else None),
            note='Old ideal clips the gap-based nominal spring before adding bias; new ideal clips each measured-anchored total force. Their local slopes match K when both are on positive branches. Nominal compression is a model coordinate, not physical penetration/contact evidence.'),
        zero_force_side_nodes=int(np.sum(force[:, :, 0] == 0.)),
        limits='Formula/FK audit only; no claim of physical contact, friction stability, or closed-loop safety. '
               'Startup logs lack saved Fdot/eps/contact fields, so those startup values are recomputed, not independently log-verified. '
               'Absent sampled contact fields are reported unavailable; model normal plus residual is never treated as independent full-contact evidence.')


def audit_case(run):
    folder = Path(run['output_dir'])
    summary_path = folder / 'summary.json'
    if not summary_path.is_file():
        return dict(case_id=run['case_id'], status='unavailable', reason='summary not saved')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    cfg, model, arms, symbolic, geom_fn, all_fn, mjb, fk = geometry_for(run, summary)
    out = ROOT / 'node_formula_audit' / run['case_id']
    out.mkdir(parents=True, exist_ok=True)
    report = dict(case_id=run['case_id'], scene=run['scene'], law=str(cfg.normal_force_law),
                  source_run_status=run['status'], model_sha256=sha(mjb), script_sha256=sha(Path(__file__)),
                  source_hashes={}, traces={})
    trajectory_path = folder / 'trajectory.npz'
    trajectory = read_npz(trajectory_path)
    if trajectory_path.is_file():
        report['source_hashes']['trajectory'] = sha(trajectory_path)
    for label in ('prediction', 'startup'):
        path = folder / (label+'.npz')
        if path.is_file():
            report['source_hashes'][label] = sha(path)
        arrays = read_npz(path)
        report['traces'][label] = audit_trace(label, arrays, cfg, model, arms, symbolic, geom_fn, all_fn, summary, run, out, trajectory)
        if label == 'prediction' and str(cfg.normal_force_law) == 'smooth_positive' and arrays:
            report['same_state_three_laws'] = same_state_three_laws(cfg, fk, arrays, out)
    report['status'] = ('failed' if any(v['status'] == 'failed' for v in report['traces'].values())
                        or report.get('same_state_three_laws', {}).get('status') == 'failed' else
                        'passed' if report['traces']['prediction']['status'] == 'passed' else 'unavailable')
    (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return report


def same_state_three_laws(cfg, fk, arrays, out):
    """Counterfactual formula comparison on unchanged smooth-baseline samples; no OCP solve."""
    times = arrays['t']
    selected = sorted(set([int(np.argmin(abs(times-target))) for target in (0., .5, 1.5)] + [len(times)-1]))
    laws = ('smooth_positive', 'ideal_unilateral', 'measured_anchored_ideal')
    functions, models = [], []
    for law in laws:
        sym = build_model(replace(cfg, normal_force_law=law), fk)
        functions.append(ca.Function('same_state_'+law, [sym.x, sym.u, sym.p], [
            sym.dynamics, sym.residual, sym.predicted_force(sym.x, sym.p), sym.force_rate_fn(sym.x, sym.p),
            ca.jacobian(sym.dynamics, sym.x), ca.jacobian(sym.residual, sym.x)]))
        models.append(sym)
    rows, records, parameter_records = [], [], []
    for index in selected:
        x, u, baseline_p = arrays['state'][index, 0], arrays['u'][index, 0], arrays['parameters_shared_full'][index, 0]
        measured = baseline_p[176:188].reshape(2, 6).copy()
        measured[:, 0] = arrays['force'][index, 0]
        values, all_parameters = [], []
        for law, model, fn in zip(laws, models, functions):
            p = np.r_[baseline_p[:225], np.zeros(3)] if law == 'measured_anchored_ideal' else baseline_p[:225].copy()
            nominal = np.asarray(model.nominal_force(x)).reshape(2, 6)
            p[176:188] = (measured-nominal).reshape(12)
            if law == 'measured_anchored_ideal':
                gap, normal = checked_anchor(float(model.contact_gap(x)), measured[:, 0]*[-1., 1.])
                p[225], p[226:228] = gap, normal
            values.append([np.asarray(v) for v in fn(x, u, p)])
            # Uniform saved width with NaN only in the explicitly absent legacy anchor extension.
            all_parameters.append(p if len(p) == 228 else np.r_[p, np.full(3, np.nan)])
        qdd = [np.r_[v[0][7:14, 0], v[0][21:28, 0]] for v in values]
        gap0 = float(models[0].contact_gap(x))
        point = dict(update=int(index), t_s=float(times[index]), gap_m=gap0,
                     measured_world_Fx_N=measured[:, 0].tolist(), laws={})
        for i, law in enumerate(laws):
            dyn, residual, force, rate, dyn_jac, res_jac = values[i]
            _, _, raw, stiffness = numpy_force(replace(cfg, normal_force_law=law), np.array([gap0]),
                                               np.array([0.]), np.asarray(all_parameters[i])[None])
            point['laws'][law] = dict(
                law_input_before_positive_part_N=raw[0].tolist(), active_normal_stiffness_N_per_m=stiffness[0].tolist(),
                current_world_Fx_N=force.ravel()[[0, 6]].tolist(), current_world_Fx_rate_N_per_s=rate.ravel()[[0, 6]].tolist(),
                qdd_rad_per_s2=qdd[i].tolist(), E=residual[:12, 0].tolist(), E_dot=residual[12:24, 0].tolist(),
                force_difference_max_N=maxabs(force-values[0][2]), qdd_difference_max_rad_per_s2=maxabs(qdd[i]-qdd[0]),
                E_difference_max=maxabs(residual[:12]-values[0][1][:12]),
                force_rate_difference_max_N_per_s=maxabs(rate-values[0][3]),
                E_dot_difference_max=maxabs(residual[12:24]-values[0][1][12:24]),
                dynamics_Jx_difference_max_mixed_units=maxabs(dyn_jac-values[0][4]),
                residual_Jx_difference_max_mixed_units=maxabs(res_jac-values[0][5]))
        rows.append(point)
        records.append(values)
        parameter_records.append(all_parameters)
    force = np.array([[v[2].ravel() for v in point] for point in records])
    rate = np.array([[v[3].ravel() for v in point] for point in records])
    dynamics = np.array([[v[0].ravel() for v in point] for point in records])
    residual = np.array([[v[1].ravel() for v in point] for point in records])
    report = dict(
        status='passed' if all(max(r['laws'][law]['force_difference_max_N'], r['laws'][law]['qdd_difference_max_rad_per_s2'],
                                 r['laws'][law]['E_difference_max']) <= 1e-8 for r in rows for law in laws) else 'failed',
        purpose='Same state, velocity, control torque, measured wrench, M/G/damping, frozen contact residual and references; reanchor only each force law.',
        measured_source='Saved baseline node-0 force for world Fx; non-x measured wrench from legacy bias slots (nominal non-x wrench is zero). Baseline is separately checked against trajectory measurement logs at strict time intersections.',
        laws=laws, samples=rows,
        units={'world_Fx':'N', 'world_Fx_dot':'N/s', 'qdd':'rad/s^2',
               'E':'Controller mixed pose/force residual; xyz channels have position-scaled units (m with configured kf), rotation channels rad.',
               'E_dot':'Corresponding residual per second.',
               'jacobians':'Full arrays saved. State column units repeat [7 rad, 7 rad/s] for each arm; dynamics rows repeat [7 rad/s, 7 rad/s^2]. Residual rows [E12, E_dot12, torque14(N m)]. Entrywise maxima mix units and are not a stability metric.'},
        limitations='No trajectory rerun and no claim that a Jacobian difference improves or harms closed-loop stability. Historical laws have no anchor extension; their final three saved parameter slots are intentionally NaN.')
    (out / 'same_state_three_laws.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    np.savez_compressed(out / 'same_state_three_laws.npz', laws=np.array(laws), update=np.array(selected), t=times[selected],
        state=arrays['state'][selected, 0], torque=arrays['u'][selected, 0], parameters_by_law=np.array(parameter_records),
        force=force, force_rate=rate, dynamics=dynamics, residual=residual,
        dynamics_Jx=np.array([[v[4] for v in point] for point in records]),
        residual_Jx=np.array([[v[5] for v in point] for point in records]))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', action='append', default=[])
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    manifest = ROOT / 'run_manifest.json'
    if not manifest.is_file():
        print(json.dumps({'status': 'waiting_for_manifest', 'path': str(manifest)}))
        return
    data = json.loads(manifest.read_text(encoding='utf-8'))
    runs = [r for r in data['runs'] if r['status'] in ('completed', 'failed') and (not args.case or r['case_id'] in args.case)]
    if args.limit is not None:
        runs = runs[:args.limit]
    reports = []
    for run in runs:
        try:
            report = audit_case(run)
        except Exception as exc:
            import traceback
            report = {'case_id': run['case_id'], 'status': 'audit_error', 'error': str(exc), 'traceback': traceback.format_exc()}
        reports.append(report)
        print(json.dumps({'case_id': run['case_id'], 'status': report['status']}, ensure_ascii=False), flush=True)
    target = ROOT / 'node_formula_audit'
    target.mkdir(exist_ok=True)
    # Preserve other cases' audits when using --case during an unfinished batch.
    by_case = {}
    for existing in target.glob('*/report.json'):
        previous = json.loads(existing.read_text(encoding='utf-8'))
        by_case[previous['case_id']] = previous
    by_case.update({r['case_id']: r for r in reports})
    aggregate = dict(manifest_state=data['state'], cases=list(by_case.values()),
                     case_statuses={k:v['status'] for k,v in by_case.items()}, script_sha256=sha(Path(__file__)))
    (target / 'audit_summary.json').write_text(json.dumps(aggregate, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    if any(r['status'] in ('failed', 'audit_error') for r in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
