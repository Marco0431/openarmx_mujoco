"""Offline audit and comparison summary for the fixed measured-anchor study."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(r'D:\RobotSim')
ROOT = BASE / 'results' / 'measured_anchored_contact_20261012'
MANIFEST = ROOT / 'run_manifest.json'


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}


def load_npz(path: Path):
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def audit_workspace_python_sources():
    snapshot_path = ROOT / 'workspace_source.json'
    snapshot = load_json(snapshot_path)
    workspace = Path(snapshot.get('workspace', ''))
    expected = snapshot.get('shared_source_sha256', {})
    actual = {}
    for path in workspace.rglob('*.py') if workspace.exists() else []:
        rel = path.relative_to(workspace).as_posix()
        actual[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    expected_norm = {k.replace('\\', '/'): v for k, v in expected.items()}
    changed = sorted(k for k in expected_norm if k in actual and actual[k] != expected_norm[k])
    missing = sorted(set(expected_norm) - set(actual))
    untracked = sorted(set(actual) - set(expected_norm))
    source_project = Path(snapshot.get('source_project', ''))
    share_root = BASE / 'nmpcc_share_20260913'
    entrypoints = {'exp2_nmpcc_latest.py', 'exp2_nmpcc_latest_grasp_pose.py'}
    prep_mismatches = []
    for path in workspace.rglob('*.py') if workspace.exists() else []:
        rel = path.relative_to(workspace)
        rel_key = rel.as_posix()
        if rel_key in expected_norm:
            expected_hash = expected_norm[rel_key]
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                prep_mismatches.append({'workspace_path': rel_key, 'expected_origin': 'workspace_source.json per-file SHA256'})
            continue
        if rel.parts[0] in ('nmpcc', 'experiments', 'diagnostics'):
            origin = source_project / rel
        elif rel.parts[0] in ('flat', 'handheld') and path.name not in entrypoints:
            overlay = source_project / rel.name
            origin = overlay if overlay.is_file() else share_root / rel
        else:
            origin = share_root / rel
        if not origin.is_file() or hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(origin.read_bytes()).digest():
            prep_mismatches.append({'workspace_path': rel.as_posix(), 'expected_origin': str(origin)})
    return {
        'snapshot_path': str(snapshot_path),
        'snapshot_sha256': hashlib.sha256(snapshot_path.read_bytes()).hexdigest() if snapshot_path.exists() else None,
        'python_file_count_expected_in_snapshot': len(expected_norm),
        'python_file_count_actual': len(actual),
        'snapshot_covered_files_unchanged': bool(not changed and not missing),
        'full_workspace_python_coverage': bool(not untracked and len(expected_norm) == len(actual)),
        'changed_files': changed,
        'missing_files': missing,
        'untracked_python_files': untracked,
        'prepare_workspace_rule_validation': 'each shared subtree overlays source_project; scene roots overlay same-named top-level source files except designated entrypoints; remaining files originate from nmpcc_share_20260913',
        'prepare_workspace_rule_mismatch_count': len(prep_mismatches),
        'prepare_workspace_rule_mismatches': prep_mismatches,
    }


def arr_stats(value):
    a = np.asarray(value)
    if not np.issubdtype(a.dtype, np.number) or not a.size:
        return {'shape': list(a.shape), 'dtype': str(a.dtype)}
    good = a[np.isfinite(a)]
    if not good.size:
        return {'shape': list(a.shape), 'finite_count': 0, 'total_count': int(a.size)}
    return {'shape': list(a.shape), 'finite_count': int(good.size), 'total_count': int(a.size),
            'min': float(np.min(good)), 'median': float(np.median(good)),
            'p95': float(np.percentile(good, 95)), 'max': float(np.max(good)),
            'rms': float(np.sqrt(np.mean(good * good)))}


def timing_stats(prediction, physics):
    t = np.asarray(prediction.get('t', []), dtype=float)
    mask = np.isfinite(t) & (t >= .1)
    result = {}
    for source, data, key in (
            ('prediction', prediction, 'prepare_ms'),
            ('prediction', prediction, 'reference_ms'),
            ('prediction', prediction, 'solver_ms'),
            ('prediction', prediction, 'control_ms'),
            ('prediction', prediction, 'joint_diagnostic_ms'),
            ('physics', physics, 'diagnostic_ms')):
        if key not in data:
            continue
        x = np.asarray(data[key], dtype=float)
        valid = mask & np.isfinite(x) if source == 'prediction' and x.shape == mask.shape else np.isfinite(x)
        x = x[valid]
        result[key] = None if not x.size else {
            'p50': float(np.percentile(x, 50)), 'p95': float(np.percentile(x, 95)), 'n': int(x.size)}
    return result


def max_or_none(value):
    if value is None:
        return None
    a = np.asarray(value, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.max(a)) if a.size else None


def motion_reference_offset(times, spec):
    t = np.asarray(times, dtype=float)
    offset = np.zeros_like(t)
    if bool(spec['hold_only']) or abs(float(spec['y_dist'])) <= 1e-12:
        return offset
    distance, speed = abs(float(spec['y_dist'])), abs(float(spec.get('v_y', -.01)))
    base = distance / speed if speed > 0 else 0.
    ramp = min(.20, .5 * base) if base > 0 else 0.
    duration = base + ramp
    start = float(spec['t_ramp']) + float(spec['t_hold'])
    end = start + duration
    tau = t - start
    accelerating = (tau >= 0) & (tau < ramp)
    cruising = (tau >= ramp) & (tau < base)
    decelerating = (tau >= base) & (tau < duration)
    sa = tau[accelerating] / ramp if ramp > 0 else np.zeros(0)
    offset[accelerating] = -speed*ramp*(sa**3-.5*sa**4)
    offset[cruising] = -speed*(tau[cruising]-.5*ramp)
    sd = (tau[decelerating]-base)/ramp if ramp > 0 else np.zeros(0)
    offset[decelerating] = -speed*(base-.5*ramp+ramp*(sd-sd**3+.5*sd**4))
    offset[t >= end] = -distance
    return offset


def motion_audit(times, box, spec, initial_y):
    """Reconstruct the exact timed reference used by PhaseReference."""
    if bool(spec['hold_only']) or abs(float(spec['y_dist'])) <= 1e-12:
        return {'applicable': False}
    t = np.asarray(times, dtype=float)
    b = np.asarray(box, dtype=float)
    if not t.size or b.ndim != 2 or len(b) != len(t):
        return {'applicable': True, 'available': False}
    distance, speed = abs(float(spec['y_dist'])), abs(float(spec.get('v_y', -.01)))
    base = distance / speed if speed > 0 else 0.
    ramp = min(.20, .5 * base) if base > 0 else 0.
    duration = base + ramp
    start = float(spec['t_ramp']) + float(spec['t_hold'])
    end = start + duration
    move_mask = (t >= start) & (t < end)
    if not move_mask.any():
        return {'applicable': True, 'available': False, 'start_s': start, 'end_s': end}
    offset = motion_reference_offset(t, spec)
    tracking_error_mm = ((b[:, 1]-initial_y)-offset)*1000.
    endpoint = int(np.argmin(np.abs(t-end)))
    travel_mm = float((b[move_mask, 1][-1]-b[move_mask, 1][0])*1000.)
    return {
        'applicable': True, 'available': True, 'window_s': [start, end],
        'samples': int(move_mask.sum()),
        'tracking_rms_mm': float(np.sqrt(np.mean(tracking_error_mm[move_mask]**2))),
        'tracking_max_abs_mm': float(np.max(np.abs(tracking_error_mm[move_mask]))),
        'signed_box_travel_mm': travel_mm,
        'endpoint_sample_time_s': float(t[endpoint]),
        'endpoint_error_mm': float(abs(tracking_error_mm[endpoint])),
    }


def min_or_none(value):
    if value is None:
        return None
    a = np.asarray(value, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.min(a)) if a.size else None


def tilt_from_vertical_deg(rotvec):
    """Axis-z deviation from vertical, excluding yaw."""
    rv = np.asarray(rotvec, dtype=float)
    if rv.ndim != 2 or rv.shape[-1] != 3:
        return np.array([])
    angle = np.linalg.norm(rv, axis=1)
    half = .5 * angle
    scale = np.zeros_like(angle)
    nz = angle > 1e-12
    scale[nz] = np.sin(half[nz]) / angle[nz]
    qx, qy = rv[:, 0]*scale, rv[:, 1]*scale
    cos_tilt = np.clip(1. - 2.*(qx*qx + qy*qy), -1., 1.)
    return np.degrees(np.arccos(cos_tilt))


def force_error_window(force, mask):
    f = np.asarray(force, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if f.ndim != 2 or len(f) != len(m) or not m.any():
        return {'samples': 0, 'signed_mean_error_N_by_side': None,
                'mae_N_by_side': None, 'rms_N_by_side': None, 'mean_side_rms_N': None}
    err = f[m] - 10.
    rms = np.sqrt(np.mean(err**2, axis=0))
    return {'samples': int(m.sum()), 'signed_mean_error_N_by_side': np.mean(err, axis=0).tolist(),
            'mae_N_by_side': np.mean(np.abs(err), axis=0).tolist(),
            'rms_N_by_side': rms.tolist(), 'mean_side_rms_N': float(np.mean(rms))}


def make_node_audit(run, prediction, joint):
    """Preserve every control update and all N+1 horizon nodes in an audit NPZ."""
    folder = Path(run['output_dir'])
    if not prediction:
        return {'available': False}
    arrays = {}
    for key in ('t', 'node_times', 'state', 'force', 'force_error', 'force_pred_dot', 'force_derivative',
                'force_rate', 'force_anchor_gap', 'force_anchor_normal',
                'force_nominal', 'force_measured', 'force_bias', 'force_anchored',
                'position_error_x', 'eps_x', 'epsd_x', 'parameters_shared_full',
                'parameter_layout', 'solver_status', 'joint_q', 'joint_v', 'joint_violation'):
        if key in prediction:
            arrays['prediction__' + key] = prediction[key]
    for key in ('contact_full_contact', 'contact_model_normal', 'contact_frozen_residual',
                'contact_tangent_contact', 'contact_correction'):
        if key in prediction:
            arrays['prediction__' + key] = prediction[key]
    for key in ('pred_t', 'pred_node_times', 'pred_joint_q', 'pred_joint_v',
                'pred_joint_violation', 'pred_violation', 'pred_solver_status'):
        if key in joint:
            arrays['joint__' + key] = joint[key]
    if arrays:
        np.savez_compressed(folder / 'prediction_node_audit.npz', **arrays)
    statuses = np.asarray(prediction.get('solver_status', []))
    force = prediction.get('force')
    state = prediction.get('state')
    violation = prediction.get('joint_violation')
    if violation is None:
        violation = joint.get('pred_joint_violation')
    if violation is None:
        violation = joint.get('pred_violation')
    detail = {
        'available': True,
        'control_updates': int(len(prediction.get('t', []))),
        'node_count': int(np.asarray(state).shape[1]) if state is not None and np.asarray(state).ndim >= 2 else None,
        'state_all_updates_all_nodes': arr_stats(state) if state is not None else None,
        'force_all_updates_all_nodes': arr_stats(force) if force is not None else None,
        'joint_violation_all_updates_all_nodes': arr_stats(violation) if violation is not None else None,
        'solver_status_bad_count': int(np.count_nonzero(statuses != 0)) if statuses.size else None,
        'solver_status_counts': {str(int(v)): int(np.count_nonzero(statuses == v)) for v in np.unique(statuses)} if statuses.size else {},
        'array_path': str(folder / 'prediction_node_audit.npz'),
    }
    # Give every solved control update a compact reference into the preserved
    # N+1 arrays; complete values remain in NPZ to avoid enormous JSON output.
    t = np.asarray(prediction.get('t', []), dtype=float)
    nodes = np.asarray(prediction.get('node_times', []), dtype=float)
    detail['control_update_index_and_node_times'] = [
        {'control_index': i, 'control_t_s': float(t[i]),
         'node_times_s': nodes[i].tolist() if nodes.ndim == 2 and i < len(nodes) else []}
        for i in range(len(t))]
    return detail


def source_fidelity(group):
    laws = {}
    for row in group:
        laws[row['law']] = row
    results = []
    base = laws.get('smooth_positive')
    for law in ('ideal_unilateral', 'measured_anchored_ideal'):
        other = laws.get(law)
        if not base or not other:
            results.append({'law': law, 'available': False})
            continue
        a, b = base.get('initial_state', {}), other.get('initial_state', {})
        state_diffs = {}
        for key in ('initial_qpos', 'initial_qvel', 'box_initial_position'):
            av, bv = a.get(key), b.get(key)
            if av is not None and bv is not None:
                state_diffs[key + '_max_abs'] = float(np.max(np.abs(np.asarray(av)-np.asarray(bv))))
        ref_a, ref_b = a.get('reference'), b.get('reference')
        results.append({
            'law': law, 'available': True,
            'same_mjb_sha256': base.get('mjb_sha256') == other.get('mjb_sha256'),
            'same_reference_file_sha256': base.get('reference_sha256') == other.get('reference_sha256'),
            'same_reference_numeric_payload': json.dumps(ref_a, sort_keys=True) == json.dumps(ref_b, sort_keys=True),
            'same_initial_qpos_qvel_and_box_position': all(
                state_diffs.get(key + '_max_abs') == 0. for key in
                ('initial_qpos', 'initial_qvel', 'box_initial_position')),
            'initial_state_max_abs_differences': state_diffs,
            'same_source_sha256_before': base.get('source_sha256_before') == other.get('source_sha256_before'),
            'same_source_sha256_after': base.get('source_sha256_after') == other.get('source_sha256_after'),
            'cache_sha256_before': other.get('cache_sha256_before'),
            'cache_sha256_after': other.get('cache_sha256_after'),
        })
    return results


def summarize_run(run):
    folder = Path(run['output_dir'])
    summary = load_json(folder / 'summary.json')
    physics = load_npz(folder / 'physics.npz')
    prediction = load_npz(folder / 'prediction.npz')
    joint = load_npz(folder / 'joint_limits.npz')
    trajectory = load_npz(folder / 'trajectory.npz')
    metrics = summary.get('metrics', {})
    initial = load_json(Path(run['evidence_prefix'] + '.reference.json'))
    force = np.asarray(physics.get('force', []), dtype=float)
    times = np.asarray(physics.get('t', []), dtype=float)
    contact_loss = force < .1 if force.size else np.zeros((0, 2), dtype=bool)
    actual_loss_steps = int(np.any(contact_loss, axis=1).sum()) if force.size else None
    actual_zero_steps = int(np.any(force <= 0., axis=1).sum()) if force.size else None
    contact_count = np.asarray(physics.get('contact_count', []))
    actual_no_contact_steps = int(np.any(contact_count == 0, axis=1).sum()) if contact_count.ndim == 2 and len(contact_count) else None
    active = np.asarray(joint.get('active', []), dtype=bool)
    lower_margin = np.asarray(joint.get('lower_margin', []), dtype=float)
    upper_margin = np.asarray(joint.get('upper_margin', []), dtype=float)
    actual_margin = np.minimum(lower_margin, upper_margin) if lower_margin.size and upper_margin.size else None
    pred_violation = joint.get('pred_joint_violation', joint.get('pred_violation',
                         prediction.get('joint_violation')))
    pred_violation_array = np.asarray(pred_violation, dtype=float) if pred_violation is not None else np.array([])
    pred_node0 = pred_violation_array[:, 0] if pred_violation_array.ndim == 3 and pred_violation_array.shape[1] else np.array([])
    pred_future = pred_violation_array[:, 1:] if pred_violation_array.ndim == 3 and pred_violation_array.shape[1] > 1 else np.array([])
    limit_torque = np.asarray(joint.get('limit_torque', []), dtype=float)
    active_joint_count = int(np.count_nonzero(active)) if active.size else None
    statuses = np.asarray(prediction.get('solver_status', []), dtype=float)
    nonfinite = []
    critical_nonfinite = []
    expected_trajectory_nan = []
    for file_name, data in (('physics', physics), ('prediction', prediction),
                            ('joint_limits', joint), ('trajectory', trajectory)):
        for key, value in data.items():
            a = np.asarray(value)
            if np.issubdtype(a.dtype, np.number) and not np.isfinite(a).all():
                nonfinite.append(f'{file_name}.{key}')
                if file_name == 'trajectory' and key in (
                        'solve_ms', 'prepare_ms', 'reference_ms', 'solver_ms', 'solver_status',
                        'qp_iter', 'solver_qp_ms', 'solver_sim_ms', 'solver_lin_ms',
                        'force_nominal', 'force_measured_control', 'force_bias', 'force_anchored',
                        'pred_force', 'pred_force_error', 'pred_position_error_x', 'pred_eps_x',
                        'pred_epsd_x', 'pred_u'):
                    expected_trajectory_nan.append(key)
                else:
                    critical_nonfinite.append(f'{file_name}.{key}')
    actual_times = np.asarray(times)
    tail = actual_times >= (actual_times[-1] - .2) if actual_times.size else np.zeros(0, dtype=bool)
    box = np.asarray(physics.get('box', []), dtype=float)
    rv = np.asarray(physics.get('rotvec', []), dtype=float)
    tips_l, tips_r = physics.get('tipL'), physics.get('tipR')
    init_box = np.asarray(initial.get('box_initial_position', box[0] if box.ndim == 2 and len(box) else []), dtype=float)
    if box.ndim == 2 and len(box) and init_box.shape == (3,):
        box_y_mm = (box[:, 1] - init_box[1]) * 1000.
        box_desc_mm = (init_box[2] - box[:, 2]) * 1000.
    else:
        box_y_mm = box_desc_mm = np.array([])
    tilt_deg = tilt_from_vertical_deg(rv)
    sync_mm = (np.asarray(tips_l)[:, 1] - np.asarray(tips_r)[:, 1]) * 1000. if tips_l is not None and tips_r is not None else np.array([])
    move_rms = {k: v for k, v in metrics.items() if ('move' in k.lower() and 'rms' in k.lower())}
    initial_y = float(init_box[1]) if init_box.shape == (3,) else float(box[0, 1]) if box.ndim == 2 and len(box) else 0.
    move_tracking = motion_audit(times, box, run['scenario'], initial_y) if box.ndim == 2 else {'applicable': True, 'available': False}
    move_offset = motion_reference_offset(times, run['scenario']) if times.size else np.array([])
    y_error_mm = (box_y_mm - move_offset*1000.) if box_y_mm.size else np.array([])
    hold_end = float(run['scenario']['t_ramp']) + float(run['scenario']['t_hold'])
    hold_mask = (times >= float(run['scenario']['t_ramp'])) & (times < hold_end) if times.size else np.zeros(0, bool)
    if run['scenario']['hold_only']:
        move_mask = np.zeros(len(times), dtype=bool)
    elif times.size:
        distance, speed = abs(float(run['scenario']['y_dist'])), .01
        duration = (distance/speed + min(.20, .5*distance/speed)) if distance else 0.
        move_mask = (times >= hold_end) & (times < hold_end + duration)
    else:
        move_mask = np.zeros(0, dtype=bool)
    output_force_fields = {}
    for fname, data in (('prediction', prediction), ('trajectory', trajectory)):
        for key, value in data.items():
            low = key.lower()
            if any(token in low for token in ('force', 'anchor', 'deriv', 'rate', 'eps_x', 'position_error')):
                output_force_fields[f'{fname}.{key}'] = arr_stats(value)
    node_audit = make_node_audit(run, prediction, joint)
    force_windows = {
        'all': force_error_window(force, np.ones(len(force), dtype=bool)),
        'hold': force_error_window(force, hold_mask),
        'move': force_error_window(force, move_mask),
        'tail_last_0p2s': force_error_window(force, tail),
    }
    failure_record = summary.get('failure') or {}
    joint_failure_keys = [k for k in joint if k.startswith('failure_')]
    joint_failure = {k: arr_stats(joint[k]) for k in joint_failure_keys}
    first_failure_status = failure_record.get('first_status', failure_record.get('status'))
    first_failure_time = failure_record.get('first_t', failure_record.get('t'))
    if first_failure_status is None and 'failure_first_status' in joint:
        first_failure_status = np.asarray(joint['failure_first_status']).tolist()
    if first_failure_time is None and 'failure_t' in joint:
        first_failure_time = np.asarray(joint['failure_t']).tolist()
    recovered_solver_failure = bool(failure_record or joint_failure_keys)
    row = {
        'sequence': run['sequence'], 'law': run['law'], 'case_id': run['case_id'],
        'case': run['case'], 'scene': run['scene'], 'eta_cop_y': run['scenario']['eta_cop_y'],
        'eta_f_x': run['scenario']['eta_f_x'], 'distance_m': float(run['scenario']['y_dist']),
        'hold_only': bool(run['scenario']['hold_only']), 'runner_status': run['status'],
        'exitcode': run.get('exitcode'), 'elapsed_wall_s': run.get('elapsed_s'),
        'solver_model': run.get('solver_model'), 'model_version': run.get('model_version'),
        'source_sha256_before': run.get('source_sha256_before'),
        'source_sha256_after': run.get('source_sha256_after'),
        'cache_sha256_before': run.get('cache_sha256_before'),
        'cache_sha256_after': run.get('cache_sha256_after'),
        'mjb_sha256': run.get('mjb_sha256'), 'reference_sha256': run.get('reference_sha256'),
        'physics_samples_2ms': int(len(force)) if force.size else 0,
        'physics_dt_median_s': float(np.median(np.diff(times))) if len(times) > 1 else None,
        'actual_contact_loss_steps_lt_0p1N': actual_loss_steps,
        'actual_contact_loss_side_samples_lt_0p1N': int(contact_loss.sum()) if force.size else None,
        'actual_no_contact_steps_contact_count_eq_0': actual_no_contact_steps,
        'actual_no_contact_side_samples_contact_count_eq_0': int(np.count_nonzero(contact_count == 0)) if contact_count.size else None,
        'actual_contact_count_side_samples_total': int(np.sum(contact_count)) if contact_count.size else None,
        'actual_contact_count_by_side_median': np.median(contact_count, axis=0).tolist() if contact_count.ndim == 2 and len(contact_count) else None,
        'actual_contact_count_by_side_max': np.max(contact_count, axis=0).tolist() if contact_count.ndim == 2 and len(contact_count) else None,
        'actual_zero_contact_steps': actual_zero_steps,
        'first_contact_loss_time_s': (float(times[np.flatnonzero(np.any(contact_loss, axis=1))[0]])
                                      if force.size and actual_loss_steps else None),
        'actual_force_peak_N': float(np.max(np.abs(force))) if force.size else None,
        'actual_force_value_rms_N_by_side': np.sqrt(np.mean(force**2, axis=0)).tolist() if force.size else None,
        'force_error_vs_10N_by_window': force_windows,
        'actual_limit_active_steps': int(np.any(active, axis=1).sum()) if active.size else None,
        'actual_limit_active_joint_samples': active_joint_count,
        'actual_limit_active_joint_duration_ms': float(active_joint_count * 2.) if active_joint_count is not None else None,
        'actual_limit_reaction_torque_peak_Nm': max_or_none(np.abs(limit_torque)) if limit_torque.size else None,
        'actual_min_joint_margin_rad': min_or_none(actual_margin),
        'actual_max_joint_violation_rad': max(0., -min_or_none(actual_margin)) if actual_margin is not None and min_or_none(actual_margin) is not None else None,
        'predicted_all_node_joint_violation_entries': int(np.count_nonzero(np.asarray(pred_violation) > 0.)) if pred_violation is not None else None,
        'predicted_all_node_joint_violation_max_rad': max_or_none(pred_violation),
        'predicted_node0_measured_joint_violation_entries': int(np.count_nonzero(pred_node0 > 0.)) if pred_node0.size else None,
        'predicted_node0_measured_joint_violation_max_rad': max_or_none(pred_node0) if pred_node0.size else None,
        'predicted_future_nodes_1_to_N_joint_violation_entries': int(np.count_nonzero(pred_future > 0.)) if pred_future.size else None,
        'predicted_future_nodes_1_to_N_joint_violation_max_rad': max_or_none(pred_future) if pred_future.size else None,
        'solver_status_bad_count': int(np.count_nonzero(statuses != 0)) if statuses.size else None,
        'solver_status_counts': {str(int(v)): int(np.count_nonzero(statuses == v)) for v in np.unique(statuses)} if statuses.size else {},
        'recovered_or_terminal_solver_failure': recovered_solver_failure,
        'first_solver_failure_time_s': first_failure_time,
        'first_solver_failure_status': first_failure_status,
        'solver_failure_record': failure_record,
        'joint_limits_failure_fields': joint_failure,
        'nonfinite_npz_fields': nonfinite,
        'critical_nonfinite_npz_fields': critical_nonfinite,
        'expected_trajectory_noncontrol_nan_fields': sorted(set(expected_trajectory_nan)),
        'box_final_y_displacement_mm': float(box_y_mm[-1]) if box_y_mm.size else None,
        'box_max_descent_mm': float(np.max(box_desc_mm)) if box_desc_mm.size else None,
        'tilt_axis_z_from_vertical_max_deg': float(np.max(tilt_deg)) if tilt_deg.size else None,
        'left_right_tip_y_sync_rms_mm': float(np.sqrt(np.mean(sync_mm**2))) if sync_mm.size else None,
        'endpoint_error_mm': metrics.get('box_y_endpoint_error_mm'),
        'commanded_move_tracking': move_tracking,
        'move_rms_metrics': move_rms,
        'tail_last_0p2s': {
            'start_s': float(times[tail][0]) if tail.any() else None,
            'end_s': float(times[tail][-1]) if tail.any() else None,
            'force_mean_N_by_side': np.mean(force[tail], axis=0).tolist() if force.size and tail.any() else None,
            'box_y_tracking_error_rms_mm': float(np.sqrt(np.mean(y_error_mm[tail]**2))) if y_error_mm.size and tail.any() else None,
            'box_y_tail_target_mm': 0. if run['scenario']['hold_only'] else -abs(float(run['scenario']['y_dist']))*1000.,
            'box_y_target_rms_error_Note': 'relative to exact hold target 0 mm or final commanded displacement -distance',
            'box_descent_max_mm': float(np.max(box_desc_mm[tail])) if box_desc_mm.size and tail.any() else None,
            'tilt_axis_z_from_vertical_max_deg': float(np.max(tilt_deg[tail])) if tilt_deg.size and tail.any() else None,
            'tip_y_sync_rms_mm': float(np.sqrt(np.mean(sync_mm[tail]**2))) if sync_mm.size and tail.any() else None,
        },
        'timing_ms_excluding_t_lt_0p1s': timing_stats(prediction, physics),
        'prediction_update_count': len(prediction.get('t', [])),
        'prediction_state_shape': list(np.shape(prediction.get('state', []))),
        'prediction_force_shape': list(np.shape(prediction.get('force', []))),
        'prediction_node_audit': node_audit,
        'measured_force_derivative_mixed_term_fields': output_force_fields,
        'production_metrics': metrics,
        'config': summary.get('config', {}),
        'initial_state': initial,
    }
    row['validation_failure'] = bool(
        run.get('exitcode') not in (None, 0) or row['solver_status_bad_count'] not in (None, 0)
        or recovered_solver_failure or actual_loss_steps not in (None, 0)
        or (run.get('law') == 'measured_anchored_ideal' and (row['actual_limit_active_steps'] or 0) > 0)
        or bool(critical_nonfinite))
    # Uniform physical safety flag across all contact laws, separate from the
    # study's existing candidate acceptance policy recorded above.
    row['physical_safety_failure'] = bool(
        (row['actual_limit_active_steps'] or 0) > 0
        or (actual_loss_steps or 0) > 0
        or (row['actual_no_contact_steps_contact_count_eq_0'] or 0) > 0
        or (row['solver_status_bad_count'] or 0) > 0
        or recovered_solver_failure
        or bool(critical_nonfinite))
    return row


def main():
    manifest = load_json(MANIFEST)
    rows = []
    for run in manifest.get('runs', []):
        if run.get('status') not in ('completed', 'failed', 'running'):
            continue
        rows.append(summarize_run(run))
    groups = {}
    for row in rows:
        groups.setdefault(row['case'], []).append(row)
    fidelity = {key: source_fidelity(group) for key, group in groups.items()}
    report = {
        'task': manifest.get('task'), 'source_manifest': str(MANIFEST),
        'simulation_performed': False, 'run_count_summarized': len(rows),
        'timing_note': 'Per-update timing samples use t >= 0.1 s, excluding two startup warmups. Wall time includes process setup, logging and physics.',
        'contact_note': 'Actual force/contact-loss counts use every available 2 ms physics sample. <0.1 N low-force samples and contact_count == 0 physical no-contact samples are reported separately.',
        'limit_note': 'Actual limit-active joint duration is joint-channel milliseconds (active joint samples × 2 ms); predicted node 0 is the measured current state and is separated from future nodes 1:N.',
        'workspace_python_source_audit': audit_workspace_python_sources(),
        'prediction_note': 'Prediction and joint-limit NPZ retain every recorded control update and all N+1 nodes. Future values are model predictions, not future open-loop plant measurements.',
        'paired_scene_fidelity': fidelity, 'runs': rows,
    }
    (ROOT / 'comparison_summary.json').write_text(
        json.dumps(clean_json(report), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
        for key, value in row['timing_ms_excluding_t_lt_0p1s'].items():
            if value:
                for stat, number in value.items():
                    flat[f'{key}_{stat}'] = number
        flat['tail_force_mean_left_N'] = (row['tail_last_0p2s']['force_mean_N_by_side'] or [None, None])[0]
        flat['tail_force_mean_right_N'] = (row['tail_last_0p2s']['force_mean_N_by_side'] or [None, None])[1]
        flat['tail_y_tracking_error_rms_mm'] = row['tail_last_0p2s']['box_y_tracking_error_rms_mm']
        flat['move_tracking_error_rms_mm'] = row['commanded_move_tracking'].get('tracking_rms_mm')
        flat['move_endpoint_error_mm'] = row['commanded_move_tracking'].get('endpoint_error_mm')
        for window, doc in row['force_error_vs_10N_by_window'].items():
            flat[f'{window}_force_error_mean_side_rms_N'] = doc.get('mean_side_rms_N')
            flat[f'{window}_force_error_left_rms_N'] = (doc.get('rms_N_by_side') or [None, None])[0]
            flat[f'{window}_force_error_right_rms_N'] = (doc.get('rms_N_by_side') or [None, None])[1]
            flat[f'{window}_force_error_left_mae_N'] = (doc.get('mae_N_by_side') or [None, None])[0]
            flat[f'{window}_force_error_right_mae_N'] = (doc.get('mae_N_by_side') or [None, None])[1]
        for key, value in row['production_metrics'].items():
            if not isinstance(value, (dict, list)):
                flat['metric_' + key] = value
        flat_rows.append(flat)
    fields = sorted(set().union(*(r.keys() for r in flat_rows))) if flat_rows else []
    with (ROOT / 'comparison_summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(flat_rows)
    if manifest.get('state') in ('completed', 'completed_with_anchored_halt'):
        before = ROOT / 'run_manifest_before_validation.json'
        if not before.exists():
            before.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        by_id = {r['case_id']: r for r in rows}
        review = {'task': manifest.get('task'), 'basis': 'offline physics, prediction, and failure-record audit',
                  'runs': []}
        for run in manifest.get('runs', []):
            row = by_id.get(run['case_id'])
            if row is None:
                continue
            run['validation_failure'] = row['validation_failure']
            run['physical_safety_failure'] = row['physical_safety_failure']
            run['validation_reasons'] = {
                'exitcode_nonzero': run.get('exitcode') not in (None, 0),
                'solver_status_bad_count': row['solver_status_bad_count'],
                'recovered_or_terminal_solver_failure': row['recovered_or_terminal_solver_failure'],
                'actual_contact_loss_steps_lt_0p1N': row['actual_contact_loss_steps_lt_0p1N'],
                'new_law_actual_limit_active_steps': (row['actual_limit_active_steps']
                                                      if run.get('law') == 'measured_anchored_ideal' else None),
                'critical_nonfinite_npz_fields': row['critical_nonfinite_npz_fields'],
            }
            run['physical_safety_reasons'] = {
                'actual_limit_active_steps': row['actual_limit_active_steps'],
                'actual_contact_loss_steps_lt_0p1N': row['actual_contact_loss_steps_lt_0p1N'],
                'actual_no_contact_steps_contact_count_eq_0': row['actual_no_contact_steps_contact_count_eq_0'],
                'solver_status_bad_count': row['solver_status_bad_count'],
                'recovered_or_terminal_solver_failure': row['recovered_or_terminal_solver_failure'],
                'critical_nonfinite_npz_fields': row['critical_nonfinite_npz_fields'],
            }
            review['runs'].append({'case_id': row['case_id'],
                                   'validation_failure': row['validation_failure'],
                                   'physical_safety_failure': row['physical_safety_failure'],
                                   'reasons': run['validation_reasons'],
                                   'physical_safety_reasons': run['physical_safety_reasons'],
                                   'first_solver_failure_time_s': row['first_solver_failure_time_s'],
                                   'first_solver_failure_status': row['first_solver_failure_status']})
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        review['validation_failure_count'] = sum(bool(r['validation_failure']) for r in review['runs'])
        review['physical_safety_failure_count'] = sum(bool(r['physical_safety_failure']) for r in review['runs'])
        (ROOT / 'physical_validation_review.json').write_text(
            json.dumps(clean_json(review), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'summary_json': str(ROOT / 'comparison_summary.json'),
                      'summary_csv': str(ROOT / 'comparison_summary.csv'),
                      'run_count': len(rows), 'fidelity': fidelity}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
