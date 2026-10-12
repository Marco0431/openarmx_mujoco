"""Prepare/run the fixed 24-case measured-anchored contact comparison.

Preparation only writes a manifest. Simulation starts only with --run, after a
separate explicit GO from the experiment owner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(r'D:\RobotSim')
HERE = Path(__file__).resolve().parent
HELPER = HERE / 'run_one_case.py'
PYTHON = Path(r'D:\acados_env\python.exe')
RESULTS = BASE / 'results' / 'measured_anchored_contact_20261012'
MANIFEST = RESULTS / 'run_manifest.json'
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
LAWS = ('smooth_positive', 'ideal_unilateral', 'measured_anchored_ideal')


def scenario(case: str, scene: str, steps: int, distance: str, hold_only: bool,
             eta_cop: str = '.8', eta_f: str = '.8', hold: str = '1') -> dict:
    return dict(case=case, scene=scene, steps=steps, y_dist=distance,
                hold_only=hold_only, eta_cop_y=eta_cop, eta_f_x=eta_f,
                t_ramp='.5', t_hold=hold)


SCENARIOS = [
    scenario('flat_hold', 'flat', 900, '.05', True),
    scenario('flat_move01', 'flat', 1550, '.01', False),
    scenario('flat_move10', 'flat', 6050, '.10', False),
    scenario('handheld_hold', 'handheld', 900, '.05', True),
    scenario('handheld_move01', 'handheld', 1550, '.01', False),
    scenario('handheld_move05', 'handheld', 3550, '.05', False),
    scenario('handheld_cop08_f02_move05', 'handheld', 3550, '.05', False, '.8', '.2'),
    scenario('handheld_cop00_f00_move05_hold0', 'handheld', 3050, '.05', False, '0', '0', '0'),
]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_tree(path: Path, exclude_codegen: bool = False) -> str:
    h = hashlib.sha256()
    for file in sorted(p for p in path.rglob('*') if p.is_file()):
        rel = file.relative_to(path)
        if '__pycache__' in rel.parts or file.suffix == '.pyc':
            continue
        if exclude_codegen and rel.parts and rel.parts[0] == 'acados_codegen':
            continue
        h.update(rel.as_posix().encode('utf-8')); h.update(b'\0')
        with file.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        h.update(b'\0')
    return h.hexdigest()


def make_command(workspace: Path, spec: dict, law: str, output: Path,
                 evidence: Path) -> list[str]:
    cmd = [str(PYTHON), '-B', '-X', 'utf8', '-u', str(HELPER),
           '--workspace', str(workspace), '--variant', spec['scene'],
           '--evidence_prefix', str(evidence), '--steps', str(spec['steps']),
           '--no_viewer', '--y_dist', spec['y_dist'], '--v_y', '-.01',
           '--force', '10', '--f_start', '10', '--t_ramp', spec['t_ramp'],
           '--t_hold', spec['t_hold'], '--N', '10', '--Tp', '.1',
           '--sim_steps', '2', '--Q', '1', '--P', '2', '--sliding_friction', '.8',
           '--noslip_iterations', '3', '--kf_scale', '.003', '--eta_dev', '.05',
           '--eta_u', '5e-8', '--contact_load_mode', 'residual',
           '--eta_cop_y', spec['eta_cop_y'], '--eta_f_x', spec['eta_f_x'],
           '--normal_force_law', law, '--output_dir', str(output)]
    # The handheld CLI exposes settling; the fixed protocol keeps it disabled.
    if spec['scene'] == 'handheld':
        cmd += ['--settle_time', '0']
    else:
        cmd += ['--sim_stages', '4']
    if spec['hold_only']:
        cmd.append('--hold_only')
    return cmd


def evidence_paths(prefix: Path) -> list[Path]:
    return [Path(str(prefix) + suffix) for suffix in
            ('.reference.json', '.model.mjb', '.result_log.npz', '.metrics.json')]


def planned_runs(workspace: Path) -> list[dict]:
    rows = []
    # Law-major order keeps the new candidate contiguous and last. Each case is
    # a fresh helper process; no parallel builds can touch the shared DLL cache.
    for law in LAWS:
        for index, spec in enumerate(SCENARIOS, 1):
            case_id = f'{law}_{spec["case"]}'
            output = RESULTS / case_id
            evidence = RESULTS / 'evidence' / case_id
            stdout = RESULTS / 'stdout' / f'{case_id}.log'
            rows.append({
                'sequence': len(rows) + 1, 'law': law, 'case': spec['case'],
                'case_id': case_id, 'scene': spec['scene'], 'scenario': dict(spec),
                'workspace': str(workspace), 'command': make_command(workspace, spec, law, output, evidence),
                'output_dir': str(output), 'evidence_prefix': str(evidence), 'stdout': str(stdout),
                'status': 'planned', 'started_local': None, 'finished_local': None,
                'elapsed_s': None, 'exitcode': None, 'error': None,
                'source_sha256_before': None, 'source_sha256_after': None,
                'cache_sha256_before': None, 'cache_sha256_after': None,
                'mjb_sha256': None, 'reference_sha256': None, 'helper_sha256': sha_file(HELPER),
                'solver_model': None, 'model_version': None, 'health': None,
            })
    return rows


def initial_manifest(workspace: Path) -> dict:
    if not workspace.is_dir():
        raise SystemExit(f'workspace does not exist: {workspace}')
    if not (workspace / 'flat').is_dir() or not (workspace / 'handheld').is_dir():
        raise SystemExit('workspace must contain flat/ and handheld/ projects')
    return {
        'task': 'fixed measured-anchored contact comparison',
        'state': 'prepared_waiting_for_GO',
        'created_local': datetime.now().astimezone().isoformat(),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'runner': str(Path(__file__).resolve()), 'runner_sha256': sha_file(Path(__file__).resolve()),
        'python_executable': str(PYTHON), 'python_version_at_prepare': sys.version,
        'helper': str(HELPER), 'helper_sha256': sha_file(HELPER),
        'results_dir': str(RESULTS), 'workspace': str(workspace),
        'fixed_thread_environment': THREAD_ENV,
        'laws': list(LAWS), 'scenario_order': [s['case'] for s in SCENARIOS],
        'case_count_expected': 24,
        'order_note': '24 cases run sequentially in separate helper processes, law-major; measured_anchored_ideal is last.',
        'fixed_parameters': {
            'force_N': 10, 'f_start_N': 10, 'ramp_s': .5, 'hold_s': 1,
            'speed_m_s': -.01, 'Q': 1, 'P': 2, 'kf_scale': .003,
            'eta_dev': .05, 'eta_u': 5e-8, 'N': 10, 'Tp_s': .1,
            'ERK_steps': 2, 'ERK_stages': 4, 'sliding_friction': .8,
            'noslip_iterations': 3, 'handheld_settle_time_s': 0,
            'flat_physics_log_dt_s': .002,
        },
        'halt_rule': 'On measured_anchored_ideal nonzero exit, solver failure, nonfinite critical arrays, or <0.1 N contact loss, retain artifacts and skip remaining measured_anchored_ideal cases for that scene only. Continue the other scene independently. Record actual joint limits without changing parameters.',
        'timing_rule': 'Report prepare/reference/solver/control median and p95 using control samples with t >= 0.1 s; process wall time is separately reported and is not solver timing.',
        'runs': planned_runs(workspace),
    }


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None


def finite_arrays(folder: Path) -> list[str]:
    import numpy as np
    bad = []
    critical = {
        'physics.npz': ('t', 'qpos', 'qvel', 'box', 'tipL', 'tipR', 'force'),
        'prediction.npz': ('t', 'state', 'force', 'force_pred_dot', 'parameters_shared_full',
                           'u', 'joint_q', 'joint_v', 'joint_violation', 'solver_status'),
        'joint_limits.npz': ('t', 'q', 'v', 'lower_margin', 'upper_margin', 'limit_torque',
                             'pred_joint_q', 'pred_joint_v', 'pred_solver_status'),
    }
    for name, keys in critical.items():
        path = folder / name
        if not path.exists():
            bad.append(f'missing:{name}')
            continue
        with np.load(path, allow_pickle=False) as z:
            for key in keys:
                if key not in z.files:
                    # Parent changes add complete all-node predictions; keep
                    # absent optional violation aliases compatible.
                    if key not in ('pred_joint_violation',):
                        bad.append(f'missing:{name}:{key}')
                    continue
                a = z[key]
                if np.issubdtype(a.dtype, np.number) and not np.isfinite(a).all():
                    bad.append(f'{name}:{key}')
    return bad


def case_health(run: dict) -> dict:
    import numpy as np
    folder = Path(run['output_dir'])
    summary = read_json(folder / 'summary.json') or {}
    metric_doc = read_json(Path(run['evidence_prefix'] + '.metrics.json')) or {}
    metrics = summary.get('metrics') or metric_doc.get('metrics') or {}
    physics = folder / 'physics.npz'
    pred = folder / 'prediction.npz'
    joint = folder / 'joint_limits.npz'
    contact_steps = contact_side_samples = zero_steps = None
    first_loss = None
    physics_dt_median = None
    if physics.exists():
        with np.load(physics, allow_pickle=False) as z:
            f = np.asarray(z['force']); t = np.asarray(z['t'])
            physics_dt_median = float(np.median(np.diff(t))) if len(t) > 1 else None
            loss = f < .1
            contact_steps = int(np.any(loss, axis=1).sum())
            contact_side_samples = int(loss.sum())
            zero_steps = int(np.any(f <= 0., axis=1).sum())
            at = np.flatnonzero(np.any(loss, axis=1))
            if at.size:
                i = int(at[0])
                first_loss = {'time_s': float(t[i]), 'physics_index': i,
                              'force_N': f[i].tolist(),
                              'loss_sides': np.flatnonzero(loss[i]).tolist(), 'threshold_N': .1}
    actual_limit_steps = predicted_bad = predicted_max = solver_bad = None
    if joint.exists():
        with np.load(joint, allow_pickle=False) as z:
            if 'active' in z.files:
                actual_limit_steps = int(np.any(z['active'], axis=1).sum())
            violation_key = 'pred_joint_violation' if 'pred_joint_violation' in z.files else 'pred_violation'
            if violation_key in z.files:
                v = np.asarray(z[violation_key])
                predicted_bad = int(np.count_nonzero(v > 0.))
                predicted_max = float(np.max(v, initial=0.))
    solver_statuses = []
    shared = None
    layout_values = []
    if pred.exists():
        with np.load(pred, allow_pickle=False) as z:
            if 'solver_status' in z.files:
                solver_statuses = np.asarray(z['solver_status']).astype(int).tolist()
            if 'parameters_shared_full' in z.files:
                shared = np.asarray(z['parameters_shared_full'])
            if 'parameter_layout' in z.files:
                layout_values = np.asarray(z['parameter_layout']).astype(str).tolist()
    solver_bad = sum(s != 0 for s in solver_statuses)
    expected_parameter_size = 228 if run['law'] == 'measured_anchored_ideal' else 225
    parameter_size = int(shared.shape[-1]) if shared is not None and shared.ndim else None
    parameter_size_mismatch = parameter_size != expected_parameter_size
    nonfinite = finite_arrays(folder)
    err = summary.get('error') or metric_doc.get('error')
    return {
        'summary_found': bool(summary), 'error': err,
        'solver_model': summary.get('solver_model') or metric_doc.get('solver_model'),
        'model_version': summary.get('model_version') or metrics.get('model_version'),
        'metrics': metrics, 'solver_status_bad_count': solver_bad,
        'solver_status_values': solver_statuses,
        'parameter_layout_values': sorted(set(layout_values)),
        'parameter_size': parameter_size,
        'parameter_size_expected': expected_parameter_size,
        'parameter_size_mismatch': parameter_size_mismatch,
        'actual_limit_active_steps': actual_limit_steps,
        'predicted_future_violation_entries': predicted_bad,
        'predicted_future_max_violation': predicted_max,
        'physics_contact_loss_steps_lt_0p1N': contact_steps,
        'physics_contact_loss_side_samples_lt_0p1N': contact_side_samples,
        'physics_zero_contact_steps': zero_steps,
        'physics_dt_median_s': physics_dt_median,
        'physics_dt_mismatch': bool(physics_dt_median is not None and abs(physics_dt_median-.002) > 2e-5),
        'first_physics_contact_loss_lt_0p1N': first_loss,
        'nonfinite_or_missing_arrays': nonfinite,
    }


def prepare(workspace: Path) -> None:
    if MANIFEST.exists():
        raise SystemExit(f'refusing to overwrite existing manifest: {MANIFEST}')
    data = initial_manifest(workspace.resolve())
    for run in data['runs']:
        ws = Path(run['workspace'])
        run['workspace_source_sha256_at_prepare'] = sha_tree(ws, exclude_codegen=True)
        cache = ws / 'acados_codegen'
        run['cache_sha256_at_prepare'] = sha_tree(cache) if cache.is_dir() else None
    RESULTS.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'manifest': str(MANIFEST), 'state': data['state'],
                      'workspace': data['workspace'], 'case_count': len(data['runs']),
                      'law_order': list(LAWS), 'scenario_order': data['scenario_order']},
                     ensure_ascii=False, indent=2))


def run_batch() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f'prepared manifest missing: {MANIFEST}; prepare first')
    data = read_json(MANIFEST)
    if data.get('state') != 'prepared_waiting_for_GO':
        raise SystemExit(f"manifest is not in prepared state: {data.get('state')}")
    if sha_file(Path(__file__).resolve()) != data.get('runner_sha256'):
        raise SystemExit('runner changed after preparation; rebuild the reviewed plan')
    if sha_file(HELPER) != data['helper_sha256']:
        raise SystemExit('helper changed after preparation; rebuild the reviewed plan')
    workspace = Path(data['workspace'])
    if not workspace.is_dir():
        raise SystemExit(f'workspace missing: {workspace}')
    for run in data['runs']:
        for path in [Path(run['output_dir']), Path(run['stdout']), *evidence_paths(Path(run['evidence_prefix']))]:
            if path.exists():
                raise SystemExit(f'refusing to overwrite existing case artifact: {path}')
    env = os.environ.copy(); env.update(THREAD_ENV)
    data['state'] = 'running'; data['run_started_local'] = datetime.now().astimezone().isoformat()
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    stopped_anchored_scenes = set()
    for run in data['runs']:
        if run['law'] == 'measured_anchored_ideal' and run['scene'] in stopped_anchored_scenes:
            run['status'] = 'skipped_after_previous_anchored_failure'
            run['error'] = f"Earlier measured_anchored_ideal case failed in {run['scene']}; artifacts retained."
            MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            continue
        ws = Path(run['workspace']); cache = ws / 'acados_codegen'
        run['status'] = 'running'; run['started_local'] = datetime.now().astimezone().isoformat()
        run['source_sha256_before'] = sha_tree(ws, exclude_codegen=True)
        run['cache_sha256_before'] = sha_tree(cache) if cache.is_dir() else None
        stdout = Path(run['stdout']); stdout.parent.mkdir(parents=True, exist_ok=True)
        Path(run['evidence_prefix']).parent.mkdir(parents=True, exist_ok=True)
        Path(run['output_dir']).parent.mkdir(parents=True, exist_ok=True)
        code = None; error = None; health = {}
        start = time.perf_counter()
        try:
            with stdout.open('x', encoding='utf-8', newline='') as log:
                log.write('COMMAND: ' + subprocess.list2cmdline(run['command']) + '\n'); log.flush()
                proc = subprocess.run(run['command'], cwd=str(BASE), env=env,
                                      stdout=log, stderr=subprocess.STDOUT, check=False)
                code = proc.returncode
            health = case_health(run)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            health = {'error': error, 'nonfinite_or_missing_arrays': []}
        run['elapsed_s'] = time.perf_counter() - start
        run['exitcode'] = code; run['error'] = error
        run['finished_local'] = datetime.now().astimezone().isoformat()
        run['source_sha256_after'] = sha_tree(ws, exclude_codegen=True)
        run['cache_sha256_after'] = sha_tree(cache) if cache.is_dir() else None
        run['health'] = health; run['solver_model'] = health.get('solver_model')
        run['model_version'] = health.get('model_version')
        for suffix, key in (('.model.mjb', 'mjb_sha256'), ('.reference.json', 'reference_sha256')):
            path = Path(run['evidence_prefix'] + suffix)
            run[key] = sha_file(path) if path.is_file() else None
        failed = (code != 0 or error is not None or health.get('error') is not None
                  or health.get('solver_status_bad_count', 0) > 0
                  or (health.get('physics_contact_loss_steps_lt_0p1N') or 0) > 0
                  or health.get('physics_dt_mismatch', False)
                  or health.get('parameter_size_mismatch', True)
                  or bool(health.get('nonfinite_or_missing_arrays')))
        run['status'] = 'failed' if failed else 'completed'
        if failed and run['law'] == 'measured_anchored_ideal':
            stopped_anchored_scenes.add(run['scene'])
            run['halt_reason'] = f"Preserve this case and skip remaining anchored cases for {run['scene']}; no parameter changes."
        MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'case_id': run['case_id'], 'status': run['status'],
                          'elapsed_s': run['elapsed_s'],
                          'actual_limit_active_steps': health.get('actual_limit_active_steps'),
                          'physics_contact_loss_steps_lt_0p1N': health.get('physics_contact_loss_steps_lt_0p1N'),
                          'solver_status_bad_count': health.get('solver_status_bad_count'),
                          'stdout': run['stdout']}, ensure_ascii=False), flush=True)
    data['state'] = 'completed_with_anchored_halt' if stopped_anchored_scenes else 'completed'
    data['run_finished_local'] = datetime.now().astimezone().isoformat()
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'manifest': str(MANIFEST), 'state': data['state']}, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description='Prepare or execute the fixed measured-anchored study.')
    ap.add_argument('--workspace', type=Path, required=True,
                    help='isolated modified workspace containing flat/ and handheld/')
    ap.add_argument('--run', action='store_true', help='execute only after explicit GO')
    args = ap.parse_args()
    if args.run:
        run_batch()
    else:
        prepare(args.workspace)


if __name__ == '__main__':
    main()
