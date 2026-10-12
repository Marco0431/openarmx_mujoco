"""固定批次实验入口：复用公开控制循环，额外保存分析所需的准确参考与模型。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

import mujoco
import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def artifact(prefix, suffix):
    # 参数名含小数点，不能用 with_suffix 截掉最后一段参数值。
    return prefix.parent / (prefix.name + suffix)


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if hasattr(value, 'value'):
        return serializable(value.value)
    if hasattr(value, '__dict__'):
        return serializable(vars(value))
    return value


def numeric_record_arrays(records):
    """Stack ordinary numeric/string row fields; dictionaries stay out of NPZ."""
    if not records:
        return {}
    keys = sorted(set.intersection(*(set(row) for row in records)))
    arrays = {}
    for key in keys:
        values = [row[key] for row in records]
        if any(isinstance(value, dict) for value in values):
            continue
        try:
            array = np.asarray(values)
        except (TypeError, ValueError):
            continue
        if array.dtype.kind in 'biufcSU':
            arrays[key] = array
    return arrays


def save_unified_outputs(output, prefix, scene, args, reference, controller,
                         result, initial, started, error, entry, model_version):
    """Save a common, lossless-enough NPZ interface for both scene variants."""
    output.mkdir(parents=True, exist_ok=True)
    # run_case's recorder exports physics/joint traces on success and failure.
    np.savez_compressed(output / 'physics.npz', **result.physics_trace)
    prediction_arrays = numeric_record_arrays(result.horizon)
    # Preserve the decomposed measured contact-load path from every update.
    # These terms are independent evidence and must not be reconstructed from
    # the model's own predicted spring force.
    if result.horizon and all(isinstance(row.get('contact_load'), dict) for row in result.horizon):
        names = sorted(set.intersection(*(set(row['contact_load']) for row in result.horizon)))
        for name in names:
            prediction_arrays['contact_' + name] = np.asarray(
                [row['contact_load'][name] for row in result.horizon])
    np.savez_compressed(output / 'prediction.npz', **prediction_arrays)
    np.savez_compressed(output / 'joint_limits.npz', **result.joint_limit_trace)
    np.savez_compressed(output / 'startup.npz', **numeric_record_arrays(result.startup))
    np.savez_compressed(output / 'trajectory.npz', **numeric_record_arrays(result.log))
    np.savez_compressed(output / 'motion_force.npz', **result.motion_force_trace)
    log_arrays = numeric_record_arrays(result.log)
    np.savez_compressed(artifact(prefix, '.result_log.npz'), **log_arrays)
    solver = controller.metadata if controller is not None else {}
    document = dict(
        args=vars(args), reference=serializable(vars(reference)), initial_state=initial,
        arms=scene.arms, box_body_name=scene.model.body(scene.ids['box_body']).name,
        scene_path=str(scene.source), scene_sha256=initial['scene_sha256'],
        physics_dt=.002, control_period_s=.006,
        config=serializable(solver.get('config', {})),
        solver_model=solver.get('name'), model_version=model_version,
        parameter_layout=solver.get('parameter_layout'), parameter_size=solver.get('parameter_size'),
        contact_face_rest_gap=scene.contact_rest_gap,
        stage_weights_diagonal=result.metadata.get('stage_weights_diagonal'),
        terminal_weights_diagonal=result.metadata.get('terminal_weights_diagonal'),
        metrics=result.metrics, error=error, failure=result.failure,
        elapsed_before_output_s=time.perf_counter()-started,
        output_dir=str(output), evidence_prefix=str(prefix),
        source_entry_file=str(entry),
    )
    (output / 'summary.json').write_text(
        json.dumps(serializable(document), ensure_ascii=False, indent=2, allow_nan=True), encoding='utf-8')
    artifact(prefix, '.metrics.json').write_text(json.dumps(serializable(dict(
        metrics=result.metrics, error=error, solver_model=solver.get('name'),
        elapsed_before_output_s=time.perf_counter()-started,
        model_version=model_version,
        parameter_layout=solver.get('parameter_layout'), parameter_size=solver.get('parameter_size'))),
        ensure_ascii=False, indent=2, allow_nan=True), encoding='utf-8')


def main():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument('--workspace', type=Path, required=True)
    own.add_argument('--variant', choices=('flat', 'handheld'), required=True)
    own.add_argument('--evidence_prefix', type=Path, required=True)
    known, rest = own.parse_known_args()
    project = known.workspace.resolve() / known.variant
    sys.path.insert(0, str(known.workspace.resolve()))
    sys.path.insert(0, str(project))
    os.chdir(project)
    from experiments.scenes import flat_cli, handheld_cli
    from experiments.scene import build_scene, initialize_reference, controller_config, controller_bounds
    from experiments.runner import run_case, runtime_options
    from experiments.data import RunPaths, RunResult
    from nmpcc.controller import Controller
    from nmpcc.ocp import MODEL_VERSION, cost_weights

    args = (flat_cli if known.variant == 'flat' else handheld_cli).make_parser().parse_args(rest)
    assert args.no_viewer and args.output_dir and not args.dynamics_diag
    assert args.sliding_friction == .8 and args.noslip_iterations == 3
    output = Path(args.output_dir).resolve()
    assert not output.exists(), output
    prefix = known.evidence_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    assert not artifact(prefix, '.reference.json').exists(), prefix
    # 与正式CLI同样的路径规则；两版在批次中都将图像与日志放入独立case目录。
    if known.variant == 'flat':
        output.mkdir(parents=True)
    entry = project / ('exp2_nmpcc_latest.py' if known.variant == 'flat'
                       else 'exp2_nmpcc_latest_grasp_pose.py')
    paths = RunPaths(project, known.workspace.resolve() / 'acados_codegen', output, entry)
    print(f'[study] variant={known.variant} output={output}', flush=True)
    scene = build_scene(known.variant, args)
    reference = initialize_reference(scene, args)
    initial = dict(args=vars(args), reference=vars(reference),
        arm_body_names=[scene.model.body(a['body_id']).name for a in scene.arms],
        box_body_name=scene.model.body(scene.ids['box_body']).name,
        box_initial_position=scene.data.xpos[scene.ids['box_body']].copy(),
        initial_qpos=scene.data.qpos.copy(), initial_qvel=scene.data.qvel.copy(),
        initial_time=float(scene.data.time), physics_dt=.002, control_every=3,
        scene_path=str(scene.source), scene_sha256=hashlib.sha256(scene.source.read_bytes()).hexdigest())
    artifact(prefix, '.reference.json').write_text(
        json.dumps(initial, default=json_value, ensure_ascii=False, indent=2), encoding='utf-8')
    mujoco.mj_saveModel(scene.model, str(artifact(prefix, '.model.mjb')), None)
    config = controller_config(scene, args)
    assert config.integrator_type == 'ERK'
    assert config.sim_method_num_steps == 2 and config.sim_method_num_stages == 4
    umin, umax, qmin, qmax = controller_bounds(scene)
    controller = Controller(config, scene.fk, cache_root=paths.cache_root,
                            umin=umin, umax=umax, qmin=qmin, qmax=qmax)
    result = RunResult()
    metadata = controller.metadata
    stage, terminal = cost_weights(config, runtime_options(args).weights)
    result.metadata.update(config=metadata['config'], solver_model=metadata['name'],
        model_version=MODEL_VERSION, parameter_layout=metadata.get('parameter_layout'),
        parameter_size=metadata.get('parameter_size'),
        contact_face_rest_gap=config.rest_gap, stage_weights_diagonal=np.diag(stage),
        terminal_weights_diagonal=np.diag(terminal))
    if known.variant == 'handheld':
        output.mkdir(parents=True)
    started = time.perf_counter()
    error = None
    try:
        # 真实控制、两次预热、限位观测与停止规则全部来自生产公共函数。
        run_case(scene, args, controller, reference, paths, result, None)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        if known.variant == 'flat':
            scene.output.save_failure(result, 'single', str(output))
        raise
    finally:
        # 即使失败，也保留已完成采样的实际位置/力/状态，不仅保存终点摘要。
        keys = sorted(set().union(*(row.keys() for row in result.log))) if result.log else []
        scalar_columns = {}
        for key in keys:
            values = [row.get(key, np.nan) for row in result.log]
            # 分析器用这些实际位置向量核对 MJB 几何重放；其余字段只导出标量。
            if key in ('tipL', 'tipR', 'box', 'rotvec') or all(np.ndim(value) == 0 for value in values):
                scalar_columns[key] = np.array(values)
        np.savez_compressed(artifact(prefix, '.result_log.npz'), **scalar_columns)
        artifact(prefix, '.metrics.json').write_text(json.dumps(
            dict(metrics=result.metrics, error=error, solver_model=metadata['name'],
                 elapsed_before_output_s=time.perf_counter()-started,
                 actual_stage_weights_diagonal=np.diag(stage),
                 actual_terminal_weights_diagonal=np.diag(terminal)),
            default=json_value, ensure_ascii=False, indent=2), encoding='utf-8')
        if known.variant == 'handheld':
            scene.output.save_run(output, args, scene.model, scene.data, scene.arms, scene.ids, result,
                reference, result.log, result.horizon, result.metrics, error, entry_file=entry,
                base_dir=project, box_center=scene.box_center, kx=scene.geometry.K_STIFF,
                rest_gap=scene.geometry.REST_GAP, scene_path=scene.source)
        # Both variants receive the same authoritative NPZ set, including when
        # the control loop failed partway through and only partial traces exist.
        save_unified_outputs(output, prefix, scene, args, reference, controller,
                             result, initial, started, error, entry, MODEL_VERSION)
        if result.metrics:
            scene.output.finish_run(result.log, result.horizon, result.metrics, args, result, 'single',
                time.perf_counter()-started, str(output), str(entry))
    print('[study] completed', flush=True)


if __name__ == '__main__':
    main()
