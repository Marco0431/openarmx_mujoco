"""两个稳定命令行入口共用的实验主流程。"""
from dataclasses import replace
from pathlib import Path
import time
import mujoco.viewer
import numpy as np
from nmpcc.controller import Controller
from nmpcc.ocp import cost_weights
from .sweeps import has_sweep, run_sweep
from .data import RunPaths, RunResult
from .scene import build_scene, initialize_reference, controller_config, controller_bounds
from .runner import run_case, runtime_options
from .scenes import flat_cli, handheld_cli


def main(variant: str, entry_file: str | Path, *, co_locate_outputs: bool = False) -> None:
    """解析选项 → 建场景 → 初始抓取 → 建控制器 → 运行 → 保存。"""
    parser = (flat_cli if variant == 'flat' else handheld_cli).make_parser()
    args = parser.parse_args()
    if not np.isfinite(args.sliding_friction) or args.sliding_friction < 0:
        parser.error('sliding_friction 必须是非负有限数')
    if args.noslip_iterations < 0:
        parser.error('noslip_iterations 必须是非负整数')
    project = Path(entry_file).resolve().parent
    default_output = project / 'results' / ('exp2_latest' if variant == 'flat' else 'exp2_latest_grasp_pose')
    paths = RunPaths(project, project.parent / 'acados_codegen', default_output, Path(entry_file).resolve())
    if variant == 'flat' and args.output_dir:
        output = Path(args.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)
        paths = replace(paths, output_root=output)
    elif co_locate_outputs and variant == 'handheld' and args.output_dir:
        # 便携入口显式要求 CSV、图片、扰动和归档同目录；目录由 execute_run 创建。
        paths = replace(paths, output_root=Path(args.output_dir).resolve())
    print(f'[command] y_dist={args.y_dist:.6f} m, v_y={args.v_y:.6f} m/s, hold_only={args.hold_only}')
    print(f'[output] CSV and Fig. 12 files: {paths.output_root}')
    scene = build_scene(variant, args)
    cache = {}
    sweep = has_sweep(args)

    def run_one(kf: float, force: float, tag: str) -> dict[str, object]:
        args.kf_scale, args.force = kf, force
        reference = initialize_reference(scene, args)
        config = controller_config(scene, args)
        # 实验扫描只在结构配置变化时创建另一控制器；每次运行公开reset。
        key = repr(config)
        controller = cache.get(key)
        if controller is None:
            umin, umax, qmin, qmax = controller_bounds(scene)
            controller = Controller(
                config, scene.fk, cache_root=paths.cache_root,
                umin=umin, umax=umax, qmin=qmin, qmax=qmax,
            )
            cache[key] = controller
            print(f'NMPCC built: K={scene.geometry.K_STIFF:.3f}, rest_gap={config.rest_gap:.6f}, N={args.N}, Tp={args.Tp:.3f}')
        return execute_run(scene, args, controller, reference, paths, tag, sweep)

    if sweep:
        run_sweep(args, run_one, str(paths.output_root))
        return
    metrics = run_one(args.kf_scale, args.force, 'single')
    print('--- latest Experiment II (summary) ---')
    for key, value in metrics.items():
        print(f'  {key}: {value}')


def execute_run(scene, args, controller: Controller, reference, paths: RunPaths,
                tag: str, sweep: bool) -> dict[str, object]:
    result = RunResult()
    metadata = controller.metadata
    stage_weights, terminal_weights = cost_weights(controller.config, runtime_options(args).weights)
    result.metadata.update(config=metadata['config'], solver_model=metadata['name'],
        contact_face_rest_gap=controller.config.rest_gap,
        stage_weights_diagonal=np.diag(stage_weights), terminal_weights_diagonal=np.diag(terminal_weights))
    viewer = None
    trace_dir = None
    if scene.variant == 'handheld':
        trace_dir = (Path(args.output_dir).resolve() if args.output_dir else
                     paths.output_root / ('trace_' + tag + '_' + time.strftime('%Y%m%d_%H%M%S')))
        if sweep and args.output_dir:
            trace_dir = trace_dir / tag
        trace_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    error = None
    try:
        if not args.no_viewer and not sweep:
            viewer = mujoco.viewer.launch_passive(scene.model, scene.data)
            viewer.cam.lookat[:] = [0., -.13, .11]
            viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 1.8, -12, 180
            viewer.sync()
            print('viewer opened; close the window to stop the experiment')
        run_case(scene, args, controller, reference, paths, result, viewer)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        if scene.variant == 'flat':
            scene.output.save_failure(result, tag, str(paths.output_root))
        raise
    finally:
        if scene.variant == 'handheld':
            scene.output.save_run(trace_dir, args, scene.model, scene.data, scene.arms, scene.ids, result,
                reference, result.log, result.horizon, result.metrics, error, entry_file=paths.entry_file,
                base_dir=paths.project_root, box_center=scene.box_center, kx=scene.geometry.K_STIFF,
                rest_gap=scene.geometry.REST_GAP, scene_path=scene.source)
            print(f'[saved] {trace_dir}')
    return scene.output.finish_run(result.log, result.horizon, result.metrics, args, result, tag,
        time.perf_counter() - started, str(paths.output_root), str(paths.entry_file))
