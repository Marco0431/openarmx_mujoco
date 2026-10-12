"""两个场景共用的闭环流程：采样、参考、求解、执行、记录、积分。"""
from pathlib import Path
import time
import mujoco
import numpy as np
from nmpcc.config import SolveOptions, RuntimeWeights
from nmpcc.data import JointState
from diagnostics.measurements import sample_dynamics
from .disturbance import ForcePulse
from .recording import Recorder

DT = .002
CONTROL_EVERY = 3


def run_case(scene, args, controller, reference, paths, result, viewer=None):
    """运行数据独立保存；任何退出路径都撤除扰动并导出已完成的记录。"""
    model, data, arms, ids = scene.model, scene.data, scene.arms, scene.ids
    required_end = reference.t_hold_end + (.3 if args.hold_only else reference.move_duration + .4)
    end_time = reference.t_hold_end + .3 if args.hold_only else max(args.steps * DT, required_end)
    pulse = ForcePulse(model, data, args, end_time, paths.output_root)
    result.disturbance_log = str(pulse.path.resolve()) if pulse.enabled else None
    recorder = Recorder(
        model, data, arms, ids, args, controller.config, reference, scene.variant, result,
    )
    # 1. 从同一次测量建立平衡种子，完成两次不推进仿真的预热。
    state = dynamics = references = seed = options = None
    try:
        controller.reset()
        options = runtime_options(args)
        diagnostic = {} if args.dynamics_diag else None
        state, dynamics = sample_current(scene, diagnostic)
        references = reference.horizon(0., controller.node_times)
        seed = controller.equilibrium_seed(state, dynamics, references, options)
        validate_seed(seed, model, arms, args.u_lim)
        recorder.joints.metadata.update(startup_seed=seed.copy(),
            startup_seed_kind='current prediction model equilibrium, not plant equilibrium')
        options.u_seed = seed
        # 两次预热共用同次采样，仿真时间不前进；核心初始化全部 N+1 状态。
        for _ in range(2):
            recorder.solution = None
            solved = controller.solve(state, dynamics, references, options)
            debug_solution(args, solved)
            recorder.solution = solved
            recorder.recovered_failure(solved, 0.)
            require_solution(solved, 'warmup')
            recorder.warmup(state, dynamics, references, solved, options)
        initial_box_y = float(data.xpos[ids['box_body']][1])
        if reference.move_duration > 0:
            print(f'[timed] ramp={args.t_ramp:.3f}s, holding={args.t_hold:.3f}s, '
                  f'moving starts at t={reference.t_hold_end:.3f}s')
        else:
            print(f'[timed] ramp={args.t_ramp:.3f}s, then holding; translation disabled')
        lower = np.concatenate([model.actuator_ctrlrange[a['act_ids'], 0] for a in arms])
        upper = np.concatenate([model.actuator_ctrlrange[a['act_ids'], 1] for a in arms])
        command = seed.copy() if scene.variant == 'handheld' else np.zeros(14)
        clipped = np.zeros(14, dtype=bool)
        saturated = np.isclose(command, lower, atol=1e-6) | np.isclose(command, upper, atol=1e-6)
        phase, reference_ms = 'ramp', 0.
        steps = max(args.steps, int(np.ceil(required_end / DT)))
        # 2. 扰动 → 采样与求解 → 执行力矩 → 积分 → 记录，保持原执行顺序。
        for step in range(steps):
            t = step * DT
            pulse.begin_step(t)
            solve_ms = prepare_ms = np.nan
            settling = step < int(round(getattr(args, 'settle_time', 0.) / DT))
            control_step = not settling and step % CONTROL_EVERY == 0
            if settling:
                from .legacy.handheld_output import settling_command
                command = settling_command(data, arms, lower, upper)
            elif control_step:
                started = time.perf_counter()
                # 本拍准备失败时不能误存上一拍的参数或预测。
                recorder.solution = None
                state = dynamics = references = None
                state, dynamics = sample_current(scene, diagnostic)
                reference_started = time.perf_counter()
                references = reference.horizon(t, controller.node_times)
                phase = reference.phase(t)
                reference_ms = (time.perf_counter() - reference_started) * 1000.
                prepare_ms = (time.perf_counter() - started) * 1000.
                options.u_seed = seed
                solved = controller.solve(state, dynamics, references, options)
                debug_solution(args, solved)
                recorder.solution = solved
                recorder.recovered_failure(solved, t)
                require_solution(solved, 'control')
                seed = solved.torque.copy()
                recorder.prediction(
                    state, dynamics, references, solved, options, t, phase, prepare_ms, reference_ms,
                )
                command = np.clip(solved.torque, lower, upper)
                clipped = np.abs(command - solved.torque) > 1e-9
                saturated = (clipped | np.isclose(command, lower, atol=1e-6)
                             | np.isclose(command, upper, atol=1e-6))
                solve_ms = (time.perf_counter() - started) * 1000.
                result.horizon[-1]['control_ms'] = solve_ms
            for side, arm in enumerate(arms):
                data.ctrl[arm['act_ids']] = command.reshape(2, 7)[side]
            if control_step and diagnostic is not None:
                recorder.dynamics(controller, state, dynamics, references, options, command, diagnostic, t)
            # 步前状态与本次 Euler 积分算出的反力配对。
            recorder.before_step()
            mujoco.mj_step(model, data)
            pulse.end_step()
            recorder.after_step(command, step)
            if viewer is not None:
                viewer.sync()
                time.sleep(DT)
            recorder.sample(
                step, t, phase, solve_ms, prepare_ms, reference_ms, command, clipped, saturated, settling,
            )
            stop_file = getattr(args, 'disturb_stop_file', None)
            if stop_file and Path(stop_file).exists():
                print('[disturbance] 已收到停止请求，撤力并保存当前日志')
                break
            if args.hold_only and t >= reference.t_hold_end + .3:
                break
        # 3. 统计观测序列；finally 对正常结束及失败统一撤力和导出。
        result.metrics = scene.output.summarize(result.log, args, initial_box_y, reference.move_duration,
            box_center=scene.box_center, hold_box_y_tol=.002, motion_timing=scene.motion_timing, physics_dt=DT)
        for key in ('prepare_ms', 'reference_ms', 'solver_ms', 'control_ms'):
            values = np.array([h[key] for h in result.horizon[10:]], dtype=float)
            if values.size:
                result.metrics[key + '_p50'] = float(np.percentile(values, 50))
                result.metrics[key + '_p95'] = float(np.percentile(values, 95))
        return result
    except Exception as exc:
        recorder.failure(exc, state, dynamics, references, seed, options)
        raise
    finally:
        try:
            pulse.release()
            pulse.save()
        finally:
            try:
                recorder.export()
            finally:
                if viewer is not None:
                    viewer.close()


def sample_current(scene, diagnostic=None):
    data, arms = scene.data, scene.arms
    state = JointState(np.array([data.qpos[a['qpos']] for a in arms]),
                       np.array([data.qvel[a['dofs']] for a in arms]), float(data.time))
    dynamics = sample_dynamics(scene.model, data, arms, scene.ids['box_geom'], diagnostic=diagnostic)
    return state, dynamics


def runtime_options(args) -> SolveOptions:
    weights = None
    if any(getattr(args, k, None) is not None for k in ('Q_rt', 'P_rt', 'ed_rt', 'eu_rt', 'ed_x_rt')):
        weights = RuntimeWeights(Q=args.Q_rt, P=args.P_rt, eta_dev=args.ed_rt, eta_u=args.eu_rt, ed_x=args.ed_x_rt)
    return SolveOptions(u_lim=args.u_lim, kf_scale=args.kf_scale, weights=weights)


def validate_seed(seed: np.ndarray, model: mujoco.MjModel, arms: list[dict],
                  u_lim: float | np.ndarray | None) -> None:
    lower = np.concatenate([model.actuator_ctrlrange[a['act_ids'], 0] for a in arms])
    upper = np.concatenate([model.actuator_ctrlrange[a['act_ids'], 1] for a in arms])
    if u_lim is not None:
        lower, upper = np.maximum(lower, -u_lim), np.minimum(upper, u_lim)
    if (not np.isfinite(seed).all() or np.isnan(lower).any() or np.isnan(upper).any()
            or np.any(lower > upper) or np.any(seed < lower) or np.any(seed > upper)):
        raise ValueError('model-equilibrium seed is outside effective torque bounds; startup stopped')


def require_solution(solved, stage: str) -> None:
    if solved.status != 0:
        raise RuntimeError(f'NMPCC {stage} status {solved.status}')
    if not np.isfinite(solved.torque).all() or not np.isfinite(solved.prediction.states).all():
        raise RuntimeError('non-finite NMPCC solution')


def debug_solution(args, solved) -> None:
    """保留手持实验 --debug_eps 的逐次求解诊断，包含两次预热。"""
    if getattr(args, 'debug_eps', False):
        residual = solved.prediction.residuals[0]
        rate = solved.prediction.residual_rates[0]
        print(f'[epsdbg] status={solved.status} eps1={residual[0, :3]} '
              f'eps2={residual[1, :3]} epsd1x={rate[0, 0]:+.5f} epsd2x={rate[1, 0]:+.5f}')

