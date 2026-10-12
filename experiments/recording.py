"""只接收命名采样与 SolveResult 的记录器，日志归 RunResult 所有。"""
from dataclasses import asdict, replace
import time
import mujoco
import numpy as np
from ik import get_ee_tip
from nmpcc.data import JointState, DynamicsSnapshot, ReferenceHorizon, SolveResult
from nmpcc.parameters import (encode_legacy_horizon, encode_shared_horizon, layout_version,
                              parameter_size, to_legacy, to_legacy_full)
from diagnostics.measurements import JointLimitRecorder, measure_contact, box_attitude, _rotvec_of_R, dynamics_phase
from .data import RunResult
from .reference import smoothstep
from .motion_trace import build_motion_trace


class Recorder:
    def __init__(self, model, data, arms, ids, args, config, reference, variant, result):
        self.model, self.data, self.arms, self.ids = model, data, arms, ids
        self.args, self.config, self.reference, self.variant = args, config, reference, variant
        self.result = result
        self.joints = JointLimitRecorder(model, data, arms, dict(
            args=vars(args).copy(), config=asdict(config), model_version='typed_nmpcc_v1',
            parameter_layout=layout_version(config), parameter_size=parameter_size(config),
            solver_model=result.metadata.get('solver_model'), joint_bounds_enabled=True,
            physics_dt=.002, control_every=3, contact_load_mode=args.contact_load_mode,
            contact_load_revision='frozen_total_minus_current_model_normal_v1' if args.contact_load_mode == 'residual' else 'tangent_v1'))
        self.physics, self.dynamics_rows = [], []
        self.motion_rows = []
        self.solution = None
        self.first_failure = None
        self.shadow = mujoco.MjData(model) if args.dynamics_diag else None

    def prediction(self, state: JointState, dynamics: DynamicsSnapshot, references: ReferenceHorizon,
                   solved: SolveResult, options, t, phase, prepare_ms, reference_ms):
        self.solution = solved
        pred = solved.prediction
        residual = pred.residuals.reshape(len(pred.states), 12)
        residual_rate = pred.residual_rates.reshape(len(pred.states), 12)
        row = dict(t=t, phase=phase, force_ref=references.forces[:, :, 0].copy(),
            reference=np.concatenate((references.poses.reshape(-1, 12), references.velocities.reshape(-1, 12), references.forces.reshape(-1, 12)), axis=1),
            force_ref_dot=references.force_rates.reshape(-1, 12).copy(), node_times=references.node_times.copy(),
            prepare_ms=prepare_ms, reference_ms=reference_ms, solver_ms=solved.statistics['solver_ms'],
            force=pred.forces.copy(), force_error=pred.force_error.copy(), position_error_x=pred.position_error_x.copy(),
            eps_x=residual.copy(), epsd_x=residual_rate.copy(), u=pred.controls.copy(),
            u_ref=dynamics.gravity.reshape(14).copy(), solver_status=solved.status)
        # 旧 parameters 字段供历史工具读取；完整布局必须单独保存，包含新律的锚点。
        shared = encode_shared_horizon(state, dynamics, references, solved.anchoring,
                                       solved.contact_load, self.config, options)
        row.update(parameters=to_legacy(shared, self.variant), parameters_shared_full=shared,
                   parameter_layout=layout_version(self.config), state=pred.states.copy())
        if getattr(pred, 'force_rates', None) is not None:
            row['force_pred_dot'] = pred.force_rates.copy()
        if 'anchor_gap' in solved.anchoring:
            row.update(force_anchor_gap=solved.anchoring['anchor_gap'].copy(),
                       force_anchor_normal=solved.anchoring['anchor_normal'].copy())
        if self.args.contact_load_mode == 'residual':
            row['contact_load'] = {key: value.reshape(-1).copy() for key, value in solved.contact_load.items()}
        started = time.perf_counter()
        q = np.concatenate((pred.states[:, :7], pred.states[:, 14:21]), axis=1)
        row.update(joint_q=q, joint_v=np.concatenate((pred.states[:, 7:14], pred.states[:, 21:28]), axis=1),
            joint_violation=np.maximum(0., np.maximum(self.joints.lo-q, q-self.joints.hi)),
            joint_diagnostic_ms=(time.perf_counter()-started)*1000.)
        self.result.horizon.append(row)
        self.recovered_failure(solved, t)

    def warmup(self, state, dynamics, references, solved, options):
        self.solution = solved
        self.recovered_failure(solved, 0.)
        shared = encode_shared_horizon(state, dynamics, references, solved.anchoring,
                                       solved.contact_load, self.config, options)
        self.result.startup.append(dict(x=solved.prediction.states.copy(), u=solved.prediction.controls.copy(),
            parameters=to_legacy(shared, self.variant), parameters_shared_full=shared,
            parameter_layout=layout_version(self.config),
            force=solved.prediction.forces.copy(), status=solved.status))

    def _legacy_failure_context(self, context):
        """只在日志边界转换参数布局；求解器返回的原始快照不作修改。"""
        converted = dict(context)
        for name in ('p_full', 'first_p_full'):
            if name in converted:
                raw_name = 'p_shared_full' if name == 'p_full' else 'first_p_shared_full'
                converted[raw_name] = np.asarray(converted[name]).copy()
                converted[name] = to_legacy_full(converted[name], self.variant)
        return converted

    def recovered_failure(self, solved, t):
        if solved.failure and self.first_failure is None:
            self.first_failure = dict(solved.failure)
            self.result.failure = dict(t=t, message='first failed solve, before rebuild',
                **self._legacy_failure_context(self.first_failure))

    def failure(self, exc, state=None, dynamics=None, references=None, seed=None, options=None):
        """终止现场放主字段，第一次求解失败独立放 first_*，二者均完整保留。"""
        context = getattr(exc, 'failure', None) or {}
        current_x = np.concatenate([part for arm in self.arms
            for part in (self.data.qpos[arm['qpos']], self.data.qvel[arm['dofs']])])
        node_count = self.config.N + 1
        # 采样或初始化尚未完成时，保留旧字段形状，以 NaN 明确表示不可用。
        failure = dict(t=float(self.data.time), x0=current_x,
            p=to_legacy(np.full((node_count, parameter_size(self.config)), np.nan), self.variant),
            p_shared_full=np.full((node_count, parameter_size(self.config)), np.nan),
            parameter_layout=layout_version(self.config),
            f_meas=np.full(2, np.nan),
            u_seed=np.full(current_x.size // 2, np.nan) if seed is None else np.asarray(seed).copy(),
            qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(),
            pred_x=np.full((node_count, current_x.size), np.nan))
        if dynamics is not None:
            measured = dynamics.contact
            failure['f_meas'] = (measured.normal_magnitude * [-1., 1.] if self.variant == 'flat'
                else np.concatenate((measured.force_world, measured.contact_torque_world), axis=1).reshape(12))
        if self.solution is not None:
            failure['pred_x'] = self.solution.prediction.states.copy()
            if state is not None and dynamics is not None and references is not None:
                failure['p_shared_full'] = encode_shared_horizon(state, dynamics, references,
                    self.solution.anchoring, self.solution.contact_load, self.config, options)
                failure['p'] = to_legacy(failure['p_shared_full'], self.variant)
        if self.first_failure is not None:
            first = self._legacy_failure_context(self.first_failure)
            failure.update({'first_' + key: value for key, value in first.items()})
        # 重建构造异常可能没有 SolveResult；公开异常上下文提供最新迭代和首失败。
        failure.update(self._legacy_failure_context(context))
        if 'p_full' in context:
            failure['p'] = to_legacy(context['p_full'], self.variant)
        failure['message'] = str(exc)
        self.result.failure = failure
        self.export()

    def before_step(self):
        self.captured = self.joints.before_step(self.data)
        self.pre_qpos, self.pre_qvel = self.data.qpos.copy(), self.data.qvel.copy()
        # 可动基座若由 mocap 驱动，也须保留同拍几何状态；不刷新主仿真。
        self.pre_mocap_pos, self.pre_mocap_quat = self.data.mocap_pos.copy(), self.data.mocap_quat.copy()

    def after_step(self, torque, step):
        self.joints.after_step(self.model, self.data, self.captured)
        self.step_contact = None
        # 两场景均保留每个 2 ms 物理步，避免 10 ms 抽样漏掉短暂单侧失接触。
        started = time.perf_counter()
        self.step_contact = measure_contact(self.model, self.data, self.arms, self.ids['box_geom'], joint_loads=False)
        counts = np.zeros(2, dtype=int)
        for contact in self.data.contact[:self.data.ncon]:
            for i, arm in enumerate(self.arms):
                if {int(contact.geom1), int(contact.geom2)} == {arm['pad_id'], self.ids['box_geom']}:
                    counts[i] += 1
        self.physics.append(dict(t=self.captured[0][0], box=self.data.xpos[self.ids['box_body']].copy(),
            qpos=self.pre_qpos, qvel=self.pre_qvel, u=torque.copy(),
            rotvec=_rotvec_of_R(self.data.xmat[self.ids['box_body']].reshape(3, 3)),
            tipL=get_ee_tip(self.data, self.arms[0]['body_id']).copy(),
            tipR=get_ee_tip(self.data, self.arms[1]['body_id']).copy(),
            force=self.step_contact.normal_magnitude.copy(), contact_count=counts,
            diagnostic_ms=(time.perf_counter()-started)*1000.))
        if step % 5 == 0:
            # 复用已有步前状态与本次 Euler 接触力，10 ms 一行；速度在退出后重放计算。
            self.motion_rows.append(dict(step=step, t=self.captured[0][0],
                legacy_csv_t=step*.002 + (.002 if self.variant == 'flat' else 0.),
                qpos=self.pre_qpos, qvel=self.pre_qvel,
                mocap_pos=self.pre_mocap_pos, mocap_quat=self.pre_mocap_quat,
                normal_force=self.step_contact.normal_magnitude.copy()))

    def sample(self, step, t, phase, solve_ms, prepare_ms, reference_ms, torque, clipped, saturated, settling=False):
        if step % 5:
            return
        contact, solved = self.step_contact, self.solution
        body = self.ids['box_body']
        box = self.data.xpos[body].copy()
        if self.variant == 'flat':
            rv, quat = box_attitude(self.data, self.ids['box_qpos'])
        else:
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, self.data.xmat[body])
            rv = _rotvec_of_R(self.data.xmat[body].reshape(3, 3))
        box_force = -contact.force_world.sum(axis=0)
        box_torque = -(contact.wrench_world[:, 3:] + np.cross(contact.reference_points_world-box, contact.force_world)).sum(axis=0)
        nodes = [0, self.args.N//2, self.args.N]
        inputs = [0, self.args.N//2, self.args.N-1]
        pred = solved.prediction
        anchoring = {k: (solved.anchoring[k][:, 0] if self.variant == 'flat'
                         else solved.anchoring[k].reshape(12)).copy()
                     for k in ('nominal', 'measured', 'bias', 'anchored')}
        residual = pred.residuals.reshape(-1, 12)
        residual_rate = pred.residual_rates.reshape(-1, 12)
        is_control = not settling and step % 3 == 0
        stats = solved.statistics
        row = dict(t=t+.002 if self.variant == 'flat' else t, phase='settle' if settling else phase,
            ramp_s=float(smoothstep(t/self.args.t_ramp)) if self.args.t_ramp > 1e-9 else 1.,
            moving_active=float(phase == 'move'), box=box, quat=quat, rotvec=rv,
            tipL=get_ee_tip(self.data,self.arms[0]['body_id']).copy(), tipR=get_ee_tip(self.data,self.arms[1]['body_id']).copy(),
            wL=(contact.force_world[0].copy(), contact.contact_torque_world[0].copy()),
            wR=(contact.force_world[1].copy(), contact.contact_torque_world[1].copy()), box_f=box_force, box_t=box_torque,
            u_cmd=torque.copy(), u_clip_mask=clipped.copy(), u_sat_mask=saturated.copy(),
            FL=contact.normal_magnitude[0], FR=contact.normal_magnitude[1],
            force_nominal=anchoring['nominal'], force_measured_control=anchoring['measured'],
            force_bias=anchoring['bias'], force_anchored=anchoring['anchored'],
            force_ref=pred.references.forces[0,:,0].copy(), pred_force=pred.forces[nodes].copy(),
            pred_force_error=pred.force_error[nodes].copy(), pred_position_error_x=pred.position_error_x[nodes].copy(),
            pred_eps_x=residual[nodes].copy(), pred_epsd_x=residual_rate[nodes].copy(), pred_u=pred.controls[inputs].copy(),
            solve_ms=solve_ms if is_control else np.nan, prepare_ms=prepare_ms if is_control else np.nan,
            reference_ms=reference_ms if is_control else np.nan, solver_ms=stats['solver_ms'] if is_control else np.nan,
            solver_status=float(solved.status) if is_control else np.nan, qp_iter=float(stats['qp_iter']) if is_control else np.nan,
            solver_qp_ms=stats['time_qp']*1000. if is_control else np.nan,
            solver_sim_ms=stats['time_sim']*1000. if is_control else np.nan,
            solver_lin_ms=stats['time_lin']*1000. if is_control else np.nan)
        # 稳定阶段尚无控制预测时，旧日志用 NaN 区分预热解与实际控制解。
        if not self.result.horizon:
            for name in ('force_nominal', 'force_measured_control', 'force_bias', 'force_anchored',
                         'pred_force', 'pred_force_error', 'pred_position_error_x',
                         'pred_eps_x', 'pred_epsd_x', 'pred_u'):
                row[name] = np.full_like(row[name], np.nan, dtype=float)
            row['force_ref'] = np.array([-self.args.f_start, self.args.f_start])
        if self.variant == 'handheld':
            ref = self.reference.sample(t)
            row.update(f_ref=ref.fmag, refLy=float(ref.refs['left'][1]), refRy=float(ref.refs['right'][1]),
                vrefLy=float(ref.vrefs['left'][1]), vrefRy=float(ref.vrefs['right'][1]),
                box_vy=float(self.pre_qvel[int(self.model.body_dofadr[body])+1]))
        else:
            print(f"t={t+.002:.3f}s  左臂对箱 F={-contact.force_world[0]} N | 力矩={-contact.contact_torque_world[0]} N·m | boxz={box[2]:.4f} tiltx={np.degrees(rv[0]):+.2f}°")
        if step == 0:
            print(f"[t0] FL={row['FL']:.4f} FR={row['FR']:.4f} boxz={box[2]:.6f} tiltx={np.degrees(rv[0]):.4f} tipL={row['tipL']}")
        self.result.log.append(row)

    def dynamics(self, controller, state, dynamics, references, options, torque, mass_snapshot, t):
        if self.shadow is None:
            return
        started = time.perf_counter()
        old = replace(dynamics, mass=mass_snapshot['mass_old'])
        solution = self.solution
        a_old = controller.acceleration(state, old, references, torque, options,
            frozen_solution=solution).reshape(14)
        a_current = controller.acceleration(state, dynamics, references, torque, options,
            frozen_solution=solution).reshape(14)
        mujoco.mj_copyData(self.shadow, self.model, self.data)
        mujoco.mj_forward(self.model, self.shadow)
        dofs = np.concatenate([a['dofs'] for a in self.arms])
        actual = self.shadow.qacc[dofs].copy()
        current_shared = encode_shared_horizon(state, dynamics, references, solution.anchoring,
                                               solution.contact_load, self.config, options)[0]
        old_shared = encode_shared_horizon(state, old, references, solution.anchoring,
                                           solution.contact_load, self.config, options)[0]
        current_p, old_p = to_legacy(current_shared, self.variant), to_legacy(old_shared, self.variant)
        self.dynamics_rows.append(dict(t=float(t), phase=dynamics_phase(self.reference,t), x=state.vector(), u=torque.copy(),
            p_old=old_p, p_current=current_p, p_old_shared_full=old_shared,
            p_current_shared_full=current_shared, parameter_layout=layout_version(self.config),
            mass_old=old.mass.copy(), mass_current=dynamics.mass.copy(),
            acceleration_old=a_old, acceleration_current=a_current, acceleration_mujoco=actual,
            error_old=a_old-actual, error_current=a_current-actual,
            sampling_diagnostic_ms=mass_snapshot['sampling_diagnostic_ms'], diagnostic_ms=(time.perf_counter()-started)*1000.))

    def export(self):
        import json
        self.joints.failure = self.result.failure
        self.result.joint_limit_trace = self.joints.export(self.result.horizon)
        self.result.physics_trace = {k: np.asarray([r[k] for r in self.physics]) for k in self.physics[0]} if self.physics else {}
        self.result.motion_force_trace = build_motion_trace(
            self.model, self.arms, self.ids['box_body'], self.motion_rows, self.reference,
            self.variant, getattr(self.args, 'settle_time', 0.))
        if self.shadow is not None:
            self.result.dynamics_trace = {k: np.asarray([r[k] for r in self.dynamics_rows]) for k in self.dynamics_rows[0]} if self.dynamics_rows else {}
            self.result.dynamics_trace['metadata_json'] = np.array(json.dumps(dict(self.joints.metadata,
                diagnostic='same-state instantaneous acceleration; only M changes'), default=lambda v: np.asarray(v).tolist()))
