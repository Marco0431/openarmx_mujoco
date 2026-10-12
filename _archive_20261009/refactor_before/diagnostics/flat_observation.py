"""本场景的运行观测器：集中记录预测节点、物理步状态和异常现场。

本模块只观测与归档，不决定控制命令；采样时刻由主程序在物理步前后明确调用。"""
import time
import numpy as np
import mujoco
from ik import get_ee_tip
from .measurements import (JointLimitRecorder, DynamicsRecorder, contact_forces,
    pad_box_wrench_3d, box_attitude, box_net_wrench)
import nmpcc_cart as nc

class Observation:
    """把一次实验的接触力、轨迹、限位及可选动力学对账整理到同一份记录。
    记录函数接收主程序已计算的数据，尽量复用预测轨迹，结束时由导出模块写盘。"""
    def __init__(self, model, data, arms, ids, args, nmpc, ref_fn, dt, control_every,
                 smoothstep, mass_snapshot, contact_snapshot, contact_load_mode):
        self.model, self.data, self.arms, self.args = model, data, arms, args
        self.nmpc, self.ref_fn = nmpc, ref_fn
        self.box_body, self.box_geom, self.box_qpos = ids['box_body'], ids['box_geom'], ids['box_qpos']
        self.dt, self.control_every, self.smoothstep = dt, control_every, smoothstep
        self.mass_snapshot, self.contact_snapshot = mass_snapshot, contact_snapshot
        self.contact_load_mode = contact_load_mode
        self.log = []
        self.horizon_log = []
        self.recorder = JointLimitRecorder(self.model, self.data, self.arms, {
            "args": vars(self.args).copy(), "config": vars(self.nmpc.cfg).copy(),
            "model_version": nc.MODEL_VERSION, "solver_model": self.nmpc._ocp.model.name,
            "joint_bounds_enabled": getattr(self.nmpc, "qmin", None) is not None,
            "physics_dt": self.dt, "control_every": self.control_every,
            "contact_load_mode": self.contact_load_mode,
            "contact_load_revision": "frozen_total_minus_current_model_normal_v1" if self.contact_snapshot is not None else "tangent_v1",
        })
        self.nmpc.joint_limit_recorder = self.recorder
        self.nmpc.joint_limit_horizon = self.horizon_log
        self.nmpc.first_failure = None
        self.dynamics_recorder = (DynamicsRecorder(self.model, self.arms, self.recorder.metadata)
                             if self.mass_snapshot is not None else None)

        self.force_nominal = np.full(2, np.nan)
        self.force_measured_control = np.full(2, np.nan)
        self.force_bias = np.full(2, np.nan)
        self.force_anchored = np.full(2, np.nan)
        self.force_ref = np.array([-self.args.f_start, self.args.f_start], dtype=float)
        self.pred_force = np.full((3, 2), np.nan)
        self.pred_force_error = np.full((3, 2), np.nan)
        self.pred_position_error_x = np.full((3, 2), np.nan)
        self.pred_eps_x = np.full((3, 2), np.nan)
        self.pred_epsd_x = np.full((3, 2), np.nan)
        self.pred_u = np.full((3, nc.NU), np.nan)

    def capture_recovered_failure(self, t):
        """保留重建前第一次失败的信息。"""
        if self.nmpc.first_failure is not None and not self.recorder.failure:
            self.recorder.failure = {"t": t, "message": "first failed solve, before rebuild",
                                **self.nmpc.first_failure}


    def capture_failure(self, exc, t, x0, p, measured, seed):
        """保存异常发生时的状态、参数和预测轨迹。"""
        self.recorder.failure = dict(t=t, x0=x0, p=p, f_meas=measured,
                                u_seed=seed, qpos=self.data.qpos.copy(),
                                qvel=self.data.qvel.copy(), message=str(exc))
        self.recorder.failure["pred_x"] = np.array([
            self.nmpc.solver.get(i, "x") for i in range(self.args.N + 1)])
        if self.nmpc.first_failure is not None:
            self.recorder.failure.update({"first_" + k: v for k, v in self.nmpc.first_failure.items()})
        self.nmpc.last_joint_limit_trace = self.recorder.export(self.horizon_log)
        if self.dynamics_recorder is not None:
            self.nmpc.last_dynamics_trace = self.dynamics_recorder.export()


    def record_prediction(self, t, phase, p, fmag, u_ref_now, prepare_ms, reference_ms):
        """记录已求出的全部节点及首、中、末节点摘要。"""
        self.force_nominal = self.nmpc.last_force_nominal.copy()
        self.force_measured_control = self.nmpc.last_force_measured.copy()
        self.force_bias = self.nmpc.last_force_bias.copy()
        self.force_anchored = self.nmpc.last_force_anchored.copy()
        self.force_ref = np.array([-fmag, fmag], dtype=float)
        state_nodes = [0, self.args.N // 2, self.args.N]
        input_nodes = [0, max(0, self.args.N // 2), self.args.N - 1]
        self.pred_force = self.nmpc.last_pred_force[state_nodes].copy()
        self.pred_force_error = self.nmpc.last_pred_force_error[state_nodes].copy()
        self.pred_position_error_x = self.nmpc.last_pred_position_error_x[state_nodes].copy()
        self.pred_eps_x = self.nmpc.last_pred_eps_x[state_nodes].copy()
        self.pred_epsd_x = self.nmpc.last_pred_epsd_x[state_nodes].copy()
        self.pred_u = self.nmpc.last_pred_u[input_nodes].copy()
        self.horizon_log.append({
            "t": t,
            "phase": phase,
            "force_ref": p[:, [150, 156]].copy(),
            "reference": p[:, 126:162].copy(),
            "force_ref_dot": p[:, nc.FREF_DOT_IDX:].copy(),
            "node_times": self.nmpc.node_times.copy(),
            "prepare_ms": prepare_ms,
            "reference_ms": reference_ms,
            "solver_ms": self.nmpc.last_solver_ms,
            "force": self.nmpc.last_pred_force.copy(),
            "force_error": self.nmpc.last_pred_force_error.copy(),
            "position_error_x": self.nmpc.last_pred_position_error_x.copy(),
            "eps_x": self.nmpc.last_pred_eps_x.copy(),
            "epsd_x": self.nmpc.last_pred_epsd_x.copy(),
            "u": self.nmpc.last_pred_u.copy(),
            "u_ref": u_ref_now.copy(),
            "solver_status": self.nmpc.last_status,
        })
        if self.contact_snapshot is not None:
            self.horizon_log[-1]["contact_load"] = {
                key: value.copy() for key, value in self.nmpc.last_contact_load_snapshot.items()}
        td = time.perf_counter()
        q_pred = self.nmpc.last_pred_x[:, np.r_[0:7, 14:21]].copy()
        self.horizon_log[-1].update(
            joint_q=q_pred,
            joint_v=self.nmpc.last_pred_x[:, np.r_[7:14, 21:28]].copy(),
            joint_violation=np.maximum(0.0, np.maximum(self.recorder.lo - q_pred, q_pred - self.recorder.hi)))
        self.horizon_log[-1]["joint_diagnostic_ms"] = (time.perf_counter() - td) * 1000.0

    def record_dynamics(self, step, t):
        """在实际命令写入后执行可选动力学对账。"""
        if self.dynamics_recorder is not None and step % self.control_every == 0:
            self.dynamics_recorder.record(self.data, self.nmpc, self.mass_snapshot, self.ref_fn, t)

    def before_step(self):
        """保存与本次物理步反力对齐的积分前状态。"""
        self.captured = self.recorder.before_step(self.data)

    def after_step(self, u_cmd):
        """读取物理步结果，记录关节限位和箱体状态。"""
        self.recorder.after_step(self.model, self.data, self.captured)

    def record_sample(self, step, t, phase, solve_ms, prepare_ms, reference_ms, u_cmd, u_clip_mask, u_sat_mask, settling=False):
        """按原频率记录用于CSV和评价的观测量。"""
        if step % 5 == 0:
            forces = contact_forces(self.model, self.data, self.arms, self.box_geom)
            w = pad_box_wrench_3d(self.model, self.data, self.arms, self.box_geom)
            rv, quat = box_attitude(self.data, self.box_qpos)

            print(f"t={t + self.dt:.3f}s  左臂对箱 F={-w['left'][0]} N | 力矩={-w['left'][1]} N·m"
                  f" | boxz={self.data.xpos[self.box_body][2]:.4f} tiltx={np.degrees(rv[0]):+.2f}°")

            fb, tb = box_net_wrench(self.model, self.data, self.arms, self.box_geom,
                                    self.data.xpos[self.box_body].copy())
            if step == 0:
                print(f"[t0] FL={forces['left']:.4f} FR={forces['right']:.4f} "
                      f"boxz={self.data.xpos[self.box_body][2]:.6f} "
                      f"tiltx={np.degrees(rv[0]):.4f} "
                      f"tipL={get_ee_tip(self.data, self.arms[0]['body_id'])}")
            self.log.append({
                "t": t + self.dt,
                "phase": phase,
                "ramp_s": self.smoothstep(t / self.args.t_ramp) if self.args.t_ramp > 1e-9 else 1.0,
                "moving_active": 1.0 if phase == "move" else 0.0,
                "box": self.data.xpos[self.box_body].copy(),
                "quat": quat.copy(),
                "rotvec": rv.copy(),
                "tipL": get_ee_tip(self.data, self.arms[0]["body_id"]).copy(),
                "tipR": get_ee_tip(self.data, self.arms[1]["body_id"]).copy(),
                "wL": w["left"], "wR": w["right"],
                "box_f": fb, "box_t": tb,
                "u_cmd": u_cmd.copy(),
                "u_clip_mask": u_clip_mask.copy(),
                "u_sat_mask": u_sat_mask.copy(),
                "FL": forces["left"], "FR": forces["right"],
                "force_nominal": self.force_nominal.copy(),
                "force_measured_control": self.force_measured_control.copy(),
                "force_bias": self.force_bias.copy(),
                "force_anchored": self.force_anchored.copy(),
                "force_ref": self.force_ref.copy(),
                "pred_force": self.pred_force.copy(),
                "pred_force_error": self.pred_force_error.copy(),
                "pred_position_error_x": self.pred_position_error_x.copy(),
                "pred_eps_x": self.pred_eps_x.copy(),
                "pred_epsd_x": self.pred_epsd_x.copy(),
                "pred_u": self.pred_u.copy(),
                "solve_ms": solve_ms if step % self.control_every == 0 else np.nan,
                "prepare_ms": prepare_ms if step % self.control_every == 0 else np.nan,
                "reference_ms": reference_ms if step % self.control_every == 0 else np.nan,
                "solver_ms": self.nmpc.last_solver_ms if step % self.control_every == 0 else np.nan,
                "solver_status": (float(self.nmpc.last_status)
                                   if step % self.control_every == 0 else np.nan),
                "qp_iter": (float(self.nmpc.last_qp_iter)
                             if step % self.control_every == 0 else np.nan),
                "solver_qp_ms": (self.nmpc.last_time_qp * 1000.0
                                  if step % self.control_every == 0 else np.nan),
                "solver_sim_ms": (self.nmpc.last_time_sim * 1000.0
                                   if step % self.control_every == 0 else np.nan),
                "solver_lin_ms": (self.nmpc.last_time_lin * 1000.0
                                   if step % self.control_every == 0 else np.nan),
            })

    def finish(self, initial_box_y, move_duration, summarize):
        """统一导出内存记录并计算指标。"""
        metrics = summarize(self.log, self.args, initial_box_y, move_duration=move_duration)
        self.nmpc.last_joint_limit_trace = self.recorder.export(self.horizon_log)
        if self.dynamics_recorder is not None:
            self.nmpc.last_dynamics_trace = self.dynamics_recorder.export()
        for key in ("prepare_ms", "reference_ms", "solver_ms", "control_ms"):
            values = np.array([h[key] for h in self.horizon_log[10:]], dtype=float)
            if values.size:
                metrics[key + "_p50"] = float(np.percentile(values, 50))
                metrics[key + "_p95"] = float(np.percentile(values, 95))
        return self.log, self.horizon_log, metrics


