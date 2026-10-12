"""独立速度/夹持力记录：退出后按实测状态重放几何，不参与控制或动力学。"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np

from ik import EE_TIP_Z, get_ee_tip


def build_motion_trace(model, arms, box_body, rows, reference, variant, settle_time=0.):
    """配对步前 qpos/qvel 与该步 Euler 接触解，计算世界系点速度。"""
    count = len(rows)
    shapes = dict(qpos=(model.nq,), qvel=(model.nv,), mocap_pos=(model.nmocap, 3),
                  mocap_quat=(model.nmocap, 4), normal_force=(2,))
    trace = {key: np.asarray([row[key] for row in rows], dtype=float).reshape((count, *shape))
             for key, shape in shapes.items()}
    trace.update(step=np.asarray([r['step'] for r in rows], dtype=np.int64),
                 t=np.asarray([r['t'] for r in rows], dtype=float),
                 legacy_csv_t=np.asarray([r['legacy_csv_t'] for r in rows], dtype=float))
    # flat 的重复运行保留 data.time；参考与扰动按本轮步号从零计时。
    trace['run_time'] = trace['step'] * float(model.opt.timestep)
    tip_velocity = np.empty((count, 2, 3))
    box_velocity = np.empty((count, 3))
    shadow = mujoco.MjData(model)
    jacobian = np.empty((3, model.nv))
    for index in range(count):
        shadow.qpos[:] = trace['qpos'][index]
        shadow.qvel[:] = trace['qvel'][index]
        shadow.mocap_pos[:] = trace['mocap_pos'][index]
        shadow.mocap_quat[:] = trace['mocap_quat'][index]
        # Jacobian 只需要这两个位置阶段；不运行 forward、接触或动力学求解。
        mujoco.mj_kinematics(model, shadow)
        mujoco.mj_comPos(model, shadow)
        for side, arm in enumerate(arms):
            mujoco.mj_jac(model, shadow, jacobian, None,
                          get_ee_tip(shadow, arm['body_id']), arm['body_id'])
            tip_velocity[index, side] = jacobian @ shadow.qvel
        mujoco.mj_jacBodyCom(model, shadow, jacobian, None, box_body)
        box_velocity[index] = jacobian @ shadow.qvel
    # PhaseReference.horizon 是纯函数，批量计算同一实际时刻的目标，不复用预测速度。
    desired = reference.horizon(0., trace['run_time'])
    phases = ['settle' if r['step'] < int(round(settle_time/.002)) else reference.phase(t)
              for r, t in zip(rows, trace['run_time'])]
    trace.update(tip_velocity_world=tip_velocity, box_velocity_world=box_velocity,
                 box_speed=np.linalg.norm(box_velocity, axis=1),
                 tip_velocity_reference_world=desired.velocities[:, :, :3].copy(),
                 normal_force_reference=np.abs(desired.forces[:, :, 0]),
                 phase=np.asarray(phases, dtype='<U8'))
    metadata = dict(schema='motion_force_v1', variant=variant, world_axes='xyz',
        velocity_units='m/s', force_units='N', physics_dt=float(model.opt.timestep),
        sample_every_steps=5, sample_period_s=5*float(model.opt.timestep),
        state_timing='pre-step qpos/qvel and mocap; contact force from that Euler step',
        t_meaning='pre-step physical data.time; step is zero-based loop index',
        run_time_meaning='step * physics_dt; run-relative pre-step time used for reference, phase, plot and disturbance',
        legacy_csv_t_meaning='same sampled step: flat step*0.002+0.002; handheld step*0.002',
        velocity_method='MuJoCo mj_kinematics + mj_comPos + point Jacobian @ measured qvel, offline',
        tip_local_offset=[0., 0., float(EE_TIP_Z)],
        arm_order=['left', 'right'], arm_body_names=[model.body(a['body_id']).name for a in arms],
        box_body_name=model.body(box_body).name, box_velocity_point='body center of mass',
        force_definition='sum of absolute contact-normal forces at each pad, same as legacy FL/FR',
        reference_definition='instantaneous PhaseReference at run_time, not held prediction or measured velocity',
        samples_are_unsmoothed=True)
    trace['metadata_json'] = np.array(json.dumps(metadata, ensure_ascii=False))
    return trace


def save_motion_trace(result, path):
    """公开实际保存路径，手持归档与自动绘图通过此字段衔接。"""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result.motion_force_trace)
    result.motion_force_path = str(path)
    print(f'[motion/force] saved {path}')


def save_motion_outputs(result, output_dir, tag, stamp, entry_file):
    """独立保存与绘图；--no_csv 只关闭旧 CSV/旧图，不关闭此记录。"""
    if not result.motion_force_trace:
        return
    output = Path(output_dir)
    if result.motion_force_path is None:
        save_motion_trace(result, output / f'exp2_latest_{tag}_{stamp}_motion_force.npz')
    if not result.motion_force_trace['t'].size:
        return
    command = [sys.executable, str(Path(entry_file).resolve().parent / 'plot_exp2_motion_force.py'),
               result.motion_force_path, '--out', str(output / f'motion_force_{tag}_{stamp}')]
    if result.disturbance_log:
        command += ['--disturbance_log', result.disturbance_log]
    try:
        subprocess.run(command, check=True)
    except Exception as exc:
        print(f'[warn] automatic motion/force generation failed: {exc}')
