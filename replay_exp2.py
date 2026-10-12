"""在 MuJoCo 窗口回放已记录的状态；不创建控制器、不重新积分。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np


PROJECT = Path(__file__).resolve().parent


def one_file(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f'{pattern}: 需要唯一文件，实际找到 {len(matches)} 个')
    return matches[0]


def load_recording(directory: Path):
    path = one_file(directory, '*motion_force.npz')
    with np.load(path, allow_pickle=False) as archive:
        trace = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(trace['metadata_json'].item()))
    timeline = trace['run_time']
    if len(timeline) < 2 or not np.isfinite(timeline).all() or np.any(np.diff(timeline) <= 0):
        raise ValueError('回放时间必须有限且递增')
    if not np.isfinite(trace['qpos']).all() or not np.isfinite(trace['qvel']).all():
        raise ValueError('记录包含非有限状态，停止回放')
    return trace, metadata


def apply_frame(model, data, trace, index):
    data.qpos[:] = trace['qpos'][index]
    data.qvel[:] = trace['qvel'][index]
    data.mocap_pos[:] = trace['mocap_pos'][index]
    data.mocap_quat[:] = trace['mocap_quat'][index]
    data.time = float(trace['t'][index])
    # 只更新可视化需要的运动学缓存，不执行 mj_step。
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


def logged_positions(directory):
    trajectory = directory / 'trajectory.npz'
    if trajectory.exists():
        with np.load(trajectory, allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in ('tipL', 'tipR', 'box')}
    matches = [path for path in directory.glob('exp2_latest_single_*.csv')
               if not path.name.endswith('_horizon.csv')]
    if len(matches) != 1:
        raise ValueError('找不到唯一的实际轨迹 CSV')
    path = matches[0]
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([[float(row[key+axis]) for axis in 'xyz'] for row in rows])
            for key in ('tipL', 'tipR', 'box')}


def load_model(directory, trace, metadata):
    snapshot = directory / 'replay_model.mjb'
    if snapshot.exists():
        model = mujoco.MjModel.from_binary_path(str(snapshot))
        source = snapshot
    else:
        # 首次回放立即快照本次实验生成的 XML；之后回放不依赖可变运行时 XML。
        name = ('exp2_slender_scene.xml' if metadata['variant'] == 'flat'
                else 'grasp_pose_handheld_nmpcc_runtime.xml')
        source = PROJECT / name
        if not source.exists() and metadata['variant'] == 'handheld':
            source = PROJECT / 'grasp_pose_scene.xml'
        model = mujoco.MjModel.from_xml_path(str(source))
    if (model.nq, model.nv) != (trace['qpos'].shape[1], trace['qvel'].shape[1]):
        raise ValueError('模型维度与记录不一致')
    positions = logged_positions(directory)
    if any(len(value) != len(trace['run_time']) for value in positions.values()):
        raise ValueError('位置记录与状态记录长度不一致')
    data = mujoco.MjData(model)
    arms = [model.body(name).id for name in metadata['arm_body_names']]
    box = model.body(metadata['box_body_name']).id
    offset = np.asarray(metadata['tip_local_offset'])
    error = 0.
    for index in (0, len(trace['run_time']) // 2, len(trace['run_time']) - 1):
        apply_frame(model, data, trace, index)
        actual = [data.xpos[body] + data.xmat[body].reshape(3, 3) @ offset for body in arms]
        actual.append(data.xpos[box])
        error = max(error, *(float(np.max(np.abs(value-positions[key][index])))
                             for key, value in zip(('tipL', 'tipR', 'box'), actual)))
    if error > 1e-9:
        raise ValueError(f'回放几何与原日志不一致：{error:g} m')
    if not snapshot.exists():
        mujoco.mj_saveModel(model, str(snapshot), None)
    (directory/'replay_validation.json').write_text(json.dumps(dict(
        recording_variant=metadata['variant'], source=str(source),
        model_snapshot=str(snapshot), geometry_check_max_m=error,
        snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        purpose='kinematic recorded-state replay; contact forces are not recomputed for analysis',
        sample_period_s=metadata['sample_period_s'],
    ), ensure_ascii=False, indent=2), encoding='utf-8')
    return model, data


def play(model, data, trace, speed):
    paused, restart, stop = threading.Event(), threading.Event(), threading.Event()
    paused.set()

    def key_pressed(key):
        if key == 32:
            paused.clear() if paused.is_set() else paused.set()
        elif key == ord('R'):
            restart.set()
        elif key == 256:
            stop.set()

    print('[replay] SPACE: play/pause; R: restart; ESC/window close: exit. Starts paused.', flush=True)
    print(f'[replay] {speed:g}x recorded motion; no NMPC solving or physics stepping.', flush=True)
    apply_frame(model, data, trace, 0)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_pressed) as viewer:
        viewer.cam.lookat[:] = [0., -.18, .20]
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 1.8, -12, 180
        playhead, previous = 0., time.perf_counter()
        duration = float(trace['run_time'][-1])
        while viewer.is_running() and not stop.is_set():
            now = time.perf_counter()
            if restart.is_set():
                playhead = 0.
                restart.clear()
            elif not paused.is_set():
                playhead = (playhead + (now-previous)*speed) % duration
            previous = now
            index = int(np.searchsorted(trace['run_time'], playhead, side='right') - 1)
            apply_frame(model, data, trace, max(0, index))
            viewer.sync()
            time.sleep(1/60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--speed', type=float, default=.5)
    parser.add_argument('--validate_only', action='store_true')
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error('speed 必须是正的有限数')
    directory = args.output_dir.resolve()
    trace, metadata = load_recording(directory)
    model, data = load_model(directory, trace, metadata)
    if not args.validate_only:
        play(model, data, trace, args.speed)


if __name__ == '__main__':
    main()
