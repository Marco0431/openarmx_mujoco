"""验证实际点速度、同拍记录与独立图；只计算小模型几何，不运行控制器。"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import matplotlib.pyplot as plt

from experiments.data import RunResult
from experiments.motion_trace import build_motion_trace, save_motion_trace, save_motion_outputs
from experiments.reference import PhaseReference
from experiments.recording import Recorder
from ik import get_ee_tip
from plot_exp2_motion_force import read_motion_trace, plot_motion_trace


class MotionTraceTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string('''<mujoco>
          <option timestep=".002"/><worldbody>
            <body name="left"><joint axis="0 1 0"/><geom type="sphere" size=".01"/></body>
            <body name="right" pos="1 0 0"><joint axis="1 0 0"/><geom type="sphere" size=".01"/></body>
            <body name="box"><freejoint/><geom type="box" size=".1 .1 .1" pos=".03 .02 .01"/></body>
          </worldbody></mujoco>''')
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:2] = [.5, -.3]
        self.data.qvel[:] = [.6, -.8, .1, -.2, .3, .4, -.5, .6]
        self.arms = [dict(body_id=self.model.body(side).id) for side in ('left', 'right')]
        self.box = self.model.body('box').id
        zero = {side: np.zeros(3) for side in ('left', 'right')}
        self.reference = PhaseReference(5., .02, .1, .5, .5, zero, zero, zero, 1.)

    def rows(self, variant='flat', count=3):
        return [dict(step=5*i, t=i*.01, legacy_csv_t=i*.01+(.002 if variant == 'flat' else 0.),
                     qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(),
                     mocap_pos=self.data.mocap_pos.copy(), mocap_quat=self.data.mocap_quat.copy(),
                     normal_force=np.array([2.+i, 3.+i])) for i in range(count)]

    def trace(self, variant='flat', count=3):
        return build_motion_trace(self.model, self.arms, self.box, self.rows(variant, count),
                                  self.reference, variant)

    def test_tip_offset_and_box_com_match_position_finite_differences(self):
        before = self.data.qpos.copy(), self.data.qvel.copy(), self.data.xpos.copy()
        # 禁止任何完整 forward：离线速度只能使用位置阶段与实际 qvel。
        with patch('mujoco.mj_forward', side_effect=AssertionError('unexpected forward')):
            trace = self.trace()
        delta = 1e-7
        points = []
        for sign in (-1, 1):
            replay = mujoco.MjData(self.model)
            replay.qpos[:] = self.data.qpos
            mujoco.mj_integratePos(self.model, replay.qpos, self.data.qvel, sign*delta)
            mujoco.mj_kinematics(self.model, replay)
            points.append(np.array([get_ee_tip(replay, a['body_id']) for a in self.arms]
                                   + [replay.xipos[self.box].copy()]))
        derivative = (points[1]-points[0])/(2*delta)
        np.testing.assert_allclose(trace['tip_velocity_world'][0], derivative[:2], atol=1e-8)
        np.testing.assert_allclose(trace['box_velocity_world'][0], derivative[2], atol=1e-8)
        self.assertGreater(np.linalg.norm(derivative[0]), .05)  # link 原点固定，tip 仍在运动。
        self.assertFalse(np.allclose(derivative[2], self.data.qvel[2:5]))  # COM 有旋转偏置。
        for observed, original in zip((self.data.qpos, self.data.qvel, self.data.xpos), before):
            np.testing.assert_array_equal(observed, original)
        np.testing.assert_array_equal(trace['tip_velocity_reference_world'], 0.)
        self.assertGreater(np.linalg.norm(trace['tip_velocity_world']), 0.)
        np.testing.assert_allclose(trace['box_speed'], np.linalg.norm(derivative[2]))

    def test_both_variants_keep_step_and_legacy_time_alignment(self):
        for variant in ('flat', 'handheld'):
            trace = self.trace(variant)
            np.testing.assert_array_equal(trace['step'], [0, 5, 10])
            np.testing.assert_allclose(trace['legacy_csv_t']-trace['t'], .002 if variant == 'flat' else 0.)
            np.testing.assert_array_equal(trace['normal_force'], [[2, 3], [3, 4], [4, 5]])
            self.assertTrue(np.all(trace['normal_force_reference'] >= 0))
            metadata = json.loads(str(trace['metadata_json'].item()))
            self.assertEqual(metadata['arm_body_names'], ['left', 'right'])
            self.assertEqual(metadata['box_velocity_point'], 'body center of mass')

    def test_recorder_pairs_pre_step_velocity_with_same_step_contact_force(self):
        mujoco.mj_kinematics(self.model, self.data)
        for variant in ('flat', 'handheld'):
            recorder = Recorder.__new__(Recorder)
            recorder.model, recorder.data, recorder.arms = self.model, self.data, self.arms
            recorder.ids = dict(box_body=self.box, box_geom=self.model.geom(2).id)
            recorder.variant, recorder.physics, recorder.motion_rows = variant, [], []
            recorder.joints = SimpleNamespace(
                before_step=lambda d: ((d.time, None, None), 0.),
                after_step=lambda *args: None)
            qpos, qvel = self.data.qpos.copy(), self.data.qvel.copy()
            recorder.before_step()
            # 模拟 Euler 返回后的状态：q/v 已更新，而缓存几何仍是步前。
            self.data.qpos[0] += .2
            self.data.qvel[:] += 7.
            contact = SimpleNamespace(normal_magnitude=np.array([4., 6.]))
            with patch('experiments.recording.measure_contact', return_value=contact) as measured:
                recorder.after_step(np.zeros(14), 5)
            self.assertEqual(measured.call_count, 1)
            row = recorder.motion_rows[0]
            np.testing.assert_array_equal(row['qpos'], qpos)
            np.testing.assert_array_equal(row['qvel'], qvel)
            np.testing.assert_array_equal(row['normal_force'], [4., 6.])
            contact.normal_magnitude[:] = 99.
            np.testing.assert_array_equal(row['normal_force'], [4., 6.])
            self.assertEqual(row['legacy_csv_t'], .012 if variant == 'flat' else .01)
            self.data.qpos[:], self.data.qvel[:] = qpos, qvel

    def test_repeated_run_keeps_absolute_time_but_uses_local_reference_and_plot(self):
        rows = self.rows()
        for row in rows:
            row['t'] += 12.
        trace = build_motion_trace(self.model, self.arms, self.box, rows, self.reference, 'flat')
        expected = self.trace()
        np.testing.assert_allclose(trace['t'], expected['t']+12.)
        np.testing.assert_array_equal(trace['run_time'], expected['run_time'])
        for key in ('phase', 'tip_velocity_reference_world', 'normal_force_reference', 'legacy_csv_t'):
            np.testing.assert_array_equal(trace[key], expected[key])
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory)/'motion_force.npz'
            np.savez_compressed(path, **trace)
            with patch('plot_exp2_motion_force.draw_disturbance', return_value='test pulse') as disturbance:
                plot_motion_trace(path, Path(directory)/'figure', Path(directory)/'pulse.npz')
            # 外力日志也是本轮时间；两者必须共用从零开始的横轴。
            self.assertEqual(len(disturbance.call_args.args[0]), 5)
            self.assertEqual(disturbance.call_args.args[0][-1].get_ylabel(), 'Normal force magnitude [N]')
            for axis in disturbance.call_args.args[0]:
                np.testing.assert_array_equal(axis.lines[0].get_xdata(), trace['run_time'])
                self.assertEqual(axis.get_xlabel(), 'Run-relative time before step [s]')
            trace['run_time'][1] = trace['run_time'][0]
            np.savez_compressed(path, **trace)
            with self.assertRaisesRegex(ValueError, 'run_time'):
                read_motion_trace(path)

    def test_empty_and_partial_trace_save_with_explicit_path(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            for count in (0, 1):
                result = RunResult(motion_force_trace=self.trace(count=count))
                path = Path(directory)/f'partial_{count}.npz'
                save_motion_trace(result, path)
                self.assertEqual(result.motion_force_path, str(path.resolve()))
                with np.load(path, allow_pickle=False) as saved:
                    self.assertEqual(saved['tip_velocity_world'].shape, (count, 2, 3))
                if count:
                    self.assertEqual(read_motion_trace(path)[0]['t'].size, 1)
                else:
                    with self.assertRaisesRegex(ValueError, '为空'):
                        read_motion_trace(path)

    def test_plot_outputs_png_svg_and_uses_saved_handheld_sidecar(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            result = RunResult(motion_force_trace=self.trace('handheld'))
            save_motion_trace(result, root/'trace'/'motion_force.npz')
            with patch('experiments.motion_trace.subprocess.run') as call:
                save_motion_outputs(result, root/'plots', 'single', 'stamp', root/'entry.py')
            self.assertIn(result.motion_force_path, call.call_args.args[0])
            self.assertFalse((root/'plots'/'exp2_latest_single_stamp_motion_force.npz').exists())
            plot_motion_trace(Path(result.motion_force_path), root/'plots'/'figure')
            self.assertGreater((root/'plots'/'figure.png').stat().st_size, 1000)
            self.assertIn('<svg', (root/'plots'/'figure.svg').read_text(encoding='utf-8'))

    def test_xyz_panels_show_only_both_actual_arms_with_consistent_colors(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            for count in (1, 3):
                trace = self.trace(count=count)
                # 每个轴、侧、采样点均给不同有符号值，防止轴索引或参考替代未被检出。
                trace['tip_velocity_world'] = (np.arange(count*6).reshape(count, 2, 3)-4)*.01
                trace['tip_velocity_reference_world'][:] = 99.
                path = root/'motion_force.npz'
                np.savez_compressed(path, **trace)
                with patch('plot_exp2_motion_force.plt.close'):
                    plot_motion_trace(path, root/'figure')
                figure = plt.gcf()
                try:
                    self.assertEqual(len(figure.axes), 5)
                    for component, axis in enumerate(figure.axes[:3]):
                        self.assertEqual(len(axis.lines), 2)
                        self.assertIn(f'world {"xyz"[component]} velocity', axis.get_title())
                        for side, line in enumerate(axis.lines):
                            np.testing.assert_array_equal(line.get_ydata(), trace['tip_velocity_world'][:, side, component]*1000.)
                            self.assertEqual(line.get_label(), ('Left', 'Right')[side]+' actual')
                            self.assertEqual(line.get_color(), figure.axes[4].lines[2*side].get_color())
                            if count == 1:
                                self.assertEqual(line.get_marker(), 'o')
                    np.testing.assert_array_equal(figure.axes[3].lines[0].get_ydata(), trace['box_speed']*1000.)
                    self.assertEqual(figure.axes[4].get_subplotspec().colspan.start, 0)
                    self.assertEqual(figure.axes[4].get_subplotspec().colspan.stop, 2)
                finally:
                    plt.close(figure)

    def test_legacy_csv_missing_sidecar_and_negative_speed_are_rejected(self):
        with self.assertRaisesRegex(ValueError, '旧 CSV'):
            read_motion_trace(Path('legacy.csv'))
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory)/'missing.npz'
            with self.assertRaisesRegex(FileNotFoundError, '缺少'):
                read_motion_trace(path)
            trace = self.trace()
            trace['box_speed'] *= -1
            np.savez_compressed(path, **trace)
            with self.assertRaisesRegex(ValueError, '非负模长'):
                read_motion_trace(path)


if __name__ == '__main__':
    unittest.main()
