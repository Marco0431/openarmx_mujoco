"""用真实 MuJoCo 数据结构和假求解器验证运行层，不编译 DLL 或运行闭环。"""
from contextlib import redirect_stdout
import io
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from experiments import cli, runner
from experiments.data import RunPaths, RunResult
from experiments.legacy import flat_output, handheld_output
from experiments.reference import PhaseReference, motion_timing
from experiments.scenes import flat_cli, handheld_cli
from nmpcc.config import ControllerConfig, NormalForceLaw
from nmpcc.data import Prediction, SolveResult
from nmpcc.parameters import FULL_SIZE, parameter_size, to_legacy, to_legacy_full


def small_scene(variant):
    """构造独立的双七关节模型，不读取或覆盖实验 XML。"""
    bodies, motors = [], []
    for side, x in (("left", -.2), ("right", .2)):
        parts = [f'<body name="{side}_0" pos="{x} 0 0">']
        for joint in range(7):
            if joint:
                parts.append(f'<body name="{side}_{joint}" pos="0 0 .05">')
            name = f"{side}_j{joint}"
            parts.append(f'<joint name="{name}" axis="0 1 0" range="-1 1"/>')
            parts.append('<geom type="sphere" size=".01" mass=".1" contype="0" conaffinity="0"/>')
            motors.append(f'<motor name="{name}_motor" joint="{name}" ctrlrange="-10 10"/>')
        parts.append(f'<geom name="{side}_pad" type="box" size=".01 .01 .01"/>')
        parts.append('</body>' * 7)
        bodies.append(''.join(parts))
    xml = '<mujoco><compiler angle="radian"/><option timestep=".002"/><worldbody>'
    xml += ''.join(bodies)
    xml += '<body name="box"><freejoint name="box_joint"/><geom name="box" type="box" size=".01 .01 .01"/></body>'
    xml += '</worldbody><actuator>' + ''.join(motors) + '</actuator></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    arms = []
    for side in ("left", "right"):
        joints = np.array([model.joint(f"{side}_j{i}").id for i in range(7)])
        arms.append(dict(side=side, joint_ids=joints, qpos=model.jnt_qposadr[joints],
            dofs=model.jnt_dofadr[joints], body_id=model.body(f"{side}_6").id,
            pad_id=model.geom(f"{side}_pad").id,
            act_ids=[model.actuator(f"{side}_j{i}_motor").id for i in range(7)]))
    ids = dict(box_body=model.body("box").id, box_geom=model.geom("box").id,
        box_qpos=int(model.jnt_qposadr[model.joint("box_joint").id]))
    mujoco.mj_forward(model, data)
    return SimpleNamespace(model=model, data=data, arms=arms, ids=ids, variant=variant,
        box_center=np.zeros(3), motion_timing=motion_timing,
        output=flat_output if variant == "flat" else handheld_output)


class FakeController:
    def __init__(self, scene, mode="ok", law=NormalForceLaw.SMOOTH_POSITIVE):
        self.config = ControllerConfig(N=2, normal_force_law=law)
        self.node_times = np.linspace(0., .1, 3)
        self.scene, self.mode, self.calls = scene, mode, []

    def reset(self):
        if self.mode == "reset":
            raise RuntimeError("injected reset failure")

    def equilibrium_seed(self, *args):
        return np.full(14, 1e6) if self.mode == "seed" else np.zeros(14)

    def solve(self, state, dynamics, references, options):
        self.calls.append((self.scene.data.time, len(references.node_times), options.u_seed.copy()))
        zeros = np.zeros((3, 2, 6))
        prediction = Prediction(np.tile(state.vector(), (3, 1)), np.zeros((2, 14)),
            references, zeros, zeros, np.zeros((3, 2)), np.zeros((3, 2)),
            np.zeros((3, 2)), np.zeros((3, 2)), zeros, zeros)
        statistics = dict(solver_ms=0., qp_iter=1, time_qp=0., time_sim=0., time_lin=0.)
        anchoring = {key: np.zeros((2, 6)) for key in ("nominal", "measured", "bias", "anchored")}
        if self.config.normal_force_law == NormalForceLaw.MEASURED_ANCHORED_IDEAL:
            anchoring.update(anchor_gap=np.array([.24]), anchor_normal=np.array([10., 11.]))
            prediction.force_rates = np.zeros((3, 2, 6))
        status = 4 if self.mode in ("warmup", "retry_failure") else 0
        first = None
        if self.mode in ("recovered", "retry_failure", "rebuild_exception"):
            first = dict(status=4, x0=state.vector(),
                p_full=np.tile(np.arange(parameter_size(self.config), dtype=float), (3, 1)),
                x=np.ones((3, 28)), u=np.ones((2, 14)))
            prediction.states[:] = 2.
        if self.mode == "rebuild_exception":
            exc = RuntimeError("injected rebuild constructor failure")
            exc.failure = dict(status=4, x0=state.vector(), p_full=first['p_full'] + 1000.,
                pred_x=np.full((3, 28), 2.), pred_u=np.full((2, 14), 2.),
                **{'first_' + key: value for key, value in first.items()})
            raise exc
        return SolveResult(np.zeros(14), prediction, status, status, statistics, first,
            anchoring, {"frozen_residual": np.zeros((2, 7))})


class RunnerContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="runner_contract_")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.stop_file = self.output / "stop"
        self.stop_file.touch()
        self.paths = RunPaths(self.output, self.output / "cache", self.output, self.output / "entry.py")

    def run_mock(self, variant, mode="ok", law=NormalForceLaw.SMOOTH_POSITIVE):
        scene = small_scene(variant)
        parser = (flat_cli if variant == "flat" else handheld_cli).make_parser()
        args = parser.parse_args(["--no_viewer", "--no_csv", "--hold_only", "--N", "2", "--steps", "1"])
        args.disturb_stop_file = str(self.stop_file)
        if variant == "handheld":
            args.settle_time = 1. if mode == "settle" else 0.
        positions = {side: np.zeros(3) for side in ("left", "right")}
        reference = PhaseReference(10., 0., 0., .5, .1, positions, positions, positions, args.f_start)
        result, controller, steps = RunResult(), FakeController(scene, mode, law), []

        def step(model, data):
            steps.append(data.time)
            data.time += .002

        error = None
        with redirect_stdout(io.StringIO()), patch.object(runner.mujoco, "mj_step", side_effect=step):
            try:
                runner.run_case(scene, args, controller, reference, self.paths, result)
            except (RuntimeError, ValueError) as exc:
                error = exc
        return scene, args, reference, result, controller, steps, error

    def test_two_warmups_precede_first_step_and_outputs_keep_shapes(self):
        for variant in ("flat", "handheld"):
            with self.subTest(variant=variant):
                _, _, _, result, controller, steps, error = self.run_mock(variant)
                self.assertIsNone(error)
                self.assertEqual(steps, [0.])
                self.assertEqual([(t, n) for t, n, _ in controller.calls], [(0., 3)] * 3)
                self.assertEqual(len(result.startup), 2)
                self.assertEqual(result.horizon[0]['parameters_shared_full'].shape, (3, FULL_SIZE))
                self.assertEqual(result.startup[0]['parameters_shared_full'].shape, (3, FULL_SIZE))
                self.assertEqual(len(result.physics_trace['t']), 1)
                self.assertEqual(result.horizon[0]["force_ref"].shape, (3, 2))
                self.assertEqual(result.horizon[0]["reference"].shape, (3, 36))
                self.assertEqual(result.horizon[0]["force_ref_dot"].shape, (3, 12))
                self.assertEqual(result.log[0]["pred_eps_x"].shape, (3, 12))
                self.assertEqual(result.log[0]["force_nominal"].shape, (2,) if variant == "flat" else (12,))

    def test_startup_failures_export_without_advancing_physics(self):
        for variant in ("flat", "handheld"):
            for mode in ("reset", "seed", "warmup"):
                with self.subTest(variant=variant, mode=mode):
                    _, _, _, result, _, steps, error = self.run_mock(variant, mode)
                    self.assertIsNotNone(error)
                    self.assertEqual(steps, [])
                    self.assertTrue(result.failure)
                    self.assertIn("failure_message", result.joint_limit_trace)
                    self.assertEqual(result.startup, [])

    def test_measured_anchor_raw_parameters_and_derivatives_are_recorded_both_scenes(self):
        for variant in ('flat', 'handheld'):
            with self.subTest(variant=variant):
                _, _, _, result, _, _, error = self.run_mock(
                    variant, law=NormalForceLaw.MEASURED_ANCHORED_IDEAL)
                self.assertIsNone(error)
                row = result.horizon[0]
                self.assertEqual(row['parameters_shared_full'].shape, (3, 228))
                np.testing.assert_array_equal(row['parameters_shared_full'][:, 225:],
                                              np.tile([.24, 10., 11.], (3, 1)))
                np.testing.assert_array_equal(row['force_anchor_normal'], [10., 11.])
                self.assertEqual(row['force_pred_dot'].shape, (3, 2, 6))
                self.assertEqual(result.startup[0]['parameters_shared_full'].shape, (3, 228))

    def test_measured_anchor_failure_preserves_raw_228_and_old_compatibility_view(self):
        for variant, old_width in (('flat', 215), ('handheld', 225)):
            with self.subTest(variant=variant):
                _, _, _, result, _, steps, error = self.run_mock(
                    variant, 'rebuild_exception', law=NormalForceLaw.MEASURED_ANCHORED_IDEAL)
                self.assertIsNotNone(error)
                self.assertEqual(steps, [])
                trace = result.joint_limit_trace
                self.assertEqual(trace['failure_p_full'].shape, (3, old_width))
                raw = np.tile(np.arange(228), (3, 1))
                np.testing.assert_array_equal(trace['failure_p_shared_full'], raw+1000.)
                np.testing.assert_array_equal(trace['failure_first_p_shared_full'], raw)

    def test_recovered_failure_keeps_legacy_full_parameter_width(self):
        for variant, width in (("flat", 215), ("handheld", 225)):
            with self.subTest(variant=variant):
                _, _, _, result, _, steps, error = self.run_mock(variant, "recovered")
                self.assertIsNone(error)
                self.assertEqual(len(steps), 1)
                trace = result.joint_limit_trace
                self.assertEqual(trace['failure_p_full'].shape, (3, width))
                np.testing.assert_array_equal(trace['failure_p_full'],
                    to_legacy_full(np.tile(np.arange(FULL_SIZE), (3, 1)), variant))
                np.testing.assert_array_equal(trace['failure_x'], 1.)

    def test_second_failure_keeps_terminal_and_first_iterates_separate(self):
        for variant, width, full_width in (("flat", 191, 215), ("handheld", 201, 225)):
            with self.subTest(variant=variant):
                _, _, _, result, _, steps, error = self.run_mock(variant, "retry_failure")
                self.assertIsNotNone(error)
                self.assertEqual(steps, [])
                trace = result.joint_limit_trace
                self.assertEqual(trace['failure_p'].shape, (3, width))
                self.assertEqual(trace['failure_first_p_full'].shape, (3, full_width))
                self.assertEqual(trace['failure_f_meas'].shape, (2,) if variant == 'flat' else (12,))
                self.assertEqual(trace['failure_u_seed'].shape, (14,))
                self.assertEqual(str(trace['failure_message']), 'NMPCC warmup status 4')
                np.testing.assert_array_equal(trace['failure_first_x'], 1.)
                np.testing.assert_array_equal(trace['failure_pred_x'], 2.)

    def test_rebuild_exception_context_preserves_both_parameter_snapshots(self):
        for variant, width, full_width in (("flat", 191, 215), ("handheld", 201, 225)):
            with self.subTest(variant=variant):
                _, _, _, result, _, steps, error = self.run_mock(variant, "rebuild_exception")
                self.assertIsNotNone(error)
                self.assertEqual(steps, [])
                trace = result.joint_limit_trace
                self.assertEqual(trace['failure_p'].shape, (3, width))
                self.assertEqual(trace['failure_p_full'].shape, (3, full_width))
                self.assertEqual(trace['failure_first_p_full'].shape, (3, full_width))
                raw = np.tile(np.arange(FULL_SIZE), (3, 1))
                np.testing.assert_array_equal(trace['failure_p'], to_legacy(raw + 1000., variant))
                np.testing.assert_array_equal(trace['failure_first_p_full'], to_legacy_full(raw, variant))
                np.testing.assert_array_equal(trace['failure_first_x'], 1.)
                np.testing.assert_array_equal(trace['failure_pred_x'], 2.)
                self.assertEqual(str(trace['failure_message']), 'injected rebuild constructor failure')

    def test_settling_without_prediction_can_be_archived(self):
        scene, args, reference, result, controller, steps, error = self.run_mock("handheld", "settle")
        self.assertIsNone(error)
        self.assertEqual(len(controller.calls), 2)
        self.assertEqual(len(steps), 1)
        self.assertFalse(result.horizon)
        self.assertTrue(np.isnan(result.log[0]["pred_force"]).all())
        with redirect_stdout(io.StringIO()), patch.object(handheld_output, "archive_sources", return_value={}):
            handheld_output.save_run(self.output, args, scene.model, scene.data, scene.arms, scene.ids,
                result, reference, result.log, result.horizon, result.metrics, None,
                entry_file=self.paths.entry_file, base_dir=self.output, box_center=np.zeros(3),
                kx=2419.3, rest_gap=.2, scene_path=self.output / "scene.xml")
        for name in ("exit_state.npz", "summary.json", "startup.npz", "physics.npz", "joint_limits.npz"):
            self.assertTrue((self.output / name).is_file(), name)

    def test_source_archive_keeps_same_named_package_modules(self):
        source = self.output / "project"
        source.mkdir()
        for name in ("entry.py", "ik.py", "grasp_pose_static.py", "nmpcc/data.py", "experiments/data.py"):
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(repr(name), encoding="utf-8")
        (source / "scene.xml").write_text("<mujoco/>", encoding="utf-8")
        hashes = handheld_output.archive_sources(self.output, source, source / "entry.py", source / "scene.xml")
        self.assertEqual(len(hashes), 6)
        for name in ("nmpcc/data.py", "experiments/data.py"):
            self.assertEqual((self.output / "source" / name).read_bytes(), (source / name).read_bytes())

    def test_saved_metadata_uses_effective_runtime_weights(self):
        args = flat_cli.make_parser().parse_args(["--no_viewer", "--no_csv"])
        args.Q_rt, args.P_rt, args.ed_rt, args.eu_rt, args.ed_x_rt = 2., 3., .4, .5, .7
        controller = SimpleNamespace(config=ControllerConfig(), metadata={"config": {}, "name": "fake"})
        output = SimpleNamespace(finish_run=lambda *args: args[2])
        scene = SimpleNamespace(variant="flat", output=output)
        captured = []
        with patch.object(cli, "run_case", side_effect=lambda *args: captured.append(args[5])):
            cli.execute_run(scene, args, controller, None, self.paths, "test", False)
        metadata = captured[0].metadata
        np.testing.assert_allclose(metadata["stage_weights_diagonal"][[0, 12, 13, 24]], [2., 1.4, .8, 1.])
        np.testing.assert_allclose(metadata["terminal_weights_diagonal"][[0, 12, 13]], [3., 2.1, 1.2])

    def test_portable_archive_finds_shared_packages_above_scene(self):
        scene = self.output / "portable" / "handheld"
        scene.mkdir(parents=True)
        for name in ("entry.py", "ik.py", "grasp_pose_static.py"):
            (scene / name).write_text(repr(name), encoding="utf-8")
        (scene / "scene.xml").write_text("<mujoco/>", encoding="utf-8")
        for package in ("nmpcc", "experiments", "diagnostics"):
            module = scene.parent / package / "data.py"
            module.parent.mkdir()
            module.write_text(repr(package), encoding="utf-8")
        handheld_output.archive_sources(self.output, scene, scene / "entry.py", scene / "scene.xml")
        for package in ("nmpcc", "experiments", "diagnostics"):
            self.assertEqual((self.output / "source" / package / "data.py").read_text(), repr(package))

    def test_handheld_colocated_output_is_an_explicit_entrypoint_option(self):
        output = self.output / 'handheld_result'
        config = ControllerConfig()
        scene = SimpleNamespace(fk={}, geometry=SimpleNamespace(K_STIFF=2419.3))
        for colocate in (False, True):
            with self.subTest(co_locate_outputs=colocate):
                captured = []
                def execute(scene, args, controller, reference, paths, tag, sweep):
                    captured.append(paths)
                    self.assertFalse(output.exists(), 'trace directory must be created by execute_run')
                    return {}
                with (redirect_stdout(io.StringIO()),
                      patch.object(sys, 'argv', ['entry', '--no_viewer', '--output_dir', str(output)]),
                      patch.object(cli, 'build_scene', return_value=scene),
                      patch.object(cli, 'initialize_reference', return_value=None),
                      patch.object(cli, 'controller_config', return_value=config),
                      patch.object(cli, 'controller_bounds', return_value=(None, None, None, None)),
                      patch.object(cli, 'Controller', return_value=SimpleNamespace(config=config)),
                      patch.object(cli, 'execute_run', side_effect=execute)):
                    cli.main('handheld', self.paths.entry_file, co_locate_outputs=colocate)
                expected = output if colocate else self.output / 'results' / 'exp2_latest_grasp_pose'
                self.assertEqual(captured[0].output_root, expected)

    def test_debug_option_prints_named_first_node_residual(self):
        prediction = SimpleNamespace(residuals=np.ones((3, 2, 6)), residual_rates=np.full((3, 2, 6), .25))
        solved = SimpleNamespace(prediction=prediction, status=0)
        output = io.StringIO()
        with redirect_stdout(output):
            runner.debug_solution(SimpleNamespace(debug_eps=True), solved)
        self.assertIn("[epsdbg] status=0", output.getvalue())
        self.assertIn("epsd1x=+0.25000 epsd2x=+0.25000", output.getvalue())


if __name__ == "__main__":
    unittest.main()
