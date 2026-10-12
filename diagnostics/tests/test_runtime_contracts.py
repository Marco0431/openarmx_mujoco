"""公共参考和采样的行为对照，不编译求解器、不运行长闭环。"""
from __future__ import annotations

import ast
import importlib.util
from dataclasses import dataclass
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from diagnostics.measurements import sample_dynamics
from experiments.reference import PhaseReference


def old_reference_type():
    """从冻结源码只加载纯参考函数，避免导入旧实验及其求解器。"""
    source = Path(__file__).resolve().parents[2] / "_archive_20261009/refactor_before/exp2_nmpcc_latest.py"
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    names = {"smoothstep", "motion_timing", "ReferenceSample", "PhaseReference"}
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    namespace = {"np": np, "dataclass": dataclass}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["PhaseReference"]


def old_sampler():
    path = Path(__file__).resolve().parents[2] / "_archive_20261009/refactor_before/measurements.py"
    spec = importlib.util.spec_from_file_location("frozen_measurements", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sample_plant_parameters


def make_reference(cls=PhaseReference, *, distance=.1, speed=-.01, ramp=.5):
    initial = {"left": np.array([-.2, .03, .1]), "right": np.array([.2, .03, .1])}
    pressed = {"left": initial["left"] + [.001, 0, 0], "right": initial["right"] + [-.001, 0, 0]}
    rotation = {side: np.array([.1, .2, -.1]) for side in initial}
    return cls(12., speed, distance, ramp, 1., initial, pressed, rotation, 10.)


def sampling_scene():
    """每臂七关节，其中前两关节产生随姿态变化的耦合惯性。"""
    bodies = []
    for side, x, rotation in (("left", -.16, "0.7071067811865476 0 0.7071067811865476 0"),
                              ("right", .16, "0.7071067811865476 0 -0.7071067811865476 0")):
        parts = [f'<body name="{side}_0" pos="{x} 0 0">']
        for joint in range(7):
            if joint:
                parts.append(f'<body name="{side}_{joint}" pos="0 0 .07">')
            axis = "0 1 0" if joint < 2 else "0 0 1"
            parts.append(f'<joint name="{side}_j{joint}" axis="{axis}" damping=".3"/>')
            parts.append('<geom type="capsule" fromto="0 0 0 0 0 .07" size=".008" mass=".1" contype="0" conaffinity="0"/>')
        parts.append(f'<geom name="{side}_pad" type="box" pos="0 0 .07" quat="{rotation}" size=".05 .05 .08" mass=".01"/>')
        parts.append('</body>' * 7)
        bodies.append(''.join(parts))
    xml = '<mujoco><option timestep=".002"/><worldbody>' + ''.join(bodies)
    xml += '<body name="box" pos="0 0 .49"><freejoint/><geom name="box" type="box" size=".1 .12 .05" mass=".5"/></body></worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    arms = []
    for side in ("left", "right"):
        joints = [model.joint(f"{side}_j{i}").id for i in range(7)]
        arms.append({"side": side, "dofs": model.jnt_dofadr[joints], "qpos": model.jnt_qposadr[joints],
                     "body_id": model.body(f"{side}_6").id, "pad_id": model.geom(f"{side}_pad").id})
    mujoco.mj_forward(model, data)
    return model, data, arms, model.geom("box").id


class ReferenceContractTests(unittest.TestCase):
    def test_batch_matches_old_samples_across_every_phase(self):
        old_type = old_reference_type()
        for distance, speed, ramp in ((.1, -.01, .5), (.001, -.01, .5), (0, -.01, .5), (.1, 0, .5), (.1, -.01, 0)):
            with self.subTest(distance=distance, speed=speed, ramp=ramp):
                old = make_reference(old_type, distance=distance, speed=speed, ramp=ramp)
                new = make_reference(distance=distance, speed=speed, ramp=ramp)
                times = np.unique(np.r_[0., .25, .5, 1.49, 1.5, 1.6, 1.7, new.move_end-.1, new.move_end, new.move_end+.4])
                horizon = new.horizon(0., times)
                for node, time in enumerate(times):
                    expected = old.sample(float(time))
                    np.testing.assert_allclose(horizon.poses[node], [expected.refs[s] for s in ("left", "right")], atol=1e-14, rtol=1e-14)
                    np.testing.assert_allclose(horizon.velocities[node], [expected.vrefs[s] for s in ("left", "right")], atol=1e-14, rtol=1e-14)
                    np.testing.assert_allclose(horizon.forces[node, :, 0], [-expected.fmag, expected.fmag], atol=1e-14)
                    np.testing.assert_allclose(horizon.force_rates[node, :, 0], [-expected.fmag_dot, expected.fmag_dot], atol=1e-14)

    def test_reference_derivatives_and_terminal(self):
        reference = make_reference()
        for time in (.25, 1., 1.6, 2., reference.move_end-.1, reference.move_end+.1):
            delta = 1e-6
            before = reference.horizon(time-delta, np.zeros(1))
            current = reference.horizon(time, np.zeros(1))
            after = reference.horizon(time+delta, np.zeros(1))
            np.testing.assert_allclose((after.poses-before.poses)/(2*delta), current.velocities, atol=1e-9)
            np.testing.assert_allclose((after.forces-before.forces)/(2*delta), current.force_rates, atol=1e-8)
        horizon = reference.horizon(1.49, np.linspace(0., .1, 11))
        self.assertEqual(horizon.poses.shape, (11, 2, 6))
        self.assertLess(horizon.poses[-1, 0, 1], horizon.poses[0, 0, 1])


class SamplingContractTests(unittest.TestCase):
    def test_named_sampling_matches_original_parameters_and_loads(self):
        model, data, arms, box = sampling_scene()
        data.qvel[arms[0]["dofs"]] = np.linspace(.01, .02, 7)
        mujoco.mj_step(model, data)
        qpos, qvel, time = data.qpos.copy(), data.qvel.copy(), data.time
        old_snapshot = {}
        old = old_sampler()(model, data, arms, box, contact_snapshot=old_snapshot)
        new = sample_dynamics(model, data, arms, box)
        np.testing.assert_array_equal(data.qpos, qpos)
        np.testing.assert_array_equal(data.qvel, qvel)
        self.assertEqual(data.time, time)
        for arm, offset in enumerate((0, 56)):
            np.testing.assert_array_equal(new.mass[arm], old[offset:offset+49].reshape(7, 7))
            np.testing.assert_array_equal(new.gravity[arm], old[offset+49:offset+56])
        np.testing.assert_array_equal(new.damping.ravel(), old[126:140])
        np.testing.assert_allclose(new.contact.normal_magnitude, old_snapshot["normal_magnitude"], atol=1e-12)
        np.testing.assert_allclose(new.contact.joint_load.ravel(), old_snapshot["full_contact"], atol=1e-10)
        np.testing.assert_allclose(new.contact.tangent_joint_load.ravel(), old_snapshot["tangent_contact"], atol=1e-10)
        np.testing.assert_allclose(np.c_[new.contact.force_world, new.contact.contact_torque_world].ravel(), old_snapshot["world_wrench"], atol=1e-12)

    def test_two_refreshes_and_velocity_restored_on_failure(self):
        model, data, arms, box = sampling_scene()
        data.qvel[:] = .03
        original = mujoco.mj_forward
        calls = []
        def refresh(m, d):
            calls.append(d.qvel.copy())
            if len(calls) == 1:
                raise RuntimeError("injected sampling failure")
            original(m, d)
        with patch.object(mujoco, "mj_forward", side_effect=refresh):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                sample_dynamics(model, data, arms, box)
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0], 0.)
        np.testing.assert_array_equal(data.qvel, .03)

    def test_common_point_wrench_reproduces_contact_joint_load(self):
        model, data, arms, box = sampling_scene()
        snapshot = sample_dynamics(model, data, arms, box)
        self.assertGreater(snapshot.contact.normal_magnitude.min(), 0.)
        for side, arm in enumerate(arms):
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jp, jr, snapshot.contact.reference_points_world[side], arm["body_id"])
            wrench = snapshot.contact.wrench_world[side]
            mapped = jp[:, arm["dofs"]].T @ wrench[:3] + jr[:, arm["dofs"]].T @ wrench[3:]
            np.testing.assert_allclose(mapped, snapshot.contact.joint_load[side], rtol=1e-10, atol=1e-10)

    def test_stale_inertia_is_refreshed_against_independent_data(self):
        model, data, arms, box = sampling_scene()
        data.qpos[arms[0]['qpos'][:2]] = [.8, -.55]
        data.qvel[arms[0]['dofs'][:2]] = [1.3, -.8]
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        saved = (data.qpos.copy(), data.qvel.copy(), data.time)
        diagnostic = {}
        original = mujoco.mj_forward
        calls = []
        def refresh(m, d):
            calls.append(d.qvel.copy())
            return original(m, d)
        with patch.object(mujoco, 'mj_forward', side_effect=refresh):
            snapshot = sample_dynamics(model, data, arms, box, diagnostic=diagnostic)
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0], 0.)
        np.testing.assert_array_equal(calls[1], saved[1])
        np.testing.assert_array_equal(data.qpos, saved[0])
        np.testing.assert_array_equal(data.qvel, saved[1])
        self.assertEqual(data.time, saved[2])
        shadow = mujoco.MjData(model)
        mujoco.mj_copyData(shadow, model, data)
        mujoco.mj_forward(model, shadow)
        mass = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, shadow, mass)
        expected = [mass[np.ix_(a['dofs'], a['dofs'])] for a in arms]
        np.testing.assert_allclose(snapshot.mass, expected, atol=1e-12, rtol=1e-12)
        self.assertGreater(np.max(np.abs(diagnostic['mass_old']-snapshot.mass)), 1e-9)

    def test_static_inertia_does_not_change(self):
        model, data, arms, box = sampling_scene()
        model.opt.gravity[:] = 0.
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        data.qpos[arms[0]['qpos'][:2]] = [.73, -.41]
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        diagnostic = {}
        sample_dynamics(model, data, arms, box, diagnostic=diagnostic)
        np.testing.assert_allclose(diagnostic['mass_old'], diagnostic['mass_current'], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
