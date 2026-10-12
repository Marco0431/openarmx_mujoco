"""Focused tests for exp2_diagnostics.sample_plant_parameters.

These tests intentionally use a tiny real MuJoCo model; no OCP compilation or
long simulation is involved.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import exp2_diagnostics as exp


XML = r"""<mujoco model="sampling">
  <option gravity="0 0 -9.81" timestep="0.002" integrator="Euler"/>
  <worldbody>
    <body name="root" pos="0 0 0">
      <joint name="hinge0" type="hinge" axis="0 1 0" damping="0.31"/>
      <geom name="link0" type="capsule" fromto="0 0 0 0 0 0.7" size="0.08" mass="1"/>
      <body name="tip" pos="0 0 0.7">
        <joint name="hinge1" type="hinge" axis="0 1 0" damping="0.47"/>
        <geom name="link1" type="capsule" fromto="0 0 0 0 0 0.6" size="0.07" mass="0.8"/>
        <geom name="pad" type="sphere" pos="0 0 0.6" size="0.04" mass="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""


def _case():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    arm = {"dofs": np.array([0, 1]), "pad_id": model.geom("pad").id,
           "body_id": model.body("tip").id}
    return model, data, [arm], -1


def test_sampling_matches_independent_shadow_and_preserves_state():
    model, data, arms, box_id = _case()
    data.qpos[:] = [0.8, -0.55]
    mujoco.mj_forward(model, data)
    # Advance once so the pre-sampling mass cache is stale and state-dependent.
    data.qvel[:] = [1.3, -0.8]
    mujoco.mj_step(model, data)
    qpos, qvel, tm = data.qpos.copy(), data.qvel.copy(), float(data.time)
    old = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, old)
    diag = {}
    packed = exp.sample_plant_parameters(model, data, arms, box_id, diagnostic=diag)
    shadow = mujoco.MjData(model)
    mujoco.mj_copyData(shadow, model, data)
    shadow.qvel[:] = 0
    mujoco.mj_forward(model, shadow)
    gravity = shadow.qfrc_bias.copy()
    current = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, shadow, current)
    assert np.array_equal(data.qpos, qpos) and np.array_equal(data.qvel, qvel)
    assert data.time == tm
    assert np.allclose(diag["mass_current"][0], current, rtol=1e-12, atol=1e-12)
    assert np.array_equal(diag["mass_old"][0], old)
    assert np.allclose(packed[:4], current.reshape(-1))
    assert np.allclose(packed[4:6], gravity)
    assert np.allclose(packed[6:8], 0.0)  # no contacts => fJ=0
    assert np.allclose(packed[8:], model.dof_damping[[0, 1]])
    assert not np.allclose(diag["mass_old"][0], diag["mass_current"][0])


def test_two_forwards_and_finally_restores_velocity():
    model, data, arms, box_id = _case()
    data.qvel[:] = [0.4, -0.2]
    real_forward = mujoco.mj_forward
    calls = []
    def wrapped(m, d):
        calls.append(d.qvel.copy())
        if len(calls) == 1:
            raise RuntimeError("injected forward failure")
        return real_forward(m, d)
    original = mujoco.mj_forward
    mujoco.mj_forward = wrapped
    try:
        try:
            exp.sample_plant_parameters(model, data, arms, box_id)
        except RuntimeError:
            pass
        else:
            raise AssertionError("injected forward failure was not propagated")
    finally:
        mujoco.mj_forward = original
    assert len(calls) == 2
    assert np.allclose(calls[0], 0.0)
    assert np.allclose(calls[1], [0.4, -0.2])
    assert np.allclose(data.qvel, [0.4, -0.2])


def test_normal_sampling_uses_exactly_two_forwards_with_restored_velocity():
    model, data, arms, box_id = _case()
    data.qvel[:] = [0.4, -0.2]
    original = mujoco.mj_forward
    calls = []
    def wrapped(m, d):
        calls.append(d.qvel.copy())
        return original(m, d)
    mujoco.mj_forward = wrapped
    try:
        exp.sample_plant_parameters(model, data, arms, box_id)
    finally:
        mujoco.mj_forward = original
    assert len(calls) == 2
    assert np.allclose(calls[0], 0.0)
    assert np.allclose(calls[1], [0.4, -0.2])


def test_diagnostic_sampling_also_uses_exactly_two_forwards():
    model, data, arms, box_id = _case()
    data.qvel[:] = [0.4, -0.2]
    original = mujoco.mj_forward
    calls = []
    def wrapped(m, d):
        calls.append(d.qvel.copy())
        return original(m, d)
    mujoco.mj_forward = wrapped
    try:
        exp.sample_plant_parameters(model, data, arms, box_id, diagnostic={})
    finally:
        mujoco.mj_forward = original
    assert len(calls) == 2
    assert np.allclose(calls[0], 0.0)
    assert np.allclose(calls[1], [0.4, -0.2])


def test_zero_velocity_static_case_has_same_old_and_current_mass():
    model, data, arms, box_id = _case()
    model.opt.gravity[:] = 0.0
    data.qpos[:] = [0.73, -0.41]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)
    data.qvel[:] = 0.0
    diag = {}
    exp.sample_plant_parameters(model, data, arms, box_id, diagnostic=diag)
    assert np.allclose(diag["mass_old"], diag["mass_current"], atol=1e-12)


def test_phase_labels_cover_all_intervals():
    ref = SimpleNamespace(t_ramp=1.0, t_hold_end=2.0, move_duration=3.0,
                          accel_time=1.0, base_time=2.0, move_end=5.0)
    assert [exp.dynamics_phase(ref, t) for t in (0, 1, 2, 3, 4.5, 5.5)] == \
        ["clamp", "hold", "acceleration", "cruise", "deceleration", "stopped"]


if __name__ == "__main__":
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


def load_tests(loader, standard_tests, pattern):
    """Expose the module-level focused checks to unittest discovery."""
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value))
    return suite
