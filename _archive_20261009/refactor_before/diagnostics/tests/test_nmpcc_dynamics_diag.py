"""Diagnostic comparisons must change only M and never forward the live plant."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

import exp2_diagnostics as diagnostics
import nmpcc_cart as nc
from test_nmpcc_revision import RecordingSolver, toy_fk


class DynamicsDiagnosticTests(unittest.TestCase):
    def test_same_state_mass_comparison_and_private_forward(self):
        bodies = ''.join(
            f'<body><inertial pos="0 0 0" mass="1" diaginertia=".2 .2 .2"/>'
            f'<joint name="j{i}" type="hinge" axis="0 0 1" damping=".2"/></body>'
            for i in range(14))
        motors = ''.join(f'<motor joint="j{i}"/>' for i in range(14))
        model = mujoco.MjModel.from_xml_string(
            f'<mujoco><option gravity="0 0 0"/><worldbody>{bodies}</worldbody>'
            f'<actuator>{motors}</actuator></mujoco>')
        data = mujoco.MjData(model)
        data.qpos[:] = np.linspace(-.1, .1, 14)
        data.qvel[:] = np.linspace(.1, .3, 14)
        mujoco.mj_forward(model, data)
        arms = [dict(dofs=np.arange(i, i+7), qpos=np.arange(i, i+7),
                     act_ids=np.arange(i, i+7)) for i in (0, 7)]
        mass = np.empty((14, 14)); mujoco.mj_fullM(model, data, mass)
        current = np.array([mass[:7, :7], mass[7:, 7:]])
        p = np.zeros(nc.NP)
        p[:49], p[56:105] = current[0].ravel(), current[1].ravel()
        p[162:176] = .2
        cfg = nc.NMPCCCartConfig(N=2, kx=0)
        with patch.object(nc, 'AcadosOcpSolver', side_effect=lambda ocp, **kw: RecordingSolver(ocp)):
            ctrl = nc.NMPCCCart(cfg, toy_fk())
        ctrl.last_parameters = np.tile(p, (3, 1))
        data.ctrl[:] = np.linspace(.2, 1.0, 14)
        fields = ('qpos', 'qvel', 'qacc', 'ctrl', 'qfrc_bias', 'qacc_warmstart')
        before = {k: getattr(data, k).copy() for k in fields}
        t = float(data.time)
        rec = diagnostics.DynamicsRecorder(model, arms, {})
        ref = SimpleNamespace(t_ramp=.5, t_hold_end=1.5, accel_time=.2,
                              base_time=10., move_duration=10.2, move_end=11.7)
        snapshot = dict(mass_old=1.1*current, mass_current=current, sampling_diagnostic_ms=0.)
        rec.record(data, ctrl, snapshot, ref, t)
        row = rec.rows[0]
        np.testing.assert_allclose(row['acceleration_current'], row['acceleration_mujoco'], atol=1e-12)
        self.assertGreater(np.linalg.norm(row['error_old']), .1)
        other = np.ones(nc.NP, bool); other[:49] = False; other[56:105] = False
        np.testing.assert_array_equal(row['p_old'][other], row['p_current'][other])
        np.testing.assert_array_equal(row['u'], data.ctrl)
        np.testing.assert_array_equal(ctrl.last_parameters, np.tile(p, (3, 1)))
        for field in fields:
            np.testing.assert_array_equal(getattr(data, field), before[field])
        mass_after = np.empty_like(mass); mujoco.mj_fullM(model, data, mass_after)
        np.testing.assert_array_equal(mass_after, mass)
        self.assertEqual(data.time, t)
        self.assertEqual(rec.export()['acceleration_current'].shape, (1, 14))


if __name__ == '__main__':
    unittest.main()
