"""Focused regression tests for optional joint position limits.

Run with ``D:/acados_env/python.exe -m unittest openarmx_mujoco.test_nmpcc_limits -v``.
All solver construction is mocked; this file never compiles an OCP or runs a scene.
"""
import unittest
from unittest.mock import patch
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import nmpcc_cart as nc
import exp2_diagnostics as diagnostics
import mujoco
from test_nmpcc_revision import toy_fk, RecordingSolver


class JointLimitTests(unittest.TestCase):
    def setUp(self):
        self.cfg = nc.NMPCCCartConfig(N=3, Tp=.1, Q=3, P=2, kf_scale=.003,
                                       kx=100, rest_gap=.5, eta_dev=.05, eta_u=5e-8)
        self.lo = np.arange(14, dtype=float) - 10
        self.hi = self.lo + 1.5
        with patch.object(nc, 'AcadosOcpSolver', side_effect=lambda ocp, **kw: RecordingSolver(ocp)):
            self.ctrl = nc.NMPCCCart(self.cfg, toy_fk(), verbose=False,
                                     qmin=self.lo, qmax=self.hi)

    def _make(self, **kw):
        with patch.object(nc, 'AcadosOcpSolver', side_effect=lambda ocp, **opts: RecordingSolver(ocp)):
            return nc.NMPCCCart(self.cfg, toy_fk(), verbose=False, **kw)

    def test_mapping_is_nonsymmetric_and_unbounded_joints_are_skipped(self):
        lo, hi = self.lo.copy(), self.hi.copy()
        lo[2], hi[2] = -np.inf, np.inf
        c = self._make(qmin=lo, qmax=hi)
        expected = np.array([0, 1, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19, 20])
        np.testing.assert_array_equal(c.solver.ocp.constraints.idxbx, expected)
        np.testing.assert_array_equal(c.solver.ocp.constraints.lbx, lo[np.isfinite(lo)])
        np.testing.assert_array_equal(c.solver.ocp.constraints.ubx, hi[np.isfinite(hi)])
        np.testing.assert_array_equal(c.solver.ocp.constraints.idxbx_e, expected)
        np.testing.assert_array_equal(c.solver.ocp.constraints.lbx_e, lo[np.isfinite(lo)])
        np.testing.assert_array_equal(c.solver.ocp.constraints.ubx_e, hi[np.isfinite(hi)])

    def test_legacy_constructor_has_no_position_constraints(self):
        c = self._make()
        self.assertIsNone(c.qmin)
        self.assertFalse(c.q_bound_mask.any())
        idx = getattr(c.solver.ocp.constraints, 'idxbx', None)
        self.assertTrue(idx is None or np.asarray(idx).size == 0)

    def test_input_validation(self):
        cases = [dict(qmin=self.lo), dict(qmax=self.hi),
                 dict(qmin=np.zeros(13), qmax=np.ones(13)),
                 dict(qmin=np.full(14, np.nan), qmax=self.hi),
                 dict(qmin=np.zeros(14), qmax=np.zeros(14)),
                 dict(qmin=np.full(14, -np.inf), qmax=np.full(14, 1.0))]
        for kw in cases:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self._make(**kw)

    def test_x0_is_measured_and_prediction_keeps_all_nodes(self):
        x = np.zeros(nc.NX); x[0] = 99.0; x[14] = -99.0
        p = np.zeros(nc.NP)
        self.ctrl.solve(x, p)
        np.testing.assert_array_equal(self.ctrl.solver.ocp.constraints.x0, np.zeros(nc.NX))
        np.testing.assert_array_equal(self.ctrl.solver.values[0, 'x'], x)
        self.assertEqual(self.ctrl.last_status, 0)
        self.assertEqual(self.ctrl.last_pred_x.shape, (self.cfg.N + 1, nc.NX))

    def test_cache_identity_changes_with_limits(self):
        a = nc._solver_identity(self.cfg, toy_fk(), qmin=self.lo, qmax=self.hi)
        b = nc._solver_identity(self.cfg, toy_fk(), qmin=self.lo, qmax=self.hi + .1)
        c = nc._solver_identity(self.cfg, toy_fk())
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertTrue(a.startswith('nmpcc_v7_'))

    def test_failure_rebuild_preserves_bounds_and_does_not_clip_x0(self):
        self.ctrl.solver.status = 4
        x = np.zeros(nc.NX); x[0] = 100
        with patch.object(nc, 'AcadosOcpSolver', side_effect=lambda ocp, **kw: RecordingSolver(ocp)):
            self.ctrl.solve(x, np.zeros(nc.NP))
        np.testing.assert_array_equal(self.ctrl.solver.ocp.constraints.lbx, self.lo)
        np.testing.assert_array_equal(self.ctrl.solver.ocp.constraints.ubx, self.hi)
        np.testing.assert_array_equal(self.ctrl.solver.values[0, 'x'], x)
        self.assertEqual(self.ctrl.first_failure['status'], 4)
        np.testing.assert_array_equal(self.ctrl.first_failure['x0'], x)
        self.assertEqual(self.ctrl.last_status, 0)

    def test_limits_do_not_change_cost_solver_options_or_dynamics(self):
        plain = self._make()
        for a, b in ((plain.W, self.ctrl.W), (plain.W_e, self.ctrl.W_e)):
            np.testing.assert_array_equal(a, b)
        for name in ('nlp_solver_type', 'qp_solver', 'integrator_type', 'hessian_approx',
                     'N_horizon', 'tf', 'sim_method_num_steps', 'sim_method_num_stages',
                     'qp_solver_iter_max', 'qp_solver_cond_N', 'nlp_solver_max_iter',
                     'regularize_method', 'levenberg_marquardt'):
            av, bv = getattr(plain.solver.ocp.solver_options, name), getattr(self.ctrl.solver.ocp.solver_options, name)
            if isinstance(av, np.ndarray) or isinstance(bv, np.ndarray):
                np.testing.assert_array_equal(av, bv)
            else:
                self.assertEqual(av, bv)
        x = np.linspace(-.1, .2, nc.NX); p = np.zeros(nc.NP); u = np.linspace(-.2, .3, nc.NU)
        p[:49] = (2*np.eye(7)).reshape(-1); p[56:105] = (3*np.eye(7)).reshape(-1)
        p[126:150] = np.linspace(-.1, .1, 24); p[150:162] = np.linspace(1., 2., 12)
        p[176:178] = [2., 3.]; p[178:191] = np.linspace(.01, .13, 13)
        a = np.asarray(plain._dynamics_fn(x, u, p), dtype=float)
        b = np.asarray(self.ctrl._dynamics_fn(x, u, p), dtype=float)
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
        np.testing.assert_allclose(a, b)


class MuJoCoLimitTests(unittest.TestCase):
    def _model(self, sparse=False):
        opt = '<option integrator="Euler" timestep="0.01"/>'
        xml = opt + '<compiler angle="radian"/><worldbody><body pos="0 0 1"><inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/><joint name="left" type="hinge" axis="0 0 1" range="-.1 .1" limited="true"/><body><inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/><joint name="right" type="hinge" axis="0 0 1" range="-.2 .2" limited="true"/></body></body></worldbody>'
        m = mujoco.MjModel.from_xml_string('<mujoco>' + xml + '</mujoco>')
        if sparse:
            m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
        return m

    def test_bounds_mapping_and_euler_limit_direction(self):
        for sparse in (False, True):
            m = self._model(sparse); d = mujoco.MjData(m)
            arms = [{'joint_ids': np.array([0]), 'qpos': np.array([0]), 'dofs': np.array([0])},
                    {'joint_ids': np.array([1]), 'qpos': np.array([1]), 'dofs': np.array([1])}]
            lo, hi = diagnostics.arm_joint_bounds(m, arms)
            np.testing.assert_array_equal(lo, [-.1, -.2]); np.testing.assert_array_equal(hi, [.1, .2])
            rec = diagnostics.JointLimitRecorder(m, d, arms, {'sparse': sparse})
            for q in (-.12, .12):
                d.qpos[:] = [q, 0.0]; d.qvel[:] = 0
                time_before = float(d.time)
                captured = rec.before_step(d); mujoco.mj_step(m, d); rec.after_step(m, d, captured)
                row = rec.rows[-1]
                self.assertAlmostEqual(row[0], time_before)
                self.assertAlmostEqual(row[1][0], q)
                np.testing.assert_array_equal(row[2], [0.0, 0.0])
                self.assertNotAlmostEqual(d.qpos[0], q)
                self.assertTrue(row[3][0] < 0 if q < 0 else row[4][0] < 0)
                self.assertTrue(row[5][0])
                self.assertNotEqual(row[6][0], 0.0)
                self.assertLess(row[6][0] * q, 0.0)
                np.testing.assert_allclose(row[6], d.qfrc_constraint, atol=1e-12)
            m.jnt_limited[1] = False
            lo, hi = diagnostics.arm_joint_bounds(m, arms)
            self.assertTrue(np.isneginf(lo[1]) and np.isposinf(hi[1]))


if __name__ == '__main__':
    unittest.main()
