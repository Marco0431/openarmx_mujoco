"""Regression checks for frozen loads, independent weights and preview references.

Run: D:/acados_env/python.exe -m unittest discover -s openarmx_mujoco -p test_nmpcc_revision.py -v
No OCP compilation or long scene initialization is required.
"""
import unittest
from unittest.mock import patch

import casadi as ca
import mujoco
import numpy as np

import nmpcc_cart as nc
import exp2_nmpcc_latest as exp
import exp2_diagnostics as diagnostics


def toy_fk():
    ql, qr = ca.SX.sym('test_ql', 7), ca.SX.sym('test_qr', 7)
    tips = ca.vertcat(ql[:6], qr[:6])
    contacts = ca.vertcat(ql[:3], qr[:3])
    return dict(tips_qt=ca.vertcat(ql, qr), tips=tips, contact_tips=contacts,
                tips_fn=ca.Function('test_tips', [ql, qr], [tips]),
                contact_tips_fn=ca.Function('test_contacts', [ql, qr], [contacts]), n_arm=7)


class RecordingSolver:
    def __init__(self, ocp, status=0):
        ocp.make_consistent()
        self.ocp, self.status = ocp, status
        self.values = {}
        self.constraints = {}
        for i in range(ocp.solver_options.N_horizon + 1):
            self.values[i, 'x'] = np.zeros(nc.NX)
            self.values[i, 'u'] = np.zeros(nc.NU)

    def set(self, i, field, value):
        self.values[i, field] = np.asarray(value).copy()

    def get(self, i, field):
        return self.values[i, field].copy()

    def constraints_set(self, i, field, value):
        self.constraints[i, field] = np.asarray(value).copy()

    def solve_for_x0(self, x0_bar, **kwargs):
        self.set(0, 'x', x0_bar)

    def get_stats(self, field):
        return np.array([0.001])


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = nc.NMPCCCartConfig(N=3, Tp=.1, Q=3, P=2, kf_scale=.003,
                                       kx=100, rest_gap=.5, eta_dev=.05, eta_u=5e-8)
        with patch.object(nc, 'AcadosOcpSolver', side_effect=lambda ocp, **kw: RecordingSolver(ocp)):
            self.ctrl = nc.NMPCCCart(self.cfg, toy_fk())
        self.x = np.zeros(nc.NX)
        self.x[0], self.x[14] = -.2, .2
        self.p = np.zeros(nc.NP)
        self.p[:49] = (2*np.eye(7)).reshape(-1)
        self.p[56:105] = (3*np.eye(7)).reshape(-1)
        self.p[nc.KF_PARAM_IDX] = .003

    def test_independent_weights(self):
        w, terminal = nc.cost_weights(self.cfg)
        self.assertEqual(w[0, 0], 3)
        self.assertEqual(terminal[0, 0], 2)
        self.cfg.Q = 7
        w2, terminal2 = nc.cost_weights(self.cfg)
        np.testing.assert_array_equal(terminal2, terminal)
        self.cfg.P = 5
        w3, terminal3 = nc.cost_weights(self.cfg)
        np.testing.assert_array_equal(w2, w3)
        self.assertEqual(terminal3[0, 0], 5)
        self.cfg.Q, self.cfg.P = 1, 2
        w, terminal = nc.cost_weights(self.cfg)
        np.testing.assert_array_equal(terminal, 2*w[:24, :24])

    def test_tangential_load_and_normal_force_not_duplicated(self):
        p0 = self.p.copy()
        p1 = p0.copy()
        left = np.arange(7, dtype=float)
        right = np.arange(7, dtype=float)+1
        p1[112:119], p1[119:126] = left, right
        a0 = np.array(self.ctrl._dynamics_fn(self.x, np.zeros(14), p0)).ravel()
        a1 = np.array(self.ctrl._dynamics_fn(self.x, np.zeros(14), p1)).ravel()
        np.testing.assert_allclose((a1-a0)[7:14], left/2, atol=1e-12)
        np.testing.assert_allclose((a1-a0)[21:28], right/3, atol=1e-12)
        normal = np.array(self.ctrl._force_nom_fn(self.x)).ravel()
        np.testing.assert_allclose(a0[[7, 21]], normal/[2, 3])
        np.testing.assert_array_equal(a0[[8, 9, 22, 23]], 0)

    def test_parameter_compatibility_and_bias_shared_once(self):
        for width in (176, 178, 179, 191):
            with self.subTest(width=width):
                legacy = self.p[:width]
                self.ctrl.solve(self.x, legacy, f_meas=[-8, 9], u_seed=np.zeros(14))
                self.assertEqual(self.ctrl.last_parameters.shape, (4, 191))
                np.testing.assert_array_equal(self.ctrl.last_parameters[:, nc.FREF_DOT_IDX:], 0)
                np.testing.assert_array_equal(self.ctrl.last_parameters[:, nc.KF_PARAM_IDX], .003)
        rows = np.tile(self.p, (4, 1))
        rows[:, 127] = np.arange(4)*.01
        rows[:, 150] = -np.arange(4)-10
        rows[:, nc.FREF_DOT_IDX] = -2
        with patch.object(self.ctrl, '_force_nom_fn', wraps=self.ctrl._force_nom_fn) as force_fn:
            self.ctrl.solve(self.x, rows, f_meas=[-8, 9])
            self.assertEqual(force_fn.call_count, 1)
        for i in range(4):
            np.testing.assert_array_equal(self.ctrl.solver.get(i, 'p')[:nc.NP],
                                          self.ctrl.last_parameters[i])
        np.testing.assert_allclose(self.ctrl.last_parameters[:, 176:178],
                                   np.tile(self.ctrl.last_force_bias, (4, 1)))
        self.assertNotEqual(self.ctrl.last_pred_eps_x[1, 1], self.ctrl.last_pred_eps_x[3, 1])
        with self.assertRaises(ValueError):
            self.ctrl.solve(self.x, rows[:-1])

    def test_force_reference_derivative(self):
        p1 = self.p.copy()
        p1[nc.FREF_DOT_IDX] = 2
        d0 = self.ctrl._diagnostic_fn(self.x, self.p, self.ctrl.Ic1, self.ctrl.Ic2,
                                       self.ctrl.If1, self.ctrl.If2)
        d1 = self.ctrl._diagnostic_fn(self.x, p1, self.ctrl.Ic1, self.ctrl.Ic2,
                                       self.ctrl.If1, self.ctrl.If2)
        self.assertAlmostEqual(float(d1[-1][0]-d0[-1][0]), .8*.003*2)

    def test_complete_residual_time_derivative_during_force_ramp(self):
        profile = ReferenceTests().profile()
        x = self.x.copy()
        x[7:14] = [.002, -.003, .001, 0, 0, 0, 0]
        x[21:28] = [-.001, -.002, .001, 0, 0, 0, 0]
        def parameters(t):
            p = self.p.copy()
            r = profile.sample(t)
            p[126:132], p[132:138] = r.refs['left'], r.refs['right']
            p[138:144], p[144:150] = r.vrefs['left'], r.vrefs['right']
            p[150], p[156] = -r.fmag, r.fmag
            p[nc.FREF_DOT_IDX], p[nc.FREF_DOT_IDX+6] = -r.fmag_dot, r.fmag_dot
            return p
        t, h = .25, 1e-6
        p = parameters(t)
        dx = np.array(self.ctrl._dynamics_fn(x, np.zeros(14), p)).ravel()
        def diagnostic(state, params):
            return self.ctrl._diagnostic_fn(state, params, self.ctrl.Ic1, self.ctrl.Ic2,
                                            self.ctrl.If1, self.ctrl.If2)
        before, after = diagnostic(x-h*dx, parameters(t-h)), diagnostic(x+h*dx, parameters(t+h))
        derivative = (np.array(after[-2])-np.array(before[-2]))/(2*h)
        np.testing.assert_allclose(derivative, np.array(diagnostic(x,p)[-1]), atol=1e-8)

    def test_failure_rebuild_receives_all_node_parameters_and_limits(self):
        self.ctrl.solver.status = 4
        rows = np.tile(self.p, (4, 1))
        rows[:, 127] = np.arange(4)*.01
        with patch.object(nc, 'AcadosOcpSolver', side_effect=lambda ocp, **kw: RecordingSolver(ocp)) as factory:
            self.ctrl.solve(self.x, rows, f_meas=[-8, 9], u_seed=np.zeros(14), u_lim=2)
            self.assertEqual(factory.call_count, 1)
        self.assertEqual(self.ctrl.last_status, 0)
        for i in range(4):
            self.assertEqual(self.ctrl.solver.get(i, 'p')[127], rows[i, 127])
        np.testing.assert_array_equal(self.ctrl.solver.constraints[0, 'ubu'], 2)

    def test_cache_identity_covers_settings_but_not_runtime_kf(self):
        fk = toy_fk()
        identity = nc._solver_identity(self.cfg, fk)
        self.assertTrue(identity.startswith('nmpcc_v7_'))
        self.cfg.kf_scale = .001
        self.assertEqual(identity, nc._solver_identity(self.cfg, fk))
        self.cfg.P = 3
        self.assertNotEqual(identity, nc._solver_identity(self.cfg, fk))
        self.assertNotEqual(nc._solver_identity(self.cfg, fk, [-2]*14, [2]*14),
                            nc._solver_identity(self.cfg, fk, [-3]*14, [3]*14))


class ReferenceTests(unittest.TestCase):
    def profile(self, **kwargs):
        p0 = {'left': np.array([-.2, 0., 0.]), 'right': np.array([.2, 0., 0.])}
        p1 = {'left': p0['left']+[.001, 0, 0], 'right': p0['right']+[-.001, 0, 0]}
        options = dict(force=12, vy=-.01, distance=.01, t_ramp=.5, t_hold=1,
                       press0=p0, press=p1, rv_ref={s: np.zeros(3) for s in p0}, f_start=10)
        options.update(kwargs)
        return exp.phase_profile(**options)[0]

    def test_analytic_derivatives_and_stage_boundaries(self):
        f = self.profile()
        self.assertAlmostEqual(f.move_duration, 1.2)
        points = [-.1, .1, .25, .499, .5, 1., 1.5, 1.6, 1.7, 2., 2.5, 2.6, 2.7, 3.]
        h = 1e-6
        for t in points:
            a, b, r = f.sample(t-h), f.sample(t+h), f.sample(t)
            for side in ('left', 'right'):
                np.testing.assert_allclose((b.refs[side]-a.refs[side])/(2*h),
                                           r.vrefs[side], atol=1e-7)
            self.assertAlmostEqual((b.fmag-a.fmag)/(2*h), r.fmag_dot, delta=3e-5)
        self.assertAlmostEqual(f.sample(f.move_end).refs['left'][1], -.01)
        np.testing.assert_array_equal(f.sample(f.move_end).vrefs['left'], 0)

    def test_disabled_motion_and_zero_ramp(self):
        for opts in (dict(distance=0), dict(vy=0)):
            f = self.profile(**opts)
            self.assertEqual(f.move_duration, 0)
            self.assertEqual(f.sample(5).refs['left'][1], 0)
        f = self.profile(t_ramp=0)
        self.assertEqual(f.sample(0).fmag, 12)
        self.assertEqual(f.sample(0).fmag_dot, 0)
        np.testing.assert_array_equal(f.sample(0).vrefs['left'], 0)

    def test_preview_distance_and_short_move(self):
        f = self.profile()
        self.assertAlmostEqual(f.sample(2.1).refs['left'][1]-f.sample(2.).refs['left'][1], -.001)
        short = self.profile(distance=.0001)
        self.assertAlmostEqual(short.sample(short.move_end).refs['left'][1], -.0001)
        np.testing.assert_array_equal(short.sample(short.move_end).vrefs['left'], 0)


class ContactTests(unittest.TestCase):
    def test_both_contact_sides_match_constraint_force(self):
        seen_sides = set()
        for order in (0, 1):
            pad = '<body name="pad_body" pos=".15 0 0"><freejoint/><geom name="pad" type="sphere" size=".1" mass="1"/></body>'
            box = '<body name="box_body"><geom name="box" type="sphere" size=".1"/></body>'
            m = mujoco.MjModel.from_xml_string('<mujoco><option gravity="0 0 0"/><default><geom friction="1 .1 .1" condim="6"/></default><worldbody>'
                                               + (pad+box if order == 0 else box+pad) + '</worldbody></mujoco>')
            d = mujoco.MjData(m)
            d.qvel[:] = [0, .2, .3, .2, .1, .1]
            mujoco.mj_forward(m, d)
            arm = dict(pad_id=m.geom('pad').id, body_id=m.body('pad_body').id,
                       dofs=list(range(6)), side='left')
            full, tangent = np.zeros(6), np.zeros(6)
            force_sum, moment_sum = np.zeros(3), np.zeros(3)
            for i in range(d.ncon):
                c = d.contact[i]
                sign = 1 if c.geom2 == arm['pad_id'] else -1
                seen_sides.add(sign)
                w = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, w)
                f, torque = sign*c.frame.reshape(3, 3).T@w[:3], sign*c.frame.reshape(3, 3).T@w[3:]
                force_sum += f
                moment_sum += torque
                mujoco.mj_applyFT(m, d, f, torque, c.pos, arm['body_id'], full)
                f[0] = 0
                mujoco.mj_applyFT(m, d, f, torque, c.pos, arm['body_id'], tangent)
            np.testing.assert_allclose(full, d.qfrc_constraint, atol=1e-10)
            actual, _ = diagnostics.contact_wrench_on_arm(m, d, arm, m.geom('box').id)
            np.testing.assert_allclose(actual, tangent, atol=1e-10)
            w = diagnostics.pad_box_wrench_3d(m, d, [arm], m.geom('box').id)['left']
            np.testing.assert_allclose(w[0], force_sum)
            np.testing.assert_allclose(w[1], moment_sum)
            box_force, _ = diagnostics.box_net_wrench(m, d, [arm], m.geom('box').id, np.zeros(3))
            np.testing.assert_allclose(box_force, -force_sum)
        self.assertEqual(seen_sides, {-1, 1})


if __name__ == '__main__':
    unittest.main()
