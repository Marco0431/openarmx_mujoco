"""Startup guesses must cover N+1 states and balance the unchanged model."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import nmpcc_cart as nc
import exp2_nmpcc_latest as exp
from test_nmpcc_revision import RecordingSolver, toy_fk


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.cfg = nc.NMPCCCartConfig(N=3, kx=100, rest_gap=.5, kf_scale=.003)
        self.factory = lambda ocp, **kw: RecordingSolver(ocp)
        with patch.object(nc, 'AcadosOcpSolver', side_effect=self.factory):
            self.c = nc.NMPCCCart(self.cfg, toy_fk())
        self.x = np.linspace(-.2, .2, nc.NX)
        self.p = np.zeros(nc.NP)
        ml = np.eye(7)*2 + np.full((7, 7), .03)
        mr = np.eye(7)*3 + np.full((7, 7), .02)
        self.p[:49], self.p[56:105] = ml.ravel(), mr.ravel()
        self.p[49:56], self.p[105:112] = np.arange(7), -np.arange(7)
        self.p[112:126] = np.linspace(-2, 3, 14)
        self.p[162:176] = np.linspace(.2, .8, 14)
        self.p[178] = .003

    def assert_terminal(self, expected):
        np.testing.assert_array_equal(self.c.solver.get(self.cfg.N, 'x'), expected)

    def test_cold_seed_and_unseeded_old_interface_cover_terminal(self):
        for seed in (None, np.arange(14)):
            self.c._warm = None
            self.c.solver.set(self.cfg.N, 'x', np.full(nc.NX, 99))
            self.c.solve(self.x, self.p, u_seed=seed)
            for k in range(self.cfg.N+1):
                np.testing.assert_array_equal(self.c.solver.get(k, 'x'), self.x)
            self.assertEqual(self.c._warm[0].shape, (self.cfg.N+1, nc.NX))
            self.assertEqual(self.c._warm[1].shape, (self.cfg.N, nc.NU))

    def test_warm_terminal_restored_without_horizon_shift(self):
        states = np.tile(self.x, (self.cfg.N+1, 1)) + np.arange(self.cfg.N+1)[:, None]
        controls = np.arange(self.cfg.N*nc.NU).reshape(self.cfg.N, nc.NU)
        self.c._warm = (states.copy(), controls.copy())
        self.c.solver.set(self.cfg.N, 'x', np.full(nc.NX, 99))
        measurement = self.x+.1
        self.c.solve(measurement, self.p)
        self.assert_terminal(states[-1])
        np.testing.assert_array_equal(self.c.solver.get(0, 'x'), measurement)
        for k in range(1, self.cfg.N):
            np.testing.assert_array_equal(self.c.solver.get(k, 'x'), states[k])
        np.testing.assert_array_equal(self.c._warm[1], controls)

    def test_failure_rebuild_initializes_terminal_cold_and_warm(self):
        for warm in (False, True):
            terminal = self.x + 2 if warm else self.x
            self.c._warm = ((np.tile(terminal, (4, 1)), np.zeros((3, 14))) if warm else None)
            self.c.solver.status = 4
            with patch.object(nc, 'AcadosOcpSolver', side_effect=self.factory):
                self.c.solve(self.x, self.p, u_seed=np.zeros(14))
            self.assert_terminal(terminal)
            np.testing.assert_array_equal(self.c.solver.get(0, 'x'), self.x)
            self.assertEqual(self.c.last_status, 0)

    def test_equilibrium_cancels_full_acceleration_legacy_and_node_parameters(self):
        original_x, original_p = self.x.copy(), self.p.copy()
        for width in (176, 178, 179, 191):
            for nodes in (False, True):
                p = self.p[:width].copy()
                if nodes:
                    p = np.tile(p, (4, 1))
                    p[1:, 49:56] += 100  # Future masses/loads cannot affect the seed.
                seed = self.c.equilibrium_seed(self.x, p, f_meas=[-8, 9])
                anchored = self.c._anchor_measured_force(self.x, p, [-8, 9])
                p0 = anchored[0] if nodes else anchored
                dx = np.asarray(self.c._dynamics_fn(self.x, seed, p0)).ravel()
                np.testing.assert_allclose(dx[np.r_[7:14, 21:28]], 0, atol=1e-11)
                np.testing.assert_array_equal(dx[np.r_[0:7, 14:21]], self.x[np.r_[7:14, 21:28]])
        np.testing.assert_array_equal(self.x, original_x)
        np.testing.assert_array_equal(self.p, original_p)

    def test_equilibrium_does_not_change_solver_iterate(self):
        before = {k:v.copy() for k,v in self.c.solver.values.items()}
        self.c.equilibrium_seed(self.x, self.p, [-10, 10])
        for key,value in before.items():
            np.testing.assert_array_equal(value, self.c.solver.values[key])

    def test_equilibrium_rejects_invalid_state_and_parameters(self):
        for x in (self.x[:-1], np.full(nc.NX, np.nan)):
            with self.assertRaises(ValueError):
                self.c.equilibrium_seed(x, self.p)
        with self.assertRaises(ValueError):
            self.c.equilibrium_seed(self.x, np.full(nc.NP, np.inf))
        with self.assertRaises(ValueError):
            self.c.equilibrium_seed(self.x, self.p, [np.nan, 10])

    def test_invalid_startup_seed_saves_failure_before_physics(self):
        class Recorder:
            def __init__(self, *a):
                self.metadata, self.failure = {}, {}
            def export(self, horizon):
                return {'failure':self.failure.copy()}
        data = SimpleNamespace(qpos=np.zeros(14), qvel=np.zeros(14), time=0.0)
        arms = [{'qpos':np.arange(7), 'dofs':np.arange(7)},
                {'qpos':np.arange(7,14), 'dofs':np.arange(7,14)}]
        # run_case 现在先建立可选扰动包装；测试仍只验证非法种子在物理步前停止。
        args = SimpleNamespace(dynamics_diag=False, kf_scale=.003, N=3, u_lim=6.0,
                               f_start=10.0, hold_only=True)
        sample = SimpleNamespace(refs={'left':np.zeros(6),'right':np.zeros(6)},
            vrefs={'left':np.zeros(6),'right':np.zeros(6)},fmag=10, fmag_dot=0, phase='hold')
        ref = SimpleNamespace(sample=lambda t:sample)
        self.c.node_times = np.linspace(0,.1,4)
        for bad in (np.full(14,np.nan), np.full(14,7.0)):
            with patch('exp2_experiments.flat_observation.JointLimitRecorder',Recorder), \
                 patch.object(exp,'sample_plant_parameters',return_value=np.zeros(140)), \
                 patch.object(exp,'contact_forces',return_value={'left':10,'right':10}), \
                 patch.object(self.c,'equilibrium_seed',return_value=bad), \
                 patch.object(exp.mujoco,'mj_step') as step:
                with self.assertRaisesRegex(ValueError,'startup stopped'):
                    exp.run_case(SimpleNamespace(),data,arms,
                        {'box_body':0,'box_geom':0,'box_qpos':0},args,self.c,ref,1.5)
                step.assert_not_called()
                self.assertEqual(self.c.last_joint_limit_trace['failure']['t'],0)
                np.testing.assert_array_equal(data.qpos,np.zeros(14))
                np.testing.assert_array_equal(data.qvel,np.zeros(14))
                self.assertEqual(data.time,0)


if __name__ == '__main__':
    unittest.main()
