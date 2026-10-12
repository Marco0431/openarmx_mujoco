"""共享核心的独立旧源码 oracle 与假求解器回归；不编译 DLL、不跑闭环。"""
from __future__ import annotations
import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from unittest.mock import patch
import casadi as ca
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from openarmx_mujoco.nmpcc import (BiasPolicy, ContactLoadMode, ContactMeasurement, Controller,
    ControllerConfig, DynamicsSnapshot, ForceFeedbackSource, NormalForceLaw, JointState, ReferenceHorizon, RuntimeWeights, SolveOptions, SolverError)
from openarmx_mujoco.nmpcc.ocp import Bounds, cost_weights, create_solver, solver_identity
from openarmx_mujoco.nmpcc.parameters import (encode, encode_legacy_horizon, encode_shared_horizon,
    from_legacy, to_legacy, to_legacy_full, view, parameter_size, layout_version)
from openarmx_mujoco.nmpcc.kinematics import mujoco_fk_symbols, pad_contact_offsets, rotvec_of_R


def toy_fk():
    ql, qr = ca.SX.sym("oracle_ql", 7), ca.SX.sym("oracle_qr", 7)
    # 非线性几何使 Jacobian/弹簧导数测试能够发现漏项。
    left = ca.vertcat(ql[0]+.03*ca.sin(ql[1]), ql[1:6])
    right = ca.vertcat(qr[0]+.04*ca.sin(qr[1]), qr[1:6])
    tips, contacts = ca.vertcat(left, right), ca.vertcat(left[:3], right[:3])
    return dict(tips_qt=ca.vertcat(ql, qr), tips=tips, contact_tips=contacts,
                tips_fn=ca.Function("oracle_tips", [ql, qr], [tips]),
                contact_tips_fn=ca.Function("oracle_contacts", [ql, qr], [contacts]), n_arm=7)


class RecordingSolver:
    def __init__(self, ocp, status=0):
        ocp.make_consistent()
        self.ocp, self.status = ocp, status
        self.values, self.constraints, self.weights = {}, {}, {}
        self.before_solve = []
        self.raise_on_solve = False
        self.corrupt_on_failure = False
        self.reset()

    def reset(self):
        for i in range(self.ocp.solver_options.N_horizon+1):
            # 故意留下无效初值，防止漏写终端节点被零初值掩盖。
            self.values[i, "x"] = np.full(28, np.nan)
            self.values[i, "u"] = np.full(14, np.nan)

    def set(self, i, key, value):
        self.values[i, key] = np.asarray(value).copy()

    def get(self, i, key):
        return self.values[i, key].copy()

    def constraints_set(self, i, key, value):
        self.constraints[i, key] = np.asarray(value).copy()

    def cost_set(self, i, key, value):
        self.weights[i, key] = np.asarray(value).copy()

    def solve_for_x0(self, x0_bar, **kwargs):
        self.before_solve.append({key: value.copy() for key, value in self.values.items()})
        self.set(0, "x", x0_bar)
        if self.status and self.corrupt_on_failure:
            for i in range(self.ocp.solver_options.N_horizon+1):
                self.set(i, "x", np.full(28, 99.))
        if self.raise_on_solve:
            raise RuntimeError("injected solver failure")

    def get_stats(self, key):
        return np.array([.001])


def original_module(variant):
    name = "nmpcc_cart.py" if variant == "flat" else "nmpcc_cart_grasp_pose.py"
    path = ROOT / "openarmx_mujoco/_archive_20261009/refactor_before" / name
    if not path.is_file():
        raise FileNotFoundError(f"independent pre-refactor oracle is required: {path}")
    spec = importlib.util.spec_from_file_location("independent_"+variant, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SharedCoreTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ControllerConfig(N=3, kx=100., rest_gap=.5, Q=3., P=2.)
        self.solvers, self.factory_calls = [], []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = JointState(np.linspace(-.2, .2, 14).reshape(2, 7), np.linspace(-.03, .04, 14).reshape(2, 7))
        contact = ContactMeasurement(np.array([10., 11.]), np.array([[-9., 1., 2.], [12., -2., 3.]]),
                                     np.arange(6).reshape(2, 3)*.01, np.ones((2, 6))*999.,
                                     np.zeros((2, 3)), np.linspace(-3., 4., 14).reshape(2, 7),
                                     np.linspace(-1., 2., 14).reshape(2, 7))
        self.dyn = DynamicsSnapshot(np.stack((np.eye(7)*2+.03, np.eye(7)*3+.02)),
                                    np.arange(14).reshape(2, 7)*.1, np.ones((2, 7))*.5, contact)
        arrays = [np.arange(48).reshape(4, 2, 6)*scale for scale in (.01, .02, .1, .03)]
        self.refs = ReferenceHorizon(np.linspace(0., .1, 4), *arrays)
        self.ctrl = self.controller(self.cfg)

    def factory(self, ocp, json_file, **kwargs):
        solver = RecordingSolver(ocp)
        self.solvers.append(solver)
        self.factory_calls.append(kwargs)
        return solver

    def controller(self, cfg):
        return Controller(cfg, toy_fk(), cache_root=self.tmp.name,
                          qmin=np.full(14, -2.), qmax=np.full(14, 2.), solver_factory=self.factory)

    def assert_close(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)

    def test_ideal_force_and_chosen_zero_boundary_derivative_are_finite(self):
        # 也允许 eps=0：理想分支必须完全不经过 sqrt(c*c+eps*eps)。
        cfg = replace(self.cfg, normal_force_law=NormalForceLaw.IDEAL_UNILATERAL, smooth_eps=0.)
        ctrl = self.controller(cfg)
        sym = ctrl.symbolic
        nominal = sym.nominal_force(sym.x)
        jac = ca.Function('ideal_force_jac', [sym.x], [ca.jacobian(nominal, sym.x)])
        for compression in (-.03, 0., .02):
            with self.subTest(compression=compression):
                x = np.zeros(28)
                x[14] = cfg.rest_gap-compression
                force = np.asarray(sym.nominal_force(x)).ravel()
                expected = np.zeros(12)
                expected[[0, 6]] = [-cfg.kx*max(compression, 0.), cfg.kx*max(compression, 0.)]
                self.assert_close(force, expected)
                actual_jac = np.asarray(jac(x))
                expected_jac = np.zeros((12, 28))
                if compression > 0:
                    expected_jac[0, [0, 1, 14, 15]] = -cfg.kx*np.array([1., .03, -1., -.04])
                    expected_jac[6] = -expected_jac[0]
                self.assert_close(actual_jac, expected_jac)
                self.assertTrue(np.isfinite(actual_jac).all())
                state = JointState(np.stack((x[:7], x[14:21])), self.state.dq)
                p = ctrl._prepare(state, self.dyn, self.refs, SolveOptions()).parameters[0]
                evaluate = ca.Function('ideal_all_jac', [sym.x, sym.u, sym.p],
                    [sym.dynamics, sym.residual, ca.jacobian(sym.dynamics, sym.x), ca.jacobian(sym.residual, sym.x)])
                for value in evaluate(state.vector(), np.zeros(14), p):
                    self.assertTrue(np.isfinite(np.asarray(value)).all())

    def test_ideal_force_rate_matches_nonlinear_geometry_derivative(self):
        cfg = replace(self.cfg, normal_force_law=NormalForceLaw.IDEAL_UNILATERAL, eta_f=np.ones((2, 6)))
        ctrl = self.controller(cfg)
        zero_refs = ReferenceHorizon(self.refs.node_times, *(np.zeros((4, 2, 6)) for _ in range(4)))
        p = encode(self.dyn, zero_refs, np.zeros((2, 6)), np.zeros((2, 7)), cfg)[0]
        for compression in (-.03, .02):
            x = self.state.vector()
            x[14] = cfg.rest_gap+x[0]+.03*np.sin(x[1])-.04*np.sin(x[15])-compression
            # 独立解析几何方向导数，不调用模型内部的压缩或力率表达式。
            dc = x[7]+.03*np.cos(x[1])*x[8]-x[21]-.04*np.cos(x[15])*x[22]
            expected_rate = np.zeros(12)
            expected_rate[[0, 6]] = np.array([-1., 1.])*cfg.kx*dc*(compression > 0)
            error_rate = np.asarray(ctrl.symbolic.diagnostic_fn(x, p)[7]).ravel()
            self.assert_close(error_rate, -cfg.kf_scale*expected_rate)
            direction = np.zeros(28)
            direction[:7], direction[14:21] = x[7:14], x[21:28]
            h = 1e-5
            finite_rate = (np.asarray(ctrl.symbolic.nominal_force(x+h*direction)).ravel()
                           -np.asarray(ctrl.symbolic.nominal_force(x-h*direction)).ravel())/(2*h)
            np.testing.assert_allclose(finite_rate, expected_rate, rtol=1e-8, atol=1e-8)
        # 在精确折点，即使法向闭合速度非零，所选力率仍为零。
        x = np.zeros(28)
        x[14], x[7] = cfg.rest_gap, .1
        self.assert_close(ctrl.symbolic.diagnostic_fn(x, p)[7], np.zeros((12, 1)))

    def test_both_force_laws_preserve_current_anchoring_and_full_load(self):
        for law in NormalForceLaw:
            cfg = replace(self.cfg, normal_force_law=law,
                          force_feedback_source=ForceFeedbackSource.WORLD_FORCE_X,
                          bias_policy=BiasPolicy.LEGACY_SIX_AXIS)
            ctrl = self.controller(cfg)
            for compression in (-.03, 0., .02):
                q = np.zeros((2, 7))
                q[1, 0] = cfg.rest_gap-compression
                state = JointState(q, self.state.dq)
                prepared = ctrl._prepare(state, self.dyn, self.refs, SolveOptions())
                self.assert_close(prepared.anchoring['anchored'], prepared.anchoring['measured'])
                self.assert_close(prepared.anchoring['anchored'][:, 0], self.dyn.contact.force_world[:, 0])
                self.assert_close(prepared.contact_load['model_normal']+prepared.contact_load['frozen_residual'], self.dyn.contact.joint_load)
                u = np.linspace(-1., 1., 14).reshape(2, 7)
                expected = np.stack([np.linalg.solve(self.dyn.mass[i], u[i]+self.dyn.contact.joint_load[i]
                    -self.dyn.gravity[i]-self.dyn.damping[i]*state.dq[i]) for i in range(2)])
                self.assert_close(ctrl.acceleration(state, self.dyn, self.refs, u.ravel()), expected)

    def test_saved_old_config_defaults_smooth_and_legacy_layout_is_unchanged(self):
        saved = asdict(self.cfg)
        saved.pop('normal_force_law')
        restored = ControllerConfig(**saved)
        self.assertEqual(restored.normal_force_law, NormalForceLaw.SMOOTH_POSITIVE)
        ctrl = self.controller(restored)
        x = self.state.vector()
        np.testing.assert_array_equal(ctrl.symbolic.nominal_force(x), self.ctrl.symbolic.nominal_force(x))
        for law in (NormalForceLaw.SMOOTH_POSITIVE, NormalForceLaw.IDEAL_UNILATERAL):
            cfg = replace(restored, normal_force_law=law)
            p = self.controller(cfg)._prepare(self.state, self.dyn, self.refs, SolveOptions()).parameters
            self.assertEqual(p.shape, (4, 225))
            for variant, width in [('flat', 191), ('handheld', 201)]:
                legacy = to_legacy(p, variant)
                self.assertEqual(legacy.shape, (4, width))
                self.assert_close(from_legacy(legacy, variant, cfg), p)

    def test_force_law_is_serializable_validated_and_isolates_cache(self):
        smooth = self.controller(replace(self.cfg, normal_force_law='smooth_positive'))
        ideal = self.controller(replace(self.cfg, normal_force_law='ideal_unilateral'))
        self.assertNotEqual(smooth.name, ideal.name)
        self.assertNotEqual(smooth._json_file.parent, ideal._json_file.parent)
        self.assertIn('shared_v3', ideal.name)
        self.assertEqual(ideal.metadata['config']['normal_force_law'], 'ideal_unilateral')
        self.assert_close(smooth.metadata['W'], ideal.metadata['W'])
        self.assert_close(smooth.metadata['W_e'], ideal.metadata['W_e'])
        with self.assertRaises(ValueError):
            replace(self.cfg, normal_force_law='unknown')

    def test_measured_anchor_total_force_has_independent_clipping_and_no_rest_gap_dependence(self):
        cfg = replace(self.cfg, normal_force_law=NormalForceLaw.MEASURED_ANCHORED_IDEAL)
        ctrl = self.controller(cfg)
        prepared = ctrl._prepare(self.state, self.dyn, self.refs, SolveOptions())
        x0, p = prepared.state, prepared.parameters[0]
        anchor = prepared.anchoring
        self.assert_close(anchor['anchor_normal'], [10., 11.])
        self.assert_close(anchor['anchored'], anchor['measured'])
        self.assert_close(prepared.parameters[:, 225:], np.tile(np.r_[anchor['anchor_gap'], anchor['anchor_normal']], (4, 1)))
        # Opening by 0.105 m unloads the left side and leaves 0.5 N on the right.
        for opening in (-.03, 0., .105, .12):
            x = x0.copy()
            x[14] += opening
            expected = np.maximum(np.array([10., 11.])-cfg.kx*opening, 0.)*[-1., 1.]
            self.assert_close(np.asarray(ctrl.symbolic.predicted_force(x, p)).ravel()[[0, 6]], expected)
            self.assert_close(np.asarray(ctrl.symbolic.diagnostic_fn(x, p)[3]).ravel(), expected)
        other = self.controller(replace(cfg, rest_gap=cfg.rest_gap+1.))
        other_p = other._prepare(self.state, self.dyn, self.refs, SolveOptions()).parameters[0]
        x = x0.copy()
        x[14] += .105
        self.assert_close(ctrl.symbolic.predicted_force(x, p), other.symbolic.predicted_force(x, other_p))

    def test_measured_anchor_zero_boundary_and_total_force_rate_jacobian(self):
        cfg = replace(self.cfg, normal_force_law=NormalForceLaw.MEASURED_ANCHORED_IDEAL,
                      eta_f=np.ones((2, 6)), kx=128.)
        ctrl = self.controller(cfg)
        refs = ReferenceHorizon(self.refs.node_times, *(np.zeros((4, 2, 6)) for _ in range(4)))
        p = encode(self.dyn, refs, np.zeros((2, 6)), np.zeros((2, 7)), cfg,
                   anchor_gap=.5, anchor_normal=np.array([8., 16.]))[0]
        sym = ctrl.symbolic
        force = sym.predicted_force(sym.x, sym.p)
        jac = ca.Function('anchored_jac', [sym.x, sym.p], [ca.jacobian(force, sym.x)])
        model_jac = ca.Function('anchored_model_jac', [sym.x, sym.u, sym.p],
            [ca.jacobian(sym.dynamics, sym.x), ca.jacobian(sym.residual, sym.x)])
        for gap, active in ((.5, [True, True]), (.5625, [False, True]), (.625, [False, False]), (.75, [False, False])):
            x = np.zeros(28)
            x[14], x[7], x[8], x[21], x[22] = gap, .02, .03, -.01, -.04
            dg = x[21]+.04*x[22]-x[7]-.03*x[8]
            expected_rate = np.zeros(12)
            expected_rate[[0, 6]] = np.array([-1., 1.])*(-cfg.kx*dg)*active
            self.assert_close(sym.force_rate_fn(x, p), expected_rate[:, None])
            self.assert_close(sym.diagnostic_fn(x, p)[7], -cfg.kf_scale*expected_rate[:, None])
            for derivative in model_jac(x, np.zeros(14), p):
                self.assertTrue(np.isfinite(np.asarray(derivative)).all())
            expected_jac = np.zeros((12, 28))
            for side, row in enumerate((0, 6)):
                if active[side]:
                    expected_jac[row, [0, 1, 14, 15]] = [-1., 1.][side]*cfg.kx*np.array([1., .03, -1., -.04])
            self.assert_close(jac(x, p), expected_jac)
        # Zero measured force at its anchor really is zero with the chosen zero derivative.
        p[226:228] = 0.
        x = np.zeros(28)
        x[14] = .5
        self.assert_close(sym.predicted_force(x, p), np.zeros((12, 1)))
        self.assert_close(jac(x, p), np.zeros((12, 28)))
        # Nonlinear geometry, away from either kink: derivative of the actual total force.
        x = self.state.vector()
        p[225], p[226:228] = float(sym.contact_gap(x)), [8., 16.]
        direction = np.zeros(28)
        direction[:7], direction[14:21] = x[7:14], x[21:28]
        h = 1e-5
        finite_rate = (np.asarray(sym.predicted_force(x+h*direction, p))
                       -np.asarray(sym.predicted_force(x-h*direction, p)))/(2*h)
        np.testing.assert_allclose(sym.diagnostic_fn(x, p)[7], -cfg.kf_scale*finite_rate, atol=1e-10, rtol=1e-8)

    def test_measured_anchor_rejects_wrong_sign_measurement_and_missing_parameters(self):
        cfg = replace(self.cfg, normal_force_law=NormalForceLaw.MEASURED_ANCHORED_IDEAL,
                      force_feedback_source=ForceFeedbackSource.WORLD_FORCE_X)
        ctrl = self.controller(cfg)
        force = self.dyn.contact.force_world.copy()
        force[0, 0] = .01
        bad = replace(self.dyn, contact=replace(self.dyn.contact, force_world=force))
        with self.assertRaisesRegex(ValueError, 'negative normal-force anchor'):
            ctrl._prepare(self.state, bad, self.refs, SolveOptions())
        force[0, 0] = 1e-12
        tiny = replace(self.dyn, contact=replace(self.dyn.contact, force_world=force))
        prepared = ctrl._prepare(self.state, tiny, self.refs, SolveOptions())
        self.assertEqual(prepared.anchoring['anchor_normal'][0], 0.)
        self.assertEqual(prepared.anchoring['measured'][0, 0], 1e-12)
        with self.assertRaisesRegex(ValueError, 'requires finite anchor_gap'):
            encode(self.dyn, self.refs, np.zeros((2, 6)), np.zeros((2, 7)), cfg)

    def test_measured_anchor_preserves_legacy_slots_and_requires_anchor_for_legacy_replay(self):
        cfg = replace(self.cfg, normal_force_law=NormalForceLaw.MEASURED_ANCHORED_IDEAL)
        ctrl = self.controller(cfg)
        prepared = ctrl._prepare(self.state, self.dyn, self.refs, SolveOptions())
        p = prepared.parameters
        self.assertEqual(p.shape, (4, 228))
        self.assertEqual(parameter_size(self.cfg), 225)
        self.assertNotEqual(layout_version(cfg), layout_version(self.cfg))
        old = encode(self.dyn, self.refs, prepared.anchoring['bias'],
                     prepared.contact_load['frozen_residual'], self.cfg)
        np.testing.assert_array_equal(p[:, :225], old)
        get = ca.Function('get_anchor_layout', [ctrl.symbolic.p],
                          [view(ctrl.symbolic.p).anchor_gap, view(ctrl.symbolic.p).anchor_normal])
        values = get(p[0])
        self.assert_close(values[0], p[0, 225])
        self.assert_close(np.asarray(values[1]).ravel(), p[0, 226:228])
        for variant in ('flat', 'handheld'):
            legacy = encode_legacy_horizon(self.state, self.dyn, self.refs, prepared.anchoring,
                                            prepared.contact_load, cfg, variant)
            self.assert_close(legacy, to_legacy(old, variant))
            self.assert_close(to_legacy_full(p, variant), to_legacy_full(old, variant))
            with self.assertRaisesRegex(ValueError, 'requires finite anchor_gap'):
                from_legacy(legacy, variant, cfg)
            self.assert_close(from_legacy(legacy, variant, cfg,
                anchor_gap=prepared.anchoring['anchor_gap'], anchor_normal=prepared.anchoring['anchor_normal']), p)

    def test_measured_anchor_six_axis_residual_seed_and_frozen_changed_state(self):
        cfg = replace(self.cfg, normal_force_law=NormalForceLaw.MEASURED_ANCHORED_IDEAL,
                      force_feedback_source=ForceFeedbackSource.WORLD_FORCE_X,
                      bias_policy=BiasPolicy.LEGACY_SIX_AXIS)
        ctrl = self.controller(cfg)
        options = SolveOptions(kf_scale=.017)
        solved = ctrl.solve(self.state, self.dyn, self.refs, options)
        self.assertEqual(solved.prediction.force_rates.shape, (4, 2, 6))
        self.assert_close(solved.anchoring['anchored'][:, 1:3], self.dyn.contact.force_world[:, 1:3])
        self.assert_close(solved.anchoring['anchored'][:, 3:], self.dyn.contact.contact_torque_world)
        self.assert_close(solved.contact_load['model_normal']+solved.contact_load['frozen_residual'], self.dyn.contact.joint_load)
        seed = ctrl.equilibrium_seed(self.state, self.dyn, self.refs, options)
        self.assert_close(ctrl.acceleration(self.state, self.dyn, self.refs, seed, options), np.zeros((2, 7)))
        q = self.state.q.copy()
        q[1, 0] += .11
        changed = JointState(q, self.state.dq)
        dynamics = replace(self.dyn, mass=self.dyn.mass*.73)
        torque = np.linspace(-2., 2., 14)
        parameters = encode_shared_horizon(changed, dynamics, self.refs, solved.anchoring, solved.contact_load, cfg, options)
        self.assert_close(parameters[:, 225:], np.tile(np.r_[solved.anchoring['anchor_gap'], solved.anchoring['anchor_normal']], (4, 1)))
        with patch.object(ctrl, '_prepare', side_effect=AssertionError('must not reanchor')):
            actual = ctrl.acceleration(changed, dynamics, self.refs, torque, options, frozen_solution=solved)
        derivative = np.asarray(ctrl.symbolic.dynamics_fn(changed.vector(), torque, parameters[0])).ravel()
        self.assert_close(actual, np.stack((derivative[7:14], derivative[21:28])))
        predicted = np.asarray(ctrl.symbolic.predicted_force(changed.vector(), parameters[0])).reshape(2, 6)
        self.assert_close(predicted[:, 1:], solved.anchoring['measured'][:, 1:])
        self.assert_close(predicted[:, 0], [0., 1.])

    def test_measured_anchor_cache_constraints_weights_and_failure_rebuild(self):
        cfg = replace(self.cfg, normal_force_law=NormalForceLaw.MEASURED_ANCHORED_IDEAL)
        ctrl = self.controller(cfg)
        self.assertNotEqual(ctrl.name, self.ctrl.name)
        self.assertNotEqual(ctrl.name, self.controller(replace(cfg, normal_force_law=NormalForceLaw.IDEAL_UNILATERAL)).name)
        self.assertEqual(ctrl.metadata['parameter_size'], 228)
        self.assert_close(ctrl.metadata['W'], self.ctrl.metadata['W'])
        self.assert_close(ctrl.metadata['W_e'], self.ctrl.metadata['W_e'])
        for field in ('idxbx', 'idxbx_e', 'lbx', 'ubx', 'lbu', 'ubu', 'lh', 'uh'):
            self.assert_close(getattr(ctrl._ocp.constraints, field), getattr(self.ctrl._ocp.constraints, field))
        failed = ctrl._solver
        failed.status = 4
        solved = ctrl.solve(self.state, self.dyn, self.refs, SolveOptions(u_lim=3., u_prev=np.zeros(14), du_max=.2))
        self.assertEqual(solved.status, 0)
        self.assertEqual(solved.failure['p_full'].shape, (4, 228))
        self.assert_close(ctrl._solver.constraints[0, 'lbu'], np.full(14, -.2))
        self.assert_close(ctrl._solver.constraints[0, 'ubu'], np.full(14, .2))
        for node in range(4):
            self.assert_close(ctrl._solver.values[node, 'p'], failed.values[node, 'p'])

    def test_original_oracle_dynamics_residual_jacobian_constraints_and_weights(self):
        for variant in ("flat", "handheld"):
            cfg = replace(self.cfg, force_feedback_source=ForceFeedbackSource.WORLD_FORCE_X,
                          bias_policy=BiasPolicy.LEGACY_SIX_AXIS) if variant == "handheld" else self.cfg
            cfg = replace(cfg, eta_cop=np.linspace(.05, .6, 12).reshape(2, 6),
                          eta_f=np.linspace(.1, .8, 12).reshape(2, 6))
            ctrl = self.controller(cfg)
            module = original_module(variant)
            args = {key: getattr(cfg, key) for key in inspect.signature(module.NMPCCCartConfig).parameters
                    if hasattr(cfg, key)}
            with patch.object(module, "AcadosOcpSolver", side_effect=lambda ocp, **kw: RecordingSolver(ocp)):
                old = module.NMPCCCart(module.NMPCCCartConfig(**args), toy_fk(),
                                       qmin=np.full(14, -2.), qmax=np.full(14, 2.))
            prepared = ctrl._prepare(self.state, self.dyn, self.refs, SolveOptions())
            legacy = to_legacy(prepared.parameters, variant)
            x, u = self.state.vector(), np.linspace(-2., 2., 14)
            old_symbol, new_symbol = old._ocp.model, ctrl.symbolic
            old_residual = ca.Function("old_residual", [old_symbol.x, old_symbol.u, old_symbol.p], [old_symbol.cost_y_expr])
            old_jac = ca.Function("old_jac", [old_symbol.x, old_symbol.u, old_symbol.p],
                                   [ca.jacobian(old_symbol.f_expl_expr, old_symbol.x),
                                    ca.jacobian(old_symbol.cost_y_expr, old_symbol.x), old_symbol.con_h_expr,
                                    ca.jacobian(old_symbol.f_expl_expr, old_symbol.u),
                                    ca.jacobian(old_symbol.cost_y_expr, old_symbol.u)])
            new_jac = ca.Function("new_jac", [new_symbol.x, new_symbol.u, new_symbol.p],
                                   [ca.jacobian(new_symbol.dynamics, new_symbol.x),
                                    ca.jacobian(new_symbol.residual, new_symbol.x), new_symbol.constraints,
                                    ca.jacobian(new_symbol.dynamics, new_symbol.u),
                                    ca.jacobian(new_symbol.residual, new_symbol.u)])
            for node in range(4):
                p, old_p = prepared.parameters[node], old._full_p(legacy[node])
                self.assert_close(new_symbol.dynamics_fn(x, u, p), old._dynamics_fn(x, u, legacy[node]))
                self.assert_close(new_symbol.residual_fn(x, u, p), old_residual(x, u, old_p))
                for a, b in zip(new_jac(x, u, p), old_jac(x, u, old_p)):
                    self.assert_close(a, b)
            self.assert_close(ctrl.metadata["W"], old.W)
            self.assert_close(ctrl.metadata["W_e"], old.W_e)
            for field in ("idxbx", "idxbx_e", "lbx", "ubx", "lbx_e", "ubx_e", "lbu", "ubu", "lh", "uh"):
                self.assert_close(getattr(ctrl._ocp.constraints, field), getattr(old._ocp.constraints, field))
            # 旧调用虽传入重力/预热力矩，u_ref 实际从未进入代价。
            for torque_reference in (np.zeros(14), np.full(14, 123.)):
                old.solve(x, legacy, u_ref=torque_reference, u_seed=u)
                for node in range(4):
                    self.assert_close(old.solver.values[node, "yref"], np.zeros(24 if node == 3 else 38))

    def test_anchoring_modes_and_residual_freezing(self):
        for mode in ContactLoadMode:
            ctrl = self.controller(replace(self.cfg, contact_load_mode=mode))
            result = ctrl.solve(self.state, self.dyn, self.refs)
            self.assert_close(result.anchoring["anchored"][:, 0], [-10., 11.])
            self.assert_close(result.anchoring["bias"][:, 1:], 0.)
            for node in range(4):
                self.assert_close(np.r_[*view(self.solvers[-1].values[node, "p"]).load], result.contact_load["frozen_residual"].ravel())
            if mode == ContactLoadMode.FULL_RESIDUAL:
                self.assert_close(result.contact_load["model_normal"]+result.contact_load["frozen_residual"], self.dyn.contact.joint_load)
                u = np.linspace(-1., 1., 14).reshape(2, 7)
                expected = np.stack([np.linalg.solve(self.dyn.mass[i], u[i]+self.dyn.contact.joint_load[i]-self.dyn.gravity[i]-self.dyn.damping[i]*self.state.dq[i]) for i in range(2)])
                self.assert_close(ctrl.acceleration(self.state, self.dyn, self.refs, u.ravel()), expected)
            else:
                self.assert_close(result.contact_load["frozen_residual"], self.dyn.contact.tangent_joint_load)
            seed = ctrl.equilibrium_seed(self.state, self.dyn, self.refs)
            self.assert_close(ctrl.acceleration(self.state, self.dyn, self.refs, seed), np.zeros((2, 7)))

    def test_acceleration_diagnostic_reuses_frozen_load_when_only_mass_changes(self):
        options = SolveOptions(kf_scale=.017, eta_f=np.full((2, 6), .3))
        solved = self.ctrl.solve(self.state, self.dyn, self.refs, options)
        parameters = np.array([self.solvers[-1].get(i, "p") for i in range(4)])
        saved_parameters = parameters.copy()
        torque = np.linspace(-2., 2., 14)
        old = replace(self.dyn, mass=self.dyn.mass*.73)
        # 对照只替换惯性，不允许重新锚定或重新冻结接触残差。
        with patch.object(self.ctrl, "_prepare", side_effect=AssertionError("diagnostic recomputed load")):
            for dynamics in (self.dyn, old):
                expected_parameters = parameters.copy()
                expected_parameters[:, :49] = dynamics.mass[0].ravel(order="F")
                expected_parameters[:, 56:105] = dynamics.mass[1].ravel(order="F")
                expected = np.asarray(self.ctrl.symbolic.dynamics_fn(
                    self.state.vector(), torque, expected_parameters[0])).ravel()
                actual = self.ctrl.acceleration(self.state, dynamics, self.refs, torque, options,
                                                frozen_solution=solved)
                np.testing.assert_array_equal(actual, np.stack((expected[7:14], expected[21:28])))
                logged = encode(dynamics, self.refs, solved.anchoring["bias"],
                                solved.contact_load["frozen_residual"], self.cfg, options)
                np.testing.assert_array_equal(logged, expected_parameters)
        np.testing.assert_array_equal(parameters, saved_parameters)
        np.testing.assert_array_equal(np.array([self.solvers[-1].get(i, "p") for i in range(4)]), saved_parameters)

    def test_cold_warm_reset_all_nodes_and_node_references(self):
        seed = np.linspace(-1., 1., 14)
        first = self.ctrl.solve(self.state, self.dyn, self.refs, SolveOptions(u_seed=seed))
        self.assertEqual(first.prediction.states.shape, (4, 28))
        for node in range(4):
            self.assert_close(first.prediction.states[node], self.state.vector())
            self.assert_close(np.r_[*view(self.solvers[-1].values[node, "p"]).poses], self.refs.poses[node].ravel())
        self.assert_close(first.prediction.controls, np.tile(seed, (3, 1)))
        changed = JointState(self.state.q+.1, self.state.dq)
        second = self.ctrl.solve(changed, self.dyn, self.refs, SolveOptions(u_seed=seed*2))
        self.assert_close(second.prediction.states[0], changed.vector())
        self.assert_close(second.prediction.states[1:], first.prediction.states[1:])
        self.assert_close(second.prediction.controls, first.prediction.controls)
        self.ctrl.reset()
        third = self.ctrl.solve(changed, self.dyn, self.refs, SolveOptions(u_seed=seed*2))
        self.assert_close(third.prediction.states, np.tile(changed.vector(), (4, 1)))
        self.assert_close(third.prediction.controls, np.tile(seed*2, (3, 1)))

    def test_failure_rebuild_restores_nodes_bounds_weights_and_scene(self):
        self.solvers[-1].status = 4
        options = SolveOptions(u_seed=np.ones(14), u_lim=3., u_prev=np.zeros(14), du_max=.2,
                               weights=RuntimeWeights(Q=5., P=7., ed_x=.3))
        result = self.ctrl.solve(self.state, self.dyn, self.refs, options)
        self.assertEqual((result.initial_status, result.status), (4, 0))
        self.assertTrue(self.factory_calls[-1]["rebuild"])
        self.assertEqual(result.failure["p_full"].shape, (4, 225))
        for node in range(4):
            self.assert_close(self.solvers[-1].values[node, "p"], result.failure["p_full"][node])
            self.assert_close(self.solvers[-1].before_solve[0][node, "x"], self.state.vector())
            self.assert_close(self.solvers[-1].values[node, "yref"], np.zeros(24 if node == 3 else 38))
            if node < 3:
                self.assert_close(self.solvers[-1].before_solve[0][node, "u"], options.u_seed)
                self.assert_close(self.solvers[-1].weights[node, "W"], cost_weights(self.cfg, options.weights)[0])
        self.assert_close(self.solvers[-1].constraints[0, "lbu"], np.full(14, -.2))
        self.assert_close(self.solvers[-1].constraints[1, "lbu"], np.full(14, -3.))
        self.assert_close(self.solvers[-1].weights[3, "W"], cost_weights(self.cfg, options.weights)[1])
        self.ctrl.solve(self.state, self.dyn, self.refs)
        self.assert_close(self.solvers[-1].weights[3, "W"], cost_weights(self.cfg)[1])
        self.assert_close(self.solvers[-1].constraints[0, "lbu"], np.full(14, -8.))

    def test_failure_rebuild_uses_last_warm_iterate_not_failed_iterate(self):
        first = self.ctrl.solve(self.state, self.dyn, self.refs, SolveOptions(u_seed=np.ones(14)))
        failed_solver = self.solvers[-1]
        failed_solver.status = 2
        failed_solver.raise_on_solve = True
        failed_solver.corrupt_on_failure = True
        result = self.ctrl.solve(self.state, self.dyn, self.refs, SolveOptions(u_seed=np.full(14, 7.)))
        self.assertEqual((result.initial_status, result.status), (4, 0))
        self.assert_close(result.failure["x"], np.full((4, 28), 99.))
        rebuilt = self.solvers[-1].before_solve[0]
        for node in range(4):
            self.assert_close(rebuilt[node, "x"], first.prediction.states[node])
            if node < 3:
                self.assert_close(rebuilt[node, "u"], first.prediction.controls[node])
        self.ctrl.reset()
        self.assertIsNone(self.ctrl.solve(self.state, self.dyn, self.refs).failure)

    def test_recreation_exception_exposes_current_and_first_failure_copies(self):
        self.solvers[-1].status = 2
        recovered = self.ctrl.solve(self.state, self.dyn, self.refs, SolveOptions(u_seed=np.ones(14)))
        first = {key: value.copy() if isinstance(value, np.ndarray) else value
                 for key, value in recovered.failure.items()}
        changed = JointState(self.state.q+.1, self.state.dq-.02)
        references = replace(self.refs, forces=self.refs.forces+3.)
        broken = self.solvers[-1]
        broken.status = 3
        cause = OSError("injected DLL load failure")
        with patch.object(self.ctrl, "_solver_factory", side_effect=cause):
            with self.assertRaises(SolverError) as captured:
                self.ctrl.solve(changed, self.dyn, references)
        error = captured.exception
        self.assertIs(error.__cause__, cause)
        failure = error.failure
        self.assertEqual(failure["status"], 3)
        self.assertEqual(failure["first_status"], 2)
        self.assert_close(failure["x0"], changed.vector())
        self.assert_close(failure["first_x0"], self.state.vector())
        self.assert_close(failure["pred_x"][0], changed.vector())
        self.assert_close(failure["p_full"], np.array([broken.get(i, "p") for i in range(4)]))
        self.assert_close(failure["pred_u"], np.array([broken.get(i, "u") for i in range(3)]))
        self.assertEqual(failure["p_full"].shape, (4, 225))
        for key, value in first.items():
            self.assert_close(failure["first_"+key], value)
        self.assertFalse(np.array_equal(failure["p_full"], failure["first_p_full"]))
        saved_current = failure["pred_x"].copy()
        broken.set(0, "x", np.zeros(28))
        recovered.failure["x0"][:] = 999.
        self.assert_close(failure["pred_x"], saved_current)
        self.assert_close(failure["first_x0"], first["x0"])

    def test_full_failure_codec_preserves_eta_after_variant_physical_layout(self):
        full = np.arange(4*225, dtype=float).reshape(4, 225)
        for variant, physical, width in (("flat", 191, 215), ("handheld", 201, 225)):
            encoded = to_legacy_full(full, variant)
            self.assertEqual(encoded.shape, (4, width))
            self.assert_close(encoded[:, :physical], to_legacy(full, variant))
            self.assert_close(encoded[:, physical:], full[:, 201:])
            self.assert_close(to_legacy_full(full[0], variant), encoded[0])
            self.assertFalse(np.shares_memory(full, encoded))
        for bad in (np.zeros(201), np.zeros((2, 215)), np.array(0.)):
            with self.assertRaises(ValueError):
                to_legacy_full(bad, "flat")
        with self.assertRaises(ValueError):
            to_legacy_full(full, "other")

    def test_process_and_terminal_weights_are_independent(self):
        process, terminal = cost_weights(self.cfg)
        process_q, terminal_q = cost_weights(self.cfg, RuntimeWeights(Q=11.))
        process_p, terminal_p = cost_weights(self.cfg, RuntimeWeights(P=13.))
        self.assert_close(process_q, process*11./self.cfg.Q)
        self.assert_close(terminal_q, terminal)
        self.assert_close(process_p, process)
        self.assert_close(terminal_p, terminal*13./self.cfg.P)
        override = RuntimeWeights(Q=5., P=7., eta_dev=.3, eta_u=.4, ed_x=.9)
        process, terminal = cost_weights(self.cfg, override)
        derivative = np.full(12, .3)
        derivative[[0, 6]] = .9
        self.assert_close(np.diag(process), 5.*np.r_[np.ones(12), derivative, np.full(14, .4)])
        self.assert_close(np.diag(terminal), 7.*np.r_[np.ones(12), derivative])
        result = self.ctrl.solve(self.state, self.dyn, self.refs, SolveOptions(weights=override))
        self.assertEqual(result.status, 0)
        self.ctrl.reset()
        self.ctrl.solve(self.state, self.dyn, self.refs)
        self.assert_close(self.solvers[-1].weights[0, "W"], cost_weights(self.cfg)[0])
        self.assert_close(self.solvers[-1].weights[3, "W"], cost_weights(self.cfg)[1])

    def test_codec_column_major_and_legacy_roundtrip(self):
        nonsymmetric = replace(self.dyn, mass=np.arange(98).reshape(2, 7, 7)+1.)
        p = encode(nonsymmetric, self.refs, np.zeros((2, 6)), np.zeros((2, 7)), self.cfg)
        self.assert_close(view(p[0]).mass[0], nonsymmetric.mass[0])
        self.assert_close(view(p[0]).mass[1], nonsymmetric.mass[1])
        for variant in ("flat", "handheld"):
            self.assert_close(from_legacy(to_legacy(p, variant), variant, self.cfg), p)

    def test_parameter_layout_all_225_slots_and_symbolic_view(self):
        bias, load = np.arange(12).reshape(2, 6)+100., np.arange(14).reshape(2, 7)+200.
        options = SolveOptions(kf_scale=.123, eta_cop=np.arange(12).reshape(2, 6)*.01,
                               eta_f=np.arange(12).reshape(2, 6)*.02)
        p = encode(self.dyn, self.refs, bias, load, self.cfg, options)
        self.assertEqual(p.shape, (4, 225))
        symbol = ca.SX.sym("parameter_layout", 225)
        symbolic_view = view(symbol)
        for node in range(4):
            expected = np.r_[self.dyn.mass[0].ravel(order="F"), self.dyn.gravity[0],
                             self.dyn.mass[1].ravel(order="F"), self.dyn.gravity[1], load.ravel(),
                             self.refs.poses[node].ravel(), self.refs.velocities[node].ravel(),
                             self.refs.forces[node].ravel(), self.dyn.damping.ravel(), bias.ravel(),
                             .123, self.refs.force_rates[node].ravel(), options.eta_cop.ravel(), options.eta_f.ravel()]
            self.assert_close(p[node], expected)
            numeric_view = view(p[node])
            for name in ("mass", "gravity", "load", "poses", "velocities", "forces", "damping", "bias", "force_rates", "eta_cop", "eta_f"):
                for arm in range(2):
                    fn = ca.Function("layout_"+name, [symbol], [getattr(symbolic_view, name)[arm]])
                    self.assert_close(np.asarray(fn(p[node])).ravel(), np.asarray(getattr(numeric_view, name)[arm]).ravel())
        self.assert_close(to_legacy(p, "handheld"), p[:, :201])
        flat = to_legacy(p, "flat")
        self.assertEqual(flat.shape, (4, 191))
        self.assert_close(flat[:, 176:178], bias[:, 0][None, :].repeat(4, axis=0))
        self.assert_close(flat[:, 178:], p[:, 188:201])

    def test_joint_and_runtime_torque_bounds(self):
        qmin, qmax = np.full(14, -np.inf), np.full(14, np.inf)
        qmin[[0, 6, 7, 13]], qmax[[0, 6, 7, 13]] = -1., 2.
        ctrl = Controller(self.cfg, toy_fk(), cache_root=self.tmp.name, qmin=qmin, qmax=qmax,
                          umin=np.full(14, -4.), umax=np.full(14, 6.), solver_factory=self.factory)
        for suffix in ("", "_e"):
            self.assert_close(getattr(ctrl._ocp.constraints, "idxbx"+suffix), [0, 6, 14, 20])
            self.assert_close(getattr(ctrl._ocp.constraints, "lbx"+suffix), np.full(4, -1.))
            self.assert_close(getattr(ctrl._ocp.constraints, "ubx"+suffix), np.full(4, 2.))
        ctrl.solve(self.state, self.dyn, self.refs, SolveOptions(u_lim=5., u_prev=np.full(14, 4.8), du_max=.5))
        solver = self.solvers[-1]
        self.assert_close(solver.constraints[0, "lbu"], np.full(14, 4.3))
        self.assert_close(solver.constraints[0, "ubu"], np.full(14, 5.))
        for node in (1, 2):
            self.assert_close(solver.constraints[node, "lbu"], np.full(14, -4.))
            self.assert_close(solver.constraints[node, "ubu"], np.full(14, 5.))
        ctrl.solve(self.state, self.dyn, self.refs)
        self.assert_close(solver.constraints[0, "lbu"], np.full(14, -4.))
        self.assert_close(solver.constraints[0, "ubu"], np.full(14, 6.))
        for options in (SolveOptions(u_lim=-1.), SolveOptions(u_lim=np.nan), SolveOptions(u_prev=np.zeros(14)),
                        SolveOptions(du_max=.1), SolveOptions(u_prev=np.zeros(14), du_max=-.1),
                        SolveOptions(u_prev=np.full(14, 10.), du_max=.1)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ctrl.solve(self.state, self.dyn, self.refs, options)
        for kwargs in ({"qmin": qmin}, {"qmin": np.zeros(14), "qmax": np.zeros(14)},
                       {"umin": np.full(14, np.nan)}, {"umax": np.ones(13)},
                       {"qmin": np.full(14, -np.inf), "qmax": np.ones(14)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Bounds.create(**kwargs)

    def test_cache_static_identity_runtime_exclusion(self):
        bounds = Bounds.create()
        identity = solver_identity(self.cfg, toy_fk(), bounds)
        self.assertIn("shared_v3", identity)
        self.assertEqual(identity, solver_identity(replace(self.cfg, kf_scale=3.), toy_fk(), bounds))
        for cfg in (replace(self.cfg, kx=101.), replace(self.cfg, contact_load_mode=ContactLoadMode.TANGENT),
                    replace(self.cfg, bias_policy=BiasPolicy.LEGACY_SIX_AXIS), replace(self.cfg, sim_method_num_steps=3)):
            self.assertNotEqual(identity, solver_identity(cfg, toy_fk(), bounds))
        self.assertNotEqual(identity, solver_identity(self.cfg, toy_fk(), Bounds.create(qmin=np.full(14, -1.), qmax=np.ones(14))))

    def test_solver_creation_and_failure_recreation_reuse_loaded_dll(self):
        for rebuild in (False, True):
            with self.subTest(rebuild=rebuild), patch("openarmx_mujoco.nmpcc.ocp.AcadosOcpSolver") as factory:
                result = create_solver(self.ctrl._ocp, self.ctrl._json_file, rebuild=rebuild)
                self.assertIs(result, factory.return_value)
                factory.assert_called_once_with(self.ctrl._ocp, json_file=str(self.ctrl._json_file),
                                                build=False, generate=False, check_reuse_possible=False, verbose=False)
            for error in (FileNotFoundError("missing cache"), OSError("unloadable cache")):
                with self.subTest(rebuild=rebuild, error=type(error).__name__):
                    replacement = object()
                    with patch("openarmx_mujoco.nmpcc.ocp.AcadosOcpSolver", side_effect=[error, replacement]) as factory:
                        self.assertIs(create_solver(self.ctrl._ocp, self.ctrl._json_file, rebuild=rebuild), replacement)
                    self.assertEqual(factory.call_count, 2)
                    self.assertFalse(factory.call_args_list[0].kwargs["build"])
                    self.assertFalse(factory.call_args_list[0].kwargs["generate"])
                    self.assertFalse(factory.call_args_list[0].kwargs["check_reuse_possible"])
                    self.assertTrue(factory.call_args_list[1].kwargs["build"])
                    self.assertTrue(factory.call_args_list[1].kwargs["generate"])

    def test_malformed_shapes_and_node_times_rejected(self):
        with self.assertRaises(ValueError):
            JointState(np.zeros(14), np.zeros((2, 7)))
        bad = replace(self.refs, node_times=self.refs.node_times+.01)
        with self.assertRaises(ValueError):
            self.ctrl.solve(self.state, self.dyn, bad)
        for kwargs in ({"N": 2.5}, {"N": True}, {"Tp": np.nan}, {"kx": np.inf}, {"Q": [1.]}):
            with self.subTest(config=kwargs), self.assertRaises(ValueError):
                ControllerConfig(**kwargs)
        for kwargs in ({"Q": np.nan}, {"P": np.inf}, {"ed_x": [1.]}):
            with self.subTest(weights=kwargs), self.assertRaises(ValueError):
                RuntimeWeights(**kwargs)
        for value in (self.state, self.dyn, self.dyn.contact):
            with self.subTest(data=type(value).__name__), self.assertRaises(ValueError):
                replace(value, time=np.nan)

    def test_real_mujoco_geometries_match_original_fk_and_contact_jacobians(self):
        random = np.random.default_rng(12)
        for variant, filename in (("flat", "exp2_slender_scene.xml"), ("handheld", "grasp_pose_scene.xml")):
            model = mujoco.MjModel.from_xml_path(str(ROOT / "openarmx_mujoco" / filename))
            data = mujoco.MjData(model)
            joint_ids = [model.joint(f"openarmx_{side}_joint{index}").id
                         for side in ("left", "right") for index in range(1, 8)]
            qpos_indices, dof_indices = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
            arms = [{"body_id": model.body(f"openarmx_{side}_link7").id,
                     "pad_id": model.geom(f"openarmx_{side}_ee_pad").id} for side in ("left", "right")]
            old_module = original_module(variant)
            offsets = pad_contact_offsets(model, arms, .1001)
            self.assert_close(offsets, old_module.pad_contact_offsets(model, arms, .1001))
            old_fk = old_module.mujoco_fk_symbols(model, arms, .1001, offsets)
            coordinates = ({"contact_offsets": offsets} if variant == "flat" else
                           {"contact_local_points": [np.array([0., 0., .1001])+offset for offset in offsets]})
            new_fk = mujoco_fk_symbols(model, arms, .1001, **coordinates)
            cfg = replace(self.cfg, force_feedback_source=ForceFeedbackSource.WORLD_FORCE_X,
                          bias_policy=BiasPolicy.LEGACY_SIX_AXIS) if variant == "handheld" else self.cfg
            ctrl = Controller(cfg, new_fk, cache_root=self.tmp.name, solver_factory=self.factory)
            args = {key: getattr(cfg, key) for key in inspect.signature(old_module.NMPCCCartConfig).parameters
                    if hasattr(cfg, key)}
            with patch.object(old_module, "AcadosOcpSolver", side_effect=lambda ocp, **kw: RecordingSolver(ocp)):
                old = old_module.NMPCCCart(old_module.NMPCCCartConfig(**args), old_fk)
            symbolic_pairs = ((old._ocp.model.x, old._ocp.model.u, old._ocp.model.p,
                               old._ocp.model.f_expl_expr, old._ocp.model.cost_y_expr),
                              (ctrl.symbolic.x, ctrl.symbolic.u, ctrl.symbolic.p,
                               ctrl.symbolic.dynamics, ctrl.symbolic.residual))
            evaluate = [ca.Function("geometry_model", [x, u, p], [f, y, ca.jacobian(f, x), ca.jacobian(y, x)])
                        for x, u, p, f, y in symbolic_pairs]
            old_j = ca.Function("old_fk_j", [old_fk["tips_qt"]], [ca.jacobian(old_fk["contact_tips"], old_fk["tips_qt"])])
            new_j = ca.Function("new_fk_j", [new_fk["tips_qt"]], [ca.jacobian(new_fk["contact_tips"], new_fk["tips_qt"])])
            for _ in range(6):
                q = random.uniform(-.6, .6, 14)
                data.qpos[qpos_indices] = q
                mujoco.mj_forward(model, data)
                np.testing.assert_array_equal(new_fk["tips_fn"](q[:7], q[7:]), old_fk["tips_fn"](q[:7], q[7:]))
                np.testing.assert_array_equal(new_fk["contact_tips_fn"](q[:7], q[7:]), old_fk["contact_tips_fn"](q[:7], q[7:]))
                np.testing.assert_array_equal(new_j(q), old_j(q))
                state = JointState(q.reshape(2, 7), self.state.dq)
                prepared = ctrl._prepare(state, self.dyn, self.refs, SolveOptions())
                tangent = encode(self.dyn, self.refs, np.zeros((2, 6)), self.dyn.contact.tangent_joint_load, cfg)
                measured = prepared.anchoring["measured"]
                old_p = old.contact_residual_parameters(state.vector(), to_legacy(tangent, variant),
                    self.dyn.contact.joint_load.ravel(), measured[:, 0] if variant == "flat" else measured.ravel())
                np.testing.assert_array_equal(to_legacy(prepared.parameters, variant), old_p)
                np.testing.assert_array_equal(ctrl.equilibrium_seed(state, self.dyn, self.refs),
                    old.equilibrium_seed(state.vector(), old_p,
                        measured[:, 0] if variant == "flat" else measured.ravel()))
                torque = random.uniform(-2., 2., 14)
                for actual, expected in zip(evaluate[1](state.vector(), torque, prepared.parameters[0]),
                                            evaluate[0](state.vector(), torque, old._full_p(old_p[0]))):
                    np.testing.assert_array_equal(actual, expected)
                poses, contacts, jacobians = [], [], []
                for arm in arms:
                    body, pad = arm["body_id"], arm["pad_id"]
                    rotation = data.xmat[body].reshape(3, 3)
                    tip = data.xpos[body]+rotation @ np.array([0., 0., .1001])
                    contact = data.geom_xpos[pad]+data.geom_xmat[pad].reshape(3, 3) @ np.array([0., 0., model.geom_size[pad, 2]])
                    poses.append(np.r_[tip, rotvec_of_R(rotation)])
                    contacts.append(contact)
                    jacobian = np.zeros((3, model.nv))
                    mujoco.mj_jac(model, data, jacobian, None, contact, body)
                    jacobians.append(jacobian[:, dof_indices])
                self.assert_close(np.asarray(new_fk["tips_fn"](q[:7], q[7:])).ravel(), np.concatenate(poses))
                self.assert_close(np.asarray(new_fk["contact_tips_fn"](q[:7], q[7:])).ravel(), np.concatenate(contacts))
                self.assert_close(new_j(q), np.vstack(jacobians))

            with self.assertRaises(ValueError):
                mujoco_fk_symbols(model, arms, .1001, offsets, contact_local_points=np.zeros((2, 3)))
            for bad in (np.zeros((2, 2)), np.full((2, 3), np.nan)):
                with self.assertRaises(ValueError):
                    mujoco_fk_symbols(model, arms, .1001, contact_local_points=bad)


if __name__ == "__main__":
    unittest.main()
