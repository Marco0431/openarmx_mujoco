"""Handheld migration regression; real MuJoCo scene, mocked acados build."""
import unittest
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch
import mujoco
import numpy as np
import casadi as ca
import nmpcc_cart as flat
import nmpcc_cart_grasp_pose as nc
import exp2_nmpcc_latest_grasp_pose as exp
import exp2_diagnostics as diag
from test_nmpcc_revision import RecordingSolver, toy_fk


class Solver(RecordingSolver):
    def cost_set(self, i, field, value):
        self.values[i, field] = np.asarray(value).copy()

    def reset(self):
        self.values.clear()
        for i in range(self.ocp.solver_options.N_horizon+1):
            self.values[i,'x']=np.zeros(28)
            self.values[i,'u']=np.zeros(14)
        self.status = 0


class HandheldTests(unittest.TestCase):
    def make(self, **config):
        cfg = nc.NMPCCCartConfig(N=3, Tp=.1, Q=3, P=2, kx=100,
            rest_gap=.5, kf_scale=.003, eta_dev=.05, eta_u=5e-8, **config)
        with patch.object(nc, 'AcadosOcpSolver', side_effect=lambda ocp, **kw: Solver(ocp)):
            return nc.NMPCCCart(cfg, toy_fk(), qmin=np.full(14,-2.), qmax=np.full(14,2.))

    def parameters(self):
        p=np.zeros(nc.NP)
        p[:49]=np.eye(7).ravel()
        p[56:105]=np.eye(7).ravel()
        p[nc.KF_PARAM_IDX]=.003
        return p

    def test_layout_and_legacy_bias_slots(self):
        c=self.make()
        self.assertTrue(c._ocp.model.name.startswith('nmpcc_grasp_v6_'))
        before=nc._solver_identity(c.cfg,toy_fk(),qmin=c.qmin,qmax=c.qmax)
        changed=nc._solver_identity(c.cfg,toy_fk(),qmin=c.qmin,qmax=c.qmax+.1)
        self.assertNotEqual(before,changed)
        self.assertEqual((nc.KF_PARAM_IDX,nc.FREF_DOT_IDX,nc.NP),(188,189,201))
        for size in (176,188,189,201):
            p=np.arange(size,dtype=float)/100
            q=c._normalize_p(p)
            np.testing.assert_array_equal(q[:size],p)
            self.assertEqual(q.shape,(201,))
            if size<=188:self.assertEqual(q[188],.003)
        with self.assertRaises(ValueError):c._normalize_p(np.zeros(191))
        with self.assertRaises(ValueError):c._normalize_p(np.zeros((3,201)))

    def test_independent_build_and_runtime_weights(self):
        c=self.make()
        np.testing.assert_array_equal(np.diag(c.W)[:12],np.full(12,3))
        np.testing.assert_array_equal(np.diag(c.W_e)[:12],np.full(12,2))
        c.set_runtime_weights(Q=7, P=2, ed_x=.1)
        terminal=c._We_rt.copy()
        c.set_runtime_weights(Q=11, P=2, ed_x=.1)
        np.testing.assert_array_equal(c._We_rt,terminal)
        stage=c._W_rt.copy()
        c.set_runtime_weights(Q=11, P=5, ed_x=.1)
        np.testing.assert_array_equal(c._W_rt,stage)
        self.assertEqual(c._We_rt[12,12],.5)

    def test_same_force_cost_and_dynamics_as_flat_controller(self):
        settings=dict(N=3,Tp=.1,Q=1,P=2,kx=2419.3,rest_gap=.24,
                      kf_scale=.003,eta_dev=.05,eta_u=5e-8)
        fk=toy_fk()
        fk['contact_tips']=ca.vertcat(fk['tips'][:3],fk['tips'][6:9])
        with patch.object(nc,'AcadosOcpSolver',side_effect=lambda ocp,**kw:Solver(ocp)), \
             patch.object(flat,'AcadosOcpSolver',side_effect=lambda ocp,**kw:Solver(ocp)):
            c=nc.NMPCCCart(nc.NMPCCCartConfig(**settings),fk)
            f=flat.NMPCCCart(flat.NMPCCCartConfig(**settings),fk)
        x=np.linspace(-.02,.03,28);u=np.linspace(-1,1,14)
        p=self.parameters();p[49:56]=np.linspace(0,1,7);p[105:112]=np.linspace(-1,0,7)
        p[112:176]=np.linspace(-.01,.02,64)
        p[nc.FREF_DOT_IDX:]=np.linspace(-.1,.1,12)
        pf=np.zeros(flat.NP);pf[:176]=p[:176];pf[flat.KF_PARAM_IDX]=.003
        pf[flat.FREF_DOT_IDX:]=p[nc.FREF_DOT_IDX:]
        p=c._anchor_measured_force(x,p,np.array([-10,0,0,0,0,0,11,0,0,0,0,0]))
        pf=f._anchor_measured_force(x,pf,[-10,11])
        fy=ca.Function('flat_transfer_cost',[f._ocp.model.x,f._ocp.model.u,f._ocp.model.p],
                       [f._ocp.model.cost_y_expr])
        np.testing.assert_allclose(c._y_fn(x,u,c._full_p(p)),fy(x,u,f._full_p(pf)),atol=1e-11)
        np.testing.assert_allclose(c._dynamics_fn(x,u,p),f._dynamics_fn(x,u,pf),atol=1e-11)
        np.testing.assert_array_equal(c.W,f.W)
        np.testing.assert_array_equal(c.W_e,f.W_e)

    def test_tangent_loads_enter_dynamics_and_reference_force_derivative(self):
        c=self.make(); x=np.zeros(28); p=self.parameters(); u=np.zeros(14)
        q=p.copy(); q[112:126]=np.arange(14)*.01
        dv=np.asarray(c._dynamics_fn(x,u,q)-c._dynamics_fn(x,u,p)).ravel()[np.r_[7:14,21:28]]
        np.testing.assert_allclose(dv,q[112:126],atol=1e-12)
        nominal=np.asarray(c._force_nom_fn(x)).ravel()
        anchored=c._anchor_measured_force(x,p,nominal)
        base=c._diagnostic_fn(x,anchored,c.Ic1,c.Ic2,c.If1,c.If2)
        anchored[nc.FREF_DOT_IDX]=2
        changed=c._diagnostic_fn(x,anchored,c.Ic1,c.Ic2,c.If1,c.If2)
        delta=np.asarray(changed[7]-base[7]).ravel()
        self.assertAlmostEqual(delta[0],.8*.003*2,places=12)

    def test_full_initialization_anchor_nodes_and_failure_rebuild(self):
        c=self.make(); x=np.arange(28)/100; p=np.tile(self.parameters(),(4,1))
        p[:,126]=np.arange(4)*.01
        f=np.arange(12)/100
        c.set_runtime_weights(Q=4, P=5)
        c.solve(x,p,f_meas=f,u_seed=np.zeros(14),u_prev=np.ones(14),du_max=.2,u_lim=3)
        self.assertEqual(c._warm[0].shape,(4,28))
        np.testing.assert_array_equal(c.solver.values[3,'x'],x)
        np.testing.assert_allclose(c.last_force_anchored,f,atol=1e-12)
        np.testing.assert_array_equal(c.last_parameters[:,176:188],np.tile(c.last_force_bias,(4,1)))
        np.testing.assert_array_equal(c.last_parameters[:,126],p[:,126])
        c.solver.status=4
        with patch.object(nc,'AcadosOcpSolver',side_effect=lambda ocp,**kw:Solver(ocp)):
            c.solve(x,p,f_meas=f,u_prev=np.ones(14),du_max=.2,u_lim=3)
        self.assertEqual(c.first_failure['status'],4)
        np.testing.assert_array_equal(c.solver.values[3,'x'],x)
        np.testing.assert_array_equal(c.solver.constraints[0,'lbu'],np.full(14,.8))
        np.testing.assert_array_equal(c.solver.constraints[0,'ubu'],np.full(14,1.2))
        np.testing.assert_array_equal(c.solver.constraints[1,'lbu'],np.full(14,-3))
        np.testing.assert_array_equal(c.solver.values[3,'W'],c._We_rt)
        np.testing.assert_array_equal(c._ocp.constraints.idxbx_e,np.r_[0:7,14:21])

    def test_residual_equilibrium_and_future_normal_load(self):
        c=self.make(); x=np.zeros(28); p=self.parameters()
        f=np.array([-10,1,2,0,0,0,10,-1,2,0,0,0],dtype=float)
        full=np.linspace(-1,1,14)
        corrected=c.contact_residual_parameters(x,p,full,f)
        dd=np.asarray(c._dynamics_fn(x,np.zeros(14),corrected)).ravel()[np.r_[7:14,21:28]]
        np.testing.assert_allclose(dd,full,atol=1e-10)
        seed=c.equilibrium_seed(x,corrected,f)
        np.testing.assert_allclose(seed,-full,atol=1e-10)
        changed=x.copy();changed[0]+=.01
        dd2=np.asarray(c._dynamics_fn(changed,np.zeros(14),corrected)).ravel()[np.r_[7:14,21:28]]
        self.assertGreater(np.max(np.abs(dd2-dd)),1e-4)

    def test_scene_mapping_sampling_and_motor_wrapper(self):
        original_bytes=exp.SCENE_XML.read_bytes()
        original=mujoco.MjModel.from_xml_path(str(exp.SCENE_XML))
        m=mujoco.MjModel.from_xml_path(str(exp.build_scene()))
        for key in ('body_mass','body_pos','body_quat','geom_size','geom_friction','jnt_range'):
            np.testing.assert_array_equal(getattr(original,key),getattr(m,key))
        d=mujoco.MjData(m); arms=exp.make_arms(m)
        self.assertTrue(m.joint(arms[0]['joint_ids'][0]).name.startswith('openarmx_right'))
        b=m.geom('box').id
        exp.disable_robot_collisions(m,arms,b);exp.soften_contacts(m,b,[a['pad_id'] for a in arms])
        exp.reset_world(m,d,arms,int(m.jnt_qposadr[m.joint('box_freejoint').id]))
        targets=exp.grasp_targets(arms)
        for a in arms: targets[a['side']][0]-=a['sign']*10/(2*exp.K_STIFF)
        exp.solve_end_grip(m,d,arms,targets)
        fk=nc.mujoco_fk_symbols(m,[dict(body_id=a['body_id'],joint_ids=a['joint_ids']) for a in arms],.1001)
        tips=np.asarray(fk['tips_fn'](*[d.qpos[a['qpos']] for a in arms])).ravel()
        for i,a in enumerate(arms):
            np.testing.assert_allclose(tips[6*i:6*i+3],exp.get_ee_tip(d,a['body_id']),atol=1e-10)
        offsets=nc.pad_contact_offsets(m,arms,.1001)
        face_fk=nc.mujoco_fk_symbols(m,[dict(body_id=a['body_id'],joint_ids=a['joint_ids']) for a in arms],
                                    .1001,contact_offsets=offsets)
        faces=np.asarray(face_fk['contact_tips_fn'](*[d.qpos[a['qpos']] for a in arms])).reshape(2,3)
        for i,a in enumerate(arms):
            gid=a['pad_id']
            actual=d.geom_xpos[gid]+d.geom_xmat[gid].reshape(3,3)@np.array([0,0,m.geom_size[gid,2]])
            np.testing.assert_allclose(faces[i],actual,atol=1e-10)
        old_compression=exp.REST_GAP-(tips[6]-tips[0])
        face_gap=exp.REST_GAP-sum(m.geom_size[a['pad_id'],2] for a in arms)
        self.assertAlmostEqual(old_compression,face_gap-(faces[1,0]-faces[0,0]),places=7)
        q,v,t=d.qpos.copy(),d.qvel.copy(),d.time
        snapshot={}
        with patch.object(diag.mujoco,'mj_forward',wraps=mujoco.mj_forward) as forward:
            diag.sample_plant_parameters(m,d,arms,b,contact_snapshot=snapshot)
        self.assertEqual(forward.call_count,2)
        np.testing.assert_array_equal(q,d.qpos);np.testing.assert_array_equal(v,d.qvel);self.assertEqual(t,d.time)
        w=diag.pad_box_wrench_3d(m,d,arms,b)
        np.testing.assert_allclose(snapshot['world_wrench'],np.concatenate([v for a in arms for v in w[a['side']]]),atol=1e-12)
        np.testing.assert_allclose(snapshot['full_contact'],d.qfrc_constraint[np.concatenate([a['dofs'] for a in arms])],atol=1e-10)
        self.assertEqual(exp.SCENE_XML.read_bytes(),original_bytes)

    def test_reference_all_phases_and_zero_motion(self):
        p0={'left':np.array([-.1,0,0]),'right':np.array([.1,0,0])}
        p1={s:p+np.array([.001 if s=='left' else -.001,0,0]) for s,p in p0.items()}
        rv={s:np.zeros(3) for s in p0}
        r,_=exp.phase_profile(12,-.01,.01,.5,.1,p0,p1,rv,f_start=10)
        for t in (.1,.49,.55,.61,.79,.81,1.61,1.79,1.81):
            a,b,c=r.sample(t-1e-6),r.sample(t),r.sample(t+1e-6)
            for s in p0:np.testing.assert_allclose((c.refs[s]-a.refs[s])/2e-6,b.vrefs[s],atol=1e-8)
            self.assertAlmostEqual((c.fmag-a.fmag)/2e-6,b.fmag_dot,places=7)
        self.assertAlmostEqual(r.move_duration,1.2)
        self.assertAlmostEqual(r.sample(r.move_end).refs['left'][1],-.01)
        np.testing.assert_array_equal(r.sample(r.move_end).vrefs['left'],np.zeros(6))
        for speed,distance in ((0,.1),(-.01,0)):
            zero,_=exp.phase_profile(12,speed,distance,0,1,p0,p1,rv,f_start=10)
            self.assertEqual(zero.move_duration,0)
            self.assertEqual(zero.sample(0).fmag_dot,0)

    def test_startup_failure_is_saved_without_stepping_or_clipping_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'failed'
            argv=['handheld','--no_viewer','--no_csv','--hold_only','--output_dir',str(output)]
            with patch.object(sys,'argv',argv), \
                 patch.object(nc,'AcadosOcpSolver',side_effect=lambda ocp,**kw:Solver(ocp)), \
                 patch.object(nc.NMPCCCart,'equilibrium_seed',return_value=np.full(14,np.nan)):
                with self.assertRaisesRegex(ValueError,'seed is outside'):
                    exp.main()
            with np.load(output/'joint_limits.npz') as q:
                self.assertEqual(float(q['failure_t']),0)
                self.assertTrue(np.isnan(q['failure_u_seed']).all())
                self.assertEqual(q['failure_pred_x'].shape,(11,28))
            with np.load(output/'exit_state.npz') as state:self.assertEqual(float(state['time']),0)


if __name__=='__main__':unittest.main()
