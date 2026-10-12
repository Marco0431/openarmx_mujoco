"""Total contact anchoring must neither double-count force nor freeze its future model."""
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
import nmpcc_cart as nc
from exp2_diagnostics import contact_wrench_on_arm, sample_plant_parameters
from test_nmpcc_revision import RecordingSolver, toy_fk

QI, VI = np.r_[0:7,14:21], np.r_[7:14,21:28]


class ContactResidualTests(unittest.TestCase):
    def setUp(self):
        with patch.object(nc,'AcadosOcpSolver',side_effect=lambda ocp,**kw:RecordingSolver(ocp)):
            self.c=nc.NMPCCCart(nc.NMPCCCartConfig(N=3,kx=100,rest_gap=.5),toy_fk())
        self.x=np.linspace(-.1,.3,28)
        self.p=np.zeros(nc.NP)
        self.ml=np.eye(7)*2+np.full((7,7),.03)
        self.mr=np.eye(7)*3+np.full((7,7),.02)
        self.p[:49]=self.ml.ravel();self.p[56:105]=self.mr.ravel()
        self.p[49:56]=np.arange(7);self.p[105:112]=-np.arange(7)
        self.p[112:126]=np.linspace(-1,2,14);self.p[162:176]=.5
        self.full=np.linspace(-3,4,14);self.u=np.linspace(1,2,14)

    def test_current_total_load_exact_for_legacy_and_node_inputs(self):
        for width in (176,178,179,191):
            for nodes in (False,True):
                p=self.p[:width].copy()
                if nodes:
                    p=np.tile(p,(4,1));p[1:,49:56]+=10
                saved=p.copy()
                corrected=self.c.contact_residual_parameters(self.x,p,self.full,[-10,11])
                p0=corrected[0] if nodes else corrected
                dd=np.asarray(self.c._dynamics_fn(self.x,self.u,p0)).ravel()[VI]
                rhs=self.u+self.full-np.r_[p0[49:56],p0[105:112]]-p0[162:176]*self.x[VI]
                expected=np.r_[np.linalg.solve(self.ml,rhs[:7]),np.linalg.solve(self.mr,rhs[7:])]
                np.testing.assert_allclose(dd,expected,atol=1e-11)
                np.testing.assert_array_equal(p,saved)
                if nodes:
                    np.testing.assert_array_equal(corrected[:,112:126],np.tile(p0[112:126],(4,1)))
                    np.testing.assert_array_equal(corrected[:,49:56],p[:,49:56])

    def test_future_model_normal_still_changes_and_bias_is_not_applied_twice(self):
        p=self.c.contact_residual_parameters(self.x,self.p,self.full,[-10,11])
        snapshot=self.c.last_contact_load_snapshot
        np.testing.assert_allclose(snapshot['frozen_residual']+snapshot['model_normal'],self.full)
        future=self.x.copy();future[0]+=.05
        now=np.asarray(self.c._dynamics_fn(self.x,self.u,p)).ravel()[VI]
        later=np.asarray(self.c._dynamics_fn(future,self.u,p)).ravel()[VI]
        self.assertGreater(np.max(abs(later-now)),1e-4)
        # solve's repeated normal-force anchoring must preserve the residual.
        self.c.solve(self.x,p,f_meas=[-10,11])
        np.testing.assert_array_equal(self.c.last_parameters[:,112:126],np.tile(p[112:126],(4,1)))
        seed=self.c.equilibrium_seed(self.x,p,[-10,11])
        np.testing.assert_allclose(np.asarray(self.c._dynamics_fn(self.x,seed,p)).ravel()[VI],0,atol=1e-11)

    def test_residual_rejects_nonfinite_or_wrong_size(self):
        for load in (np.zeros(13),np.full(14,np.nan)):
            with self.assertRaises(ValueError):self.c.contact_residual_parameters(self.x,self.p,load,[-10,10])
        with self.assertRaises(ValueError):self.c.contact_residual_parameters(self.x[:-1],self.p,self.full,[-10,10])
        with self.assertRaises(ValueError):self.c.contact_residual_parameters(self.x,self.p,self.full,[np.inf,10])

    def test_failure_rebuild_preserves_corrected_parameters_and_weights(self):
        corrected=self.c.contact_residual_parameters(self.x,self.p,self.full,[-10,11])
        stage,terminal=self.c.W.copy(),self.c.W_e.copy()
        self.c.solver.status=4
        with patch.object(nc,'AcadosOcpSolver',side_effect=lambda ocp,**kw:RecordingSolver(ocp)):
            self.c.solve(self.x,corrected,f_meas=[-10,11],u_seed=np.zeros(14))
        self.assertEqual(self.c.last_status,0)
        np.testing.assert_array_equal(self.c.last_parameters[:,112:126],np.tile(corrected[112:126],(4,1)))
        np.testing.assert_array_equal(self.c.first_failure['p_full'][:,112:126],np.tile(corrected[112:126],(4,1)))
        np.testing.assert_array_equal(self.c.W,stage)
        np.testing.assert_array_equal(self.c.W_e,terminal)

    def test_full_contact_matches_mujoco_generalized_constraint_load(self):
        # One free pad, tilted fixed box, friction including rolling/spin torques.
        fixed='<body name="boxbody" quat=".98006658 0 0 .19866933"><geom name="box" type="box" size=".1 .1 .1" condim="6" friction="1 .1 .1"/></body>'
        moving='<body name="padbody" pos=".12895 .05452 .02"><freejoint/><geom name="pad" type="sphere" size=".05" mass="1" condim="6" friction="1 .1 .1"/></body>'
        for bodies in (fixed+moving,moving+fixed):
            m=mujoco.MjModel.from_xml_string('<mujoco><option gravity="0 0 0"/><worldbody>'+bodies+'</worldbody></mujoco>')
            d=mujoco.MjData(m);d.qvel[:]=[0,.02,-.03,.1,.2,.3]
            mujoco.mj_forward(m,d)
            arm=dict(dofs=np.arange(6),pad_id=m.geom('pad').id,body_id=m.body('padbody').id)
            snapshot={}
            tangent,_=contact_wrench_on_arm(m,d,arm,m.geom('box').id,snapshot=snapshot)
            self.assertGreater(snapshot['normal_magnitude'],0)
            np.testing.assert_allclose(snapshot['full_contact'],d.qfrc_constraint,atol=1e-10)
            self.assertGreater(np.max(abs(tangent-snapshot['full_contact'])),1e-4)
            before=d.qpos.copy(),d.qvel.copy(),d.time
            raw={}
            real=mujoco.mj_forward
            with patch.object(mujoco,'mj_forward',wraps=real) as forward:
                sample_plant_parameters(m,d,[arm],m.geom('box').id,contact_snapshot=raw)
                self.assertEqual(forward.call_count,2)
            for expected,actual in zip(before,(d.qpos,d.qvel,d.time)):
                np.testing.assert_array_equal(expected,actual)
            np.testing.assert_allclose(raw['full_contact'],d.qfrc_constraint,atol=1e-10)


if __name__=='__main__':unittest.main()
