import sys
from pathlib import Path
import numpy as np
import mujoco

base = Path(r'D:\RobotSim\openarmx_mujoco')
sys.path.insert(0, str(base))
import exp2_nmpcc_latest_grasp_pose as exp

model = mujoco.MjModel.from_xml_path(str(base / 'grasp_pose_scene.xml'))
data = mujoco.MjData(model)
arms = exp.make_arms(model)
for a in arms:
    a['q_rest'] = np.asarray(a['q_rest'], dtype=float)
    data.qpos[a['qpos']] = a['q_rest']
data.qvel[:] = 0
mujoco.mj_forward(model, data)
print('root euler rotation: local +x -> world', np.array([0.0, -1.0, 0.0]))
print('root euler rotation: local +y -> world', np.array([1.0, 0.0, 0.0]))
for a in arms:
    print(a['side'], 'body', a['body_id'], 'tip', np.array2string(exp.get_ee_tip(data, a['body_id']), precision=9))
print('box XML pos', data.xpos[exp.name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, 'box')])
print('body dof adr', model.body_dofadr[exp.name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, 'box')])
print('root body id', exp.name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, 'openarmx_robot_yaw_root'))
print('body_pos root local', model.body_pos[exp.name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, 'openarmx_robot_yaw_root')])
print('body_quat root', model.body_quat[exp.name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, 'openarmx_robot_yaw_root')])
for a in arms:
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
    tip = exp.get_ee_tip(data, a['body_id'])
    mujoco.mj_jac(model, data, jacp, jacr, tip, a['body_id'])
    print(a['side'], 'Jpos rows xyz / arm dofs')
    print(np.array2string(jacp[:, a['dofs']], precision=5, suppress_small=True))

# Verify the generated phase reference without calling a solver.
press = {a['side']: exp.get_ee_tip(data, a['body_id']).copy() for a in arms}
rv = {a['side']: np.zeros(3) for a in arms}
fn, tend = exp.phase_profile(20.0, 0.01, 0.01, 0.1, 0.1, press, press, rv, f_start=10.0)
for t in (0.0, 0.2, 0.7, 1.3):
    refs, vrefs, _, phase, _ = fn(t)
    print('profile', t, phase, 'left_ref', refs['left'][:3], 'vx', vrefs['left'][0])
