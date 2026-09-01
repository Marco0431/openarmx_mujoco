#!/usr/bin/env python3
"""calib_contact.py —— 实测 MuJoCo 软接触刚度 k1（用于预测模型式(27) 的 kx 标定）

方法：位置伺服把左臂端面目标沿 x 内移 δ（1→10mm），稳定后读垫片-箱接触力 f(δ)；
  每侧刚度 k1 = Δf/Δδ（线性段拟合）；模型式(27) 的压缩量是"两臂间距差"（两侧串联），
  故 kx = k1/2（且力参考 25N 对应的每臂压深 δ* = 25/(2·kx)）。
运行: D:\\acados_env\\python.exe -u calib_contact.py
"""
import numpy as np
import mujoco

from ik import ARM_CONFIGS, EE_TIP_Z, get_ee_tip, name_to_id
from grasp_pose_static import BOX_HALF, BOX_MASS, GRIP_CLEARANCE, solve_grasp_pose
from ikforce import IK_DT, K_POS, K_D, K_DQ, K_PQ, MAX_NULL_TORQUE
from ikforce_grasp import GRASP_Z, build_scene
from exp_box_nmpcc_cart import soften_contacts, arm_damping, pad_contact_forces

model, data = build_scene()
box_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
box_geom_id = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
box_qpos = int(model.jnt_qposadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")])
box_qvel = int(model.jnt_dofadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")])
arms = []
for cfg in ARM_CONFIGS:
    body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["jacobi_body"])
    jids = [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg["joint_names"]]
    dofs = [int(model.jnt_dofadr[j]) for j in jids]
    qpos = [int(model.jnt_qposadr[j]) for j in jids]
    apos = [name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_motor") for n in cfg["joint_names"]]
    arms.append({"side": cfg["side"], "sign": 1.0 if cfg["side"] == "left" else -1.0,
                 "q_rest": cfg["q_rest"], "body_id": body_id, "joint_ids": jids,
                 "dofs": dofs, "qpos": qpos, "act_ids": apos,
                 "pad_id": name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"openarmx_{cfg['side']}_ee_pad")})
soften_contacts(model, box_geom_id, [a["pad_id"] for a in arms])
arm_damping(model, [d for a in arms for d in a["dofs"]], 0.5)

for a in arms:
    data.qpos[a["qpos"]] = a["q_rest"].copy()
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)
mid = np.mean([get_ee_tip(data, a["body_id"]) for a in arms], axis=0)
data.qpos[box_qpos:box_qpos + 3] = [mid[0], mid[1], GRASP_Z]
data.qpos[box_qpos + 3: box_qpos + 7] = [1.0, 0, 0, 0]
mujoco.mj_forward(model, data)
q_grasp = solve_grasp_pose(model, data, arms, box_id, iters=800, tilt=0.0)
for a in arms:
    data.qpos[a["qpos"]] = q_grasp[a["side"]]
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)
box_start = data.xpos[box_id].copy()
left = arms[0]
tip0 = get_ee_tip(data, left["body_id"])

from ikforce import solve_torque_position  # 位置伺服（无力指令）

def run_delta(delta_mm):
    """左臂端面目标沿 x 内移 δ，位置伺服稳定 1s，返回垫片接触力。箱每步冻结。"""
    target = tip0.copy()
    target[0] -= delta_mm / 1000.0            # 左臂（+x 侧）向内 = −x
    for t in range(500):                       # 1.0 s
        mujoco.mj_forward(model, data)
        saved = data.qpos[box_qpos:box_qpos + 7].copy()
        tau, _ = solve_torque_position(model, data, left, target, use_nullspace=True)
        data.ctrl[left["act_ids"]] = tau
        # 右臂保持 q_grasp（重力补偿）
        tau2 = data.qfrc_bias[arms[1]["dofs"]].copy() - 0.8 * data.qvel[arms[1]["dofs"]]
        data.ctrl[arms[1]["act_ids"]] = tau2
        mujoco.mj_step(model, data)
        data.qpos[box_qpos:box_qpos + 7] = saved
        data.qvel[box_qvel:box_qvel + 6] = 0.0
    fc = pad_contact_forces(model, data, arms, box_geom_id)
    # 端面实际位置（压进再读）
    tip = get_ee_tip(data, left["body_id"])
    return fc["left"], tip

print("δ(mm)  F(N)   tip_x")
deltas = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
pts = []
for d in deltas:
    F, tip = run_delta(d)
    pts.append((d, F))
    print(f"{d:5.1f} {F:7.1f} {tip[0]:.4f}")
# 线性段拟合（取 F 在 5~35N 之间的点）
sel = [(d, F) for d, F in pts if 5.0 < F < 35.0]
if len(sel) >= 2:
    ds = np.array([s[0] for s in sel]); Fs = np.array([s[1] for s in sel])
    k1_N_mm = np.polyfit(ds, Fs, 1)[0]
    k1 = k1_N_mm * 1000.0                        # N/m（单侧）
    kx = k1 / 2.0                                # 模型（两侧串联）
    print(f"\nk1（单侧实测） = {k1:.0f} N/m")
    print(f"kx_model（=k1/2） = {kx:.0f} N/m")
    print(f"25N 对应每臂压深 δ* = {25.0 / (2 * kx) * 1000:.2f} mm；总压缩 c* = {25.0 / kx * 1000:.2f} mm")
else:
    print("标定区间内点数不足（检查接触/伺服）")