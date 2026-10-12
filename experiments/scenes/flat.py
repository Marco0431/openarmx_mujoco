"""平放场景的几何和初始抓取。"""
import mujoco
import numpy as np
from ik import clamp_qpos, name_to_id
from exp2_slender_scene import (BOX_CENTER, CAN_HALF_H, CAN_OFFSETS_X, IK_ITERS, SLENDER_HALF,
    build_grasped_scene, grasp_targets, solve_end_grip)
K_STIFF = 2793.166
REST_GAP = 0.507537

def reset_world(model, data, arms, box_qpos) -> None:
    """恢复初始关节和箱体姿态，清零速度并刷新模型。
    这是每次实验开始时的复位，不是控制循环内的状态修正。"""
    for a in arms:
        data.qpos[a["qpos"]] = a["q_rest"]
        clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    data.qpos[box_qpos:box_qpos + 3] = BOX_CENTER
    data.qpos[box_qpos + 3:box_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    for k, ox in enumerate(CAN_OFFSETS_X):
        cq = int(model.jnt_qposadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                               f"can{k}_freejoint")])
        data.qpos[cq:cq + 3] = [BOX_CENTER[0] + ox, BOX_CENTER[1],
                                BOX_CENTER[2] + SLENDER_HALF[2] + CAN_HALF_H + 0.002]
        data.qpos[cq + 3:cq + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)


