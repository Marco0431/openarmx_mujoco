#!/usr/bin/env python3
"""_probe_grip_z.py —— 探针：垫片接触高度对箱体滚转稳定性的影响（一次性实验，验完删）

把垫片在 link7 局部系内沿世界竖直方向偏移 dz（正=向上），冻结双臂，
给箱体一个初始绕夹持轴(x)滚转扰动 phi0，步进物理，观察 rvx 演化：
  - rvx 收敛回 0 → 重力是恢复力矩（摆式稳定）
  - rvx 增大 → 重力放大滚转（倒摆/顶重不稳定）
"""
from __future__ import annotations
import numpy as np
import mujoco
from exp2_slender_scene import (BOX_CENTER, SLENDER_HALF, build_scene, grasp_targets,
                                make_arms, solve_end_grip)
from ik import EE_TIP_Z, name_to_id

DT = 0.002


def make_pad_fix_vertical(model, data, arms, dz: float) -> dict:
    """与 compute_pad_fix 同构，但垫片沿世界竖直方向额外偏移 dz（link7 局部系内表达）。"""
    fix = {}
    for a in arms:
        R = data.xmat[a["body_id"]].reshape(3, 3)
        n_world = np.array([-a["sign"], 0.0, 0.0])            # 垫片法向：纯水平指向箱内
        n_local = R.T @ n_world
        n_local /= np.linalg.norm(n_local)
        v_local = R.T @ np.array([0.0, 0.0, 1.0])             # 世界 +z 在 link7 局部系
        v_local /= np.linalg.norm(v_local)
        # 旋转矩阵：z' = n_local；y' 取世界竖直方向的投影保证垫片竖直摆放
        zp = n_local
        ref = v_local
        if abs(float(zp @ ref)) > 0.95:
            ref = np.array([1.0, 0.0, 0.0])
        xp = np.cross(ref, zp); xp /= np.linalg.norm(xp)
        yp = np.cross(zp, xp)
        R_pad = np.column_stack([xp, yp, zp])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R_pad.reshape(-1))
        pos = np.array([0.0, 0.0, EE_TIP_Z]) + dz * v_local
        fix[a["side"]] = (quat, pos)
    return fix


def run_case(dz: float, phi0: float, steps: int = 1500):
    pad_fix = None
    if abs(dz) > 1e-9:
        # 两阶段：先裸模型 IK 出抓取姿态，再带 dz 垫片重建
        sc = build_scene(None)
        m = mujoco.MjModel.from_xml_path(str(sc))
        d = mujoco.MjData(m)
        arms = make_arms(m)
        for a in arms:
            d.qpos[a["qpos"]] = a["q_rest"].copy()
        d.qvel[:] = 0.0
        bq = int(m.jnt_qposadr[name_to_id(m, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")])
        d.qpos[bq:bq + 3] = BOX_CENTER
        d.qpos[bq + 3:bq + 7] = [1.0, 0.0, 0.0, 0.0]
        from exp2_slender_scene import CAN_COUNT, CAN_HALF_H, CAN_OFFSETS_X, SLENDER_HALF as SH
        for k, ox in enumerate(CAN_OFFSETS_X):
            cq = int(m.jnt_qposadr[name_to_id(m, mujoco.mjtObj.mjOBJ_JOINT, f"can{k}_freejoint")])
            d.qpos[cq:cq + 3] = [BOX_CENTER[0] + ox, BOX_CENTER[1],
                                 BOX_CENTER[2] + SH[2] + CAN_HALF_H + 0.002]
            d.qpos[cq + 3:cq + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(m, d)
        solve_end_grip(m, d, arms, grasp_targets(arms))    # 先收敛到抓取姿态
        pad_fix = make_pad_fix_vertical(m, d, arms, dz)

    sc = build_scene(pad_fix)
    m = mujoco.MjModel.from_xml_path(str(sc))
    d = mujoco.MjData(m)
    arms = make_arms(m)
    for a in arms:
        d.qpos[a["qpos"]] = a["q_rest"].copy()
    bq = int(m.jnt_qposadr[name_to_id(m, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")])
    d.qpos[bq:bq + 3] = BOX_CENTER
    d.qpos[bq + 3:bq + 7] = [1.0, 0.0, 0.0, 0.0]
    from exp2_slender_scene import CAN_COUNT, CAN_HALF_H, CAN_OFFSETS_X, SLENDER_HALF as SH
    for k, ox in enumerate(CAN_OFFSETS_X):
        cq = int(m.jnt_qposadr[name_to_id(m, mujoco.mjtObj.mjOBJ_JOINT, f"can{k}_freejoint")])
        d.qpos[cq:cq + 3] = [BOX_CENTER[0] + ox, BOX_CENTER[1],
                             BOX_CENTER[2] + SH[2] + CAN_HALF_H + 0.002]
        d.qpos[cq + 3:cq + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    targets = grasp_targets(arms)
    solve_end_grip(m, d, arms, targets)
    # 预紧压入（与主程序一致）：每侧向内压入 delta = F/(2*K_STIFF)
    K_STIFF = 2793.166
    press = {a["side"]: targets[a["side"]] + np.array([-a["sign"] * 10.0 / (2.0 * K_STIFF), 0.0, 0.0])
             for a in arms}
    solve_end_grip(m, d, arms, press)

    # 冻结双臂（保存 qpos，每步强制写回）
    qpos_frozen = {side: d.qpos[a["qpos"]].copy() for side, a in zip(("left", "right"), arms)}
    qadr = {a["side"]: a["qpos"] for a in arms}
    dofs = [a["dofs"] for a in arms]

    # 给箱体初始滚转扰动
    d.qpos[bq + 3:bq + 7] = [np.cos(phi0 / 2), np.sin(phi0 / 2), 0.0, 0.0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    hist = []
    for i in range(steps):
        for a, dof in zip(arms, dofs):
            d.qpos[qadr[a["side"]]] = qpos_frozen[a["side"]]
            d.qvel[dof] = 0.0
        mujoco.mj_step(m, d)
        q = d.qpos[bq + 3:bq + 7]
        w, x, y, z = q
        phi = 2.0 * np.arctan2(float(x), float(w))   # 绕 x 滚转角
        hist.append((d.time, phi, float(d.qpos[bq + 2])))
    return np.array(hist)


if __name__ == "__main__":
    for dz in (0.0, +0.02, -0.02):
        h = run_case(dz, phi0=0.05)
        t, phi, boxz = h.T
        phi_deg = np.degrees(phi)
        print(f"dz={dz:+5.2f} m: phi0={np.degrees(phi[0]):+5.2f}° -> "
              f"t=0.5 phi={phi_deg[250]:+7.3f}° | t=1.0 phi={phi_deg[500]:+7.3f}° | "
              f"t=1.5 phi={phi_deg[750]:+7.3f}° | t=3.0 phi={phi_deg[1499]:+7.3f}° | "
              f"boxz t=3.0 = {boxz[1499]:.4f}")
