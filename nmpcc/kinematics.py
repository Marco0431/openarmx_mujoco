"""MuJoCo 静态几何到 CasADi 运动学；长度 m，角度 rad，世界坐标。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
from .data import Array
import casadi as ca

def _rotvec_of_R_sx(R: ca.SX) -> ca.SX:
    tr = (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0
    theta = ca.acos(ca.fmax(ca.fmin(tr, 1.0), -1.0))
    vee = ca.vertcat(R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1])
    small = theta < 1e-6
    return ca.if_else(small, 0.5 * vee, 0.5 * theta / ca.sin(theta) * vee)


def _quat2mat_sx(q: ca.SX) -> ca.SX:
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        ca.horzcat(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        ca.horzcat(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def _rot_axis(axis: ca.SX, q: ca.SX) -> ca.SX:
    k = axis / ca.norm_2(axis)
    K = ca.vertcat(
        ca.horzcat(0.0, -k[2], k[1]),
        ca.horzcat(k[2], 0.0, -k[0]),
        ca.horzcat(-k[1], k[0], 0.0))
    return ca.SX.eye(3) + ca.sin(q) * K + (1 - ca.cos(q)) * (K @ K)


def _quat_to_mat_np(q: Array) -> Array:
    """将 MuJoCo wxyz 四元数转换为旋转矩阵。"""
    w, x, y, z = np.asarray(q, dtype=float)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pad_contact_offsets(model: Any, arms: Sequence[dict[str, Any]], ee_tip_z: float) -> list[Array]:
    """返回左右 pad 朝箱体接触面中心相对 link7 tip 的局部偏移。

    ``geom_pos`` 是 pad 中心相对 link7 原点的位置，``geom_size[2]``
    是 pad 半厚度；pad 的局部 +z 面是实际朝向箱体的受力面。
    """
    offsets = []
    tip_offset = np.array([0.0, 0.0, float(ee_tip_z)])
    for arm in arms:
        pad_id = int(arm["pad_id"])
        pad_pos = np.asarray(model.geom_pos[pad_id], dtype=float)
        pad_R = _quat_to_mat_np(model.geom_quat[pad_id])
        pad_face = pad_pos + pad_R @ np.array(
            [0.0, 0.0, float(model.geom_size[pad_id, 2])])
        offsets.append(pad_face - tip_offset)
    return offsets


@dataclass(slots=True)
class LinkTransform:
    position: Array
    quaternion: Array
    axis: Array
    jointed: bool


def _collect_chain(model: Any, body_id: int) -> list[LinkTransform]:
    body_ids = []
    body = int(body_id)
    while body != 0:
        body_ids.append(body)
        body = int(model.body_parentid[body])
    chain = []
    for body in reversed(body_ids):
        joint = int(model.body_jntadr[body])
        jointed = joint >= 0
        axis = model.jnt_axis[joint].copy() if jointed else np.zeros(3)
        axis /= np.linalg.norm(axis) + 1e-12
        chain.append(LinkTransform(model.body_pos[body].copy(), model.body_quat[body].copy(), axis, jointed))
    return chain


def _chain_transform(chain: Sequence[LinkTransform], q: ca.SX) -> ca.SX:
    transform, index = ca.SX.eye(4), 0
    for link in chain:
        rotation = _quat2mat_sx(ca.SX(link.quaternion))
        joint_rotation = _rot_axis(ca.SX(link.axis), q[index]) if link.jointed else ca.SX.eye(3)
        index += int(link.jointed)
        transform = transform @ ca.vertcat(
            ca.horzcat(rotation @ joint_rotation, ca.reshape(ca.SX(link.position), 3, 1)),
            ca.horzcat(ca.SX.zeros(1, 3), ca.SX.ones(1, 1)))
    return transform


def mujoco_fk_symbols(model: Any, arm_cfgs: Sequence[dict[str, Any]], ee_tip_z: float,
                      contact_offsets: Sequence[Array] | None = None, *,
                      contact_local_points: Sequence[Array] | None = None) -> dict[str, Any]:
    """提取静态机器人几何；tip 为轨迹点，pad 接触面中心用于弹簧和力 Jacobian。

    contact_offsets 相对 link7 tip；contact_local_points 相对 link7 原点。
    两种输入互斥，均按左右臂顺序给出两个局部三维坐标。
    保留原模型的逐连杆乘法顺序以便数值对照。
    """
    if len(arm_cfgs) != 2:
        raise ValueError("exactly two arm configurations are required")
    if contact_offsets is not None and contact_local_points is not None:
        raise ValueError("contact_offsets and contact_local_points are mutually exclusive")
    if contact_offsets is None and contact_local_points is None:
        contact_offsets = (np.zeros(3), np.zeros(3))
    points = np.asarray(contact_offsets if contact_local_points is None else contact_local_points, float)
    if points.shape != (2, 3) or not np.isfinite(points).all():
        raise ValueError("contact coordinates must contain two finite 3D points")
    chains = [_collect_chain(model, int(arm["body_id"])) for arm in arm_cfgs]
    counts = [sum(link.jointed for link in chain) for chain in chains]
    if counts != [7, 7]:
        raise ValueError(f"expected seven joints per arm, got {counts}")
    joints = [ca.SX.sym("qL", 7), ca.SX.sym("qR", 7)]
    poses, contacts = [], []
    for chain, q, point in zip(chains, joints, points):
        transform = _chain_transform(chain, q)
        tip = transform @ ca.vertcat(0., 0., ee_tip_z, 1.)
        poses.append(ca.vertcat(tip[:3], _rotvec_of_R_sx(transform[:3, :3])))
        if contact_local_points is None:
            contacts.append(tip[:3] + transform[:3, :3] @ ca.SX(point.reshape(3, 1)))
        else:
            contacts.append((transform @ ca.vertcat(ca.SX(point), 1.))[:3])
    tips, contact_tips = ca.vertcat(*poses), ca.vertcat(*contacts)
    return {"tips_qt": ca.vertcat(*joints), "tips": tips,
            "tips_fn": ca.Function("fk_tips", joints, [tips]),
            "contact_tips": contact_tips,
            "contact_tips_fn": ca.Function("fk_contact_tips", joints, [contact_tips]), "n_arm": 7}


def rotvec_of_R(R: Array) -> Array:
    """世界旋转矩阵 (3,3) → 旋转向量（主值，与符号实现一致）。"""
    R = np.asarray(R, float)
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-6:
        return 0.5 * vee
    return 0.5 * theta / np.sin(theta) * vee


