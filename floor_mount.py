#!/usr/bin/env python3
"""floor_mount.py —— OpenArmX 双臂"地面安装"模型生成器（论文 Fig.11 安装方式）

把两条 7-DOF 手臂从躯干（openarmx_robot_yaw_root）上拆下来，分别装到地面基座上、
面对面摆放（左臂在 -x 侧伸向 +x，右臂在 +x 侧伸向 -x）——对应论文里两台 Panda
基座直接立在台面/地面上的布局。

- 臂的关节链 / 惯性 / 网格 / 端面垫片完全不变（ik.py generate_runtime_xml 已加垫片）；
- 只替换安装位：link1 的 pos 从肩部 (0,±0.089,0.735) 改为基座 (±base_x,0,z_mount)；
- 关节名不变 → ARM_CONFIGS / IK / NMPCC（mujoco_fk_symbols）全部直接兼容；
- 附带基座立柱 + 底盘（纯视觉，contype=0）。
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ik import BASE_DIR, ROBOT_YAW_LEFT_90, generate_runtime_xml
from ikforce import strip_collisions
from grasp_pose_static import strip_ee_geoms

FLOOR_RUNTIME = BASE_DIR / "openarmx_floor_runtime.xml"


def generate_floor_runtime_xml(base_x: float = 0.45, z_mount: float = 0.06,
                               mount_roll: float = 0.0, mount_pitch: float = 0.0,
                               mount_yaw: float | dict = 0.0,
                               pad_fix: dict | None = None,
                               pad_half_w: float = 0.030, pad_half_h: float = 0.030,
                               out: Path = FLOOR_RUNTIME) -> Path:
    """生成地面安装版 runtime XML。

    base_x : 基座中心到原点距离（左 -base_x，右 +base_x）；箱长 ≈ 2·base_x − 2·臂前伸量。
    z_mount: 臂 link1（joint1）离地高度。
    mount_roll / mount_pitch / mount_yaw: 臂安装滚转/俯仰/偏航角（rad，滚转/偏航支持
              dict 每侧独立取值）——调整安装姿态；link0 基座壳跟随 yaw 但不跟随 roll/pitch。
    pad_fix : 可选 {side: (quat_wxyz, pos_xyz)} —— 垫片斜角安装（斜角转接板）。
    pad_half_w / pad_half_h: 垫片横向/竖直半尺寸（m）。加宽垫片 = 更大接触面 → 更强
              摩擦容量与抗滚转力臂（箱面 12×8cm，默认 6×6cm）。
    """
    runtime_xml = generate_runtime_xml()          # 重新从 URDF 生成（含端面垫片）
    strip_ee_geoms(runtime_xml)                   # 删手/指网格（无夹具顶夹）
    strip_collisions(runtime_xml)                 # 删除 link0-link7 自带碰撞网格，只保留 ee_pad 与场景物体

    tree = ET.parse(runtime_xml)
    root = tree.getroot()
    for g in root.iter("geom"):                   # 垫片尺寸参数化（默认 6cm 见方保持不变）
        if (g.get("name") or "").endswith("_ee_pad"):
            g.set("size", f"{pad_half_w:.4f} {pad_half_h:.4f} 0.004")
    worldbody = root.find("worldbody")
    yaw_root = next(b for b in worldbody.findall("body")
                    if b.get("name") == "openarmx_robot_yaw_root")

    # 摘下两条臂的 link1 子树 + 各自的 link0 基座壳视觉网格
    arms, link0_geoms = {}, {}
    for side in ("left", "right"):
        arm = next(b for b in yaw_root.findall("body")
                   if b.get("name") == f"openarmx_{side}_link1")
        yaw_root.remove(arm)
        arms[side] = arm
        link0_geoms[side] = [g for g in yaw_root.findall("geom")
                             if f"openarmx_{side}_link0_visual" in (g.get("name") or "")]
    worldbody.remove(yaw_root)                    # 躯干（躯干视觉网格）移除

    yaw0 = float(ROBOT_YAW_LEFT_90)
    for side, sx in (("left", -base_x), ("right", base_x)):
        # 支持每侧独立滚转/偏航：标量=双臂同值；dict={...}=每侧独立
        roll_s = mount_roll.get(side, 0.0) if isinstance(mount_roll, dict) else mount_roll
        yaw_s = mount_yaw.get(side, 0.0) if isinstance(mount_yaw, dict) else mount_yaw
        wrap = ET.Element("body", {
            "name": f"openarmx_{side}_floor_base",
            "pos": f"{sx:.4f} 0 0",
            "euler": f"0 0 {yaw0 + yaw_s:.6f}",   # 基座 yaw：-90° 基础 + 可选偏航调整
        })
        # 底盘 + 立柱（纯视觉）
        ET.SubElement(wrap, "geom", {
            "name": f"{side}_base_plate", "type": "box",
            "pos": "0 0 0.012", "size": "0.11 0.11 0.012",
            "rgba": "0.22 0.24 0.28 1", "contype": "0", "conaffinity": "0", "group": "1",
        })
        ET.SubElement(wrap, "geom", {
            "name": f"{side}_base_post", "type": "cylinder",
            "pos": f"0 0 {z_mount / 2:.4f}", "size": f"0.055 {max(z_mount / 2, 0.012):.4f}",
            "rgba": "0.30 0.32 0.36 1", "contype": "0", "conaffinity": "0", "group": "1",
        })
        # link0 基座壳视觉：躯干版中相对 j1 偏移 (0, ∓0.058, 0)，保持不滚转（它是落地基座）
        off_y = -0.058 if side == "left" else 0.058
        for g in link0_geoms[side]:
            g.set("pos", f"0 {off_y:.4f} {z_mount:.4f}")
            wrap.append(g)                        # 留在 wrap 层：随 yaw、不随 roll/pitch
        arm = arms[side]
        arm.set("pos", f"0 0 {z_mount:.4f}")      # 替换肩部偏移；quat 保持（关节1轴向不变）
        if pad_fix and side in pad_fix:           # 垫片斜角安装（转接板）
            quat_wxyz, pos_xyz = pad_fix[side]
            link7 = next(b for b in arm.iter("body") if b.get("name") == f"openarmx_{side}_link7")
            pad = next(g for g in link7.findall("geom") if g.get("name") == f"openarmx_{side}_ee_pad")
            pad.set("pos", f"{pos_xyz[0]:.6f} {pos_xyz[1]:.6f} {pos_xyz[2]:.6f}")
            pad.set("quat", f"{quat_wxyz[0]:.8f} {quat_wxyz[1]:.8f} {quat_wxyz[2]:.8f} {quat_wxyz[3]:.8f}")
        mount = ET.Element("body", {
            "name": f"{side}_arm_mount", "pos": "0 0 0",
            "euler": f"{roll_s:.6f} {mount_pitch:.6f} 0",   # 安装姿态：只转臂，不转立柱/基座壳
        })
        mount.append(arm)
        wrap.append(mount)
        worldbody.append(wrap)

    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


if __name__ == "__main__":
    import mujoco
    p = generate_floor_runtime_xml()
    m = mujoco.MjModel.from_xml_path(str(p))
    print(f"✅ 地面安装模型: {p.name} | nq={m.nq} nv={m.nv} nbody={m.nbody} ngeom={m.ngeom}")