#!/usr/bin/env python3
"""双臂 IK 静态夹箱姿态 (grasp_pose_static.py) —— 夹箱第一步

思路：
  用"位置 + 姿态 IK"（ik.py 的 solve_pose_ik）在纯运动学层面摆出要求的夹箱姿态：
  左右臂最后的柱状杆（link7）水平、其顶面垫片**面对面**压住箱子的两个侧面，
  垫片中心（杆顶面中点）正好对准箱子两个侧面的中点（err≈0.3mm）。
  关键点（实测验证）：
  1. 垫片必须"指向箱内"（desired_dir = −侧向符号）——这是"对夹"的方向；
  2. 箱子中心比休息位端面中点抬高 BOX_Z_OFFSET 时，平放对夹才在腕部可达范围内；
  3. 端面挂载的手/指网格（沿 +Z 前伸 0.66m 的"手"）一律删掉（strip_ee_geoms），
     采用"无末端夹具·平端面顶夹"。
  纯静态：不推进动力学（全程不调 mj_step，qvel/qacc 清零只做正运动学），
  IK 收敛后定格展示。这个姿态就是后续接力/力矩控制抓箱的起点。

箱子：比较宽、尺寸合适，半尺寸 (x=夹持厚度, y=宽, z=高) = (12, 20, 10)cm
运行：
    python grasp_pose_static.py --check     # 只构建加载
    python grasp_pose_static.py --headless  # 无 GUI：收敛 IK 后打印贴合报表
    python grasp_pose_static.py             # 打开 viewer 展示静态夹持姿态（R 复位, G 重新夹持）
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from ik import (
    ARM_CONFIGS,
    BASE_DIR,
    clamp_qpos,
    clamp_target,
    generate_runtime_xml,
    get_ee_tip,
    name_to_id,
    solve_pose_ik,
)

# ---------- 箱子与夹持参数 ----------
BOX_HALF = np.array([0.12, 0.20, 0.10], dtype=np.float64)   # 箱半尺寸（x=夹持厚度 y=宽 z=高）
BOX_MASS = 1.5                                              # 箱子质量 (kg)
BOX_FRICTION = "5 0.05 0.005"                               # 大摩擦：为后面力控/顶夹预留
# 平放对夹：垫片水平、左右面对面压箱（desired_dir = −侧向符号，见 run_clamp_ik）
BOX_Z_OFFSET = 0.10           # 箱子中心相对休息位端面中点抬高 0.1m：该高度下"平放对夹"可达且精确
# 端面挂载的手/指网格会被 strip_ee_geoms 全删（本项目是"无末端夹具·平端面顶夹"）。
# 垫片厚 8mm、半厚 4mm：净空设为 4mm → 垫片前端面(接触面)恰好贴在箱面上（接触、零穿入）
GRIP_CLEARANCE = 0.004
IK_ITERS = 800             # IK 收敛迭代上限（纯运动学，跑满很快）
SCENE_XML = BASE_DIR / "grasp_pose_scene.xml"


def strip_ee_geoms(runtime_xml: Path) -> None:
    """移除端面挂载的手/指网格 geom（名字含 hand 或 finger 的）。

    runtime 模型里手部网格沿 link7 +Z 前伸约 0.66m（网格资产自带 ~0.66 偏移），
    只要端面贴到箱面，手部就会整段穿进箱子、甚至穿出另一侧；URDF 里另有
    *_finger 指节（prismatic 关节）一旦进模型同样碍事。
    本演示是"无末端夹具·平端面顶夹"（与 ik.py 的 EE_TIP_Z 定义一致），把手/指全删，
    只留 link7 平端 + 垫片，姿态才能干净地对夹箱面。
    """
    tree = ET.parse(runtime_xml)
    root = tree.getroot()
    removed = 0
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            nm = geom.get("name", "")
            if "hand" in nm or "finger" in nm:
                body.remove(geom)
                removed += 1
    tree.write(runtime_xml, encoding="utf-8", xml_declaration=True)
    print(f"✅ 已移除 {removed} 个手/指网格（无夹具顶夹演示）")


def build_scene() -> Path:
    """生成 runtime XML（含端面垫片，删掉手/指网格）并写静态夹箱场景：重力开、宽箱子放两臂之间。"""
    runtime_xml = generate_runtime_xml()
    strip_ee_geoms(runtime_xml)
    SCENE_XML.write_text(
        f"""<mujoco model="grasp_pose_static_scene">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" iterations="500" solver="Newton" cone="elliptic"/>

  <asset>
    <texture name="studio_sky" type="skybox" builtin="gradient"
             rgb1="0.08 0.10 0.14" rgb2="0.01 0.012 0.018"
             width="512" height="512"/>
    <texture name="floor_grid" type="2d" builtin="checker"
             rgb1="0.16 0.18 0.21" rgb2="0.24 0.27 0.31"
             width="1024" height="1024" mark="edge" markrgb="0.55 0.60 0.68"/>
    <material name="floor_mat" texture="floor_grid" texrepeat="7 7"
              reflectance="0.22" shininess="0.35" specular="0.25"/>
    <material name="wall_mat" rgba="0.10 0.12 0.16 1" reflectance="0.08"/>
    <material name="box_mat" rgba="0.85 0.55 0.20 1" shininess="0.6" specular="0.4"/>
  </asset>

  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.24 0.24 0.26" specular="0.25 0.25 0.25"/>
    <map znear="0.01" zfar="8"/>
    <scale forcewidth="0.06" contactwidth="0.05"/>
    <rgba haze="0.03 0.035 0.045 1"/>
  </visual>

  <worldbody>
    <light name="key_light" pos="-1.4 -2.2 3.2" dir="0.35 0.55 -1" directional="true"
           diffuse="0.95 0.92 0.86" specular="0.35 0.35 0.35"/>
    <light name="fill_light" pos="1.8 1.4 2.2" dir="-0.45 -0.35 -1" directional="true"
           diffuse="0.35 0.45 0.65" specular="0.12 0.12 0.16"/>
    <light name="rim_light" pos="0 2.2 1.8" dir="0 -1 -0.45" directional="true"
           diffuse="0.55 0.65 0.85" specular="0.20 0.22 0.28"/>

    <geom name="floor" type="plane" pos="0 0 -0.012" size="3.0 3.0 0.02" material="floor_mat"/>
    <geom name="back_wall" type="box" pos="0 0.95 0.78" size="2.6 0.025 0.80" material="wall_mat"
          contype="0" conaffinity="0"/>

    <!-- 箱子：freejoint 先锁在原位（本步不推进动力学，所以它静止） -->
    <body name="box" pos="0.5 0 0.72">
      <freejoint name="box_freejoint"/>
      <geom name="box" type="box" size="{BOX_HALF[0]} {BOX_HALF[1]} {BOX_HALF[2]}"
            material="box_mat" mass="{BOX_MASS}"
            friction="{BOX_FRICTION}" condim="3" contype="1" conaffinity="1"
            solref="0.002 1" solimp="0.999 0.999 0.001"/>
    </body>
  </worldbody>

  <include file="{runtime_xml.name}"/>
</mujoco>
""",
        encoding="utf-8",
    )
    return SCENE_XML


def solve_grasp_pose(
    model, data, arms, box_id, iters: int = IK_ITERS, tilt: float = 0.0
) -> dict[str, np.ndarray]:
    """纯运动学收敛到夹箱姿态：柱顶垫片面对面压箱两侧、中心对准箱侧面中点。

    tilt：垫片法向俯仰角（0=纯平放对夹；>0 时法向 = (−sign·cosα, 0, +sinα)，
    内倾且微微上托——托住箱体，侧倾被动稳定）。返回每臂关节角（供力矩级零空间引用）。
    """
    box_x = float(data.xpos[box_id][0])
    grasp_y, grasp_z = float(data.xpos[box_id][1]), float(data.xpos[box_id][2])
    for _ in range(iters):
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_forward(model, data)
        sols = []
        for arm in arms:
            target_pos = clamp_target(
                [box_x + arm["sign"] * (BOX_HALF[0] + GRIP_CLEARANCE), grasp_y, grasp_z]
            )
            # 兼容两种臂字典命名：grasp_pose_static 用 jac_body_id，ikforce_grasp 用 body_id
            #（注意不能用 dict.get(key, arm["body_id"])——默认参数会立即求值导致 KeyError）
            body_id = arm["jac_body_id"] if "jac_body_id" in arm else arm["body_id"]
            current_pos = get_ee_tip(data, body_id)
            # 斜角夹持：垫片法向 = (−sign·cosα, 0, +sinα)——向内且微微上托，托住箱体
            desired_dir = np.array(
                [-arm["sign"] * np.cos(tilt), 0.0, np.sin(tilt)], dtype=np.float64
            )
            dq, _ = solve_pose_ik(
                model, data, body_id, target_pos, desired_dir, current_pos,
                arm["dofs"], arm["qpos"], arm["q_rest"],
            )
            sols.append((arm, dq))
        for arm, dq in sols:
            data.qpos[arm["qpos"]] += dq
            clamp_qpos(model, data, arm["qpos"], arm["joint_ids"])
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    return {arm["side"]: data.qpos[arm["qpos"]].copy() for arm in arms}


def main() -> None:
    parser = argparse.ArgumentParser(description="双臂 IK 静态夹箱姿态。")
    parser.add_argument("--check", action="store_true", help="Only build and load the model")
    parser.add_argument("--headless", action="store_true", help="无 GUI：收敛 IK 后打印贴合报表")
    parser.add_argument("--iters", type=int, default=IK_ITERS, help="IK 收敛迭代上限")
    args = parser.parse_args()

    scene_xml = build_scene()
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)

    box_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_qpos = int(model.jnt_qposadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")])
    box_qvel = int(model.jnt_dofadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")])

    arms = []
    for cfg in ARM_CONFIGS:
        jac_body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["jacobi_body"])
        joint_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg["joint_names"]]
        dofs = [int(model.jnt_dofadr[j]) for j in joint_ids]
        qpos = [int(model.jnt_qposadr[j]) for j in joint_ids]
        # runtime yaw 后：左臂沿 +x、右臂沿 -x 伸出，夹持轴是 x
        sign = 1.0 if cfg["side"] == "left" else -1.0
        arms.append(
            {
                "side": cfg["side"],
                "sign": sign,
                "q_rest": cfg["q_rest"],
                "jac_body_id": jac_body_id,
                "joint_ids": joint_ids,
                "dofs": dofs,
                "qpos": qpos,
                "pad_id": name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"openarmx_{cfg['side']}_ee_pad"),
            }
        )

    def reset_robot() -> None:
        """双臂回休息位，箱子放回端面中点并下移 BOX_Z_OFFSET（姿态水平）。"""
        for arm in arms:
            data.qpos[arm["qpos"]] = arm["q_rest"].copy()
            clamp_qpos(model, data, arm["qpos"], arm["joint_ids"])
            data.qvel[arm["dofs"]] = 0.0
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_forward(model, data)

        mid = np.mean([get_ee_tip(data, arm["jac_body_id"]) for arm in arms], axis=0)
        data.qpos[box_qpos : box_qpos + 3] = mid + np.array([0.0, 0.0, BOX_Z_OFFSET])
        data.qpos[box_qpos + 3 : box_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[box_qvel : box_qvel + 6] = 0.0
        mujoco.mj_forward(model, data)

    def run_clamp_ik(iters: int) -> None:
        """纯运动学收敛到"平放对夹"姿态（复用模块级 solve_grasp_pose）。"""
        solve_grasp_pose(model, data, arms, box_id, iters)

    reset_robot()
    print("✅ 静态夹箱场景已加载")
    print(f"nq={model.nq} nv={model.nv} nbody={model.nbody} ngeom={model.ngeom}")
    print(f"箱子半尺寸={BOX_HALF} 全尺寸={2*BOX_HALF} 质量={BOX_MASS}kg")

    if args.check:
        return

    run_clamp_ik(args.iters)

    # ---------- 贴合报表 ----------
    def arm_box_penetrations(arm) -> list[tuple[str, int]]:
        """该臂所有碰撞网格中真正落在箱体 AABB 内的顶点几何（几何名, 顶点数）。空列表=完全净空。"""
        hx, hy, hz = BOX_HALF
        bx, by, bz = data.xpos[box_id]
        lo = np.array([bx - hx, by - hy, bz - hz])
        hi = np.array([bx + hx, by + hy, bz + hz])
        hits = []
        for g in range(model.ngeom):
            gname = model.geom(g).name or ""
            # 只统计碰撞几何（contype 默认=1），跳过 pad、手部、视觉副本
            if not gname.startswith(f"openarmx_{arm['side']}_") or "_ee_pad" in gname or "_visual_part" in gname:
                continue
            m = model.geom_dataid[g]
            if m < 0 or "collision" not in gname:
                continue
            verts = model.mesh_vert[model.mesh_vertadr[m] : model.mesh_vertadr[m] + model.mesh_vertnum[m]]
            xm = data.geom_xmat[g].reshape(3, 3)
            world = verts @ xm.T + data.geom_xpos[g]
            n_inside = int(np.all((world > lo) & (world < hi), axis=1).sum())
            if n_inside > 0:
                hits.append((gname, n_inside))
        return hits

    box_pos = data.xpos[box_id]
    for arm in arms:
        tip = get_ee_tip(data, arm["jac_body_id"])
        target = np.array(
            [box_pos[0] + arm["sign"] * (BOX_HALF[0] + GRIP_CLEARANCE), box_pos[1], box_pos[2]]
        )
        err = np.linalg.norm(tip - target)
        pad_xpos = data.geom_xpos[arm["pad_id"]]
        normal = data.xmat[arm["jac_body_id"]].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
        normal /= np.linalg.norm(normal)
        ddir = np.array([-arm["sign"], 0.0, 0.0])                 # 平放对夹：垫片指向箱内
        align = float(abs(float(normal @ ddir)))                      # 法向与"面对面"方向的对齐度
        x_in = float(normal @ np.array([-arm["sign"], 0.0, 0.0]))     # 指向箱内的夹持力 x 分量
        pad_front = pad_xpos + 0.004 * normal                         # 垫片前端面（接触面）位置
        face_x = box_pos[0] + arm["sign"] * BOX_HALF[0]              # 箱侧面所在 x 平面
        contact_gap = (pad_front[0] - face_x) * arm["sign"]           # 前端面到箱面距离：0=正好贴住
        pen = arm_box_penetrations(arm)
        pen_txt = "0（完全净空✓）" if not pen else "、".join(f"{n}:{c}顶点" for n, c in pen)
        print(
            f"[{arm['side']}] 端面误差={err*1000:.2f} mm | 垫片中心={pad_xpos.round(4)} "
            f"| 法向={normal.round(3)} 对齐度={align:.4f} 夹持x分量={x_in:.3f} "
            f"| 垫片前端贴面间隙={contact_gap*1000:.2f} mm | 链路网格穿箱顶点={pen_txt}"
        )
    print(f"箱子中心={box_pos.round(4)} | 半宽x={BOX_HALF[0]} | 垫片净空={GRIP_CLEARANCE*1000:.1f} mm")

    if args.headless:
        print("完成（静态夹箱姿态已收敛）。")
        return

    state = {"rest": False, "grasp": False}

    def key_callback(keycode: int) -> None:
        try:
            key = chr(keycode).upper()
        except ValueError:
            return
        if key == "R":
            state["rest"] = True
            print("👉 复位到休息位")
        elif key == "G":
            state["grasp"] = True
            print("👉 重新收敛夹箱姿态")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = [0.4, 0.0, 0.65]
        viewer.cam.distance = 2.6
        viewer.cam.elevation = -14
        viewer.cam.azimuth = 180
        mujoco.mj_forward(model, data)
        while viewer.is_running():
            if state["rest"]:
                reset_robot()
                state["rest"] = False
            elif state["grasp"]:
                run_clamp_ik(args.iters)
                state["grasp"] = False
            viewer.sync()


if __name__ == "__main__":
    main()