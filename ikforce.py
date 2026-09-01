#!/usr/bin/env python3
"""OpenArmX 双臂 力矩级笛卡尔位置控制 教学版 (ikforce.py)

========== 文件总览 ==========
从 ik.py 的"位置级 IK 驱动"升级到"力矩级位置控制"的第一步（不做论文、不做柔顺）:
双臂各自用 PD + 动力学补偿（计算力矩思路）在真实重力下立住，并追踪同一个目标点。

控制律（每臂、每控制周期）:
    τ = qfrc_bias + Jᵀ·Λ·(K_p·e_pos − K_d·J·q̇) − K_dq·q̇
        + N·[K_pq·(q_rest − q) − K_dq·q̇]      （零空间姿势任务，可 --no_nullspace 关闭）
- qfrc_bias = C + G，从 MuJoCo 数值读取，无需自己推导动力学
- Λ = (J·M⁻¹·Jᵀ)⁻¹ 是任务空间惯量矩阵（opspace 的 Mx）：
  任务力乘 Λ 后闭环 ẍ = K_p·e + K_d·ė，刚度/阻尼/轴间解耦全工作空间恒定
- 刚度 K_p 调硬：目标是把位置跟准，与"阻抗/柔顺"的让位行为相反
- N 是动力学一致的零空间投影 N = I − Jᵀ(J·M⁻¹·Jᵀ)⁻¹·J·M⁻¹：
  末端位置任务优先，剩余冗余自由度拿去向休息位靠（姿势"弹簧"）。
  原理同 ik.py 的 (I − J†J)，但力矩级必须用质量矩阵 M 加权，才能保证
  零空间力矩不扰动末端的力/运动（纯运动学投影在力矩级只是近似）。
- 对应论文式(7) baseline 的 u_comp + u_trac 两项，为后面加协同/力项打基础

运行:
    python ikforce.py --check                  # 只构建加载
    python ikforce.py                          # 打开 MuJoCo 被动 viewer，鼠标拖红球设目标
    python ikforce.py --headless --steps 3000  # 无 GUI：稳定保持 + 正弦轨迹跟踪验证

========== 文件总览 ==========
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from ik import (
    ARM_CONFIGS,
    clamp_qpos,
    clamp_target,
    generate_runtime_xml,
    get_ee_tip,
    name_to_id,
    write_scene,
)

# ---------- 控制参数 ----------
K_POS = 1500.0           # 笛卡尔位置刚度 (N/m)：硬刚度，追求跟踪精度
K_D = 30.0               # 笛卡尔速度阻尼 (N·s/m)：约 2·sqrt(K_POS)·ζ，ζ≈0.4
K_DQ = 0.8               # 附加关节粘性阻尼 (N·m·s/rad)，抑制关节抖动
K_PQ = 2.0               # 零空间姿势弹簧刚度 (N·m/rad)：把冗余关节拉向休息位
MAX_TASK_FORCE = 500.0   # 笛卡尔目标力软限幅 (N)：防止 Jᵀ 放大后超出关节力矩上限
MAX_NULL_TORQUE = 10.0   # 投影后零空间姿势力矩软限幅 (N·m)：防轻量臂上投影放大致饱和抖动
IK_DT = 0.002            # 主循环步长（与场景 timestep 一致）
GRAVITY = np.array([0.0, 0.0, -9.81])   # 力矩控制必须有真实重力

# 头部无负载：纯跟踪验证，不加箱子（加箱子是下一步"力控"的事）


def inject_motors(runtime_xml: Path) -> None:
    """给 runtime XML 注入 14 个力矩电机（左右各 7）。

    URDF 转出的模型只有关节（带 actuatorfrcrange 属性）没有 actuator，
    力矩控制必须有 motor，forcerange 直接沿用关节的 actuatorfrcrange。
    """
    tree = ET.parse(runtime_xml)
    root = tree.getroot()
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")

    for side in ("left", "right"):
        for i in range(1, 8):
            joint_name = f"openarmx_{side}_joint{i}"
            joint = root.find(f".//joint[@name='{joint_name}']")
            if joint is None:
                raise ValueError(f"runtime XML 中找不到关节 {joint_name}")
            frc = joint.get("actuatorfrcrange", "-120 120")
            ET.SubElement(
                actuator,
                "motor",
                {
                    "name": f"{joint_name}_motor",
                    "joint": joint_name,
                    "forcerange": frc,
                    "ctrlrange": frc,  # ctrl 同样限幅，避免注入超出量程的命令
                },
            )
    tree.write(runtime_xml, encoding="utf-8", xml_declaration=True)
    print("✅ 已注入 14 个力矩电机")


def strip_collisions(runtime_xml: Path) -> None:
    """移除自动生成的 *_collision 碰撞 geom。

    当前 URDF 转换是"实体即视觉"（每个视觉零件生成同名碰撞网格，供抓箱接触用），
    但 ikforce 是自由空间演示、无物体接触，这些全尺寸碰撞网格会造成大量
    自碰撞和残留手部碰撞，把双臂别在半路——这正是之前稳态残差 35mm 的来源。
    ee_pad（名称不含 _collision）保留。
    """
    tree = ET.parse(runtime_xml)
    root = tree.getroot()
    removed = 0
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("name", "").endswith("_collision"):
                body.remove(geom)
                removed += 1
    tree.write(runtime_xml, encoding="utf-8", xml_declaration=True)
    print(f"✅ 已移除 {removed} 个 *_collision 碰撞 geom（自由空间演示）")


def build_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """生成 runtime XML（含端面垫片 + 电机），写场景，加载并打开重力。"""
    runtime_xml = generate_runtime_xml()
    inject_motors(runtime_xml)
    strip_collisions(runtime_xml)
    scene_xml = write_scene(runtime_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    model.opt.gravity[:] = GRAVITY
    data = mujoco.MjData(model)
    return model, data


def solve_torque_position(
    model,
    data,
    arm: dict,
    target_pos: np.ndarray,
    use_nullspace: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """力矩级笛卡尔位置控制：PD + 重力/科氏补偿 (+零空间姿势任务)，返回该臂关节力矩。

    每臂独立求解（双臂无耦合），对应论文式(7) 的 u_comp + u_trac。
    零空间: 7 自由度冗余臂在 3 维位置任务下有 4 维冗余，
    用动力学一致投影 N = I − Jᵀ(J M⁻¹ Jᵀ)⁻¹ J M⁻¹ 把"往休息位靠"的
    姿势弹簧放进零空间，不干扰末端任务。
    """
    body_id = arm["body_id"]
    dofs = arm["dofs"]

    tip = get_ee_tip(data, body_id)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, tip, body_id)
    jac = jacp[:, dofs]                     # 平移雅可比 (3,7)

    # 质量矩阵 M：mj_fullM 从 data.M 展开，取该臂 7×7 块。
    # 基座固接、两臂自由度彼此独立 → M 按臂分块对角，块内即该臂惯量矩阵。
    m = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, m)
    mm = m[np.ix_(dofs, dofs)]
    m_inv = np.linalg.inv(mm)                        # 关节空间 M⁻¹ (7×7)

    # 任务空间(操作空间)惯量矩阵 Λ = (J·M⁻¹·Jᵀ)⁻¹ —— 即 opspace 的 Mx。
    # 任务力乘 Λ → 闭环 ẍ = K_p·e + K_d·ė，刚度/阻尼/各轴解耦全工作空间恒定。
    lam_inv = jac @ m_inv @ jac.T
    if np.linalg.det(lam_inv) > 1e-6:                # 良态直接求逆
        lam = np.linalg.inv(lam_inv)
    else:                                            # 近奇异用伪逆兜底（同 opspace pinv）
        lam = np.linalg.pinv(lam_inv, rcond=1e-6)

    e_pos = target_pos - tip                         # 位置误差（目标 − 当前位置）
    x_dot = jac @ data.qvel[dofs]                    # 末端速度
    force = lam @ (K_POS * e_pos - K_D * x_dot)      # 操作空间阻抗力：JᵀΛF（opspace 同款）

    f_norm = np.linalg.norm(force)
    if f_norm > MAX_TASK_FORCE and f_norm > 1e-9:
        force = force / f_norm * MAX_TASK_FORCE   # 软限幅，防 Jᵀ 放大爆力矩

    tau = (
        data.qfrc_bias[dofs].copy()        # C + G 前馈补偿
        + jac.T @ force                    # 任务空间 PD 映射到关节
        - K_DQ * data.qvel[dofs]           # 附加关节阻尼
    )

    if use_nullspace:
        # 动力学一致零空间投影: N = I − Jᵀ Λ J M⁻¹  (使 J·M⁻¹·N·τ=0)
        null_proj = np.eye(len(dofs)) - jac.T @ lam @ jac @ m_inv

        # 姿势"弹簧-阻尼": 关节拉向休息位
        tau_posture = (
            K_PQ * (arm["q_rest"] - data.qpos[arm["qpos"]])
            - K_DQ * data.qvel[dofs]
        )
        tau_null = null_proj @ tau_posture
        n_null = np.linalg.norm(tau_null)
        if n_null > MAX_NULL_TORQUE and n_null > 1e-9:
            # 轻量臂 M⁻¹ 数值很大，投影算子把姿势弹簧放大数十倍、顶到电机力上限，
            # 限幅反复翻转造成"抽动"。软限幅保住姿势任务又不烧限幅（同 MAX_TASK_FORCE 思路）
            tau_null = tau_null / n_null * MAX_NULL_TORQUE
        tau = tau + tau_null

    # 限幅交给写 ctrl 处（MuJoCo 内部按 ctrlrange 裁剪，与 forcerange 一致）。
    # 这里返回裁剪前的 tau，才能统计真实饱和（原先在函数内裁剪后再读 ctrl 永远测不到）
    return tau, e_pos


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenArmX 双臂力矩级笛卡尔位置控制教学版。")
    parser.add_argument("--check", action="store_true", help="Only build and load the model")
    parser.add_argument("--headless", action="store_true", help="无 GUI：稳定保持 + 正弦轨迹跟踪验证")
    parser.add_argument("--steps", type=int, default=4000, help="Headless 总步数")
    parser.add_argument("--no_nullspace", action="store_true", help="关闭零空间姿势任务（对照用）")
    args = parser.parse_args()

    # 零空间开关：全局标志，传给每步求解
    USE_NULLSPACE = not args.no_nullspace

    model, data = build_model()

    # ---------- 预解析 id ----------
    target_body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "ik_target")
    target_mocap_id = model.body_mocapid[target_body_id]

    arms = []
    for cfg in ARM_CONFIGS:
        body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["jacobi_body"])
        joint_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg["joint_names"]]
        dofs = [int(model.jnt_dofadr[j]) for j in joint_ids]
        qpos = [int(model.jnt_qposadr[j]) for j in joint_ids]
        act_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_motor") for n in cfg["joint_names"]]
        viz_body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["viz_body"])
        arms.append(
            {
                "side": cfg["side"],
                "q_rest": cfg["q_rest"],
                "body_id": body_id,
                "joint_ids": joint_ids,
                "dofs": dofs,
                "qpos": qpos,
                "act_ids": act_ids,
                "viz_mocap_id": model.body_mocapid[viz_body_id],
            }
        )

    def reset_robot() -> None:
        """复位到休息位，目标球放到两臂端面中点，速度清零。"""
        for arm in arms:
            data.qpos[arm["qpos"]] = arm["q_rest"].copy()
            clamp_qpos(model, data, arm["qpos"], arm["joint_ids"])
            data.qvel[arm["dofs"]] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)

        tips = [get_ee_tip(data, arm["body_id"]).copy() for arm in arms]
        target_pos = clamp_target(np.mean(tips, axis=0))
        data.mocap_pos[target_mocap_id] = target_pos
        for arm, tip in zip(arms, tips):
            data.mocap_pos[arm["viz_mocap_id"]] = tip
        mujoco.mj_forward(model, data)

    reset_robot()
    print("✅ 力矩级位置控制 ikforce 已加载")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")

    if args.check:
        return

    state = {"running": True, "pause": False, "reset": False}

    def key_callback(keycode: int) -> None:
        try:
            key = chr(keycode).upper()
        except ValueError:
            return
        if key == "R":
            state["reset"] = True
            print("👉 复位")
        elif key == "P":
            state["pause"] = not state["pause"]
            print(f"👉 控制 {'暂停' if state['pause'] else '继续'}")
        elif key == "Q":
            state["running"] = False
            print("👉 退出")

    def step_control(target_pos: np.ndarray) -> tuple[list, list]:
        """解力矩 -> 写 ctrl -> 推进动力学一步，返回 (末端误差列表, (臂,未裁剪力矩)列表)。"""
        mujoco.mj_forward(model, data)
        taus = []
        errs = []
        for arm in arms:
            tau, e = solve_torque_position(model, data, arm, target_pos, USE_NULLSPACE)
            taus.append((arm, tau))
            errs.append(e)
        for arm, tau in taus:
            data.ctrl[arm["act_ids"]] = tau
        mujoco.mj_step(model, data)
        return errs, taus

    # ---------- 主循环 ----------
    if args.headless:
        # 参考轨迹：先静止保持让末端收敛到中点，随后正弦来回，验证跟踪
        hold_steps = 1200
        settle_window = 400          # 保持阶段末段，用于统计稳态误差
        amp, freq = 0.05, 0.8
        start = data.mocap_pos[target_mocap_id].copy()
        errs = {arm["side"]: [] for arm in arms}
        posture = {arm["side"]: [] for arm in arms}
        sat_ratio = {arm["side"]: 0 for arm in arms}

        for t in range(args.steps):
            if t < hold_steps:
                ref = start.copy()
            else:
                ref = start + np.array([amp * np.sin(freq * (t - hold_steps) * IK_DT), 0.0, 0.0])
            ref = clamp_target(ref)
            data.mocap_pos[target_mocap_id] = ref

            e_list, raw_taus = step_control(ref)
            for (arm, tau_raw), e in zip(raw_taus, e_list):
                errs[arm["side"]].append(np.linalg.norm(e))
                posture[arm["side"]].append(np.linalg.norm(data.qpos[arm["qpos"]] - arm["q_rest"]))
                low, high = model.actuator_ctrlrange[arm["act_ids"]].T
                if np.any(tau_raw < low - 1e-9) or np.any(tau_raw > high + 1e-9):   # 对裁剪前的 tau 判饱和
                    sat_ratio[arm["side"]] += 1

        for arm in arms:
            side = arm["side"]
            arr = np.array(errs[side])
            settle = np.mean(arr[hold_steps - settle_window:hold_steps]) * 1000   # 稳态保持误差 (mm)
            mean_err = np.mean(arr[hold_steps:]) * 1000
            max_err = np.max(arr[hold_steps:]) * 1000
            posture_mean = np.mean(posture[side][hold_steps:])
            print(
                f"[{side}] 稳态保持={settle:.2f} mm | 跟踪误差 均值={mean_err:.2f} mm 最大={max_err:.2f} mm "
                f"| 姿势偏差 均值={posture_mean:.4f} rad "
                f"| 力矩饱和次数={sat_ratio[side]}/{args.steps}"
            )
        print("完成。")
        return

    last_debug = 0.0
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = [0.0, -0.02, 0.58]
        viewer.cam.distance = 2.25
        viewer.cam.elevation = -16
        viewer.cam.azimuth = 180

        while viewer.is_running() and state["running"]:
            if state["reset"]:
                reset_robot()
                state["reset"] = False
                viewer.sync()
                continue

            target_pos = clamp_target(data.mocap_pos[target_mocap_id].copy())
            data.mocap_pos[target_mocap_id] = target_pos

            if not state["pause"]:
                step_control(target_pos)
            else:
                mujoco.mj_forward(model, data)

            # 可视化球跟随端面
            for arm in arms:
                data.mocap_pos[arm["viz_mocap_id"]] = get_ee_tip(data, arm["body_id"])

            now = time.time()
            if now - last_debug > 0.5:
                errs = []
                for arm in arms:
                    tip = get_ee_tip(data, arm["body_id"])
                    errs.append(f"{arm['side']}={np.linalg.norm(target_pos - tip) * 1000:.1f}mm")
                print(f"[DEBUG] target={target_pos.round(4)} " + " ".join(errs))
                last_debug = now

            viewer.sync()
            time.sleep(IK_DT)


if __name__ == "__main__":
    main()