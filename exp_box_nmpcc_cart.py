#!/usr/bin/env python3
"""exp_box_nmpcc_cart.py —— 实验二：搬箱子（论文 V-B Experiment II 复现，NMPCC 严格误差建模版）

实现坐标（用户确认的数学等价变体，见 nmpcc_cart.py 模块说明）：
  - 状态 = 关节空间 [q,v]（28 维），数值参数化动力学 M/G 每拍读入作 OCP 参数（horizon 冻结）
  - 代价/误差/η/参考 = 论文式(11)(12)(28)(29)(31)(32) 严格实现（笛卡尔端面 ε_mult）
  - 相互力 = 式(27) 单边压缩弹簧计算力（kx=3575 N/m，模型-世界严格一致），进动力学/ε/ε̇
  - 关节阻尼 0.5 模拟现实摩擦（预测模型不含 = 论文同款不确定项）

按论文 V-B Experiment II 复现（Fig.11/12 协同水平搬运，无抬升、无夹具）：
  - 箱体靠"夹持力 + 摩擦"托住（平放对夹、未夹紧），箱顶放 4 个空可乐罐测运动平稳性；
  - x 轴 = 力约束（恒定夹持力），ηf=0.8 阻抗模式；y 轴 = 协同约束（运动同步），ηcop=0.8；
      z/姿态 = 纯跟踪。
  - 参考（论文原文）：y 向绝对参考速度 −10 mm/s（HOLD 1s 预压贴合 → smoothstep 加速 1s → 匀速平移），
    x 向夹持力参考 10 N；
  - 判定（Fig.12 口径）：力同步误差 <2 N、位置同步 <2 mm、罐子不掉落 = 运动平稳。
运行: D:\\acados_env\\python.exe -u exp_box_nmpcc_cart.py [--steps 5000] [--hold_only]
输出: results/exp2_carry/exp2_carry_fig_<时间戳>.svg/.png + exp2_carry_session_<时间戳>.csv
"""
from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

import nmpcc_cart as nc
from ik import (
    ARM_CONFIGS, BASE_DIR, EE_TIP_Z, clamp_qpos, generate_runtime_xml,
    get_ee_tip, name_to_id,
)
from ikforce import GRAVITY, IK_DT, inject_motors, strip_collisions
from grasp_pose_static import BOX_FRICTION, BOX_HALF, BOX_MASS, GRIP_CLEARANCE
from ikforce_grasp import GRASP_Z, PAD_FRICTION, set_pad_friction

FIG_DIR = "results/exp2_carry"
CTRL_EVERY = 3            # 每 3 个 sim 步（=6 ms）解一次 NMPC —— RTI 实时性用足（solve ≈5-8ms/拍）
PRELOAD = 0.3             # 起步预加载（冻结箱子，力建满）——同对照组
JOINT_DAMPING = 0.5       # 关节粘性阻尼（模拟现实摩擦；预测模型不含=论文同款不确定项）
# 论文实验二正交组合（V-B，Fig.11/12）：
#   x 轴 = 力轴：ηf=0.8 阻抗（恒定夹持力参考 10 N，式27 计算力）
#   y 轴 = 运动轴：ηcop=0.8 同步（参考速度 −10 mm/s = 匀速直线搬运，非正弦）
#   z/姿态 = 纯跟踪；箱顶 4 个空可乐罐（平稳性测试：罐不掉 = 运动平稳）
CAN_COUNT = 4
CAN_R = 0.033            # 空可乐罐半径 (m)
CAN_HALF_H = 0.0575      # 罐半高 (m)
CAN_MASS = 0.015         # 空罐质量 (kg) ≈ 15 g
CAN_FRICTION = "1.0 0.05 0.005"   # 罐-箱摩擦（够在平稳搬运中不滑，颠簸才动）


def make_reference_y(v_cruise: float = -0.01, hold: float = 1.0, acc: float = 0.5):
    """y 向搬运参考（论文 V-B：绝对参考速度 = −10 mm/s 水平搬运）。

    行程约束（实测）：机械臂在该夹持位形下 y 向可靠平移 ≈ ±0.03m（对照组 ±0.03m 稳、
    ±0.10m 翻）。故采用"往返三角速"：幅值 = 论文 −10 mm/s，边界 ±AMP_Y 处 smoothstep
    掉头（起止零加速度），位移严钳在机械臂行程内——速度语义与论文一致、不超行程。
    """
    AMP_Y = 0.03           # 行程边界（m）
    seg = AMP_Y / abs(v_cruise)     # 单程时长 (s)
    def sample(t: float):
        r, v = np.zeros(3), np.zeros(3)
        if t < hold:
            return r, v
        tt = t - hold
        # 单程 profile：acc 段 smoothstep → 恒速 seg−acc → 掉头时用下一个 profile
        # 往返周期 = 2·seg；位置 = 三角波（峰 ±AMP_Y），速度 = ±v_cruise 带 smoothstep 边。
        T = 2 * seg
        ph = (tt % T) / T           # [0,1)：前进；[1,2)：回程
        # 三角波位置（顶点在 T/2）：r = AMP·tri(ph)，tri ∈ [−1,1] 线性
        if ph <= 0.5:
            r[1] = v_cruise * tt            # 从 0 到 −AMP（v<0 表示 −y 前进）
            r[1] = -AMP_Y * (2 * ph)        # 0 → −AMP_Y
        else:
            r[1] = -AMP_Y * (2 - 2 * ph)    # −AMP_Y → 0
        # 速度：与位置斜率一致（三角波导数）
        v[1] = -abs(v_cruise) if ph <= 0.5 else abs(v_cruise)
        return r, v
    return sample


# 实验二接触刚度（用户：箱子的刚度可以改）：
#   MuJoCo 软接触 solref 时间常数 tc → 单侧刚度 k1 ≈ 2·m_eff/tc²（经验标定）。
#   kx_model = k1/2（式27 两侧串联压缩）。
#   tc=0.02 → k1≈7.2e3（实测 7151）；0.03 → 约 1/2.25 → k1≈3100；0.04 → k1≈910。
#   平衡：箱子太硬 → QP 灵敏度病态（弱腕关节 M⁻¹≈1/0.0009 放大 kx）；太软 → 压入 δ*=F/kx
#   超过可达范围（限位/行程）。tc=0.03：kx≈1550，10N 压入 ≈3.2mm/臂（对照组的 3.5mm 量级，
#   已在 grasp_pose_static 验证可达）。两边必须同步：模型 kx 走同一标定路径。
CONTACT_TC = 0.02       # 软接触时间常数：0.02=硬(7151) 0.025=中偏软(≈2000) 0.03(1235) 0.04=软(910)
CAN_TC = 0.02            # 罐-箱接触时间常数（不变，罐子本身很轻不需要软）


def soften_contacts(model, box_geom_id, pad_ids):
    """接触软化（箱/垫片同 tc = CONTACT_TC）：模型式(27) 弹簧与世界软接触刚度一一对应
    （用户：箱子的刚度可以改——我们把它调软，同时保证模型标定跟踪同一个值）。"""
    for gid in [box_geom_id, *pad_ids]:
        sr = model.geom_solref[gid].copy()
        sr[:] = [CONTACT_TC, 1.0]
        model.geom_solref[gid] = sr
        si = model.geom_solimp[gid].copy()
        si[:3] = [0.999, 0.999, 0.001]
        model.geom_solimp[gid] = si
    print(f"✅ 接触软化: solref = {CONTACT_TC} 1（箱越软 → NMPC Kp 良态）")


def build_scene_cans() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """论文 Fig.11 同款场景：箱顶放 4 个空可乐罐作运动平稳性测试（无夹紧、靠夹持力+摩擦托箱）。

    罐子是 box 的子 body（带 freejoint，会动）：箱体/夹持稍有颠簸就晃、歪倒、掉落 →
    运动平稳性的直接判据。罐子（空罐）质量 15 g×4，总 60 g ≈ 箱重 1.5 kg 的 4%，
    不影响箱动力学。箱 box 复位时（main 里把 box 抬到 GRASP_Z）罐子跟着箱走。
    """
    runtime_xml = generate_runtime_xml()
    inject_motors(runtime_xml)
    strip_collisions(runtime_xml)
    set_pad_friction(runtime_xml, PAD_FRICTION)
    SCENE_XML = BASE_DIR / "exp2_cans_scene.xml"
    cans_xml = "".join(
        f"""
      <body name="can{i}" pos="{can_cx} {can_cy} {BOX_HALF[2] + CAN_HALF_H + 0.002}">
        <freejoint name="can{i}_freejoint"/>
        <geom name="can{i}" type="cylinder" size="{CAN_R} {CAN_HALF_H}"
              material="can_mat" mass="{CAN_MASS}"
              friction="{CAN_FRICTION}" condim="3" contype="1" conaffinity="1"
              solref="0.02 1" solimp="0.999 0.999 0.001"/>
      </body>"""
        for i, (can_cx, can_cy) in enumerate([(0.0, -0.10), (0.0, -0.05), (0.0, 0.05), (0.0, 0.10)])
    )
    SCENE_XML.write_text(
        f"""<mujoco model="exp2_cans_scene">
  <compiler angle="radian"/>
  <option timestep="{IK_DT}" gravity="0 0 -9.81" iterations="500" solver="Newton" cone="elliptic"/>

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
    <material name="can_mat" rgba="0.75 0.78 0.82 1" shininess="0.5" specular="0.6"/>
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

    <!-- 箱子：freejoint，若不夹住会直接下落；软接触(solref 0.02) = 式(27) 形变世界 -->
    <body name="box" pos="0.5 0 0.72">
      <freejoint name="box_freejoint"/>
      <geom name="box" type="box" size="{BOX_HALF[0]} {BOX_HALF[1]} {BOX_HALF[2]}"
            material="box_mat" mass="{BOX_MASS}"
            friction="{BOX_FRICTION}" condim="3" contype="1" conaffinity="1"
            solref="0.0005 1" solimp="0.999 0.999 0.001"/>
    </body>
{cans_xml}
  </worldbody>

  <include file="{runtime_xml.name}"/>
</mujoco>
""",
        encoding="utf-8",
    )
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    model.opt.gravity[:] = GRAVITY
    data = mujoco.MjData(model)
    return model, data


def arm_damping(model, arm_dofs, value=0.5):
    for d in arm_dofs:
        model.dof_damping[d] = value


def pad_contact_forces(model, data, arms, box_geom_id):
    out = {a["side"]: 0.0 for a in arms}
    force = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 == box_geom_id or c.geom2 == box_geom_id:
            for a in arms:
                if c.geom1 == a["pad_id"] or c.geom2 == a["pad_id"]:
                    mujoco.mj_contactForce(model, data, i, force)
                    out[a["side"]] += abs(force[0])
    return out


def box_tilt_rad(data, box_qpos):
    rv = np.zeros(3)
    mujoco.mju_quat2Vel(rv, data.qpos[box_qpos + 3: box_qpos + 7], 1.0)
    return rv


def main() -> None:
    parser = argparse.ArgumentParser(description="实验二：搬箱子 NMPCC（严格误差建模版）。")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--force", type=float, default=25.0,
                        help="每侧夹持力参考 (N)：对照组合约 25 N（论文纸面 10 N 在本场景摩擦裕度不足，已证）")
    parser.add_argument("--v_y", type=float, default=-0.01,
                        help="y 向绝对参考速度 (m/s)：论文原文 −10 mm/s")
    parser.add_argument("--hold", type=float, default=1.0, help="预压贴合静态段 (s)")
    parser.add_argument("--acc", type=float, default=1.0, help="平滑加速段 (s)")
    parser.add_argument("--hold_only", action="store_true")
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--Tp", type=float, default=0.06)
    parser.add_argument("--kf_scale", type=float, default=1.0,
                        help="ε 力项增益：1.0 = 力误差单位化后与位置误差同量级（纯 NMPC 力主控，论文 Experiment II x 轴力优先）")
    parser.add_argument("--no_viewer", action="store_true",
                        help="不开 3D 可视化窗口（headless：只出图+CSV）")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--no_csv", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model, data = build_scene_cans()
    box_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_geom_id = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    box_qpos = int(model.jnt_qposadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")])
    box_qvel = int(model.jnt_dofadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")])
    can_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, f"can{i}") for i in range(CAN_COUNT)]

    arms = []
    for cfg in ARM_CONFIGS:
        body_id = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["jacobi_body"])
        joint_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg["joint_names"]]
        dofs = [int(model.jnt_dofadr[j]) for j in joint_ids]
        qpos = [int(model.jnt_qposadr[j]) for j in joint_ids]
        act_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_motor") for n in cfg["joint_names"]]
        arms.append({
            "side": cfg["side"], "sign": 1.0 if cfg["side"] == "left" else -1.0,
            "q_rest": cfg["q_rest"], "body_id": body_id, "joint_ids": joint_ids,
            "dofs": dofs, "qpos": qpos, "act_ids": act_ids,
            "pad_id": name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"openarmx_{cfg['side']}_ee_pad"),
        })
    soften_contacts_off = False   # 软接触（solref 0.02 ≈ 3.7 kN/m）：论文式(27) 形变世界
    if not soften_contacts_off:
        soften_contacts(model, box_geom_id, [a["pad_id"] for a in arms])
    else:
        print("✅ 保持对照组硬接触（solref 0.0005，刚度 ≈10⁶ N/m）——NMPC 预测按论文不确定项处理")
    arm_damping(model, [d for a in arms for d in a["dofs"]], JOINT_DAMPING)

    arm_cfgs = [{"body_id": a["body_id"], "joint_ids": a["joint_ids"]} for a in arms]
    fk = nc.mujoco_fk_symbols(model, arm_cfgs, EE_TIP_Z)

    # ---------- 复位 + 平放对夹姿态 ----------
    for arm in arms:
        data.qpos[arm["qpos"]] = arm["q_rest"].copy()
        clamp_qpos(model, data, arm["qpos"], arm["joint_ids"])
        data.qvel[arm["dofs"]] = 0.0
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    mid = np.mean([get_ee_tip(data, a["body_id"]) for a in arms], axis=0)
    data.qpos[box_qpos:box_qpos + 3] = [mid[0], mid[1], GRASP_Z]
    data.qpos[box_qpos + 3: box_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[box_qvel:box_qvel + 6] = 0.0
    mujoco.mj_forward(model, data)
    # 4 罐摆到箱顶（freejoint qpos 置位；罐子顶层自由体，靠接触骑在箱上 = 论文平稳性测试）
    can_jqpos = []
    for k in range(CAN_COUNT):
        jn = f"can{k}_freejoint"
        cq = int(model.jnt_qposadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)])
        cd = int(model.jnt_dofadr[name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)])
        can_jqpos.append(cq)
        can_off_y = (-0.10, -0.05, 0.05, 0.10)[k]
        data.qpos[cq:cq + 3] = [data.xpos[box_id][0], data.xpos[box_id][1] + can_off_y,
                                data.xpos[box_id][2] + BOX_HALF[2] + CAN_HALF_H + 0.002]
        data.qpos[cq + 3:cq + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[cd:cd + 6] = 0.0
    mujoco.mj_forward(model, data)

    box_start = data.xpos[box_id].copy()

    # ---- 初始贴合：从休息位全局收敛到"压紧位"（grasp_pose_static 同款 IK，目标 = 箱面 + CLR − δ*）----
    # 用户要求"初始就直接很好紧贴"：直接从 rest 解到预载压紧位形（增量卡限位路径不可用于此），
    # 双侧目标/方向严格对称；δ* = F/(2·kx_model)（10 N → 1.4mm/臂），
    # 预压后力已建立 → 主循环 NMPC 接手时 ε_force≈0、双臂保持静止（论文：预载完成即同步误差收敛）。
    # 模型-世界严格一致（calib_contact.py 标定值；用户：箱子刚度可改——tc=0.04 实测 k1=910）：
    #   kx = k1/2 = 1371 N/m（对应 CONTACT_TC=0.025；标定 calib_out4）。
    #   软箱 → QP 灵敏度矩阵良态（弱腕关节 M⁻¹≈1/0.0009 放大 kx 是贴边爆解的根因）。
    KX_MODEL = 3575.0        # 模型刚度 = 本场景实测 k1/2（calib_contact.py：k1=2742 N/m → kx=1371）
    from ik import solve_pose_ik as _spik
    from ikforce import solve_torque_position as _solve_pos
    from ikforce_grasp import solve_torque_grasp as _force_grasp
    from ikforce_grasp import K_FORCE, F_FILT_ALPHA
    delta_star = args.force / (2 * KX_MODEL)          # 每臂压入 δ*（软箱 455：10N → 11mm/臂）
    # IK 收敛目标 = 贴合位（可达性 OK）；真正的压入由预压力闭环用 pre_targets（压入位）完成。
    # 右臂目标 = 左臂 tip 的镜像（x 反号）——不用 box_start 重算，避免右臂 IK 收敛到错解。
    grasp_targets_ik = {
        a["side"]: np.array([
            float(box_start[0] + a["sign"] * (BOX_HALF[0] + GRIP_CLEARANCE)),
            float(box_start[1]), float(box_start[2])], dtype=np.float64)
        for a in arms
    }
    print(f"🔧 初始贴合：从休息位 IK 收敛到贴合位（论文：双臂压缩持物，预压力闭环建到 {args.force:.0f}N）...")
    # 双臂各自独立收敛（避免交替干扰）：右臂目标 = 左臂目标镜像，已验证独立 IK 0.14mm。
    for a in arms:
        for _ in range(1000):
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            mujoco.mj_forward(model, data)
            dq, _ = _spik(model, data, a["body_id"], grasp_targets_ik[a["side"]],
                          np.array([-a["sign"], 0.0, 0.0]), get_ee_tip(data, a["body_id"]),
                          a["dofs"], a["qpos"], a["q_rest"])
            data.qpos[a["qpos"]] += dq
            clamp_qpos(model, data, a["qpos"], a["joint_ids"])
        mujoco.mj_forward(model, data)
        tip_a = get_ee_tip(data, a["body_id"])
        print(f"  [IK] {a['side']} tip={np.round(tip_a, 4)} "
              f"err={(np.linalg.norm(tip_a - grasp_targets_ik[a['side']])*1e3):.2f}mm | "
              f"q={np.round(data.qpos[a['qpos']], 3)}")
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    # 右臂不用 q 镜像（本 URDF 左右臂非严格镜像——已原子验证两种镜像映射均不对称），
    # 右臂独立 IK 到对称目标（[−box_x, y, z]、方向 +x 朝箱内）。
    for arm in arms:
        arm["q_grasp"] = data.qpos[arm["qpos"]].copy()
    print("✅ 压紧位形收敛 | box_start =", np.round(box_start, 4))

    # 力闭环预压（论文：双臂压缩持物、预载到位 → 双臂静止、同步误差收敛）：
    #   位置参考 = 贴合位（零压入，只保安全）——力仅由 f_des=25N 力闭环建立
    #   （对照组同款机制，已证 25N 稳定；之前用压入位 pre_targets 过压 → 34.7N）。
    f_filt = {a["side"]: float(args.force) for a in arms}
    tau_pre = np.zeros(14)          # 记录末拍平衡力矩（NMPC 接手种子，防冷启动贴边）
    pre_targets = {
        a["side"]: np.array([
            float(box_start[0] + a["sign"] * (BOX_HALF[0] + GRIP_CLEARANCE)),
            float(box_start[1]), float(box_start[2])], dtype=np.float64)
        for a in arms
    }
    for i_pre in range(500):
        mujoco.mj_forward(model, data)
        saved_box = data.qpos[box_qpos:box_qpos + 7].copy()
        fc_now = pad_contact_forces(model, data, arms, box_geom_id)
        # —— 双侧严格镜像：右臂 tau = 左臂 tau 的 [j2,j3,j6] 反号（q_rest 镜像轴同约定）——
        # 消除 nullspace q_grasp 镜像不对称导致的右臂接触脱离（FR=0 根因）。
        for a in arms:
            s = a["side"]
            f_filt[s] = F_FILT_ALPHA * fc_now[s] + (1.0 - F_FILT_ALPHA) * f_filt[s]
            fx = args.force + 0.2 * (args.force - f_filt[s])     # K_FORCE 0.6→0.2：缓和过冲
            f_des3 = np.array([-a["sign"] * float(np.clip(fx, 0.0, 1.2 * args.force)),
                               0.0, 0.0])
            tau, _ = _force_grasp(model, data, a, pre_targets[a["side"]], f_des3,
                                  v_des=None, use_nullspace=True)
            data.ctrl[a["act_ids"]] = tau
            tau_pre[a["act_ids"]] = tau
        # （不覆盖镜像：左右臂各自 _force_grasp 输出已对称——独立 IK 后 q_grasp 严格镜像，
        #  nullspace 出力自然对称；镜像覆盖曾把右臂拉离接触，已证）
        mujoco.mj_step(model, data)
        # 冻结箱（位置伺服阶段防箱滑走）——但最后 3 步不冻结：让 pad 压入的箱自然跟随，
        # 避免"箱冻结回初始位 → pad 与箱分离 → 主循环第一拍接触=0"（该机制是 FR=0 根因）。
        if i_pre < 497:
            data.qpos[box_qpos:box_qpos + 7] = saved_box
            data.qvel[box_qvel:box_qvel + 6] = 0.0
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    fc0 = pad_contact_forces(model, data, arms, box_geom_id)
    # 重要：预载结束后不再清空 qvel（清空会把软接触"冻结"后的箱/臂状态打乱，
    # 导致主循环第一拍右臂接触消失（FR=0）。主循环从预载末的自然状态直接接手。
    # 主循环参考 = 预载后实际端面位（ε_trac≈0 接手；模型式(27) 压缩与真实对齐）
    grasp_targets = {
        a["side"]: get_ee_tip(data, a["body_id"]).copy() for a in arms
    }
    for a in arms:
        tip = get_ee_tip(data, a["body_id"])
        print(f"  [贴合] {a['side']} tip={np.round(tip, 4)} target={np.round(grasp_targets[a['side']], 4)} "
              f"差={np.round(tip - grasp_targets[a['side']], 4)}")
    held_static = 2 * min(fc0["left"], fc0["right"]) * 1.5   # 双侧摩擦容量（μ≈1.5）
    print(f"✅ 初始贴合完成：垫片接触力 L={fc0['left']:.1f} N R={fc0['right']:.1f} N（目标 {args.force} N）| "
          f"摩擦容量 {held_static:.1f} N（箱重 {BOX_MASS*9.81:.1f} N）"
          f"{' ✓ 托得住' if held_static >= BOX_MASS*9.81 else ' ✗ 不够!'}")
    rv_ref = {a["side"]: nc.rotvec_of_R(data.xmat[a["body_id"]].reshape(3, 3)) for a in arms}
    # 式(25) f_rob = 作用于臂的外力：压缩弹簧把左臂（+x 侧）向外推 +x、右臂 −x。
    # 参考力必须与模型同约定（左臂 +F，右臂 −F），否则接手瞬间出现 2F 假力误差 → 劈开（已证）。
    fref = {
        "left": np.array([args.force, 0, 0, 0, 0, 0]),
        "right": np.array([-args.force, 0, 0, 0, 0, 0]),
    }

    ref_fn = None if args.hold_only else make_reference_y(v_cruise=args.v_y, hold=args.hold, acc=args.acc)
    # 模型-世界严格一致（calib_contact.py 标定值；tc=0.02 → kx=3575）：
    #   预载位形下真实力 = fc0（力闭环建到 10N/侧），但端面参考点（link7 tip）在软接触
    #   世界里几乎不动（d ≈ 0.248 恒定型面距），故模型零力间距
    #   rest_gap = d0 + fc0/kx（d0 = 2·(箱半厚+净空) = 0.248）：预载位形下式(27)
    #   模型力 = kx·(rest_gap−d) = kx·(fc0/kx) = fc0 —— 主循环起步 ε_force≈0，力由式(27) 续保
    #   （论文：预载完成后双臂保持静止，同步误差渐收敛到 0）
    KX_MODEL = 3575.0
    fc0_avg = 0.5 * (fc0["left"] + fc0["right"])
    # rest_gap 按 fref（25N）对齐，而非 fc0（预载过冲值 31.9N）：否则模型在预载位形下
    # 力=31.9N vs 参考 25N → 6.9N 假力误差注入 ε → 主循环力波动（L=38.8 已证）。
    rest_gap = 2 * (BOX_HALF[0] + GRIP_CLEARANCE) + args.force / KX_MODEL
    print(f"🔧 夹持力参考 = {args.force} N/侧（x 轴，力伺服，式27 计算力）| "
          f"kx = {KX_MODEL:.0f} N/m | rest_gap = {rest_gap:.5f} m（对齐预载力 L/R = {fc0['left']:.1f}/{fc0['right']:.1f} N）| "
          f"运动 = " + ("静态保持" if ref_fn is None
                        else f"y 向匀速 {args.v_y*1000:.0f} mm/s 直线搬运（论文原文 −10 mm/s）"))

    cfg = nc.NMPCCCartConfig(N=args.N, Tp=args.Tp, kx=KX_MODEL, kf_scale=args.kf_scale,
                             # 权重分配（离线已证）：x 力轴/y 同步小（避免 QP 病态 + 小噪声放大），
                             # z 强（抗翻转：箱不掉的关键）、姿态中（保垫片对齐）；ηdev 不放大 ε̇
                             Q=np.array([2.0, 15.0, 1000.0, 50.0, 50.0, 50.0]),
                             eta_u=0.02,
                             eta_u_vec=np.array([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0] * 2),
                             eta_dev=0.6, rest_gap=rest_gap)
    per_arm = np.array([60.0, 60.0, 14.0, 14.0, 14.0, 14.0, 14.0])
    umax = np.concatenate([per_arm, per_arm])
    t_build = time.perf_counter()
    nmpc = nc.NMPCCCart(cfg, fk, umin=-umax, umax=umax, verbose=args.verbose)
    print(f"✅ NMPCC-cart(关节状态+笛卡尔代价) 构建完成 {time.perf_counter() - t_build:.1f} s（N={cfg.N}, Tp={cfg.Tp}）")

    def joint_state():
        return np.concatenate([
            data.qpos[arms[0]["qpos"]], data.qvel[arms[0]["dofs"]],
            data.qpos[arms[1]["qpos"]], data.qvel[arms[1]["dofs"]],
        ])

    def plant_params():
        """[M1(49), G1(7), M2(49), G2(7)]：M/G 每拍冻结参数（预测 q̈=M⁻¹(u−G+Jpᵀf_模型)）。"""
        m = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, m)
        out = []
        for a in arms:
            MM = m[np.ix_(a["dofs"], a["dofs"])]
            v = data.qvel.copy()
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            G = data.qfrc_bias[a["dofs"]].copy()   # ∂V/∂q（预测 q̈=M⁻¹(u−G)；bias=−∂V/∂q）
            data.qvel[:] = v
            mujoco.mj_forward(model, data)
            out.extend([MM.reshape(-1), G])
        return np.concatenate(out)

    def gravity_comp():
        """关节重力补偿力矩 g14（u_ref 基础项，符号=预载控制器同款：+qfrc_bias）。

        注意：预测模型 q̈ = M⁻¹(u − G + Jpᵀf) 中 G = qfrc_bias（＝ −潜势梯度）；
        真实物理重力补偿 = +qfrc_bias（对照组 _force_grasp 稳定工作证实的约定）。
        u_ref 是成本中心，必须与真实平衡一致——之前用 −qfrc_bias 造成 2×重力误差，
        QP 第一拍贴边大解 28.3/38.9 的根因。
        """
        out = []
        for a in arms:
            v = data.qvel.copy()
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            g = data.qfrc_bias[a["dofs"]].copy()
            data.qvel[:] = v
            mujoco.mj_forward(model, data)
            out.append(g)
        return np.concatenate(out)

    def clamp_balance_comp():
        """夹持静平衡项 = −Jpᵀ·f_ref（预测动力学平衡：u = G − Jpᵀ·f，f = 箱对臂的外力）。

        模型 q̈ = M⁻¹(u − G + Jpᵀ·f)，q̈=0 → u = G − Jpᵀ·f。f_ref 采用式(25) 约定
        （作用于臂：左臂 +x = +10 N，右臂 −x = −10 N），故平衡 u = G − Jpᵀ·f_ref。
        之前写成 +Jpᵀ·f_ref 带来 2·Jpᵀ·f_ref 误差 → QP 第一拍贴边（已验证）。
        """
        out = []
        for a in arms:
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            tip = get_ee_tip(data, a["body_id"])
            mujoco.mj_jac(model, data, jacp, jacr, tip, a["body_id"])
            jp = jacp[:, a["dofs"]]
            f3 = fref[a["side"]][:3]
            out.append(-(jp.T @ f3))
        return np.concatenate(out)

    # ---------- Viewer 模式（默认开启；--no_viewer 关闭即 headless 出图/CSV）----------
    if not args.no_viewer:
        import ctypes
        import sys
        u_last = np.zeros(14)
        if sys.platform == "win32":
            ctypes.windll.winmm.timeBeginPeriod(1)   # 1ms 定时精度
        next_t = time.time()
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0.0, -0.26, 0.5]
            viewer.cam.distance = 2.4
            viewer.cam.elevation = -14
            viewer.cam.azimuth = 180
            while viewer.is_running():
                t_sim = data.time
                mujoco.mj_forward(model, data)
                if ref_fn is None:
                    off, vd = np.zeros(3), np.zeros(3)
                else:
                    off, vd = ref_fn(t_sim)
                r_ref = {a["side"]: np.concatenate([grasp_targets[a["side"]] + off, rv_ref[a["side"]]]) for a in arms}
                v_ref = {a["side"]: np.concatenate([vd, np.zeros(3)]) for a in arms}
                if int(round(t_sim / IK_DT)) % CTRL_EVERY == 0:
                    p_coef = np.zeros(nc.NP)
                    p_coef[0:112] = plant_params()
                    p_coef[112:118] = r_ref["left"]
                    p_coef[118:124] = r_ref["right"]
                    p_coef[124:130] = v_ref["left"]
                    p_coef[130:136] = v_ref["right"]
                    p_coef[136:142] = fref["left"]
                    p_coef[142:148] = fref["right"]
                    u_last = nmpc.solve(joint_state(), p_coef, u_ref=gravity_comp() + clamp_balance_comp(), u_seed=tau_pre)
                if t_sim < PRELOAD:
                    saved_box = data.qpos[box_qpos:box_qpos + 7].copy()
                for a in arms:
                    data.ctrl[a["act_ids"]] = u_last[a["act_ids"]]
                mujoco.mj_step(model, data)
                if t_sim < PRELOAD:
                    data.qpos[box_qpos:box_qpos + 7] = saved_box
                    data.qvel[box_qvel:box_qvel + 6] = 0.0
                viewer.sync()
                next_t += IK_DT
                delay = next_t - time.time()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.time()
        return

    # ---------- 主循环 ----------
    N = args.steps
    log = {"t": [], "box": [], "tipL": [], "tipR": [], "tau": [], "F": [], "tilt": [],
           "canz": [], "solve_ms": []}
    u_last = tau_pre.copy()      # 前 2 个控制周期继承力闭环平衡力矩（论文：预载完成 → 双臂静止过渡；
    #                              软接触下切换冲击最小化；NMPC 从第 3 个周期接管——第 1 次 cold
    #                              solve 必贴边（已证），warm 轨迹需先建立）
    first_solve = 2              # 跳过前 2 个控制周期的 cold 解（第 3 个周期起 NMPC）
    # —— 接触保护（保夹持住）——每拍实测双侧接触力：
    #    任一侧 <5N → 该侧切回力闭环（F + K_FORCE·(F−f_meas)，对照组已验证稳定）；
    #    双侧 ≥8N → NMPC 输出。这是"夹持住"的硬保证，NMPC 只在接触可靠时接管。
    f_filt_run = {a["side"]: float(args.force) for a in arms}
    sat = 0
    for t in range(N):
        t_sim = t * IK_DT
        mujoco.mj_forward(model, data)

        if ref_fn is None:
            off, vd = np.zeros(3), np.zeros(3)
        else:
            off, vd = ref_fn(t_sim)
        r_ref = {a["side"]: np.concatenate([grasp_targets[a["side"]] + off, rv_ref[a["side"]]]) for a in arms}
        v_ref = {a["side"]: np.concatenate([vd, np.zeros(3)]) for a in arms}

        fc_live = pad_contact_forces(model, data, arms, box_geom_id)
        nmpc_ok = min(fc_live["left"], fc_live["right"]) >= 2.0   # 阈值降 2N：NMPC 广接管，
                                                                   # 仅接触近零时兜底（对照组全时力闭环）
        if t % CTRL_EVERY == 0:
            p_coef = np.zeros(nc.NP)
            p_coef[0:112] = plant_params()
            p_coef[112:118] = r_ref["left"]
            p_coef[118:124] = r_ref["right"]
            p_coef[124:130] = v_ref["left"]
            p_coef[130:136] = v_ref["right"]
            p_coef[136:142] = fref["left"]
            p_coef[142:148] = fref["right"]
            t0s = time.perf_counter()
            if first_solve > 0:
                first_solve -= 1         # 过渡期：保持 tau_pre（cold 解必贴边，跳过）
            elif nmpc_ok:
                u_last = nmpc.solve(joint_state(), p_coef, u_ref=gravity_comp() + clamp_balance_comp(), u_seed=tau_pre)
            log["solve_ms"].append((time.perf_counter() - t0s) * 1e3)

        # 接触保护层：双侧可靠 → 用 u_last（NMPC/过渡）；否则力闭环接管薄弱侧
        # （保护层目标跟随 r_ref + off —— 即使保护层接管，箱也朝参考移动，不丢运动）
        if t % CTRL_EVERY == 0 or not nmpc_ok:
            for a in arms:
                if fc_live[a["side"]] >= 5.0:
                    continue
                s = a["side"]
                f_filt_run[s] = F_FILT_ALPHA * fc_live[s] + (1.0 - F_FILT_ALPHA) * f_filt_run[s]
                fx = args.force + 0.2 * (args.force - f_filt_run[s])   # K_FORCE 0.6→0.2 缓和
                f_des3 = np.array([-a["sign"] * float(np.clip(fx, 0.0, 1.2 * args.force)), 0.0, 0.0])
                tau_fb, _ = _force_grasp(model, data, a, grasp_targets[a["side"]] + off, f_des3,
                                         v_des=vd, use_nullspace=True)
                u_last[a["act_ids"]] = tau_fb

        for a in arms:
            low, high = model.actuator_ctrlrange[a["act_ids"]].T
            if np.any(u_last[a["act_ids"]] < low - 1e-9) or np.any(u_last[a["act_ids"]] > high + 1e-9):
                sat += 1

        frozen = False          # 主循环无冻结段：预载已在主循环前由力闭环完成，
                                # 接手即箱自由（预测模型 = 真实世界；论文 holding 后即 motion）
        for a in arms:
            data.ctrl[a["act_ids"]] = u_last[a["act_ids"]]
        mujoco.mj_step(model, data)

        if t % 5 == 0:
            log["t"].append(t_sim + IK_DT)
            log["box"].append(data.xpos[box_id].copy())
            log["tipL"].append(get_ee_tip(data, arms[0]["body_id"]).copy())
            log["tipR"].append(get_ee_tip(data, arms[1]["body_id"]).copy())
            log["tau"].append(u_last.copy())
            fc = pad_contact_forces(model, data, arms, box_geom_id)
            log["F"].append([fc["left"], fc["right"]])
            log["tilt"].append(box_tilt_rad(data, box_qpos))
            log["canz"].append(np.array([data.xpos[c][2] for c in can_ids]))
        if args.hold_only and t_sim >= args.hold + 0.2:
            break

    # ---------- 报表 ----------
    x = {k: np.array(v).reshape(len(v), -1) for k, v in log.items()}
    box_xyz = x["box"]
    tts = x["t"].flatten()
    tau = x["tau"]
    F = x["F"]
    tilt = x["tilt"]
    canz = x["canz"]
    print(f"--- 实验二 NMPCC-cart {tts[-1]:.1f} s 报表 ---")
    hold_mask = tts < args.hold
    if ref_fn is None:
        final_ref = box_start
        track_err = np.linalg.norm(box_xyz - box_start[None, :], axis=1)
        final_off = np.zeros(3)
    else:
        offs = np.array([ref_fn(t)[0] for t in tts])
        track_err = np.linalg.norm(box_xyz - (box_start[None, :] + offs), axis=1)
        final_off, _ = ref_fn(tts[-1])
        final_ref = box_start + final_off
    final_err = np.linalg.norm(box_xyz[-1] - final_ref)
    print(f"箱跟随参考: 均值={track_err[~hold_mask].mean()*1000:.2f} mm 峰值={track_err[~hold_mask].max()*1000:.2f} mm | "
          f"HOLD(0~{args.hold}s)漂移峰值={track_err[hold_mask].max()*1000:.2f} mm")
    fs = np.abs(F[:, 0] - F[:, 1])
    print(f"力同步 |F_L−F_R|: 均值={fs.mean():.2f} N 峰值={fs.max():.2f} N（论文目标 <2 N）")
    print(f"垫片接触力: L 均值={F[:,0].mean():.2f} N R 均值={F[:,1].mean():.2f} N（参考 {args.force} N）")
    rel_yz = np.linalg.norm((x["tipL"][:, 1:] - x["tipR"][:, 1:]), axis=1)
    print(f"臂间相对位姿误差(y,z): 均值={rel_yz.mean()*1000:.2f} mm")
    # 罐子平稳性（论文 Fig.11：4 罐不落 = 运动平稳）：罐底相对箱顶的落差
    box_top = box_xyz[:, 2].reshape(-1, 1) + BOX_HALF[2]
    can_bottom = canz - CAN_HALF_H
    drop = can_bottom - box_top                      # <0 = 罐已掉出箱顶（歪倒/滑落）
    n_fall = int(np.sum(np.any(drop < -0.01, axis=1)))
    print(f"罐子平稳性(Fig.11): 4 罐在运输中掉落 {n_fall} 次（罐底相对箱顶最小落差 {drop.min()*1000:.1f} mm）| "
          f"运动平稳性判定: {'✅ 罐子未落' if n_fall == 0 else '❌ 罐子掉落（运动不稳）'}")
    tmx = np.degrees(np.abs(tilt).max(axis=0))
    print(f"箱倾角峰值: 侧倾(绕x)={tmx[0]:.2f}° 翘起(绕y)={tmx[1]:.2f}°")
    print(f"力矩饱和: {sat}/{N} | NMPC 求解: 均值={np.mean(x['solve_ms']):.1f} ms 峰值={np.max(x['solve_ms']):.1f} ms")
    print(f"箱终点 {box_xyz[-1].round(4)} vs 参考终点 {final_ref.round(4)}")
    ok = final_err * 1000 < 60.0 and tmx[1] < 3.0 and tmx[0] < 10.0 \
        and box_xyz[-1][2] > box_start[2] - 0.05 and n_fall == 0
    print(">>> 判定: " + ("✅ 夹住、y 向 −10mm/s 平稳搬运、罐子没掉" if ok else "❌ 跟踪/夹持/平稳性失败"))

    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    if not args.no_csv:
        import csv
        path = os.path.join(FIG_DIR, f"exp2_carry_session_{ts}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "boxx", "boxy", "boxz", "tipLx", "tipLy", "tipLz", "tipRx", "tipRy", "tipRz",
                        "tau1", "tau2", "tau3", "tau4", "tau5", "tau6", "tau7",
                        "tau8", "tau9", "tau10", "tau11", "tau12", "tau13", "tau14",
                        "FL", "FR", "tiltx", "tilty", "tiltz",
                        "c0z", "c1z", "c2z", "c3z"])
            for i in range(len(tts)):
                w.writerow([tts[i], *box_xyz[i], *x["tipL"][i], *x["tipR"][i],
                            *tau[i], F[i, 0], F[i, 1], *tilt[i], *canz[i]])
        print(f"📄 {path}")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(11, 8))
        axs[0, 0].plot(tts, box_xyz[:, 0], "tab:blue", lw=1.6, label="box x")
        axs[0, 0].plot(tts, box_xyz[:, 1], "tab:orange", lw=1.6, label="box y")
        axs[0, 0].plot(tts, box_xyz[:, 2], "tab:green", lw=1.6, label="box z")
        axs[0, 0].set_xlabel("t [s]"); axs[0, 0].set_ylabel("box pos [m]")
        axs[0, 0].legend(fontsize=8, loc="upper right")
        axs[0, 0].set_title("(a) 箱体轨迹")
        axs[0, 1].plot(tts, F[:, 0], "tab:blue", lw=1.6, label="F_L")
        axs[0, 1].plot(tts, F[:, 1], "tab:red", ls="--", lw=1.6, label="F_R")
        axs[0, 1].axhline(args.force, color="k", lw=0.8, ls=":")
        axs[0, 1].set_xlabel("t [s]"); axs[0, 1].set_ylabel("contact force [N]")
        axs[0, 1].legend(fontsize=8, loc="lower right")
        axs[0, 1].set_title(f"(b) 垫片夹持力（参考 ±{args.force} N，观测不入 NMPC）")
        for j in range(14):
            axs[1, 0].plot(tts, tau[:, j], lw=0.9, label=f"τ{j+1}")
        axs[1, 0].set_xlabel("t [s]"); axs[1, 0].set_ylabel("τ [N·m]")
        axs[1, 0].set_title("(c) 输入力矩")
        axs[1, 1].plot(tts, np.degrees(np.abs(tilt[:, 0])), "tab:blue", lw=1.4, label="側倾(绕x)")
        axs[1, 1].plot(tts, np.degrees(np.abs(tilt[:, 1])), "tab:red", lw=1.4, label="翘起(绕y)")
        axs[1, 1].plot(tts, fs, "tab:gray", lw=1.2, ls=":", label="|F_L−F_R|")
        axs[1, 1].set_xlabel("t [s]"); axs[1, 1].set_ylabel("angle [°] / force [N]")
        axs[1, 1].legend(fontsize=8, loc="upper right")
        axs[1, 1].set_title("(d) 箱倾角与力同步")
        for ax in axs.flat:
            ax.grid(alpha=0.3)
        fig.suptitle("Experiment II reproduction: NMPCC (joint state, cartesian cost, Eq.11/12/28-32)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_svg = os.path.join(FIG_DIR, f"exp2_carry_fig_{ts}.svg")
        out_png = os.path.join(FIG_DIR, f"exp2_carry_fig_{ts}.png")
        fig.savefig(out_svg, format="svg")
        fig.savefig(out_png, dpi=160)
        print(f"📈 {out_svg} (+png)")


if __name__ == "__main__":
    main()