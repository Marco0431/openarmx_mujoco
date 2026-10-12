"""手持场景的几何、左右映射和启动预压；不包含控制循环。"""
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from ik import ARM_CONFIGS, BASE_DIR, clamp_qpos, get_ee_tip, name_to_id, solve_pose_ik
from diagnostics.measurements import contact_forces
BOX_HALF = np.array([0.12, 0.20, 0.10])
BOX_CENTER = np.array([0.5, 0.0, 0.72])
GRIP_CLEARANCE = 0.004
IK_ITERS = 1500
SCENE_XML = BASE_DIR / 'grasp_pose_scene.xml'
K_STIFF = 2419.3
REST_GAP = 2 * (BOX_HALF[0] + GRIP_CLEARANCE)

def build_scene() -> Path:
    """从固定抓取场景建立运行模型，补齐缺失的关节力矩电机。
    电机限幅继承模型的物理范围，原始场景文件不作覆盖。"""
    if not SCENE_XML.is_file():
        raise FileNotFoundError(SCENE_XML)
    original = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    root = ET.Element('mujoco', model='grasp_pose_nmpcc_runtime')
    ET.SubElement(root, 'include', file=str(SCENE_XML.resolve()))
    actuator = ET.SubElement(root, 'actuator')
    missing = 0
    for physical_side in ('left', 'right'):
        for i in range(1, 8):
            name = f'openarmx_{physical_side}_joint{i}'
            jid = name_to_id(original, mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_ACTUATOR, name + '_motor')
            if aid >= 0:
                continue
            if not original.jnt_actfrclimited[jid]:
                raise ValueError(f'{name} lacks an explicit physical torque range')
            bounds = ' '.join(format(float(v), '.17g') for v in original.jnt_actfrcrange[jid])
            ET.SubElement(actuator, 'motor', name=name + '_motor', joint=name,
                          gear='1', ctrllimited='true', forcelimited='true',
                          ctrlrange=bounds, forcerange=bounds)
            missing += 1
    if not missing:
        return SCENE_XML
    runtime = BASE_DIR / 'grasp_pose_handheld_nmpcc_runtime.xml'
    content = ET.tostring(root, encoding='unicode')
    if not runtime.exists() or runtime.read_text(encoding='utf-8') != content:
        runtime.write_text(content, encoding='utf-8')
    return runtime


def make_arms(model):
    """收集双臂关节、执行器和垫片索引，并转换到控制器的左右排序。
    手持场景的物理左右臂与控制器左右约定相反，后续数据统一使用此映射。"""
    arms = []
    # 物理左臂在 +x，物理右臂在 -x；转换成控制器 left/right 约定。
    side_of = {"left": "right", "right": "left"}
    for cfg in ARM_CONFIGS:
        side = side_of[cfg["side"]]
        joint_ids = [name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg["joint_names"]]
        arms.append({
            "side": side, "sign": -1.0 if side == "left" else 1.0,
            "q_rest": cfg["q_rest"],
            "body_id": name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["jacobi_body"]),
            "joint_ids": joint_ids,
            "dofs": [int(model.jnt_dofadr[j]) for j in joint_ids],
            "qpos": [int(model.jnt_qposadr[j]) for j in joint_ids],
            "pad_id": name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                 f"openarmx_{cfg['side']}_ee_pad"),
        })
    return sorted(arms, key=lambda a: 0 if a["side"] == "left" else 1)


def grasp_targets(arms, center=None):
    """根据箱体中心和手臂所在侧生成对夹目标点。"""
    center = BOX_CENTER if center is None else np.asarray(center)
    return {a["side"]: np.array(
        [center[0] + a["sign"] * (BOX_HALF[0] + GRIP_CLEARANCE), center[1], center[2]])
        for a in arms}


def solve_end_grip(model, data, arms, targets, iters=IK_ITERS):
    """用位置与朝向 IK 建立初始夹持姿态；仅用于初始化，不运行动态控制。"""
    for a in arms:
        for _ in range(iters):
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            mujoco.mj_forward(model, data)
            dq, _ = solve_pose_ik(
                model, data, a["body_id"], targets[a["side"]],
                np.array([-a["sign"], 0.0, 0.0]), get_ee_tip(data, a["body_id"]),
                a["dofs"], a["qpos"], a["q_rest"])
            data.qpos[a["qpos"]] += dq
            clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)


def calibrate_initial_preload(model, data, arms, box_geom_id, target_force,
                              target_fn, tolerance=0.25, max_iter=12):
    """按实测接触力迭代修正初始压入量，返回初始夹持目标和对应实测力。
    只校准本次启动的几何预压，不改接触刚度或控制器增益。"""
    lo = 0.0
    hi = max(3.0 * float(target_force), 5.0)
    best = None
    for _ in range(max_iter):
        command_force = 0.5 * (lo + hi)
        solve_end_grip(model, data, arms, target_fn(command_force),
                       max(250, IK_ITERS // 3))
        measured = contact_forces(model, data, arms, box_geom_id)
        mean_force = 0.5 * (measured["left"] + measured["right"])
        error = mean_force - target_force
        if best is None or abs(error) < abs(best[0]):
            best = (error, command_force, measured)
        if abs(error) <= tolerance:
            break
        if error < 0.0:
            lo = command_force
        else:
            hi = command_force
    if best is None or abs(best[0]) > max(tolerance, 0.05 * target_force):
        raise RuntimeError(
            f"t=0 preload calibration failed: target={target_force:.3f} N, best={best}")
    command_force = best[1]
    solve_end_grip(model, data, arms, target_fn(command_force), IK_ITERS)
    measured = contact_forces(model, data, arms, box_geom_id)
    print(f"[t0 preload] target={target_force:.3f}N geometry_command={command_force:.3f}N "
          f"static=({measured['left']:.3f},{measured['right']:.3f})N")
    return target_fn(command_force), measured


def disable_robot_collisions(model, arms, box_geom_id):
    """设置机器人与箱体的碰撞筛选，保留本实验需要的垫片接触。"""
    pads = {a["pad_id"] for a in arms}
    for gid in range(model.ngeom):
        if gid == box_geom_id or gid in pads:
            continue
        if (model.geom(gid).name or "").startswith("openarmx_"):
            model.geom_contype[gid] = 0
            model.geom_conaffinity[gid] = 0


def soften_contacts(model, box_geom_id, pad_ids):
    """统一箱体和垫片的接触求解参数，保持当前标定所用的软接触设置。"""
    for gid in [box_geom_id, *pad_ids]:
        model.geom_solref[gid] = [0.02, 1.0]
        model.geom_solimp[gid][:3] = [0.999, 0.999, 0.001]



def arm_damping(model, dofs, value=0.5):
    """设置指定关节的粘性阻尼；预测参数随后从模型读取同一组系数。"""
    for d in dofs:
        model.dof_damping[d] = value


def inject_motors(runtime_xml: Path) -> None:
    """为指定运行模型补齐力矩电机；已有电机保留不重复创建。"""
    tree = ET.parse(runtime_xml)
    root = tree.getroot()
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    existing = {motor.get("name") for motor in actuator.findall("motor")}
    added = 0
    for side in ("left", "right"):
        for i in range(1, 8):
            joint_name = f"openarmx_{side}_joint{i}"
            motor_name = f"{joint_name}_motor"
            if motor_name in existing:
                continue
            joint = root.find(f".//joint[@name='{joint_name}']")
            if joint is None:
                raise ValueError(f"runtime XML 中找不到关节 {joint_name}")
            force_range = joint.get("actuatorfrcrange", "-120 120")
            ET.SubElement(actuator, "motor", {
                "name": motor_name,
                "joint": joint_name,
                "forcerange": force_range,
                "ctrlrange": force_range,
            })
            added += 1
    if added:
        tree.write(runtime_xml, encoding="utf-8", xml_declaration=True)
        print(f"[scene] injected {added} torque motors into {runtime_xml.name}")



def reset_world(model, data, arms, box_qpos) -> None:
    """恢复初始关节和箱体姿态，清零速度并刷新模型。
    这是每次实验开始时的复位，不是控制循环内的状态修正。"""
    global BOX_CENTER
    data.time = 0.0
    data.ctrl[:] = 0.0
    data.qacc_warmstart[:] = 0.0
    for a in arms:
        data.qpos[a["qpos"]] = a["q_rest"]
        clamp_qpos(model, data, a["qpos"], a["joint_ids"])
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    # 根据休息姿态设置箱体中心，保证初始抓取目标沿用已验证的位置。
    tips_mid = np.mean([get_ee_tip(data, a["body_id"]) for a in arms], axis=0)
    BOX_CENTER = tips_mid + np.array([0.0, 0.0, 0.10])
    data.qpos[box_qpos:box_qpos + 3] = BOX_CENTER
    data.qpos[box_qpos + 3:box_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


