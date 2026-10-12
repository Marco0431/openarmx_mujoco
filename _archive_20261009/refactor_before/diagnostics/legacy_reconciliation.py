"""历史启动几何对账工具，不参与当前控制循环。

包含旧场景加载流程与旧标定常数，仅为保留旧接口归档。
其打印结论不能代替当前实验日志中的接触、限位和失持判断。
"""
from __future__ import annotations
import argparse
import mujoco
import numpy as np
from ik import get_ee_tip, name_to_id
from .measurements import contact_forces

SCENE_CHECK_CONST = {
    "slender": {
        "calib_intercept": 0.8,   # N：标定 delta=0 处的软接触底数（delta=0 实测 ≈0.99N，拟合截距 ≈0.8N）
        "pad_half_t": 0.004,      # 垫片半厚（与场景一致）
    },
    "grasp_pose": {
        "calib_intercept": 0.8,
        "pad_half_t": 0.004,
        "box_mass": 1.5,          # 箱质量 kg（与 grasp_pose_scene.xml 一致）
    },
}


def press_delta_mm(force, k_stiff):
    """重压目标压入量（mm）= F/(2K)/1e-3。"""
    return force / (2.0 * k_stiff) * 1000.0


def reconciliation(model, data, arms, box_qpos, fk, base_targets, press_light,
                   force, scene) -> None:
    """启动几何/FK 对账：箱位姿、垫片贴合、IK 误差、符号 FK(vs MuJoCo) 误差、
    标定预期接触力 vs 实测接触力，单块打印——启动即验证几何/控制器假设一致。

    ``scene``：dict(box_center(3), box_half(3), box_mass, k_stiff,
    calib_intercept, pad_half_t, rest_gap)，由加载函数给出场景常数。
    """
    box_center = np.asarray(scene["box_center"], dtype=np.float64)
    box_half = np.asarray(scene["box_half"], dtype=np.float64)
    box_mass = float(scene["box_mass"])
    k_stiff = float(scene["k_stiff"])
    calib_intercept = float(scene["calib_intercept"])
    pad_half_t = float(scene["pad_half_t"])
    rest_gap = float(scene["rest_gap"])
    print("=" * 78)
    print("启动几何/FK 对账 (reconciliation)")
    print("=" * 78)
    bq = box_qpos
    print(f"[箱] freejoint pos={np.round(data.qpos[bq:bq + 3], 6)} "
          f"目标={box_center} ok={np.allclose(data.qpos[bq:bq + 3], box_center, atol=1e-4)}")
    print(f"[箱] quat={np.round(data.qpos[bq + 3:bq + 7], 6)}"
          f" | 质量={box_mass}kg 重心高={box_center[2]}m")
    qL = data.qpos[arms[0]["qpos"]].copy()
    qR = data.qpos[arms[1]["qpos"]].copy()
    fk_r = np.asarray(fk["tips_fn"](qL, qR)).reshape(-1)
    for i, a in enumerate(arms):
        side = a["side"]
        tip = get_ee_tip(data, a["body_id"])
        tgt = press_light[side]
        err_mm = np.linalg.norm(tip - tgt) * 1000.0
        fkp = fk_r[i * 6:i * 6 + 3]
        fk_err_mm = np.linalg.norm(fkp - tip) * 1000.0
        pad_R = data.geom_xmat[a["pad_id"]].reshape(3, 3)
        n = pad_R @ np.array([0.0, 0.0, 1.0])
        n /= (np.linalg.norm(n) + 1e-12)
        align = float(n @ np.array([-a["sign"], 0.0, 0.0]))
        pad_c = data.geom_xpos[a["pad_id"]]
        front_x = pad_c[0] + pad_half_t * n[0]
        face_x = box_center[0] + a["sign"] * box_half[0]
        gap_mm = (front_x - face_x) * a["sign"] * 1000.0
        pad_z_lo, pad_z_hi = pad_c[2] - 0.030, pad_c[2] + 0.030
        box_z_lo, box_z_hi = box_center[2] - box_half[2], box_center[2] + box_half[2]
        print(f"[{side}] IK err={err_mm:.3f}mm | FK-vs-MuJoCo err={fk_err_mm:.4f}mm | "
              f"垫片法向对齐(n·入箱)={align:.4f} | 垫片-箱端面间隙={gap_mm:+.3f}mm")
        print(f"[{side}] 垫片面 z∈[{pad_z_lo:.4f},{pad_z_hi:.4f}] 箱面 z∈[{box_z_lo:.4f},{box_z_hi:.4f}]"
              f" | {'箱面下缘<垫片下缘:纯摩擦持重(需法向>0.9N)' if box_z_lo < pad_z_lo else '箱面下缘受垫片托载'}")
    d_tip = float(get_ee_tip(data, arms[1]["body_id"])[0]
                  - get_ee_tip(data, arms[0]["body_id"])[0])
    qL = data.qpos[arms[0]["qpos"]].copy()
    qR = data.qpos[arms[1]["qpos"]].copy()
    contact = np.asarray(fk["contact_tips_fn"](qL, qR)).reshape(-1)
    d_contact = float(contact[3] - contact[0])
    print(f"[几何] tip中心间距 d_tip={d_tip:.6f} m；"
          f"pad接触面间距 d_contact={d_contact:.6f} m vs REST_GAP={rest_gap:.6f} m "
          f"(接触面差 {(d_contact - rest_gap) * 1000:+.3f} mm → 零压入接触量)")
    press_in = press_light["left"][0] - base_targets["left"][0]
    expect = k_stiff * 2.0 * abs(press_in) + calib_intercept
    forces = contact_forces(model, data, arms,
                            name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "box"))
    print(f"[预紧] 起始轻压 |{press_in * 1000:.3f}| mm | 标定预期力 "
          f"K_side*δ+{calib_intercept} = {expect:.2f} N | 实测 FL={forces['left']:.2f}N "
          f"FR={forces['right']:.2f}N | 目标力 {force:.1f} N")
    ok_light = forces["left"] > 0.3 and forces["right"] > 0.3
    print(f"[对账] {'✓ 轻接触成立（摩擦持重安全）' if ok_light else '✗ 无接触/欠压！箱可能下落'}"
          f" | 重压目标 δ={press_delta_mm(force, k_stiff):.3f} mm（标定 F={force:.0f}N 对应）")
    print("=" * 78)


def _load_slender(f_start):
    """加载地面细长箱场景并按 f_start 收敛轻压位（与 exp2_nmpcc_latest.main 同款启动路径）。"""
    import exp2_nmpcc_latest as exp
    import exp2_slender_scene as sc
    import nmpcc_cart as nc

    scene_xml, model, data, arms = sc.build_grasped_scene(sc.IK_ITERS)
    box_qpos = int(model.jnt_qposadr[
        name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")])
    exp.reset_world(model, data, arms, box_qpos)
    base_targets = sc.grasp_targets(arms)

    def press_for(force_n):
        delta = force_n / (2.0 * exp.K_STIFF)
        return {a["side"]: base_targets[a["side"]]
                + np.array([-a["sign"] * delta, 0.0, 0.0]) for a in arms}

    press_light = press_for(f_start)
    sc.solve_end_grip(model, data, arms, press_light, sc.IK_ITERS)
    contact_offsets = nc.pad_contact_offsets(model, arms, 0.1001)
    fk = nc.mujoco_fk_symbols(
        model,
        [{"body_id": a["body_id"], "joint_ids": a["joint_ids"]} for a in arms],
        0.1001,
        contact_offsets=contact_offsets,
    )
    cc = SCENE_CHECK_CONST["slender"]
    check = {
        "box_center": sc.BOX_CENTER, "box_half": sc.SLENDER_HALF,
        "box_mass": sc.BOX_MASS, "k_stiff": exp.K_STIFF,
        "calib_intercept": cc["calib_intercept"], "pad_half_t": cc["pad_half_t"],
        "rest_gap": exp.REST_GAP,
    }
    return model, data, arms, box_qpos, fk, base_targets, press_light, check


def _load_grasp_pose(f_start):
    """加载空中抓取姿态场景并按 f_start 收敛轻压位（与 exp2_nmpcc_latest_grasp_pose 同款路径）。"""
    import exp2_nmpcc_latest_grasp_pose as exp
    import nmpcc_cart_grasp_pose as nc

    scene_xml = exp.build_scene()
    exp.inject_motors(exp.SCENE_XML.parent / "openarmx_ik_runtime.xml")
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    arms = exp.make_arms(model)
    box_qpos = int(model.jnt_qposadr[
        name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")])
    box_geom = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    exp.disable_robot_collisions(model, arms, box_geom)
    exp.soften_contacts(model, box_geom, [a["pad_id"] for a in arms])
    # reset_world 按休息位重算 BOX_CENTER，随后必须读取模块内最新 BOX_CENTER 重算目标。
    exp.reset_world(model, data, arms, box_qpos)
    base_targets = exp.grasp_targets(arms, exp.BOX_CENTER)

    def press_for(force_n):
        delta = force_n / (2.0 * exp.K_STIFF)
        return {a["side"]: base_targets[a["side"]]
                + np.array([-a["sign"] * delta, 0.0, 0.0]) for a in arms}

    press_light = press_for(f_start)
    exp.solve_end_grip(model, data, arms, press_light, exp.IK_ITERS)
    fk = nc.mujoco_fk_symbols(
        model,
        [{"body_id": a["body_id"], "joint_ids": a["joint_ids"]} for a in arms],
        0.1001,
    )
    cc = SCENE_CHECK_CONST["grasp_pose"]
    check = {
        "box_center": exp.BOX_CENTER, "box_half": exp.BOX_HALF,
        "box_mass": cc["box_mass"], "k_stiff": exp.K_STIFF,
        "calib_intercept": cc["calib_intercept"], "pad_half_t": cc["pad_half_t"],
        "rest_gap": exp.REST_GAP,
    }
    return model, data, arms, box_qpos, fk, base_targets, press_light, check


def main() -> None:
    parser = argparse.ArgumentParser(
        description="实验二 NMPCC 检测工具：启动几何/FK 对账 + 接触力标定核对（独立运行）")
    parser.add_argument("--scene", choices=["slender", "grasp_pose"],
                        default="slender",
                        help="检查哪个场景：地面细长箱（默认）或空中抓取姿态")
    parser.add_argument("--force", type=float, default=None,
                        help="重压目标夹持力 (N)；默认沿用对应主程序默认值（slender=15, grasp_pose=10）")
    parser.add_argument("--f_start", type=float, default=None,
                        help="轻压启动夹持力 (N)；默认 10")
    args = parser.parse_args()

    if args.scene == "slender":
        default_force, loader = 15.0, _load_slender
    else:
        default_force, loader = 10.0, _load_grasp_pose
    force = default_force if args.force is None else args.force
    f_start = 10.0 if args.f_start is None else args.f_start

    print(f"[diagnostics] scene={args.scene} force={force} N f_start={f_start} N")
    model, data, arms, box_qpos, fk, base_targets, press_light, check = loader(f_start)
    reconciliation(model, data, arms, box_qpos, fk, base_targets, press_light,
                   force, check)

