"""场景组装只负责模型、抓取、参考起点与控制配置。"""
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from pathlib import Path
import mujoco
import numpy as np
from ik import name_to_id
from nmpcc.config import ControllerConfig, ForceFeedbackSource, BiasPolicy, ContactLoadMode, NormalForceLaw
from nmpcc.kinematics import pad_contact_offsets, mujoco_fk_symbols, rotvec_of_R
from diagnostics.measurements import arm_joint_bounds
from .reference import PhaseReference, motion_timing
from .legacy import flat_output, handheld_output


@dataclass(slots=True)
class Scene:
    variant: str
    model: mujoco.MjModel
    data: mujoco.MjData
    arms: list[dict]
    ids: dict[str, int]
    source: Path
    geometry: ModuleType
    output: ModuleType
    fk: dict
    contact_rest_gap: float
    box_center: np.ndarray
    motion_timing = staticmethod(motion_timing)


def build_scene(variant: str, args) -> Scene:
    if variant not in ('flat', 'handheld'):
        raise ValueError('variant must be flat or handheld')
    # 1. 仅加载所选场景的几何和资源，两个入口不互相依赖。
    geometry = import_module(f'.scenes.{variant}', __package__)
    if variant == 'flat':
        import exp2_slender_scene
        exp2_slender_scene.GRIP_Z_OFFSET = args.grip_z
        exp2_slender_scene.PAD_HALF_W = args.pad_half_w
        exp2_slender_scene.PAD_HALF_H = args.pad_half_h
        source, model, data, arms = geometry.build_grasped_scene(geometry.IK_ITERS)
    else:
        source = geometry.build_scene()
        model = mujoco.MjModel.from_xml_path(str(source))
        data = mujoco.MjData(model)
        arms = geometry.make_arms(model)
    # 2. 建立执行器和箱体索引，再应用该场景的接触设置。
    for arm in arms:
        arm['act_ids'] = [
            name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'{model.joint(j).name}_motor')
            for j in arm['joint_ids']
        ]
    body = name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, 'box')
    geom = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, 'box')
    joint = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, 'box_freejoint')
    ids = dict(box_body=body, box_geom=geom, box_qpos=int(model.jnt_qposadr[joint]))
    if variant == 'handheld':
        geometry.disable_robot_collisions(model, arms, geom)
        geometry.soften_contacts(model, geom, [a['pad_id'] for a in arms])
    model.geom_friction[[geom] + [a['pad_id'] for a in arms], 0] = args.sliding_friction
    model.opt.noslip_iterations = args.noslip_iterations
    print(f'[contact] box/pads sliding_friction={args.sliding_friction:g}, noslip_iterations={model.opt.noslip_iterations}')
    # 平放旧流程在建立符号FK前先建立一次预压，保留初始化顺序。
    if variant == 'flat':
        geometry.reset_world(model, data, arms, ids['box_qpos'])
        targets = geometry.grasp_targets(arms)
        geometry.solve_end_grip(model, data, arms,
            pressed_targets(targets, arms, args.f_start, geometry.K_STIFF), geometry.IK_ITERS)
    offsets = pad_contact_offsets(model, arms, .1001)
    rest_gap = geometry.REST_GAP
    if variant == 'handheld':
        rest_gap -= sum(model.geom_size[a['pad_id'], 2] for a in arms)
    arm_configs = [dict(body_id=a['body_id'], joint_ids=a['joint_ids']) for a in arms]
    if variant == 'handheld':
        # 手持原模型先组合连杆局部接触点，再一次齐次变换；保留浮点运算顺序。
        local_points = [np.array([0., 0., .1001]) + offset for offset in offsets]
        fk = mujoco_fk_symbols(model, arm_configs, .1001, contact_local_points=local_points)
    else:
        fk = mujoco_fk_symbols(model, arm_configs, .1001, contact_offsets=offsets)
    for arm in arms:
        model.dof_damping[arm['dofs']] = .5
    return Scene(variant, model, data, arms, ids, Path(source), geometry,
        flat_output if variant == 'flat' else handheld_output, fk, rest_gap, geometry.BOX_CENTER.copy())


def initialize_reference(scene: Scene, args) -> PhaseReference:
    g, model, data, arms = scene.geometry, scene.model, scene.data, scene.arms
    g.reset_world(model, data, arms, scene.ids['box_qpos'])
    scene.box_center = g.BOX_CENTER.copy()
    targets = g.grasp_targets(arms)
    def press(force):
        return pressed_targets(targets, arms, force, g.K_STIFF)
    if scene.variant == 'handheld':
        start, _ = g.calibrate_initial_preload(model, data, arms, scene.ids['box_geom'], args.f_start, press)
    else:
        start = press(args.f_start)
        g.solve_end_grip(model, data, arms, start, g.IK_ITERS)
    rotation = {a['side']:rotvec_of_R(data.xmat[a['body_id']].reshape(3, 3)) for a in arms}
    return PhaseReference(args.force, 0. if args.hold_only else args.v_y, args.y_dist,
        args.t_ramp, args.t_hold, start, press(args.force), rotation, args.f_start)


def controller_config(scene: Scene, args) -> ControllerConfig:
    is_handheld = scene.variant == 'handheld'
    stiffness = args.kx_model if is_handheld else (args.kx if args.kx is not None else scene.geometry.K_STIFF)
    return ControllerConfig(N=args.N, Tp=args.Tp,
        eta_cop=np.tile([0., args.eta_cop_y, 0., 0., 0., 0.], (2, 1)),
        eta_f=np.tile([getattr(args, 'If1_x', .8), 0., 0., 0., 0., 0.], (2, 1)),
        Q=args.Q, P=args.P, eta_dev=args.eta_dev, eta_u=args.eta_u,
        kx=stiffness, rest_gap=scene.contact_rest_gap, kf_scale=args.kf_scale, vmax=.05, vz_max=args.vz_max,
        sim_method_num_steps=args.sim_steps, sim_method_num_stages=getattr(args, 'sim_stages', 4),
        qp_solver=getattr(args, 'qp_solver', 'PARTIAL_CONDENSING_HPIPM'),
        qp_solver_iter_max=getattr(args, 'qp_solver_iter_max', 15),
        qp_solver_cond_N=getattr(args, 'qp_solver_cond_N', None), nlp_solver_max_iter=getattr(args, 'nlp_solver_max_iter', 20),
        force_feedback_source=ForceFeedbackSource.WORLD_FORCE_X if is_handheld else ForceFeedbackSource.NORMAL_MAGNITUDE,
        bias_policy=BiasPolicy.LEGACY_SIX_AXIS if is_handheld else BiasPolicy.X_ONLY,
        normal_force_law=NormalForceLaw(getattr(args, 'normal_force_law', NormalForceLaw.SMOOTH_POSITIVE)),
        contact_load_mode=ContactLoadMode.FULL_RESIDUAL if args.contact_load_mode == 'residual' else ContactLoadMode.TANGENT)


def controller_bounds(scene: Scene) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model, arms = scene.model, scene.arms
    maximum = np.concatenate([model.actuator_ctrlrange[a['act_ids'], 1] for a in arms])
    lower_q, upper_q = arm_joint_bounds(model, arms)
    return - maximum, maximum, lower_q, upper_q


def pressed_targets(targets: dict[str, np.ndarray], arms: list[dict],
                    force: float, stiffness: float) -> dict[str, np.ndarray]:
    delta = force / (2. * stiffness)
    return {a['side']:targets[a['side']] + np.array([-a['sign'] * delta, 0., 0.]) for a in arms}
