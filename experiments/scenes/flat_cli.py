"""平放场景命令行：保留历史默认，并支持显式指定静态协作/力系数。"""
import argparse
import math
from experiments.legacy import flat_output as experiment_tools
from experiments import disturbance as disturbance_tools
K_STIFF = 2793.166


def unit_coefficient(value):
    """协作与力混合系数仅接受 [0, 1] 内的有限数，两场景共用此校验。"""
    number = float(value)
    if not math.isfinite(number) or not 0. <= number <= 1.:
        raise argparse.ArgumentTypeError('系数必须是 [0, 1] 内的有限数')
    return number


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--force", type=float, default=10, help="稳定夹持力参考 (N)")
    parser.add_argument("--kf_scale", type=float, default=0.003,
                        help="力误差增益，直接乘力误差；当前验证值为 0.003")
    parser.add_argument("--kx", type=float, default=None,
                        help="OCP 夹持弹簧刚度 (N/m)；None=用标定 K_STIFF；"
                             "模型力=kx*max(0,rest_gap-接触间距)，与 MuJoCo 软接触刚度标定对齐")
    parser.add_argument("--qp_solver", type=str, default="PARTIAL_CONDENSING_HPIPM",
                        help="QP 求解器（PARTIAL_CONDENSING_HPIPM/FULL_CONDENSING_HPIPM/"
                             "FULL_CONDENSING_QPOASES 等）")
    parser.add_argument("--qp_solver_iter_max", type=int, default=15)
    parser.add_argument("--qp_solver_cond_N", type=int, default=None,
                        help="部分凝聚块数（仅 PARTIAL_CONDENSING_HPIPM 生效；None=不设置）")
    parser.add_argument("--nlp_solver_max_iter", type=int, default=20)
    parser.add_argument("--Q", type=float, default=1.0,
                        help="阶段代价独立权重")
    parser.add_argument("--P", type=float, default=2.0,
                        help="终端代价独立权重")
    parser.add_argument("--eta_dev", type=float, default=0.05,
                        help="误差导数相对权重 eta_dev")
    parser.add_argument("--eta_u", type=float, default=5e-8,
                        help="输入相对权重 eta_u")
    parser.add_argument("--u_lim", type=float, default=None,
                        help="运行时输入盒约束 ±u_lim (Nm)：标准 MPC 输入限幅 u∈U，"
                             "只改 lbu/ubu 数值，免重新编译；None=不额外钳制，"
                             "沿用物理 actuator_ctrlrange 结构界")
    parser.add_argument("--eta_cop_y", type=unit_coefficient, default=0.8,
                        help="协作映射 eta_cop 的 y 分量；0=绝对 y 跟踪，0.8=同步优先")
    parser.add_argument("--eta_f_x", "--If1_x", dest="If1_x", type=unit_coefficient, default=0.8,
                        help="两臂 eta_f 的 x 分量；本次运行固定使用，默认0.8；1=纯力误差")
    parser.add_argument("--t_ramp", type=float, default=0.5, help="10N→目标力的平滑时间 (s)")
    parser.add_argument("--t_hold", type=float, default=1, help="固定 Holding 时间，结束后直接 Moving (s)")
    parser.add_argument("--f_start", type=float, default=10.0,
                        help="启动夹持力参考 (N)，同时用于初始 IK 压紧和控制器首拍")
    parser.add_argument("--grip_z", type=float, default=0.02,
                        help="垫片接触高度相对箱体重心的竖直偏移 (m)；正=重心上方（摆式稳定），"
                             "负=重心下方")
    parser.add_argument("--pad_half_w", type=float, default=0.060,
                        help="垫片横向半尺寸 (m)；0.060 = 覆盖箱面全宽（12cm）")
    parser.add_argument("--pad_half_h", type=float, default=0.030,
                        help="垫片竖直半尺寸 (m)；加大可增强抗滚转力臂")
    parser.add_argument("--v_y", type=float, default=-0.01)
    parser.add_argument("--y_dist", type=float, default=0.5)
    parser.add_argument("--vz_max", type=float, default=0.05,
                        help="OCP 末端 z 速度约束上限 (m/s)")
    parser.add_argument("--hold_only", action="store_true")
    parser.add_argument("--sliding_friction", type=float, default=0.8,
                        help="箱体及两侧夹持垫的滑动摩擦系数；两侧同时设置，避免取较大值覆盖")
    parser.add_argument("--noslip_iterations", type=int, default=3,
                        help="MuJoCo 每个物理步的 NoSlip 最大迭代次数；0 为关闭")
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--Tp", type=float, default=0.10)
    parser.add_argument("--sim_steps", type=int, default=2,
                        help="acados ERK 每个预测区间的积分步数")
    parser.add_argument("--sim_stages", type=int, default=4,
                        help="acados ERK 积分级数；当前验证配置为 4")
    parser.add_argument("--contact_load_mode", choices=("tangent", "residual"), default="residual",
                        help="接触预测载荷：默认 residual 冻结完整实测载荷减当前模型法向载荷；tangent 回退原切向载荷")
    parser.add_argument("--no_viewer", action="store_true", help="关闭 MuJoCo 可视化窗口")
    experiment_tools.add_arguments(parser)
    disturbance_tools.add_arguments(parser)
    parser.add_argument('--output_dir', default=None, help='本次实验输出目录（须为新目录）')
    return parser

