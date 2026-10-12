"""手持场景命令行：保留历史默认，并支持显式指定静态协作/力系数。"""
import argparse
from nmpcc.config import NormalForceLaw
from experiments.legacy import handheld_output as experiment_tools
from experiments import disturbance as disturbance_tools
from .flat_cli import unit_coefficient
K_STIFF = 2419.3

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--force", type=float, default=10, help="稳定夹持力参考 (N)")
    parser.add_argument("--kf_scale", type=float, default=0.003,
                        help="力误差直接乘此增益，与平放版一致；不再除以接触刚度")
    parser.add_argument("--Q", type=float, default=1.0, help="阶段代价的独立整体权重 Q")
    parser.add_argument("--P", type=float, default=2.0, help="独立终端权重 P")
    parser.add_argument("--eta_dev", type=float, default=0.05, help="误差导数相对权重 eta_dev")
    parser.add_argument("--eta_u", type=float, default=5e-8, help="输入相对权重 eta_u")
    parser.add_argument("--u_lim", type=float, default=None,
                        help="求解时 u 全 horizon 硬限幅 (N·m)；为力矩优化提供有限可行域、"
                             "防止力矩偏移过大（标准 MPC 输入约束 u∈U）")
    parser.add_argument("--kx_model", type=float, default=K_STIFF,
                        help="预测弹簧刚度 (N/m)，默认保留手持标定2419.3；力残差不再除以kx。"
                             "本轮不调整该参数或物理接触标定。")
    parser.add_argument("--normal_force_law", choices=[law.value for law in NormalForceLaw],
                        default='smooth_positive',
                        help="法向力预测：smooth_positive=旧平滑；ideal_unilateral=旧截断；"
                             "measured_anchored_ideal=测量预载加间距增量后截断。默认保留旧平滑")
    parser.add_argument("--eta_cop_y", type=unit_coefficient, default=0.8,
                        help="协作映射 eta_cop 的 y 分量；0=绝对 y 跟踪，0.8=同步优先")
    parser.add_argument("--If1_x", "--eta_f_x", dest="If1_x", type=unit_coefficient, default=0.8,
                        help="两臂 eta_f 的 x 分量；本次运行固定使用，默认0.8；1=纯力误差")

    parser.add_argument("--t_ramp", type=float, default=0.5, help="10N→目标力的平滑时间 (s)")
    parser.add_argument("--t_hold", type=float, default=1.0, help="固定 Holding 时间，结束后直接 Moving (s)")
    parser.add_argument("--f_start", type=float, default=10.0,
                        help="启动夹持力参考 (N)，同时用于初始 IK 压紧和控制器首拍")
    parser.add_argument("--v_y", type=float, default=-0.01,
                        help="世界坐标 -y 前进速度 (m/s)，正值表示沿 -y 前进")
    parser.add_argument("--y_dist", type=float, default=0.05,
                        help="沿世界 -y 的前进距离 (m)")
    parser.add_argument("--vz_max", type=float, default=0.05,
                        help="OCP 末端 z 速度约束上限 (m/s)")
    parser.add_argument("--settle_time", type=float, default=0.0,
                        help="t0 沉降时长 (s)：用全接触平衡扭矩静置，让软接触竖直摩擦发育、"
                             "接住箱子后再交给控制器；0=关闭")
    parser.add_argument("--hold_only", action="store_true")
    parser.add_argument("--sliding_friction", type=float, default=0.8,
                        help="箱体及两侧夹持垫的滑动摩擦系数；两侧同时设置，避免取较大值覆盖")
    parser.add_argument("--noslip_iterations", type=int, default=3,
                        help="MuJoCo 每个物理步的 NoSlip 最大迭代次数；0 为关闭")
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--Tp", type=float, default=0.10)
    parser.add_argument("--sim_steps", type=int, default=2,
                        help="acados ERK 每个预测区间的积分步数")
    parser.add_argument("--no_viewer", action="store_true", help="关闭 MuJoCo 可视化窗口")
    parser.add_argument('--contact_load_mode', choices=['residual', 'tangent'], default='residual',
                        help='默认residual，与平放版一致；tangent仅供载荷近似对照')
    parser.add_argument('--output_dir', default=None, help='new directory for complete NPZ diagnostics')
    experiment_tools.add_arguments(parser)
    disturbance_tools.add_arguments(parser)
    return parser

