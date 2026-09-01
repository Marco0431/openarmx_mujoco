#!/usr/bin/env python3
"""mpc_tutorial.py —— 一个完整可跑的 MPC（格式教学，最简可运行版）

车型：双积分器  x = [p, v]（位置、速度），u = [F]（推力），质量 m=1。
目标：从 (0,0) 出发，跟踪一条参考轨迹 pref(t)（含参考速度）。
演示 MPC 的完整闭环格式：
  ① 模型定义 → ② OCP 配置(代价/约束/求解器) → ③ 编译 → ④ 闭环循环(set x0/yref → solve → 取u → 仿真步进) → ⑤ 记录
相比 nmpcc_cart.py，这里去掉机械臂的 FK/合作指标等业务细节，只保留 MPC 骨架，方便你对照格式。

运行： D:\\acados_env\\python.exe mpc_tutorial.py
"""
import numpy as np
from casadi import SX, vertcat
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# ========== 问题设定 ==========
m = 1.0          # 质量
Tf = 0.4         # 预测时域（s）
N = 20           # 预测步数
Fmax = 5.0       # 力矩/推力上限
DT = Tf / N      # 每段时长
TOTAL = 80       # 闭环总步数（80*DT = 1.6s）

# ========== ① 模型（casadi 符号动力学） ==========
x = SX.sym('x', 2)          # 状态 x = [p, v]
u = SX.sym('u', 1)          # 控制 u = [F]
f_expl = vertcat(x[1], u[0] / m)     # ẋ = [v; F/m]

model = AcadosModel()
model.name = 'cart_mpc'
model.x = x
model.u = u
model.f_expl_expr = f_expl

# ========== ② OCP 配置 ==========
ocp = AcadosOcp()
ocp.model = model
ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf

nx, nu = 2, 1
ny = nx + nu          # 跑动残差维度 [x, u]

# ②-a 代价（NONLINEAR_LS：残差 + 权重 + 参考）
Q = np.diag([100.0, 1.0])          # 状态权重（位置重罚，速度轻罚）
R = np.diag([0.01])                # 输入权重（省力）
P = np.diag([1000.0, 10.0])        # 终端权重（终点强调到位）

ocp.cost.cost_type = 'NONLINEAR_LS'
ocp.cost.cost_type_e = 'NONLINEAR_LS'
model.cost_y_expr = vertcat(x, u)      # 跑动残差 y = [x, u]
model.cost_y_expr_e = x                # 终端残差 y_e = [x]（无 u）
W = np.zeros((ny, ny))
W[:nx, :nx] = Q
W[nx:, nx:] = R
ocp.cost.W = W
ocp.cost.W_e = P
ocp.cost.yref = np.zeros(ny)           # 参考（循环里逐节点 set）
ocp.cost.yref_e = np.zeros(nx)

# ②-b 约束（输入限幅）
ocp.constraints.idxbu = np.array([0])
ocp.constraints.lbu = np.array([-Fmax])
ocp.constraints.ubu = np.array([+Fmax])
ocp.constraints.x0 = np.zeros(nx)      # 初始状态占位（循环里用 lbx/ubx 覆盖）

# ②-c 求解器选项（实时）
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP_RTI'   # 或 'SQP'（完整收敛）

# ========== ③ 编译（生成 C 代码 + gcc → DLL + 加载） ==========
solver = AcadosOcpSolver(ocp, json_file='cart_mpc.json', build=True, generate=True)
print("✅ 编译完成，solver 已加载")

# ========== ④ 闭环循环 ==========
def ref(t):
    """参考轨迹：匀速 0.4 m/s，位置线性增加。"""
    return np.array([0.4 * t, 0.4])       # [p_ref, v_ref]

x_now = np.array([0.0, 0.0])
log = []
for k in range(TOTAL):
    t_now = k * DT
    # 4-a 固定初始状态（当前实测状态）
    solver.set(0, 'lbx', x_now); solver.set(0, 'ubx', x_now)
    # 4-b 逐节点装参考（MPC 带 preview：把未来 N 步的目标都告诉它）
    for i in range(N):
        solver.set(i, 'yref', np.concatenate([ref(t_now + i * DT), np.zeros(nu)]))
    solver.set(N, 'yref', ref(t_now + N * DT))       # 终端节点只有状态参考
    # 4-c 求解
    status = solver.solve()
    u0 = solver.get(0, 'u')[0]
    # 4-d 仿真被控对象一步（欧拉，与模型一致）
    x_now = x_now + DT * np.array([x_now[1], u0 / m])
    log.append((t_now, x_now[0], x_now[1], u0))
    if k % 10 == 0:
        print(f"t={t_now:.2f}  p={x_now[0]:.3f}  v={x_now[1]:.3f}  u={u0:+.3f}  status={status}")

# ========== ⑤ 结果 ==========
arr = np.array(log)
p_end = arr[-1, 1]; pref_end = 0.4 * (TOTAL * DT)
err = abs(p_end - pref_end)
print(f"\n 终点位置={p_end:.3f} m，参考={pref_end:.3f} m，跟踪误差={err*1e3:.1f} mm")
print(" 越界(正/负)出现次数:", int((arr[:, 3] > Fmax + 1e-9).sum()), "/", int((arr[:, 3] < -Fmax - 1e-9).sum()))
np.savez('mpc_tutorial_log.npz', t=arr[:, 0], pos=arr[:, 1], vel=arr[:, 2], u=arr[:, 3])
print(" 已保存 mpc_tutorial_log.npz")
print(" >>> 结果: " + ("✅ 跟踪收敛、力矩未越界" if err < 5e-3 else "❌ 需调参"))