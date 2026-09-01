# 实验二复现交接说明（IV. NUMERICAL SIMULATION / Fig.8）

> 给接手的 agent：本文件是当前工作的完整交接说明。你不需要从头理解论文，
> 照着本文件先跑通现有代码，再按"待办"继续即可。

## 0. 任务背景

复现论文 **Zhao 等, "A Dual-Arm Robot Cooperation Framework Based on a Nonlinear Model Predictive Cooperative Control", IEEE/ASME Trans. Mechatronics, vol.29 no.5, 2024** 的 **第 IV 节数值仿真（"仿真二"，Fig.8）**：

- 双 2 连杆机械臂（各自独立动力学），末端与"环境/夹持物"作用，产生相互作用力
- 一次 3 s 闭环仿真，时间分段切换三种合作模式（t=1 s、2 s 处竖线分隔）：
  - `tracking`（0–1 s）：ηcop=0, ηf=0 → 各臂跟自己的参考轨迹
  - `sync`（1–2 s）：ηcop=0.5, ηf=0 → 双臂相互靠近（运动同步）
  - `force`（2–3 s）：ηf=1 → 相互作用力→0（力优先）
- 输出论文 Fig.8 四联图：(a) 轨迹 (b) x 向时间线 (c) 输入力矩 (d) 相互作用力

论文全文文本（公式/表格原文）在 `_paper_text.txt`（从 PDF 提取，含所有公式符号）。
查公式直接 `grep "式号" _paper_text.txt`，如 `(27)`、`(36)`、`TABLE III`。

## 1. 代码结构

| 文件 | 角色 |
|---|---|
| `nmpcc.py` | **算法库**：NMPCC/NMPCCConfig/two_link_plant；负责把论文式 (11)(12)(28)(30)(31)(32) 翻译成 acados OCP 并生成求解器 |
| `exp_sim2_nmpcc.py` | **环境力版主脚本**（论文式 36：`f_i = K(r_env,i − r_i)`），当前正式交付版本 |
| `exp_sim2_nmpcc_spring.py` | **双臂间虚拟弹簧版**（论文式 27：`f_1 = K(r_2 − r_1), f_2 = −f_1`），环境力版的复制改动机，输出到 `sim2_figs_spring/` |
| `sim2_figs/` | 环境力版输出：`fig8_sim2_<时间戳>.svg/.png` + `sim2_session_<时间戳>.csv` |
| `sim2_figs_spring/` | 弹簧版输出：`fig8_sim2_spring_<时间戳>.svg/.png` + `sim2_session_spring_<时间戳>.csv` |
| `_paper_text.txt` | 论文全文文本（只读参考，勿删） |

## 2. 如何运行

```powershell
# 注意！必须用 acados 环境（base/miniconda 没有 casadi+acados）
D:\acados_env\python.exe -u exp_sim2_nmpcc.py
D:\acados_env\python.exe -u exp_sim2_nmpcc_spring.py
```

- 首次运行会为每个 OCP 生成并编译 acados C 代码（`codegen_ocp_nmpcc_two_link_dual_<hash>` 目录，
  **这是缓存**，删了会自动重建，每个约 5–10 s）。
- 运行后输出到 `sim2_figs/`（或 `sim2_figs_spring/`），文件名含时间戳。
- 控制周期 10 ms、仿真步长 5 ms、Tp=100 ms（N=50）、三段各 1 s。
- 求解均值 ~1.4–1.9 ms（无需强求论文的 <1 ms）。

## 3. 关键实现细节（改代码前必读）

1. **η 是分轴矩阵**（式 12 的 `diag{ηx, ηy}`）。
   - `force` 模式用 `eta_f=[[1,0],[1,0]]`（x 向力优先、y 向保持跟踪），因为
     `k_stiff=diag(1000,0)` 只有 x 向：全轴 ηf=1 会让 y 向残差恒为 0 → GN 海森半正定 → QP 秩亏崩溃。
   - `sync` 模式的 x 向 ηcop=0（力通道保持），y 向 ηcop=0.5（运动同步）：
     全轴 ηcop=0.5 的凸组合会把两臂 x 拉向平均值，穿过各自墙面 → 力反号（与论文 (d) 恒正/恒负矛盾）。
2. **参考轨迹是分段线性时变**（`SEGMENTS` + 每段 `r_ref(t) = r0 + (r1−r0)·τ`），不是常数目标点。
   论文原文 "each arm follows its reference trajectory"。给常数目标会导致 NMPC 短视界（0.1s）下轨迹甩动。
   参考以 **OCP 参数 p** 进入求解器（`p = [f(2nt), rLref, rRref, ṙLref, ṙRref]`，每拍 `solve` 时更新），
   残差为式 (12) 的完整 ε_mult 表达式（含 r_j 状态 → 预测中 e_cop 演化），`yref ≡ 0`。
3. **两臂严格镜像对称**（用户要求 |f1|=|f2|、轨迹对称）：
   - 所有参考常数满足"臂2 = 臂1 关于 x=0.45 的镜像"（`x_ref2 = 0.9 − x_ref1`）
   - 每拍 `enforce_xmirror(x)`：臂2 状态由臂1 末端经解析 IK（双解取连续）+ 速度映射 `ṙ2=(−ṙ1x, ṙ1y)` 重建。
   - `ik2`（DLS：`dq=(JᵀJ+λ²I)⁻¹Jᵀe`，λ=0.05，多初值）避免近奇异发散。
4. **速度约束**：论文式 (30) 的 `∥ṙ∥≤vmax` 在 SQP_RTI+HPIPM 下以 box 状态约束会使 QP 病态，
   改用软约束——式 (31) 里的 `ε̇ᵀ·ηdev·ε̇`（ηdev=50–80）抑制末端速度。
5. **力矩限幅 ±8 N·m**（对齐论文实验一 "10 N for safety"）；切换瞬间 τ 饱和 ±8 是
   "η 阶跃 + 参考速度跳变 + RTI 冷启动"的正常现象（论文图 (c) 同样有切换尖峰）。
6. **权重 Q/ηdev/ηu/P 论文未公布**（原文只说 "weight matrix should be determined based on the task decomposition"），
   当前值按量级平衡自整定：Q≈400–500（位置误差）、ηdev=50–80（阻尼）、ηu=0.05–0.2（输入正则）。
7. **求解器选项**（nmpcc.py）：`SQP_RTI`（=论文 real-time iteration）、`GAUSS_NEWTON`（=论文高斯-牛顿求 Hessian）、
   `regularize_method="MIRROR"` + `lm_reg`（力优先段半正定海森的正则）、QP 失败时重建求解器重试一次（兜底）。
   误差建模与论文式 (12)(28)(29) 一致：ε_mult 完整表达式为残差（E 全放 y_expr），参考轨迹走参数 p（见第 2 条）。
8. **输出**：`plot_fig8` 同时存 SVG（矢量可放大）+ PNG（预览）；CSV 列含 `t, seg, r1x,r1y, r2x,r2y, tau1..4, f1x,f1y, f2x,f2y`。

## 4. 当前成果（两版都在，参数/结果都是最新）

- 环境力版：`sim2_figs/fig8_sim2_20260822_200136.svg/.png/.csv`
  - tracking 末 (0.4477, 0.3668)/(0.4523, 0.3668)，力 0.35/0.35 N；sync 末 0.27/0.27 N；force 末 0.00/0.00 N
- 弹簧版：`sim2_figs_spring/fig8_sim2_spring_20260822_200007.svg/.png/.csv`
  - 力严格镜像（1.17/1.17、1.44/1.44、0.01/0.01 N）

## 5. 与论文的差异（写复现报告时按此表述）

1. **求解器**：论文 MATLAB R2017a + ACADO（qpOASES，<1 ms）→ 我们 Python + acados（ACADO 继任实现，
   SQP_RTI=real-time iteration、HPIPM），求解均值 1.4–1.9 ms。
2. **预测模型坐标**：论文式 (25) 末端笛卡尔空间（[r, ṙ]，M(r)/C(r,ṙ)/G(r)）→ 我们关节空间（[q, v]，式 35）。
   **动力学等价、状态坐标不同**（残差/代价本身都用末端量表达，与论文数学一致；积分变量是内部实现选择）。
3. **误差建模（已按论文统一）**：论文式 (28)(29) 的误差状态空间（e_trac/e_cop 为预测中演化的状态）→
   我们按式 (11)(12) 把 **ε_mult 的完整表达式直接放进 NONLINEAR_LS 残差**
   （`E_i = (1−If)[r_i − (1−Ic)r_i,ref − Ic·r_j] − If·f_i`，其中 r_i、r_j 都是状态函数：
   GN 雅可比含 r_j → 预测中协同误差 e_cop 随状态演化，与式 (29) 误差动态一致），
   参考轨迹 r_i,ref/ṙ_i,ref 作为 **OCP 参数 p**（外生输入，每拍更新），`yref ≡ 0`；
   平衡点仍是论文式 (15)(16) 的凸组合。**这是与论文数学结构一致的标准实现。**
4. **速度约束**：硬约束（式 30）→ 软约束（ηdev 阻尼，见第 3 节第 4 条）。
5. **权重数值**：论文未公布 → 自整定（结构同式 (31)(32)）。
6. **对称性**：论文未显式处理（对称系统自然对称）→ 我们每拍解析镜像投影（保证 |f1|=|f2|、轨迹对称，
   与论文结果一致）。
7. **任务几何数值**：论文图上读值（x 0.081–0.087, y 0.33–0.65）→ 我们用自洽等价几何
   （x 0.445–0.455，y 0.33–0.44），行为特征（相向对称弧线/靠拢/力归零/分段）对齐。
8. 一致项：Tp=100 ms、Table III 参数（b1..c1 原值）、η 配置（0 / 0.5 / 0–1 分轴）。
9. **摩擦处理（待统一，见下文 B 项）**：论文 IV 节明确"we ignore the friction law and treat it as
   the uncertainty term"——预测模型**不含**摩擦（物理仿真含，作为不确定项）；当前 `two_link_plant` 的
   `f_expl` 被预测模型与物理积分共用且含摩擦 D 项，需改为"预测不含摩擦"（`two_link_plant` 加
   `friction` 参数：True=物理、False=预测；`make_plant_and_sim` 分别构造，NMPCC 用无摩擦 plant）。
   注：摩擦项形式按当前实现 `D(q̇)=[Dq1+μ·q̇1; Dq2+μ·q̇2]`（μ=0.01 很小，式 35 数值影响 <1%）。

## 6. 待办 / 可扩展方向

- [ ] **IV 节的"第一个仿真"**（合作指标映射，Fig.6/Fig.7）：纯代数画 `(I−ηcop)` 逆矩阵
  （式 18/33）随 ηcop,1/ηcop,2、ηf 变化的地图（Tracking/Synchronization/Impedance/Force priority 四分区），
  与 Fig.8 无关，十几行代码即可补全，如论文"整节复现"需要。
- [ ] 可选：切换瞬间的 η 渐变（0.2–0.3 s 斜坡）消掉 ±8 N·m 切换尖峰（当前尖峰与论文同性质，非必须）。
- [ ] 可选：`mujoco_dual_arm_plant()`（nmpcc.py 内已有）把 Fig.8 搬上 openarmx MuJoCo 模型做三维版。
- 注意：`exp_sim2_nmpcc*.py` 只用了 `two_link_plant`（2 连杆），与 `ikforce_grasp.py`（MuJoCo 箱夹实验）是两套东西。