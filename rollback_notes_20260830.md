# 实验二 NMPCC 代码修改记录与回退说明（2026-08-30 晚）

> 本文件记录 2026-08-30 晚间对 `openarmx_mujoco/` 的全部改动、当前最终状态、
> 验证结果与回退映射。所有改动均已通过 `py_compile` 与 `--hold_only` 冒烟测试。
> 行数会随后续编辑变化，仅为量级参考。

---

## 0. 一句话总结

1. 检测类函数抽取到新文件 `exp2_diagnostics.py`（可独立运行启动对账），主程序只保留仿真与作图；
2. 控制层最终状态：**`clamp_balance_comp` 夹持预紧保留（启动预热 + 每拍 `u_ref` 都叠加）**，
   `y_guard` 附加保护层移除；grasp_pose 的 `startup_hold`/`handoff` 笛卡尔保持层**保留**；
3. `nmpcc_cart.py` 修复 2 个既有 bug 并**恢复 acados 编译复用**（不再每次重编译）。

---

## 1. 改动时间线（三个阶段）

### 阶段一：检测函数抽取（约 22:30）
- 新建 `exp2_diagnostics.py`：`contact_wrench_on_arm` / `contact_forces` /
  `pad_box_wrench_3d` / `box_net_wrench` / `box_attitude` / `press_delta_mm` /
  `reconciliation`（场景常数参数化，两场景共用），自带 `main()` 可独立运行：
  - `D:\acados_env\python.exe -u exp2_diagnostics.py`（slender，默认）
  - `D:\acados_env\python.exe -u exp2_diagnostics.py --scene grasp_pose`
- 两个主程序删除上述函数定义与 `reconciliation` 调用（含 sweep 分支），
  控制环/日志需要的测量函数改为 `from exp2_diagnostics import ...`；
  同时删除每拍刷屏的 `[PROFILE]` 打印、仅对账用的常量与未用导入。

### 阶段二：控制层精简尝试（约 23:06）——**已部分退回**
- 曾移除 `clamp_balance_comp`、`y_guard_torque`、grasp_pose 的
  `cartesian_hold_torque` 与 startup_hold/handoff 机制。
- **冒烟测试证明不可行**：slender 箱体在 ramp 阶段掉落 82mm、接触力归零；
  grasp_pose 箱体掉落 307mm。结论见第 4 节。

### 阶段三：修复与最终状态（约 23:23–23:45）
- 恢复 `clamp_balance_comp`（两文件：预热种子 + 每拍 `u_ref`）及
  `--clamp_ff_scale` / `--clamp_force_kp` 两个参数与 summarize 两个字段；
- grasp_pose 恢复 `cartesian_hold_torque` 与 startup_hold/handoff 机制
  （`--startup_hold` / `--handoff_time` / `--handoff_clamp_scale` /
  `--handoff_hold_floor` 四个参数、`rv_ref` 形参）；
- `y_guard`（"gate"）保持移除；
- `nmpcc_cart.py` 恢复 acados 编译复用。

---

## 2. 各文件当前最终状态

### `exp2_diagnostics.py`（新增，约 336 行）
- 全部"只起检测作用"的函数：接触力偶/姿态测量 5 个 + `press_delta_mm` +
  `reconciliation`（`scene` dict 携带箱尺寸/标定常数/rest_gap）。
- `main()` 自动加载场景（slender / grasp_pose 两个加载器）+ 执行对账。
- 独立运行验证过，输出与原主程序启动打印数值一致。

### `exp2_nmpcc_latest.py`（约 767 行，原 1201 行）
- 保留：`phase_profile`、`reset_world`、`run_case`、`summarize`、`main`、
  CSV 落盘、Fig.12 自动绘图、sweep/NMPC_CACHE（进程内求解器复用）。
- 控制环（`run_case`）：
  - 预热：`u_seed = gravity_comp() + clamp_balance_comp(fmag0)`，预求解 2 次；
  - 每拍：`u_last = nmpc.solve(..., u_ref=gravity_comp() + clamp_balance_comp(fmag), u_seed=u_seed)`，
    `u_cmd = clip(u_last)`。
- 已移除：`reconciliation`/`press_delta_mm`/接触测量函数定义（用
  `exp2_diagnostics`）、`y_guard_torque`、`--y_guard_*` 参数、
  summarize 中 `y_guard_*` 字段、`[PROFILE]` 打印。
- CLI 现有：`--force --kf_scale --q_x --q_y --q_z --q_rv --eta_cop_y
  --clamp_ff_scale --clamp_force_kp --t_ramp --t_hold --f_start --v_y --y_dist
  --hold_only --N --Tp --sim_steps --no_csv --no_viewer --sweep_* --repeats`。

### `exp2_nmpcc_latest_grasp_pose.py`（约 960 行，原 1201 行）
- 同 latest.py 的检测函数抽取 + `y_guard` 移除 + `clamp_balance_comp` 保留。
- 额外保留：`cartesian_hold_torque` 与 startup_hold/handoff 机制
  （`t < startup_hold` 时 gravity+clamp+笛卡尔保持；之后 handoff 平滑交给
  NMPCC，`handoff_hold_floor=0.75` 保留 75% 保持项；y_guard 项已从
  `u_raw = u_center + handoff_s * correction` 中去掉）。
- `f_meas=measured_force()` 保留（`nmpcc_cart_grasp_pose` 的实测力锚定输入）。
- 文档字符串已修正为该场景（箱 0.24×0.40×0.20 m、K=3575、rest_gap=0.248）。

### `nmpcc_cart.py`（518 行，3 处改动）
1. **`solve()` 状态码修复（既有 bug）**：本机 acados 的 `solve_for_x0`
   返回的是 `u0`（数组）而非状态码，`status != 0` 对数组判值直接崩溃。
   已在主求解与 rebuild 两处改为 `status = int(self.solver.status)`。
2. **补齐求解统计（既有 bug）**：新增 `last_time_sim` / `last_time_lin`
   记录（主程序日志引用，原来缺失会 `AttributeError`）。
3. **acados 编译复用（恢复）**：`__init__` 中先
   `AcadosOcpSolver(..., build=False, generate=False)` 加载已编译求解器，
   捕获 `(FileNotFoundError, OSError)` 时才 `build=True` 重编译。
   实测整个 run 墙钟 ~6.6s（含 casadi 导入 + 450 步仿真），复用生效。

---

## 3. 验证结果（--hold_only --no_viewer --no_csv --steps 100 冒烟测试）

| 场景 | 指标 | 数值 | 说明 |
|---|---|---|---|
| slender | z_sag | **+0.69 mm** | 箱体保持（失败案例：-82mm 掉落） |
| slender | FL/FR | 18.10 / 18.09 N | 夹持力保持 |
| slender | box_y_hold_max | 42.5 mm | **既有行为**（y_guard 关闭后即如此，与本次改动无关） |
| grasp_pose | z_sag | **-2.83 mm** | 箱体保持（失败案例：-307mm 掉落） |
| grasp_pose | hold_dy | 0.20 mm | |
| grasp_pose | FL/FR | 10.97 / 10.97 N | |
| grasp_pose | tilt | ≈0.01° | |

---

## 4. 关键设计结论（回退时不要再次踩坑）

1. **`clamp_balance_comp` 不是"只在启动时需要"，每拍都需要。**
   - 论文 x 轴是力阻抗：位置系数 `(1−ηf)²·Q_x ≈ 20`，力残差系数
     `ηf·kf_scale/kx ≈ 1.4e-5`，优化器靠力残差顶不出 10–15N 夹持力矩，
     每拍都会向输入正则化的方向"泄力"。
   - `u_ref = gravity + clamp_ff` 让"保持夹持力矩"成为输入跟踪代价的最优解；
     去掉后几拍内接触力归零、箱体掉落（实测 -82mm / -307mm）。
   - 预热种子只解决接手瞬间，不解决稳态维持。
2. **`y_guard`（"gate"）已按需求移除**（默认本来就是关闭的，不影响基线）。
3. **grasp_pose 的 startup_hold/handoff 必须保留**：空中场景（1.5kg 箱无托底）
   去掉后启动阶段箱体掉落 307mm；恢复后 hold_dy=0.20mm。
4. **力项 `/kx` 归一化的意义**（`ηf·kf_scale·(f−fref)/kx`）：
   把力误差换算成等价压缩量误差（`(f−fref)/kx = c−cref`），单位回到米；
   使力反馈闭环增益与接触刚度无关（不除则增益 ∝kx、Hessian ∝kx²≈7.8e6，
   条件数爆炸）。`kf_scale` 就是"力/位置平衡旋钮"，无需再加新系数：
   - 平衡条件：`ηf·kf_scale = (1−ηf)` → `kf_scale = 0.25`（当前 ηf=0.8）；
   - **注意 0.25 恰好等于稳定性判据 `(ηf/(1−ηf))·kf_scale/kx < 1/k_接触`
     的临界值**，建议从 0.15~0.2 起步实验；
   - 下一步可跑 `--sweep_kf 0.05,0.1,0.15,0.2,0.25` 验证纯 NMPCC 能否自己
     夹住（若能，`clamp_ff_scale=0` 消融即可去掉前馈）。

---

## 5. 回退映射（scratch/ 备份 → 目标文件）

| 目标文件 | 可用备份 | 备份内容 |
|---|---|---|
| `exp2_nmpcc_latest.py` | `scratch/exp2_nmpcc_latest_before_layer_cleanup_20260830.py` | 阶段一之后、阶段二精简之前（含 clamp + y_guard，无 diagnostics 抽取前状态不对——此备份已是抽取后） |
| `exp2_nmpcc_latest_grasp_pose.py` | `scratch/exp2_nmpcc_latest_grasp_pose_before_layer_cleanup_20260830.py` | 阶段一之后、阶段二之前（含 clamp + y_guard + handoff） |
| `exp2_nmpcc_latest_grasp_pose.py`（更早） | `scratch/exp2_nmpcc_latest_grasp_pose_before_handoff_fix_20260830.py` | 今天白天 handoff 修复前的版本（55.9KB） |
| `nmpcc_cart.py` | `scratch/nmpcc_cart_before_baseline_restore_20260830.py` | baseline restore 之前（注意：可能不含 acados 复用） |
| `nmpcc_cart_grasp_pose.py` | `scratch/nmpcc_cart_grasp_pose_before_cleanup_20260830.py` | 今天白天 cleanup 前 |

**回退提示：**
- 要回到"今晚全部改动之前"的 `exp2_nmpcc_latest.py`（1201 行原始版）：**没有备份**
  （阶段一改动前未留档）。grasp_pose 可用 `before_handoff_fix` 近似（白天状态）。
- `nmpcc_cart.py` 若只想去掉 acados 复用：把 `__init__` 里 try/except 的
  `build=False, generate=False` 块改回单行 `build=True, generate=True` 即可，
  status 修复和统计补齐建议保留（否则主程序跑不起来）。
- 带哈希校验的 acados 复用（"OCP formulation matches previous via hash"）
  在全部备份中已不存在，不可恢复；当前是 try/except 简化版。注意：简化版
  跨进程更改 Q/N/Tp 等公式参数不会自动重编译（与 grasp 控制器同款限制）。

---

## 6. 已知遗留 / 下一步建议

1. kf_scale 扫描（0.05→0.25）验证纯 NMPCC 夹持 + 极限环风险；
2. slender 场景 y 向 42.5mm 漂移为既有行为，若需要抑制需另做（重调
   `q_y`/`eta_cop_y` 或恢复 y_guard）；
3. 如需哈希校验版 acados 复用需重新实现（备份中已丢失）。
