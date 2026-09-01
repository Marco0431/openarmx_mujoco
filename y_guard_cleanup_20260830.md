# y_guard 清除记录（2026-08-30 23:59）

> 应要求把 `y_guard`（附加 y 保护层）从全部活动文件中清除，并顺带修复两个旧变体
> 文件里被并发工作流误写损坏的 `build_p` 一行。

---

## 1. y_guard 是什么（清除前）

执行器层附加项，不属于论文 NMPCC 代价。读取**箱体中心**的 y 位置/速度，
与参考做 PD 得到限幅力 `fy = kp·(y_ref − box_y) + kd·(vy_ref − box_vy)`，
通过末端雅可比 y 行平分映射到双臂关节力矩，加在 NMPCC 输出之上：

```python
u_raw = u_last + y_guard_torque(refs_now, vrefs_now)
```

三个 CLI 参数 `--y_guard_kp`（默认 0）、`--y_guard_kd`（默认 0）、
`--y_guard_fmax`（默认 20）**默认全为零，从未实际生效**；仅当显式开启时才对
箱体 y 漂移施加闭环保护。summarize 里另有 `y_guard_enabled/kp/kd/fmax` 指标字段。

## 2. 清除范围（活动文件全部清零）

| 文件 | 状态 |
|---|---|
| `exp2_nmpcc_latest.py` | 今晚 23:06 控制层精简时已删除，保持删除 |
| `exp2_nmpcc_latest_grasp_pose.py` | 同上 |
| `exp2_nmpcc_latest_grasppose_scene.py` | 本次删除（函数 + u_raw 加项 + 指标 + 3 个参数） |
| `exp2_nmpcc_grasppose.py` | 本次删除（同上） |
| `scratch/*.py` 备份 | **有意不动**（回退素材） |

每个文件删除的内容一致：
1. `y_guard_torque(refs, vrefs)` 函数定义（约 25 行）；
2. 控制环里 `u_raw = u_last + y_guard_torque(...)` 改为 `u_raw = u_last`，
   同时删除不再使用的 `refs_now, vrefs_now, _, _, _ = reference_for(t)` 行；
3. `summarize` 指标字典里的 `y_guard_enabled` / `y_guard_kp` / `y_guard_kd` /
   `y_guard_fmax` 四个字段；
4. `main()` 里 `--y_guard_kp` / `--y_guard_kd` / `--y_guard_fmax` 三个 argparse 参数。

行为影响：零。三个参数默认值全是 0，删除前后任何运行结果不变。

## 3. 顺带修复：两个旧变体文件的 build_p 损坏行

`exp2_nmpcc_latest_grasppose_scene.py` 与 `exp2_nmpcc_grasppose.py` 的
`build_p` 里残留一行并发工作流误写产生的工具调用 JSON 碎片（一个超长行，
`p[150:156]` 赋值之后整段被 `#` 注释吞掉）。后果：**`p[156:162]`（右臂力参考）
从未被赋值**。已按该碎片中"new_string"的意图恢复为：

```python
p[150:156] = [-fmag, 0, 0, 0, 0, 0]    # 与 exp2_nmpcc_latest 同约定：left=−x 侧 → f_ref=−F
p[156:162] = [fmag, 0, 0, 0, 0, 0]     # right=+x 侧 → f_ref=+F
```

（注意：这两个旧文件因此从"人形抱箱 ±F 约定"翻转为与
`exp2_nmpcc_latest.py` 一致的 left=−x/−F、right=+x/+F 约定，这是那次被
截断的编辑本来要做的改动。）

## 4. 验证

- 两个文件 `py_compile` 通过；
- 全库 grep（排除 `scratch/`）：`y_guard` 零命中（两个主程序 docstring 里
  保留的"y_guard 附加保护层已移除"是历史说明文字，不算代码）；
- `p[150:156]` / `p[156:162]` 两行已恢复且符号约定正确；
- 临时清理脚本 `_fix_yguard_cleanup.py` 已删除。

## 5. 备注

- 若以后需要压箱体 y 漂移（地面版保持阶段约 42.5mm 的既有漂移），可参考
  `scratch/exp2_nmpcc_latest_before_layer_cleanup_20260830.py` 里的原实现
  重新加回，但需重新整定 kp/kd 才有实际效果。
- 回退时 `scratch/` 备份仍含 y_guard（默认零，不影响行为），如回退后想再次
  清除，参照本记录第 2 节逐项删除即可。
